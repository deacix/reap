# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Resolve a model's MoE layout from ``MODEL_ATTRS``.

The lane reads the same table the research entry points read
(``reap.model_util.MODEL_ATTRS``) and takes two shapes:

- the fused experts transformers 5.x gives its ``@use_experts_implementation``
  families (DeepSeek-V4): the experts module holds ``gate_up_proj`` /
  ``down_proj`` as 3-D parameters indexed by expert and applies the
  family's gate through ``_apply_gate``;
- the loop-based experts of an entry marked ``legwork_loop`` (MiMo-V2, the
  checkpoint's own model code): ``experts`` is a ``ModuleList`` of MLPs and
  the router returns ``(topk_idx, topk_weight)``. The lane observes these
  through the router and prunes them by slicing the source checkpoint
  (``reap.legwork.slice``), never in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from reap.model_util import MODEL_ATTRS

HASH_KIND = "hash_moe"
TOPK_KIND = "moe"


@dataclass(frozen=True)
class MoeLayer:
    """One decoder layer's MoE block and the modules the lane edits."""

    index: int
    block: nn.Module
    experts: nn.Module
    router: nn.Module
    #: ``"moe"`` for a learned top-k router, ``"hash_moe"`` for a frozen
    #: token-id table (``tid2eid``) — DeepSeek-V4's bootstrap layers.
    kind: str
    #: Fused 3-D expert parameters; ``False`` for a ``ModuleList`` of experts.
    fused: bool = True

    @property
    def num_experts(self) -> int:
        if isinstance(self.experts, nn.ModuleList):
            return len(self.experts)
        return int(self.experts.num_experts)


def model_attrs(model: nn.Module) -> dict:
    """The ``MODEL_ATTRS`` entry for ``model`` — by class name first, then by
    ``config.model_type`` — or a ``ValueError`` naming what is supported."""
    class_name = model.__class__.__name__
    entry = MODEL_ATTRS.get(class_name)
    if entry is None:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        entry = MODEL_ATTRS.get(model_type) if model_type else None
    if entry is None:
        supported = sorted(k for k in MODEL_ATTRS if k[:1].isupper())
        raise ValueError(
            f"{class_name} is not a REAP-supported architecture; supported: {supported}"
        )
    if not entry.get("fused") and not entry.get("legwork_loop"):
        raise ValueError(
            f"{class_name} uses loop-based experts; the Legwork lane prunes the fused "
            "families and MiMoV2ForCausalLM only — use the research entry points."
        )
    return entry


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The decoder stack: under ``base_model_prefix``, or ``model.model`` for
    remote code that declares no prefix (MiMo-V2's)."""
    prefix = getattr(model, "base_model_prefix", "") or ""
    for base in (getattr(model, prefix, None) if prefix else None, getattr(model, "model", None), model):
        layers = getattr(base, "layers", None) if base is not None else None
        if isinstance(layers, nn.ModuleList):
            return layers
    raise ValueError(f"{model.__class__.__name__} exposes no decoder layers")


def moe_layers(model: nn.Module) -> list[MoeLayer]:
    """Every decoder layer holding an MoE block of the entry's shape, in
    layer order (a dense layer's MLP has no router and is skipped)."""
    attrs = model_attrs(model)
    loop = bool(attrs.get("legwork_loop"))
    layer_types_attr = attrs.get("layer_types")
    layer_types = (
        list(getattr(model.config, layer_types_attr)) if layer_types_attr else None
    )
    hash_kind = attrs.get("hash_layer_type", HASH_KIND)
    found: list[MoeLayer] = []
    for index, layer in enumerate(_decoder_layers(model)):
        block = getattr(layer, attrs["moe_block"], None)
        if block is None:
            continue
        experts = getattr(block, attrs["experts"], None)
        router = getattr(block, attrs["router"], None)
        if experts is None or router is None:
            continue
        if loop:
            if not isinstance(experts, nn.ModuleList):
                continue
        elif not hasattr(experts, "gate_up_proj"):
            continue
        kind = TOPK_KIND
        if layer_types is not None and index < len(layer_types) and layer_types[index] == hash_kind:
            kind = HASH_KIND
        elif attrs.get("hash_table") and hasattr(router, attrs["hash_table"]):
            kind = HASH_KIND
        found.append(
            MoeLayer(
                index=index, block=block, experts=experts, router=router, kind=kind, fused=not loop
            )
        )
    if not found:
        raise ValueError(f"{model.__class__.__name__} has no MoE layers to prune")
    return found


def hash_routed_layers(model: nn.Module) -> list[int]:
    """The layer indices whose routing is a frozen token-id table."""
    return [layer.index for layer in moe_layers(model) if layer.kind == HASH_KIND]


def num_routed_experts(model: nn.Module) -> int:
    attrs = model_attrs(model)
    return int(getattr(model.config, attrs["num_experts"]))
