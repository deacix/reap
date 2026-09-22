# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Resolve a model's MoE layout from ``MODEL_ATTRS``.

The lane reads the same table the research entry points read
(``reap.model_util.MODEL_ATTRS``) and requires the fused-experts shape
transformers 5.x gives its ``@use_experts_implementation`` families: the
experts module holds ``gate_up_proj`` / ``down_proj`` as 3-D parameters
indexed by expert and applies the family's gate through ``_apply_gate``.
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

    @property
    def num_experts(self) -> int:
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
    if not entry.get("fused"):
        raise ValueError(
            f"{class_name} uses loop-based experts; the Legwork lane prunes the fused "
            "families only (DeepseekV4ForCausalLM) — use the research entry points."
        )
    return entry


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, model.base_model_prefix, model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError(f"{model.__class__.__name__} exposes no decoder layers")
    return layers


def moe_layers(model: nn.Module) -> list[MoeLayer]:
    """Every decoder layer holding a fused MoE block, in layer order."""
    attrs = model_attrs(model)
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
        if experts is None or router is None or not hasattr(experts, "gate_up_proj"):
            continue
        kind = TOPK_KIND
        if layer_types is not None and index < len(layer_types) and layer_types[index] == hash_kind:
            kind = HASH_KIND
        elif attrs.get("hash_table") and hasattr(router, attrs["hash_table"]):
            kind = HASH_KIND
        found.append(MoeLayer(index=index, block=block, experts=experts, router=router, kind=kind))
    if not found:
        raise ValueError(f"{model.__class__.__name__} has no fused MoE layers to prune")
    return found


def hash_routed_layers(model: nn.Module) -> list[int]:
    """The layer indices whose routing is a frozen token-id table."""
    return [layer.index for layer in moe_layers(model) if layer.kind == HASH_KIND]


def num_routed_experts(model: nn.Module) -> int:
    attrs = model_attrs(model)
    return int(getattr(model.config, attrs["num_experts"]))
