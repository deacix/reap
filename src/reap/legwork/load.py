# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Reload a pruned checkpoint, ragged ones included.

A uniform prune (the default) reloads with the stock
``AutoModelForCausalLM.from_pretrained``. ``--skip-layers`` produces a
*ragged* checkpoint whose skipped layers kept their full width; stock
transformers builds every layer at ``config.n_routed_experts`` and
refuses the wider tensors on the size mismatch. This helper widens those
layers while the model is built: it
wraps the MoE block's constructor for the duration of ``from_pretrained``
so each block reads its own width from ``reap_pruning.n_routed_experts_per_layer``,
and transformers' own loader (key conversion, dtype rules, device maps)
does the rest. transformers-side evaluation only — vLLM builds every
layer from one ``n_routed_experts``.
"""

from __future__ import annotations

import copy
import inspect
import json
import pathlib
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from reap.legwork.arch import model_attrs, moe_layers


def read_pruning_record(path: str | pathlib.Path) -> dict[str, Any] | None:
    config = json.loads((pathlib.Path(path) / "config.json").read_text(encoding="utf-8"))
    record = config.get("reap_pruning")
    return record if isinstance(record, dict) else None


@contextmanager
def _widened_blocks(config, widths: list[int]) -> Iterator[None]:
    """Patch the MoE block class so layer ``i`` is built ``widths[i]`` wide."""
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)
    attrs = model_attrs(skeleton)
    layers = moe_layers(skeleton)
    if len(widths) != len(layers):
        raise ValueError(
            f"reap_pruning names {len(widths)} layer widths, the model has {len(layers)} MoE layers"
        )
    width_by_layer = {layer.index: int(width) for layer, width in zip(layers, widths)}
    block_cls = type(layers[0].block)
    parameters = list(inspect.signature(block_cls.__init__).parameters)
    if "layer_idx" not in parameters:
        raise ValueError(
            f"{block_cls.__name__} takes no layer_idx; the lane cannot widen its layers one by one"
        )
    original_init = block_cls.__init__
    num_experts_attr = attrs["num_experts"]

    def widened_init(self, cfg, layer_idx, *args, **kwargs):
        width = width_by_layer.get(int(layer_idx))
        if width is not None and width != int(getattr(cfg, num_experts_attr)):
            cfg = copy.deepcopy(cfg)
            setattr(cfg, num_experts_attr, width)
        original_init(self, cfg, layer_idx, *args, **kwargs)

    block_cls.__init__ = widened_init
    try:
        yield
    finally:
        block_cls.__init__ = original_init


def load_pruned(path: str | pathlib.Path, dtype: torch.dtype | None = None, device_map: Any = None):
    """``from_pretrained`` for a pruned checkpoint; widens skipped layers."""
    from transformers import AutoConfig, AutoModelForCausalLM

    path = pathlib.Path(path)
    kwargs: dict[str, Any] = {"dtype": dtype if dtype is not None else "auto"}
    if device_map is not None:
        kwargs["device_map"] = device_map
    record = read_pruning_record(path)
    widths = [int(w) for w in (record or {}).get("n_routed_experts_per_layer") or []]
    if len(set(widths)) <= 1:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    config = AutoConfig.from_pretrained(path)
    with _widened_blocks(config, widths):
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model.eval()
