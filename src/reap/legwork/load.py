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
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

import torch

from reap.legwork.arch import model_attrs, moe_layers


def read_pruning_record(path: str | pathlib.Path) -> dict[str, Any] | None:
    config = json.loads((pathlib.Path(path) / "config.json").read_text(encoding="utf-8"))
    record = config.get("reap_pruning")
    return record if isinstance(record, dict) else None


def is_fp8_checkpoint(path: str | pathlib.Path) -> bool:
    """Whether the checkpoint ships FP8 weights (``quantization_config.quant_method``
    ``fp8``; DeepSeek-V4's FP4-packed experts ride the same config)."""
    try:
        config = json.loads((pathlib.Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    quant = config.get("quantization_config")
    return isinstance(quant, dict) and str(quant.get("quant_method", "")).lower() == "fp8"


def dequantize_kwargs(path: str | pathlib.Path, dtype: str = "auto") -> dict[str, Any]:
    """What ``--dequantize`` adds to ``from_pretrained``: an FP8 checkpoint
    loads every weight in BF16, the load transformers picks by itself under
    compute capability 8.9, so a GPU the lane installs no FP8 kernels for
    (an AMD ROCm board, which reports 9.x) runs plain BF16 matmuls. The
    loading-attributes ``FineGrainedFP8Config`` only flips ``dequantize`` on
    the checkpoint's own scale layout. Any other checkpoint loads as it is."""
    if not is_fp8_checkpoint(path):
        return {}
    from transformers import FineGrainedFP8Config

    kwargs: dict[str, Any] = {"quantization_config": FineGrainedFP8Config(dequantize=True)}
    if dtype == "auto":
        kwargs["dtype"] = torch.bfloat16
    return kwargs


def loaded_dequantized(model, path: str | pathlib.Path) -> bool:
    """Whether an FP8 checkpoint came up in float: ``--dequantize``, or
    transformers' own fallback on a GPU under compute capability 8.9."""
    return is_fp8_checkpoint(path) and not getattr(model, "is_quantized", False)


_SCALE_SOURCE = ".weight_scale_inv$"
_ANCHORED_WEIGHT = ".weight$"


def drop_dequantize_codec(model) -> int:
    """transformers keeps the load-time weight conversions on the model to
    revert them on save, and the FP8 loader's dequantize op reverses into a
    re-quantize: every 2-D weight the block tiles, the embeddings and the
    head included, would come back out as FP8 codes beside a
    ``weight_scale_inv``, under a config that no longer names FP8, so a plain
    reload reads garbage (deacix/legwork#24439). Undo what the dequantizing
    load did to the conversions so the model saves as the float checkpoint it
    holds, in the checkpoint's own layout: the dequantize-only converter goes,
    and each model converter it was prepended to (an expert merge) gets back
    its weight sources, without the scale sources and the ``$`` anchors the
    quantizer added. -> the number of dequantize ops dropped."""
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    conversions = getattr(model, "_weight_conversions", None)
    if not isinstance(conversions, list):
        return 0
    kept, dropped = [], 0
    for conversion in conversions:
        operations = getattr(conversion, "operations", None)
        if not isinstance(operations, list) or not any(
            isinstance(op, Fp8Dequantize) for op in operations
        ):
            kept.append(conversion)
            continue
        remaining = [op for op in operations if not isinstance(op, Fp8Dequantize)]
        dropped += len(operations) - len(remaining)
        if not remaining:
            continue
        sources = [
            pattern[:-1] if pattern.endswith(_ANCHORED_WEIGHT) else pattern
            for pattern in conversion._original_source_patterns
            if not pattern.endswith(_SCALE_SOURCE)
        ]
        restored = WeightConverter(
            source_patterns=sources,
            target_patterns=list(conversion._original_target_patterns),
            operations=remaining,
            force_cpu=conversion.force_cpu,
        )
        for name in ("scope_prefix", "base_model_prefix"):
            if hasattr(conversion, name):
                setattr(restored, name, getattr(conversion, name))
        kept.append(restored)
    model._weight_conversions = kept
    return dropped


@contextmanager
def float_forward(model) -> Iterator[None]:
    """BF16 autocast on every device the model sits on. DeepSeek-V4 keeps its
    norms and hyper-connections in float32, so a dequantized model's float32
    activations meet BF16 linears, which a plain forward refuses
    (``expected m1 and m2 to have the same dtype``); the FP8 path's linears
    cast their own input."""
    device_types = sorted({param.device.type for param in model.parameters()} - {"meta"})
    with ExitStack() as stack:
        for device_type in device_types:
            stack.enter_context(torch.autocast(device_type, dtype=torch.bfloat16))
        yield


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
