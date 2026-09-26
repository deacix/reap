# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Write a pruned MiMo-V2 checkpoint by slicing its source's own tensors.

A MiMo reference ships quantized (MXFP4 experts, FP8 blocks with a
TP-chunked fused ``qkv_proj``), and its pruned build keeps that precision
and that layout: nothing is dequantized or re-quantized, so the serving
lanes load it the way they load the source. Per MoE layer:

- the kept experts' tensors (``mlp.experts.<old>.*``, weights and scales)
  are copied under dense new ids, in ascending old-id order;
- the router's ``mlp.gate.weight`` rows and ``mlp.gate.e_score_correction_bias``
  entries are sliced to the same order;
- every other tensor is copied byte for byte.

``config.json`` gets ``n_routed_experts``; the index keeps the source's
``metadata`` (``save_format``, ``tp_size``) with ``total_size`` recomputed;
the model code, the tokenizer and every other top-level file and
subdirectory are copied. ``drop_drafts`` leaves out the multi-token
prediction layers (``model.mtp.*``, ``num_nextn_predict_layers: 0``) and the
``dflash/`` drafter.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from reap.legwork.checkpoint import (
    DEFAULT_SHARD_BYTES,
    ShardWriter,
    TensorReader,
    copy_files,
    read_index,
)

EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
ROUTER = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$")
DRAFT_PREFIXES = ("model.mtp.",)
DRAFT_LAYER = re.compile(r"^model\.mtp\.layers\.(\d+)\.")
DRAFT_DIRS = ("dflash",)


def source_draft_layers(source: str | pathlib.Path) -> int:
    """How many multi-token-prediction layers (``model.mtp.layers.<n>.*``) the source carries."""
    weight_map, _ = read_index(source)
    return len({match.group(1) for name in weight_map if (match := DRAFT_LAYER.match(name))})


@dataclass(frozen=True)
class SourceLayer:
    """One MoE layer as the source's ``config.json`` names it (the fields
    ``reap.legwork.prune.resolve_kept_plan`` reads)."""

    index: int
    num_experts: int
    kind: str = "moe"


def source_moe_layers(config: Mapping[str, Any]) -> list[SourceLayer]:
    """The MoE layers ``moe_layer_freq`` marks, each ``n_routed_experts`` wide."""
    freq = config.get("moe_layer_freq")
    if not isinstance(freq, list) or not freq:
        raise ValueError("config.json names no moe_layer_freq list; --slice-source reads MiMo-V2 checkpoints")
    experts = int(config["n_routed_experts"])
    return [SourceLayer(index, experts) for index, flag in enumerate(freq) if flag]


@dataclass
class SliceReport:
    out: str
    keep: int
    experts_before: int
    copied: int = 0
    sliced: int = 0
    dropped: int = 0
    total_size: int = 0


def slice_source(
    source: str | pathlib.Path,
    out: str | pathlib.Path,
    kept: Mapping[int, Sequence[int]],
    drop_drafts: bool = False,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    progress: Callable[[float], None] | None = None,
) -> SliceReport:
    source, out = pathlib.Path(source), pathlib.Path(out)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    layers = source_moe_layers(config)
    experts_before = layers[0].num_experts
    widths = {len(ids) for ids in kept.values()}
    if set(kept) != {layer.index for layer in layers}:
        raise ValueError(
            f"the kept sets name layers {sorted(kept)}; the MoE layers are {[l.index for l in layers]}"
        )
    if len(widths) != 1:
        raise ValueError(f"every MoE layer keeps the same number of experts, got {sorted(widths)}")
    keep = widths.pop()
    renumber: dict[int, dict[int, int]] = {}
    order: dict[int, torch.Tensor] = {}
    for index, ids in kept.items():
        ascending = sorted(int(i) for i in ids)
        if len(set(ascending)) != len(ascending) or not all(0 <= i < experts_before for i in ascending):
            raise ValueError(f"layer {index}: the kept ids must be unique and below {experts_before}")
        renumber[index] = {old: new for new, old in enumerate(ascending)}
        order[index] = torch.tensor(ascending, dtype=torch.long)
    reader = TensorReader(source)
    writer = ShardWriter(out, shard_bytes)
    report = SliceReport(out=str(out), keep=keep, experts_before=experts_before)
    names = reader.names()
    try:
        for position, name in enumerate(names):
            if progress is not None:
                progress(95.0 * position / len(names))
            if drop_drafts and name.startswith(DRAFT_PREFIXES):
                report.dropped += 1
                continue
            expert = EXPERT.match(name)
            if expert is not None and int(expert.group(1)) in renumber:
                new = renumber[int(expert.group(1))].get(int(expert.group(2)))
                if new is None:
                    report.dropped += 1
                else:
                    writer.add(
                        f"model.layers.{expert.group(1)}.mlp.experts.{new}.{expert.group(3)}",
                        reader.get(name),
                    )
                    report.copied += 1
                continue
            router = ROUTER.match(name)
            if router is not None and int(router.group(1)) in order:
                tensor = reader.get(name)
                writer.add(name, tensor.index_select(0, order[int(router.group(1))]))
                report.sliced += 1
                continue
            writer.add(name, reader.get(name))
            report.copied += 1
        writer.close(metadata={k: v for k, v in reader.metadata.items() if k != "total_size"})
    finally:
        reader.close()
    report.total_size = writer.total_size
    config["n_routed_experts"] = keep
    if drop_drafts:
        config["num_nextn_predict_layers"] = 0
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_files(source, out, skip_dirs=DRAFT_DIRS if drop_drafts else ())
    if progress is not None:
        progress(100.0)
    return report
