# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Write a pruned checkpoint by slicing its source's own tensors.

A MiMo-V2 or DeepSeek-V4 reference ships quantized (MXFP4 or FP4 experts,
FP8 blocks elsewhere), and its pruned build keeps that precision and that
layout: nothing is dequantized or re-quantized and no model is built, so
every loader reads the build the way it reads the source. Per MoE layer:

- the kept experts' tensors (weights and scales alike) are copied under
  dense new ids, in ascending old-id order;
- the router's rows and its score-correction bias entries are sliced to the
  same order;
- a hash-routed layer's token table (DeepSeek-V4's ``tid2eid``) is remapped
  onto the kept set the way the in-memory prune remaps it;
- every other tensor is copied byte for byte.

The layouts, by the source's ``model_type``:

- MiMo-V2 (``model.layers.N.mlp.experts.<id>.*``): ``moe_layer_freq`` names
  the MoE layers. ``drop_drafts`` leaves out the multi-token prediction
  layers (``model.mtp.*``, ``num_nextn_predict_layers: 0``) and the
  ``dflash/`` drafter.
- DeepSeek-V4 (DeepSeek's own names, ``layers.N.ffn.experts.<id>.*``): every
  decoder layer is MoE, the first ``num_hash_layers`` (or the
  ``hash_moe`` entries of ``mlp_layer_types``) routed by table. The draft
  layers (``mtp.*``) are never carried, since a draft layer's own experts
  would need a kept set a kept plan cannot name: ``num_nextn_predict_layers``
  becomes 0. This is the NVIDIA prune's writer (deacix/legwork#25045):
  transformers 5.17 saves an FP8 model under ``model.``-prefixed names and
  loses its experts' scale grids.

``config.json`` gets ``n_routed_experts``; the index keeps the source's
``metadata`` (``save_format``, ``tp_size``) with ``total_size`` recomputed;
the model code, the tokenizer and every other top-level file and
subdirectory are copied.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from reap.legwork.arch import HASH_KIND
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

V4_MODEL_TYPE = "deepseek_v4"
V4_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")
V4_ROUTER = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias)$")
V4_HASH_TABLE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.tid2eid$")
V4_DRAFT_PREFIX = "mtp."
V4_DRAFT_LAYER = re.compile(r"^mtp\.(\d+)\.")
#: transformers' own default when a DeepSeek-V4 config names neither word.
V4_DEFAULT_HASH_LAYERS = 3


def _model_type(source: str | pathlib.Path) -> str | None:
    config_path = pathlib.Path(source) / "config.json"
    if not config_path.is_file():
        return None
    return json.loads(config_path.read_text(encoding="utf-8")).get("model_type")


def source_draft_layers(source: str | pathlib.Path) -> int:
    """How many multi-token-prediction layers the source carries
    (``model.mtp.layers.<n>.*`` in MiMo-V2, ``mtp.<n>.*`` in DeepSeek-V4)."""
    weight_map, _ = read_index(source)
    draft = V4_DRAFT_LAYER if _model_type(source) == V4_MODEL_TYPE else DRAFT_LAYER
    return len({match.group(1) for name in weight_map if (match := draft.match(name))})


@dataclass(frozen=True)
class SourceLayer:
    """One MoE layer as the source's ``config.json`` names it (the fields
    ``reap.legwork.prune.resolve_kept_plan`` reads)."""

    index: int
    num_experts: int
    kind: str = "moe"


def source_moe_layers(config: Mapping[str, Any]) -> list[SourceLayer]:
    """The source's MoE layers, each ``n_routed_experts`` wide: every
    DeepSeek-V4 decoder layer (the hash-routed ones marked), or the layers a
    MiMo-V2 ``moe_layer_freq`` marks."""
    experts = int(config["n_routed_experts"])
    if config.get("model_type") == V4_MODEL_TYPE:
        layers = int(config["num_hidden_layers"])
        kinds = config.get("mlp_layer_types")
        if not isinstance(kinds, list):
            hashed = config.get("num_hash_layers", V4_DEFAULT_HASH_LAYERS)
            kinds = [HASH_KIND if index < hashed else "moe" for index in range(layers)]
        if len(kinds) < layers:
            raise ValueError(f"config.json names {len(kinds)} mlp_layer_types for {layers} layers")
        return [SourceLayer(index, experts, kinds[index]) for index in range(layers)]
    freq = config.get("moe_layer_freq")
    if not isinstance(freq, list) or not freq:
        raise ValueError(
            "config.json names no moe_layer_freq list; "
            "--slice-source reads MiMo-V2 and DeepSeek-V4 checkpoints"
        )
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
    if config.get("model_type") == V4_MODEL_TYPE:
        try:
            _slice_deepseek_v4(reader, writer, layers, renumber, order, report, progress)
            writer.close(metadata={k: v for k, v in reader.metadata.items() if k != "total_size"})
        finally:
            reader.close()
        report.total_size = writer.total_size
        config["n_routed_experts"] = keep
        config["num_nextn_predict_layers"] = 0
        _write_config(out, config)
        copy_files(source, out)
        if progress is not None:
            progress(100.0)
        return report
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
    _write_config(out, config)
    copy_files(source, out, skip_dirs=DRAFT_DIRS if drop_drafts else ())
    if progress is not None:
        progress(100.0)
    return report


def _write_config(out: pathlib.Path, config: Mapping[str, Any]) -> None:
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slice_deepseek_v4(
    reader: TensorReader,
    writer: ShardWriter,
    layers: Sequence[SourceLayer],
    renumber: Mapping[int, Mapping[int, int]],
    order: Mapping[int, torch.Tensor],
    report: SliceReport,
    progress: Callable[[float], None] | None,
) -> None:
    """DeepSeek-V4's arm of ``slice_source``, over DeepSeek's own names."""
    # prune.py imports this module, so its remap is read at call time.
    from reap.legwork.prune import remap_hash_table

    hashed = {layer.index for layer in layers if layer.kind == HASH_KIND}
    tables = {int(match.group(1)) for name in reader.names() if (match := V4_HASH_TABLE.match(name))}
    if tables != hashed:
        stray = sorted(tables ^ hashed)
        raise ValueError(
            f"layer {stray[0]}: config.json and the source's hash tables disagree on which layers "
            f"route by table (the config names {sorted(hashed)}, the tensors {sorted(tables)})"
        )
    names = reader.names()
    for position, name in enumerate(names):
        if progress is not None:
            progress(95.0 * position / len(names))
        if name.startswith(V4_DRAFT_PREFIX):
            report.dropped += 1
            continue
        expert = V4_EXPERT.match(name)
        if expert is not None and int(expert.group(1)) in renumber:
            new = renumber[int(expert.group(1))].get(int(expert.group(2)))
            if new is None:
                report.dropped += 1
            else:
                writer.add(f"layers.{expert.group(1)}.ffn.experts.{new}.{expert.group(3)}", reader.get(name))
                report.copied += 1
            continue
        router = V4_ROUTER.match(name)
        if router is not None and int(router.group(1)) in order:
            writer.add(name, reader.get(name).index_select(0, order[int(router.group(1))]))
            report.sliced += 1
            continue
        table = V4_HASH_TABLE.match(name)
        if table is not None and int(table.group(1)) in order:
            index = int(table.group(1))
            weight = reader.get(f"layers.{index}.ffn.gate.weight")
            writer.add(name, remap_hash_table(reader.get(name), order[index], weight))
            report.sliced += 1
            continue
        writer.add(name, reader.get(name))
        report.copied += 1
