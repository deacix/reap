# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Write a dequantized DeepSeek-V4 prune as the FP8 build vLLM serves.

A prune on a ROCm host loads the FP8 reference in BF16 and saves the pruned
tree through transformers (``reap-prune --dequantize``). vLLM 0.30.0 serves a
DeepSeek-V4 checkpoint only in DeepSeek's own storage, and transformers 5.17
writes neither half of it:

- names: its save reverses its load renames by swapping each rule's patterns,
  so ``embed.weight``, ``hc_head_fn`` and ``layers.N.attn.kv_norm.weight`` come
  back as ``model.embed_tokens.weight``, ``model.hc_head.hc_fn`` and
  ``model.layers.N.attn.norm.weight``, and every other tensor gains ``model.``;
- the config: it writes its own words (``mlp_layer_types`` with ``hash_moe``,
  ``compress_rates``, ``rope_parameters``) and drops DeepSeek's
  (``num_hash_layers``, ``compress_ratios``, ``rope_scaling``); vLLM's
  ``deepseek_v4`` config refuses the first and its model reads the second;
- the weights: vLLM's V4 quantization takes ``quant_method: fp8`` with
  ``expert_dtype`` ``fp4`` or ``fp8`` and nothing else.

The build is the DeepSeek-V4-Flash-Base layout (``expert_dtype: "fp8"``):

- every tensor under the reference's own name, checked both ways (each written
  name is one the reference has; each reference tensor but the ``.scale``
  grids, the draft layers and the pruned experts is written);
- each weight the reference stores beside a ``.scale`` as ``F8_E4M3`` tiles of
  the reference's ``weight_block_size`` with an ``F32`` ``.scale``
  (``reap.legwork.quant``); every other tensor in the reference's own dtype;
- ``config.json``: the reference's, with the tree's ``n_routed_experts``,
  ``num_nextn_predict_layers: 0`` (no draft layers are carried),
  ``expert_dtype: "fp8"`` and the tree's ``reap_pruning`` record.

Only the reference's index, tensor headers and config are read, and the tree
is streamed tensor by tensor on the CPU.

CLI::

    python -m reap.legwork.fp8 --model <pruned> --source <reference> --out <dir> [--shard-gib 4]

Prints ``STAGE_PROGRESS <pct>`` lines and a final ``REAP_RESULT {json}``:
``out``, ``tensors``, ``quantized``, ``bytes`` and ``weight_block_size``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from reap.legwork.checkpoint import (
    DEFAULT_SHARD_BYTES,
    ShardWriter,
    TensorReader,
    copy_files,
    read_index,
)
from reap.legwork.progress import stage_progress
from reap.legwork.quant import quantize_fp8_blocks

SCALE = ".scale"
WEIGHT = ".weight"
DRAFT_PREFIX = "mtp."
EXPERT = re.compile(r"^layers\.\d+\.ffn\.experts\.(\d+)\.")

#: transformers 5.17's saved names for the DeepSeek-V4 tensors its save does
#: not rename back, in the order they are tried; the rest only gain ``model.``.
_DEEPSEEK_NAMES = (
    (re.compile(r"^model\.embed_tokens\.weight$"), "embed.weight"),
    (re.compile(r"^model\.hc_head\.hc_(fn|base|scale)$"), r"hc_head_\1"),
    (re.compile(r"^model\.(layers\.\d+\.attn)\.norm\.weight$"), r"\1.kv_norm.weight"),
    (re.compile(r"^model\."), ""),
)

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass
class Fp8BuildReport:
    out: str
    tensors: int = 0
    quantized: int = 0
    bytes: int = 0
    weight_block_size: list[int] = field(default_factory=list)


def deepseek_name(name: str) -> str:
    """A tensor name as DeepSeek's checkpoint writes it."""
    for pattern, replacement in _DEEPSEEK_NAMES:
        renamed, hits = pattern.subn(replacement, name)
        if hits:
            return renamed
    return name


def _pruned_away(name: str, keep: int) -> bool:
    expert = EXPERT.match(name)
    return expert is not None and int(expert.group(1)) >= keep


def _reference_dtypes(source: pathlib.Path, weight_map: dict[str, str]) -> dict[str, str]:
    """Every reference tensor's safetensors dtype, read from the shard headers."""
    from safetensors import safe_open

    dtypes: dict[str, str] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(source / shard), framework="pt") as handle:
            names = handle.keys()
            for name in names:
                dtypes[name] = handle.get_slice(name).get_dtype()
    return dtypes


def _as_reference(tensor: torch.Tensor, dtype_name: str, name: str) -> torch.Tensor:
    dtype = _DTYPES.get(dtype_name)
    if dtype is None or tensor.is_floating_point() != dtype.is_floating_point:
        raise ValueError(
            f"{name} is {dtype_name} in the reference and {tensor.dtype} in the pruned tree, "
            "and the reference stores no .scale beside it"
        )
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _block(quantization: dict[str, Any], source: pathlib.Path) -> int:
    block = quantization.get("weight_block_size")
    square = (
        isinstance(block, list)
        and len(block) == 2
        and all(isinstance(side, int) and side > 0 for side in block)
        and block[0] == block[1]
    )
    if quantization.get("quant_method") != "fp8" or not square:
        raise ValueError(
            f"{source} stores no FP8 blocks (quantization_config {quantization!r}): the build "
            "takes its tiles from an FP8 reference with square weight_block_size"
        )
    return int(block[0])


def build_fp8(
    model: str | pathlib.Path,
    source: str | pathlib.Path,
    out: str | pathlib.Path,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    progress: Callable[[float], None] | None = None,
) -> Fp8BuildReport:
    model, source, out = pathlib.Path(model), pathlib.Path(source), pathlib.Path(out)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    reference_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    tree_quantization = config.get("quantization_config")
    if tree_quantization:
        method = (
            tree_quantization.get("quant_method") if isinstance(tree_quantization, dict) else None
        )
        raise ValueError(
            f"{model} is already quantized ({method}): the build starts from the float tree "
            "a dequantized prune writes"
        )
    record = dict(config.get("reap_pruning") or {})
    widths = set(record.get("n_routed_experts_per_layer") or [])
    if record.get("ragged") or len(widths) > 1:
        raise ValueError(
            "the pruned tree keeps layers at different widths (ragged): vLLM builds every "
            "layer from one n_routed_experts"
        )
    block = _block(reference_config.get("quantization_config") or {}, source)
    keep = config.get("n_routed_experts")
    if not isinstance(keep, int) or keep <= 0:
        raise ValueError(f"{model}/config.json names no n_routed_experts")

    reader = TensorReader(model)
    weight_map, _ = read_index(source)
    reference = set(weight_map)
    names: dict[str, str] = {}
    for name in reader.names():
        target = deepseek_name(name)
        if target not in reference:
            raise ValueError(f"the pruned tensor {name} (read as {target}) is not in {source}")
        if target in names:
            raise ValueError(f"{names[target]} and {name} both read as {target}")
        names[target] = name
    expected = {
        name
        for name in reference
        if not name.endswith(SCALE)
        and not name.startswith(DRAFT_PREFIX)
        and not _pruned_away(name, keep)
    }
    missing = sorted(expected - set(names))
    if missing:
        shown = ", ".join(missing[:3])
        raise ValueError(
            f"the pruned tree lacks {len(missing)} of the reference's tensors: {shown}"
        )
    extra = sorted(set(names) - expected)
    if extra:
        shown = ", ".join(extra[:3])
        raise ValueError(f"the pruned tree carries tensors the build cannot place: {shown}")

    dtypes = _reference_dtypes(source, weight_map)
    report = Fp8BuildReport(out=str(out), weight_block_size=[block, block])
    writer = ShardWriter(out, shard_bytes)
    ordered = sorted(names, key=lambda target: (reader.weight_map[names[target]], names[target]))
    try:
        for position, target in enumerate(ordered):
            tensor = reader.get(names[target])
            scale = target[: -len(WEIGHT)] + SCALE if target.endswith(WEIGHT) else None
            if scale is not None and scale in reference:
                if tensor.dim() != 2:
                    raise ValueError(
                        f"{target} has shape {tuple(tensor.shape)}; FP8 tiles take a matrix"
                    )
                fp8, scale_inv = quantize_fp8_blocks(tensor, block)
                writer.add(target, fp8)
                writer.add(scale, scale_inv)
                report.quantized += 1
            else:
                writer.add(target, _as_reference(tensor, dtypes[target], target))
            if progress is not None:
                progress(95.0 * (position + 1) / len(ordered))
        writer.close()
    finally:
        reader.close()

    report.tensors = len(writer.weight_map)
    report.bytes = writer.total_size
    built = dict(reference_config)
    built["n_routed_experts"] = keep
    built["num_nextn_predict_layers"] = 0
    built["expert_dtype"] = "fp8"
    built["reap_pruning"] = {
        **record,
        "fp8_build": {
            "weight_block_size": [block, block],
            "quantized": report.quantized,
            "source": str(source),
        },
    }
    (out / "config.json").write_text(
        json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copy_files(model, out, include_dirs=False)
    if progress is not None:
        progress(100.0)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reap-fp8",
        description="Write a dequantized DeepSeek-V4 prune as the FP8 build vLLM serves.",
    )
    parser.add_argument(
        "--model", required=True, help="the pruned float tree (reap-prune --dequantize)"
    )
    parser.add_argument(
        "--source", required=True, help="the FP8 reference the tree was pruned from"
    )
    parser.add_argument("--out", required=True, help="where the build goes")
    parser.add_argument(
        "--shard-gib", type=float, default=DEFAULT_SHARD_BYTES / (1 << 30), help="shard size bound"
    )
    return parser


def main(argv: list[str] | None = None, progress: Callable[[float], None] = stage_progress) -> int:
    args = build_parser().parse_args(argv)
    progress(0)
    try:
        report = build_fp8(
            args.model,
            args.source,
            args.out,
            shard_bytes=int(args.shard_gib * (1 << 30)),
            progress=progress,
        )
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"reap-fp8: {error}") from error
    print("REAP_RESULT " + json.dumps(asdict(report)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
