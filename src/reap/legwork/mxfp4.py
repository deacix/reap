# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Write an edited MiMo-V2 tree as the native MXFP4 build SGLang serves.

The abliterate lane edits the BF16 working copy (``reap-materialize``'s
layout: split attention, BF16, no draft layers) and the build restores the
reference's storage for the serving lane:

- every edited residual writer re-encoded to the storage the source keeps
  for it — each routed expert's down projection to MXFP4 (E2M1 + E8M0,
  block 32), the dense down projection to FP8 blocks, the attention
  output projections carried as BF16 (``reap.legwork.quant``);
- every other tensor byte-identical from the source: the fused ``qkv_proj``
  in its TP chunks, the unedited expert projections, the router, the
  embeddings, the norms, the towers and the draft layers the working copy
  never carried;
- ``config.json``: the source's, with a ``reap_mxfp4`` record; the shard
  index keeps the source's metadata (``save_format: mxfp4``, ``tp_size``).

Only the edited tree's edited tensors and the source's index, tensor
headers and config are read, and the build streams tensor by tensor on
the CPU.

CLI::

    python -m reap.legwork.mxfp4 --model <edited> --source <reference> --out <dir> [--shard-gib 4]

Prints ``STAGE_PROGRESS <pct>`` lines and a final ``REAP_RESULT {json}``:
``out``, ``tensors``, ``quantized``, ``bytes`` and ``mxfp4_block_size``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass
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
from reap.legwork.quant import MXFP4_BLOCK, quantize_fp8_blocks, quantize_mxfp4

FP8_SCALE = ".weight_scale_inv"
MXFP4_SCALE = ".weight_scale"
WEIGHT = ".weight"

#: The residual writers the ablation edits, by native tensor name: every
#: attention output projection, the dense down projection, and every routed
#: (and shared) expert's down projection. Everything else rides verbatim.
_EDITED_PATTERNS = (
    re.compile(r"\.self_attn\.o_proj\.weight$"),
    re.compile(r"\.mlp\.down_proj\.weight$"),
    re.compile(r"\.mlp\.experts\.\d+\.down_proj\.weight$"),
    re.compile(r"\.shared_expert(?:s)?\.down_proj\.weight$"),
)


@dataclass
class Mxfp4BuildReport:
    out: str
    tensors: int = 0
    quantized: int = 0
    bytes: int = 0
    mxfp4_block_size: int = MXFP4_BLOCK


def is_edited_target(name: str) -> bool:
    """Whether a native tensor name is a residual writer the ablation edits."""
    return any(pattern.search(name) for pattern in _EDITED_PATTERNS)


def _layout(quantization: Any, source: pathlib.Path) -> tuple[int, int]:
    """The ``(fp8 block, mxfp4 block)`` the source stores, or a refusal."""
    blocks = quantization.get("weight_block_size") if isinstance(quantization, dict) else None
    square = (
        isinstance(blocks, list)
        and len(blocks) == 2
        and all(isinstance(side, int) and side > 0 for side in blocks)
        and blocks[0] == blocks[1]
    )
    mxfp4 = quantization.get("mxfp4_block_size") if isinstance(quantization, dict) else None
    if (
        not isinstance(quantization, dict)
        or quantization.get("quant_method") != "fp8"
        or quantization.get("store_dtype") != "mxfp4"
        or not square
        or not isinstance(mxfp4, int)
        or mxfp4 <= 0
    ):
        raise ValueError(
            f"{source} stores no MiMo-V2 MXFP4/FP8 layout (quantization_config "
            f"{quantization!r}): the build takes its formats from an MXFP4 reference"
        )
    return int(blocks[0]), int(mxfp4)


def build_mxfp4(
    model: str | pathlib.Path,
    source: str | pathlib.Path,
    out: str | pathlib.Path,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    progress: Callable[[float], None] | None = None,
) -> Mxfp4BuildReport:
    model, source, out = pathlib.Path(model), pathlib.Path(source), pathlib.Path(out)
    edited_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if edited_config.get("quantization_config"):
        raise ValueError(
            f"{model} is already quantized: the build starts from the BF16 tree "
            "the working copy edits"
        )
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    fp8_block, mxfp4_block = _layout(source_config.get("quantization_config"), source)
    edited = TensorReader(model)
    native = TensorReader(source)
    _, source_meta = read_index(source)
    report = Mxfp4BuildReport(out=str(out), mxfp4_block_size=mxfp4_block)
    writer = ShardWriter(out, shard_bytes)
    weights = [
        name
        for name in native.names()
        if not name.endswith(FP8_SCALE) and not name.endswith(MXFP4_SCALE)
    ]
    ordered = sorted(weights, key=lambda name: (native.weight_map[name], name))
    reencoded: set[str] = set()
    total = len(ordered)
    try:
        for position, target in enumerate(ordered):
            if not is_edited_target(target):
                writer.add(target, native.get(target))
            else:
                if target not in edited:
                    raise ValueError(f"the edited tree has no tensor {target}")
                tensor = edited.get(target).float()
                stem = target[: -len(WEIGHT)] if target.endswith(WEIGHT) else None
                mxfp4_scale = stem + MXFP4_SCALE if stem is not None else None
                fp8_scale = stem + FP8_SCALE if stem is not None else None
                if mxfp4_scale is not None and mxfp4_scale in native:
                    packed, scales = quantize_mxfp4(tensor, mxfp4_block)
                    writer.add(target, packed)
                    writer.add(mxfp4_scale, scales)
                    reencoded.add(stem)
                    report.quantized += 1
                elif fp8_scale is not None and fp8_scale in native:
                    fp8, scale_inv = quantize_fp8_blocks(tensor, fp8_block)
                    writer.add(target, fp8)
                    writer.add(fp8_scale, scale_inv)
                    reencoded.add(stem)
                    report.quantized += 1
                else:
                    writer.add(target, tensor.to(native.get(target).dtype))
            if progress is not None:
                progress(95.0 * (position + 1) / total)
        for name in native.names():
            suffix = MXFP4_SCALE if name.endswith(MXFP4_SCALE) else FP8_SCALE
            if not name.endswith(suffix):
                continue
            if name[: -len(suffix)] not in reencoded:
                writer.add(name, native.get(name))
        writer.close(metadata={k: v for k, v in source_meta.items() if k != "total_size"})
    finally:
        edited.close()
        native.close()

    report.tensors = len(writer.weight_map)
    report.bytes = writer.total_size
    built = dict(source_config)
    built["reap_mxfp4"] = {"quantized": report.quantized, "source": str(source)}
    (out / "config.json").write_text(
        json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copy_files(source, out, include_dirs=True)
    if progress is not None:
        progress(100.0)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reap-mxfp4",
        description="Write an edited MiMo-V2 tree as the native MXFP4 build SGLang serves.",
    )
    parser.add_argument(
        "--model", required=True, help="the edited BF16 tree (the working copy, edited)"
    )
    parser.add_argument(
        "--source", required=True, help="the native reference the tree was edited from"
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
        report = build_mxfp4(
            args.model,
            args.source,
            args.out,
            shard_bytes=int(args.shard_gib * (1 << 30)),
            progress=progress,
        )
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"reap-mxfp4: {error}") from error
    print("REAP_RESULT " + json.dumps(asdict(report)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
