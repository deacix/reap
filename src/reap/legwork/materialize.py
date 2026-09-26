# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Build the BF16 working copy of a MiMo-V2 checkpoint (the prune lane's fetch).

The Hub checkpoint cannot run on its own model code: ``modeling_mimo_v2.py``
has no FP8 or MXFP4 handling and reads a fused ``qkv_proj`` as one
contiguous ``[Q|K|V]``, while the checkpoint stores it in TP chunks. The
working copy is what that code runs:

- every FP8 block weight (``weight_scale_inv``) and MXFP4 expert
  (``weight_scale``) dequantized to BF16 (``reap.legwork.quant``);
- each fused ``qkv_proj`` split into ``q_proj`` / ``k_proj`` / ``v_proj``
  from its ``metadata.tp_size`` chunks (``reap.legwork.qkv``);
- ``config.json`` with ``attention_projection_layout: "split"``, no
  ``quantization_config``, ``dtype: bfloat16`` and a ``legwork_working_copy``
  record;
- the multi-token-prediction draft layers (``model.mtp.*``) left out: the
  model code never builds them and the router statistics never reach them;
- the model code, the tokenizer and every other top-level file copied (the
  subdirectories, such as the DFlash drafter, are not).

Only ``reap-collect`` reads the working copy. The pruned build is written
by slicing the source itself (``reap-prune --slice-source``), so it keeps
the reference's own precision and the source's layout.

CLI::

    python -m reap.legwork.materialize --source <dir> --out <dir> [--shard-gib 4]

Prints ``STAGE_PROGRESS <pct>`` lines and a final ``REAP_RESULT {json}``:
``out``, ``tensors``, ``bytes``, ``dequantized``, ``split_qkv``,
``dropped`` and ``tp_size``.
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

from reap.legwork.checkpoint import DEFAULT_SHARD_BYTES, ShardWriter, TensorReader, copy_files
from reap.legwork.progress import stage_progress
from reap.legwork.qkv import layer_geometry, split_fused_qkv
from reap.legwork.quant import dequantize_fp8_blocks, dequantize_mxfp4

FP8_SCALE = ".weight_scale_inv"
MXFP4_SCALE = ".weight_scale"
FUSED_QKV = re.compile(r"^(model\.layers\.(\d+)\.self_attn\.)qkv_proj\.weight$")
#: Tensors the working copy leaves out: the draft layers.
DROPPED_PREFIXES = ("model.mtp.",)
WORKING_DTYPE = torch.bfloat16


@dataclass
class MaterializeReport:
    out: str
    tensors: int = 0
    bytes: int = 0
    dequantized: int = 0
    split_qkv: int = 0
    dropped: int = 0
    tp_size: int = 1


def working_copy_config(config: dict[str, Any], source: str, tp_size: int) -> dict[str, Any]:
    """The working copy's ``config.json``: split attention, no quantization."""
    out = dict(config)
    out["attention_projection_layout"] = "split"
    out.pop("quantization_config", None)
    out["dtype"] = "bfloat16"
    out.pop("torch_dtype", None)
    out["legwork_working_copy"] = {
        "source": source,
        "tp_size": tp_size,
        "dropped_prefixes": list(DROPPED_PREFIXES),
    }
    return out


def materialize(
    source: str | pathlib.Path,
    out: str | pathlib.Path,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    progress: Callable[[float], None] | None = None,
) -> MaterializeReport:
    source, out = pathlib.Path(source), pathlib.Path(out)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    reader = TensorReader(source)
    fused = config.get("attention_projection_layout") == "fused_qkv"
    tp_size = int(reader.metadata.get("tp_size") or 1)
    report = MaterializeReport(out=str(out), tp_size=tp_size)
    writer = ShardWriter(out, shard_bytes)
    names = [
        name
        for name in reader.names()
        if not name.endswith(FP8_SCALE) and not name.endswith(MXFP4_SCALE)
    ]
    try:
        for position, name in enumerate(names):
            if name.startswith(DROPPED_PREFIXES):
                report.dropped += 1
                continue
            tensor = reader.get(name)
            stem = name[: -len(".weight")] if name.endswith(".weight") else None
            match = FUSED_QKV.match(name) if fused else None
            if match is not None:
                geometry = layer_geometry(config, int(match.group(2)), tp_size)
                q, k, v = split_fused_qkv(tensor, reader.get(stem + FP8_SCALE), geometry)
                for projection, dense in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                    writer.add(f"{match.group(1)}{projection}.weight", dense.to(WORKING_DTYPE))
                report.split_qkv += 1
                report.dequantized += 1
            elif tensor.dtype == torch.float8_e4m3fn:
                if stem is None or stem + FP8_SCALE not in reader:
                    raise ValueError(f"{name} is FP8 and has no {FP8_SCALE} beside it")
                dense = dequantize_fp8_blocks(tensor, reader.get(stem + FP8_SCALE))
                writer.add(name, dense.to(WORKING_DTYPE))
                report.dequantized += 1
            elif tensor.dtype == torch.uint8 and stem is not None and stem + MXFP4_SCALE in reader:
                dense = dequantize_mxfp4(tensor, reader.get(stem + MXFP4_SCALE))
                writer.add(name, dense.to(WORKING_DTYPE))
                report.dequantized += 1
            else:
                writer.add(name, tensor)
            if progress is not None:
                progress(95.0 * (position + 1) / len(names))
        writer.close()
    finally:
        reader.close()
    report.tensors = len(writer.weight_map)
    report.bytes = writer.total_size
    (out / "config.json").write_text(
        json.dumps(working_copy_config(config, str(source), tp_size), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copy_files(source, out, include_dirs=False)
    if progress is not None:
        progress(100.0)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reap-materialize",
        description="Dequantize a MiMo-V2 checkpoint into the BF16 working copy its model code runs.",
    )
    parser.add_argument("--source", required=True, help="the Hub checkpoint directory")
    parser.add_argument("--out", required=True, help="where the working copy goes")
    parser.add_argument(
        "--shard-gib", type=float, default=DEFAULT_SHARD_BYTES / (1 << 30), help="shard size bound"
    )
    return parser


def main(argv: list[str] | None = None, progress: Callable[[float], None] = stage_progress) -> int:
    args = build_parser().parse_args(argv)
    progress(0)
    try:
        report = materialize(
            args.source, args.out, shard_bytes=int(args.shard_gib * (1 << 30)), progress=progress
        )
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"reap-materialize: {error}") from error
    print("REAP_RESULT " + json.dumps(asdict(report)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
