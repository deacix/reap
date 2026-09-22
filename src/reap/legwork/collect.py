# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Collect router stats over a calibration set (the lane's first stage).

CLI::

    python -m reap.legwork.collect --model <dir> --calib <set.jsonl> \
        --out <router-stats.pt> [--seq-len 2048] [--max-samples N] \
        [--layers 3,4,5] [--device cpu|cuda|auto] [--dtype auto|bfloat16|float32]

The calibration set is JSON Lines; each row is one sample in one of four
shapes: ``{"text": "..."}``, ``{"messages": [{"role", "content"}, ...]}``
(rendered with the tokenizer's chat template), ``{"prompt": "...",
"completion": "..."}`` (concatenated) or ``{"input_ids": [...]}``
(pre-tokenized; needs no tokenizer). Samples run one at a time,
truncated to ``--seq-len``, so no padding token ever enters the stats.

Prints ``STAGE_PROGRESS <pct>`` lines and a final ``REAP_RESULT {json}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Callable, Iterator

import torch

from reap.legwork.observer import RouterStatsObserver, save_router_stats
from reap.legwork.progress import parse_layer_list, stage_progress


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: pathlib.Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_rows(path: pathlib.Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: not JSON ({error.msg})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: a calibration row is a JSON object")
            yield number, row


class SampleEncoder:
    """Turn calibration rows into token-id lists, loading the tokenizer on
    the first row that needs one."""

    def __init__(self, model_dir: str, seq_len: int):
        self.model_dir = model_dir
        self.seq_len = seq_len
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return self._tokenizer

    def encode(self, row: dict[str, Any], where: str) -> list[int]:
        if "input_ids" in row:
            ids = row["input_ids"]
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise ValueError(f"{where}: input_ids must be a list of integers")
            return ids[: self.seq_len]
        if "messages" in row:
            text = self.tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        elif "text" in row:
            text = row["text"]
        elif "prompt" in row and "completion" in row:
            text = f"{row['prompt']}{row['completion']}"
        else:
            raise ValueError(
                f"{where}: a row carries text, messages, prompt+completion or input_ids"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: the sample text is empty")
        encoded = self.tokenizer(text, truncation=True, max_length=self.seq_len, add_special_tokens=True)
        return list(encoded["input_ids"])


def _load_model(path: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device != "cpu":
        kwargs["device_map"] = device
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reap-collect", description="Record REAP router stats over a calibration set."
    )
    parser.add_argument("--model", required=True, help="the source checkpoint directory")
    parser.add_argument("--calib", required=True, help="the calibration set (JSON Lines)")
    parser.add_argument("--out", required=True, help="where the router-stats file goes")
    parser.add_argument("--seq-len", type=int, default=2048, help="tokens per sample after truncation")
    parser.add_argument("--max-samples", type=int, default=None, help="stop after this many rows")
    parser.add_argument("--layers", default="", help="comma-separated layer indices to observe (default: every MoE layer)")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or auto (accelerate device_map)")
    return parser


def main(argv: list[str] | None = None, progress: Callable[[float], None] = stage_progress) -> int:
    args = build_parser().parse_args(argv)
    calib = pathlib.Path(args.calib)
    if not calib.is_file():
        raise SystemExit(f"reap-collect: calibration set not found: {calib}")
    layers = parse_layer_list(args.layers)
    total = count_rows(calib)
    if args.max_samples is not None:
        total = min(total, max(args.max_samples, 0))
    if total == 0:
        raise SystemExit(f"reap-collect: {calib} holds no calibration rows")
    progress(0)
    model = _load_model(args.model, args.dtype, args.device)
    device = next(model.parameters()).device
    encoder = SampleEncoder(args.model, args.seq_len)
    observer = RouterStatsObserver(model, layers or None)
    progress(5)
    tokens = 0
    seen = 0
    try:
        with torch.no_grad():
            for number, row in iter_rows(calib):
                if args.max_samples is not None and seen >= args.max_samples:
                    break
                ids = encoder.encode(row, f"{calib}:{number}")
                if not ids:
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                model(input_ids=input_ids, use_cache=False)
                observer.note_sample()
                tokens += len(ids)
                seen += 1
                progress(5 + 90.0 * seen / total)
    finally:
        observer.close()
    state = observer.state()
    state["calibration"] = {
        "path": str(calib),
        "sha256": _sha256(calib),
        "samples": seen,
        "tokens": tokens,
        "seq_len": args.seq_len,
    }
    save_router_stats(state, args.out)
    progress(100)
    result = {
        "out": str(args.out),
        "samples": seen,
        "tokens": tokens,
        "layers": sorted(state["layers"].keys()),
        "model_type": state["model_type"],
    }
    print("REAP_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
