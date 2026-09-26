# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Collect router stats over a calibration set (the lane's first stage).

CLI::

    python -m reap.legwork.collect --model <dir> --calib <set.jsonl> \
        --out <router-stats.pt> [--map <expert-map.json.gz>] [--seq-len 2048] \
        [--max-samples N] [--layers 3,4,5] [--device cpu|cuda|auto] \
        [--dtype auto|bfloat16|float32] [--trust-remote-code]

``--trust-remote-code`` loads the checkpoint's own model code: a MiMo-V2
working copy (``reap-materialize``), whose ``modeling_mimo_v2.py`` rides the
snapshot. The caller passes it only for a snapshot at a pinned revision;
``reap.legwork.compat`` adapts that code's mask calls to the installed
transformers.

The calibration set is JSON Lines; each row is one sample in one of four
shapes: ``{"text": "..."}``, ``{"messages": [{"role", "content"}, ...]}``
(rendered with the tokenizer's chat template), ``{"prompt": "...",
"completion": "..."}`` (concatenated) or ``{"input_ids": [...]}``
(pre-tokenized; needs no tokenizer). Samples run one at a time,
truncated to ``--seq-len``, so no padding token ever enters the stats.

A ``messages`` row may carry ``tools``: OpenAI-style function schemas
(``{"type": "function", "function": {"name", "description",
"parameters"}}``) the template receives as ``tools=``. Assistant turns may
carry ``tool_calls`` (``{"type": "function", "function": {"name",
"arguments": {...}}}``) and tool results arrive as ``{"role": "tool"}``
turns. The rendered text keeps the template's own special tokens (no
second BOS). When the template raises, or the tokenizer has none, the
row is rendered as plain text instead (role headers, the contents, tool
calls and the tools list as JSON) and counted:
``calibration.template_fallbacks`` in the stats, ``template_fallbacks`` in
the result.

Any row may name its ``source`` (a string id, default ``"unlabeled"``) and
whether it is ``private`` (default false; the worker marks a customer's
own dataset rows private). Both reach the observer with the sample
(``RouterStatsObserver.begin_sample``); the stats and the result list
every source's rows and tokens. A private row feeds the saliency terms
(pooled and per source) but never the top-token sketch or the exemplars.

``--map`` also writes the expert map (``reap.legwork.expert_map``, gzip
JSON): every observed expert's rank, saliency terms, per-source split,
``protected`` flag, top tokens and exemplars, decoded with the model's
tokenizer (loaded before the calibration starts, so a missing one fails
fast).

Prints ``STAGE_PROGRESS <pct>`` lines and a final ``REAP_RESULT {json}``:
``out``, ``samples``, ``tokens``, ``layers``, ``model_type``,
``template_fallbacks``, ``sources`` (``[{id, private, rows, tokens}]``) and
``map`` (``{path, sha256, bytes}`` of the written map, ``null`` without
``--map``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Callable, Iterator

import torch

from reap.legwork.compat import patch_remote_code
from reap.legwork.expert_map import build_expert_map, write_expert_map
from reap.legwork.observer import DEFAULT_SOURCE, RouterStatsObserver, save_router_stats
from reap.legwork.progress import parse_layer_list, stage_progress

#: How many template fallbacks are named on stderr; the rest only count.
FALLBACK_WARNINGS = 5


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


def row_source(row: dict[str, Any], where: str) -> tuple[str, bool]:
    """A row's ``(source, private)``: ``"unlabeled"`` and false unless the
    row says otherwise (``null`` reads as absent)."""
    source = row.get("source")
    if source is None:
        source = DEFAULT_SOURCE
    elif not isinstance(source, str) or not source.strip():
        raise ValueError(f"{where}: source is a non-empty string id")
    private = row.get("private")
    if private is None:
        private = False
    elif not isinstance(private, bool):
        raise ValueError(f"{where}: private is true or false")
    return source, private


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            else:
                parts.append(_json(part))
        return "\n".join(parts)
    return _json(content)


def render_plain(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
    """The template-free rendering of a ``messages`` row: the tools list as
    JSON, then one block per turn — a ``<role>:`` header, the content, the
    turn's tool calls as JSON."""
    blocks: list[str] = []
    if tools:
        blocks.append("tools:\n" + _json(tools))
    for message in messages:
        lines = [f"{message.get('role')}:"]
        content = _content_text(message.get("content"))
        if content:
            lines.append(content)
        if message.get("tool_calls"):
            lines.append(_json(message["tool_calls"]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class SampleEncoder:
    """Turn calibration rows into token-id lists, loading the tokenizer on
    the first row that needs one; counts the ``messages`` rows the chat
    template could not render (``template_fallbacks``)."""

    def __init__(self, model_dir: str, seq_len: int):
        self.model_dir = model_dir
        self.seq_len = seq_len
        self._tokenizer = None
        self.template_fallbacks = 0

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return self._tokenizer

    def render_messages(self, row: dict[str, Any], where: str) -> tuple[str, bool]:
        """``(text, templated)``: the chat template's rendering, or the plain
        one (counted) when the template raises or the tokenizer has none."""
        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or not messages
            or not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages)
        ):
            raise ValueError(f"{where}: messages is a non-empty list of {{role, content}} objects")
        tools = row.get("tools")
        if tools is not None and (
            not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)
        ):
            raise ValueError(f"{where}: tools is a list of function schemas")
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            kwargs: dict[str, Any] = {"tools": tools} if tools else {}
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, **kwargs
                )
                return text, True
            except Exception as error:  # a template raises whatever its author wrote
                reason = f"the chat template failed ({type(error).__name__}: {str(error)[:160]})"
        else:
            reason = "the tokenizer has no chat template"
        self.template_fallbacks += 1
        if self.template_fallbacks <= FALLBACK_WARNINGS:
            print(
                f"reap-collect: {where}: {reason}; rendered the row as plain text",
                file=sys.stderr,
                flush=True,
            )
        return render_plain(messages, tools), False

    def encode(self, row: dict[str, Any], where: str) -> list[int]:
        if "input_ids" in row:
            ids = row["input_ids"]
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise ValueError(f"{where}: input_ids must be a list of integers")
            return ids[: self.seq_len]
        # A template's rendering already carries the model's special tokens
        # (transformers' own tokenize=True path adds none either).
        add_special_tokens = True
        if "messages" in row:
            text, templated = self.render_messages(row, where)
            add_special_tokens = not templated
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
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.seq_len, add_special_tokens=add_special_tokens
        )
        return list(encoded["input_ids"])


def _load_model(path: str, dtype: str, device: str, trust_remote_code: bool = False):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device != "cpu":
        kwargs["device_map"] = device
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    if trust_remote_code:
        patch_remote_code(model)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reap-collect", description="Record REAP router stats over a calibration set."
    )
    parser.add_argument("--model", required=True, help="the source checkpoint directory")
    parser.add_argument("--calib", required=True, help="the calibration set (JSON Lines)")
    parser.add_argument("--out", required=True, help="where the router-stats file goes")
    parser.add_argument("--map", default=None, help="also write the expert map here (gzip JSON)")
    parser.add_argument("--seq-len", type=int, default=2048, help="tokens per sample after truncation")
    parser.add_argument("--max-samples", type=int, default=None, help="stop after this many rows")
    parser.add_argument("--layers", default="", help="comma-separated layer indices to observe (default: every MoE layer)")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or auto (accelerate device_map)")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="load the checkpoint's own model code (MiMo-V2's working copy); the caller "
        "passes it only for a snapshot at a pinned revision",
    )
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
    encoder = SampleEncoder(args.model, args.seq_len)
    map_tokenizer = None
    if args.map:
        try:
            map_tokenizer = encoder.tokenizer
        except Exception as error:
            raise SystemExit(
                f"reap-collect: --map decodes token ids with the model's tokenizer, and "
                f"{args.model} holds none that loads ({type(error).__name__})"
            ) from error
    model = _load_model(args.model, args.dtype, args.device, args.trust_remote_code)
    device = next(model.parameters()).device
    observer = RouterStatsObserver(model, layers or None)
    progress(5)
    tokens = 0
    seen = 0
    try:
        with torch.no_grad():
            for number, row in iter_rows(calib):
                if args.max_samples is not None and seen >= args.max_samples:
                    break
                where = f"{calib}:{number}"
                source, private = row_source(row, where)
                ids = encoder.encode(row, where)
                if not ids:
                    continue
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                observer.begin_sample(ids, source=source, private=private)
                model(input_ids=input_ids, use_cache=False)
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
        "template_fallbacks": encoder.template_fallbacks,
    }
    save_router_stats(state, args.out)
    map_info = None
    if args.map:
        try:
            map_info = write_expert_map(build_expert_map(state, map_tokenizer), args.map)
        except ValueError as error:
            raise SystemExit(
                f"reap-collect: the expert map was not written ({error}); "
                f"the router stats are at {args.out}"
            ) from error
    progress(100)
    result = {
        "out": str(args.out),
        "samples": seen,
        "tokens": tokens,
        "layers": sorted(state["layers"].keys()),
        "model_type": state["model_type"],
        "template_fallbacks": encoder.template_fallbacks,
        "sources": state["sources"],
        "map": map_info,
    }
    print("REAP_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
