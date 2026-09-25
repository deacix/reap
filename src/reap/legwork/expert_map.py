# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The expert map: every observed expert's saliency, where it came from and
a readable sample of what it serves, written as gzip JSON by
``reap-collect --map``.

Schema (camelCase, ``version`` 1; floats keep full float64 precision)::

    {"version": 1,
     "model": {"modelType", "modelClass", "experts", "topK"},
     "calibration": {"sha256", "samples", "tokens", "seqLen", "templateFallbacks"},
     "protectNormRatio": 20,
     "sources": [{"id", "kind": "corpus" | "dataset", "rows", "tokens"}],
     "scopes": [
       {"scope": "L<layer>", "layer", "kind": "moe" | "hash_moe", "tokens",
        "experts": [
          {"id", "rank", "count", "weightSum", "reapSum", "maxNorm", "protected",
           "bySource": {"<source id>": [count, reapSum]},
           "topTokens": [["<token>", count]],
           "exemplars": [{"source", "before", "token", "after"}]}]}]}

Every expert of every observed MoE layer appears, sorted by id. ``rank`` is
the expert's 1-based place in ``select_kept``'s order over the pooled REAP
score (``reapSum / max(count, 1)``, ties to the lower id), so the experts
kept at width ``W`` are exactly those with ``rank <= W``. ``protected``
flags an outlier: ``maxNorm >= PROTECT_NORM_RATIO`` times the median
``maxNorm`` over the scope's experts with ``count > 0``. ``bySource``
omits sources that never reached the expert. ``topTokens`` (count
descending, then token id) and ``exemplars`` (best first; ``before`` and
``after`` are the 24 tokens around the routed ``token``) are decoded with
the model's tokenizer and read only non-private rows: a private source (a
customer's own dataset, ``kind: "dataset"``) feeds the counts and
``bySource`` only. The pruned build carries no draft blocks, so every
scope is a backbone layer (``L<n>``).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pathlib
import statistics
from typing import Any, Sequence

import torch

from reap.legwork.prune import saliency, saliency_order

MAP_VERSION = 1
#: An expert is ``protected`` when its max output norm is at least this
#: many times the median max norm of its scope's routed experts.
PROTECT_NORM_RATIO = 20


def expert_ranks(scores: torch.Tensor) -> list[int]:
    """Each expert's 1-based place in ``select_kept``'s order."""
    ranks = [0] * int(scores.numel())
    for place, expert in enumerate(saliency_order(scores).tolist(), start=1):
        ranks[expert] = place
    return ranks


def protected_experts(
    max_norm: torch.Tensor, count: torch.Tensor, ratio: float = PROTECT_NORM_RATIO
) -> list[bool]:
    """``max_norm >= ratio * median(max_norm of the experts with count > 0)``
    per expert (the median of an even count is the mean of the middle two);
    nothing is protected when no expert was routed or that median is 0."""
    norms = [float(v) for v in max_norm.tolist()]
    routed = [norm for norm, hits in zip(norms, count.tolist()) if hits > 0]
    if not routed:
        return [False] * len(norms)
    median = statistics.median(routed)
    if median <= 0:
        return [False] * len(norms)
    return [norm >= ratio * median for norm in norms]


class _Decoder:
    """Token ids to text with the model's tokenizer; single ids cached."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._single: dict[int, str] = {}

    def text(self, ids: Sequence[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def token(self, token_id: int) -> str:
        text = self._single.get(token_id)
        if text is None:
            text = self._single[token_id] = self.text([token_id])
        return text


def _exemplar(entry: dict[str, Any], decode: _Decoder) -> dict[str, str]:
    window, offset = entry["window"], int(entry["offset"])
    return {
        "source": entry["source"],
        "before": decode.text(window[:offset]),
        "token": decode.token(int(window[offset])),
        "after": decode.text(window[offset + 1 :]),
    }


def _scope(index: int, layer: dict[str, Any], decode: _Decoder) -> dict[str, Any]:
    n = int(layer["num_experts"])
    count = layer["count"]
    ranks = expert_ranks(saliency(layer, "reap"))
    protected = protected_experts(layer["max_norm"], count)
    counts = [int(v) for v in count.tolist()]
    weight_sums = layer["weight_sum"].double().tolist()
    reap_sums = layer["reap_sum"].double().tolist()
    max_norms = layer["max_norm"].double().tolist()
    by_source = {
        source: (terms["count"].tolist(), terms["reap_sum"].double().tolist())
        for source, terms in sorted((layer.get("by_source") or {}).items())
    }
    top_tokens = layer.get("top_tokens") or [[] for _ in range(n)]
    exemplars = layer.get("exemplars") or [[] for _ in range(n)]
    experts = []
    for e in range(n):
        experts.append(
            {
                "id": e,
                "rank": ranks[e],
                "count": counts[e],
                "weightSum": weight_sums[e],
                "reapSum": reap_sums[e],
                "maxNorm": max_norms[e],
                "protected": protected[e],
                "bySource": {
                    source: [int(c[e]), r[e]] for source, (c, r) in by_source.items() if c[e] > 0
                },
                "topTokens": [[decode.token(int(t)), int(k)] for t, k in top_tokens[e]],
                "exemplars": [_exemplar(entry, decode) for entry in exemplars[e]],
            }
        )
    return {
        "scope": f"L{index}",
        "layer": int(index),
        "kind": layer["kind"],
        "tokens": int(layer["tokens"]),
        "experts": experts,
    }


def build_expert_map(stats: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """The expert map of a router-stats state (``RouterStatsObserver.state``
    plus collect's ``calibration``); ``tokenizer`` decodes the token ids."""
    decode = _Decoder(tokenizer)
    layers = stats.get("layers") or {}
    calibration = stats.get("calibration") or {}
    experts = stats.get("experts")
    if experts is None:
        experts = max((int(layer["num_experts"]) for layer in layers.values()), default=0)
    return {
        "version": MAP_VERSION,
        "model": {
            "modelType": stats.get("model_type"),
            "modelClass": stats.get("model_class"),
            "experts": int(experts),
            "topK": stats.get("top_k"),
        },
        "calibration": {
            "sha256": calibration.get("sha256"),
            "samples": calibration.get("samples", stats.get("samples", 0)),
            "tokens": calibration.get("tokens", 0),
            "seqLen": calibration.get("seq_len"),
            "templateFallbacks": calibration.get("template_fallbacks", 0),
        },
        "protectNormRatio": PROTECT_NORM_RATIO,
        "sources": [
            {
                "id": source["id"],
                "kind": "dataset" if source["private"] else "corpus",
                "rows": int(source["rows"]),
                "tokens": int(source["tokens"]),
            }
            for source in stats.get("sources") or []
        ],
        "scopes": [_scope(index, layers[index], decode) for index in sorted(layers)],
    }


def write_expert_map(expert_map: dict[str, Any], path: str | pathlib.Path) -> dict[str, Any]:
    """Write the map as gzip JSON (no timestamp or name in the gzip header,
    so one map is one byte string); returns ``{path, sha256, bytes}``. A
    non-finite float refuses (``ValueError``): JSON has none."""
    path = pathlib.Path(path)
    payload = json.dumps(
        expert_map, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(payload)
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def read_expert_map(path: str | pathlib.Path) -> dict[str, Any]:
    with gzip.open(pathlib.Path(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)
