# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Write ``fixtures/keep-plan-pooled.json``, the pooled-REAP parity fixture.

The API resolves a keep plan from an expert map with a weighted score,
``sum_s (w_s / T_s) * reapSum_s / sum_s (w_s / T_s) * count_s`` (``T_s`` a
source's tokens, ``w_s`` its weight). With every weight the source's token
share that is the fork's pooled REAP score, so the plan must keep
``select_kept(saliency(pooled), keep)`` in every scope. ``test_kept_plan.py``
checks ``expected`` against the fork's own selection and prunes a tiny V4
with it; the API's keep-plan tests read a byte-identical copy
(deacix/legwork#23171).

Each scope is a seeded draw of every expert's per-source ``[count,
reapSum]`` under top-2 routing (a source's counts sum to twice its tokens,
none above its tokens), redrawn until the scope shows its case. Counts are
integers, reapSums multiples of 1/16 and every ``w_s / T_s`` is exactly
1/1024, so both scores come out as the same doubles in any implementation,
ties included. Run from the repository root::

    python3 tests/legwork/make_keep_plan_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
from typing import Any, Callable

import torch

from reap.legwork.expert_map import expert_ranks
from reap.legwork.prune import saliency, select_kept

FIXTURE = pathlib.Path(__file__).with_name("fixtures") / "keep-plan-pooled.json"
SEED = 23167
EXPERTS = 8
KEEP = 4
TOP_K = 2
#: ``(id, kind, rows, tokens)``. The token counts are powers of two summing
#: to one, so every source's token share over its tokens is exactly 1/1024.
SOURCES = (("code", "corpus", 4, 512), ("chat", "corpus", 2, 256), ("mine", "dataset", 2, 256))
SOURCE_IDS = tuple(source[0] for source in SOURCES)
TOKENS = {source[0]: source[3] for source in SOURCES}
TOTAL_TOKENS = sum(TOKENS.values())
#: The range of a drawn per-source mean ``reapSum / count``, in sixteenths.
MEAN_SIXTEENTHS = (4, 40)
ATTEMPTS = 10_000

#: One expert's ``{source: (count, reapSum)}``; a source that never reached
#: the expert is absent.
Split = dict[str, tuple[int, float]]


def _draw(rng: random.Random, reach: dict[str, list[int]], pinned: dict[int, Split]) -> list[Split] | None:
    """One scope: the ``pinned`` experts' splits as given, then each source's
    remaining routed slots spread over the experts it ``reach``es, each at a
    drawn mean. None when the draw breaks top-2 routing (an expert left
    without a slot, or a count above its source's tokens)."""
    scope: list[Split] = [dict(pinned.get(expert, {})) for expert in range(EXPERTS)]
    for source in SOURCE_IDS:
        tokens = TOKENS[source]
        slots = tokens * TOP_K - sum(split[source][0] for split in pinned.values() if source in split)
        free = reach[source]
        weights = [rng.randint(1, 3) for _ in free]
        total = sum(weights)
        counts = [slots * weight // total for weight in weights]
        by_remainder = sorted(range(len(free)), key=lambda i: (-(slots * weights[i] % total), i))
        for i in by_remainder[: slots - sum(counts)]:
            counts[i] += 1
        for expert, count in zip(free, counts):
            if not 1 <= count <= tokens:
                return None
            scope[expert][source] = (count, count * rng.randint(*MEAN_SIXTEENTHS) / 16)
    return scope


def _pooled(scope: list[Split]) -> dict[str, torch.Tensor]:
    return {
        "count": torch.tensor([sum(c for c, _ in split.values()) for split in scope], dtype=torch.long),
        "reap_sum": torch.tensor(
            [float(sum(r for _, r in split.values())) for split in scope], dtype=torch.float64
        ),
    }


def _alone(scope: list[Split], source: str) -> list[int]:
    """The ids ``source``'s own counts would keep."""
    terms = [split.get(source, (0, 0.0)) for split in scope]
    stats = {
        "count": torch.tensor([c for c, _ in terms], dtype=torch.long),
        "reap_sum": torch.tensor([r for _, r in terms], dtype=torch.float64),
    }
    return select_kept(saliency(stats), KEEP).tolist()


def _reach(rng: random.Random, experts: list[int]) -> dict[str, list[int]]:
    """``code`` and ``chat`` reach every one of ``experts``; the private
    dataset, the smallest source, a drawn five or more of them."""
    pool = list(experts)
    rng.shuffle(pool)
    mine = sorted(pool[: rng.randint(min(5, len(pool)), len(pool))])
    return {"code": list(experts), "chat": list(experts), "mine": mine}


def _ties(scores: torch.Tensor) -> int:
    """How many reached experts share their score with a lower id."""
    positive = [score for score in scores.tolist() if score > 0]
    return len(positive) - len(set(positive))


def _no_ties(scope: list[Split], scores: torch.Tensor, kept: list[int]) -> bool:
    return _ties(scores) == 0


def _tie_on_the_boundary(scope: list[Split], scores: torch.Tensor, kept: list[int]) -> bool:
    tie = float(scores[3])
    return (
        _ties(scores) == 1
        and float(scores[5]) == tie
        and sum(score > tie for score in scores.tolist()) == KEEP - 1
    )


def _one_source_kept(scope: list[Split], scores: torch.Tensor, kept: list[int]) -> bool:
    return 6 in kept and _ties(scores) == 0


def _pooling_changes_the_pick(scope: list[Split], scores: torch.Tensor, kept: list[int]) -> bool:
    return _ties(scores) == 0 and all(kept != _alone(scope, source) for source in SOURCE_IDS)


def _mean_not_count_or_sum(scope: list[Split], scores: torch.Tensor, kept: list[int]) -> bool:
    pooled = _pooled(scope)
    count, reap_sum = pooled["count"].tolist(), pooled["reap_sum"].tolist()
    others = [e for e in range(EXPERTS) if e != 1]
    return (
        _ties(scores) == 0
        and 1 not in kept
        and all(count[1] > count[e] and reap_sum[1] > reap_sum[e] for e in others)
    )


ALL = list(range(EXPERTS))
#: Per scope ``L<index>``: the case it shows, which experts each source
#: reaches (the pinned experts aside), the pinned splits, and the test a
#: draw must pass.
CASES: list[dict[str, Any]] = [
    {
        "case": "Experts 1 and 6 were reached by no source: each scores 0 (never NaN), and "
        "they rank last, the lower id first.",
        "reach": lambda rng: _reach(rng, [0, 2, 3, 4, 5, 7]),
        "pinned": {},
        "accept": _no_ties,
    },
    {
        "case": "Experts 3 and 5 tie exactly on the keep boundary (225 / 160 each, from "
        "different per-source splits): the lower id, 3, is kept.",
        "reach": lambda rng: _reach(rng, [0, 1, 2, 4, 6, 7]),
        "pinned": {
            3: {"code": (120, 180.0), "chat": (40, 45.0)},
            5: {"code": (60, 75.0), "chat": (60, 90.0), "mine": (40, 60.0)},
        },
        "accept": _tie_on_the_boundary,
    },
    {
        "case": "Expert 6 was reached by one source alone (chat): its bySource holds one "
        "entry, and its score keeps it.",
        "reach": lambda rng: _reach(rng, [0, 1, 2, 3, 4, 5, 7]),
        "pinned": {6: {"chat": (90, 202.5)}},
        "accept": _one_source_kept,
    },
    {
        "case": "Pooling changes the pick: the kept set differs from the one each source "
        "alone would keep (see alone).",
        "reach": lambda rng: _reach(rng, ALL),
        "pinned": {},
        "accept": _pooling_changes_the_pick,
    },
    {
        "case": "Only experts 2, 5 and 7 were reached, fewer than keep: the score-0 tie for "
        "the last slot goes to the lowest id, 0.",
        "reach": lambda rng: {source: [2, 5, 7] for source in SOURCE_IDS},
        "pinned": {},
        "accept": _no_ties,
    },
    {
        "case": "Expert 1 was routed the most and holds the largest reapSum, but its mean is "
        "the lowest: REAP ranks the mean, so it is dropped.",
        "reach": lambda rng: _reach(rng, [0, 2, 3, 4, 5, 6, 7]),
        "pinned": {1: {"code": (450, 225.0), "chat": (200, 112.5), "mine": (150, 65.625)}},
        "accept": _mean_not_count_or_sum,
    },
    {
        "case": "A seeded draw with no constructed case.",
        "reach": lambda rng: _reach(rng, ALL),
        "pinned": {},
        "accept": _no_ties,
    },
    {
        "case": "A seeded draw with no constructed case.",
        "reach": lambda rng: _reach(rng, ALL),
        "pinned": {},
        "accept": _no_ties,
    },
]


def _until(rng: random.Random, case: dict[str, Any]) -> list[Split]:
    accept: Callable[[list[Split], torch.Tensor, list[int]], bool] = case["accept"]
    for _ in range(ATTEMPTS):
        scope = _draw(rng, case["reach"](rng), case["pinned"])
        if scope is None:
            continue
        scores = saliency(_pooled(scope))
        if accept(scope, scores, select_kept(scores, KEEP).tolist()):
            return scope
    raise RuntimeError(f"no draw in {ATTEMPTS} showed the case: {case['case']}")


def build_fixture(seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    scopes: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    for index, case in enumerate(CASES):
        scope = _until(rng, case)
        pooled = _pooled(scope)
        scores = saliency(pooled)
        ranks = expert_ranks(scores)
        name = f"L{index}"
        scopes.append(
            {
                "scope": name,
                "case": case["case"],
                "experts": [
                    {
                        "id": expert,
                        "rank": ranks[expert],
                        "count": int(pooled["count"][expert]),
                        "reapSum": float(pooled["reap_sum"][expert]),
                        "protected": False,
                        "bySource": {s: [split[s][0], split[s][1]] for s in SOURCE_IDS if s in split},
                    }
                    for expert, split in enumerate(scope)
                ],
                "alone": {source: _alone(scope, source) for source in SOURCE_IDS},
            }
        )
        expected.append({"scope": name, "experts": select_kept(scores, KEEP).tolist()})
    return {
        "version": 1,
        "about": "Pooled-REAP parity for kept plans (deacix/legwork#23171). expected is the "
        "kept.json reap-prune --kept reads, and the plan the API's keep-plan resolver must "
        "return for these scopes with weights. Generated: edit the generator, never this file.",
        "generator": "tests/legwork/make_keep_plan_fixture.py",
        "seed": seed,
        "rule": "Per scope an expert's score is the sum of its bySource reapSums over the sum "
        "of its bySource counts, 0 when no source reached it. expected keeps the keep highest "
        "scores, ties to the lower id, listed ascending; rank is the 1-based place in that "
        "order. The weighted score with weights (the sources' token shares) is the same double.",
        "keep": KEEP,
        "experts": EXPERTS,
        "topK": TOP_K,
        "sources": [
            {"id": source, "kind": kind, "rows": rows, "tokens": tokens}
            for source, kind, rows, tokens in SOURCES
        ],
        "weights": {source: TOKENS[source] / TOTAL_TOKENS for source in SOURCE_IDS},
        "scopes": scopes,
        "expected": {"version": 1, "keep": KEEP, "scopes": expected},
    }


def _one_line(value: Any) -> bool:
    """True when ``value`` holds no list of objects or lists."""
    if isinstance(value, dict):
        return all(_one_line(item) for item in value.values())
    if isinstance(value, list):
        return not any(isinstance(item, (dict, list)) for item in value)
    return True


def render(value: Any, level: int = 0) -> str:
    """The fixture's JSON text: two-space indents, and a list of objects or
    lists puts each element on its own line, whole when it holds no such
    list itself."""
    pad = "  " * level
    if isinstance(value, dict) and value:
        items = [f"{pad}  {json.dumps(key)}: {render(item, level + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and not _one_line(value):
        items = [
            f"{pad}  {json.dumps(item) if _one_line(item) else render(item, level + 1)}"
            for item in value
        ]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value)


def fixture_text(seed: int = SEED) -> str:
    return render(build_fixture(seed)) + "\n"


def main() -> int:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(fixture_text(), encoding="utf-8")
    print(FIXTURE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
