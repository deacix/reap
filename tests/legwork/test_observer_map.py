# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Router stats v2 and the expert map on the tiny V4: the map's keys and
types, per-source terms summing to the pooled ones, the top-token sketch
exact on a small deterministic case, private rows kept out of the sketch
and the exemplars, the protected rule, the map's rank order equal to
``select_kept``'s, and v1 stats still loading and pruning."""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import math
import random

import pytest
import torch

from reap.legwork import collect
from reap.legwork.arch import moe_layers
from reap.legwork.expert_map import (
    PROTECT_NORM_RATIO,
    build_expert_map,
    expert_ranks,
    protected_experts,
    read_expert_map,
    write_expert_map,
)
from reap.legwork.observer import (
    EXEMPLAR_CONTEXT,
    EXEMPLARS,
    STATS_VERSION,
    TOP_TOKENS,
    RouterStatsObserver,
    TopTokenSketch,
    load_router_stats,
    routed_expert_norms,
    save_router_stats,
)
from reap.legwork.prune import prune_model, saliency, select_kept

from tests.legwork.tiny_v4 import EXPERTS, TOP_K, VOCAB, build_tiny_tokenizer, build_tiny_v4

PUBLIC = range(4, 200)
PRIVATE = range(200, VOCAB)
SAMPLES = 19


def _calibration_rows(seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(SAMPLES - 1):
        if i % 3 == 0:
            rows.append({"input_ids": [rng.choice(PUBLIC) for _ in range(40)], "source": "code"})
        elif i % 3 == 1:
            words = " ".join(f"t{rng.choice(PUBLIC)}" for _ in range(30))
            rows.append({"text": words, "source": "chat"})
        else:
            ids = [rng.choice(PRIVATE) for _ in range(40)]
            rows.append({"input_ids": ids, "source": "mine", "private": True})
    rows.append({"input_ids": [rng.choice(PUBLIC) for _ in range(12)]})
    return rows


@pytest.fixture(scope="module")
def collected(tmp_path_factory, tiny_v4_dir):
    tmp = tmp_path_factory.mktemp("collect-map")
    calib = tmp / "calib.jsonl"
    calib.write_text(
        "".join(json.dumps(row) + "\n" for row in _calibration_rows()), encoding="utf-8"
    )
    stats_path = tmp / "stats.pt"
    map_path = tmp / "maps" / "experts.json.gz"
    argv = [
        "--model", str(tiny_v4_dir),
        "--calib", str(calib),
        "--out", str(stats_path),
        "--map", str(map_path),
        "--seq-len", "64",
    ]
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert collect.main(argv, progress=lambda pct: None) == 0
    line = [l for l in stdout.getvalue().splitlines() if l.startswith("REAP_RESULT ")][-1]
    return {
        "result": json.loads(line[12:]),
        "stats": load_router_stats(stats_path),
        "map": read_expert_map(map_path),
        "map_path": map_path,
    }


def test_collect_reports_the_map_it_wrote(collected):
    result, map_path = collected["result"], collected["map_path"]
    data = map_path.read_bytes()
    assert result["map"] == {
        "path": str(map_path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    assert result["template_fallbacks"] == 0
    assert [s["id"] for s in result["sources"]] == ["chat", "code", "mine", "unlabeled"]
    # No timestamp in the gzip header: one map is one byte string.
    again = write_expert_map(collected["map"], map_path.parent / "again.json.gz")
    assert again["sha256"] == result["map"]["sha256"]


def test_the_stats_file_is_version_2(collected):
    stats = collected["stats"]
    assert stats["version"] == STATS_VERSION == 2
    assert (stats["experts"], stats["top_k"], stats["vocab_size"]) == (EXPERTS, TOP_K, VOCAB)
    assert stats["sketch"] == {"top_tokens": 16, "capacity": 256, "exemplars": 3, "context": 24}
    assert sum(s["rows"] for s in stats["sources"]) == stats["samples"] == SAMPLES
    for layer in stats["layers"].values():
        assert sorted(layer["by_source"]) == ["chat", "code", "mine", "unlabeled"]
        for terms in layer["by_source"].values():
            assert terms["count"].dtype == torch.long and terms["count"].shape == (EXPERTS,)
            assert terms["reap_sum"].dtype == torch.float64 and terms["reap_sum"].shape == (EXPERTS,)
        assert len(layer["top_tokens"]) == len(layer["exemplars"]) == EXPERTS
        for tops in layer["top_tokens"]:
            assert len(tops) <= TOP_TOKENS
            assert tops == sorted(tops, key=lambda pair: (-pair[1], pair[0]))
        for rows in layer["exemplars"]:
            assert len(rows) <= EXEMPLARS
            assert len({row["sample"] for row in rows}) == len(rows)  # one per sample
            assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)


def test_the_map_carries_every_expert_with_typed_fields(collected):
    expert_map, stats = collected["map"], collected["stats"]
    assert list(expert_map) == ["version", "model", "calibration", "protectNormRatio", "sources", "scopes"]
    assert expert_map["version"] == 1
    assert expert_map["model"] == {
        "modelType": "deepseek_v4",
        "modelClass": "DeepseekV4ForCausalLM",
        "experts": EXPERTS,
        "topK": TOP_K,
    }
    calibration = expert_map["calibration"]
    assert list(calibration) == ["sha256", "samples", "tokens", "seqLen", "templateFallbacks"]
    assert calibration["sha256"] == stats["calibration"]["sha256"] and len(calibration["sha256"]) == 64
    assert (calibration["samples"], calibration["seqLen"], calibration["templateFallbacks"]) == (
        SAMPLES, 64, 0,
    )
    assert calibration["tokens"] == sum(s["tokens"] for s in expert_map["sources"])
    assert expert_map["protectNormRatio"] == PROTECT_NORM_RATIO == 20
    assert [s["scope"] for s in expert_map["scopes"]] == ["L0", "L1"]
    assert [s["kind"] for s in expert_map["scopes"]] == ["hash_moe", "moe"]
    for scope in expert_map["scopes"]:
        assert list(scope) == ["scope", "layer", "kind", "tokens", "experts"]
        layer = stats["layers"][scope["layer"]]
        assert scope["tokens"] == layer["tokens"] == calibration["tokens"]
        assert [e["id"] for e in scope["experts"]] == list(range(EXPERTS))
        for e in scope["experts"]:
            assert list(e) == [
                "id", "rank", "count", "weightSum", "reapSum", "maxNorm",
                "protected", "bySource", "topTokens", "exemplars",
            ]
            assert isinstance(e["rank"], int) and isinstance(e["count"], int)
            assert isinstance(e["protected"], bool)
            for key in ("weightSum", "reapSum", "maxNorm"):
                assert isinstance(e[key], float)
            # Full float64 precision: the JSON value is the stats value.
            assert e["reapSum"] == float(layer["reap_sum"][e["id"]])
            assert e["weightSum"] == float(layer["weight_sum"][e["id"]])
            assert e["maxNorm"] == float(layer["max_norm"][e["id"]])
            for count, reap_sum in e["bySource"].values():
                assert isinstance(count, int) and isinstance(reap_sum, float)
            for token, count in e["topTokens"]:
                assert isinstance(token, str) and isinstance(count, int)
            for exemplar in e["exemplars"]:
                assert list(exemplar) == ["source", "before", "token", "after"]
                assert all(isinstance(value, str) for value in exemplar.values())
    assert any(e["topTokens"] and e["exemplars"] for s in expert_map["scopes"] for e in s["experts"])


def test_by_source_sums_equal_the_pooled_terms(collected):
    for layer in collected["stats"]["layers"].values():
        terms = layer["by_source"].values()
        assert torch.equal(sum(t["count"] for t in terms), layer["count"])
        assert torch.allclose(sum(t["reap_sum"] for t in terms), layer["reap_sum"], rtol=1e-12, atol=0)
    for scope in collected["map"]["scopes"]:
        for e in scope["experts"]:
            assert all(count > 0 for count, _ in e["bySource"].values())
            assert sum(count for count, _ in e["bySource"].values()) == e["count"]
            assert math.isclose(
                sum(reap for _, reap in e["bySource"].values()), e["reapSum"], rel_tol=1e-12
            )


def test_private_rows_never_reach_top_tokens_or_exemplars(collected):
    stats, expert_map = collected["stats"], collected["map"]
    for layer in stats["layers"].values():
        assert int(layer["by_source"]["mine"]["count"].sum()) == 6 * 40 * TOP_K
        for tops in layer["top_tokens"]:
            assert all(token < PRIVATE.start for token, _ in tops)
        for rows in layer["exemplars"]:
            for row in rows:
                assert row["source"] != "mine"
                assert all(token < PRIVATE.start for token in row["window"])
    kinds = {s["id"]: s["kind"] for s in expert_map["sources"]}
    assert kinds == {"chat": "corpus", "code": "corpus", "mine": "dataset", "unlabeled": "corpus"}
    private_words = {f"t{i}" for i in PRIVATE}
    for scope in expert_map["scopes"]:
        assert any("mine" in e["bySource"] for e in scope["experts"])
        for e in scope["experts"]:
            assert not private_words & {token for token, _ in e["topTokens"]}
            for exemplar in e["exemplars"]:
                assert exemplar["source"] != "mine"
                words = " ".join((exemplar["before"], exemplar["token"], exemplar["after"])).split()
                assert not private_words & set(words)


def test_rank_order_is_select_kept_order(collected):
    for scope in collected["map"]["scopes"]:
        scores = saliency(collected["stats"]["layers"][scope["layer"]], "reap")
        ranks = {e["id"]: e["rank"] for e in scope["experts"]}
        assert sorted(ranks.values()) == list(range(1, EXPERTS + 1))
        for width in range(1, EXPERTS + 1):
            kept = sorted(expert for expert, rank in ranks.items() if rank <= width)
            assert kept == select_kept(scores, width).tolist()


def test_ranks_break_ties_to_the_lower_id():
    scores = torch.tensor([0.5, 0.0, 0.5, 0.25, 0.5, 0.0], dtype=torch.float64)
    assert expert_ranks(scores) == [1, 5, 2, 4, 3, 6]


def test_the_protected_rule():
    def flags(norms, counts):
        return protected_experts(torch.tensor(norms), torch.tensor(counts))

    # Median of [1, 2, 60] is 2: 60 >= 20 * 2.
    assert flags([1.0, 2.0, 60.0], [1, 1, 1]) == [False, False, True]
    # Exactly the ratio is protected; just below it is not.
    assert flags([1.0, 2.0, 40.0], [1, 1, 1]) == [False, False, True]
    assert flags([1.0, 2.0, 39.5], [1, 1, 1]) == [False, False, False]
    # An even count takes the mean of the middle two: (2 + 4) / 2 = 3, and
    # 40 < 60 (the lower middle, 2, would have protected it).
    assert flags([1.0, 2.0, 4.0, 40.0], [1, 1, 1, 1]) == [False] * 4
    # Unrouted experts stay out of the median and are never protected.
    assert flags([2.0, 0.0, 2.0, 0.0, 40.0], [3, 0, 3, 0, 3]) == [False] * 4 + [True]
    assert flags([0.0, 0.0], [0, 0]) == [False, False]
    assert flags([0.0, 0.0, 5.0], [1, 1, 1]) == [False, False, False]  # a zero median


def test_the_map_flags_protected_experts(tmp_path):
    layer = {
        "kind": "moe",
        "num_experts": 4,
        "tokens": 10,
        "count": torch.tensor([3, 4, 0, 3]),
        "weight_sum": torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64),
        "reap_sum": torch.tensor([0.5, 0.25, 0.0, 9.0], dtype=torch.float64),
        "max_norm": torch.tensor([0.5, 0.25, 0.0, 10.0]),
    }
    stats = {"version": 1, "model_type": "deepseek_v4", "layers": {3: layer}}
    expert_map = build_expert_map(stats, build_tiny_tokenizer())
    (scope,) = expert_map["scopes"]
    assert scope["scope"] == "L3" and scope["layer"] == 3
    assert [e["protected"] for e in scope["experts"]] == [False, False, False, True]
    assert [e["rank"] for e in scope["experts"]] == [2, 3, 4, 1]
    assert all(e["bySource"] == {} and e["topTokens"] == [] for e in scope["experts"])


def test_the_sketch_is_exact_on_a_small_deterministic_case():
    """Two experts over a 10-token vocab, one routed slot per token; capacity
    3 forces a merge and a cut after every pass. The leaders stay exact and
    the tie between tokens 4 and 6 keeps the lower id."""
    sketch = TopTokenSketch(num_experts=2, vocab_size=10, capacity=3, top=2)
    wide = TopTokenSketch(num_experts=2, vocab_size=10, capacity=10, top=10)
    oracle: collections.Counter = collections.Counter()
    for noise in (0, 1, 2, 3, 4, 8):
        routed = (
            [(0, 5)] * 4 + [(0, 7)] * 3 + [(0, noise)]
            + [(1, 2)] * 5 + [(1, 4)] * 3 + [(1, 6)] * 3 + [(1, 9)] * 2
            + [(2, 1)]  # an id past num_experts is skipped
        )
        experts = torch.tensor([[e] for e, _ in routed])
        tokens = torch.tensor([t for _, t in routed])
        sketch.add(experts, tokens)
        wide.add(experts, tokens)
        oracle.update((e, t) for e, t in routed if e < 2)

    def ranked(expert: int, top: int) -> list[list[int]]:
        pairs = [[t, c] for (e, t), c in oracle.items() if e == expert]
        return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:top]

    assert sketch.top_tokens() == [[[5, 24], [7, 18]], [[2, 30], [4, 18]]]
    assert sketch.top_tokens() == [ranked(0, 2), ranked(1, 2)]
    # Wide enough to never cut: every token, exactly.
    assert wide.top_tokens() == [ranked(0, 10), ranked(1, 10)]


def test_the_observer_sketch_and_exemplars_match_a_brute_force(tiny_v4_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_v4_dir, dtype=torch.float32).eval()
    layers = moe_layers(model)
    captured: dict[int, list] = {layer.index: [] for layer in layers}

    def capture(index: int):
        def hook(module, args, output):
            captured[index].append(tuple(a.detach().clone() for a in args[:3]))

        return hook

    handles = [layer.experts.register_forward_hook(capture(layer.index)) for layer in layers]
    observer = RouterStatsObserver(model)
    rng = random.Random(5)
    samples: list[tuple[list[int], bool]] = []
    with torch.no_grad():
        for i in range(9):
            private = i % 3 == 2
            ids = [rng.choice(PRIVATE if private else PUBLIC) for _ in range(rng.randrange(20, 60))]
            samples.append((ids, private))
            observer.begin_sample(ids, source="mine" if private else "code", private=private)
            model(input_ids=torch.tensor([ids]), use_cache=False)
    observer.close()
    for handle in handles:
        handle.remove()
    state = observer.state()
    for layer in layers:
        counts: collections.Counter = collections.Counter()
        best: dict[int, list] = collections.defaultdict(list)
        for serial, ((ids, private), (hidden, index, weights)) in enumerate(
            zip(samples, captured[layer.index])
        ):
            if private:
                continue
            for token, experts in zip(ids, index.tolist()):
                counts.update((expert, token) for expert in experts)
            with torch.no_grad():
                for expert, token_idx, w, norms in routed_expert_norms(
                    layer.experts, hidden, index, weights
                ):
                    contribution = w.double() * norms.double()
                    j = int(contribution.argmax())
                    best[expert].append((float(contribution[j]), serial, int(token_idx[j])))
        assert sorted(best) == list(range(EXPERTS))  # every expert served public tokens
        stats_layer = state["layers"][layer.index]
        for e in range(EXPERTS):
            expected = sorted(
                ([t, c] for (x, t), c in counts.items() if x == e), key=lambda p: (-p[1], p[0])
            )[:TOP_TOKENS]
            assert stats_layer["top_tokens"][e] == expected
            exemplars = stats_layer["exemplars"][e]
            assert [(r["score"], r["sample"], r["position"]) for r in exemplars] == sorted(
                best[e], key=lambda b: (-b[0], b[1])
            )[:EXEMPLARS]
            for row in exemplars:
                ids = samples[row["sample"]][0]
                start = max(0, row["position"] - EXEMPLAR_CONTEXT)
                assert row["window"] == ids[start : row["position"] + EXEMPLAR_CONTEXT + 1]
                assert row["offset"] == row["position"] - start
                assert row["window"][row["offset"]] == ids[row["position"]]


def test_begin_sample_guards_the_forward_it_names():
    model = build_tiny_v4()
    observer = RouterStatsObserver(model)
    with torch.no_grad():
        observer.begin_sample([5, 6, 7])
        with pytest.raises(ValueError, match="routed 4 tokens, begin_sample named 3"):
            model(input_ids=torch.tensor([[5, 6, 7, 8]]), use_cache=False)
        observer.begin_sample([5, 6, 7])
        model(input_ids=torch.tensor([[5, 6, 7]]), use_cache=False)
        with pytest.raises(ValueError, match="a second forward pass for one sample"):
            model(input_ids=torch.tensor([[5, 6, 7]]), use_cache=False)
    observer.close()


def test_v1_stats_still_load_and_prune(tmp_path):
    model = build_tiny_v4()
    observer = RouterStatsObserver(model)
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for _ in range(4):
            model(input_ids=torch.randint(4, VOCAB, (1, 16), generator=generator), use_cache=False)
            observer.note_sample()
    observer.close()
    state = observer.state()
    v1_keys = ("kind", "num_experts", "tokens", "count", "weight_sum", "reap_sum", "max_norm")
    v1 = {
        "version": 1,
        "model_class": state["model_class"],
        "model_type": state["model_type"],
        "samples": state["samples"],
        "layers": {i: {k: layer[k] for k in v1_keys} for i, layer in state["layers"].items()},
    }
    path = save_router_stats(v1, tmp_path / "v1.pt")
    loaded = load_router_stats(path)
    assert loaded["version"] == 1 and "sources" not in loaded
    expected = [select_kept(saliency(loaded["layers"][i]), 4).tolist() for i in sorted(loaded["layers"])]
    report = prune_model(model, loaded, keep=4)
    assert [layer.kept for layer in report.layers] == expected
    assert report.n_routed_experts_per_layer == [4, 4]
    with pytest.raises(ValueError, match="version 1 or 2"):
        load_router_stats(save_router_stats({**v1, "version": 3}, tmp_path / "v3.pt"))
