# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Prune an explicit kept list (``reap-prune --kept``): the plan's checks,
the slice with the hash remap unchanged, the record and REAP_RESULT, and
the pooled-REAP parity fixture whose byte-identical copy the API's
keep-plan resolver reads."""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import re
from typing import Any

import pytest
import torch

from reap.legwork import prune
from reap.legwork.arch import moe_layers
from reap.legwork.expert_map import expert_ranks
from reap.legwork.load import read_pruning_record
from reap.legwork.observer import save_router_stats
from reap.legwork.prune import (
    KEPT_METHOD,
    parse_kept_plan,
    prune_model,
    read_kept_plan,
    remap_hash_table,
    resolve_kept_plan,
    saliency,
    select_kept,
)

from tests.legwork.make_keep_plan_fixture import FIXTURE, fixture_text
from tests.legwork.tiny_v4 import EXPERTS, VOCAB, build_tiny_tokenizer, build_tiny_v4

KEEP = 4
#: One layer per fixture scope, in DeepSeek-V4's order: three hash-routed
#: bootstrap layers, then top-k layers.
FIXTURE_LAYERS = ("hash_moe",) * 3 + ("moe",) * 5
GOOD = (("L0", [0, 2, 5, 7]), ("L1", [6, 1, 4, 3]))


def _plan(*scopes: tuple[Any, Any], keep: Any = KEEP, version: Any = 1) -> dict[str, Any]:
    return {"version": version, "keep": keep, "scopes": [{"scope": s, "experts": e} for s, e in scopes]}


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _pooled(scope: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "count": torch.tensor([e["count"] for e in scope["experts"]], dtype=torch.long),
        "reap_sum": torch.tensor([e["reapSum"] for e in scope["experts"]], dtype=torch.float64),
    }


def _prune(*argv: str) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert prune.main(list(argv), progress=lambda pct: None) == 0
    line = [line for line in stdout.getvalue().splitlines() if line.startswith("REAP_RESULT ")][-1]
    return json.loads(line[len("REAP_RESULT ") :])


def _refused(*argv: str) -> str:
    with pytest.raises(SystemExit) as refusal:
        prune.main(list(argv), progress=lambda pct: None)
    return str(refusal.value)


def test_the_fixture_is_the_generators_output():
    """The committed file is byte for byte what the seeded generator writes,
    so its copy in the API pins the same scopes."""
    assert FIXTURE.read_bytes() == fixture_text().encode("utf-8")


def test_expected_is_the_pooled_reap_pick():
    fixture = _fixture()
    keep, tokens = fixture["keep"], {s["id"]: s["tokens"] for s in fixture["sources"]}
    assert len(fixture["scopes"]) >= 6
    assert (fixture["expected"]["version"], fixture["expected"]["keep"]) == (1, keep)
    expected = {s["scope"]: s["experts"] for s in fixture["expected"]["scopes"]}
    assert list(expected) == [s["scope"] for s in fixture["scopes"]]
    for scope in fixture["scopes"]:
        experts = scope["experts"]
        assert [e["id"] for e in experts] == list(range(fixture["experts"]))
        for e in experts:
            assert e["protected"] is False
            assert all(count > 0 for count, _ in e["bySource"].values())
            assert e["count"] == sum(count for count, _ in e["bySource"].values())
            assert e["reapSum"] == sum(reap for _, reap in e["bySource"].values())
        # Top-k routing: a source's counts sum to its tokens times top_k.
        for source, total in tokens.items():
            counts = [e["bySource"][source][0] for e in experts if source in e["bySource"]]
            assert sum(counts) == total * fixture["topK"] and max(counts) <= total
        scores = saliency(_pooled(scope))
        assert expected[scope["scope"]] == select_kept(scores, keep).tolist()
        assert [e["rank"] for e in experts] == expert_ranks(scores)
        for source, ids in scope["alone"].items():
            terms = [e["bySource"].get(source, [0, 0.0]) for e in experts]
            alone = {
                "count": torch.tensor([count for count, _ in terms]),
                "reap_sum": torch.tensor([reap for _, reap in terms], dtype=torch.float64),
            }
            assert ids == select_kept(saliency(alone), keep).tolist()
    assert parse_kept_plan(fixture["expected"], keep) == {
        int(name[1:]): ids for name, ids in expected.items()
    }


def test_token_share_weights_give_the_pooled_scores_to_the_bit():
    """The API's keep-plan score, ``sum_s (w_s/T_s)*reapSum_s /
    sum_s (w_s/T_s)*count_s``, with the fixture's weights (the token
    shares) is the pooled score as the same double in either summation
    order, so it keeps the same ids, ties included."""
    fixture = _fixture()
    tokens = {s["id"]: s["tokens"] for s in fixture["sources"]}
    weights = fixture["weights"]
    assert weights == {source: count / sum(tokens.values()) for source, count in tokens.items()}
    expected = {s["scope"]: s["experts"] for s in fixture["expected"]["scopes"]}
    for scope in fixture["scopes"]:
        pooled = saliency(_pooled(scope)).tolist()
        weighted = []
        for e in scope["experts"]:
            terms = [
                ((weights[s] / tokens[s]) * reap, (weights[s] / tokens[s]) * count)
                for s, (count, reap) in e["bySource"].items()
            ]
            scores = set()
            for order in (terms, terms[::-1]):
                numerator = sum(term[0] for term in order)
                denominator = sum(term[1] for term in order)
                scores.add(numerator / denominator if denominator else 0.0)
            assert scores == {pooled[e["id"]]}
            weighted.append(scores.pop())
        ranked = sorted(range(len(weighted)), key=lambda i: (-weighted[i], i))
        assert sorted(ranked[: fixture["keep"]]) == expected[scope["scope"]]


def test_the_fixture_shows_each_edge_case():
    fixture = _fixture()
    keep = fixture["keep"]
    expected = {s["scope"]: set(s["experts"]) for s in fixture["expected"]["scopes"]}
    shown: collections.Counter = collections.Counter()
    for scope in fixture["scopes"]:
        kept, experts = expected[scope["scope"]], scope["experts"]
        scores = saliency(_pooled(scope)).tolist()
        reached = {e["id"] for e in experts if e["count"] > 0}
        unreached = sorted(e["id"] for e in experts if e["count"] == 0)
        for e in experts:
            if e["count"] == 0:
                assert e["bySource"] == {} and e["reapSum"] == 0.0 and scores[e["id"]] == 0.0
        if unreached and len(reached) >= keep:
            assert not kept & set(unreached)
            shown["no source reached an expert"] += 1
        if len(reached) < keep:
            assert kept == reached | set(unreached[: keep - len(reached)])
            shown["a score-0 tie for a kept slot"] += 1
        for a in range(len(experts)):
            for b in range(a + 1, len(experts)):
                if scores[a] == scores[b] > 0 and (a in kept) != (b in kept):
                    assert a in kept
                    shown["an exact tie on the boundary"] += 1
        if any(len(e["bySource"]) == 1 and e["id"] in kept for e in experts):
            shown["a kept expert one source reached"] += 1
        if all(set(ids) != kept for ids in scope["alone"].values()):
            shown["pooling changes the pick"] += 1
        top = max(experts, key=lambda e: e["count"])
        if top["id"] not in kept and all(top["reapSum"] > e["reapSum"] for e in experts if e is not top):
            shown["the most-routed expert dropped"] += 1
    assert len(shown) == 6, shown


@pytest.fixture(scope="module")
def fixture_model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny-v4-fixture-layers")
    build_tiny_v4(mlp_layer_types=FIXTURE_LAYERS).save_pretrained(out)
    build_tiny_tokenizer().save_pretrained(out)
    return out


def test_prune_keeps_exactly_the_fixture_plan(tmp_path, fixture_model_dir):
    """The fixture's expected plan, pruned into a V4 with one layer per
    scope: the checkpoint holds exactly those experts, each layer's hash
    table remapped as before, and it reloads with stock transformers."""
    from transformers import AutoModelForCausalLM

    expected = _fixture()["expected"]
    kept_path = tmp_path / "kept.json"
    kept_path.write_text(json.dumps(expected), encoding="utf-8")
    sha256 = hashlib.sha256(kept_path.read_bytes()).hexdigest()
    out = tmp_path / "pruned"
    result = _prune(
        "--model", str(fixture_model_dir),
        "--kept", str(kept_path),
        "--out", str(out),
        "--keep", str(KEEP),
        "--skip-layers", "",
    )
    assert result == {
        "out": str(out),
        "keep": KEEP,
        "experts_before": EXPERTS,
        "layers": len(FIXTURE_LAYERS),
        "skipped_layers": [],
        "ragged": False,
        "method": KEPT_METHOD,
        "draft_blocks_source": 0,
        "draft_blocks_carried": 0,
        "kept_plan_sha256": sha256,
    }
    record = read_pruning_record(out)
    assert record["method"] == "kept"
    assert record["kept_plan"] == {"sha256": sha256, "path": str(kept_path)}
    assert record["stats"] is None and record["calibration"] is None
    kept = {int(scope["scope"][1:]): scope["experts"] for scope in expected["scopes"]}
    assert {layer["index"]: layer["kept"] for layer in record["layers"]} == kept
    assert record["hash_routed_layers"] == [0, 1, 2]

    source = AutoModelForCausalLM.from_pretrained(fixture_model_dir, dtype=torch.float32).eval()
    pruned = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    assert pruned.config.n_routed_experts == KEEP
    for before, after in zip(moe_layers(source), moe_layers(pruned), strict=True):
        ids = torch.tensor(kept[before.index])
        assert torch.equal(after.experts.gate_up_proj, before.experts.gate_up_proj[ids])
        assert torch.equal(after.experts.down_proj, before.experts.down_proj[ids])
        assert torch.equal(after.router.weight, before.router.weight[ids])
        table = getattr(before.router, "tid2eid", None)
        if table is not None:
            assert torch.equal(after.router.tid2eid, remap_hash_table(table, ids, before.router.weight))
        else:
            bias = before.router.e_score_correction_bias[ids]
            assert torch.equal(after.router.e_score_correction_bias, bias)
    with torch.no_grad():
        logits = pruned(input_ids=torch.randint(4, VOCAB, (1, 10)), use_cache=False).logits
    assert torch.isfinite(logits).all()


def test_the_plan_wins_over_the_stats_in_any_order(tmp_path, tiny_v4_dir):
    """With ``--stats`` beside ``--kept`` the plan still decides (REAP would
    keep 4..7 here); scopes and ids may come in any order, and the record
    keeps the stats' provenance."""
    layer = {
        "kind": "moe",
        "num_experts": EXPERTS,
        "tokens": EXPERTS,
        "count": torch.ones(EXPERTS, dtype=torch.long),
        "weight_sum": torch.ones(EXPERTS, dtype=torch.float64),
        "reap_sum": torch.arange(EXPERTS, dtype=torch.float64),
        "max_norm": torch.ones(EXPERTS),
    }
    assert select_kept(saliency(layer), KEEP).tolist() == [4, 5, 6, 7]
    stats_path = save_router_stats(
        {"version": 2, "layers": {0: layer, 1: layer}, "calibration": {"samples": 3}},
        tmp_path / "stats.pt",
    )
    kept_path = tmp_path / "kept.json"
    kept_path.write_text(json.dumps(_plan(("L1", [3, 0, 2, 1]), ("L0", [6, 1, 0, 3]))), encoding="utf-8")
    out = tmp_path / "pruned"
    result = _prune(
        "--model", str(tiny_v4_dir),
        "--stats", str(stats_path),
        "--kept", str(kept_path),
        "--out", str(out),
        "--keep", str(KEEP),
    )
    assert result["method"] == "kept"
    assert result["kept_plan_sha256"] == hashlib.sha256(kept_path.read_bytes()).hexdigest()
    record = read_pruning_record(out)
    assert [layer["kept"] for layer in record["layers"]] == [[0, 1, 3, 6], [0, 1, 2, 3]]
    assert record["stats"] == str(stats_path) and record["calibration"] == {"samples": 3}


@pytest.mark.parametrize(
    "plan, message",
    [
        pytest.param([], "the plan is not a JSON object", id="not-an-object"),
        pytest.param(_plan(*GOOD, version=2), "reap-prune reads version 1", id="version-2"),
        pytest.param(_plan(*GOOD, version=True), "reap-prune reads version 1", id="version-bool"),
        pytest.param(_plan(*GOOD, keep="4"), "the plan's keep is '4', not an expert count", id="keep-str"),
        pytest.param(_plan(*GOOD, keep=5), "the plan keeps 5 experts per scope, --keep is 4", id="keep"),
        pytest.param({"version": 1, "keep": 4, "scopes": {}}, "scopes are not a list", id="scopes"),
        pytest.param({"version": 1, "keep": 4, "scopes": ["L0"]}, "scopes[0] names no scope", id="entry"),
        pytest.param(
            _plan(*GOOD, ("D0", [0, 1, 2, 3])),
            "scope D0 is a draft block: the pruned checkpoint carries no draft blocks",
            id="draft-block",
        ),
        *[
            pytest.param(_plan(GOOD[0], (scope, [1, 3, 4, 6])), "is unknown", id=f"scope-{scope!r}")
            for scope in ("X1", "L01", "l1", "L1\n", "L-1", "L\u0661", "", "D01")
        ],
        pytest.param(_plan(*GOOD, ("L0", [1, 3, 4, 6])), "scope L0 is listed twice", id="scope-twice"),
        pytest.param(_plan(("L0", "0,2,5,7"), GOOD[1]), "experts is not a list", id="experts-str"),
        pytest.param(_plan(("L0", [0, 2, 5, 7.0]), GOOD[1]), "expert id 7.0 is not an integer", id="float"),
        pytest.param(_plan(("L0", [0, 2, 5, True]), GOOD[1]), "expert id True is not an integer", id="bool"),
        pytest.param(_plan(("L0", [0, 2, 5, "7"]), GOOD[1]), "expert id '7' is not an integer", id="str"),
        pytest.param(_plan(("L0", [0, 2, 5, -1]), GOOD[1]), "scope L0: expert -1 is out of range", id="neg"),
        pytest.param(
            _plan(("L0", [0, 2, 2, 7]), GOOD[1]), "scope L0: experts [2] are listed twice", id="dup"
        ),
        pytest.param(_plan(("L0", [0, 2, 5]), GOOD[1]), "3 experts listed, the plan keeps 4", id="short"),
        pytest.param(_plan(("L0", [0, 2, 5, 6, 7]), GOOD[1]), "5 experts listed", id="long"),
    ],
)
def test_parse_refuses_a_malformed_plan(plan, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_kept_plan(plan, KEEP)


def test_resolve_checks_the_plan_against_the_model():
    layers = moe_layers(build_tiny_v4())
    resolved = resolve_kept_plan(_plan(*GOOD), layers, KEEP)
    assert {index: ids.tolist() for index, ids in resolved.items()} == {0: [0, 2, 5, 7], 1: [1, 3, 4, 6]}
    refusals = [
        (_plan(*GOOD, ("L7", [0, 1, 2, 3])), "scope L7 is unknown: the model has no MoE layer 7 "
         "(its MoE layers are [0, 1])"),
        (_plan(GOOD[0]), "the plan lists no experts for MoE layers [1]"),
        (_plan(), "the plan lists no experts for MoE layers [0, 1]"),
        (_plan(GOOD[0], ("L1", [1, 3, 4, 8])), "scope L1: expert 8 is out of range; layer 1 has 8 experts"),
    ]
    for plan, message in refusals:
        with pytest.raises(ValueError, match=re.escape(message)):
            resolve_kept_plan(plan, layers, KEEP)


def test_prune_model_checks_the_whole_plan_before_it_slices():
    model = build_tiny_v4()
    with pytest.raises(ValueError, match="cannot be combined with --skip-layers"):
        prune_model(model, None, KEEP, skip_layers=[0], kept_plan=_plan(*GOOD))
    with pytest.raises(ValueError, match="expert 8 is out of range"):
        prune_model(model, None, KEEP, kept_plan=_plan(GOOD[0], ("L1", [1, 3, 4, 8])))
    with pytest.raises(ValueError, match="pass stats or a kept plan"):
        prune_model(model, None, KEEP)
    assert [layer.num_experts for layer in moe_layers(model)] == [EXPERTS, EXPERTS]
    report = prune_model(model, None, KEEP, kept_plan=_plan(*GOOD))
    assert report.method == KEPT_METHOD and not report.ragged
    assert [layer.kept for layer in report.layers] == [[0, 2, 5, 7], [1, 3, 4, 6]]
    assert [layer.num_experts for layer in moe_layers(model)] == [KEEP, KEEP]


def test_the_cli_refuses_a_bad_request_before_it_loads_the_model(tmp_path):
    """Every refusal lands before the model loads: the model path does not
    exist, so reaching the load would raise something else."""
    plan = tmp_path / "kept.json"
    plan.write_text(json.dumps(_plan(*GOOD)), encoding="utf-8")
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(_plan(*GOOD, ("D0", [0, 1, 2, 3]))), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    latin = tmp_path / "latin.json"
    latin.write_bytes(b'{"version": 1, "keep": 4, "scopes": [], "note": "caf\xe9"}')
    base = ["--model", str(tmp_path / "no-model"), "--out", str(tmp_path / "out")]
    four = [*base, "--keep", str(KEEP)]
    assert _refused(*four, "--kept", str(plan), "--skip-layers", "0") == (
        "reap-prune: --kept lists every MoE layer's experts, so it cannot be combined with --skip-layers"
    )
    assert _refused(*four, "--kept", str(plan), "--method", "reap") == (
        "reap-prune: --kept names the experts and --method ranks them; pass one"
    )
    assert _refused(*four) == "reap-prune: --stats is required unless --kept names the experts"
    assert _refused(*four, "--kept", str(tmp_path / "missing.json")) == (
        f"reap-prune: --kept {tmp_path / 'missing.json'} cannot be read (No such file or directory)"
    )
    assert _refused(*four, "--kept", str(broken)) == (
        f"reap-prune: --kept {broken}: the plan is not JSON (line 1, column 2)"
    )
    assert _refused(*four, "--kept", str(latin)) == f"reap-prune: --kept {latin}: the plan is not UTF-8 text"
    assert "the pruned checkpoint carries no draft blocks" in _refused(*four, "--kept", str(draft))
    assert _refused(*base, "--keep", "6", "--kept", str(plan)) == (
        f"reap-prune: --kept {plan}: the plan keeps 4 experts per scope, --keep is 6"
    )
    assert not (tmp_path / "out").exists()


def test_read_kept_plan_hashes_the_file_bytes(tmp_path):
    compact, spaced = tmp_path / "compact.json", tmp_path / "spaced.json"
    compact.write_text(json.dumps(_plan(*GOOD), separators=(",", ":")), encoding="utf-8")
    spaced.write_text(json.dumps(_plan(*GOOD), indent=2), encoding="utf-8")
    plan, sha256 = read_kept_plan(compact, KEEP)
    same, other = read_kept_plan(spaced, KEEP)
    assert plan == same == _plan(*GOOD)
    assert sha256 == hashlib.sha256(compact.read_bytes()).hexdigest() != other
