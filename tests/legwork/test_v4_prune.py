# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The Legwork lane on a tiny random DeepSeek-V4, CPU only: the model
entry, the observer, the pruner (uniform and ragged), the hash-table
remap, the reload, and both CLIs' STAGE_PROGRESS contract."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from reap.legwork import (
    RouterStatsObserver,
    hash_routed_layers,
    load_router_stats,
    model_attrs,
    moe_layers,
    prune_model,
    remap_hash_table,
    saliency,
    save_router_stats,
    select_kept,
)
from reap.legwork.load import load_pruned, read_pruning_record
from reap.legwork.progress import parse_layer_list
from reap.legwork.prune import save_pruned
from reap.model_util import MODEL_ATTRS

from tests.legwork.tiny_v4 import EXPERTS, LAYERS, TOP_K, VOCAB, build_tiny_v4

KEEP = 4


def test_model_attrs_carry_the_deepseek_v4_entry():
    entry = MODEL_ATTRS["DeepseekV4ForCausalLM"]
    assert MODEL_ATTRS["deepseek_v4"] is entry
    assert entry["experts"] == "experts"
    assert entry["router"] == "gate"
    assert entry["shared_expert"] == "shared_experts"
    assert entry["num_experts_per_tok"] == "num_experts_per_tok"
    assert entry["num_experts"] == "n_routed_experts"
    assert entry["fused"] is True


def test_moe_layers_name_the_hash_routed_bootstrap():
    model = build_tiny_v4()
    assert model_attrs(model) is MODEL_ATTRS["deepseek_v4"]
    layers = moe_layers(model)
    assert [layer.index for layer in layers] == list(range(LAYERS))
    assert [layer.kind for layer in layers] == ["hash_moe", "moe"]
    assert hash_routed_layers(model) == [0]
    assert all(layer.num_experts == EXPERTS for layer in layers)


def _observe(model, samples: int = 6, seq_len: int = 16, seed: int = 3):
    observer = RouterStatsObserver(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(samples):
            ids = torch.randint(4, VOCAB, (1, seq_len), generator=generator)
            model(input_ids=ids, use_cache=False)
            observer.note_sample()
    observer.close()
    return observer.state()


def test_observer_counts_every_routed_slot():
    model = build_tiny_v4()
    state = _observe(model, samples=4, seq_len=10)
    assert state["version"] == 1
    assert state["model_type"] == "deepseek_v4"
    assert state["samples"] == 4
    for index in range(LAYERS):
        layer = state["layers"][index]
        assert layer["num_experts"] == EXPERTS
        assert layer["tokens"] == 40
        # Every token lands top_k slots; a hash layer's table has no duplicates.
        assert int(layer["count"].sum()) == 40 * TOP_K
        assert torch.all(layer["reap_sum"] >= 0)
        assert torch.all(layer["weight_sum"] >= 0)
    # A closed observer records nothing more.
    with torch.no_grad():
        model(input_ids=torch.randint(4, VOCAB, (1, 8)), use_cache=False)
    assert state["layers"][1]["tokens"] == 40


def test_saliency_methods_and_stable_selection():
    stats = {
        "count": torch.tensor([4, 0, 2, 2, 1, 8, 3, 3]),
        "weight_sum": torch.tensor([1.0, 0.0, 0.5, 0.5, 0.1, 2.0, 0.9, 0.9], dtype=torch.float64),
        "reap_sum": torch.tensor([2.0, 0.0, 1.0, 1.0, 0.1, 4.0, 1.5, 1.5], dtype=torch.float64),
    }
    reap = saliency(stats, "reap")
    assert reap[1] == 0  # never routed: scores zero, never a NaN
    assert torch.allclose(reap[[0, 2, 5]], torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64))
    # Ties keep the lower id; the result is ascending.
    assert select_kept(reap, 4).tolist() == [0, 2, 3, 5]
    assert select_kept(saliency(stats, "frequency"), 3).tolist() == [0, 5, 6]
    assert select_kept(saliency(stats, "weighted_frequency"), 2).tolist() == [0, 5]
    with pytest.raises(ValueError):
        select_kept(reap, 9)
    with pytest.raises(ValueError):
        saliency(stats, "magic")


def test_remap_hash_table_prefers_the_nearest_kept_row_without_duplicates():
    router = torch.eye(4)
    router[3] = torch.tensor([0.9, 0.1, 0.0, 0.0])  # expert 3 sits beside expert 0
    kept = torch.tensor([0, 1])
    table = torch.tensor([[3, 1], [2, 3], [0, 3], [1, 0]])
    remapped = remap_hash_table(table, kept, router)
    assert remapped.tolist() == [[0, 1], [0, 1], [0, 1], [1, 0]]
    # [0, 3]: expert 3 wants expert 0, already in the row -> the next-nearest
    # kept expert (1) fills the slot; kept ids are renumbered onto 0..keep-1.
    assert remapped.dtype == table.dtype
    assert int(remapped.max()) < kept.numel()


def _assert_pruned_shapes(model, keep: int, hidden: int = 64):
    for layer in moe_layers(model):
        assert layer.num_experts == keep
        assert tuple(layer.experts.gate_up_proj.shape)[0] == keep
        assert tuple(layer.experts.down_proj.shape)[0] == keep
        assert tuple(layer.router.weight.shape) == (keep, hidden)
        if hasattr(layer.router, "tid2eid"):
            assert int(layer.router.tid2eid.max()) < keep
            assert tuple(layer.router.tid2eid.shape) == (VOCAB, TOP_K)
        else:
            assert tuple(layer.router.e_score_correction_bias.shape) == (keep,)


def test_prune_to_a_uniform_width_and_reload_with_stock_transformers(tmp_path, tiny_v4_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_v4_dir, dtype=torch.float32).eval()
    stats = _observe(model)
    report = prune_model(model, stats, keep=KEEP)
    assert report.keep == KEEP and report.experts_before == EXPERTS
    assert not report.ragged and report.skipped_layers == []
    assert report.n_routed_experts_per_layer == [KEEP, KEEP]
    assert model.config.n_routed_experts == KEEP
    _assert_pruned_shapes(model, KEEP)
    out = save_pruned(model, tmp_path / "pruned", report, source_dir=tiny_v4_dir)
    assert (out / "tokenizer.json").is_file()
    record = read_pruning_record(out)
    assert record["keep"] == KEEP and record["hash_routed_layers"] == [0]
    assert record["layers"][1]["kept"] == report.layers[1].kept

    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    assert reloaded.config.n_routed_experts == KEEP
    _assert_pruned_shapes(reloaded, KEEP)
    ids = torch.randint(4, VOCAB, (2, 12))
    with torch.no_grad():
        logits = reloaded(input_ids=ids, use_cache=False).logits
    assert tuple(logits.shape) == (2, 12, VOCAB)
    assert torch.isfinite(logits).all()
    # load_pruned is the same stock load on a uniform checkpoint.
    same = load_pruned(out)
    with torch.no_grad():
        assert torch.allclose(same(input_ids=ids, use_cache=False).logits, logits)


def test_skip_layers_leaves_a_ragged_checkpoint_the_lane_loader_reads(tmp_path, tiny_v4_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_v4_dir, dtype=torch.float32).eval()
    stats = _observe(model)
    report = prune_model(model, stats, keep=KEEP, skip_layers=[0])
    assert report.ragged and report.skipped_layers == [0]
    assert report.n_routed_experts_per_layer == [EXPERTS, KEEP]
    layers = moe_layers(model)
    assert layers[0].num_experts == EXPERTS and layers[1].num_experts == KEEP
    saved_wide = layers[0].experts.gate_up_proj.detach().clone()
    out = save_pruned(model, tmp_path / "ragged", report)
    assert read_pruning_record(out)["ragged"] is True
    # Stock transformers builds every layer at n_routed_experts and refuses
    # the wider tensors on the size mismatch.
    with pytest.raises(RuntimeError):
        AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    # The lane loader widens the skipped layer and reads its real weights.
    reloaded = load_pruned(out, dtype=torch.float32)
    widths = [layer.num_experts for layer in moe_layers(reloaded)]
    assert widths == [EXPERTS, KEEP]
    assert torch.equal(moe_layers(reloaded)[0].experts.gate_up_proj.detach(), saved_wide)
    ids = torch.randint(4, VOCAB, (1, 9))
    with torch.no_grad():
        logits = reloaded(input_ids=ids, use_cache=False).logits
        before = model(input_ids=ids, use_cache=False).logits
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits, before, atol=1e-5)


def test_prune_refuses_bad_requests():
    model = build_tiny_v4()
    stats = _observe(model, samples=2, seq_len=8)
    with pytest.raises(ValueError, match="between 1 and 8"):
        prune_model(model, stats, keep=9)
    with pytest.raises(ValueError, match="below the router's top-k"):
        prune_model(model, stats, keep=1)
    with pytest.raises(ValueError, match="without an MoE block"):
        prune_model(model, stats, keep=4, skip_layers=[7])
    with pytest.raises(ValueError, match="carry no layer 1"):
        prune_model(model, {"layers": {0: stats["layers"][0]}}, keep=4)


def test_parse_layer_list():
    assert parse_layer_list("") == []
    assert parse_layer_list("2, 0,1,1") == [0, 1, 2]
    with pytest.raises(ValueError):
        parse_layer_list("a")
    with pytest.raises(ValueError):
        parse_layer_list("-1")


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, timeout=600
    )


def test_the_two_clis_emit_progress_lines_and_a_result(tmp_path, tiny_v4_dir, calibration_jsonl):
    stats_path = tmp_path / "router-stats.pt"
    collect = _run(
        "reap.legwork.collect",
        "--model", str(tiny_v4_dir),
        "--calib", str(calibration_jsonl),
        "--out", str(stats_path),
        "--seq-len", "16",
    )
    assert collect.returncode == 0, collect.stderr
    lines = collect.stdout.splitlines()
    progress = [line for line in lines if line.startswith("STAGE_PROGRESS ")]
    assert progress[0] == "STAGE_PROGRESS 0.0" and progress[-1] == "STAGE_PROGRESS 100.0"
    assert all(0 <= float(line.split()[1]) <= 100 for line in progress)
    result = json.loads([line for line in lines if line.startswith("REAP_RESULT ")][-1][12:])
    assert result["samples"] == 12 and result["layers"] == [0, 1]
    stats = load_router_stats(stats_path)
    assert stats["calibration"]["samples"] == 12 and stats["calibration"]["sha256"]

    out_dir = tmp_path / "pruned"
    prune = _run(
        "reap.legwork.prune",
        "--model", str(tiny_v4_dir),
        "--stats", str(stats_path),
        "--out", str(out_dir),
        "--keep", str(KEEP),
        "--skip-layers", "",
    )
    assert prune.returncode == 0, prune.stderr
    lines = prune.stdout.splitlines()
    progress = [line for line in lines if line.startswith("STAGE_PROGRESS ")]
    assert progress[0] == "STAGE_PROGRESS 0.0" and progress[-1] == "STAGE_PROGRESS 100.0"
    result = json.loads([line for line in lines if line.startswith("REAP_RESULT ")][-1][12:])
    assert result == {
        "out": str(out_dir),
        "keep": KEEP,
        "experts_before": EXPERTS,
        "layers": LAYERS,
        "skipped_layers": [],
        "ragged": False,
        "method": "reap",
    }
    record = read_pruning_record(out_dir)
    assert record["calibration"]["samples"] == 12
    assert record["source"] == str(tiny_v4_dir)

    ragged = _run(
        "reap.legwork.prune",
        "--model", str(tiny_v4_dir),
        "--stats", str(stats_path),
        "--out", str(tmp_path / "ragged"),
        "--keep", str(KEEP),
        "--skip-layers", "0",
    )
    assert ragged.returncode == 0, ragged.stderr
    assert "ragged" in ragged.stderr
    assert json.loads(ragged.stdout.splitlines()[-1][12:])["skipped_layers"] == [0]
