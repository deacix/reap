# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""A DeepSeek-V4 prune sliced from its source's own tensors (``--slice-source``).

The first NVIDIA prunes of DeepSeek-V4-Flash (deacix/legwork#25045) kept the
reference in FP8 and FP4 while they pruned it, then saved it through
transformers 5.17. That save writes the tensors under ``model.``-prefixed
DeepSeek names, which the quantizer's transformers 5.14 cannot map, and it
cannot write the FP8 experts' scale grids back at all: they came out under
``.experts.<n>.w1.weight`` without their layer, every layer's on the same
names, at the full width. The slice never builds the model. Every tensor keeps
the reference's name, dtype and bytes; the kept experts' weights and ``.scale``
grids move to dense ids together; the routers and the hash tables follow them.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest
import torch

from reap.legwork import collect
from reap.legwork import prune as prune_cli
from reap.legwork.checkpoint import TensorReader
from reap.legwork.fp8 import deepseek_name
from reap.legwork.observer import load_router_stats
from reap.legwork.prune import remap_hash_table, saliency, select_kept
from reap.legwork.slice import source_draft_layers, source_moe_layers
from tests.legwork.tiny_v4_reference import (
    EXPERTS,
    QUANTIZED,
    REFERENCE_CONFIG,
    schema_of,
    write_reference,
)

KEEP = 4
#: Kept ids that are never a prefix of the experts, so a slice that kept the
#: first ``KEEP`` ids, or the scales of the first ``KEEP``, fails.
PLAN = {0: [1, 3, 4, 6], 1: [0, 2, 5, 7], 2: [2, 3, 6, 7]}
EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")


def _quiet(_pct: float) -> None:
    return None


def _tensors(path: pathlib.Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Every tensor's ``(dtype, shape, bytes)``, across the shards."""
    from safetensors import safe_open

    out: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            names = list(handle.keys())
            for name in names:
                tensor = handle.get_tensor(name).contiguous()
                raw = tensor.view(torch.uint8).numpy().tobytes()
                out[name] = (str(tensor.dtype), tuple(tensor.shape), raw)
    return out


def _write_plan(path: pathlib.Path, plan: dict[int, list[int]] = PLAN) -> pathlib.Path:
    scopes = [{"scope": f"L{layer}", "experts": ids} for layer, ids in sorted(plan.items())]
    path.write_text(json.dumps({"version": 1, "keep": KEEP, "scopes": scopes}), encoding="utf-8")
    return path


def _slice(model: pathlib.Path, out: pathlib.Path, *selection: str) -> pathlib.Path:
    argv = ["--model", str(model), *selection, "--out", str(out), "--keep", str(KEEP), "--slice-source"]
    assert prune_cli.main(argv, progress=_quiet) == 0
    return out


@pytest.fixture(scope="module")
def v4_reference(tmp_path_factory) -> tuple[pathlib.Path, pathlib.Path]:
    return write_reference(tmp_path_factory.mktemp("v4-slice-reference"))


@pytest.fixture(scope="module")
def plan_path(tmp_path_factory) -> pathlib.Path:
    return _write_plan(tmp_path_factory.mktemp("v4-slice-plan") / "kept.json")


@pytest.fixture(scope="module")
def sliced(v4_reference, plan_path, tmp_path_factory) -> pathlib.Path:
    _, reference = v4_reference
    return _slice(reference, tmp_path_factory.mktemp("v4-sliced"), "--kept", str(plan_path))


def test_the_source_layout_reads_deepseeks_config_words(v4_reference):
    _, reference = v4_reference
    config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    layers = source_moe_layers(config)
    assert [(layer.index, layer.num_experts, layer.kind) for layer in layers] == [
        (0, EXPERTS, "hash_moe"),
        (1, EXPERTS, "moe"),
        (2, EXPERTS, "moe"),
    ]
    # transformers' own words for the same layout read the same.
    words = {k: v for k, v in config.items() if k != "num_hash_layers"}
    words["mlp_layer_types"] = ["hash_moe", "moe", "moe"]
    assert [layer.kind for layer in source_moe_layers(words)] == ["hash_moe", "moe", "moe"]
    assert source_draft_layers(reference) == 1


def test_a_v4_slice_names_every_tensor_as_the_reference_does(v4_reference, sliced):
    _, reference = v4_reference
    source, built = _tensors(reference), _tensors(sliced)
    assert not [name for name in built if name.startswith(("model.", "mtp.", "."))]
    assert set(built) <= set(source)
    assert schema_of(built) == schema_of(source)
    for name in built:
        if QUANTIZED.search(name):
            assert name[: -len(".weight")] + ".scale" in built, name
    experts = {int(EXPERT.match(name).group(2)) for name in built if EXPERT.match(name)}
    assert experts == set(range(KEEP))


def test_the_kept_experts_keep_their_weights_and_their_scales(v4_reference, sliced):
    _, reference = v4_reference
    source, built = _tensors(reference), _tensors(sliced)
    for layer, ids in PLAN.items():
        for new, old in enumerate(ids):
            for leaf in ("w1", "w2", "w3"):
                for part in ("weight", "scale"):
                    got = built[f"layers.{layer}.ffn.experts.{new}.{leaf}.{part}"]
                    want = source[f"layers.{layer}.ffn.experts.{old}.{leaf}.{part}"]
                    assert got == want, (layer, new, leaf, part)
    assert built["layers.1.ffn.experts.0.w1.weight"][0] == "torch.int8"
    assert built["layers.1.ffn.experts.0.w1.scale"][0] == "torch.float8_e8m0fnu"


def test_every_other_tensor_is_copied_byte_for_byte(v4_reference, sliced):
    _, reference = v4_reference
    source, built = _tensors(reference), _tensors(sliced)
    for name, value in source.items():
        if EXPERT.match(name) or ".ffn.gate." in name or name.startswith("mtp."):
            continue
        assert built[name] == value, name


def test_the_routers_follow_the_kept_order_and_the_hash_table_is_remapped(v4_reference, sliced):
    _, reference = v4_reference
    source, built = TensorReader(reference), TensorReader(sliced)
    for layer, ids in PLAN.items():
        order = torch.tensor(ids)
        gate = f"layers.{layer}.ffn.gate"
        assert torch.equal(built.get(f"{gate}.weight"), source.get(f"{gate}.weight")[order])
        if f"{gate}.bias" in source:
            assert torch.equal(built.get(f"{gate}.bias"), source.get(f"{gate}.bias")[order])
    table = built.get("layers.0.ffn.gate.tid2eid")
    want = remap_hash_table(
        source.get("layers.0.ffn.gate.tid2eid"),
        torch.tensor(PLAN[0]),
        source.get("layers.0.ffn.gate.weight"),
    )
    assert torch.equal(table, want)
    assert int(table.min()) >= 0 and int(table.max()) < KEEP
    assert all(len(set(row)) == len(row) for row in table.tolist())
    assert "layers.0.ffn.gate.bias" not in built and "layers.1.ffn.gate.tid2eid" not in built


def test_the_config_is_the_references_with_the_prunes_edits(v4_reference, sliced, plan_path):
    _, reference = v4_reference
    config = json.loads((sliced / "config.json").read_text(encoding="utf-8"))
    record = config.pop("reap_pruning")
    want = dict(REFERENCE_CONFIG, n_routed_experts=KEEP, num_nextn_predict_layers=0)
    assert config == want
    assert record["method"] == "kept" and record["keep"] == KEEP and record["experts_before"] == EXPERTS
    assert record["hash_routed_layers"] == [0]
    assert record["n_routed_experts_per_layer"] == [KEEP] * 3 and record["ragged"] is False
    assert {layer["index"]: layer["kept"] for layer in record["layers"]} == PLAN
    assert record["draft_blocks"] == {"source": 1, "carried": 0}
    assert record["kept_plan"]["path"] == str(plan_path)
    assert record["sliced"]["dropped"] > 0
    index = json.loads((sliced / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == sum(len(value[2]) for value in _tensors(sliced).values())
    assert (sliced / "tokenizer.json").is_file() or (sliced / "tokenizer_config.json").is_file()


def test_the_sliced_tree_loads_in_transformers_with_nothing_missing(sliced):
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(sliced, output_loading_info=True)
    assert not info["missing_keys"], sorted(info["missing_keys"])[:5]
    assert not info["unexpected_keys"], sorted(info["unexpected_keys"])[:5]
    assert not info["mismatched_keys"]
    assert model.config.n_routed_experts == KEEP
    ids = torch.randint(4, 256, (2, 12), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    assert tuple(logits.shape) == (2, 12, 256) and bool(torch.isfinite(logits).all())


def test_the_slice_is_the_in_memory_prune_of_the_float_twin(v4_reference, plan_path, tmp_path):
    """On a float checkpoint both paths run: the slice writes, under DeepSeek's
    names, the very tensors the in-memory prune holds."""
    float_dir, _ = v4_reference
    sliced = _slice(float_dir, tmp_path / "sliced", "--kept", str(plan_path))
    in_memory = tmp_path / "in-memory"
    argv = ["--model", str(float_dir), "--kept", str(plan_path), "--out", str(in_memory)]
    assert prune_cli.main([*argv, "--keep", str(KEEP)], progress=_quiet) == 0
    held = TensorReader(in_memory)
    written = TensorReader(sliced)
    assert {deepseek_name(name) for name in held.names()} == set(written.names())
    for name in held.names():
        assert torch.equal(held.get(name).float(), written.get(deepseek_name(name)).float()), name


def test_a_v4_slice_ranks_on_router_stats_as_the_in_memory_prune_does(
    v4_reference, calibration_jsonl, tmp_path
):
    float_dir, reference = v4_reference
    stats_path = tmp_path / "router-stats.pt"
    # float32 keeps this BF16 fixture's forward off the mixed-precision path,
    # as the FP8 build's fixture does; the statistics only rank the experts.
    argv = ["--model", str(float_dir), "--calib", str(calibration_jsonl), "--out", str(stats_path)]
    argv += ["--seq-len", "64", "--dtype", "float32"]
    assert collect.main(argv, progress=_quiet) == 0
    sliced = _slice(reference, tmp_path / "sliced", "--stats", str(stats_path))
    stats = load_router_stats(stats_path)
    record = json.loads((sliced / "config.json").read_text(encoding="utf-8"))["reap_pruning"]
    assert record["method"] == "reap"
    for layer in record["layers"]:
        want = select_kept(saliency(stats["layers"][layer["index"]], "reap"), KEEP).tolist()
        assert layer["kept"] == want


def test_the_cli_reports_a_sliced_v4_result(v4_reference, plan_path, tmp_path):
    _, reference = v4_reference
    out = tmp_path / "cli"
    argv = ["--model", str(reference), "--kept", str(plan_path), "--out", str(out)]
    run = subprocess.run(
        [sys.executable, "-m", "reap.legwork.prune", *argv, "--keep", str(KEEP), "--slice-source"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert any(line.startswith("STAGE_PROGRESS ") for line in lines)
    result_line = next(line for line in lines if line.startswith("REAP_RESULT "))
    result = json.loads(result_line.split(" ", 1)[1])
    assert result["sliced"] is True and result["total_size"] > 0
    shape = (result["keep"], result["experts_before"], result["layers"], result["ragged"])
    assert shape == (KEEP, EXPERTS, 3, False)
    assert (result["draft_blocks_source"], result["draft_blocks_carried"]) == (1, 0)


def test_a_hash_table_the_config_does_not_name_is_refused(v4_reference, plan_path, tmp_path):
    _, reference = v4_reference
    edited = tmp_path / "edited"
    edited.mkdir()
    for entry in reference.iterdir():
        (edited / entry.name).write_bytes(entry.read_bytes())
    config = json.loads((edited / "config.json").read_text(encoding="utf-8"))
    config["num_hash_layers"] = 0
    (edited / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="layer 0"):
        _slice(edited, tmp_path / "out", "--kept", str(plan_path))
