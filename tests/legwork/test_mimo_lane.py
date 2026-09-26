# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The MiMo-V2 arm of the lane on a tiny V2.6-layout checkpoint: the BF16
working copy runs on Xiaomi's own model code, collect observes it through
its router and renders rows through its chat template, and
``reap-prune --slice-source`` writes a build that keeps the source's
formats, bytes and layout and serves exactly as the source does with the
dropped experts masked out of its router."""

from __future__ import annotations

import gzip
import json
import pathlib
import random

import pytest
import torch
from safetensors import safe_open

from reap.legwork import collect as collect_cli
from reap.legwork import prune as prune_cli
from reap.legwork.checkpoint import TensorReader
from reap.legwork.compat import patch_remote_code
from reap.legwork.materialize import materialize
from reap.legwork.observer import load_router_stats
from tests.legwork.tiny_mimo import (
    EXPERTS,
    MOE_LAYERS,
    TOP_K,
    TP,
    build_tiny_mimo,
    quantize_dequantize,
    write_hub_checkpoint,
)

KEEP = 4


def _quiet(_pct: float) -> None:
    return None


def _load(path: pathlib.Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=torch.bfloat16
    ).eval()
    patch_remote_code(model)
    return model


def _tensors(path: pathlib.Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Every tensor's ``(dtype, shape, bytes)``, across the shards."""
    index = json.loads((path / "model.safetensors.index.json").read_text())
    out: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(path / shard), framework="pt") as handle:
            names = list(handle.keys())
            for name in names:
                tensor = handle.get_tensor(name).contiguous()
                raw = tensor.view(torch.uint8).numpy().tobytes()
                out[name] = (str(tensor.dtype), tuple(tensor.shape), raw)
    return out


@pytest.fixture(scope="module")
def tiny_mimo():
    return build_tiny_mimo(seed=3)


@pytest.fixture(scope="module")
def hub_dir(tiny_mimo, tmp_path_factory) -> pathlib.Path:
    return write_hub_checkpoint(tiny_mimo, tmp_path_factory.mktemp("mimo-hub"))


@pytest.fixture(scope="module")
def working_copy(hub_dir, tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("mimo-working-copy")
    materialize(hub_dir, out, shard_bytes=64 << 10)
    return out


@pytest.fixture(scope="module")
def calibration(tmp_path_factory) -> pathlib.Path:
    rng = random.Random(11)
    path = tmp_path_factory.mktemp("mimo-calib") / "calibration.jsonl"
    tools = [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}}]
    with path.open("w", encoding="utf-8") as handle:
        for sample in range(6):
            words = " ".join(f"t{rng.randrange(4, 256)}" for _ in range(12))
            if sample % 2:
                row = {"messages": [{"role": "user", "content": words}], "tools": tools, "source": "code"}
            else:
                row = {"text": words, "source": "chat"}
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture(scope="module")
def router_stats(working_copy, calibration, tmp_path_factory) -> pathlib.Path:
    folder = tmp_path_factory.mktemp("mimo-stats")
    out = folder / "router-stats.pt"
    argv = ["--model", str(working_copy), "--calib", str(calibration), "--out", str(out)]
    argv += ["--map", str(folder / "expert-map.json.gz"), "--trust-remote-code", "--seq-len", "64"]
    assert collect_cli.main(argv, progress=_quiet) == 0
    return out


def test_the_working_copy_is_split_bf16_without_drafts(hub_dir, working_copy):
    config = json.loads((working_copy / "config.json").read_text())
    assert config["attention_projection_layout"] == "split"
    assert "quantization_config" not in config
    assert config["legwork_working_copy"]["tp_size"] == TP
    tensors = _tensors(working_copy)
    assert not any(name.endswith(("weight_scale", "weight_scale_inv")) for name in tensors)
    assert not any("qkv_proj" in name or name.startswith("model.mtp.") for name in tensors)
    for layer in range(3):
        for projection in ("q_proj", "k_proj", "v_proj"):
            dtype, _, _ = tensors[f"model.layers.{layer}.self_attn.{projection}.weight"]
            assert dtype == "torch.bfloat16"
    assert (working_copy / "modeling_mimo_v2.py").is_file()
    assert not (working_copy / "dflash").exists()


def test_the_working_copy_runs_the_models_own_code(tiny_mimo, working_copy):
    reference = quantize_dequantize(tiny_mimo)
    model = _load(working_copy)
    ids = torch.randint(3, 256, (2, 20), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        got = model(ids).logits
        want = reference(ids).logits
        original = tiny_mimo(ids).logits
    # The same BF16 weights on the same code: bit for bit. The unquantized
    # model differs, so the formats were really applied.
    assert torch.equal(got, want), float((got.float() - want.float()).abs().max())
    assert not torch.equal(got, original)


def test_collect_observes_mimo_through_its_router(router_stats):
    stats = load_router_stats(router_stats)
    assert sorted(stats["layers"]) == list(MOE_LAYERS)
    assert stats["model_type"] == "mimo_v2"
    # MiMo's own chat template rendered every messages row, tools included.
    assert stats["calibration"]["template_fallbacks"] == 0
    for index in MOE_LAYERS:
        layer = stats["layers"][index]
        assert layer["num_experts"] == EXPERTS
        # Every routed token reaches exactly top-k experts.
        assert int(layer["count"].sum()) == layer["tokens"] * TOP_K
        assert sorted(layer["by_source"]) == ["chat", "code"]
        assert float(layer["reap_sum"].sum()) > 0


def test_collect_writes_mimos_expert_map(router_stats):
    with gzip.open(router_stats.parent / "expert-map.json.gz", "rt", encoding="utf-8") as handle:
        expert_map = json.load(handle)
    scopes = expert_map.get("scopes") or []
    assert [scope.get("scope") for scope in scopes] == [f"L{i}" for i in MOE_LAYERS]
    assert all(len(scope.get("experts") or []) == EXPERTS for scope in scopes)


@pytest.fixture(scope="module")
def sliced(hub_dir, router_stats, tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("mimo-sliced")
    code = prune_cli.main(
        ["--model", str(hub_dir), "--stats", str(router_stats), "--out", str(out), "--keep", str(KEEP), "--slice-source"],
        progress=_quiet,
    )
    assert code == 0
    return out


def test_slice_keeps_non_expert_bytes(hub_dir, sliced):
    source = _tensors(hub_dir)
    built = _tensors(sliced)
    config = json.loads((sliced / "config.json").read_text())
    assert config["n_routed_experts"] == KEEP
    assert config["attention_projection_layout"] == "fused_qkv"
    record = config["reap_pruning"]
    assert record["keep"] == KEEP and record["experts_before"] == EXPERTS
    assert record["draft_blocks"] == {"source": 1, "carried": 1}
    kept = {layer["index"]: layer["kept"] for layer in record["layers"]}
    for name, value in source.items():
        if ".mlp.experts." in name or ".mlp.gate." in name:
            continue
        assert built[name] == value, name
    for layer, ids in kept.items():
        assert ids == sorted(ids) and len(ids) == KEEP
        for new, old in enumerate(ids):
            for suffix in ("gate_proj.weight", "gate_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale"):
                assert built[f"model.layers.{layer}.mlp.experts.{new}.{suffix}"] == source[
                    f"model.layers.{layer}.mlp.experts.{old}.{suffix}"
                ]
        assert f"model.layers.{layer}.mlp.experts.{KEEP}.up_proj.weight" not in built
    index = json.loads((sliced / "model.safetensors.index.json").read_text())
    assert index["metadata"]["save_format"] == "mxfp4" and index["metadata"]["tp_size"] == TP
    assert index["metadata"]["total_size"] == sum(
        len(value[2]) for value in built.values()
    )
    assert (sliced / "modeling_mimo_v2.py").is_file() and (sliced / "dflash" / "config.json").is_file()


def test_slice_routers_follow_the_kept_order(hub_dir, sliced):
    record = json.loads((sliced / "config.json").read_text())["reap_pruning"]
    source, built = TensorReader(hub_dir), TensorReader(sliced)
    for layer in record["layers"]:
        order = torch.tensor(layer["kept"])
        for tensor in ("weight", "e_score_correction_bias"):
            name = f"model.layers.{layer['index']}.mlp.gate.{tensor}"
            assert torch.equal(built.get(name), source.get(name)[order])


def test_a_sliced_build_serves_as_the_source_does_with_the_dropped_experts_masked(
    working_copy, sliced, tmp_path_factory
):
    pruned_copy = tmp_path_factory.mktemp("mimo-sliced-working-copy")
    materialize(sliced, pruned_copy)
    pruned = _load(pruned_copy)
    masked = _load(working_copy)
    record = json.loads((sliced / "config.json").read_text())["reap_pruning"]
    with torch.no_grad():
        for layer in record["layers"]:
            gate = masked.model.layers[layer["index"]].mlp.gate
            dropped = [e for e in range(EXPERTS) if e not in layer["kept"]]
            gate.e_score_correction_bias[dropped] = -1e4
    ids = torch.randint(3, 256, (2, 20), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        got = pruned(ids).logits.float()
        want = masked(ids).logits.float()
    assert torch.allclose(got, want, atol=1e-3, rtol=0), float((got - want).abs().max())


def test_drop_drafts_leaves_out_the_draft_layers_and_the_drafter(hub_dir, router_stats, tmp_path):
    out = tmp_path / "no-drafts"
    assert (
        prune_cli.main(
            ["--model", str(hub_dir), "--stats", str(router_stats), "--out", str(out), "--keep", str(KEEP), "--slice-source", "--drop-drafts"],
            progress=_quiet,
        )
        == 0
    )
    config = json.loads((out / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 0
    assert config["reap_pruning"]["draft_blocks"] == {"source": 1, "carried": 0}
    assert not any(name.startswith("model.mtp.") for name in _tensors(out))
    assert not (out / "dflash").exists()


def test_slice_refuses_what_it_cannot_write(hub_dir, router_stats, tmp_path):
    for argv, message in (
        (["--keep", "1"], "top-k"),
        (["--keep", "4", "--skip-layers", "1"], "uniform"),
    ):
        with pytest.raises(SystemExit, match=message):
            prune_cli.main(
                ["--model", str(hub_dir), "--stats", str(router_stats), "--out", str(tmp_path / "x"), "--slice-source", *argv],
                progress=_quiet,
            )
    with pytest.raises(SystemExit, match="--slice-source"):
        prune_cli.main(
            ["--model", str(hub_dir), "--stats", str(router_stats), "--out", str(tmp_path / "y"), "--keep", "4", "--drop-drafts"],
            progress=_quiet,
        )


def test_the_remote_code_patch_is_idempotent(tiny_mimo):
    assert patch_remote_code(tiny_mimo) == []
