# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The native MXFP4 writer (issue deacix/legwork#28563) on a tiny V2.6-layout
checkpoint: an edited BF16 tree re-encodes only its edited residual writers
to the source's storage — expert down projections to MXFP4, the dense down
projection to FP8 blocks, o_proj carried as BF16 — while every unedited
tensor rides byte-identical from the source, and the build serves like the
edited tree once materialized."""

from __future__ import annotations

import json
import pathlib

import pytest
import torch
from safetensors import safe_open

from reap.legwork import mxfp4 as mxfp4_cli
from reap.legwork.checkpoint import ShardWriter, TensorReader
from reap.legwork.compat import patch_remote_code
from reap.legwork.materialize import materialize
from reap.legwork.quant import dequantize_fp8_blocks, dequantize_mxfp4
from tests.legwork.tiny_mimo import (
    EXPERTS,
    MOE_LAYERS,
    TP,
    build_tiny_mimo,
    write_hub_checkpoint,
)


def _quiet(_pct: float) -> None:
    return None


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


def _edited_targets() -> list[str]:
    """Every residual writer the ablation edits on the tiny layout."""
    names = []
    for layer in range(3):
        names.append(f"model.layers.{layer}.self_attn.o_proj.weight")
    names.append("model.layers.0.mlp.down_proj.weight")
    for layer in MOE_LAYERS:
        for expert in range(EXPERTS):
            names.append(f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight")
    return names


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
def edited(working_copy, tmp_path_factory) -> pathlib.Path:
    """The working copy with every ablation target perturbed, same sharding."""
    out = tmp_path_factory.mktemp("mimo-edited")
    reader = TensorReader(working_copy)
    writer = ShardWriter(out, 64 << 10)
    try:
        for name in reader.names():
            tensor = reader.get(name).float()
            if name in _edited_targets():
                tensor = tensor + 1.0
            writer.add(name, tensor.to(torch.bfloat16))
        writer.close()
    finally:
        reader.close()
    for leaf in ("config.json", "generation_config.json"):
        (out / leaf).write_bytes((working_copy / leaf).read_bytes())
    for leaf in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        (out / leaf).write_bytes((working_copy / leaf).read_bytes())
    return out


@pytest.fixture(scope="module")
def built(edited, hub_dir, tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("mimo-mxfp4")
    assert (
        mxfp4_cli.main(
            ["--model", str(edited), "--source", str(hub_dir), "--out", str(out)],
            progress=_quiet,
        )
        == 0
    )
    return out


def test_edited_targets_decide_what_reencodes():
    assert mxfp4_cli.is_edited_target("model.layers.1.self_attn.o_proj.weight")
    assert mxfp4_cli.is_edited_target("model.layers.0.mlp.down_proj.weight")
    assert mxfp4_cli.is_edited_target("model.layers.2.mlp.experts.383.down_proj.weight")
    assert not mxfp4_cli.is_edited_target("model.layers.1.mlp.experts.0.gate_proj.weight")
    assert not mxfp4_cli.is_edited_target("model.layers.1.mlp.experts.0.up_proj.weight")
    assert not mxfp4_cli.is_edited_target("model.layers.1.self_attn.qkv_proj.weight")
    assert not mxfp4_cli.is_edited_target("model.layers.1.self_attn.q_proj.weight")
    assert not mxfp4_cli.is_edited_target("model.embed_tokens.weight")
    assert not mxfp4_cli.is_edited_target("lm_head.weight")
    assert not mxfp4_cli.is_edited_target("model.mtp.layers.0.mlp.gate_proj.weight")
    assert not mxfp4_cli.is_edited_target("model.layers.1.mlp.gate.weight")


def test_mxfp4_reencodes_edited_tensors_to_the_sources_storage(built):
    written = TensorReader(built)
    try:
        for layer in MOE_LAYERS:
            for expert in range(EXPERTS):
                name = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
                scale = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale"
                assert (written.get(name).dtype, written.get(scale).dtype) == (
                    torch.uint8,
                    torch.uint8,
                ), (layer, expert)
        dense = "model.layers.0.mlp.down_proj.weight"
        assert written.get(dense).dtype == torch.float8_e4m3fn
        assert written.get(dense[: -len(".weight")] + ".weight_scale_inv").dtype == torch.float32
    finally:
        written.close()


def test_unedited_tensors_ride_byte_identical_from_the_source(built, hub_dir):
    source = _tensors(hub_dir)
    built_tensors = _tensors(built)
    dense = "model.layers.0.mlp.down_proj.weight"
    for name, value in source.items():
        if ".mlp.experts." in name and ".down_proj.weight" in name:
            continue  # re-encoded from the edited tree
        if name in (dense, dense[: -len(".weight")] + ".weight_scale_inv"):
            continue  # re-encoded from the edited tree
        if ".self_attn.o_proj.weight" in name:
            continue  # edited values, carried as BF16
        assert built_tensors[name] == value, name


def test_edited_values_survive_the_reencode(built, hub_dir, edited):
    native, written, edited_reader = (
        TensorReader(hub_dir),
        TensorReader(built),
        TensorReader(edited),
    )
    try:
        for layer in MOE_LAYERS:
            name = f"model.layers.{layer}.mlp.experts.0.down_proj.weight"
            stem = name[: -len(".weight")]
            decoded = dequantize_mxfp4(written.get(name), written.get(stem + ".weight_scale"))
            want = edited_reader.get(name).float()
            before = dequantize_mxfp4(native.get(name), native.get(stem + ".weight_scale"))
            assert torch.allclose(decoded, want, atol=0.5, rtol=0), layer
            assert not torch.allclose(decoded, before, atol=0.5, rtol=0), layer
        dense = "model.layers.0.mlp.down_proj.weight"
        dense_stem = dense[: -len(".weight")]
        decoded = dequantize_fp8_blocks(
            written.get(dense), written.get(dense_stem + ".weight_scale_inv")
        )
        assert torch.allclose(decoded, edited_reader.get(dense).float(), atol=0.1)
        for layer in range(3):
            name = f"model.layers.{layer}.self_attn.o_proj.weight"
            assert written.get(name).dtype == torch.bfloat16
            assert torch.equal(written.get(name), edited_reader.get(name))
    finally:
        native.close()
        written.close()
        edited_reader.close()


def test_mxfp4_build_config_index_and_files(built, hub_dir):
    config = json.loads((built / "config.json").read_text())
    source_config = json.loads((hub_dir / "config.json").read_text())
    assert config["quantization_config"] == source_config["quantization_config"]
    assert config["attention_projection_layout"] == "fused_qkv"
    assert "legwork_working_copy" not in config
    assert config["n_routed_experts"] == source_config["n_routed_experts"]
    record = config["reap_mxfp4"]
    assert record["quantized"] == len(_edited_targets()) - 3  # every down_proj; o_proj rides BF16
    index = json.loads((built / "model.safetensors.index.json").read_text())
    assert index["metadata"]["save_format"] == "mxfp4" and index["metadata"]["tp_size"] == TP
    assert (built / "modeling_mimo_v2.py").is_file()
    assert (built / "dflash" / "config.json").is_file()
    assert any(name.startswith("model.mtp.") for name in _tensors(built))


def test_mxfp4_build_serves_like_the_edited_tree(built, edited, tmp_path_factory):
    from transformers import AutoModelForCausalLM

    materialized = tmp_path_factory.mktemp("mimo-mxfp4-working-copy")
    materialize(built, materialized)
    edited_model = AutoModelForCausalLM.from_pretrained(
        edited, trust_remote_code=True, dtype=torch.bfloat16
    ).eval()
    built_model = AutoModelForCausalLM.from_pretrained(
        materialized, trust_remote_code=True, dtype=torch.bfloat16
    ).eval()
    patch_remote_code(edited_model)
    patch_remote_code(built_model)
    ids = torch.randint(3, 256, (2, 20), generator=torch.Generator().manual_seed(8))
    with torch.no_grad():
        got = built_model(ids).logits.float()
        want = edited_model(ids).logits.float()
    # The re-quant noise floor on a tiny random model: the average tracks
    # (measured ~0.14), the peak stays bounded, and the outputs differ, so
    # the formats were really applied. Precision rides the tensor tests.
    diff = (got - want).abs()
    assert float(diff.mean()) < 0.5, float(diff.mean())
    assert float(diff.max()) < 10.0, float(diff.max())
    assert not torch.equal(got, want)


def test_mxfp4_refuses_what_it_cannot_write(hub_dir, edited, tmp_path):
    with pytest.raises(SystemExit, match="quantization"):
        mxfp4_cli.main(
            ["--model", str(edited), "--source", str(edited), "--out", str(tmp_path / "x")],
            progress=_quiet,
        )
    with pytest.raises(SystemExit, match="config.json"):
        mxfp4_cli.main(
            [
                "--model",
                str(tmp_path / "missing"),
                "--source",
                str(hub_dir),
                "--out",
                str(tmp_path / "y"),
            ],
            progress=_quiet,
        )


def test_mxfp4_refuses_an_edited_tree_missing_a_target(edited, hub_dir, tmp_path):
    broken = tmp_path / "broken"
    reader = TensorReader(edited)
    writer = ShardWriter(broken, 64 << 10)
    try:
        for name in reader.names():
            if name == "model.layers.1.mlp.experts.0.down_proj.weight":
                continue
            writer.add(name, reader.get(name))
        writer.close()
    finally:
        reader.close()
    (broken / "config.json").write_bytes((edited / "config.json").read_bytes())
    with pytest.raises(SystemExit, match="has no tensor"):
        mxfp4_cli.main(
            ["--model", str(broken), "--source", str(hub_dir), "--out", str(tmp_path / "z")],
            progress=_quiet,
        )
