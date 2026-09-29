# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The FP8 build of a dequantized DeepSeek-V4 prune (``reap.legwork.fp8``).

The first live AMD prune (deacix/legwork, job a8176daa, 2026-09-29) built its
checkpoint and failed the serve proof: vLLM refused the config transformers
had rewritten. These tests pin what transformers writes, reproduce vLLM's
refusal on it, and hold the build to DeepSeek's own layout: DeepSeek's names,
FP8 tiles with a float32 ``.scale`` for every weight the reference scales, the
reference's config with the prune's edits.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch

from tests.legwork.tiny_v4_reference import (
    BLOCK,
    DEEPSEEK_V4_SCHEMA,
    QUANTIZED,
    REFERENCE_CONFIG,
    deepseek_dtype,
    schema_of,
    write_reference,
)

KEEP = 4


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, timeout=600
    )


def _tensors(directory: pathlib.Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for shard in sorted(pathlib.Path(directory).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            names = handle.keys()
            for name in names:
                out[name] = handle.get_tensor(name)
    return out


def _config(directory: pathlib.Path) -> dict:
    return json.loads((pathlib.Path(directory) / "config.json").read_text(encoding="utf-8"))


def vllm_config(config: dict):
    """vLLM 0.30.0's own ``deepseek_v4`` config
    (``vllm/transformers_utils/configs/deepseek_v4.py``) without its vision
    fields: every other key rides ``**kwargs`` into ``PretrainedConfig``,
    whose validation vLLM's serve met."""
    from transformers import PretrainedConfig

    class DeepseekV4Config(PretrainedConfig):
        model_type = "deepseek_v4"

        def __init__(
            self,
            max_position_embeddings=1048576,
            rope_scaling=None,
            rope_parameters=None,
            rope_theta=10000.0,
            **kwargs,
        ):
            self.max_position_embeddings = max_position_embeddings
            self.rope_scaling = rope_scaling
            self.rope_theta = rope_theta
            self.rope_parameters = rope_scaling or rope_parameters
            super().__init__(**kwargs)

    return DeepseekV4Config(**config)


@pytest.fixture(scope="session")
def v4_reference(tmp_path_factory) -> tuple[pathlib.Path, pathlib.Path]:
    return write_reference(tmp_path_factory.mktemp("v4-reference"))


@pytest.fixture(scope="session")
def v4_pruned(tmp_path_factory, v4_reference, calibration_jsonl) -> pathlib.Path:
    """The tree a ROCm prune writes: DeepSeek's storage loaded ``--dequantize``."""
    float_dir, reference_dir = v4_reference
    root = tmp_path_factory.mktemp("v4-pruned")
    stats = root / "router-stats.pt"
    # The statistics only rank the experts; float32 keeps this BF16 fixture's
    # forward off the mixed-precision path (the dequantized collect's
    # `float_forward` covers the real one).
    collect = _run(
        "reap.legwork.collect",
        "--model",
        str(float_dir),
        "--calib",
        str(calibration_jsonl),
        "--out",
        str(stats),
        "--seq-len",
        "16",
        "--dtype",
        "float32",
    )
    assert collect.returncode == 0, collect.stderr
    prune = _run(
        "reap.legwork.prune",
        "--model",
        str(reference_dir),
        "--stats",
        str(stats),
        "--out",
        str(root / "pruned"),
        "--keep",
        str(KEEP),
        "--dequantize",
    )
    assert prune.returncode == 0, prune.stderr
    return root / "pruned"


@pytest.fixture(scope="session")
def v4_build(tmp_path_factory, v4_reference, v4_pruned) -> tuple[pathlib.Path, str]:
    _, reference_dir = v4_reference
    out = tmp_path_factory.mktemp("v4-build") / "fp8"
    run = _run(
        "reap.legwork.fp8",
        "--model",
        str(v4_pruned),
        "--source",
        str(reference_dir),
        "--out",
        str(out),
    )
    assert run.returncode == 0, run.stderr
    return out, run.stdout


def test_the_fixture_is_deepseeks_own_layout(v4_reference):
    float_dir, reference_dir = v4_reference
    reference = _tensors(reference_dir)
    assert schema_of(reference) == set(DEEPSEEK_V4_SCHEMA)
    assert schema_of(_tensors(float_dir)) == {
        n for n in DEEPSEEK_V4_SCHEMA if not n.endswith(".scale")
    }
    expert, expert_scale = (
        reference["layers.1.ffn.experts.0.w1.weight"],
        reference["layers.1.ffn.experts.0.w1.scale"],
    )
    assert expert.dtype == torch.int8 and expert_scale.dtype == torch.float8_e8m0fnu
    assert expert_scale.shape[1] * 32 == expert.shape[1] * 2
    linear, linear_scale = (
        reference["layers.1.attn.wq_a.weight"],
        reference["layers.1.attn.wq_a.scale"],
    )
    assert linear.dtype == torch.float8_e4m3fn and linear_scale.dtype == torch.float8_e8m0fnu
    assert reference["layers.0.ffn.gate.tid2eid"].dtype == torch.int64
    assert reference["layers.1.hc_attn_fn"].dtype == torch.float32


def test_transformers_saves_the_prune_in_its_own_names_and_words(v4_pruned):
    names = set(_tensors(v4_pruned))
    assert "model.embed_tokens.weight" in names and "embed.weight" not in names
    assert "model.layers.0.attn.norm.weight" in names
    assert "model.hc_head.hc_fn" in names
    config = _config(v4_pruned)
    assert config["mlp_layer_types"] == ["hash_moe", "moe", "moe"]
    for dropped in ("num_hash_layers", "compress_ratios", "rope_scaling"):
        assert dropped not in config


def test_vllm_refuses_the_saved_config_and_reads_the_reference(v4_pruned, v4_reference):
    with pytest.raises(Exception, match="mlp_layer_types"):
        vllm_config(_config(v4_pruned))
    parsed = vllm_config(_config(v4_reference[1]))
    assert parsed.num_hash_layers == 1 and parsed.compress_ratios == [0, 4, 128]


def test_the_build_names_every_tensor_as_deepseek_does(v4_build, v4_reference):
    out, _ = v4_build
    reference = set(_tensors(v4_reference[1]))
    expected = {
        name
        for name in reference
        if not name.startswith("mtp.")
        and not any(
            f".experts.{e}." in name for e in range(KEEP, REFERENCE_CONFIG["n_routed_experts"])
        )
    }
    assert set(_tensors(out)) == expected


def test_the_build_quantizes_the_scaled_weights_and_keeps_the_rest(
    v4_build, v4_pruned, v4_reference
):
    from reap.legwork.fp8 import deepseek_name
    from reap.legwork.quant import dequantize_fp8_blocks

    out, _ = v4_build
    built = _tensors(out)
    pruned = {deepseek_name(name): tensor for name, tensor in _tensors(v4_pruned).items()}
    quantized = 0
    for name, tensor in built.items():
        if name.endswith(".scale"):
            continue
        source = pruned[name]
        if QUANTIZED.search(name):
            scale = built[name[: -len(".weight")] + ".scale"]
            assert tensor.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
            rows, cols = tensor.shape
            assert tuple(scale.shape) == (-(-rows // BLOCK), -(-cols // BLOCK))
            error = (dequantize_fp8_blocks(tensor, scale, BLOCK) - source.float()).abs().max()
            assert error <= 0.07 * source.float().abs().max(), name
            quantized += 1
        else:
            assert tensor.dtype == deepseek_dtype(name), name
            assert torch.equal(tensor, source.to(tensor.dtype)), name
    assert quantized > 0


def test_the_build_config_is_the_references_with_the_prunes_edits(v4_build, v4_reference):
    out, _ = v4_build
    config, reference = _config(out), _config(v4_reference[1])
    assert set(config) == set(reference) | {"reap_pruning"}
    changed = {key for key in reference if config[key] != reference[key]}
    assert changed == {"n_routed_experts", "num_nextn_predict_layers", "expert_dtype"}
    assert config["n_routed_experts"] == KEEP
    assert config["num_nextn_predict_layers"] == 0
    assert config["expert_dtype"] == "fp8"
    assert config["reap_pruning"]["keep"] == KEEP
    assert config["reap_pruning"]["fp8_build"]["weight_block_size"] == [BLOCK, BLOCK]
    vllm_config(config)


def test_the_cli_reports_progress_and_a_result(v4_build):
    out, stdout = v4_build
    lines = stdout.splitlines()
    assert any(line.startswith("STAGE_PROGRESS ") for line in lines)
    result = json.loads(
        next(line for line in lines if line.startswith("REAP_RESULT ")).split(" ", 1)[1]
    )
    assert result["out"] == str(out)
    assert result["weight_block_size"] == [BLOCK, BLOCK]
    assert result["quantized"] > 0 and result["tensors"] > result["quantized"]
    assert (
        result["bytes"]
        == json.loads((out / "model.safetensors.index.json").read_text())["metadata"]["total_size"]
    )


def _copy(tree: pathlib.Path, into: pathlib.Path) -> pathlib.Path:
    shutil.copytree(tree, into)
    return into


def _refusal(model: pathlib.Path, source: pathlib.Path, out: pathlib.Path) -> str:
    run = _run(
        "reap.legwork.fp8", "--model", str(model), "--source", str(source), "--out", str(out)
    )
    assert run.returncode != 0
    assert "REAP_RESULT" not in run.stdout
    return run.stderr


def _edit_config(tree: pathlib.Path, **changes) -> None:
    config = _config(tree)
    config.update(changes)
    (tree / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_a_quantized_tree_is_refused(tmp_path, v4_pruned, v4_reference):
    tree = _copy(v4_pruned, tmp_path / "pruned")
    _edit_config(tree, quantization_config={"quant_method": "fp8"})
    assert "already quantized" in _refusal(tree, v4_reference[1], tmp_path / "out")


def test_a_ragged_tree_is_refused(tmp_path, v4_pruned, v4_reference):
    tree = _copy(v4_pruned, tmp_path / "pruned")
    record = _config(tree)["reap_pruning"]
    _edit_config(tree, reap_pruning={**record, "ragged": True})
    assert "ragged" in _refusal(tree, v4_reference[1], tmp_path / "out")


def test_a_reference_without_fp8_blocks_is_refused(tmp_path, v4_pruned, v4_reference):
    reference = _copy(v4_reference[1], tmp_path / "reference")
    config = _config(reference)
    del config["quantization_config"]
    (reference / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert "FP8 blocks" in _refusal(v4_pruned, reference, tmp_path / "out")


def _rewrite(tree: pathlib.Path, edit) -> None:
    from safetensors.torch import save_file

    tensors = _tensors(tree)
    edit(tensors)
    for shard in tree.glob("*.safetensors"):
        shard.unlink()
    (tree / "model.safetensors.index.json").unlink(missing_ok=True)
    save_file(tensors, str(tree / "model.safetensors"), metadata={"format": "pt"})


def test_a_tensor_the_reference_does_not_name_is_refused(tmp_path, v4_pruned, v4_reference):
    tree = _copy(v4_pruned, tmp_path / "pruned")
    _rewrite(tree, lambda t: t.__setitem__("model.layers.0.attn.stray.weight", torch.zeros(2)))
    assert "model.layers.0.attn.stray.weight" in _refusal(tree, v4_reference[1], tmp_path / "out")


def test_a_reference_tensor_the_tree_lacks_is_refused(tmp_path, v4_pruned, v4_reference):
    tree = _copy(v4_pruned, tmp_path / "pruned")
    _rewrite(tree, lambda t: t.pop("model.layers.1.attn.norm.weight"))
    assert "layers.1.attn.kv_norm.weight" in _refusal(tree, v4_reference[1], tmp_path / "out")
