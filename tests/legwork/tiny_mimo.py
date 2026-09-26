# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The Legwork lane's MiMo-V2 CPU fixture.

``build_tiny_mimo`` builds a tiny random MiMo-V2 from Xiaomi's own model
code (``fixtures/mimo_v2``, vendored at
XiaomiMiMo/MiMo-V2.6-Flash-RL@5711b26) through transformers'
trust_remote_code path, in the V2.6 shape: a dense layer 0, then two
sliding-window MoE layers with twice the full-attention KV heads, 8 routed
experts at top-2, attention sinks and a partial rotary.

``write_hub_checkpoint`` stores it the way the Hub checkpoint is stored: a
fused FP8 ``qkv_proj`` in ``TP`` chunks, FP8 blocks for the dense MLP,
MXFP4 experts, BF16 norms, router, ``o_proj`` and embeddings, an F32
``e_score_correction_bias``, multi-token-prediction tensors the model code
ignores, a ``dflash/`` folder, two shards and an index carrying
``save_format: mxfp4`` and ``tp_size``. ``quantize_dequantize`` is the same
model with every stored weight passed through its format, the reference a
working copy must reproduce.
"""

from __future__ import annotations

import copy
import json
import pathlib
import shutil
import tempfile

import torch

from reap.legwork.compat import patch_remote_code
from reap.legwork.qkv import fuse_qkv, layer_geometry, split_fused_qkv
from reap.legwork.quant import (
    dequantize_fp8_blocks,
    dequantize_mxfp4,
    quantize_fp8_blocks,
    quantize_mxfp4,
)
from tests.legwork.tiny_v4 import build_tiny_tokenizer

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "mimo_v2"
CODE_FILES = ("configuration_mimo_v2.py", "modeling_mimo_v2.py")
TP = 2
EXPERTS = 8
TOP_K = 2
MOE_LAYERS = (1, 2)

TINY_CONFIG = {
    "architectures": ["MiMoV2ForCausalLM"],
    "model_type": "mimo_v2",
    "auto_map": {
        "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
        "AutoModel": "modeling_mimo_v2.MiMoV2Model",
        "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
    },
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 48,
    "v_head_dim": 32,
    "swa_num_attention_heads": 8,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 48,
    "swa_v_head_dim": 32,
    "sliding_window": 16,
    "sliding_window_size": 16,
    "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True,
    "hybrid_layer_pattern": [0, 1, 1],
    "partial_rotary_factor": 0.334,
    "attention_value_scale": 0.707,
    "n_routed_experts": EXPERTS,
    "moe_intermediate_size": 64,
    "num_experts_per_tok": TOP_K,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "moe_layer_freq": [0, 1, 1],
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "swa_rope_theta": 10000.0,
    "layernorm_epsilon": 1e-6,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "num_nextn_predict_layers": 1,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "dtype": "bfloat16",
}

QUANTIZATION_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "store_dtype": "mxfp4",
    "mxfp4_block_size": 32,
    "weight_block_size": [128, 128],
    "ignored_layers": [f"model.layers.{i}.self_attn.o_proj" for i in range(3)],
}


def write_model_code(out: pathlib.Path) -> None:
    for name in CODE_FILES:
        shutil.copy2(FIXTURE_DIR / name, out / name)


def build_tiny_tokenizer_with_template():
    tokenizer = build_tiny_tokenizer()
    tokenizer.chat_template = (FIXTURE_DIR / "chat_template.jinja").read_text(encoding="utf-8")
    return tokenizer


def build_tiny_mimo(seed: int = 0):
    """The tiny BF16 MiMo-V2 (split attention projections), every weight
    seeded; its model code comes from a scratch directory holding the
    vendored files, as a working copy's does."""
    from transformers import AutoConfig, AutoModelForCausalLM

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="tiny-mimo-code-"))
    write_model_code(scratch)
    config = dict(TINY_CONFIG, attention_projection_layout="split")
    (scratch / "config.json").write_text(json.dumps(config), encoding="utf-8")
    auto_config = AutoConfig.from_pretrained(scratch, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(auto_config, trust_remote_code=True, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() >= 2:
                noise = torch.randn(parameter.shape, generator=generator) * 0.08
            elif name.endswith("norm.weight"):
                noise = 1.0 + torch.randn(parameter.shape, generator=generator) * 0.05
            else:
                noise = torch.randn(parameter.shape, generator=generator) * 0.1
            parameter.copy_(noise.to(parameter.dtype))
    patch_remote_code(model)
    return model.eval()


def _hub_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The Hub checkpoint's tensors for a split BF16 state dict."""
    tensors: dict[str, torch.Tensor] = {}
    config = TINY_CONFIG
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        q = state.pop(prefix + "self_attn.q_proj.weight").float()
        k = state.pop(prefix + "self_attn.k_proj.weight").float()
        v = state.pop(prefix + "self_attn.v_proj.weight").float()
        fused, scale = fuse_qkv(q, k, v, layer_geometry(config, layer, TP))
        tensors[prefix + "self_attn.qkv_proj.weight"] = fused
        tensors[prefix + "self_attn.qkv_proj.weight_scale_inv"] = scale
    for name in list(state):
        tensor = state[name]
        if ".mlp.experts." in name and name.endswith("_proj.weight"):
            packed, scales = quantize_mxfp4(tensor.float())
            tensors[name] = packed
            tensors[name[: -len(".weight")] + ".weight_scale"] = scales
        elif name.startswith("model.layers.0.mlp.") and name.endswith("_proj.weight"):
            fp8, scale = quantize_fp8_blocks(tensor.float())
            tensors[name] = fp8
            tensors[name[: -len(".weight")] + ".weight_scale_inv"] = scale
        elif name.endswith("e_score_correction_bias"):
            tensors[name] = tensor.float()
        else:
            tensors[name] = tensor
    # One multi-token-prediction layer the model code never builds.
    generator = torch.Generator().manual_seed(99)
    mtp_fp8, mtp_scale = quantize_fp8_blocks(torch.randn(256, 128, generator=generator))
    tensors["model.mtp.layers.0.mlp.gate_proj.weight"] = mtp_fp8
    tensors["model.mtp.layers.0.mlp.gate_proj.weight_scale_inv"] = mtp_scale
    tensors["model.mtp.layers.0.enorm.weight"] = torch.ones(128, dtype=torch.bfloat16)
    return tensors


def write_hub_checkpoint(model, out: pathlib.Path | str) -> pathlib.Path:
    """``model`` stored in the Hub checkpoint's formats, in two shards."""
    from safetensors.torch import save_file

    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    tensors = _hub_tensors(state)
    names = sorted(tensors)
    halves = (names[: len(names) // 2], names[len(names) // 2 :])
    weight_map: dict[str, str] = {}
    total = 0
    for number, part in enumerate(halves, start=1):
        shard = f"model_pp0_ep0_shard{number - 1}.safetensors"
        save_file({name: tensors[name].contiguous() for name in part}, str(out / shard))
        for name in part:
            weight_map[name] = shard
            total += tensors[name].numel() * tensors[name].element_size()
    index = {"metadata": {"save_format": "mxfp4", "total_size": total, "tp_size": TP}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    config = dict(TINY_CONFIG, attention_projection_layout="fused_qkv", quantization_config=QUANTIZATION_CONFIG)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "generation_config.json").write_text(json.dumps({"bos_token_id": 1, "eos_token_id": 2}), encoding="utf-8")
    write_model_code(out)
    build_tiny_tokenizer_with_template().save_pretrained(out)
    (out / "dflash").mkdir(exist_ok=True)
    (out / "dflash" / "config.json").write_text(json.dumps({"drafter": "dflash"}), encoding="utf-8")
    return out


def quantize_dequantize(model):
    """``model`` with every stored weight passed through its Hub format and
    back: the weights a correct working copy holds."""
    reference = copy.deepcopy(model)
    config = TINY_CONFIG
    with torch.no_grad():
        for layer in range(config["num_hidden_layers"]):
            attention = reference.model.layers[layer].self_attn
            geometry = layer_geometry(config, layer, TP)
            fused, scale = fuse_qkv(
                attention.q_proj.weight.float(),
                attention.k_proj.weight.float(),
                attention.v_proj.weight.float(),
                geometry,
            )
            q, k, v = split_fused_qkv(fused, scale, geometry)
            for projection, dense in ((attention.q_proj, q), (attention.k_proj, k), (attention.v_proj, v)):
                projection.weight.copy_(dense.to(projection.weight.dtype))
        for name, parameter in reference.named_parameters():
            if ".mlp.experts." in name and name.endswith("_proj.weight"):
                dense = dequantize_mxfp4(*quantize_mxfp4(parameter.float()))
            elif name.startswith("model.layers.0.mlp.") and name.endswith("_proj.weight"):
                dense = dequantize_fp8_blocks(*quantize_fp8_blocks(parameter.float()))
            else:
                continue
            parameter.copy_(dense.to(parameter.dtype))
    return reference.eval()
