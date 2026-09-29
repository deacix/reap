# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""A tiny DeepSeek-V4 stored the way DeepSeek ships it (the FP8 build's CPU
fixture): DeepSeek's tensor names and dtypes, the reference's config words
(``num_hash_layers``, ``compress_ratios``, ``rope_scaling``, ``expert_dtype``),
three layers (sliding, compressed-sparse with an indexer, heavily compressed)
and 8 routed experts. Two copies of one model:

- ``float/``: every weight in BF16 and no ``quantization_config``, the model
  the collect's statistics are read over;
- ``reference/``: DeepSeek-V4-Flash's own storage, the checkpoint a ROCm prune
  dequantizes (``--dequantize``) and the FP8 build reads names, dtypes and
  config from: attention and shared-expert weights as ``F8_E4M3`` tiles with
  power-of-two ``F8_E8M0`` ``.scale`` grids, routed experts as ``I8``-packed
  E2M1 pairs with one ``F8_E8M0`` ``.scale`` per 32 inputs, and one draft-layer
  (``mtp.``) expert the prune never carries.
"""

from __future__ import annotations

import json
import pathlib
import re
import tempfile

import torch

from tests.legwork.tiny_v4 import build_tiny_tokenizer

#: DeepSeek-V4-Flash-0731's tensor names (revision 7872f01b, its
#: ``model.safetensors.index.json``) with layer and expert ids as ``N`` and the
#: draft layers left out: the build must write exactly these shapes of name.
DEEPSEEK_V4_SCHEMA = (
    "embed.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "head.weight",
    "layers.N.attn.attn_sink",
    "layers.N.attn.compressor.ape",
    "layers.N.attn.compressor.norm.weight",
    "layers.N.attn.compressor.wgate.weight",
    "layers.N.attn.compressor.wkv.weight",
    "layers.N.attn.indexer.compressor.ape",
    "layers.N.attn.indexer.compressor.norm.weight",
    "layers.N.attn.indexer.compressor.wgate.weight",
    "layers.N.attn.indexer.compressor.wkv.weight",
    "layers.N.attn.indexer.weights_proj.weight",
    "layers.N.attn.indexer.wq_b.scale",
    "layers.N.attn.indexer.wq_b.weight",
    "layers.N.attn.kv_norm.weight",
    "layers.N.attn_norm.weight",
    "layers.N.attn.q_norm.weight",
    "layers.N.attn.wkv.scale",
    "layers.N.attn.wkv.weight",
    "layers.N.attn.wo_a.scale",
    "layers.N.attn.wo_a.weight",
    "layers.N.attn.wo_b.scale",
    "layers.N.attn.wo_b.weight",
    "layers.N.attn.wq_a.scale",
    "layers.N.attn.wq_a.weight",
    "layers.N.attn.wq_b.scale",
    "layers.N.attn.wq_b.weight",
    "layers.N.ffn.experts.N.w1.scale",
    "layers.N.ffn.experts.N.w1.weight",
    "layers.N.ffn.experts.N.w2.scale",
    "layers.N.ffn.experts.N.w2.weight",
    "layers.N.ffn.experts.N.w3.scale",
    "layers.N.ffn.experts.N.w3.weight",
    "layers.N.ffn.gate.bias",
    "layers.N.ffn.gate.tid2eid",
    "layers.N.ffn.gate.weight",
    "layers.N.ffn_norm.weight",
    "layers.N.ffn.shared_experts.w1.scale",
    "layers.N.ffn.shared_experts.w1.weight",
    "layers.N.ffn.shared_experts.w2.scale",
    "layers.N.ffn.shared_experts.w2.weight",
    "layers.N.ffn.shared_experts.w3.scale",
    "layers.N.ffn.shared_experts.w3.weight",
    "layers.N.hc_attn_base",
    "layers.N.hc_attn_fn",
    "layers.N.hc_attn_scale",
    "layers.N.hc_ffn_base",
    "layers.N.hc_ffn_fn",
    "layers.N.hc_ffn_scale",
    "norm.weight",
)

#: The weights DeepSeek stores with a ``.scale`` (the schema's ``.scale`` rows).
QUANTIZED = re.compile(
    r"\.(attn\.(wq_a|wq_b|wkv|wo_a|wo_b)|attn\.indexer\.wq_b"
    r"|ffn\.shared_experts\.w[123]|ffn\.experts\.\d+\.w[123])\.weight$"
)

BLOCK = 16
EXPERTS = 8

#: The reference's config: DeepSeek-V4-Flash-0731's own keys and words, tiny.
REFERENCE_CONFIG: dict = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "expert_dtype": "fp4",
    "hc_eps": 1e-06,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 20,
    "head_dim": 32,
    "hidden_act": "silu",
    "hidden_size": 64,
    "index_head_dim": 16,
    "index_n_heads": 2,
    "index_topk": 8,
    "initializer_range": 0.02,
    "max_position_embeddings": 512,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 32,
    "n_routed_experts": EXPERTS,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 3,
    "num_hash_layers": 1,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 2,
    "o_lora_rank": 16,
    "q_lora_rank": 16,
    "qk_rope_head_dim": 8,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [BLOCK, BLOCK],
    },
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 2,
        "original_max_position_embeddings": 256,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 16,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": 256,
    "compress_rope_theta": 160000,
    "compress_ratios": [0, 4, 128],
    "dspark_block_size": 5,
    "dspark_noise_token_id": 250,
    "dspark_target_layer_ids": [0, 1, 2],
    "dspark_markov_rank": 16,
}

_F32 = re.compile(r"(attn_sink|\.ape|gate\.bias|hc_)")


def deepseek_dtype(name: str) -> torch.dtype:
    """The dtype DeepSeek stores a non-quantized tensor in."""
    if name.endswith("tid2eid"):
        return torch.int64
    return torch.float32 if _F32.search(name) else torch.bfloat16


def _deepseek_name(saved: str) -> str:
    """A transformers-saved name (``model.``-prefixed, a few leaves renamed)
    as DeepSeek names it; checked against ``DEEPSEEK_V4_SCHEMA`` by the suite."""
    if saved == "model.embed_tokens.weight":
        return "embed.weight"
    head = re.fullmatch(r"model\.hc_head\.hc_(fn|base|scale)", saved)
    if head:
        return f"hc_head_{head.group(1)}"
    name = re.sub(r"^model\.", "", saved)
    return re.sub(r"^(layers\.\d+\.attn)\.norm\.weight$", r"\1.kv_norm.weight", name)


def schema_of(names) -> set[str]:
    return {re.sub(r"\.\d+\.", ".N.", name) for name in names if not name.startswith("mtp.")}


def fp8_ue8m0(weight: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """``(F8_E4M3 tiles, F8_E8M0 .scale)`` with each tile's scale rounded up
    to a power of two, as DeepSeek-V4-Flash stores its FP8 weights."""
    rows, cols = weight.shape
    grid_rows, grid_cols = -(-rows // block), -(-cols // block)
    tiles = weight.float().reshape(grid_rows, block, grid_cols, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max)))
    fp8 = (tiles / scale[:, None, :, None]).reshape(rows, cols).to(torch.float8_e4m3fn)
    return fp8.contiguous(), scale.to(torch.float8_e8m0fnu).contiguous()


def fp4_experts(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(I8 packed E2M1 pairs, F8_E8M0 .scale per 32 inputs)``: the OCP MX
    layout of ``reap.legwork.quant`` in the dtypes DeepSeek-V4-Flash stores."""
    from reap.legwork.quant import quantize_mxfp4

    packed, scales = quantize_mxfp4(weight.float())
    return packed.view(torch.int8), scales.view(torch.float8_e8m0fnu)


def _deepseek_storage(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    stem = name[: -len(".weight")]
    weight, scale = fp4_experts(tensor) if ".experts." in name else fp8_ue8m0(tensor)
    return {name: weight, stem + ".scale": scale}


def write_reference(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """``(float_dir, reference_dir)`` of one seeded tiny V4 under ``root``."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    quantized_keys = ("quantization_config", "expert_dtype")
    float_config = {k: v for k, v in REFERENCE_CONFIG.items() if k not in quantized_keys}
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(DeepseekV4Config.from_dict(float_config)).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for layer in model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "tid2eid"):
                table = torch.stack(
                    [torch.randperm(EXPERTS, generator=generator)[:2] for _ in range(256)]
                )
                gate.tid2eid.copy_(table)
    with tempfile.TemporaryDirectory() as scratch:
        model.save_pretrained(scratch)
        tensors = {}
        for shard in sorted(pathlib.Path(scratch).glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as handle:
                saved_names = handle.keys()
                for saved in saved_names:
                    name = _deepseek_name(saved)
                    tensor = handle.get_tensor(saved).to(deepseek_dtype(name))
                    tensors[name] = tensor.contiguous()

    float_dir, reference_dir = root / "float", root / "reference"
    float_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    save_file(tensors, str(float_dir / "model.safetensors"), metadata={"format": "pt"})
    (float_dir / "config.json").write_text(json.dumps(float_config, indent=2), encoding="utf-8")
    build_tiny_tokenizer().save_pretrained(float_dir)

    reference: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if QUANTIZED.search(name):
            reference.update(_deepseek_storage(name, tensor))
        else:
            reference[name] = tensor
    draft = tensors["layers.1.ffn.experts.0.w1.weight"]
    reference.update(_deepseek_storage("mtp.0.ffn.experts.0.w1.weight", draft))
    save_file(reference, str(reference_dir / "model.safetensors"), metadata={"format": "pt"})
    (reference_dir / "config.json").write_text(
        json.dumps(REFERENCE_CONFIG, indent=2), encoding="utf-8"
    )
    build_tiny_tokenizer().save_pretrained(reference_dir)
    return float_dir, reference_dir
