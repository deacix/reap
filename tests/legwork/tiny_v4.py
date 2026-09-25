# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The Legwork lane's CPU fixture: a tiny random DeepSeek-V4 (2 layers,
8 routed experts, hidden 64, layer 0 hash-routed) and a word-level
tokenizer over its vocabulary."""

from __future__ import annotations

import torch

VOCAB = 256
LAYERS = 2
EXPERTS = 8
HIDDEN = 64
TOP_K = 2


def tiny_v4_config(mlp_layer_types=("hash_moe", "moe"), experts=EXPERTS, top_k=TOP_K):
    from transformers import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        moe_intermediate_size=32,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=16,
        num_experts_per_tok=top_k,
        n_routed_experts=experts,
        n_shared_experts=1,
        max_position_embeddings=512,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        sliding_window=16,
        mlp_layer_types=list(mlp_layer_types),
        num_nextn_predict_layers=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def build_tiny_v4(seed: int = 0, experts: int = EXPERTS, top_k: int = TOP_K):
    from transformers import DeepseekV4ForCausalLM

    torch.manual_seed(seed)
    model = DeepseekV4ForCausalLM(tiny_v4_config(experts=experts, top_k=top_k)).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "tid2eid"):
                # A real table never lists one expert twice for a token.
                table = torch.stack(
                    [torch.randperm(experts, generator=generator)[:top_k] for _ in range(VOCAB)]
                )
                gate.tid2eid.copy_(table)
    return model


def build_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for index in range(4, VOCAB):
        vocab[f"t{index}"] = index
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
