# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""MiMo-V2's storage formats: MXFP4 experts, FP8 blocks and the fused,
TP-chunked ``qkv_proj``, each against an independent reading."""

from __future__ import annotations

import pytest
import torch

from reap.legwork.qkv import QkvGeometry, fuse_qkv, layer_geometry, split_fused_qkv
from reap.legwork.quant import (
    E2M1_VALUES,
    dequantize_fp8_blocks,
    dequantize_mxfp4,
    quantize_fp8_blocks,
    quantize_mxfp4,
)


def test_mxfp4_packs_the_even_element_in_the_low_nibble():
    # vLLM's `_downcast_to_mxfp`: `evens | (odds << 4)`; E8M0 127 is 2**0.
    packed = torch.tensor([[0x21, 0xF9]], dtype=torch.uint8).repeat(1, 8)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    dense = dequantize_mxfp4(packed, scales)
    assert dense[0, :4].tolist() == [0.5, 1.0, -0.5, -6.0]
    assert dequantize_mxfp4(packed, scales + 3)[0, 1].item() == 8.0


def test_mxfp4_roundtrip_is_exact_on_representable_values():
    generator = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (8, 64), generator=generator)
    exponents = torch.randint(-6, 6, (8, 2), generator=generator)
    values = torch.tensor(E2M1_VALUES)[codes]
    # Each 32-block reaches 6.0 so its shared exponent is recovered exactly.
    values[:, 0] = 6.0
    values[:, 32] = -6.0
    weight = values * torch.pow(2.0, exponents.float()).repeat_interleave(32, dim=1)
    packed, scales = quantize_mxfp4(weight)
    assert packed.shape == (8, 32) and scales.shape == (8, 2)
    assert torch.equal(dequantize_mxfp4(packed, scales), weight)


def test_mxfp4_rounds_to_the_nearest_code_and_ties_to_even():
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(16, 96, generator=generator)
    packed, scales = quantize_mxfp4(weight)
    back = dequantize_mxfp4(packed, scales)
    step = torch.pow(2.0, scales.float() - 127).repeat_interleave(32, dim=1)
    # The nearest code is at most half the widest gap (4 -> 6) away, and the
    # MX exponent `floor(log2(amax)) - 2` saturates a block's top values
    # between 6 and 8 units at 6: never more than 2 units off.
    assert ((back - weight).abs() <= step * 2.0 + 1e-7).all()
    ties = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] + [6.0] * 25])
    tied_packed, tied_scales = quantize_mxfp4(ties)
    assert dequantize_mxfp4(tied_packed, tied_scales)[0, :7].tolist() == [
        0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0,
    ]


def test_fp8_blocks_roundtrip_with_partial_tiles():
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(200, 300, generator=generator)
    fp8, scale = quantize_fp8_blocks(weight)
    assert fp8.dtype == torch.float8_e4m3fn and scale.shape == (2, 3)
    back = dequantize_fp8_blocks(fp8, scale)
    assert torch.allclose(back, weight, rtol=0.07, atol=1e-3)
    with pytest.raises(ValueError, match="scale grid"):
        dequantize_fp8_blocks(fp8, scale[:1])


def _sglang_deinterleave(weight, scale, tp, q_rows, k_rows, v_rows, block=128):
    """SGLang 0.5.20's reading (`_resolve_deferred_qkv_scale_inv` and
    `_deinterleave_qkv_shards`), restated: chunk both tensors by the
    checkpoint's TP, dequantize each chunk on its own grid, gather Q, K, V."""
    dense = []
    for w, s in zip(weight.chunk(tp, dim=0), scale.chunk(tp, dim=0)):
        expanded = s.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
        dense.append(w.float() * expanded[: w.shape[0], : w.shape[1]])
    qs = [d[:q_rows] for d in dense]
    ks = [d[q_rows : q_rows + k_rows] for d in dense]
    vs = [d[q_rows + k_rows :] for d in dense]
    return torch.cat(qs + ks + vs)


@pytest.mark.parametrize(
    ("kv_heads", "rows", "scale_rows"),
    [(4, 13568, 108), (8, 14848, 116)],
    ids=["full-attention", "sliding-window"],
)
def test_deinterleaves_tp_chunked_qkv_like_sglang(kv_heads, rows, scale_rows):
    # MiMo-V2.6-Flash's two attention shapes at its index's tp_size 4, over a
    # narrow hidden size; the row counts and scale rows are the checkpoint's.
    geometry = QkvGeometry(heads=64, kv_heads=kv_heads, head_dim=192, v_head_dim=128, tp=4)
    generator = torch.Generator().manual_seed(kv_heads)
    q = torch.randn(64 * 192, 128, generator=generator)
    k = torch.randn(kv_heads * 192, 128, generator=generator)
    v = torch.randn(kv_heads * 128, 128, generator=generator)
    fused, scale = fuse_qkv(q, k, v, geometry)
    assert tuple(fused.shape) == (rows, 128) and tuple(scale.shape) == (scale_rows, 1)
    split_q, split_k, split_v = split_fused_qkv(fused, scale, geometry)
    reference = _sglang_deinterleave(
        fused, scale, 4, geometry.q_rows, geometry.k_rows, geometry.v_rows
    )
    assert torch.equal(torch.cat((split_q, split_k, split_v)), reference)
    for split, original in ((split_q, q), (split_k, k), (split_v, v)):
        assert torch.allclose(split, original, rtol=0.07, atol=1e-2)


def test_layer_geometry_reads_the_hybrid_pattern():
    config = {
        "hybrid_layer_pattern": [0, 1],
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 192,
        "v_head_dim": 128,
        "swa_num_attention_heads": 64,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 192,
        "swa_v_head_dim": 128,
    }
    assert layer_geometry(config, 0, 4).rows_per_chunk == 3392
    assert layer_geometry(config, 1, 4).rows_per_chunk == 3712
    with pytest.raises(ValueError, match="divisible"):
        layer_geometry(config, 1, 3)
