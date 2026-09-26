# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""MiMo-V2's fused ``qkv_proj``, pre-sharded for tensor parallelism.

A checkpoint whose ``config.json`` names ``attention_projection_layout:
"fused_qkv"`` stores each layer's Q, K and V as one FP8 tensor cut into
``tp`` chunks (``model.safetensors.index.json`` ``metadata.tp_size``: 4 on
MiMo-V2.6-Flash, 8 on Pro), one per rank of the TP layout it was exported
for. Each chunk holds that rank's heads as ``[Q_r | K_r | V_r]``:

    [Q_0 | K_0 | V_0 | Q_1 | K_1 | V_1 | ... | Q_{tp-1} | K_{tp-1} | V_{tp-1}]

with ``(heads / tp) * head_dim`` Q rows, ``(kv_heads / tp) * head_dim`` K
rows and ``(kv_heads / tp) * v_head_dim`` V rows, and its 128x128 FP8
scales are tiled on the chunk's own grid (``ceil(rows_per_chunk / 128)``
scale rows per chunk). A MiMo-V2.6-Flash full-attention layer (4 KV heads)
is 4 x 3392 rows with a ``[108, 32]`` scale; a sliding-window layer (8 KV
heads) is 4 x 3712 rows with ``[116, 32]``. This is SGLang 0.5.20's
reading (``_deinterleave_qkv_shards``) and vLLM's since
vllm-project/vllm#57508.

The working copy splits it into the ``q_proj`` / ``k_proj`` / ``v_proj``
the checkpoint's own ``modeling_mimo_v2.py`` builds under
``attention_projection_layout: "split"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from reap.legwork.quant import FP8_BLOCK, dequantize_fp8_blocks, quantize_fp8_blocks


@dataclass(frozen=True)
class QkvGeometry:
    """One attention layer's heads and the TP layout its fused tensor was cut for."""

    heads: int
    kv_heads: int
    head_dim: int
    v_head_dim: int
    tp: int

    def __post_init__(self) -> None:
        if self.tp < 1 or self.heads % self.tp or self.kv_heads % self.tp:
            raise ValueError(
                f"a fused qkv_proj cut for tp={self.tp} needs heads ({self.heads}) and "
                f"kv heads ({self.kv_heads}) divisible by it"
            )

    @property
    def q_rows(self) -> int:
        return (self.heads // self.tp) * self.head_dim

    @property
    def k_rows(self) -> int:
        return (self.kv_heads // self.tp) * self.head_dim

    @property
    def v_rows(self) -> int:
        return (self.kv_heads // self.tp) * self.v_head_dim

    @property
    def rows_per_chunk(self) -> int:
        return self.q_rows + self.k_rows + self.v_rows

    def scale_rows_per_chunk(self, block: int = FP8_BLOCK) -> int:
        return -(-self.rows_per_chunk // block)


def layer_geometry(config: dict, layer: int, tp: int) -> QkvGeometry:
    """The geometry of ``layer``'s attention: its ``hybrid_layer_pattern``
    entry picks the sliding-window (``1``) or full-attention (``0``) heads."""
    pattern = config.get("hybrid_layer_pattern") or []
    swa = layer < len(pattern) and int(pattern[layer]) == 1
    if swa:
        head_dim = int(config.get("swa_head_dim") or config["head_dim"])
        return QkvGeometry(
            heads=int(config.get("swa_num_attention_heads") or config["num_attention_heads"]),
            kv_heads=int(config.get("swa_num_key_value_heads") or config["num_key_value_heads"]),
            head_dim=head_dim,
            v_head_dim=int(config.get("swa_v_head_dim") or head_dim),
            tp=tp,
        )
    head_dim = int(config["head_dim"])
    return QkvGeometry(
        heads=int(config["num_attention_heads"]),
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=head_dim,
        v_head_dim=int(config.get("v_head_dim") or head_dim),
        tp=tp,
    )


def split_fused_qkv(
    weight: torch.Tensor, scale_inv: torch.Tensor, geometry: QkvGeometry, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize each chunk on its own scale grid and gather every chunk's
    Q, then K, then V: the split projections in head order, float32."""
    rows = geometry.rows_per_chunk
    scale_rows = geometry.scale_rows_per_chunk(block)
    if weight.shape[0] != rows * geometry.tp:
        raise ValueError(
            f"a fused qkv_proj of {weight.shape[0]} rows is not {geometry.tp} chunks of {rows}"
        )
    if scale_inv.shape[0] != scale_rows * geometry.tp:
        raise ValueError(
            f"its scale has {scale_inv.shape[0]} rows, not {geometry.tp} chunks of {scale_rows}"
        )
    qs, ks, vs = [], [], []
    for chunk in range(geometry.tp):
        w = weight[chunk * rows : (chunk + 1) * rows]
        s = scale_inv[chunk * scale_rows : (chunk + 1) * scale_rows]
        dense = dequantize_fp8_blocks(w, s, block)
        qs.append(dense[: geometry.q_rows])
        ks.append(dense[geometry.q_rows : geometry.q_rows + geometry.k_rows])
        vs.append(dense[geometry.q_rows + geometry.k_rows :])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def fuse_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, geometry: QkvGeometry, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """The inverse, for the CPU suite's tiny checkpoints: cut split Q, K and
    V into ``tp`` chunks, quantize each chunk on its own grid, concatenate."""
    weights, scales = [], []
    for chunk in range(geometry.tp):
        dense = torch.cat(
            (
                q[chunk * geometry.q_rows : (chunk + 1) * geometry.q_rows],
                k[chunk * geometry.k_rows : (chunk + 1) * geometry.k_rows],
                v[chunk * geometry.v_rows : (chunk + 1) * geometry.v_rows],
            )
        )
        fp8, scale = quantize_fp8_blocks(dense, block)
        weights.append(fp8)
        scales.append(scale)
    return torch.cat(weights), torch.cat(scales)
