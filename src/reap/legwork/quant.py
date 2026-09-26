# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The two storage formats Xiaomi's MiMo-V2 checkpoints ship, both ways.

- **FP8 blocks** (``quantization_config.weight_block_size [128, 128]``):
  an ``F8_E4M3`` weight and an ``F32`` ``weight_scale_inv`` holding one
  multiplier per 128x128 tile (the last tile of a row or a column may be
  partial). ``w = fp8 * scale_inv`` tile by tile.
- **MXFP4** (``store_dtype: mxfp4``, 32-element blocks, the OCP MX spec):
  a ``U8`` weight packing two E2M1 values per byte along the input axis,
  the even element in the low nibble (vLLM's ``_downcast_to_mxfp``:
  ``evens | (odds << 4)``), and a ``U8`` ``weight_scale`` holding one E8M0
  exponent per 32 inputs: ``w = e2m1 * 2 ** (scale - 127)``.

The quantizers exist for the CPU suite's tiny checkpoints; the lane itself
only dequantizes (``reap.legwork.materialize``).
"""

from __future__ import annotations

import torch

FP8_BLOCK = 128
MXFP4_BLOCK = 32
E8M0_BIAS = 127
#: E2M1 codes 0..15: sign bit 3, exponent bits 2-1, mantissa bit 0.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
#: The largest E2M1 exponent (6 = 1.5 * 2**2): the MX shared scale is
#: ``floor(log2(amax)) - 2``.
E2M1_EMAX = 2


def _tile(scale: torch.Tensor, rows: int, cols: int, block: int) -> torch.Tensor:
    """One multiplier per element from a per-tile grid, cropped to the weight."""
    return scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:rows, :cols]


def dequantize_fp8_blocks(
    weight: torch.Tensor, scale_inv: torch.Tensor, block: int = FP8_BLOCK
) -> torch.Tensor:
    """``weight`` (``float8_e4m3fn`` [rows, cols]) times its per-tile
    ``scale_inv`` ([ceil(rows/block), ceil(cols/block)]), in float32."""
    rows, cols = weight.shape
    expected = (-(-rows // block), -(-cols // block))
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"an FP8 weight of shape {tuple(weight.shape)} needs a {expected} scale grid, "
            f"got {tuple(scale_inv.shape)}"
        )
    return weight.to(torch.float32) * _tile(scale_inv.to(torch.float32), rows, cols, block)


def quantize_fp8_blocks(
    weight: torch.Tensor, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8, scale_inv)`` with each tile's amax at the E4M3 maximum."""
    rows, cols = weight.shape
    grid_rows, grid_cols = -(-rows // block), -(-cols // block)
    padded = torch.zeros(grid_rows * block, grid_cols * block, dtype=torch.float32)
    padded[:rows, :cols] = weight.to(torch.float32)
    tiles = padded.view(grid_rows, block, grid_cols, block)
    finfo = torch.finfo(torch.float8_e4m3fn)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale_inv = amax / finfo.max
    quantized = (tiles / scale_inv[:, None, :, None]).clamp(finfo.min, finfo.max)
    fp8 = quantized.view(grid_rows * block, grid_cols * block)[:rows, :cols].to(torch.float8_e4m3fn)
    return fp8.contiguous(), scale_inv.contiguous()


def dequantize_mxfp4(
    packed: torch.Tensor, scales: torch.Tensor, block: int = MXFP4_BLOCK
) -> torch.Tensor:
    """``packed`` (``uint8`` [rows, cols/2]) and its E8M0 ``scales``
    (``uint8`` [rows, cols/block]) as a float32 [rows, cols] weight."""
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(f"MXFP4 tensors are uint8, got {packed.dtype} and {scales.dtype}")
    rows, half = packed.shape
    cols = half * 2
    if cols % block or tuple(scales.shape) != (rows, cols // block):
        raise ValueError(
            f"an MXFP4 weight of {rows} x {cols} needs a ({rows}, {cols // block}) scale grid, "
            f"got {tuple(scales.shape)}"
        )
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    values = torch.stack((table[low], table[high]), dim=-1).reshape(rows, cols)
    exponent = scales.to(torch.int32) - E8M0_BIAS
    multiplier = torch.pow(2.0, exponent.to(torch.float32)).repeat_interleave(block, dim=1)
    return values * multiplier


def quantize_mxfp4(
    weight: torch.Tensor, block: int = MXFP4_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(packed, scales)`` in the OCP MX layout: per 32 inputs one E8M0
    exponent ``floor(log2(amax)) - 2``, each value rounded to the nearest
    E2M1 code (ties to the even code)."""
    rows, cols = weight.shape
    if cols % block or cols % 2:
        raise ValueError(f"MXFP4 packs rows of a multiple of {block} inputs, got {cols}")
    blocks = weight.to(torch.float32).view(rows, cols // block, block)
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp(min=torch.finfo(torch.float32).tiny))) - E2M1_EMAX,
        torch.full_like(amax, -E8M0_BIAS),
    ).clamp(-E8M0_BIAS, E8M0_BIAS)
    scaled = blocks / torch.pow(2.0, exponent)[..., None]
    magnitudes = torch.tensor(E2M1_VALUES[:8], dtype=torch.float32)
    distance = (scaled.abs()[..., None] - magnitudes).abs()
    # The nearest magnitude; an exact tie sits between two neighbouring codes
    # and goes to the even one.
    tied = distance == distance.min(dim=-1, keepdim=True).values
    even = tied & (torch.arange(8) % 2 == 0)
    code = torch.where(even.any(dim=-1), even.float().argmax(dim=-1), tied.float().argmax(dim=-1))
    sign = (scaled < 0) & (code > 0)
    codes = (code + 8 * sign.long()).view(rows, cols).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = (exponent + E8M0_BIAS).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()
