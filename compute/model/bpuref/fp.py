"""Bit-level helpers for the BPU number formats (docs/numerics.md).

fp32 arithmetic in the reference is plain numpy float32: IEEE-754 binary32 with
round-to-nearest-even and subnormals, which is exactly what the RTL implements.
The only difference is NaN encoding: the RTL always emits the canonical quiet
NaN, so comparisons go through `f32_equal`.
"""

import numpy as np

F32_QNAN = 0x7FC00000
BF16_QNAN = 0x7FC0


def f32_from_bits(bits) -> np.ndarray:
    """Reinterpret uint32 bit patterns as float32."""
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def f32_bits(x) -> np.ndarray:
    """Reinterpret float32 values as uint32 bit patterns."""
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def bf16_to_f32(b) -> np.ndarray:
    """Exact bf16 -> fp32 widening (bf16 is the top half of binary32)."""
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x) -> np.ndarray:
    """fp32 -> bf16, round to nearest even; NaNs become the canonical bf16 NaN."""
    u = f32_bits(x).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    is_nan = ((u >> 23) & 0xFF == 0xFF) & ((u & 0x7FFFFF) != 0)
    return np.where(is_nan, BF16_QNAN, rounded).astype(np.uint16)


def f32_equal(rtl_bits, ref) -> np.ndarray:
    """Element-wise bit equality under the RTL NaN rule (any NaN -> canonical)."""
    rtl_bits = np.asarray(rtl_bits, dtype=np.uint32)
    ref = np.asarray(ref, dtype=np.float32)
    return np.where(np.isnan(ref), rtl_bits == F32_QNAN, rtl_bits == f32_bits(ref))
