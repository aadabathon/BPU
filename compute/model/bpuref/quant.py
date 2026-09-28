"""Placeholder quantizers so tests and experiments can make realistic data.

These are NOT the frozen rules. Weight quantization belongs to ml-models (it
will likely be GPTQ/AWQ), and activation quantization becomes the FVU `quantize`
op once its rounding and clamping are specified (docs/numerics.md, open decisions).
"""

from __future__ import annotations

import numpy as np

from .fp import bf16_to_f32, f32_to_bf16
from .qmv import GROUP, W4, W_RANGE


def quantize_weights_rtn(w, wfmt: int = W4):
    """Symmetric round-to-nearest per (row, 64-group). Returns (codes, bf16 scales)."""
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    lo, hi = W_RANGE[wfmt]
    blocks = w.reshape(n, k // GROUP, GROUP)
    amax = np.abs(blocks).max(axis=2)
    scales = f32_to_bf16(np.where(amax > 0, amax / hi, 1.0).astype(np.float32))
    s = bf16_to_f32(scales)[:, :, None]
    codes = np.clip(np.rint(blocks / s), lo, hi).astype(np.int8)
    return codes.reshape(n, k), scales


def quantize_activations(x):
    """Symmetric absmax per 64-group into [-127, 127]. Returns (codes, bf16 scales)."""
    x = np.asarray(x, dtype=np.float32)
    blocks = x.reshape(-1, GROUP)
    amax = np.abs(blocks).max(axis=1)
    scales = f32_to_bf16(np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32))
    s = bf16_to_f32(scales)[:, None]
    codes = np.clip(np.rint(blocks / s), -127, 127).astype(np.int8)
    return codes.reshape(-1), scales
