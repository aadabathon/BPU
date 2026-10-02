"""Quantizers.

quantize_activations is the hardware's activation quantization exactly: it runs the
same FVU op sequence the compiler emits (bpuref.qwen.Compiler._quant):
RAMAX -> VMULS fp32(1/127) -> VRBF16 -> SFU rcp -> VMULG -> VQCLAMP. An all-zero
group gets scale 0 and codes 0.

quantize_weights_rtn is a placeholder so tests can make realistic weights; weight
quantization belongs to ml-models (likely GPTQ/AWQ) and only has to deliver codes
and bf16 scales in the agreed format (docs/numerics.md).
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
    """a8 codes and bf16 scales of x (length a multiple of 64), bit-exact with the
    compiled FVU sequence. Returns (int8 codes, uint16 bf16 scales)."""
    from . import fvu as F
    from . import sfu
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    k = x.size
    g = k // GROUP
    gp = -(-g // 64) * 64
    sc, t, codes, inv127 = k, k + gp, k + 2 * gp, 2 * k + 2 * gp
    spm = np.zeros(inv127 + 64, dtype=np.float32)
    spm[:k] = x
    spm[inv127] = np.float32(1.0 / 127.0)
    for op in (F.FvuOp(F.RAMAX, g, GROUP, d=sc, a=0, a_stride=GROUP, d_stride=1),
               F.FvuOp(F.VMULS, 1, g, d=sc, a=sc, s=inv127),
               F.FvuOp(F.VRBF16, 1, g, d=sc, a=sc),
               F.FvuOp(F.VSFU, 1, g, d=t, a=sc, func=sfu.RCP),
               F.FvuOp(F.VMULG, 1, k, d=codes, a=0, t=t),
               F.FvuOp(F.VQCLAMP, 1, k, d=codes, a=codes)):
        F.execute(op, spm)
    c = spm[codes:codes + k]
    return c.astype(np.int8), f32_to_bf16(spm[sc:sc + g])
