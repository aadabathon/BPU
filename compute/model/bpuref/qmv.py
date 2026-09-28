"""QMV reference: quantized matrix-vector product and its stream layouts.

`qmv_ref` defines the arithmetic (docs/numerics.md, "QMV"). The packing
functions define the RTL stream contract (docs/engine-ops.md, "layout L0").
Every hardware configuration must produce `qmv_ref`'s output bit for bit.
"""

from __future__ import annotations

import numpy as np

from .fp import bf16_to_f32

GROUP = 64
W4, W8 = 0, 1
W_RANGE = {W4: (-8, 7), W8: (-128, 127)}


def qmv_ref(w_codes, w_scales, x_codes, x_scales) -> np.ndarray:
    """y[n] = sum over groups g, in increasing g, of f32(isum[n,g]) * (sw[n,g] * sx[g]).

    w_codes:  int [N, K]    int4 or int8 weight codes
    w_scales: uint16 [N, K/64]  bf16 bit patterns, one per (row, group)
    x_codes:  int [K]       int8 activation codes
    x_scales: uint16 [K/64] bf16 bit patterns, one per group
    Returns float32 [N].
    """
    w = np.asarray(w_codes, dtype=np.int64)
    x = np.asarray(x_codes, dtype=np.int64)
    n, k = w.shape
    if k % GROUP or x.shape != (k,):
        raise ValueError(f"K={k} must be a multiple of {GROUP} and match x")
    kg = k // GROUP

    # Exact integer group sums; |isum| <= 2^20, so the int -> fp32 cast is exact.
    isum = np.einsum("ngk,gk->ng", w.reshape(n, kg, GROUP), x.reshape(kg, GROUP))
    sw = bf16_to_f32(w_scales).reshape(n, kg)
    sx = bf16_to_f32(x_scales).reshape(kg)

    with np.errstate(all="ignore"):
        scale = sw * sx[None, :]                 # fp32 mul
        prod = isum.astype(np.float32) * scale    # fp32 mul
        acc = np.zeros(n, dtype=np.float32)
        for g in range(kg):                       # fixed order: increasing g
            acc = acc + prod[:, g]                # fp32 add
    return acc


def _pack(values, bits: int) -> int:
    mask = (1 << bits) - 1
    word = 0
    for i, v in enumerate(values):
        word |= (int(v) & mask) << (bits * i)
    return word


def pack_weight_stream(w_codes, w_scales, lanes: int, row_interleave: int, wfmt: int):
    """Serialize weights into (beats, scales) in layout L0 order.

    for rb < N/R: for g < K/64: for r < R: for c < chunks: beat(row rb*R + r, g, c)
    A beat is Lanes nibbles (Lanes*4 bits): Lanes int4 codes (W4), or Lanes/2
    int8 codes as little-endian bytes (W8). One bf16 scale per (row, group).
    """
    w = np.asarray(w_codes, dtype=np.int64)
    ws = np.asarray(w_scales, dtype=np.uint16)
    n, k = w.shape
    lo, hi = W_RANGE[wfmt]
    if w.min(initial=0) < lo or w.max(initial=0) > hi:
        raise ValueError("weight code out of range for format")
    if n % row_interleave:
        raise ValueError(f"N={n} must be a multiple of RowInterleave={row_interleave}")
    per_beat = lanes if wfmt == W4 else lanes // 2
    bits = 4 if wfmt == W4 else 8
    chunks = GROUP // per_beat

    beats, scales = [], []
    for rb in range(n // row_interleave):
        for g in range(k // GROUP):
            for r in range(row_interleave):
                row = rb * row_interleave + r
                for c in range(chunks):
                    k0 = g * GROUP + c * per_beat
                    beats.append(_pack(w[row, k0:k0 + per_beat], bits))
                scales.append(int(ws[row, g]))
    return beats, scales


def pack_x_words(x_codes, lanes: int) -> list[int]:
    """Activation buffer words: element k lives in word k // Lanes, byte k % Lanes."""
    x = np.asarray(x_codes, dtype=np.int64)
    return [_pack(x[i:i + lanes], 8) for i in range(0, len(x), lanes)]
