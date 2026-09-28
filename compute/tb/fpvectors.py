"""fp32 operand generators aimed at the corners of IEEE-754 rounding.

Each generator returns (a, b) as uint32 bit-pattern arrays. The mix covers the
special values, uniform random bit patterns, and targeted regions: ties,
cancellation, subnormal inputs and outputs, and the overflow boundary.
"""

from __future__ import annotations

import numpy as np

SPECIALS = np.array(
    [
        0x00000000, 0x80000000,              # +-0
        0x7F800000, 0xFF800000,              # +-inf
        0x7FC00000, 0x7F800001, 0xFFC12345,  # qNaN, sNaN, negative NaN with payload
        0x00000001, 0x80000001,              # +-min subnormal
        0x007FFFFF, 0x807FFFFF,              # +-max subnormal
        0x00800000, 0x80800000,              # +-min normal
        0x7F7FFFFF, 0xFF7FFFFF,              # +-max normal
        0x3F800000, 0xBF800000,              # +-1
        0x3F800001, 0x3F7FFFFF,              # 1 + ulp, 1 - ulp/2
        0x3FC00000, 0x40000000,              # 1.5, 2
        0x00400000, 0x34000000, 0x4B800000,  # 2^-127, 2^-23, 2^24
    ],
    dtype=np.uint32,
)


def _compose(sign, exp, man) -> np.ndarray:
    return ((sign.astype(np.uint32) << 31) | (exp.astype(np.uint32) << 23)
            | man.astype(np.uint32)).astype(np.uint32)


def _rand_fp(rng, n, exp_lo, exp_hi, sparse=False) -> np.ndarray:
    sign = rng.integers(0, 2, n)
    exp = rng.integers(exp_lo, exp_hi + 1, n)
    man = rng.integers(0, 1 << 23, n)
    if sparse:
        # Keep only a few leading mantissa bits so products/sums are often exact ties.
        keep = rng.integers(1, 12, n)
        man = (man >> (23 - keep)) << (23 - keep)
    return _compose(sign, exp, man)


def _special_cross():
    a, b = np.meshgrid(SPECIALS, SPECIALS)
    return a.ravel(), b.ravel()


def mul_pairs(rng, n: int):
    parts = [_special_cross()]
    q = max(n // 8, 1)
    parts.append((rng.integers(0, 1 << 32, q, dtype=np.uint64).astype(np.uint32),
                  rng.integers(0, 1 << 32, q, dtype=np.uint64).astype(np.uint32)))
    parts.append((_rand_fp(rng, q, 100, 154), _rand_fp(rng, q, 100, 154)))
    parts.append((_rand_fp(rng, q, 100, 154, sparse=True), _rand_fp(rng, q, 100, 154, sparse=True)))
    # Overflow boundary: ea + eb - 127 around 254.
    ea = rng.integers(130, 255, q)
    eb = np.clip(381 - ea + rng.integers(-3, 4, q), 1, 254)
    parts.append((_compose(rng.integers(0, 2, q), ea, rng.integers(0, 1 << 23, q)),
                  _compose(rng.integers(0, 2, q), eb, rng.integers(0, 1 << 23, q))))
    # Underflow into subnormals: ea + eb - 127 in [-26, 2].
    ea = rng.integers(1, 127, q)
    eb = np.clip(127 - ea + rng.integers(-26, 3, q), 1, 254)
    parts.append((_compose(rng.integers(0, 2, q), ea, rng.integers(0, 1 << 23, q)),
                  _compose(rng.integers(0, 2, q), eb, rng.integers(0, 1 << 23, q))))
    # Subnormal x normal (normalization of subnormal inputs).
    parts.append((_rand_fp(rng, q, 0, 0), _rand_fp(rng, q, 100, 254)))
    parts.append((_rand_fp(rng, q, 0, 0, sparse=True), _rand_fp(rng, q, 120, 180, sparse=True)))
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    return a, b


def add_pairs(rng, n: int):
    parts = [_special_cross()]
    q = max(n // 8, 1)
    parts.append((rng.integers(0, 1 << 32, q, dtype=np.uint64).astype(np.uint32),
                  rng.integers(0, 1 << 32, q, dtype=np.uint64).astype(np.uint32)))
    # Exponent differences 0..30 around a random base.
    base = rng.integers(40, 220, q)
    diff = rng.integers(0, 31, q)
    parts.append((_compose(rng.integers(0, 2, q), base, rng.integers(0, 1 << 23, q)),
                  _compose(rng.integers(0, 2, q), base - diff, rng.integers(0, 1 << 23, q))))
    # Sparse mantissas: exact ties and round-to-even decisions.
    a = _rand_fp(rng, q, 120, 135, sparse=True)
    parts.append((a, _rand_fp(rng, q, 120, 135, sparse=True)))
    # Massive cancellation: b = -(a +- few ulps).
    a = _rand_fp(rng, q, 1, 254)
    b = (a ^ 0x80000000).astype(np.int64) + rng.integers(-3, 4, q)
    parts.append((a, np.clip(b, 0, 0xFFFFFFFF).astype(np.uint32)))
    # Subnormal region and the subnormal/normal boundary.
    parts.append((_rand_fp(rng, q, 0, 2), _rand_fp(rng, q, 0, 2)))
    # Overflow boundary.
    parts.append((_rand_fp(rng, q, 252, 254), _rand_fp(rng, q, 250, 254)))
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    return a, b
