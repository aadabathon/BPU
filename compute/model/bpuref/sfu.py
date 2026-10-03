"""Special-function unit reference: rcp, rsqrt, exp2, exp, log2 on fp32.

Every function is range reduction + a table-driven quadratic + packing, all in
integer arithmetic, so this model and the RTL (rtl/sfu/) agree bit for bit.
The coefficient tables are generated here (fit_tables) and written to
rtl/sfu/bpu_sfu_rom.sv by gen_sfu_rom.py; the tables *are* the spec.

Conventions (docs/numerics.md, "SFU"):
  * Inputs: subnormals are treated as zero (DAZ).
  * Outputs: results below the normal range flush to zero (FTZ).
  * NaN results are the canonical quiet NaN.
  * Results are approximations with bounded error (see error_report), not
    correctly rounded, but deterministic.
"""

from __future__ import annotations

import functools

import numpy as np

F32_QNAN = 0x7FC00000
F32_INF = 0x7F800000

# Function codes (the RTL func_i encoding).
RCP, RSQRT, EXP2, EXP, LOG2 = 0, 1, 2, 3, 4
FUNC_NAMES = {RCP: "rcp", RSQRT: "rsqrt", EXP2: "exp2", EXP: "exp", LOG2: "log2"}

# Table ids.
T_RCP, T_RSQ_EVEN, T_RSQ_ODD, T_EXP2, T_LOG_POS, T_LOG_NEG = range(6)
N_TABLES = 6

SEG_BITS = 7            # 128 segments per table
DELTA_BITS = 17         # table argument u is 24 bits: 7 index + 17 offset
POLY_FRAC = 30          # polynomial output fraction bits
FX = 24                 # exp: fixed-point fraction bits of the input
LOG2E_FRAC = 30
LOG2E_Q = round(np.log2(np.e) * 2**LOG2E_FRAC)   # 1549082005
LOG_FRAC = 24 + POLY_FRAC                         # log: fraction bits of t*h


# ---------------------------------------------------------------------------
# Table domains: g(u) for u in [0, 1), the value the table approximates
# ---------------------------------------------------------------------------

def _g(table: int, u: np.ndarray) -> np.ndarray:
    """Exact target value (float64) at normalized table argument u in [0, 1)."""
    if table == T_RCP:
        return 2.0 / (1.0 + u)
    if table == T_RSQ_EVEN:
        return 2.0 / np.sqrt(1.0 + u)
    if table == T_RSQ_ODD:
        return np.sqrt(2.0) / np.sqrt(1.0 + u)
    if table == T_EXP2:
        return np.exp2(u)
    if table == T_LOG_POS:            # h(t) = log2(1+t)/t at t = u/2 in [0, 0.5)
        t = u / 2.0
        return np.where(t == 0, 1.0 / np.log(2.0), np.log1p(t) / np.log(2.0) / np.where(t == 0, 1, t))
    if table == T_LOG_NEG:            # h(t) at t = -v, v = u/4 + 2^-24 in (0, 0.25]
        v = u / 4.0 + 2.0**-24
        return np.log1p(-v) / np.log(2.0) / (-v)
    raise ValueError(table)


# ---------------------------------------------------------------------------
# Quadratic evaluator (exact integer semantics shared with the RTL)
# ---------------------------------------------------------------------------

def poly_eval(c0, c1, c2, delta):
    """y = c0 + floor(c1*d / 2^17) + floor(c2*floor(d*d / 2^17) / 2^17), all int64."""
    d = np.asarray(delta, dtype=np.int64)
    sq = (d * d) >> DELTA_BITS
    return (np.asarray(c0, np.int64) + ((np.asarray(c1, np.int64) * d) >> DELTA_BITS)
            + ((np.asarray(c2, np.int64) * sq) >> DELTA_BITS))


def tables() -> np.ndarray:
    """The frozen coefficient tables (sfu_tables_data.py), shape (N_TABLES, 128, 3)."""
    from .sfu_tables_data import SFU_TABLES
    return _as_array(SFU_TABLES)


@functools.lru_cache(maxsize=None)
def _as_array(data) -> np.ndarray:
    return np.array(data, dtype=np.int64)


def fit_tables() -> np.ndarray:
    """Coefficient tables, shape (N_TABLES, 128, 3) int64.

    Per segment: least-squares quadratic on Chebyshev nodes, coefficients rounded
    to integers, then c0 re-centred so the evaluator's error (measured on every
    offset) is balanced. exp2 segment 0 is pinned so exp2(0) == 1 exactly.
    """
    nseg, dmax = 1 << SEG_BITS, 1 << DELTA_BITS
    scale = 2.0**POLY_FRAC
    deltas = np.arange(dmax, dtype=np.int64)
    out = np.zeros((N_TABLES, nseg, 3), dtype=np.int64)
    nodes = 0.5 - 0.5 * np.cos(np.pi * (np.arange(24) + 0.5) / 24)
    for tbl in range(N_TABLES):
        for i in range(nseg):
            ys = _g(tbl, (i + nodes) / nseg)
            a2, a1, a0 = np.polyfit(nodes, ys, 2)
            c = [round(a0 * scale), round(a1 * scale), round(a2 * scale)]
            exact = np.round(_g(tbl, (i + deltas / dmax) / nseg) * scale)
            err = poly_eval(c[0], c[1], c[2], deltas) - exact
            if tbl == T_EXP2 and i == 0:
                c[0] = 1 << POLY_FRAC
            else:
                c[0] -= int(np.floor((err.max() + err.min()) / 2))
            out[tbl, i] = c
    return out


def table_lookup(table, u24):
    """Evaluate table(s) at 24-bit arguments u24 (vectorized)."""
    t = tables()
    table = np.asarray(table, np.int64)
    u24 = np.asarray(u24, np.int64)
    idx = u24 >> DELTA_BITS
    c = t[table, idx]
    return poly_eval(c[..., 0], c[..., 1], c[..., 2], u24 & ((1 << DELTA_BITS) - 1))


def _round_pack(sign, exp, y):
    """y in [2^30, 2^31] (after clamping) -> fp32 bits with exponent field exp.
    Round to nearest even on the 7 dropped bits; a carry bumps the exponent."""
    drop = POLY_FRAC - 23
    keep = y >> drop
    rem = y & ((1 << drop) - 1)
    half = 1 << (drop - 1)
    up = (rem > half) | ((rem == half) & (keep & 1))
    mag = ((exp.astype(np.int64) << 23) | (keep & 0x7FFFFF)) + up
    return ((sign.astype(np.int64) << 31) | mag).astype(np.uint32)


def _clamp_unit(y):
    return np.clip(y, 1 << POLY_FRAC, (1 << (POLY_FRAC + 1)) - 1)


# ---------------------------------------------------------------------------
# The functions
# ---------------------------------------------------------------------------

def sfu(func: int, a_bits) -> np.ndarray:
    """Apply SFU function `func` to fp32 bit patterns; returns fp32 bit patterns."""
    a = np.asarray(a_bits, dtype=np.uint32).astype(np.int64)
    sign = (a >> 31) & 1
    e = (a >> 23) & 0xFF
    m = a & 0x7FFFFF
    nan = (e == 0xFF) & (m != 0)
    inf = (e == 0xFF) & (m == 0)
    zero = e == 0                         # DAZ: subnormals count as zero
    out = np.zeros(a.shape, dtype=np.uint32)

    if func == RCP:
        exp_field = np.where(m == 0, 254 - e, 253 - e)
        y = np.where(m == 0, 1 << POLY_FRAC, _clamp_unit(table_lookup(T_RCP, m << 1)))
        res = _round_pack(sign, np.maximum(exp_field, 0), y)
        res = np.where(exp_field <= 0, sign << 31, res)                      # FTZ
        res = np.where(zero, (sign << 31) | F32_INF, res)
        res = np.where(inf, sign << 31, res)
        return np.where(nan, F32_QNAN, res).astype(np.uint32)

    if func == RSQRT:
        E = e - 127
        odd = (E & 1) == 1
        h = E >> 1                                                           # floor(E/2)
        exact = (~odd) & (m == 0)
        exp_field = np.where(exact, 127 - h, 126 - h)
        y = _clamp_unit(table_lookup(np.where(odd, T_RSQ_ODD, T_RSQ_EVEN), m << 1))
        y = np.where(exact, 1 << POLY_FRAC, y)
        res = _round_pack(sign * 0, exp_field, y)
        res = np.where(zero, (sign << 31) | F32_INF, res)                    # +-0 -> +-inf
        res = np.where(inf & (sign == 0), 0, res)                            # +inf -> +0
        res = np.where(sign.astype(bool) & ~zero, F32_QNAN, res)             # x < 0 -> NaN
        return np.where(nan, F32_QNAN, res).astype(np.uint32)

    if func in (EXP2, EXP):
        # Fixed point with FX fraction bits, truncated toward zero; |x| >= 256 saturates.
        big = e >= 127 + 8
        sh = e - 126                                   # left shift of the 24-bit significand
        sig = np.where(zero, 0, m | (1 << 23))
        mag = np.where(sh >= 0, sig << np.clip(sh, 0, 63), sig >> np.clip(-sh, 0, 63))
        X = np.where(sign == 1, -mag, mag)
        Y = (X * LOG2E_Q) >> LOG2E_FRAC if func == EXP else X
        n = Y >> FX
        f = Y & ((1 << FX) - 1)
        y = _clamp_unit(table_lookup(T_EXP2, f))
        res = _round_pack(np.zeros_like(sign), np.clip(n + 127, 0, 255), y)
        res = np.where(n >= 128, F32_INF, res)
        res = np.where(n < -126, 0, res)                                     # FTZ
        res = np.where(big, np.where(sign == 1, 0, F32_INF), res)
        res = np.where(inf, np.where(sign == 1, 0, F32_INF), res)
        return np.where(nan, F32_QNAN, res).astype(np.uint32)

    if func == LOG2:
        E = e - 127
        lo = m < (1 << 22)                            # 1.m < 1.5: t = m in [0, 0.5)
        t_pos = m << 1                                # t with 24 fraction bits
        v_int = (1 << 23) - m                         # 1.m >= 1.5: t = -(1 - m)/2, v = -t
        # Table arguments (both branches are evaluated; each is masked to its range).
        u_pos = np.where(lo, m << 2, 0)                                  # u = 2t
        u_neg = np.where(lo, 0, (v_int - 1) << 2)                        # u = 4(v - 2^-24)
        h = np.where(lo, table_lookup(T_LOG_POS, u_pos), table_lookup(T_LOG_NEG, u_neg))
        t = np.where(lo, t_pos, -v_int)
        ep = np.where(lo, E, E + 1)
        R = (ep << LOG_FRAC) + t * h                   # fixed point, LOG_FRAC fraction bits
        res = _fixed_to_f32(R, LOG_FRAC)
        res = np.where(zero, 0xFF800000, res)                                # log2(+-0) = -inf
        res = np.where(inf & (sign == 0), F32_INF, res)
        res = np.where(sign.astype(bool) & ~zero, F32_QNAN, res)             # x < 0 -> NaN
        return np.where(nan, F32_QNAN, res).astype(np.uint32)

    raise ValueError(func)


def _fixed_to_f32(R, frac: int):
    """Signed fixed-point (frac fraction bits, |R| < 2^62) -> fp32, round to nearest even."""
    R = np.asarray(R, dtype=np.int64)
    sign = (R < 0).astype(np.int64)
    mag = np.abs(R)
    out = np.zeros(R.shape, dtype=np.uint32)
    nz = mag != 0
    msb = np.zeros(R.shape, dtype=np.int64)
    msb[nz] = np.floor(np.log2(mag[nz].astype(np.float64))).astype(np.int64)
    # log2 in float64 can be off by one near powers of two; fix exactly.
    msb = np.where(nz & ((mag >> np.clip(msb, 0, 62)) == 0), msb - 1, msb)
    msb = np.where(nz & ((mag >> np.clip(msb + 1, 0, 62)) != 0), msb + 1, msb)
    drop = np.maximum(msb - 23, 0)
    keep = mag >> drop
    rem = mag & ((np.int64(1) << drop) - 1)
    half = np.where(drop > 0, np.int64(1) << np.maximum(drop - 1, 0), 0)
    up = (drop > 0) & ((rem > half) | ((rem == half) & ((keep & 1) == 1)))
    keep = keep << np.maximum(23 - msb, 0)             # small values: shift up (exact)
    exp_field = msb - frac + 127
    mag32 = (exp_field << 23) + (keep & 0x7FFFFF) + up
    out = np.where(nz, (sign << 31) | mag32, 0)
    return out.astype(np.uint32)


# ---------------------------------------------------------------------------
# Accuracy report (vs float64)
# ---------------------------------------------------------------------------

def _exact(func, x):
    with np.errstate(all="ignore"):
        if func == RCP:
            return 1.0 / x
        if func == RSQRT:
            return 1.0 / np.sqrt(x)
        if func == EXP2:
            return np.exp2(x)
        if func == EXP:
            return np.exp(x)
        return np.log2(x)


def ulp_error(func, a_bits):
    """Error of sfu() in units of the fp32 ulp of the exact result (finite normals only)."""
    a = np.asarray(a_bits, np.uint32)
    x = a.view(np.float32).astype(np.float64)
    got = sfu(func, a).view(np.float32).astype(np.float64)
    ref = _exact(func, x)
    ok = np.isfinite(ref) & np.isfinite(got) & (np.abs(ref) >= 2.0**-126) & (ref != 0)
    ulp = 2.0 ** (np.floor(np.log2(np.abs(ref[ok]))) - 23)
    return np.abs(got[ok] - ref[ok]) / ulp, x[ok]


def error_report(n: int = 2_000_000, seed: int = 0) -> dict[str, float]:
    """Max ulp error per function on random inputs over each function's useful range."""
    rng = np.random.default_rng(seed)
    def rand_f32(lo_e, hi_e, positive=True):
        e = rng.integers(lo_e, hi_e + 1, n).astype(np.uint32)
        mm = rng.integers(0, 1 << 23, n).astype(np.uint32)
        s = np.zeros(n, np.uint32) if positive else rng.integers(0, 2, n).astype(np.uint32)
        return (s << 31) | (e << 23) | mm
    report = {}
    report["rcp"] = ulp_error(RCP, rand_f32(1, 252, False))[0].max()
    report["rsqrt"] = ulp_error(RSQRT, rand_f32(1, 254))[0].max()
    report["exp2 (|x| < 126)"] = ulp_error(EXP2, np.float32(rng.uniform(-126, 127, n)).view(np.uint32))[0].max()
    report["exp (|x| < 87)"] = ulp_error(EXP, np.float32(rng.uniform(-87, 88, n)).view(np.uint32))[0].max()
    report["exp (|x| < 20)"] = ulp_error(EXP, np.float32(rng.uniform(-20, 20, n)).view(np.uint32))[0].max()
    report["log2 (all x)"] = ulp_error(LOG2, rand_f32(1, 254))[0].max()
    x = np.float32(rng.uniform(1.0, 2.0, n)).view(np.uint32)
    report["log2 (1 <= x < 2)"] = ulp_error(LOG2, x)[0].max()
    return report
