"""FP32 vector unit (FVU) reference: the op set and its exact semantics.

The FVU works on a scratchpad (SPM) of fp32 elements. Every op is 2-D: `rows`
rows of `cols` elements. Operands are addressed per row:

    vector operand  X[r, j] = spm[x + r * x_stride + j]      (x, x_stride: multiples of 64)
    row scalar      S[r]    = spm[s + r * s_stride]          (any element address)
    group scalar    T[r, g] = spm[t + r * t_stride + j // 64]

Rules that make results independent of the hardware lane count:
  * Element-wise ops are exact fp32 (RNE, subnormals) or SFU ops, per element.
  * Sums (RSUM, RDOT) use one canonical tree: pad the row with +0 to
    P = max(64, next_pow2(cols)) and add adjacent pairs level by level.
  * VVECMAT accumulates sequentially over rows, starting from +0.
  * Max is order-independent (total-order key; NaN ranks lowest).

Writes touch exactly the op's elements (the RTL uses per-lane write masks), so an
op may update part of a row in place (e.g. RoPE on the first dims of each head).
The destination may equal a source exactly (in place) but not partially overlap it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sfu as sfu_mod
from .fp import F32_QNAN, bf16_to_f32, f32_to_bf16
from .qmv import f32_order_key

# Opcodes (the RTL encoding, bpu_compute_pkg::Fvu*)
VADD, VSUB, VMUL, VMULS, VADDS, VAXPY, VMULADD, VMULG = 0, 1, 2, 3, 4, 5, 6, 7
VSFU, VRBF16, VCOPY, VPERM, VQCLAMP, VSEL = 8, 9, 10, 11, 12, 13
RSUM, RDOT, RMAX, RAMAX = 16, 17, 18, 19
VVECMAT = 24

ELEMENTWISE = {VADD, VSUB, VMUL, VMULS, VADDS, VAXPY, VMULADD, VMULG, VSFU, VRBF16, VCOPY,
               VPERM, VQCLAMP, VSEL}
REDUCTIONS = {RSUM, RDOT, RMAX, RAMAX}
OP_NAMES = {v: k for k, v in globals().items() if k.isupper() and isinstance(v, int)
            and k.startswith(("V", "R")) and k not in ("ELEMENTWISE", "REDUCTIONS")}

# Which operands each op reads.
USES = {
    VADD: "ab", VSUB: "ab", VMUL: "ab", VMULS: "as", VADDS: "as", VAXPY: "abs",
    VMULADD: "abc", VMULG: "at", VSFU: "a", VRBF16: "a", VCOPY: "a", VPERM: "a",
    VQCLAMP: "a", VSEL: "abcs", RSUM: "a", RDOT: "ab", RMAX: "a", RAMAX: "a", VVECMAT: "as",
}


@dataclass(frozen=True)
class FvuOp:
    op: int
    rows: int
    cols: int
    d: int
    a: int = 0
    b: int = 0
    c: int = 0
    s: int = 0
    t: int = 0
    d_stride: int = 0
    a_stride: int = 0
    b_stride: int = 0
    c_stride: int = 0
    s_stride: int = 0
    t_stride: int = 0
    func: int = 0          # VSFU: bpuref.sfu function code
    half: int = 32         # VPERM: partner distance (power of two)

    def __str__(self):
        return f"{OP_NAMES.get(self.op, self.op)}[{self.rows}x{self.cols}] d={self.d}"


def validate(op: FvuOp, spm_size: int) -> None:
    """Static legality checks (the RTL rejects the same things)."""
    uses = USES[op.op]
    if op.rows < 1 or op.cols < 1:
        raise ValueError(f"{op}: empty shape")
    vec_fields = [("d", op.d, op.d_stride)] if op.op not in REDUCTIONS else []
    vec_fields += [(n, getattr(op, n), getattr(op, n + "_stride")) for n in "abc" if n in uses]
    for name, base, stride in vec_fields:
        if base % 64 or stride % 64:
            raise ValueError(f"{op}: vector operand {name} base/stride must be multiples of 64")
    if op.op == VPERM and (op.half & (op.half - 1) or op.cols % (2 * op.half)):
        raise ValueError(f"{op}: VPERM needs power-of-two half dividing cols/2")
    if op.op == VVECMAT and op.d_stride:
        raise ValueError(f"{op}: VVECMAT writes one vector (d_stride must be 0)")
    # Every element the op touches must exist (the hardware would wrap the address).
    for name, idx in [("d", _write_set(op))] + list(_read_sets(op).items()):
        if np.asarray(idx).size and int(np.max(idx)) >= spm_size:
            raise ValueError(f"{op}: operand {name} reaches element {int(np.max(idx))} of a {spm_size}-element SPM")
    check_hazards(op)


def _write_set(op: FvuOp) -> np.ndarray:
    R, C = op.rows, op.cols
    if op.op in REDUCTIONS:
        return (op.d + op.d_stride * np.arange(R))[:, None]
    if op.op == VVECMAT:
        return np.broadcast_to(op.d + np.arange(C), (R, C))
    return op.d + op.d_stride * np.arange(R)[:, None] + np.arange(C)[None, :]


def _read_sets(op: FvuOp) -> dict[str, np.ndarray]:
    R, C = op.rows, op.cols
    uses, reads = USES[op.op], {}
    cols = np.arange(C)[None, :]
    rows = np.arange(R)[:, None]
    for n in "abc":
        if n in uses:
            j = cols ^ op.half if (op.op == VPERM and n == "a") else cols
            reads[n] = getattr(op, n) + getattr(op, n + "_stride") * rows + j
    if "s" in uses:
        reads["s"] = np.broadcast_to(op.s + op.s_stride * rows, (R, C))
    if "t" in uses:
        reads["t"] = op.t + op.t_stride * rows + cols // 64
    if op.op == VVECMAT:
        reads["acc"] = _write_set(op)[1:]        # rows >= 1 read the accumulator ...
    return reads


def check_hazards(op: FvuOp) -> None:
    """Reject ops whose writes could be observed by their own reads: the hardware
    streams rows and words, so a written element may be read back only as the same
    (row, column) operand position (exact in-place). VVECMAT's accumulator reads are
    the one intended read-after-write and are sequenced by the hardware."""
    W = _write_set(op)
    if op.op in ELEMENTWISE and np.unique(W).size != W.size:
        raise ValueError(f"{op}: destination rows overlap each other")
    for name, Rd in _read_sets(op).items():
        if name == "acc":
            continue
        hits = np.isin(Rd, W)
        if not hits.any():
            continue
        if name in "abc" and op.op in ELEMENTWISE:
            # Exact in-place is fine: every aliased element is read at the position it is written.
            if np.array_equal(Rd[hits], W[hits]):
                continue
        e = int(np.asarray(Rd)[hits][0])
        raise ValueError(f"{op}: operand {name} reads element {e}, which the op also writes")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _rows(spm, base, stride, rows, cols):
    idx = base + stride * np.arange(rows)[:, None] + np.arange(cols)[None, :]
    return spm[idx]


def _scalars(spm, base, stride, rows):
    return spm[base + stride * np.arange(rows)][:, None]


def tree_sum(x: np.ndarray) -> np.ndarray:
    """Canonical row sum: pad each row with +0 to max(64, next_pow2(cols)), then
    add adjacent pairs level by level (fp32)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    p = max(64, 1 << (n - 1).bit_length())
    x = np.concatenate([x, np.zeros(x.shape[:-1] + (p - n,), np.float32)], axis=-1)
    with np.errstate(all="ignore"):
        while x.shape[-1] > 1:
            x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def order_max(x: np.ndarray) -> np.ndarray:
    """Row max under the total-order key; all-NaN rows give the canonical NaN."""
    key = f32_order_key(x)
    i = np.argmax(key, axis=-1)
    out = np.take_along_axis(x, i[..., None], axis=-1)[..., 0].astype(np.float32)
    allnan = key.max(axis=-1) == 0
    return np.where(allnan, np.uint32(F32_QNAN).view(np.float32), out)


def execute(op: FvuOp, spm: np.ndarray) -> None:
    """Apply one op to the scratchpad (numpy float32 array) in place."""
    validate(op, spm.size)
    R, C = op.rows, op.cols
    uses = USES[op.op]
    A = _rows(spm, op.a, op.a_stride, R, C) if "a" in uses else None
    B = _rows(spm, op.b, op.b_stride, R, C) if "b" in uses else None
    Cm = _rows(spm, op.c, op.c_stride, R, C) if "c" in uses else None
    S = _scalars(spm, op.s, op.s_stride, R) if "s" in uses else None

    with np.errstate(all="ignore"):
        if op.op in ELEMENTWISE:
            if op.op == VADD:
                out = A + B
            elif op.op == VSUB:
                out = A - B
            elif op.op == VMUL:
                out = A * B
            elif op.op == VMULS:
                out = A * S
            elif op.op == VADDS:
                out = A + S
            elif op.op == VAXPY:
                out = (S * A) + B
            elif op.op == VMULADD:
                out = (A * B) + Cm
            elif op.op == VMULG:
                g = np.arange(C)[None, :] // 64
                T = spm[op.t + op.t_stride * np.arange(R)[:, None] + g]
                out = A * T
            elif op.op == VSFU:
                out = sfu_mod.sfu(op.func, A.view(np.uint32)).view(np.float32)
            elif op.op == VRBF16:
                out = bf16_to_f32(f32_to_bf16(A))
            elif op.op == VCOPY:
                out = A.copy()
            elif op.op == VPERM:
                out = A[:, np.arange(C) ^ op.half]
            elif op.op == VQCLAMP:
                q = np.clip(np.rint(A), -127, 127) + np.float32(0)       # +0 for zero
                out = np.where(np.isnan(A), np.float32(0), q).astype(np.float32)
            elif op.op == VSEL:
                out = np.where(A > S, B, Cm)
            idx = op.d + op.d_stride * np.arange(R)[:, None] + np.arange(C)[None, :]
            spm[idx] = out.astype(np.float32)
        elif op.op in REDUCTIONS:
            if op.op == RSUM:
                res = tree_sum(A)
            elif op.op == RDOT:
                res = tree_sum(A * B)
            elif op.op == RMAX:
                res = order_max(A)
            else:   # RAMAX
                res = order_max(np.abs(A))
            spm[op.d + op.d_stride * np.arange(R)] = res.astype(np.float32)
        elif op.op == VVECMAT:
            acc = np.zeros(C, dtype=np.float32)
            for r in range(R):
                acc = acc + (S[r, 0] * A[r])
            spm[op.d + np.arange(C)] = acc
        else:
            raise ValueError(f"unknown opcode {op.op}")


def run(ops, spm: np.ndarray) -> np.ndarray:
    for op in ops:
        execute(op, spm)
    return spm
