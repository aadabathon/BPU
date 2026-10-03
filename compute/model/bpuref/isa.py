"""Command descriptors (mirror of rtl/common/bpu_isa_pkg.sv; provisional encodings).

A descriptor is {unit, tag, wait mask, body}; addresses and strides in the body are
32-bit element (fp32) addresses into the shared SRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UNIT_VEC, UNIT_MAT, UNIT_MEM = 0, 1, 2
UNIT_NAMES = {UNIT_VEC: "vector", UNIT_MAT: "matrix", UNIT_MEM: "memory"}

BODY_W = 432
# Vector body: op, func, half_log2, rows, cols, then d a b c s t ds as bs cs ss ts (32 bits each)
VB_OP, VB_FUNC, VB_HALF, VB_ROWS, VB_COLS, VB_ADDR = 0, 5, 8, 13, 29, 45
VB_FIELDS = ("d", "a", "b", "c", "s", "t", "d_stride", "a_stride", "b_stride", "c_stride",
             "s_stride", "t_stride")
# Matrix body: wid, wfmt, argmax, k, n (24 bits), then x, xs, y
MB_WID, MB_WFMT, MB_ARGMAX, MB_K, MB_N, MB_X, MB_XS, MB_Y = 0, 16, 17, 18, 34, 58, 90, 122


def _put(body: int, off: int, width: int, val: int) -> int:
    if val < 0 or val >= 1 << width:
        raise ValueError(f"field at bit {off} does not fit {width} bits: {val}")
    return body | (val << off)


def vec_body(op) -> int:
    """bpuref.fvu.FvuOp -> vector body."""
    b = 0
    b = _put(b, VB_OP, 5, op.op)
    b = _put(b, VB_FUNC, 3, op.func)
    b = _put(b, VB_HALF, 5, op.half.bit_length() - 1)
    b = _put(b, VB_ROWS, 16, op.rows)
    b = _put(b, VB_COLS, 16, op.cols)
    for i, name in enumerate(VB_FIELDS):
        b = _put(b, VB_ADDR + 32 * i, 32, getattr(op, name) & 0xFFFFFFFF)
    return b


def mat_body(op, wid: int, wfmt: int) -> int:
    """bpuref.qwen.QmvOp -> matrix body (wid = the weight tensor's id for the memory side)."""
    b = 0
    b = _put(b, MB_WID, 16, wid)
    b = _put(b, MB_WFMT, 1, wfmt)
    b = _put(b, MB_ARGMAX, 1, int(op.argmax))
    b = _put(b, MB_K, 16, op.k)
    b = _put(b, MB_N, 24, op.n)
    b = _put(b, MB_X, 32, op.x)
    b = _put(b, MB_XS, 32, op.xs)
    b = _put(b, MB_Y, 32, op.y)
    return b


@dataclass
class Cmd:
    """One descriptor plus the scheduling facts behind it."""
    unit: int
    tag: int
    wait: int                       # bit mask over tags
    op: object
    body: int = 0
    deps: list = field(default_factory=list)   # indices of the commands waited for
