"""bpu_fvu on the shared SRAM (tb/hdl/bpu_fvu_sys.sv) against bpuref.fvu: directed
corner ops and random legal op sequences, with competing host traffic on the same
SRAM while every op runs, then a comparison of the whole test region. Also: illegal
descriptors and out-of-range addresses complete with an error instead of hanging.
Configuration via BPU_FVU_CFG; test region size via BPU_SPM_ELEMS (the SRAM is twice
that: the upper half takes the competing writes)."""

import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

import busmodels as bm
from bpuref import fvu as F
from bpuref import sfu
from bpuref.configs import FVU_CONFIGS
from bpuref.fp import f32_equal

_name = os.environ.get("BPU_FVU_CFG", "tiny")
# Test variants (e.g. "nofwd", "gate") pass their lane count explicitly.
CFG = FVU_CONFIGS.get(_name, FVU_CONFIGS["tiny"])
V = int(os.environ.get("BPU_FVU_VLANES", CFG.vlanes))
SPM = int(os.environ.get("BPU_SPM_ELEMS", "2048"))
SRAM_WORDS = 2 * SPM // V            # tb/test_rtl.py: twice the test region
SRC_END = SPM // 2                 # vector sources in [0, VEC_END), scalars in [VEC_END, SRC_END),
VEC_END = SRC_END // 2             # destinations in [SRC_END, SPM); in-place ops write a source


def random_spm(rng):
    x = rng.normal(0, 2, SPM).astype(np.float32)
    x[rng.random(SPM) < 0.02] = 0.0
    x[rng.random(SPM) < 0.02] = -0.0
    x[rng.random(SPM) < 0.005] = np.inf
    x[rng.random(SPM) < 0.005] = -np.inf
    x[rng.random(SPM) < 0.005] = np.nan
    x[rng.random(SPM) < 0.02] = np.float32(1e-40)            # subnormal
    return x


def random_op(rng):
    """A legal random op (rejection-sampled against bpuref.fvu.validate, which also
    rejects read/write aliasing)."""
    while True:
        op = _candidate_op(rng)
        try:
            F.validate(op, SPM)
            return op
        except ValueError:
            continue


def _candidate_op(rng):
    op = int(rng.choice(sorted(F.USES)))
    uses = F.USES[op]
    cols = int(rng.choice([1, 3, 7, 16, 33, 64, 100, 128]))
    if op == F.VPERM:
        half = int(rng.choice([1, 2, 4, 8, 16, 32]))
        cols = max(2 * half, cols - cols % (2 * half))
    else:
        half = 32
    span = -(-cols // 64) * 64
    rows = int(rng.integers(1, 5))
    stride = span * int(rng.integers(1, 3)) if rows > 1 else 0
    if rows * max(stride, span) > VEC_END // 2:
        rows, stride = 1, 0
    region = lambda lo, hi: int(rng.integers(lo // 64, (hi - rows * max(stride, span)) // 64 + 1)) * 64
    kw = dict(op=op, rows=rows, cols=cols, half=half, func=int(rng.integers(0, 5)))
    for n in "abc":
        kw[n] = region(0, VEC_END) if n in uses else 0
        kw[n + "_stride"] = stride if n in uses else 0
    for n in "st":
        kw[n] = int(rng.integers(VEC_END, SRC_END - 64))
        kw[n + "_stride"] = int(rng.integers(0, 3)) if n in uses else 0
    if op in F.REDUCTIONS:
        kw["d"] = int(rng.integers(SRC_END, SPM - 8 * rows))
        kw["d_stride"] = int(rng.integers(1, 5))
    elif op == F.VVECMAT:
        kw["d"] = region(SRC_END, SPM)
        kw["d_stride"] = 0
    else:
        in_place = rng.random() < 0.25
        kw["d"] = kw["a"] if in_place else region(SRC_END, SPM)
        kw["d_stride"] = stride
    return F.FvuOp(**kw)


def directed_ops():
    """Each opcode on awkward shapes, plus the corner behaviours. The shapes are laid
    out for a 2048-element SPM; smaller SPMs (gate-level) keep the ops that fit."""
    ops = []
    for op in _directed_candidates():
        try:
            F.validate(op, SPM)
            ops.append(op)
        except ValueError:
            assert SPM < 2048, f"directed op does not fit: {op}"
    return ops


def _directed_candidates():
    ops = []
    D = SRC_END
    for op in sorted(F.USES):
        base = dict(a=0, b=128, c=256, s=VEC_END + 5, t=VEC_END + 64)
        if op in F.REDUCTIONS:
            ops.append(F.FvuOp(op, 3, 70, d=D + 3, d_stride=1, a_stride=128, b_stride=128, **base))
            ops.append(F.FvuOp(op, 1, 1, d=D + 9, **base))
        elif op == F.VVECMAT:
            ops.append(F.FvuOp(op, 5, 70, d=D + 64, a_stride=128, s_stride=1, **base))
            ops.append(F.FvuOp(op, 7, 3, d=D + 256, a_stride=64, s_stride=1, **base))   # short rows: hazard spacing
        elif op == F.VPERM:
            for half in (1, 8, 32):
                ops.append(F.FvuOp(op, 2, 64, d=D + 384, d_stride=64, a_stride=64, half=half, **base))
        else:
            for f in range(5) if op == F.VSFU else [0]:
                ops.append(F.FvuOp(op, 2, 67, d=D + 512, d_stride=128, a_stride=128, b_stride=128,
                                   c_stride=128, s_stride=1, t_stride=2, func=f, **base))
    # VVECMAT with many short rows: accumulator reads race the previous row's writes,
    # so every forwarding case (and none) occurs across the configurations.
    for cols in (1, V, 2 * V + 1, 3 * V, 64):
        ops.append(F.FvuOp(F.VVECMAT, 8, cols, d=D + 640, a=0, a_stride=64, s=VEC_END + 9, s_stride=1))
    # In-place partial-row update (RoPE-style): only the first 16 of 64 elements change.
    ops.append(F.FvuOp(F.VMULS, 2, 16, d=0, a=0, s=VEC_END + 7, d_stride=64, a_stride=64))
    return ops


async def issue(dut, op):
    """Send one op and wait for its completion; returns the error flag."""
    await bm.wait_high(dut, dut.cmd_ready_o)
    dut.cmd_valid_i.value = 1
    dut.cmd_op_i.value = op.op
    dut.cmd_func_i.value = op.func
    dut.cmd_half_log2_i.value = op.half.bit_length() - 1
    dut.cmd_rows_i.value = op.rows
    dut.cmd_cols_i.value = op.cols
    for n in "dabcst":
        getattr(dut, f"cmd_{n}_i").value = getattr(op, n) & 0xFFFFFFFF
        getattr(dut, f"cmd_{n}s_i").value = getattr(op, n + "_stride") & 0xFFFFFFFF
    await RisingEdge(dut.clk_i)
    dut.cmd_valid_i.value = 0
    for _ in range(1_000_000):
        await ReadOnly()
        if dut.done_o.value == 1:
            err = int(dut.err_o.value)
            await RisingEdge(dut.clk_i)
            return err
        await RisingEdge(dut.clk_i)
    raise AssertionError(f"op never completed: {op}")


async def run_and_compare(dut, host, ops, spm0, rng, noise=True):
    ref = spm0.copy()
    await host.write_elems(0, spm0)
    debug = os.environ.get("BPU_FVU_DEBUG") == "1"
    for i, op in enumerate(ops):
        F.execute(op, ref)
        if noise:
            task = cocotb.start_soon(host.noise(rng, (0, SRAM_WORDS), (SPM // V, SRAM_WORDS)))
        err = await issue(dut, op)
        if noise:
            host.noise_on = False
            await task
        assert err == 0, f"legal op #{i} reported an error: {op}"
        if debug:                       # compare after every op to find the first divergence
            now = await host.read_elems(0, SPM)
            diff = np.flatnonzero(~f32_equal(now, ref))
            if diff.size:
                dut._log.error(f"op #{i} {op!r} diverges at {diff[:8].tolist()}")
                break
    got = await host.read_elems(0, SPM)
    ok = f32_equal(got, ref)
    bad = np.flatnonzero(~ok)
    for e in bad[:12]:
        dut._log.error(f"spm[{e}]: rtl {got[e]:08x} ref {ref.view(np.uint32)[e]:08x}")
    assert bad.size == 0, f"{bad.size} / {SPM} elements differ"


INPUTS = lambda dut: (dut.cmd_valid_i, dut.host_rd_valid_i, dut.host_wr_valid_i)


@cocotb.test()
async def directed(dut):
    rng = np.random.default_rng(7)
    await bm.reset(dut, INPUTS(dut))
    host = bm.SramHost(dut, V)
    ops = directed_ops()
    await run_and_compare(dut, host, ops, random_spm(rng), rng)
    dut._log.info(f"[{_name}] {len(ops)} of {len(_directed_candidates())} directed ops "
                  f"(those that fit a {SPM}-element test region) bit-exact under competing traffic")


@cocotb.test()
async def random_sequences(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    await bm.reset(dut, INPUTS(dut))
    host = bm.SramHost(dut, V)
    for trial in range(3):
        ops = [random_op(rng) for _ in range(40)]
        await run_and_compare(dut, host, ops, random_spm(rng), rng, noise=trial != 1)
    dut._log.info(f"[{_name}] 3 x 40 random ops bit-exact (two trials under competing traffic)")


@cocotb.test()
async def illegal_ops_complete_with_error(dut):
    """Malformed descriptors are consumed without running; addresses past the SRAM
    run but report an error. Either way the op completes, so nothing waits forever."""
    await bm.reset(dut, INPUTS(dut))
    beyond = SRAM_WORDS * V
    for op in (F.FvuOp(F.VADD, 1, 8, d=0, a=1, b=0),           # misaligned vector
               F.FvuOp(F.VADD, 0, 8, d=0, a=0, b=0),           # empty
               F.FvuOp(31, 1, 8, d=0, a=0),                   # unknown opcode
               F.FvuOp(F.VPERM, 1, 24, d=0, a=0, half=8),     # cols not a multiple of 2*half
               F.FvuOp(F.VSFU, 1, 8, d=0, a=64, func=5),      # no such SFU function
               F.FvuOp(F.VCOPY, 2, 64, d=0, a=beyond - 64, a_stride=64, d_stride=64),  # reads past the end
               F.FvuOp(F.RSUM, 1, 64, d=beyond + 3, a=0)):    # writes past the end
        err = await issue(dut, op)
        assert err == 1, f"no error for {op}"
        assert dut.busy_o.value == 0
    # A legal op afterwards still works.
    assert await issue(dut, F.FvuOp(F.VCOPY, 1, 8, d=128, a=0)) == 0
