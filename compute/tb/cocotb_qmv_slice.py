"""bpu_qmv_slice against bpuref.qmv_ref, bit for bit.

The runner selects the hardware configuration with BPU_QMV_CFG=<name>. Every
test runs unchanged at every configuration; that is the point.
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

from bpuref.configs import QMV_SLICE_CONFIGS
from bpuref.fp import f32_bits, f32_equal, f32_to_bf16
from bpuref.qmv import GROUP, W4, W8, W_RANGE, pack_weight_stream, pack_x_words, qmv_ref
from bpuref.quant import quantize_activations, quantize_weights_rtn

CFG = QMV_SLICE_CONFIGS[os.environ.get("BPU_QMV_CFG", "fpga")]


class Op:
    """One QMV operation: data, its reference result, and its streams."""

    def __init__(self, wfmt, w, ws, x, xs):
        self.wfmt, self.w, self.ws, self.x, self.xs = wfmt, w, ws, x, xs
        self.n, self.k = w.shape
        self.ref = qmv_ref(w, ws, x, xs)
        self.beats, self.scales = pack_weight_stream(w, ws, CFG.lanes, CFG.row_interleave, wfmt)
        self.x_words = pack_x_words(x, CFG.lanes)


def realistic_scales(rng, shape):
    """bf16 scales of the magnitude real quantized tensors have, either sign."""
    mag = 2.0 ** rng.uniform(-14, 0, shape)
    sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    return f32_to_bf16((sign * mag).astype(np.float32))


def random_op(rng, *, wfmt=None, max_groups=6, max_rowblocks=3, scales=realistic_scales):
    wfmt = int(rng.integers(0, 2)) if wfmt is None else wfmt
    kg = int(rng.integers(1, min(CFG.max_k // GROUP, max_groups) + 1))
    n = int(rng.integers(1, max_rowblocks + 1)) * CFG.row_interleave
    k = kg * GROUP
    lo, hi = W_RANGE[wfmt]
    w = rng.integers(lo, hi + 1, (n, k))
    x = rng.integers(-128, 128, k)
    return Op(wfmt, w, scales(rng, (n, kg)), x, scales(rng, (kg,)))


# ---------------------------------------------------------------------------
# Bus functional model
# ---------------------------------------------------------------------------

async def reset(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    for sig in (dut.x_we_i, dut.xs_we_i, dut.cmd_valid_i, dut.w_valid_i, dut.ws_valid_i, dut.y_ready_i,
                dut.status_clr_i):
        sig.value = 0
    dut.rst_ni.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def source(dut, valid, ready, data, items, rng, p_idle):
    """Valid/ready source. Holds valid and data until the handshake completes.

    Returns the number of cycles from the first to the last handshake, inclusive.
    """
    i, asserted, cycle, first = 0, False, 0, None
    while i < len(items):
        if not asserted and rng.random() >= p_idle:
            data.value = items[i]
            valid.value = 1
            asserted = True
        await ReadOnly()
        fire = asserted and ready.value == 1
        await RisingEdge(dut.clk_i)
        if fire:
            first = cycle if first is None else first
            last = cycle
            i += 1
            asserted = False
            valid.value = 0
        cycle += 1
    return last - first + 1


async def sink(dut, count, rng, p_stall, out):
    while len(out) < count:
        dut.y_ready_i.value = int(rng.random() >= p_stall)
        await ReadOnly()
        if dut.y_valid_o.value == 1 and dut.y_ready_i.value == 1:
            out.append(int(dut.y_data_o.value))
        await RisingEdge(dut.clk_i)
    dut.y_ready_i.value = 0


async def run_ops(dut, ops, rng, p_idle=0.2, p_stall=0.2):
    """Run operations back to back and check every output against the reference.

    Returns, per operation, the cycles spent accepting its weight beats.
    """
    got, beat_cycles = [], []
    total = sum(op.n for op in ops)
    sink_task = cocotb.start_soon(sink(dut, total, rng, p_stall, got))

    for op in ops:
        # The activation buffer may only be written while no operation is streaming.
        while True:
            await ReadOnly()
            if dut.cmd_ready_o.value == 1:
                break
            await RisingEdge(dut.clk_i)
        await RisingEdge(dut.clk_i)

        # Lanes <= 64, so there are at least as many x words as groups.
        for addr, word in enumerate(op.x_words):
            dut.x_we_i.value = 1
            dut.x_waddr_i.value = addr
            dut.x_wdata_i.value = word
            dut.xs_we_i.value = int(addr < len(op.xs))
            if addr < len(op.xs):
                dut.xs_waddr_i.value = addr
                dut.xs_wdata_i.value = int(op.xs[addr])
            await RisingEdge(dut.clk_i)
        dut.x_we_i.value = 0
        dut.xs_we_i.value = 0

        dut.cmd_valid_i.value = 1
        dut.cmd_wfmt_i.value = op.wfmt
        dut.cmd_ngroups_i.value = op.k // GROUP
        dut.cmd_nrowblk_i.value = op.n // CFG.row_interleave
        await RisingEdge(dut.clk_i)          # cmd_ready_o is high: accepted
        dut.cmd_valid_i.value = 0

        w_task = cocotb.start_soon(source(dut, dut.w_valid_i, dut.w_ready_o, dut.w_data_i,
                                          op.beats, rng, p_idle))
        ws_task = cocotb.start_soon(source(dut, dut.ws_valid_i, dut.ws_ready_o, dut.ws_data_i,
                                           op.scales, rng, p_idle))
        beat_cycles.append(await w_task)
        await ws_task

    await sink_task
    for _ in range(4):
        await RisingEdge(dut.clk_i)
    assert dut.busy_o.value == 0, "slice still busy after all outputs were drained"

    ref = np.concatenate([op.ref for op in ops])
    ok = f32_equal(np.array(got, dtype=np.uint32), ref)
    bad = np.flatnonzero(~ok)
    for j in bad[:10]:
        dut._log.error(f"row {j}: rtl {got[j]:08x} ref {int(f32_bits(ref[j])):08x}")
    assert bad.size == 0, f"{bad.size} / {len(ref)} rows mismatched"
    dut._log.info(f"[{CFG.name}] {len(ops)} ops, {len(ref)} rows bit-exact")
    return beat_cycles


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def random_ops_with_backpressure(dut):
    """Mixed W4/W8 operations of random shape; random stream gaps and output stalls."""
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    await reset(dut)
    ops = [random_op(rng) for _ in range(16)]
    await run_ops(dut, ops, rng)
    assert int(dut.perf_beats_o.value) == sum(len(op.beats) for op in ops)
    assert int(dut.err_cmd_o.value) == 0
    # Realistic scales never overflow, so the sticky flags must stay clear.
    assert int(dut.flag_nan_o.value) == 0 and int(dut.flag_inf_o.value) == 0


@cocotb.test()
async def full_throughput(dut):
    """With no gaps or stalls the slice accepts exactly one weight beat per cycle.

    Holds when a row block (K/64 * R * chunks beats) outlasts the pipeline latency,
    which every Qwen3.5 shape does (K >= 2048); docs/engine-ops.md has the bound.
    """
    rng = np.random.default_rng(2)
    await reset(dut)
    kg = min(CFG.max_k // GROUP, 32)
    ops = []
    for wfmt in (W4, W8):
        op = random_op(rng, wfmt=wfmt, max_groups=kg, max_rowblocks=4)
        ops.append(op if op.k == kg * GROUP else Op(wfmt, *_widen(rng, op, kg)))
    cycles = await run_ops(dut, ops, rng, p_idle=0.0, p_stall=0.0)
    for op, c in zip(ops, cycles):
        assert c == len(op.beats), f"{len(op.beats)} beats took {c} cycles"
    # The counters agree: every running cycle accepted a beat.
    assert int(dut.perf_beats_o.value) == sum(len(op.beats) for op in ops)
    stalls = [int(s.value) for s in (dut.perf_stall_w_o, dut.perf_stall_ws_o, dut.perf_stall_out_o)]
    assert stalls == [0, 0, 0], f"unexpected stall cycles {stalls}"


def _widen(rng, op, kg):
    """Same op with exactly kg groups (full-size reduction)."""
    k = kg * GROUP
    lo, hi = W_RANGE[op.wfmt]
    return (rng.integers(lo, hi + 1, (op.n, k)), realistic_scales(rng, (op.n, kg)),
            rng.integers(-128, 128, k), realistic_scales(rng, (kg,)))


@cocotb.test()
async def qwen_projection_shapes(dut):
    """Realistic quantized data at Qwen3.5-2B reduction lengths (K = 2048, 6144)."""
    rng = np.random.default_rng(3)
    await reset(dut)
    ops = []
    # Configurations too small for Qwen's K use their largest K instead.
    for k in [k for k in (2048, 6144) if k <= CFG.max_k] or [CFG.max_k]:
        n = 2 * CFG.row_interleave
        for wfmt in (W4, W8):
            w, ws = quantize_weights_rtn(rng.normal(0, 0.02, (n, k)).astype(np.float32), wfmt)
            x, xs = quantize_activations(rng.normal(0, 1.0, k).astype(np.float32))
            ops.append(Op(wfmt, w, ws, x, xs))
    await run_ops(dut, ops, rng)


@cocotb.test()
async def extreme_values(dut):
    """Largest group sums, and scales that overflow, underflow and produce NaN/inf."""
    rng = np.random.default_rng(4)
    await reset(dut)

    def wild_scales(rng, shape):
        exp = rng.choice([1, 2, 60, 127, 190, 254, 0, 255], shape)   # incl. subnormal, inf/NaN
        man = rng.integers(0, 128, shape)
        sign = rng.integers(0, 2, shape)
        return ((sign << 15) | (exp << 7) | man).astype(np.uint16)

    ops = []
    for wfmt in (W4, W8):
        lo, _ = W_RANGE[wfmt]
        kg = min(CFG.max_k // GROUP, 4)
        n = CFG.row_interleave
        w = np.full((n, kg * GROUP), lo)
        x = np.full(kg * GROUP, -128)
        ops.append(Op(wfmt, w, realistic_scales(rng, (n, kg)), x, realistic_scales(rng, (kg,))))
        ops.append(random_op(rng, wfmt=wfmt, scales=wild_scales))
    await run_ops(dut, ops, rng)
    ref = np.concatenate([op.ref for op in ops])
    assert int(dut.flag_nan_o.value) == int(np.isnan(ref).any())
    assert int(dut.flag_inf_o.value) == int(np.isinf(ref).any())

    # status_clr_i clears the sticky flags and the counters.
    await RisingEdge(dut.clk_i)
    dut.status_clr_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.status_clr_i.value = 0
    await ReadOnly()
    assert int(dut.flag_nan_o.value) == 0 and int(dut.flag_inf_o.value) == 0
    assert int(dut.perf_beats_o.value) == 0


@cocotb.test()
async def illegal_commands_are_rejected(dut):
    """Bad shapes are consumed without running, raise err_cmd_o, and leave the slice usable."""
    rng = np.random.default_rng(5)
    await reset(dut)
    max_groups = CFG.max_k // GROUP
    for ngroups, nrowblk in ((0, 1), (max_groups + 1, 1), (1, 0)):
        if ngroups >= 1 << len(dut.cmd_ngroups_i):
            continue                      # not even encodable at this configuration
        dut.cmd_valid_i.value = 1
        dut.cmd_wfmt_i.value = W4
        dut.cmd_ngroups_i.value = ngroups
        dut.cmd_nrowblk_i.value = nrowblk
        await ReadOnly()
        assert dut.cmd_ready_o.value == 1
        await RisingEdge(dut.clk_i)
        dut.cmd_valid_i.value = 0
        await RisingEdge(dut.clk_i)
        await ReadOnly()
        assert dut.err_cmd_o.value == 1, f"ngroups={ngroups} nrowblk={nrowblk} not flagged"
        assert dut.busy_o.value == 0 and dut.cmd_ready_o.value == 1
        await RisingEdge(dut.clk_i)
        dut.status_clr_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.status_clr_i.value = 0
    # A legal operation still runs correctly afterwards.
    await run_ops(dut, [random_op(rng)], rng)
    assert dut.err_cmd_o.value == 0
