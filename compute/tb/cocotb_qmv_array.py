"""bpu_qmv_array against bpuref (qmv_ref + argmax_ref), bit for bit.

The runner selects the configuration with BPU_QMV_CFG=<name> (QMV_ARRAY_CONFIGS).
"""

import os

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge

from bpuref.configs import QMV_ARRAY_CONFIGS
from bpuref.fp import f32_bits, f32_equal, f32_from_bits
from bpuref.qmv import GROUP, W4, W8, W_RANGE, argmax_ref, pack_array_streams, pack_x_words, qmv_ref
from bpuref.quant import quantize_activations, quantize_weights_rtn
import busmodels as bm
from stimulus import realistic_scales

CFG = QMV_ARRAY_CONFIGS[os.environ.get("BPU_QMV_CFG", "fpga")]
NS, LANES, R = CFG.nslice, CFG.slice.lanes, CFG.slice.row_interleave
ROWS_PER_BLOCK = NS * R


class Op:
    def __init__(self, wfmt, w, ws, x, xs, argmax=False):
        self.wfmt, self.w, self.ws, self.x, self.xs, self.argmax = wfmt, w, ws, x, xs, argmax
        self.n, self.k = w.shape
        self.ref = qmv_ref(w, ws, x, xs)
        streams = pack_array_streams(w, ws, NS, LANES, R, wfmt)
        self.beats = [b for b, _ in streams]
        self.scales = [s for _, s in streams]
        self.x_words = pack_x_words(x, LANES)

    def expected(self):
        if self.argmax:
            i = argmax_ref(self.ref)
            return [(i, int(f32_bits(self.ref[i])))]
        return [(i, int(f32_bits(v))) for i, v in enumerate(self.ref)]


def random_op(rng, *, wfmt=None, max_groups=4, max_blocks=2, argmax=False):
    wfmt = int(rng.integers(0, 2)) if wfmt is None else wfmt
    kg = int(rng.integers(1, min(CFG.slice.max_k // GROUP, max_groups) + 1))
    n = int(rng.integers(1, max_blocks + 1)) * ROWS_PER_BLOCK
    lo, hi = W_RANGE[wfmt]
    return Op(wfmt, rng.integers(lo, hi + 1, (n, kg * GROUP)), realistic_scales(rng, (n, kg)),
              rng.integers(-128, 128, kg * GROUP), realistic_scales(rng, (kg,)), argmax)


INPUTS = lambda dut: (dut.x_we_i, dut.xs_we_i, dut.cmd_valid_i, dut.cmd_argmax_i, dut.w_valid_i,
                      dut.ws_valid_i, dut.y_ready_i, dut.status_clr_i)


async def run_ops(dut, ops, rng, p_idle=0.2, p_stall=0.2):
    got = []
    expected = [e for op in ops for e in op.expected()]
    sink = cocotb.start_soon(bm.sink(dut, dut.y_valid_o, dut.y_ready_i, (dut.y_index_o, dut.y_data_o),
                                     len(expected), rng, p_stall, got))
    for op in ops:
        await bm.wait_high(dut, dut.cmd_ready_o)
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
        dut.cmd_nrowblk_i.value = op.n // ROWS_PER_BLOCK
        dut.cmd_argmax_i.value = int(op.argmax)
        await RisingEdge(dut.clk_i)
        dut.cmd_valid_i.value = 0
        w = cocotb.start_soon(bm.multi_source(dut, dut.w_valid_i, dut.w_ready_o, dut.w_data_i,
                                              LANES * 4, op.beats, rng, p_idle))
        ws = cocotb.start_soon(bm.multi_source(dut, dut.ws_valid_i, dut.ws_ready_o, dut.ws_data_i,
                                               16, op.scales, rng, p_idle))
        await w
        await ws
    await sink
    for _ in range(4):
        await RisingEdge(dut.clk_i)
    assert dut.busy_o.value == 0

    bad = 0
    for j, ((gi, gv), (ei, ev)) in enumerate(zip(got, expected)):
        ok = gi == ei and bool(f32_equal(np.uint32(gv), f32_from_bits(np.uint32(ev))))
        if not ok:
            bad += 1
            if bad <= 10:
                dut._log.error(f"#{j}: rtl ({gi}, {gv:08x}) ref ({ei}, {ev:08x})")
    assert bad == 0, f"{bad} / {len(expected)} results mismatched"
    dut._log.info(f"[{CFG.name}] {len(ops)} ops, {len(expected)} results bit-exact")


@cocotb.test()
async def stream_mode_random(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    await bm.reset(dut, INPUTS(dut))
    await run_ops(dut, [random_op(rng) for _ in range(6)], rng)


@cocotb.test()
async def argmax_mode(dut):
    """Argmax with random data, exact ties across slices, and NaN rows."""
    rng = np.random.default_rng(11)
    await bm.reset(dut, INPUTS(dut))
    ops = [random_op(rng, argmax=True) for _ in range(3)]

    # Ties: several identical rows (identical outputs) spread across slices.
    tie = random_op(rng, wfmt=W4, argmax=True, max_blocks=2)
    top = int(np.argmax(tie.ref))
    for dup in rng.choice(tie.n, size=min(4, tie.n), replace=False):
        tie.w[dup], tie.ws[dup] = tie.w[top], tie.ws[top]
    ops.append(Op(tie.wfmt, tie.w, tie.ws, tie.x, tie.xs, argmax=True))

    # NaN rows never win; an all-NaN result picks row 0.
    nan = random_op(rng, wfmt=W8, argmax=True)
    nan.ws[rng.integers(0, nan.n, 3), 0] = 0x7FC0
    ops.append(Op(nan.wfmt, nan.w, nan.ws, nan.x, nan.xs, argmax=True))
    alln = random_op(rng, wfmt=W4, argmax=True)
    alln.xs[0] = 0x7FC0
    ops.append(Op(alln.wfmt, alln.w, alln.ws, alln.x, alln.xs, argmax=True))

    await run_ops(dut, ops, rng)


@cocotb.test()
async def qwen_shapes(dut):
    """Realistic quantized data at K = 2048 (or the largest K the config holds), both modes."""
    rng = np.random.default_rng(12)
    await bm.reset(dut, INPUTS(dut))
    k = 2048 if CFG.slice.max_k >= 2048 else CFG.slice.max_k
    n = 2 * ROWS_PER_BLOCK
    ops = []
    for wfmt, argmax in ((W4, False), (W8, True)):
        w, ws = quantize_weights_rtn(rng.normal(0, 0.02, (n, k)).astype(np.float32), wfmt)
        x, xs = quantize_activations(rng.normal(0, 1.0, k).astype(np.float32))
        ops.append(Op(wfmt, w, ws, x, xs, argmax))
    await run_ops(dut, ops, rng)


@cocotb.test()
async def illegal_command(dut):
    rng = np.random.default_rng(13)
    await bm.reset(dut, INPUTS(dut))
    dut.cmd_valid_i.value = 1
    dut.cmd_ngroups_i.value = 0
    dut.cmd_nrowblk_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.cmd_valid_i.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    assert dut.err_cmd_o.value == 1 and dut.busy_o.value == 0 and dut.cmd_ready_o.value == 1
    await bm.pulse(dut, dut.status_clr_i)
    await run_ops(dut, [random_op(rng)], rng)
    assert dut.err_cmd_o.value == 0
