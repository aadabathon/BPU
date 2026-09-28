"""Full Qwen3.5 decode (tiny same-structure model) on bpu_compute_top, bit-exact
against bpuref.qwen.BpuQwen: after every token the whole SPM and the chosen token
must match. The testbench plays the host (SPM image, per-token inputs, the op
stream) and the memory side (weight streams on request)."""

import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

import busmodels as bm
from bpuref import fvu as F
from bpuref.configs import TOP_CONFIGS
from bpuref.fp import f32_equal
from bpuref.qmv import pack_array_streams
from bpuref.qwen import TINY, BpuQwen, QmvOp, quantize, random_weights

CFG = TOP_CONFIGS[os.environ.get("BPU_TOP_CFG", "tiny")]
V = CFG.fvu.vlanes
NS, LANES, R = CFG.nslice, CFG.qmv.lanes, CFG.qmv.row_interleave
SPM = int(os.environ.get("BPU_SPM_ELEMS", "65536"))
STEPS = int(os.environ.get("BPU_STEPS", "3"))
TOKENS = [5, 300, 42, 7, 99, 256]


class Host:
    def __init__(self, dut):
        self.dut = dut

    async def write(self, base, values):
        """Write fp32 values at element address `base` (word-masked)."""
        dut, bits = self.dut, np.asarray(values, np.float32).view(np.uint32)
        first, last = base // V, (base + len(bits) - 1) // V
        for w in range(first, last + 1):
            word, mask = 0, 0
            for l in range(V):
                e = w * V + l - base
                if 0 <= e < len(bits):
                    word |= int(bits[e]) << (32 * l)
                    mask |= 1 << l
            dut.host_we_i.value = 1
            dut.host_waddr_i.value = w
            dut.host_wmask_i.value = mask
            dut.host_wdata_i.value = word
            await RisingEdge(dut.clk_i)
        dut.host_we_i.value = 0

    async def read(self, n):
        dut, out = self.dut, np.zeros(-(-n // V) * V, np.uint32)
        for w in range(len(out) // V):
            dut.host_re_i.value = 1
            dut.host_raddr_i.value = w
            await RisingEdge(dut.clk_i)
            dut.host_re_i.value = 0
            await ReadOnly()
            word = int(dut.host_rdata_o.value)
            for l in range(V):
                out[w * V + l] = (word >> (32 * l)) & 0xFFFFFFFF
            await RisingEdge(dut.clk_i)
        return out[:n]


async def memory_side(dut, qw, names, rng):
    """Answer weight requests: pad the tensor to whole row blocks, stream layout L0."""
    dut.wreq_ready_i.value = 0
    while True:
        await ReadOnly()
        if dut.wreq_valid_o.value == 1:
            wid, nrb, wfmt = int(dut.wreq_id_o.value), int(dut.wreq_nrowblk_o.value), int(dut.wreq_wfmt_o.value)
            await RisingEdge(dut.clk_i)
            dut.wreq_ready_i.value = 1
            await RisingEdge(dut.clk_i)
            dut.wreq_ready_i.value = 0
            codes, scales, fmt = qw[names[wid]]
            assert fmt == wfmt
            rows = nrb * NS * R
            pc = np.zeros((rows, codes.shape[1]), np.int64)
            ps = np.zeros((rows, scales.shape[1]), np.uint16)
            pc[:len(codes)], ps[:len(scales)] = codes, scales
            streams = pack_array_streams(pc, ps, NS, LANES, R, wfmt)
            w = cocotb.start_soon(bm.multi_source(dut, dut.w_valid_i, dut.w_ready_o, dut.w_data_i,
                                                  LANES * 4, [b for b, _ in streams], rng, 0.1))
            s = cocotb.start_soon(bm.multi_source(dut, dut.ws_valid_i, dut.ws_ready_o, dut.ws_data_i,
                                                  16, [x for _, x in streams], rng, 0.1))
            await w
            await s
        else:
            await RisingEdge(dut.clk_i)


async def issue(dut, op, qw, wid):
    await bm.wait_high(dut, dut.cmd_ready_o)
    dut.cmd_valid_i.value = 1
    if isinstance(op, QmvOp):
        dut.cmd_qmv_i.value = 1
        dut.cmd_wid_i.value = wid[op.weight]
        dut.cmd_wfmt_i.value = qw[op.weight][2]
        dut.cmd_argmax_i.value = int(op.argmax)
        dut.cmd_k_i.value = op.k
        dut.cmd_n_i.value = op.n
        dut.cmd_x_i.value, dut.cmd_xs_i.value, dut.cmd_y_i.value = op.x, op.xs, op.y
    else:
        dut.cmd_qmv_i.value = 0
        dut.cmd_op_i.value = op.op
        dut.cmd_func_i.value = op.func
        dut.cmd_half_log2_i.value = op.half.bit_length() - 1
        dut.cmd_rows_i.value = op.rows
        dut.cmd_cols_i.value = op.cols
        for n in "dabcst":
            getattr(dut, f"cmd_{n}_i").value = getattr(op, n)
            getattr(dut, f"cmd_{n}s_i").value = getattr(op, n + "_stride")
    await RisingEdge(dut.clk_i)
    dut.cmd_valid_i.value = 0


@cocotb.test()
async def tiny_qwen_decode(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    w = random_weights(TINY, seed=11)
    qw = quantize(TINY, w)
    names = sorted(qw)
    wid = {n: i for i, n in enumerate(names)}
    ref = BpuQwen(TINY, w, qw)
    comp, mem = ref.comp, ref.comp.mem
    assert mem.size <= SPM

    await bm.reset(dut, (dut.cmd_valid_i, dut.host_we_i, dut.host_re_i, dut.status_clr_i,
                         dut.w_valid_i, dut.ws_valid_i, dut.wreq_ready_i))
    cocotb.start_soon(memory_side(dut, qw, names, rng))
    host = Host(dut)
    await host.write(0, comp.initial_spm())

    for step in range(STEPS):
        tok = TOKENS[step]
        pos = ref.pos
        for addr, vals in comp.host_inputs(tok, pos).items():
            await host.write(addr, vals)
        ops = comp.step_program(pos)
        expect = ref.step(tok)
        start = cocotb.utils.get_sim_time("ns") if hasattr(cocotb, "utils") else 0
        for op in ops:
            await issue(dut, op, qw, wid)
        await bm.wait_high(dut, dut.cmd_ready_o)
        for _ in range(4):
            await RisingEdge(dut.clk_i)
        assert dut.err_o.value == 0, "an operation was rejected"
        got = await host.read(mem.size)
        bad = np.flatnonzero(~f32_equal(got, ref.spm[:mem.size]))
        names_at = {e: n for n, (b, sz) in mem.regions.items() for e in range(b, b + sz)}
        for e in bad[:12]:
            dut._log.error(f"step {step}: spm[{e}] ({names_at.get(int(e), '?')}): "
                           f"rtl {got[e]:08x} ref {ref.spm.view(np.uint32)[e]:08x}")
        assert bad.size == 0, f"step {step}: {bad.size} / {mem.size} SPM elements differ"
        rtl_tok = int(got[mem["token"]].view(np.float32))
        assert rtl_tok == expect, f"step {step}: token {rtl_tok} != {expect}"
        n_q = sum(isinstance(o, QmvOp) for o in ops)
        dut._log.info(f"[{CFG.name}] step {step}: token {tok} -> {rtl_tok}; "
                      f"{len(ops)} ops ({n_q} QMV) bit-exact, whole SPM matches")
