"""Full Qwen3.5 decode (tiny same-structure model) on bpu_core, bit-exact against
bpuref.qwen.BpuQwen: after every token the whole program region of the shared SRAM
and the chosen token must match.

The testbench plays the control SoC (the step program as tagged descriptors with
dependency masks, bpuref.sched), and the memory manager: SRAM image and per-token
inputs through its SRAM ports, weight streams on request, and the memory-unit
commands (accepted and completed after a random delay)."""

import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb.utils import get_sim_time

import busmodels as bm
from bpuref.configs import TOP_CONFIGS
from bpuref.fp import f32_equal
from bpuref.isa import UNIT_MAT, UNIT_MEM, UNIT_VEC, Cmd
from bpuref.perf import CycleModel
from bpuref.qmv import pack_array_streams
from bpuref.qwen import TINY, BpuQwen, QmvOp, quantize, random_weights
from bpuref.sched import schedule

CFG = TOP_CONFIGS[os.environ.get("BPU_TOP_CFG", "tiny")]
V = CFG.fvu.vlanes
NS, LANES, R = CFG.nslice, CFG.qmv.lanes, CFG.qmv.row_interleave
STEPS = int(os.environ.get("BPU_STEPS", "3"))
TOKENS = [5, 300, 42, 7, 99, 256]


async def memory_weights(dut, qw, names, rng):
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


async def memory_commands(dut, rng, log):
    """Memory-unit commands: accept, then complete after a random delay."""
    dut.mm_cmd_ready_i.value = 0
    dut.mm_done_i.value = 0
    dut.mm_err_i.value = 0
    while True:
        dut.mm_cmd_ready_i.value = int(rng.random() < 0.7)
        await ReadOnly()
        fire = dut.mm_cmd_valid_o.value == 1 and dut.mm_cmd_ready_i.value == 1
        body = int(dut.mm_cmd_body_o.value) if fire else 0
        await RisingEdge(dut.clk_i)
        if fire:
            dut.mm_cmd_ready_i.value = 0
            for _ in range(int(rng.integers(1, 20))):
                await RisingEdge(dut.clk_i)
            log.append(("done", body, get_sim_time("ns")))
            dut.mm_done_i.value = 1
            await RisingEdge(dut.clk_i)
            dut.mm_done_i.value = 0


async def run_cmds(dut, cmds, mon):
    """Push descriptors in order (each when the sequencer accepts it), then wait until
    every command has completed. Returns the cycles taken."""
    start = get_sim_time("ns")
    for c in cmds:
        dut.desc_valid_i.value = 1
        dut.desc_unit_i.value = c.unit
        dut.desc_tag_i.value = c.tag
        dut.desc_wait_i.value = c.wait
        dut.desc_body_i.value = c.body
        while True:
            await ReadOnly()
            ok = dut.desc_ready_o.value == 1
            await RisingEdge(dut.clk_i)
            if ok:
                mon.accepted.append((c, get_sim_time("ns")))
                break
    dut.desc_valid_i.value = 0
    while True:
        await ReadOnly()
        if int(dut.pending_o.value) == 0:
            break
        await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    return int((get_sim_time("ns") - start) // 10)


class Monitor:
    """Watches completions: records each command's completion time and error bit, and
    checks that no command completes before the commands it waits for."""

    def __init__(self, dut, ntags):
        self.dut, self.ntags = dut, ntags
        self.accepted = []
        self.tag_cmd_order = []
        self.tag_cmd = {}                 # tag -> index of the command holding it
        self.done_at = {}                 # command index -> time
        self.errors = []

    async def run(self):
        dut = self.dut
        while True:
            await ReadOnly()
            cpl = int(dut.cpl_o.value)
            now = get_sim_time("ns")
            # descriptors accepted so far own their tags
            while len(self.tag_cmd_order) < len(self.accepted):
                i = len(self.tag_cmd_order)
                self.tag_cmd_order.append(i)
                self.tag_cmd[self.accepted[i][0].tag] = i
            await RisingEdge(dut.clk_i)
            if cpl:
                await ReadOnly()
                err = int(dut.err_o.value)
                for t in range(self.ntags):
                    if (cpl >> t) & 1:
                        i = self.tag_cmd[t]
                        self.done_at[i] = now
                        if (err >> t) & 1:
                            self.errors.append(self.accepted[i][0].op)
                await RisingEdge(dut.clk_i)


def check_order(cmds, done_at):
    """Every command finished after the commands it depends on."""
    for j, c in enumerate(cmds):
        for i in c.deps:
            assert done_at[i] < done_at[j] or c.unit == cmds[i].unit, \
                f"command {j} ({c.op}) completed before its dependency {i} ({cmds[i].op})"


@cocotb.test()
async def tiny_qwen_decode(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    w = random_weights(TINY, seed=11)
    qw = quantize(TINY, w)
    names = sorted(qw)
    ref = BpuQwen(TINY, w, qw)
    comp, mem = ref.comp, ref.comp.mem
    assert mem.size <= CFG.elems()

    await bm.reset(dut, (dut.desc_valid_i, dut.mm_rd_valid_i, dut.mm_wr_valid_i, dut.mm_cmd_ready_i,
                         dut.mm_done_i, dut.w_valid_i, dut.ws_valid_i, dut.wreq_ready_i))
    mm = bm.SramHost(dut, V, prefix="mm_")
    cocotb.start_soon(memory_weights(dut, qw, names, rng))
    cocotb.start_soon(memory_commands(dut, rng, []))
    await mm.write_elems(0, comp.initial_spm())

    for step in range(STEPS):
        tok = TOKENS[step]
        pos = ref.pos
        for addr, vals in comp.host_inputs(tok, pos).items():
            await mm.write_elems(addr, np.asarray(vals, np.float32))
        ops = comp.step_program(pos)
        cmds = schedule(ops, CFG.ntags, qw, names)
        expect = ref.step(tok)

        mon = Monitor(dut, CFG.ntags)
        task = cocotb.start_soon(mon.run())
        cycles = await run_cmds(dut, cmds, mon)
        for _ in range(3):
            await RisingEdge(dut.clk_i)
        task.cancel()
        assert not mon.errors, f"step {step}: commands completed with an error: {mon.errors[:3]}"
        assert len(mon.done_at) == len(cmds), f"{len(cmds) - len(mon.done_at)} completions missing"
        check_order(cmds, mon.done_at)

        got = await mm.read_elems(0, mem.size)
        bad = np.flatnonzero(~f32_equal(got, ref.spm[:mem.size]))
        names_at = {e: n for n, (b, sz) in mem.regions.items() for e in range(b, b + sz)}
        for e in bad[:12]:
            dut._log.error(f"step {step}: sram[{e}] ({names_at.get(int(e), '?')}): "
                           f"rtl {got[e]:08x} ref {ref.spm.view(np.uint32)[e]:08x}")
        assert bad.size == 0, f"step {step}: {bad.size} / {mem.size} elements differ"
        rtl_tok = int(got[mem["token"]].view(np.float32))
        assert rtl_tok == expect, f"step {step}: token {rtl_tok} != {expect}"
        nq = sum(isinstance(c.op, QmvOp) for c in cmds)
        nw = sum(1 for c in cmds if c.wait)
        dut._log.info(f"[{CFG.name}] step {step}: token {tok} -> {rtl_tok}; {len(cmds)} commands "
                      f"({nq} matrix, {nw} with cross-unit waits) bit-exact, SRAM region matches; "
                      f"{cycles} cycles")
        dump = os.environ.get("BPU_PERF_DUMP")
        if dump and step == 1:                    # per-command completion times, for calibration
            import json
            t0 = min(t for _, t in mon.accepted)
            with open(dump, "w") as fh:
                json.dump([(str(c.op), c.unit, int((mon.done_at[i] - t0) // 10)) for i, c in enumerate(cmds)], fh)
        model = CycleModel(CFG, beat_rate=0.9).schedule(cmds)["total"]
        err = model / cycles - 1
        dut._log.info(f"[{CFG.name}] step {step}: cycle model {model} ({100 * err:+.1f}%)")
        if os.environ.get("BPU_MODEL_CHECK", "1") == "1":
            assert abs(err) < 0.05, f"cycle model off by {100 * err:+.1f}%: recalibrate bpuref/perf.py"


@cocotb.test()
async def dependencies_across_units(dut):
    """Memory-unit commands interleaved with vector work: a command never starts before
    the commands it waits for complete, tags are reused, and a reserved unit errors."""
    from bpuref import fvu as F
    from bpuref.isa import vec_body
    rng = np.random.default_rng(3)
    await bm.reset(dut, (dut.desc_valid_i, dut.mm_rd_valid_i, dut.mm_wr_valid_i, dut.mm_cmd_ready_i,
                         dut.mm_done_i, dut.w_valid_i, dut.ws_valid_i, dut.wreq_ready_i))
    log = []
    cocotb.start_soon(memory_commands(dut, rng, log))
    cmds = []
    for j in range(40):
        tag = j % CFG.ntags
        if j % 3 == 2:
            wait = 1 << cmds[-1].tag if cmds else 0
            cmds.append(Cmd(UNIT_MEM, tag, wait, ("mem", j), body=j, deps=[j - 1]))
        else:
            op = F.FvuOp(F.VCOPY, 1, 64, d=128 + 64 * (j % 4), a=0)
            wait = 1 << cmds[-1].tag if cmds and cmds[-1].unit == UNIT_MEM else 0
            deps = [j - 1] if wait else []
            cmds.append(Cmd(UNIT_VEC, tag, wait, op, vec_body(op), deps))
    mon = Monitor(dut, CFG.ntags)
    task = cocotb.start_soon(mon.run())
    await run_cmds(dut, cmds, mon)
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    task.cancel()
    assert not mon.errors
    assert len(mon.done_at) == len(cmds)
    check_order(cmds, mon.done_at)
    # Reserved unit: completes with an error.
    mon = Monitor(dut, CFG.ntags)
    task = cocotb.start_soon(mon.run())
    await run_cmds(dut, [Cmd(3, 0, 0, "reserved")], mon)
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    task.cancel()
    assert mon.errors == ["reserved"], mon.errors
