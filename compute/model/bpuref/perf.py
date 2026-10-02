"""Cycle model of bpu_core running a scheduled program.

Per command, the engine cycles come from the RTL's issue rules: operand reads per
read port, collector-slot turnaround, write-buffer and reduction credit loops,
VVECMAT row recurrence, the reduction merge, the QMV load gearbox and beats per
slice. On top, a replay of the sequencer: descriptors accepted in order (tag reuse,
queue depth), each unit in order, dependencies across units, units in parallel.
It is calibrated against the measured cycles of the tiny-Qwen RTL test (the decode
test asserts that it stays within 5%), then used to project Qwen3.5-2B.

    python -m bpuref.perf            # calibration table + 2B projection
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product

from . import fvu as F
from .configs import TOP_CONFIGS, TopConfig
from .isa import UNIT_MAT, UNIT_MEM, UNIT_VEC
from .qmv import GROUP
from .qwen import QWEN35_2B, TINY, Compiler, QmvOp, QwenConfig, quantize, random_weights
from .sched import schedule as make_cmds


def _pipe(mask: int) -> int:
    return bin(mask).count("1")


def _pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _expected_max_load(k: int, nbanks: int) -> float:
    """E[largest number of k uniform requests landing on one of nbanks banks]."""
    if k <= 1 or nbanks == 1:
        return float(k if nbanks == 1 else min(k, 1))
    tot = 0
    for combo in product(range(nbanks), repeat=k):
        tot += max(combo.count(b) for b in set(combo))
    return tot / nbanks ** k


class CycleModel:
    MERGE_PER_NODE = 1.08   # pipelined reduction merge, measured (tb/cocotb_fvu_reduce.py)
    SEQ_GAP = 2             # sequencer: done -> next issue on the same unit
    FVU_TAIL = 5            # FVU fixed cost beyond the read latency and lanes
    ARGMAX_TAIL = 30        # argmax: the array's final merge across slices
    QMV_TAIL = 5            # matrix unit fixed cost beyond pipeline and writes

    def __init__(self, cfg: TopConfig, beat_rate: float = 1.0, attn_on_qmv: bool = False,
                 rd_ports: int | None = None, fwd: bool = True):
        """Knobs: beat_rate = weight beats per cycle per slice (1.0 = memory keeps up);
        attn_on_qmv = attention K.q and p^T V on the QMV engine (an INT8 KV cache
        streamed like weights); rd_ports / fwd override the vector unit's read ports and
        VVECMAT forwarding."""
        self.cfg, self.beat_rate, self.attn_on_qmv = cfg, beat_rate, attn_on_qmv
        f, q = cfg.fvu, cfg.qmv
        self.V = f.vlanes
        self.np = rd_ports or f.rd_ports
        self.acc_depth = f.acc_depth if fwd else 0
        self.la = _pipe(f.add_pipe)
        self.ltot = f.ltot
        self.rdlat = cfg.sram.rd_lat
        self.q_lat = (1 + q.prod_reg + (0 if not q.tree_reg_every else (q.lanes.bit_length() - 1) // q.tree_reg_every)
                      + 1 + max(_pipe(q.mul_pipe), q.i2f_reg) + _pipe(q.mul_pipe) + _pipe(q.add_pipe) + 2)
        nb = cfg.sram.nbanks
        self._conflict = {k: _expected_max_load(k, nb) for k in range(1, 4)}

    # -- vector unit ------------------------------------------------------------
    def _port_reads(self, need: dict) -> int:
        """(port cycles, expected extra cycles from bank conflicts) to fetch one item."""
        cls = ({"s": 0, "t": 0, "a": 0, "b": 0, "c": 0} if self.np == 1 else
               {"s": 0, "t": 0, "a": 0, "b": 1, "c": 1} if self.np == 2 else
               {"s": 0, "t": 0, "a": 0, "b": 1, "c": 2})
        per = defaultdict(int)
        for k, n in need.items():
            per[cls[k]] += n
        if not per:
            return 0, 0.0
        conc = min(len(per), 3)                     # reads in flight together
        return max(per.values()), (self._conflict[conc] - 1 if conc > 1 else 0.0)

    def fvu(self, op: F.FvuOp) -> int:
        f, V, uses = self.cfg.fvu, self.V, F.USES[op.op]
        words = -(-op.cols // V)
        red = op.op in F.REDUCTIONS
        vm = op.op == F.VVECMAT
        wrow = max(64, _pow2(op.cols)) // V if red else words
        fwd = vm and self.acc_depth and words <= self.acc_depth
        nsub = V // f.nsfu if op.op == F.VSFU else 1

        def row_cost(r):
            c = 0.0
            for w in range(wrow):
                pad = red and w >= words
                need = {"a": int(not pad)}
                need["s"] = int("s" in uses and w == 0)
                need["t"] = int(op.op == F.VMULG and (w * V) % 64 == 0)
                need["b"] = int(("b" in uses and not pad) or (vm and r > 0 and not fwd))
                need["c"] = int("c" in uses and not pad)
                port, conflict = self._port_reads({k: n for k, n in need.items() if n})
                # A slot turns over every RdLat + 3 cycles (allocate, request, response,
                # launch), stretched by bank conflicts among its reads.
                slot = (self.rdlat + 3 + conflict) / f.nslot
                c += max(port + conflict, nsub, slot)
            if vm and r > 0:
                loop = self.ltot + 2 if fwd else self.ltot + 5 + self.rdlat
                c = max(c, loop)
            return c

        cycles = row_cost(0) + (op.rows - 1) * row_cost(1) if op.rows > 1 else row_cost(0)
        items = op.rows * wrow
        if red:
            lt = (V.bit_length() - 1) * self.la
            per_node = (self.ltot + lt + 4) / f.red_fifo
            if op.op in (F.RSUM, F.RDOT):
                per_node = max(per_node, self.MERGE_PER_NODE)
            cycles = max(cycles, items * per_node)
        else:
            writes = items * nsub if not (vm and fwd) else words * nsub
            cycles = max(cycles, writes * (self.ltot + 3) / f.wb_depth)
        tail = self.rdlat + self.ltot + self.FVU_TAIL
        if red:
            tail += (V.bit_length() - 1) * self.la + 3
            if op.op in (F.RSUM, F.RDOT):
                tail += (wrow.bit_length() - 1) * max(self.la, 1) + 1
        return int(cycles + tail)

    # -- matrix unit --------------------------------------------------------------
    def qmv(self, op: QmvOp, wfmt: int = 0) -> int:
        cfg, q, V = self.cfg, self.cfg.qmv, self.V
        steps_per_word = max(1, V // q.lanes)
        g = op.k // GROUP
        load = (op.k // V) * steps_per_word + g + self.rdlat + 3
        rows_blk = cfg.nslice * q.row_interleave
        nrowblk = -(-op.n // rows_blk)
        cpg = (GROUP // q.lanes) * (2 if wfmt else 1)
        beats = nrowblk * g * q.row_interleave * cpg
        stream = max(beats / self.beat_rate, nrowblk * rows_blk if not op.argmax else 0)
        tail = self.q_lat + (cfg.nslice + 3 + self.ARGMAX_TAIL if op.argmax else 2) + self.QMV_TAIL
        return int(load + stream + tail)

    def attention_on_qmv(self, op: F.FvuOp) -> int:
        macs = op.rows * op.cols
        return int(macs / (self.cfg.nslice * self.cfg.qmv.lanes)) + self.q_lat + 20

    def duration(self, op, qw=None, context=None) -> tuple[str, int]:
        if (self.attn_on_qmv and context and not isinstance(op, QmvOp) and getattr(op, "rows", 0) == context
                and op.op in (F.RDOT, F.VVECMAT)):
            return "attn_qmv", self.attention_on_qmv(op)
        if isinstance(op, QmvOp):
            return "matrix", self.qmv(op, qw[op.weight][2] if qw else 0)
        if isinstance(op, F.FvuOp):
            kind = ("vec_red" if op.op in F.REDUCTIONS else "vec_vecmat" if op.op == F.VVECMAT else "vec_ew")
            return kind, self.fvu(op)
        return "memory", 0

    # -- the sequencer ---------------------------------------------------------------
    def schedule(self, cmds, qw=None, context=None) -> dict:
        """Replay the sequencer over scheduled commands; returns cycles (total and busy
        cycles per unit kind)."""
        cfg = self.cfg
        accept = [0] * len(cmds)
        issue = [0] * len(cmds)
        done = [0] * len(cmds)
        unit_free = defaultdict(int)
        tag_free = defaultdict(int)
        unit_issues = defaultdict(list)            # per unit: issue times, in order
        busy = defaultdict(int)
        t_acc = 0
        for j, c in enumerate(cmds):
            a = max(t_acc, tag_free[c.tag])
            q = unit_issues[c.unit]
            if len(q) >= cfg.qdepth:                # the queue entry frees when it issues
                a = max(a, q[-cfg.qdepth] + 1)
            accept[j] = a
            t_acc = a + 1
            kind, dur = self.duration(c.op, qw, context)
            s = max(a + 1, unit_free[c.unit], max((done[i] + 1 for i in c.deps), default=0))
            issue[j] = s
            done[j] = s + dur
            unit_free[c.unit] = done[j] + self.SEQ_GAP
            tag_free[c.tag] = done[j] + 1
            q.append(s)
            busy[kind] += dur
        out = {k: int(v) for k, v in busy.items()}
        out["total"] = int(max(done) + 1) if cmds else 0
        self.last_done = done
        return out

    def program(self, ops, qw=None, context=None, ntags=None) -> dict:
        return self.schedule(make_cmds(ops, ntags or self.cfg.ntags, qw), qw, context)


def tiny_program():
    w = random_weights(TINY, seed=11)
    qw = quantize(TINY, w)
    comp = Compiler(TINY, w, qw)
    return comp.step_program(1), qw


def projection(cfg: QwenConfig, top: TopConfig, context: int, clock_hz: float = 250e6,
               beat_rate: float = 1.0, **knobs) -> dict:
    """Modelled cycles per decoded token at a context length. The compiler needs
    only shapes (weights are used for SRAM images, not for programs)."""
    import dataclasses
    small = dataclasses.replace(cfg, max_pos=context)
    ops = Compiler(small, None, None).step_program(context - 1)
    res = CycleModel(top, beat_rate, **knobs).program(ops, context=context)
    res["tokens_per_s"] = clock_hz / res["total"]
    return res


if __name__ == "__main__":
    ops, qw = tiny_program()
    print("tiny-Qwen decode step, modelled cycles per configuration:")
    for name, top in TOP_CONFIGS.items():
        m = CycleModel(top, beat_rate=0.9).program(ops, qw)
        print(f"  {name:5s}", m)
    print("Qwen3.5-2B at the fpga config, 250 MHz, tokens/s:")
    scen = [("current RTL", {}), ("+ attention on QMV", {"attn_on_qmv": True})]
    print(f"  {'':22s}" + "".join(f"{c:>10d}" for c in (128, 2048, 8192)))
    for label, knobs in scen:
        row = [projection(QWEN35_2B, TOP_CONFIGS["fpga"], c, **knobs)["tokens_per_s"] for c in (128, 2048, 8192)]
        print(f"  {label:22s}" + "".join(f"{v:10.1f}" for v in row))
    m = projection(QWEN35_2B, TOP_CONFIGS["fpga"], 2048)
    print("  breakdown at 2048 (busy cycles per unit kind; units overlap):",
          {k: v for k, v in m.items() if k != "tokens_per_s"})
