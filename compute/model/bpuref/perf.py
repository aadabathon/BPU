"""Cycle model of bpu_compute_top running a compiled program.

Counts cycles from the RTL's actual issue rules (per-item SPM reads, reduction
merge rate, VVECMAT row spacing, QMV gearbox, beats per slice) rather than from
peak rates. It is calibrated against the measured op-stream cycles of the
tiny-Qwen RTL test (compute_top), then used to project Qwen3.5-2B.

    python -m bpuref.perf            # calibration table + 2B projection
"""

from __future__ import annotations

from collections import defaultdict

from . import fvu as F
from .configs import TOP_CONFIGS, TopConfig
from .qmv import GROUP
from .qwen import QWEN35_2B, TINY, Compiler, QmvOp, QwenConfig, quantize, random_weights


def _pipe(mask: int) -> int:
    return bin(mask).count("1")


def _pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


class CycleModel:
    OP_OVERHEAD = 3        # dispatcher + issue handshake between operations

    def __init__(self, cfg: TopConfig, beat_rate: float = 1.0, spm_ports: int = 1,
                 merge_per_node: float | None = None, attn_on_qmv: bool = False):
        """Knobs beyond the current RTL (all default to it): spm_ports = operand reads
        per cycle; merge_per_node = reduction merge cycles per word node (None = the RTL's
        3 + AddLatency); attn_on_qmv = attention K.q and p^T V run on the QMV engine at its
        MAC rate (i.e. an INT8 KV cache streamed like weights)."""
        self.cfg = cfg
        self.beat_rate = beat_rate            # weight beats per cycle per slice (1.0 = HBM keeps up)
        self.spm_ports = spm_ports
        self.merge_per_node = merge_per_node
        self.attn_on_qmv = attn_on_qmv
        f, q = cfg.fvu, cfg.qmv
        self.V = f.vlanes
        self.la = _pipe(f.add_pipe)
        self.ltot = max(_pipe(f.mul_pipe) + self.la, _pipe(f.sfu_pipe))
        self.vm_period = self.ltot + 6
        self.q_lat = (1 + q.prod_reg + (0 if not q.tree_reg_every else (q.lanes.bit_length() - 1) // q.tree_reg_every)
                      + 1 + max(_pipe(q.mul_pipe), q.i2f_reg) + _pipe(q.mul_pipe) + _pipe(q.add_pipe) + 2)

    # -- FVU ----------------------------------------------------------------
    def fvu(self, op: F.FvuOp) -> int:
        V, uses = self.V, F.USES[op.op]
        words = -(-op.cols // V)
        red = op.op in F.REDUCTIONS
        wrow = max(64, _pow2(op.cols)) // V if red else words
        cycles = 0
        for r in range(op.rows):
            row = 0
            for w in range(wrow):
                pad = red and w >= words
                reads = 0
                reads += ("s" in uses and w == 0)
                reads += (op.op == F.VMULG and (w * V) % 64 == 0)
                reads += (not pad)
                reads += (("b" in uses and not pad) or (op.op == F.VVECMAT and r > 0))
                reads += ("c" in uses and not pad)
                row += max(1, -(-reads // self.spm_ports))
            if op.op == F.VVECMAT and r > 0:
                row = max(row, self.vm_period)
            cycles += row
        if red and op.op in (F.RSUM, F.RDOT):
            # Merge FSM: pop + push + one merge of (La + 1) cycles per word node.
            per_node = self.merge_per_node if self.merge_per_node is not None else 3 + self.la
            cycles = max(cycles, int(op.rows * wrow * per_node))
        drain = 2 + self.ltot + (1 + (V.bit_length() - 1) * self.la + 3 if red else 1)
        return cycles + drain + self.OP_OVERHEAD

    # -- QMV ----------------------------------------------------------------
    def qmv(self, op: QmvOp, wfmt: int = 0) -> int:
        cfg, q = self.cfg, self.cfg.qmv
        chunk = min(self.V, q.lanes)
        load = op.k // chunk + 2 + op.k // GROUP + 2 + 3
        rows_blk = cfg.nslice * q.row_interleave
        nrowblk = -(-op.n // rows_blk)
        cpg = (GROUP // q.lanes) * (2 if wfmt else 1)
        beats = nrowblk * (op.k // GROUP) * q.row_interleave * cpg
        stream = max(beats / self.beat_rate, nrowblk * rows_blk if not op.argmax else 0)
        tail = self.q_lat + (cfg.nslice + 3 if op.argmax else 2)
        return int(load + stream + tail) + self.OP_OVERHEAD

    def attention_on_qmv(self, op: F.FvuOp) -> int:
        macs = op.rows * op.cols
        return int(macs / (self.cfg.nslice * self.cfg.qmv.lanes)) + self.q_lat + 20

    def program(self, ops, qw=None, context: int | None = None) -> dict[str, int]:
        out = defaultdict(int)
        for op in ops:
            if (self.attn_on_qmv and context and not isinstance(op, QmvOp)
                    and op.rows == context and op.op in (F.RDOT, F.VVECMAT)):
                out["attn_qmv"] += self.attention_on_qmv(op)
                continue
            if isinstance(op, QmvOp):
                out["qmv"] += self.qmv(op, qw[op.weight][2] if qw else 0)
            else:
                kind = ("fvu_red" if op.op in F.REDUCTIONS else
                        "fvu_vecmat" if op.op == F.VVECMAT else "fvu_ew")
                out[kind] += self.fvu(op)
        out["total"] = sum(out.values())
        return dict(out)


def tiny_program():
    w = random_weights(TINY, seed=11)
    qw = quantize(TINY, w)
    comp = Compiler(TINY, w, qw)
    return comp.step_program(1), qw


def projection(cfg: QwenConfig, top: TopConfig, context: int, clock_hz: float = 250e6,
               beat_rate: float = 1.0, **knobs) -> dict:
    """Modelled cycles per decoded token at a context length. The compiler needs
    only shapes (weights are used for SPM images, not for programs)."""
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
        print(f"  {name:5s}", {k: int(v) for k, v in m.items()})
    print("Qwen3.5-2B at the fpga config, 250 MHz, tokens/s (cumulative improvements):")
    scen = [("current RTL", {}),
            ("+ 3-port SPM", {"spm_ports": 3}),
            ("+ pipelined merge", {"spm_ports": 3, "merge_per_node": 1}),
            ("+ attention on QMV", {"spm_ports": 3, "merge_per_node": 1, "attn_on_qmv": True})]
    print(f"  {'':22s}" + "".join(f"{c:>10d}" for c in (128, 2048, 8192)))
    for label, knobs in scen:
        row = [projection(QWEN35_2B, TOP_CONFIGS["fpga"], c, **knobs)["tokens_per_s"] for c in (128, 2048, 8192)]
        print(f"  {label:22s}" + "".join(f"{v:10.1f}" for v in row))
    m = projection(QWEN35_2B, TOP_CONFIGS["fpga"], 2048)
    print("  current RTL breakdown at 2048:", {k: int(v) for k, v in m.items() if k != "tokens_per_s"})
