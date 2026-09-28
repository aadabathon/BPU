"""bpu_fp32_mul / bpu_fp32_add against numpy float32, bit for bit.

Selected by the runner via BPU_FP_OP=mul|add. One operand pair per cycle, with
random bubbles to exercise the valid pipeline.
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

import fpvectors
from bpuref.fp import f32_bits, f32_equal, f32_from_bits


@cocotb.test()
async def matches_numpy_float32(dut):
    op = os.environ["BPU_FP_OP"]
    n = int(os.environ.get("BPU_FP_N", "40000"))
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))

    a, b = (fpvectors.mul_pairs if op == "mul" else fpvectors.add_pairs)(rng, n)
    with np.errstate(all="ignore"):
        fa, fb = f32_from_bits(a), f32_from_bits(b)
        ref = fa * fb if op == "mul" else fa + fb

    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    dut.valid_i.value = 0
    dut.rst_ni.value = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    bubbles = rng.random(len(a) * 2) < 0.1
    got = []
    i = cycle = 0
    while len(got) < len(a):
        await RisingEdge(dut.clk_i)
        if i < len(a) and not bubbles[cycle % len(bubbles)]:
            dut.a_i.value = int(a[i])
            dut.b_i.value = int(b[i])
            dut.valid_i.value = 1
            i += 1
        else:
            dut.valid_i.value = 0
        await ReadOnly()
        if dut.valid_o.value == 1:
            got.append(int(dut.y_o.value))
        cycle += 1
        assert cycle < 2 * len(a) + 100, "outputs stopped arriving"

    ok = f32_equal(np.array(got, dtype=np.uint32), ref)
    bad = np.flatnonzero(~ok)
    for j in bad[:10]:
        dut._log.error(f"{op} {a[j]:08x} {b[j]:08x}: rtl {got[j]:08x} ref {int(f32_bits(ref[j])):08x}")
    assert bad.size == 0, f"{bad.size} / {len(a)} mismatches"
    dut._log.info(f"{op}: {len(a)} vectors bit-exact")
