"""bpu_sfu against bpuref.sfu, bit for bit. Functions are interleaved cycle by cycle."""

import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

import busmodels as bm
import fpvectors
from bpuref import sfu


def sfu_vectors(rng, n):
    """(func, a) pairs: specials for every function, random bits, and each
    function's hard regions."""
    funcs, vals = [], []

    def add(func, a):
        a = np.asarray(a, dtype=np.uint32)
        funcs.append(np.full(a.shape, func))
        vals.append(a)

    f32 = lambda x: np.asarray(x, dtype=np.float32).view(np.uint32)
    for func in sfu.FUNC_NAMES:
        add(func, fpvectors.SPECIALS)
        add(func, rng.integers(0, 1 << 32, n // 10, dtype=np.uint64).astype(np.uint32))
    add(sfu.RCP, f32(rng.uniform(-4, 4, n // 10)))
    add(sfu.RCP, rng.integers(0x7E800000, 0x7F800000, 200, dtype=np.uint64))       # FTZ edge
    add(sfu.RSQRT, f32(np.exp(rng.uniform(-80, 80, n // 10))))
    for func in (sfu.EXP, sfu.EXP2):
        add(func, f32(rng.uniform(-150, 150, n // 10)))
        add(func, f32(rng.uniform(-1e-3, 1e-3, n // 20)))
        add(func, f32(np.concatenate([np.linspace(-127, -125, 400), np.linspace(125, 129, 400),
                                      np.linspace(-88, -86, 400), np.linspace(87, 89, 400)])))
    one = 0x3F800000
    add(sfu.LOG2, np.arange(one - 2000, one + 2000))                                # both sides of 1
    add(sfu.LOG2, np.arange(0x3FC00000 - 1000, 0x3FC00000 + 1000))                  # 1.5 branch point
    add(sfu.LOG2, f32(np.exp(rng.uniform(-80, 80, n // 10))))
    funcs, vals = np.concatenate(funcs), np.concatenate(vals)
    perm = rng.permutation(len(vals))
    return funcs[perm], vals[perm]


@cocotb.test()
async def matches_reference(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    funcs, vals = sfu_vectors(rng, int(os.environ.get("BPU_SFU_N", "40000")))
    ref = np.empty(len(vals), dtype=np.uint32)
    for f in sfu.FUNC_NAMES:
        sel = funcs == f
        ref[sel] = sfu.sfu(f, vals[sel])

    await bm.reset(dut, (dut.valid_i,))
    bubbles = rng.random(len(vals) * 2) < 0.1
    got, i, cycle = [], 0, 0
    while len(got) < len(vals):
        await RisingEdge(dut.clk_i)
        if i < len(vals) and not bubbles[cycle % len(bubbles)]:
            dut.func_i.value = int(funcs[i])
            dut.a_i.value = int(vals[i])
            dut.valid_i.value = 1
            i += 1
        else:
            dut.valid_i.value = 0
        await ReadOnly()
        if dut.valid_o.value == 1:
            got.append(int(dut.y_o.value))
        cycle += 1
        assert cycle < 2 * len(vals) + 100, "outputs stopped arriving"

    got = np.array(got, dtype=np.uint32)
    bad = np.flatnonzero(got != ref)
    for j in bad[:10]:
        dut._log.error(f"{sfu.FUNC_NAMES[int(funcs[j])]}({vals[j]:08x}): rtl {got[j]:08x} ref {ref[j]:08x}")
    assert bad.size == 0, f"{bad.size} / {len(vals)} mismatches"
    dut._log.info(f"{len(vals)} vectors bit-exact")
