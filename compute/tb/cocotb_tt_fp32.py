"""tt_um_bpu_fp32 (the Tiny Tapeout learning run) driven through its byte-wide host
protocol, exactly as the demo board would, bit-exact against numpy float32."""

import os

import cocotb
import numpy as np
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge

from bpuref.fp import f32_bits, f32_equal
from fpvectors import SPECIALS

CMD_A, CMD_B, CMD_ADD, CMD_SUB, CMD_MUL, CMD_LOOP = range(6)


async def strobe(dut, cmd, idx=0, data=0):
    """One host command: set the controls, then a slow STROBE pulse."""
    dut.ui_in.value = data
    dut.uio_in.value = (idx << 4) | (cmd << 1)
    await ClockCycles(dut.clk, 3)
    dut.uio_in.value = (idx << 4) | (cmd << 1) | 1
    await ClockCycles(dut.clk, 4)
    dut.uio_in.value = (idx << 4) | (cmd << 1)
    await ClockCycles(dut.clk, 2)


async def load(dut, cmd, value):
    for i in range(4):
        await strobe(dut, cmd, i, (value >> (8 * i)) & 0xFF)


async def read_result(dut):
    r = 0
    for i in range(4):
        dut.uio_in.value = i << 4
        await ClockCycles(dut.clk, 3)
        await ReadOnly()
        r |= int(dut.uo_out.value) << (8 * i)
        await RisingEdge(dut.clk)
    return r


def vectors(rng, n):
    pool = [int(x) for x in SPECIALS]                                         # bit patterns
    a = [int(x) for x in rng.integers(0, 2**32, n, dtype=np.uint64)]          # any bits
    b = [int(x) for x in rng.integers(0, 2**32, n, dtype=np.uint64)]
    g = f32_bits(rng.normal(0, 1, 2 * n).astype(np.float32)).tolist()        # typical values
    a += g[:n] + [pool[i % len(pool)] for i in range(len(pool) ** 2)]
    b += g[n:] + [pool[i // len(pool)] for i in range(len(pool) ** 2)]
    # The first vectors docs/info.md suggests for silicon bring-up:
    # 1 + 2^-24 (tie -> 1.0), min subnormal * 0.5 (tie -> +0), inf - inf (NaN).
    a += [0x3F800000, 0x00000001, 0x7F800000]
    b += [0x33800000, 0x3F000000, 0x7F800000]
    return list(zip(a, b))


@cocotb.test()
async def host_protocol_bit_exact(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    from cocotb.clock import Clock
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)
    assert int(dut.uio_oe.value) == 0, "all bidirectional pins must be inputs"

    pairs = vectors(rng, int(os.environ.get("BPU_TT_N", "200")))
    for a, b in pairs:
        fa, fb = np.uint32(a).view(np.float32), np.uint32(b).view(np.float32)
        await load(dut, CMD_A, a)
        await load(dut, CMD_B, b)
        with np.errstate(all="ignore"):
            expect = {CMD_ADD: fa + fb, CMD_SUB: fa - fb, CMD_MUL: fa * fb,
                      CMD_LOOP: np.float32(fa)}
        for cmd, ref in expect.items():
            await strobe(dut, cmd)
            got = await read_result(dut)
            ok = bool(f32_equal(got, ref)) if cmd != CMD_LOOP else got == a
            assert ok, f"cmd {cmd}: a={a:08x} b={b:08x}: got {got:08x}, want {int(f32_bits(ref)):08x}"
    dut._log.info(f"{len(pairs)} operand pairs x add/sub/mul/loopback bit-exact through the pins")
