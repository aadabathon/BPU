"""Valid/ready bus functional models shared by the cocotb testbenches.

All models follow the same phase discipline: drive in the writable phase right
after a rising edge, sample in ReadOnly, then advance to the next rising edge.
Sources obey AXI-Stream rules (valid and data held until the handshake).
"""

import cocotb
from cocotb.triggers import ReadOnly, RisingEdge


async def reset(dut, idle_inputs=(), cycles=2):
    """Start a 10 ns clock, hold rst_ni low for `cycles`, zero the given inputs."""
    from cocotb.clock import Clock
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    for sig in idle_inputs:
        sig.value = 0
    dut.rst_ni.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def source(dut, valid, ready, data, items, rng, p_idle):
    """Single valid/ready source. Returns cycles from first to last handshake (inclusive)."""
    i, asserted, cycle, first, last = 0, False, 0, None, 0
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
    return 0 if first is None else last - first + 1


async def multi_source(dut, valid, ready, data, width, queues, rng, p_idle):
    """Independent sources packed into one vector: lane s drives bit s of valid/ready
    and bits [s*width, (s+1)*width) of data."""
    n = len(queues)
    idx, asserted, held = [0] * n, [False] * n, [0] * n
    while any(idx[s] < len(queues[s]) for s in range(n)):
        vbits = dword = 0
        for s in range(n):
            if not asserted[s] and idx[s] < len(queues[s]) and rng.random() >= p_idle:
                asserted[s] = True
                held[s] = queues[s][idx[s]]
            if asserted[s]:
                vbits |= 1 << s
                dword |= held[s] << (s * width)
        valid.value = vbits
        data.value = dword
        await ReadOnly()
        rbits = int(ready.value)
        await RisingEdge(dut.clk_i)
        for s in range(n):
            if asserted[s] and (rbits >> s) & 1:
                idx[s] += 1
                asserted[s] = False
    valid.value = 0


async def sink(dut, valid, ready, fields, count, rng, p_stall, out):
    """Accept `count` transfers, randomly withholding ready. Appends a tuple of the
    integer values of `fields` (or the single value) per transfer to `out`."""
    while len(out) < count:
        ready.value = int(rng.random() >= p_stall)
        await ReadOnly()
        if valid.value == 1 and ready.value == 1:
            vals = tuple(int(f.value) for f in fields)
            out.append(vals if len(vals) > 1 else vals[0])
        await RisingEdge(dut.clk_i)
    ready.value = 0


async def wait_high(dut, sig):
    """Return in the writable phase of the first cycle where `sig` is high."""
    while True:
        await ReadOnly()
        if sig.value == 1:
            break
        await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)


async def pulse(dut, sig):
    await RisingEdge(dut.clk_i)
    sig.value = 1
    await RisingEdge(dut.clk_i)
    sig.value = 0
