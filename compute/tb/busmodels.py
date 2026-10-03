"""Valid/ready bus functional models shared by the cocotb testbenches.

All models follow the same phase discipline: drive in the writable phase right
after a rising edge, sample in ReadOnly, then advance to the next rising edge.
Sources obey AXI-Stream rules (valid and data held until the handshake).
"""

import cocotb
import numpy as np
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


class SramHost:
    """A host read port and write port on bpu_sram_shared (or a system exposing them):
    pipelined word reads/writes, and random competing traffic while an op runs.
    `prefix` names the ports: <prefix>rd_valid_i, <prefix>rd_ready_o, <prefix>rd_addr_i,
    <prefix>rd_rvalid_o, <prefix>rd_rdata_o, <prefix>wr_valid_i, ... ."""

    def __init__(self, dut, lanes, prefix="host_"):
        self.dut, self.lanes = dut, lanes
        g = lambda n: getattr(dut, prefix + n)
        self.rv, self.rr, self.ra = g("rd_valid_i"), g("rd_ready_o"), g("rd_addr_i")
        self.rrv, self.rd = g("rd_rvalid_o"), g("rd_rdata_o")
        self.wv, self.wr, self.wa = g("wr_valid_i"), g("wr_ready_o"), g("wr_addr_i")
        self.wm, self.wd = g("wr_mask_i"), g("wr_data_i")
        self.rv.value = 0
        self.wv.value = 0
        self.noise_on = False

    def pack(self, bits32):
        word = 0
        for l, b in enumerate(bits32):
            word |= int(b) << (32 * l)
        return word

    async def write_words(self, base, words, masks=None):
        full = (1 << self.lanes) - 1
        i = 0
        while i < len(words):
            self.wv.value, self.wa.value = 1, base + i
            self.wm.value = full if masks is None else masks[i]
            self.wd.value = words[i]
            await ReadOnly()
            acc = self.wr.value == 1
            await RisingEdge(self.dut.clk_i)
            if acc:
                i += 1
        self.wv.value = 0

    async def write_elems(self, base_elem, values):
        """fp32 values (or uint32 bits) at an element address (word-aligned base)."""
        bits = np.asarray(values).view(np.uint32) if np.asarray(values).dtype != np.uint32 else values
        V = self.lanes
        assert base_elem % V == 0
        n = -(-len(bits) // V)
        words, masks = [], []
        for w in range(n):
            chunk = bits[w * V:(w + 1) * V]
            words.append(self.pack(chunk))
            masks.append((1 << len(chunk)) - 1)
        await self.write_words(base_elem // V, words, masks)

    async def read_words(self, base, n):
        out, i = [], 0
        while len(out) < n:
            self.rv.value = int(i < n)
            self.ra.value = base + min(i, n - 1)
            await ReadOnly()
            if self.rrv.value == 1:
                out.append(int(self.rd.value))
            acc = i < n and self.rr.value == 1
            await RisingEdge(self.dut.clk_i)
            if acc:
                i += 1
            if i >= n:
                self.rv.value = 0
        return out

    async def read_elems(self, base_elem, n):
        V = self.lanes
        words = await self.read_words(base_elem // V, -(-n // V))
        out = np.zeros(len(words) * V, dtype=np.uint32)
        for w, word in enumerate(words):
            for l in range(V):
                out[w * V + l] = (word >> (32 * l)) & 0xFFFFFFFF
        return out[:n]

    async def noise(self, rng, rd_words, wr_words, p=0.5):
        """Until noise_on is cleared: random reads anywhere in rd_words (a range of word
        addresses) and random full-word writes inside wr_words (memory nobody else uses)."""
        self.noise_on = True
        rq = wq = None
        while self.noise_on or rq is not None or wq is not None:
            if rq is None and self.noise_on and rng.random() < p:
                rq = int(rng.integers(rd_words[0], rd_words[1]))
            if wq is None and self.noise_on and wr_words and rng.random() < p:
                wq = int(rng.integers(wr_words[0], wr_words[1]))
            self.rv.value, self.wv.value = int(rq is not None), int(wq is not None)
            if rq is not None:
                self.ra.value = rq
            if wq is not None:
                self.wa.value, self.wm.value = wq, (1 << self.lanes) - 1
                self.wd.value = int.from_bytes(rng.bytes(4 * self.lanes), "little")
            await ReadOnly()
            racc, wacc = self.rr.value == 1, self.wr.value == 1
            await RisingEdge(self.dut.clk_i)
            if rq is not None and racc:
                rq = None
            if wq is not None and wacc:
                wq = None
        self.rv.value = self.wv.value = 0
        for _ in range(4):                       # let the last responses drain
            await RisingEdge(self.dut.clk_i)
