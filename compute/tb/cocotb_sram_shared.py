"""bpu_sram_shared: random traffic on every read and write port against a reference
memory that applies the visibility rule cycle by cycle (a read sees every write
accepted in an earlier cycle, never one accepted in the same cycle). Checks
response latency and order, out-of-range handling, the round-robin wait bound,
and that requests to different banks never wait for each other.
Configuration via BPU_SRAM_* environment variables (see test_rtl.py)."""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

from bpuref.sram import bank_of

E = lambda k, d: int(os.environ.get("BPU_SRAM_" + k, d))
NB, BW, LANES, NRD, NWR = E("NBANKS", 4), E("BANKWORDS", 64), E("LANES", 2), E("NRD", 3), E("NWR", 2)
OUTREG, HASH, AW = E("OUTREG", 0), E("HASH", 1), 16
CAP = NB * BW
RDLAT = 1 + OUTREG
W = LANES * 32
MASK_ALL = (1 << LANES) - 1
MEM = [0] * CAP          # the reference memory lives as long as the simulation (all tests)


def rand_word(rng):
    return int.from_bytes(rng.bytes(W // 8), "little")


def field(sig, i, width):
    return (int(sig.value) >> (i * width)) & ((1 << width) - 1)


def set_field(cur, i, width, val):
    m = ((1 << width) - 1) << (i * width)
    return (cur & ~m) | ((val << (i * width)) & m)


class Bench:
    def __init__(self, dut, rng):
        self.dut, self.rng = dut, rng
        self.mem = MEM
        self.cycle = 0
        self.rd_q = [[] for _ in range(NRD)]          # expected (due_cycle, data, oob)
        self.wait = {("r", p): 0 for p in range(NRD)} | {("w", p): 0 for p in range(NWR)}
        self.max_wait = dict(self.wait)
        self.errors = 0
        self.reads = self.writes = 0
        # Drive state: per port current request (None = idle)
        self.rd_req = [None] * NRD
        self.wr_req = [None] * NWR

    def drive(self):
        d = self.dut
        v = a = 0
        for p, r in enumerate(self.rd_req):
            if r is not None:
                v |= 1 << p
                a = set_field(a, p, AW, r)
        d.rd_valid_i.value, d.rd_addr_i.value = v, a
        v = a = m = dat = 0
        for p, w in enumerate(self.wr_req):
            if w is not None:
                v |= 1 << p
                a = set_field(a, p, AW, w[0])
                m = set_field(m, p, LANES, w[1])
                dat = set_field(dat, p, W, w[2])
        d.wr_valid_i.value, d.wr_addr_i.value = v, a
        d.wr_mask_i.value, d.wr_data_i.value = m, dat

    def sample_and_check(self):
        """In ReadOnly: account the handshakes of this cycle against the model."""
        d = self.dut
        rready, wready = int(d.rd_ready_o.value), int(d.wr_ready_o.value)
        rvalid = int(d.rd_rvalid_o.value)
        rd_oob, wr_oob = int(d.rd_oob_o.value), int(d.wr_oob_o.value)
        # Responses due now
        for p in range(NRD):
            got_v = (rvalid >> p) & 1
            exp = self.rd_q[p][0] if self.rd_q[p] and self.rd_q[p][0][0] == self.cycle else None
            if exp is None:
                if got_v:
                    self.fail(f"read port {p}: unexpected response at cycle {self.cycle}")
                continue
            self.rd_q[p].pop(0)
            if not got_v:
                self.fail(f"read port {p}: missing response at cycle {self.cycle}")
            elif field(d.rd_rdata_o, p, W) != exp[1]:
                self.fail(f"read port {p}: data {field(d.rd_rdata_o, p, W):x} want {exp[1]:x}")
        # Reads accepted this cycle see the memory before this cycle's writes
        for p, r in enumerate(self.rd_req):
            if r is None:
                continue
            oob = r >= CAP
            if ((rd_oob >> p) & 1) != oob:
                self.fail(f"read port {p}: oob flag wrong for addr {r}")
            if (rready >> p) & 1:
                self.rd_q[p].append((self.cycle + RDLAT, 0 if oob else self.mem[r], oob))
                self.rd_req[p] = None
                self.reads += 1
                self.wait[("r", p)] = 0
            else:
                self.wait[("r", p)] += 1
        for p, w in enumerate(self.wr_req):
            if w is None:
                continue
            addr, mask, data = w
            oob = addr >= CAP
            if ((wr_oob >> p) & 1) != oob:
                self.fail(f"write port {p}: oob flag wrong for addr {addr}")
            if (wready >> p) & 1:
                if not oob:
                    old = self.mem[addr]
                    for l in range(LANES):
                        if (mask >> l) & 1:
                            old = set_field(old, l, 32, (data >> (32 * l)) & 0xFFFFFFFF)
                    self.mem[addr] = old
                self.wr_req[p] = None
                self.writes += 1
                self.wait[("w", p)] = 0
            else:
                self.wait[("w", p)] += 1
        for k, v in self.wait.items():
            self.max_wait[k] = max(self.max_wait[k], v)
            bound = (NRD if k[0] == "r" else NWR) - 1
            if v > bound:
                self.fail(f"{k}: waited {v} cycles (round-robin bound {bound})")

    def fail(self, msg):
        self.errors += 1
        if self.errors <= 10:
            self.dut._log.error(msg)


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    dut.rd_valid_i.value = 0
    dut.wr_valid_i.value = 0
    dut.rst_ni.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def run(dut, bench, cycles, new_rd, new_wr):
    for _ in range(cycles):
        for p in range(NRD):
            if bench.rd_req[p] is None:
                bench.rd_req[p] = new_rd(p)
        for p in range(NWR):
            if bench.wr_req[p] is None:
                bench.wr_req[p] = new_wr(p)
        bench.drive()
        await ReadOnly()
        bench.sample_and_check()
        await RisingEdge(dut.clk_i)
        bench.cycle += 1
    # drain responses
    bench.rd_req = [None] * NRD
    bench.wr_req = [None] * NWR
    for _ in range(RDLAT + 2):
        bench.drive()
        await ReadOnly()
        bench.sample_and_check()
        await RisingEdge(dut.clk_i)
        bench.cycle += 1


@cocotb.test()
async def random_traffic(dut):
    """Hot addresses (lots of read-after-write and bank conflicts) plus out-of-range."""
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    await setup(dut)
    b = Bench(dut, rng)
    hot = [int(x) for x in rng.choice(CAP, size=min(CAP, 3 * NB), replace=False)]

    def addr():
        r = rng.random()
        if r < 0.03:
            return int(rng.integers(CAP, 1 << AW))          # out of range
        if r < 0.6:
            return hot[int(rng.integers(len(hot)))]
        return int(rng.integers(CAP))

    new_rd = lambda p: addr() if rng.random() < 0.7 else None
    new_wr = lambda p: ((addr(), int(rng.integers(1, MASK_ALL + 1)), rand_word(rng))
                        if rng.random() < 0.6 else None)
    await run(dut, b, 4000, new_rd, new_wr)
    assert b.errors == 0, f"{b.errors} errors"
    dut._log.info(f"[{NB} banks, {NRD}R/{NWR}W, RdLat {RDLAT}, hash {HASH}] {b.reads} reads, "
                  f"{b.writes} writes checked; max waits {max(b.max_wait.values())}")


@cocotb.test()
async def distinct_banks_never_wait(dut):
    """Every port streaming through its own bank: every request is accepted at once."""
    if NB < max(NRD, NWR):
        return
    rng = np.random.default_rng(5)
    await setup(dut)
    b = Bench(dut, rng)
    by_bank = {k: [a for a in range(CAP) if bank_of(a, NB, BW, bool(HASH)) == k] for k in range(NB)}
    new_rd = lambda p: int(rng.choice(by_bank[p]))
    new_wr = lambda p: (int(rng.choice(by_bank[NB - 1 - p])), MASK_ALL, rand_word(rng))
    await run(dut, b, 500, new_rd, new_wr)
    assert b.errors == 0
    assert max(b.max_wait.values()) == 0, f"distinct banks waited: {b.max_wait}"
    assert int(dut.perf_rd_wait_o.value) == 0 and int(dut.perf_wr_wait_o.value) == 0
