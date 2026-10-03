"""bpu_fvu_reduce on its own: random rows of every tree depth, sums and maxes mixed
back to back, credit-limited input with random gaps, bit-exact against the
canonical pairwise tree and the order-key max.

The unit's contract is wider than what bpu_fvu uses (one op at a time, rows at
least 64 elements): here consecutive rows differ in length (any power of two
words, including one word) and in kind, so every pairing and root case of the
pipelined merge is exercised. Configuration via BPU_RED_LANES / BPU_RED_FIFO."""

import os

import cocotb
import numpy as np
from cocotb.triggers import ReadOnly, RisingEdge

import busmodels as bm
from bpuref.fp import F32_QNAN, f32_bits, f32_equal
from bpuref.fvu import order_max

LANES = int(os.environ.get("BPU_RED_LANES", "4"))
FIFO = int(os.environ.get("BPU_RED_FIFO", "8"))
ADD_LAT = bin(int(os.environ.get("BPU_RED_ADDPIPE", "7"))).count("1")
TREE_LAT = (LANES.bit_length() - 1) * ADD_LAT       # lane adder tree ahead of the FIFO


def pair_tree(x: np.ndarray) -> np.float32:
    """Adjacent pairs level by level over a power-of-two vector (fp32)."""
    x = np.asarray(x, np.float32)
    with np.errstate(all="ignore"):
        while x.size > 1:
            x = x[0::2] + x[1::2]
    return x[0]


def random_values(rng, n):
    x = (rng.normal(0, 1, n) * np.exp2(rng.integers(-40, 40, n))).astype(np.float32)
    x[rng.random(n) < 0.03] = 0.0
    x[rng.random(n) < 0.03] = -0.0
    x[rng.random(n) < 0.02] = np.float32(3e-41)              # subnormal
    x[rng.random(n) < 0.004] = np.inf
    x[rng.random(n) < 0.004] = -np.inf
    x[rng.random(n) < 0.004] = np.nan
    return x


def random_rows(rng, n_rows, max_log2_words):
    """[(is_max, words[W][LANES], masks[W][LANES], expected)] with W a power of two."""
    rows = []
    for _ in range(n_rows):
        w = 1 << int(rng.integers(0, max_log2_words + 1))
        vals = random_values(rng, w * LANES).reshape(w, LANES)
        mask = rng.random((w, LANES)) < 0.9
        if rng.random() < 0.2:
            mask[w // 2:] = False                            # padding words
        is_max = bool(rng.random() < 0.3)
        if is_max:
            sel = vals[mask]
            exp = order_max(sel[None, :])[0] if sel.size else np.uint32(F32_QNAN).view(np.float32)
        else:
            exp = pair_tree(np.where(mask, vals, np.float32(0)).reshape(-1))
        rows.append((is_max, vals, mask, np.float32(exp)))
    return rows


def pack(vals_row, mask_row):
    bits = f32_bits(vals_row)
    word = 0
    for lane in range(LANES):
        word |= int(bits[lane]) << (32 * lane)
    m = sum(int(b) << lane for lane, b in enumerate(mask_row))
    return word, m


async def drive(dut, rows, rng, p_idle):
    """Send every word, never more in flight than FIFO credits allow."""
    credits, cycles = FIFO, 0
    items = []
    for r, (is_max, vals, mask, _) in enumerate(rows):
        for w in range(vals.shape[0]):
            items.append((is_max, w == 0, w == vals.shape[0] - 1, r, *pack(vals[w], mask[w])))
    i = 0
    while i < len(items):
        send = credits > 0 and rng.random() >= p_idle
        if send:
            is_max, first, last, dest, word, m = items[i]
            dut.in_valid_i.value = 1
            dut.in_max_i.value = int(is_max)
            dut.in_first_i.value = int(first)
            dut.in_last_i.value = int(last)
            dut.in_dest_i.value = dest
            dut.in_mask_i.value = m
            dut.in_vals_i.value = word
            i += 1
        else:
            dut.in_valid_i.value = 0
        await ReadOnly()
        credits += int(dut.pop_o.value) - int(send)
        await RisingEdge(dut.clk_i)
        cycles += 1
    dut.in_valid_i.value = 0
    return cycles


async def collect(dut, results):
    while True:
        await ReadOnly()
        if dut.wr_valid_o.value == 1:
            addr = int(dut.wr_addr_o.value)
            assert addr not in results, f"row {addr} written twice"
            results[addr] = int(dut.wr_data_o.value)
        await RisingEdge(dut.clk_i)


async def run_rows(dut, rows, rng, p_idle):
    results = {}
    mon = cocotb.start_soon(collect(dut, results))
    cycles = await drive(dut, rows, rng, p_idle)
    for _ in range(10000):
        await ReadOnly()
        if dut.busy_o.value == 0 and len(results) == len(rows):
            break
        await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    mon.cancel()
    assert len(results) == len(rows), f"{len(rows) - len(results)} rows never finished"
    bad = [r for r, row in enumerate(rows) if not f32_equal(results[r], row[3])]
    for r in bad[:8]:
        dut._log.error(f"row {r} ({'max' if rows[r][0] else 'sum'}, {rows[r][1].shape[0]} words): "
                       f"rtl {results[r]:08x} ref {f32_bits(rows[r][3]):08x}")
    assert not bad, f"{len(bad)} / {len(rows)} rows differ"
    return cycles


INPUTS = lambda dut: (dut.in_valid_i, dut.in_max_i, dut.in_first_i, dut.in_last_i)


@cocotb.test()
async def random_rows_mixed(dut):
    rng = np.random.default_rng(int(os.environ.get("BPU_SEED", "1")))
    await bm.reset(dut, INPUTS(dut))
    for p_idle in (0.0, 0.3, 0.7):
        rows = random_rows(rng, 150, 7)
        await run_rows(dut, rows, rng, p_idle)
    dut._log.info(f"[lanes {LANES}, fifo {FIFO}] 450 mixed rows bit-exact")


@cocotb.test()
async def merge_throughput(dut):
    """Back-to-back sum rows: the merge takes about one word node per cycle. The
    driver returns credits one cycle after a pop, so full rate also needs a FIFO
    that covers the lane tree's latency."""
    rng = np.random.default_rng(3)
    await bm.reset(dut, INPUTS(dut))
    for log2w in (1, 3, 6):
        rows = []
        for _ in range(max(8, 512 >> log2w)):
            w = 1 << log2w
            vals = random_values(rng, w * LANES).reshape(w, LANES)
            mask = np.ones((w, LANES), bool)
            rows.append((False, vals, mask, pair_tree(vals.reshape(-1))))
        words = sum(r[1].shape[0] for r in rows)
        cycles = await run_rows(dut, rows, rng, 0.0)
        dut._log.info(f"[lanes {LANES}, fifo {FIFO}] rows of {1 << log2w} words: "
                      f"{cycles / words:.3f} cycles per word node")
        if FIFO >= TREE_LAT + 3:
            assert cycles <= 1.15 * words + 16, f"merge too slow: {cycles} cycles for {words} words"
