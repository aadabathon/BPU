# Performance: measured, modelled, projected

## Measured (RTL simulation)

`tb/cocotb_compute_top.py` runs a full decode step of the tiny same-structure
Qwen3.5 (hidden 128, 3 DeltaNet + 1 attention layer, vocab 512; 443 operations,
32 of them QMV) on `bpu_compute_top`, bit-exact. Cycles for one step's operation
stream (position 1):

| Configuration | QMV | FVU | Cycles / token |
|---|---|---|---|
| fpga | 32 slices × 64 lanes | 16 lanes, 4 shared SFUs, 3 SPM read ports | 26,077 |
| asic | 1 × 16 | 2 lanes, 1 shared SFU, 1 read port | 179,839 |
| tiny | 3 × 4 | 4 lanes, 2 shared SFUs, 2 read ports, combinational | 117,315 |

Before the pipelined reduction merge, the multi-port SPM and VVECMAT accumulator
forwarding, the same stream took 34,038 / 182,578 / 139,787 cycles. Sharing SFUs
cost 1–4% of the cycles when it was introduced (see Area and timing).

## Cycle model

`bpuref/perf.py` counts cycles from the RTL's actual issue rules:
- the SPM reads each item needs (`SpmReadPorts` per cycle);
- the reduction merge rate (about 1.08 cycles per word node, measured by
  `tb/cocotb_fvu_reduce.py`) and the reduction credit loop
  (`RedFifoDepth` credits per lane-pipeline + lane-tree + 4 cycles);
- VVECMAT row spacing (`Latency + 1` cycles, with accumulator forwarding);
- the QMV gearbox, beats per slice, and pipeline tails.

| Configuration | Measured | Modelled | Error |
|---|---|---|---|
| fpga | 26,077 | 25,346 | −2.8% |
| asic | 179,839 | 180,167 | +0.2% |
| tiny | 117,315 | 117,102 | −0.2% |

## Projection: Qwen3.5-2B at the fpga configuration, 250 MHz

`python -m bpuref.perf` compiles the real 2B decode step (shapes only) and applies
the calibrated model. It assumes weights arrive at one beat per cycle per slice
(32 HBM pseudo-channels keeping up). Tokens/s:

| | 128 ctx | 2K ctx | 8K ctx |
|---|---|---|---|
| First RTL (1 read port, serial reduction merge) | 54.5 | 15.1 | 4.6 |
| 1 read port, pipelined merge (the asic shape) | 60.5 | 24.6 | 8.5 |
| **Current fpga RTL (3 read ports, pipelined merge, VVECMAT forwarding)** | **89.4** | **42.3** | **15.7** |
| + attention K·q and pᵀV on the QMV engine (INT8 KV) | 123.6 | 93.6 | 85.9 |

**FVU width** is the next lever and needs no new RTL. `VLanes` goes up to 64, and
the `wide` FVU configuration (64 lanes, 16 shared SFUs, 3 read ports) is verified
in simulation like the others. Projected tokens/s:

| FVU lanes | 128 ctx | 2K ctx | 8K ctx | + attention on QMV (128 / 2K / 8K) |
|---|---|---|---|---|
| 16 (fpga) | 89.4 | 42.3 | 15.7 | 123.6 / 93.6 / 85.9 |
| 32 | 118.1 | 68.0 | 28.8 | 164.3 / 121.0 / 112.4 |
| 64 (wide) | 136.5 | 86.6 | 39.9 | 196.8 / 138.8 / 130.4 |

Whether 64 lanes fit F2 at 250 MHz alongside 32 QMV slices is a Vivado question.
It is roughly 64 fp32 multipliers and adders, 16 SFUs, and 3 SPM copies 2048 bits wide.

Where the current fpga RTL spends a token:

| Work | 128 ctx | 2K ctx |
|---|---|---|
| Element-wise (DeltaNet state decay/update, norms, SiLU…) | 0.98M | 1.02M |
| QMV | 0.97M | 0.97M |
| VVECMAT (DeltaNet state reads, attention pᵀV) | 0.70M | 2.17M |
| Reductions (attention scores: one RDOT row per position per head) | 0.15M | 1.76M |
| **Total** | **2.80M** | **5.92M** |

**The QMV engine is not the bottleneck.** At 0.97M cycles/token it alone would
allow ~260 tok/s, close to the 272 tok/s MAC bound (2048 MACs/cycle vs 1.88G MACs/token).
The vector unit is the bottleneck. Remaining steps, in order of payoff:

1. **Move attention onto QMV.** Store the KV cache as INT8 with per-group scales
   (a numerics decision: it needs an accuracy check by ml-models). Then K·q is a
   QMV op over the cache and pᵀV is QMV with Vᵀ; the QMV W8 mode already computes
   both. This is what makes long context viable.
2. **Widen the FVU** (above).
3. **A fused delta-rule op**: two passes over each head's 128×128 state instead of
   four (decay + Sᵀk, then rank-1 update + Sᵀq). DeltaNet state traffic is 43% of
   a 128-context token. It needs a small local VVECMAT accumulator, because each
   fused item would otherwise write both the state and the accumulator through the
   single SPM write port.

A fused delta-rule op can keep every result bit (the same operations in the same
order). Attention-on-QMV changes the KV format.
The pipelined merge, the read ports and VVECMAT forwarding (all done) changed no
bits either: the RTL stays bit-exact with `bpuref` at every configuration.

**Reduction FIFO sizing.** Every reduction word holds a credit from issue until
the merge pops it, a loop of about `Ltot + log2(VLanes)·AddLatency + 4` cycles
(22 at fpga). `RedFifoDepth` must cover it for full rate. The fpga configuration
uses 24 entries. Each entry is one 32-bit value plus control, because the word max
travels as its order key and the key is inverted at the end.

## Area and timing

**sky130** (`scripts/synth_sky130.sh`): the asic configuration mapped onto
`sky130_fd_sc_hd` cells, typical corner, SRAMs excluded. These are pre-layout
numbers: no wires, placement or clock tree, and ABC's delay is area-oriented.
Useful for sizing a tapeout, not for sign-off.

| Block | Area | Cells | ABC critical path |
|---|---|---|---|
| `bpu_qmv_slice` (16 lanes) | 0.111 mm² | 18,951 | 20.7 ns |
| `bpu_sfu` | 0.134 mm² | 25,236 | 19.7 ns |
| `bpu_fvu` (2 lanes, 1 shared SFU) | 0.319 mm² | 51,734 | ~27–35 ns |
| `bpu_compute_top` (asic, 1 shared SFU) | **0.428 mm²** | 68,067 | ~24–31 ns |
| (before the pipelined merge's narrower FIFO) | 0.458 mm² | 72,202 | |
| (same with one SFU per lane, before) | 0.575 mm² | 94,376 | |

ABC's results move by a few percent from run to run and with small RTL changes.
Read these as ±5%.

So the compute logic of the tapeout candidate is about **0.43 mm²** plus SRAM
macros for the SPM and the QMV activation buffer, which fits comfortably in a
Caravel-class ~10 mm² user area. The asic pipelining settings reach roughly
30–40 MHz before wires. Deeper pipeline settings (the same parameters the fpga
configuration uses) trade flops for clock.

**Generic cells** (`scripts/synth_yosys.sh`, technology-independent), asic configuration:

| Block | Cells | Flops |
|---|---|---|
| `bpu_qmv_slice` | 14.7K | 887 |
| `bpu_qmv_array` (1 slice) | 16.3K | 1,145 |
| `bpu_sfu` | 17.9K | 197 |
| `bpu_fvu` (2 lanes, 1 shared SFU) | 44.5K | 2,761 |
| `bpu_compute_top` (1 shared SFU) | 61.8K | 3,991 |

**SFU sharing** (`SfuLanes`): the SFU is the largest per-lane cost. Sharing one
SFU across the asic configuration's two FVU lanes saves 20% of the compute top's
area for about 1.5% more cycles on the tiny model. The fpga configuration shares
4 SFUs among 16 lanes. A tapeout would also size the SPM for its model; the tiny
model needs about 42K fp32 elements.

**SPM read ports** (`SpmReadPorts`): each extra port is another copy of the SPM
storage (writes go to every copy). On F2 that is block RAM, which is plentiful,
and it is worth +34% tokens/s at 128 context and +64% at 2K. The asic
configuration keeps one port; there the SRAM macro area would dominate.
