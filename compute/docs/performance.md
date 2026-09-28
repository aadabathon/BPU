# Performance: measured, modelled, projected

## Measured (RTL simulation)

`tb/cocotb_compute_top.py` runs a full decode step of the tiny same-structure
Qwen3.5 (hidden 128, 3 DeltaNet + 1 attention layer, vocab 512; 443 operations,
32 of them QMV) on `bpu_compute_top`, bit-exact. Cycles for one step's operation
stream (position 1):

| Configuration | QMV | FVU | Cycles / token |
|---|---|---|---|
| fpga | 32 slices × 64 lanes | 16 lanes, 4 shared SFUs, 3 SPM read ports | 29,867 |
| asic | 1 × 16 | 2 lanes, 1 shared SFU, 1 read port | 179,839 |
| tiny | 3 × 4 | 4 lanes, 2 shared SFUs, 2 read ports, combinational | 117,315 |

Before the pipelined reduction merge and the multi-port SPM (serial merge, one
read port), the same stream took 34,038 / 182,578 / 139,787 cycles. Sharing SFUs
cost 1–4% of the cycles when it was introduced (see Area and timing).

## Cycle model

`bpuref/perf.py` counts cycles from the RTL's actual issue rules:
- the SPM reads each item needs (`SpmReadPorts` per cycle);
- the reduction merge rate (about 1.08 cycles per word node, measured by
  `tb/cocotb_fvu_reduce.py`) and the reduction credit loop
  (`RedFifoDepth` credits per lane-pipeline + lane-tree + 4 cycles);
- VVECMAT row spacing;
- the QMV gearbox, beats per slice, and pipeline tails.

| Configuration | Measured | Modelled | Error |
|---|---|---|---|
| fpga | 29,867 | 29,136 | −2.4% |
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
| **Current fpga RTL (3 read ports, pipelined merge)** | **80.9** | **40.3** | **15.4** |
| + attention K·q and pᵀV on the QMV engine (INT8 KV) | 123.6 | 84.4 | 78.0 |

Where the current fpga RTL spends a 2K-context token (6.2M cycles):

| Work | Cycles |
|---|---|
| VVECMAT (attention pᵀV, DeltaNet state reads) | 2.46M |
| Reductions (attention scores: one RDOT row per position per head) | 1.76M |
| Element-wise (DeltaNet state decay/update, norms, SiLU…) | 1.02M |
| QMV | 0.97M |

**The QMV engine is not the bottleneck.** At 0.97M cycles/token it alone would
allow ~260 tok/s, close to the 272 tok/s MAC bound (2048 MACs/cycle vs 1.88G MACs/token).
The vector unit is the bottleneck. Remaining steps, in order of payoff:

1. **Move attention onto QMV.** Store the KV cache as INT8 with per-group scales
   (a numerics decision: it needs an accuracy check by ml-models). Then K·q is a
   QMV op over the cache and pᵀV is QMV with Vᵀ. This is what makes long context
   viable.
2. **Forward the VVECMAT accumulator** inside the lane pipeline instead of through
   the SPM. Rows are spaced by `Ltot + 6` cycles today so each row reads the
   accumulator after the previous row wrote it. DeltaNet state rows are 8 words
   at 16 lanes, so they wait on that spacing.
3. **A fused delta-rule op** that reads each head's 128×128 state once instead of
   four times.

None of these changes a result bit, except attention-on-QMV, which changes the
KV format. The pipelined merge and the read ports (done) did not change any bit
either: the RTL stays bit-exact with `bpuref` at every configuration.

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
| `bpu_fvu` (2 lanes, 1 shared SFU) | 0.305 mm² | 49,615 | ~30–35 ns |
| `bpu_compute_top` (asic, 1 shared SFU) | **0.437 mm²** | 70,906 | ~24–28 ns |
| (before the pipelined merge's narrower FIFO) | 0.458 mm² | 72,202 | |
| (same with one SFU per lane, before) | 0.575 mm² | 94,376 | |

So the compute logic of the tapeout candidate is about **0.44 mm²** plus SRAM
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
| `bpu_fvu` (2 lanes, per-lane SFUs) | 63.3K | 3,436 |
| `bpu_compute_top` (per-lane SFUs) | 79.4K | 4,090 |

**SFU sharing** (`SfuLanes`): the SFU is the largest per-lane cost. Sharing one
SFU across the asic configuration's two FVU lanes saves 20% of the compute top's
area for about 1.5% more cycles on the tiny model. The fpga configuration shares
4 SFUs among 16 lanes. A tapeout would also size the SPM for its model; the tiny
model needs about 42K fp32 elements.

**SPM read ports** (`SpmReadPorts`): each extra port is another copy of the SPM
storage (writes go to every copy). On F2 that is block RAM, which is plentiful,
and it is worth +34% tokens/s at 128 context and +64% at 2K. The asic
configuration keeps one port; there the SRAM macro area would dominate.
