# Performance: measured, modelled, projected

## Measured (RTL simulation)

`tb/cocotb_compute_top.py` runs a full decode step of the tiny same-structure
Qwen3.5 (hidden 128, 3 DeltaNet + 1 attention layer, vocab 512; 443 operations,
32 of them QMV) on `bpu_compute_top`, bit-exact. Cycles for one step's operation
stream (position 1):

| Configuration | QMV | FVU | Cycles / token |
|---|---|---|---|
| fpga | 32 slices × 64 lanes | 16 lanes, 4 shared SFUs | 34,038 |
| asic | 1 × 16 | 2 lanes, 1 shared SFU | 182,578 |
| tiny | 3 × 4 | 4 lanes, 2 shared SFUs, combinational | 139,787 |

With one SFU per lane the cycle counts were 32,853 / 179,827 / 138,384; sharing costs 1–4%.

## Cycle model

`bpuref/perf.py` counts cycles from the RTL's actual issue rules:
- the SPM reads each item needs (one per cycle);
- the reduction merge rate (3 + AddLatency cycles per word node);
- VVECMAT row spacing;
- the QMV gearbox, beats per slice, and pipeline tails.

| Configuration | Measured | Modelled | Error |
|---|---|---|---|
| fpga | 34,038 | 33,317 | −2.1% |
| asic | 182,578 | 182,954 | +0.2% |
| tiny | 139,787 | 138,836 | −0.7% |

## Projection: Qwen3.5-2B at the fpga configuration, 250 MHz

`python -m bpuref.perf` compiles the real 2B decode step (shapes only) and applies
the calibrated model. It assumes weights arrive at one beat per cycle per slice
(32 HBM pseudo-channels keeping up). Each row adds one improvement on top of the
previous ones, in tokens/s:

| | 128 ctx | 2K ctx | 8K ctx |
|---|---|---|---|
| **Current RTL** | **54.5** | **15.1** | **4.6** |
| + 3-port SPM (all operand reads in one cycle) | 68.7 | 17.8 | 5.3 |
| + pipelined reduction merge (1 node/cycle) | 81.3 | 41.1 | 15.9 |
| + attention K·q and pᵀV on the QMV engine (INT8 KV) | 123.8 | 84.5 | 78.2 |

Where the current RTL spends a 2K-context token (16.6M cycles):

| Work | Cycles |
|---|---|
| Reductions (attention scores: one RDOT row per position per head) | 9.6M |
| VVECMAT (attention pᵀV, DeltaNet state reads) | 4.5M |
| Element-wise (DeltaNet state decay/update, norms, SiLU…) | 1.5M |
| QMV | 0.97M |

**The QMV engine is not the bottleneck.** At 0.97M cycles/token it alone would
allow ~260 tok/s, close to the 272 tok/s MAC bound (2048 MACs/cycle vs 1.88G MACs/token).
The vector unit is the bottleneck. Recommended next steps, in order of payoff:

1. **Move attention onto QMV.** Store the KV cache as INT8 with per-group scales
   (a numerics decision: it needs an accuracy check by ml-models). Then K·q is a
   QMV op over the cache and pᵀV is QMV with Vᵀ. This is what makes long context
   viable.
2. **Pipeline the reduction merge.** Rows are independent, so interleave several
   rows' merge stacks around the adder the way QMV interleaves rows.
3. **Multi-port or banked SPM**, so 2- and 3-operand ops issue one item per cycle.
4. **A fused delta-rule op**, which reads each head's 128×128 state once instead
   of four times. After items 1–3 this is the remaining DeltaNet cost.

All four keep the numerics contract, and none changes a result bit, except
attention-on-QMV, which changes the KV format.

## Area and timing

**sky130** (`scripts/synth_sky130.sh`): the asic configuration mapped onto
`sky130_fd_sc_hd` cells, typical corner, SRAMs excluded. These are pre-layout
numbers: no wires, placement or clock tree, and ABC's delay is area-oriented.
Useful for sizing a tapeout, not for sign-off.

| Block | Area | Cells | ABC critical path |
|---|---|---|---|
| `bpu_qmv_slice` (16 lanes) | 0.111 mm² | 18,951 | 20.7 ns |
| `bpu_sfu` | 0.134 mm² | 25,236 | 19.7 ns |
| `bpu_fvu` (2 lanes, 1 shared SFU) | 0.327 mm² | 50,876 | ~30–35 ns |
| `bpu_compute_top` (asic, 1 shared SFU) | **0.458 mm²** | 72,202 | ~24–28 ns |
| (same with one SFU per lane) | 0.575 mm² | 94,376 | |

So the compute logic of the tapeout candidate is about **0.46 mm²** plus SRAM
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
