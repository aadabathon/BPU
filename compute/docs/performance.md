# Performance: measured, modelled, projected

## Measured (RTL simulation)

`tb/cocotb_compute_top.py` runs a full decode step of the tiny same-structure
Qwen3.5 (hidden 128, 3 DeltaNet + 1 attention layer, vocab 512; 443 operations,
32 of them QMV) on `bpu_compute_top`, bit-exact. Cycles for one step's operation
stream (position 1):

| Configuration | QMV | FVU | Cycles / token |
|---|---|---|---|
| fpga | 32 slices × 64 lanes | 16 lanes | 32,853 |
| asic | 1 × 16 | 2 lanes | 179,827 |
| tiny | 3 × 4 | 4 lanes, combinational | 138,384 |

## Cycle model

`bpuref/perf.py` counts cycles from the RTL's actual issue rules:
- the SPM reads each item needs (one per cycle);
- the reduction merge rate (3 + AddLatency cycles per word node);
- VVECMAT row spacing;
- the QMV gearbox, beats per slice, and pipeline tails.

| Configuration | Measured | Modelled | Error |
|---|---|---|---|
| fpga | 32,853 | 32,132 | −2.2% |
| asic | 179,827 | 180,203 | +0.2% |
| tiny | 138,384 | 137,433 | −0.7% |

## Projection: Qwen3.5-2B at the fpga configuration, 250 MHz

`python -m bpuref.perf` compiles the real 2B decode step (shapes only) and applies
the calibrated model. It assumes weights arrive at one beat per cycle per slice
(32 HBM pseudo-channels keeping up). Each row adds one improvement on top of the
previous ones, in tokens/s:

| | 128 ctx | 2K ctx | 8K ctx |
|---|---|---|---|
| **Current RTL** | **56** | **15** | **4.6** |
| + 3-port SPM (all operand reads in one cycle) | 71 | 18 | 5.3 |
| + pipelined reduction merge (1 node/cycle) | 85 | 42 | 16 |
| + attention K·q and pᵀV on the QMV engine (INT8 KV) | 131 | 89 | 83 |

Where the current RTL spends a 2K-context token (16.4M cycles):

| Work | Cycles |
|---|---|
| Reductions (attention scores: one RDOT row per position per head) | 9.6M |
| VVECMAT (attention pᵀV, DeltaNet state reads) | 4.5M |
| Element-wise (DeltaNet state decay/update, norms, SiLU…) | 1.4M |
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

## Area (Yosys, generic cells, SRAMs excluded)

`scripts/synth_yosys.sh` at the asic configuration:

| Block | Cells | Flops |
|---|---|---|
| `bpu_qmv_slice` (16 lanes) | 14.7K | 887 |
| `bpu_qmv_array` (1 slice) | 16.3K | 1,145 |
| `bpu_sfu` | 17.9K | 197 |
| `bpu_fvu` (2 lanes) | 63.3K | 3,436 |
| `bpu_compute_top` | 79.4K | 4,090 |

The SFU is the largest per-lane cost. Sharing one SFU across FVU lanes (a planned
`SfuLanes` parameter) is the main area lever for a tapeout configuration. The
compute top at the asic configuration is roughly 0.5–1 mm² of logic in sky130 (a
generic-cell estimate), plus SPM/activation SRAM. A tapeout would size the SPM for
the tiny model (≈ 42K elements ≈ 168 KB fp32 today) or smaller.
