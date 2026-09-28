# Compute roadmap: Qwen3.5-2B decode

Scope: the RTL that does the arithmetic for every Qwen3.5-2B decode operation.
It is bit-exact with `bpuref`, runs at the F2 size, and scales down for tapeout.
Out of scope: DMA/HBM, the command source (ISA/RISC-V/host), and the SoC.
`bpu_compute_top` defines the seam: an operation stream in, weight requests and
streams from memory.

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| C0 | Contract + scaffolding: numerics and op specs, `bpuref`, toolchain, config sweeps, lint, synthesis | **done** | docs, `scripts/` |
| C1 | QMV slice: IEEE fp32 units, W4/W8 × A8 dot, row-interleaved accumulate, credits, flags, counters | **done** | slice tests, formal proofs, fp32 soak, gate-level sim |
| C2 | QMV array: row striping, in-order merge, argmax (LM head), logical row count | **done** | array tests at 32 / 1 / 3 slices |
| C3 | SFU: rcp, rsqrt, exp2, exp, log2, bit-exact tables, ≤ 1.44 ulp | **done** | 138M-vector soak, gate-level sim |
| C4 | FVU core: element-wise, conversions, quantize, canonical reductions, SPM | **done** | FVU tests at 16 / 2 / 4 lanes |
| C5 | FVU 2-D ops and Qwen blocks: VVECMAT, per-row scalars, VPERM (RoPE), conv, delta rule, attention | **done** | same, plus the compiled Qwen program |
| C6 | Compute top: operation stream, QMV ↔ SPM gearbox, weight requests; full Qwen decode | **done** | tiny-Qwen decode bit-exact, 3 tokens, 3 configs |
| C7 | Hardening | **partly** | lint, Yosys synthesis, gate-level sim of every engine, unbounded control proofs (QMV, FVU) done; timing and PnR not started |
| C8 | FVU throughput: pipelined reduction merge, multi-port SPM, VVECMAT forwarding | **done** | reduce unit test, FVU + decode bit-exact; 54.5 → 89.4 tok/s projected (2B, 128 ctx) |

## Qwen3.5-2B coverage

Every decode operation is implemented, runs in the compiled program, and is
checked bit-exact on the RTL:
- all ten projection types and the tied LM head with greedy argmax;
- embedding dequantize and activation quantize;
- RMSNorm (`1 + w`) and gated RMSNorm (`w`);
- causal conv + SiLU; q/k l2norm and scaling; the β and α gates (sigmoid, softplus with threshold);
- the gated delta rule;
- per-head Q/K RMSNorm, partial RoPE and KV append;
- GQA attention with softmax; the output gate; SiLU·up; residuals.

## Next, in priority order

1. **Performance** (see [performance.md](performance.md)). The fpga RTL projects to
   89 / 42 / 16 tok/s on Qwen3.5-2B at 128 / 2K / 8K context. The pipelined
   merge, the 3-port SPM and VVECMAT forwarding are done. Remaining:
   * attention on QMV with an INT8 KV cache: 124 / 94 / 86 tok/s. The QMV W8 mode
     already computes it; it needs an ml-models accuracy check and a KV layout
     from rtl-memory;
   * a wider FVU: 32 lanes → 118 / 68 / 29 tok/s, 64 lanes → 137 / 87 / 40 (a
     parameter; the 64-lane `wide` configuration is verified in simulation);
   * a fused delta-rule op (two passes over each head's state instead of four).
     It needs a local VVECMAT accumulator, because otherwise the single SPM write
     port limits it.
2. **Timing closure.**
   * Out-of-context Vivado runs at 250 MHz for `bpu_qmv_slice` (fpga config) and `bpu_fvu_lane`.
   * OpenLane 2 on sky130 for the asic config.
   * The fp32 multiplier's normalize stage and the adder's post-add shift are the known long paths.
3. **Area for tapeout.** SFU sharing (`SfuLanes`) is done: −20% compute-top area
   at the asic configuration. Remaining levers:
   * fusing int→fp32 with the QMV product multiplier (bit-identical);
   * sizing the SPM for the chosen tapeout model.

   Current estimate: 0.43 mm² sky130 logic plus SRAM macros (performance.md).
4. **Formal coverage of data.** The control logic of both engines is proven
   unbounded. Data-dependent correctness (VVECMAT forwarding, the merge's pairing
   order) rests on bit-exact simulation. A symbolic-data model at a tiny shape
   would close that gap.
5. **Capacity: where the 2B model's state lives.** The tiny model fits the SPM; the
   2B model does not. In fp32:
   * DeltaNet state: 18 layers × 16 heads × 128×128 = 18.9 MB;
   * KV cache: 24 KB per position = 3.1 MB at 128, 50 MB at 2K, 201 MB at 8K context.

   F2's VU47P has about 34 MB of UltraRAM plus 9 MB of block RAM. Two workable
   plans, both needing rtl-memory:
   * **(A) States on chip, KV in HBM.** A ~20 MB SPM (one copy) holds the DeltaNet
     states. The KV cache stays in HBM and attention runs on QMV (INT8 KV), which
     streams it exactly like weights. The QMV side needs no new RTL. What's
     missing: the numerics decision and a DMA path for the KV append.
   * **(B) Stream everything.** The FVU gets a streaming operand port: the
     sequencer stalls issue until a stream beat is ready, and the lanes never stall.
     Per token, DeltaNet state traffic is about 113 MB (four reads and two writes of
     18.9 MB), about 0.3 ms at HBM rates, so bandwidth is not the limit. This is
     the natural plan for a tapeout with no large SRAM.

   Read ports at that capacity: replication (`SpmReadPorts`, done) multiplies
   memory, so a 20 MB SPM would use one copy. Region banking would give ports
   without copies. However, rows of one op can then issue at different rates, so
   VVECMAT needs per-item accumulator hazard tracking instead of the current
   row spacing. Decide after (A) versus (B).

## Tapeout track

* **Candidate:** `bpu_compute_top` at the asic configuration: 1 QMV slice × 16
  lanes and a 2-lane FVU with one shared SFU. That is about 0.43 mm² of sky130
  logic (pre-layout, typical corner) plus SRAMs, running the tiny model bit-exact
  against the FPGA build and the reference.
* **Early learning run (ready to submit):** `compute/tapeout/tt` wraps the
  unchanged fp32 adder and multiplier for a Tiny Tapeout shuttle, behind a
  byte-wide host protocol. It is 0.052 mm² of sky130 cells, so a 4x2-tile slot.
  A pin-level test checks it bit-exact, and `assemble.sh` builds the submission
  tree. It teaches the flow and checks the arithmetic every BPU datapath uses,
  on real silicon.

## Needed from other teams

| From | What | Why |
|---|---|---|
| architecture + ml-models | confirm W4A8 + bf16 scales; KV precision (INT8 enables the biggest speedup) | numerics freeze |
| ml-models | real-checkpoint quality of these numerics (perplexity) | accuracy sign-off |
| rtl-memory | agreement on layout L0 and the `wreq_*` protocol | weight streaming |
| architecture / rtl-control | who produces the operation stream (compiler list, RISC-V, list-walker) | integration |
