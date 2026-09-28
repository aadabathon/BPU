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
| C7 | Hardening | **partly** | lint, Yosys synthesis, gate-level sim, formal (QMV, FVU) done; timing and PnR not started |
| C8 | FVU throughput: pipelined reduction merge, multi-port SPM | **done** | reduce unit test, FVU + decode bit-exact; 54.5 → 80.9 tok/s projected (2B, 128 ctx) |

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
   81 / 40 / 15 tok/s on Qwen3.5-2B at 128 / 2K / 8K context. The pipelined merge
   and the 3-port SPM are done. Remaining:
   * attention on QMV with an INT8 KV cache: 124 / 84 / 78 tok/s. The QMV W8 mode
     already computes it; it needs an ml-models accuracy check and a KV layout
     from rtl-memory;
   * VVECMAT accumulator forwarding: about +10% at short context;
   * a fused delta-rule op.
2. **Timing closure.**
   * Out-of-context Vivado runs at 250 MHz for `bpu_qmv_slice` (fpga config) and `bpu_fvu_lane`.
   * OpenLane 2 on sky130 for the asic config.
   * The fp32 multiplier's normalize stage and the adder's post-add shift are the known long paths.
3. **Area for tapeout.** SFU sharing (`SfuLanes`) is done: −20% compute-top area
   at the asic configuration. Remaining levers:
   * fusing int→fp32 with the QMV product multiplier (bit-identical);
   * sizing the SPM for the chosen tapeout model.

   Current estimate: 0.44 mm² sky130 logic plus SRAM macros (performance.md).
4. **FVU formal:** BMC and covers pass. The unbounded PDR proof converges on all
   but one property in 40 minutes (see verification.md).
5. **Streaming operands for full-size tensors.** A real 2B model's KV cache and
   128×128 states per head exceed an on-chip SPM, so the FVU needs a streaming
   operand port or tiling support in the compiler.

## Tapeout track

* **Candidate:** `bpu_compute_top` at the asic configuration: 1 QMV slice × 16
  lanes and a 2-lane FVU with one shared SFU. That is about 0.44 mm² of sky130
  logic (pre-layout, typical corner) plus SRAMs, running the tiny model bit-exact
  against the FPGA build and the reference.
* **Early learning run:** `bpu_qmv_dot` + `bpu_fp32_*` at a tiny configuration on a
  Tiny Tapeout shuttle, to learn the flow first.

## Needed from other teams

| From | What | Why |
|---|---|---|
| architecture + ml-models | confirm W4A8 + bf16 scales; KV precision (INT8 enables the biggest speedup) | numerics freeze |
| ml-models | real-checkpoint quality of these numerics (perplexity) | accuracy sign-off |
| rtl-memory | agreement on layout L0 and the `wreq_*` protocol | weight streaming |
| architecture / rtl-control | who produces the operation stream (compiler list, RISC-V, list-walker) | integration |
