# Compute roadmap: Qwen3.5-2B decode

Scope: the RTL that does the arithmetic for every Qwen3.5-2B decode operation.
It has to be bit-exact with `bpuref`, run at the F2 size and scale down for tapeout.
Out of scope: DMA, HBM, descriptor queues, the RISC-V, the host
(rtl-memory / rtl-control / soc / fpga).

Every milestone ends on **evidence**: tests that pass at every named configuration
(`fpga`, `asic`, `tiny`), lint clean, and synthesis numbers.

## Milestones

| # | Milestone | Qwen3.5 ops unlocked | Exit criteria | Status |
|---|---|---|---|---|
| C0 | **Contract + scaffolding**: numerics spec, op spec, `bpuref`, toolchain, config-sweep runner, lint, Yosys check | – | docs v0; reference self-tests; lint clean at all configs | **done** |
| C1 | **QMV slice** (the first MAC): IEEE fp32 mul/add, int2fp, W4/W8 dot, row-interleaved fp32 accumulation, valid/ready + credits | every projection (one slice) | fp32 units bit-exact vs numpy (~35K vectors each, 3 pipeline configs); slice bit-exact at K = 2048/6144, W4+W8; 1 beat/cycle measured; bugs deliberately injected are caught | **done (simulation)** |
| C2 | **QMV array + epilogues**: NSlice row striping, argmax epilogue, x double-buffering; freeze layout L0 with rtl-memory | full projection set incl. LM head + greedy sample | 32-slice array bit-exact; head argmax over 248,320 rows matches reference; throughput = NSlice beats/cycle | next |
| C3 | **SFU**: exp2, log2, rcp, rsqrt as tables + polynomial/Newton; tables/coefficients defined in `bpuref` | prerequisite for all nonlinear ops | bit-exact vs `bpuref`; per-function max-ulp error vs float64 published; ml-models signs off on sigmoid/SiLU/softplus/exp accuracy on real activation ranges | |
| C4 | **FVU core**: element-wise, conversions, `quant`/`dequant`, canonical reductions, scratchpad, `VLanes` parameter | RMSNorm, residual, SiLU·up, activation quantize, L2 norm, embedding dequant | each op bit-exact at every VLanes; canonical reduction tree proven lane-count independent | |
| C5 | **FVU 2-D + Qwen blocks**: `matvec`/`vecmat`/`rank1`, `rope`, `conv_step`; operand streaming for big tensors | DeltaNet step, causal conv, gates, gated RMSNorm, decode attention (GQA, online softmax), output gate | one DeltaNet layer and one attention layer of the tiny-Qwen config bit-exact vs `bpuref` layer models | |
| C6 | **compute_dispatch + counters**: op decode, QMV↔FVU dependency tracking, completion/fault, per-engine busy/stall/beat counters | whole decode step | full tiny-Qwen decode token in RTL simulation matches `bpuref`; a real Qwen3.5-2B layer (vectors from ml-models) matches | |
| C7 | **Hardening for silicon + FPGA**: OpenLane run at the asic config, VU47P out-of-context timing at 250 MHz, SymbiYosys stream-protocol proofs, gate-level simulation | – | area/timing report per config; formal proofs pass; the tapeout configuration is chosen | |

C3 does not depend on C2, so they can run in parallel if two people are available.

## Qwen3.5-2B coverage tracker

| Op | Engine | Milestone | Status |
|---|---|---|---|
| All 10 projection types (K = 2048/6144) | QMV slice | C1 | done (single slice) |
| LM head 2048 → 248,320 + greedy argmax | QMV array + argmax | C2 | |
| Activation quantize (a8 + bf16 group scales) | FVU `quant` | C4 | placeholder in `bpuref.quant` |
| Embedding lookup (tied table) | QMV / FVU `dequant` | C4 | |
| RMSNorm (hidden), per-head Q/K RMSNorm | FVU + SFU `rsqrt` | C3/C4 | |
| Residual add, SiLU(gate)·up | FVU + SFU | C3/C4 | |
| Causal conv1d k=4 + SiLU | FVU `conv_step` | C5 | |
| q/k L2 norm, β/α gates | FVU + SFU (`exp`, `softplus`, `sigmoid`) | C5 | |
| DeltaNet recurrent state step | FVU `vecmat` + `rank1` | C5 | |
| Gated RMSNorm | FVU + SFU | C5 | |
| Partial RoPE | FVU `rope` | C5 | |
| Decode attention (GQA, online softmax) | FVU `matvec`/`vecmat` + SFU | C5 | |
| Attention output gate | FVU + SFU `sigmoid` | C5 | |

## Tapeout track (parallel with C2–C7)

* **Candidate chip ("mini-BPU")**: 1 QMV slice (16 lanes) + FVU (2–4 lanes) +
  SFU + scratchpad + Wishbone wrapper, running the tiny-Qwen config bit-exactly
  against the FPGA build.
* **Early learning run**: the `bpu_qmv_dot` + `bpu_fp32_*` datapath at a tiny
  configuration on a Tiny Tapeout shuttle, to learn the flow before chipIgnite.
* **Current data point** (generic Yosys cells from `scripts/synth_yosys.sh`,
  SRAMs excluded):

  | Block | Cells |
  |---|---|
  | slice, asic config (16 lanes) | ~14.3K incl. ~760 flops, plus a 2 KiB activation SRAM |
  | slice, fpga config (64 lanes) | ~31.9K incl. ~2.8K flops |
  | `bpu_qmv_dot`, 16 / 64 lanes | ~5.0K / ~20.7K |
  | `bpu_fp32_mul`, `bpu_fp32_add`, `bpu_int2fp32` | ~4.8K, ~1.5K, ~0.4K |

  At the asic size, MAC lanes and the fp32 product multiplier cost about the same.
  **Area optimization, planned for C7:** fuse `int2fp32` and the product multiplier into one
  int22 × fp32 multiplier. The result is bit-identical, because the conversion is
  exact, so rounding the exact product once is the same computation.

## Needed from other teams

| From | What | Needed by |
|---|---|---|
| architecture + ml-models | W4A8 vs W4A16 decision; bf16 vs E8M0 scales | before freezing C1 (the RTL assumes W4A8 + bf16) |
| rtl-memory + ml-compiler | agreement on layout L0 (or an adapter spec) | C2 |
| ml-models | activation-quantize rule; SFU accuracy targets; confirmed Qwen3.5 tensor names/norm conventions; a tiny-Qwen config + per-layer golden vectors | C3–C6 |
| architecture / rtl-control | command descriptor encoding and completion protocol | C6 |
