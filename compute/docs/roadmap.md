# Compute roadmap: Qwen3.5-2B decode

Scope: the compute and shared-memory half of the BPU block diagram. That means the
vector and matrix units, the shared SRAM with its arbiter, and the command sequencer
and scoreboard. It is bit-exact with `bpuref`, runs at the F2 size, and scales down
for tapeout.

Out of scope: the control SoC (it produces descriptors), the memory manager
(rtl-memory), HBM and the shell. `bpu_core` defines the seams: descriptors in, a
scoreboard out, and the memory manager's command, SRAM and weight-stream ports.

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| C0 | Contract + scaffolding: numerics and op specs, `bpuref`, toolchain, config sweeps, lint, synthesis | **done** | docs, `scripts/` |
| C1 | QMV slice: IEEE fp32 units, W4/W8 × A8 dot, row-interleaved accumulate, credits, flags, counters | **done** | slice tests, formal proofs, fp32 soak, gate-level sim |
| C2 | QMV array: row striping, in-order merge, argmax (LM head), logical row count | **done** | array tests at 16 / 1 / 3 slices |
| C3 | SFU: rcp, rsqrt, exp2, exp, log2, bit-exact tables, ≤ 1.44 ulp | **done** | 138M-vector soak, gate-level sim |
| C4 | FVU core: element-wise, conversions, quantize, canonical reductions | **done** | FVU tests |
| C5 | FVU 2-D ops and Qwen blocks: VVECMAT, per-row scalars, VPERM (RoPE), conv, delta rule, attention | **done** | same, plus the compiled Qwen program |
| C6 | Integrated decode: QMV ↔ memory gearbox, weight requests; full Qwen decode | **done** | tiny-Qwen decode bit-exact, 3 tokens, 3 configs |
| C7 | Hardening | **partly** | lint, Yosys synthesis, gate-level sim, unbounded control proofs done; timing and PnR not started |
| C8 | FVU throughput: pipelined reduction merge, VVECMAT forwarding | **done** | reduce unit test, FVU + decode bit-exact |
| C9 | Shared-SRAM architecture (block diagram): banked shared SRAM, request/response engines, sequencer + scoreboard, `bpu_core` | **done** | SRAM vs a cycle-accurate reference memory, FVU under competing traffic, decode as 443 tagged descriptors bit-exact at 3 configs, formal (SRAM, sequencer, FVU on the SRAM), gate-level FVU + SRAM ([review-response.md](review-response.md)) |

## Qwen3.5-2B coverage

Every decode operation is implemented, runs in the compiled program, and is checked
bit-exact on the RTL:
- all ten projection types and the tied LM head with greedy argmax;
- embedding dequantize and activation quantize;
- RMSNorm (`1 + w`) and gated RMSNorm (`w`);
- causal conv + SiLU; q/k l2norm and scaling; the β and α gates (sigmoid, softplus with threshold);
- the gated delta rule;
- per-head Q/K RMSNorm, partial RoPE and KV append;
- GQA attention with softmax; the output gate; SiLU·up; residuals.

## Next, in priority order

1. **Capacity: where the 2B model's state lives.** It is the first thing the shared
   SRAM forces, and it is rtl-memory's design. In fp32:
   * DeltaNet state: 18 layers × 16 heads × 128×128 = 18 MiB;
   * KV cache: 24 KiB per position (3 MiB at 128, 48 MiB at 2K context).

   The diagram's 8 MiB SRAM holds neither. The core is ready for either plan, since
   the memory manager has its own SRAM ports and commands in the sequencer with
   dependencies:
   * **(a) stage per layer.** The memory manager streams each layer's state (1 MiB)
     and KV slice into the SRAM before the layer, and back after. That is about
     113 MiB of state traffic per token, ~0.3 ms at HBM rates;
   * **(b) attention on the matrix unit with an INT8 KV cache** streamed like weights,
     which also lifts long-context speed (below).
2. **Performance** (see [performance.md](performance.md)). On the diagram's 16 HBM
   interfaces, the fpga configuration projects 68 / 36 / 15 tok/s at 128 / 2K / 8K
   context. Weight bandwidth (16 slices) and the vector unit share the time. Levers:
   * attention on QMV (INT8 KV): 88 / 70 / 64 tok/s. The QMV W8 mode already
     computes it; it needs an accuracy check by ml-models and a KV layout;
   * 32 HBM pseudo-channels instead of 16 (if the shell exposes them): 89 / 42 / 15;
   * a 32-lane FVU: 81 / 53 / 25. The 512 B/cycle SRAM budget caps the vector unit
     around 32 lanes;
   * **compiler overlap**: units now run concurrently, but the compiled step reuses
     a few scratch buffers, so most work is still a dependency chain (about 5%
     overlap). Double-buffering the scratch regions lets the next projection's
     load and the current vector work overlap;
   * a fused delta-rule op with a local accumulator.
3. **Timing closure.**
   * Out-of-context Vivado at 250 MHz for `bpu_qmv_slice`, `bpu_fvu_lane`, and the
     SRAM crossbar (32 banks × 5 read ports at 512 bits is the new wide structure);
   * register the FVU operand collector's request selection: its mapped path doubled
     with the shared-memory rework;
   * OpenLane 2 on sky130 for the asic configuration.
4. **Area for tapeout.** The asic `bpu_core` is ~0.72 mm² of sky130 logic plus SRAM
   macros ([performance.md](performance.md)); the shared-memory machinery added
   ~0.29 mm² over the serial top. Levers:
   * an address-width parameter for the FVU's slots, row bases and buffers;
   * 1-deep sequencer queues;
   * `AccDepth = 0` at the asic size;
   * SRAM sizing for the chosen tapeout model.
5. **Formal coverage of data.** The control logic of every block is proven
   unbounded, and the shared SRAM's data integrity is proven for a symbolic
   address. The engines' data paths rest on bit-exact simulation; a symbolic-data
   model at a tiny shape would close that.

## Tapeout track

* **Candidate:** `bpu_core` at the asic configuration: 1 QMV slice × 16 lanes, a
  2-lane FVU with one shared SFU, a 4-bank shared SRAM (512 KiB in the tiny-model
  build), the sequencer. That is ~0.72 mm² of logic plus SRAM macros, and it runs
  the tiny model bit-exact against the FPGA build and the reference.
* **Early learning run (ready to submit):** `compute/tapeout/tt` wraps the unchanged
  fp32 adder and multiplier for a Tiny Tapeout shuttle behind a byte-wide host
  protocol. It is 0.052 mm² of sky130 cells, a 4x2-tile slot, with a pin-level test
  and `assemble.sh` to build the submission tree.

## Needed from other teams

| From | What | Why |
|---|---|---|
| architecture + ml-models | activation type (A8 today), quantization format and group size, whether Q8.24 / FP16 / INT16 are needed, KV precision | numerics freeze |
| ml-models | real-checkpoint quality of these numerics (perplexity) | accuracy sign-off |
| rtl-memory | memory-manager command format; capacity plan (staging vs streaming); weights straight from HBM or through the SRAM; HBM interface count | integration, bandwidth |
| architecture / rtl-control | descriptor format and tag count (`bpu_isa_pkg` is provisional); who generates dependency masks (`bpuref.sched` shows one way) | integration |
