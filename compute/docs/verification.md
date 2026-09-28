# Verification evidence

Every claim in these docs and the command that reproduces it. Run from `compute/`
with the OSS CAD Suite environment sourced (`source ~/tools/oss-cad-suite/environment`).
CI (`.github/workflows/compute.yml`) runs all of it except the Hugging Face check
and the unbounded FVU proof (it needs about 16 GB).

| Claim | Evidence | Command |
|---|---|---|
| RTL is clean SystemVerilog at every configuration | Verilator `-Wall` lint of every top at every configuration, plus the 64-lane FVU and the reduction unit's edge shapes | `scripts/lint.sh` |
| The open ASIC flow accepts the RTL | Yosys (slang frontend) synthesis of every block | `scripts/synth_yosys.sh` |
| fp32 mul/add are IEEE RNE with subnormals | 100M random vectors each vs the host FPU; all 2^32 bf16×bf16 products; all 2^22 int→fp32 | `scripts/soak_fp32.sh 100000000`, `BPU_SOAK_BF16=1 scripts/soak_fp32.sh` |
| SFU matches its spec | ~138M vectors: every mantissa of each function's critical binades + random, bit-exact | `scripts/soak_sfu.sh` |
| SFU accuracy | ≤ 0.75 / 0.58 / 1.19 / 1.44 / 0.52 ulp (rcp / rsqrt / exp2 / exp / log2) vs float64 | `pytest model/tests/test_sfu_ref.py` |
| SFU RTL ROM = frozen tables | generated from `sfu_tables_data.py` and checked | `python -m bpuref.gen_sfu_rom --check` |
| QMV slice is correct and full-throughput | bit-exact at 3 configs: random shapes, backpressure, Qwen K = 2048/6144, extreme values, 1 beat/cycle, illegal commands, flags, counters | `pytest -k qmv_slice` |
| QMV slice control is safe for all inputs | unbounded PDR proofs (no output overflow, credit bounds, scale/beat pairing, counters in range, output stable under backpressure) at 3 configs; 40-cycle BMC; covers. Proof engines race (rIC3 and ABC PDR): PDR alone is sensitive to netlist ordering and once took over an hour on R=4, while rIC3 takes 32 s | `cd formal && sby -f qmv_slice.sby` |
| QMV dot product is the textbook dot product | SAT equivalence for all inputs, W4 and W8, 2 and 4 lanes | `cd formal && sby -f qmv_dot.sby` |
| QMV array | bit-exact at 32 / 1 / 3 slices: stream and argmax (ties, NaN), padding rows dropped, Qwen shapes | `pytest -k qmv_array` |
| FVU | bit-exact at 16 / 2 / 4 / 64 lanes with 3 / 1 / 2 / 3 SPM read ports: directed corner ops for every opcode, VVECMAT short-row accumulator races (forwarding), 3 × 40 random legal op sequences (several seeds), illegal descriptors | `pytest -k fvu`, `BPU_SEED=n pytest -k fvu` |
| FVU reduction unit | bit-exact at 4 shapes (1–16 lanes, combinational to fully pipelined adders, 2–24-entry FIFO): 450 random rows of every tree depth incl. one-word rows, sums and maxes mixed back to back, random input gaps; merge throughput 1.00–1.10 cycles per word | `pytest -k fvu_reduce` |
| FVU control is safe | BMC 30 cycles at 1 and 2 read ports: reduction credits and FIFO never overflow, pipeline count bounded, one SPM writer per cycle, external port gated while running, one port per staging register, merge levels in range; covers: an op completes, a reduction writes its result | `cd formal && sby -f fvu.sby bmc bmc_p2 cover` |
| FVU control is safe for all time | unbounded proof of all 8 properties above (2 lanes, 2-entry FIFO). The rIC3 portfolio takes about 9 minutes and peaks near 16 GB, so it runs locally, not in CI. ABC PDR alone stalls on the merge-level bound, and single rIC3 engines (IC3, k-induction) did not finish in 30 minutes | `cd formal && sby -f fvu.sby prove` |
| Qwen3.5 semantics | float64 reference = Hugging Face transformers 5.17 to ~2e-7 over 5–6 decode steps | `pytest model/tests/test_qwen_ref.py` (needs torch), `python -m bpuref.hf_check` |
| The compiled program computes Qwen3.5 | W4A8 program vs float64 model with the same weights: logit cosine > 0.999, argmax agrees | `pytest model/tests/test_qwen_ref.py` |
| Programs are hazard-free | every FVU op of the compiled program passes `validate()` (alignment, bounds, read/write aliasing) | same |
| **The RTL runs Qwen3.5** | tiny-Qwen decode, 3 tokens, fpga / asic / tiny: whole SPM and chosen token bit-exact after every token | `pytest -k compute_top` |
| The cycle model behind the 2B projection tracks the RTL | within 5% of the measured cycles on every decode step at every configuration (today −2.8% / +0.2% / −0.2%) | same |
| The Tiny Tapeout wrapper works through its pins | host byte protocol, 979 operand pairs (random bits, typical values, all pairs of 24 specials, the bring-up vectors) × add / sub / mul / loopback, bit-exact | `pytest -k tt_fp32` |
| Synthesis preserves function | gate-level netlists (Yosys, flattened, memories mapped to flops) of the SFU, the QMV slice, the FVU reduction unit and a 2-lane FVU with a shared SFU and a 2048-element SPM pass their unchanged cocotb tests (the FVU: all 36 directed ops and 3 × 40 random ops). The FVU is mapped without ABC, which takes hours on its flattened combinational fp32 logic | `BPU_GATES=1 pytest tb/test_gates.py` |
| The tests can fail | bugs injected by hand during development (adder tie rounding, W8 low-nibble sign, SFU log2 rounding) were each caught by the suite | re-inject and run the matching `pytest -k` |

## What is not verified yet

* **Timing.** No Vivado run (F2) and no OpenLane/sky130 run. Generic-gate depth is
  only a proxy. Unknown until someone runs those flows.
* **Real-checkpoint quality.** Accuracy of W4A8 + these numerics on the real
  Qwen3.5-2B (perplexity, tasks) belongs to ml-models.
* **Full-size integration.** The full 2B model has not run in RTL simulation (it
  would take days); the tiny model exercises every operation and code path.
  Real-size shapes are covered per engine (K = 2048/6144 on QMV; 128×128 state
  ops are the same ops).
* **FVU data hazards are checked by simulation, not formally.** The formal model
  cuts out all data, so VVECMAT forwarding and the reduction pairing order rest on
  the bit-exact tests (directed races, random sequences, the decode).
