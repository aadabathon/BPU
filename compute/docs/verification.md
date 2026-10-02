# Verification evidence

Every claim in these docs and the command that reproduces it. Run from `compute/`
with the OSS CAD Suite environment sourced (`source ~/tools/oss-cad-suite/environment`).
CI (`.github/workflows/compute.yml`) runs all of it except the Hugging Face check and
the FVU's unbounded proof (an hour).

| Claim | Evidence | Command |
|---|---|---|
| RTL is clean SystemVerilog at every configuration | Verilator `-Wall` lint of 32 targets: every block at every named configuration (generated from `bpuref/configs.py`) plus edge shapes | `scripts/lint.sh` |
| The open ASIC flow accepts the RTL | Yosys (slang frontend) synthesis of every block, including `bpu_core` | `scripts/synth_yosys.sh` |
| fp32 mul/add are IEEE RNE with subnormals | 100M random vectors each vs the host FPU; all 2^32 bf16×bf16 products; all 2^22 int→fp32 | `scripts/soak_fp32.sh 100000000`, `BPU_SOAK_BF16=1 scripts/soak_fp32.sh` |
| SFU matches its spec | ~138M vectors: every mantissa of each function's critical binades + random, bit-exact | `scripts/soak_sfu.sh` |
| SFU accuracy | ≤ 0.75 / 0.58 / 1.19 / 1.44 / 0.52 ulp (rcp / rsqrt / exp2 / exp / log2) vs float64 | `pytest model/tests/test_sfu_ref.py` |
| SFU RTL ROM = frozen tables | generated from `sfu_tables_data.py` and checked | `python -m bpuref.gen_sfu_rom --check` |
| **Shared SRAM** behaves as specified | random traffic on every port of 3 shapes (1–8 banks, 2–5 read and 2–3 write ports, with and without the output register and hashing), several seeds, against a reference memory that applies the visibility rule cycle by cycle: data, response latency and order, out-of-range flags; the round-robin wait bound is reached and never exceeded; streams in different banks never wait | `pytest -k sram_shared` |
| **Shared SRAM** is correct for all time | unbounded proofs (with and without the output register): one grant per bank and port kind, round-robin wait bound, fixed response latency, out-of-range flags, and **data integrity** for a symbolic address (every read returns the last write accepted before it) | `cd formal && sby --sequential -f sram_shared.sby` |
| **Sequencer** issues in dependency order | unbounded proof: queued and running commands are pending, remaining dependencies are pending and never the command itself, one command in flight per unit, a tracked command never issues before the commands it waited for complete; cover: a command waits, then issues | `cd formal && sby --sequential -f cmd_seq.sby` |
| QMV slice is correct and full-throughput | bit-exact at 3 configs: random shapes, backpressure, Qwen K = 2048/6144, extreme values, 1 beat/cycle, illegal commands, flags, counters | `pytest -k qmv_slice` |
| QMV slice control is safe for all inputs | unbounded proofs (no output overflow, credit bounds, scale/beat pairing, counters in range, output stable under backpressure) at 3 configs; 40-cycle BMC; covers | `cd formal && sby --sequential -f qmv_slice.sby` |
| QMV dot product is the textbook dot product | SAT equivalence for all inputs, W4 and W8, 2 and 4 lanes | `cd formal && sby --sequential -f qmv_dot.sby` |
| QMV array | bit-exact at 16 / 1 / 3 slices: stream and argmax (ties, NaN), padding rows dropped, Qwen shapes | `pytest -k qmv_array` |
| **FVU on the shared SRAM** | bit-exact at 16 / 2 / 4 / 64 lanes (1–3 read ports, 2–8 collector slots), plus a variant with VVECMAT forwarding off. Every op runs while a second client hammers the same SRAM with random reads and writes. Directed corner ops for every opcode, VVECMAT short-row accumulator races, 3 × 40 random legal op sequences (several seeds). Illegal descriptors, unknown SFU functions and out-of-range addresses complete with an error. Mutation check: removing the "wait for the previous row's write" gate fails the tests | `pytest -k fvu`, `BPU_SEED=n pytest -k fvu` |
| FVU reduction unit | bit-exact at 4 shapes (1–16 lanes, combinational to fully pipelined adders, 2–24-entry FIFO): 450 random rows of every tree depth, sums and maxes mixed back to back, random input gaps; merge throughput 1.00–1.10 cycles per word | `pytest -k fvu_reduce` |
| FVU control is safe | on a real shared SRAM with a free competing client: BMC 25 cycles at 1 read port and 22 at 2 read ports. Checked: credits and buffers never overflow or underflow, collector bookkeeping, read requests stable until accepted, the round-robin wait bound on every FVU port, the SRAM's own grant rules. Covers: an op completes, a reduction writes its result, a forwarded VVECMAT completes | `cd formal && sby --sequential -f fvu.sby bmc bmc_p2 cover` (CI runs bmc and cover) |
| FVU control for all time | unbounded (IC3 + PDR, addresses below 256): 27 of 30 properties proved in an hour. Open: the merge-level bound (unchanged logic, proved earlier with the rIC3 portfolio), the forwarding-FIFO bound and request stability. All three hold for 25 cycles, and the first two are also checked in every simulation | `cd formal && sby -f fvu.sby prove` |
| Qwen3.5 semantics | float64 reference = Hugging Face transformers 5.17 to ~2e-7 over 5–6 decode steps | `pytest model/tests/test_qwen_ref.py` (needs torch), `python -m bpuref.hf_check` |
| The compiled program computes Qwen3.5 | W4A8 program vs float64 model with the same weights: logit cosine > 0.999, argmax agrees | `pytest model/tests/test_qwen_ref.py` |
| Programs are hazard-free | every FVU op passes `validate()` (alignment, bounds, read/write aliasing); `bpuref.sched` derives cross-unit dependencies from read/write footprints | same |
| Activation quantization has one definition | `bpuref.quant.quantize_activations` equals the compiled FVU sequence, including all-zero groups | `pytest model/tests/test_bpuref.py` |
| **The core runs Qwen3.5** | tiny-Qwen decode on `bpu_core`, 3 tokens, fpga / asic / tiny. Each step runs as 443 tagged descriptors through the sequencer (112 with cross-unit waits); the SRAM region and chosen token are bit-exact after every token, and every command completes, without error, after the commands it depends on | `pytest -k core` |
| Cross-unit dependencies and tag reuse | vector commands interleaved with memory-manager commands (random delays) on reused tags: completion order respects every dependency; the reserved unit completes with an error | same |
| The cycle model behind the 2B projection tracks the RTL | within 5% of the measured cycles on every decode step at every configuration (today +0.1% / −0.0% / −1.1%) | same |
| The Tiny Tapeout wrapper works through its pins | host byte protocol, 979 operand pairs × add / sub / mul / loopback, bit-exact | `pytest -k tt_fp32` |
| Synthesis preserves function | gate-level netlists (Yosys, flattened, memories mapped to flops) pass their unchanged cocotb tests: the SFU, the QMV slice, the FVU reduction unit, and the FVU together with a 4-bank shared SRAM (all 36 directed ops and 3 × 40 random ops under competing traffic). The FVU + SRAM is mapped without ABC, which takes hours on its flattened combinational fp32 logic | `BPU_GATES=1 pytest tb/test_gates.py` |
| The tests can fail | bugs injected by hand were each caught: adder tie rounding, W8 low-nibble sign, SFU log2 rounding, and the VVECMAT accumulator gate | re-inject and run the matching `pytest -k` |

## What is not verified yet

* **Timing.** No Vivado run (F2) and no OpenLane/sky130 run. Generic-gate depth is
  only a proxy. The SRAM crossbar (32 banks × 5 read ports × 512 bits at the fpga
  configuration) is a new wide structure to watch.
* **Real-checkpoint quality.** Accuracy of W4A8 + these numerics on the real
  Qwen3.5-2B (perplexity, tasks) belongs to ml-models.
* **Full-size integration.** The full 2B model has not run in RTL simulation (it
  would take days, and its state does not fit the SRAM without the memory manager);
  the tiny model exercises every operation and code path.
* **Data hazards are checked by simulation, not formally.** The FVU's formal model
  cuts out all data, so VVECMAT forwarding and accumulator ordering and the
  reduction pairing order rest on the bit-exact tests (directed races, a mutation
  check, random sequences under contention, the decode). The shared SRAM's data
  integrity is proven.
* **The memory manager** is outside the core; its behaviour in the tests is a cocotb
  model (SRAM image and inputs, weight streams, commands with random delays).
