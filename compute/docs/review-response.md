# Response to the RTL reuse review (2 October 2026)

The review (`NOTES_FOR_AGENTS_README/BPU_RTL_Reuse_Review.md`) compared the compute
RTL with the block diagram (`meeting_docs/zeb_block_diagram.png`). Its main finding
was that memory ownership and timing, not the arithmetic, kept the RTL from fitting
a shared-SRAM architecture. This page lists each point and what changed. The block
diagram is itself a sketch, so choices it leaves open are marked provisional, not
settled.

## Done

| Review point | Change |
|---|---|
| FVU owns its scratchpad; fixed one-cycle reads; writes cannot be refused (§2) | `bpu_fvu` has no memory. Operand-collector slots fetch through request/response read ports with in-order tag FIFOs; items launch only when their operands are in; a credited write buffer feeds a write port that may wait. This is the review's strategy 2, keeping the arithmetic pipelines stall-free |
| VVECMAT forwarding and row spacing depend on timing (§2) | Short rows forward the accumulator through a FIFO (no SRAM round trip). Long rows read it back, and each read waits for the previous row's write of that word to be **accepted**, counted rather than timed. A mutation test confirms the regression catches removing the gate |
| `SpmReadPorts` replicates memory (§2) | Removed. `bpu_sram_shared` is a banked shared SRAM with per-bank round-robin arbitration and XOR-hashed bank mapping; the FVU's read ports are clients like any other |
| Serial top, no scoreboard, no IDs or completion (§3) | `bpu_cmd_seq` + `bpu_scoreboard`: tagged descriptors with wait masks, one queue per unit (vector, matrix, memory manager), units concurrent, completion per tag. Wait masks are resolved at acceptance, so reusing a tag never aliases a newer command |
| "Done" must mean visible (§3) | Every engine reports done only after its last SRAM write is accepted. The SRAM's rule is a write accepted in cycle t is seen by reads accepted after t |
| QMV slice ready means accepted, not done (§3) | `bpu_qmv_engine` reports done after its writes, and takes a new command only once the array is ready (the slices finish padding rows after the last real result, which the regression caught) |
| QMV behind a memory adapter (§4) | `bpu_qmv_engine`: loads activations and scales through an SRAM read port (credits, gearbox), writes packed results through a write port, keeps the weight-stream interface for the memory manager |
| 16 × 256-bit HBM vs 32 slices (§Architecture, §4) | The fpga configuration has 16 slices to match the diagram; 32 is a parameter. Projections cover both (performance.md) |
| RTL accepts addresses that wrap (§5) | Addresses past the SRAM no longer wrap: the SRAM accepts them without touching a bank and flags them, and the engine completes the command with an error. The review's example (VCOPY whose second source row starts at element 2048) is in the FVU tests and completes with an error. Read/write aliasing stays a software rule (`bpuref.fvu.validate`, which now also checks bounds) |
| Unsupported SFU function codes accepted (§5) | Rejected (consumed with an error) |
| `quant.py` placeholder disagrees with the compiled quantizer (§5) | `quantize_activations` now runs the compiled FVU sequence itself; a test pins it to the compiler's output, including the all-zero-group case |
| Separate numerics from provisional encodings (§6) | `bpu_compute_pkg` = numerics; `bpu_isa_pkg` (and `bpuref/isa.py`) = opcodes, function codes, descriptor layout |
| Hand-mirrored configuration tables | Lint and synthesis targets are generated from `bpuref/configs.py` |

## Not changed, and why

| Review point | Status |
|---|---|
| Keep the serial top as a reference (§Recommendation) | Replaced, not kept beside the new top. The golden reference is `bpuref`, which the RTL matches bit for bit at every configuration. A second top would mean maintaining a second FVU. The serial top remains in git history |
| Q8.24 (matrix, vector), INT16 and FP16 (vector) | Not built. Each needs a numerics spec first: product and accumulator widths, rounding, overflow and saturation, conversions. Each also needs a reference model and tests. FP32 compute with bf16 storage already covers Qwen3.5. Needs a team decision (roadmap) |
| BF16 activations into the matrix unit (§1) | Matrix activations stay int8 (A8). A bf16-activation path is a different multiplier and a numerics decision |
| VQCLAMP codes stored as fp32 (4× storage) (§1) | Unchanged; the codes are a temporary, hidden-sized vector. Packing them is a small gearbox change if SRAM space gets tight |
| Matrix–matrix / prefill (§4) | Out of scope for decode; QMV handles repeated vectors without tiled reuse |
| SRAM capacity: DeltaNet state is 18 MiB vs 8 MiB SRAM (§2) | Agreed. The core leaves room for it (memory-manager port and commands), but streaming or staging state is rtl-memory's design (roadmap) |
| Formal proofs are control-only (§5) | Still true for the engines, and stated per harness. New: the shared SRAM is proven unbounded including **data integrity** for a symbolic address, and the sequencer's dependency ordering is proven unbounded. The FVU on the SRAM passes BMC and covers; 27 of its 30 control properties are proven unbounded (verification.md lists the open three). Engine data correctness rests on bit-exact simulation and gate-level runs |
| Cost of the rework | The asic `bpu_core` is ~0.72 mm² of sky130 logic versus 0.43 mm² for the serial top. Most of the difference is the FVU's operand collector and buffering, plus the sequencer queues. The FVU's longest path also doubled. Both have known levers (performance.md) |
| `QmvGroup` 64 is spread across ports and code (§1) | Unchanged; group size 64 is part of the numerics contract |

## Decisions still needed from the team

The review's list stands. With this design:
- **Settled in the RTL, but provisional:**
  - element (fp32) addresses;
  - interleaved, hashed bank mapping;
  - fixed read latency with in-order responses;
  - "done = writes accepted";
  - wait-mask dependencies over 16 tags.
- **Still open:**
  - activation type;
  - quantization format, group size and scale type;
  - which of Q8.24, FP16 and INT16 are needed;
  - KV precision;
  - the memory manager's command format and capacity plan;
  - whether weights stream from HBM directly or through the SRAM;
  - the HBM channel count.
