# BPU RTL reuse review

Reviewed 2 October 2026: the uploaded `compute.zip` (24 SystemVerilog source files, about 4,166 lines including the generated ROM), its reference models, tests, documentation, and archived formal results. This is the compute subtree, not a checkout of the complete private BPU repository. The comparison baseline is Zeb's supplied block diagram and his comments about FP32 vector support, INT4/INT8 matrix support, optional Q8.24, centralized SRAM arbitration, and software-specified dependencies. That baseline is an architectural sketch, not a finished interface or numerical specification. Differences below are implementation choices to reconcile, not automatically mistakes.

## Recommendation

Reuse the arithmetic modules and their tests. Keep the existing integrated top as a numerical and simulation reference. For Zeb's shared-memory architecture, adapt the QMV engine behind a memory adapter and rebuild the FVU's memory-facing controller around request/response handshakes. Do not adopt the top's serial scheduling, local scratchpad organization, or quantization contract as team requirements without deciding them explicitly.

The substantial work here is real: signed nibble arithmetic, configurable arithmetic pipelines, group scaling, output credits, ordered reductions, special-function approximation, and an executable reference compiler. The largest integration mismatch is memory ownership and timing, rather than the underlying integer dot product or FP32 arithmetic.

## Architecture comparison

| Area | Zeb's sketch | Uploaded implementation | Consequence |
|---|---|---|---|
| Command issue | Commands to matrix, vector, and memory units; scoreboard eligibility | `bpu_compute_top` dispatches exactly one operation at a time | Existing top is a useful serial bring-up implementation; independent engine issue needs a different integration layer |
| Dependency tracking | Scoreboard updated by compute and memory | Busy/ready signals and sticky flags; no job ID, dependency ID, or tagged completion interface | Add per-engine command identity and completion reporting; software dependencies can simplify scheduling |
| Memory ownership | Shared 8 MiB SRAM, 32 banks, centralized arbitration | FVU instantiates its scratchpad internally; QMV has local activation/scale buffers | FVU must be separated from storage or wrapped with a deliberately reserved local-memory interface |
| Memory timing | Arbitration must account for competing users | Fixed one-cycle reads; unconditional scheduled writes | Requires request acceptance, response tracking, write buffering, and admission credits |
| Matrix types | INT4, INT8, possible Q8.24 | W4/W8 weights × signed A8 activations; integer group sum, FP32 scaling and accumulation | Q8.24 and floating activation modes require new arithmetic paths |
| Vector types | INT16, Q8.24, FP16, FP32 | FP32 arithmetic, BF16 rounding, quantization clamp | No native INT16/Q8.24/FP16 modes |
| Matrix operations | Matrix unit, with operation set still unspecified | Primarily quantized matrix–vector products for single-token decode | Matrix–matrix can be decomposed into repeated vector products, but efficient tiled reuse/prefill is additional work |
| Weight delivery | Memory manager stages/transfers tensors | Tensor-ID request followed by slice-specific packed code and BF16-scale streams | Memory manager needs a tensor table and exact stream-layout adapter |
| Bandwidth configuration | Diagram labels 16 × 256-bit HBM interfaces | Named FPGA config has 32 × 256-bit slice inputs | Do not inherit a 32-stream full-rate performance claim for a 16-stream system without remapping/rate modelling |

The diagram's HBM label might count controllers differently from the repo's pseudo-channels. This is an unresolved interface-count question, not proof that either physical platform description is wrong.

## 1. Numerical behavior you would inherit

### Matrix datapath

`rtl/qmv/bpu_qmv_dot.sv` implements an exact signed dot product. W4 uses one signed nibble per lane. W8 reconstructs a byte from an unsigned low nibble and signed high nibble:

`signed_byte = unsigned(low_nibble) + 16 * signed(high_nibble)`.

Both terms multiply the same signed INT8 activation. This shares small multipliers between W4 and W8; W8 consumes half as many weights per fixed-width beat. It is an especially useful module to study and reuse.

`bpu_qmv_slice.sv` sums each 64-element quantization group into a signed 22-bit value, converts that exact sum to FP32, multiplies BF16 weight and activation scales, multiplies the group sum by the combined scale, then accumulates groups in a fixed order using FP32 addition:

`y[n] = sum_g FP32(INTSUM[n,g] * FP32(sw[n,g] * sx[g]))`.

The individual INT4 × INT8 products are not each rounded to FP32. The integer sum is exact inside the group; rounding enters the scaling and inter-group accumulation path. There is no fused multiply-add.

BF16 in this engine is primarily a scale encoding. Activations entering the matrix engine are INT8, not BF16. This differs from an INT4-weight/BF16-activation plan. To preserve that alternative plan, one would need a floating activation multiplication path or a separately agreed activation quantizer; reinterpreting the operand bits would be incorrect.

The format contract is symmetric quantization with no zero-point input, group size 64, signed two's-complement codes, low nibble first, and BF16 scales. A different checkpoint format cannot be assumed compatible merely because it says “4-bit.” It needs matching metadata and layout, or conversion/repacking and possibly requantization.

`QmvGroup` looks centralized, but changing it alone is insufficient: top-level ports, index shifts, alignment checks, compiler operations, and packing code contain explicit 64-element assumptions.

### Vector datapath

`bpu_fvu_lane.sv` supplies FP32 addition/subtraction, multiplication, scalar multiplication, multiply-then-add, selection, copy, absolute value, BF16 rounding, and integer quantization clamp. `VRBF16` rounds a value to BF16 precision and keeps it in a 32-bit FP32 container with the low 16 bits zero. `VQCLAMP` returns an exact integer-valued FP32 value in [-127,127]; it does not store a packed INT8 byte. The top later converts those values into the QMV activation buffers. That is a 4× storage cost for the temporary codes relative to packed bytes.

The SFU supports reciprocal, reciprocal square root, exponential base two, exponential, and logarithm base two. The RTL and coefficient tables define an approximation, not correctly rounded IEEE transcendental operations. Subnormal SFU inputs are treated as zero and subnormal outputs are flushed to zero; the ordinary FP32 add/multiply modules support subnormals. Reuse the ROM, reference model, generator, and accuracy tests together.

The FVU reduction unit implements a canonical adjacent-pair sum tree so lane counts do not change numerical results. Its merge controller is more involved than an ordinary running accumulator, but useful if reproducibility across FPGA/ASIC configurations is a requirement. Changing reduction order or introducing fused arithmetic changes the reference contract and may change model outputs.

### Q8.24

Q8.24 is fixed point: under the usual signed 32-bit convention, the encoded signed integer represents `raw / 2^24` (confirm whether the sign is included in the “8” with the team). A product has 48 fractional bits before rescaling. Correct support needs specified product width, accumulator width, rounding, overflow/saturation, and conversion rules. Neither the INT8 multiplier nor FP32 modules become Q8.24 units by changing a port label.

## 2. Memory adaptation is the major rewrite

`bpu_fvu.sv`, around lines 225–367, records which operands were requested and stages their data on a fixed schedule. `ret_slot_q <= issue ? pick : 0` assumes the requested read happened. The read data arrives one cycle later, and the item enters arithmetic two cycles after its last read. There is no read-request-ready input or response-valid input.

Its write path around lines 504–522 chooses arithmetic results, reduction results, or idle external writes. There is no write-ready input. The external port is available only while the FVU is idle; it is not a concurrent DMA or shared-memory client port.

Therefore, replacing the internal memory wrappers with an arbitrated SRAM connection alone is insufficient. On a denied/delayed read, the controller would advance with stale or unrelated data. On a denied write, results would be lost. Existing fixed row spacing and the two-entry VVECMAT forwarding buffer also depend on these timing guarantees.

Two legitimate adaptation strategies:

1. Reserve guaranteed read/write slots for an operation and preserve the existing timing contract. This keeps more controller RTL, but restricts sharing and requires an explicit reservation mechanism.
2. Make operand fetch request/response driven, launch only when all operands are staged, and reserve result-buffer capacity before launch. Buffer writes until accepted/committed. Track VVECMAT accumulator hazards against actual writes/forwarded results rather than elapsed cycles alone. This is more flexible and fits shared arbitration better.

For the second strategy, keep fixed-latency arithmetic pipelines running. Backpressure belongs before launch and at buffered output boundaries. Do not globally freeze arithmetic without also freezing every valid bit, destination tag, scale, and operand delay consistently.

`SpmReadPorts=3` instantiates three complete scratchpad copies with broadcast writes. It is memory replication, not arbitration over three ports or banking of one shared SRAM. Allocating an 8 MiB scratchpad this way consumes 24 MiB. Zeb's 32-bank design needs a banking/address mapping and conflict policy instead.

Using this archive's own model dimensions, DeltaNet persistent FP32 state alone is `18*16*128*128*4 = 18 MiB`; the FP32 key/value cache adds 3 MiB at 128 positions. Thus the complete resident-state organization cannot fit the sketch's 8 MiB SRAM. External-memory streaming, layer staging, or a different precision/storage policy is necessary. This conclusion follows from the archive's configuration; it is not independent verification of the real checkpoint dimensions.

## 3. Command and completion integration

The compute top around lines 175–181 requires both engines ready before accepting any operation. It then runs a QMV load/issue/collect sequence or waits for the FVU to drain. It has no scoreboard and deliberately serializes independent work. It can remain a useful baseline while the team develops scheduling.

Retain engine-local sequencing. A matrix engine still needs row/group/chunk counters, and a vector engine still needs operand fetch and row traversal. These do not conflict with a global command sequencer: the global layer decides which operation may begin, and the local layer executes it.

Add command IDs and a completion/error interface around each engine. For an initial version, one command in flight per engine plus explicit software predecessor IDs is enough; register renaming is not required merely to issue accelerator commands. Memory conflicts and visibility still need a defined policy.

Be careful with `bpu_qmv_slice.cmd_ready_o`: it is `!run_q`, while `busy_o` also includes reserved output credits. Ready can rise once the last input beat is accepted while arithmetic/results are still outstanding. It is an admission signal, not completion. The array serializes results more conservatively, but integration must still distinguish input accepted, result produced, write accepted, and result visible to a dependent command.

When writes move behind the memory manager, “done” must correspond to the team's visibility guarantee. An arithmetic result entering a write FIFO does not alone make its destination safe to read. Read/write requests and completion reports need enough identity to update the correct scoreboard entry.

The existing descriptor ports are an internal micro-operation interface, not a packed instruction encoding or ISA decoder. Keep semantic operations and translate the eventual ISA into them. Opcode numbers, strides, address units, and descriptor widths remain team decisions.

## 4. Weight streaming and matrix scope

`bpu_qmv_array.sv` broadcasts activations to slices and stripes output row n onto slice `n % NSlice`. Each slice receives beats in L0 order: row block → 64-element group → interleaved row → chunk. A weight scale is consumed with its row-group's final chunk. The two valid/ready streams already tolerate input gaps and output backpressure under the engine's credit rules.

This makes QMV substantially easier to adapt than FVU. The memory manager can supply compatible buffered streams. It must resolve tensor IDs, fetch/repack codes and scales, maintain their order, pad output rows, and share the available bandwidth. Buffering must preserve source-valid/data stability until acceptance.

The named FPGA configuration has 32 slices × 64 INT4 lanes. Each slice's code beat is 256 bits: 1,024 bytes/cycle aggregate at full rate. A literal interpretation of the diagram's 16 × 256-bit interfaces supplies 512 bytes/cycle. Reducing NSlice, time-multiplexing the streams, or confirming a controller/pseudo-channel distinction resolves this; performance needs recalculation afterward.

QMV computes `y = W*x`; it is not a systolic GEMM engine. Repeated vectors compute multiple columns correctly, but the controller does not provide tiled reuse across those columns. Prefill and batched matrix multiplication require additional scheduling/storage decisions. The FVU's `VVECMAT` is FP32 vector–matrix accumulation, not a general INT4/INT8 GEMM mode.

## 5. Concrete validation gaps

The Python `fvu.validate()` checks operand ranges and aliasing. RTL `cmd_legal`, lines 134–143, checks opcode, nonempty shape, alignment, and some permutation geometry, but does not check all reached addresses or aliasing. Top QMV legality similarly does not validate complete x/xs/y regions. Hardware address arithmetic can wrap.

Example: with 2,048 scratchpad elements, VCOPY with two 64-element rows, source base 1,984, source stride 64, destination 0, destination stride 64 passes the RTL's shape/alignment checks. The second source row begins at element 2,048 and wraps in the hardware address width. Python validation rejects it. This is a code-derived example, not a claimed simulated failing test.

The FVU also does not explicitly reject unsupported SFU function codes. Decide whether the software descriptor producer is trusted to enforce these restrictions or whether RTL should reject them. The existing tests for illegal FVU operations focus on alignment, empty shape, unknown opcode, and permutation shape, not comprehensive bounds/aliasing validation.

The generic activation quantization helper in `model/bpuref/quant.py` calls itself a placeholder and uses division and a scale of 1 for all-zero groups. The actual compiler `_quant()` uses FVU operations and the SFU reciprocal, with zero scale for an all-zero group. They need not agree bit for bit. Use the actual compiled quantization contract for hardware comparisons; do not substitute the helper silently.

The archive contains saved formal PASS results, including an FVU proof. The harness cuts out arithmetic/memory data and bounds shapes to small cases. These establish scoped control properties, not a proof of complete Qwen numerical correctness, all sizes, arbitration behavior, or memory visibility. They were inspected, not rerun for this review.

## 6. Reuse inventory

| Files | Decision | Conditions |
|---|---|---|
| `common/bpu_delay.sv`, `bpu_fifo.sv`, `bpu_lzc*.sv` | Reuse | Respect fixed latency, FIFO depth and parameter restrictions |
| `qmv/bpu_add_tree.sv`, `bpu_qmv_dot.sv` | Reuse first | Signed W4/W8 × A8; power-of-two lane count; preserve tests |
| `fp/bpu_fp32_add.sv`, `bpu_fp32_mul.sv` | Reuse | Fixed-latency, no ready port; nearest-even; canonical NaN; no exception flags/FMA |
| `fp/bpu_int2fp32.sv` | Reuse within its contract | Exact only for signed input widths 2–25; not a general arbitrary-width converter |
| `sfu/bpu_sfu*.sv` | Reuse as a set | Coefficient tables, generator, approximation semantics and accuracy tests travel together |
| `fvu/bpu_fvu_lane.sv`, `bpu_fvu_qround.sv`, sum/max trees | Reuse | FP32 semantics and inherited precision rules |
| `fvu/bpu_fvu_reduce.sv` | Adapt around output | Preserve canonical reduction; buffer completed writes because no write-ready port |
| `qmv/bpu_qmv_slice.sv`, `bpu_qmv_array.sv` | Reuse with adapters | Matching quantization/layout, guarded local buffer writes, tagged completion, bandwidth configuration |
| `common/bpu_sram_1r1w*.sv` | Reference/local-buffer wrappers | Not a replacement for shared arbitration; collision behavior and actual macros must be defined |
| `fvu/bpu_fvu.sv` | Substantial controller adaptation | Externalize/replace storage and fixed read/write timing assumptions |
| `top/bpu_compute_top.sv` | Keep as baseline; replace system integration | Serial dispatcher and FVU-owned scratchpad differ from Zeb's organization |
| `common/bpu_compute_pkg.sv` | Split/review constants | Separate accepted numerical rules from provisional opcode/interface assignments |
| `model/bpuref`, `tb`, `formal` | Keep alongside copied RTL | Tests are a major part of the reuse value; update the reference for intentional numerical changes |

## 7. Practical implementation order

1. Write a provisional numerical contract: W4/W8 × A8, group 64, BF16 scales, FP32 output/accumulation; label it provisional rather than presenting it as Zeb's settled decision.
2. Copy the dot product, integer adder tree, FP32 add/multiply/converter, delay/FIFO helpers, and their tests. Learn the signed-nibble decomposition and pipeline-tag alignment before editing them.
3. Bring up one QMV slice with local activation buffers and a mocked weight stream. Preserve its output credit accounting and accumulation-latency parameter check.
4. Add command-ID/completion wrapping and a memory adapter. Increase slices only after the memory stream organization is agreed.
5. Reuse FVU lanes/SFU/reductions behind an operand collector and buffered result writer compatible with arbitration. Keep the existing FVU as a numerical comparison target.
6. Integrate with the scoreboard/sequencer, then implement any agreed Q8.24, FP16, or GEMM extensions as separate tested modes.

Minimum team decisions before final integration: activation type; quantization format/group/scale rules; element versus byte addresses; bank mapping and masks; read latency/response ordering; write completion/visibility; command IDs/dependencies; and whether “matrix unit” must support efficient prefill/GEMM.

## Verification performed

- Python reference suite: 39 passed, 1 skipped. The skipped test is the Hugging Face cross-check because torch is unavailable.
- Generated SFU ROM agrees with its frozen coefficient tables.
- FP32 multiplier: 1,000,000 random mixed/corner-oriented vectors against host arithmetic, zero mismatches.
- FP32 adder: 1,000,000 such vectors, zero mismatches.
- Signed INT22-to-FP32: all 4,194,304 inputs, zero mismatches.
- Full supplied lint script: clean under Verilator 5.038. An initial older 5.020 attempt produced width warnings in the top and was incompatible with cocotb 2.1; that was a toolchain mismatch, not a demonstrated functional defect.
- Eighteen selected RTL regression cases passed under Verilator 5.038/cocotb 2.1: FP32 add/multiply at three pipeline masks each, SFU at three masks, tiny FVU, all four reduction configurations, Tiny Tapeout arithmetic wrapper, tiny QMV slice, tiny QMV array, and integrated tiny-Qwen decode. The integrated test checks the entire scratchpad and chosen token after each of three tokens. Two initial builds required correcting a compiler precompiled-header configuration; no RTL changes were made.

No Vivado timing/resource closure, ASIC place-and-route, fresh formal run, full-size real-checkpoint execution, quantization-quality evaluation, or shared-memory integration was performed. The archive's token-rate and area figures remain projections/pre-layout estimates, not measured performance of Zeb's architecture.
