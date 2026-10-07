# Matrix unit: working notes

Last updated 2026-10-07. These are working notes for whoever builds the BPU matrix unit,
meaning the owner and any agent helping them. They collect what the source documents say,
what the repo already has, what is still open, and what to do next.

## 1. Scope

- The task is to **build the matrix unit**, following the compute meeting slides
  (`meeting_docs/Compute Team Meeting - 10-03-26.pdf`).
- The vector unit (a RISC-V core driving a SIMD array) is **on hold** until two things are
  settled: the ISA, and which RISC-V core gets dropped in.
- New RTL follows `NOTES_FOR_AGENTS_README/SystemVerilog Coding Standards.md`.

## 2. Source documents

| Document | What it is | Status |
|---|---|---|
| `meeting_docs/Compute Team Meeting - 10-03-26.pdf` | Matrix and vector unit microarchitecture (17 slides) | The spec for this work |
| `meeting_docs/zeb_simd_ISA.pdf` | Zeb's SIMD ISA: a 32-bit command word with 7 families | Proposal; its "Signatures" tab (the full op list) is missing from the PDF |
| `meeting_docs/zeb_block_diagram.png` | BPU block diagram (also slide 2) | GPT-drawn; "hand-wavy" |
| `NOTES_FOR_AGENTS_README/SystemVerilog Coding Standards.md` | Team coding rules, v1.0 | Applies to new code and to PRs |
| `NOTES_FOR_AGENTS_README/BPU_RTL_Reuse_Review.md` | Review of the old RTL against the block diagram | Answered in `compute/docs/review-response.md` |

## 3. Control path (how work reaches the units)

- **Top:** the Control SoC (a RISC-V with its own SRAM) sends descriptors to the **command
  sequencer**. The sequencer handles issue, dependencies and completion, using a scoreboard
  (slide 2).
- **Vector path:** the command sequencer sends commands to the **local RISC-V core**, which
  resolves them into custom SIMD instructions for the **SIMD array** (the owner's
  understanding, 2026-10-07; slides 10–12 say the same: "A local RISC-V core controls SIMD
  operations and SRAM transfers").
- **Matrix path, to confirm with the team:**
  - The slides show the sequencer dispatching the matrix unit directly: "The sequencer
    dispatches both units; they exchange data through SRAM" (slide 12).
  - Zeb's doc says a "command issuer … sends work to both the SIMD unit and the matrix unit".
  - Either way, the matrix unit takes commands from a control block and exchanges data with
    the vector side **through the shared SRAM**. The open question is only who sends its
    commands and in what format.
- **Repo today:** there is no RISC-V. The host or testbench sends tagged descriptors to
  `bpu_cmd_seq`, which dispatches directly to `bpu_fvu` (vector), `bpu_qmv_engine` (matrix)
  and the memory manager.

## 4. What the slides specify for the matrix unit

### Data types (slides 3–4)

| Operand | Format | Notes |
|---|---|---|
| Weights | signed INT4 or INT8 | dense projections |
| Activations | signed INT8 | dense projection inputs |
| Raw dot sums | INT32 | exact |
| State S, v, Δ | INT32 Q8.24 | value = code × 2⁻²⁴ |
| Q, K | signed INT16.Ff, f = 14–24 | f shared across 128 codes (block floating point), per token and head |
| Decay deficit | UINT16.Ff, f = 15–40 | per token and head |
| Beta | UINT16.F15 | |

- Block scaling **will be included**, but its implementation and parameters are undecided.
- Q8.24 × INT16.Ff → round-to-nearest-even of (product / 2ᶠ) → Q8.24.

### Datapath (slides 5–6, 8)

- The pipeline is: SRAM reads → stage and unpack → PE array → accumulate and stage → SRAM writes.
- Commands carry the operation, operand views, dimensions, dtypes, fraction bits and
  arithmetic profile.
- Dense operations use narrow integer products. **Each PE is a 2×2 INT4 multiply or one INT8
  multiply**, configured by a mode control.
- The recurrence uses **serial** wide fixed-point products: the PE keeps its operands and
  combines narrow partial products over several steps, with sign correction and no
  intermediate rounding.
- Profiles:

  | Profile | Operands | Product bits | Shift (round to nearest even) |
  |---|---|---|---|
  | BYTE_DOT | INT8 × INT4 or INT8 × INT8 | — | none (exact INT32 sums) |
  | QSTATE_S16 | Q8.24 × signed INT16.Ff | 48 | f |
  | QSTATE_U16 | Q8.24 × UINT16.Ff | 48 | f |
  | Q824_ORDERED | Q8.24 × Q8.24 | 64 | 24 |

  The serial profiles run: complete product → shift and round → saturate to INT32 →
  ordered accumulation.
- The matrix unit reports completion and range errors to the scoreboard.

### Commands (slides 7, 9)

- Dense, using BYTE_DOT:
  - `C ← M.GEMM(A, B)` (prefill)
  - `y ← M.GEMV(W, x)` (decode)
  - `C ← M.OUTER(a, b)` (rank-one product)
- Recurrence, using QSTATE_S16 and running after the vector unit's decay step:
  - `mem ← M.GEMV(Sᵀ, k, shift=kf)`
  - then the vector unit computes the error and Δ (slide 15)
  - `S ← M.OUTER(k, Δ, shift=kf, accumulate=true)`
  - `out ← M.GEMV(Sᵀ, q, shift=qf)`
- Each product is rounded separately, and reductions follow **increasing logical K**. State
  is committed after every token, and dependent tokens stay in order.

### Explicitly undecided (slides 5, 8, 17)

- array size, buffering and dataflow
- serial multiplier schedule and cycle counts
- block scaling
- final ISA and instruction format

## 5. What the repo has today (`compute/`, branch `claude/compute-roadmap`, PR #1)

### Hierarchy

```
bpu_core (rtl/top)                    u_mat at rtl/top/bpu_core.sv:201
└─ bpu_qmv_engine                     matrix unit: shared-SRAM client, load → issue → collect
   └─ bpu_qmv_array                   NSlice slices; row n → slice n % NSlice; argmax mode
      └─ bpu_qmv_slice ×NSlice        one stripe of rows, one weight beat per cycle
         ├─ bpu_qmv_dot               exact integer dot (one 5×8 multiplier per lane)
         │  └─ bpu_add_tree
         └─ bpu_int2fp32, bpu_fp32_mul, bpu_fp32_add, bpu_fifo, bpu_sram_1r1w, bpu_delay
```

### Math and configurations

- **Math today:** y[n] = Σ over groups g (in increasing order) of
  isum[n,g] × (weight_scale[n,g] × act_scale[g]).
  - isum is the exact integer sum over 64 elements.
  - Scales are bf16; accumulation is FP32.
  - Weights are W4 or W8; activations are A8.
  - The golden model is `compute/model/bpuref/qmv.py` (`qmv_ref`).
- **Configurations** (`bpuref/configs.py`):
  - fpga: 16 slices × 64 lanes, row interleave 4, K up to 6144
  - asic: 1 × 16, interleave 1, K up to 2048
  - tiny: 3 × 4, interleave 2, K up to 256
- **Throughput at fpga:** 1,024 INT4 MACs per cycle, which is 512 B/cycle. That matches
  16 × 256-bit HBM channels at 250 MHz, and caps decode at about 130 tok/s.

### Verification

- `pytest -k qmv_slice`, `-k qmv_array`, `-k core` (full tiny-Qwen decode, bit-exact)
- formal `qmv_dot.sby` (equivalence for all inputs) and `qmv_slice.sby` (unbounded control proofs)
- gate-level slice

### What carries over to the new design

- the exact integer dot and adder tree (essentially BYTE_DOT before scaling)
- slice/array striping
- the engine's SRAM-client logic: credits, done only after writes are accepted, error completion
- argmax for the LM head
- the test and formal structure
- the habit of writing the golden model first

### What changes

- INT32 output (scaling waits on the block-scaling decision)
- fracturable 2×2 INT4 / INT8 PE
- serial Q8.24 multiplier with shift, round and saturate
- M.OUTER (read-modify-write of S in SRAM)
- M.GEMM
- operands from SRAM instead of streamed weights (if decided)
- a new command format

### Not done by the matrix unit today

- Attention (q·K, p·V) and the DeltaNet recurrence run on the vector unit in FP32
  (`bpuref/qwen.py:461`).

## 6. Port interface

Draft diagrams, all in `compute/docs/`. The `.excalidraw` files are the editable source;
open them at excalidraw.com via ☰ → Open.

- **`matrix-unit-ports.excalidraw` / `.svg`:** today's ports on `bpu_qmv_engine`.
  - Clock/reset
  - Command: valid/ready, plus 8 payload fields
  - Completion: `done_o`, `err_o`
  - One shared-SRAM read port and one write port
  - Weight-stream request plus one weight stream and one scale stream per slice
  - Status

  It also has a proposed `mat_cmd_t` packed struct (op, profile, a/b types, shift,
  accumulate, argmax, m/n/k, a/b/c views, wid) and the open decisions.
- **`matrix-unit-inside.excalidraw` / `.svg`:** a decision-free bubble view of the inside.
  - Control: command intake → sequencer → completion
  - Data: fetch → stage → PE array → accumulate → post-process → write

Port protocol facts worth remembering:

- **Ready/valid:** a transfer happens on the edge where both are high. The producer holds
  valid and the payload until accepted.
- **SRAM read:** the response comes exactly RdLat cycles after the request is accepted, in
  order. Reserve buffer space before asking (credits).
- **SRAM write:** a write accepted in cycle t is visible to reads accepted after t.
- **`done_o`:** pulses only once the last result write has been accepted.

## 7. Zeb's SIMD ISA (summary)

- **Word layout** (32 bits, stored least-significant byte first):
  `FAMILY[31:29] SIG[28:22] D[21:18] A[17:13] B[12:8] C[7:3] PM[2:0]`
- **Families:**
  - 000 CONTROL (SETVL, PAND, PNOT)
  - 001 ALU (SIG = function × 8 + type)
  - 010 PRODUCT (addend/shift field)
  - 011 REDUCE (result to a scalar register)
  - 100 CAST (SIG = from × 8 + to; exponent field)
  - 101 SPECIAL (quotient, divisor, remainder)
  - 110 DATA (lane index)
  - 111 is an error
- **Registers:**
  - V0–V7: one value per lane
  - S0–S7: scalars
  - P0–P7: predicates
  - codes 16–19: constants (0, −0.0, 1, all-ones)
  - Cells are 64 bits; a result fills its low 16, 32 or 64 bits and zeroes the rest.
- **Lane rule:** a lane executes only if its index < VL and its predicate bit (PM) is true.
  Other lanes keep their values.
- **Issue model:** a separate command issuer replays stored command words (a lookup table),
  repeats sequences, and waits for earlier writes before issuing readers. The SIMD unit
  "does not fetch a program or branch by itself".
  - Separate memory logic moves data between memory and registers and fetches lookup-table
    entries.
- **Types:** F16, F32, I8, U8, I16, U16, I32, I64. U32/U64 can be stored and used in
  arithmetic, but are not CAST endpoints.
- **Errors:** an unknown FAMILY/SIG errors before any result is written. Unused fields must
  be zero.

### Differences from the slides

| Topic | Slides | Zeb |
|---|---|---|
| Loops | RISC-V branches (BEQ/BNE/BLT) | issuer replays sequences; no branches |
| Loads and stores | U.LOAD / U.STORE instructions | "separate memory logic" |
| Types | lists INT4 (pack/unpack) | no INT4 |
| Matrix commands | defined | not defined |
| Encodings | unspecified | fixed |

## 8. Open questions

**For the team:**

1. **Matrix command path.** Does the matrix unit get commands straight from the sequencer, or
   through the RISC-V / issuer? Who defines the matrix command format? Today's proposal is
   `mat_cmd_t`.
2. **Weight source.** Are weights read from the shared SRAM (slide 5) or streamed from HBM by
   the memory manager (repo today)? This decides whether the weight-stream ports exist.
3. **Block scaling.** Where do scales live, and does the matrix unit apply them or only return
   INT32?
4. **Address units.** The repo addresses SRAM in fp32 elements (4 bytes). INT4/8/16 data in
   SRAM needs byte addressing and byte write masks.
5. **Read bandwidth.** OUTER reads S, k and Δ, and GEMV reads A and x. How many SRAM read
   ports does the matrix unit get?
6. **Saturation.** Is it an error completion or a sticky flag?
7. **Array size, buffering, dataflow and serial-multiplier schedule.** All undecided in the
   slides.

**For Zeb:**

1. Please share the Signatures tab (the actual op list).
2. Who is the command issuer: the Control SoC RISC-V, the local RISC-V, or a hardware sequencer?
3. What does the issuer send the matrix unit?
4. Is PRODUCT's addend/shift field an immediate? If so, 5 bits reaches 31, but the decay
   deficit uses f up to 40.
5. Where does INT4 packing happen (there is no INT4 CAST type)?
6. Branches (slides) or replay (Zeb)?

## 9. Coding-standard gaps in today's matrix-unit RTL

These come from the gap check against the standard on 2026-10-04.

- **Fixed constants** should be ALL_CAPS: `QmvGroup`, `QmvIsumW`, `WfmtW4/W8`, and the `Mb*`
  descriptor offsets.
- **Command:** a flat 154-bit descriptor body; it should become a packed `mat_cmd_t`.
- **Several assignments per register in one `always_ff`.** Move these to the
  `always_comb _d` / `_q <= _d` pattern:
  - `bpu_qmv_engine.sv:255` (14 registers)
  - `bpu_qmv_slice.sv:138` (5), `:400`, `:424`
  - `bpu_qmv_array.sv:222` (5), `:174`
- **Waivers without a reason:** `lint_off SYNCASYNCNET` at `bpu_qmv_engine.sv:400` and
  `bpu_qmv_slice.sv:474`.
- **Assertions** are unlabelled immediate asserts inside `` `ifdef FORMAL ``. They need
  `Name_A:` labels.
- **Layout:**
  - several ports on one line (`bpu_qmv_engine.sv:43`)
  - declarations after logic (engine 34, slice 31, array 20)
  - one line over 100 characters in the slice
- **Cryptic names** (`rq_q`, `cw_q`, `sj_q`, `xstep_q`, `lhave_q`) and the `bpu_` /
  `qmv` prefixes.
- **No documented reset/quiesce behavior and no mid-operation reset test.** The standard
  requires both.

## 10. Suggested next steps (in order)

1. **Take section 8 to the team and to Zeb.** Questions 1–4 block the interface.
2. **Write the golden model first:** a bit-exact Python reference in `bpuref` for BYTE_DOT,
   QSTATE_S16/U16 and Q824_ORDERED, including round-to-nearest-even, saturation and K order.
   It becomes the contract the RTL has to match.
3. **Write the command package and an empty module.** The package holds `mat_cmd_t` and the
   op/profile enums. The module has the agreed ports and, until it's built out, completes
   every command with an "unsupported" error, as the standard requires of stubs.
4. **Build the BYTE_DOT path:** a fracturable PE (2×2 INT4 / INT8) with INT32 output, reusing
   the dot/adder-tree structure.
5. **Add the serial Q8.24 multiplier and the shift, round and saturate stage.**
6. **Add M.OUTER** (read-modify-write of S, with ordering against its own writes), then **M.GEMM**.

## 11. Glossary

| Term | Meaning |
|---|---|
| Decode / prefill | generating one token at a time / processing the whole prompt at once |
| GEMV / GEMM | matrix × vector / matrix × matrix |
| OUTER | outer product: every a[i]·b[j]; no sum; "accumulate" adds it into an existing matrix |
| MAC | one multiply-accumulate |
| PE | processing element: a multiplier plus an adder |
| Reuse | how many times each fetched number is used. GEMV is low (memory-bound); GEMM is high (compute-bound) |
| Q8.24 | INT32 read as code / 2²⁴ |
| INT16.Ff | INT16 read as code / 2ᶠ |
| RNE | round to nearest, ties to even |
| Saturation | clamp to the min/max value instead of wrapping around |
| Block floating point | a group of values sharing one exponent or fraction count (here 128 Q/K codes share f) |
| View | base address + strides describing where a matrix sits in memory |
| VL / predicate | how many SIMD lanes are active / per-lane on-off mask |
| Scoreboard | tracks which commands are pending, so dependent ones wait |
| DeltaNet recurrence | per head, a 128×128 state S. Per token: decay S, mem = Sᵀk, Δ = β(v − mem), S += k ⊗ Δ, out = Sᵀq |

## 12. Reading list

- **What's being multiplied:** 3Blue1Brown's transformer videos; Jay Alammar, *The
  Illustrated Transformer*.
- **Accelerators:**
  - H.T. Kung, *Why Systolic Architectures?* (1982)
  - Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit* (2017)
- **Dataflows:** Sze, Chen, Yang, Emer, *Efficient Processing of Deep Neural Networks: A
  Tutorial and Survey*.
- **Memory-bound vs compute-bound:** Williams, Waterman, Patterson, *Roofline* (2009).
- **Open-source designs:** Gemmini (Berkeley; a systolic array attached to a RISC-V core);
  NVDLA.
- **Integer quantization:** Jacob et al., *Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference* (2018).
- **The recurrence:** Yang et al., *Gated Delta Networks* (2024).

## 13. Repo state at the time of writing

- **Branch** `claude/compute-roadmap`, PR #1 open (https://github.com/aadabathon/BPU/pull/1).
  Nothing from 2026-10-04 to 2026-10-07 has been committed.
- **Untracked files:**
  - this note
  - the four diagram files in `compute/docs/`
  - `meeting_docs/zeb_simd_ISA.pdf`
  - `meeting_docs/Compute Team Meeting - 10-03-26.pdf`
  - `NOTES_FOR_AGENTS_README/SystemVerilog Coding Standards.md`
