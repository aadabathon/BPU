# Interfaces and operations (v2: shared-SRAM architecture)

The compute team's contract with everyone else: the descriptor format (architecture,
rtl-control, the control SoC), the shared-SRAM protocol and the memory manager's
duties (rtl-memory), tensor layouts (ml-compiler) and the golden model (ml-models).
Every operation matches `bpuref` bit for bit ([numerics.md](numerics.md)).

The organization follows the BPU block diagram (`meeting_docs/zeb_block_diagram.png`):
a command sequencer and scoreboard issue work to a matrix unit, a vector unit and a
memory manager, which all exchange tensor data through one banked shared SRAM.
The diagram is a sketch, so where it leaves a choice open, the choice here is
labelled as provisional.

## Conventions

* **Streams and requests** use valid/ready with AXI-Stream rules: a source holds
  valid and payload until the transfer, and ready may depend on valid, never the
  reverse.
* **Arithmetic pipelines never stall.** Engines admit work only when its results
  have a place (credits), so latencies can change with parameters.
* **Completion means visibility.** An engine reports a command done only after its
  last write has been accepted by the shared SRAM, so any later command sees it.
* **Ports are flat vectors.** No SV interfaces or structs on module boundaries
  (Verilator, Icarus, Yosys/slang, SymbiYosys and Vivado all accept the RTL).
* **Reset** is asynchronous and active low, on control state only.
* **Memories** only through `bpu_sram_1r1w` / `bpu_sram_1r1w_be` (macro wrappers on silicon).
* **Naming**: lowRISC style (`_i`/`_o`, `_q`, `u_`, `g_`), CamelCase parameters,
  `bpu_` prefix. Numerics constants live in `bpu_compute_pkg`; provisional encodings
  (opcodes, function codes, descriptor layout) in `bpu_isa_pkg` / `bpuref/isa.py`.

## Hierarchy

```
bpu_core                      the compute + shared-SRAM half of the block diagram
├── bpu_cmd_seq               command sequencer: descriptors -> per-unit queues -> issue
│   └── bpu_scoreboard        pending / error bit per tag
├── bpu_fvu                   vector unit: fp32, shared-SRAM client (no private memory)
│   ├── bpu_fvu_lane x V      fp32 mul -> add, bf16 round, quantize clamp, select (+ own SFU)
│   ├── bpu_sfu x SfuLanes    shared SFU bank when SfuLanes < VLanes
│   └── bpu_fvu_reduce        lane adder tree + pipelined canonical merge; order-key max
├── bpu_qmv_engine            matrix unit: shared-SRAM client around the QMV array
│   └── bpu_qmv_array         NSlice slices, in-order merge, argmax
│       └── bpu_qmv_slice xN  int dot (W4/W8 x A8) -> fp32 scale -> row-interleaved accumulate
└── bpu_sram_shared           NBanks of 1R1W SRAM behind a request/response crossbar
```

Outside the core: the control SoC (writes descriptors, reads the scoreboard) and the
memory manager (owned by rtl-memory; see below).

## Descriptors and the sequencer (`bpu_cmd_seq`)

A descriptor is `{unit, tag, wait, body}`:

| Field | Meaning |
|---|---|
| `unit` | 0 vector, 1 matrix, 2 memory manager (3 reserved: completes at once with an error) |
| `tag` | names the command in the scoreboard (`NTags`, 16 by default). A descriptor waits at the door while an earlier command with the same tag is still pending, so tags can be reused freely |
| `wait` | bit mask of tags this command depends on. It is resolved at acceptance against the commands pending at that moment and then only shrinks as they complete, so a bit always means a specific earlier command, never a later reuse of the tag |
| `body` | the unit's operation (`bpu_isa_pkg::BodyW` = 432 bits); addresses and strides are 32-bit element addresses |

* Descriptors are accepted in program order into one queue per unit (`QDepth`).
  Each unit runs one command at a time, in order; the units run concurrently.
* Commands to the same unit are ordered by construction. Software only needs wait
  bits for dependencies on other units. `bpuref.sched` derives them from each op's
  read and write footprint (read after write, write after read, write after write).
* The scoreboard exposes `pending_o`, `err_o` (status of the last completion per tag)
  and `cpl_o` (completions this cycle). A completed command's results are visible to
  everything issued after it.

**Vector body** (LSB first): `op[5] func[3] half_log2[5] rows[16] cols[16]`, then
`d a b c s t ds as bs cs ss ts`, 32 bits each. **Matrix body**: `wid[16] wfmt argmax
k[16] n[24] x[32] xs[32] y[32]`. **Memory body**: opaque to the core; rtl-memory
defines it.

## Shared SRAM (`bpu_sram_shared`)

* `NBanks` banks of `BankWords` words; a word is `VLanes` fp32 elements (64 bytes at
  the fpga configuration: 32 banks × 256 KiB = 8 MiB).
* Clients have independent read and write ports: request `valid/ready` + word
  address (+ lane mask and data for writes); read response `rvalid/rdata` exactly
  `RdLat = 1 + OutReg` cycles after acceptance, in request order, no backpressure.
* **Visibility**: a write accepted in cycle t is seen by every read accepted after
  t. A read accepted in the same cycle as a write to the same word gets the old data.
* **Arbitration**: each bank takes one read and one write per cycle; competing
  requests are granted round-robin per bank (a waiting request is served within
  NRd − 1, or NWr − 1, other grants of that bank). The bank of a word is its address
  modulo NBanks XOR-folded with the row bits, so streams whose bases differ by a
  multiple of NBanks still spread out.
* **Out-of-range** requests are accepted at once, never touch a bank (reads return
  zero, writes are dropped) and raise `*_oob_o`; engines turn that into an error on
  the running command.
* Ports in `bpu_core`: reads `[vector × FRdPorts | matrix | memory manager]`, writes
  `[vector | matrix | memory manager]`.

## Memory manager (outside the core; rtl-memory)

The core expects the memory manager to:

1. Take its commands from the sequencer (`mm_cmd_valid_o/ready_i`, `mm_cmd_body_o`)
   and report each one done (`mm_done_i`, `mm_err_i`) once its SRAM writes are accepted.
2. Move data with its own SRAM read and write port (`mm_rd_*`, `mm_wr_*`): HBM ↔ SRAM
   and SRAM ↔ SRAM transfers, the per-token inputs, KV and state streaming.
3. Answer the matrix unit's weight requests: on `wreq` (*tensor `wid`, `nrowblk` row
   blocks, `k/64` groups, format `wfmt`*), stream each slice's beats in layout L0
   (below) on `w_*` and the bf16 weight scales on `ws_*`, padding rows zero. Weights
   are used once per token, so they can stream straight from HBM without landing
   in the SRAM; staging them through the SRAM also works but costs SRAM bandwidth.

## Vector unit (`bpu_fvu`)

Operand addressing: a vector operand is `X[r,j] = mem[x + r·x_stride + j]` (64-aligned
base and stride); a row scalar is `S[r] = mem[s + r·s_stride]`; a group scalar is
`T[r,j] = mem[t + r·t_stride + j/64]`.

| Op | Result | Reads |
|---|---|---|
| `VADD` `VSUB` `VMUL` | a+b, a-b, a·b | a b |
| `VMULS` `VADDS` | a·S, a+S | a s |
| `VAXPY` | (S·a) + b | a b s |
| `VMULADD` | (a·b) + c | a b c |
| `VMULG` | a·T (per-64-group scalar: dequantize, quantize) | a t |
| `VSFU` | rcp / rsqrt / exp2 / exp / log2 (a) | a |
| `VRBF16` `VCOPY` `VPERM` | bf16(a), a, a[j xor half] | a |
| `VQCLAMP` | clamp(rne(a), ±127) | a |
| `VSEL` | a > S ? b : c | a b c s |
| `RSUM` `RDOT` | canonical tree sum of a, of a·b → one element per row | a (b) |
| `RMAX` `RAMAX` | max of a, of \|a\| → one element per row | a |
| `VVECMAT` | d[j] = Σ_r S[r]·a[r,j], sequential over rows | a s |

How it runs:

* A walker allocates one **operand-collector slot** per word of the op (`NSlot`
  slots of fetch-ahead). Each of the `RdPorts` read ports fetches its class of
  operands (port 0: s, t, a; port 1: b; port 2: c; one port serves all) for the
  slots in order; per-port tag FIFOs route the in-order responses, so any SRAM
  latency or contention works.
* The oldest slot **launches** when its operands are in, the write buffer has room
  for its result (credits), and, for reductions, the reduction FIFO has a credit.
  Row scalars and group scalars are fetched once and reused by the following
  words. A VSFU word with a shared SFU bank launches once per lane group.
* Results go through a `WbDepth` write buffer to the write port; reductions pass
  through `bpu_fvu_reduce` (about one word per cycle).
* **VVECMAT**: when a row fits the `AccDepth`-word forwarding FIFO, row r+1 takes
  row r's results from the FIFO and only the last row is written. Longer rows read
  the accumulator back from the SRAM, and the read of word w of row r+1 waits until
  the write of word w of row r has been **accepted** (counted, not timed).
* **Errors**: malformed descriptors (unknown opcode or SFU function, misalignment,
  empty shape, bad VPERM geometry) complete immediately with `err_o`; an operand that
  reaches past the SRAM runs but completes with `err_o`. Aliasing between an op's
  reads and writes beyond exact in-place updates is a software rule
  (`bpuref.fvu.validate`), not checked in hardware.

## Matrix unit (`bpu_qmv_engine`, `bpu_qmv_array`, `bpu_qmv_slice`)

**Command** `wid, wfmt, argmax, k, n, x, xs, y`:

1. Load: read `k` int8 codes (exact fp32 integers at `x`, 64-aligned) and `k/64`
   bf16 scales (fp32 at `xs`) from the shared SRAM into the array's activation
   buffers, with credits for a small response buffer; a gearbox moves
   min(VLanes, Lanes) codes and one scale per cycle. The activation buffers stay in
   the engine: every weight row re-reads x, which the shared SRAM could not feed.
2. Start the array (rows padded to whole row blocks; `n` real rows) and raise
   `wreq_*` to the memory manager.
3. Write results to the SRAM, packed into words: `y[i]` for `i < n`, or in argmax
   mode `y[0] = index` (as fp32) and `y[1] = value`.

The array:

* Global row `n` is computed by slice `n % NSlice` as its local row `n / NSlice`;
  slice `s` takes the weight stream of HBM interface `s`.
* **Layout L0**, per slice:
  `for rb: for g < K/64: for r < R: for chunk: beat(local row rb·R + r, group g, chunk)`.
  The weight scale for (row, g) is consumed with that row-group's last chunk. A W4
  beat holds `Lanes` int4 codes (element i in bits [4i+3:4i]); a W8 beat holds
  `Lanes/2` int8 codes as little-endian bytes. On F2 one W4 beat is one 256-bit HBM
  transfer = one 64-weight group. Executable definition: `bpuref.qmv.pack_array_streams`.
* **Throughput**: one beat per cycle per slice whenever a row block outlasts the
  pipeline (every Qwen3.5 shape). The `full_throughput` test measures it.
* **Status**: `flag_nan_o`, `flag_inf_o` for the current command; per-slice counters
  in the array.

## Qwen3.5 decode on these operations

`bpuref.qwen.Compiler.step_program(pos)` emits the decode step and `bpuref.sched`
turns it into descriptors. The step is verified against Hugging Face (math) and runs
bit-exact on `bpu_core` (tiny model). Per layer:

| Block | Operations |
|---|---|
| RMSNorm (`1 + w`) | RDOT → VMULS 1/n → VADDS eps → rsqrt → VMULS → VMUL |
| Activation quantize | RAMAX → VMULS 1/127 → VRBF16 → rcp → VMULG → VQCLAMP |
| Projections | QMV (one quantize feeds several QMVs: qkv/z/b/a, q/k/v, gate/up) |
| Causal conv (k=4) + SiLU | VMUL + 3 × VMULADD over the 3-slot history, 3 × VCOPY, SiLU |
| q/k l2norm, q/√dk | RDOT (rows = heads) → VADDS → rsqrt → VMULS (per-row scalar) |
| β, α gates | sigmoid; exp → +1 → log2 → ·ln2 → VSEL(>20) = softplus; exp(A_log)·sp·(−1) → exp |
| Delta rule, per head | VMULS (S·α) → VVECMAT (Sᵀk) → VSUB, VMULS (δ) → VAXPY (S += k δᵀ) → VVECMAT (Sᵀq) |
| Gated RMSNorm, silu(z) | RDOT (rows = heads) … VMUL(w) → SiLU(z) → VMUL |
| Q/K per-head RMSNorm, partial RoPE | RMSNorm (rows = heads) → VPERM (half = rot/2) → VMUL(±sin) → VMULADD(cos) |
| KV append, attention per head | VCOPY → RDOT (K·q, rows = positions) → scale → RMAX → exp(s − m) → RSUM → rcp → VMULS → VVECMAT (pᵀV) |
| Output gate, MLP, residual | sigmoid → VMUL; SiLU(gate)·up; VADD |
| LM head | final RMSNorm → quantize → QMV argmax (tied embedding) |
| Embedding | the memory manager writes the row's codes + scales; VMULG dequantizes |
