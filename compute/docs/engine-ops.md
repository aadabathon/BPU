# Compute engine interfaces and operations (v1)

The compute team's contract with everyone upstream: the ISA/command format
(architecture, rtl-control), tensor layouts (ml-compiler, rtl-memory) and the
golden model (ml-models). Every operation below matches `bpuref` bit for bit
([numerics.md](numerics.md)).

## Conventions (all engines)

* **Streams** use valid/ready with AXI-Stream rules: a source holds valid and data
  until the transfer happens, and ready may depend on valid, never the reverse.
* **Arithmetic pipelines never stall.** Engines admit work only when the results
  have a guaranteed place (credits), so latency can change with parameters.
* **Ports are flat vectors.** No SV interfaces or structs on module boundaries, so
  Verilator, Icarus, Yosys/slang, SymbiYosys and Vivado all accept the RTL.
* **Reset** is asynchronous and active low, on control state only.
* **Memories** only through `bpu_sram_1r1w` / `bpu_sram_1r1w_be` (macro wrappers on silicon).
* **Errors**: malformed commands are consumed without running and raise a sticky
  `err_cmd_o`; `status_clr_i` clears flags and counters.
* **Naming**: lowRISC style (`_i`/`_o`, `_q`, `u_`, `g_`), CamelCase parameters,
  and a `bpu_` prefix on every module.

## Hierarchy

```
bpu_compute_top           operation stream -> FVU or QMV; shares the FVU scratchpad
├── bpu_fvu               fp32 vector unit (owns the SPM)
│   ├── bpu_fvu_lane x V  fp32 mul -> add, bf16 round, quantize clamp, select (+ own SFU)
│   ├── bpu_sfu x SfuLanes shared SFU bank when SfuLanes < VLanes
│   └── bpu_fvu_reduce    lane adder tree + canonical merge stack; order-key max
└── bpu_qmv_array         NSlice slices, in-order merge, argmax
    └── bpu_qmv_slice xN  int dot (W4/W8 x A8) -> fp32 scale -> row-interleaved accumulate
```

## bpu_compute_top: the operation stream

Each operation is one handshake on `cmd_*` (`cmd_qmv_i` selects the kind), and
operations run strictly in order. This is where an ISA or control layer plugs in:
a RISC-V, a hardware list-walker or the host can all produce this stream.

**FVU operation.** Fields `op, func, half_log2, rows, cols` plus six operands
(`d a b c s t`), each with a base element address and a row stride. Semantics:
`bpuref.fvu`.

**QMV operation.** Fields `wid, wfmt, argmax, k, n, x, xs, y`:
1. Read `k` int8 codes (stored as exact fp32 integers at `x`, 64-aligned) and
   `k/64` bf16 scales (stored as fp32 at `xs`) from the SPM into the array's
   activation buffers. A gearbox moves min(VLanes, Lanes) codes per cycle.
2. Start the array (rows padded to whole row blocks; `n` real rows) and raise
   `wreq_*`: *weight tensor `wid`, `nrowblk` row blocks, `k/64` groups, format
   `wfmt`*. The memory side answers by streaming each slice's beats in layout L0
   (below), with padding rows zero.
3. Write results into the SPM: `y[i]` for `i < n`, or in argmax mode
   `y[0] = index` (as fp32) and `y[1] = value`.

**Host port** (`host_*`): word-wide SPM access with lane masks, available whenever
`cmd_ready_o` is high. It carries per-token inputs (embedding codes + scales, RoPE
tables) and lets the host read results.

## bpu_qmv_array / bpu_qmv_slice

* Global row `n` is computed by slice `n % NSlice` as its local row `n / NSlice`.
  On F2, slice `s` is fed by HBM pseudo-channel `s`.
* A command covers `NSlice · R · nrowblk` rows, of which the first `nrows` are real.
* **Layout L0**, per slice:
  `for rb: for g < K/64: for r < R: for chunk: beat(local row rb·R + r, group g, chunk)`.
  The weight scale for (row, g) is consumed with that row-group's last chunk.
  A W4 beat holds `Lanes` int4 codes (element i in bits [4i+3:4i]); a W8 beat holds
  `Lanes/2` int8 codes as little-endian bytes. On F2 one W4 beat is one 256-bit HBM
  transfer = exactly one 64-weight group. The executable definition is
  `bpuref.qmv.pack_array_streams`.
* **Throughput**: one beat per cycle per slice, as long as a row block outlasts the
  pipeline, which every Qwen3.5 shape does. The `full_throughput` test measures it.
* **Status**: sticky `err_cmd_o`, `flag_nan_o` and `flag_inf_o`; per-slice counters
  for beats, stalls on missing weights, stalls on missing scales, and stalls on
  output credits.

## bpu_fvu: operations

Operand addressing: a vector operand is `X[r,j] = spm[x + r·x_stride + j]` (64-aligned
base and stride); a row scalar is `S[r] = spm[s + r·s_stride]`; a group scalar is
`T[r,j] = spm[t + r·t_stride + j/64]`.

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

The sequencer issues one SPM read per cycle per operand an item needs (S at a row
start, T at a group start, then a, b, c), writes back with per-lane masks, and
spaces VVECMAT rows so no accumulator is read before the previous row wrote it.
With `SfuLanes < VLanes`, a VSFU word is issued as `VLanes/SfuLanes` sub-items,
each feeding one lane group to the shared SFU bank. Results are identical and
only the cycle count changes.

## Qwen3.5 decode on these operations

`bpuref.qwen.Compiler.step_program(pos)` emits the whole decode step. Verified
against Hugging Face (math) and run bit-exact on the RTL (tiny model). Per layer:

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
| Embedding | host writes the row's codes + scales; VMULG dequantizes |
