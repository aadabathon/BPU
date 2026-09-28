# Compute engine operations (v0 draft)

This is the compute team's contract with everyone upstream: the ISA and command
format (architecture / rtl-control), the tensor layouts (ml-compiler / rtl-memory)
and the golden model (ml-models). The ISA, whatever form it takes, lowers to the
operations below. Compute guarantees that each operation matches `bpuref` bit for
bit (see [numerics.md](numerics.md)).

Status legend: **built** = RTL plus tests passing; **planned** = specified here, no RTL yet.

## Interface conventions (all engines)

* **Streams** use valid/ready. A transfer happens on a cycle where both are high.
  A source must not drop `valid` or change `data` until the transfer happens
  (AXI-Stream rules). `ready` may depend on `valid`, never the reverse.
* **Arithmetic pipelines inside an engine are valid-only and never stall.** Each
  engine admits work only when it has reserved room for the results (credits), so
  latency can change with parameters without anything upstream noticing.
* **Ports are flat vectors** (`logic [N*W-1:0]`), with no SV interfaces or structs
  on module boundaries. That keeps Verilator, Icarus, Yosys/slang, Vivado and the
  OpenLane flow all happy.
* **Reset** is asynchronous and active low (`rst_ni`), and only on control state.
  Datapath flops have no reset, so they map to SRLs/DSP registers on the FPGA and
  smaller cells on the ASIC.
* **Memories** are instantiated only through technology wrappers
  (`bpu_sram_1r1w`). No vendor primitives in engine RTL.
* **Naming** follows the lowRISC style: `_i`/`_o` ports, `_q` flops, `u_` instances,
  `g_` generate blocks, CamelCase parameters, and a `bpu_` prefix on every module.

## QMV: quantized matrix-vector engine

### `bpu_qmv_slice` (built)

Computes `y = W·x` for the rows streamed through it; `W` is `w4` or `w8`
with bf16 group scales, and `x` is `a8` with bf16 group scales.

| Parameter | Meaning | fpga | asic | tiny |
|---|---|---|---|---|
| `Lanes` | int4 MAC lanes (a W8 beat carries Lanes/2 weights) | 64 | 16 | 4 |
| `RowInterleave` (R) | rows accumulated in parallel | 4 | 1 | 2 |
| `MaxK` | activation buffer capacity | 6144 | 2048 | 256 |
| `OutFifoDepth` | output FIFO / credit pool | 2R | 2R | 2R |
| `ProdReg`, `TreeRegEvery`, `I2fReg`, `MulPipe`, `AddPipe` | pipeline registers | deep | shallow | none |

**Operation sequence**

1. While `cmd_ready_o` is high, write the activations: `x_*` takes words of `Lanes`
   int8 codes (element `k` is in word `k / Lanes`, byte `k % Lanes`), and `xs_*`
   takes one bf16 scale per 64-element group.
2. Handshake a command: `{wfmt, ngroups = K/64, nrowblk = N/R}`. N must be a
   multiple of R; the compiler pads with zero rows.
3. Stream weight beats on `w_*` and weight scales on `ws_*` in **layout L0**:

   ```
   for rb in 0 .. N/R-1:
     for g in 0 .. K/64-1:
       for r in 0 .. R-1:                     # row = rb*R + r
         for c in 0 .. chunks-1:              # chunks = 64/Lanes (W4) or 128/Lanes (W8)
           beat(row, elements g*64 + c*per_beat ...)
         scale(row, g)                        # consumed with the last chunk
   ```

   Beat bit layout: W4 puts element `i` of the chunk in bits `[4i+3:4i]`. W8
   puts element `j` in byte `j`, little-endian.
4. `y_*` delivers one fp32 per row, in row order.

`bpuref.qmv.pack_weight_stream` / `pack_x_words` are the executable definition
of this layout. The memory team's weight packer must produce exactly these streams.
On F2, one W4 beat (Lanes = 64) is 256 bits, the width of one HBM pseudo-channel
transfer, which carries exactly one 64-weight quantization group.

**Throughput.** One beat per cycle, as long as the stream sources and the output
consumer keep up and a row block lasts longer than the pipeline. Roughly:
`(K/64) * R * chunks >= pipeline latency + R`, where the latency is about 16
cycles at the fpga configuration. Every Qwen3.5 projection (K ≥ 2048) is far
above this bound; raise `OutFifoDepth` for short-K work.
The `full_throughput` test asserts exactly one beat per cycle.

**Ordering hazard.** Row interleave exists so that updates to the same row's
fp32 accumulator are at least `AddLatency + 1` cycles apart. The elaboration check
enforces `R * 64/Lanes >= AddLatency + 1`.

### QMV array and epilogues (planned, milestone C2)

* `bpu_qmv_array`: `NSlice` slices. Rows are striped so slice `s` owns rows
  `≡ s (mod NSlice)`, the activation vector is broadcast to every slice, and the
  outputs merge in row order. On F2, slice `s` is fed by HBM pseudo-channel `s`.
* **Argmax epilogue** for the LM head: each slice keeps a running (max, index),
  and the array reduces them; the smallest index wins ties. This avoids shipping
  248,320 logits.
* **Accumulator-init epilogue** (`acc` starts at a given fp32 value, e.g. a
  residual or bias). *This changes the arithmetic*: `init + Σp` is not
  `(Σp) + init` in fp32. It needs a spec entry before use.
* **Activation double-buffering**: load the next `x` while the current op streams.

## FVU: fp32 vector unit (planned, milestones C4–C5)

Operands come from a local scratchpad or are streamed through the memory port.
Every operation below is exact to the fp32 rules in numerics.md, with
canonical reduction order.

| Group | Operations |
|---|---|
| Element-wise | `add`, `sub`, `mul`, `scale` (vector × scalar), `axpy` (a·x + y), `select`/`copy` |
| Conversion | `cvt.bf16→f32`, `cvt.f32→bf16` (RNE), `dequant` (int8 + bf16 group scale → f32), `quant` (f32 → a8 codes + bf16 group scales, the input to QMV) |
| Reduction | `sum`, `sumsq`, `max`, `argmax` (canonical order) |
| Special function (SFU) | `exp2`, `log2`, `rcp`, `rsqrt`, plus fixed compositions `exp`, `sigmoid`, `silu`, `softplus` |
| 2-D | `matvec` (y = M·x), `vecmat` (y = xᵀ·M), `rank1` (M = a·M + u·vᵀ) |
| Sequence helpers | `rope` (rotate pairs using a cos/sin table), `conv_step` (4-tap causal conv with a history shift) |

## Qwen3.5-2B decode → engine operations

Shapes are derived from the SiliconBadgers report's MAC counts and "187 linear
calls" (18×5 + 6×4 + 24×3 + 1). Tensor names and the norm conventions must be
confirmed against the pinned `modeling_qwen3_5.py` before RTL depends on them.

### Projections (all QMV, K is a multiple of 64 everywhere)

| Projection | K | N | Per token |
|---|---|---|---|
| DeltaNet `in_proj_qkv` | 2048 | 6144 | ×18 |
| DeltaNet `in_proj_z` | 2048 | 2048 | ×18 |
| DeltaNet `in_proj_b`, `in_proj_a` | 2048 | 16 + 16 (merge into one 32-row op) | ×18 |
| DeltaNet `out_proj` | 2048 | 2048 | ×18 |
| Attention `q_proj` (query + output gate) | 2048 | 4096 | ×6 |
| Attention `k_proj`, `v_proj` | 2048 | 512 each | ×6 |
| Attention `o_proj` | 2048 | 2048 | ×6 |
| MLP `gate_proj`, `up_proj` | 2048 | 6144 each | ×24 |
| MLP `down_proj` | 6144 | 2048 | ×24 |
| LM head (tied to the embedding) + argmax | 2048 | 248,320 | ×1 |

### Everything else (FVU unless noted)

| Qwen op | Engine sequence |
|---|---|
| Embedding lookup | QMV row fetch of the tied table, or `dequant` of one row |
| RMSNorm (2048; convention `w` vs `1+w` to confirm) | `sumsq` → `scale` by 1/2048 → `add` eps → `rsqrt` → `scale` → `mul` weight |
| Activation quantize before each projection | `quant` |
| Causal conv1d (k=4, 6144 ch) + SiLU | `conv_step` → `silu` |
| q/k L2 norm (16 heads × 128), q × 1/√128 | `sumsq` → `rsqrt` → `scale` |
| β = σ(b), α = exp(−exp(A_log)·softplus(a + dt_bias)) (16 values) | `add`, `softplus`, `mul`, `exp`, `sigmoid` |
| DeltaNet state step (16 heads, S 128×128 fp32) | `vecmat` (Sᵀk, Sᵀq) → δ = β(v − αSᵀk) → o = αSᵀq + δ(k·q) → `rank1` (S = αS + kδᵀ) |
| Gated RMSNorm: norm(o) ⊙ SiLU(z) | RMSNorm sequence → `silu` → `mul` |
| Q/K per-head RMSNorm (256), partial RoPE (64 of 256 dims) | RMSNorm sequence → `rope` |
| Decode attention (8 Q heads / 2 KV heads, GQA) | `matvec` (K·q) → online softmax (`max`, `exp`, `sum`, rescale) → `vecmat` (pᵀV) |
| Attention output gate | `sigmoid` → `mul` |
| MLP SiLU(gate) ⊙ up | `silu` → `mul` |
| Residual add | `add` |
| Greedy sampling | QMV argmax epilogue on the LM head |

## Draft command descriptor (for the ISA/control discussion)

Not frozen. It lists the fields compute needs, so the ISA team can fit them into
a real encoding.

| Field | QMV | FVU |
|---|---|---|
| `opcode` | `qmv` | one of the operations above |
| `wfmt` | w4 / w8 | – |
| shape | `ngroups` (K/64), `nrowblk` (N/R) | length, rows/cols for 2-D ops |
| operands | weight + scale stream ids, x buffer | scratchpad addresses or stream ids, scalar immediate (fp32) |
| result | output stream id | scratchpad address or stream id |
| epilogue | none / argmax (/ acc-init, once specified) | – |
| sync | completion tag (sequence number) | completion tag |
