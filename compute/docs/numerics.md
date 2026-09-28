# Compute numerics contract (v0 draft)

This is the arithmetic contract for every compute engine. The Python reference
model in `compute/model/bpuref` implements it, the RTL has to match that model
bit for bit, and every test compares the two.

Status: **v0 draft**. Anything marked *open* waits on a decision from another
team (see [Open decisions](#open-decisions)). Every change needs a matching
change to `bpuref` and to the tests.

## Principle: parameterize throughput, never numerics

The same RTL is built at very different sizes:

| Configuration | Purpose | Example |
|---|---|---|
| `fpga` | AWS F2 (VU47P + HBM) | 64 MAC lanes/slice, 4 interleaved rows, 250 MHz pipelining |
| `asic` | tapeout candidate | 16 lanes, 1 row, shallow pipelines |
| `tiny` | fast regressions, corner cases | 4 lanes, fully combinational units |

**For the same inputs, every configuration produces bit-identical outputs.**
One set of golden vectors therefore covers FPGA, silicon and simulation. To
keep that property:

1. Anything that affects results is a **spec constant**, not a module parameter:
   formats, group size, rounding, accumulation order and special-function
   algorithms. These live in `bpu_compute_pkg` and `bpuref`.
2. Module parameters only change **how fast** results are produced: lane counts,
   interleave factors, buffer depths and pipeline registers.
3. Datapaths assign lanes to *independent* outputs, or to integer sums (which
   are exact in any order). Floating-point accumulation for a given output is
   always sequential, in a fixed order.

## Number formats

| Name | Encoding | Used for |
|---|---|---|
| `fp32` | IEEE-754 binary32 | accumulators, vector-engine math, recurrent state |
| `bf16` | top 16 bits of binary32 | quantization scales; KV cache (*open*) |
| `w4` | signed two's-complement int4, range [-8, 7] | weights |
| `w8` | signed two's-complement int8, range [-128, 127] | sensitive weights (e.g. LM head, *open*); INT8 KV (*open*) |
| `a8` | signed int8, hardware range [-128, 127]; quantizer emits [-127, 127] | activations entering the QMV engine |

### FP32 arithmetic rules

* Round to nearest, ties to even (RNE). No other rounding modes.
* **Subnormals are fully supported** on both input and output (no flush-to-zero).
  That makes numpy `float32` a bit-exact reference with no emulation layer.
* Every NaN result is the canonical quiet NaN `0x7FC00000`. Payloads are not
  propagated. Tests only require "NaN in, NaN out".
* No exception flags and no traps.
* Signed zeros follow IEEE rules: `x + (-x) = +0`, `(-0) + (-0) = -0`, and the
  sign of a product is `sa ^ sb`.
* Add and multiply are separate operations, each rounded. There is no fused
  multiply-add in v0: a separately rounded mul + add can be vectorized in
  numpy exactly, and an FMA cannot.

### bf16 to fp32 conversion

Exact: `fp32 = bf16 << 16`. When the compute engines produce bf16 values
(activation scales, and KV if bf16), they round from fp32 to bf16 with RNE.

## QMV: quantized matrix-vector product

Computes `y[N] = W[N,K] · x[K]`, where `W` is `w4` or `w8` with one bf16 scale
per (row, group), and `x` is `a8` with one bf16 scale per group.

* Group size `G = 64` along K (`bpu_compute_pkg::QmvGroup`). K must be a
  multiple of 64. Every Qwen3.5-2B reduction length (2048, 6144) already is.
* **Exact semantics** for each row `n`, with `KG = K / 64` groups:

```
acc = +0.0                                     # fp32
for g in 0 .. KG-1:                            # strictly increasing g
    isum = sum_{k in group g} W[n,k] * x[k]    # exact integer, |isum| <= 2^20
    sc   = f32(sw[n,g]) * f32(sx[g])           # fp32 mul (exact unless it over/underflows)
    p    = f32(isum) * sc                      # int -> fp32 is exact (|isum| < 2^24), mul rounds RNE
    acc  = acc + p                             # fp32 add, RNE
y[n] = acc
```

* The integer group sum is exact, so the MAC lane count and adder-tree shape
  cannot change results.
* The fp32 part runs once per group per row, always in increasing `g`.
* Hardware may process several rows at once (row interleave `R`) and spread
  rows across slices; neither changes the per-row order.

### Why row interleave exists

The accumulation above is a loop-carried dependency through a pipelined fp32
adder. When a slice finishes one group per cycle (64 lanes, W4), updates to
the same row's accumulator must be at least `AddLatency + 1` cycles apart. The
slice therefore interleaves `R` rows: the weight stream visits group `g` for
rows `r = 0..R-1`, then group `g+1`, and so on. That meets the constraint with
no extra accumulators and no change to the arithmetic. Required:
`R * (64 / Lanes) >= AddLatency + 1`.

## Vector engine (FVU): planned rules

Not implemented yet. These rules are recorded now so they don't get decided
by accident later.

* All arithmetic in fp32, following the rules above.
* Element-wise ops are trivially identical across lane counts.
* **2-D ops** (`M·x`, `xᵀ·M`, rank-1 update): lanes map to output elements, and
  each output accumulates sequentially along the reduction index. The result
  is identical at any lane count.
* **Full-vector sums** (RMSNorm mean of squares, L2 norm, softmax denominators)
  use one canonical order: a pairwise binary tree over 64-element blocks
  (zero-padded), then sequential accumulation across blocks. Any power-of-two
  lane count ≤ 64 can reproduce this tree exactly with a small partial-sum stack.
* **Max/argmax** is order-independent except for ties: the smallest index wins.

## Special-function unit (SFU): planned rules

Four primitives cover all of Qwen3.5: `exp2`, `log2`, `rcp` (1/x) and `rsqrt`.
Every other function is a fixed composition of these plus fp32 add/mul:

| Function | Composition (fixed; the order is part of the spec) |
|---|---|
| `exp(x)` | `exp2(x * log2(e))` |
| `sigmoid(x)` | `rcp(1 + exp(-x))` |
| `silu(x)` | `x * sigmoid(x)` |
| `softplus(x)` | `x` if `x > 20`, else `log2(1 + exp(x)) * ln(2)` |
| RMSNorm scale | `rsqrt(mean(x²) + eps)` |

Each primitive is table-plus-polynomial or seed-plus-Newton, with its table
contents and coefficients defined *in `bpuref`*. That keeps the SFU bit-exact,
not just "within tolerance". Error bounds (in ulp, compared against float64)
are measured and published per function for ml-models to sign off.

## Open decisions

| Decision | Current assumption | Blocks | Owner |
|---|---|---|---|
| Activation precision (W4A8 vs W4A16) | W4A8 blockwise | QMV datapath (built for A8) | architecture + ml-models |
| Scale format (bf16 vs power-of-two E8M0) | bf16 | QMV scale path (one multiplier vs an exponent add) | architecture |
| Weight format per tensor (which tensors get w8) | w4 everywhere, w8 supported | nothing (QMV supports both) | ml-models |
| KV cache precision (bf16 vs int8) | bf16 | whether attention can run on QMV later | architecture |
| Activation quantization rule (scale rounding, clamp) | absmax/127 → bf16, round codes RNE, clamp ±127 (placeholder) | FVU quantize op | compute + ml-models |
| Norm weight convention (`w` vs `1 + w`) | confirm against the pinned `modeling_qwen3_5.py` | FVU RMSNorm sequence | ml-models |
