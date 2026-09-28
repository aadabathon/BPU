# Compute numerics contract (v1)

This is the arithmetic contract for every compute engine. The Python reference
(`compute/model/bpuref`) implements it, and the RTL matches that reference bit for
bit at every hardware configuration. Tests compare every bit (NaN payloads
excepted), never "within tolerance". Accuracy relative to the real model is a
separate, measured question; see [Accuracy](#accuracy-against-the-model).

## Principle: parameterize throughput, never numerics

| Configuration | Purpose | QMV | FVU |
|---|---|---|---|
| `fpga` | AWS F2 (VU47P + HBM) | 32 slices × 64 lanes, row interleave 4 | 16 lanes, deep pipelines |
| `asic` | tapeout candidate | 1 slice × 16 lanes | 2 lanes, shallow pipelines |
| `tiny` | corner-case regressions | 3 slices × 4 lanes | 4 lanes, every unit combinational |

**For the same inputs, every configuration produces bit-identical results.** Three rules make that hold:

1. Anything that affects results is a **spec constant**: formats, group size,
   rounding, accumulation order, special-function tables. Module parameters only
   change how fast results come out.
2. Lanes map to *independent* outputs or to *integer* sums, which are exact in any order.
3. Floating-point accumulation into one output is sequential in a fixed order, or
   follows one canonical tree that every lane count reproduces exactly.

## Number formats

| Name | Encoding | Used for |
|---|---|---|
| `fp32` | IEEE-754 binary32 | all vector math, accumulators, recurrent state, KV cache (v1) |
| `bf16` | top 16 bits of binary32 | quantization scales |
| `w4` / `w8` | signed int4 [-8, 7] / int8 [-128, 127] | weights |
| `a8` | signed int8; the quantizer emits [-127, 127] | activations entering QMV |

### FP32 arithmetic (multiply, add, convert)

* Round to nearest, ties to even. **Subnormals fully supported** (no flush), so
  numpy `float32` is the bit-exact reference.
* NaN results are always the canonical quiet NaN `0x7FC00000`.
* Signed zeros follow IEEE rules (`x + (-x) = +0`, `(-0) + (-0) = -0`).
* No fused multiply-add: every product and sum is rounded separately.

Verified by 100M random vectors per unit against the host FPU, all 2^32 bf16×bf16
products, and all 2^22 int22→fp32 conversions (`scripts/soak_fp32.sh`).

## QMV: quantized matrix-vector product

`y[n] = Σ over groups g (increasing), of f32(isum[n,g]) · (f32(sw[n,g]) · f32(sx[g]))`, where:

* `isum` is the exact integer dot product of the group's 64 weights and 64 activations;
* `acc` starts at +0.0, and every product and add is fp32 RNE.

Row interleave `R` hides the adder latency without changing the order:
`R · 64/Lanes ≥ AddLatency + 1`.

**Argmax** (greedy sampling on the LM head): the largest value under a total-order
key, with `-inf < … < -0 < +0 < … < +inf` and NaN ranked lowest; ties go to the
smallest row index. Rows beyond the logical row count (padding up to whole row
blocks) never win.

## FVU: fp32 vector unit

Ops are 2-D (rows × cols) with per-row strided operands; see
[engine-ops.md](engine-ops.md) for the op list.

* **Element-wise** ops are the fp32 rules above per element. `VSUB` is `a + (-b)`;
  `VAXPY` is `(s·a) + b`; `VMULADD` is `(a·b) + c`.
* **Sums** (`RSUM`, `RDOT`): pad the row with +0 to `P = max(64, next_pow2(cols))`,
  then add adjacent pairs level by level. The lane adder tree plus the pipelined merge
  reproduce exactly this tree at any lane count. `RDOT` rounds each product first.
* **Max** (`RMAX`, `RAMAX`): total-order key as above; an all-NaN row gives canonical NaN.
* **`VVECMAT`** `d[j] = Σ_r s[r]·a[r,j]`: sequential over rows, starting from +0.
* **`VSEL`** `a > s ? b : c`: IEEE greater-than (false when either is NaN; `+0 == -0`).
* **`VRBF16`**: round to bf16 (nearest even); NaN becomes canonical.
* **`VQCLAMP`**: round half to even to an integer, clamp to ±127, NaN → 0, and zero is +0.
* **Hazards**: an op's writes may alias its reads only exactly in place (same element
  at the same row/column position). `bpuref.fvu.validate` enforces this, and every
  compiled program is checked.

### Activation quantization (the rule, now fixed)

For each 64-element group of `x`:
1. `amax = max|x|` (`RAMAX`).
2. `s = bf16(amax · fp32(1/127))` (`VMULS`, `VRBF16`).
3. `inv = rcp(s)` (SFU).
4. `code = VQCLAMP(x · inv)`.

If `amax = 0`, then `s = 0` and `inv = +inf`; `0·inf = NaN`, and `VQCLAMP` turns NaN
into 0, so an all-zero group gets all-zero codes. The codes and bf16 scales are
exactly what the QMV engine consumes.

## SFU: special functions

Five functions: `rcp`, `rsqrt`, `exp2`, `exp`, `log2`. Each is integer range
reduction, a **128-segment quadratic from a frozen coefficient table**, and
fixed-point packing. The tables are the spec (`bpuref/sfu_tables_data.py`); the
RTL ROM is generated from them and CI checks that they match.

* Subnormal inputs read as zero (DAZ); results below the normal range flush to zero (FTZ).
  Neither occurs in Qwen3.5's uses.
* `exp` multiplies by log2(e) in fixed point (Q·24 × Q·30) before the exp2 core,
  so large arguments keep accuracy.
* `log2` computes `e' + t·h(t)` with `t ∈ [-0.25, 0.5)`, so results near 1 keep
  full *relative* precision on both sides.
* Compositions used by programs: `sigmoid = rcp(1 + exp(-x))`, `silu = x·sigmoid(x)`,
  `softplus = x > 20 ? x : ln2·log2(1 + exp(x))`.

Measured error against float64 (random inputs over each function's range, plus
dense sweeps):

| Function | Max error |
|---|---|
| rcp | 0.75 ulp |
| rsqrt | 0.58 ulp |
| exp2 | 1.19 ulp |
| exp | 1.44 ulp |
| log2 (incl. x → 1 from either side) | 0.52 ulp |

## Accuracy against the model

* The float64 reference of Qwen3.5 (`bpuref.qwen.Float64Qwen`) matches Hugging Face
  transformers 5.17 to about 2e-7 relative over multi-token decode. HF computes
  norms and the DeltaNet core in fp32, which sets that floor.
* The compiled BPU program (W4 weights, A8 activations, fp32 vector math, SFU)
  tracks the float64 model run with the same dequantized weights at logit cosine
  > 0.9997 on a random-weight tiny model.
* **Still open:** whole-model quality of the chosen quantization on the real 2B
  checkpoint (perplexity, downstream tasks). That is an ml-models measurement.

## Open decisions

| Decision | v1 choice | Notes |
|---|---|---|
| Activation precision | W4A8, 64-element groups | QMV also supports W8 weights |
| Scale format | bf16 | E8M0 would turn the scale multiply into an exponent add |
| Which tensors use W8 | none | per-tensor choice, carried in each QMV op |
| KV cache precision | fp32 in the SPM | bf16 storage = `VRBF16` + the top 16 bits; INT8 would let attention run on QMV (see performance.md) |
