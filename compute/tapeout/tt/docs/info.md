## How it works

This tile carries the two arithmetic primitives every datapath of the Badger
Processing Unit (a Qwen3.5 inference accelerator) is built from: an IEEE-754
binary32 adder and multiplier. The RTL is the unchanged `bpu_fp32_add` and
`bpu_fp32_mul`, made combinational: round to nearest even, full subnormal
support, canonical quiet NaN (0x7FC00000) for every NaN result.

The first silicon goal is to confirm that these units, and the open flow that
built them, produce the same bits as the reference model (numpy float32) on real
sky130 parts.

A host (the demo board's microcontroller) talks to three 32-bit registers A, B
and R through a byte-wide protocol. Every command acts on a rising edge of
STROBE (`uio[0]`), which is synchronized to the clock:

| CMD (`uio[3:1]`) | Action |
|---|---|
| 0 | `A.byte[IDX] <= ui_in` |
| 1 | `B.byte[IDX] <= ui_in` |
| 2 | `R <= A + B` |
| 3 | `R <= A - B` |
| 4 | `R <= A * B` |
| 5 | `R <= A` (I/O loopback) |

`IDX` (`uio[5:4]`) selects a byte, 0 being the least significant. `uo_out`
always shows `R.byte[IDX]`.

## How to test

For each command, set CMD, IDX and `ui_in`, then raise STROBE, hold it for a few
clocks, and lower it. To read R, step IDX through 0..3 and read `uo_out`.
Compare against numpy:

```python
import numpy as np
a, b = np.float32(1.5), np.float32(-2.25)
expected = {"add": a + b, "sub": a - b, "mul": a * b}   # any NaN reads as 0x7FC00000
```

Good first vectors: `1 + 2^-24` (a rounding tie: expect exactly 1.0), the
smallest subnormal times 0.5 (a tie to even: expect +0), and `inf - inf`
(expect 0x7FC00000).

The pin-level cocotb test in the BPU repository (`compute/tb/cocotb_tt_fp32.py`)
checks 976 operand pairs this way. The pairs are uniform random bits, typical
values, and every pair of 24 special values.

## External hardware

None: the demo board's microcontroller drives the pins.
