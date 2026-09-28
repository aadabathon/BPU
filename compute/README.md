# BPU compute engines

RTL for the arithmetic of Qwen3.5-2B decode on the BPU:
- a **quantized matrix-vector engine** (QMV) for every projection and the LM head;
- an **fp32 vector unit** (FVU) with a **special-function unit** (SFU) for everything else;
- a **compute top** that runs the whole decode step as one operation stream.

One parameterized RTL base targets AWS F2 and a smaller tapeout, and every size
produces bit-identical results.

**Status:** a full decode of a tiny same-structure Qwen3.5 (DeltaNet + attention
layers, MLPs, tied LM head with greedy argmax) runs on `bpu_compute_top`
**bit-exact** against the Python reference at all three hardware configurations.
The reference itself matches Hugging Face's Qwen3.5 to ~2e-7.

| Doc | What it is |
|---|---|
| [docs/numerics.md](docs/numerics.md) | the arithmetic contract: formats, rounding, order, SFU accuracy, open decisions |
| [docs/engine-ops.md](docs/engine-ops.md) | interfaces, operation set, stream layouts, Qwen3.5 → operations |
| [docs/verification.md](docs/verification.md) | every claim and the command that reproduces it |
| [docs/performance.md](docs/performance.md) | measured cycles, calibrated model, 2B projection, area |
| [docs/roadmap.md](docs/roadmap.md) | milestones C0–C7, coverage, next steps, cross-team needs |

## Layout

```
rtl/                SystemVerilog (compile order: rtl/compute.f)
  common/           package, delay line, tree LZC, FIFO, SRAM wrappers
  fp/               IEEE fp32 multiplier, adder, int->fp32
  qmv/              adder tree, W4/W8 dot product, slice, array (+argmax)
  sfu/              special-function unit + generated coefficient ROM
  fvu/              vector unit: lanes, reduction unit, sequencer
  top/              bpu_compute_top
model/bpuref/       bit-exact reference: fp, qmv, sfu, fvu, qwen (model + compiler), perf, configs
model/tests/        reference self-tests (incl. Hugging Face cross-check when torch is present)
tb/                 cocotb testbenches, pytest runners (RTL and gate-level), soak harnesses
formal/             SymbiYosys proofs (QMV slice protocol, dot-product equivalence)
scripts/            lint, synthesis, soak
```

## Running

Toolchain: [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) (the 2026-09-27
build, pinned in CI), numpy, and a C++ compiler plus make for Verilator. On Windows, use WSL.

```bash
source ~/tools/oss-cad-suite/environment && python3 -m pip install numpy
sudo apt install -y build-essential        # Verilator needs g++ and make
```

From `compute/`:

```bash
pytest                                     # reference tests + every RTL test at every configuration
pytest -k compute_top                      # the full Qwen decode on the RTL
BPU_SEED=7 pytest -k "fvu or qmv"          # other random stimulus
WAVES=1 pytest -k "fvu and tiny"           # waveforms in the build directory
scripts/lint.sh                            # Verilator -Wall
scripts/synth_yosys.sh                     # Yosys synthesis + cell counts
scripts/soak_fp32.sh 100000000             # fp32 units vs host FPU
scripts/soak_sfu.sh                        # SFU vs reference, ~138M vectors
(cd formal && sby -f qmv_slice.sby)        # unbounded proofs
BPU_GATES=1 pytest tb/test_gates.py        # gate-level simulation (slow)
(cd model && python3 -m bpuref.hf_check)   # vs Hugging Face (needs torch + transformers)
(cd model && python3 -m bpuref.perf)       # cycle model + 2B projection
```

In WSL, `BPU_BUILD_ROOT=~/bpu_build` keeps simulator builds off `/mnt/c`, which is much faster.

## Rules of the road

1. **Parameters change speed, never results.** A change that alters any output bit
   is a numerics change: update `docs/numerics.md` and `bpuref` first.
2. **Every module is tested at every named configuration** (`bpuref/configs.py`,
   mirrored in `scripts/`).
3. **The reference is the spec.** RTL comparisons are bit-exact; tolerances belong
   only to model-accuracy checks.
4. **Portable RTL.** Flat-vector ports, no vendor primitives, memories through
   wrappers, `-Wall` clean, synthesizable by Yosys.
5. **Programs are checked.** `bpuref.fvu.validate` rejects misaligned operands and
   read/write aliasing; the compiler's output must pass it.
