# BPU compute

RTL for the compute and shared-memory half of the BPU block diagram, aimed at
Qwen3.5-2B decode:
- a **vector unit** (FVU, fp32, with a special-function unit) for norms, activations,
  attention, the DeltaNet state update and everything else that is not a projection;
- a **matrix unit** (QMV array, W4/W8 × A8 with bf16 group scales) for every
  projection and the LM head;
- a **banked shared SRAM** that both units, and the memory manager, read and write
  through request/response ports;
- a **command sequencer and scoreboard** that issue tagged descriptors to the units,
  honour their dependencies and report completion.

One parameterized RTL base targets AWS F2 and a smaller tapeout, and every size
produces bit-identical results.

**Status:** a full decode of a tiny same-structure Qwen3.5 (DeltaNet + attention
layers, MLPs, tied LM head with greedy argmax) runs on `bpu_core` as 443 tagged
descriptors, **bit-exact** against the Python reference at all three hardware
configurations. The reference itself matches Hugging Face's Qwen3.5 to ~2e-7.

| Doc | What it is |
|---|---|
| [docs/engine-ops.md](docs/engine-ops.md) | interfaces: descriptors, shared-SRAM protocol, memory-manager duties, engine operations |
| [docs/numerics.md](docs/numerics.md) | the arithmetic contract: formats, rounding, order, SFU accuracy, open decisions |
| [docs/verification.md](docs/verification.md) | every claim and the command that reproduces it |
| [docs/performance.md](docs/performance.md) | measured cycles, calibrated model, 2B projection, area |
| [docs/roadmap.md](docs/roadmap.md) | milestones, next steps, cross-team needs |
| [docs/review-response.md](docs/review-response.md) | the RTL reuse review, point by point |

## Layout

```
rtl/                SystemVerilog (compile order: rtl/compute.f)
  common/           numerics package, provisional ISA package, delay line, LZC, FIFO, SRAM wrappers
  fp/               IEEE fp32 multiplier, adder, int->fp32
  qmv/              adder tree, W4/W8 dot product, slice, array (+argmax), matrix-unit engine
  sfu/              special-function unit + generated coefficient ROM
  fvu/              vector unit: lanes, reduction unit, operand collector / sequencer
  mem/              banked shared SRAM with arbitration
  ctrl/             command sequencer, scoreboard
  top/              bpu_core
model/bpuref/       bit-exact reference: fp, qmv, sfu, fvu, qwen (model + compiler), isa
                    (descriptors), sched (dependencies), sram, perf, configs
model/tests/        reference self-tests (incl. Hugging Face cross-check when torch is present)
tb/                 cocotb testbenches, pytest runners (RTL and gate-level), soak harnesses
tb/hdl/             testbench-only RTL (the FVU on a shared SRAM)
formal/             SymbiYosys proofs (shared SRAM, sequencer, FVU on the SRAM, QMV slice, dot product)
scripts/            lint, synthesis, soak
tapeout/tt/         Tiny Tapeout learning run: the fp32 units behind a byte-wide protocol
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
pytest -k core                             # the full Qwen decode on bpu_core
BPU_SEED=7 pytest -k "fvu or sram"         # other random stimulus
WAVES=1 pytest -k "fvu and tiny"           # waveforms in the build directory
scripts/lint.sh                            # Verilator -Wall, every configuration
scripts/synth_yosys.sh                     # Yosys synthesis + cell counts
SKY130_LIB=.../sky130_fd_sc_hd__tt_025C_1v80.lib scripts/synth_sky130.sh   # sky130 area/delay estimate
scripts/soak_fp32.sh 100000000             # fp32 units vs host FPU
scripts/soak_sfu.sh                        # SFU vs reference, ~138M vectors
(cd formal && for f in sram_shared cmd_seq qmv_slice qmv_dot fvu; do sby --sequential -f $f.sby; done)
BPU_GATES=1 pytest tb/test_gates.py        # gate-level simulation (~5 min)
(cd model && python3 -m bpuref.hf_check)   # vs Hugging Face (needs torch + transformers)
(cd model && python3 -m bpuref.perf)       # cycle model + 2B projection
```

In WSL, `BPU_BUILD_ROOT=~/bpu_build` keeps simulator builds off `/mnt/c`, which is much
faster. The unbounded proofs use the rIC3 portfolio, which takes up to ~16 GB: run
formal tasks one at a time (`--sequential`).

## Rules of the road

1. **Parameters change speed, never results.** A change that alters any output bit
   is a numerics change: update `docs/numerics.md` and `bpuref` first.
2. **Every module is tested at every named configuration** (`bpuref/configs.py`;
   lint and synthesis targets are generated from it).
3. **The reference is the spec.** RTL comparisons are bit-exact; tolerances belong
   only to model-accuracy checks.
4. **Portable RTL.** Flat-vector ports, no vendor primitives, memories through
   wrappers, `-Wall` clean, synthesizable by Yosys.
5. **Completion means visible.** An engine reports a command done only after its
   writes have been accepted by the shared SRAM.
6. **Programs are checked.** `bpuref.fvu.validate` rejects out-of-range or misaligned
   operands and read/write aliasing; `bpuref.sched` derives every cross-unit
   dependency from read/write footprints.
