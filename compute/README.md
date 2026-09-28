# BPU compute engines

RTL for the arithmetic of Qwen3.5-2B inference: the quantized matrix-vector
engine (QMV), and later the fp32 vector unit (FVU) and special-function unit (SFU).
The same parameterized RTL targets AWS F2 and a smaller tapeout, and every size
produces bit-identical results.

| Doc | What it is |
|---|---|
| [docs/numerics.md](docs/numerics.md) | The arithmetic contract: formats, rounding, accumulation order, open decisions |
| [docs/engine-ops.md](docs/engine-ops.md) | Engine interfaces, stream layouts, the op list, Qwen3.5 → op mapping |
| [docs/roadmap.md](docs/roadmap.md) | Milestones C0–C7, Qwen coverage tracker, tapeout track, cross-team needs |

## Status

* **Built and verified in simulation:** IEEE fp32 multiplier and adder, int→fp32,
  pipelined adder tree, W4/W8 dot product, `bpu_qmv_slice`.
* **Verified:** bit-exact against the Python reference at three hardware
  configurations (`fpga`, `asic`, `tiny`), with random backpressure, at Qwen
  reduction lengths, one beat per cycle, and with extreme values. Verilator
  `-Wall` lint clean; synthesizes with Yosys (slang frontend).
* **Next:** C2, the QMV array and argmax epilogue (see the roadmap).

## Layout

```
rtl/              SystemVerilog (compile order in rtl/compute.f)
  common/         package, delay line, LZC, FIFO, SRAM wrapper
  fp/             fp32 mul/add, int->fp32
  qmv/            adder tree, dot product, slice
model/bpuref/     bit-exact Python reference + named hardware configs
model/tests/      reference self-tests (no simulator)
tb/               cocotb testbenches + pytest runner (sweeps every config)
scripts/          lint.sh (Verilator), synth_yosys.sh (generic synthesis)
```

## Running

Toolchain: [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build)
(Verilator, Icarus, Yosys + slang, cocotb, pytest) plus numpy. On Windows, use WSL.

```bash
# one-time: unpack OSS CAD Suite to ~/tools, then add numpy to its Python
source ~/tools/oss-cad-suite/environment && python3 -m pip install numpy
# Verilator also needs a C++ compiler and make (Icarus does not)
sudo apt install -y build-essential
```

From `compute/`:

```bash
source ~/tools/oss-cad-suite/environment
pytest                              # reference self-tests + all RTL tests (Verilator)
SIM=icarus pytest                   # same tests on Icarus
pytest -k "qmv and tiny"            # one configuration
WAVES=1 pytest -k "qmv and tiny"    # with waveforms (in the build directory)
BPU_SEED=7 pytest                   # different random stimulus
scripts/lint.sh                     # Verilator -Wall, every top at every config
scripts/synth_yosys.sh asic fpga    # generic synthesis + cell counts
```

In WSL, `BPU_BUILD_ROOT=~/bpu_build` keeps simulator builds off `/mnt/c`, which
is much faster.

## Rules of the road

1. **Parameters change speed, never results.** A new parameter must not change
   any output bit. If it would, it's a numerics change: update
   `docs/numerics.md` and `bpuref` first.
2. **Every module is tested at every named configuration** (`bpuref/configs.py`,
   mirrored in `scripts/lint.sh` and `scripts/synth_yosys.sh`).
3. **The reference is the spec.** RTL is compared bit for bit, never "within
   tolerance". Accuracy tolerances belong to the ml-models evaluation, not RTL tests.
4. **Keep it portable.** Flat-vector ports, no vendor primitives, memories only
   through wrappers, lint-clean with `-Wall`, and it must synthesize in Yosys.
