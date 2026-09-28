"""pytest entry point: builds each RTL top at each configuration and runs its cocotb tests.

    cd compute && pytest            # Verilator by default
    SIM=icarus pytest               # another simulator
    WAVES=1 pytest -k tiny          # dump waveforms into the build directory
    BPU_BUILD_ROOT=~/bpu_build pytest   # build outside the repo (e.g. off /mnt/c in WSL)
"""

import os
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from bpuref.configs import QMV_SLICE_CONFIGS

COMPUTE = Path(__file__).resolve().parents[1]
RTL = COMPUTE / "rtl"
SIM = os.environ.get("SIM", "verilator")
WAVES = os.environ.get("WAVES", "0") == "1"


def rtl_sources() -> list[Path]:
    lines = (RTL / "compute.f").read_text().splitlines()
    return [RTL / ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("//")]


def run(top: str, test_module: str, parameters: dict, tag: str, env: dict) -> None:
    build_root = Path(os.environ.get("BPU_BUILD_ROOT", COMPUTE / "sim_build")).expanduser()
    build_dir = build_root / f"{top}-{tag}"
    build_args = ["--x-assign", "unique", "--x-initial", "unique"] if SIM == "verilator" else []
    runner = get_runner(SIM)
    runner.build(
        sources=rtl_sources(),
        hdl_toplevel=top,
        parameters=parameters,
        build_args=build_args,
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        waves=WAVES,
    )
    runner.test(
        hdl_toplevel=top,
        test_module=test_module,
        build_dir=build_dir,
        test_dir=build_dir,
        extra_env={"BPU_SEED": os.environ.get("BPU_SEED", "1"), **env},
        timescale=("1ns", "1ps"),
        waves=WAVES,
    )


@pytest.mark.parametrize("pipe", [0b111, 0b000, 0b101])
@pytest.mark.parametrize("op", ["mul", "add"])
def test_fp32(op: str, pipe: int) -> None:
    run(f"bpu_fp32_{op}", "cocotb_fp32", {"PipeMask": f"3'b{pipe:03b}"}, f"p{pipe:03b}",
        {"BPU_FP_OP": op})


@pytest.mark.parametrize("cfg", list(QMV_SLICE_CONFIGS))
def test_qmv_slice(cfg: str) -> None:
    run("bpu_qmv_slice", "cocotb_qmv_slice", QMV_SLICE_CONFIGS[cfg].hdl_parameters(), cfg,
        {"BPU_QMV_CFG": cfg})
