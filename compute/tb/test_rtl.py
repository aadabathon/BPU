"""pytest entry point: builds each RTL top at each configuration and runs its cocotb tests.

    cd compute && pytest            # Verilator by default
    SIM=icarus pytest               # another simulator
    WAVES=1 pytest -k tiny          # dump waveforms into the build directory
    BPU_BUILD_ROOT=~/bpu_build pytest   # build outside the repo (e.g. off /mnt/c in WSL)
"""

import dataclasses
import os
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from bpuref.configs import (FVU_CONFIGS, QMV_ARRAY_CONFIGS, QMV_SLICE_CONFIGS, TOP_CONFIGS,
                             fvu_test_sram)

COMPUTE = Path(__file__).resolve().parents[1]
RTL = COMPUTE / "rtl"
TB_HDL = COMPUTE / "tb" / "hdl"
SIM = os.environ.get("SIM", "verilator")
WAVES = os.environ.get("WAVES", "0") == "1"


def rtl_sources() -> list[Path]:
    lines = (RTL / "compute.f").read_text().splitlines()
    return [RTL / ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("//")]


def run(top: str, test_module: str, parameters: dict, tag: str, env: dict,
        extra_sources: list[Path] = ()) -> None:
    build_root = Path(os.environ.get("BPU_BUILD_ROOT", COMPUTE / "sim_build")).expanduser()
    build_dir = build_root / f"{top}-{tag}"
    build_args = ["--x-assign", "unique", "--x-initial", "unique"] if SIM == "verilator" else []
    runner = get_runner(SIM)
    runner.build(
        sources=rtl_sources() + list(extra_sources),
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


@pytest.mark.parametrize("pipe", [0b11111, 0b00000, 0b01010])
def test_sfu(pipe: int) -> None:
    run("bpu_sfu", "cocotb_sfu", {"PipeMask": f"5'b{pipe:05b}"}, f"p{pipe:05b}", {})


# The named configurations, plus "nofwd": tiny with VVECMAT forwarding disabled, so
# every accumulator read goes through the SRAM and must wait for the previous row's
# write (short rows and competing writes make that race real).
FVU_TEST_CONFIGS = {**FVU_CONFIGS, "nofwd": dataclasses.replace(FVU_CONFIGS["tiny"], name="nofwd",
                                                               acc_depth=0, nslot=4)}


@pytest.mark.parametrize("cfg", list(FVU_TEST_CONFIGS))
def test_fvu(cfg: str) -> None:
    """The FVU on a shared SRAM with a competing host port (tb/hdl/bpu_fvu_sys.sv)."""
    spm = 2048
    f = FVU_TEST_CONFIGS[cfg]
    run("bpu_fvu_sys", "cocotb_fvu", {**f.hdl_parameters(), **fvu_test_sram(f, spm).hdl_parameters()},
        cfg, {"BPU_FVU_CFG": cfg, "BPU_FVU_VLANES": str(f.vlanes), "BPU_SPM_ELEMS": str(spm)},
        [TB_HDL / "bpu_fvu_sys.sv"])


# Shared SRAM shapes: (banks, words/bank, lanes, read ports, write ports, OutReg, Hash)
SRAM_CASES = {"b4_r3w2": (4, 64, 2, 3, 2, 0, 1), "b1_r2w2_reg": (1, 64, 1, 2, 2, 1, 1),
              "b8_r5w3_nohash": (8, 32, 4, 5, 3, 1, 0)}


@pytest.mark.parametrize("case", list(SRAM_CASES))
def test_sram_shared(case: str) -> None:
    nb, bw, lanes, nrd, nwr, oreg, hsh = SRAM_CASES[case]
    run("bpu_sram_shared", "cocotb_sram_shared",
        {"NBanks": nb, "BankWords": bw, "Lanes": lanes, "NRd": nrd, "NWr": nwr, "AW": 16,
         "OutReg": f"1'b{oreg}", "Hash": f"1'b{hsh}"}, case,
        {f"BPU_SRAM_{k}": str(v) for k, v in
         dict(NBANKS=nb, BANKWORDS=bw, LANES=lanes, NRD=nrd, NWR=nwr, OUTREG=oreg, HASH=hsh).items()})


# (lanes, AddPipe, FIFO depth): the three configurations' shapes plus a 1-lane edge case
FVU_REDUCE_CASES = {"l16_p111_f24": (16, 0b111, 24), "l2_p010_f8": (2, 0b010, 8),
                    "l4_p000_f2": (4, 0b000, 2), "l1_p101_f4": (1, 0b101, 4)}


@pytest.mark.parametrize("case", list(FVU_REDUCE_CASES))
def test_fvu_reduce(case: str) -> None:
    lanes, pipe, fifo = FVU_REDUCE_CASES[case]
    run("bpu_fvu_reduce", "cocotb_fvu_reduce",
        {"Lanes": lanes, "AW": 16, "AddPipe": f"3'b{pipe:03b}", "FifoDepth": fifo}, case,
        {"BPU_RED_LANES": str(lanes), "BPU_RED_FIFO": str(fifo), "BPU_RED_ADDPIPE": str(pipe)})


def test_tt_fp32() -> None:
    """The Tiny Tapeout learning-run wrapper, through its pins."""
    run("tt_um_bpu_fp32", "cocotb_tt_fp32", {}, "tt", {},
        [COMPUTE / "tapeout" / "tt" / "src" / "tt_um_bpu_fp32.sv"])


@pytest.mark.parametrize("cfg", list(QMV_SLICE_CONFIGS))
def test_qmv_slice(cfg: str) -> None:
    run("bpu_qmv_slice", "cocotb_qmv_slice", QMV_SLICE_CONFIGS[cfg].hdl_parameters(), cfg,
        {"BPU_QMV_CFG": cfg})


@pytest.mark.parametrize("cfg", list(QMV_ARRAY_CONFIGS))
def test_qmv_array(cfg: str) -> None:
    run("bpu_qmv_array", "cocotb_qmv_array", QMV_ARRAY_CONFIGS[cfg].hdl_parameters(), cfg,
        {"BPU_QMV_CFG": cfg})


@pytest.mark.parametrize("cfg", list(TOP_CONFIGS))
def test_core(cfg: str) -> None:
    """bpu_core: a full tiny-Qwen decode as tagged descriptors, bit-exact, plus
    cross-unit dependency checks with the memory unit."""
    run("bpu_core", "cocotb_core", TOP_CONFIGS[cfg].hdl_parameters(), cfg,
        {"BPU_TOP_CFG": cfg, "BPU_STEPS": os.environ.get("BPU_STEPS", "3"),
         "BPU_MODEL_CHECK": os.environ.get("BPU_MODEL_CHECK", "1")})
