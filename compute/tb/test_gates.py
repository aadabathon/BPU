"""Gate-level simulation: synthesize with Yosys to generic gates (memories mapped to
flops, everything flattened) and run the unchanged cocotb tests on the netlist.
Shows that synthesis preserves function, the check a tapeout needs.

Opt-in (slow): BPU_GATES=1 pytest tb/test_gates.py
"""

import os
import subprocess
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from test_rtl import COMPUTE, RTL, rtl_sources

pytestmark = pytest.mark.skipif(os.environ.get("BPU_GATES") != "1", reason="set BPU_GATES=1")

TARGETS = {
    # name: (top, parameter overrides, cocotb module, extra env)
    "sfu": ("bpu_sfu", {"PipeMask": "5'b01010"}, "cocotb_sfu", {"BPU_SFU_N": "20000"}),
    "qmv_slice_tiny": ("bpu_qmv_slice",
                       {"Lanes": 4, "RowInterleave": 2, "MaxK": 256, "ProdReg": "1'b0",
                        "TreeRegEvery": 1, "I2fReg": "1'b0", "MulPipe": "3'b000", "AddPipe": "3'b000"},
                       "cocotb_qmv_slice", {"BPU_QMV_CFG": "tiny"}),
    # 2 lanes sharing one SFU (the tapeout shape); combinational units keep the netlist small.
    "fvu_small": ("bpu_fvu",
                  {"VLanes": 2, "SfuLanes": 1, "SpmWords": 256, "MulPipe": "3'b000", "AddPipe": "3'b000",
                   "SfuPipe": "5'b00000", "RedFifoDepth": 2},
                  "cocotb_fvu", {"BPU_FVU_CFG": "gate", "BPU_SPM_ELEMS": "512"}),
    "fvu_reduce": ("bpu_fvu_reduce", {"Lanes": 2, "AW": 16, "AddPipe": "3'b010", "FifoDepth": 8},
                   "cocotb_fvu_reduce", {"BPU_RED_LANES": "2", "BPU_RED_FIFO": "8", "BPU_RED_ADDPIPE": "2"}),
}


def synthesize(name: str, top: str, params: dict, out: Path) -> Path:
    netlist = out / f"{name}.v"
    gparams = " ".join(f"-G{k}={v}" for k, v in params.items())
    files = " ".join(str(p) for p in rtl_sources())
    script = f"""
        read_slang -DSYNTHESIS --top {top} {gparams} {files}
        synth -top {top} -flatten
        memory_map
        opt -full
        techmap; opt -fast
        abc -g AND,NAND,OR,NOR,XOR,XNOR,MUX
        opt_clean
        rename -top {top}
        write_verilog -noattr {netlist}
    """
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["yosys", "-q", "-m", "slang", "-p", script], check=True, cwd=RTL)
    return netlist


@pytest.mark.parametrize("name", list(TARGETS))
def test_gate_level(name: str) -> None:
    top, params, module, env = TARGETS[name]
    root = Path(os.environ.get("BPU_BUILD_ROOT", COMPUTE / "sim_build")).expanduser() / "gates"
    netlist = synthesize(name, top, params, root)
    build_dir = root / name
    runner = get_runner("verilator")
    runner.build(sources=[netlist], hdl_toplevel=top, build_dir=build_dir,
                 build_args=["-Wno-fatal", "--x-assign", "unique", "--x-initial", "unique"],
                 timescale=("1ns", "1ps"))
    runner.test(hdl_toplevel=top, test_module=module, build_dir=build_dir, test_dir=build_dir,
                extra_env={"BPU_SEED": "1", **env}, timescale=("1ns", "1ps"))
