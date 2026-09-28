"""Named hardware configurations of the compute engines.

All configurations must produce identical results (docs/numerics.md); they differ
only in throughput and pipelining. compute/scripts/lint.sh mirrors this table.
"""

from __future__ import annotations

from dataclasses import dataclass

from .qmv import GROUP


def _pipe3_latency(mask: int) -> int:
    return bin(mask & 0b111).count("1")


@dataclass(frozen=True)
class QmvSliceConfig:
    name: str
    lanes: int              # int4 MAC lanes per slice
    row_interleave: int     # rows in flight (R)
    max_k: int              # largest reduction length supported by the x buffer
    prod_reg: int = 1
    tree_reg_every: int = 2
    i2f_reg: int = 1
    mul_pipe: int = 0b111
    add_pipe: int = 0b111
    out_fifo_depth: int | None = None

    @property
    def add_latency(self) -> int:
        return _pipe3_latency(self.add_pipe)

    def validate(self) -> None:
        """Mirror of the slice's elaboration checks."""
        if self.lanes < 2 or self.lanes > GROUP or self.lanes & (self.lanes - 1):
            raise ValueError(f"{self.name}: lanes must be a power of two in [2, {GROUP}]")
        if self.max_k % GROUP or self.max_k < 2 * GROUP:
            raise ValueError(f"{self.name}: max_k must be a multiple of {GROUP} and >= {2 * GROUP}")
        if self.row_interleave * (GROUP // self.lanes) < self.add_latency + 1:
            raise ValueError(f"{self.name}: row_interleave too small for the accumulator latency")

    def hdl_parameters(self) -> dict[str, int | str]:
        """Parameter overrides for the RTL; sized literals keep lint width-clean."""
        params: dict[str, int | str] = {
            "Lanes": self.lanes,
            "RowInterleave": self.row_interleave,
            "MaxK": self.max_k,
            "ProdReg": f"1'b{self.prod_reg}",
            "TreeRegEvery": self.tree_reg_every,
            "I2fReg": f"1'b{self.i2f_reg}",
            "MulPipe": f"3'b{self.mul_pipe:03b}",
            "AddPipe": f"3'b{self.add_pipe:03b}",
        }
        if self.out_fifo_depth is not None:
            params["OutFifoDepth"] = self.out_fifo_depth
        return params


@dataclass(frozen=True)
class QmvArrayConfig:
    name: str
    nslice: int
    slice: QmvSliceConfig

    def hdl_parameters(self) -> dict[str, int | str]:
        return {"NSlice": self.nslice, **self.slice.hdl_parameters()}


@dataclass(frozen=True)
class FvuConfig:
    name: str
    vlanes: int
    mul_pipe: int = 0b111
    add_pipe: int = 0b111
    sfu_pipe: int = 0b11111
    red_fifo: int = 8
    sfu_lanes: int | None = None     # SFUs shared by the lanes (None: one per lane)
    spm_ports: int = 1               # SPM read ports (replicated storage): operand reads per cycle

    @property
    def nsfu(self) -> int:
        return self.sfu_lanes or self.vlanes

    def hdl_parameters(self, spm_elems: int) -> dict[str, int | str]:
        return {"VLanes": self.vlanes, "SpmWords": spm_elems // self.vlanes,
                "MulPipe": f"3'b{self.mul_pipe:03b}", "AddPipe": f"3'b{self.add_pipe:03b}",
                "SfuPipe": f"5'b{self.sfu_pipe:05b}", "RedFifoDepth": self.red_fifo,
                "SfuLanes": self.nsfu, "SpmReadPorts": self.spm_ports}


FVU_CONFIGS = {
    # red_fifo covers the reduction credit loop (lane pipeline + lane tree + 4) for full rate
    "fpga": FvuConfig("fpga", vlanes=16, sfu_lanes=4, red_fifo=24, spm_ports=3),
    "asic": FvuConfig("asic", vlanes=2, mul_pipe=0b010, add_pipe=0b010, sfu_pipe=0b01010, sfu_lanes=1),
    "tiny": FvuConfig("tiny", vlanes=4, mul_pipe=0b000, add_pipe=0b000, sfu_pipe=0b00000, red_fifo=2,
                      sfu_lanes=2, spm_ports=2),
    # F2 option: the widest FVU (projected 137 tok/s at 128 ctx vs 89 at 16 lanes).
    "wide": FvuConfig("wide", vlanes=64, sfu_lanes=16, red_fifo=32, spm_ports=3),
}


QMV_SLICE_CONFIGS = {
    # AWS F2: one 256-bit HBM beat (64 int4 codes = one group) per cycle at ~250 MHz.
    "fpga": QmvSliceConfig("fpga", lanes=64, row_interleave=4, max_k=6144),
    # Tapeout candidate: narrow datapath, shallow pipelines for a slow clock.
    "asic": QmvSliceConfig("asic", lanes=16, row_interleave=1, max_k=2048,
                           tree_reg_every=0, mul_pipe=0b010, add_pipe=0b010),
    # Regression corner: minimum lanes, every unit combinational.
    "tiny": QmvSliceConfig("tiny", lanes=4, row_interleave=2, max_k=256, prod_reg=0,
                           tree_reg_every=1, i2f_reg=0, mul_pipe=0b000, add_pipe=0b000),
}

for _cfg in QMV_SLICE_CONFIGS.values():
    _cfg.validate()


@dataclass(frozen=True)
class TopConfig:
    """bpu_compute_top: a QMV array and an FVU sharing the FVU scratchpad."""
    name: str
    nslice: int
    qmv: QmvSliceConfig
    fvu: FvuConfig

    def hdl_parameters(self, spm_elems: int) -> dict[str, int | str]:
        q, f = self.qmv, self.fvu
        return {"NSlice": self.nslice, "Lanes": q.lanes, "RowInterleave": q.row_interleave,
                "MaxK": q.max_k, "QProdReg": f"1'b{q.prod_reg}", "QTreeRegEvery": q.tree_reg_every,
                "QI2fReg": f"1'b{q.i2f_reg}", "QMulPipe": f"3'b{q.mul_pipe:03b}",
                "QAddPipe": f"3'b{q.add_pipe:03b}", "VLanes": f.vlanes,
                "SpmWords": spm_elems // f.vlanes, "FMulPipe": f"3'b{f.mul_pipe:03b}",
                "FAddPipe": f"3'b{f.add_pipe:03b}", "FSfuPipe": f"5'b{f.sfu_pipe:05b}",
                "RedFifoDepth": f.red_fifo, "FSfuLanes": f.nsfu, "FSpmPorts": f.spm_ports}

QMV_ARRAY_CONFIGS = {
    # F2: one slice per HBM pseudo-channel.
    "fpga": QmvArrayConfig("fpga", nslice=32, slice=QMV_SLICE_CONFIGS["fpga"]),
    # Tapeout: a single slice.
    "asic": QmvArrayConfig("asic", nslice=1, slice=QMV_SLICE_CONFIGS["asic"]),
    # Non-power-of-two slice count exercises the round-robin merge and argmax reduce.
    "tiny": QmvArrayConfig("tiny", nslice=3, slice=QMV_SLICE_CONFIGS["tiny"]),
}


TOP_CONFIGS = {
    "fpga": TopConfig("fpga", nslice=32, qmv=QMV_SLICE_CONFIGS["fpga"], fvu=FVU_CONFIGS["fpga"]),
    "asic": TopConfig("asic", nslice=1, qmv=QMV_SLICE_CONFIGS["asic"], fvu=FVU_CONFIGS["asic"]),
    "tiny": TopConfig("tiny", nslice=3, qmv=QMV_SLICE_CONFIGS["tiny"], fvu=FVU_CONFIGS["tiny"]),
}
