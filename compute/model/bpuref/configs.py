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
