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
class SramConfig:
    """bpu_sram_shared: nbanks banks of bank_words words; a word is the FVU's VLanes elements."""
    nbanks: int
    bank_words: int
    out_reg: int = 0          # registered read data: latency 2 instead of 1
    hash: int = 1             # XOR-fold row bits into the bank index

    @property
    def rd_lat(self) -> int:
        return 1 + self.out_reg

    def words(self) -> int:
        return self.nbanks * self.bank_words

    def hdl_parameters(self, prefix: str = "") -> dict[str, int | str]:
        return {prefix + "NBanks": self.nbanks, prefix + "BankWords": self.bank_words,
                prefix + "OutReg": f"1'b{self.out_reg}", prefix + "Hash": f"1'b{self.hash}"}


@dataclass(frozen=True)
class FvuConfig:
    name: str
    vlanes: int
    mul_pipe: int = 0b111
    add_pipe: int = 0b111
    sfu_pipe: int = 0b11111
    red_fifo: int = 8
    sfu_lanes: int | None = None     # SFUs shared by the lanes (None: one per lane)
    rd_ports: int = 1                # operand read ports on the shared SRAM
    nslot: int = 4                   # operand-collector slots (fetch-ahead)
    wb_depth: int = 8                # write buffer entries
    acc_depth: int = 8               # VVECMAT forwarding FIFO words (rows up to this many words)

    @property
    def nsfu(self) -> int:
        return self.sfu_lanes or self.vlanes

    @property
    def ltot(self) -> int:
        return max(_pipe3_latency(self.mul_pipe) + _pipe3_latency(self.add_pipe),
                   bin(self.sfu_pipe & 0b11111).count("1"))

    def hdl_parameters(self, prefix: str = "") -> dict[str, int | str]:
        p = prefix
        return {p + "VLanes": self.vlanes, p + "MulPipe": f"3'b{self.mul_pipe:03b}",
                p + "AddPipe": f"3'b{self.add_pipe:03b}", p + "SfuPipe": f"5'b{self.sfu_pipe:05b}",
                p + "RedFifoDepth": self.red_fifo, p + "SfuLanes": self.nsfu,
                p + "RdPorts": self.rd_ports, p + "NSlot": self.nslot, p + "WbDepth": self.wb_depth,
                p + "AccDepth": self.acc_depth}


FVU_CONFIGS = {
    # red_fifo covers the reduction credit loop and wb_depth the write-credit loop
    # (lane pipeline + a few cycles) for one item per cycle.
    "fpga": FvuConfig("fpga", vlanes=16, sfu_lanes=4, red_fifo=24, rd_ports=3, nslot=8, wb_depth=12,
                      acc_depth=16),
    "asic": FvuConfig("asic", vlanes=2, mul_pipe=0b010, add_pipe=0b010, sfu_pipe=0b01010, sfu_lanes=1),
    # Regression corner: combinational units, minimum buffering (every credit path stalls).
    "tiny": FvuConfig("tiny", vlanes=4, mul_pipe=0b000, add_pipe=0b000, sfu_pipe=0b00000, red_fifo=2,
                      sfu_lanes=2, rd_ports=2, nslot=2, wb_depth=2, acc_depth=4),
    # F2 option: the widest FVU.
    "wide": FvuConfig("wide", vlanes=64, sfu_lanes=16, red_fifo=32, rd_ports=3, nslot=8, wb_depth=12,
                      acc_depth=4),
}


def fvu_test_sram(f: FvuConfig, elems: int) -> SramConfig:
    """Shared SRAM for the FVU's own tests: 4 banks holding twice the test's elements
    (the upper half takes the host's competing writes)."""
    return SramConfig(nbanks=4, bank_words=2 * elems // (4 * f.vlanes), out_reg=int(f.name == "fpga"))


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
    """bpu_core: sequencer + scoreboard, vector unit, matrix unit (QMV array) and the
    shared SRAM they all work out of."""
    name: str
    nslice: int
    qmv: QmvSliceConfig
    fvu: FvuConfig
    sram: SramConfig
    ntags: int = 16
    qdepth: int = 2

    def elems(self) -> int:
        return self.sram.words() * self.fvu.vlanes

    def hdl_parameters(self) -> dict[str, int | str]:
        q = self.qmv
        return {"NBanks": self.sram.nbanks, "BankWords": self.sram.bank_words,
                "SOutReg": f"1'b{self.sram.out_reg}", "SHash": f"1'b{self.sram.hash}",
                "VLanes": self.fvu.vlanes, "RedFifoDepth": self.fvu.red_fifo,
                "FMulPipe": f"3'b{self.fvu.mul_pipe:03b}", "FAddPipe": f"3'b{self.fvu.add_pipe:03b}",
                "FSfuPipe": f"5'b{self.fvu.sfu_pipe:05b}", "FSfuLanes": self.fvu.nsfu,
                "FRdPorts": self.fvu.rd_ports, "FNSlot": self.fvu.nslot, "FWbDepth": self.fvu.wb_depth,
                "FAccDepth": self.fvu.acc_depth,
                "NSlice": self.nslice, "Lanes": q.lanes, "RowInterleave": q.row_interleave,
                "MaxK": q.max_k, "QProdReg": f"1'b{q.prod_reg}", "QTreeRegEvery": q.tree_reg_every,
                "QI2fReg": f"1'b{q.i2f_reg}", "QMulPipe": f"3'b{q.mul_pipe:03b}",
                "QAddPipe": f"3'b{q.add_pipe:03b}", "NTags": self.ntags, "QDepth": self.qdepth}

QMV_ARRAY_CONFIGS = {
    # F2: one slice per 256-bit HBM interface (16 in the block diagram; 32 if the shell
    # exposes every pseudo-channel - a parameter change).
    "fpga": QmvArrayConfig("fpga", nslice=16, slice=QMV_SLICE_CONFIGS["fpga"]),
    # Tapeout: a single slice.
    "asic": QmvArrayConfig("asic", nslice=1, slice=QMV_SLICE_CONFIGS["asic"]),
    # Non-power-of-two slice count exercises the round-robin merge and argmax reduce.
    "tiny": QmvArrayConfig("tiny", nslice=3, slice=QMV_SLICE_CONFIGS["tiny"]),
}


TOP_CONFIGS = {
    # 8 MiB shared SRAM: 32 banks x 256 KiB, 64-byte words (16 fp32 lanes).
    "fpga": TopConfig("fpga", nslice=16, qmv=QMV_SLICE_CONFIGS["fpga"], fvu=FVU_CONFIGS["fpga"],
                      sram=SramConfig(nbanks=32, bank_words=4096, out_reg=1), qdepth=4),
    # Tapeout candidate: 512 KiB in 4 banks.
    "asic": TopConfig("asic", nslice=1, qmv=QMV_SLICE_CONFIGS["asic"], fvu=FVU_CONFIGS["asic"],
                      sram=SramConfig(nbanks=4, bank_words=16384)),
    # Regression corner: two banks (constant conflicts), minimum buffering.
    "tiny": TopConfig("tiny", nslice=3, qmv=QMV_SLICE_CONFIGS["tiny"], fvu=FVU_CONFIGS["tiny"],
                      sram=SramConfig(nbanks=2, bank_words=16384), ntags=8, qdepth=1),
}


def _gargs(params: dict) -> str:
    return " ".join(f'"-G{k}={v}"' for k, v in params.items())


def lint_targets() -> list[str]:
    """One "top -Gparam=value ..." line per named configuration (scripts/lint.sh)."""
    lines = []
    for c in QMV_SLICE_CONFIGS.values():
        lines.append(f"bpu_qmv_slice {_gargs(c.hdl_parameters())}")
    for c in QMV_ARRAY_CONFIGS.values():
        lines.append(f"bpu_qmv_array {_gargs(c.hdl_parameters())}")
    for c in FVU_CONFIGS.values():
        lines.append(f"bpu_fvu {_gargs(c.hdl_parameters())}")
    lines.append(f"bpu_fvu {_gargs({**FVU_CONFIGS['tiny'].hdl_parameters(), 'AccDepth': 0})}")
    for c in TOP_CONFIGS.values():
        lines.append(f"bpu_core {_gargs(c.hdl_parameters())}")
    return lines


def hdl_params(kind: str, name: str) -> dict:
    """Parameters of one named configuration of a block: kind is slice, array, fvu or core."""
    table = {"slice": QMV_SLICE_CONFIGS, "array": QMV_ARRAY_CONFIGS, "fvu": FVU_CONFIGS, "core": TOP_CONFIGS}
    return table[kind][name].hdl_parameters()


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--lint"]:
        print("\n".join(lint_targets()))
    elif sys.argv[1:2] == ["--params"]:           # --params <kind> <name>: -G arguments
        print(" ".join(f"-G{k}={v}" for k, v in hdl_params(sys.argv[2], sys.argv[3]).items()))
    else:
        sys.exit("usage: python -m bpuref.configs --lint | --params <slice|array|fvu|core> <name>")
