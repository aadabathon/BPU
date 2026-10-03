"""Shared SRAM addressing (mirror of rtl/mem/bpu_sram_shared.sv).

Words are `lanes` fp32 elements; a word address maps to (bank, row). The bank is
the address modulo nbanks, XOR-folded with the row bits when `hash` is set.
"""

from __future__ import annotations


def _log2(x: int) -> int:
    return x.bit_length() - 1


def bank_of(addr: int, nbanks: int, bank_words: int, hash: bool = True) -> int:
    bb, rb = _log2(nbanks), _log2(bank_words)
    if bb == 0:
        return 0
    b = addr & (nbanks - 1)
    row = (addr >> bb) & (bank_words - 1)
    if hash:
        for i in range(rb):
            b ^= ((row >> i) & 1) << (i % bb)
    return b


def in_range(addr: int, nbanks: int, bank_words: int) -> bool:
    return addr < nbanks * bank_words
