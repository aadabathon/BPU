"""Turn a compiled program (FVU and QMV ops) into tagged command descriptors.

Dependencies come from the memory each op reads and writes, tracked per 64-element
block (a superset of the true element dependencies, so always safe): read after
write, write after write and write after read. Commands to the same unit execute
in order, so only dependencies on other units become wait bits, and per other
unit only the latest one matters. Tags rotate through ntags; the sequencer holds
a descriptor back while its tag is still pending, so a wait bit on a reused tag
still means "that command and everything before it on its unit has completed".
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import fvu as F
from .isa import UNIT_MAT, UNIT_VEC, Cmd, mat_body, vec_body
from .qmv import GROUP

BLOCK = 64


def _blocks(idx) -> np.ndarray:
    a = np.asarray(idx).ravel()
    return np.unique(a // BLOCK) if a.size else np.zeros(0, dtype=np.int64)


def _span(base: int, n: int) -> np.ndarray:
    return np.arange(base // BLOCK, (base + n - 1) // BLOCK + 1)


def footprint(op):
    """(unit, read blocks, write blocks) of one op."""
    if isinstance(op, F.FvuOp):
        reads = [_blocks(v) for v in F._read_sets(op).values()]
        rb = np.unique(np.concatenate(reads)) if reads else np.zeros(0, dtype=np.int64)
        return UNIT_VEC, rb, _blocks(F._write_set(op))
    reads = np.unique(np.concatenate([_span(op.x, op.k), _span(op.xs, op.k // GROUP)]))
    return UNIT_MAT, reads, _span(op.y, 2 if op.argmax else op.n)


def schedule(ops, ntags: int = 16, qw: dict | None = None, names: list | None = None) -> list[Cmd]:
    """Descriptors for `ops`. With qw (quantized weights) and names (the weight-id order
    the memory side uses), matrix bodies carry the weight id and format."""
    cmds: list[Cmd] = []
    last_writer: dict[int, int] = {}
    readers: dict[int, dict[int, int]] = defaultdict(dict)       # block -> {unit: latest reader}
    wid = {n: i for i, n in enumerate(names)} if names else {}
    for j, op in enumerate(ops):
        unit, rb, wb = footprint(op)
        deps: set[int] = set()
        for b in rb.tolist():
            if b in last_writer:
                deps.add(last_writer[b])
        for b in wb.tolist():
            if b in last_writer:
                deps.add(last_writer[b])
            deps.update(readers[b].values())
        latest: dict[int, int] = {}
        for i in deps:
            u = cmds[i].unit
            if u != unit:
                latest[u] = max(latest.get(u, -1), i)
        wait = 0
        for i in latest.values():
            wait |= 1 << cmds[i].tag
        if unit == UNIT_VEC:
            body = vec_body(op)
        else:
            w = qw[op.weight][2] if qw else 0
            body = mat_body(op, wid.get(op.weight, 0), w)
        cmds.append(Cmd(unit, j % ntags, wait, op, body, sorted(latest.values())))
        for b in rb.tolist():
            readers[b][unit] = j
        for b in wb.tolist():
            last_writer[b] = j
            readers[b] = {}
    return cmds
