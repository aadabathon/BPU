"""Stream SFU soak vectors (func, a, expected y) as little-endian uint32 triples on stdout.

Exhaustive over every mantissa in each function's critical binades, plus random
inputs. Reference results come from bpuref.sfu (the spec); sfu_soak.cpp compares.
"""

import sys

import numpy as np

from bpuref import sfu

CHUNK = 1 << 20


def binade(exp_field, sign=0):
    m = np.arange(1 << 23, dtype=np.uint32)
    return (np.uint32(sign) << 31) | (np.uint32(exp_field) << 23) | m


def regions(rng):
    yield sfu.RCP, binade(127)
    yield sfu.RCP, binade(127, sign=1)
    yield sfu.RSQRT, binade(127)                  # even exponent
    yield sfu.RSQRT, binade(128)                  # odd exponent
    yield sfu.LOG2, binade(126)                   # [0.5, 1): the t < 0 branch near 1
    yield sfu.LOG2, binade(127)                   # [1, 2): both branches
    for func in (sfu.EXP, sfu.EXP2):
        yield func, binade(127)                   # x in [1, 2)
        yield func, binade(127, sign=1)
        yield func, binade(133, sign=1)           # x in (-128, -64]: underflow edge
        yield func, binade(133)                   # overflow edge for exp
    for func in sfu.FUNC_NAMES:
        yield func, rng.integers(0, 1 << 32, 4 * CHUNK, dtype=np.uint64).astype(np.uint32)


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    out = sys.stdout.buffer
    total = 0
    for func, a in regions(rng):
        for i in range(0, len(a), CHUNK):
            chunk = a[i:i + CHUNK]
            rec = np.empty((len(chunk), 3), dtype="<u4")
            rec[:, 0] = func
            rec[:, 1] = chunk
            rec[:, 2] = sfu.sfu(func, chunk)
            out.write(rec.tobytes())
            total += len(chunk)
    print(f"generated {total} vectors", file=sys.stderr)


if __name__ == "__main__":
    main()
