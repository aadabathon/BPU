"""Self-checks of the SFU reference (no simulator needed)."""

import numpy as np
import pytest

from bpuref import gen_sfu_rom, sfu

f32 = lambda x: np.asarray(x, dtype=np.float32).view(np.uint32)
QNAN, INF, NINF = 0x7FC00000, 0x7F800000, 0xFF800000


def test_generated_rom_matches_frozen_tables():
    assert gen_sfu_rom.main(["--check"]) == 0


@pytest.mark.parametrize("func,bound", [(sfu.RCP, 1.0), (sfu.RSQRT, 1.0), (sfu.EXP2, 1.5),
                                        (sfu.EXP, 2.0), (sfu.LOG2, 1.0)])
def test_accuracy_bounds(func, bound):
    rng = np.random.default_rng(func)
    if func in (sfu.EXP, sfu.EXP2):
        a = f32(rng.uniform(-80, 80, 200_000))
    elif func == sfu.RCP:
        a = f32(rng.uniform(-1e4, 1e4, 200_000))
    else:
        a = f32(np.exp(rng.uniform(-60, 60, 200_000)))
    err, _ = sfu.ulp_error(func, a)
    assert err.max() <= bound, f"{sfu.FUNC_NAMES[func]}: {err.max():.3f} ulp"


def test_exact_values():
    assert sfu.sfu(sfu.RCP, f32([1.0, 2.0, -0.5]))[()].tolist() == f32([1.0, 0.5, -2.0]).tolist()
    assert sfu.sfu(sfu.RSQRT, f32([1.0, 4.0, 0.25]))[()].tolist() == f32([1.0, 0.5, 2.0]).tolist()
    assert sfu.sfu(sfu.EXP2, f32([0.0, 1.0, -3.0]))[()].tolist() == f32([1.0, 2.0, 0.125]).tolist()
    assert sfu.sfu(sfu.EXP, f32([0.0]))[0] == f32(1.0)
    assert sfu.sfu(sfu.LOG2, f32([1.0, 2.0, 0.5, 1024.0]))[()].tolist() == f32([0.0, 1.0, -1.0, 10.0]).tolist()


@pytest.mark.parametrize("func,x,y", [
    (sfu.RCP, [0.0, -0.0, np.inf, -np.inf, 1e-40], [INF, NINF, 0, 0x80000000, INF]),       # DAZ
    (sfu.RCP, [3e38], [0]),                                                                # FTZ
    (sfu.RSQRT, [0.0, -0.0, np.inf, -1.0, -np.inf], [INF, NINF, 0, QNAN, QNAN]),
    (sfu.EXP, [np.inf, -np.inf, 89.0, -104.0, 1000.0, -1000.0], [INF, 0, INF, 0, INF, 0]),
    (sfu.EXP2, [128.0, -127.0, -126.0], [INF, 0, 0x00800000]),
    (sfu.LOG2, [0.0, -0.0, np.inf, -1.0, 1e-40], [NINF, NINF, INF, QNAN, NINF]),
])
def test_special_cases(func, x, y):
    assert sfu.sfu(func, f32(x)).tolist() == y


@pytest.mark.parametrize("func", list(sfu.FUNC_NAMES))
def test_nan_in_nan_out(func):
    assert np.all(sfu.sfu(func, np.array([0x7FC00000, 0x7F800001, 0xFFFFFFFF], np.uint32)) == QNAN)
