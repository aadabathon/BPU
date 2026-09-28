"""Self-checks of the FVU reference semantics."""

import numpy as np
import pytest

from bpuref import fvu as F

f32 = lambda x: np.asarray(x, dtype=np.float32)


def test_tree_sum_pads_to_64_and_pairs_adjacent():
    x = f32([1e8, 1.0, -1e8, 1.0])
    # ((1e8 + 1) + (-1e8 + 1)) + zeros: adjacent pairs first.
    expect = (f32(1e8) + f32(1.0)) + (f32(-1e8) + f32(1.0))
    assert F.tree_sum(x[None])[0] == expect
    # A lone -0 is padded with +0 (P' >= 64), so the sum is +0.
    assert np.signbit(F.tree_sum(f32([-0.0])[None])[0]) == False  # noqa: E712


def test_order_max_nan_lowest_and_all_nan():
    assert F.order_max(f32([[np.nan, -np.inf, -3.0]]))[0] == -3.0
    assert np.isnan(F.order_max(f32([[np.nan, np.nan]]))[0])
    assert np.signbit(F.order_max(f32([[-0.0, 0.0]]))[0]) == False  # noqa: E712


def test_vvecmat_is_sequential_from_plus_zero():
    spm = np.zeros(320, np.float32)
    spm[0:64] = 1e8
    spm[64:128] = -1e8
    spm[128:192] = 1.0
    spm[256:259] = 1.0
    F.execute(F.FvuOp(F.VVECMAT, 3, 64, d=192, a=0, s=256, a_stride=64, s_stride=1), spm)
    assert spm[192] == np.float32(np.float32(np.float32(0 + 1e8) - 1e8) + 1.0)


def test_qclamp_rounding_and_limits():
    spm = np.zeros(128, np.float32)
    spm[:9] = [0.5, 1.5, 2.5, -0.5, 127.4, 127.5, -200.0, np.nan, -0.2]
    F.execute(F.FvuOp(F.VQCLAMP, 1, 9, d=64, a=0), spm)
    assert spm[64:73].tolist() == [0.0, 2.0, 2.0, 0.0, 127.0, 127.0, -127.0, 0.0, 0.0]
    out = spm[64:73]
    assert not np.signbit(out[out == 0]).any()                     # zeros are +0


@pytest.mark.parametrize("op", [
    F.FvuOp(F.VADD, 2, 64, d=64, a=0, b=0, a_stride=64, d_stride=64),       # d row 0 is a's row 1
    F.FvuOp(F.VMULS, 1, 8, d=0, a=0, s=3),                                  # scalar inside d
    F.FvuOp(F.RSUM, 2, 8, d=64, a=0, a_stride=64, d_stride=1),              # result lands in a row read later
    F.FvuOp(F.VADD, 1, 8, d=1, a=0, b=0),                                   # misaligned vector
])
def test_validate_rejects_hazards_and_misalignment(op):
    with pytest.raises(ValueError):
        F.validate(op, 1024)


def test_validate_accepts_exact_in_place():
    F.validate(F.FvuOp(F.VMULS, 4, 64, d=0, a=0, s=900, a_stride=64, d_stride=64), 1024)
    F.validate(F.FvuOp(F.VAXPY, 4, 64, d=0, a=512, b=0, s=900, b_stride=64, d_stride=64, s_stride=1), 1024)
