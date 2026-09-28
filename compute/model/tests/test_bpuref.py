"""Self-checks of the reference model (no simulator needed)."""

import numpy as np
import pytest

from bpuref.fp import bf16_to_f32, f32_bits, f32_to_bf16
from bpuref.qmv import GROUP, W4, W8, pack_weight_stream, pack_x_words, qmv_ref
from bpuref.quant import quantize_activations, quantize_weights_rtn


def test_bf16_round_to_nearest_even():
    cases = {
        0x3F800000: 0x3F80,   # exact
        0x3F808000: 0x3F80,   # tie, even stays
        0x3F818000: 0x3F82,   # tie, odd rounds up
        0x3F808001: 0x3F81,   # above tie
        0x7F7FFFFF: 0x7F80,   # rounds up to inf
        0x7FC00001: 0x7FC0,   # NaN -> canonical
    }
    x = np.array(list(cases), dtype=np.uint32).view(np.float32)
    assert list(f32_to_bf16(x)) == list(cases.values())
    assert f32_bits(bf16_to_f32(np.uint16(0x3F80)))[()] == 0x3F800000


def test_qmv_ref_is_exact_for_small_integers():
    rng = np.random.default_rng(0)
    w = rng.integers(-8, 8, (5, 3 * GROUP))
    x = rng.integers(-127, 128, 3 * GROUP)
    one = f32_to_bf16(np.float32(1.0))
    y = qmv_ref(w, np.full((5, 3), one), x, np.full(3, one))
    assert np.array_equal(y, (w @ x).astype(np.float32))   # all partial sums < 2^24


@pytest.mark.parametrize("wfmt", [W4, W8])
def test_qmv_ref_tracks_dequantized_matmul(wfmt):
    rng = np.random.default_rng(1)
    k = 2048
    w, ws = quantize_weights_rtn(rng.normal(0, 0.02, (16, k)).astype(np.float32), wfmt)
    x, xs = quantize_activations(rng.normal(0, 1, k).astype(np.float32))
    wd = w.reshape(16, -1, GROUP) * bf16_to_f32(ws)[:, :, None].astype(np.float64)
    xd = x.reshape(-1, GROUP) * bf16_to_f32(xs)[:, None].astype(np.float64)
    exact = np.einsum("ngk,gk->n", wd, xd)
    y = qmv_ref(w, ws, x, xs).astype(np.float64)
    assert np.max(np.abs(y - exact)) <= 1e-5 * np.max(np.abs(exact))


def _unpack(word, bits, count):
    vals = [(word >> (bits * i)) & ((1 << bits) - 1) for i in range(count)]
    return [v - (1 << bits) if v >> (bits - 1) else v for v in vals]


@pytest.mark.parametrize("lanes,r", [(64, 4), (16, 1), (4, 2)])
@pytest.mark.parametrize("wfmt", [W4, W8])
def test_layout_l0_round_trip(lanes, r, wfmt):
    rng = np.random.default_rng(2)
    n, k = 2 * r, 2 * GROUP
    lo, hi = (-8, 7) if wfmt == W4 else (-128, 127)
    w = rng.integers(lo, hi + 1, (n, k))
    ws = rng.integers(0, 1 << 16, (n, k // GROUP)).astype(np.uint16)
    beats, scales = pack_weight_stream(w, ws, lanes, r, wfmt)

    per_beat = lanes if wfmt == W4 else lanes // 2
    bits = 4 if wfmt == W4 else 8
    chunks = GROUP // per_beat
    rebuilt = np.zeros_like(w)
    it, sit = iter(beats), iter(scales)
    for rb in range(n // r):
        for g in range(k // GROUP):
            for rr in range(r):
                row = rb * r + rr
                for c in range(chunks):
                    k0 = g * GROUP + c * per_beat
                    rebuilt[row, k0:k0 + per_beat] = _unpack(next(it), bits, per_beat)
                assert next(sit) == ws[row, g]
    assert np.array_equal(rebuilt, w)

    x = rng.integers(-128, 128, k)
    words = pack_x_words(x, lanes)
    assert np.array_equal(np.concatenate([_unpack(wd, 8, lanes) for wd in words]), x)
