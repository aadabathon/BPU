"""Test data generators shared by the QMV testbenches."""

import numpy as np

from bpuref.fp import f32_to_bf16


def realistic_scales(rng, shape):
    """bf16 scales of the magnitude real quantized tensors have, either sign."""
    mag = 2.0 ** rng.uniform(-14, 0, shape)
    sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    return f32_to_bf16((sign * mag).astype(np.float32))


def wild_scales(rng, shape):
    """bf16 scales across the whole exponent range, incl. subnormal, inf and NaN."""
    exp = rng.choice([1, 2, 60, 127, 190, 254, 0, 255], shape)
    man = rng.integers(0, 128, shape)
    sign = rng.integers(0, 2, shape)
    return ((sign << 15) | (exp << 7) | man).astype(np.uint16)
