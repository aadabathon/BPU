"""Bit-exact reference model for the BPU compute engines (docs/numerics.md)."""

from .configs import QMV_SLICE_CONFIGS, QmvSliceConfig
from .fp import F32_QNAN, bf16_to_f32, f32_bits, f32_equal, f32_from_bits, f32_to_bf16
from .qmv import GROUP, W4, W8, pack_weight_stream, pack_x_words, qmv_ref
