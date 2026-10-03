"""The Qwen3.5 reference: model math vs Hugging Face, and compiled programs vs the math."""

import numpy as np
import pytest

from bpuref import fvu as F
from bpuref.qwen import TINY, BpuQwen, Compiler, Float64Qwen, dequantized, quantize, random_weights

TOKENS = (5, 17, 300, 42, 7)


def test_float64_model_matches_hugging_face():
    pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from bpuref import hf_check
    try:
        from transformers.models import qwen3_5  # noqa: F401
    except ImportError:
        pytest.skip(f"transformers {tf.__version__} has no qwen3_5")
    w = random_weights(TINY, seed=3)
    ref = Float64Qwen(TINY, w)
    ours = [ref.step(t) for t in TOKENS]
    theirs = hf_check.hf_decode(hf_check.hf_model(TINY, w), TOKENS)
    for a, b in zip(ours, theirs):
        # HF runs norms and the DeltaNet core in fp32 even in a float64 model.
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-5


def test_compiled_program_tracks_the_model():
    """Same (dequantized) weights: only activation quantization, fp32 and SFU error remain."""
    w = random_weights(TINY, seed=4)
    qw = quantize(TINY, w)
    ref, bpu = Float64Qwen(TINY, dequantized(w, qw)), BpuQwen(TINY, w, qw)
    for t in TOKENS:
        f = ref.step(t)
        tok, lg = bpu.step(t, logits=True)
        assert f @ lg / (np.linalg.norm(f) * np.linalg.norm(lg)) > 0.999
        assert tok == int(np.argmax(lg))


def test_programs_are_hazard_free():
    w = random_weights(TINY, seed=5)
    comp = Compiler(TINY, w, quantize(TINY, w))
    for pos in (0, 1, TINY.max_pos - 1):
        for op in comp.step_program(pos, logits=True):
            if isinstance(op, F.FvuOp):
                F.validate(op, comp.mem.size)
