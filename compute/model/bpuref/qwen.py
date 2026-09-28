"""Qwen3.5 text decode on the BPU compute engines: model config, weights, a float64
reference of the model math, and the compiler from one decode step to a program of
QMV + FVU operations over the FVU scratchpad (SPM).

Semantics follow the pinned Hugging Face implementation (transformers 5.17,
models/qwen3_5/modeling_qwen3_5.py):
  * RMSNorm:       x * rsqrt(mean(x^2) + eps) * (1 + w)          (zero-centred weight)
  * gated RMSNorm: w * (x * rsqrt(mean(x^2) + eps)) * silu(z)    (plain weight)
  * DeltaNet:      causal conv (k=4, last 3 inputs) + SiLU, split [q | k | v],
                   l2norm(q), l2norm(k), q /= sqrt(dk), beta = sigmoid(b),
                   g = -exp(A_log) * softplus(a + dt_bias), per-head state S (dk x dv):
                   S *= exp(g); v_hat = S^T k; d = (v - v_hat) * beta; S += k d^T; o = S^T q
  * Attention:     q_proj rows per head = [query(hd) | gate(hd)], per-head RMSNorm on q/k,
                   rotate-half RoPE on the first hd/4 dims, softmax(q k^T / sqrt(hd)) v,
                   output * sigmoid(gate)
  * Layer:         h += mixer(norm(h)); h += mlp(norm(h)); logits = embed @ norm(h)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fvu as F
from . import sfu
from .fp import f32_bits
from .qmv import GROUP, W4, argmax_ref, qmv_ref
from .quant import quantize_weights_rtn


# ---------------------------------------------------------------------------
# Configuration and weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QwenConfig:
    hidden: int
    intermediate: int
    layer_types: tuple
    vocab: int
    lin_heads: int
    lin_dk: int
    lin_dv: int
    attn_heads: int
    kv_heads: int
    head_dim: int
    max_pos: int
    conv_k: int = 4
    rotary_frac: float = 0.25
    rope_theta: float = 1e7
    eps: float = 1e-6

    @property
    def key_dim(self):
        return self.lin_heads * self.lin_dk

    @property
    def value_dim(self):
        return self.lin_heads * self.lin_dv

    @property
    def conv_dim(self):
        return 2 * self.key_dim + self.value_dim

    @property
    def rot_dim(self):
        return int(self.head_dim * self.rotary_frac)


QWEN35_2B = QwenConfig(hidden=2048, intermediate=6144,
                       layer_types=("linear", "linear", "linear", "full") * 6, vocab=248320,
                       lin_heads=16, lin_dk=128, lin_dv=128, attn_heads=8, kv_heads=2,
                       head_dim=256, max_pos=8192)

# Same structure, small enough for full-model RTL simulation.
TINY = QwenConfig(hidden=128, intermediate=256, layer_types=("linear", "linear", "linear", "full"),
                  vocab=512, lin_heads=2, lin_dk=64, lin_dv=64, attn_heads=2, kv_heads=1,
                  head_dim=64, max_pos=16)


def random_weights(cfg: QwenConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Qwen-like random init (normal 0.02 linears; A_log, dt_bias, norms as in HF init,
    plus small noise on norm weights so they are exercised)."""
    rng = np.random.default_rng(seed)
    n = lambda *shape, std=0.02: rng.normal(0, std, shape).astype(np.float32)
    w = {"embed": n(cfg.vocab, cfg.hidden), "final_norm": n(cfg.hidden, std=0.1)}
    for i, kind in enumerate(cfg.layer_types):
        p = f"l{i}."
        w[p + "in_norm"] = n(cfg.hidden, std=0.1)
        w[p + "post_norm"] = n(cfg.hidden, std=0.1)
        if kind == "linear":
            w[p + "qkv"] = n(cfg.conv_dim, cfg.hidden)
            w[p + "z"] = n(cfg.value_dim, cfg.hidden)
            w[p + "b"] = n(cfg.lin_heads, cfg.hidden)
            w[p + "a"] = n(cfg.lin_heads, cfg.hidden)
            w[p + "conv"] = n(cfg.conv_dim, cfg.conv_k, std=0.3)
            w[p + "dt_bias"] = (1.0 + n(cfg.lin_heads, std=0.1)).astype(np.float32)
            w[p + "A_log"] = np.log(rng.uniform(0.01, 16, cfg.lin_heads)).astype(np.float32)
            w[p + "gnorm"] = (1.0 + n(cfg.lin_dv, std=0.1)).astype(np.float32)
            w[p + "out"] = n(cfg.hidden, cfg.value_dim)
        else:
            w[p + "q"] = n(2 * cfg.attn_heads * cfg.head_dim, cfg.hidden)
            w[p + "k"] = n(cfg.kv_heads * cfg.head_dim, cfg.hidden)
            w[p + "v"] = n(cfg.kv_heads * cfg.head_dim, cfg.hidden)
            w[p + "o"] = n(cfg.hidden, cfg.attn_heads * cfg.head_dim)
            w[p + "q_norm"] = n(cfg.head_dim, std=0.1)
            w[p + "k_norm"] = n(cfg.head_dim, std=0.1)
        w[p + "gate"] = n(cfg.intermediate, cfg.hidden)
        w[p + "up"] = n(cfg.intermediate, cfg.hidden)
        w[p + "down"] = n(cfg.hidden, cfg.intermediate)
    return w


LINEAR_NAMES = ("qkv", "z", "b", "a", "out", "q", "k", "v", "o", "gate", "up", "down")


def quantize(cfg: QwenConfig, w: dict, wfmt: int = W4) -> dict[str, tuple]:
    """Quantize every linear (and the tied embedding/head) to (codes, bf16 scales, wfmt)."""
    q = {"embed": (*quantize_weights_rtn(w["embed"], wfmt), wfmt)}
    for name, t in w.items():
        if name.split(".")[-1] in LINEAR_NAMES:
            q[name] = (*quantize_weights_rtn(t, wfmt), wfmt)
    return q


def dequantized(w: dict, qw: dict) -> dict:
    """Float weights equal to the quantized ones (for comparing math, not quantization)."""
    from .fp import bf16_to_f32
    out = dict(w)
    for name, (codes, scales, _) in qw.items():
        n, k = codes.shape
        out[name] = (codes.reshape(n, k // GROUP, GROUP)
                     * bf16_to_f32(scales)[:, :, None]).reshape(n, k).astype(np.float32)
    return out


def rope_tables(cfg: QwenConfig, pos: int):
    """cos/sin for the rotary dims at one position, as HF computes them (fp32)."""
    rd = cfg.rot_dim
    inv_freq = (1.0 / (cfg.rope_theta ** (np.arange(0, rd, 2, dtype=np.float32) / rd))).astype(np.float32)
    freqs = (inv_freq * np.float32(pos)).astype(np.float32)
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)]).astype(np.float32)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)]).astype(np.float32)
    return cos, sin


# ---------------------------------------------------------------------------
# Float64 reference of the model math (validated against Hugging Face)
# ---------------------------------------------------------------------------

class Float64Qwen:
    """Single-sequence decode, float64, mirroring the HF module structure."""

    def __init__(self, cfg: QwenConfig, w: dict):
        self.cfg, self.w = cfg, {k: v.astype(np.float64) for k, v in w.items()}
        self.pos = 0
        self.conv, self.S, self.K, self.V = {}, {}, {}, {}
        for i, kind in enumerate(cfg.layer_types):
            if kind == "linear":
                self.conv[i] = np.zeros((cfg.conv_k - 1, cfg.conv_dim))
                self.S[i] = np.zeros((cfg.lin_heads, cfg.lin_dk, cfg.lin_dv))
            else:
                self.K[i] = np.zeros((0, cfg.kv_heads, cfg.head_dim))
                self.V[i] = np.zeros((0, cfg.kv_heads, cfg.head_dim))

    def _norm(self, x, w, eps):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + w)

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    def _deltanet(self, i, x):
        c, w, p = self.cfg, self.w, f"l{i}."
        mixed = w[p + "qkv"] @ x
        z = (w[p + "z"] @ x).reshape(c.lin_heads, c.lin_dv)
        b, a = w[p + "b"] @ x, w[p + "a"] @ x
        hist = np.concatenate([self.conv[i], mixed[None]], axis=0)       # (k, conv_dim)
        conv = self._silu(np.einsum("kc,ck->c", hist, w[p + "conv"]))
        self.conv[i] = hist[1:]
        q, k, v = np.split(conv, [c.key_dim, 2 * c.key_dim])
        q = q.reshape(c.lin_heads, c.lin_dk)
        k = k.reshape(c.lin_heads, c.lin_dk)
        v = v.reshape(c.lin_heads, c.lin_dv)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        q = q / np.sqrt(c.lin_dk)
        beta = 1.0 / (1.0 + np.exp(-b))
        g = -np.exp(w[p + "A_log"]) * np.logaddexp(0.0, a + w[p + "dt_bias"])
        S = self.S[i] * np.exp(g)[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", S, k)
        delta = (v - kv_mem) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        self.S[i] = S
        o = np.einsum("hkv,hk->hv", S, q)
        o = o / np.sqrt(np.mean(o * o, axis=-1, keepdims=True) + c.eps) * w[p + "gnorm"]
        o = o * self._silu(z)
        return w[p + "out"] @ o.reshape(-1)

    def _attention(self, i, x):
        c, w, p = self.cfg, self.w, f"l{i}."
        qg = (w[p + "q"] @ x).reshape(c.attn_heads, 2 * c.head_dim)
        q, gate = qg[:, :c.head_dim], qg[:, c.head_dim:]
        k = (w[p + "k"] @ x).reshape(c.kv_heads, c.head_dim)
        v = (w[p + "v"] @ x).reshape(c.kv_heads, c.head_dim)
        q, k = self._norm(q, w[p + "q_norm"], c.eps), self._norm(k, w[p + "k_norm"], c.eps)
        cos, sin = (t.astype(np.float64) for t in rope_tables(c, self.pos))
        rd, h = c.rot_dim, c.rot_dim // 2

        def rope(t):
            r = t[:, :rd]
            rot = np.concatenate([-r[:, h:], r[:, :h]], axis=-1)
            return np.concatenate([r * cos + rot * sin, t[:, rd:]], axis=-1)

        q, k = rope(q), rope(k)
        self.K[i] = np.concatenate([self.K[i], k[None]])
        self.V[i] = np.concatenate([self.V[i], v[None]])
        group = c.attn_heads // c.kv_heads
        out = np.zeros((c.attn_heads, c.head_dim))
        for hq in range(c.attn_heads):
            kv = hq // group
            s = self.K[i][:, kv] @ q[hq] / np.sqrt(c.head_dim)
            pr = np.exp(s - s.max())
            out[hq] = (pr / pr.sum()) @ self.V[i][:, kv]
        out = out * (1.0 / (1.0 + np.exp(-gate)))
        return w[p + "o"] @ out.reshape(-1)

    def step(self, token: int) -> np.ndarray:
        """Feed one token, return the logits for the next one."""
        c, w = self.cfg, self.w
        h = w["embed"][token].copy()
        for i, kind in enumerate(c.layer_types):
            p = f"l{i}."
            x = self._norm(h, w[p + "in_norm"], c.eps)
            h = h + (self._deltanet(i, x) if kind == "linear" else self._attention(i, x))
            x = self._norm(h, w[p + "post_norm"], c.eps)
            h = h + w[p + "down"] @ (self._silu(w[p + "gate"] @ x) * (w[p + "up"] @ x))
        self.pos += 1
        self.last_hidden = h
        return w["embed"] @ self._norm(h, w["final_norm"], c.eps)


# ---------------------------------------------------------------------------
# Programs: QMV + FVU operations over the SPM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QmvOp:
    """y[0:n] = W x, with x given as a8 codes (K elements) + bf16 scales (K/64) in the SPM.
    argmax=True writes (index, value) of the best row to y[0], y[1] instead."""
    weight: str
    x: int
    xs: int
    k: int
    n: int
    y: int
    argmax: bool = False

    def __str__(self):
        return f"QMV[{self.weight} {self.n}x{self.k}{' argmax' if self.argmax else ''}] y={self.y}"


@dataclass
class MemMap:
    """Bump allocator: 64-aligned regions, sizes rounded up to 64 elements."""
    size: int = 0
    regions: dict = field(default_factory=dict)

    def alloc(self, name: str, n: int) -> int:
        base = self.size
        self.regions[name] = (base, n)
        self.size += -(-n // 64) * 64
        return base

    def __getitem__(self, name):
        return self.regions[name][0]


# fp32 constants the programs use, stored in the SPM.
def _constants(cfg: QwenConfig) -> dict[str, float]:
    return {"one": 1.0, "neg_one": -1.0, "eps": cfg.eps, "l2_eps": 1e-6, "twenty": 20.0,
            "ln2": float(np.float32(np.log(2.0))), "inv127": float(np.float32(1.0 / 127.0)),
            "inv_hidden": 1.0 / cfg.hidden, "inv_hd": 1.0 / cfg.head_dim, "inv_dv": 1.0 / cfg.lin_dv,
            "inv_sqrt_dk": float(np.float32(1.0 / np.sqrt(cfg.lin_dk))),
            "attn_scale": float(np.float32(1.0 / np.sqrt(cfg.head_dim)))}


class Compiler:
    """Lowers Qwen3.5 decode steps to QMV + FVU programs for a fixed memory map."""

    def __init__(self, cfg: QwenConfig, w: dict, qw: dict):
        self.cfg, self.w, self.qw = cfg, w, qw
        c = cfg
        m = self.mem = MemMap()
        for name in _constants(c):
            m.alloc("k_" + name, 1)
        # Model constants used by FVU ops
        for i, kind in enumerate(c.layer_types):
            p = f"l{i}."
            m.alloc(p + "in_norm1", c.hidden)
            m.alloc(p + "post_norm1", c.hidden)
            if kind == "linear":
                for k in range(c.conv_k):
                    m.alloc(p + f"conv_w{k}", c.conv_dim)
                m.alloc(p + "dt_bias", c.lin_heads)
                m.alloc(p + "A_log", c.lin_heads)
                m.alloc(p + "gnorm", c.lin_dv)
                # Persistent state
                for k in range(c.conv_k - 1):
                    m.alloc(p + f"conv_s{k}", c.conv_dim)
                m.alloc(p + "S", c.lin_heads * c.lin_dk * c.lin_dv)
            else:
                m.alloc(p + "q_norm1", c.head_dim)
                m.alloc(p + "k_norm1", c.head_dim)
                m.alloc(p + "Kc", c.kv_heads * c.max_pos * c.head_dim)
                m.alloc(p + "Vc", c.kv_heads * c.max_pos * c.head_dim)
        m.alloc("final_norm1", c.hidden)
        # Per-token inputs written by the host / memory side
        m.alloc("emb_codes", c.hidden)
        m.alloc("emb_scales", c.hidden // GROUP)
        m.alloc("rope_cos", c.rot_dim)
        m.alloc("rope_sin", c.rot_dim)                       # signed: [-sin_lo, +sin_hi]
        # Scratch
        big = max(c.conv_dim, c.intermediate, 2 * c.attn_heads * c.head_dim, c.hidden)
        for name in ("h", "xn", "t0", "t1", "t2", "t3", "codes"):
            m.alloc(name, big)
        m.alloc("scales", big // GROUP)
        m.alloc("scal", 64)                                  # scalar scratch
        m.alloc("scal2", 64)
        m.alloc("heads", 64)                                 # per-head scalars
        m.alloc("heads2", 64)
        m.alloc("scores", c.max_pos)
        m.alloc("out", max(c.value_dim, c.attn_heads * c.head_dim))
        m.alloc("mix", big)
        m.alloc("logits", c.vocab)
        m.alloc("token", 2)

    # -- memory image -------------------------------------------------------

    def initial_spm(self) -> np.ndarray:
        """SPM contents before the first step: constants, FVU-side weights, zero state."""
        c, w, m = self.cfg, self.w, self.mem
        spm = np.zeros(m.size, dtype=np.float32)
        for name, v in _constants(c).items():
            spm[m["k_" + name]] = np.float32(v)
        put = lambda name, v: spm.__setitem__(slice(m[name], m[name] + len(v)), v)
        f32 = lambda x: np.asarray(x, dtype=np.float32)
        for i, kind in enumerate(c.layer_types):
            p = f"l{i}."
            put(p + "in_norm1", f32(1.0) + w[p + "in_norm"])
            put(p + "post_norm1", f32(1.0) + w[p + "post_norm"])
            if kind == "linear":
                for k in range(c.conv_k):
                    put(p + f"conv_w{k}", np.ascontiguousarray(w[p + "conv"][:, k]))
                put(p + "dt_bias", w[p + "dt_bias"])
                put(p + "A_log", w[p + "A_log"])
                put(p + "gnorm", w[p + "gnorm"])
            else:
                put(p + "q_norm1", f32(1.0) + w[p + "q_norm"])
                put(p + "k_norm1", f32(1.0) + w[p + "k_norm"])
        put("final_norm1", f32(1.0) + w["final_norm"])
        return spm

    def host_inputs(self, token: int, pos: int) -> dict[int, np.ndarray]:
        """Per-step SPM writes by the host/memory side: embedding row (codes + scales, the
        memory port converts int codes to fp32) and the RoPE tables for this position."""
        c, m = self.cfg, self.mem
        codes, scales, _ = self.qw["embed"]
        from .fp import bf16_to_f32
        cos, sin = rope_tables(c, pos)
        h = c.rot_dim // 2
        sin_signed = np.concatenate([-sin[:h], sin[h:]]).astype(np.float32)
        return {m["emb_codes"]: codes[token].astype(np.float32),
                m["emb_scales"]: bf16_to_f32(scales[token]),
                m["rope_cos"]: cos, m["rope_sin"]: sin_signed}

    # -- program building blocks ------------------------------------------------

    def _k(self, name):
        return self.mem["k_" + name]

    def _quant(self, x, k):
        """a8 quantization of x[0:k] -> (codes, scales) regions (see fvu module)."""
        m, ops = self.mem, []
        g = k // GROUP
        sc, codes = m["scales"], m["codes"]
        ops += [F.FvuOp(F.RAMAX, g, GROUP, d=sc, a=x, a_stride=GROUP, d_stride=1),
                F.FvuOp(F.VMULS, 1, g, d=sc, a=sc, s=self._k("inv127")),
                F.FvuOp(F.VRBF16, 1, g, d=sc, a=sc),
                F.FvuOp(F.VSFU, 1, g, d=m["t3"], a=sc, func=sfu.RCP),
                F.FvuOp(F.VMULG, 1, k, d=codes, a=x, t=m["t3"]),
                F.FvuOp(F.VQCLAMP, 1, k, d=codes, a=codes)]
        return ops, codes, sc

    def _qmv(self, x, k, targets):
        """Quantize x once, then one QMV per (weight, n, y) target."""
        ops, codes, sc = self._quant(x, k)
        return ops + [QmvOp(name, codes, sc, k, n, y) for name, n, y in targets]

    def _rmsnorm(self, x, w1, n, y, rows=1, x_stride=0, y_stride=0, inv_n="inv_hidden"):
        """y = x * rsqrt(mean(x^2) + eps) * w1 (w1 = 1 + weight), per row."""
        s = self.mem["heads"]
        return [F.FvuOp(F.RDOT, rows, n, d=s, a=x, b=x, a_stride=x_stride, b_stride=x_stride, d_stride=1),
                F.FvuOp(F.VMULS, 1, rows, d=s, a=s, s=self._k(inv_n)),
                F.FvuOp(F.VADDS, 1, rows, d=s, a=s, s=self._k("eps")),
                F.FvuOp(F.VSFU, 1, rows, d=s, a=s, func=sfu.RSQRT),
                F.FvuOp(F.VMULS, rows, n, d=y, a=x, s=s, a_stride=x_stride, d_stride=y_stride, s_stride=1),
                F.FvuOp(F.VMUL, rows, n, d=y, a=y, b=w1, a_stride=y_stride, d_stride=y_stride)]

    def _sigmoid(self, x, y, n, rows=1, x_stride=0, y_stride=0):
        k = self._k
        return [F.FvuOp(F.VMULS, rows, n, d=y, a=x, s=k("neg_one"), a_stride=x_stride, d_stride=y_stride),
                F.FvuOp(F.VSFU, rows, n, d=y, a=y, func=sfu.EXP, a_stride=y_stride, d_stride=y_stride),
                F.FvuOp(F.VADDS, rows, n, d=y, a=y, s=k("one"), a_stride=y_stride, d_stride=y_stride),
                F.FvuOp(F.VSFU, rows, n, d=y, a=y, func=sfu.RCP, a_stride=y_stride, d_stride=y_stride)]

    def _silu(self, x, y, n):
        return self._sigmoid(x, y, n) + [F.FvuOp(F.VMUL, 1, n, d=y, a=x, b=y)]

    # -- layers ---------------------------------------------------------------

    def _deltanet(self, i):
        c, m, k = self.cfg, self.mem, self._k
        p = f"l{i}."
        H, nh, dk, dv = c.hidden, c.lin_heads, c.lin_dk, c.lin_dv
        xn, mixed, z, t2, t3 = m["xn"], m["t0"], m["t1"], m["t2"], m["t3"]
        heads, heads2, scal, out = m["heads"], m["heads2"], m["scal"], m["out"]
        ops = []
        ops += self._qmv(xn, H, [(p + "qkv", c.conv_dim, mixed), (p + "z", c.value_dim, z),
                                 (p + "b", nh, heads), (p + "a", nh, heads2)])
        # Causal conv over (s0, s1, s2, x) with weights (w0..w3), then SiLU.
        s = [m[p + f"conv_s{j}"] for j in range(c.conv_k - 1)]
        wc = [m[p + f"conv_w{j}"] for j in range(c.conv_k)]
        n = c.conv_dim
        ops += [F.FvuOp(F.VMUL, 1, n, d=t2, a=s[0], b=wc[0]),
                F.FvuOp(F.VMULADD, 1, n, d=t2, a=s[1], b=wc[1], c=t2),
                F.FvuOp(F.VMULADD, 1, n, d=t2, a=s[2], b=wc[2], c=t2),
                F.FvuOp(F.VMULADD, 1, n, d=t2, a=mixed, b=wc[3], c=t2),
                F.FvuOp(F.VCOPY, 1, n, d=s[0], a=s[1]),
                F.FvuOp(F.VCOPY, 1, n, d=s[1], a=s[2]),
                F.FvuOp(F.VCOPY, 1, n, d=s[2], a=mixed)]
        ops += self._silu(t2, mixed, n)                          # mixed = conv output [q | k | v]
        q, kk, v = mixed, mixed + c.key_dim, mixed + 2 * c.key_dim
        # l2norm(q) * 1/sqrt(dk), l2norm(k), in place
        for vec, extra in ((q, True), (kk, False)):
            ops += [F.FvuOp(F.RDOT, nh, dk, d=scal, a=vec, b=vec, a_stride=dk, b_stride=dk, d_stride=1),
                    F.FvuOp(F.VADDS, 1, nh, d=scal, a=scal, s=k("l2_eps")),
                    F.FvuOp(F.VSFU, 1, nh, d=scal, a=scal, func=sfu.RSQRT),
                    F.FvuOp(F.VMULS, nh, dk, d=vec, a=vec, s=scal, a_stride=dk, d_stride=dk, s_stride=1)]
            if extra:
                ops += [F.FvuOp(F.VMULS, 1, c.key_dim, d=vec, a=vec, s=k("inv_sqrt_dk"))]
        # Gates: heads <- beta = sigmoid(b); heads2 <- alpha = exp(-exp(A_log) * softplus(a + dt_bias))
        ops += self._sigmoid(heads, heads, nh)
        a_ = heads2
        sp = m["scal2"]
        ops += [F.FvuOp(F.VADD, 1, nh, d=a_, a=a_, b=m[p + "dt_bias"]),
                F.FvuOp(F.VSFU, 1, nh, d=sp, a=a_, func=sfu.EXP),
                F.FvuOp(F.VADDS, 1, nh, d=sp, a=sp, s=k("one")),
                F.FvuOp(F.VSFU, 1, nh, d=sp, a=sp, func=sfu.LOG2),
                F.FvuOp(F.VMULS, 1, nh, d=sp, a=sp, s=k("ln2")),
                F.FvuOp(F.VSEL, 1, nh, d=sp, a=a_, b=a_, c=sp, s=k("twenty")),   # softplus threshold
                F.FvuOp(F.VSFU, 1, nh, d=a_, a=m[p + "A_log"], func=sfu.EXP),
                F.FvuOp(F.VMUL, 1, nh, d=a_, a=a_, b=sp),
                F.FvuOp(F.VMULS, 1, nh, d=a_, a=a_, s=k("neg_one")),
                F.FvuOp(F.VSFU, 1, nh, d=a_, a=a_, func=sfu.EXP)]
        # Recurrent step, per head
        S0 = m[p + "S"]
        for h in range(nh):
            Sh, kh, qh, vh, oh = S0 + h * dk * dv, kk + h * dk, q + h * dk, v + h * dv, out + h * dv
            ops += [F.FvuOp(F.VMULS, dk, dv, d=Sh, a=Sh, s=heads2 + h, a_stride=dv, d_stride=dv),
                    F.FvuOp(F.VVECMAT, dk, dv, d=t3, a=Sh, s=kh, a_stride=dv, s_stride=1),
                    F.FvuOp(F.VSUB, 1, dv, d=t3, a=vh, b=t3),
                    F.FvuOp(F.VMULS, 1, dv, d=t3, a=t3, s=heads + h),
                    F.FvuOp(F.VAXPY, dk, dv, d=Sh, a=t3, b=Sh, s=kh, b_stride=dv, d_stride=dv, s_stride=1),
                    F.FvuOp(F.VVECMAT, dk, dv, d=oh, a=Sh, s=qh, a_stride=dv, s_stride=1)]
        # Gated RMSNorm per head, then silu(z)
        ops += [F.FvuOp(F.RDOT, nh, dv, d=scal, a=out, b=out, a_stride=dv, b_stride=dv, d_stride=1),
                F.FvuOp(F.VMULS, 1, nh, d=scal, a=scal, s=k("inv_dv")),
                F.FvuOp(F.VADDS, 1, nh, d=scal, a=scal, s=k("eps")),
                F.FvuOp(F.VSFU, 1, nh, d=scal, a=scal, func=sfu.RSQRT),
                F.FvuOp(F.VMULS, nh, dv, d=out, a=out, s=scal, a_stride=dv, d_stride=dv, s_stride=1),
                F.FvuOp(F.VMUL, nh, dv, d=out, a=out, b=m[p + "gnorm"], a_stride=dv, d_stride=dv)]
        ops += self._silu(z, t2, c.value_dim)
        ops += [F.FvuOp(F.VMUL, 1, c.value_dim, d=out, a=out, b=t2)]
        ops += self._qmv(out, c.value_dim, [(p + "out", H, m["mix"])])
        return ops

    def _attention(self, i, pos):
        c, m, k = self.cfg, self.mem, self._k
        p = f"l{i}."
        H, nq, nkv, hd, rd = c.hidden, c.attn_heads, c.kv_heads, c.head_dim, c.rot_dim
        xn, qg, kv, vv, qn = m["xn"], m["t0"], m["t1"], m["t2"], m["t3"]
        out, sc = m["out"], m["scores"]
        T = pos + 1
        ops = []
        ops += self._qmv(xn, H, [(p + "q", 2 * nq * hd, qg), (p + "k", nkv * hd, kv),
                                 (p + "v", nkv * hd, vv)])
        ops += self._rmsnorm(qg, m[p + "q_norm1"], hd, qn, rows=nq, x_stride=2 * hd, y_stride=hd, inv_n="inv_hd")
        ops += self._rmsnorm(kv, m[p + "k_norm1"], hd, kv, rows=nkv, x_stride=hd, y_stride=hd, inv_n="inv_hd")
        # RoPE on the first rd dims of each head: x*cos + perm(x)*[-sin, sin]
        tmp = m["mix"]
        for vec, rows in ((qn, nq), (kv, nkv)):
            ops += [F.FvuOp(F.VPERM, rows, rd, d=tmp, a=vec, a_stride=hd, d_stride=hd, half=rd // 2),
                    F.FvuOp(F.VMUL, rows, rd, d=tmp, a=tmp, b=m["rope_sin"], a_stride=hd, d_stride=hd),
                    F.FvuOp(F.VMULADD, rows, rd, d=vec, a=vec, b=m["rope_cos"], c=tmp,
                            a_stride=hd, c_stride=hd, d_stride=hd)]
        # Append this position's K and V
        Kc, Vc = m[p + "Kc"], m[p + "Vc"]
        ops += [F.FvuOp(F.VCOPY, nkv, hd, d=Kc + pos * hd, a=kv, a_stride=hd, d_stride=c.max_pos * hd),
                F.FvuOp(F.VCOPY, nkv, hd, d=Vc + pos * hd, a=vv, a_stride=hd, d_stride=c.max_pos * hd)]
        group = nq // nkv
        scal = m["scal"]
        for h in range(nq):
            kvh = h // group
            Kh, Vh = Kc + kvh * c.max_pos * hd, Vc + kvh * c.max_pos * hd
            ops += [F.FvuOp(F.RDOT, T, hd, d=sc, a=Kh, b=qn + h * hd, a_stride=hd, d_stride=1),
                    F.FvuOp(F.VMULS, 1, T, d=sc, a=sc, s=k("attn_scale")),
                    F.FvuOp(F.RMAX, 1, T, d=scal, a=sc),
                    F.FvuOp(F.VMULS, 1, 1, d=scal, a=scal, s=k("neg_one")),
                    F.FvuOp(F.VADDS, 1, T, d=sc, a=sc, s=scal),
                    F.FvuOp(F.VSFU, 1, T, d=sc, a=sc, func=sfu.EXP),
                    F.FvuOp(F.RSUM, 1, T, d=scal, a=sc),
                    F.FvuOp(F.VSFU, 1, 1, d=scal, a=scal, func=sfu.RCP),
                    F.FvuOp(F.VMULS, 1, T, d=sc, a=sc, s=scal),
                    F.FvuOp(F.VVECMAT, T, hd, d=out + h * hd, a=Vh, s=sc, a_stride=hd, s_stride=1)]
        # Output gate: out *= sigmoid(gate), gate_h = qg[h*2hd + hd : ...]
        ops += self._sigmoid(qg + hd, tmp, hd, rows=nq, x_stride=2 * hd, y_stride=hd)
        ops += [F.FvuOp(F.VMUL, nq, hd, d=out, a=out, b=tmp, a_stride=hd, b_stride=hd, d_stride=hd)]
        ops += self._qmv(out, nq * hd, [(p + "o", H, m["mix"])])
        return ops

    def step_program(self, pos: int, logits: bool = False) -> list:
        """Ops for one decode step at position `pos` (host inputs already written)."""
        c, m = self.cfg, self.mem
        H, h, xn = c.hidden, m["h"], m["xn"]
        ops = [F.FvuOp(F.VMULG, 1, H, d=h, a=m["emb_codes"], t=m["emb_scales"])]
        for i, kind in enumerate(c.layer_types):
            p = f"l{i}."
            ops += self._rmsnorm(h, m[p + "in_norm1"], H, xn)
            ops += self._deltanet(i) if kind == "linear" else self._attention(i, pos)
            ops += [F.FvuOp(F.VADD, 1, H, d=h, a=h, b=m["mix"])]
            ops += self._rmsnorm(h, m[p + "post_norm1"], H, xn)
            I = c.intermediate
            ops += self._qmv(xn, H, [(p + "gate", I, m["t0"]), (p + "up", I, m["t1"])])
            ops += self._silu(m["t0"], m["t2"], I)
            ops += [F.FvuOp(F.VMUL, 1, I, d=m["t2"], a=m["t2"], b=m["t1"])]
            ops += self._qmv(m["t2"], I, [(p + "down", H, m["mix"])])
            ops += [F.FvuOp(F.VADD, 1, H, d=h, a=h, b=m["mix"])]
        ops += self._rmsnorm(h, m["final_norm1"], H, xn)
        q_ops, codes, sc = self._quant(xn, H)
        ops += q_ops + [QmvOp("embed", codes, sc, H, c.vocab, m["token"], argmax=True)]
        if logits:
            ops += [QmvOp("embed", codes, sc, H, c.vocab, m["logits"])]
        return ops


def execute_qmv(op: QmvOp, spm: np.ndarray, qw: dict) -> None:
    from .fp import f32_to_bf16
    codes, scales, _ = qw[op.weight]
    x = spm[op.x:op.x + op.k].astype(np.int64)
    xs = f32_to_bf16(spm[op.xs:op.xs + op.k // GROUP])
    y = qmv_ref(codes[:op.n], scales[:op.n], x, xs)
    if op.argmax:
        i = argmax_ref(y)
        spm[op.y] = np.float32(i)
        spm[op.y + 1] = y[i]
    else:
        spm[op.y:op.y + op.n] = y


def run_program(ops, spm: np.ndarray, qw: dict) -> np.ndarray:
    for op in ops:
        if isinstance(op, QmvOp):
            execute_qmv(op, spm, qw)
        else:
            F.execute(op, spm)
    return spm


class BpuQwen:
    """Decode with the compiled program: the bit-exact reference of the hardware."""

    def __init__(self, cfg: QwenConfig, w: dict, qw: dict):
        self.cfg, self.comp, self.qw = cfg, Compiler(cfg, w, qw), qw
        self.spm = self.comp.initial_spm()
        self.pos = 0

    def step(self, token: int, logits: bool = False):
        for addr, vals in self.comp.host_inputs(token, self.pos).items():
            self.spm[addr:addr + len(vals)] = vals
        run_program(self.comp.step_program(self.pos, logits), self.spm, self.qw)
        m = self.comp.mem
        self.pos += 1
        nxt = int(self.spm[m["token"]])
        if logits:
            return nxt, self.spm[m["logits"]:m["logits"] + self.cfg.vocab].copy()
        return nxt
