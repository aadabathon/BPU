"""Cross-check bpuref.qwen against the Hugging Face Qwen3.5 implementation.

    python -m bpuref.hf_check            # prints agreement for a few decode steps

Needs torch + transformers (>= 5.17, with models/qwen3_5). Hugging Face computes the
norms and the DeltaNet core in fp32 even in a float64 model, so agreement with the
float64 reference is ~1e-6 relative, not 1e-12.
"""

from __future__ import annotations

import numpy as np

from .qwen import TINY, BpuQwen, Float64Qwen, QwenConfig, dequantized, quantize, random_weights


def hf_model(cfg: QwenConfig, w: dict):
    import torch
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    half = cfg.rot_dim // 2
    sec = [half - 2 * (half // 3), half // 3, half // 3]         # sums to rot_dim / 2
    hf_cfg = Qwen3_5TextConfig(
        vocab_size=cfg.vocab, hidden_size=cfg.hidden, intermediate_size=cfg.intermediate,
        num_hidden_layers=len(cfg.layer_types),
        layer_types=["linear_attention" if t == "linear" else "full_attention" for t in cfg.layer_types],
        linear_num_key_heads=cfg.lin_heads, linear_num_value_heads=cfg.lin_heads,
        linear_key_head_dim=cfg.lin_dk, linear_value_head_dim=cfg.lin_dv,
        linear_conv_kernel_dim=cfg.conv_k, num_attention_heads=cfg.attn_heads,
        num_key_value_heads=cfg.kv_heads, head_dim=cfg.head_dim, rms_norm_eps=cfg.eps,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta,
                         "partial_rotary_factor": cfg.rotary_frac, "mrope_section": sec,
                         "mrope_interleaved": True},
        tie_word_embeddings=True, attn_implementation="eager")
    model = Qwen3_5ForCausalLM(hf_cfg).double().eval()
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64))
    sd = {"model.embed_tokens.weight": t(w["embed"]), "model.norm.weight": t(w["final_norm"]),
          "lm_head.weight": t(w["embed"])}
    for i, kind in enumerate(cfg.layer_types):
        p, L = f"l{i}.", f"model.layers.{i}."
        sd[L + "input_layernorm.weight"] = t(w[p + "in_norm"])
        sd[L + "post_attention_layernorm.weight"] = t(w[p + "post_norm"])
        for ours, theirs in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            sd[L + f"mlp.{theirs}.weight"] = t(w[p + ours])
        if kind == "linear":
            A = L + "linear_attn."
            for ours, theirs in (("qkv", "in_proj_qkv"), ("z", "in_proj_z"), ("b", "in_proj_b"),
                                 ("a", "in_proj_a"), ("out", "out_proj")):
                sd[A + theirs + ".weight"] = t(w[p + ours])
            sd[A + "conv1d.weight"] = t(w[p + "conv"])[:, None, :]
            sd[A + "dt_bias"] = t(w[p + "dt_bias"])
            sd[A + "A_log"] = t(w[p + "A_log"])
            sd[A + "norm.weight"] = t(w[p + "gnorm"])
        else:
            A = L + "self_attn."
            for ours in ("q", "k", "v", "o"):
                sd[A + f"{ours}_proj.weight"] = t(w[p + ours])
            sd[A + "q_norm.weight"] = t(w[p + "q_norm"])
            sd[A + "k_norm.weight"] = t(w[p + "k_norm"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    return model


def hf_decode(model, tokens):
    """Logits after each token, feeding one token at a time with the cache."""
    import torch
    out, past = [], None
    with torch.no_grad():
        for tok in tokens:
            r = model(input_ids=torch.tensor([[tok]]), past_key_values=past, use_cache=True)
            past = r.past_key_values
            out.append(r.logits[0, -1].double().numpy())
    return out


def compare(cfg: QwenConfig = TINY, tokens=(5, 17, 300, 42, 7, 99), seed: int = 0) -> dict:
    w = random_weights(cfg, seed)
    ref = Float64Qwen(cfg, w)
    ours = [ref.step(t) for t in tokens]
    theirs = hf_decode(hf_model(cfg, w), tokens)
    rel = [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(ours, theirs)]

    qw = quantize(cfg, w)
    fref = Float64Qwen(cfg, dequantized(w, qw))
    bpu = BpuQwen(cfg, w, qw)
    cos, argmax_agree = [], []
    for t in tokens:
        f = fref.step(t)
        tok, lg = bpu.step(t, logits=True)
        cos.append(float(f @ lg / (np.linalg.norm(f) * np.linalg.norm(lg))))
        argmax_agree.append(tok == int(np.argmax(lg)))
    return {"hf_vs_float64_rel_err": rel, "bpu_vs_float64_cosine": cos,
            "bpu_argmax_is_logit_argmax": argmax_agree}


if __name__ == "__main__":
    for k, v in compare().items():
        print(f"{k:32s}", [round(x, 8) if isinstance(x, float) else x for x in v])
