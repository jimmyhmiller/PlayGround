"""
Full 30-layer decode-step parity case.

Models one autoregressive decode step on top of a 15-token prefill, through
all 30 layers of the T3 Llama backbone + final RMSNorm.

For each layer L we record the post-RoPE K/V after the prefill pass — these
are the cache contents the decode step's attention will read from.

Outputs (per dtype tag):
  Globals:
    x_decode_<tag>.bin           (1, 1, 1024)
    cos_decode_<tag>.bin         (1, 1, 64)
    sin_decode_<tag>.bin         (1, 1, 64)
    final_norm_<tag>.bin         (1024,)
    expected_<tag>.bin           (1, 1, 1024)   post-final-norm output
  Per-layer L in 0..29:
    layer{L}/k_hist_<tag>.bin    (1, 16, 15, 64)
    layer{L}/v_hist_<tag>.bin    (1, 16, 15, 64)

We do NOT re-dump per-layer weights here — the existing forward/layer{L}/*.bin
fixtures already cover those, and the Mojo test will reuse them.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
WEIGHTS = FIX / "llama_t3_weights_fp32.safetensors"
ACTS = FIX / "activations_fp32.npz"
OUT = FIX / "forward_decode"
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 1024
N_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE = 4096
N_LAYERS = 30
T_HIST = 15
T_TOTAL = T_HIST + 1   # 16
BATCH = 1
EPS = 1e-5
SCALE = HEAD_DIM ** -0.5


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag, raw = 1, arr.astype(np.uint16, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def fp32_to_bf16_bits(arr_fp32: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(arr_fp32)).to(torch.bfloat16)
    return t.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight.to(torch.float32) * x).to(in_dtype)


def apply_rope(q, k, cos, sin):
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    def rotate_half(t):
        x1 = t[..., : t.shape[-1] // 2]
        x2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos_e) + (rotate_half(q) * sin_e), (k * cos_e) + (rotate_half(k) * sin_e)


def silu_mul(gate, up):
    return F.silu(gate) * up


def load_layer_weights(state: dict, L: int, dtype: torch.dtype) -> dict:
    pfx = f"layers.{L}."
    return {
        "in_norm": state[pfx + "input_layernorm.weight"].to(dtype),
        "post_norm": state[pfx + "post_attention_layernorm.weight"].to(dtype),
        "qw": state[pfx + "self_attn.q_proj.weight"].t().contiguous().to(dtype),
        "kw": state[pfx + "self_attn.k_proj.weight"].t().contiguous().to(dtype),
        "vw": state[pfx + "self_attn.v_proj.weight"].t().contiguous().to(dtype),
        "ow": state[pfx + "self_attn.o_proj.weight"].t().contiguous().to(dtype),
        "gate_w": state[pfx + "mlp.gate_proj.weight"].t().contiguous().to(dtype),
        "up_w": state[pfx + "mlp.up_proj.weight"].t().contiguous().to(dtype),
        "down_w": state[pfx + "mlp.down_proj.weight"].t().contiguous().to(dtype),
    }


def block_prefill(x, cos, sin, mask, w):
    B, S, _ = x.shape
    r1 = x
    x_n = rms_norm(x, w["in_norm"], EPS)
    q = (x_n @ w["qw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = (x_n @ w["kw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    v = (x_n @ w["vw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    logits = (q @ k.transpose(-1, -2)) * SCALE
    logits = logits + mask.view(1, 1, S, S)
    probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    attn = probs @ v
    attn = attn.transpose(1, 2).contiguous().view(B, S, HIDDEN) @ w["ow"]
    x = r1 + attn
    r2 = x
    x_n = rms_norm(x, w["post_norm"], EPS)
    mlp_out = silu_mul(x_n @ w["gate_w"], x_n @ w["up_w"]) @ w["down_w"]
    return r2 + mlp_out, k, v


def block_decode(x, cos, sin, k_hist, v_hist, w):
    B = x.shape[0]
    r1 = x
    x_n = rms_norm(x, w["in_norm"], EPS)
    q = (x_n @ w["qw"]).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
    k_new = (x_n @ w["kw"]).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
    v_new = (x_n @ w["vw"]).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
    q, k_new = apply_rope(q, k_new, cos, sin)
    k_all = torch.cat([k_hist, k_new], dim=2)
    v_all = torch.cat([v_hist, v_new], dim=2)
    logits = (q @ k_all.transpose(-1, -2)) * SCALE
    probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    attn = probs @ v_all
    attn = attn.transpose(1, 2).contiguous().view(B, 1, HIDDEN) @ w["ow"]
    x = r1 + attn
    r2 = x
    x_n = rms_norm(x, w["post_norm"], EPS)
    mlp_out = silu_mul(x_n @ w["gate_w"], x_n @ w["up_w"]) @ w["down_w"]
    return r2 + mlp_out


def make_causal_mask(seq: int, dtype: torch.dtype) -> torch.Tensor:
    finfo = torch.finfo(dtype)
    m = torch.zeros(seq, seq, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), finfo.min)
    return m


def dump_for_dtype(state: dict, x_input: np.ndarray, dtype: torch.dtype, tag: str) -> None:
    # Run 30-layer prefill on the first 15 tokens collecting per-layer K/V,
    # then 30-layer decode on the 16th token reusing those caches.
    x_full = torch.from_numpy(x_input.astype(np.float32))[:, :T_TOTAL, :].to(dtype)
    x_prefill = x_full[:, :T_HIST, :]
    x_decode = x_full[:, T_HIST:T_TOTAL, :]

    cfg = LlamaConfig(
        vocab_size=8, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS, head_dim=HEAD_DIM,
        num_hidden_layers=N_LAYERS, rms_norm_eps=EPS,
        rope_theta=500_000.0,
        rope_scaling=dict(
            factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0,
            original_max_position_embeddings=8192, rope_type="llama3",
        ),
        max_position_embeddings=131072,
        mlp_bias=False, attention_bias=False, hidden_act="silu",
        attn_implementation="eager",
    )
    rope = LlamaRotaryEmbedding(cfg).eval()
    dummy = torch.zeros(BATCH, T_TOTAL, HEAD_DIM, dtype=dtype)
    pos_pre = torch.arange(T_HIST, dtype=torch.long).unsqueeze(0)
    pos_dec = torch.tensor([[T_HIST]], dtype=torch.long)
    with torch.no_grad():
        cos_pre, sin_pre = rope(dummy[:, :T_HIST, :], pos_pre)
        cos_dec, sin_dec = rope(dummy[:, :1, :], pos_dec)

    mask_pre = make_causal_mask(T_HIST, dtype)

    # ---- Prefill 30 layers, collecting per-layer K/V history ----
    k_hists = []
    v_hists = []
    h = x_prefill
    with torch.no_grad():
        for L in range(N_LAYERS):
            w = load_layer_weights(state, L, dtype)
            h, k, v = block_prefill(h, cos_pre, sin_pre, mask_pre, w)
            k_hists.append(k)
            v_hists.append(v)

    # ---- Decode 30 layers ----
    h = x_decode
    with torch.no_grad():
        for L in range(N_LAYERS):
            w = load_layer_weights(state, L, dtype)
            h = block_decode(h, cos_dec, sin_dec, k_hists[L], v_hists[L], w)

    # ---- Final RMSNorm ----
    final_norm_w = state["norm.weight"].to(dtype)
    with torch.no_grad():
        expected = rms_norm(h, final_norm_w, EPS)

    def transform(arr_fp32: np.ndarray) -> np.ndarray:
        if tag == "fp32":
            return arr_fp32.astype(np.float32)
        return fp32_to_bf16_bits(arr_fp32).reshape(arr_fp32.shape)

    # Globals.
    write_tensor(OUT / f"x_decode_{tag}.bin", transform(x_decode.float().numpy()))
    write_tensor(OUT / f"cos_decode_{tag}.bin", transform(cos_dec.float().numpy()))
    write_tensor(OUT / f"sin_decode_{tag}.bin", transform(sin_dec.float().numpy()))
    write_tensor(OUT / f"final_norm_{tag}.bin", transform(final_norm_w.float().numpy()))
    write_tensor(OUT / f"expected_{tag}.bin", transform(expected.float().numpy()))

    # Per-layer caches.
    for L in range(N_LAYERS):
        layer_dir = OUT / f"layer{L}"
        write_tensor(layer_dir / f"k_hist_{tag}.bin", transform(k_hists[L].float().numpy()))
        write_tensor(layer_dir / f"v_hist_{tag}.bin", transform(v_hists[L].float().numpy()))

    print(f"[{tag}] expected[0,0,:4] = {expected.float()[0,0,:4].numpy()}")


def main() -> None:
    state = load_file(str(WEIGHTS))
    acts = np.load(ACTS)
    dump_for_dtype(state, acts["input_embeds"], torch.float32, "fp32")
    dump_for_dtype(state, acts["input_embeds"], torch.bfloat16, "bf16")


if __name__ == "__main__":
    main()
