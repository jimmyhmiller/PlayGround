"""
Multi-step argmax-decode parity case.

Simulates a simplified version of T3.inference():
  - Start from a fixed sequence of speech token ids (length T_PREFILL).
  - Embed them via speech_emb + speech_pos_emb at positions 0..T_PREFILL-1.
  - 30-layer prefill → final_norm → speech_head → argmax over the last row
    → first generated token id (at position T_PREFILL).
  - For N_STEPS - 1 additional steps: embed the new token + speech_pos_emb at
    its position, run a 30-layer decode step, argmax again.

We omit CFG, top-p, min-p, repetition penalty, and temperature — argmax is
deterministic and bit-stable across Mojo↔Torch when the underlying matmuls
agree, which is exactly what we want for parity.

Outputs:
  initial_ids_<tag>.bin       (T_PREFILL,) int64    fixed input prompt
  expected_ids_<tag>.bin      (N_STEPS,)  int64     argmax token sequence
  speech_emb_<tag>.bin        (8194, 1024)
  speech_pos_emb_<tag>.bin    (4100, 1024)
  speech_head_<tag>.bin       (8194, 1024)
  prefill_hidden_<tag>.bin    (1, T_PREFILL, 1024)  post-final-norm prefill output
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "generate"
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 1024
N_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE = 4096
N_LAYERS = 30
V_SPEECH = 8194
P_SPEECH = 4100
T_PREFILL = 15
N_STEPS = 8                     # total generated tokens (prefill argmax + 7 decodes)
BATCH = 1
EPS = 1e-5
SCALE = HEAD_DIM ** -0.5


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag, raw = 1, arr.astype(np.uint16, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
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


def rms_norm(x, weight, eps):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight.to(torch.float32) * x).to(in_dtype)


def apply_rope(q, k, cos, sin):
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    def rh(t):
        x1 = t[..., : t.shape[-1] // 2]
        x2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos_e) + (rh(q) * sin_e), (k * cos_e) + (rh(k) * sin_e)


def block_prefill(x, cos, sin, mask, w):
    B, S, _ = x.shape
    r1 = x
    x_n = rms_norm(x, w["in_norm"], EPS)
    q = (x_n @ w["qw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = (x_n @ w["kw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    v = (x_n @ w["vw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    logits = (q @ k.transpose(-1, -2)) * SCALE + mask.view(1, 1, S, S)
    probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    attn = (probs @ v).transpose(1, 2).contiguous().view(B, S, HIDDEN) @ w["ow"]
    x = r1 + attn
    r2 = x
    x_n = rms_norm(x, w["post_norm"], EPS)
    return r2 + F.silu(x_n @ w["gate_w"]) * (x_n @ w["up_w"]) @ w["down_w"], k, v


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
    attn = (probs @ v_all).transpose(1, 2).contiguous().view(B, 1, HIDDEN) @ w["ow"]
    x = r1 + attn
    r2 = x
    x_n = rms_norm(x, w["post_norm"], EPS)
    return r2 + F.silu(x_n @ w["gate_w"]) * (x_n @ w["up_w"]) @ w["down_w"], k_new, v_new


def load_layer(state, L, dtype):
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


def make_causal_mask(seq, dtype):
    finfo = torch.finfo(dtype)
    m = torch.zeros(seq, seq, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), finfo.min)
    return m


def dump_for_dtype(state, t3_state, dtype, tag):
    speech_emb = t3_state["speech_emb.weight"].to(dtype)
    speech_pos = t3_state["speech_pos_emb.emb.weight"].to(dtype)
    speech_head = t3_state["speech_head.weight"].to(dtype)
    final_norm_w = state["norm.weight"].to(dtype)
    layer_weights = [load_layer(state, L, dtype) for L in range(N_LAYERS)]

    # Fixed deterministic prompt: a mix of speech token ids in-range.
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    initial_ids = torch.randint(0, V_SPEECH - 2, (T_PREFILL,), generator=g, dtype=torch.int64)
    initial_ids = initial_ids.unsqueeze(0)  # (1, T_PREFILL)

    # ---- Embed initial ids ----
    pos_ids = torch.arange(T_PREFILL, dtype=torch.int64)
    x = speech_emb[initial_ids.flatten()].view(1, T_PREFILL, HIDDEN)
    x = x + speech_pos[pos_ids].unsqueeze(0)

    # RoPE for prefill (positions 0..T_PREFILL-1).
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
    dummy = torch.zeros(1, T_PREFILL + N_STEPS, HEAD_DIM, dtype=dtype)
    positions_full = torch.arange(T_PREFILL + N_STEPS, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos_full, sin_full = rope(dummy, positions_full)
    cos_pre, sin_pre = cos_full[:, :T_PREFILL], sin_full[:, :T_PREFILL]
    mask_pre = make_causal_mask(T_PREFILL, dtype)

    # ---- 30-layer prefill, collecting K/V history ----
    k_hists, v_hists = [], []
    h = x
    with torch.no_grad():
        for L in range(N_LAYERS):
            h, k, v = block_prefill(h, cos_pre, sin_pre, mask_pre, layer_weights[L])
            k_hists.append(k)
            v_hists.append(v)
        prefill_hidden = rms_norm(h, final_norm_w, EPS)  # (1, T_PREFILL, H)

    # ---- LM head on the last position → argmax → first generated token ----
    logits0 = prefill_hidden[:, -1:, :] @ speech_head.T  # (1, 1, V)
    next_id = int(logits0.argmax(dim=-1).item())
    generated = [next_id]

    # ---- N_STEPS - 1 decode steps ----
    with torch.no_grad():
        for step in range(N_STEPS - 1):
            cur_pos = T_PREFILL + step       # position of token being processed
            tok = torch.tensor([[next_id]], dtype=torch.int64)
            emb = speech_emb[tok.flatten()].view(1, 1, HIDDEN) + speech_pos[cur_pos].view(1, 1, HIDDEN)
            cos_dec = cos_full[:, cur_pos:cur_pos + 1]
            sin_dec = sin_full[:, cur_pos:cur_pos + 1]

            h = emb
            for L in range(N_LAYERS):
                h, k_new, v_new = block_decode(h, cos_dec, sin_dec,
                                                k_hists[L], v_hists[L], layer_weights[L])
                k_hists[L] = torch.cat([k_hists[L], k_new], dim=2)
                v_hists[L] = torch.cat([v_hists[L], v_new], dim=2)
            h = rms_norm(h, final_norm_w, EPS)
            logits = h @ speech_head.T
            next_id = int(logits.argmax(dim=-1).item())
            generated.append(next_id)

    expected_ids = np.array(generated, dtype=np.int64)
    print(f"[{tag}] initial_ids = {initial_ids.flatten().tolist()}")
    print(f"[{tag}] generated   = {generated}")

    def transform(arr_fp32: np.ndarray) -> np.ndarray:
        if tag == "fp32":
            return arr_fp32.astype(np.float32)
        return fp32_to_bf16_bits(arr_fp32).reshape(arr_fp32.shape)

    write_tensor(OUT / f"initial_ids_{tag}.bin", initial_ids.flatten().numpy().astype(np.int64))
    write_tensor(OUT / f"expected_ids_{tag}.bin", expected_ids)
    write_tensor(OUT / f"speech_emb_{tag}.bin", transform(speech_emb.float().numpy()))
    write_tensor(OUT / f"speech_pos_emb_{tag}.bin", transform(speech_pos.float().numpy()))
    write_tensor(OUT / f"speech_head_{tag}.bin", transform(speech_head.float().numpy()))
    write_tensor(OUT / f"prefill_hidden_{tag}.bin", transform(prefill_hidden.float().numpy()))

    # RoPE cos/sin for the full range (prefill + all decode positions).
    write_tensor(OUT / f"cos_full_{tag}.bin", transform(cos_full.float().numpy()))
    write_tensor(OUT / f"sin_full_{tag}.bin", transform(sin_full.float().numpy()))
    # Prefill causal mask.
    write_tensor(OUT / f"mask_prefill_{tag}.bin",
                 transform(mask_pre.float().numpy()) if tag == "fp32"
                 else fp32_to_bf16_bits(mask_pre.float().numpy()).reshape(mask_pre.shape))
    # Final norm weight (also in forward fixture but copy here for self-containment).
    write_tensor(OUT / f"final_norm_{tag}.bin", transform(final_norm_w.float().numpy()))


def main() -> None:
    WEIGHTS = FIX / "llama_t3_weights_fp32.safetensors"
    state = load_file(str(WEIGHTS))
    ckpt = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="t3_cfg.safetensors")
    t3_state = load_file(ckpt)

    dump_for_dtype(state, t3_state, torch.float32, "fp32")


if __name__ == "__main__":
    main()
