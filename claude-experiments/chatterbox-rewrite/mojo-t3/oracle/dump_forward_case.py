"""
Full Llama-backbone forward parity case using real T3 weights.

We test the full 30-layer Llama backbone, ending at the final RMSNorm output
(== last_hidden_state). The speech_head is a separate matmul we'll add later.

Per-layer weights dumped (all pre-transposed for direct A @ B matmul):
  layer{L}/in_norm.bin        (1024,)
  layer{L}/post_norm.bin      (1024,)
  layer{L}/qw.bin             (1024, 1024)
  layer{L}/kw.bin             (1024, 1024)
  layer{L}/vw.bin             (1024, 1024)
  layer{L}/ow.bin             (1024, 1024)
  layer{L}/gate_w.bin         (1024, 4096)
  layer{L}/up_w.bin           (1024, 4096)
  layer{L}/down_w.bin         (4096, 1024)
for L in 0..29.

Plus globals:
  input_embeds.bin     (1, 16, 1024)    the embed-table output (we already
                                         have this from extract.py)
  final_norm.bin       (1024,)          weight of the final RMSNorm
  cos.bin, sin.bin     (1, 16, 64)      RoPE table for SEQ=16
  mask.bin             (16, 16)         causal mask
  expected.bin         (1, 16, 1024)    last_hidden_state from extract.py

We reuse activations_fp32.npz for input_embeds + last_hidden_state
(no need to re-run HF; those fixtures are already pinned to the same model
state and are guaranteed deterministic).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
WEIGHTS = FIX / "llama_t3_weights_fp32.safetensors"
ACTS = FIX / "activations_fp32.npz"
ACTS_BF16 = FIX / "activations_bf16.npz"
OUT = FIX / "forward"
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 1024
N_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE = 4096
N_LAYERS = 30
SEQ = 16
BATCH = 1


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag = 0
        raw = arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag = 1
        raw = arr.astype(np.uint16, copy=False).tobytes()
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


def make_causal_mask_fp32(seq: int) -> np.ndarray:
    finfo = np.finfo(np.float32)
    m = np.zeros((seq, seq), dtype=np.float32)
    for i in range(seq):
        for j in range(i + 1, seq):
            m[i, j] = finfo.min
    return m


def make_causal_mask_bf16_bits(seq: int) -> np.ndarray:
    # bf16's most-negative finite is approximately -3.39e38 (matches fp32 finfo.min
    # when cast to bf16). We round-trip through torch to get the exact bit pattern.
    finfo = torch.finfo(torch.bfloat16)
    m = torch.zeros(seq, seq, dtype=torch.bfloat16)
    for i in range(seq):
        for j in range(i + 1, seq):
            m[i, j] = finfo.min
    return m.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def dump_weights_per_layer(state: dict, dtype_tag: str, transform):
    """transform: callable np_fp32 → np_bf16_bits (or identity for fp32)."""
    for L in range(N_LAYERS):
        pfx = f"layers.{L}."
        in_norm = state[pfx + "input_layernorm.weight"].float().numpy()
        post_norm = state[pfx + "post_attention_layernorm.weight"].float().numpy()
        qw = state[pfx + "self_attn.q_proj.weight"].t().contiguous().float().numpy()
        kw = state[pfx + "self_attn.k_proj.weight"].t().contiguous().float().numpy()
        vw = state[pfx + "self_attn.v_proj.weight"].t().contiguous().float().numpy()
        ow = state[pfx + "self_attn.o_proj.weight"].t().contiguous().float().numpy()
        gate_w = state[pfx + "mlp.gate_proj.weight"].t().contiguous().float().numpy()
        up_w = state[pfx + "mlp.up_proj.weight"].t().contiguous().float().numpy()
        down_w = state[pfx + "mlp.down_proj.weight"].t().contiguous().float().numpy()

        layer_dir = OUT / f"layer{L}"
        write_tensor(layer_dir / f"in_norm_{dtype_tag}.bin", transform(in_norm))
        write_tensor(layer_dir / f"post_norm_{dtype_tag}.bin", transform(post_norm))
        write_tensor(layer_dir / f"qw_{dtype_tag}.bin", transform(qw))
        write_tensor(layer_dir / f"kw_{dtype_tag}.bin", transform(kw))
        write_tensor(layer_dir / f"vw_{dtype_tag}.bin", transform(vw))
        write_tensor(layer_dir / f"ow_{dtype_tag}.bin", transform(ow))
        write_tensor(layer_dir / f"gate_w_{dtype_tag}.bin", transform(gate_w))
        write_tensor(layer_dir / f"up_w_{dtype_tag}.bin", transform(up_w))
        write_tensor(layer_dir / f"down_w_{dtype_tag}.bin", transform(down_w))


def main() -> None:
    state = load_file(str(WEIGHTS))
    acts_fp32 = np.load(ACTS)
    acts_bf16 = np.load(ACTS_BF16)

    # RoPE table — same setup as block fixture, reused unchanged across layers.
    cfg = LlamaConfig(
        vocab_size=8, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS, head_dim=HEAD_DIM,
        num_hidden_layers=N_LAYERS, rms_norm_eps=1e-5,
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
    positions = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)

    # ---- fp32 ----
    dummy_x = torch.zeros(BATCH, SEQ, HEAD_DIM, dtype=torch.float32)
    with torch.no_grad():
        cos_fp32, sin_fp32 = rope(dummy_x, positions)

    write_tensor(OUT / "input_embeds_fp32.bin",
                 acts_fp32["input_embeds"].astype(np.float32))
    write_tensor(OUT / "expected_fp32.bin",
                 acts_fp32["last_hidden_state"].astype(np.float32))
    write_tensor(OUT / "cos_fp32.bin", cos_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "sin_fp32.bin", sin_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "mask_fp32.bin", make_causal_mask_fp32(SEQ))
    write_tensor(OUT / "final_norm_fp32.bin",
                 state["norm.weight"].float().numpy().astype(np.float32))

    print(f"[fp32] weights for {N_LAYERS} layers ...")
    dump_weights_per_layer(state, "fp32", lambda a: a.astype(np.float32))

    print(f"[fp32] input_embeds[0,0,:4] = {acts_fp32['input_embeds'][0,0,:4]}")
    print(f"[fp32] expected[0,0,:4]     = {acts_fp32['last_hidden_state'][0,0,:4]}")

    # ---- bf16 ----
    dummy_x_bf16 = dummy_x.to(torch.bfloat16)
    with torch.no_grad():
        cos_bf16, sin_bf16 = rope(dummy_x_bf16, positions)

    # Input/expected from the bf16 oracle pass (already bf16-rounded).
    write_tensor(OUT / "input_embeds_bf16.bin",
                 fp32_to_bf16_bits(acts_bf16["input_embeds"].astype(np.float32))
                 .reshape(acts_fp32["input_embeds"].shape))
    write_tensor(OUT / "expected_bf16.bin",
                 fp32_to_bf16_bits(acts_bf16["last_hidden_state"].astype(np.float32))
                 .reshape(acts_fp32["last_hidden_state"].shape))
    write_tensor(OUT / "cos_bf16.bin",
                 fp32_to_bf16_bits(cos_bf16.float().numpy())
                 .reshape(cos_fp32.shape))
    write_tensor(OUT / "sin_bf16.bin",
                 fp32_to_bf16_bits(sin_bf16.float().numpy())
                 .reshape(sin_fp32.shape))
    write_tensor(OUT / "mask_bf16.bin", make_causal_mask_bf16_bits(SEQ))
    final_norm_fp32 = state["norm.weight"].float().numpy()
    write_tensor(OUT / "final_norm_bf16.bin",
                 fp32_to_bf16_bits(final_norm_fp32).reshape(final_norm_fp32.shape))

    print(f"[bf16] weights for {N_LAYERS} layers ...")
    dump_weights_per_layer(state, "bf16",
                           lambda a: fp32_to_bf16_bits(a).reshape(a.shape))

    print(f"[bf16] expected[0,0,:4] (decoded) = {acts_bf16['last_hidden_state'][0,0,:4]}")


if __name__ == "__main__":
    main()
