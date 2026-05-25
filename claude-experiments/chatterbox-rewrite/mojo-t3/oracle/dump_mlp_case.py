"""
Produce self-contained MLP parity case using real T3 weights.

T3 LlamaMLP (SwiGLU):
  gate = x @ gate_proj.weight.T              # (rows, 4096)
  up   = x @ up_proj.weight.T                # (rows, 4096)
  out  = (silu(gate) * up) @ down_proj.weight.T   # (rows, 1024)

We pre-transpose all three weight matrices so the Mojo side can use
straight matmul (C = A @ B) without needing a B^T variant.

Fixture inputs:
  x               (rows, hidden)        from layer 0's actual residual stream
                                         post-attention (so the MLP sees realistic
                                         activations, not random noise).

Oracle runs HF's LlamaMLP module loaded with real weights.

Binary format (same as other oracles):
  i64 rank, i64[rank] shape, i32 tag, raw bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
WEIGHTS = FIX / "llama_t3_weights_fp32.safetensors"
ACTS = FIX / "activations_fp32.npz"
OUT = FIX / "mlp"
OUT.mkdir(parents=True, exist_ok=True)

# Mirror of T3 Llama config (only the fields LlamaMLP uses).
HIDDEN = 1024
INTERMEDIATE = 4096
ROWS = 16  # batch*seq from our standard fixture (1, 16, 1024)


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag = 0
        raw = arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag = 1
        raw = arr.astype(np.uint16, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def fp32_to_bf16_bits(arr_fp32: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(arr_fp32)).to(torch.bfloat16)
    return t.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def build_mlp(state: dict, dtype: torch.dtype) -> LlamaMLP:
    cfg = LlamaConfig(
        vocab_size=8, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_attention_heads=16, num_key_value_heads=16, head_dim=64,
        num_hidden_layers=30, rms_norm_eps=1e-5, mlp_bias=False,
        hidden_act="silu",
    )
    mlp = LlamaMLP(cfg)
    mlp.gate_proj.weight.data = state["layers.0.mlp.gate_proj.weight"].to(dtype)
    mlp.up_proj.weight.data = state["layers.0.mlp.up_proj.weight"].to(dtype)
    mlp.down_proj.weight.data = state["layers.0.mlp.down_proj.weight"].to(dtype)
    return mlp.to(dtype=dtype, device="cpu").eval()


def main() -> None:
    state = load_file(str(WEIGHTS))
    acts = np.load(ACTS)

    # Use input_embeds as the MLP input. Realistic magnitudes from the actual
    # forward pass; better signal than random noise for catching numeric drift.
    x_fp32 = torch.from_numpy(acts["input_embeds"].astype(np.float32)).reshape(ROWS, HIDDEN)

    # ---- fp32 ----
    mlp_fp32 = build_mlp(state, torch.float32)
    with torch.no_grad():
        out_fp32 = mlp_fp32(x_fp32)

    # Pre-transpose weights so Mojo can do straight A @ B matmul.
    gate_w_fp32 = state["layers.0.mlp.gate_proj.weight"].t().contiguous().float()  # (1024, 4096)
    up_w_fp32 = state["layers.0.mlp.up_proj.weight"].t().contiguous().float()       # (1024, 4096)
    down_w_fp32 = state["layers.0.mlp.down_proj.weight"].t().contiguous().float()   # (4096, 1024)

    write_tensor(OUT / "x_fp32.bin", x_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "gate_w_fp32.bin", gate_w_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "up_w_fp32.bin", up_w_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "down_w_fp32.bin", down_w_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "expected_fp32.bin", out_fp32.numpy().astype(np.float32))

    print(f"[fp32] x         {tuple(x_fp32.shape)}")
    print(f"[fp32] gate_w.T  {tuple(gate_w_fp32.shape)}")
    print(f"[fp32] up_w.T    {tuple(up_w_fp32.shape)}")
    print(f"[fp32] down_w.T  {tuple(down_w_fp32.shape)}")
    print(f"[fp32] out       {tuple(out_fp32.shape)}")
    print(f"  x[0,:4]   = {x_fp32[0,:4].numpy()}")
    print(f"  out[0,:4] = {out_fp32[0,:4].numpy()}")

    # ---- bf16 ----
    mlp_bf16 = build_mlp(state, torch.bfloat16)
    x_bf16 = x_fp32.to(torch.bfloat16)
    with torch.no_grad():
        out_bf16 = mlp_bf16(x_bf16)

    write_tensor(OUT / "x_bf16.bin",
                 fp32_to_bf16_bits(x_bf16.float().numpy()).reshape(x_fp32.shape))
    write_tensor(OUT / "gate_w_bf16.bin",
                 fp32_to_bf16_bits(gate_w_fp32.numpy()).reshape(gate_w_fp32.shape))
    write_tensor(OUT / "up_w_bf16.bin",
                 fp32_to_bf16_bits(up_w_fp32.numpy()).reshape(up_w_fp32.shape))
    write_tensor(OUT / "down_w_bf16.bin",
                 fp32_to_bf16_bits(down_w_fp32.numpy()).reshape(down_w_fp32.shape))
    write_tensor(OUT / "expected_bf16.bin",
                 fp32_to_bf16_bits(out_bf16.float().numpy()).reshape(out_fp32.shape))

    print(f"[bf16] out[0,:4] (decoded) = {out_bf16.float()[0,:4].numpy()}")


if __name__ == "__main__":
    main()
