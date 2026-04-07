"""
Export GPT-2 weights and a reference forward pass for our tensor DSL.

Usage:
    pip install torch transformers numpy
    python export_gpt2.py

Outputs to gpt2_weights/ directory:
    - weights.bin: all weight tensors concatenated as f32
    - manifest.json: describes each tensor's name, shape, offset, and size
    - reference_input.bin: token ids as f32
    - reference_output.bin: expected logits as f32
"""

import json
import os
import struct

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def export():
    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Config
    config = model.config
    n_layer = config.n_layer      # 12
    n_embd = config.n_embd        # 768
    n_head = config.n_head        # 12
    vocab_size = config.vocab_size  # 50257

    print(f"Config: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, vocab_size={vocab_size}")

    # Test input
    text = "Hello, world"
    tokens = tokenizer.encode(text)
    print(f"Input: {text!r} -> tokens: {tokens}")
    seq_len = len(tokens)

    # Run reference forward pass
    with torch.no_grad():
        input_ids = torch.tensor([tokens], dtype=torch.long)
        output = model(input_ids)
        logits = output.logits  # [1, seq_len, vocab_size]

    print(f"Output logits shape: {logits.shape}")
    print(f"Top-5 predictions for last token:")
    last_logits = logits[0, -1]
    top5 = torch.topk(last_logits, 5)
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        word = tokenizer.decode([idx.item()])
        print(f"  {i}: {word!r} (idx={idx.item()}, logit={val.item():.3f})")

    # Collect weights in the order our DSL expects
    sd = model.state_dict()

    os.makedirs("gpt2_weights", exist_ok=True)

    manifest = []
    all_data = bytearray()

    def add_tensor(name, tensor):
        """Add a tensor to the binary file and manifest."""
        data = tensor.detach().cpu().float().numpy()
        flat = data.flatten().astype(np.float32)
        offset = len(all_data)
        raw = flat.tobytes()
        all_data.extend(raw)
        manifest.append({
            "name": name,
            "shape": list(data.shape),
            "offset": offset,
            "size_bytes": len(raw),
            "n_elements": len(flat),
        })
        print(f"  {name}: {list(data.shape)} ({len(flat)} elements)")

    print("\nExporting weights...")

    # 0: tokens (will be provided at runtime)
    # 1: wte [vocab_size, n_embd]
    add_tensor("wte", sd["transformer.wte.weight"])

    # 2: wpe [1024, n_embd] — full positional embedding table
    add_tensor("wpe", sd["transformer.wpe.weight"])

    # For each layer
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"

        # LayerNorm 1 (pre-attention)
        add_tensor(f"ln1_g_{i}", sd[f"{prefix}.ln_1.weight"])
        add_tensor(f"ln1_b_{i}", sd[f"{prefix}.ln_1.bias"])

        # Attention QKV weight: Conv1D stores [in, out], transpose to [out, in]
        # so the K dimension is contiguous in memory for fast matmul.
        attn_w = sd[f"{prefix}.attn.c_attn.weight"]  # [n_embd, 3*n_embd] (Conv1D)
        attn_b = sd[f"{prefix}.attn.c_attn.bias"]    # [3*n_embd]
        add_tensor(f"attn_qkv_w_{i}", attn_w.t().contiguous())
        add_tensor(f"attn_qkv_b_{i}", attn_b)

        # Attention output projection
        proj_w = sd[f"{prefix}.attn.c_proj.weight"]  # [n_embd, n_embd] (Conv1D)
        proj_b = sd[f"{prefix}.attn.c_proj.bias"]    # [n_embd]
        add_tensor(f"attn_proj_w_{i}", proj_w.t().contiguous())
        add_tensor(f"attn_proj_b_{i}", proj_b)

        # LayerNorm 2 (pre-MLP)
        add_tensor(f"ln2_g_{i}", sd[f"{prefix}.ln_2.weight"])
        add_tensor(f"ln2_b_{i}", sd[f"{prefix}.ln_2.bias"])

        # MLP fc (expand) — transpose for fast matmul
        fc_w = sd[f"{prefix}.mlp.c_fc.weight"]  # [n_embd, 4*n_embd] (Conv1D)
        fc_b = sd[f"{prefix}.mlp.c_fc.bias"]    # [4*n_embd]
        add_tensor(f"mlp_fc_w_{i}", fc_w.t().contiguous())
        add_tensor(f"mlp_fc_b_{i}", fc_b)

        # MLP proj (contract) — transpose for fast matmul
        mp_w = sd[f"{prefix}.mlp.c_proj.weight"]  # [4*n_embd, n_embd] (Conv1D)
        mp_b = sd[f"{prefix}.mlp.c_proj.bias"]    # [n_embd]
        add_tensor(f"mlp_proj_w_{i}", mp_w.t().contiguous())
        add_tensor(f"mlp_proj_b_{i}", mp_b)

    # Final layer norm
    add_tensor("ln_f_g", sd["transformer.ln_f.weight"])
    add_tensor("ln_f_b", sd["transformer.ln_f.bias"])

    # Write binary weights
    with open("gpt2_weights/weights.bin", "wb") as f:
        f.write(all_data)
    print(f"\nWrote weights.bin: {len(all_data)} bytes ({len(all_data) / 1024 / 1024:.1f} MB)")

    # Write manifest
    with open("gpt2_weights/manifest.json", "w") as f:
        json.dump({
            "config": {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "seq_len": seq_len,
            },
            "tensors": manifest,
        }, f, indent=2)
    print("Wrote manifest.json")

    # Write reference input (token ids as f32)
    input_f32 = np.array(tokens, dtype=np.float32)
    with open("gpt2_weights/reference_input.bin", "wb") as f:
        f.write(input_f32.tobytes())
    print(f"Wrote reference_input.bin: {tokens}")

    # Write reference output (full logits for all positions)
    output_f32 = logits[0].detach().cpu().float().numpy().flatten().astype(np.float32)
    with open("gpt2_weights/reference_output.bin", "wb") as f:
        f.write(output_f32.tobytes())
    print(f"Wrote reference_output.bin: {logits.shape} ({len(output_f32)} elements)")

    print("\nDone! Files in gpt2_weights/")


if __name__ == "__main__":
    export()
