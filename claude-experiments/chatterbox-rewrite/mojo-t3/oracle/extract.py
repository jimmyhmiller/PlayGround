"""
Oracle extraction for Mojo T3 / Llama port.

Loads the actual T3 Llama backbone (Llama_520M config) with the real Chatterbox
weights, runs a fixed input through it, and dumps per-layer activations to disk
in both fp32 and bf16. These fixtures become the ground truth for parity tests.

What we dump per layer L:
  - input to L (residual stream entering L)
  - output of L's attention sublayer (post-attn residual add)
  - output of L's MLP sublayer (post-MLP residual add, == input to L+1)
  - q/k/v projections (post-RoPE) for L
  - attention weights output (pre-o_proj)

We also dump:
  - the input embedding (inputs_embeds)
  - the final RMSNorm output
  - logits-equivalent: hidden_states[-1]
  - all weights as a flat safetensors file (so Mojo loads from the same source)

Two passes: one in fp32 (CPU), one in bf16 (CPU bf16 too, deterministic).
We use CPU for both to avoid cuDNN/atomics nondeterminism.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaModel

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# Mirror of chatterbox/src/chatterbox/models/t3/llama_configs.py:LLAMA_520M_CONFIG_DICT
# Inlined so the oracle has no dependency on installing the chatterbox package.
LLAMA_520M_CONFIG_DICT = dict(
    vocab_size=8,
    max_position_embeddings=131072,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    attn_implementation="sdpa",
    head_dim=64,
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    num_key_value_heads=16,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=dict(
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
        rope_type="llama3",
    ),
    rope_theta=500000.0,
    torch_dtype="bfloat16",
    use_cache=True,
)


def load_t3_llama_state_dict() -> dict[str, torch.Tensor]:
    """Pull the T3 checkpoint and extract the Llama backbone weights only.

    The T3 state dict has keys like `tfmr.layers.0.self_attn.q_proj.weight`.
    We strip the `tfmr.` prefix so the dict can load directly into LlamaModel.
    """
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="t3_cfg.safetensors")
    full = load_file(ckpt_path)
    if "model" in full:
        full = full["model"][0]

    backbone: dict[str, torch.Tensor] = {}
    for k, v in full.items():
        if k.startswith("tfmr."):
            backbone[k[len("tfmr."):]] = v
    if not backbone:
        raise RuntimeError("no tfmr.* keys found in t3 checkpoint — layout changed?")
    return backbone


def build_llama(dtype: torch.dtype) -> LlamaModel:
    cfg_dict = dict(LLAMA_520M_CONFIG_DICT)
    cfg_dict["torch_dtype"] = str(dtype).replace("torch.", "")
    cfg = LlamaConfig(**cfg_dict)
    model = LlamaModel(cfg)
    state = load_t3_llama_state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading Llama: {unexpected[:5]}...")
    benign = {"rotary_emb.inv_freq"}
    real_missing = [k for k in missing if not any(k.endswith(b) for b in benign)]
    if real_missing:
        raise RuntimeError(f"missing keys when loading Llama: {real_missing[:5]}...")
    model = model.to(dtype=dtype, device="cpu").eval()
    return model


def make_fixed_input(seq_len: int, hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    x = torch.randn(1, seq_len, hidden_size, generator=g, dtype=torch.float32)
    return x.to(dtype=dtype)


def dump_for_dtype(dtype: torch.dtype, tag: str) -> dict[str, np.ndarray]:
    """Run a forward pass, capture activations via hooks, return as a flat dict."""
    torch.manual_seed(0)
    model = build_llama(dtype)
    cfg = model.config
    seq_len = 16
    inputs_embeds = make_fixed_input(seq_len, cfg.hidden_size, dtype)

    captured: dict[str, torch.Tensor] = {"input_embeds": inputs_embeds.detach()}

    def hook_residual(layer_idx: int):
        def fn(_module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[f"layer.{layer_idx}.residual_out"] = hs.detach()
        return fn

    def hook_attn(layer_idx: int):
        def fn(_module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[f"layer.{layer_idx}.attn_out"] = hs.detach()
        return fn

    def hook_mlp(layer_idx: int):
        def fn(_module, _args, output):
            captured[f"layer.{layer_idx}.mlp_out"] = output.detach()
        return fn

    def hook_input_norm(layer_idx: int):
        def fn(_module, _args, output):
            captured[f"layer.{layer_idx}.input_layernorm_out"] = output.detach()
        return fn

    def hook_post_attn_norm(layer_idx: int):
        def fn(_module, _args, output):
            captured[f"layer.{layer_idx}.post_attention_layernorm_out"] = output.detach()
        return fn

    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(hook_residual(i)))
        handles.append(layer.self_attn.register_forward_hook(hook_attn(i)))
        handles.append(layer.mlp.register_forward_hook(hook_mlp(i)))
        handles.append(layer.input_layernorm.register_forward_hook(hook_input_norm(i)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook_post_attn_norm(i)))
    handles.append(model.norm.register_forward_hook(
        lambda _m, _a, o: captured.update({"final_norm_out": o.detach()})
    ))

    with torch.inference_mode():
        out = model(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)

    captured["last_hidden_state"] = out.last_hidden_state.detach()

    for h in handles:
        h.remove()

    flat = {}
    for k, v in captured.items():
        v_cpu = v.to(dtype=torch.float32 if dtype == torch.bfloat16 else dtype).cpu().numpy()
        flat[k] = v_cpu
    flat["__meta_dtype"] = np.array([tag], dtype="U16")
    return flat


def write_npz(path: Path, flat: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)


def write_weights(path: Path) -> None:
    state = load_t3_llama_state_dict()
    state_fp32 = {k: v.to(torch.float32).contiguous() for k, v in state.items()}
    save_file(state_fp32, str(path))


def write_meta(path: Path) -> None:
    meta = {
        "config": LLAMA_520M_CONFIG_DICT,
        "fixture": {
            "seq_len": 16,
            "batch": 1,
            "hidden_size": LLAMA_520M_CONFIG_DICT["hidden_size"],
            "input_seed": "0xC0FFEE",
            "input_distribution": "torch.randn fp32 then cast",
        },
        "tolerances": {
            "fp32": {
                "rms_norm": 1e-6,
                "rope": 1e-5,
                "attn": 1e-4,
                "mlp": 1e-4,
                "single_layer": 1e-3,
                "full_forward": 1e-2,
            },
            "bf16": {
                "rms_norm": 1e-3,
                "rope": 1e-3,
                "attn": 5e-2,
                "mlp": 5e-2,
                "single_layer": 1e-1,
                "full_forward": 5e-1,
            },
        },
        "captured_keys_per_layer": [
            "layer.{i}.input_layernorm_out",
            "layer.{i}.post_attention_layernorm_out",
            "layer.{i}.attn_out",
            "layer.{i}.mlp_out",
            "layer.{i}.residual_out",
        ],
        "global_keys": ["input_embeds", "final_norm_out", "last_hidden_state"],
    }
    path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    print("→ extracting weights (fp32)")
    write_weights(FIXTURE_DIR / "llama_t3_weights_fp32.safetensors")

    print("→ running fp32 forward, capturing activations")
    flat_fp32 = dump_for_dtype(torch.float32, "fp32")
    write_npz(FIXTURE_DIR / "activations_fp32.npz", flat_fp32)

    print("→ running bf16 forward, capturing activations")
    flat_bf16 = dump_for_dtype(torch.bfloat16, "bf16")
    write_npz(FIXTURE_DIR / "activations_bf16.npz", flat_bf16)

    print("→ writing meta.json")
    write_meta(FIXTURE_DIR / "meta.json")

    print(f"done. fixtures in {FIXTURE_DIR}")
    print(f"  weights:     llama_t3_weights_fp32.safetensors")
    print(f"  fp32 acts:   activations_fp32.npz  ({len(flat_fp32)} tensors)")
    print(f"  bf16 acts:   activations_bf16.npz  ({len(flat_bf16)} tensors)")


if __name__ == "__main__":
    main()
