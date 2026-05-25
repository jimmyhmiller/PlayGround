"""Dump upstream chatterbox layer-0 attention intermediates for parity comparison.

Hooks self_attn.q_proj/k_proj/v_proj/o_proj on layer 0 to capture inputs/outputs.
Also captures RoPE'd q/k via monkey-patching apply_rotary_pos_emb.
"""
import os, sys
import torch
import numpy as np

OUT = "/tmp/t3_dump"
os.makedirs(OUT, exist_ok=True)

from chatterbox.tts import ChatterboxTTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device=device)
tts.t3.tfmr.config._attn_implementation = "eager"
for layer in tts.t3.tfmr.layers:
    layer.self_attn.config._attn_implementation = "eager"
print("[dump-l0] forced eager attention")

tts.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav",
                         exaggeration=0.5)


def save(name, t):
    arr = t.detach().cpu().float().numpy().astype(np.float32)
    np.save(f"{OUT}/upstream_l0_{name}.npy", arr)
    print(f"  saved {name} shape={arr.shape}")


layer0 = tts.t3.tfmr.layers[0]

captures = {}
done = [False]   # only capture the FIRST prefill call

def hook_input_q(mod, inp, out):
    if done[0]: return
    captures["xnorm"] = inp[0]
    captures["qlin"] = out
def hook_klin(mod, inp, out):
    if done[0]: return
    captures["klin"] = out
def hook_vlin(mod, inp, out):
    if done[0]: return
    captures["vlin"] = out
def hook_oproj(mod, inp, out):
    if done[0]: return
    captures["av_flat"] = inp[0]
    captures["attnout"] = out
    done[0] = True   # done after o_proj completes

layer0.self_attn.q_proj.register_forward_hook(hook_input_q)
layer0.self_attn.k_proj.register_forward_hook(hook_klin)
layer0.self_attn.v_proj.register_forward_hook(hook_vlin)
layer0.self_attn.o_proj.register_forward_hook(hook_oproj)

# Patch eager_attention_forward to capture qklogits, attnprobs, av.
import transformers.models.llama.modeling_llama as llm_mod
orig_eager = llm_mod.eager_attention_forward

def patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    # query, key, value shape: (B, H, S, Dh)
    captures["qrope"] = query.clone()
    captures["krope"] = key.clone()
    captures["vperm"] = value.clone()
    # Mimic eager_attention_forward but capture intermediates.
    from transformers.models.llama.modeling_llama import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
    captures["qklogits"] = attn_weights.clone()
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    captures["attnprobs"] = attn_weights.clone()
    attn_output = torch.matmul(attn_weights, value_states)
    captures["av"] = attn_output.clone()
    attn_output = attn_output.transpose(1, 2).contiguous()
    # Restore original for subsequent layers (only capture L0).
    llm_mod.eager_attention_forward = orig_eager
    return attn_output, attn_weights

llm_mod.eager_attention_forward = patched_eager

# Also capture post-attn residual (input to post_attention_layernorm).
def hook_post_ln(mod, inp, out):
    captures.setdefault("postattn", inp[0])
layer0.post_attention_layernorm.register_forward_hook(hook_post_ln)

# Run inference to trigger captures.
with torch.no_grad():
    _ = tts.generate("Hello world.", cfg_weight=0.5, temperature=0.8, top_p=0.95,
                     min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)

print(f"[dump-l0] captured keys: {sorted(captures.keys())}")
for name, t in captures.items():
    # Keep only cond batch (index 0) like our Mojo dump.
    if t.shape[0] >= 2:
        t = t[0:1]
    save(name, t)
