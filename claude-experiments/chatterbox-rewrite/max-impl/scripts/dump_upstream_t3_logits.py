"""Dump upstream chatterbox T3's per-step cond+uncond raw logits.

Strategy: monkey-patch torch.softmax in the generation loop's window to
intercept the post-CFG logits, save to disk, and let generation continue
normally. We also dump the sampled token at each step so the Mojo side
can force-feed those tokens for a teacher-forcing comparison.

Output:
  /tmp/t3_dump/upstream_logits.npz with keys:
    text_ids: (T_TEXT,)
    sampled_tokens: (N_STEPS,)
    logits_cond:    (N_STEPS, V)     pre-warpers, post-CFG
    logits_uncond:  (N_STEPS, V)     raw uncond branch
    logits_cfg:     (N_STEPS, V)     cond + cfg * (cond - uncond)
"""
import os, sys, json
import torch
import numpy as np

OUT_DIR = "/tmp/t3_dump"
os.makedirs(OUT_DIR, exist_ok=True)

# Set torch seed for reproducible sampling.
SEED = 0xDEADBEEF & 0xFFFFFFFF
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from chatterbox.tts import ChatterboxTTS

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
TEXT = "Hello world."

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device=device)
# Force eager attention so HF uses naive QK→softmax→AV (matching our implementation),
# not SDPA which may have AMD-specific numerical differences.
import os
if os.environ.get("FORCE_EAGER", "1") == "1":
    tts.t3.tfmr.config._attn_implementation = "eager"
    for layer in tts.t3.tfmr.layers:
        layer.self_attn.config._attn_implementation = "eager"
    print("[dump] forced eager attention on T3 backbone")
tts.prepare_conditionals(REF, exaggeration=0.5)

# Monkey-patch into the loop. We'll wrap T3.inference's per-step block.
import chatterbox.models.t3.t3 as t3_module
orig_inference = t3_module.T3.inference

# Collect step-by-step data.
DUMP = {"steps": []}


def patched_inference(self, *, t3_cond, text_tokens, initial_speech_tokens=None,
                      prepend_prompt_speech_tokens=None, num_return_sequences=1,
                      max_new_tokens=None, stop_on_eos=True, do_sample=True,
                      temperature=0.8, top_p=0.95, min_p=0.05, length_penalty=1.0,
                      repetition_penalty=1.2, cfg_weight=0.5):
    """Same as upstream inference, but dumps cond/uncond/cfg logits per step."""
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    from chatterbox.models.t3.t3 import _ensure_BOT_EOT
    from tqdm import tqdm

    _ensure_BOT_EOT(text_tokens, self.hp)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

    if initial_speech_tokens is None:
        initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    embeds, len_cond = self.prepare_input_embeds(
        t3_cond=t3_cond, text_tokens=text_tokens,
        speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight,
    )

    if not getattr(self, "compiled", False):
        patched_model = T3HuggingfaceBackend(
            config=self.cfg, llama=self.tfmr, speech_enc=self.speech_emb,
            speech_head=self.speech_head, alignment_stream_analyzer=None,
        )
        self.patched_model = patched_model
        self.compiled = True

    bos = self.hp.start_speech_token
    bos_token = torch.tensor([[bos]], dtype=torch.long, device=self.device)
    bos_embed = self.speech_emb(bos_token) + self.speech_pos_emb.get_fixed_embedding(0)
    if cfg_weight > 0.0:
        bos_embed = torch.cat([bos_embed, bos_embed])
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

    generated_ids = bos_token.clone()
    predicted = []

    from transformers.generation.logits_process import (
        TemperatureLogitsWarper, TopPLogitsWarper, MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
    )
    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
    top_p_warper = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1) if top_p < 1.0 else None
    min_p_warper = MinPLogitsWarper(min_p=min_p, min_tokens_to_keep=1)

    output = self.patched_model(
        inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True,
        output_attentions=False, output_hidden_states=True, return_dict=True,
    )
    # DEBUG: dump per-layer hidden states for cond batch (index 0).
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        import numpy as _np
        for L, h in enumerate(output.hidden_states):
            # h shape: (B=2, S, D). Save cond batch only.
            arr = h[0].cpu().float().numpy().astype(_np.float32)
            # L=0 is INPUT embedding; L=1..n_layers is after each layer.
            # Save as -1 for input, 0..n_layers-1 for layer outputs.
            label = L - 1
            _np.save(f"/tmp/t3_dump/upstream_layer_{label}.npy", arr)
        print(f"[dump] saved {len(output.hidden_states)} hidden states (cond batch)")
    past = output.past_key_values

    V = output.logits.shape[-1]
    MAX_N = max_new_tokens if max_new_tokens is not None else 200

    for i in tqdm(range(MAX_N), desc="Sampling"):
        logits_step = output.logits[:, -1, :]   # (2, V)
        cond_raw = logits_step[0:1, :].clone()
        uncond_raw = logits_step[1:2, :].clone()
        cfg = torch.as_tensor(cfg_weight, device=cond_raw.device, dtype=cond_raw.dtype)
        logits_cfg = cond_raw + cfg * (cond_raw - uncond_raw)

        # Apply warpers (matches upstream loop).
        ids_for_proc = generated_ids[:1, ...]
        logits_after_rep = repetition_penalty_processor(ids_for_proc, logits_cfg)
        logits_after_temp = logits_after_rep / temperature if temperature != 1.0 else logits_after_rep
        logits_after_min_p = min_p_warper(ids_for_proc, logits_after_temp)
        logits_final = top_p_warper(ids_for_proc, logits_after_min_p) if top_p_warper else logits_after_min_p

        probs = torch.softmax(logits_final, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        DUMP["steps"].append({
            "cond_raw": cond_raw[0].cpu().float().numpy(),
            "uncond_raw": uncond_raw[0].cpu().float().numpy(),
            "logits_cfg": logits_cfg[0].cpu().float().numpy(),
            "next_token": int(next_token.item()),
        })

        predicted.append(next_token)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if next_token.view(-1) == self.hp.stop_speech_token:
            break

        next_token_embed = self.speech_emb(next_token)
        next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
        next_token_embed = torch.cat([next_token_embed, next_token_embed])
        output = self.patched_model(
            inputs_embeds=next_token_embed, past_key_values=past,
            output_attentions=False, output_hidden_states=True, return_dict=True,
        )
        past = output.past_key_values

    predicted_tokens = torch.cat(predicted, dim=1) if predicted else generated_ids[:, 1:]
    return predicted_tokens


t3_module.T3.inference = patched_inference

# Re-run generate (no watermark needed, we don't care about audio here).
import torch.nn.functional as F
_orig_generate = ChatterboxTTS.generate

# Match seed at sample time, too.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

wav = tts.generate(TEXT, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                   min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)

# Save.
N = len(DUMP["steps"])
V = DUMP["steps"][0]["cond_raw"].shape[0]
cond = np.stack([s["cond_raw"] for s in DUMP["steps"]])
uncond = np.stack([s["uncond_raw"] for s in DUMP["steps"]])
cfg = np.stack([s["logits_cfg"] for s in DUMP["steps"]])
sampled = np.array([s["next_token"] for s in DUMP["steps"]], dtype=np.int64)

# Also dump the text_ids that went through tokenizer.
text_ids = tts.tokenizer.text_to_tokens(TEXT)[0].cpu().numpy()

np.savez(f"{OUT_DIR}/upstream_logits.npz",
         text_ids=text_ids, sampled_tokens=sampled,
         logits_cond=cond, logits_uncond=uncond, logits_cfg=cfg)

# Also dump cond_emb (post-T3CondEnc output) for parity.
with torch.no_grad():
    cond_emb = tts.t3.prepare_conditioning(tts.conds.t3).cpu().float().numpy()
    # text tokens with START/STOP
    text_ids_with_se = tts.tokenizer.text_to_tokens(TEXT)  # (1, T)
    # add START/STOP via _ensure_BOT_EOT logic — just pad
    import torch.nn.functional as F
    sot = tts.t3.hp.start_text_token
    eot = tts.t3.hp.stop_text_token
    t_ids_padded = F.pad(text_ids_with_se, (1, 0), value=sot)
    t_ids_padded = F.pad(t_ids_padded, (0, 1), value=eot)
    t_ids_padded = t_ids_padded.to(tts.device)
    text_emb_up = tts.t3.text_emb(t_ids_padded) + tts.t3.text_pos_emb(t_ids_padded)
    text_emb_up = text_emb_up.cpu().float().numpy()
    np.save(f"{OUT_DIR}/upstream_text_emb.npy", text_emb_up)

    # bos_emb
    import torch as _t
    bos_token = _t.tensor([[tts.t3.hp.start_speech_token]], dtype=_t.long, device=tts.device)
    bos_emb_up = tts.t3.speech_emb(bos_token) + tts.t3.speech_pos_emb.get_fixed_embedding(0)
    bos_emb_up = bos_emb_up.cpu().float().numpy()
    np.save(f"{OUT_DIR}/upstream_bos_emb.npy", bos_emb_up)

np.save(f"{OUT_DIR}/upstream_cond_emb.npy", cond_emb)
print(f"[dump] upstream cond_emb shape={cond_emb.shape}")
print(f"[dump] upstream text_emb shape={text_emb_up.shape}")
print(f"[dump] upstream bos_emb shape={bos_emb_up.shape}")

# Also dump speaker_emb and cond_prompt_speech_tokens to compare conditioning inputs.
speaker_emb = tts.conds.t3.speaker_emb.cpu().float().numpy()
cond_prompt = tts.conds.t3.cond_prompt_speech_tokens.cpu().long().numpy()
emotion = tts.conds.t3.emotion_adv.cpu().float().numpy()
np.savez(f"{OUT_DIR}/upstream_t3_cond.npz",
         speaker_emb=speaker_emb, cond_prompt=cond_prompt, emotion=emotion)
print(f"[dump] saved T3 conditioning: speaker_emb shape={speaker_emb.shape}, cond_prompt shape={cond_prompt.shape}")
print(f"[dump] saved {N} steps to {OUT_DIR}/upstream_logits.npz")
print(f"[dump] text_ids: {text_ids.tolist()}")
print(f"[dump] sampled: {sampled[:20].tolist()}...")
