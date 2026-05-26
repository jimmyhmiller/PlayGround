"""Dump upstream chatterbox CFM intermediates for chunk 53 text.

Saves: speech_tokens, mu, spks, cond, final_mel, audio. We'll re-feed these
into Mojo's CFM to find where they diverge.
"""
import os, sys, json, time, types
import numpy as np
import torch

import chatterbox.tts as tts_mod
from chatterbox.tts import ChatterboxTTS

TEXT = ("What then could have motivated Carnap's heroic efforts on the conceptual "
        "side of epistemology, when hope of certainty on the doctrinal side was "
        "abandoned? There were two good reasons still.")
REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
OUT_DIR = "/tmp/cfm_diag"
os.makedirs(OUT_DIR, exist_ok=True)

# Force deterministic sampling so we can reproduce.
SEED = 0xDEADBEEF + 53
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals(REF, exaggeration=0.5)

# Patch the s3gen flow.decoder.forward to capture CFM inputs and final mel.
captured = {}
s3gen = tts.s3gen
cfm = s3gen.flow.decoder   # CausalConditionalCFM

@torch.inference_mode()
def patched_cfm_forward(self_cfm, mu, mask, n_timesteps, temperature=1.0,
                        spks=None, cond=None, noised_mels=None, meanflow=False):
    captured["mu"] = mu.detach().cpu().float().numpy().copy()
    captured["mask"] = mask.detach().cpu().float().numpy().copy()
    captured["spks"] = spks.detach().cpu().float().numpy().copy() if spks is not None else None
    captured["cond"] = cond.detach().cpu().float().numpy().copy() if cond is not None else None
    captured["n_timesteps"] = int(n_timesteps)
    captured["t_scheduler"] = self_cfm.t_scheduler
    captured["inference_cfg_rate"] = float(self_cfm.inference_cfg_rate)

    z = torch.randn_like(mu)
    if noised_mels is not None:
        prompt_len = mu.size(2) - noised_mels.size(2)
        z = z.clone()
        z[..., prompt_len:] = noised_mels
    captured["z_init"] = z.detach().cpu().float().numpy().copy()

    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    if (not meanflow) and (self_cfm.t_scheduler == "cosine"):
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    captured["t_span"] = t_span.detach().cpu().float().numpy().copy()

    x = self_cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask,
                              spks=spks, cond=cond, meanflow=meanflow)
    captured["final_x"] = x.detach().cpu().float().numpy().copy()
    return x, None
cfm.forward = types.MethodType(patched_cfm_forward, cfm)

# Also capture the speech_tokens that T3 produced.
import chatterbox.models.t3.t3 as t3_mod
orig_inference = t3_mod.T3.inference

@torch.inference_mode()
def patched_t3(self, **kw):
    out = orig_inference(self, **kw)
    # out shape: (1, T) speech tokens
    captured["speech_tokens"] = out.detach().cpu().numpy().reshape(-1).tolist()
    return out
t3_mod.T3.inference = patched_t3

t0 = time.perf_counter()
wav = tts.generate(TEXT, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                    min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
print(f"upstream: {time.perf_counter()-t0:.1f}s wall, audio={audio.size/24000:.2f}s", flush=True)

# Save everything.
np.savez(f"{OUT_DIR}/upstream.npz",
         text=np.array([TEXT]),
         audio=audio,
         speech_tokens=np.array(captured["speech_tokens"], dtype=np.int64),
         mu=captured["mu"],
         mask=captured["mask"],
         spks=captured["spks"],
         cond=captured["cond"],
         z_init=captured["z_init"],
         t_span=captured["t_span"],
         final_x=captured["final_x"],
         n_timesteps=np.array([captured["n_timesteps"]]),
         t_scheduler=np.array([captured["t_scheduler"]]),
         inference_cfg_rate=np.array([captured["inference_cfg_rate"]]),
)
print(f"saved {OUT_DIR}/upstream.npz", flush=True)
print(f"  speech_tokens: {len(captured['speech_tokens'])}")
print(f"  mu: {captured['mu'].shape}, spks: {None if captured['spks'] is None else captured['spks'].shape}, cond: {None if captured['cond'] is None else captured['cond'].shape}")
print(f"  n_timesteps: {captured['n_timesteps']}, scheduler: {captured['t_scheduler']}, cfg_rate: {captured['inference_cfg_rate']}")
print(f"  final_x: {captured['final_x'].shape}")
