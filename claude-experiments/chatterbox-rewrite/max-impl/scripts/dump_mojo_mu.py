"""Run Mojo's flow encoder on Mojo's bad tokens. Save mu/spks/cond/x_init.
We patch op_flow's forward to also capture mu before CFM runs.
Simplest: call op_flow with our handle, capture mu via instrumented patched_decode.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['CHATTERBOX_BF16'] = '1'
os.environ['CHATTERBOX_CFM_STEPS'] = '5'
os.environ['CHATTERBOX_T3_FUSE_QKV'] = '1'
os.environ['CHATTERBOX_T3_FUSE_MLP'] = '1'

import numpy as np
from chatterbox_mojo import ChatterboxTTS
import chatterbox_mojo.tts as tts_mod

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
mojo_tokens = np.load('/tmp/cfm_diag/mojo_bad_tokens.npy').tolist()
print(f"loaded mojo tokens: {len(mojo_tokens)}", flush=True)

# Instrument _s3gen_decode to capture mu, cond, spks, x_init.
# To get mu separately, we need to call upsample encoder directly OR add a hook in op_flow.
# Easier: do the full op_flow call BUT also do a separate call to grab just mu.
# Simpler still: just compare the final mel and CFM inputs we can already grab.

captured = {}

def patched_decode(self, speech_tokens, cfg_rate=0.7, n_steps=0, noise_seed=0xC0FFEE):
    import os as _os
    if n_steps == 0:
        n_steps = int(_os.environ.get("CHATTERBOX_CFM_STEPS", 5))
    from max.driver import Buffer
    from max.dtype import DType
    gpu, cpu = self._gpu, self._cpu
    c = self.conds
    T_GEN = len(speech_tokens)
    T_TOTAL_TOKEN = 250 + T_GEN
    T_TOTAL_MEL = 2 * T_TOTAL_TOKEN
    T_OUT_MEL = T_TOTAL_MEL - 500

    prompt_token_host = c.prompt_token.to(cpu).to_numpy().reshape(-1)
    tok_combined = np.concatenate([
        prompt_token_host.astype(np.int64),
        np.array(speech_tokens, dtype=np.int64),
    ]).reshape(1, T_TOTAL_TOKEN)
    tok_buf_i64 = Buffer.from_numpy(tok_combined).to(gpu)

    mel_out = Buffer(shape=(1, 80, T_OUT_MEL), dtype=DType.float32, device=gpu)
    flow_config = {
        "B": 1, "T_token": T_TOTAL_TOKEN, "T_prompt_mel": 500,
        "T_total_mel": T_TOTAL_MEL, "T_out_mel": T_OUT_MEL,
        "n_steps": int(n_steps), "cfg_rate": float(cfg_rate),
        "noise_seed": int(noise_seed),
    }
    import op_flow, op_hift
    op_flow.forward(self._flow_h, tok_buf_i64, c.spks, c.prompt_feat, mel_out, flow_config)
    captured["final_mel"] = mel_out.to(cpu).to_numpy().reshape(1, 80, T_OUT_MEL).copy()
    captured["spks"] = c.spks.to(cpu).to_numpy().reshape(1, 80).copy()
    captured["prompt_feat"] = c.prompt_feat.to(cpu).to_numpy().reshape(1, 500, 80).copy()
    captured["tokens"] = tok_combined.copy()
    captured["T_TOTAL_MEL"] = T_TOTAL_MEL

    T_HIFT = T_OUT_MEL * 120 + 1
    T_AUDIO = (T_HIFT - 1) * 4
    audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
    op_hift.forward(self._hift_h, mel_out, audio_out, 1, T_OUT_MEL)
    return audio_out.to(cpu).to_numpy().reshape(-1).copy()

tts_mod.ChatterboxTTS._s3gen_decode = patched_decode

tts = ChatterboxTTS.from_pretrained(use_bf16=True)
tts.prepare_conditionals(REF, exaggeration=0.5)
audio = tts._s3gen_decode(mojo_tokens)
print(f"mojo audio: {audio.size/24000:.2f}s", flush=True)

diff = np.abs(np.diff(audio))
print(f"mojo loud thumps: {int(np.sum(diff>0.8))}", flush=True)

np.savez("/tmp/cfm_diag/mojo_on_mojo.npz",
         final_mel=captured["final_mel"], spks=captured["spks"],
         prompt_feat=captured["prompt_feat"], tokens=captured["tokens"])
print("saved /tmp/cfm_diag/mojo_on_mojo.npz", flush=True)
