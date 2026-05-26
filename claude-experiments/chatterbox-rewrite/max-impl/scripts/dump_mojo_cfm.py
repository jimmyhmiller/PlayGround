"""Re-feed upstream's speech_tokens into Mojo, dump our CFM intermediates."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CHATTERBOX_BF16"] = "1"
os.environ["CHATTERBOX_T3_FUSE_QKV"] = "1"
os.environ["CHATTERBOX_T3_FUSE_MLP"] = "1"
# Match upstream: 10 CFM steps.
os.environ["CHATTERBOX_CFM_STEPS"] = "10"

import numpy as np
from chatterbox_mojo import ChatterboxTTS

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
upstream = np.load("/tmp/cfm_diag/upstream.npz", allow_pickle=True)
upstream_tokens_raw = upstream["speech_tokens"].tolist()
# Mirror upstream's drop_invalid_tokens + <6561 filter.
EOS = 6562
upstream_tokens = []
for t in upstream_tokens_raw:
    if t == EOS:
        break
    if t < 6561:
        upstream_tokens.append(int(t))
print(f"loaded {len(upstream_tokens_raw)} raw, filtered to {len(upstream_tokens)}", flush=True)

tts = ChatterboxTTS.from_pretrained(use_bf16=True)
tts.prepare_conditionals(REF, exaggeration=0.5)

# Now hook into our _s3gen_decode to dump mu, cond, spks, final mel.
import chatterbox_mojo.tts as tts_mod

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
    mel_host = mel_out.to(cpu).to_numpy().reshape(1, 80, T_OUT_MEL).copy()
    captured["out_mel"] = mel_host
    captured["T_OUT_MEL"] = T_OUT_MEL
    captured["T_TOTAL_MEL"] = T_TOTAL_MEL
    captured["T_PROMPT_MEL"] = 500
    # Also dump spks + prompt_feat that we used.
    captured["spks_used"] = c.spks.to(cpu).to_numpy().reshape(1, 80).copy()
    captured["prompt_feat_used"] = c.prompt_feat.to(cpu).to_numpy().reshape(1, 500, 80).copy()
    captured["prompt_token_used"] = prompt_token_host.copy()

    T_HIFT = T_OUT_MEL * 120 + 1
    T_AUDIO = (T_HIFT - 1) * 4
    audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
    op_hift.forward(self._hift_h, mel_out, audio_out, 1, T_OUT_MEL)
    return audio_out.to(cpu).to_numpy().reshape(-1).copy()

tts_mod.ChatterboxTTS._s3gen_decode = patched_decode

# Call _s3gen_decode directly with the upstream tokens.
audio = tts._s3gen_decode(upstream_tokens)
print(f"mojo audio: {audio.size/24000:.2f}s", flush=True)

# Detect thumps.
diff = np.abs(np.diff(audio.astype(np.float32)))
big = np.flatnonzero(diff > 0.8)
print(f"mojo loud thumps in re-fed audio: {len(big)}", flush=True)

# Save.
np.savez("/tmp/cfm_diag/mojo.npz",
         speech_tokens=np.array(upstream_tokens, dtype=np.int64),
         audio=audio,
         out_mel=captured["out_mel"],
         T_OUT_MEL=np.array([captured["T_OUT_MEL"]]),
         T_TOTAL_MEL=np.array([captured["T_TOTAL_MEL"]]),
         T_PROMPT_MEL=np.array([captured["T_PROMPT_MEL"]]),
         spks_used=captured["spks_used"],
         prompt_feat_used=captured["prompt_feat_used"],
         prompt_token_used=captured["prompt_token_used"],
)
print("saved /tmp/cfm_diag/mojo.npz")
