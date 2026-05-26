"""Render Quine chunks one by one, capturing mel + audio per chunk.
When a chunk produces a LOUD thump (|step| > 0.8), save the mel + audio.
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CHATTERBOX_BF16"] = "1"
os.environ["CHATTERBOX_CFM_STEPS"] = "5"
os.environ["CHATTERBOX_T3_FUSE_QKV"] = "1"
os.environ["CHATTERBOX_T3_FUSE_MLP"] = "1"

import numpy as np
from chatterbox_mojo import ChatterboxTTS
import chatterbox_mojo.tts as tts_mod

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

captured = {"mel": None}

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
    captured["mel"] = mel_out.to(cpu).to_numpy().reshape(80, T_OUT_MEL).copy()
    T_HIFT = T_OUT_MEL * 120 + 1
    T_AUDIO = (T_HIFT - 1) * 4
    audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
    op_hift.forward(self._hift_h, mel_out, audio_out, 1, T_OUT_MEL)
    return audio_out.to(cpu).to_numpy().reshape(-1).copy()

tts_mod.ChatterboxTTS._s3gen_decode = patched_decode


def find_loud_thumps(audio, sr=24000, thresh=0.8):
    diff = np.abs(np.diff(audio.astype(np.float32)))
    big = np.flatnonzero(diff > thresh)
    if len(big) == 0:
        return []
    gaps = np.diff(big) if len(big) > 1 else np.array([])
    starts = np.concatenate([[0], np.flatnonzero(gaps > sr // 20) + 1]) if len(big) > 1 else np.array([0])
    return [(int(big[s]), float(diff[big[s]])) for s in starts]


def main():
    body = json.load(open("/home/jimmyhmiller/audiobooks/m4b/Quine-Epistemology-Naturalized.chapters.json"))[0]["body"]
    body = body[:15000]
    last = body.rfind(". ")
    body = body[:last+1]

    # Naive chunking by sentence then by size cap (matches paper-audiobooks behavior closely enough).
    import re
    sents = re.split(r'(?<=[.!?])\s+', body)
    chunks = []
    cur = ""
    for s in sents:
        if not s.strip(): continue
        if cur and len(cur) + 1 + len(s) > 250:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur.strip())
    print(f"[diag] {len(chunks)} chunks", flush=True)

    tts = ChatterboxTTS.from_pretrained(use_bf16=True)
    tts.prepare_conditionals(REF, exaggeration=0.5)

    out_dir = Path("/tmp/loud_thumps")
    out_dir.mkdir(exist_ok=True)
    n_loud = 0
    for i, text in enumerate(chunks):
        captured["mel"] = None
        wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                          min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
                          rng_seed=0xDEADBEEF + i)
        audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
        mel = captured["mel"]
        loud = find_loud_thumps(audio)
        if loud:
            n_loud += 1
            max_md = max(d for _, d in loud)
            print(f"  chunk {i:>3} ({len(text):>3} chars): {len(loud)} loud, max|d|={max_md:.2f}: {text[:60]!r}", flush=True)
            np.savez(str(out_dir / f"chunk_{i:03d}.npz"),
                     mel=mel, audio=audio,
                     thumps=np.array([s for s, _ in loud], dtype=np.int64),
                     text=np.array([text]))
    print(f"\n[diag] {n_loud}/{len(chunks)} chunks had loud thumps", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
