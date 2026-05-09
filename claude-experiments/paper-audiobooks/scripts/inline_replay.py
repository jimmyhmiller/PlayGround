"""Capture-and-replay-immediately: run model.generate() to produce both an
anomalous waveform and its speech tokens, then in the same process and
immediately, re-call s3gen on those exact tokens. If the immediate replay
produces clean audio, the muffled-output bug is in the *first* s3gen call
made by generate(), not in the tokens themselves.

This is the same idea as bisect_t3_s3gen.py but tighter: no break between
producing the anomalous output and replaying.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import welch

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
CHUNK = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)
MAX_TRIALS = 30  # keep going until we find an anomalous run


def stats(audio: np.ndarray, sr: int) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid": float(np.sum(f * p) / tot),
        "below_300": float(np.sum(p[f < 300]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "dur": len(audio) / sr,
    }


def is_anomalous(s: dict) -> bool:
    return (s["centroid"] < 700 and s["below_300"] > 0.5) or s["rms"] < 0.04


def gen_with_capture(model: ChatterboxTTS) -> tuple[torch.Tensor, np.ndarray]:
    """Replicate model.generate() and return (speech_tokens, raw_wav). No
    watermarker."""
    text = punc_norm(CHUNK)
    text_tokens = model.tokenizer.text_to_tokens(text).to(model.device)
    cfg_weight = 0.5
    if cfg_weight > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3, text_tokens=text_tokens,
            max_new_tokens=1000, temperature=0.8, cfg_weight=cfg_weight,
            repetition_penalty=1.2, min_p=0.05, top_p=1.0,
        )
        speech_tokens = speech_tokens[0]
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)

        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens, ref_dict=model.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
    return speech_tokens.detach().cpu(), wav


def replay_s3gen(model: ChatterboxTTS, tokens: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        st = tokens.to(model.device)
        wav, _ = model.s3gen.inference(speech_tokens=st, ref_dict=model.conds.gen)
        return wav.squeeze(0).detach().cpu().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    out = Path("inline_replay_results")
    out.mkdir(exist_ok=True)
    sr = int(model.sr)

    print(f"\nGenerating until we hit an anomaly (max {MAX_TRIALS} attempts)\n",
          flush=True)
    for trial in range(1, MAX_TRIALS + 1):
        t0 = time.time()
        tokens, wav = gen_with_capture(model)
        elapsed = time.time() - t0
        s = stats(wav, sr)
        anom = is_anomalous(s)
        sf.write(out / f"trial{trial:02d}_orig.wav", wav, sr)
        np.save(out / f"trial{trial:02d}_tokens.npy", tokens.numpy())
        flag = " ANOMALY" if anom else ""
        print(f"trial {trial:2d}: gen={elapsed:5.1f}s n_tokens={len(tokens):4d} "
              f"rms={s['rms']:.4f} cen={s['centroid']:5.0f}Hz "
              f"<300={s['below_300']:.3f}{flag}", flush=True)
        if not anom:
            continue

        # ANOMALY — immediately replay 5 times in same process, same conds.
        print(f"  >>> anomalous trial — immediately replaying s3gen 5x with same tokens", flush=True)
        for r in range(1, 6):
            replay_wav = replay_s3gen(model, tokens)
            rs = stats(replay_wav, sr)
            r_anom = is_anomalous(rs)
            sf.write(out / f"trial{trial:02d}_replay{r}.wav", replay_wav, sr)
            r_flag = " ANOMALY" if r_anom else " clean"
            print(f"    replay {r}: rms={rs['rms']:.4f} cen={rs['centroid']:5.0f}Hz "
                  f"<300={rs['below_300']:.3f}{r_flag}", flush=True)
        # Done — we have the data we need.
        print("\n=== DIAGNOSTIC ===", flush=True)
        print("  If all 5 replays are clean: the bug is in the FIRST s3gen call", flush=True)
        print("    inside generate() — something about that specific call differs", flush=True)
        print("    from a fresh s3gen.inference() on the same tokens.", flush=True)
        print("  If replays are also anomalous: bug is in the tokens themselves;", flush=True)
        print("    s3gen deterministic from tokens. Means t3 is the bug.", flush=True)
        return

    print(f"\nNo anomaly in {MAX_TRIALS} trials. Try again or increase MAX_TRIALS.")


if __name__ == "__main__":
    main()
