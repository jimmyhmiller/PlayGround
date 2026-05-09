"""Hypothesis: t3 leaves GPU memory in a fragmented/noisy state that causes
s3gen's MIOpen kernel selection to pick a different (occasionally buggy)
kernel for the first call. Clearing the cache between t3 and s3gen should
mimic the "clean replay" GPU state and eliminate anomalies.

Method: replicate model.generate() but call torch.cuda.empty_cache() between
the t3 call and the s3gen call. Run many trials, count anomalies.

If anomaly rate drops to ~0%, GPU memory state is the cause. If it stays
the same as baseline (~17%), the cache isn't the issue.
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
TRIALS = 12


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


def gen_with_cache_clear(model: ChatterboxTTS, *, clear_cache: bool) -> np.ndarray:
    """Replicate generate() but optionally torch.cuda.empty_cache() between t3 and s3gen."""
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

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens, ref_dict=model.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
    return wav


def run_block(model: ChatterboxTTS, label: str, *, clear_cache: bool) -> int:
    sr = int(model.sr)
    out = Path("clear_cache_test") / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label}: {TRIALS} trials, clear_cache={clear_cache} ===\n", flush=True)
    anomalies = 0
    for trial in range(1, TRIALS + 1):
        t0 = time.time()
        wav = gen_with_cache_clear(model, clear_cache=clear_cache)
        elapsed = time.time() - t0
        s = stats(wav, sr)
        anom = is_anomalous(s)
        if anom:
            anomalies += 1
        sf.write(out / f"trial{trial:02d}.wav", wav, sr)
        flag = " ANOMALY" if anom else ""
        print(f"  trial {trial:2d}: gen={elapsed:5.1f}s rms={s['rms']:.4f} "
              f"cen={s['centroid']:5.0f}Hz <300={s['below_300']:.3f}{flag}",
              flush=True)
    return anomalies


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    no_clear = run_block(model, "A_no_clear", clear_cache=False)
    with_clear = run_block(model, "B_with_clear", clear_cache=True)

    print(f"\n=== SUMMARY ===")
    print(f"  WITHOUT empty_cache: {no_clear}/{TRIALS} anomalies")
    print(f"  WITH    empty_cache: {with_clear}/{TRIALS} anomalies")
    print()
    print("If with_clear is much lower: GPU memory state IS the cause.")
    print("If similar: GPU cache isn't the discriminator.")


if __name__ == "__main__":
    main()
