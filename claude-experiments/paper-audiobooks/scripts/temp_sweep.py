"""Temperature sweep for chatterbox: does lower temp reduce anomalies?

Runs the same 361-char chunk at multiple temperatures, multiple trials each,
and scores each output for the muffled signature. Bypasses our subprocess
wrapper and calls chatterbox directly so we can pass temperature/seed.

Run from the chatterbox venv (it has chatterbox + torch installed):
  /home/jimmyhmiller/.cache/paper-audiobooks/venvs/chatterbox/bin/python \
    scripts/temp_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import welch

# Reduce HIP allocator fragmentation, same as the production worker.
import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
CHUNK = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)


def stats(audio: np.ndarray, sr: int) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid": float(np.sum(f * p) / tot),
        "below_300": float(np.sum(p[f < 300]) / tot),
        "mid": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "dur": len(audio) / sr,
    }


def is_anomalous(s: dict) -> bool:
    return (s["centroid"] < 700 and s["below_300"] > 0.5) or s["rms"] < 0.04


def main() -> None:
    print(f"loading chatterbox on {torch.cuda.is_available() and 'cuda' or 'cpu'}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    out_dir = Path("temp_sweep")
    out_dir.mkdir(exist_ok=True)

    # Load voice conditionals once
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    temps = [0.4, 0.6, 0.8, 1.0]
    trials = 6

    summary = []
    for temp in temps:
        print(f"\n=== temperature={temp} ===", flush=True)
        anomalies = 0
        for trial in range(1, trials + 1):
            with torch.inference_mode():
                wav = model.generate(
                    CHUNK,
                    audio_prompt_path=VOICE_REF,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=temp,
                )
            audio = wav.squeeze().cpu().numpy().astype("float32")
            sr = int(model.sr)
            s = stats(audio, sr)
            anom = is_anomalous(s)
            if anom:
                anomalies += 1
            path = out_dir / f"temp{temp}_t{trial}.wav"
            sf.write(path, audio, sr)
            flag = " ANOMALY" if anom else ""
            print(f"  trial {trial}: dur={s['dur']:5.1f}s rms={s['rms']:.4f} "
                  f"cen={s['centroid']:5.0f}Hz <300={s['below_300']:.3f} "
                  f"mid={s['mid']:.3f}{flag}", flush=True)
        summary.append((temp, anomalies))

    print("\n=== SUMMARY ===", flush=True)
    for temp, anom in summary:
        print(f"  temperature={temp}: {anom}/{trials} anomalies", flush=True)


if __name__ == "__main__":
    main()
