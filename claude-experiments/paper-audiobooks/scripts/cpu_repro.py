"""Reproduce the chatterbox muffled-output bug on CPU.

Goal: distinguish "GPU/MIOpen kernel choice causes the bug" from "the bug
is in chatterbox itself / model weights / sampling." If CPU produces the
same anomaly rate the GPU runs do (~17-50%), the GPU is exonerated and we
look at the model. If CPU is clean (0/N), the working theory in
CHATTERBOX_DEBUG.md (ROCm SDPA / MIOpen autotune) gets stronger.

Run inside the chatterbox venv directly — this script does NOT use the
subprocess JSON protocol the production backend uses. Fewer moving parts:

    ~/.cache/paper-audiobooks/venvs/chatterbox/bin/python scripts/cpu_repro.py

Output: 15 wavs in ./cpu_repro_clips/ + per-trial detector scores printed
to stdout.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force CPU before importing torch so no HIP context is created.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np
import soundfile as sf
import torch
from scipy.signal import welch

assert not torch.cuda.is_available(), "CPU-only run requested but cuda is visible"

from chatterbox.tts import ChatterboxTTS


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

# Same 361-char chunk that produced the original muffled stretch in book 2.
CHUNK19 = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)

N_TRIALS = 15
START_TRIAL = 5  # resume after a power-cycle that lost trials 1-4 (wavs preserved on disk)
OUT_DIR = Path("cpu_repro_clips")


def stats(audio: np.ndarray, sr: int) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "300_to_2k": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "duration_s": len(audio) / sr,
    }


def is_anomaly(s: dict) -> bool:
    # Same detector rule used throughout CHATTERBOX_DEBUG.md.
    return (s["centroid_hz"] < 700 and s["below_300hz"] > 0.5) or s["rms"] < 0.04


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"torch={torch.__version__} device=cpu threads={torch.get_num_threads()}", flush=True)

    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cpu")
    print(f"[load] model loaded in {time.time()-t0:.1f}s on cpu, sr={model.sr}", flush=True)

    anomalies = 0
    summary: list[tuple[int, dict, bool]] = []

    for trial in range(START_TRIAL, N_TRIALS + 1):
        t0 = time.time()
        wav = model.generate(
            CHUNK19,
            audio_prompt_path=VOICE_REF,
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        gen_dt = time.time() - t0
        audio = wav.squeeze().cpu().numpy().astype("float32")
        path = OUT_DIR / f"cpu_trial{trial:02d}.wav"
        sf.write(path, audio, model.sr)
        s = stats(audio, model.sr)
        bad = is_anomaly(s)
        if bad:
            anomalies += 1
        flag = "ANOMALY" if bad else "ok"
        print(
            f"  trial {trial:2d}: gen={gen_dt:5.1f}s dur={s['duration_s']:.1f}s "
            f"rms={s['rms']:.4f} centroid={s['centroid_hz']:4.0f}Hz "
            f"<300Hz={s['below_300hz']:.3f} 300-2k={s['300_to_2k']:.3f}  [{flag}]",
            flush=True,
        )
        summary.append((trial, s, bad))

    n_run = N_TRIALS - START_TRIAL + 1
    print(f"\nDone. {anomalies}/{n_run} anomalies on CPU (trials {START_TRIAL}-{N_TRIALS}).", flush=True)


if __name__ == "__main__":
    main()
