"""Run model.generate() 100 times on the same chunk and count anomalies.

If anomaly rate is ~0/100, the bug appears to have been fixed (somehow) and
we need to figure out what changed. If it's nonzero, we still have a
reproducible bug to track.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import welch

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
TRIALS = 100


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
    # Two failure modes seen in earlier runs: muffled (low centroid + high <300)
    # and rushed/whisper (low rms or unusually high centroid). Catch both.
    if s["rms"] < 0.04:
        return True
    if s["centroid"] < 700 and s["below_300"] > 0.5:
        return True
    if s["centroid"] > 1300 and s["rms"] > 0.15:  # the loud-shouty cap-hit mode
        return True
    return False


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    out = Path("big_run_100")
    out.mkdir(exist_ok=True)
    sr = int(model.sr)

    print(f"\nRunning {TRIALS} trials...\n", flush=True)
    anomalies = 0
    anomaly_trials: list[int] = []
    rms_list, cen_list = [], []
    for trial in range(1, TRIALS + 1):
        t0 = time.time()
        with torch.inference_mode():
            wav = model.generate(
                CHUNK,
                audio_prompt_path=VOICE_REF,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
            )
        elapsed = time.time() - t0
        audio = wav.squeeze().cpu().numpy().astype("float32")
        s = stats(audio, sr)
        anom = is_anomalous(s)
        if anom:
            anomalies += 1
            anomaly_trials.append(trial)
            sf.write(out / f"trial{trial:03d}_ANOMALY.wav", audio, sr)
        else:
            sf.write(out / f"trial{trial:03d}.wav", audio, sr)
        rms_list.append(s["rms"])
        cen_list.append(s["centroid"])
        flag = " ANOMALY" if anom else ""
        if trial % 10 == 0 or anom:
            print(f"trial {trial:3d}: gen={elapsed:5.1f}s rms={s['rms']:.4f} "
                  f"cen={s['centroid']:5.0f}Hz <300={s['below_300']:.3f}{flag}",
                  flush=True)

    print(f"\n=== SUMMARY ===")
    print(f"  total trials: {TRIALS}")
    print(f"  anomalies: {anomalies}/{TRIALS} ({anomalies/TRIALS*100:.1f}%)")
    if anomaly_trials:
        print(f"  anomaly trials: {anomaly_trials}")
    print(f"  rms: mean={np.mean(rms_list):.4f} std={np.std(rms_list):.4f}")
    print(f"  cen: mean={np.mean(cen_list):.0f}Hz std={np.std(cen_list):.0f}")


if __name__ == "__main__":
    main()
