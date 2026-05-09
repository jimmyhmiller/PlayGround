"""Sweep cfg_weight × top_p on chatterbox to find settings that eliminate
muffled-output anomalies on the same 361-char chunk that fails at default
settings.

Theory: extrapolating CFG (cfg_weight=0.5) plus loose top_p (=1.0) lets the
sampler occasionally pick logit configurations that produce muffled audio.
Lower cfg + tighter top_p should reduce the anomaly rate.
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    out_dir = Path("cfg_topp_sweep")
    out_dir.mkdir(exist_ok=True)

    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    settings = [
        # cfg=0.0 is broken in chatterbox (bos_embed mismatch); skip it.
        (0.1, 1.0),  # near-zero CFG — minimal extrapolation
        (0.25, 1.0),
        (0.25, 0.9),
        (0.5, 1.0),  # current default
        (0.5, 0.9),
        (0.5, 0.8),  # tighter top_p to prune long tail more
    ]
    trials = 6

    summary = []
    for cfg, tp in settings:
        print(f"\n=== cfg={cfg} top_p={tp} ===", flush=True)
        anomalies = 0
        for trial in range(1, trials + 1):
            with torch.inference_mode():
                wav = model.generate(
                    CHUNK,
                    audio_prompt_path=VOICE_REF,
                    exaggeration=0.5,
                    cfg_weight=cfg,
                    temperature=0.8,
                    top_p=tp,
                )
            audio = wav.squeeze().cpu().numpy().astype("float32")
            sr = int(model.sr)
            s = stats(audio, sr)
            anom = is_anomalous(s)
            if anom:
                anomalies += 1
            sf.write(out_dir / f"cfg{cfg}_tp{tp}_t{trial}.wav", audio, sr)
            flag = " ANOMALY" if anom else ""
            print(f"  trial {trial}: dur={s['dur']:5.1f}s rms={s['rms']:.4f} "
                  f"cen={s['centroid']:5.0f}Hz <300={s['below_300']:.3f}{flag}",
                  flush=True)
        summary.append((cfg, tp, anomalies))

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'cfg':>5} {'top_p':>6} {'anomalies':>10}")
    for cfg, tp, anom in summary:
        print(f"{cfg:>5} {tp:>6} {anom:>5}/{trials}")


if __name__ == "__main__":
    main()
