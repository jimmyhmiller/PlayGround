"""Test whether forcing attn_implementation='eager' eliminates the muffled-output
anomalies we see at default settings.

Theory: ROCm's experimental SDPA kernels (Flash Efficient / Mem Efficient
attention) occasionally produce numerically incorrect attention scores, which
the autoregressive sampler commits to as bad speech tokens, which the s3gen
vocoder smears into muffled audio. Switching to eager attention removes those
experimental kernels.

Runs the same 361-char chunk, 12 trials, default sampling params, with eager
attention forced on the t3 transformer.
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading chatterbox on {device}", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Force eager attention on the t3 transformer (Llama backbone).
    # The model is already loaded, so we mutate the config and re-create
    # attention layers if needed. Simplest route: set both attrs and rely on
    # HF's runtime path-selection in scaled_dot_product_attention to honour
    # the config.
    cfg = model.t3.tfmr.config
    print(f"BEFORE: _attn_implementation={getattr(cfg, '_attn_implementation', None)} "
          f"output_attentions={cfg.output_attentions}", flush=True)
    cfg._attn_implementation = "eager"
    cfg.attn_implementation = "eager"
    # Some HF versions check this attribute on each layer too:
    for layer in model.t3.tfmr.layers:
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "eager"
    print(f"AFTER:  _attn_implementation={cfg._attn_implementation}", flush=True)

    out = Path("eager_attn_test")
    out.mkdir(exist_ok=True)
    model.prepare_conditionals(VOICE_REF, exaggeration=0.5)

    print(f"\nRunning {TRIALS} trials on chunk19 (361 chars) with eager attention\n", flush=True)
    anomalies = 0
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
        sr = int(model.sr)
        s = stats(audio, sr)
        anom = is_anomalous(s)
        if anom:
            anomalies += 1
        sf.write(out / f"trial{trial:02d}.wav", audio, sr)
        flag = " ANOMALY" if anom else ""
        print(f"  trial {trial:2d}: gen={elapsed:5.1f}s dur={s['dur']:5.1f}s "
              f"rms={s['rms']:.4f} cen={s['centroid']:5.0f}Hz "
              f"<300={s['below_300']:.3f}{flag}", flush=True)

    print(f"\n=== EAGER ATTN RESULT: {anomalies}/{TRIALS} anomalies ===", flush=True)
    print("(chatterbox SDPA baseline at default: 4/24 = ~17%)", flush=True)


if __name__ == "__main__":
    main()
