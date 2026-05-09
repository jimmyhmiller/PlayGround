"""Test whether short (<250 char) chunks also produce occasional distortion.

Theory under test: chatterbox issue #424 says hallucinations only happen above
~350 chars. If true, short chunks should ALWAYS come out clean. Run several
short chunks at multiple lengths, many trials each, score each output for the
"muffled" signature (low spectral centroid + high <300Hz energy + low RMS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from paper_audiobooks.tts import get_backend, SAMPLE_RATE


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

# Real text from the Plantinga preface, hand-trimmed to specific lengths.
SHORT_50 = "This book is about Christian belief and warrant."  # 48 chars
SHORT_100 = (
    "This book is about Christian belief and warrant. Plantinga argues for proper function."
)  # 86 chars
SHORT_150 = (
    "This book is about Christian belief and warrant. Plantinga argues for proper function "
    "as the foundation of epistemology."
)  # 119 chars
SHORT_200 = (
    "This book is about Christian belief and warrant. Plantinga argues that proper function "
    "is the foundation of warrant, which in turn is what distinguishes knowledge from belief."
)  # 175 chars
SHORT_250 = (
    "This book is about Christian belief and warrant. Plantinga argues that proper function "
    "is the foundation of warrant, which in turn is what distinguishes knowledge from belief, "
    "and warrant is necessary for genuine knowledge."
)  # 224 chars

CASES = [
    ("len_50", SHORT_50),
    ("len_100", SHORT_100),
    ("len_150", SHORT_150),
    ("len_200", SHORT_200),
    ("len_250", SHORT_250),
]
TRIALS = 8


def stats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
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
    """Match the muffled signature: low centroid, low-freq dominance, lower rms."""
    return (s["centroid"] < 700 and s["below_300"] > 0.5) or s["rms"] < 0.04


def main() -> None:
    out = Path("repro_clips_short")
    out.mkdir(exist_ok=True)
    backend = get_backend("chatterbox")

    print(f"\nTesting {len(CASES)} text lengths × {TRIALS} trials each\n")
    summary: list[tuple[str, int, int, list[dict]]] = []
    for name, text in CASES:
        print(f"\n=== {name} ({len(text)} chars) ===")
        results: list[dict] = []
        anomalies = 0
        for trial in range(1, TRIALS + 1):
            audio = backend.synthesize_chunk(text, voice=VOICE_REF)
            path = out / f"{name}_t{trial}.wav"
            sf.write(path, audio, SAMPLE_RATE)
            s = stats(audio)
            results.append(s)
            flag = " ANOMALY" if is_anomalous(s) else ""
            if is_anomalous(s):
                anomalies += 1
            print(f"  trial {trial}: dur={s['dur']:5.1f}s rms={s['rms']:.4f} "
                  f"cen={s['centroid']:5.0f}Hz <300={s['below_300']:.3f} "
                  f"mid={s['mid']:.3f}{flag}")
        summary.append((name, len(text), anomalies, results))

    print("\n=== SUMMARY ===")
    print(f"{'name':<10} {'chars':>6} {'anomalies':>10} / {TRIALS}")
    for name, n_chars, anomalies, _ in summary:
        print(f"{name:<10} {n_chars:>6} {anomalies:>10} / {TRIALS}")


if __name__ == "__main__":
    main()
