"""Reproduce TTS distortion by re-synthesizing chunks of various lengths.

Theory: chatterbox produces muffled/low-frequency-dominated audio when given
chunks longer than its safe context (~250-300 chars). To verify, render the
same Plantinga sentence as one long 394-char chunk vs split into 2 shorter
chunks on a comma boundary, and compare spectral content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

from paper_audiobooks.tts import get_backend, SAMPLE_RATE


VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

# The exact 394-char chunk from the audiobook that landed in the distorted region.
LONG_CHUNK = (
    "Many have argued that the Christian doctrine of three divine persons with one nature "
    "cannot be coherently stated; many have claimed that it is not logically possible that "
    "a human being, Jesus of Nazareth, should also be the second person of the divine "
    "Trinity, and many have thought it impossible that one person's suffering, even if "
    "that person is divine, should atone for someone else's sins."
)

# Two-way split of the same sentence on the second semicolon.
SHORT_A = (
    "Many have argued that the Christian doctrine of three divine persons with one nature "
    "cannot be coherently stated; many have claimed that it is not logically possible that "
    "a human being, Jesus of Nazareth, should also be the second person of the divine Trinity."
)
SHORT_B = (
    "And many have thought it impossible that one person's suffering, even if that person "
    "is divine, should atone for someone else's sins."
)


def low_freq_dom(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    f, p = welch(audio, sr, nperseg=2048)
    tot = float(np.sum(p)) + 1e-12
    return {
        "centroid_hz": float(np.sum(f * p) / tot),
        "below_300hz": float(np.sum(p[f < 300]) / tot),
        "300_to_2k": float(np.sum(p[(f >= 300) & (f < 2000)]) / tot),
        "rms": float(np.sqrt(np.mean(audio**2))),
        "duration_s": len(audio) / sr,
    }


def main() -> None:
    out_dir = Path("repro_clips")
    out_dir.mkdir(exist_ok=True)
    backend = get_backend("chatterbox")

    cases = [
        ("long_394char", LONG_CHUNK),
        ("short_a_249char", SHORT_A),
        ("short_b_141char", SHORT_B),
    ]
    for name, text in cases:
        print(f"\n=== {name} ({len(text)} chars) ===")
        print(f"  text: {text[:80]!r}...")
        audio = backend.synthesize_chunk(text, voice=VOICE_REF)
        path = out_dir / f"{name}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        stats = low_freq_dom(audio)
        print(f"  {path}")
        print(f"  duration={stats['duration_s']:.2f}s rms={stats['rms']:.4f}")
        print(f"  spectral centroid: {stats['centroid_hz']:.0f} Hz")
        print(f"  energy <300Hz: {stats['below_300hz']:.3f}, 300-2k: {stats['300_to_2k']:.3f}")


if __name__ == "__main__":
    main()
