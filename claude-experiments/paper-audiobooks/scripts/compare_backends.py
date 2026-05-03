"""Render the same passage through every working TTS backend, concatenate to one mp3.

Each backend gets a Kokoro-spoken intro ("Now reading with X.") so you can hear the
boundaries without watching a clock.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from paper_audiobooks.tts import SAMPLE_RATE, get_backend, render_audio

PASSAGE = (
    "Various attempts have been made in recent years to state necessary and sufficient conditions "
    "for someone's knowing a given proposition. The attempts have often been such that they can be "
    "stated in a form similar to the following: a person S knows that proposition P if and only if "
    "P is true, S believes that P, and S is justified in believing that P."
)

OUT = Path("output/backend-comparison.mp3")
BRITISH_REF = Path("output/_british-ref.wav")

# Per-backend voice override. None = use the backend's default. For voice-cloning
# backends (chatterbox, f5) we can pass a path to a reference wav.
BACKENDS_TO_TRY = [
    ("kokoro", None),
    ("chatterbox", None),   # use Resemble's built-in default voice (no cloning)
    ("f5", None),           # use F5's bundled English example as reference
    # higgs intentionally excluded: the bosonai/higgs-audio-v2-tokenizer keys don't
    # match what transformers v5 expects, so the audio decoder loads uninitialized
    # and outputs near-silent noise. Re-enable when upstream is fixed.
]


def announce(text: str) -> np.ndarray:
    """Render the announcement using Kokoro (fast, deterministic)."""
    k = get_backend("kokoro")
    return render_audio(text, backend=k, voice="bm_daniel")


def make_british_ref() -> Path:
    """Pre-render a Kokoro bm_daniel clip we can hand to voice-cloning backends."""
    if BRITISH_REF.exists():
        return BRITISH_REF
    BRITISH_REF.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "The British Museum is one of the largest and most comprehensive collections in the world. "
        "It houses approximately eight million works dedicated to human history, art, and culture."
    )
    audio = announce(text)
    sf.write(BRITISH_REF, audio, SAMPLE_RATE)
    return BRITISH_REF


def main() -> int:
    pieces: list[np.ndarray] = []
    silence_short = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    silence_long = np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)

    make_british_ref()  # pre-render so chatterbox/f5 can clone the voice

    for name, voice_override in BACKENDS_TO_TRY:
        print(f"\n=== {name} ===")
        try:
            t0 = time.time()
            backend = get_backend(name)
            voice = voice_override or backend.info.default_voice
            intro = announce(f"Now reading with {name}.")
            pieces.append(intro)
            pieces.append(silence_short)
            audio = render_audio(PASSAGE, backend=backend, voice=voice)
            dur = len(audio) / SAMPLE_RATE
            print(f"  rendered {dur:.1f}s in {time.time()-t0:.1f}s")
            pieces.append(audio)
            pieces.append(silence_long)
        except Exception:
            print(f"  SKIP {name}: ", end="")
            traceback.print_exc()
            failed = announce(f"Skipping {name}, it failed.")
            pieces.append(failed)
            pieces.append(silence_long)

    if not pieces:
        print("nothing rendered")
        return 1
    full = np.concatenate(pieces)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(tmp_path, full, SAMPLE_RATE)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
             "-codec:a", "libmp3lame", "-b:a", "96k", str(OUT)],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"\nwrote {OUT} ({len(full)/SAMPLE_RATE:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
