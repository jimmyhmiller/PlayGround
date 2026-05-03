"""Generate a longer comparison clip for selected voices."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from paper_audiobooks.tts import _chunk_text

SAMPLE_RATE = 24000

VOICES = ["am_michael", "bm_daniel"]
SCRIPT_PATH = Path("output/gettier.script.md")
OUT = Path("output/long_samples.mp3")
MAX_CHARS = 2400


def synth_voice(voice: str, text: str) -> np.ndarray:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=voice[0])
    pieces: list[np.ndarray] = []
    intro = f"Sample for voice {voice.replace('_', ' ')}."
    for chunk in _chunk_text(intro):
        for _gs, _ps, audio in pipeline(chunk, voice=voice):
            pieces.append(np.asarray(audio, dtype=np.float32))
    pieces.append(np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.float32))
    for chunk in _chunk_text(text):
        for _gs, _ps, audio in pipeline(chunk, voice=voice):
            pieces.append(np.asarray(audio, dtype=np.float32))
        pieces.append(np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)


def main() -> None:
    if not SCRIPT_PATH.exists():
        sys.exit(f"missing {SCRIPT_PATH}")
    text = SCRIPT_PATH.read_text().strip()[:MAX_CHARS]

    gap = np.zeros(int(SAMPLE_RATE * 1.2), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for i, voice in enumerate(VOICES):
        print(f"[{i+1}/{len(VOICES)}] {voice}")
        pieces.append(synth_voice(voice, text))
        if i < len(VOICES) - 1:
            pieces.append(gap)

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
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
