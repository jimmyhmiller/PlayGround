"""Generate a single mp3 sampling every Kokoro voice.

Each voice speaks:
  1. its own id
  2. a short neutral sample line
then a brief pause before the next voice.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi
from kokoro import KPipeline

SAMPLE_RATE = 24000
SAMPLE_LINE = (
    "Knowledge is justified true belief, except when it is not. "
    "This is a short sample of my voice."
)
OUT = Path("output/voice_samples.mp3")

LANG_PREFIX_TO_CODE = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Brazilian Portuguese
    "z": "z",  # Mandarin
}


def list_voices() -> list[str]:
    files = HfApi().list_repo_files("hexgrad/Kokoro-82M")
    all_voices = sorted(f.replace("voices/", "").replace(".pt", "") for f in files if f.startswith("voices/"))
    return [v for v in all_voices if v[0] in ("a", "b")]


def say_name(voice_id: str) -> str:
    """Render voice id as a natural spoken phrase."""
    return voice_id.replace("_", " ")


def synth(pipeline: KPipeline, text: str, voice: str) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text, voice=voice):
        pieces.append(np.asarray(audio, dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)


def main() -> None:
    voices = list_voices()
    print(f"sampling {len(voices)} voices")

    pipelines: dict[str, KPipeline | None] = {}
    gap_short = np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.float32)
    gap_long = np.zeros(int(SAMPLE_RATE * 0.9), dtype=np.float32)

    all_pieces: list[np.ndarray] = []
    skipped: list[tuple[str, str]] = []
    for i, voice in enumerate(voices, 1):
        lang = LANG_PREFIX_TO_CODE.get(voice[0], "a")
        if lang not in pipelines:
            try:
                pipelines[lang] = KPipeline(lang_code=lang)
            except Exception as exc:
                pipelines[lang] = None
                print(f"  !! lang {lang!r} unavailable: {exc}")
        pipe = pipelines[lang]
        if pipe is None:
            skipped.append((voice, f"lang {lang!r} unavailable"))
            continue
        print(f"[{i:>2}/{len(voices)}] {voice}")
        try:
            name_audio = synth(pipe, say_name(voice), voice)
            line_audio = synth(pipe, SAMPLE_LINE, voice)
        except Exception as exc:
            print(f"  !! {voice} failed: {exc}")
            skipped.append((voice, str(exc)))
            continue
        all_pieces.extend([name_audio, gap_short, line_audio, gap_long])

    full = np.concatenate(all_pieces) if all_pieces else np.zeros(1, dtype=np.float32)
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
    if skipped:
        print(f"skipped {len(skipped)}:")
        for v, why in skipped:
            print(f"  {v}: {why}")


if __name__ == "__main__":
    main()
