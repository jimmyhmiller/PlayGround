"""TTS worker subprocess.

Reads JSON {"text": str, "voice": str, "out_path": str, "backend": str} on stdin,
writes the synthesized audio to out_path as a float32 .npy at 24kHz, prints "ok"
on success.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def main() -> int:
    try:
        req = json.loads(sys.stdin.read())
        from paper_audiobooks.tts import get_backend, render_audio  # noqa: E402

        backend = get_backend(req.get("backend", "kokoro"))
        voice = req.get("voice") or backend.info.default_voice
        out = Path(req["out_path"])
        # Per-chunk cache lives next to the chapter cache file.
        chunk_cache_dir = out.parent / "chunks"
        audio = render_audio(
            req["text"], backend=backend, voice=voice,
            chunk_cache_dir=chunk_cache_dir,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, audio.astype(np.float32))
        print("ok")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
