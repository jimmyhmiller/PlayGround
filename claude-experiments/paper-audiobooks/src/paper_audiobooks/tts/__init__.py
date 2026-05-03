"""TTS — text → audio with pluggable backends.

Public surface:
- SAMPLE_RATE: target sample rate for all backends
- chunk_text(text, max_chars): shared sentence chunking
- get_backend(name) / list_backends(): registry access
- render_audio(text, *, backend, voice): synthesize via a Backend instance
- synthesize(text, out_path, *, backend_name, voice): one-shot CLI helper
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .backends.base import Backend, BackendInfo

SAMPLE_RATE = 24000


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text on sentence boundaries into chunks at most max_chars long.

    Most TTS backends have a per-call context limit; this keeps each call safe.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def render_audio(text: str, *, backend: Backend, voice: str) -> np.ndarray:
    """Synthesize text via a loaded Backend, returning a float32 waveform at SAMPLE_RATE."""
    import sys, time
    pieces: list[np.ndarray] = []
    silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
    chunks = chunk_text(text, max_chars=backend.info.max_chunk_chars)
    print(f"  [render] {len(chunks)} chunk(s), max_chunk_chars={backend.info.max_chunk_chars}", file=sys.stderr, flush=True)
    for i, chunk in enumerate(chunks, 1):
        t0 = time.time()
        print(f"  [render] chunk {i}/{len(chunks)} ({len(chunk)} chars): start", file=sys.stderr, flush=True)
        audio = backend.synthesize_chunk(chunk, voice=voice)
        dt = time.time() - t0
        print(f"  [render] chunk {i}/{len(chunks)} ({len(chunk)} chars): "
              f"{len(audio)/SAMPLE_RATE:.1f}s audio in {dt:.1f}s wall", file=sys.stderr, flush=True)
        pieces.append(audio.astype(np.float32))
        pieces.append(silence)
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)


def get_backend(name: str) -> Backend:
    """Instantiate a backend by name. Loads model on first call."""
    cls = _registry()[name]
    return cls()


def list_backends() -> list[BackendInfo]:
    return [cls.info for cls in _registry().values()]


def _registry() -> dict[str, type[Backend]]:
    # Lazy imports so users without optional deps can still use the others.
    from .backends import kokoro as kokoro_mod
    reg: dict[str, type[Backend]] = {kokoro_mod.KokoroBackend.info.name: kokoro_mod.KokoroBackend}
    try:
        from .backends import higgs as higgs_mod
        reg[higgs_mod.HiggsBackend.info.name] = higgs_mod.HiggsBackend
    except Exception:
        pass
    try:
        from .backends import chatterbox as chatterbox_mod
        reg[chatterbox_mod.ChatterboxBackend.info.name] = chatterbox_mod.ChatterboxBackend
    except Exception:
        pass
    try:
        from .backends import f5 as f5_mod
        reg[f5_mod.F5Backend.info.name] = f5_mod.F5Backend
    except Exception:
        pass
    return reg


def synthesize(
    text: str,
    out_path: Path,
    *,
    backend_name: str = "kokoro",
    voice: str | None = None,
    bitrate: str = "96k",
) -> Path:
    """Render text to an audio file using the named backend."""
    backend = get_backend(backend_name)
    if voice is None:
        voice = backend.info.default_voice
    full = render_audio(text, backend=backend, voice=voice)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(tmp_path, full, SAMPLE_RATE)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
                 "-codec:a", "libmp3lame", "-b:a", bitrate, str(out_path)],
                check=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        sf.write(out_path, full, SAMPLE_RATE)
    return out_path


# Back-compat shims so existing callers keep working.
def make_pipeline():  # pragma: no cover — legacy
    return get_backend("kokoro")


_chunk_text = chunk_text  # legacy private name
