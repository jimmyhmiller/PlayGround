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


MAX_ANOMALY_RETRIES = 3


def render_audio(text: str, *, backend: Backend, voice: str) -> np.ndarray:
    """Synthesize text via a loaded Backend, returning a float32 waveform at SAMPLE_RATE.

    When CHATTERBOX_HALT_ON_ANOMALY=1, each rendered chunk is scored with the
    chatterbox muffled-output detector. Bad chunks are retried up to
    MAX_ANOMALY_RETRIES times (the bug is per-call nondeterministic so retries
    usually succeed). After the retries are exhausted, dump the bad chunk +
    capture buffer and raise ChatterboxAnomalyHalt."""
    import os, sys, time
    pieces: list[np.ndarray] = []
    silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
    chunks = chunk_text(text, max_chars=backend.info.max_chunk_chars)
    print(f"  [render] {len(chunks)} chunk(s), max_chunk_chars={backend.info.max_chunk_chars}", file=sys.stderr, flush=True)

    halt_on_anomaly = os.environ.get("CHATTERBOX_HALT_ON_ANOMALY") == "1"
    is_anomalous = None
    halt_exc_cls = None
    if halt_on_anomaly:
        from .backends.chatterbox import _is_anomalous, ChatterboxAnomalyHalt
        is_anomalous = _is_anomalous
        halt_exc_cls = ChatterboxAnomalyHalt

    for i, chunk in enumerate(chunks, 1):
        t0 = time.time()
        print(f"  [render] chunk {i}/{len(chunks)} ({len(chunk)} chars): start", file=sys.stderr, flush=True)
        audio = _render_chunk_with_retry(
            chunk, backend=backend, voice=voice,
            is_anomalous=is_anomalous, halt_exc_cls=halt_exc_cls,
            chunk_idx=i, total_chunks=len(chunks),
        )
        dt = time.time() - t0
        print(f"  [render] chunk {i}/{len(chunks)} ({len(chunk)} chars): "
              f"{len(audio)/SAMPLE_RATE:.1f}s audio in {dt:.1f}s wall", file=sys.stderr, flush=True)
        pieces.append(audio.astype(np.float32))
        pieces.append(silence)
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)


def _render_chunk_with_retry(text, *, backend, voice, is_anomalous, halt_exc_cls,
                             chunk_idx, total_chunks):
    """Synthesize a single chunk. If is_anomalous is set, retry up to
    MAX_ANOMALY_RETRIES times when the detector flags the output. On final
    failure, dump artifacts and raise halt_exc_cls."""
    import json, os, sys
    from pathlib import Path
    import time as _time
    from .backends.chatterbox import DEFAULT_HALT_DIR, HALT_DIR_ENV

    last_audio = None
    last_stats = None
    n_attempts = 1 + (MAX_ANOMALY_RETRIES if is_anomalous is not None else 0)
    for attempt in range(1, n_attempts + 1):
        audio = backend.synthesize_chunk(text, voice=voice)
        if is_anomalous is None:
            return audio
        bad, stats = is_anomalous(audio, SAMPLE_RATE)
        if not bad:
            if attempt > 1:
                print(f"  [render] chunk {chunk_idx}/{total_chunks}: "
                      f"recovered on attempt {attempt}/{n_attempts} "
                      f"(prior stats={last_stats})", file=sys.stderr, flush=True)
            return audio
        last_audio = audio
        last_stats = stats
        print(f"  [render] chunk {chunk_idx}/{total_chunks}: anomaly on attempt "
              f"{attempt}/{n_attempts}: {stats}; retrying" if attempt < n_attempts
              else f"  [render] chunk {chunk_idx}/{total_chunks}: anomaly on attempt "
                   f"{attempt}/{n_attempts}: {stats}; out of retries",
              file=sys.stderr, flush=True)

    # All attempts failed — dump and raise.
    import soundfile as sf
    halt_dir = Path(os.environ.get(HALT_DIR_ENV) or DEFAULT_HALT_DIR)
    halt_dir.mkdir(parents=True, exist_ok=True)
    slug = _time.strftime("%Y%m%d-%H%M%S")
    wav_path = halt_dir / f"halt-{slug}.wav"
    txt_path = halt_dir / f"halt-{slug}.txt"
    stats_path = halt_dir / f"halt-{slug}.json"
    dumps_dir = halt_dir / f"halt-{slug}-dumps"
    sf.write(wav_path, last_audio, SAMPLE_RATE)
    txt_path.write_text(text, encoding="utf-8")
    stats_path.write_text(json.dumps(
        {**last_stats, "n_chars": len(text), "attempts": n_attempts,
         "chunk_idx": chunk_idx, "total_chunks": total_chunks},
        indent=2,
    ), encoding="utf-8")
    dump_note = ""
    try:
        backend.dump_intermediates(dumps_dir)
        dump_note = f"\n  dumps: {dumps_dir}"
    except Exception as exc:
        dump_note = f"\n  dumps: <failed: {exc}>"
    raise halt_exc_cls(
        f"chatterbox anomaly persisted across {n_attempts} attempts: {last_stats}\n"
        f"  wav:   {wav_path}\n"
        f"  text:  {txt_path}\n"
        f"  stats: {stats_path}{dump_note}\n"
        f"  chars: {len(text)}  (chunk {chunk_idx}/{total_chunks})"
    )


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
        from .backends import chatterbox_mojo as chatterbox_mojo_mod
        reg[chatterbox_mojo_mod.ChatterboxMojoBackend.info.name] = chatterbox_mojo_mod.ChatterboxMojoBackend
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
