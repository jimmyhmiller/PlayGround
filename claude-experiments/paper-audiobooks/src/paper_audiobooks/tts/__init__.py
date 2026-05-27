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


def _detect_thumps(audio: np.ndarray) -> tuple[bool, dict]:
    """Detect catastrophic CFM oscillation in a chunk: ±0.99-clipped samples
    appearing in clusters (the signature of our bf16 + s3gen instability —
    audio jumps wildly to ±1.0, gets clipped by HiFT's audio_limit=0.99).

    Returns (is_bad, stats).
    """
    a = audio.astype(np.float32)
    diff = np.abs(np.diff(a))
    n_big = int(np.sum(diff > 0.8))
    n_huge = int(np.sum(diff > 1.5))
    n_clipped = int(np.sum(np.abs(a) > 0.985))
    n_samples = a.size
    pct_clipped = 100.0 * n_clipped / max(n_samples, 1)
    # Bad if ≥3 huge jumps OR ≥0.05% of samples are clipped (a 12s chunk × 24kHz
    # = 288k samples; 0.05% = 144 clipped samples → clear oscillation, vs normal
    # speech where occasional loud peaks produce maybe 1-2 clipped samples).
    bad = (n_huge >= 3) or (pct_clipped > 0.05)
    return bad, {
        "thumps_gt_0p8": n_big, "huge_gt_1p5": n_huge,
        "clipped_samples": n_clipped, "pct_clipped": pct_clipped,
        "n_samples": n_samples,
    }


def _chunk_cache_key(backend: Backend, voice: str, text: str) -> str:
    """Stable cache key for one chunk. Includes backend name, voice, and the
    exact text so a different render config produces a different cache file.

    This is intentionally not version-aware — if you change backend parameters
    (cfg_weight, temperature, etc.) and want to invalidate, just delete the
    chunks/ directory under .tts-cache.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(backend.info.name.encode("utf-8"))
    h.update(b"\x00")
    h.update(voice.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:24]


def _load_chunk_cache(cache_dir, key: str):
    if cache_dir is None:
        return None
    p = cache_dir / f"chunk-{key}.npy"
    if not p.exists():
        return None
    try:
        a = np.load(p).astype(np.float32)
    except Exception:
        # Corrupt — wipe and re-render.
        p.unlink(missing_ok=True)
        return None
    # Validate: if a cached chunk has thumps, it was saved before the per-chunk
    # QA gate was added. Treat as a miss so we resynthesize cleanly.
    bad, _stats = _detect_thumps(a)
    if bad:
        p.unlink(missing_ok=True)
        return None
    return a


def _save_chunk_cache(cache_dir, key: str, audio: np.ndarray) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Atomic-ish: write to .tmp then rename.
    p = cache_dir / f"chunk-{key}.npy"
    tmp = cache_dir / f"chunk-{key}.npy.tmp"
    try:
        np.save(tmp, audio.astype(np.float32))
        tmp.replace(p)
    except Exception:
        # Best-effort; don't fail synthesis just because cache write failed.
        try: tmp.unlink(missing_ok=True)
        except Exception: pass


def render_audio(text: str, *, backend: Backend, voice: str,
                 chunk_cache_dir=None) -> np.ndarray:
    """Synthesize text via a loaded Backend, returning a float32 waveform at SAMPLE_RATE.

    When CHATTERBOX_HALT_ON_ANOMALY=1, each rendered chunk is scored with the
    chatterbox muffled-output detector. Bad chunks are retried up to
    MAX_ANOMALY_RETRIES times (the bug is per-call nondeterministic so retries
    usually succeed). After the retries are exhausted, dump the bad chunk +
    capture buffer and raise ChatterboxAnomalyHalt.

    `chunk_cache_dir`: if given, each successfully-synthesized chunk is cached
    as a separate .npy file keyed by hash(backend+voice+text). On a re-run we
    skip resynthesis for chunks already cached. This is much finer-grained
    than the per-chapter cache and saves work on partial chapter renders.
    """
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

    # Lookup any cached chunks (so we skip them, and feed only misses to the backend).
    keys = [_chunk_cache_key(backend, voice, c) for c in chunks]
    cached = [_load_chunk_cache(chunk_cache_dir, k) for k in keys]
    n_hits = sum(1 for c in cached if c is not None)
    if chunk_cache_dir is not None and n_hits > 0:
        print(f"  [render] chunk cache: {n_hits}/{len(chunks)} hits", file=sys.stderr, flush=True)

    # Per-chunk thump retry budget. Each retry uses a perturbed seed; chunks
    # that thump tend to be seed-sensitive (the underlying CFM bug is bf16
    # + specific T3 token sequence → about 5% of seeds for an affected chunk
    # produce wild oscillation). A few retries with new seeds clears almost all.
    MAX_THUMP_RETRIES = int(os.environ.get("CHATTERBOX_THUMP_RETRIES", "5"))

    def _retry_chunk_until_clean(local_idx: int, global_idx: int, text: str) -> np.ndarray:
        """Re-synthesize one chunk with new seeds until no thumps detected."""
        for attempt in range(1, MAX_THUMP_RETRIES + 1):
            seed = 0xDEADBEEF + global_idx + attempt * 0x1000000
            try:
                a = backend.synthesize_chunk(text, voice=voice, rng_seed=seed)
            except TypeError:
                # Backend doesn't support rng_seed — can't recover; surface raw render.
                a = backend.synthesize_chunk(text, voice=voice)
                return a
            bad, stats = _detect_thumps(a)
            if not bad:
                print(f"  [thump-retry] chunk {global_idx+1}: recovered on attempt {attempt} "
                      f"(seed=0x{seed:x}, {stats})", file=sys.stderr, flush=True)
                return a
            print(f"  [thump-retry] chunk {global_idx+1} attempt {attempt}/{MAX_THUMP_RETRIES} "
                  f"still bad: {stats}", file=sys.stderr, flush=True)
        # All retries failed — log loudly but return last attempt (don't crash the book).
        print(f"  [thump-retry] !!! chunk {global_idx+1} EXHAUSTED retries; "
              f"shipping defective audio. Text: {text[:120]!r}", file=sys.stderr, flush=True)
        return a

    # Fast path: if there's no per-chunk anomaly retry, let the backend render
    # the cache-miss chunks together (it may parallelize across workers).
    if is_anomalous is None and hasattr(backend, "synthesize_many"):
        miss_indices = [i for i, c in enumerate(cached) if c is None]
        miss_texts = [chunks[i] for i in miss_indices]
        if miss_texts:
            t0 = time.time()
            print(f"  [render] batch-synthesizing {len(miss_texts)} chunks", file=sys.stderr, flush=True)
            # Don't cache yet — we'll cache after thump-checking.
            try:
                new_audios = backend.synthesize_many(miss_texts, voice=voice)
            except TypeError:
                new_audios = backend.synthesize_many(miss_texts, voice=voice)
            dt = time.time() - t0
            total_audio_s = sum(len(a) / SAMPLE_RATE for a in new_audios)
            print(f"  [render] batch done: {total_audio_s:.1f}s audio in {dt:.1f}s wall "
                  f"(rtf={total_audio_s/dt:.2f}x)", file=sys.stderr, flush=True)

            # Thump-check every chunk; retry the bad ones serially with new seeds.
            n_bad = 0
            for local_idx, audio in enumerate(new_audios):
                global_idx = miss_indices[local_idx]
                bad, stats = _detect_thumps(audio)
                if bad:
                    n_bad += 1
                    print(f"  [thump] chunk {global_idx+1} bad: {stats}; retrying",
                          file=sys.stderr, flush=True)
                    audio = _retry_chunk_until_clean(local_idx, global_idx, chunks[global_idx])
                _save_chunk_cache(chunk_cache_dir, keys[global_idx], audio)
                cached[global_idx] = audio.astype(np.float32)
            if n_bad:
                print(f"  [thump] {n_bad}/{len(new_audios)} chunks needed thump retry",
                      file=sys.stderr, flush=True)
        for a in cached:
            pieces.append(a.astype(np.float32))
            pieces.append(silence)
        return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)

    for i, chunk in enumerate(chunks, 1):
        if cached[i-1] is not None:
            pieces.append(cached[i-1].astype(np.float32))
            pieces.append(silence)
            continue
        t0 = time.time()
        print(f"  [render] chunk {i}/{len(chunks)} ({len(chunk)} chars): start", file=sys.stderr, flush=True)
        audio = _render_chunk_with_retry(
            chunk, backend=backend, voice=voice,
            is_anomalous=is_anomalous, halt_exc_cls=halt_exc_cls,
            chunk_idx=i, total_chunks=len(chunks),
        )
        # Thump check regardless of anomaly setting — this is a different bug.
        bad, stats = _detect_thumps(audio)
        if bad:
            print(f"  [thump] chunk {i} bad: {stats}; retrying", file=sys.stderr, flush=True)
            audio = _retry_chunk_until_clean(0, i-1, chunk)
        dt = time.time() - t0
        print(f"  [render] chunk {i}/{len(chunks)} ({len(chunk)} chars): "
              f"{len(audio)/SAMPLE_RATE:.1f}s audio in {dt:.1f}s wall", file=sys.stderr, flush=True)
        _save_chunk_cache(chunk_cache_dir, keys[i-1], audio)
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
