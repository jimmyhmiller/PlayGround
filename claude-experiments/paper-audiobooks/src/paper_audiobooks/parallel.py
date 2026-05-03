"""Sequential TTS rendering. One chapter at a time, one chatterbox subprocess
at a time. The "parallel" in the module name is historical — concurrent TTS
on AMD ROCm causes MIOpen crashes ("chatterbox child closed stdout
unexpectedly"), so we render strictly serially.

Renders are cached on disk per (paper, chapter index) so the pipeline can be
killed and resumed without re-doing chapters that already finished.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .chapters import Chapter


def _die_with_parent() -> None:
    """preexec_fn: SIGKILL this process if the parent dies. Linux only."""
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def render_chapters_parallel(
    chapter_texts: list[tuple[Chapter, str]],
    *,
    voice: str,
    workers: int = 1,  # ignored; kept for API compatibility
    backend: str = "kokoro",
    cache_dir: Path,
    progress_cb=None,
) -> list[tuple[Chapter, np.ndarray]]:
    """Render (chapter, spoken_text) pairs sequentially; cache and resume.

    Each rendered chapter is saved as `cache_dir/chapter-NNNN.npy`. On a re-run,
    chapters that already have a cached file are skipped.

    `workers` is accepted but ignored — rendering is always serial. Concurrency
    on AMD ROCm causes MIOpen crashes; on NVIDIA the marginal speedup isn't
    worth the complexity (a single chatterbox child already keeps the GPU busy).
    """
    if workers != 1:
        print(
            f"[tts] note: workers={workers} ignored — pipeline is sequential by design",
            file=sys.stderr,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[np.ndarray] = []

    for i, (chap, text) in enumerate(chapter_texts):
        out_path = cache_dir / f"chapter-{i:04d}.npy"

        # Reuse cached render if present.
        if out_path.exists():
            try:
                audio = np.load(out_path).astype(np.float32)
                if progress_cb is not None:
                    progress_cb(i, chap, cached=True)
                results.append(audio)
                continue
            except Exception:
                # Corrupt cache entry — re-render.
                out_path.unlink(missing_ok=True)

        # Render this chapter, retrying transient failures.
        results.append(_render_one(i, chap, text, voice, backend, out_path, progress_cb))

    return [(chap, results[i]) for i, (chap, _) in enumerate(chapter_texts)]


def _render_one(
    idx: int,
    chap: Chapter,
    text: str,
    voice: str,
    backend: str,
    out_path: Path,
    progress_cb,
    *,
    max_attempts: int = 3,
) -> np.ndarray:
    """Render one chapter via a fresh tts_worker subprocess. Retry on transient
    failures (chatterbox children sometimes die mid-chunk on AMD; a fresh
    subprocess usually succeeds)."""
    req = json.dumps({
        "text": text, "voice": voice, "out_path": str(out_path), "backend": backend,
    })
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        # Stream stderr live (so we see [child] logs in real time) but capture stdout.
        proc = subprocess.Popen(
            [sys.executable, "-m", "paper_audiobooks.tts_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
            preexec_fn=_die_with_parent,
        )
        stdout, _ = proc.communicate(input=req)
        if proc.returncode == 0 and out_path.exists():
            audio = np.load(out_path).astype(np.float32)
            if progress_cb is not None:
                progress_cb(idx, chap, cached=False)
            return audio
        last_error = f"rc={proc.returncode}; stdout={stdout!r}"
        print(
            f"[tts] chapter {idx} ({chap.title!r}) attempt {attempt}/{max_attempts} failed: "
            f"{last_error[:200]}",
            file=sys.stderr,
        )
    raise RuntimeError(
        f"TTS worker failed for chapter {idx} ({chap.title}) after {max_attempts} attempts; "
        f"last_error={last_error}"
    )
