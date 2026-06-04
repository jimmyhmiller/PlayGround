"""Chatterbox-Mojo backend — uses the Mojo rewrite at
chatterbox-rewrite/max-impl. 2x faster than the PyTorch upstream backend,
0% WER on round-trip whisper testing.

Spawns a long-lived child process running ChatterboxTTS in the max-impl
pixi env (via `pixi run python`). Communication is JSON RPC over stdin/stdout
with binary float32 audio payloads (size-prefixed).

Voice handling:
  - "default" → uses the built-in default voice (whatever the child loads
     via prepare_conditionals once at startup)
  - any other value treated as an absolute path to a 24kHz .wav reference

Re-uses chatterbox_mojo.pool._Worker for transport.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import Backend, BackendInfo


# Resolve the max-impl repo path. If the user moves the repo, set
# CHATTERBOX_MOJO_REPO=/path/to/chatterbox-rewrite/max-impl to override.
_DEFAULT_REPO = Path(
    "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/max-impl"
)
REPO_PATH = Path(os.environ.get("CHATTERBOX_MOJO_REPO", str(_DEFAULT_REPO)))

# Default voice ref — same as paper-audiobooks default-voice.
DEFAULT_VOICE_REF = os.path.expanduser("~/.config/paper-audiobooks/default-voice.wav")


def _load_worker_cls():
    """Add the max-impl repo to sys.path and import the Worker class.

    We do this lazily so the import is cheap when this module is loaded but
    chatterbox_mojo isn't actually used. The chatterbox_mojo.pool module
    imports torch + the wrapper, so it's not free at import time.
    """
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))
    from chatterbox_mojo.pool import _Worker
    return _Worker


def _load_worker_pool_cls():
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))
    from chatterbox_mojo.pool import WorkerPool
    return WorkerPool


# How many worker subprocesses to spawn for cross-chunk parallelism.
# 2 is verified safe on AMD Strix Halo (gfx1151) — 3+ hits hipErrorOutOfMemory.
N_WORKERS = int(os.environ.get("CHATTERBOX_MOJO_WORKERS", "2"))


class ChatterboxMojoBackend(Backend):
    info = BackendInfo(
        name="chatterbox-mojo",
        default_voice="default",
        max_chunk_chars=250,
        description="Chatterbox via the Mojo rewrite (max-impl) — ~2x faster than the PyTorch backend on AMD.",
    )

    def __init__(self) -> None:
        if not REPO_PATH.exists():
            raise RuntimeError(
                f"chatterbox-mojo repo not found at {REPO_PATH}. "
                f"Set CHATTERBOX_MOJO_REPO=/abs/path to override."
            )
        self._worker = None       # single-worker fallback (used by synthesize_chunk path)
        self._pool = None         # WorkerPool (used by synthesize_many path)
        self._worker_voice = None  # the voice the child loaded conditionals for
        self._pool_voice = None

    def _resolve_voice_ref(self, voice: str) -> str:
        if voice and voice != "default":
            p = Path(voice).expanduser()
            if p.is_file():
                return str(p)
        # Fallback to the default voice.
        if Path(DEFAULT_VOICE_REF).is_file():
            return DEFAULT_VOICE_REF
        raise RuntimeError(
            f"voice={voice!r} not a file and default {DEFAULT_VOICE_REF} doesn't exist"
        )

    def _ensure_worker(self, voice_ref: str):
        """Spin up the child if not running, or recycle it if the requested
        voice changed (child holds conditionals for one voice only).
        """
        needs_restart = (
            self._worker is None
            or self._worker_voice != voice_ref
            or (self._worker is not None and self._worker.proc.poll() is not None)
        )
        if needs_restart:
            if self._worker is not None:
                try:
                    self._worker.close()
                except Exception:
                    pass
            _Worker = _load_worker_cls()
            self._worker = _Worker(
                tag="mojo",
                voice_ref=voice_ref,
                use_bf16=True,
                cfm_steps=10,
            )
            self._worker_voice = voice_ref

    def _ensure_pool(self, voice_ref: str):
        """Spin up the WorkerPool (or recycle on voice change)."""
        needs_restart = (
            self._pool is None
            or self._pool_voice != voice_ref
        )
        if needs_restart:
            if self._pool is not None:
                try: self._pool.close()
                except Exception: pass
            WorkerPool = _load_worker_pool_cls()
            self._pool = WorkerPool(
                n_workers=N_WORKERS,
                voice_ref=voice_ref,
                use_bf16=True,
                cfm_steps=10,
            )
            self._pool_voice = voice_ref

    def synthesize_chunk(self, text: str, *, voice: str, rng_seed: int | None = None) -> np.ndarray:
        from .. import SAMPLE_RATE  # 24000
        voice_ref = self._resolve_voice_ref(voice)
        self._ensure_worker(voice_ref)
        kwargs = dict(
            cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05,
            repetition_penalty=1.2, exaggeration=0.5,
        )
        if rng_seed is not None:
            kwargs["rng_seed"] = int(rng_seed)
        audio = self._worker.synthesize(text, **kwargs)
        # The Mojo wrapper outputs at 24kHz, same as SAMPLE_RATE. No resample.
        if SAMPLE_RATE != 24000:
            from .higgs import _resample
            audio = _resample(audio, 24000, SAMPLE_RATE)
        return audio.astype(np.float32, copy=False)

    def synthesize_many(self, texts: list, *, voice: str,
                        on_chunk_done=None) -> list:
        """Parallel render across N_WORKERS subprocesses (default 2).

        `on_chunk_done(idx, audio)`: called from a worker thread immediately
        after chunk `idx` completes (useful for incremental caching).
        """
        from .. import SAMPLE_RATE
        if N_WORKERS <= 1:
            out = []
            for i, t in enumerate(texts):
                a = self.synthesize_chunk(t, voice=voice)
                out.append(a)
                if on_chunk_done is not None:
                    try: on_chunk_done(i, a)
                    except Exception: pass
            return out
        voice_ref = self._resolve_voice_ref(voice)
        self._ensure_pool(voice_ref)
        if SAMPLE_RATE == 24000:
            # No resample needed — pass callback straight through.
            return self._pool.synthesize_many(
                texts, on_chunk_done=on_chunk_done,
                cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05,
                repetition_penalty=1.2, exaggeration=0.5,
            )
        from .higgs import _resample
        def _wrap_cb(idx, audio_24k):
            if on_chunk_done is None: return
            try: on_chunk_done(idx, _resample(audio_24k, 24000, SAMPLE_RATE))
            except Exception: pass
        audios = self._pool.synthesize_many(
            texts, on_chunk_done=_wrap_cb,
            cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05,
            repetition_penalty=1.2, exaggeration=0.5,
        )
        return [_resample(a, 24000, SAMPLE_RATE).astype(np.float32, copy=False) for a in audios]

    def __del__(self) -> None:
        try:
            if getattr(self, "_worker", None) is not None:
                self._worker.close()
        except Exception:
            pass
        try:
            if getattr(self, "_pool", None) is not None:
                self._pool.close()
        except Exception:
            pass
