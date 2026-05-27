"""Multi-process ChatterboxTTS worker pool.

Spawns N child processes, each holding its own ChatterboxTTS model. The
parent dispatches work via JSON RPC over stdin/stdout — each child returns
synthesized audio as raw float32 bytes (size-prefixed).

This gets cross-utterance throughput where in-process batching can't (our
single-instance pipeline is already compute-bound at B=1 due to large mel-time
dims, so batching inside one process makes things slower; but two processes
can each push the GPU's parallel compute units).

Verified on AMD Strix Halo (gfx1151) with 2 workers: both finish without
HSA exceptions and total wall is ~55-70% of sequential.

Usage:
    pool = WorkerPool(n_workers=2, voice_ref="default-voice.wav", use_bf16=True)
    audios = pool.synthesize_many(["sentence 1", "sentence 2", ...])
    pool.close()
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np


_CHILD_SCRIPT_TEMPLATE = r'''
import sys, os, struct, json
import numpy as np
sys.path.insert(0, "__ROOT__")
# Tell the wrapper to use bf16 if requested.
if "__USE_BF16__" == "1":
    os.environ["CHATTERBOX_BF16"] = "1"
os.environ["CHATTERBOX_CFM_STEPS"] = "__CFM_STEPS__"

from chatterbox_mojo import ChatterboxTTS

tag = sys.argv[1]
voice_ref = sys.argv[2]

# Print log lines to stderr so stdout stays a clean binary channel.
def _log(msg):
    sys.stderr.write(f"[worker-{tag}] {msg}\n")
    sys.stderr.flush()

_log("loading model")
tts = ChatterboxTTS.from_pretrained()
tts.prepare_conditionals(voice_ref, exaggeration=0.5)
_log("ready")
# Signal ready: a single line on stdout (the only line that's not binary).
sys.stdout.write("READY\n")
sys.stdout.flush()

# Protocol: parent sends one JSON request per line on stdin. Each request is
# {"req_id": int, "text": str, "cfg_weight": float, ...} OR {"shutdown": true}.
# Response: parent reads from stdout
#   * 4-byte little-endian uint32: response header length
#   * <hlen> bytes UTF-8 JSON {"req_id": int, "n_samples": int, "error": optional}
#   * if no error: <n_samples * 4> bytes float32 audio data
while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        _log(f"bad json: {e}")
        continue
    if req.get("shutdown"):
        _log("shutdown")
        break

    req_id = req["req_id"]
    text = req["text"]
    try:
        wav = tts.generate(
            text,
            cfg_weight=req.get("cfg_weight", 0.5),
            temperature=req.get("temperature", 0.8),
            top_p=req.get("top_p", 0.95),
            min_p=req.get("min_p", 0.05),
            repetition_penalty=req.get("repetition_penalty", 1.2),
            exaggeration=req.get("exaggeration", 0.5),
            rng_seed=req.get("rng_seed", 0xDEADBEEF),
        )
        arr = wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        n_samples = arr.size
        hdr = json.dumps({"req_id": req_id, "n_samples": int(n_samples)}).encode("utf-8")
    except Exception as e:
        hdr = json.dumps({"req_id": req_id, "error": str(e)}).encode("utf-8")
        n_samples = 0
        arr = np.zeros(0, dtype=np.float32)

    # Write header length + header + (optional) audio bytes.
    sys.stdout.buffer.write(struct.pack("<I", len(hdr)))
    sys.stdout.buffer.write(hdr)
    if n_samples > 0:
        sys.stdout.buffer.write(arr.tobytes())
    sys.stdout.buffer.flush()

_log("exiting")
'''


_MAX_IMPL_ROOT = str(Path(__file__).resolve().parent.parent)


class _Worker:
    """One child subprocess holding a ChatterboxTTS instance.

    Thread-safe via an internal lock — only one request in flight at a time
    per worker. Use a WorkerPool to fan out across many workers.
    """

    def __init__(self, tag: str, voice_ref: str, *, use_bf16: bool = True,
                 cfm_steps: int = 10, env: Optional[dict] = None):
        script = (
            _CHILD_SCRIPT_TEMPLATE
            .replace("__ROOT__", _MAX_IMPL_ROOT)
            .replace("__USE_BF16__", "1" if use_bf16 else "0")
            .replace("__CFM_STEPS__", str(cfm_steps))
        )
        spawn_env = os.environ.copy()
        if env:
            spawn_env.update(env)
        # Each worker should also know the bf16 flag (the script reads it).
        spawn_env["CHATTERBOX_BF16"] = "1" if use_bf16 else "0"
        spawn_env["CHATTERBOX_CFM_STEPS"] = str(cfm_steps)

        # We launch via `pixi run python` so the child uses the same MAX/Mojo env.
        self.proc = subprocess.Popen(
            ["pixi", "run", "python", "-c", script, tag, voice_ref],
            cwd=_MAX_IMPL_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=spawn_env,
            bufsize=0,  # unbuffered binary
        )
        self.tag = tag
        self._lock = threading.Lock()
        # Wait for READY (a single text line).
        ready = self.proc.stdout.readline()
        if ready.strip() != b"READY":
            raise RuntimeError(f"worker-{tag} didn't start: got {ready!r}")

    def _read_exact(self, n: int) -> bytes:
        """read exactly n bytes from child stdout (bufsize=0 returns short)."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                raise RuntimeError(f"worker-{self.tag} closed unexpectedly (after {n-remaining}/{n} bytes)")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """Blocking synth. Returns float32 numpy audio at 24kHz."""
        with self._lock:
            req = {"req_id": 0, "text": text, **kwargs}
            self.proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
            raw_hlen = self._read_exact(4)
            hlen = struct.unpack("<I", raw_hlen)[0]
            hdr_bytes = self._read_exact(hlen)
            hdr = json.loads(hdr_bytes.decode("utf-8"))
            if "error" in hdr:
                raise RuntimeError(f"worker-{self.tag} error: {hdr['error']}")
            n_samples = hdr["n_samples"]
            if n_samples == 0:
                return np.zeros(0, dtype=np.float32)
            audio_bytes = self._read_exact(n_samples * 4)
            return np.frombuffer(audio_bytes, dtype=np.float32).copy()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"shutdown": True}).encode("utf-8") + b"\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class WorkerPool:
    """N worker processes. Round-robins requests."""

    def __init__(self, n_workers: int, voice_ref: str, *,
                 use_bf16: bool = True, cfm_steps: int = 10,
                 stagger_load_s: float = 2.0):
        """Stagger model loads to avoid burst memory allocation on shared GPU pools.
        On AMD Strix Halo, two instances loading simultaneously can race the GTT
        allocator and hit OOM; loading them one after another with a small delay
        is reliable.
        """
        self.workers = []
        for i in range(n_workers):
            if i > 0:
                time.sleep(stagger_load_s)
            w = _Worker(f"w{i}", voice_ref, use_bf16=use_bf16, cfm_steps=cfm_steps)
            self.workers.append(w)

    def synthesize_many(self, texts: list[str], *, on_chunk_done=None,
                        **kwargs) -> list[np.ndarray]:
        """Synthesize a batch of texts in parallel across workers. Preserves
        ordering (output[i] corresponds to texts[i]).

        `on_chunk_done(idx, audio)`: optional callback invoked from the worker
        thread immediately after a chunk completes. Use this for incremental
        caching so a mid-batch crash doesn't lose all work.
        """
        n = len(texts)
        results: list[Optional[np.ndarray]] = [None] * n
        errors: list[Optional[Exception]] = [None] * n

        def _do_one(idx: int):
            worker = self.workers[idx % len(self.workers)]
            try:
                audio = worker.synthesize(texts[idx], rng_seed=0xDEADBEEF + idx, **kwargs)
                results[idx] = audio
                if on_chunk_done is not None:
                    try: on_chunk_done(idx, audio)
                    except Exception: pass  # never let callback failure poison the batch
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=_do_one, args=(i,), daemon=True)
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, e in enumerate(errors):
            if e is not None:
                raise RuntimeError(f"chunk {i} failed") from e
        return results  # type: ignore[return-value]

    def close(self):
        for w in self.workers:
            w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
