"""Chatterbox TTS backend (Resemble AI).

Beats ElevenLabs in blind tests at 63.75% preference. Voice-cloning + emotion
control. Has its own venv to avoid version conflicts.

Setup:
    uv venv ~/.cache/paper-audiobooks/venvs/chatterbox --python 3.11
    VIRTUAL_ENV=~/.cache/paper-audiobooks/venvs/chatterbox uv pip install chatterbox-tts soundfile

The child process is long-lived: model loads once, then we feed it one JSON
request per line on stdin and read one JSON ack per line on stdout. This avoids
paying the ~4s model load per chunk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import Backend, BackendInfo

VENV_DIR = Path(os.path.expanduser("~/.cache/paper-audiobooks/venvs/chatterbox"))
VENV_PYTHON = VENV_DIR / "bin" / "python"


def _die_with_parent() -> None:
    """preexec_fn: ask the kernel to SIGKILL this process if the parent dies.
    Linux-only (PR_SET_PDEATHSIG = 1, SIGKILL = 9). Without this, killing the
    pipeline parent leaves orphan chatterbox children spinning on the GPU."""
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass  # best-effort; don't block child startup

# Per-chunk timeout once the model is loaded. ~250-char chunk on GPU is ~10-30s
# typical; we allow 5min as a hard ceiling so a real GPU hang doesn't block forever.
CHUNK_TIMEOUT_SECONDS = 300

_CHILD_SCRIPT = r"""
import json, os, sys, time, traceback

# Save the real stdout for protocol acks, then redirect fd 1 to stderr so any
# library that prints to stdout during import/model-load (chatterbox does:
# "loaded PerthNet (Implicit) at step 250,000") doesn't corrupt our JSON channel.
_protocol_stdout = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = os.fdopen(1, "w", buffering=1)

print(f"[child] booting at {time.strftime('%H:%M:%S')}", file=sys.stderr, flush=True)
import numpy as np
import soundfile as sf
import torch

print(f"[child] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}", file=sys.stderr, flush=True)

from chatterbox.tts import ChatterboxTTS
print("[child] loading model...", file=sys.stderr, flush=True)
t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
print(f"[child] model loaded in {time.time()-t0:.1f}s on {device}", file=sys.stderr, flush=True)

# Signal ready so the parent can start sending requests.
_protocol_stdout.write(json.dumps({"ready": True}) + "\n")
_protocol_stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        if req.get("shutdown"):
            print("[child] shutdown requested", file=sys.stderr, flush=True)
            break
        text = req["text"]
        out_path = req["out_path"]
        audio_prompt = req.get("voice_ref_path") or None
        exaggeration = float(req.get("exaggeration", 0.5))
        cfg_weight = float(req.get("cfg_weight", 0.5))

        print(f"[child] generate(): {len(text)}chars voice_ref={'yes' if audio_prompt else 'no'}", file=sys.stderr, flush=True)
        t0 = time.time()
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        gen_dt = time.time() - t0
        audio = wav.squeeze().cpu().numpy().astype("float32")
        sr = int(model.sr)
        sf.write(out_path, audio, sr)
        print(f"[child] done in {gen_dt:.1f}s, {audio.shape[0]/sr:.1f}s audio", file=sys.stderr, flush=True)
        _protocol_stdout.write(json.dumps({"sample_rate": sr, "samples": int(audio.shape[0])}) + "\n")
        _protocol_stdout.flush()
    except Exception as exc:
        traceback.print_exc()
        _protocol_stdout.write(json.dumps({"error": str(exc)}) + "\n")
        _protocol_stdout.flush()
"""


class _ChildHandle:
    """Wraps the long-lived chatterbox child process."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [str(VENV_PYTHON), "-c", _CHILD_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,  # line-buffered
            preexec_fn=_die_with_parent,  # kill child if parent dies (no orphans)
        )
        # Wait for the {"ready": true} ack so we don't send chunks before the model is up.
        ready_line = self.proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("chatterbox child exited before becoming ready")
        try:
            ack = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"chatterbox child sent bad ready line: {ready_line!r}") from exc
        if not ack.get("ready"):
            raise RuntimeError(f"chatterbox child failed to start: {ack!r}")
        self._lock = threading.Lock()

    def synth(self, *, text: str, voice_ref_path: str | None, out_path: Path) -> None:
        req = {"text": text, "out_path": str(out_path)}
        if voice_ref_path:
            req["voice_ref_path"] = voice_ref_path
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f"chatterbox child died (rc={self.proc.returncode})")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            # Wait for the ack with a timeout. We can't easily timeout readline(),
            # so use a watchdog thread that kills the child if it overruns.
            timer = threading.Timer(CHUNK_TIMEOUT_SECONDS, self._kill_on_timeout)
            timer.start()
            try:
                ack_line = self.proc.stdout.readline()
            finally:
                timer.cancel()

            if not ack_line:
                raise RuntimeError("chatterbox child closed stdout unexpectedly")
            ack = json.loads(ack_line)
            if "error" in ack:
                raise RuntimeError(f"chatterbox child error: {ack['error']}")

    def _kill_on_timeout(self) -> None:
        print(
            f"[chatterbox] TIMEOUT after {CHUNK_TIMEOUT_SECONDS}s — killing child",
            file=sys.stderr,
            flush=True,
        )
        try:
            self.proc.kill()
        except Exception:
            pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class ChatterboxBackend(Backend):
    info = BackendInfo(
        name="chatterbox",
        default_voice="default",
        max_chunk_chars=250,
        description="Chatterbox (Resemble AI) — emotion control, voice clone, beats ElevenLabs in blind tests.",
    )

    def __init__(self) -> None:
        if not VENV_PYTHON.exists():
            raise RuntimeError(
                f"Chatterbox venv not found at {VENV_DIR}. See chatterbox.py docstring."
            )
        self._child: _ChildHandle | None = None

    def _ensure_child(self) -> _ChildHandle:
        if self._child is None or self._child.proc.poll() is not None:
            self._child = _ChildHandle()
        return self._child

    def synthesize_chunk(self, text: str, *, voice: str) -> np.ndarray:
        from .. import SAMPLE_RATE
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        voice_ref = None
        if voice and voice != "default" and Path(voice).expanduser().is_file():
            voice_ref = str(Path(voice).expanduser())
        try:
            child = self._ensure_child()
            child.synth(text=text, voice_ref_path=voice_ref, out_path=tmp_path)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError("chatterbox produced an empty wav")
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            tmp_path.unlink(missing_ok=True)

        if sr != SAMPLE_RATE:
            from .higgs import _resample
            audio = _resample(audio, sr, SAMPLE_RATE)
        return audio.astype(np.float32)

    def __del__(self) -> None:
        try:
            if getattr(self, "_child", None) is not None:
                self._child.close()
        except Exception:
            pass
