"""Client for the persistent marker extraction server (`extract_server.py`).

Spawns one long-lived `extract_server` subprocess, waits for it to finish
loading marker's models, then sends documents one at a time over JSON lines.
The server's stderr (marker progress bars, tracebacks) is streamed to the
parent's stderr so you still see live progress.

If the server dies mid-run, `extract()` raises ExtractServerDead; the caller
can restart (make a new ExtractClient) and retry — one dead doc costs a model
reload, not the whole run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path


class ExtractServerDead(RuntimeError):
    pass


def _die_with_parent() -> None:
    """preexec_fn: SIGKILL this process if the parent dies (Linux)."""
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class ExtractClient:
    def __init__(self, *, ready_timeout: float = 600.0, log_file=None) -> None:
        """Start the server and block until its models are loaded.

        `log_file`: optional file object to tee the server's stderr into (in
        addition to the parent's stderr).
        """
        self._log_file = log_file
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "paper_audiobooks.extract_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=_die_with_parent,
        )
        # Pump the server's stderr to our stderr (+ optional log) in a thread so
        # marker progress is visible and the pipe never blocks.
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, daemon=True,
        )
        self._stderr_thread.start()
        self._await_ready(ready_timeout)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            if self._log_file is not None:
                try:
                    self._log_file.write(line)
                    self._log_file.flush()
                except Exception:
                    pass

    def _readline_json(self) -> dict | None:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            return {}
        try:
            return json.loads(line)
        except Exception:
            return {}

    def _await_ready(self, timeout: float) -> None:
        # The server prints {"ready": true} once models are loaded. We read
        # lines until we see it (skipping any stray output). If the process
        # dies first, raise.
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ExtractServerDead(
                    f"extract server exited during startup (rc={self.proc.returncode})"
                )
            msg = self._readline_json()
            if msg is None:
                raise ExtractServerDead("extract server closed stdout during startup")
            if msg.get("ready"):
                return
        raise ExtractServerDead("extract server did not become ready in time")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def extract(self, source: Path, out_path: Path,
                page_range: list[int] | None = None, *, req_id: int = 0) -> str:
        """Extract one document; returns the markdown. Raises ExtractServerDead
        if the server isn't alive, or RuntimeError on a per-doc extraction error
        (server stays up for the next doc)."""
        if not self.alive():
            raise ExtractServerDead("extract server is not running")
        assert self.proc.stdin is not None
        req = {
            "id": req_id,
            "source_path": str(source),
            "out_path": str(out_path),
            "page_range": page_range,
        }
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise ExtractServerDead("extract server stdin closed (process died)")

        resp = self._readline_json()
        if resp is None:
            raise ExtractServerDead("extract server closed stdout mid-request")
        if not resp.get("ok"):
            # Per-doc failure — server is still alive for the next doc.
            raise RuntimeError(resp.get("error", "unknown extract error"))
        return out_path.read_text()

    def close(self) -> None:
        try:
            if self.alive() and self.proc.stdin is not None:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()

    def __enter__(self) -> "ExtractClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
