"""Auto-launch and supervise a local llama.cpp server."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEFAULT_BINARY = Path("/home/jimmyhmiller/Documents/Code/open-source/llama.cpp/build/bin/llama-server")
DEFAULT_MODEL = Path(
    "/home/jimmyhmiller/.cache/llama.cpp/unsloth_Qwen3.6-35B-A3B-MTP-GGUF_Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
)
DEFAULT_CTX = 131072
DEFAULT_ALIAS = "qwen3"


def _is_up(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _wait_until_ready(base_url: str, proc: subprocess.Popen[bytes], deadline: float) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        if _is_up(base_url):
            return
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become ready in time")


@contextmanager
def ensure_server(
    base_url: str,
    *,
    binary: Path = DEFAULT_BINARY,
    model: Path = DEFAULT_MODEL,
    ctx: int = DEFAULT_CTX,
    alias: str = DEFAULT_ALIAS,
    startup_timeout: float = 300.0,
    log_path: Path | None = None,
):
    """Yield once a llama.cpp server is reachable at base_url.

    If an external server is already running there, do nothing.
    Otherwise, start one and terminate it on exit.
    """
    if _is_up(base_url):
        yield False
        return

    if not binary.exists():
        raise FileNotFoundError(f"llama-server binary not found at {binary}")
    if not model.exists():
        raise FileNotFoundError(f"model not found at {model}")

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8080)

    cmd = [
        str(binary), "-m", str(model),
        "--host", host, "--port", port,
        "-ngl", "999", "-c", str(ctx),
        "--jinja", "-fit", "off", "-a", alias,
    ]

    log_f = open(log_path, "ab") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        stdout=log_f if log_path else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        _wait_until_ready(base_url, proc, time.monotonic() + startup_timeout)
        yield True
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        if log_path:
            log_f.close()
