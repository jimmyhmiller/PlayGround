"""F5-TTS backend.

Excellent zero-shot voice cloning from short reference audio.

Setup:
    uv venv ~/.cache/paper-audiobooks/venvs/f5 --python 3.11
    VIRTUAL_ENV=~/.cache/paper-audiobooks/venvs/f5 uv pip install f5-tts soundfile
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import Backend, BackendInfo

VENV_DIR = Path(os.path.expanduser("~/.cache/paper-audiobooks/venvs/f5"))
VENV_PYTHON = VENV_DIR / "bin" / "python"

_CHILD_SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

req = json.loads(sys.stdin.read())

from f5_tts.api import F5TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = F5TTS(device=device)

# F5 always needs a reference voice (even for "zero-shot"). Use bundled English
# default unless a custom path is provided.
import f5_tts
_pkg_dir = Path(list(f5_tts.__path__)[0])
default_ref = (_pkg_dir / "infer/examples/basic/basic_ref_en.wav").resolve()
default_ref_text = "Some call me nature, others call me mother nature."

ref_path = req.get("voice_ref_path") or str(default_ref)
ref_text = req.get("voice_ref_text") or (default_ref_text if ref_path == str(default_ref) else "")

audio, sr, _ = tts.infer(
    ref_file=ref_path,
    ref_text=ref_text,
    gen_text=req["text"],
    speed=req.get("speed", 1.0),
)
audio = np.asarray(audio, dtype="float32")
sf.write(req["out_path"], audio, sr)
print(json.dumps({"sample_rate": int(sr), "samples": int(audio.shape[0])}))
"""


class F5Backend(Backend):
    info = BackendInfo(
        name="f5",
        default_voice="default",
        max_chunk_chars=300,
        description="F5-TTS — flow-matching, excellent zero-shot voice cloning.",
    )

    def __init__(self) -> None:
        if not VENV_PYTHON.exists():
            raise RuntimeError(
                f"F5 venv not found at {VENV_DIR}. See f5.py docstring."
            )

    def synthesize_chunk(self, text: str, *, voice: str) -> np.ndarray:
        from .. import SAMPLE_RATE
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        req: dict = {"text": text, "out_path": str(tmp_path)}
        if voice and voice != "default" and Path(voice).expanduser().is_file():
            req["voice_ref_path"] = str(Path(voice).expanduser())
        try:
            r = subprocess.run(
                [str(VENV_PYTHON), "-c", _CHILD_SCRIPT],
                input=json.dumps(req), text=True, capture_output=True,
            )
            if r.returncode != 0 or tmp_path.stat().st_size == 0:
                raise RuntimeError(
                    f"f5 child failed (rc={r.returncode}):\n"
                    f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
                )
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            tmp_path.unlink(missing_ok=True)

        if sr != SAMPLE_RATE:
            from .higgs import _resample
            audio = _resample(audio, sr, SAMPLE_RATE)
        return audio.astype(np.float32)
