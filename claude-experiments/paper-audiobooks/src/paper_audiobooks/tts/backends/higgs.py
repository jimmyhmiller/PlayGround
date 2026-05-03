"""Higgs Audio v2 backend.

Uses the native transformers integration (transformers >= 5.3 ships
HiggsAudioV2ForConditionalGeneration), so we don't depend on the boson_multimodal
package's stale model code. Runs in its own venv to avoid the rest of our project
having to upgrade transformers.

Setup (one-time):
    uv venv ~/.cache/paper-audiobooks/venvs/higgs --python 3.10
    uv pip install --python ~/.cache/paper-audiobooks/venvs/higgs/bin/python \\
        'transformers>=5.3' tokenizers accelerate torch torchaudio soundfile

Voice: pass a path to a reference wav for voice-cloning, or "default" for
zero-shot smart-voice (model picks).
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

VENV_DIR = Path(os.path.expanduser("~/.cache/paper-audiobooks/venvs/higgs"))
VENV_PYTHON = VENV_DIR / "bin" / "python"

_CHILD_SCRIPT = r"""
import json, sys
import numpy as np
import soundfile as sf
import torch

req = json.loads(sys.stdin.read())

from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

model_id = req.get("model_id", "bosonai/higgs-audio-v2-generation-3B-base")
device_map = "auto" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, device_map=device_map)
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(model_id, device_map=device_map)

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "scene",  "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
    {"role": "user",   "content": [{"type": "text", "text": req["text"]}]},
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    sampling_rate=24000,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.get("max_new_tokens", 2048),
        do_sample=True,
        temperature=req.get("temperature", 0.8),
        top_p=req.get("top_p", 0.95),
    )
decoded = processor.batch_decode(outputs)

# Higgs's processor returns intermediate codebook tokens; only `processor.save_audio`
# knows how to render them through the audio tokenizer to a real waveform.
# So we let it write the wav, then re-read.
processor.save_audio(decoded, req["out_path"])
audio, sr = sf.read(req["out_path"], dtype="float32")
print(json.dumps({"sample_rate": int(sr), "samples": int(len(audio))}))
"""


class HiggsBackend(Backend):
    info = BackendInfo(
        name="higgs",
        default_voice="default",
        max_chunk_chars=350,
        description="Higgs Audio v2 (transformers native) — 3B audio LM, audiobook-trained.",
    )

    def __init__(self) -> None:
        if not VENV_PYTHON.exists():
            raise RuntimeError(
                f"Higgs venv not found at {VENV_DIR}. See higgs.py docstring for setup."
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
                    f"higgs child failed (rc={r.returncode}):\n"
                    f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
                )
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            tmp_path.unlink(missing_ok=True)

        if sr != SAMPLE_RATE:
            audio = _resample(audio, sr, SAMPLE_RATE)
        return audio.astype(np.float32)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    try:
        import soxr
        return soxr.resample(x, src, dst).astype(np.float32)
    except ImportError:
        import math
        n_out = int(math.ceil(len(x) * dst / src))
        idx = np.linspace(0, len(x) - 1, n_out)
        lo = idx.astype(np.int64)
        hi = np.clip(lo + 1, 0, len(x) - 1)
        frac = (idx - lo).astype(np.float32)
        return ((1 - frac) * x[lo] + frac * x[hi]).astype(np.float32)
