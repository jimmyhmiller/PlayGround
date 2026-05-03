"""Kokoro TTS backend (the default — fast, predictable, broad voice library)."""
from __future__ import annotations

import numpy as np

from .base import Backend, BackendInfo


class KokoroBackend(Backend):
    info = BackendInfo(
        name="kokoro",
        default_voice="bm_daniel",
        max_chunk_chars=500,
        description="Kokoro 82M ONNX-based TTS — steady tone, fast, 50+ voices.",
    )

    def __init__(self) -> None:
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code="a")

    def synthesize_chunk(self, text: str, *, voice: str) -> np.ndarray:
        # Kokoro emits 24kHz natively (matches our SAMPLE_RATE).
        pieces: list[np.ndarray] = []
        for _gs, _ps, audio in self._pipeline(text, voice=voice):
            pieces.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
