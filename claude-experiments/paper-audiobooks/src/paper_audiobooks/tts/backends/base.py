"""Backend interface that every TTS engine implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    name: str             # registry key, e.g. "kokoro"
    default_voice: str    # voice id used when caller doesn't specify one
    max_chunk_chars: int  # safe upper bound for a single synthesis call
    description: str      # one-line human description


class Backend(ABC):
    info: BackendInfo  # subclasses set as a class attr

    @abstractmethod
    def synthesize_chunk(self, text: str, *, voice: str) -> np.ndarray:
        """Render a single text chunk to a float32 waveform at the package SAMPLE_RATE.

        Backends are responsible for resampling internally if their native rate differs.
        """

    def list_voices(self) -> list[str]:  # pragma: no cover — optional override
        return [self.info.default_voice]

    def dump_intermediates(self, target_dir: Path) -> None:  # pragma: no cover — optional
        """Optional: when render_audio gives up on a chunk after retries, it
        calls this to ask the backend to flush any in-memory diagnostic state
        (e.g. captured intermediates from the last failed call). Default: no-op."""
        return None
