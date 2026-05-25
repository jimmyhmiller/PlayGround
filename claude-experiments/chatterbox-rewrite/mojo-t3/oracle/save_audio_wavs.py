"""
Save the HiFiGAN audio fixtures as playable WAV files for subjective listening.
Run after dump_hifigan_case.py and after the Mojo test produces its output.
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import wave

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "hifigan"
OUT_DIR = ROOT / "tests" / "fixtures" / "hifigan_wavs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 24000


def read_fp32(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        assert tag == 0, f"expected fp32 tag 0, got {tag}"
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def save_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    audio = audio.flatten()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path}  ({len(audio)/sr:.2f}s @ {sr}Hz, peak={np.abs(audio).max():.3f})")


def main():
    save_wav(OUT_DIR / "upstream_full.wav",       read_fp32(FIX / "expected_wav.bin"))
    save_wav(OUT_DIR / "upstream_decode_zeros.wav",   read_fp32(FIX / "expected_wav_decode_zeros.bin"))
    save_wav(OUT_DIR / "upstream_decode_real.wav",    read_fp32(FIX / "expected_wav_decode_real.bin"))


if __name__ == "__main__":
    main()
