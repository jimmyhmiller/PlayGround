"""
Convert the Mojo-produced raw fp32 audio and the upstream-produced fp32 audio
into playable WAV files so we can listen side-by-side.
"""
from __future__ import annotations
import struct
import wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "real"
OUT_DIR = ROOT / "tests" / "fixtures" / "real_wavs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SR = 24000


def read_fixture(path: Path) -> np.ndarray:
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
    # Upstream's real audio.
    up = read_fixture(FIX / "real_audio_upstream.bin")
    save_wav(OUT_DIR / "upstream_real.wav", up)

    # Mojo's output, in our standard fixture format.
    data = read_fixture(FIX / "mojo_audio.bin")
    save_wav(OUT_DIR / "mojo_real.wav", data)

    # Diff signal (upstream - mojo) for analysis.
    n = min(len(up.flatten()), len(data))
    diff = up.flatten()[:n] - data[:n]
    save_wav(OUT_DIR / "diff.wav", diff)
    print(f"diff stats: max abs {np.abs(diff).max():.4e}  rms {np.sqrt((diff**2).mean()):.4e}")


if __name__ == "__main__":
    main()
