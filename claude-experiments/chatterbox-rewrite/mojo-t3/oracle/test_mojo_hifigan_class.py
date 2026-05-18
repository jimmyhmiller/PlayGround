"""
Test the MojoHifigan class as paper-audiobooks would use it: pass in a real
mel + real source STFT (numpy), get back audio.
"""
import sys
import struct
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from mojo_hifigan_py import MojoHifigan


def read_fixture(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        assert tag == 0
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def main():
    FIX = Path("tests/fixtures/real")

    # Load a real cloned-voice mel and the upstream source STFT.
    mel = read_fixture(FIX / "real_mel.bin")
    s_stft = read_fixture(FIX / "real_s_stft_cat.bin")
    upstream_audio = read_fixture(FIX / "real_audio_upstream.bin")

    print(f"input mel: {mel.shape}")
    print(f"input s_stft: {s_stft.shape}")
    print(f"upstream audio: {upstream_audio.shape}")

    # paper-audiobooks-style call.
    mh = MojoHifigan(work_dir=".")
    audio = mh.synthesize(mel, s_stft)
    print(f"\nmojo audio: {audio.shape}")

    diff = np.abs(audio.flatten() - upstream_audio.flatten())
    print(f"\nvs upstream Chatterbox:")
    print(f"  max abs: {diff.max():.4e}")
    print(f"  mean abs: {diff.mean():.4e}")
    print(f"  rms: {np.sqrt((diff**2).mean()):.4e}")

    # Save WAVs.
    import wave
    OUT = Path("tests/fixtures/real_wavs")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sig in [("mojo_class.wav", audio), ("upstream_class.wav", upstream_audio)]:
        pcm = (np.clip(sig.flatten(), -0.99, 0.99) * 32767.0).astype(np.int16)
        with wave.open(str(OUT / name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(pcm.tobytes())
        print(f"wrote {OUT / name}")

    if diff.max() < 0.3:
        print("\n✓ paper-audiobooks-style integration: PASS (audible-quality match)")
    else:
        print(f"\n✗ FAILED: max diff {diff.max()} > 0.3")


if __name__ == "__main__":
    main()
