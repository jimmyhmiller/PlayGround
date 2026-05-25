"""Dump 24kHz mel input/output for parity testing."""
import os, struct
import numpy as np
import librosa
import torch


def write_tensor(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def main():
    OUT = "weights/s3gen_prompt/mel24k_diag"
    os.makedirs(OUT, exist_ok=True)

    # Use a deterministic short audio (1 second).
    torch.manual_seed(0)
    n = 24000
    wav = torch.randn(n) * 0.1

    # Run upstream's mel_spectrogram directly.
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram
    spec = mel_spectrogram(wav, n_fft=1920, num_mels=80, sampling_rate=24000,
                            hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False)
    print(f"24k mel: shape={spec.shape} mean-abs={spec.abs().mean().item():.4f}")
    write_tensor(f"{OUT}/wav_24k.bin", wav.numpy())
    write_tensor(f"{OUT}/mel_24k.bin", spec.numpy())


if __name__ == "__main__":
    main()
