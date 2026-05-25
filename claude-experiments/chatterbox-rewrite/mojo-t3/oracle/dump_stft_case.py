"""
STFT / iSTFT parity case matching the HiFiGAN configuration:
  n_fft = 16, hop_length = 4, window = hann, center=True, return_complex=True

We dump a small input signal, the STFT output (real & imag), and the iSTFT
round-trip of the same data. The Mojo kernels must match these bit-tolerantly.

Binary format matches the others (rank, shape, tag, payload).
"""

from __future__ import annotations
import struct
from pathlib import Path

import numpy as np
import torch


def hann_window(n: int) -> np.ndarray:
    # Periodic Hann window matching scipy.signal.get_window("hann", n, fftbins=True)
    # and torch.hann_window(n, periodic=True): w[k] = 0.5 - 0.5 * cos(2*pi*k/n).
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def get_window(name: str, n: int, fftbins: bool = True) -> np.ndarray:
    assert name == "hann" and fftbins, "only hann (periodic) supported"
    return hann_window(n).astype(np.float32)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "stft"
OUT.mkdir(parents=True, exist_ok=True)

N_FFT = 16
HOP = 4
T_IN = 64       # samples; chosen to be > n_fft to exercise multiple frames


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag, raw = 1, arr.astype(np.uint16, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def main() -> None:
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    x = torch.randn(1, T_IN, generator=g, dtype=torch.float32) * 0.1
    window = torch.from_numpy(get_window("hann", N_FFT, fftbins=True).astype(np.float32))

    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=window, return_complex=True)
    # spec shape: (B=1, n_freq=N_FFT//2+1, n_frames)
    real = spec.real.contiguous()
    imag = spec.imag.contiguous()
    n_freq = N_FFT // 2 + 1
    n_frames = real.shape[2]
    print(f"[stft] x{tuple(x.shape)} -> spec{tuple(spec.shape)} (real+imag)")
    print(f"  n_freq={n_freq} n_frames={n_frames}")

    write_tensor(OUT / "x.bin", x.numpy().astype(np.float32))
    write_tensor(OUT / "window.bin", window.numpy().astype(np.float32))
    write_tensor(OUT / "real.bin", real.numpy().astype(np.float32))
    write_tensor(OUT / "imag.bin", imag.numpy().astype(np.float32))

    # iSTFT round-trip on a perturbed spec (so it's not just identity).
    # We'll use the original spec and compare iSTFT(spec) ≈ x (within
    # boundary effects: torch.istft drops the first/last n_fft//2 samples
    # because center=True padded them in).
    inverse = torch.istft(spec, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                          window=window, length=T_IN)
    print(f"[istft] spec{tuple(spec.shape)} -> inv{tuple(inverse.shape)}")
    write_tensor(OUT / "istft_expected.bin", inverse.numpy().astype(np.float32))

    write_tensor(OUT / "meta.bin",
                 np.array([1, T_IN, N_FFT, HOP, n_freq, n_frames], dtype=np.int64))


if __name__ == "__main__":
    main()
