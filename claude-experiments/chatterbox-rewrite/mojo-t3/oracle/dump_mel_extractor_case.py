"""
Dump upstream Chatterbox mel-spectrogram of the reference voice WAV so Mojo
can verify its own mel extractor.

Writes tests/fixtures/mel_extractor/:
  ref_wav.bin             (1, T_samples)   raw audio in [-1, 1]
  ref_mel.bin             (1, num_mels, T_mel)
  mel_basis.bin           (80, n_freq=961)
  hann_window.bin         (1920,)
  meta.bin                int64: n_fft, hop, win, num_mels, sr, fmin, fmax_int

Note: librosa's mel filterbank is fixed (a function of sr, n_fft, num_mels,
fmin, fmax). We dump it so Mojo can use it directly instead of recomputing
the filter math.
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import torch
import wave

import sys
sys.path.insert(0, "../chatterbox/src")
# Try to import the upstream mel_spectrogram. Fall back to inline copy if
# librosa isn't here.
try:
    from chatterbox.models.s3gen.utils.mel import mel_spectrogram
except Exception:
    mel_spectrogram = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "mel_extractor"
OUT.mkdir(parents=True, exist_ok=True)

REF_WAV = Path("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")

N_FFT = 1920
HOP = 480
WIN = 1920
NUM_MELS = 80
SR = 24000
FMIN = 0
FMAX = 8000


def write_tensor(path, arr):
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(arr.dtype)
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def main():
    # Load reference WAV via stdlib wave module.
    with wave.open(str(REF_WAV), "rb") as wf:
        nch = wf.getnchannels()
        sr_in = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    assert sampwidth == 2, f"expected 16-bit PCM, got {sampwidth} bytes"
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, nch)
    if nch > 1:
        pcm = pcm.mean(axis=1)
    else:
        pcm = pcm[:, 0]
    audio = pcm.astype(np.float32) / 32768.0
    if sr_in != SR:
        # Cheap polyphase resample for the oracle.
        import scipy.signal as sig
        audio = sig.resample_poly(audio, SR, sr_in).astype(np.float32)
    wav = torch.from_numpy(audio).unsqueeze(0)
    # Truncate to ~10s.
    wav = wav[:, : 10 * SR]
    print(f"wav shape: {tuple(wav.shape)}  dtype={wav.dtype}")
    write_tensor(OUT / "ref_wav.bin", wav.numpy().astype(np.float32))

    # Compute mel via upstream.
    if mel_spectrogram is not None:
        mel = mel_spectrogram(
            wav.numpy(), n_fft=N_FFT, num_mels=NUM_MELS, sampling_rate=SR,
            hop_size=HOP, win_size=WIN, fmin=FMIN, fmax=FMAX, center=False,
        )
    else:
        # Inline librosa+torch reproduction.
        from librosa.filters import mel as librosa_mel_fn
        mb = librosa_mel_fn(sr=SR, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX)
        mb_t = torch.from_numpy(mb).float()
        hw = torch.hann_window(WIN)
        y = torch.nn.functional.pad(
            wav.unsqueeze(1), (int((N_FFT - HOP) / 2), int((N_FFT - HOP) / 2)),
            mode="reflect",
        ).squeeze(1)
        spec = torch.view_as_real(torch.stft(
            y, N_FFT, hop_length=HOP, win_length=WIN, window=hw,
            center=False, pad_mode="reflect", normalized=False, onesided=True,
            return_complex=True,
        ))
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        mel = torch.matmul(mb_t, spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))
    print(f"mel shape: {tuple(mel.shape)}")
    write_tensor(OUT / "ref_mel.bin", mel.numpy().astype(np.float32))

    # Also dump the mel basis (librosa filter bank) and hann window so Mojo
    # uses the same fixed coefficients.
    from librosa.filters import mel as librosa_mel_fn
    mb = librosa_mel_fn(sr=SR, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX)
    write_tensor(OUT / "mel_basis.bin", mb.astype(np.float32))
    hw = torch.hann_window(WIN).numpy()
    write_tensor(OUT / "hann_window.bin", hw.astype(np.float32))

    write_tensor(OUT / "meta.bin", np.array(
        [N_FFT, HOP, WIN, NUM_MELS, SR, FMIN, FMAX], dtype=np.int64,
    ))


if __name__ == "__main__":
    main()
