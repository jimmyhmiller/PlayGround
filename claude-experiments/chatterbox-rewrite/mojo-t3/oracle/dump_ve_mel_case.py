"""Dump VE mel extractor fixture: wav → 40-bin mel (no log, no normalize).

Avoids librosa by hand-rolling the librosa-compatible STFT + mel filterbank
in numpy. This keeps the pixi env light and matches what we'll do in Mojo.
"""
import os, struct
import numpy as np
import torch


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "ve_mel")
os.makedirs(OUT_DIR, exist_ok=True)


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


# ---- librosa-compatible STFT (center=True, pad_mode='reflect') ----
def hann_periodic(n):
    return 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft(wav, n_fft=400, hop=160, win_size=400):
    """Match librosa.stft(n_fft, hop, win_length=win_size, center=True,
    pad_mode='reflect', window='hann'). Returns (n_bins, T) complex."""
    window = hann_periodic(win_size)  # librosa: scipy.signal.get_window('hann', win_length, fftbins=True)
    # librosa pads window centered in n_fft if win_length < n_fft (here win_size==n_fft so no-op).
    pad = n_fft // 2
    wav = np.pad(wav, pad, mode="reflect")
    n_frames = 1 + (len(wav) - n_fft) // hop
    spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for f in range(n_frames):
        frame = wav[f * hop : f * hop + n_fft] * window
        spec[:, f] = np.fft.rfft(frame, n=n_fft)
    return spec


# ---- librosa-compatible mel filterbank (slaney norm) ----
def mel_to_hz(m):
    f_min, f_sp = 0.0, 200.0 / 3
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    m = np.asarray(m, dtype=np.float32)
    return np.where(m >= min_log_mel,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)),
                    f_min + f_sp * m).astype(np.float32)


def hz_to_mel(hz):
    f_min, f_sp = 0.0, 200.0 / 3
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.asarray(hz, dtype=np.float32)
    return np.where(hz >= min_log_hz,
                    min_log_mel + np.log(hz / min_log_hz) / logstep,
                    (hz - f_min) / f_sp).astype(np.float32)


def mel_filter(sr=16000, n_fft=400, n_mels=40, fmin=0.0, fmax=8000.0):
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')"""
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2, dtype=np.float32)
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mel_pts = np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float32)
    mel_f = mel_to_hz(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # Slaney normalization.
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[: n_mels])
    weights *= enorm[:, None]
    return weights


def reference_mel(wav, sr=16000, n_fft=400, hop=160, win_size=400,
                  n_mels=40, fmin=0.0, fmax=8000.0, mel_power=2.0):
    """Matches chatterbox VoiceEncoder melspectrogram() with hp defaults."""
    spec = stft(wav, n_fft=n_fft, hop=hop, win_size=win_size)
    mag = np.abs(spec) ** mel_power
    bank = mel_filter(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = bank @ mag                    # (n_mels, T)
    return mel


def main():
    # Use the default-voice reference audio if available, otherwise synthesize.
    default_wav = os.path.expanduser("~/.config/paper-audiobooks/default-voice.wav")
    sr = 16000
    if os.path.exists(default_wav):
        import wave
        w = wave.open(default_wav, "rb")
        frames = w.readframes(w.getnframes())
        wav_full = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        in_sr = w.getframerate()
        w.close()
        # Light naive resample to 16 kHz if needed (linear).
        if in_sr != sr:
            n_in = len(wav_full)
            n_out = int(round(n_in * sr / in_sr))
            x = np.linspace(0, n_in - 1, n_out).astype(np.float32)
            wav_full = np.interp(x, np.arange(n_in), wav_full).astype(np.float32)
        # Use a 2-second chunk (32000 samples).
        chunk = min(len(wav_full), 2 * sr)
        wav = wav_full[:chunk]
    else:
        rng = np.random.default_rng(0)
        wav = rng.standard_normal(2 * sr).astype(np.float32) * 0.1

    print(f"wav: shape={wav.shape}, sr={sr}, dtype={wav.dtype}")
    mel = reference_mel(wav, sr=sr)          # (40, T)
    print(f"mel: shape={mel.shape}, min={mel.min():.6f}, max={mel.max():.6f}")
    # Save (B=1, T_frames, N_MEL).
    mel_btm = mel.T[None, :, :]              # (1, T, 40)

    # Mel filter bank.
    bank = mel_filter()
    print(f"bank: shape={bank.shape}, max={bank.max():.4f}")

    # Reflect-padded signal length so the Mojo side can preallocate.
    pad = 400 // 2
    n_padded = len(wav) + 2 * pad

    write_tensor(os.path.join(OUT_DIR, "wav.bin"), wav[None, :])     # (1, L)
    write_tensor(os.path.join(OUT_DIR, "mel.bin"), mel_btm)
    write_tensor(os.path.join(OUT_DIR, "bank.bin"), bank)
    with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
        f.write(f"L={len(wav)}\nL_padded={n_padded}\nT_frames={mel.shape[1]}\nN_BINS={1 + 400//2}\nN_MEL=40\nN_FFT=400\nHOP=160\n")
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
