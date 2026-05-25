"""Dump librosa.resample 24kHz->16kHz for parity testing."""
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
    OUT = "weights/s3gen_prompt/resample_diag"
    os.makedirs(OUT, exist_ok=True)

    # Use the default voice WAV.
    ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
    wav_24, sr = librosa.load(ref_path, sr=24000)
    print(f"input wav: shape={wav_24.shape} sr={sr}")
    write_tensor(f"{OUT}/wav_24k.bin", wav_24)

    wav_16 = librosa.resample(wav_24, orig_sr=24000, target_sr=16000)
    print(f"resampled to 16k: shape={wav_16.shape}")
    write_tensor(f"{OUT}/wav_16k_soxr.bin", wav_16)

    # Also dump the "scipy" version for comparison (uses polyphase with fir filter).
    wav_16_scipy = librosa.resample(wav_24, orig_sr=24000, target_sr=16000, res_type="scipy")
    write_tensor(f"{OUT}/wav_16k_scipy.bin", wav_16_scipy)
    print(f"scipy resampled: shape={wav_16_scipy.shape}")

    diff_soxr_scipy = np.abs(wav_16 - wav_16_scipy).max()
    print(f"max-abs soxr vs scipy: {diff_soxr_scipy:.6f}")

    # Also dump the speaker embedding through the upstream CAMPPlus speaker_encoder.
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained("cpu")
    spk_enc = m.s3gen.speaker_encoder
    spk_enc.eval()
    with torch.inference_mode():
        emb = spk_enc.inference([torch.from_numpy(wav_16)])
    print(f"speaker emb from upstream wav_16: mean-abs={emb.abs().mean().item():.4f}")
    write_tensor(f"{OUT}/speaker_emb_from_wav.bin", emb.cpu().numpy())


if __name__ == "__main__":
    main()
