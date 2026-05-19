"""Dump s3tokenizer mel + tokens for parity testing."""
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


def write_i64(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 2))
        f.write(arr.tobytes())


def main():
    OUT = "weights/s3gen_prompt/s3tok_diag"
    os.makedirs(OUT, exist_ok=True)

    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained("cpu")
    s3tok = m.s3gen.tokenizer
    s3tok.eval()

    ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
    wav_24, _ = librosa.load(ref_path, sr=24000)
    wav_16 = librosa.resample(wav_24, orig_sr=24000, target_sr=16000)
    print(f"wav_16 shape: {wav_16.shape}")
    write_tensor(f"{OUT}/wav_16k.bin", wav_16)

    with torch.inference_mode():
        wav_16_t = torch.from_numpy(wav_16).unsqueeze(0)
        # log_mel_spectrogram
        log_mel = s3tok.log_mel_spectrogram(wav_16_t)  # (1, 128, T)
        # full forward
        tokens, tok_lens = s3tok(wav_16_t)

    print(f"log_mel shape: {log_mel.shape} mean={log_mel.mean().item():.4f}")
    print(f"tokens shape: {tokens.shape} (first 20: {tokens[0, :20].tolist()})")
    write_tensor(f"{OUT}/log_mel_16k.bin", log_mel.numpy())
    write_i64(f"{OUT}/tokens.bin", tokens.long().cpu().numpy())

    print(f"Wrote to {OUT}/")


if __name__ == "__main__":
    main()
