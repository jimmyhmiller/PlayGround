"""Dump Kaldi fbank input/output for parity testing."""
import os, struct
import numpy as np
import torch
import torchaudio.compliance.kaldi as Kaldi


def write_tensor(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def main():
    OUT = "weights/s3gen_prompt/kaldi_diag"
    os.makedirs(OUT, exist_ok=True)

    torch.manual_seed(0)
    n_samples = 16000 + 240   # ~1.01s
    wav = torch.randn(1, n_samples) * 0.1
    print(f"input wav shape: {wav.shape}")

    out = Kaldi.fbank(wav, num_mel_bins=80)
    print(f"kaldi output: shape={out.shape} mean-abs={out.abs().mean().item():.4f} "
          f"min={out.min().item():.3f} max={out.max().item():.3f}")

    # Also after subtract_mean
    out_sub = out - out.mean(dim=0, keepdim=True)
    print(f"after subtract-col-mean: mean-abs={out_sub.abs().mean().item():.4f}")

    write_tensor(f"{OUT}/wav.bin", wav.squeeze(0).numpy())
    write_tensor(f"{OUT}/fbank_raw.bin", out.numpy())
    write_tensor(f"{OUT}/fbank_after_mean.bin", out_sub.numpy())
    print(f"Wrote to {OUT}/")


if __name__ == "__main__":
    main()
