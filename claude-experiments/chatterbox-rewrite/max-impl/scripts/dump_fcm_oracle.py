"""Dump FCM input/output for a known mel input to test Mojo's FCM impl."""
import os, struct
import numpy as np
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
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained("cpu")
    spk_enc = m.s3gen.speaker_encoder
    fcm = spk_enc.head
    spk_enc.eval()

    OUT = "weights/s3gen_prompt/fcm_diag"
    os.makedirs(OUT, exist_ok=True)

    # Use a fixed deterministic input.
    torch.manual_seed(0)
    B, FEAT, T = 1, 80, 64
    x = torch.randn(B, FEAT, T)
    print(f"FCM input: shape={x.shape}")

    with torch.inference_mode():
        out = fcm(x)
        # Full speaker_encoder forward expects (B, T, F) then permutes to (B, F, T).
        # So we feed x.transpose(1,2) → (B, T, F).
        full = spk_enc(x.transpose(1, 2))
    print(f"FCM output: shape={out.shape} mean-abs={out.abs().mean().item():.4f}")
    print(f"Full speaker_enc output: shape={full.shape} mean-abs={full.abs().mean().item():.4f}")

    write_tensor(f"{OUT}/fcm_input.bin", x.numpy())
    write_tensor(f"{OUT}/fcm_output.bin", out.numpy())
    write_tensor(f"{OUT}/speaker_emb.bin", full.numpy())
    print(f"Wrote to {OUT}/")


if __name__ == "__main__":
    main()
