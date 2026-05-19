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
    fcm = m.s3gen.speaker_encoder.head
    fcm.eval()

    OUT = "weights/s3gen_prompt/fcm_diag"
    os.makedirs(OUT, exist_ok=True)

    # Use a fixed deterministic input.
    torch.manual_seed(0)
    B, FEAT, T = 1, 80, 64
    x = torch.randn(B, FEAT, T)
    print(f"FCM input: shape={x.shape}")

    with torch.inference_mode():
        out = fcm(x)
    print(f"FCM output: shape={out.shape} mean-abs={out.abs().mean().item():.4f}")

    write_tensor(f"{OUT}/fcm_input.bin", x.numpy())
    write_tensor(f"{OUT}/fcm_output.bin", out.numpy())
    print(f"Wrote to {OUT}/")


if __name__ == "__main__":
    main()
