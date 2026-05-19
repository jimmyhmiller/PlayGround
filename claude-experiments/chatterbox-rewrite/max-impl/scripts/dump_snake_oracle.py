"""Dump Snake activation oracle: y = x + (1/alpha) * sin^2(x*alpha)."""
import os, struct
import numpy as np
import torch

CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
OUT = "weights/hift_parity"


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def main():
    from chatterbox.models.s3gen.hifigan import Snake

    torch.manual_seed(0)
    C, T = 256, 32
    x = torch.zeros(1, C, T)
    for ci in range(C):
        for ti in range(T):
            x[0, ci, ti] = np.sin(ci * 0.05 + ti * 0.1) * 0.3

    # Use the trained alpha values from resblocks[0].activations1[0].
    from safetensors.torch import safe_open
    p = os.path.join(CKPT, "s3gen.safetensors")
    with safe_open(p, framework="pt") as f:
        alpha = f.get_tensor("mel2wav.resblocks.0.activations1.0.alpha").clone()
    print("alpha shape:", alpha.shape, "min/max:", alpha.min().item(), alpha.max().item())

    snake = Snake(C, alpha_logscale=False)
    with torch.no_grad():
        snake.alpha.copy_(alpha)
    snake.eval()

    with torch.inference_mode():
        y = snake(x)

    print("y mean-abs:", y.abs().mean().item())
    write_tensor(f"{OUT}/snake_x.bin", x.numpy())
    write_tensor(f"{OUT}/snake_alpha.bin", alpha.numpy())
    write_tensor(f"{OUT}/snake_y.bin", y.numpy())


if __name__ == "__main__":
    main()
