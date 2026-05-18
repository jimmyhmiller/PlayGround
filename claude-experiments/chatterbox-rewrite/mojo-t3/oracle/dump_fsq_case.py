"""Dump FSQ codebook fixture: random x → expected discrete indices."""
import os, struct
import numpy as np
import torch


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "fsq")
os.makedirs(OUT_DIR, exist_ok=True)


def write_fp32(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def write_i32(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int32))
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        # No i32 tag in our current fixture format; reuse fp32 tag and read raw.
        # We'll define a custom i32 reader.
        f.write(struct.pack("<i", 3))
        f.write(arr.tobytes())


class FSQCodebook(torch.nn.Module):
    def __init__(self, dim, level=3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level

    def encode(self, x):
        h = self.project_down(x).float()
        h = h.tanh() * 0.9990000128746033
        h = h.round() + 1
        powers = torch.pow(self.level, torch.arange(8).to(h.dtype))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        return mu.int()


def main():
    torch.manual_seed(0)
    B, T, D = 2, 16, 1280
    fsq = FSQCodebook(dim=D).eval()
    x = torch.randn(B, T, D)
    # Flatten and encode (s3tokenizer rearranges to "... d -> (...) d").
    h_flat = x.reshape(-1, D)
    with torch.inference_mode():
        idx = fsq.encode(h_flat).reshape(B, T)
    print("x:", x.shape, "idx:", idx.shape, "min/max:", int(idx.min()), int(idx.max()))
    write_fp32(os.path.join(OUT_DIR, "x.bin"), x.numpy())
    write_i32(os.path.join(OUT_DIR, "idx.bin"), idx.numpy())
    write_fp32(os.path.join(OUT_DIR, "project_down_w.bin"), fsq.project_down.weight.detach().numpy())
    write_fp32(os.path.join(OUT_DIR, "project_down_b.bin"), fsq.project_down.bias.detach().numpy())
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
