"""Convert s3gen weight .bin files to bf16 alongside.

Targets: every Linear weight file (`weight.bin`) in weights/s3gen/flow/
that's part of a 2D Linear weight. Skips conv weights (1D bias / 4D conv kernels).

File format (matches src/fixture.mojo):
  i64 rank
  i64 shape[i] for i in [0, rank)
  i32 tag         (0 = fp32, 1 = bf16)
  payload (4 bytes/elem for f32, 2 bytes/elem for bf16)
"""
import os, sys, struct
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "weights" / "s3gen"


def read_f32(path: Path):
    with open(path, "rb") as f:
        data = f.read()
    rank = struct.unpack_from("<q", data, 0)[0]
    off = 8
    shape = []
    for _ in range(rank):
        shape.append(struct.unpack_from("<q", data, off)[0])
        off += 8
    tag = struct.unpack_from("<i", data, off)[0]
    off += 4
    if tag != 0:
        return None, None
    n = 1
    for s in shape:
        n *= s
    arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).copy().reshape(shape)
    return arr, shape


def write_bf16(path: Path, arr: np.ndarray):
    shape = list(arr.shape)
    bf16 = torch.from_numpy(arr).to(torch.bfloat16)
    raw = bf16.contiguous().view(torch.uint16).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(shape)))
        for s in shape:
            f.write(struct.pack("<q", s))
        f.write(struct.pack("<i", 1))   # tag = 1 (bf16)
        f.write(raw)


def main():
    converted = 0
    skipped = 0
    for fp in ROOT.rglob("*.bin"):
        if fp.name.endswith(".bf16.bin"):
            continue
        # We want 2D Linear weights only. Read header to check rank=2.
        with open(fp, "rb") as f:
            head = f.read(16)
        if len(head) < 16:
            continue
        rank = struct.unpack_from("<q", head, 0)[0]
        if rank != 2:
            skipped += 1
            continue
        # Check the file is f32 (tag=0).
        arr, shape = read_f32(fp)
        if arr is None:
            skipped += 1
            continue
        out = fp.with_name(fp.stem + ".bf16.bin")
        if out.exists() and out.stat().st_mtime > fp.stat().st_mtime:
            continue
        write_bf16(out, arr)
        converted += 1
        if converted < 5 or converted % 50 == 0:
            print(f"  [{converted}] {fp.relative_to(ROOT)}  shape={shape}")
    print(f"done — converted {converted}, skipped {skipped} non-2D")


if __name__ == "__main__":
    sys.exit(main() or 0)
