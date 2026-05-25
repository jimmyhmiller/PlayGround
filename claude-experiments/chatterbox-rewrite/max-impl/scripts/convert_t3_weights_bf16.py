"""Convert T3 fp32 weight .bin files to bf16 alongside.

For each weights/t3/**/*.bin matching the T3 Linear weight files (qw, kw, vw,
ow, gate_w, up_w, down_w, speech_emb_w, speech_head_w, text_emb_w),
read the f32 fixture, round-to-nearest cast to bfloat16, write back with
tag=1 (bf16). RMSNorm weights stay f32.

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

ROOT = Path(__file__).resolve().parent.parent / "weights" / "t3"

# Files we want to convert (Linear weights only — leave norms alone).
TARGET_BASENAMES = {
    "qw", "kw", "vw", "ow", "gate_w", "up_w", "down_w",
    "speech_emb_w", "speech_head_w", "text_emb_w", "text_pos_w", "speech_pos_w",
}


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
    total = 0
    converted = 0
    for fp in ROOT.rglob("*.bin"):
        if fp.name.endswith(".bf16.bin"):
            continue
        base = fp.stem    # filename without .bin
        if base not in TARGET_BASENAMES:
            continue
        total += 1
        out = fp.with_name(fp.stem + ".bf16.bin")
        if out.exists() and out.stat().st_mtime > fp.stat().st_mtime:
            continue
        arr, shape = read_f32(fp)
        if arr is None:
            print(f"skip {fp}: not f32")
            continue
        write_bf16(out, arr)
        converted += 1
        if converted < 5 or converted % 30 == 0:
            print(f"  [{converted}] {fp.relative_to(ROOT)}  shape={shape}  → {out.name}")
    print(f"done — converted {converted}/{total} files")


if __name__ == "__main__":
    sys.exit(main() or 0)
