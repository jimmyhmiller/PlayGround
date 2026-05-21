"""Phase B feasibility test: op_campplus on GPU buffers.

Run from project root:
    pixi run python -m chatterbox_mojo.test_campplus

Loads the existing CAMPPlus parity fixture (weights/campplus_parity/x.bin
+ expected.bin), runs xvector_forward through the .so, and compares to the
upstream torch expected output.

PASS criteria: cos_sim >= 0.999, rel_l2 < 0.01.
"""
import struct
import sys
from pathlib import Path

import numpy as np

import chatterbox_mojo  # noqa: F401 — sys.path bootstrap
import mojo.importer  # noqa: F401
import op_campplus  # type: ignore — compiled on demand

from max.driver import Accelerator, CPU, Buffer
from max.dtype import DType


WEIGHTS_BASE = "weights/s3gen/speaker_encoder"
PARITY_DIR = Path("weights/campplus_parity")


def load_fixture(path: Path) -> np.ndarray:
    """Read fixture format: 8-byte rank, then 8-byte dims, 4-byte dtype tag, fp32 data."""
    with open(path, "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        if tag != 0:
            raise ValueError(f"expected fp32 tag 0, got {tag}")
        n = 1
        for d in shape:
            n *= d
        data = np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(shape)
    return data


def main() -> int:
    print("[test] loading parity fixtures...")
    x = load_fixture(PARITY_DIR / "x.bin")              # (1, 320, 16)
    expected = load_fixture(PARITY_DIR / "expected.bin")  # (1, 192)
    print(f"[test]   x.shape={x.shape}  expected.shape={expected.shape}")

    print("[test] creating GPU accelerator + buffers...")
    gpu = Accelerator()
    print(f"[test]   {gpu}")
    dctx_ptr = gpu._device_context_ptr()
    print(f"[test]   device_context_ptr=0x{dctx_ptr:x}")

    # Stage input as a host CPU buffer, then move to GPU via Buffer.to().
    x_contig = np.ascontiguousarray(x, dtype=np.float32)
    host_in = Buffer.from_numpy(x_contig).to(gpu)
    in_buf = host_in  # already on GPU
    out_buf = Buffer(shape=(1, 192), dtype=DType.float32, device=gpu)

    print("[test] init_op (loads ~937 weight files to GPU)...")
    handle = op_campplus.init_op(WEIGHTS_BASE, dctx_ptr)
    print(f"[test]   handle=0x{handle:x}")

    try:
        print("[test] xvector_forward...")
        op_campplus.xvector_forward(handle, in_buf, out_buf, 1, 16)

        # Pull back result and compare.
        got = out_buf.to_numpy().reshape(1, 192)
        print(f"[test]   got.shape={got.shape}  dtype={got.dtype}")

        diff = got - expected
        max_abs = float(np.abs(diff).max())
        rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(expected))
        cos_sim = float(
            (got.ravel() @ expected.ravel())
            / (np.linalg.norm(got) * np.linalg.norm(expected) + 1e-30)
        )
        print(f"[test]   max_abs={max_abs:.6e}  rel_l2={rel_l2:.6e}  cos_sim={cos_sim:.6f}")
        print(f"[test]   first 8 got:      {got.ravel()[:8]}")
        print(f"[test]   first 8 expected: {expected.ravel()[:8]}")

        if cos_sim >= 0.999 and rel_l2 < 0.01:
            print("[PASS] op_campplus parity")
            return 0
        else:
            print(f"[FAIL] parity below threshold (need cos_sim>=0.999, rel_l2<0.01)")
            return 1
    finally:
        op_campplus.destroy_op(handle)


if __name__ == "__main__":
    sys.exit(main())
