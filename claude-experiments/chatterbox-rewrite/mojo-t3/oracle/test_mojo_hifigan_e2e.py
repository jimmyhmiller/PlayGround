"""
End-to-end Python→Mojo→Python parity test of the Mojo HiFiGAN extension.

For each function the .so exposes:
  conv_pre_step(mel, weights_dir) -> (1, 512, 32)
  ups0_step(x_lrelu, weights_dir) -> (1, 256, 256)

We feed real upstream input, call Mojo, and check against the upstream-dumped
intermediate.
"""
import sys
import struct
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

import mojo_hifigan
print("imported:", mojo_hifigan.__name__)


FIX = Path("tests/fixtures/hifigan")
W = FIX / "weights"


def read_fixture(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        shape = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        assert tag == 0, f"expected fp32 tag 0, got {tag}"
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)


def main():
    mel = read_fixture(FIX / "mel.bin")                              # (1, 80, 32)
    expected_pre = read_fixture(FIX / "stage_after_conv_pre.bin")     # (1, 512, 32)
    expected_lrelu = read_fixture(FIX / "stage_up0_after_lrelu.bin")  # (1, 512, 32)
    expected_ups = read_fixture(FIX / "stage_up0_after_transposed_conv.bin")  # (1, 256, 256)

    print(f"input mel: {mel.shape}")

    # 1. conv_pre via Mojo.
    out_pre = mojo_hifigan.conv_pre_step(mel, str(W))
    print(f"conv_pre output: {out_pre.shape}  dtype={out_pre.dtype}")
    diff = np.abs(out_pre - expected_pre)
    print(f"  vs upstream: max abs = {diff.max():.4e}  mean = {diff.mean():.4e}")
    assert diff.max() < 1e-4, f"conv_pre parity failed: {diff.max()}"

    # 2. leaky_relu (in Python — trivial; matches upstream's torch.nn.functional.leaky_relu).
    out_lrelu = np.where(out_pre > 0, out_pre, 0.1 * out_pre)
    diff = np.abs(out_lrelu - expected_lrelu)
    print(f"leaky_relu: max abs = {diff.max():.4e}")

    # 3. ups[0] via Mojo.
    out_ups = mojo_hifigan.ups0_step(out_lrelu, str(W))
    print(f"ups[0] output: {out_ups.shape}")
    diff = np.abs(out_ups - expected_ups)
    print(f"  vs upstream: max abs = {diff.max():.4e}  mean = {diff.mean():.4e}")
    assert diff.max() < 1e-3, f"ups[0] parity failed: {diff.max()}"

    print("\n✓ Python ↔ Mojo HiFiGAN integration: PASS")


if __name__ == "__main__":
    main()
