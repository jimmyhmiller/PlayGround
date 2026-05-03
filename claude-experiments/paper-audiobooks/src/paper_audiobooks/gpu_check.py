"""Verify torch is using the AMD GPU on this machine.

    uv run python -m paper_audiobooks.gpu_check

If this prints `cuda: False`, the venv has the wrong torch build (probably
the PyPI CUDA build silently picked up by a transitive dep). See CLAUDE.md.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    import torch

    print(f"torch:          {torch.__version__}")
    print(f"cuda build:     {torch.version.cuda}")
    print(f"hip build:      {torch.version.hip}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count:   {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print()
        print("FAIL: no GPU visible to torch.")
        print("This machine has an AMD Radeon 8060S (gfx1151).")
        print("If torch.version.hip is None, you have the CUDA build — re-pin")
        print("torch to the rocm index (see pyproject.toml + CLAUDE.md) and")
        print("`uv sync` again.")
        return 1

    name = torch.cuda.get_device_name(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"device:         {name} ({arch})")

    # Tiny matmul to make sure HIP kernels actually launch.
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"2048x2048 matmul: {dt * 1000:.1f} ms (sum={float(c.sum()):.2f})")
    print()
    print("OK: torch is using the GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
