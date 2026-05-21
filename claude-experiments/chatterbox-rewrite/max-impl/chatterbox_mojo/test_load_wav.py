"""Phase A feasibility test: build + load op_load_wav.so, decode a WAV.

Run from project root:
    pixi run python -m chatterbox_mojo.test_load_wav <path/to/wav>

Exits 0 on success. Validates:
1. mojo.importer compiles the .so cold
2. The .so loads and exposes get_wav_size + load_wav_into
3. max.driver.Buffer allocation works
4. Mojo can read the buffer pointer and write into it
5. Decoded samples are sane (in [-1, 1])
"""
import sys
from pathlib import Path

# Bootstrap path setup so ops/op_load_wav/ is on sys.path.
import chatterbox_mojo  # noqa: F401 — side effect: extends sys.path

import mojo.importer  # noqa: F401 — registers the .mojo loader
import op_load_wav  # type: ignore — compiled on demand

from max.driver import Accelerator, CPU, Buffer
from max.dtype import DType


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m chatterbox_mojo.test_load_wav <wav>", file=sys.stderr)
        return 2
    wav_path = sys.argv[1]
    if not Path(wav_path).is_file():
        print(f"file not found: {wav_path}", file=sys.stderr)
        return 2

    print(f"[test] reading header: {wav_path}")
    n_samples, sample_rate = op_load_wav.get_wav_size(wav_path)
    print(f"[test]   n_samples={n_samples}  sample_rate={sample_rate}")

    if n_samples <= 0:
        print("[FAIL] n_samples <= 0", file=sys.stderr)
        return 1

    # Allocate a host-side float32 buffer of the right size.
    host = CPU()
    buf = Buffer(shape=(n_samples,), dtype=DType.float32, device=host)
    print(f"[test] allocated buffer: shape={buf.shape} dtype={buf.dtype} device={buf.device}")

    sr2 = op_load_wav.load_wav_into(buf, wav_path)
    print(f"[test]   decoded; sample_rate from op={sr2}")
    if sr2 != sample_rate:
        print(f"[FAIL] sample rate mismatch", file=sys.stderr)
        return 1

    # Pull samples back as numpy purely for *validation* — no compute, just
    # sanity checks on min/max and a non-zero-energy heuristic.
    arr = buf.to_numpy()
    print(f"[test]   numpy view: shape={arr.shape} dtype={arr.dtype}")
    mn, mx = float(arr.min()), float(arr.max())
    rms = float((arr ** 2).mean() ** 0.5)
    print(f"[test]   min={mn:.4f}  max={mx:.4f}  rms={rms:.4f}")
    if mn < -1.001 or mx > 1.001:
        print(f"[FAIL] samples out of [-1, 1]", file=sys.stderr)
        return 1
    if rms < 1e-6:
        print(f"[FAIL] zero-energy audio (likely decode error)", file=sys.stderr)
        return 1

    print("[PASS] op_load_wav round-trip works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
