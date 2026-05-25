"""Test whether HiFTGenerator.decode is deterministic on GPU across repeated
calls with identical input — and how that depends on MIOpen find-mode.

The bug we're chasing: GPU output rms varies 0.028 to 0.067 across calls
with same input. We confirmed this by comparing saved gpu_out.pt vs
inline_gpu_out.pt from the same bundle — two consecutive forwards in the
same process gave different outputs.

This script reproduces the test with controlled MIOpen env vars. Set them
before invoking, e.g.:
    MIOPEN_FIND_MODE=2 python repeatability_test.py captures/grind/run_<id>/
    MIOPEN_DEBUG_CONV_WINOGRAD=0 python repeatability_test.py captures/grind/run_<id>/
"""
from __future__ import annotations

import argparse, copy, json, os, sys
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()

    # Mirror env to record what we tested.
    miopen_env = {k: v for k, v in os.environ.items() if k.startswith("MIOPEN_")}

    here = Path(__file__).resolve().parents[1]
    cb_src = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    sys.path.insert(0, str(cb_src))
    from chatterbox.models.s3gen.hifigan import HiFTGenerator  # type: ignore
    try:
        from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor  # type: ignore
        f0p = ConvRNNF0Predictor()
    except Exception:
        f0p = None

    init_kwargs = json.load((bundle / "hifigan_init_kwargs.json").open())
    state = torch.load(bundle / "hifigan_state_dict.pt", weights_only=False, map_location="cpu")
    mel = torch.load(bundle / "mel_in.pt", weights_only=False)
    s_cache = torch.load(bundle / "s_cache.pt", weights_only=False)

    model = HiFTGenerator(f0_predictor=f0p, **init_kwargs).eval()
    model.load_state_dict(state, strict=False)
    model = model.to(args.device)

    mel = mel.to(args.device)
    s_cache = s_cache.to(args.device)

    print(f"[repeat] device={args.device} mel_shape={tuple(mel.shape)}")
    print(f"[repeat] miopen env: {miopen_env}")
    print(f"\n{'trial':>5s}  {'rms':>8s}  {'cent':>6s}  {'<300':>5s}  {'ratio':>7s}")

    rms_list = []
    centroids = []
    below = []
    out_first = None
    diffs_to_first = []

    for i in range(args.n):
        torch.manual_seed(0)  # control the SineGen randoms
        with torch.inference_mode():
            wav, _ = model.inference(mel, s_cache)
        arr = wav.detach().to(torch.float32).cpu().reshape(-1).numpy()
        rms = float((arr ** 2).mean() ** 0.5)
        # Quick spectral metrics.
        from scipy.signal import welch
        f, p = welch(arr, model.sampling_rate, nperseg=2048)
        tot = float(p.sum()) + 1e-12
        cent = float((f * p).sum() / tot)
        below_300 = float(p[f < 300].sum() / tot)
        if out_first is None:
            out_first = arr
            diff_first = 0.0
        else:
            diff_first = float(((out_first - arr) ** 2).mean() ** 0.5)
        rms_list.append(rms)
        centroids.append(cent)
        below.append(below_300)
        diffs_to_first.append(diff_first)
        print(f"{i:>5d}  {rms:.4f}  {cent:>5.0f}  {below_300:.2f}  {diff_first:.4f}")

    print(f"\n[repeat] rms range: {min(rms_list):.4f} - {max(rms_list):.4f}")
    print(f"[repeat] rms stddev: {(sum((r-sum(rms_list)/len(rms_list))**2 for r in rms_list)/len(rms_list))**0.5:.4e}")
    print(f"[repeat] max diff to first run: {max(diffs_to_first):.4e}")

    out = bundle / "repeatability.json"
    out.write_text(json.dumps({
        "n": args.n, "device": args.device, "miopen_env": miopen_env,
        "rms": rms_list, "centroid_hz": centroids, "below_300hz": below,
        "diff_to_first_run": diffs_to_first,
    }, indent=2))
    print(f"[repeat] wrote {out}")


if __name__ == "__main__":
    main()
