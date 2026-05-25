"""Check whether torch.stft (rocFFT) is the source of divergence.

The bisect showed source_downs.0 is the first layer to diverge from fp32
noise (3.6e-4 mean_abs). But standalone replay of source_downs.0 with clean
input agrees with CPU to fp32 noise. So the divergence in the full forward
must come from source_downs.0 receiving a DIFFERENT input — and that input
is `s_stft` from torch.stft inside HiFTGenerator._stft.

This script:
1. Loads the captured s_cache (input to _stft) from the bundle.
2. Computes s_stft on CPU and GPU N times.
3. Reports the diff distribution.

If torch.stft on GPU is deterministic and matches CPU to fp32 noise, the
divergence is elsewhere. If it diverges, this is the upstream root cause
of the resblocks.3 amplification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()

    # Reconstruct the inputs the way HiFTGenerator.inference does. We need:
    #   - speech_feat (mel) -> f0_predictor -> f0
    #   - f0 -> f0_upsamp -> ... -> m_source -> s
    #   - s with cache_source merged
    # Easiest: just replay model.inference once on CPU and capture s before
    # _stft. Then we can compute _stft on both devices.

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
    state = torch.load(bundle / "hifigan_state_dict.pt", map_location="cpu",
                       weights_only=False)
    mel = torch.load(bundle / "mel_in.pt", weights_only=False).cpu()
    cache = torch.load(bundle / "s_cache.pt", weights_only=False).cpu()

    model = HiFTGenerator(f0_predictor=f0p, **init_kwargs).eval().cpu()
    model.load_state_dict(state, strict=False)

    # Run the source pipeline manually (mirroring inference()) up to the
    # point _stft would consume `s`.
    with torch.inference_mode():
        f0 = model.f0_predictor(mel)
        s = model.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = model.m_source(s)
        s = s.transpose(1, 2)
        if cache.shape[2] != 0:
            s[:, :, :cache.shape[2]] = cache

    # Now compute _stft on CPU vs GPU and compare. The window is on the
    # device the input is on, per the model code.
    n_fft = init_kwargs["istft_params"]["n_fft"]
    hop_len = init_kwargs["istft_params"]["hop_len"]

    from scipy.signal import get_window
    window_cpu = torch.from_numpy(
        get_window("hann", n_fft, fftbins=True).astype(np.float32)
    )

    def stft(x_dev, win_dev):
        spec = torch.stft(x_dev, n_fft, hop_len, n_fft, window=win_dev,
                          return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    s_cpu = s.squeeze(1).cpu()
    s_gpu = s_cpu.to(args.device)
    win_gpu = window_cpu.to(args.device)

    print(f"[check_stft] s shape={tuple(s_cpu.shape)} dtype={s_cpu.dtype}")

    # Run CPU stft N times — should be fully deterministic.
    cpu_real_first = cpu_imag_first = None
    cpu_var = 0.0
    for i in range(args.n):
        r, im = stft(s_cpu, window_cpu)
        if i == 0:
            cpu_real_first = r.clone()
            cpu_imag_first = im.clone()
        else:
            cpu_var = max(cpu_var,
                          (r - cpu_real_first).abs().max().item(),
                          (im - cpu_imag_first).abs().max().item())
    print(f"[check_stft] CPU stft self-determinism over {args.n} runs: {cpu_var:.3e}")

    # GPU stft N times.
    print(f"\n[check_stft] GPU stft trials:")
    print(f"{'trial':>5s}  {'real_max':>11s}  {'imag_max':>11s}  "
          f"{'rdiff_max':>11s}  {'idiff_max':>11s}  {'mean':>11s}")
    diffs = []
    for i in range(args.n):
        r_gpu, im_gpu = stft(s_gpu, win_gpu)
        r_gpu_cpu = r_gpu.cpu()
        im_gpu_cpu = im_gpu.cpu()
        rdiff = (r_gpu_cpu - cpu_real_first).abs()
        idiff = (im_gpu_cpu - cpu_imag_first).abs()
        d_max = max(rdiff.max().item(), idiff.max().item())
        d_mean = (rdiff.mean().item() + idiff.mean().item()) / 2
        print(f"{i:>5d}  {r_gpu_cpu.abs().max():.3e}  "
              f"{im_gpu_cpu.abs().max():.3e}  "
              f"{rdiff.max():.3e}  {idiff.max():.3e}  {d_mean:.3e}")
        diffs.append({"max": d_max, "mean": d_mean})

    means = [d["mean"] for d in diffs]
    maxes = [d["max"] for d in diffs]
    print(f"\n[check_stft] GPU stft over {args.n} trials:")
    print(f"  mean abs diff in [{min(means):.3e}, {max(means):.3e}]")
    print(f"  max  abs diff in [{min(maxes):.3e}, {max(maxes):.3e}]")

    out_path = bundle / "check_stft.json"
    out_path.write_text(json.dumps({
        "n": args.n,
        "device": args.device,
        "cpu_self_determinism_max": cpu_var,
        "trials": diffs,
    }, indent=2))
    print(f"[check_stft] wrote {out_path}")


if __name__ == "__main__":
    main()
