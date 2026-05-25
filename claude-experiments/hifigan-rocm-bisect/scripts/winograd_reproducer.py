"""Small reproducer for the Winograd numerical bug on AMD gfx1151.

Runs a sweep of conv shapes that the MIOpen trace showed using
ConvBinWinogradRxSf3x2 / ConvBinWinogradRxSf2x3g1 solvers during
chatterbox s3gen+HiFiGAN inference. For each shape:
- with default MIOpen (Winograd enabled): record GPU output.
- compare to CPU output (deterministic ground truth).

Designed to be invoked twice in two processes:
    # 1. baseline (Winograd allowed):
    python3 scripts/winograd_reproducer.py --out /tmp/winograd_on.pt

    # 2. control (Winograd disabled):
    MIOPEN_DEBUG_CONV_WINOGRAD=0 python3 scripts/winograd_reproducer.py \\
        --out /tmp/winograd_off.pt

Then a third script diffs the two output sets to identify shapes where
Winograd produces materially different output from the non-Winograd
fallback (which we trust as the reference, since GemmFwd* solvers don't
have known gfx1151 bugs).

The shapes below come straight from the trace
miopen_trace_logs/trace_clean_phase1_trial06_tokens.log lines like:
  "Returning an invoker for problem 512x1x761x1x4x512x1x758x1xNCHWxFP32x0x0x1x1x1x1x1xFxDefault and solver ConvBinWinogradRxSf3x2"

Format of the problem string:
  C_in x H_in x W_in x N=1 x K_h*K_w x C_out x H_out x W_out x N=1 x layout x dtype x pad_h x pad_w x stride_h x stride_w x dilation_h x dilation_w x groups x direction x mode

We take the W (time) axis as the only spatial axis, treat as Conv1d.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F


# Shapes pulled from the trace. (in_ch, out_ch, kernel, time_in, dilation, padding)
# Picked the ones that showed up using Winograd. From trace_clean for trial06.
SHAPES = [
    # First-row from trace: 512x1x761x1x4x512x1x758x1 pad=0,stride=1,dil=1,grp=1
    # Conv1d(512->512, k=4, padding=0, stride=1, dilation=1) on T=761 -> T=758
    (512, 512, 4, 761, 1, 0, 1),
    # 512x1x1520x1x5x512x1x1516x1 pad=0,stride=1,dil=1
    (512, 512, 5, 1520, 1, 0, 1),
    # 512x1x760x1x3x512x1x758x1
    (512, 512, 3, 760, 1, 0, 1),
    # 512x1x1016x1x3x512x1x1016x1 pad=1,stride=1,dil=1
    (512, 512, 3, 1016, 1, 1, 1),
    # 256x1x8128x1x3x256x1x8128x1 pad=1
    (256, 256, 3, 8128, 1, 1, 1),
    # 256x1x8128x1x7x256x1x8128x1 pad=3
    (256, 256, 7, 8128, 1, 3, 1),
    # 256x1x8128x1x11x256x1x8128x1 pad=5
    (256, 256, 11, 8128, 1, 5, 1),
    # 128x1x40640x1x3x128x1x40640x1 pad=1
    (128, 128, 3, 40640, 1, 1, 1),
    # 128x1x40640x1x7x128x1x40640x1 pad=3
    (128, 128, 7, 40640, 1, 3, 1),
    # 128x1x40640x1x11x128x1x40640x1 pad=5
    (128, 128, 11, 40640, 1, 5, 1),
    # HiFiGAN ResBlock dilated Conv1d's — these were the actual hot path.
    # ResBlock(channels, kernel, dilations=[1,3,5]), kernel∈{3,7,11}
    # For dilated conv: padding = (k-1)*dilation/2 to keep T_out=T_in.
    # 256ch ResBlocks at T=8128 (post first upsample, before second)
    (256, 256, 3, 8128, 3, 3, 1),    # k=3 dil=3 pad=3
    (256, 256, 3, 8128, 5, 5, 1),    # k=3 dil=5 pad=5
    (256, 256, 7, 8128, 3, 9, 1),    # k=7 dil=3 pad=9
    (256, 256, 7, 8128, 5, 15, 1),   # k=7 dil=5 pad=15
    (256, 256, 11, 8128, 3, 15, 1),  # k=11 dil=3 pad=15
    (256, 256, 11, 8128, 5, 25, 1),  # k=11 dil=5 pad=25
    # 128ch ResBlocks at T=40640
    (128, 128, 3, 40640, 3, 3, 1),
    (128, 128, 3, 40640, 5, 5, 1),
    (128, 128, 7, 40640, 3, 9, 1),
    (128, 128, 7, 40640, 5, 15, 1),
    (128, 128, 11, 40640, 3, 15, 1),
    (128, 128, 11, 40640, 5, 25, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=3, help="trials per shape")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output .pt path")
    args = ap.parse_args()

    print(f"[repro] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",
          file=sys.stderr)
    print(f"[repro] device={args.device}, {len(SHAPES)} shapes × {args.n} trials",
          file=sys.stderr)

    # Pick a single deterministic input + weight per shape so two runs are comparable.
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    results = []
    for shape_idx, (C_in, C_out, K, T_in, dil, pad, stride) in enumerate(SHAPES):
        T_out = (T_in + 2 * pad - dil * (K - 1) - 1) // stride + 1
        x = torch.randn(1, C_in, T_in, generator=rng, dtype=torch.float32)
        w = torch.randn(C_out, C_in, K, generator=rng, dtype=torch.float32) * 0.02

        # CPU reference (deterministic).
        with torch.inference_mode():
            cpu_out = F.conv1d(x, w, bias=None, stride=stride,
                               padding=pad, dilation=dil)

        # GPU N times.
        x_gpu = x.to(args.device)
        w_gpu = w.to(args.device)
        gpu_outs = []
        for trial in range(args.n):
            with torch.inference_mode():
                go = F.conv1d(x_gpu, w_gpu, bias=None, stride=stride,
                              padding=pad, dilation=dil)
            gpu_outs.append(go.detach().to("cpu", torch.float32))

        # Diff between GPU and CPU.
        diffs = []
        for go in gpu_outs:
            d = (go - cpu_out).abs()
            diffs.append({
                "max_abs": float(d.max()),
                "mean_abs": float(d.mean()),
            })
        # Diff between GPU trials (intra-GPU determinism).
        intra = (gpu_outs[0] - gpu_outs[-1]).abs() if len(gpu_outs) > 1 else None
        intra_max = float(intra.max()) if intra is not None else 0.0

        rec = {
            "shape": (C_in, C_out, K, T_in, dil, pad, stride),
            "T_out": T_out,
            "input_abs_max": float(x.abs().max()),
            "weight_abs_max": float(w.abs().max()),
            "cpu_out_abs_max": float(cpu_out.abs().max()),
            "gpu_out_abs_max": float(gpu_outs[0].abs().max()),
            "diffs_per_trial": diffs,
            "intra_gpu_max": intra_max,
        }
        results.append(rec)

        # Print one-liner.
        worst_max = max(d["max_abs"] for d in diffs)
        worst_mean = max(d["mean_abs"] for d in diffs)
        print(f"[repro] shape{shape_idx} {rec['shape']} -> "
              f"diff max={worst_max:.3e} mean={worst_mean:.3e} "
              f"cpu_max={rec['cpu_out_abs_max']:.3e} "
              f"intra_gpu={intra_max:.3e}",
              file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.out)
    print(f"[repro] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
