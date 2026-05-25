"""Test if Winograd produces wrong output for the SAVED real-pass inputs.

Earlier `winograd_reproducer.py` ran random fp32 inputs through Winograd-using
shapes and saw fp32-noise diffs only — Winograd is numerically fine for random
inputs of those shapes. But during real chatterbox inference, Winograd fires
the bug ~10%. So the trigger involves the actual input value distribution.

This script loads gpu_intermediates.pt from a captured bad bundle and re-runs
each layer on GPU with default MIOpen (Winograd allowed) vs falls back to
golden CPU. Diffs compared against the same with Winograd disabled.

Run twice:
    python3 scripts/winograd_repro_real_inputs.py captures/grind/run_<id>/ \\
        --out /tmp/wino_real_on.pt
    MIOPEN_DEBUG_CONV_WINOGRAD=0 python3 scripts/winograd_repro_real_inputs.py \\
        captures/grind/run_<id>/ --out /tmp/wino_real_off.pt

Then compare. If on-GPU output differs from CPU when Winograd is enabled, but
matches CPU when disabled, we've localized the bug to a specific Winograd
kernel + the specific input it ran on.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch


def _add_chatterbox_to_path():
    here = Path(__file__).resolve().parents[1]
    cb_src = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    if cb_src.exists() and str(cb_src) not in sys.path:
        sys.path.insert(0, str(cb_src))


def _resolve(model, name):
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    _add_chatterbox_to_path()
    from chatterbox.models.s3gen.hifigan import HiFTGenerator  # type: ignore
    try:
        from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor  # type: ignore
        f0p = ConvRNNF0Predictor()
    except Exception:
        f0p = None

    init_kwargs = json.load((bundle / "hifigan_init_kwargs.json").open())
    state = torch.load(bundle / "hifigan_state_dict.pt", map_location="cpu",
                       weights_only=False)

    # We need the input to each ResBlock. The cpu_intermediates.pt has the
    # outputs at each named submodule (CPU). For ResBlocks.{i*num_kernels+j},
    # the input is approx ups[i]'s output added with si — for our purposes,
    # the prior layer's output is close enough as a "real" input.
    cpu_int_path = bundle / "cpu_intermediates.pt"
    if not cpu_int_path.exists():
        sys.exit(f"missing {cpu_int_path}")
    cpu_int = torch.load(cpu_int_path, map_location="cpu", weights_only=False)

    # Build CPU + GPU model.
    model_cpu = HiFTGenerator(f0_predictor=f0p, **init_kwargs).eval().cpu()
    model_cpu.load_state_dict(state, strict=False)
    import copy
    model_gpu = copy.deepcopy(model_cpu).to(args.device)

    # For each ResBlock 0..8, take the appropriate prior probe as input proxy.
    # ResBlocks 0,1,2 sit after ups.0 (T=8016). Use ups.0 output as input.
    # ResBlocks 3,4,5 sit after ups.1 (T=40080). Use ups.1 output.
    # ResBlocks 6,7,8 sit after ups.2 (T=120240). Use ups.2 output.
    layers = []
    for i in range(9):
        rb_name = f"resblocks.{i}"
        if i < 3:
            input_probe = "ups.0"
        elif i < 6:
            input_probe = "ups.1"
        else:
            input_probe = "ups.2"
        if input_probe in cpu_int:
            layers.append((rb_name, input_probe))

    print(f"[wino_real] {len(layers)} (layer, input_probe) pairs", file=sys.stderr)

    results = []
    for rb_name, ipname in layers:
        x_cpu = cpu_int[ipname]
        x_gpu = x_cpu.to(args.device)
        layer_cpu = _resolve(model_cpu, rb_name)
        layer_gpu = _resolve(model_gpu, rb_name)

        with torch.inference_mode():
            ref_cpu = layer_cpu(x_cpu)

        # GPU N times.
        gpu_outs = []
        for _ in range(args.n):
            with torch.inference_mode():
                go = layer_gpu(x_gpu)
            gpu_outs.append(go.detach().to("cpu", torch.float32))

        diffs = []
        for go in gpu_outs:
            d = (go - ref_cpu).abs()
            diffs.append({"max_abs": float(d.max()), "mean_abs": float(d.mean())})

        worst = max(d["max_abs"] for d in diffs)
        worst_mean = max(d["mean_abs"] for d in diffs)
        intra = (gpu_outs[0] - gpu_outs[-1]).abs().max().item() if len(gpu_outs) > 1 else 0.0
        rec = {
            "layer": rb_name,
            "input_probe": ipname,
            "input_shape": list(x_cpu.shape),
            "input_abs_max": float(x_cpu.abs().max()),
            "cpu_out_abs_max": float(ref_cpu.abs().max()),
            "diffs_per_trial": diffs,
            "intra_gpu_max": intra,
        }
        results.append(rec)
        print(f"[wino_real] {rb_name} (input shape {tuple(x_cpu.shape)}, "
              f"input_max {x_cpu.abs().max():.2f}): "
              f"diff max={worst:.3e} mean={worst_mean:.3e} intra_gpu={intra:.3e}",
              file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.out)
    print(f"[wino_real] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
