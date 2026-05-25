"""Replay a single ResBlock or conv from a captured bundle.

Loads the bundle's `cpu_intermediates.pt` (or gpu_intermediates.pt) to recover
the actual input to a chosen submodule on the bad pass, reconstructs that
submodule from the state_dict, and runs it standalone on CPU vs GPU. Diffs
the result.

Reasoning:
- The post-hoc bisect showed a 470x mean_abs jump from resblocks.2 to
  resblocks.3 on a captured bad pass. That tells us divergence first
  amplifies catastrophically at resblocks.3, but doesn't tell us whether
  resblocks.3 itself is broken or whether it's just amplifying tiny upstream
  errors.
- This script tests that: feed the *clean CPU input* to resblocks.3 on
  GPU and CPU. If GPU diverges from CPU even with clean input, the bug
  is inside resblocks.3. If GPU matches CPU on clean input, the upstream
  amplification hypothesis is correct and we need to look earlier.

The bug is per-call nondeterministic, so we run N times to estimate
the divergence distribution rather than relying on a single trial.

Usage:
    python3 scripts/replay_layer.py captures/grind/run_<id>/ \
        --layer resblocks.3 --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _add_chatterbox_to_path() -> None:
    here = Path(__file__).resolve().parents[1]
    candidate = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _resolve_named_module(model, name: str):
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _previous_probe_name(layer: str) -> str | None:
    """Best-effort: return the probe whose output is approximately this
    layer's input in HiFTGenerator.decode forward order. For dilated
    resblocks within an upsample stage (resblocks.{3i,3i+1,3i+2}), the
    input to all three is the *same* tensor `x = ups[i](...) + si`, which
    is not recorded directly; the closest probe is the prior resblock,
    but the input is actually the prior `x` value. For our purposes
    (perturbing input by the GPU-CPU delta), the prior-resblock approximation
    is good enough since the HiFiGAN decode sums their outputs anyway.
    """
    # Map layer name -> previous probe whose output we can use as a proxy
    # for this layer's input. ResBlocks within an upsample stage all share
    # input; we use the prior resblock's output as a stand-in for "the
    # noise riding into this layer."
    if layer.startswith("resblocks."):
        idx = int(layer.split(".")[1])
        if idx == 0:
            return "ups.0"
        return f"resblocks.{idx - 1}"
    if layer.startswith("source_resblocks."):
        idx = int(layer.split(".")[1])
        return f"source_downs.{idx}"
    if layer == "source_downs.0":
        return None  # input is s_stft, not a probe
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--layer", required=True,
                    help="dotted path, e.g. resblocks.3")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-source", choices=["cpu", "gpu"], default="cpu",
                    help="which intermediates to use as input. Default cpu = "
                         "clean reference; tests whether the layer ITSELF is "
                         "broken given clean input.")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()

    _add_chatterbox_to_path()
    from chatterbox.models.s3gen.hifigan import HiFTGenerator  # type: ignore
    try:
        from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor  # type: ignore
        f0_predictor = ConvRNNF0Predictor()
    except Exception:
        f0_predictor = None

    init_kwargs = json.load((bundle / "hifigan_init_kwargs.json").open())
    state = torch.load(bundle / "hifigan_state_dict.pt", map_location="cpu",
                       weights_only=False)

    # Build CPU model, load weights, then deepcopy to GPU.
    model_cpu = HiFTGenerator(f0_predictor=f0_predictor, **init_kwargs).eval().cpu()
    missing, unexpected = model_cpu.load_state_dict(state, strict=False)
    print(f"[replay] state_dict missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)

    layer_cpu = _resolve_named_module(model_cpu, args.layer)
    print(f"[replay] layer {args.layer}: {type(layer_cpu).__name__}", file=sys.stderr)

    import copy
    model_gpu = copy.deepcopy(model_cpu).to(args.device)
    layer_gpu = _resolve_named_module(model_gpu, args.layer)

    # Find the saved input to this layer. The hooks recorded *outputs* of
    # each named submodule, but a layer's input is the prior layer's output
    # in the forward order. Just use the layer's own output as the GROUND-
    # TRUTH cpu reference, and reconstruct the input as the previous probe's
    # output.
    inters_path = bundle / f"{args.input_source}_intermediates.pt"
    if not inters_path.exists():
        sys.exit(f"missing {inters_path}; bundle was captured before per-layer "
                 f"intermediates were saved")
    inters = torch.load(inters_path, map_location="cpu", weights_only=False)

    # The layer's own output is its key in the dict.
    if args.layer not in inters:
        sys.exit(f"layer {args.layer} not in intermediates; available: "
                 f"{sorted(inters.keys())}")

    # Get the layer's input. Default: hook the layer in a fresh CPU run.
    # With --use-saved-input, take the saved bad-pass intermediates as input
    # (only works for layers whose input is the *output* of a prior probe in
    # the decode forward order — for resblocks.3 that's resblocks.2 + the
    # source-branch addition; for source_downs.0 it's s_stft which we don't
    # have). For non-ResBlock layers, this fallback to the CPU-fresh-hook is
    # often what you want anyway.
    mel = torch.load(bundle / "mel_in.pt", weights_only=False)
    s_cache = torch.load(bundle / "s_cache.pt", weights_only=False)

    layer_input_cpu = None
    if args.input_source == "gpu":
        # Hook into the GPU layer to capture its input by re-running... no,
        # the bug is per-call nondeterministic. Instead use the saved
        # gpu_intermediates: the input to layer N is approximately the output
        # of the prior layer in forward order. For resblocks.3, that's the
        # `x = x + si` step in decode — which is resblocks.2's output (last
        # of three resblocks for upsample i=1) PLUS source_resblocks.1's
        # output. We don't have that sum recorded directly.
        #
        # Simpler approach: rerun the FULL forward on a fresh CPU model with
        # a hook that captures the input to the target layer. CPU is
        # deterministic so this gives one valid "clean" input. Then
        # PERTURB it by adding (gpu_intermediates[prev] - cpu_intermediates[prev])
        # to mimic the real GPU input the bad pass saw.
        gi_path = bundle / "gpu_intermediates.pt"
        ci_path = bundle / "cpu_intermediates.pt"
        if not gi_path.exists() or not ci_path.exists():
            sys.exit("--input-source gpu needs gpu_intermediates.pt + cpu_intermediates.pt")
        gi = torch.load(gi_path, map_location="cpu", weights_only=False)
        ci = torch.load(ci_path, map_location="cpu", weights_only=False)
        prev_name = _previous_probe_name(args.layer)
        if prev_name is None or prev_name not in gi or prev_name not in ci:
            print(f"[replay] cannot determine previous probe for {args.layer} "
                  f"(tried {prev_name!r}); falling back to CPU-fresh-hook input",
                  file=sys.stderr)
        else:
            # Capture clean input via fresh CPU hook, then perturb.
            captured = {}
            def grab(_m, inp, _o):
                captured["x"] = inp[0].detach().clone()
            h = layer_cpu.register_forward_hook(grab)
            try:
                with torch.inference_mode():
                    model_cpu.inference(mel.cpu(), s_cache.cpu())
            finally:
                h.remove()
            base = captured["x"]
            # The relevant perturbation is "what GPU saw minus what CPU saw"
            # at the prior probe — but for ResBlocks the input is a sum of
            # the prior resblock's output and source_resblocks. The simplest
            # honest test: perturb base by gi[prev] - ci[prev] of MATCHING
            # shape; if shapes don't match, just use base.
            delta = gi[prev_name] - ci[prev_name]
            if delta.shape == base.shape:
                layer_input_cpu = base + delta
                print(f"[replay] using CPU-fresh-hook input + delta from "
                      f"{prev_name} (delta mean_abs={delta.abs().mean():.3e})",
                      file=sys.stderr)
            else:
                print(f"[replay] perturbation shape {tuple(delta.shape)} != "
                      f"input shape {tuple(base.shape)}; using clean CPU input",
                      file=sys.stderr)
                layer_input_cpu = base

    if layer_input_cpu is None:
        captured_input = {}
        def grab_input(_m, inp, _out):
            captured_input["x"] = inp[0].detach().clone()
        h = layer_cpu.register_forward_hook(grab_input)
        try:
            with torch.inference_mode():
                model_cpu.inference(mel.cpu(), s_cache.cpu())
        finally:
            h.remove()
        if "x" not in captured_input:
            sys.exit("hook never fired — wrong path?")
        layer_input_cpu = captured_input["x"]
    print(f"[replay] layer input shape={tuple(layer_input_cpu.shape)} "
          f"dtype={layer_input_cpu.dtype}", file=sys.stderr)

    # Run the layer N times on GPU, comparing to CPU ground truth (also run
    # N times to verify CPU is deterministic).
    print(f"\n[replay] running {args.n} trials of {args.layer} ...")
    print(f"{'trial':>5s}  {'cpu_max':>11s}  {'gpu_max':>11s}  {'diff_max':>11s}  "
          f"{'diff_mean':>11s}  {'diff_rel':>11s}")

    cpu_results = []
    gpu_results = []
    diffs = []

    layer_input_gpu = layer_input_cpu.to(args.device)

    with torch.inference_mode():
        for i in range(args.n):
            cpu_out = layer_cpu(layer_input_cpu)
            gpu_out = layer_gpu(layer_input_gpu)
            gpu_out_cpu = gpu_out.detach().to("cpu", torch.float32)
            d = (cpu_out - gpu_out_cpu).abs()
            rel = d / (cpu_out.abs() + 1e-8)
            print(f"{i:>5d}  {cpu_out.abs().max().item():.3e}  "
                  f"{gpu_out_cpu.abs().max().item():.3e}  "
                  f"{d.max().item():.3e}  {d.mean().item():.3e}  "
                  f"{rel.max().item():.3e}")
            cpu_results.append(cpu_out)
            gpu_results.append(gpu_out_cpu)
            diffs.append({"max": float(d.max()), "mean": float(d.mean()),
                          "rel_max": float(rel.max())})

    # Verify CPU determinism across trials.
    cpu_var = sum(((cpu_results[0] - r).abs().max().item()
                   for r in cpu_results[1:]), 0.0)
    print(f"\n[replay] CPU max-abs deviation across {args.n} trials: {cpu_var:.3e}",
          file=sys.stderr)

    # Summarize GPU divergence.
    means = [d["mean"] for d in diffs]
    maxes = [d["max"] for d in diffs]
    print(f"[replay] GPU diff over {args.n} trials: "
          f"mean_abs in [{min(means):.3e}, {max(means):.3e}], "
          f"max_abs in [{min(maxes):.3e}, {max(maxes):.3e}]")

    out_path = bundle / f"replay_{args.layer.replace('.', '_')}.json"
    out_path.write_text(json.dumps({
        "layer": args.layer,
        "n": args.n,
        "input_source": args.input_source,
        "device": args.device,
        "input_shape": list(layer_input_cpu.shape),
        "input_dtype": str(layer_input_cpu.dtype),
        "cpu_determinism_max_dev": cpu_var,
        "trials": diffs,
    }, indent=2))
    print(f"[replay] wrote {out_path}")


if __name__ == "__main__":
    main()
