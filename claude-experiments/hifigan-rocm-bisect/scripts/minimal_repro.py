"""Phase C: collapse the bisect result to a single functional call.

Two subcommands:

  extract   captures/run_<id>/  --layer ups.0
            Reads the bundle, runs CPU inference up to the named layer,
            captures (input, weight, bias, stride, padding, dilation,
            groups, op_name) for that layer, and writes
            captures/run_<id>/minimal_repro.pt

  run       captures/run_<id>/minimal_repro.pt  [--device cuda]
            Loads the saved tensors and calls
            torch.nn.functional.conv1d / conv_transpose1d on CPU and on
            the chosen device, prints a diff. Reproduces (or falsifies)
            the layer-level divergence at the single-op level.

The minimal_repro.pt file is the artifact every deeper layer (2-6) consumes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ------------------ extract path setup ------------------------------


def _add_chatterbox_to_path() -> None:
    here = Path(__file__).resolve().parents[1]
    candidate = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


# --------------------------- extract -------------------------------


def _resolve_named_module(model, name: str):
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _conv_layer_descriptor(mod: torch.nn.Module) -> dict:
    """Return the tensors/attrs needed to call the right F.conv* function."""
    cls = type(mod).__name__
    weight = mod.weight.detach().clone()
    bias = mod.bias.detach().clone() if mod.bias is not None else None
    desc = {
        "op_name": (
            "conv_transpose1d" if "ConvTranspose" in cls else "conv1d"
            if "Conv1d" in cls else cls.lower()
        ),
        "module_class": cls,
        "weight": weight,
        "bias": bias,
        "stride": tuple(mod.stride),
        "padding": tuple(mod.padding),
        "dilation": tuple(mod.dilation),
        "groups": int(mod.groups),
    }
    if "ConvTranspose" in cls:
        desc["output_padding"] = tuple(mod.output_padding)
    return desc


def cmd_extract(args):
    bundle = Path(args.bundle).resolve()
    layer_name = args.layer

    _add_chatterbox_to_path()
    from chatterbox.models.s3gen.hifigan import HiFTGenerator  # type: ignore

    state = torch.load(bundle / "hifigan_state_dict.pt", map_location="cpu", weights_only=False)
    init_kwargs = json.load((bundle / "hifigan_init_kwargs.json").open())
    mel_in = torch.load(bundle / "mel_in.pt", map_location="cpu", weights_only=False)
    s_cache = torch.load(bundle / "s_cache.pt", map_location="cpu", weights_only=False)

    # Build CPU model
    try:
        from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor  # type: ignore
        f0_predictor = ConvRNNF0Predictor()
    except Exception:
        f0_predictor = None
    model = HiFTGenerator(f0_predictor=f0_predictor, **init_kwargs).eval().cpu()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[extract] state_dict missing={len(missing)} unexpected={len(unexpected)}",
              file=sys.stderr)

    target = _resolve_named_module(model, layer_name)
    if not isinstance(target, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        sys.exit(
            f"--layer must point at a Conv1d / ConvTranspose1d (got {type(target).__name__}). "
            "If the broken layer is a ResBlock or container, pick a specific conv inside it."
        )

    captured: dict = {}

    def hook(_m, inp, _out):
        # inp is a tuple; first element is the input tensor.
        captured["input"] = inp[0].detach().clone()

    h = target.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.inference(mel_in.cpu(), s_cache.cpu())
    finally:
        h.remove()

    if "input" not in captured:
        sys.exit(f"hook on {layer_name} never fired — wrong path?")

    desc = _conv_layer_descriptor(target)
    desc["input"] = captured["input"]
    desc["layer_name"] = layer_name

    out = bundle / "minimal_repro.pt"
    torch.save(desc, out)
    print(
        f"[extract] {layer_name} ({desc['module_class']}): "
        f"input={tuple(desc['input'].shape)} weight={tuple(desc['weight'].shape)} "
        f"stride={desc['stride']} padding={desc['padding']} "
        f"dilation={desc['dilation']} groups={desc['groups']}"
    )
    print(f"[extract] wrote {out}")


# ----------------------------- run ---------------------------------


def _call_op(desc: dict, device: str) -> torch.Tensor:
    inp = desc["input"].to(device)
    w = desc["weight"].to(device)
    b = desc["bias"].to(device) if desc["bias"] is not None else None

    if desc["op_name"] == "conv1d":
        return F.conv1d(
            inp, w, bias=b,
            stride=desc["stride"], padding=desc["padding"],
            dilation=desc["dilation"], groups=desc["groups"],
        )
    if desc["op_name"] == "conv_transpose1d":
        return F.conv_transpose1d(
            inp, w, bias=b,
            stride=desc["stride"], padding=desc["padding"],
            output_padding=desc.get("output_padding", (0,)),
            groups=desc["groups"], dilation=desc["dilation"],
        )
    sys.exit(f"unsupported op_name: {desc['op_name']}")


def cmd_run(args):
    desc = torch.load(args.minimal_repro, map_location="cpu", weights_only=False)
    print(
        f"[run] {desc.get('layer_name', '?')} ({desc['module_class']}): "
        f"input={tuple(desc['input'].shape)} weight={tuple(desc['weight'].shape)} "
        f"stride={desc['stride']} padding={desc['padding']} "
        f"dilation={desc['dilation']} groups={desc['groups']}"
    )

    cpu_out = _call_op(desc, "cpu")
    gpu_out = _call_op(desc, args.device)

    diff = (cpu_out - gpu_out.cpu()).abs()
    rel = diff / (cpu_out.abs() + 1e-8)
    print(
        f"[run] cpu: shape={tuple(cpu_out.shape)} "
        f"min={cpu_out.min().item():.4e} max={cpu_out.max().item():.4e} "
        f"abs_mean={cpu_out.abs().mean().item():.4e}"
    )
    print(
        f"[run] gpu: shape={tuple(gpu_out.shape)} "
        f"min={gpu_out.min().item():.4e} max={gpu_out.max().item():.4e} "
        f"abs_mean={gpu_out.abs().mean().item():.4e}"
    )
    print(
        f"[run] diff: max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} "
        f"max_rel={rel.max().item():.3e}"
    )

    # Repeat with synthetic random input of the same shape — does the bug
    # depend on input values, or only on shape?
    rand_input = torch.randn_like(desc["input"])
    rand_desc = dict(desc)
    rand_desc["input"] = rand_input
    cpu_r = _call_op(rand_desc, "cpu")
    gpu_r = _call_op(rand_desc, args.device)
    diff_r = (cpu_r - gpu_r.cpu()).abs()
    rel_r = diff_r / (cpu_r.abs() + 1e-8)
    print(
        f"[run] random-input diff: max_abs={diff_r.max().item():.3e} "
        f"mean_abs={diff_r.mean().item():.3e} "
        f"max_rel={rel_r.max().item():.3e}"
    )


# -------------------------- entry ----------------------------------


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract a single conv layer from a bundle")
    e.add_argument("bundle")
    e.add_argument("--layer", required=True,
                   help="dotted module path, e.g. ups.0, conv_pre, "
                        "resblocks.0.convs1.0, source_downs.1")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("run", help="run the saved single-op reproducer")
    r.add_argument("minimal_repro")
    r.add_argument("--device", default="cuda")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
