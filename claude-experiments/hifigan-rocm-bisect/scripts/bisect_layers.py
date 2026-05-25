"""Phase B offline layer-by-layer bisect.

Loads a captured bundle (from Phase A) and runs HiFTGenerator.inference on
both CPU and GPU with forward hooks on every relevant submodule. Diffs each
probe pair and prints a table. The first row whose abs-max jumps several
orders of magnitude above the prior rows is the broken layer.

Usage:
    python scripts/bisect_layers.py captures/run_<id>/
    python scripts/bisect_layers.py captures/run_<id>/ --device cuda
    python scripts/bisect_layers.py captures/run_<id>/ --no-hooks  # final-only

Env knobs that may influence MIOpen behavior (set externally):
    MIOPEN_USER_DB_PATH=/tmp/empty_<id>     # force clean solver search
    MIOPEN_DEBUG_CONV_WINOGRAD=0            # disable winograd family
    MIOPEN_DEBUG_CONV_DIRECT=0              # etc.
    MIOPEN_LOG_LEVEL=6 MIOPEN_ENABLE_LOGGING=1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


def _add_chatterbox_to_path() -> None:
    """Add the editable chatterbox checkout to sys.path so we import the
    same HiFTGenerator definition the capture used. We resolve relative to
    THIS project directory so we don't rely on cwd."""
    here = Path(__file__).resolve().parents[1]
    candidate = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    if not candidate.exists():
        # Try a couple of fallbacks
        for up in (here.parent.parent / "chatterbox-rewrite" / "chatterbox" / "src",):
            if up.exists():
                candidate = up
                break
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _load_bundle(bundle: Path) -> dict:
    return {
        "mel_in": torch.load(bundle / "mel_in.pt", map_location="cpu", weights_only=False),
        "s_cache": torch.load(bundle / "s_cache.pt", map_location="cpu", weights_only=False),
        "state_dict": torch.load(
            bundle / "hifigan_state_dict.pt", map_location="cpu", weights_only=False
        ),
        "init_kwargs": json.load((bundle / "hifigan_init_kwargs.json").open()),
        "metadata": json.load((bundle / "metadata.json").open())
        if (bundle / "metadata.json").exists()
        else {},
    }


def _build_module(init_kwargs: dict, state_dict: dict):
    _add_chatterbox_to_path()
    from chatterbox.models.s3gen.hifigan import HiFTGenerator  # type: ignore

    # f0_predictor is required at construction time. We don't have it as a
    # standalone artifact, but the state_dict contains its weights. We
    # construct the same predictor module the chatterbox stack uses; if that
    # import fails the user can override via --f0-predictor.
    f0_predictor = _load_default_f0_predictor(state_dict)

    model = HiFTGenerator(f0_predictor=f0_predictor, **init_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[bisect] state_dict mismatch — missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
        if missing:
            print("  missing[:5]:", missing[:5], file=sys.stderr)
        if unexpected:
            print("  unexpected[:5]:", unexpected[:5], file=sys.stderr)
    model.eval()
    return model


def _load_default_f0_predictor(state_dict: dict):
    """Find and instantiate the f0_predictor used by chatterbox's HiFTGenerator.

    The state_dict has keys prefixed `f0_predictor.*` — we mirror those.
    chatterbox bundles a ConvRNNF0Predictor (path varies by version), so we
    try the canonical import first.
    """
    try:
        from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor  # type: ignore
        return ConvRNNF0Predictor()
    except Exception:
        pass
    # Fallback: build a minimal stub that has the right param shapes.
    # Without the real class we can't reproduce f0_predictor numerics; but
    # the bug we're hunting lives in HiFiGAN convs/upsamples, which run on
    # `speech_feat` directly via decode(). The predictor's output `f0` only
    # feeds the source branch; if that branch matches between CPU and GPU
    # we're fine, and if it doesn't we'll see it at source_downs. Print a
    # warning so the user knows.
    print(
        "[bisect] WARNING: could not import ConvRNNF0Predictor; using identity stub. "
        "Source branch will be incorrect. Bisect of main branch (conv_pre, ups, "
        "resblocks, conv_post) is still valid.",
        file=sys.stderr,
    )

    class _IdentityF0(torch.nn.Module):
        def forward(self, x):
            # speech_feat shape (B, mel, T); f0 should be (B, T_frames)
            return torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)

    return _IdentityF0()


# --------------------------- probes --------------------------------


def _names_to_probe(model) -> list[str]:
    names = ["conv_pre"]
    for i in range(len(model.ups)):
        names.append(f"ups.{i}")
    names.append("reflection_pad")
    for i in range(len(model.source_downs)):
        names.append(f"source_downs.{i}")
        names.append(f"source_resblocks.{i}")
    for i in range(len(model.resblocks)):
        names.append(f"resblocks.{i}")
    names.append("conv_post")
    return names


def _register_hooks(model, names: list[str], buf: list):
    handles = []
    name_to_mod = dict(model.named_modules())
    for name in names:
        if name not in name_to_mod:
            print(f"[bisect] WARNING: no submodule named {name!r}; skipping", file=sys.stderr)
            continue
        mod = name_to_mod[name]

        def make_hook(n):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                buf.append((n, t.detach().to(torch.float32).cpu().clone()))
            return fn

        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles


def _patch_istft_probe(model, buf: list, tag: str):
    """_istft isn't a Module, so attach a wrapper that records (mag, phase, out)."""
    original = model._istft

    def wrapped(magnitude, phase):
        buf.append((f"{tag}._istft.in.magnitude", magnitude.detach().to(torch.float32).cpu().clone()))
        buf.append((f"{tag}._istft.in.phase", phase.detach().to(torch.float32).cpu().clone()))
        out = original(magnitude, phase)
        buf.append((f"{tag}._istft.out", out.detach().to(torch.float32).cpu().clone()))
        return out

    model._istft = wrapped  # bind on the instance
    return original


# --------------------------- run -----------------------------------


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_mismatch": True, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    d = (a - b).abs()
    rel = d / (a.abs() + 1e-8)
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "max_rel": float(rel.max().item()),
        "shape": list(a.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--device", default="cuda", help="GPU device for the second run")
    ap.add_argument("--no-hooks", action="store_true",
                    help="skip per-layer hooks; only diff the final waveform")
    args = ap.parse_args()

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        sys.exit(f"not a directory: {bundle}")

    b = _load_bundle(bundle)
    mel_in = b["mel_in"]
    s_cache = b["s_cache"]

    print(f"[bisect] mel_in shape={tuple(mel_in.shape)} dtype={mel_in.dtype}")
    print(f"[bisect] s_cache shape={tuple(s_cache.shape)}")
    print(f"[bisect] init_kwargs: {b['init_kwargs']}")

    cpu_model = _build_module(b["init_kwargs"], b["state_dict"]).cpu()
    gpu_model = copy.deepcopy(cpu_model).to(args.device)

    cpu_buf, gpu_buf = [], []
    if not args.no_hooks:
        names = _names_to_probe(cpu_model)
        _register_hooks(cpu_model, names, cpu_buf)
        _register_hooks(gpu_model, names, gpu_buf)
        _patch_istft_probe(cpu_model, cpu_buf, "cpu")
        _patch_istft_probe(gpu_model, gpu_buf, "gpu")

    with torch.inference_mode():
        cpu_out, _ = cpu_model.inference(mel_in.cpu(), s_cache.cpu())
        gpu_out, _ = gpu_model.inference(mel_in.to(args.device), s_cache.to(args.device))

    print()
    print(f"{'probe':40s}  {'shape':>20s}  {'max_abs':>11s}  {'mean_abs':>11s}  {'max_rel':>11s}")
    print("-" * 110)

    if not args.no_hooks:
        # Strip the cpu/gpu tag from istft probes so we can pair them.
        def _strip(name: str) -> str:
            return name.replace("cpu.", "").replace("gpu.", "")

        cpu_dict: dict[str, torch.Tensor] = {}
        for n, t in cpu_buf:
            cpu_dict[_strip(n)] = t
        for n, t_gpu in gpu_buf:
            key = _strip(n)
            if key not in cpu_dict:
                continue
            t_cpu = cpu_dict[key]
            d = _diff(t_cpu, t_gpu)
            if "shape_mismatch" in d:
                print(f"{key:40s}  SHAPE MISMATCH cpu={d['shape_a']} gpu={d['shape_b']}")
            else:
                print(
                    f"{key:40s}  {str(d['shape']):>20s}  "
                    f"{d['max_abs']:.3e}  {d['mean_abs']:.3e}  {d['max_rel']:.3e}"
                )

    final = _diff(cpu_out.cpu(), gpu_out.cpu())
    if "shape_mismatch" in final:
        print(f"{'FINAL':40s}  SHAPE MISMATCH cpu={final['shape_a']} gpu={final['shape_b']}")
    else:
        print(
            f"{'FINAL':40s}  {str(final['shape']):>20s}  "
            f"{final['max_abs']:.3e}  {final['mean_abs']:.3e}  {final['max_rel']:.3e}"
        )

    # Persist the diff table next to the bundle
    out_path = bundle / "bisect_diffs.json"
    serial = {
        "device": args.device,
        "no_hooks": args.no_hooks,
        "final": final,
    }
    if not args.no_hooks:
        per_probe = {}
        for n, t in cpu_buf:
            key = n.replace("cpu.", "").replace("gpu.", "")
            if key in per_probe:
                continue
            t_gpu = next((tt for nn, tt in gpu_buf if nn.replace("gpu.", "").replace("cpu.", "") == key), None)
            if t_gpu is None:
                continue
            per_probe[key] = _diff(t, t_gpu)
        serial["per_probe"] = per_probe
    out_path.write_text(json.dumps(serial, indent=2))
    print(f"\n[bisect] wrote {out_path}")


if __name__ == "__main__":
    main()
