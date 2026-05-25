"""Replay a captured bad-pass bundle with the FULL on-disk state restored:
MIOpen user DB + kernel cache from the moment of capture, plus the RNG
state from just before the bad call. Tries to reproduce the bad output
deterministically.

If this works (replay rms ~= captured gpu rms), we have a deterministic
repro and can move to layered analysis with confidence. If it doesn't,
we've shown the bug requires non-on-disk process state — pinning down
exactly what would be the next investigation.

Usage:
    python3 scripts/replay_with_state.py captures/grind/run_<id>/ --n 5

ENV must NOT have MIOPEN_USER_DB_PATH or MIOPEN_CUSTOM_CACHE_DIR pre-set
externally — this script controls them.
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, tempfile
from pathlib import Path

# Set MIOpen env BEFORE importing torch.
def _setup_miopen_from_snapshot(bundle: Path) -> tuple[str, str]:
    snap = bundle / "miopen_snapshot"
    if not snap.exists():
        sys.exit(f"bundle has no miopen_snapshot/ at {snap}")
    user_src = snap / "user_db"
    cache_src = snap / "cache"

    tmp_root = Path(tempfile.mkdtemp(prefix="miopen_replay_"))
    user_dst = tmp_root / "user"
    cache_dst = tmp_root / "cache"
    user_dst.mkdir()
    cache_dst.mkdir()
    if user_src.exists():
        for p in user_src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(user_src)
                tgt = user_dst / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tgt)
    if cache_src.exists():
        for p in cache_src.rglob("*"):
            if p.is_file():
                rel = p.relative_to(cache_src)
                tgt = cache_dst / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tgt)
    os.environ["MIOPEN_USER_DB_PATH"] = str(user_dst)
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(cache_dst)
    return str(user_dst), str(cache_dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-rng", action="store_true",
                    help="don't restore the captured RNG state")
    ap.add_argument("--no-miopen", action="store_true",
                    help="don't restore the MIOpen snapshot (control)")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()

    if not args.no_miopen:
        u, c = _setup_miopen_from_snapshot(bundle)
        print(f"[replay] MIOPEN_USER_DB_PATH={u}", file=sys.stderr)
        print(f"[replay] MIOPEN_CUSTOM_CACHE_DIR={c}", file=sys.stderr)
    else:
        print("[replay] MIOpen snapshot NOT restored (control)", file=sys.stderr)

    # Match capture-time HIP allocator config.
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    print(f"[replay] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",
          file=sys.stderr)

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
    state = torch.load(bundle / "hifigan_state_dict.pt", weights_only=False,
                       map_location="cpu")
    mel = torch.load(bundle / "mel_in.pt", weights_only=False).to(args.device)
    sc = torch.load(bundle / "s_cache.pt", weights_only=False).to(args.device)

    m = HiFTGenerator(f0_predictor=f0p, **init_kwargs).eval()
    m.load_state_dict(state, strict=False)
    m = m.to(args.device)

    # Read the captured "bad" rms from metadata so we can compare.
    meta = json.load((bundle / "metadata.json").open())
    captured_gpu_rms = meta["gpu_metrics"]["rms"]
    captured_cpu_rms = meta["cpu_metrics"]["rms"]
    print(f"[replay] captured: gpu_rms={captured_gpu_rms:.4f} cpu_rms={captured_cpu_rms:.4f}",
          file=sys.stderr)

    rng_state = None
    if not args.no_rng:
        rng_path = bundle / "rng_state.pt"
        if rng_path.exists():
            rng_state = torch.load(rng_path, weights_only=False)
            print("[replay] RNG state will be restored before each call",
                  file=sys.stderr)
        else:
            print("[replay] no rng_state.pt in bundle (older capture)",
                  file=sys.stderr)

    print(f"\n{'trial':>5s}  {'rms':>8s}  {'cent':>6s}  {'<300':>5s}  {'classify':>10s}")
    rmses = []
    bad_count = 0
    for i in range(args.n):
        if rng_state is not None:
            torch.set_rng_state(rng_state["cpu"])
            for d, st in enumerate(rng_state["cuda"]):
                if d < torch.cuda.device_count():
                    torch.cuda.set_rng_state(st, d)
        with torch.inference_mode():
            wav, _ = m.inference(mel, sc)
        arr = wav.detach().to("cpu", torch.float32).reshape(-1).numpy()
        rms = float((arr ** 2).mean() ** 0.5)
        rmses.append(rms)
        from scipy.signal import welch
        f, p = welch(arr, m.sampling_rate, nperseg=2048)
        tot = float(p.sum()) + 1e-12
        cent = float((f * p).sum() / tot)
        below = float(p[f < 300].sum() / tot)
        bad = (cent < 700 and below > 0.5) or rms < 0.04
        if bad: bad_count += 1
        print(f"{i:>5d}  {rms:.4f}  {cent:>5.0f}  {below:.2f}  "
              f"{'BAD' if bad else 'ok'}")

    rms_mean = sum(rmses) / len(rmses)
    print(f"\n[replay] {args.n} trials: bad={bad_count}/{args.n}, "
          f"rms mean={rms_mean:.4f} (captured bad was {captured_gpu_rms:.4f}, "
          f"clean was {captured_cpu_rms:.4f})")
    # Save outcome
    out_path = bundle / f"replay_with_state_{'with' if rng_state else 'no'}rng_"\
                       f"{'with' if not args.no_miopen else 'no'}miopen.json"
    out_path.write_text(json.dumps({
        "n": args.n, "device": args.device,
        "restored_rng": rng_state is not None,
        "restored_miopen": not args.no_miopen,
        "captured_gpu_rms": captured_gpu_rms,
        "captured_cpu_rms": captured_cpu_rms,
        "rmses": rmses,
        "bad_count": bad_count,
    }, indent=2))
    print(f"[replay] wrote {out_path}")


if __name__ == "__main__":
    main()
