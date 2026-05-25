"""Full-pipeline replay test.

Loads ChatterboxTTS exactly like real_capture.py (so the process has T3 +
s3gen + HiFiGAN all instantiated and the same MIOpen pre-warmup as the
grind), then injects a saved mel + s_cache from a known-bad capture
DIRECTLY into mel2wav.inference(). Skips T3+s3gen execution entirely.

If anomaly rate in this setup matches the live full-pipeline rate
(~10-30%), then T3+s3gen *execution* during the call is NOT necessary —
just having the chatterbox process loaded (instantiating those modules)
is sufficient to prime the bug.

If anomaly rate is 0%, then T3+s3gen actually running during the call
(as part of model.generate()) is necessary.

Usage:
    python3 scripts/full_pipeline_replay.py captures/grind/run_<id>/ --n 20
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["mel-only", "full-generate"],
                    default="mel-only",
                    help="mel-only: feed saved mel through model.s3gen.mel2wav.inference. "
                         "full-generate: run full model.generate() on chunk19.")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()

    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    here = Path(__file__).resolve().parents[1]
    cb_src = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    sys.path.insert(0, str(cb_src))

    import torch
    print(f"[full_replay] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",
          file=sys.stderr)

    print("[full_replay] loading ChatterboxTTS (full pipeline)...", file=sys.stderr)
    from chatterbox.tts import ChatterboxTTS  # type: ignore
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=args.device)
    print(f"[full_replay] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    mel = torch.load(bundle / "mel_in.pt", weights_only=False).to(args.device)
    sc = torch.load(bundle / "s_cache.pt", weights_only=False).to(args.device)

    meta = json.load((bundle / "metadata.json").open())
    captured_gpu_rms = meta["gpu_metrics"]["rms"]
    captured_cpu_rms = meta["cpu_metrics"]["rms"]
    print(f"[full_replay] captured: gpu_rms={captured_gpu_rms:.4f} "
          f"cpu_rms={captured_cpu_rms:.4f}", file=sys.stderr)

    from scipy.signal import welch
    sr = int(model.sr)

    print(f"\n[full_replay] mode={args.mode} n={args.n}")
    print(f"{'trial':>5s}  {'rms':>8s}  {'cent':>6s}  {'<300':>5s}  {'classify':>10s}")
    bad = 0
    rmses = []
    for i in range(args.n):
        if args.mode == "mel-only":
            with torch.inference_mode():
                w, _ = model.s3gen.mel2wav.inference(speech_feat=mel, cache_source=sc)
        else:
            # full-generate: synthesize chunk19 fresh
            text = (
                "The objection goes as follows: according to Christian belief, we human "
                "beings have been created by an all-powerful, all-knowing God who loves us "
                "enough to send his son, the second person of the divine Trinity, to "
                "suffer and die on our account; but given the devastating amount and "
                "variety of human suffering and evil in our sad world, this simply can't "
                "be true."
            )
            voice_ref = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
            w = model.generate(text, audio_prompt_path=voice_ref,
                               exaggeration=0.5, cfg_weight=0.5)
        arr = w.detach().to("cpu", torch.float32).reshape(-1).numpy()
        rms = float((arr ** 2).mean() ** 0.5)
        f, p = welch(arr, sr, nperseg=2048)
        tot = float(p.sum()) + 1e-12
        cent = float((f * p).sum() / tot)
        below = float(p[f < 300].sum() / tot)
        is_bad = (cent < 700 and below > 0.5) or rms < 0.04
        if is_bad: bad += 1
        rmses.append(rms)
        print(f"{i:>5d}  {rms:.4f}  {cent:>5.0f}  {below:.2f}  "
              f"{'BAD' if is_bad else 'ok'}")

    print(f"\n[full_replay] bad={bad}/{args.n}, rms range "
          f"[{min(rmses):.4f}, {max(rmses):.4f}]")

    out = bundle / f"full_pipeline_replay_{args.mode}.json"
    out.write_text(json.dumps({
        "mode": args.mode,
        "n": args.n,
        "bad": bad,
        "rmses": rmses,
        "captured_gpu_rms": captured_gpu_rms,
        "captured_cpu_rms": captured_cpu_rms,
    }, indent=2))
    print(f"[full_replay] wrote {out}")


if __name__ == "__main__":
    main()
