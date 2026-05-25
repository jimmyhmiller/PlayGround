"""Synthetic-mel capture runner.

Loads the real chatterbox S3Gen weights (mel2wav = HiFTGenerator), generates
random mels of the canonical shape (1, 80, T), runs inference under the
capture_hook patch, and lets the hook persist any bad-run bundles to
captures/run_<id>/.

Per PLAN_LAYER1.md Phase A "option 3" (synthetic mel + real weights). The
plan flags this may not trigger the bug if it's input-dependent. We run N
trials over a sweep of T values to give the bug as many chances as possible.

Usage:
    python3 scripts/synth_capture.py                  # defaults: T=998, N=10
    python3 scripts/synth_capture.py --t 998 --n 50
    python3 scripts/synth_capture.py --t-sweep 500,750,998,1200 --n 5
    python3 scripts/synth_capture.py --force          # save every run

Env knobs are forwarded to capture_hook:
    HIFIGAN_BISECT_OUT, HIFIGAN_BISECT_BAD_RATIO, HIFIGAN_BISECT_FORCE
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def _add_chatterbox_to_path() -> None:
    here = Path(__file__).resolve().parents[1]
    candidate = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    if not candidate.exists():
        sys.exit(f"chatterbox source not found at {candidate}")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _load_hifigan(device: str):
    """Load HiFTGenerator with default chatterbox weights via S3Gen.

    Returns (mel2wav_module, sample_rate)."""
    _add_chatterbox_to_path()
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from chatterbox.models.s3gen import S3Gen, S3GEN_SR  # type: ignore

    print("[synth] downloading s3gen.safetensors from ResembleAI/chatterbox ...", flush=True)
    ckpt_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="s3gen.safetensors")
    print(f"[synth] got {ckpt_path}")

    s3gen = S3Gen()
    state = load_file(ckpt_path)
    missing, unexpected = s3gen.load_state_dict(state, strict=False)
    # `S3Gen` declares some persistent-buffer ignores; missing keys for
    # those are expected.
    print(f"[synth] s3gen state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    s3gen = s3gen.to(device).eval()
    return s3gen.mel2wav, S3GEN_SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--t", type=int, default=998, help="mel time dimension")
    ap.add_argument("--t-sweep", type=str, default=None,
                    help="comma-separated list of T values to sweep")
    ap.add_argument("--n", type=int, default=10, help="trials per T")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="save every run regardless of ratio (sets HIFIGAN_BISECT_FORCE)")
    ap.add_argument("--mel-min", type=float, default=-11.616,
                    help="lower bound of synthetic mel values (matches BACKGROUND.md)")
    ap.add_argument("--mel-max", type=float, default=1.191,
                    help="upper bound of synthetic mel values (matches BACKGROUND.md)")
    ap.add_argument("--no-capture", action="store_true",
                    help="don't install the capture hook; just print metrics")
    args = ap.parse_args()

    if args.force:
        os.environ["HIFIGAN_BISECT_FORCE"] = "1"

    # Install the capture hook BEFORE we touch the module.
    if not args.no_capture:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import capture_hook  # type: ignore
        # We set the flag for clarity even though we call install() directly.
        os.environ.setdefault("HIFIGAN_BISECT_CAPTURE", "1")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[synth] WARNING: torch.cuda.is_available() is False — the hook does CPU comparison "
              "via deepcopy, but we expected a GPU here. Check your ROCm/PyTorch wheel.",
              file=sys.stderr)

    torch.manual_seed(args.seed)
    mel2wav, sr = _load_hifigan(args.device)
    print(f"[synth] HiFTGenerator on {args.device}, sr={sr}")

    if not args.no_capture:
        capture_hook.install()  # type: ignore[name-defined]

    ts = (
        [int(x) for x in args.t_sweep.split(",")] if args.t_sweep else [args.t]
    )
    cache_source = torch.zeros(1, 1, 0, device=args.device)

    n_total = 0
    for T in ts:
        for trial in range(args.n):
            n_total += 1
            mel = torch.empty(1, 80, T, device=args.device).uniform_(args.mel_min, args.mel_max)
            t0 = time.time()
            with torch.inference_mode():
                _wav, _s = mel2wav.inference(speech_feat=mel, cache_source=cache_source)
            dt = time.time() - t0
            print(f"[synth] T={T} trial={trial} elapsed={dt:.2f}s")

    print(f"[synth] done; {n_total} trials. Check captures/ for bundles.")


if __name__ == "__main__":
    main()
