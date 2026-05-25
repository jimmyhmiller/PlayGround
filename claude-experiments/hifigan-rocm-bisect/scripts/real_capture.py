"""Real-state chatterbox capture driver.

Runs ChatterboxTTS.from_pretrained on GPU with the capture_hook installed,
then synthesizes a known-bad text chunk N times. Each generate() call routes
through HiFTGenerator.inference, which the hook compares CPU vs GPU and
persists a bundle to captures/run_<id>/ when GPU output is muffled.

Per PLAN_LAYER1.md Phase A "option 1": run a short separate chatterbox
session with HiFiGAN-on-GPU enabled on text known to trigger muffled output.

This script does NOT touch paper-audiobooks. It just imports chatterbox
directly with weights downloaded from HuggingFace (or HF cache).

Usage:
    python3 scripts/real_capture.py                       # 5 trials, existing MIOpen cache
    python3 scripts/real_capture.py --n 10
    python3 scripts/real_capture.py --clean-cache         # fresh MIOPEN_USER_DB_PATH
    python3 scripts/real_capture.py --force               # save every run, not just bad ones
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path


# The exact 361-char chunk from paper-audiobooks/scripts/repro_chunk19.py.
# Plan calls out this text by name as a known-bad input.
CHUNK19 = (
    "The objection goes as follows: according to Christian belief, we human "
    "beings have been created by an all-powerful, all-knowing God who loves us "
    "enough to send his son, the second person of the divine Trinity, to "
    "suffer and die on our account; but given the devastating amount and "
    "variety of human suffering and evil in our sad world, this simply can't "
    "be true."
)

VOICE_REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"


def _setup_clean_miopen_cache() -> str:
    """Force MIOpen to use a fresh user-db path. Returns the path so we can
    persist it in metadata.
    """
    tmp = f"/tmp/miopen_clean_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    os.makedirs(tmp, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = tmp
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = tmp  # for kernel-binary cache
    print(f"[real_capture] forcing MIOPEN_USER_DB_PATH={tmp}", file=sys.stderr)
    return tmp


def _add_chatterbox_to_path() -> None:
    here = Path(__file__).resolve().parents[1]
    candidate = here.parent / "chatterbox-rewrite" / "chatterbox" / "src"
    if not candidate.exists():
        sys.exit(f"chatterbox source not found at {candidate}")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="trials of the chunk")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clean-cache", action="store_true",
                    help="force fresh MIOPEN_USER_DB_PATH (cold solver search)")
    ap.add_argument("--force", action="store_true",
                    help="save every run, not just GPU<0.6*CPU runs")
    ap.add_argument("--text", default=None, help="override the chunk text")
    ap.add_argument("--voice-ref", default=VOICE_REF)
    ap.add_argument("--out-suffix", default=None,
                    help="append to capture dir name, e.g. 'cache' or 'clean'")
    args = ap.parse_args()

    # Set env BEFORE we touch anything torch/MIOpen-related.
    if args.clean_cache:
        _setup_clean_miopen_cache()

    if args.force:
        os.environ["HIFIGAN_BISECT_FORCE"] = "1"

    if args.out_suffix:
        # Route the capture_hook to a subdir so cache vs clean runs land separately.
        proj = Path(__file__).resolve().parents[1]
        out = proj / "captures" / args.out_suffix
        out.mkdir(parents=True, exist_ok=True)
        os.environ["HIFIGAN_BISECT_OUT"] = str(out)

    # Import + install hook BEFORE chatterbox. capture_hook itself only patches
    # HiFTGenerator after install() is called, so it's safe to import early.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import capture_hook  # type: ignore

    _add_chatterbox_to_path()
    import torch
    print(f"[real_capture] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}",
          file=sys.stderr)

    # We must import chatterbox before install() so HiFTGenerator is in
    # sys.modules and the patch lands on the live class.
    from chatterbox.tts import ChatterboxTTS  # type: ignore

    capture_hook.install()

    print(f"[real_capture] loading ChatterboxTTS on {args.device} ...", file=sys.stderr)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=args.device)
    print(f"[real_capture] loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    text = args.text if args.text else CHUNK19
    print(f"[real_capture] text len={len(text)}: {text[:60]!r}...", file=sys.stderr)
    print(f"[real_capture] voice_ref={args.voice_ref}", file=sys.stderr)

    for trial in range(args.n):
        t0 = time.time()
        wav = model.generate(
            text,
            audio_prompt_path=args.voice_ref,
            exaggeration=0.5,
            cfg_weight=0.5,
        )
        dt = time.time() - t0
        n_samples = wav.numel()
        sr = int(model.sr)
        rms = float((wav.detach().to(torch.float32).cpu().reshape(-1) ** 2).mean().sqrt())
        print(
            f"[real_capture] trial={trial} elapsed={dt:.1f}s "
            f"samples={n_samples} dur={n_samples/sr:.1f}s rms={rms:.4f}",
            file=sys.stderr,
        )

    print("[real_capture] done. Check captures/ for bundles.", file=sys.stderr)


if __name__ == "__main__":
    main()
