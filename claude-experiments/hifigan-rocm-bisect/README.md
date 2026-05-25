# hifigan-rocm-bisect

Standalone investigation project. Goal: dive from "HiFiGAN-on-GPU produces
muffled audio on AMD gfx1151" all the way down to the actual broken
kernel / instruction / silicon behavior.

This project is intentionally separate from `paper-audiobooks` so that
running experiments here does not interfere with the audiobook pipeline
running there. Treat the audiobook project as read-only context.

## Where things live

- `BACKGROUND.md` — what we already know about the bug, captured before
  this project was created. Read first.
- `PLAN_LAYER1.md` — concrete plan for Layer 1 (Python → single broken
  aten op). Read second.
- `LAYERS.md` — the six-layer funnel from PyTorch op down to silicon.
  All deeper layers will get their own `PLAN_LAYER<N>.md` once the
  preceding layer's results are in.
- `RESULTS_LAYER1.md` — written when Layer 1 finishes.
- `captures/` — saved (mel_in, state_dict, gpu_out, cpu_out) bundles
  from real bad runs. Created during Layer 1 Phase A.
- `scripts/` — the capture hook and offline bisect script. Empty until
  we start executing Layer 1.

## Source we'll be reading and modifying (read-only references)

- HiFiGAN source (vendored, editable): `../chatterbox-rewrite/chatterbox/src/chatterbox/models/s3gen/hifigan.py`
- HiFiGAN source (installed venv): `~/.cache/paper-audiobooks/venvs/chatterbox/lib/python3.11/site-packages/chatterbox/models/s3gen/hifigan.py`
- Audiobook project (do not modify): `../paper-audiobooks/`
- Existing debug log: `../paper-audiobooks/CHATTERBOX_DEBUG.md`

## Hardware / stack assumptions

- AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151, RDNA 3.5,
  "Strix Halo", 112 GB unified memory.
- PyTorch 2.11.0+rocm7.2.
- ROCm 7.2.2, MIOpen 1.0.70202.
- No NVIDIA GPU. `nvidia-smi` is absent. Do not infer "no GPU" from
  that.

## Execution rules for future sessions

- Don't run anything until a plan in this directory has been read and
  approved.
- Don't import or modify code under `../paper-audiobooks/` or
  `../chatterbox-rewrite/`. If we need to monkey-patch HiFiGAN, the
  patch lives in this project and gets injected via env var.
- The audiobook pipeline may be running concurrently. GPU experiments
  here are allowed (user has approved) but should be short.
