# Background — what we know before Layer 1 begins

## The bug, in one paragraph

On AMD gfx1151 (Strix Halo iGPU), HiFiGAN's GPU code path intermittently
produces audio at roughly half the amplitude of HiFiGAN-on-CPU given the
exact same mel input, with most of the missing energy concentrated in the
300 Hz - 2 kHz speech-formant band. Within a single Python process, once
the bad behavior appears for a given input shape it persists across
inferences with that shape; across processes it varies. The waveform is
not NaN/Inf — just wrong, plausible-looking numbers. Audibly muffled.

## Evidence already gathered (not by this project)

Three results pinned the bug to exactly the HiFiGAN stage:

1. **Inputs to HiFiGAN match.** Captured every intermediate s3gen produces.
   The mel that HiFiGAN consumes was shape `(1, 80, 998)`, range
   `[-11.616, 1.191]`. CPU vs GPU mel: max-abs 4.6e-4, mean-abs 1.7e-6 —
   relative error ~1e-7, fp32 rounding noise. They are the same mel.
2. **Outputs of HiFiGAN do not match.** Same captured run:
   - GPU HiFiGAN: rms 0.0304, centroid 654 Hz, <300 Hz energy 0.679,
     audibly muffled.
   - CPU HiFiGAN given the GPU mel: rms 0.0658, centroid 811 Hz,
     <300 Hz energy 0.310, audibly clean.
   - CPU HiFiGAN given the CPU mel: rms 0.0658 (identical to the
     GPU-mel CPU run).
   2.2x amplitude difference, spectral balance shifted toward bass.
3. **By elimination.** Mels match; waveforms diverge; HiFiGAN is what
   runs between them. So HiFiGAN is the bug.

Older symptoms in `paper-audiobooks/CHATTERBOX_DEBUG.md` are consistent:

- Process-state-dependent anomaly rate — HiFiGAN sees a fixed set of
  shapes per inference, MIOpen autotunes per-shape, so a buggy kernel
  choice early in a process sticks for the rest of it.
- Saved-tokens replay produces clean audio sometimes — replay is a
  fresh kernel invocation that may pick a different MIOpen kernel.
- "Same seed → 0.42 max-abs diff" — HiFiGAN has no RNG, but its kernel
  choice is nondeterministic, so identical seeds can still yield
  different waveforms.

The current pipeline runs HiFiGAN on CPU as a workaround. We are NOT
trying to undo that workaround. We are trying to understand the root
cause.

## What HiFiGAN actually is, structurally

`HiFTGenerator` in
`chatterbox/models/s3gen/hifigan.py`. ~474 lines, plain `torch.nn`. The
hot path is `decode(x, s)` (hifigan.py:412). For input mel
`(1, 80, 998)`:

- `conv_pre`: `Conv1d(80→512, k=7)`.
- Two upsample stages (`self.ups`):
  - Stage 0: `ConvTranspose1d(512→256, k=16, s=8)`.
  - Stage 1: `ConvTranspose1d(256→128, k=16, s=8)`.
- After each upsample, three `ResBlock`s in parallel (summed and
  averaged). Each `ResBlock` is six `Conv1d(c→c, k∈{3,7,11},
  dilation∈{1,3,5}, padding=…)` plus `Snake` activations.
- `conv_post`: `Conv1d(channels → n_fft+2, k=7)`.
- `_istft` via `torch.istft` (rocFFT, not MIOpen).

Conv shapes seen per inference are a small, fixed set — exactly the kind
of workload MIOpen autotunes once and then reuses.

## Hypothesis going into Layer 1

A MIOpen solver for either:

- `ConvTranspose1d(stride=8, kernel=16)` at the upsample layers, or
- `Conv1d(dilation∈{1,3,5}, kernel∈{3,7,11})` inside `ResBlock`,

silently produces wrong numbers on gfx1151 for at least one of those
specific shapes. Strongest precedents in MIOpen issue tracker:

- `ROCm/MIOpen#3735` — `conv_wino_fury_RxS` crashed on gfx1151,
  disabled per-shape via PR #3685. Maintainer comment: *"rooted in a
  low level issue specific to this chip."*
- `ROCm/MIOpen#2492` — Winograd kernel produced 36.7% mismatched
  elements (silent wrong values). Maintainer comment: *"Winograd
  kernels are by design aiming performance by sacrificing numerical
  accuracy."*
- `ROCm/rocm-libraries#3559` — gfx1151 picked a bad solver,
  workaround `MIOPEN_DEBUG_CONV_DIRECT=0`.

No public report matches our exact symptom (silent half-amplitude
muffled output). Likely a new bug; the dive is to prove it.

## Existing artifacts in `paper-audiobooks/` we can use as input

- `bisect_results/phase1_trial*_tokens.npy` and
  `inline_replay_results/trial*_tokens.npy` — T3 token sequences. These
  are upstream of s3gen, not HiFiGAN. Useful for reproducing bad runs
  by replaying through s3gen, but go through nondeterministic
  flow-matching first.
- `*_clips/*.wav` — captured downstream waveforms. Useful for spectral
  detector validation, not for bisecting HiFiGAN itself.
- No (mel_in, hifigan_state_dict, gpu_out, cpu_out) bundle exists. The
  earlier in-process debug ran the comparison live and didn't persist
  intermediates. Layer 1 has to capture one before it can bisect.
