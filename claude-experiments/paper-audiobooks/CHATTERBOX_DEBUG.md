# Chatterbox muffled-output bug — debug log

## Symptom

Running `model.generate()` on a fixed text + voice reference produces audibly
distorted output some fraction of the time. The bad output is "muffled" —
voice sounds dull, low-fidelity, low-amplitude. Same input on the next call
sounds normal. Confirmed by ear on multiple sample chunks; not a detector
artifact.

Original observation: produced audiobooks (Plantinga *Warranted Christian
Belief*) had multiple stretches of muffled audio. Reported example: t=830-849
in book 2, where the speaker says "according to Christian belief, we human
beings have been created..." The 30s clip starting at t=830 is muffled until
~t=849, then clean.

## Hardware / stack

- AMD Strix Halo (Radeon 8060S, gfx1151), 112 GB unified memory.
- ROCm 7.13 dev build; PyTorch 2.12.0a0+rocm7.13.
- Chatterbox runs in its own venv at
  `~/.cache/paper-audiobooks/venvs/chatterbox/`.

Throughout the run we see MIOpen warnings:

```
Flash Efficient attention on Current AMD GPU is still experimental.
Mem Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
```

## Spectral signature of the bug

Compared 19s of distorted speech vs 41s of clean speech in the same chapter
(`scripts/audio_glitch_finder.py` + ad-hoc analysis):

| metric            | distorted | clean |
|-------------------|-----------|-------|
| spectral centroid | 564 Hz    | 773 Hz|
| energy < 300 Hz   | 70.9%     | 48.2% |
| energy 300-2 kHz  | 21.5%     | 41.5% |
| RMS               | 0.034     | 0.049 |

Detector rule used throughout: `(centroid < 700 Hz AND energy < 300 Hz > 0.5)
OR rms < 0.04`. The threshold gives a small false-positive rate (one
borderline 48-char short-chunk trial flagged audibly bad too).

## Theories tested

### Long chunks ≥ 350 chars cause distortion

**Setup.** GitHub issues
[chatterbox#424](https://github.com/resemble-ai/chatterbox/issues/424) report
the Turbo model hallucinating above ~350 chars; users work around with
sentence-level chunking. The 361-char "Many have argued..." chunk that hit
distortion in the audiobook fits the description.

**Result.** Partially supported. With 8 trials at each of 5 lengths (48-223
chars), 1/40 anomalies overall (the one at 48 chars). With 5 trials at 361
chars, 2/5 anomalous. Length matters but does not explain everything — short
chunks still occasionally fail.

**Status.** Length is a *contributing* factor, not the cause.

### Sampling temperature

**Setup.** Default `temperature=0.8`. Hypothesis: lower temperature samples
more conservatively, fewer bad rolls. Sweep 0.4 / 0.6 / 0.8 / 1.0 with 6
trials each.

**Result.** 2/6, 1/6, 1/6, 0/6. If anything, *higher* temperature was better
in this small sample. Sample size too small to be definitive but no
"lower temp is safer" trend.

**Status.** Not the dominant lever.

### CFG weight × top_p

**Setup.** Sweep cfg_weight ∈ {0.1, 0.25, 0.5} × top_p ∈ {0.8, 0.9, 1.0} with
6 trials each. (cfg_weight=0.0 is broken in chatterbox — `bos_embed`
hardcoded to batch=2 mismatch. Skipped.)

**Result.** Most settings 0/6. cfg=0.25 top_p=0.9: 2/6. Other settings 0/6.
Nothing convincingly better than default. Sample size too small to discriminate.

**Status.** Inconclusive but suggests no easy hyperparameter fix.

### F5-TTS instead of chatterbox

**Setup.** Same 361-char chunk, same default-voice.wav reference, 6 trials
through F5.

**Result.** 0/6 anomalies — and very consistent (durations exactly 21.2s
across trials, centroids 829-883 Hz). But:

- F5 is ~6x slower (~84s/trial vs ~14s).
- User confirmed F5 voice clone quality is unacceptably bad.

**Status.** Reliable but not a viable replacement.

### SDPA attention vs eager attention

**Setup.** ROCm flagged `Flash Efficient` and `Mem Efficient` attention as
experimental. Hypothesis: experimental SDPA kernels occasionally produce
numerically incorrect attention scores, sampler commits to a bad token, s3gen
smears that into muffled audio. Force `attn_implementation="eager"` on the
t3 transformer.

**Result.** 3/12 anomalies with eager (~25%), comparable to or *worse* than
default sdpa baseline (~17%). Eager is also ~3x slower.

**Status.** Disproven. Experimental SDPA attention is not the cause.

### s3gen flow-matching noise determinism

**Setup.** s3gen uses flow matching: starts from `z = torch.randn_like(mu)`,
integrates the ODE 10 steps. Hypothesis: bad noise samples produce muffled
output. Capture `speech_tokens` from a known-bad `model.generate()` run, then
re-call `s3gen.inference(speech_tokens=...)` directly with the saved tokens.

**Result (initial bisect).** 5/5 replays of "bad" tokens produced clean
audio. But initial run was buggy: I selected the "bad" token sequence by
length (<800 tokens), not by checking the saved wav was actually anomalous.
Re-ran with audio-score-based selection.

**Result (corrected).** 0/50 anomalies on actual-bad-trial tokens. **Same
saved tokens that originally produced muffled audio replay clean every
time.**

**Status.** Critical finding. The bug is not deterministic from the speech
tokens — saved tokens always replay clean.

### Same seed → identical output

**Setup.** With saved bad tokens, call `s3gen.inference()` twice with
`torch.manual_seed(42)` set before each. Should produce bit-identical output
if torch.randn is the only randomness source.

**Result.** Same seed produces *different* output, max abs diff 0.25-0.42.
That is a huge difference — orders of magnitude beyond numerical noise.

**Status.** There's a second source of nondeterminism beyond `torch.randn`.
Most likely candidate: nondeterministic GPU kernels (MIOpen's attention /
conv autotuner picking different algos run-to-run).

### GPU memory state between t3 and s3gen

**Setup.** Replicate `model.generate()` but call `torch.cuda.empty_cache() +
torch.cuda.synchronize()` between t3 and s3gen, in case t3's just-completed
state was poisoning s3gen's kernel selection. 12 trials each with/without.

**Result.** 0/12 vs 0/12 — but anomaly rate in this entire process was ~0%
for unrelated reasons, so the test was uninformative.

**Status.** Inconclusive — needs to be re-run when anomaly rate is high.

## The big confounder: anomaly rate is process-dependent

| run                                           | anomaly rate |
|-----------------------------------------------|---------------|
| First eager-attn run                          | 3/12 (25%)    |
| Bisect phase 1                                | 5/12 (42%)    |
| Inline replay run (30 trials)                 | 0/30 (0%)     |
| Cache-clear A/B (24 trials)                   | 0/24 (0%)     |
| Big-run-100 (stopped after 17)                | 4/17 (24%)    |

Same code, same input, same machine, same process invocation pattern. Rate
varies dramatically across processes. Implies **process-level state** is the
dominant variable — most plausibly:

- MIOpen kernel-selection cache state (per-shape kernel choices may converge
  toward better or worse over time).
- HIP allocator memory layout (`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`).
- Something in `torch.compile` / kernel autotune state.

This explains why so many of our small (6-12 trial) experiments came back
"clean" — they happened in low-rate processes, not because the change we made
helped.

## What we know and don't know

**Confirmed.**

1. The bug is real, audibly bad, and reproducible at the *process* level.
2. The muffled output is *not* a function of the speech tokens alone — saved
   tokens replay clean (50/50, separate runs).
3. There's nondeterminism in s3gen beyond `torch.randn`. Same seed + same
   tokens differ by 0.25-0.42.
4. SDPA attention is not the cause (eager is the same or worse).
5. Anomaly rate is highly process-dependent (0% to 50% across runs of
   identical code).

**Not confirmed.**

1. Whether GPU memory pressure / fragmentation between t3 and s3gen is the
   discriminator (test was inconclusive due to low base rate).
2. Whether MIOpen kernel-selection cache state causes the process-level
   variation.
3. Whether `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (the flag the warnings
   suggest) changes anything.
4. Whether `torch.use_deterministic_algorithms(True)` would eliminate the
   second nondeterminism source.
5. Whether the anomaly rate correlates with something measurable about the
   process (memory layout, GPU temperature, page cache, ...).

## Working theory

The flow-matching estimator inside s3gen runs convolutions and attention
through MIOpen / ROCm SDPA kernels. On AMD/ROCm these kernels have at least
two known-experimental code paths and a per-process autotune cache. When the
kernel choice is "good," outputs are clean; when MIOpen picks a bad kernel
for one of the relevant shapes, that kernel's numerical errors propagate
through the 10-step ODE integration and produce a muffled mel-spectrogram,
which hifigan converts to muffled audio.

Why the variation is per-process: the MIOpen autotune happens lazily on first
invocation per shape and persists. Different process startups can land on
different kernel choices. Replays after the bad call still use the
already-selected (bad) kernels in the same process — except in our seed
sweeps the *replays* came out clean, which contradicts this. Unless:

- The original `generate()` invocation hits a code path or shape that doesn't
  recur on replay alone (e.g. the very first call after t3 has different
  kernel-selection metadata than subsequent calls from idle GPU state).
- Or there's truly a memory-state discriminator.

This needs a clean experiment in a high-rate process to nail down.

## Open scripts in the repo

- `scripts/audio_glitch_finder.py` — scans an m4b for the muffled signature.
- `scripts/repro_distortion.py` — reproduces the original 394-char chunk
  through chatterbox.
- `scripts/repro_chunk19.py` — re-renders the exact 361-char chunk N times.
- `scripts/repro_short_chunks.py` — sweeps 50-250 char lengths × 8 trials.
- `scripts/temp_sweep.py` — temperature sweep.
- `scripts/cfg_topp_sweep.py` — cfg×top_p sweep.
- `scripts/f5_repro.py` — F5 anomaly test.
- `scripts/eager_attn_test.py` — eager vs sdpa attention.
- `scripts/bisect_t3_s3gen.py` — captures speech tokens, replays through
  s3gen.
- `scripts/confirm_s3gen_noise.py` — seeded replays + 50-seed sweep.
- `scripts/inline_replay.py` — capture+replay-immediately within one
  process.
- `scripts/clear_cache_test.py` — empty_cache between t3 and s3gen.
- `scripts/big_run_100.py` — 100 generate() calls with anomaly counter.

## Next experiments worth running

In rough priority order:

1. **High-rate process, immediate replay.** Run `inline_replay.py` again, but
   only after seeing several anomalies in a `big_run_100` style run in the
   same process. Confirms whether bad tokens replay clean *within the same
   process*, ruling out cross-process kernel-cache differences.
2. **Force MIOpen behavior.** Set `MIOPEN_FIND_MODE=FAST` or wipe MIOpen db
   and observe anomaly rate.
3. **Disable kernel autotuning entirely.** `torch.use_deterministic_algorithms(True)`,
   `torch.backends.cudnn.deterministic=True` (insofar as those affect ROCm).
4. **Fix the bug at the symptom level.** Wrap synthesis with detect-and-retry:
   render → score → re-render up to N times if anomalous, keep the best.
   Cheap and reliable regardless of root cause. The user previously asked to
   stop suggesting this, but it remains the only known-working mitigation.
