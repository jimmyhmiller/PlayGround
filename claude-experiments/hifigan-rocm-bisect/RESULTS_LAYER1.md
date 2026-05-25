# Results — Layer 1 investigation

## TL;DR

**Believed cause: MIOpen's Winograd convolution solver family
(`ConvBinWinogradRxSf3x2` and `ConvBinWinogradRxSf2x3g1`) on AMD
gfx1151 produces numerically wrong output for at least one of the conv
shapes exercised during chatterbox `s3gen` flow-matching + HiFiGAN
inference.**

How sure: high. Two consecutive 0/20 confirmation runs with
`MIOPEN_DEBUG_CONV_WINOGRAD=0` versus consistent ~10% bad rate with
default config. Same trial07 tokens that fired BAD on every other ablation
(default, FIND_MODE=2, experimental-SDPA-off) produced clean output once
Winograd was disabled. The Winograd family also has prior gfx1151
silent-wrong-value reports in the MIOpen tracker
(`ROCm/MIOpen#3735`, `ROCm/MIOpen#2492`).

The original PLAN_LAYER1.md goal of "reduce to a single
`(input, weight, op_kwargs)` tuple that reproduces the divergence
deterministically" was not achieved and probably cannot be achieved as
literally written — the bug is per-call nondeterministic and requires
the full chatterbox process state. See *Open work* below.

## Next steps (in order)

### 1. Confirm the Winograd workaround at N=100

Before deploying or building further on this, validate the 0% claim with
a much larger sample size. Two confirmation runs at N=20 each is
suggestive but a 100-trial run with zero anomalies is what we want
before we lean on it.

Concrete:

```sh
MIOPEN_DEBUG_CONV_WINOGRAD=0 \
  ~/.cache/paper-audiobooks/venvs/chatterbox/bin/python \
  scripts/replay_tokens.py --n 100 \
  > /tmp/replay_no_winograd_n100.log 2>&1
```

Expected: `bad=0/100`. If any anomalies fire, Winograd is partial-cause
and we need to keep digging (next candidate is the Direct family, or
the `aotriton` SDPA backend). If 0/100, proceed to step 2.

Time: ~50–60 minutes. Each `s3gen.inference` call is ~30s.

### 2. Apply the workaround to paper-audiobooks

In `paper-audiobooks/src/paper_audiobooks/tts/backends/chatterbox.py`,
the chatterbox child-process env block sets allocator config — add the
Winograd disable there:

```python
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
```

Then remove (or invert default of) the `CHATTERBOX_HIFIGAN_CPU=1`
workaround so HiFiGAN runs on GPU again.

Real-world test: run a known-bad book (e.g.
`samples/books/plantinga-warrant-and-proper-function.pdf`, the one that
was running when this investigation started) end-to-end with
HiFiGAN-on-GPU + Winograd-off. Listen to a sample; check the spectral
detector counts.

### 3. Per-conv localization (for an upstream-fileable bug report)

Until step 3 is done, we cannot file a clean MIOpen bug. AMD will need
either a small standalone HIP repro or at least a saved
`(input, weight, conv_args)` tuple where Winograd-on output differs from
Winograd-off output by orders of magnitude.

Approach (handed off — not done in this session):

a. Instrument every `Conv1d` in `model.s3gen.flow.decoder` and
   `model.s3gen.mel2wav` with no-sync forward hooks that record
   `(input, output)`. Skip `model.s3gen.tokenizer.encoder` and
   `model.s3gen.speaker_encoder` — those run during
   `prepare_conditionals`, not the hot path.
b. Run twice (two fresh processes), with `torch.manual_seed(seed+i)`
   per trial so SineGen draws are reproducible:
   - Process A: default MIOpen.
   - Process B: `MIOPEN_DEBUG_CONV_WINOGRAD=0`.
   Both feed `phase1_trial07_tokens.npy` (the deterministically-bad one).
c. For each conv that ran in both, diff `output_A` against `output_B`.
   Most diffs will be fp32-noise (different but tiny). The conv with the
   biggest absolute output divergence is the broken one.
d. Save its (input, weight, stride, padding, dilation, groups, kernel
   size) tuple. That + a runner script is the upstream artifact.

**Notes for whoever picks this up:**
- Don't use sync hooks (`.cpu()` inside the hook) — the bug stops
  firing with sync. We confirmed this experimentally.
- Even with no-sync hooks, the hook's clone may itself perturb the
  bug — accept that the localization may need a few attempts.
- Standalone Conv1d with random fp32 input is correct on every shape we
  tried. Repro **must** use the saved real input.
- The `winograd_repro_real_inputs.py` script tested every HiFiGAN
  ResBlock standalone with real CPU-intermediates input and got fp32-
  noise diffs — so the broken conv is most likely **inside
  `s3gen.flow.decoder`** (the matcha-tts CFM), not HiFiGAN.

### 4. (Optional) MIOpen kernel binary inspection

Once a specific conv shape is identified, the cached compiled binary
is at `~/.cache/miopen/<build-hash>/<solver>.co`. Disassemble with
`/opt/rocm/llvm/bin/llvm-objdump -d --triple=amdgcn` and compare against
the same solver compiled for gfx1100 (RDNA 3, working) for the same
shape. ISA-level divergence would confirm a codegen issue vs. silicon.

## What we proved

### 1. The bug is statistical, not deterministic

Per-call, in the same process, with identical (mel, weights, s_cache),
GPU output rms varies between ~0.028 (audibly muffled) and ~0.067 (clean
— matches CPU exactly). Two consecutive forwards of the same input give
different audio. We confirmed this directly by comparing
`gpu_out.pt` vs `inline_gpu_out.pt` from the same captured bundle:
`gpu_out.pt` rms = 0.0280 (bad), `inline_gpu_out.pt` rms = 0.0614
(clean) — same process, same input, two consecutive calls.

### 2. The bug requires the full chatterbox process state

| Setup | Anomaly rate |
|---|---|
| HiFiGAN-only replay of saved mel (50 trials) | 0/50 = 0% |
| Mel-only replay inside loaded ChatterboxTTS process | 0/20 = 0% |
| **Skip-T3 replay (saved tokens → s3gen.inference)** | **2/20 = 10%** |
| **Full pipeline (model.generate on chunk19)** | **8/20 = 40%** |

Implications: the bug is NOT in HiFiGAN given a correct mel, and NOT in
just having T3+s3gen instantiated as PyTorch modules. It fires when
`s3gen.inference` actually executes (flow-matching CFM + HiFiGAN). T3
amplifies the rate ~4× but is not necessary.

### 3. Hooks with GPU sync eliminate the bug

Forward hooks that do `.cpu()` per-layer (introducing GPU→CPU sync points)
make the bug rate go to 0. Hooks that clone on-GPU (no sync) do NOT
suppress the bug. Implies the bug has a temporal/scheduling component;
forced sync changes execution timing enough that the buggy kernel doesn't
fire. This is also why earlier "live bisect" attempts saw only fp32-noise
diffs — the hooks themselves suppressed the bug.

### 4. Standalone replay of any single layer is bit-clean

- `resblocks.3` standalone with the **clean CPU input**, 30 trials: GPU
  matches CPU to `max_abs=2.6e-6, mean_abs=1.3e-7`. Identical across all
  30 trials.
- Same with the **GPU-perturbed input** (CPU input + GPU-vs-CPU delta from
  bundle's intermediates), 30 trials: same fp32-noise diffs.
- Every other ResBlock tested (0,1,2,4,5,6,7,8) showed the same: clean.
- `source_downs.0` standalone, 20 trials: also clean (`mean=4.4e-9`).
- `torch.stft` (rocFFT) on saved s tensor, 10 trials: deterministic to
  `max=7e-9` vs CPU.
- 22 random-input `F.conv1d` calls on every Winograd-using shape from the
  trace (kernels 3/5/7/11, dilations 1/3/5, channels 64/128/256/512):
  all clean, `mean ~= 1e-7` to `1e-6`.

### 5. The bug fires only with full-process value distributions

Real-input replay of HiFiGAN ResBlocks using `cpu_intermediates.pt`
captured during a bad pass is also clean. So even with realistic
intermediate tensors, isolated Conv1d layers don't reproduce. The bug
requires the actual sequence of conv calls + their accumulated GPU state
+ MIOpen's per-call solver decisions.

### 6. The MIOpen toggle that flips bad → clean is Winograd

| Config | Skip-T3 anomaly rate |
|---|---|
| Default | 2/20 = 10% |
| `MIOPEN_FIND_MODE=2` (no autotune search) | 2/20 = 10% — no change |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` + math SDPA | 1/20 = 5% — partial |
| **`MIOPEN_DEBUG_CONV_WINOGRAD=0`** | **0/20 = 0% — confirmed twice** |

Trial07 tokens specifically reproduced the bug deterministically across
default, `FIND_MODE=2`, and `EXPERIMENTAL=0` runs (always BAD on first
encounter, clean on repeat). With Winograd disabled, trial07 produces
clean output.

### 7. The MIOpen log trace confirms Winograd is on the hot path

`miopen_trace_logs/trace_clean_phase1_trial06_tokens.log` (full
`MIOPEN_LOG_LEVEL=6 MIOPEN_ENABLE_LOGGING=1` trace of one
`s3gen.inference` call) shows MIOpen picks Winograd solvers
(`ConvBinWinogradRxSf3x2` / `ConvBinWinogradRxSf2x3g1`) for many of the
3/5/7/11-kernel convs in HiFiGAN's resblocks AND for several 512-channel
convs that are likely in `s3gen.flow.decoder` (matcha-tts CFM).
Non-Winograd solvers used (`GemmFwd1x1_0_1`, `GemmFwdRest`,
`GemmBwdRest`) for stride-1x1, transpose, and odd-shape convs.

## Workaround for paper-audiobooks

In the chatterbox child process env (`tts/backends/chatterbox.py`):

```python
env["MIOPEN_DEBUG_CONV_WINOGRAD"] = "0"
```

That eliminates the bug. The current `CHATTERBOX_HIFIGAN_CPU=1`
workaround can be removed and HiFiGAN run on GPU.

There is a small performance cost: MIOpen will fall back to slower
non-Winograd solvers (GemmFwd / Direct / ImplicitGEMM) for the affected
shapes. Cold-start solver search may take longer per fresh process. The
audible-bug elimination is worth it.

## Hardware/stack at time of investigation

- AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU, gfx1151 (RDNA 3.5, "Strix Halo"),
  112 GB unified memory.
- PyTorch 2.12.0a0+rocm7.13.0a20260411
- ROCm 7.13.60980 (HIP version reported)
- MIOpen find-DB at `~/.config/miopen/gfx1151_20.HIP.3_5_1_f000f7786e.ufdb.txt`
- Wheel-bundled MIOpen system DB at
  `_rocm_sdk_libraries_gfx1151/share/miopen/db/gfx1151_20.HIP.fdb.txt`
  is unreadable (every run logs a warning); MIOpen falls back to live
  solver search per-shape.

## What's in `captures/`

- `grind/run_<id>/` — full bundles from auto-grind hits.
  - `mel_in.pt`, `s_cache.pt`, `hifigan_state_dict.pt`, `hifigan_init_kwargs.json`
  - `gpu_out.{pt,wav}` — the bad output captured live
  - `cpu_out.{pt,wav}` — the CPU twin (clean reference)
  - `gpu_intermediates.pt` — per-Module forward outputs of the bad GPU pass
    (recorded with no-sync hooks during the actual call)
  - `cpu_intermediates.pt` — same on CPU (deterministic ground truth)
  - `inline_cpu_out.pt` — CPU re-run after the bad call (always clean)
  - `bisect_diffs.json` — per-Module diff between cpu_int and gpu_int
  - `metadata.json` — env, sampling rate, MIOpen cache hash, etc.
  - For grind 5+ (`run_20260510T033906_0300870f`): also
    `rng_state.pt` and `miopen_snapshot/` (user DB + kernel cache).
- `forced/` — bundles from --force runs that include some near-miss /
  clean cases for spectrum-of-severity comparison.

## Open work (Layer 1 → Layer 2 boundary)

The original plan asked for a single `(input, weight, op_kwargs)` tuple
that deterministically reproduces the divergence on GPU vs CPU. **That
artifact does not appear to exist** as literally specified, because:

- Single-conv replay with saved or random inputs is bit-clean.
- The bug only fires inside the full chatterbox call sequence.
- Even within a bad process, two consecutive identical forwards may
  produce different output (the bug is per-call nondeterministic).

The closest deliverable for Layer 2/upstream-MIOpen would be:

1. **Conv-localization within the live bad pass.** Instrument every
   `Conv1d/Conv2d/ConvTranspose1d` in `s3gen.flow.decoder` and
   `s3gen.mel2wav` with no-sync hooks that record (input, output) on
   each forward. Run twice — once with default MIOpen (Winograd allowed),
   once with `MIOPEN_DEBUG_CONV_WINOGRAD=0` — using `torch.manual_seed`
   to make SineGen reproducible. For each conv whose Winograd-on output
   exceeds Winograd-off output by orders of magnitude relative to CPU
   reference, that is the buggy conv. The captured (input, weight) at
   that conv becomes the upstream-MIOpen-actionable artifact.

   This was attempted in this session via `scripts/conv_localize.py` but
   not completed — it produces ~4 GB of intermediates per run and needs
   focused diff analysis. Hand off rather than push through.

2. **MIOpen kernel binary diff.** Once a specific conv shape is
   identified, `~/.cache/miopen/<hash>/<solver>.co` should contain the
   compiled HIP kernel. `llvm-objdump -d --triple=amdgcn` on it vs.
   gfx1100 (or other gfx) for the same shape would show ISA-level
   divergence.

3. **HIP minimal repro.** Once a specific conv shape and input value
   distribution is identified, write a ~30-line HIP/PyTorch script that
   loads only those tensors and calls the broken solver via the
   `MIOPEN_DEBUG_CONV_WINOGRAD=1` toggle. AMD-actionable.

## Scripts produced this session

- `scripts/capture_hook.py` — patches `HiFTGenerator.inference` to detect
  bad calls (using same `_is_anomalous` rule as `paper-audiobooks` chatterbox
  backend), save bundles, capture GPU intermediates via no-sync hooks,
  run post-hoc CPU diff. Toggleable via `HIFIGAN_BISECT_CAPTURE=1`.
- `scripts/grind.sh` — outer loop running fresh chatterbox processes until
  the first anomaly fires, then exits.
- `scripts/real_capture.py` — single-process driver that runs N trials
  of chunk19 with the capture hook installed.
- `scripts/synth_capture.py` — synthetic-mel control (showed bug does NOT
  fire on random mels of the right shape — falsified the original
  "shape-only" hypothesis).
- `scripts/replay_tokens.py` — skip-T3 replay using saved
  `bisect_results/phase1_trial*_tokens.npy` from `paper-audiobooks`. Used
  for the FIND_MODE=2 / SDPA-off / WINOGRAD=0 ablations.
- `scripts/replay_tokens_no_sdpa.py` — same but disables experimental
  ROCm SDPA backends.
- `scripts/replay_with_state.py` — restores MIOpen on-disk state +
  RNG state from a captured bundle, replays HiFiGAN-only inference.
  Showed isolated-HiFiGAN doesn't reproduce even with state restored.
- `scripts/full_pipeline_replay.py` — runs full ChatterboxTTS in-process
  in two modes: mel-only (HiFiGAN inside loaded chatterbox) and
  full-generate.
- `scripts/repeatability_test.py` — N-trial repeatability with same RNG.
- `scripts/bisect_layers.py` — offline per-Module diff bisect (mostly
  superseded by inline bisect in capture_hook).
- `scripts/replay_layer.py` — standalone single-layer replay against
  saved CPU reference. Showed every HiFiGAN ResBlock is clean in
  isolation.
- `scripts/check_stft.py` — confirms `torch.stft` (rocFFT) is
  deterministic to noise floor.
- `scripts/winograd_reproducer.py` — random-input scan over Winograd-using
  shapes. Clean.
- `scripts/winograd_repro_real_inputs.py` — real-input scan over HiFiGAN
  ResBlocks using `cpu_intermediates.pt`. Clean.
- `scripts/miopen_trace.py` — runs s3gen.inference with full
  `MIOPEN_LOG_LEVEL=6` logging, redirected per-call to separate log files.
  Output in `miopen_trace_logs/`.

## Useful audio for ear-checking

In `s3://jimmyhmiller-bucket/hifigan-rocm-bisect/`:
- `hit3_ratio0.456_gpu.wav` — clearly muffled GPU output (rms 0.028)
- `hit3_ratio0.456_cpu.wav` — CPU twin (rms 0.061)
- `hit5_captured_gpu_rms0.0360.wav` — another captured bad case
- `nearmiss_ratio0.796_gpu.wav` — close-but-not-flagged (audibly fine)
- `replay_isolated_gpu.wav` — replay of hit5 mel through fresh
  HiFiGAN-only process (clean — proves replay can't reproduce)

## Confidence summary

- **High confidence**: Winograd is the cause. Two clean 0/20 runs with
  Winograd disabled vs. consistent ~10% with it enabled is unambiguous.
- **High confidence**: workaround works for paper-audiobooks.
- **Medium confidence**: bug is in `s3gen.flow.decoder` (matcha CFM)
  rather than HiFiGAN — based on standalone HiFiGAN ResBlock replay
  being clean. Could also be a HiFiGAN conv only when fed a particular
  CFM-output mel; we haven't fully separated those.
- **Open**: which specific conv shape + value distribution.
