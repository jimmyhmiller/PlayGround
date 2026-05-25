# HiFiGAN root-cause dive — Layer 1 plan

Goal of Layer 1: take "the bug is somewhere inside `HiFTGenerator.decode()`"
and reduce it to "the bug is `aten::<op_name>` with this exact (input,
weight, stride, dilation, padding, dtype) tuple." That tuple is the
artifact every deeper layer needs.

This plan does **not** cover MIOpen solvers, kernel source, ISA, or
silicon — those are Layers 2-6 and get planned after Layer 1 produces a
result.

## What we know going in

- Symptom: HiFiGAN-on-GPU output has ~half the amplitude of HiFiGAN-on-CPU
  on identical mel input, with energy shifted into <300 Hz. Confirmed
  numerically (rms 0.0304 vs 0.0658, centroid 654 Hz vs 811 Hz) and
  audibly.
- Inputs match: GPU and CPU mels for the same s3gen run differ by max-abs
  4.6e-4 (fp32 rounding noise). Bug is strictly between mel and waveform.
- Process-state dependent: once a process produces a bad waveform from a
  given mel, replays from saved tokens may pick a different MIOpen kernel
  and produce a different (sometimes correct) waveform.
- HiFiGAN is `chatterbox/models/s3gen/hifigan.py` (474 lines). The hot
  path is `HiFTGenerator.decode(x, s)` at hifigan.py:412. Two
  `ConvTranspose1d` upsamples (k=16, s=8), three `ResBlock`s per upsample
  (each: 6 dilated `Conv1d` with k∈{3,7,11}, dilation∈{1,3,5}), one
  `Conv1d` post, then `torch.istft`.

## What we don't have yet

- A saved (mel_in, hifigan_state_dict, hifigan_init_kwargs, gpu_out,
  cpu_out) bundle. The repo has token-level captures (upstream of s3gen)
  and `.wav` outputs (downstream of HiFiGAN), but nothing at the
  HiFiGAN boundary. The earlier in-process debug ran the comparison live
  and didn't persist intermediates.

So Layer 1 has two phases: (A) capture once, (B) bisect offline against
the capture. Phase B never touches chatterbox or s3gen — it loads weights
+ a mel and calls `HiFTGenerator.decode` directly.

## Phase A — capture a bad-run bundle

Goal: produce one directory `scripts/hifigan_bisect/captures/run_<id>/`
containing:

- `mel_in.pt` — the exact tensor passed to `HiFTGenerator.inference` /
  `decode` on the bad run. Shape (1, 80, T). fp32, on whatever device the
  bad call ran on, then `.cpu()`-saved.
- `s_cache.pt` — the `cache_source` arg if non-empty (usually zeros).
- `hifigan_state_dict.pt` — `mel2wav.state_dict()` after model load.
  ~30-60 MB. Captured once per process; doesn't change.
- `hifigan_init_kwargs.json` — the constructor kwargs for `HiFTGenerator`
  (in_channels, base_channels, upsample_rates, kernel sizes, etc.) so
  Phase B can reconstruct the module without importing chatterbox.
- `gpu_out.wav` and `gpu_out.pt` — the bad waveform.
- `cpu_out.wav` and `cpu_out.pt` — same `decode()` call rerun on CPU
  immediately after, with the model deepcopied to CPU. This is the
  ground truth for Phase B's diff.
- `metadata.json` — torch version, ROCm version, MIOpen version,
  `HSA_OVERRIDE_GFX_VERSION`, `MIOPEN_*` env vars set, `~/.cache/miopen`
  contents hash, process pid, wall-clock timestamp.

How: a small monkey-patch loaded by the chatterbox child process. It
wraps `HiFTGenerator.inference` to:

1. Save inputs and the GPU output.
2. Re-run the same call on a CPU clone of the module (deepcopy +
   `.cpu()`).
3. Compute spectral metrics (rms, centroid, <300 Hz energy ratio) on
   both outputs.
4. If GPU rms < 0.6 × CPU rms (i.e. clearly bad), persist the bundle
   and log the path. Otherwise discard.

The patch lives in `scripts/hifigan_bisect/capture_hook.py` and gets
loaded via a `CHATTERBOX_HIFIGAN_CAPTURE=1` env var the chatterbox
backend already plumbs through (need to verify — if it doesn't, we add
one line to `tts/backends/chatterbox.py` to export it).

Stop condition: one good bundle. The book run has been producing
muffled chunks regularly, so we should get one within a chapter or two.

## Phase B — offline layer-by-layer bisect

Goal: for each named tensor `decode()` produces, compute
`max_abs(gpu_layer_out - cpu_layer_out)` and find the first layer where
that diverges past fp32 noise (~1e-4 relative).

Single script, ~150 lines, no chatterbox imports. Pseudocode:

```
load mel_in, s_cache, state_dict, init_kwargs
hf_cpu = HiFTGenerator(**init_kwargs); hf_cpu.load_state_dict(...); hf_cpu.eval().cpu()
hf_gpu = deepcopy(hf_cpu).to("cuda")

probes = []
def hook(name):
    def fn(mod, inp, out): probes.append((name, out.detach().cpu()))
    return fn

# register forward hooks on every named submodule we care about:
#   conv_pre, ups[0], ups[1], reflection_pad, source_downs[i],
#   source_resblocks[i], resblocks[i*3 + j] for i in 0..1, j in 0..2,
#   conv_post, and inside _istft we add manual probes since istft
#   isn't a Module.

run_with_hooks(hf_cpu,  mel_in.cpu(),       s_cache.cpu())  -> cpu_probes
run_with_hooks(hf_gpu,  mel_in.to("cuda"),  s_cache.to("cuda")) -> gpu_probes

for (n_cpu, t_cpu), (n_gpu, t_gpu) in zip(cpu_probes, gpu_probes):
    diff = (t_cpu - t_gpu).abs()
    print(n_cpu, diff.max().item(), diff.mean().item(),
          (diff / (t_cpu.abs() + 1e-8)).max().item())
```

Output is a table. The first row whose max-abs diff jumps several orders
of magnitude above the previous rows is the broken layer.

Expected outcomes (in order of likelihood):

1. Divergence first appears at `ups[0]` or `ups[1]` — a
   `ConvTranspose1d(k=16, s=8)` is wrong. **Most likely.**
2. Divergence first appears inside a `ResBlock` — a dilated `Conv1d` is
   wrong. The shape catalogue (kernel ∈ {3,7,11}, dilation ∈ {1,3,5}) is
   small enough to bisect to a single conv in one more pass.
3. Divergence first appears at `_istft` output but every conv before it
   matches. Would mean the bug is in `torch.istft` / rocFFT, not
   MIOpen. Surprising but informative.
4. Divergence is small everywhere and only blows up at `audio_limit`
   clamping — would falsify the "wrong conv" hypothesis entirely. Very
   unlikely given the spectral signature, but the script will say so.

## Phase C — collapse to a single functional call

Once Phase B names a layer (e.g. `ups[0]`):

1. Extract its weight, bias, stride, padding, dilation, groups from the
   state dict + module attributes. Save to
   `scripts/hifigan_bisect/captures/run_<id>/minimal_repro.pt`.
2. Write a 30-line script that loads only that tensor pair and calls
   `torch.nn.functional.conv_transpose1d` (or `conv1d`) on GPU and CPU,
   diffs the result. If it reproduces the divergence, this *is* the
   minimal repro.
3. If it doesn't reproduce — meaning the wrongness needs preceding
   layers' state to trigger — keep narrowing: include the immediately
   preceding op's output as the input. Repeat until the smallest
   self-contained pair triggers the bug.

The Phase C output (one tensor pair + one functional call) is what
Layer 2 onwards consumes. It's also what gets attached to a MIOpen
issue if we ever file one.

## Risks and how the plan handles them

- **The bug doesn't repro on the captured mel**: process-state-dependent
  per the existing notes. If Phase B's GPU run on the captured mel is
  *correct*, we have no Phase B to do. Mitigation: the capture hook
  already discards good bundles; we only persist when GPU is clearly
  bad. But "bad in the original process, good in a fresh process" is
  possible. If it happens, Phase A re-runs with the capture hook
  loading the saved mel into a fresh chatterbox process repeatedly until
  one of those fresh processes also produces bad output. That gives us a
  bundle where the bad result is reproducible from a cold start.
- **MIOpen find-db caching makes results non-reproducible across runs**:
  Phase B records `~/.cache/miopen` hash in metadata, and the bisect
  script can be invoked with `MIOPEN_USER_DB_PATH=/tmp/empty_<id>` to
  force a clean solver search. We compare bad-cache vs clean-cache
  results explicitly.
- **The hooks themselves perturb solver selection** (forward hooks add
  CPU sync points, which can change MIOpen's autotune choices):
  Phase B has a hook-free reference run that just compares final
  output, executed before the hooked run. If the hook-free run is bad
  but the hooked run is good, we know the hooks are perturbing the
  bug — we'd switch from forward hooks to inserting `.clone()`-and-
  save calls directly into a copy of `HiFTGenerator.decode` instead.

## Deliverables

- `scripts/hifigan_bisect/capture_hook.py` — runs in chatterbox child.
- `scripts/hifigan_bisect/bisect_layers.py` — offline diff script.
- `scripts/hifigan_bisect/captures/run_<id>/` — at least one captured
  bundle.
- `scripts/hifigan_bisect/RESULTS_LAYER1.md` — the diff table from
  Phase B and the named broken layer.
- `scripts/hifigan_bisect/captures/run_<id>/minimal_repro.pt` — the
  tensor pair Phase C identified.

When `RESULTS_LAYER1.md` exists, Layer 1 is done and we re-plan for
Layer 2 (MIOpen solver identification).
