# Layer 1 plan — Python → single broken aten op

Goal: take "the bug is somewhere inside `HiFTGenerator.decode()`" and
reduce it to "the bug is `aten::<op_name>` with this exact (input,
weight, stride, padding, dilation, dtype) tuple." That tuple is the
artifact every deeper layer needs.

This plan does not cover MIOpen solvers, kernel source, ISA, or
silicon — those are Layers 2-6 in `LAYERS.md` and get planned after
Layer 1 produces a result.

## Three phases

### Phase A — capture one bad-run bundle

We don't yet have a saved (mel_in, hifigan_state_dict, gpu_out, cpu_out)
bundle. The audiobook project has T3 token captures (upstream of s3gen)
and `.wav` outputs (downstream of HiFiGAN), but nothing at the HiFiGAN
boundary. Phase A produces one such bundle.

Deliverable: `captures/run_<id>/` containing:

- `mel_in.pt` — exact tensor passed to `HiFTGenerator.inference` /
  `decode` on the bad run. Shape `(1, 80, T)`. fp32. Saved as `.cpu()`.
- `s_cache.pt` — the `cache_source` arg if non-empty (usually zeros).
- `hifigan_state_dict.pt` — `mel2wav.state_dict()` after model load.
  Captured once per process, doesn't change. ~30-60 MB.
- `hifigan_init_kwargs.json` — `HiFTGenerator` constructor kwargs so
  Phase B can reconstruct the module without importing chatterbox.
- `gpu_out.pt` and `gpu_out.wav` — the bad waveform.
- `cpu_out.pt` and `cpu_out.wav` — same `decode()` call rerun on CPU
  immediately after, with the model deepcopied to CPU. Ground truth
  for Phase B.
- `metadata.json` — torch / ROCm / MIOpen versions, `HSA_OVERRIDE_*`
  env, all `MIOPEN_*` env vars, hash of `~/.cache/miopen/` contents,
  process pid, wall-clock timestamp.

How:

- A monkey-patch loaded by the chatterbox child process. Lives in
  `scripts/capture_hook.py` in this project.
- Wraps `HiFTGenerator.inference` to:
  1. Save inputs and the GPU output.
  2. Re-run the same call on a CPU clone of the module (deepcopy +
     `.cpu()`).
  3. Compute spectral metrics (rms, centroid, <300 Hz energy ratio) on
     both outputs.
  4. If `gpu_rms < 0.6 * cpu_rms`, persist the bundle and log the path.
     Otherwise discard.
- Loaded via env var (e.g. `HIFIGAN_BISECT_CAPTURE=1`) the chatterbox
  TTS backend in paper-audiobooks already plumbs through to the child
  (verify in `tts/backends/chatterbox.py` before relying on it; if
  it doesn't, we add one line of plumbing).

Stop condition: one good bundle.

Risk: the audiobook pipeline currently runs HiFiGAN on CPU as a
workaround. Phase A needs to temporarily route HiFiGAN back to GPU long
enough to capture one bad run. Three options for how:

1. Wait for the current book run to finish, then run a short separate
   chatterbox session with HiFiGAN-on-GPU re-enabled, on text known to
   trigger muffled output (e.g. the 361-char "Many have argued..."
   chunk from `CHATTERBOX_DEBUG.md`).
2. Run the capture session in parallel on the same GPU. The user has
   said GPU contention is fine, but it'll be slow.
3. Skip live capture entirely: synthesize a random mel of shape
   `(1, 80, 998)` and use the real HiFiGAN weights. Fast, but the bug
   is shape-and-input dependent — the synthetic input may or may not
   trigger it.

Option 1 is cleanest. Option 3 is plan B if the bug turns out to fire
on arbitrary inputs of the right shape (which is consistent with
"MIOpen picks a bad solver per shape" — input values shouldn't matter,
only shape should).

### Phase B — offline layer-by-layer bisect

Pure offline. No chatterbox import. Loads the capture bundle and runs
the bisect.

Script: `scripts/bisect_layers.py`. ~150 lines. Pseudocode:

```
mel_in        = torch.load("captures/run_<id>/mel_in.pt")
s_cache       = torch.load("captures/run_<id>/s_cache.pt")
state_dict    = torch.load("captures/run_<id>/hifigan_state_dict.pt")
init_kwargs   = json.load(open("captures/run_<id>/hifigan_init_kwargs.json"))

# We import HiFTGenerator from the editable copy at
# ../chatterbox-rewrite/chatterbox/src/chatterbox/models/s3gen/hifigan.py
# rather than from the installed venv, so we can monkey-patch decode()
# without touching the audiobook project's venv.
hf_cpu = HiFTGenerator(**init_kwargs)
hf_cpu.load_state_dict(state_dict)
hf_cpu.eval().cpu()
hf_gpu = copy.deepcopy(hf_cpu).to("cuda")

probes_cpu, probes_gpu = [], []
def make_hook(buf, name):
    def fn(mod, inp, out): buf.append((name, out.detach().cpu().clone()))
    return fn

# Register forward hooks on every named submodule we care about:
#   conv_pre
#   ups[0], ups[1]
#   reflection_pad
#   source_downs[0], source_downs[1]
#   source_resblocks[0], source_resblocks[1]
#   resblocks[0..5]   (i*num_kernels + j for i∈{0,1}, j∈{0,1,2})
#   conv_post
# Inside _istft we add manual probes (it's not a Module, so no hook).

with torch.inference_mode():
    cpu_out = hf_cpu.inference(mel_in.cpu(),     s_cache.cpu())
    gpu_out = hf_gpu.inference(mel_in.to("cuda"), s_cache.to("cuda"))

# Diff every probe pair:
for (n_cpu, t_cpu), (n_gpu, t_gpu) in zip(probes_cpu, probes_gpu):
    diff = (t_cpu - t_gpu).abs()
    rel  = diff / (t_cpu.abs() + 1e-8)
    print(f"{n_cpu:30s}  max={diff.max():.3e}  mean={diff.mean():.3e}  "
          f"rel_max={rel.max():.3e}")
```

Output: a table. The first row whose `max` jumps several orders of
magnitude above prior rows (typical noise should be ~1e-4 absolute,
~1e-6 relative) is the broken layer.

Expected outcomes (in priority order):

1. Divergence first appears at `ups[0]` or `ups[1]` — a
   `ConvTranspose1d(k=16, s=8)` is wrong. **Most likely.**
2. Divergence first appears inside a `ResBlock` — a dilated `Conv1d`
   is wrong. The shape catalogue is small enough to bisect to a single
   conv in one more pass.
3. Divergence first appears at `_istft` output but every conv before
   it matches. Bug is in `torch.istft` / rocFFT, not MIOpen.
4. Divergence is small everywhere and only blows up at `audio_limit`
   clamping. Falsifies the "wrong conv" hypothesis. Very unlikely
   given the spectral signature.

### Phase C — collapse to a single functional call

Once Phase B names a layer (e.g. `ups[0]`):

1. Extract its weight, bias, stride, padding, dilation, groups from
   the state dict and module attributes. Save to
   `captures/run_<id>/minimal_repro.pt` as a single dict:
   `{input, weight, bias, stride, padding, dilation, groups, op_name}`.
2. Write `scripts/minimal_repro.py` (~30 lines) that loads only that
   dict and calls `torch.nn.functional.conv_transpose1d` (or
   `conv1d`) on GPU and CPU, diffs the result.
3. If the repro reproduces the divergence, this is the artifact
   Layers 2-6 consume.
4. If it doesn't reproduce — meaning the wrongness needs preceding
   layers' state to trigger (unlikely for a stateless conv, but
   possible if e.g. weight-norm parametrization isn't materialized) —
   include the full preceding subgraph as the input. Repeat until the
   smallest self-contained pair triggers the bug.

## Risks and how the plan handles them

- **The captured mel doesn't repro on the captured weights**:
  process-state-dependent per existing notes. Mitigation: capture hook
  records `~/.cache/miopen/` hash. If Phase B's GPU run on the bundle
  is correct, re-run Phase A in fresh processes (cold MIOpen cache)
  until one produces bad output that's reproducible from a cold start.
- **MIOpen find-db caching makes results non-reproducible across runs**:
  Phase B can be invoked with `MIOPEN_USER_DB_PATH=/tmp/empty_<id>` to
  force clean solver search. Compare bad-cache vs clean-cache results
  explicitly. If clean-cache always produces correct output, the bug
  is a poisoned-cache bug, not a solver bug — different (smaller)
  problem.
- **Forward hooks perturb solver selection**: hooks add CPU sync
  points which can change MIOpen autotune choices. Mitigation:
  Phase B does a hook-free reference run first (just final output
  diff). If hook-free is bad but hooked is good, switch from forward
  hooks to inserting `.clone()`-and-save calls directly into a copy of
  `HiFTGenerator.decode` (one local copy of the file in this project,
  imported instead of the chatterbox version).
- **The bug is entirely deterministic given fixed MIOpen cache + fixed
  input**: best case. Means Phase B can run repeatedly without
  variance and Phase C's minimal repro is stable.
- **The bug requires specific tensor values, not just shapes**:
  unlikely for an MIOpen kernel selection bug, but possible if the bug
  is a numerical instability in a specific solver. Phase C will
  notice — if random-input minimal repro doesn't reproduce but the
  saved-input one does, that tells us values matter and we capture a
  values-preserving repro.

## Deliverables produced by Layer 1

- `scripts/capture_hook.py` — runs in the chatterbox child.
- `scripts/bisect_layers.py` — offline bisect.
- `scripts/minimal_repro.py` — single-op reproducer.
- `captures/run_<id>/` — at least one captured bundle.
- `captures/run_<id>/minimal_repro.pt` — the tensor pair Phase C
  identified.
- `RESULTS_LAYER1.md` — diff table from Phase B, named broken layer,
  whether Phase C's minimal repro reproduces, and any oddities.

When `RESULTS_LAYER1.md` exists, Layer 1 is done. Re-plan Layer 2
(MIOpen solver identification) from those results.
