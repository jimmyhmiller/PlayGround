# Mojo Compile OOM Investigation

## Status: RESOLVED (2026-05-21)

The monolithic synth files (`synthesize_from_wav.mojo`, `preprocess_voice.mojo`,
and friends) have been removed in favor of per-op `.so` modules orchestrated
from Python. See `REFACTOR_STATUS.md` and `README.md`.

| Metric | Before (monolith) | After (per-op `.so`) |
|---|---|---|
| Cold-compile peak RSS | 125 GB OOM-kill | 3.0 GB (largest op: `op_flow`) |
| Distinct compilation units | 1 | 9 |
| End-to-end works | yes (warm cache only) | yes (cold + warm) |

The root cause analysis below is preserved for historical reference.

---

## Symptom

Compiling `tests/synthesize_from_wav.mojo` (or any file that imports the
`voice_encoder_inference` symbol from `src/voice_encoder.mojo`) makes the
`mojo` compiler process consume **~125 GB RSS** before getting OOM-killed
(exit 137). System has 128 GB RAM, 32 GB swap. The compiler peaks at
the system limit and dies.

Reproduces with `Mojo Opus 4.7 (1M context)` toolchain
`mojo-compiler-1.0.0b2.dev2026051806-release` (current pinned via pixi).

```text
exit=137
peak process: mojo at 125375 MB
```

## What works

- `tests/test_voice_encoder_inference.mojo` (small test file that
  imports + calls `voice_encoder_inference`) compiles fine in seconds
  and runs in ~5s, producing the correct embedding (cos_sim=0.989 vs
  upstream).
- `tests/synthesize_end_to_end.mojo` and `tests/synthesize_from_wav.mojo`
  WITHOUT the `voice_encoder_inference` import both compile fine.

So neither the function alone nor the synth binary alone is the problem.
It's the combination.

## What changed when OOM started

Adding this single symbol to `src/voice_encoder.mojo`:

```mojo
def voice_encoder_inference(
    mut ctx: DeviceContext,
    mut ve: VoiceEncoder,
    mut mel_full_buf: DeviceBuffer[DType.float32],
    mut embed_out: DeviceBuffer[DType.float32],
    t_frames: Int,
    frame_step: Int = 77,
    partial_frames: Int = 160,
    n_mels: Int = 40,
    embed_dim: Int = 256,
) raises:
    ...
    for pi in range(n_partials):
        with pi_box.map_to_host() as h:
            h[0] = Int32(pi)
        elementwise[slice_fn, simd_width=1, target="gpu"](...)
        voice_encoder_forward(ctx, ve, ve_in, ve_out, 1, partial_frames)
        elementwise[copy_fn, simd_width=1, target="gpu"](...)
```

Where `slice_fn` and `copy_fn` are closures defined ONCE before the loop,
both capture a pointer to a 1-element int buffer (`pi_box`), and the loop
just updates that buffer each iteration.

Simply IMPORTING this function into `synthesize_from_wav.mojo` (no call)
is enough to OOM the compile. Replacing the import with `voice_encoder_forward`
makes compile succeed.

## Hypotheses

### H1: Specialization explosion across closures

Each `elementwise[fn, simd_width=1, target="gpu"]` call instantiates
the closure as a comptime parameter. If the compiler is monomorphizing
each occurrence of `slice_fn` per use-site across all files, and the
combinatorial set blows up...

But the same pattern exists in dozens of other files without OOM.
And the closure is defined ONCE here (outside the loop), so use-sites
should be 1.

### H2: Closure captures cross-module specialization

`voice_encoder_inference` captures these into its closures:
- `mp` = `mel_full_buf.unsafe_ptr()` — UnsafePointer[Float32, ...]
- `vp` = `ve_in.unsafe_ptr()` — UnsafePointer[Float32, ...]
- `pip` = `pi_box.unsafe_ptr()` — UnsafePointer[Int32, ...]
- a few captured Int parameters

When the compiler imports this symbol into a file that ALSO has many
other elementwise kernels with similar capture shapes, maybe it
re-monomorphizes the closure templates per call site somehow?

### H3: `pi_box.map_to_host() as h` inside loop

```mojo
for pi in range(n_partials):
    with pi_box.map_to_host() as h:
        h[0] = Int32(pi)
```

The `with ... as h:` block creates a context manager. Each iteration
of the `for pi in range(n_partials)` may be triggering comptime
unrolling of the with-block lifetime tracking. If `n_partials` is a
runtime Int but the loop body has comptime structure (kernel launches),
the compiler may attempt to specialize per-iteration.

### H4: GPU target + AMD codegen path

On AMD gfx1151 (Strix Halo) the LLVM AMDGPU backend may have a path
that's poorly optimized for the kernel signatures we're generating.
Some prior cases in this project showed the AMD pass killing on N=1
shapes. Combine that with hundreds of compiled elementwise instances
in the same binary and the linker / opt pass may explode.

## Mitigation: split big files into smaller binaries

Doesn't address root cause. Workaround:
- Keep `voice_encoder_inference` in its own module that the big synth
  doesn't import.
- Run multi-partial VE as a separate Mojo binary that dumps the embed
  to disk; load it from disk in the synth binary.

Ugly but works.

## Reproduction steps

```bash
cd /home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/max-impl

# Confirm compile succeeds WITHOUT the symbol import:
pixi run mojo run -I src -I tests tests/synthesize_from_wav.mojo
# Expected: exit 0, ~20s wall, audio.wav saved

# Now add `voice_encoder_inference` to the import:
sed -i 's|from voice_encoder import voice_encoder_forward|from voice_encoder import voice_encoder_inference|' tests/synthesize_from_wav.mojo

# Compile again — OOM:
pixi run mojo run -I src -I tests tests/synthesize_from_wav.mojo
# Expected: exit 137, mojo process at ~125 GB RSS
```

## Next diagnostic steps

1. **Bisect what part of the function triggers it.** Reduce
   `voice_encoder_inference` to a one-line stub `pass` body and re-test:
   does the OOM still happen with the symbol imported? If yes → H2 is
   wrong, the issue is symbol enumeration not specialization.
2. **Try `@no_inline`** on `voice_encoder_inference` to force out-of-line
   compilation.
3. **Try `mojo build`** with `-mllvm -opt-bisect-limit=N` to identify
   the LLVM pass that explodes (if it's in opt not in MLIR).
4. **Reproduce with a minimal repro** — strip both files down to the
   bare elements needed to trigger.

## Investigation log

| step | observation |
|---|---|
| 0 | baseline synthesize_from_wav.mojo compiles in ~10s, ~3GB RAM |
| 1 | added voice_encoder_inference symbol; import alone → OOM at 125GB |
| 2 | removing import (even with function still in voice_encoder.mojo) → compiles fine |
| 3 | test_voice_encoder_inference.mojo (same import) compiles fine |
| 4 | difference: synth file has ~25 other elementwise capture closures |
| 5 | stub'd `voice_encoder_inference` body with `pass`; import works fine |
| 6 | restored full body; with import but call commented → compiles fine |
| 7 | re-enabled the actual call → **compiles fine and runs to completion in 14.6s** |

## Update — possibly a transient / cache-related issue

After cleaning the `/tmp/mojo*.so` files, the compile that previously
OOM'd at 125 GB now succeeds in 14.6s and produces audio. So this was
likely:
- a transient compile-cache corruption from a partial Mojo invocation, OR
- accumulated junk in the compile cache that the compiler kept linking
  in across runs.

Cleanup that helped: `rm -f /tmp/mojo*.so`

The bug stands as a real risk — the Mojo compiler shouldn't OOM on
65 GB worth of cached / stale state — but for development purposes
clearing `/tmp/mojo*.so` resolves it. File a repro upstream if it
recurs without the stale-cache trigger.

## Conclusion

The OOM is not deterministic given the same source; it correlates with
stale compile artifacts in `/tmp`. The multi-partial `voice_encoder_inference`
function compiles and runs correctly after cleaning the cache.

## Separate issue: clipping audio with Mojo voice pipeline

Even with the OOM resolved, the synth produces clipping audio (max=0.99)
when run with the full Mojo voice pipeline. Diagnostic:

| stage | metric | value |
|---|---|---|
| Mojo polyphase resample 24k→16k | rel_l2 vs librosa.soxr | ~11% |
| Mojo CAMPPlus → spks (post-affine, 80-d) | cos_sim vs upstream | 0.991 |
|  | rel_l2 | 14% |
|  | max-abs diff per coord | 0.036 |
| Mojo VoiceEncoder multi-partial (256-d) | cos_sim vs upstream | 0.989 |
|  | max-abs diff per coord | 0.049 |

Both speaker embeddings (CAMPPlus 80-d for CFM, VoiceEncoder 256-d for
T3CondEnc) drift by ~10-14% rel_l2 from upstream due primarily to the
resampler not being bit-exact to librosa.resample(res_type='soxr_hq').

When BOTH drifted embeddings are fed into the pipeline:
- T3 generates a different speech_token sequence (55 tokens vs 41 for
  single-partial vs 59 for upstream-exact).
- CFM produces a mel that's quantitatively close but in a regime where
  HiFT amplifies it past the audio clip rail.

The chain is correct algorithmically but loses fidelity at the resampler
boundary. Path to fix:

### Why each fix path is hard right now

| Path | Status | Notes |
|---|---|---|
| Implement soxr_hq pure-Mojo | Months of work | soxr is a multi-stage polyphase resampler with several thousand lines of C. The HQ quality uses 1024-tap Kaiser-windowed sinc filters with carefully tuned coefficients and multi-rate decomposition. Replicating bit-exact requires either reading the coefficients out of libsoxr's runtime state or re-deriving the design. |
| Call libsoxr.so.0 via FFI | **Blocked by Mojo nightly** | Mojo 1.0.0b2.dev2026051806 doesn't expose `external_call` or `DLHandle` from `std.sys.ffi` or `std.sys.intrinsics`. Verified by grep + import attempts. The stdlib *source* at `modular/mojo/stdlib/stdlib/sys/ffi.mojo` has both, but they're not in the `std.mojopkg` shipped here. |
| Use scipy.signal.resample_poly equivalent | Possible | Cleaner algorithm than soxr but `librosa.resample(res_type='scipy')` differs from `'soxr_hq'` by ~5% rel_l2 — still won't match upstream. |
| Precompute voice profile once | Works today | Run upstream's `prepare_conditionals` once per voice (in chatterbox venv), dump `prompt_token`, `prompt_feat`, `embedding`, `cond_emb` to disk. Mojo binary loads these at inference. This is what production TTS deployments do. |
| Pin upstream's exact 16kHz output | Works today | Have user supply both 24kHz and 16kHz versions of the reference (resampled with soxr offline). Mojo skips the resample step. |

### Decision

For "bit-exact upstream behavior", the realistic path with current Mojo
is **precompute voice profile**: take a wav, run upstream's
`prepare_conditionals` in Python once, save the resulting tensors to
disk. The Mojo TTS binary then loads them and runs inference.

This is exactly the production deployment pattern for TTS systems —
voice profiles are computed once and cached as small files.

## UPDATE — REPRODUCED and ROOT-CAUSED

Built a measurement script (`/tmp/measure_compile.sh`) that tracks peak
mojo RSS during compile. Ran it on the same source file under different
cache states:

| cache state | peak RSS | result |
|---|---|---|
| populated (4.8 GB existing) | 2.2 GB | success, ~15s |
| populated, run 5× back-to-back | 2.2 GB each | all success |
| **empty / cleared** | **122.7 GB** | **OOM (exit 137)** |
| restored | 2.2 GB | success again |

So the 125 GB peak is **cold-compile cost**, not stale state. Mojo's
on-disk cache at
`.pixi/envs/default/share/max/cache/.mojo_cache/mojo/transform/...`
(4.8 GB, 6903 entries, individual entries up to ~52 MB each) amortizes
the cost. Once the cache is warm, peak RSS is ~2 GB.

### Why cold compile blows up

The synth file imports 26 modules. Across them: **163 `elementwise[...]`
kernel sites**. Each is a comptime closure capture (often capturing 5-10
pointer/Int values), monomorphized into a unique kernel function. Each
unique monomorphization gets:
1. Mojo IR generation
2. Lowered to MLIR
3. Lowered to LLVM IR (typically 10-100× source size for templated GPU
   code)
4. LLVM optimization passes (the memory hog — opt holds the whole IR in
   memory plus working sets for each pass)
5. AMDGPU codegen
6. Cached to disk

If all 163 (+ their inner instances from MAX abstractions like
`linalg.matmul`, `nn.softmax`) are materialized simultaneously during a
single pass-manager invocation, working set explodes. Per-kernel cost is
small (~50 MB cache entry) but in-memory working set during LLVM opt is
likely several GB per kernel held concurrently.

### Reproduction

```bash
cd max-impl
# Clear cache
rm -rf .pixi/envs/default/share/max/cache/.mojo_cache
# Cold compile of large file → OOM at ~122 GB
pixi run mojo run -I src -I tests tests/synthesize_from_wav.mojo
# (gets killed)

# Restore cache (or rebuild it incrementally by compiling smaller files first)
pixi run mojo run -I src -I tests tests/test_voice_encoder_inference.mojo
pixi run mojo run -I src -I tests tests/test_resampler_soxr.mojo
# ... once enough monomorphizations are cached, the big file can compile
pixi run mojo run -I src -I tests tests/synthesize_from_wav.mojo
# (succeeds at ~2 GB)
```

### Possible mitigations (without changing Mojo)

1. **Warm the cache by compiling smaller files first** — incrementally
   builds monomorphizations that the big file will hit.
2. **Split the synth into stage 1 + stage 2 binaries** — each is small
   enough to cold-compile under the limit. This is what
   `preprocess_voice.mojo` + `synthesize_end_to_end.mojo` already do.
3. **Reduce kernel call sites** — factor out repeated elementwise
   patterns into reusable utilities (so a transpose used 5 times
   monomorphizes once not five times). May help but unclear how much.

## UPDATE — found the FFI path

Probing Mojo nightly 1.0.0b2.dev2026051806:

- `std.sys.ffi` is **not exposed** to user code.
- BUT `std.subprocess.run(cmd)` IS exposed and works.
- libsoxr is installed system-wide (apt `libsoxr0`) AND ffmpeg is built
  with `--enable-libsoxr`.
- `ffmpeg -af aresample=resampler=soxr:precision=20 -ar 16000` produces
  **bit-exact** output (max-abs=0.0, rel_l2=0.0) vs
  `librosa.resample(res_type='soxr_hq')`.

Implemented as `src/resampler_soxr.mojo` →
`soxr_resample_24k_to_16k(ctx, in_buf, out_buf, n_in, n_out)`. Verified
bit-exact in `tests/test_resampler_soxr.mojo`.

### Remaining OOM

Importing **both** `voice_encoder_inference` AND `soxr_resample_24k_to_16k`
into `synthesize_from_wav.mojo` triggers the compile-time OOM. Each works
in isolation. Suspect: cumulative monomorphization across `elementwise[...]`
closures in this large file. Workarounds:

1. **Split synthesize_from_wav into two binaries**: stage 1 does voice
   preprocessing (subprocess+soxr resample → mel → CAMPPlus → VE
   inference → save embeddings to disk), stage 2 does the TTS inference
   (load embeddings → T3 → CFM → HiFT → audio). The two binaries don't
   import each other's heavy code so compile stays manageable.
2. **Use upstream-precomputed voice profile** for the heaviest path
   (still uses Python upstream once per voice).
3. **Pre-resample the WAV with ffmpeg externally** so synthesize_from_wav
   reads two WAV files (24k and 16k) — pure-Mojo at runtime, ffmpeg is
   just a build-time data preparation step.
