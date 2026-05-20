# Mojo Compile OOM Investigation

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
1. Implement soxr_hq-equivalent resampling (significant work).
2. OR: precompute the speaker embeddings once per voice and cache them
   to disk (treats voice as a fixed input, not derived at inference).
3. OR: make CFM/HiFT robust to small embedding noise (model change, not
   pipeline).
