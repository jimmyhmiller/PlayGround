# Next Session — Pickup Points

## Goal restated

A full Mojo Chatterbox clone that produces audio matching upstream Chatterbox using a cloned voice, drop-in for the paper-audiobooks pipeline. **Currently incomplete.**

## What's verified working with real Chatterbox weights

| Component | Parity vs upstream | Test file |
|---|---|---|
| T3 backbone forward (prefill, 30 layers) | max abs 9e-6 fp32, 0.09 bf16 | `tests/test_forward.mojo` |
| T3 backbone forward (decode, 30 layers) | max abs 3e-6 fp32 | `tests/test_forward_decode.mojo` |
| T3 multi-step argmax generation (8 tokens) | bit-exact tokens | `tests/test_generate.mojo` |
| HiFiGAN conv_pre on real (80→512, 32) input | max abs 1.1e-6 | `tests/test_hifigan_conv_pre.mojo::test_conv_pre_fp32` |
| HiFiGAN leaky_relu after conv_pre | bit-exact | same file |
| HiFiGAN ups[0] (512→256, 32→256) on real weights | max abs 7e-7 | same file `::test_ups0_fp32` |
| HiFiGAN conv_post + mag/phase + iSTFT → audio | max abs 3.4e-5 vs upstream audio | same file `::test_final_stages_fp32` |

## What's verified at toy scale only

- ResBlock (snake + conv1d + snake + conv1d + residual, chained 3 dilations): tested at 8 channels, 32 length. Real HiFiGAN ResBlocks operate at 256/128/64 channels with various K=3/7/11 kernel sizes — should work but unverified.
- STFT/iSTFT at n_fft=16 — tested with random signal, used successfully in `test_final_stages_fp32` above.

## Pickup chunks, prioritized

### Chunk 1: HiFiGAN mid-stages (ResBlock chain at real channel counts)
Build `tests/test_hifigan_resblock_chain.mojo`:
- Input: `stage_up0_after_transposed_conv.bin` (1, 256, 256)
- For each of 9 resblocks in `resblocks.0..8` (at the i=0 upsample stage), but actually only `resblocks.0`, `resblocks.1`, `resblocks.2` since `num_kernels=3` per stage. Load weights from `weights/resblocks__0__convs1__0__weight.bin` etc.
- Run snake → conv1d → snake → conv1d → +residual for each dilation [1,3,5] within each resblock.
- Mean the 3 resblock outputs together.
- Skip the source path for now (assume s=0; just verify the resblock branch on its own).
- Compare to **a new intermediate to dump from upstream**: the result of the resblocks-mean step before adding `si`. Modify `oracle/dump_hifigan_case.py` to dump it.

### Chunk 2: Source branch (s=zeros)
The source_downs[0] conv on zero input becomes a constant equal to the conv's bias propagated through the conv. Then source_resblocks[0] processes it. Build:
- Intermediate dump: `stage_up0_si.bin` (the `si` tensor before the `x = x + si` step)
- Mojo path: emit zeros, run conv1d (the source_down), run a real-scale ResBlock chain, compare.

### Chunk 3: Full HiFiGAN decode (s=zeros) — the big composition
Wire everything: conv_pre → 3× (lrelu → ups[i] → +si → 3 resblocks → mean) → reflection_pad → final lrelu → conv_post → mag/phase → iSTFT. Compare audio to `expected_wav_decode_zeros.bin`. Should match at max abs ~ few * 1e-4 given accumulated kernel error.

### Chunk 4: F0 + SourceModuleHnNSF (real source signal)
Now `s != 0`. Build:
- F0 predictor: 5× (conv1d → ELU) + linear → abs. ELU kernel needed (new).
- f0_upsample: trivial repeat-pad upsample.
- SineGen: cumsum + sin + uniform noise. Cumsum kernel and pseudo-random kernel needed; using a fixed seeded phase from the oracle keeps this deterministic.
- Tanh, linear, etc.
- Compare to `expected_wav_decode_real.bin` and `expected_wav.bin`.

### Chunk 5: S3Gen CFM stack (token → mel)
Largest single remaining piece (~3000 LOC upstream). See `S3GEN_PLAN.md`. Components in priority order:
- ConditionalDecoder (decoder.py, 333 LOC) — U-Net w/ AdaLN, attention, time embedding.
- CausalConditionalCFM (flow_matching.py, 246 LOC) — Euler ODE solver.
- UpsampleConformerEncoder (transformer/, ~1700 LOC) — conv subsample + rel-pos attention + FFN + upsample. New kernels: conv2d, LayerNorm, relative-position-bias attention.

### Chunk 6: Voice cloning conditioning
- CAMPPlus xvector (xvector.py, 428 LOC) — TDNN + BN + pooling.
- Mel extractor (utils/mel.py) — STFT + mel filterbank matmul.
- Conditioning embedding path through cond_enc (Perceiver-style, in T3).

### Chunk 7: End-to-end driver
Compose T3 (existing) + S3Gen (chunks 5-6) + HiFiGAN (chunks 1-4). Compare WAV bytes against upstream Chatterbox on a fixed prompt + cloned voice. Spectral similarity threshold TBD.

## Notes / gotchas

- Upstream `remove_weight_norm()` calls the deprecated hook API; new torch versions need `parametrize.remove_parametrizations(mod, name, leave_parametrized=True)`. See `dump_hifigan_case.py`.
- The chatterbox package has heavy deps not in our pixi env. Load source modules with `importlib.util.spec_from_file_location` directly, bypassing `chatterbox/__init__.py`.
- All conv kernels currently use `block_dim=1` which is slow at large grids. ups[0] takes 179ms — full HiFiGAN will be too slow without parallelizing inside blocks. Optimization for after correctness.
- MIOpen Winograd corruption on gfx1151 is **not** a problem for us — Mojo bypasses MIOpen entirely.
