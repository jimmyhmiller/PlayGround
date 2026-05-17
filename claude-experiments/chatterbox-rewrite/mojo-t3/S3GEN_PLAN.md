# S3Gen Mojo Port — Plan

## Goal

Port chatterbox's S3Gen vocoder (speech tokens → 24 kHz audio) to Mojo, matching upstream output. Sidesteps the MIOpen-Winograd corruption that bites upstream chatterbox on gfx1151 (see `paper-audiobooks/hifigan-rocm-bisect/RESULTS_LAYER1.md`).

## Pipeline overview (upstream)

```
speech_tokens (B, T)
  │
  ▼  CausalMaskedDiffWithXvec.inference   [flow.py]
  ├── token_emb (lookup)                                  ← simple
  ├── + xvector (CAMPPlus speaker emb)                    ← xvector.py 428 LOC
  ├── encoder = UpsampleConformerEncoder (6 blocks)       ← transformer/* ~1700 LOC
  │   • conv subsample
  │   • rel-pos self-attn × 6
  │   • positional FFN × 6
  │   • upsample (rate 2)
  │
  ▼  → mu (B, 80, T*2)   [the "mean" of the CFM target]
  │
  ▼  CausalConditionalCFM.solve_euler     [flow_matching.py]
  ├── Sample z ~ N(0, I), shape (B, 80, T*2)
  ├── For each of N_TIMESTEPS (default 10):
  │     v = estimator(z, mu, t, prompt_feat, spk_emb)     ← decoder.py 333 LOC
  │     z = z + (t_{i+1} - t_i) * v
  │
  ▼  → mel (B, 80, T*2)
  │
  ▼  HiFTGenerator.inference              [hifigan.py 474 LOC]
  ├── conv_pre
  ├── 4× upsample blocks (transposed conv + ResBlock×3)
  ├── conv_post + final tanh / istft
  │
  ▼  → audio (B, T_audio @ 24kHz)
```

## Module map → Mojo kernel needs

| Module | LOC | Upstream file | Mojo kernels needed | Status |
|---|---|---|---|---|
| Speech-token embedding | trivial | `flow.py` | embed_lookup (already have) | reuse |
| CAMPPlus speaker encoder | 428 | `xvector.py` | conv1d, BN, TDNN, mean+std pooling | TODO |
| Mel-spectrogram extractor | small | `utils/mel.py` | STFT, mel-filterbank matmul | TODO |
| UpsampleConformerEncoder | 318 | `transformer/upsample_encoder.py` | conv subsample, rel-pos attn, FFN, transposed conv upsample | TODO |
| Conformer rel-pos attention | 330 | `transformer/attention.py` | new QK kernel (rel-pos additive bias), reuse softmax/AV | TODO |
| Positional FFN (Swish) | 115 | `transformer/positionwise_feed_forward.py` | linear→swish→linear; mostly matmul + silu | TODO |
| Conv subsampling | 383 | `transformer/subsampling.py` | conv2d (stride-2) + ReLU | TODO |
| ConditionalDecoder | 333 | `decoder.py` | U-Net style: conv1d, AdaLN, attention, time emb | TODO |
| CausalConditionalCFM | 246 | `flow_matching.py` | Euler ODE step (just a scaled-add); orchestrates `estimator` | TODO |
| HiFTGenerator (vocoder) | 474 | `hifigan.py` | conv1d, transposed-conv1d, ResBlock (dilated conv), STFT, LeakyReLU | **HIGHEST PRIORITY** |
| S3Tokenizer | external | `s3tokenizer/` | Used at encode time (cloning); not needed if we accept pre-tokenized speech | defer |

## Kernel inventory we'll need to add

These don't exist yet in `mojo-t3/src/`:

- **conv1d** (general stride, dilation, padding, groups=1)
- **transposed_conv1d** (for HiFiGAN upsampling layers)
- **batch_norm1d** (CAMPPlus uses BN)
- **layer_norm** (Conformer uses LN, not RMSNorm)
- **leaky_relu** (HiFiGAN activation)
- **swish/silu_only** (we have silu_mul which is gated; need plain silu for Conformer FFN)
- **stft / istft** (mel feature extractor + HiFiGAN output)
- **mean+std pooling** (CAMPPlus)
- **rel-pos QK kernel** (Conformer attention adds a learned relative-position bias before softmax — differs from Llama RoPE)
- **AdaLN** (ConditionalDecoder — affine-modulated LayerNorm)

Most are mechanical; conv1d is the big one and the load-bearing kernel for HiFiGAN.

## Priority order (best ROI first)

1. **HiFiGAN** alone — given mel input from upstream torch, produce audio in Mojo. This is the piece that breaks upstream on gfx1151 (Winograd corruption); a Mojo replacement of just HiFiGAN, called from upstream's CFM output, would be an immediate win for paper-audiobooks even without the rest of the rewrite. Building blocks: conv1d, transposed_conv1d, ResBlock (3 dilated convs + skip), LeakyReLU, STFT/iSTFT.

2. **ConditionalDecoder + CFM solver** — the largest single matmul-heavy piece. Given a working HiFiGAN, this is the next gate to producing audio without upstream.

3. **UpsampleConformerEncoder** — substantial port; relies on conv1d (shared with HiFiGAN), LayerNorm, relative-position attention, FFN-swish.

4. **CAMPPlus xvector + mel extractor** — voice cloning. Needed for parity-of-voice; the rest can be tested with a fixed pre-computed `embedding` tensor.

5. **End-to-end audio parity** — compare Mojo waveform to upstream Chatterbox waveform on the same prompt + cloned voice.

## Immediate next steps

- Build `oracle/dump_hifigan_case.py`: dump HiFiGAN weights + a small mel tensor + the upstream output waveform.
- Implement `src/conv.mojo` with conv1d + transposed_conv1d + a Mojo HiFiGAN driver.
- Parity test mel→audio.

That's the discrete first chunk we can land with a clear pass/fail gate.
