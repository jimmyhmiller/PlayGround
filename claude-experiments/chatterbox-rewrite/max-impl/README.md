# Chatterbox on MAX (Mojo)

End-to-end TTS pipeline reimplemented on top of MAX's Mojo abstractions.

## Layout

- `src/` — Mojo source. Thin Module wrappers around `nn.*`/`linalg.*`
  kernel functions, model architectures composed from these Modules.
- `tests/` — Parity tests against existing `../mojo-t3/tests/fixtures/`
  oracle outputs (numpy/torch reference).
- `scripts/` — Python helpers (weight conversion from upstream
  safetensors, fixture dumps if needed).

## Layer reuse from MAX

We use the following Mojo packages from the MAX install
(`$PIXI/lib/mojo/*.mojopkg`):

| Need | Use |
|---|---|
| matmul | `linalg.matmul.matmul` |
| batched matmul | `linalg.bmm.batched_matmul` |
| layer norm | `nn.normalization.layer_norm` (GPU) |
| RMSNorm | `nn.normalization.rms_norm` |
| softmax | `nn.softmax.softmax` |
| RoPE | `nn.rope.apply_rope` / `nn.fused_qk_rope.*` |
| KV cache | `kv_cache` package |
| gather/scatter | `nn.gather_scatter` |
| concat | `nn.concat.concat` |
| activations | `nn.activations.{relu, elu, leaky_relu}` |
| GELU | written in Mojo (`std.math.erf`-based) — small |
| pooling | `nn.pool` |
| pad | `nn.pad`, `nn.pad_gpu` |
| top-k / sampling | `nn.topk`, `nn.sampling` |

## Module pattern

```mojo
struct Linear:
    var weight: TileTensor[...]   # (out, in)
    var bias:   TileTensor[...]   # (out,)
    def __call__(self, mut ctx: DeviceContext, x: TileTensor, out: TileTensor):
        linalg.matmul.matmul[target="gpu"](out, x, self.weight, dctx)
        bias_add(out, self.bias, ...)
```

## Phasing

0. Project skeleton + sanity test that `nn.softmax` runs.
1. VoiceEncoder + s3tokenizer.
2. T3 (Llama-30L + cond_enc + perceiver) + KV-cached generation.
3. s3gen (UpsampleConformerEncoder + CFM + HiFiGAN).
4. End-to-end synthesize binary.
5. Weight loading from upstream safetensors.

## Reused from `../mojo-t3`

- Fixture loader format (`src/fixture.mojo` equivalent) — reuse the format,
  paths point to existing fixture files.
- WAV I/O (load_wav, save_wav).
- BPE tokenizer (pure-Mojo).
- All `oracle/dump_*.py` numpy reference dumps.
- All `tests/fixtures/` weight + golden-output files.
