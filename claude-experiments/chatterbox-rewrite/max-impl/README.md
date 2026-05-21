# Chatterbox on MAX (Mojo)

End-to-end TTS pipeline reimplemented on top of MAX's Mojo abstractions,
split into per-op `.so` modules orchestrated from Python (matches MAX's own
`_interpreter_ops` pattern).

## Run

```bash
pixi run python -m chatterbox_mojo <ref.wav> "<text>" <out.wav>
```

First run compiles each op `.so` lazily via `mojo.importer` (cached under
`ops/op_*/__mojocache__/`). Subsequent runs load from cache.

## Layout

- `chatterbox_mojo/` — Python orchestrator (no compute; loads ops and
  chains their `Buffer`s on the GPU).
- `ops/op_*/` — per-op Mojo `.so` modules. Each has:
    - `op_<name>.mojo` — `PyInit_*` + `PythonModuleBuilder` bindings,
      `init_op` / `forward` / `destroy_op` functions.
    - `weights_<name>.mojo` — model-specific weight loaders (extracted
      from `src/weights.mojo`).
    - Symlinks to relevant `src/*.mojo` source files.
- `src/` — canonical Mojo source for kernels and model definitions.
- `tests/` — parity tests against existing `../mojo-t3/tests/fixtures/`
  oracle outputs.
- `scripts/` — Python helpers (weight conversion, op dep setup).

## Why per-op .so

The monolithic test harness (`tests/synthesize_from_wav.mojo`, since
removed) hit a 125 GB peak RSS cold-compile OOM driven by 163
`elementwise[fn, ...]` sites across 26 source files. Each is a unique
closure type the Mojo compiler must monomorphize and hold simultaneously
during the LLVM opt pass.

Splitting into smaller `.so` modules — exactly how MAX's own kernel ops
are packaged — bounds the per-binary monomorphization graph. The largest
op (`op_flow`) cold-compiles at 3.0 GB peak RSS. Each `.so` is loaded by
CPython's extension-module loader and reads/writes GPU device buffers
directly via raw pointer handoff (`max.driver.Buffer._data_ptr()`).

See `REFACTOR_STATUS.md` and `MOJO_COMPILE_OOM.md` for details.

## Ops

| `.so` | Role |
|---|---|
| `op_load_wav` | WAV header + decode + soxr resample (subprocess to ffmpeg) |
| `op_write_wav` | Write float32 mono buffer as 16-bit PCM WAV |
| `op_text_tokenize` | BPE tokenize → list[int] |
| `op_audio_in` | s3tokenizer log-mel + s3tokenize + 24k mel + kaldi fbank + VE mel + VE forward |
| `op_campplus` | FCM + xvector backbone → 192-d speaker embedding |
| `op_spk_affine` | L2 normalize + Linear(192, 80) → spks |
| `op_t3` | T3CondEnc + Llama-30L autoregressive speech-token generator |
| `op_flow` | UpsampleConformerEncoder + CFM Euler |
| `op_hift` | NSF-HiFiGAN F0 + sine source + iSTFT → audio |

## Layer reuse from MAX stdlib

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

## Reused from `../mojo-t3`

- Fixture loader format (`src/fixture.mojo`).
- WAV I/O.
- BPE tokenizer (pure-Mojo).
- All `oracle/dump_*.py` numpy reference dumps.
- All `tests/fixtures/` weight + golden-output files.
