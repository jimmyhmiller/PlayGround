# MAX-style per-op `.so` refactor — status

Started 2026-05-20, on branch `master`. Goal: split the 125 GB cold-compile
monolith into per-op Mojo `.so` modules loaded by a Python orchestrator —
mirroring MAX's `_interpreter_ops` architecture. See plan at
`/home/jimmyhmiller/.claude/plans/noble-greeting-thompson.md`.

## Architectural validation — DONE

The per-op model works in our Mojo nightly:

| Metric | Monolith | op_campplus.so (Phase B) |
|---|---|---|
| Cold-compile peak RSS | 125 GB OOM-kill | 483 MB |
| GPU device-buffer handoff | n/a | zero-copy via `_data_ptr()` |
| DeviceContext sharing | implicit | via int address |
| Parity vs upstream | bit-exact | cos_sim 1.000000, rel_l2 4e-6 |
| State across calls | implicit | heap-allocated `OpState` + int handle |

The toolchain works end-to-end:
- Mojo `.so` with `@export def PyInit_<name>` + `PythonModuleBuilder`
- Python orchestrator does `import mojo.importer; import op_<name>` —
  the `.so` builds on-demand and is loaded via CPython's extension-module
  loader
- `max.driver.Buffer._data_ptr()` returns a raw GPU pointer; Mojo wraps
  it as a non-owning `DeviceBuffer[dtype]` and reads/writes directly
- `Accelerator()._device_context_ptr()` returns an int address that Mojo
  reconstructs into a shared `DeviceContext` via
  `DeviceContextPtr(opaque).get_device_context()`

## Ops built

| Op | Status | Lines | Phase |
|---|---|---|---|
| `op_load_wav.so` | DONE + tested | 280 | A |
| `op_campplus.so` | DONE + parity-verified | 170 (+150 weight loader) | B |
| `op_write_wav.so` | DONE (compiles) | 87 | C |
| `op_text_tokenize.so` | DONE (compiles) | 70 | C |
| `op_spk_affine.so` | DONE (compiles) | 115 | C |

## Ops remaining

Scope was originally 17 ops; consolidated to **9 total**:

| Op | What it does | est. elementwise sites | Status |
|---|---|---|---|
| `op_audio_in.so` | kaldi_fbank + mel_24k + s3tokenizer + voice_encoder | ~28 | dir created, not written |
| `op_t3.so` | text_to_input_ids + t3_cond_enc + t3_generate (CFG + sampling, autoregressive) | ~15 | dir created, not written |
| `op_flow.so` | upsample_conformer_encoder + cfm_solve_euler | ~29 | dir created, not written |
| `op_hift.so` | hift_decode_full (mel → wav) + source net + iSTFT | ~18 | dir created, not written |

Each follows the same template as `op_campplus`:
1. `init_op(weights_base_path, device_context_ptr) -> handle (int)`
2. `forward(handle, in_bufs..., out_bufs..., shape_args...) -> None`
3. `destroy_op(handle) -> None`

Each needs:
- Symlinks to relevant `src/*.mojo` files via `scripts/setup_op_deps.py`
- An inlined weight loader (extracted from `src/weights.mojo`) — the same
  pattern as `ops/op_campplus/weights_campplus.mojo`

## Phase D: orchestrator

Not yet written. `chatterbox_mojo/__main__.py` will:
1. Open Accelerator, get device context ptr
2. Call `init_op` on every op once at startup; cache handles
3. Run the pipeline:
   - load_wav → resample 24k→16k
   - audio_in: kaldi_fbank, mel_24k, s3tokenizer ×2, voice_encoder
   - campplus → spk_affine → spks (80-d)
   - text_tokenize → t3 generate (CFG, autoregressive)
   - flow encoder + CFM Euler (10 steps)
   - hift → audio
   - write_wav

## Why this was scoped down

The original 17-op plan was over-engineered. With Phase B's 483 MB peak
RSS, we have huge headroom under any sane memory budget — there's no
benefit to splitting `flow_encoder` from `cfm_step` into separate `.so`s
when both fit easily in one. Consolidating to 9 ops:
- Reduces engineering burden 2x
- Keeps cold compile of any one `.so` well under 5 GB (expected)
- Matches MAX's grouping pattern (their `matmul_ops.mojo` covers both
  Matmul and BatchMatmul; not "MatmulOp" and "BatchMatmulOp" separately)

## Reproduction

```bash
# Phase A: WAV load
pixi run python -m chatterbox_mojo.test_load_wav /home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav

# Phase B: CAMPPlus parity
pixi run python -m chatterbox_mojo.test_campplus
```

Both pass.

## Next session

1. Write `ops/op_audio_in/op_audio_in.mojo` (and weight loader)
2. Write `ops/op_t3/op_t3.mojo` (and weight loader)
3. Write `ops/op_flow/op_flow.mojo` (and weight loader)
4. Write `ops/op_hift/op_hift.mojo` (and weight loader)
5. Write `chatterbox_mojo/__main__.py`
6. Run end-to-end, compare to existing `max_impl_from_wav.wav`
