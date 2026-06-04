# chatterbox-mojo backend

A drop-in replacement for the upstream `chatterbox` backend, using the Mojo
rewrite at `../chatterbox-rewrite/max-impl`. Roughly **2× faster** on AMD
Strix Halo (gfx1151) with **0% WER** difference on round-trip Whisper testing.

## Usage

```bash
paper-audiobooks one some.pdf --tts-backend chatterbox-mojo
```

## How it works

`paper_audiobooks/tts/backends/chatterbox_mojo.py` defines
`ChatterboxMojoBackend`. It spawns one long-lived child process via
`pixi run python` (so the child uses the Mojo + MAX env), reusing
`chatterbox_mojo.pool._Worker` for transport. JSON RPC over stdin/stdout
with binary float32 audio payloads (size-prefixed).

Voice handling matches `chatterbox`:
  - `"default"` → uses `~/.config/paper-audiobooks/default-voice.wav`
  - any other path → treated as a 24kHz .wav reference

## Config knobs

The wrapper hard-codes the recommended fast-path defaults:
- `use_bf16=True` (matrix-core GEMM)
- `cfm_steps=5` (Euler steps in CFM)
- T3 QKV+MLP fusion on

Override via env:
- `CHATTERBOX_MOJO_REPO=/path/to/chatterbox-rewrite/max-impl` if the
  repo is in a non-default location.

## Limitations vs upstream

- No `dump_intermediates()` hook yet — the upstream backend's anomaly-capture
  harness isn't ported. If you set `CHATTERBOX_HALT_ON_ANOMALY=1`, the
  detector still runs on the **output audio** (it's audio-level), but no
  internal CFM dumps will be produced on halt.
- Each worker process loads its own model (no shared weights). 2 workers
  fit; 3+ OOMs the GPU memory pool.
