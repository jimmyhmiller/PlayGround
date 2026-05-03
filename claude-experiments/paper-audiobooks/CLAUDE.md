# paper-audiobooks

PDF / DJVU / EPUB → markdown → LLM-rewritten script → local TTS → m4b audiobook.

## Hardware: AMD Strix Halo (gfx1151, RDNA 3.5) — use ROCm, NEVER assume CPU

This machine is an **AMD Ryzen AI MAX+ 395 with Radeon 8060S iGPU
(gfx1151, RDNA 3.5, 112 GB unified memory)**. There is no NVIDIA GPU.

`nvidia-smi` is not on PATH. **That does not mean "no GPU."** If you find
yourself about to write a sentence that says "no GPU detected" or "running
on CPU," stop. Check `pyproject.toml` and `uv pip show torch` first.

The failure mode this exists to prevent: torch from PyPI is the CUDA build
(`+cuXXX`). On AMD it loads fine, reports `cuda.is_available() == False`,
and silently falls back to CPU. Marker then crawls at ~2s/page on CPU,
turning a 5-minute job into a 4-hour job. **Don't accept that fallback —
fix the torch install.**

### The correct setup (already pinned)

`pyproject.toml` pins `torch` / `torchvision` / `torchaudio` to PyTorch's
**rocm7.2** wheel index via `[tool.uv.sources]`. This wheel ships native
kernels for gfx1151, so no `HSA_OVERRIDE_GFX_VERSION` is needed.

System-wide ROCm install lives under `/opt/rocm` (currently 7.2.x).

### Verify before running

```sh
uv run python -m paper_audiobooks.gpu_check
```

Expected output ends with `OK: torch is using the GPU.` and reports
`device: AMD Radeon 8060S (gfx1151)` plus a fast matmul (~30-100ms).

The CLI commands (`one`, `batch`) call `_assert_gpu()` at startup, so
they'll refuse to run on CPU rather than silently degrading.

### What to do if `cuda: False`

1. Run `uv pip show torch`. If the version doesn't end in `+rocm7.2`, the
   wrong wheel is installed.
2. Don't pip-install torch directly. Run:
   ```sh
   uv sync --reinstall-package torch \
           --reinstall-package torchvision \
           --reinstall-package torchaudio
   ```
3. Don't remove the `[tool.uv.sources]` torch pin in `pyproject.toml` —
   that's what keeps the rocm wheel selected.

### Older ROCm wheels

If you ever need to fall back to rocm6.x wheels (which lack gfx1151
kernels), gfx1151 can sometimes masquerade as gfx1100 with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` — but in practice the rocm6.4 wheels
fail with `hipErrorNoBinaryForGpu` even with the override. Use rocm7.x or
later. (See `scripts/gpu_env.sh`.)

## Pipeline at a glance

- `extract.py` — uses marker-pdf for layout-aware PDF→markdown (GPU OCR).
- `chapters.py` — splits markdown into chapters; supports `--first-chapter`
  (drops front matter, picks first real chapter).
- `rewrite.py` — calls a local llama.cpp server to rewrite each chapter
  for spoken audio.
- `tts/` — pluggable backends: kokoro, chatterbox, f5, higgs.
- `pipeline.py` — stages `extract → rewrite → TTS → m4b`.
- CLI: `paper-audiobooks one <pdf>` or `paper-audiobooks batch <pdfs...>`.

## Pipeline is strictly sequential — no concurrency anywhere

Every concurrent code path in this pipeline blew up on AMD ROCm
(MIOpen "unspecified launch failure", "chatterbox child closed stdout
unexpectedly", catastrophic sampling-rate collapse to ~30s/iter). The pipeline
now runs strictly serially:

- Extraction: one marker subprocess at a time.
- Rewrite: chapters processed in order through one llama-server.
- TTS: one chatterbox subprocess at a time, one chapter at a time.

`--tts-workers` and `--extract-workers` are kept as flags for back-compat but
ignored. Don't reintroduce concurrency without a very good reason — even on
NVIDIA the marginal speedup wasn't worth the breakage on AMD.

Useful flags:
- `--first-chapter` — only synthesize chapter 1 of a book.
- `--all-chapters` — synthesize every substantive chapter as a separate m4b track.
- `--max-pages N` — cap marker extraction (defaults to 60 with `--first-chapter`).

## Don't

- Don't conclude "no GPU" from `nvidia-smi` being missing.
- Don't run a CPU pipeline as a "fallback" — fix torch first.
- Don't `pip install torch` ad-hoc; let uv resolve from the pinned ROCm index.
- Don't unpin `[tool.uv.sources]` in pyproject.toml.
