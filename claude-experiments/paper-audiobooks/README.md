# paper-audiobooks

Turn a PDF, EPUB, or DJVU into a chaptered M4B audiobook with a single command.

```sh
uv run paper-audiobooks my-book.pdf --all-chapters
```

That extracts the document, splits it into chapters, runs each chapter
through a local LLM to make it sound like spoken English (drops citations,
expands abbreviations, etc.), and synthesizes audio with chatterbox-tts.
Output lands in `output/<stem>.m4b` with a chapter marker per section.

## Pipeline

```
source (.pdf|.epub|.djvu)
  ──► extract       (marker-pdf for PDF/DJVU, ebooklib for EPUB)
  ──► split         (header-aware chaptering, see chapters.py)
  ──► rewrite       (local llama.cpp + Qwen3.5-35B-A3B, OpenAI-compatible)
  ──► tts           (chatterbox / f5 / higgs / kokoro)
  ──► m4b           (ffmpeg, with embedded chapter markers)
```

Every stage is **strictly sequential** — no thread pools, no extract pipelining,
one chatterbox subprocess at a time. Concurrency caused MIOpen kernel crashes
on AMD ROCm and the marginal speedup wasn't worth it. See `CLAUDE.md`.

## Common commands

```sh
# Whole book → one m4b with chapter markers
uv run paper-audiobooks book.pdf --all-chapters

# Just the first real chapter (skip preface/contents)
uv run paper-audiobooks book.pdf --first-chapter

# A short academic paper (no flag — treats whole doc as one chapter)
uv run paper-audiobooks paper.pdf

# Many sources in one llama-server lifetime
uv run paper-audiobooks batch *.pdf --all-chapters

# Verify the GPU is actually being used (and not silently falling back to CPU)
uv run python -m paper_audiobooks.gpu_check
```

## Resumability

Every stage caches:

- `output/<stem>.md` — extracted markdown
- `output/<stem>.chapters.json` — rewritten chapter scripts
- `output/.tts-cache/<stem>/chapter-NNNN.npy` — rendered audio per chapter

If the pipeline is killed mid-run, just re-invoke — finished work is reused.
You can also force-skip stages with `--skip-extract` / `--skip-rewrite`.

Each TTS chapter has a 3-attempt retry budget for transient chatterbox flakes.

## Hardware

This is built for **AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151)**.
ROCm 7.2 wheels for PyTorch are pinned in `pyproject.toml` via
`[tool.uv.sources]` — gfx1151 is supported natively.

If you see `cuda: False` in `gpu_check`, the wrong torch wheel is installed.
Don't `pip install torch` directly; use `uv sync --reinstall-package torch …`.

For NVIDIA: change the index URL in `pyproject.toml` to a CUDA channel and
re-lock. Concurrency limits could probably be relaxed too, but the code
doesn't expose flags for it.

The rewrite LLM (`qwen3.5-35B-A3B`, ~37 GB VRAM at Q8) is served by
`llama.cpp` with the Vulkan backend. Path is hardcoded in `server.py`:

- binary: `~/Documents/Code/open-source/llama.cpp/build-vulkan/bin/llama-server`
- model: `~/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf`

`paper-audiobooks` auto-starts and shuts down llama-server unless one is
already running on `127.0.0.1:8080`.

## Wall-time budget

On Strix Halo, per source-document:

- **Extract** (marker, GPU OCR): ~10–30 sec/page
- **Rewrite** (Qwen3.5 35B): a few seconds per chapter
- **TTS** (chatterbox): **~1.4× realtime** — 1 hour of audio takes ~85 min wall

For a 35K-word academic book: ~10 hours of audio, ~13–14 hours of GPU time.

## TTS backends

```sh
uv run paper-audiobooks backends
```

- `chatterbox` (default) — Resemble AI, voice cloning, ~500M, ~1.5 GB VRAM
- `kokoro` — fixed voices, fastest, ~0.3 GB VRAM
- `f5` — flow-matching voice clone, ~1.5 GB VRAM (separate venv)
- `higgs` — Higgs Audio v2 3B, audiobook-trained, ~7-8 GB VRAM

Set `PAPER_AUDIOBOOKS_BACKEND=kokoro` to change the default.
For voice cloning (chatterbox/f5/higgs), put a reference wav at
`~/.config/paper-audiobooks/default-voice.wav` or pass `--voice <path>`.

## Layout

```
src/paper_audiobooks/
  cli.py              # `one` and `batch` commands; auto-routes bare paths to `one`
  pipeline.py         # extract → rewrite → tts stage functions
  extract.py          # PDF (marker), EPUB (ebooklib), DJVU (ddjvu→PDF)
  extract_worker.py   # subprocess shim so torch state isn't shared across runs
  chapters.py         # header-aware chapter splitting + select_first_chapter / select_content_chapters
  rewrite.py          # OpenAI-compatible call to llama.cpp server
  server.py           # auto-launch + supervise llama-server
  tts/                # backend registry + chunked synth helpers
  tts/backends/       # one file per backend (chatterbox, kokoro, f5, higgs)
  parallel.py         # sequential TTS render loop with retry + on-disk cache
  gpu_check.py        # `python -m paper_audiobooks.gpu_check`
```

## DJVU support

Requires `djvulibre-bin`:

```sh
sudo apt install -y djvulibre-bin
```

DJVUs are converted to PDF via `ddjvu -format=pdf`, then handed to marker —
so layout-aware OCR still applies.
