# paper_audiobooks — package internals

Source for the `paper-audiobooks` CLI: turn a **PDF / EPUB / DJVU** into a
chaptered **M4B** (or MP3) audiobook. Document → markdown → LLM-rewritten
spoken script → local TTS → ffmpeg mux with chapter markers.

This file documents *how the code is wired*. For hardware/GPU rules (ROCm
wheels, "never assume CPU"), see the **root `CLAUDE.md`** — those constraints
override everything here.

## Pipeline stages

```
source.(pdf|epub|djvu)
  → extract   PDF/DJVU via marker-pdf (GPU OCR); EPUB via ebooklib  → output/<stem>.md
  → split     PDF outline if present, else header heuristics         → list[Chapter]
  → rewrite   local llama.cpp (OpenAI-compatible) per chapter        → output/<stem>.chapters.json
  → tts       pluggable backend, chunked + cached                    → output/.tts-cache/...
  → mux       ffmpeg, embedded chapter markers                       → output/<stem>.m4b
```

`cli.py` orchestrates; `pipeline.py` holds the stage functions
(`stage_extract`, `stage_extract_subproc`, `stage_rewrite`, `stage_tts`).
`PipelinePaths` / `paths_for()` derive all output paths from the source stem.

## Module map

| File | Role |
|------|------|
| `cli.py` | Click commands `one` / `batch` / `corpus` / `backends`. `main()` auto-routes a bare path to `one`. Audio commands call `_assert_gpu()` at startup. |
| `corpus.py` | Bulk corpus builder — extract + chapter-split + metadata enrich, **no audio**. One full-content JSON record per doc under `output/corpus/<stem>.json` + metadata-only `index.jsonl`. |
| `metadata_lookup.py` | Online metadata enrichment: ISBN/DOI extraction from text, Open Library + Google Books (books), Crossref (DOIs/papers), and an LLM title/author fallback via llama.cpp. |
| `pipeline.py` | Stage functions + `PipelinePaths`. `stage_extract_subproc` runs marker in a child process so torch state isn't shared across documents. |
| `extract.py` | `extract_markdown()` — marker (PDF/DJVU), ebooklib (EPUB). DJVU is `ddjvu -format=pdf`'d first. |
| `extract_worker.py` | `python -m` JSON-stdin shim that runs one extraction and exits (subprocess isolation for `batch`). |
| `chapters.py` | `Chapter` dataclass; `split_by_pdf_toc` / `split_by_headers`; `select_first_chapter` / `select_content_chapters`; `spoken_chapter_text`; `write_chapters_m4b` (ffmpeg mux). |
| `rewrite.py` | `rewrite_for_audio()` — OpenAI-compatible call to the llama.cpp server; strips citations, expands abbreviations, etc. |
| `server.py` | `ensure_server()` — auto-launch + supervise llama-server (context manager); reuses an existing server on `127.0.0.1:8080`. Binary/model paths are hardcoded (see root README). |
| `paper_metadata.py` | Pull title/author out of the extracted markdown for m4b tags. |
| `parallel.py` | `render_chapters_parallel()` — **serial** chapter render loop with per-chapter `.npy` cache, 3-attempt subprocess retry, thump validation. Name is historical. |
| `tts_worker.py` | `python -m` JSON-stdin shim that renders one chapter in a fresh process (isolates chatterbox crashes). |
| `tts/__init__.py` | Backend registry (`get_backend`/`list_backends`), `chunk_text`, `render_audio` (chunking + per-chunk cache + thump retry). `SAMPLE_RATE = 24000`. |
| `tts/backends/*.py` | One `Backend` subclass per engine: `kokoro`, `chatterbox`, `chatterbox_mojo`, `f5`, `higgs`. |
| `gpu_check.py` | `python -m paper_audiobooks.gpu_check` — verifies torch sees the GPU. |

## Concurrency: there is none, by design

Every stage is **strictly sequential** — one marker subprocess, one chapter
rewrite at a time, one TTS child at a time. `--tts-workers` and
`--extract-workers` are accepted but **ignored** (back-compat only). Concurrent
TTS on AMD gfx1151 crashes (MIOpen "unspecified launch failure", child stdout
death, sampling-rate collapse). Do not reintroduce concurrency.

The one apparent exception: the **`chatterbox-mojo`** backend's `synthesize_many`
runs a small `WorkerPool` (default 2 child processes) for *cross-chunk*
parallelism inside a single chapter — verified safe; 3+ workers OOM the GPU
pool. Set `CHATTERBOX_MOJO_WORKERS` to change.

## Caching & resumability

Three cache layers, all on disk — kill and re-invoke to resume:

1. **Extract** — `output/<stem>.md`. Skipped if present (`--skip-extract`).
2. **Rewrite** — `output/<stem>.chapters.json`. Skipped if present (`--skip-rewrite`).
3. **TTS** — two tiers under `output/.tts-cache/<stem>/`:
   - per-chapter `chapter-NNNN.npy` (in `parallel.py`)
   - per-chunk `chunk-<hash>.npy` (in `render_audio`, keyed on backend+voice+text)

Cache entries are **thump-validated on load** (`_detect_thumps`): clipped/
oscillating audio is discarded and re-rendered. Changing render params
(cfg_weight, temperature, …) does *not* auto-invalidate — delete the cache dir.

## TTS backends

Registered lazily (missing optional deps are skipped silently). `info.name`:

- **`chatterbox`** (default) — Resemble AI, voice cloning, ~500M.
- **`chatterbox-mojo`** — the Mojo rewrite at `../chatterbox-rewrite/max-impl`,
  ~2× faster on gfx1151, 0% WER drift. Spawns a `pixi run python` child over
  JSON-RPC. See `CHATTERBOX_MOJO.md`. Override repo path with
  `CHATTERBOX_MOJO_REPO`.
- **`kokoro`** — fixed voices, fastest, smallest.
- **`f5`**, **`higgs`** — flow-matching / audiobook-trained voice clones.

Pick with `--tts-backend` or `PAPER_AUDIOBOOKS_BACKEND`. Voice resolution
(`_resolve_voice`): explicit `--voice` → `PAPER_AUDIOBOOKS_VOICE` wav (clone
backends only) → backend's built-in default. Default clone voice lives at
`~/.config/paper-audiobooks/default-voice.wav`.

## The "thump" problem (chatterbox)

bf16 + s3gen CFM occasionally oscillates to ±1.0 and HiFT clips it to ±0.99 —
audible loud thumps on ~1% of chunks. Mitigation is layered:

- `_detect_thumps()` flags it (huge sample-jumps and/or %clipped over threshold).
- Bad chunks are re-synthesized with **perturbed RNG seeds** (≈5% of seeds for
  an affected chunk thump; a few retries clear almost all). `CHATTERBOX_THUMP_RETRIES`.
- Caches reject thumpy audio on load so a bad render can't be reused.

Separately, `CHATTERBOX_HALT_ON_ANOMALY=1` enables the *muffled-output*
detector with `MAX_ANOMALY_RETRIES` retries and an artifact dump on final
failure (different bug from thumps). Background on both: `CHATTERBOX_DEBUG.md`.

## Corpus subsystem (`corpus` command — no audio)

`paper-audiobooks corpus <files...>` is the data-gathering half of the
pipeline, split out from TTS. For each document it reuses the *same* extraction
(`stage_extract_subproc`, cached `.md`) and chaptering (`split_by_pdf_toc` /
`split_by_headers` → `select_content_chapters`) as the audio path, then:

1. Scrapes local metadata (`paper_metadata.extract_paper_metadata`) + ISBNs/DOIs
   (`metadata_lookup.extract_isbns` / `extract_dois`) from the text.
2. Enriches online (`metadata_lookup.enrich`): Open Library + Google Books by
   ISBN (subjects = the recommendation signal), Crossref by DOI, Google Books
   title search as last resort. All best-effort — a failed lookup never raises.
3. Optional `--llm-title-fallback`: when the scraped title `looks_like_bad_title`
   *and* online lookup found none, the local llama.cpp server reads the real
   title/author from the document head. Auto-starts llama-server (`ensure_server`).
4. Writes a full-content record (`CorpusRecord`) — `title_source` is one of
   `enriched` / `local` / `llm`.

Records are keyed by `content_sha256` of the extracted markdown + a
`CORPUS_SCHEMA_VERSION`; `is_record_fresh()` skips unchanged docs (use `--force`
to rebuild). `rebuild_index()` regenerates the bodies-stripped `index.jsonl`.

**⚠️ Corpus records hold the ENTIRE text of copyrighted books.** They live under
`output/corpus/` (git-ignored, listed explicitly in `.gitignore`) and **must
never be published or uploaded.** The intended next step is local vector search
over these records for recommendations — keep it local.

Extraction needs the GPU (marker); the metadata/enrich half does not, so the
command only `_assert_gpu()`s when a doc has no cached `.md`.

## Subprocess isolation pattern

Extraction and TTS each run in **fresh `python -m` child processes** fed JSON
on stdin (`extract_worker.py`, `tts_worker.py`). This keeps torch/model state
from leaking between documents and lets a crashed chatterbox child be retried
cleanly. Children use `PR_SET_PDEATHSIG` (`_die_with_parent`) so they're
SIGKILLed if the parent dies — no orphaned GPU processes.

## Conventions

- All audio is float32 at `SAMPLE_RATE = 24000`; backends resample to it.
- Don't silently swallow errors or return placeholder audio — fail loudly
  (the thump/anomaly paths log and retry, but exhaustion is surfaced).
- `git` commit messages in this repo are always literally `"Changes"`.
