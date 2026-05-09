"""paper-audiobooks CLI: PDF -> audiobook."""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

# Sanity-check torch is on the GPU before doing real work. With rocm7.2
# wheels, gfx1151 (Strix Halo) is supported natively and no override is
# needed — but if someone reinstalls torch from PyPI by accident, we want
# to fail loudly rather than silently fall back to CPU.
def _assert_gpu() -> None:
    import torch
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "torch.cuda.is_available() is False. This machine has an AMD Radeon "
        "8060S (gfx1151). The most likely cause is that the venv has the "
        "PyPI CUDA torch build instead of the rocm7.2 build. Run: "
        "`uv sync --reinstall-package torch --reinstall-package torchvision "
        "--reinstall-package torchaudio` and re-run gpu_check. "
        "See CLAUDE.md for the full story."
    )

import click

from .chapters import (
    Chapter,
    select_content_chapters,
    select_first_chapter,
    split_by_headers,
    split_by_pdf_toc,
)


def _split_chapters(markdown: str, source: Path) -> list[Chapter]:
    """Use the PDF outline when present (TOC is ground truth), otherwise
    fall back to header-heuristic splitting. EPUB/DJVU and outline-less PDFs
    take the heuristic path.
    """
    if source.suffix.lower() == ".pdf":
        toc_chapters = split_by_pdf_toc(markdown, source)
        if toc_chapters:
            click.echo(f"[chapters] using PDF outline: {len(toc_chapters)} chapter(s)")
            return toc_chapters
        click.echo("[chapters] no usable PDF outline; falling back to header heuristics")
    return split_by_headers(markdown)
from .pipeline import (
    announce_chapters,
    paths_for,
    stage_extract,
    stage_extract_subproc,
    stage_rewrite,
    stage_tts,
)
from .server import ensure_server

# User-configurable defaults.
# Env vars override; falls back to the built-in defaults below.
DEFAULT_BACKEND = os.environ.get("PAPER_AUDIOBOOKS_BACKEND", "chatterbox")
DEFAULT_VOICE_REF = Path(
    os.environ.get(
        "PAPER_AUDIOBOOKS_VOICE",
        os.path.expanduser("~/.config/paper-audiobooks/default-voice.wav"),
    )
)


@click.group()
def cli() -> None:
    pass


@cli.command("one")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("--llm-base-url", default="http://127.0.0.1:8080", help="llama.cpp server base URL")
@click.option("--tts-backend", default=DEFAULT_BACKEND, show_default=True,
              help="TTS backend: kokoro, chatterbox, f5, higgs (must be installed)")
@click.option("--voice", default=None, help="Voice id or reference wav path; default = configured user voice")
@click.option("--skip-extract", is_flag=True, help="Reuse existing .md")
@click.option("--skip-rewrite", is_flag=True, help="Reuse existing .chapters.json")
@click.option("--no-auto-llm", is_flag=True, help="Don't start llama-server")
@click.option("--format", "fmt", type=click.Choice(["m4b", "mp3"]), default="m4b")
@click.option("--tts-workers", type=int, default=1, show_default=True,
              help="Ignored — TTS is always sequential. Kept for back-compat.")
@click.option("--first-chapter", is_flag=True,
              help="Only synthesize the first real chapter (skip front matter).")
@click.option("--all-chapters", is_flag=True,
              help="Synthesize every substantive chapter as separate m4b tracks "
                   "(drops preface/contents/etc. but keeps the rest).")
@click.option("--max-pages", type=int, default=None,
              help="Only extract the first N pages (defaults to 60 with --first-chapter).")
def one(source: Path, out_dir: Path, llm_base_url: str, tts_backend: str, voice: str | None,
        skip_extract: bool, skip_rewrite: bool, no_auto_llm: bool,
        fmt: str, tts_workers: int, first_chapter: bool, all_chapters: bool,
        max_pages: int | None) -> None:
    """Process a single PDF / EPUB / DJVU."""
    _assert_gpu()
    if first_chapter and all_chapters:
        raise click.UsageError("--first-chapter and --all-chapters are mutually exclusive")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = paths_for(source, out_dir, fmt)
    voice = _resolve_voice(tts_backend, voice)

    if first_chapter and max_pages is None:
        max_pages = 60
    markdown = stage_extract(paths, skip=skip_extract, max_pages=max_pages)
    raw_chapters = _split_chapters(markdown, paths.source)
    if first_chapter:
        raw_chapters = select_first_chapter(raw_chapters)
        click.echo(f"[first-chapter] selected: {raw_chapters[0].title!r}")
    elif all_chapters:
        before = len(raw_chapters)
        raw_chapters = select_content_chapters(raw_chapters)
        click.echo(f"[all-chapters] kept {len(raw_chapters)}/{before} chapters")
    announce_chapters(raw_chapters)

    need_rewrite = not (skip_rewrite and paths.chapters.exists())
    llm_ctx = (
        ensure_server(llm_base_url, log_path=paths.log)
        if need_rewrite and not no_auto_llm
        else nullcontext(False)
    )

    with llm_ctx as started:
        if need_rewrite:
            click.echo(
                f"[llm] {'started llama-server' if started else 'reusing existing server'} "
                f"at {llm_base_url}"
            )
        rewritten = stage_rewrite(paths, raw_chapters, llm_base_url=llm_base_url, skip=skip_rewrite)

    stage_tts(paths, rewritten, voice=voice, workers=tts_workers, fmt=fmt, backend=tts_backend)


def _resolve_voice(backend_name: str, voice: str | None) -> str:
    """Voice resolution order:
    1. explicit --voice flag
    2. configured DEFAULT_VOICE_REF wav (only used by voice-cloning backends)
    3. backend's built-in default voice
    """
    if voice:
        return voice
    if backend_name in {"chatterbox", "f5", "higgs"} and DEFAULT_VOICE_REF.is_file():
        return str(DEFAULT_VOICE_REF)
    from .tts import get_backend
    return get_backend(backend_name).info.default_voice


@cli.command("backends")
def backends_cmd() -> None:
    """List available TTS backends."""
    from .tts import list_backends
    for info in list_backends():
        click.echo(f"  {info.name:12s}  default-voice={info.default_voice:12s}  {info.description}")


@cli.command("batch")
@click.argument("sources", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("--llm-base-url", default="http://127.0.0.1:8080", help="llama.cpp server base URL")
@click.option("--tts-backend", default=DEFAULT_BACKEND, show_default=True,
              help="TTS backend: kokoro, chatterbox, f5, higgs")
@click.option("--voice", default=None, help="Voice id or reference wav path; default = configured user voice")
@click.option("--skip-existing", is_flag=True, help="Skip papers whose audio already exists")
@click.option("--no-auto-llm", is_flag=True, help="Don't start llama-server")
@click.option("--format", "fmt", type=click.Choice(["m4b", "mp3"]), default="m4b")
@click.option("--tts-workers", type=int, default=1, show_default=True,
              help="Ignored — TTS is always sequential. Kept for back-compat.")
@click.option("--extract-workers", type=int, default=0, show_default=True,
              help="Ignored — extraction is always sequential. Kept for back-compat.")
@click.option("--first-chapter", is_flag=True,
              help="Only synthesize the first real chapter (skip front matter).")
@click.option("--all-chapters", is_flag=True,
              help="Synthesize every substantive chapter as separate m4b tracks.")
@click.option("--max-pages", type=int, default=None,
              help="Only extract the first N pages (defaults to 60 with --first-chapter).")
def batch(sources: tuple[Path, ...], out_dir: Path, llm_base_url: str, tts_backend: str, voice: str | None,
          skip_existing: bool, no_auto_llm: bool, fmt: str,
          tts_workers: int, extract_workers: int, first_chapter: bool,
          all_chapters: bool, max_pages: int | None) -> None:
    """Process many PDFs/EPUBs/DJVUs strictly sequentially through extract → rewrite → TTS.

    The LLM is started once and shared across all papers. Each paper finishes
    fully before the next one starts — no concurrency anywhere.
    """
    _assert_gpu()
    if first_chapter and all_chapters:
        raise click.UsageError("--first-chapter and --all-chapters are mutually exclusive")
    sources = tuple(sources)
    if not sources:
        click.echo("no source files given")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    voice = _resolve_voice(tts_backend, voice)

    if first_chapter and max_pages is None:
        max_pages = 60

    plans = [paths_for(p, out_dir, fmt) for p in sources]

    if skip_existing:
        before = len(plans)
        plans = [p for p in plans if not p.audio.exists()]
        click.echo(f"[batch] {before - len(plans)} already done; {len(plans)} to process")

    if not plans:
        click.echo("[batch] nothing to do")
        return

    click.echo(f"[batch] {len(plans)} paper(s); LLM at {llm_base_url}; tts_workers={tts_workers} extract_workers={extract_workers}")

    llm_ctx = (
        ensure_server(llm_base_url, log_path=out_dir / "llama-server.log",
                      startup_timeout=600.0)
        if not no_auto_llm
        else nullcontext(False)
    )

    with llm_ctx as started:
        click.echo(f"[llm] {'started' if started else 'reusing'} server")

        for i, paths in enumerate(plans):
            click.echo(f"[batch] === paper {i + 1}/{len(plans)}: {paths.source.name} ===")
            try:
                click.echo(f"[batch] extract {i + 1}/{len(plans)}: {paths.source.name}")
                markdown = stage_extract_subproc(paths, max_pages=max_pages)
            except Exception as exc:
                click.echo(f"[batch] FAILED extract {paths.source.name}: {exc}")
                continue

            try:
                raw_chapters = _split_chapters(markdown, paths.source)
                if first_chapter:
                    raw_chapters = select_first_chapter(raw_chapters)
                    click.echo(f"[first-chapter] selected: {raw_chapters[0].title!r}")
                elif all_chapters:
                    before = len(raw_chapters)
                    raw_chapters = select_content_chapters(raw_chapters)
                    click.echo(f"[all-chapters] kept {len(raw_chapters)}/{before} chapters")
                announce_chapters(raw_chapters)
                rewritten = stage_rewrite(paths, raw_chapters, llm_base_url=llm_base_url)
                stage_tts(paths, rewritten, voice=voice, workers=1, fmt=fmt, backend=tts_backend)
            except Exception as exc:
                click.echo(f"[batch] FAILED rewrite/tts {paths.source.name}: {exc}")
                continue

        click.echo("[batch] all done")


def main() -> None:
    """Entrypoint that auto-routes: `paper-audiobooks foo.pdf` -> `one foo.pdf`."""
    import sys
    argv = sys.argv[1:]
    subcommands = {"one", "batch", "backends", "--help", "-h"}
    if argv and not argv[0].startswith("-") and argv[0] not in subcommands:
        sys.argv = [sys.argv[0], "one", *argv]
    cli()


if __name__ == "__main__":
    main()
