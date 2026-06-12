"""High-level pipeline functions used by both single-paper and batch CLIs."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click
import numpy as np
import soundfile as sf

from .chapters import Chapter, split_by_headers, spoken_chapter_text, write_chapters_m4b
from .extract import extract_markdown
from .paper_metadata import extract_paper_metadata
from .parallel import render_chapters_parallel
from .rewrite import rewrite_for_audio
from .tts import SAMPLE_RATE


@dataclass
class PipelinePaths:
    source: Path  # input document: .pdf / .epub / .djvu
    md: Path
    chapters: Path
    audio: Path
    log: Path


def paths_for(source: Path, out_dir: Path, fmt: str) -> PipelinePaths:
    stem = source.stem
    return PipelinePaths(
        source=source,
        md=out_dir / f"{stem}.md",
        chapters=out_dir / f"{stem}.chapters.json",
        audio=out_dir / f"{stem}.{fmt}",
        log=out_dir / f"{stem}.llama-server.log",
    )


def _repo_output_dir() -> Path:
    """The pipeline repo's `output/` dir, where the corpus lives, regardless of
    the pipeline's -o out_dir. Allow override via $PAPER_AUDIOBOOKS_OUTPUT.
    `pipeline.py` is at <repo>/src/paper_audiobooks/, so the repo root is three
    parents up."""
    import os
    env = os.environ.get("PAPER_AUDIOBOOKS_OUTPUT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "output"


def stage_extract(
    paths: PipelinePaths,
    *,
    skip: bool = False,
    max_pages: int | None = None,
    corpus_out_dir: Path | None = None,
    use_corpus: bool = True,
) -> str:
    if skip and paths.md.exists():
        click.echo(f"[extract] reusing {paths.md}")
        return paths.md.read_text()

    # Corpus reuse: if we already extracted this document's full text into the
    # corpus (the search/index store), rebuild the .md from it instead of
    # re-running the slow extractor. Only when not page-limited (the corpus is
    # always the full document). Falls through to fresh extraction on any miss.
    #
    # The corpus is keyed to the REPO's output dir, which is NOT necessarily the
    # pipeline's -o out_dir (e.g. the watcher writes m4b to ~/audiobooks/m4b).
    # So default the lookup to the repo output dir, not paths.md.parent.
    if use_corpus and max_pages is None:
        from . import corpus as corpus_mod
        cdir = corpus_out_dir if corpus_out_dir is not None else _repo_output_dir()
        record = corpus_mod.find_record_by_stem(cdir, paths.source)
        if record is not None:
            md = corpus_mod.reconstruct_markdown_from_record(record)
            if md:
                title = record.get("title") or paths.source.stem
                click.echo(
                    f"[extract] reusing corpus extraction for {title!r} "
                    f"({len(md)} chars, {record.get('n_chapters')} chapters) "
                    f"-> {paths.md}"
                )
                paths.md.parent.mkdir(parents=True, exist_ok=True)
                paths.md.write_text(md)
                return md

    click.echo(f"[extract] {paths.source} -> {paths.md}"
               + (f" (first {max_pages} pages)" if max_pages else ""))
    page_range = list(range(max_pages)) if max_pages else None
    markdown = extract_markdown(paths.source, page_range=page_range)
    paths.md.parent.mkdir(parents=True, exist_ok=True)
    paths.md.write_text(markdown)
    return markdown


def _die_with_parent() -> None:
    """preexec_fn: SIGKILL this process if the parent dies. Linux only."""
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def stage_extract_subproc(paths: PipelinePaths, *, max_pages: int | None = None) -> str:
    """Extract via a separate subprocess so multiple extractions don't share torch state.

    The worker's full stdout+stderr is streamed to the parent's stderr live
    (so progress bars and error tracebacks are visible) AND tee'd to
    `paths.log.with_suffix('.extract.log')` so the diagnostic survives even
    if the parent's output buffer is truncated. On non-zero exit, the tail
    of the captured log is included in the raised exception.
    """
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    if paths.md.exists():
        return paths.md.read_text()
    paths.md.parent.mkdir(parents=True, exist_ok=True)
    req: dict = {"source_path": str(paths.source), "out_path": str(paths.md)}
    if max_pages:
        req["page_range"] = list(range(max_pages))

    log_path = paths.md.with_suffix(".extract.log")
    captured: list[str] = []
    with open(log_path, "w", buffering=1) as logf:
        proc = _subprocess.Popen(
            [_sys.executable, "-m", "paper_audiobooks.extract_worker"],
            stdin=_subprocess.PIPE,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,  # merge stderr into stdout — tracebacks land here
            text=True,
            bufsize=1,
            preexec_fn=_die_with_parent,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(_json.dumps(req))
        proc.stdin.close()
        # Stream merged stdout+stderr line-by-line, mirror to log + parent stderr.
        for line in proc.stdout:
            _sys.stderr.write(line)
            _sys.stderr.flush()
            logf.write(line)
            captured.append(line)
        rc = proc.wait()
    if rc != 0:
        # Capture the final ~80 lines of output for the exception message —
        # plenty for a Python traceback, not so much that it floods batch logs.
        # Negative rc on POSIX = killed by signal (e.g. -9 = SIGKILL/OOM).
        sig_note = f" (killed by signal {-rc})" if rc < 0 else ""
        tail = "".join(captured[-80:]).rstrip()
        raise RuntimeError(
            f"extract worker failed for {paths.source} rc={rc}{sig_note}.\n"
            f"Full log: {log_path}\n"
            f"--- last lines ---\n{tail}"
        )
    return paths.md.read_text()


def stage_rewrite(
    paths: PipelinePaths,
    raw_chapters: list[Chapter],
    *,
    llm_base_url: str,
    skip: bool = False,
) -> list[Chapter]:
    if skip and paths.chapters.exists():
        click.echo(f"[rewrite] reusing {paths.chapters}")
        return [Chapter(**c) for c in json.loads(paths.chapters.read_text())]

    rewritten: list[Chapter] = []
    for i, chap in enumerate(raw_chapters, 1):
        click.echo(f"[rewrite] {i}/{len(raw_chapters)}: {chap.title}")
        body = rewrite_for_audio(
            f"# {chap.title}\n\n{chap.body}", base_url=llm_base_url,
        )
        rewritten.append(Chapter(title=chap.title, body=body))
    paths.chapters.parent.mkdir(parents=True, exist_ok=True)
    paths.chapters.write_text(json.dumps(
        [{"title": c.title, "body": c.body} for c in rewritten], indent=2,
    ))
    return rewritten


def stage_tts(
    paths: PipelinePaths,
    chapters: list[Chapter],
    *,
    voice: str,
    workers: int,
    fmt: str,
    backend: str = "kokoro",
) -> Path:
    total = len(chapters)
    chapter_texts = [
        (c, spoken_chapter_text(c, i + 1, total)) for i, c in enumerate(chapters)
    ]
    cache_dir = paths.audio.parent / ".tts-cache" / paths.source.stem
    click.echo(
        f"[tts] backend={backend} voice={voice} | synthesizing {total} chapter(s) "
        f"with {workers} worker(s) | cache={cache_dir}"
    )

    completed = {"n": 0, "cached": 0}

    def _on_done(idx: int, chap: Chapter, *, cached: bool = False) -> None:
        completed["n"] += 1
        if cached:
            completed["cached"] += 1
        tag = "cached" if cached else "rendered"
        click.echo(f"[tts] {completed['n']}/{total} ({tag}): {chap.title}")

    segments = render_chapters_parallel(
        chapter_texts, voice=voice, workers=workers, backend=backend,
        cache_dir=cache_dir, progress_cb=_on_done,
    )
    if completed["cached"]:
        click.echo(f"[tts] {completed['cached']}/{total} chapters reused from cache")

    paths.audio.parent.mkdir(parents=True, exist_ok=True)
    fallback_title = paths.source.stem.replace("-", " ").replace("_", " ")
    title = fallback_title
    author: str | None = None
    if paths.md.exists():
        meta = extract_paper_metadata(paths.md.read_text())
        if meta.title:
            title = meta.title
        if meta.author:
            author = meta.author
    click.echo(f"[meta] title={title!r} author={author!r}")
    if fmt == "m4b":
        write_chapters_m4b(
            segments, sample_rate=SAMPLE_RATE, out_path=paths.audio,
            metadata_title=title,
            metadata_author=author,
        )
    else:
        _write_mp3(segments, paths.audio)
    click.echo(f"done: {paths.audio}")
    return paths.audio


def _write_mp3(segments, out_path: Path) -> None:
    full = np.concatenate([audio for _, audio in segments])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(tmp_path, full, SAMPLE_RATE)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
             "-codec:a", "libmp3lame", "-b:a", "96k", str(out_path)],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def announce_chapters(chapters: list[Chapter]) -> None:
    click.echo(f"[chapters] {len(chapters)} section(s): "
               + ", ".join(c.title[:40] for c in chapters[:6])
               + (" ..." if len(chapters) > 6 else ""))
