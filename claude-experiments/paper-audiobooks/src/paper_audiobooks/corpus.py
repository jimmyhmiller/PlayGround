"""Bulk corpus builder — chapter-split every document and enrich its metadata,
with NO audio synthesis.

This is the data-gathering half of the pipeline, split out from TTS. For each
source document it:

  1. Extracts full markdown (reusing the existing marker/ebooklib path, with
     the same on-disk `<stem>.md` cache the audio pipeline uses).
  2. Splits it into chapters (PDF outline first, header heuristics otherwise —
     identical logic to the audio pipeline, so corpus chapters == audio chapters).
  3. Scrapes local metadata from the document text (title, author, ISBN, DOI).
  4. Enriches online (Open Library + Google Books, Crossref for DOIs) to pull
     subjects/categories/description — the signal we'll embed for vector search.
  5. Writes ONE JSON record per document holding the FULL extracted contents.

The goal is a corpus we vector-search over later for recommendations.

⚠️  PRIVACY / COPYRIGHT: these records contain the *entire* text of each book.
    They are written under `output/corpus/`, which is git-ignored, and MUST NOT
    be published or uploaded anywhere. Treat them as a private local library
    index.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .chapters import (
    Chapter,
    select_content_chapters,
    split_by_headers,
    split_by_pdf_toc,
)
from .metadata_lookup import (
    BookMetadata,
    enrich,
    extract_dois,
    extract_isbns,
    llm_extract_title_author,
    looks_like_bad_title,
)
from .paper_metadata import extract_paper_metadata

# Schema version — bump when the record shape changes so a re-run can detect
# stale records and rebuild them.
CORPUS_SCHEMA_VERSION = 2


def corpus_dir(out_dir: Path) -> Path:
    return out_dir / "corpus"


def record_path(out_dir: Path, source: Path) -> Path:
    return corpus_dir(out_dir) / f"{source.stem}.json"


def index_path(out_dir: Path) -> Path:
    """The flat index: one line per document, metadata only (no bodies)."""
    return corpus_dir(out_dir) / "index.jsonl"


@dataclass
class CorpusChapter:
    title: str
    body: str
    n_chars: int


@dataclass
class CorpusRecord:
    schema_version: int
    source_path: str
    source_name: str
    stem: str
    content_sha256: str           # hash of the extracted markdown (change detection)
    # Local (scraped-from-document) metadata:
    local_title: str | None
    local_author: str | None
    isbns: list[str]
    dois: list[str]
    # Online-enriched metadata (None if every lookup failed / was skipped):
    enriched: dict | None
    # Best-effort resolved fields (enriched wins, falls back to local):
    title: str | None
    authors: list[str]
    subjects: list[str]
    description: str | None
    publish_year: int | None
    title_source: str  # "enriched" | "local" | "llm"
    # Full content:
    n_chapters: int
    total_chars: int
    chapters: list[dict]          # [{title, body, n_chars}, ...]

    def to_index_entry(self) -> dict:
        """A bodies-stripped view for index.jsonl."""
        return {
            "stem": self.stem,
            "source_name": self.source_name,
            "title": self.title,
            "authors": self.authors,
            "subjects": self.subjects,
            "publish_year": self.publish_year,
            "n_chapters": self.n_chapters,
            "total_chars": self.total_chars,
            "isbns": self.isbns,
            "dois": self.dois,
            "has_description": bool(self.description),
            "enriched_source": (self.enriched or {}).get("source"),
            "title_source": self.title_source,
        }


def _split_chapters(markdown: str, source: Path) -> list[Chapter]:
    """Mirror cli._split_chapters: PDF outline is ground truth; else heuristics."""
    if source.suffix.lower() == ".pdf":
        toc_chapters = split_by_pdf_toc(markdown, source)
        if toc_chapters:
            return toc_chapters
    return split_by_headers(markdown)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_record(
    source: Path,
    markdown: str,
    *,
    content_chapters_only: bool = True,
    do_enrich: bool = True,
    enrich_timeout: float = 10.0,
    llm_title_fallback: bool = False,
    llm_base_url: str = "http://127.0.0.1:8080",
) -> CorpusRecord:
    """Build a full corpus record from already-extracted markdown.

    `content_chapters_only`: drop front/back matter (preface, index, etc.) via
    select_content_chapters — the same filter `--all-chapters` uses. Set False
    to keep every split section verbatim.

    `llm_title_fallback`: when the scraped title looks like garbage AND online
    enrichment didn't resolve a real title, ask the local llama.cpp server to
    read the title/author out of the document opening. Requires a running
    llama-server at `llm_base_url`.
    """
    raw_chapters = _split_chapters(markdown, source)
    chapters = (
        select_content_chapters(raw_chapters) if content_chapters_only else raw_chapters
    )
    # If filtering nuked everything (very short paper), keep the raw split.
    if not chapters:
        chapters = raw_chapters

    local = extract_paper_metadata(markdown)
    isbns = extract_isbns(markdown)
    dois = extract_dois(markdown)

    enriched: BookMetadata | None = None
    if do_enrich:
        enriched = enrich(
            title=local.title,
            author=local.author,
            isbns=isbns,
            dois=dois,
            timeout=enrich_timeout,
        )

    # Resolve best-effort fields: enriched wins, fall back to local scrape.
    title = (enriched.title if enriched and enriched.title else None) or local.title
    authors = (
        enriched.authors if enriched and enriched.authors
        else ([local.author] if local.author else [])
    )

    # LLM fallback: only when we have no online-resolved title and the scraped
    # one looks bad. This is where "read the title/author if they don't exist"
    # lives — it rescues OCR'd papers whose heuristic title is a copyright line
    # or running header.
    llm_title_used = False
    enriched_has_title = bool(enriched and enriched.title)
    if llm_title_fallback and not enriched_has_title and looks_like_bad_title(title):
        llm_title, llm_authors = llm_extract_title_author(markdown, base_url=llm_base_url)
        if llm_title:
            title = llm_title
            llm_title_used = True
        if llm_authors and not authors:
            authors = llm_authors
    subjects = enriched.subjects if enriched else []
    description = enriched.description if enriched else None
    publish_year = enriched.publish_year if enriched else None

    if llm_title_used:
        title_source = "llm"
    elif enriched_has_title:
        title_source = "enriched"
    else:
        title_source = "local"

    corpus_chapters = [
        CorpusChapter(title=c.title, body=c.body, n_chars=len(c.body)) for c in chapters
    ]
    total_chars = sum(c.n_chars for c in corpus_chapters)

    return CorpusRecord(
        schema_version=CORPUS_SCHEMA_VERSION,
        source_path=str(source.resolve()),
        source_name=source.name,
        stem=source.stem,
        content_sha256=_sha256(markdown),
        local_title=local.title,
        local_author=local.author,
        isbns=isbns,
        dois=dois,
        enriched=enriched.to_dict() if enriched else None,
        title=title,
        authors=authors,
        subjects=subjects,
        description=description,
        publish_year=publish_year,
        title_source=title_source,
        n_chapters=len(corpus_chapters),
        total_chars=total_chars,
        chapters=[asdict(c) for c in corpus_chapters],
    )


def write_record(out_dir: Path, record: CorpusRecord) -> Path:
    path = record_path(out_dir, Path(record.source_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False))
    tmp.replace(path)
    return path


def load_record(out_dir: Path, source: Path) -> CorpusRecord | None:
    path = record_path(out_dir, source)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CorpusRecord(**data)
    except Exception:
        return None


def is_record_fresh(out_dir: Path, source: Path, markdown: str) -> bool:
    """True if an existing record matches the current schema AND the current
    extracted content (so we can skip re-enriching unchanged documents)."""
    rec = load_record(out_dir, source)
    if rec is None:
        return False
    if rec.schema_version != CORPUS_SCHEMA_VERSION:
        return False
    return rec.content_sha256 == _sha256(markdown)


_INDEX_FIELDS = (
    "stem", "source_name", "title", "authors", "subjects", "publish_year",
    "n_chapters", "total_chars", "isbns", "dois", "title_source",
)


def _index_entry_from_dict(d: dict) -> dict:
    """Bodies-stripped index entry built straight from a record's JSON dict.

    Doesn't reconstruct a CorpusRecord, so it survives schema drift (extra or
    missing fields) instead of silently dropping the record."""
    entry = {k: d.get(k) for k in _INDEX_FIELDS}
    entry["has_description"] = bool(d.get("description"))
    entry["enriched_source"] = (d.get("enriched") or {}).get("source")
    return entry


def rebuild_index(out_dir: Path) -> Path:
    """Regenerate index.jsonl from every record JSON in the corpus tree.

    Records mirror their S3 keys into nested subdirs (output/corpus/pdfs/.../x.json),
    so this walks the whole tree, not just the top level."""
    cdir = corpus_dir(out_dir)
    idx = index_path(out_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for jf in sorted(cdir.rglob("*.json")):
        if jf.name == "index.jsonl" or jf.name.endswith(".json.tmp"):
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if "stem" not in d:
            continue
        entries.append(_index_entry_from_dict(d))
    tmp = idx.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.replace(idx)
    return idx
