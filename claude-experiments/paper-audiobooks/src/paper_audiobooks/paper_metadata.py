"""Extract paper-level metadata (title, authors) from marker markdown output.

Marker (the PDF-to-markdown OCR/layout tool) produces markdown that almost
always begins with a `# Title` line and frequently has an author line near
the top. Both are noisy: PDF character-position artifacts insert random
spaces ("JONATH AN LEW IS"), authors come prefixed with "by", "Edited by",
or just appear as ALL-CAPS names, and front matter from publishers
(copyright pages, ISBN data) can sit between the title and authors.

This module:
1. Picks the strongest candidate `# Title` from the first ~50 lines.
2. Scans nearby lines for author signals.
3. Cleans up the spacing damage (collapses single-char gaps inside words).

It's a best-effort heuristic — paper PDFs are too varied for a perfect
parser. Anything we can't confidently extract returns None and the pipeline
falls back to the filename stem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# How many lines from the top of the markdown to consider.
_SCAN_LINES = 60
# Author hint patterns. Match the line that introduces author(s).
_AUTHOR_PREFIX = re.compile(
    r"^\s*\*?\s*(?:by|edited\s+by|author[s]?:|written\s+by)\s+(.+?)\s*\*?\s*$",
    re.IGNORECASE,
)
# Lines that look like a CC-of-many-uppercase-words author block (the PDF
# extraction often capitalizes author names on title pages).
_ALLCAPS_AUTHOR = re.compile(r"^\s*[A-Z][A-Z\s\.\-]{6,}$")
# Garbage we don't want as a title.
_BAD_TITLE_PATTERNS = [
    re.compile(r"^downloaded\s+from", re.IGNORECASE),
    re.compile(r"^\s*chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^contents$", re.IGNORECASE),
    re.compile(r"^abstract$", re.IGNORECASE),
    re.compile(r"^introduction$", re.IGNORECASE),
]


@dataclass
class PaperMetadata:
    title: str | None = None
    author: str | None = None


def _strip_md(s: str) -> str:
    """Remove markdown emphasis and link syntax from a line."""
    s = re.sub(r"\[(.+?)\]\([^)]*\)", r"\1", s)  # [text](url) -> text
    s = re.sub(r"[*_`]+", "", s)
    return s.strip()


def _fix_spacing(s: str) -> str:
    """Collapse the kind of letter-spacing artifact PDF text extraction
    produces: "T I T L E" → "TITLE", "L E W I S" → "LEWIS".

    Only collapses runs of 3+ single letters separated by single spaces.
    Anything else (real words separated by spaces) is left alone.
    """
    def merge(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    # Match: word boundary, then ≥3 single letters separated by single
    # spaces. Captures the whole run so we can squash it.
    s = re.sub(r"\b(?:[A-Za-z](?: |$)){3,}", merge, s)
    # That can leave trailing residue like "LEWIS " with a leading space
    # from the join — clean it up.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _looks_bad(line: str) -> bool:
    for pat in _BAD_TITLE_PATTERNS:
        if pat.match(line):
            return True
    return False


def _is_image_only(line: str) -> bool:
    return bool(re.match(r"^\s*!\[", line))


def _smart_title_case(s: str) -> str:
    """Title-case an ALL-CAPS string but leave already-mixed-case alone."""
    if not s.isupper():
        return s
    small = {"a", "an", "the", "and", "but", "or", "for", "nor", "on",
             "at", "to", "from", "by", "of", "in", "is", "as", "if"}
    words = s.split()
    out = []
    for i, w in enumerate(words):
        # Preserve obvious initialisms (single uppercase letters, acronyms ≤3)
        if len(w) <= 3 and all(c.isupper() or c in ".-?!,:" for c in w) and i not in (0, len(words) - 1):
            out.append(w)
            continue
        lw = w.lower()
        if i not in (0, len(words) - 1) and lw in small:
            out.append(lw)
        else:
            out.append(lw.capitalize())
    return " ".join(out)


def _extract_title(lines: list[str]) -> str | None:
    """First reasonable `# Title` heading in the top of the doc."""
    candidates: list[str] = []
    for raw in lines:
        s = _strip_md(raw)
        if not s or _is_image_only(s):
            continue
        m = re.match(r"^(#{1,3})\s+(.+)$", s)
        if not m:
            continue
        body = m.group(2).strip()
        if _looks_bad(body):
            continue
        # Skip `# By Author` headings — they're author lines, not titles.
        if re.match(r"^(?:by|edited\s+by)\s+", body, re.IGNORECASE):
            continue
        if len(body) < 3 or len(body) > 200:
            continue
        candidates.append(body)
        if m.group(1) == "#":
            return _smart_title_case(_fix_spacing(body).strip())
    return _smart_title_case(_fix_spacing(candidates[0]).strip()) if candidates else None


def _extract_author(lines: list[str], title: str | None) -> str | None:
    """Author block near the top of the doc."""
    # First pass: explicit "by ..." / "Edited by ..." line. Also matches
    # `# By Author` headings — strip the leading # before testing.
    for raw in lines:
        s = _strip_md(raw)
        if not s:
            continue
        s_no_hash = re.sub(r"^#{1,3}\s+", "", s)
        m = _AUTHOR_PREFIX.match(s_no_hash)
        if m:
            return _normalize_authors(m.group(1))

    # Second pass: ALL-CAPS line below the title that isn't itself the title.
    seen_title = title is None
    for raw in lines:
        s = _strip_md(raw)
        if not s:
            continue
        if not seen_title and title and title.lower() in s.lower():
            seen_title = True
            continue
        if not seen_title:
            continue
        if _ALLCAPS_AUTHOR.match(s):
            return _normalize_authors(s)
    return None


def _normalize_authors(s: str) -> str:
    # Split on between-name separators BEFORE collapsing intra-name spacing,
    # so dash-separated authors ("JONATH AN LEW IS - JONG ER IC SCHW ITZGEBEL")
    # don't get glued into one mega-word.
    s = re.sub(r"\s+[-·—–]\s+", "|", s)
    s = re.sub(r"\s*&\s*", "|", s)
    s = re.sub(r"\s+and\s+", "|", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"[|,]", s) if p.strip()]
    cleaned = []
    for p in parts:
        p = _fix_spacing(p)
        if p.isupper():
            p = " ".join(w.capitalize() for w in p.split())
        cleaned.append(p)
    return ", ".join(cleaned).strip()


def extract_paper_metadata(markdown: str) -> PaperMetadata:
    lines = markdown.splitlines()[:_SCAN_LINES]
    title = _extract_title(lines)
    author = _extract_author(lines, title)
    return PaperMetadata(title=title, author=author)
