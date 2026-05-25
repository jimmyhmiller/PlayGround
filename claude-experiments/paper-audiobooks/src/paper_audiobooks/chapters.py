"""Chapter-aware splitting of extracted markdown and M4B assembly."""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
# Real section numbering: "1.", "2.3", "4.1.2", "1)", "Section 4:". Page numbers
# (running-heads) look like "47 Title" — bare integer + space + words — so we
# require either a punctuation suffix after the number OR a multi-level number.
NUMBERED_TITLE_RE = re.compile(
    r"^\s*(?:section\s+)?(?:\d+\.\d+(?:\.\d+)*|\d+\.|\d+\))[.:]?\s+\S",
    re.IGNORECASE,
)

# Section titles to drop entirely (matched case-insensitively against cleaned title).
SKIP_TITLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^references?$",
        r"^bibliograph(y|ical references)$",
        r"^works\s+cited$",
        r"^literature\s+cited$",
        r"^citations$",
        r"^notes?$",
        r"^endnotes?$",
        r"^footnotes?$",
        r"^(\[[^\]]*\]\s*)?index(es|of\s+.+)?$",
        r"^subject\s+index$",
        r"^author\s+index$",
        r"^name\s+index$",
        r"^index\s+of\s+",
        r"^acknowledgements?$",
        r"^acknowledgments?$",
        r"^about\s+the\s+authors?$",
        r"^author\s+biograph(y|ies)$",
        r"^funding$",
        r"^conflict\s+of\s+interest$",
        r"^supplementary\s+material$",
        r"^supporting\s+information$",
        r"^appendix(\s.*)?$",
        r"^abbreviations?$",
    ]
]

# Front-matter chapter titles: skip these when picking the "first real chapter"
# of a book. Distinct from SKIP_TITLE_PATTERNS, which is applied to *all* runs;
# these are only filtered when --first-chapter is requested.
#
# NOTE: "Introduction" is NOT here. A book's introduction is often the first
# substantive content — usually what the user means by "first major thing".
# The distinction is preface/foreword/contents (non-content) vs. introduction
# (content). Editor/translator introductions ARE filtered though, since those
# are about-the-edition material.
FRONT_MATTER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^cover$",
        r"^about\s+this\s+book$",
        r"^preface(\s.*)?$",
        r"^foreword(\s.*)?$",
        r"^contents$",
        r"^table\s+of\s+contents$",
        r"^dedication$",
        r"^epigraph$",
        r"^frontispiece$",
        r"^copyright$",
        r"^title\s+page$",
        r"^half[\s-]title$",
        r"^colophon$",
        r"^list\s+of\s+(figures|tables|illustrations|abbreviations|contributors)$",
        r"^abstract$",
        r"^summary$",
        r"^preliminaries$",
        r"^editor['’]?s?\s+(note|introduction|preface)$",
        r"^translator['’]?s?\s+(note|introduction|preface)$",
        r"^author['’]?s?\s+(note|preface)$",
        r"^note\s+to\s+the\s+(reader|second\s+edition|third\s+edition)$",
        r"^how\s+to\s+(use|read)\s+this\s+book$",
    ]
]

# "Part" / "Book" dividers separate groups of chapters; they have no real body
# of their own, so we don't want to pick one as the "first chapter".
PART_DIVIDER_RE = re.compile(
    r"^\s*(part|book|section)\s+(\d+|[ivxlcdm]+|one|two|three|four|five)\b",
    re.IGNORECASE,
)

# A title that's just a bare number ("1", "2.", "Page 3") is almost always a
# stray page header that marker promoted to H1, not a real chapter title.
BARE_NUMERIC_TITLE_RE = re.compile(r"^\s*(?:page\s+)?\d+\.?\s*$", re.IGNORECASE)

# Dedication: "For Susan", "To my parents", "In memory of X". These are short
# (1-6 words) and almost never have real body content.
DEDICATION_RE = re.compile(
    r"^\s*(?:for|to|in\s+memory\s+of|in\s+memoriam)\s+[A-Z][\w\s,.&'’-]{0,40}$",
    re.IGNORECASE,
)

# An ALL-CAPS short title with no punctuation is almost always an author byline
# or a re-print of the book title on the title page ("DEREK PARFIT", "MORAL
# REALISM"). Real chapter titles either have lowercase letters or are long.
ALL_CAPS_NAME_RE = re.compile(r"^\s*[A-Z][A-Z\s.\-']{1,40}$")

# A title that's JUST a person's full name (first + last, optional middle).
# Catches forewords and intros titled by their author: "Samuel Scheffler",
# "Bradford Skow", "T. M. Scanlon".
#
# Heuristic: 2-3 capitalized words, each at least 4 letters long (rules out
# common short title-case words like "A", "An", "The", "Of", "Why", "Few",
# "New"), with no all-cap words and no punctuation other than middle initials.
# We accept "John Searle" and "Samuel Scheffler" but reject "Mathematical
# Explanation" — even though structurally similar, "Mathematical" is unlikely
# to be a first name. A blocklist of common chapter-title head words handles
# the remaining false positives.
_NON_NAME_HEAD_WORDS = {
    "introduction", "preface", "foreword", "synopsis", "summary",
    "abstract", "contents", "acknowledgments", "acknowledgements",
    "conclusion", "epilogue", "prologue",
    "lecture", "chapter", "part", "book", "section", "appendix",
    "mathematical", "philosophical", "logical", "moral", "ethical",
    "natural", "physical", "scientific", "rational", "empirical",
    "theoretical", "practical", "metaphysical", "epistemic", "semantic",
    "syntactic", "linguistic", "psychological", "social", "political",
    "historical", "personal", "general", "special",
    "modern", "ancient", "classical", "contemporary",
    "more", "another", "further", "additional", "various",
}


def _is_person_name_title(title: str) -> bool:
    t = title.strip()
    parts = t.split()
    if not (2 <= len(parts) <= 3):
        return False
    if parts[0].lower() in _NON_NAME_HEAD_WORDS:
        return False
    for p in parts:
        # Allow middle initials like "M." or "T."
        if re.fullmatch(r"[A-Z]\.?", p):
            continue
        # Real name word: capitalized, all letters, >= 4 chars (enough to look name-y),
        # not all caps, no apostrophes/dashes for now.
        if not re.fullmatch(r"[A-Z][a-z]{3,}", p):
            return False
    return True

# Boilerplate phrases that show up on copyright/imprint pages. If a chapter's
# body has multiple of these, it's the copyright page, not the first chapter.
_COPYRIGHT_MARKERS = (
    "oxford university press",
    "cambridge university press",
    "all rights reserved",
    "library of congress",
    "isbn",
    "published in the united states",
    "first published",
    "copyright ©",
    "no part of this publication may be reproduced",
    "british library cataloguing",
    "printed in the united states of america",
    "printed and bound in",
    "a catalog record for this book",
    "registered trademark",
)


def _is_front_matter(title: str) -> bool:
    if BARE_NUMERIC_TITLE_RE.match(title):
        return True
    if DEDICATION_RE.match(title):
        return True
    if ALL_CAPS_NAME_RE.match(title):
        return True
    return any(p.search(title) for p in FRONT_MATTER_PATTERNS)


def _looks_like_person_byline(title: str, body: str) -> bool:
    """Detect 'guest author byline' chapters: the title is just a person's
    name. Catches things like "Samuel Scheffler" (foreword author) so we don't
    mistake the foreword for the book's first chapter.
    """
    return _is_person_name_title(title)


def _is_part_divider(title: str) -> bool:
    return bool(PART_DIVIDER_RE.match(title))


def _looks_like_copyright_page(body: str, *, threshold: int = 3) -> bool:
    """A body dominated by publisher/imprint boilerplate is the copyright page."""
    lower = body.lower()
    hits = sum(1 for marker in _COPYRIGHT_MARKERS if marker in lower)
    return hits >= threshold

# A line that looks like a bibliography entry: starts with author name(s) or number,
# contains a year in parens or comma-year, often has all-caps surnames or italics.
BIB_LINE_RE = re.compile(
    r"""
    (^\s*\d+\.\s+[A-Z])               # 1. Author...
    | (^\s*\[\d+\]\s+[A-Z])           # [1] Author...
    | (\b\(\d{4}[a-z]?\)\B)           # (1999) or (1999a)
    | (\b[A-Z][a-z]+,\s*[A-Z]\.\s)    # Smith, J. ...
    | (\bvol\.?\s*\d+\b)              # vol. 12
    | (\bpp?\.\s*\d+(–|-)\d+)         # pp. 12-34
    """,
    re.VERBOSE,
)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Markdown link: [text](url) -> text. Handle TOC-style links like [Title](#page-23-0).
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# A trailing standalone page number (digits only, possibly preceded by spaces).
_TRAILING_PAGENUM_RE = re.compile(r"\s+\d+\s*$")
# A TOC-style "header": contains a markdown link to a page anchor and ends with a page number.
_TOC_HEADER_RE = re.compile(r"\]\(#?page[-_]\d", re.IGNORECASE)


def _clean_title(title: str) -> str:
    """Strip HTML tags, markdown links, decorations, and stray whitespace from a header."""
    t = _HTML_TAG_RE.sub("", title).strip()
    # Replace markdown links with their text. Do this before stripping ** etc.
    t = _MD_LINK_RE.sub(lambda m: m.group(1), t)
    for marker in ("**", "__", "*", "_"):
        if t.startswith(marker) and t.endswith(marker) and len(t) > 2 * len(marker):
            t = t[len(marker):-len(marker)].strip()
    # Drop trailing page numbers ("Introduction 3" -> "Introduction").
    t = _TRAILING_PAGENUM_RE.sub("", t).strip()
    return t


def _is_toc_header(raw_title: str) -> bool:
    """A header that's actually a TOC entry (link to #page anchor)."""
    return bool(_TOC_HEADER_RE.search(raw_title))


def _drop_toc_clusters(
    headers: list[tuple[int, int, int, str, str]],
    *,
    min_cluster_size: int = 5,
    max_gap: int = 1500,
) -> list[tuple[int, int, int, str, str]]:
    """Drop runs of headers that are tightly packed (= TOC page).

    Conservative: only drops a cluster if (a) at least 5 headers are within ~1500
    chars of each other AND (b) keeping at most half the headers leaves something
    meaningful (we don't drop if it would empty everything out).
    """
    if len(headers) < min_cluster_size:
        return headers
    in_cluster = [False] * len(headers)
    i = 0
    while i < len(headers):
        j = i
        while j + 1 < len(headers) and headers[j + 1][0] - headers[j][1] < max_gap:
            j += 1
        run_len = j - i + 1
        if run_len >= min_cluster_size:
            for k in range(i, j + 1):
                in_cluster[k] = True
        i = j + 1
    kept = [h for h, c in zip(headers, in_cluster) if not c]
    # Safety: if filtering removed everything, keep originals.
    return kept if kept else headers


def _detect_running_heads(headers: list[tuple[int, int, int, str]]) -> set[str]:
    """Identify titles that are page running-head artifacts.

    A running head looks like "<page-num> <book-title>" or "<page-num> <chapter-title>"
    appearing at the top of every page after the first chapter starts. We detect them by
    finding suffixes that appear repeatedly with different leading numbers.
    """
    from collections import Counter
    suffixes: Counter[str] = Counter()
    for _, _, _, title in headers:
        # If title starts with a number, take the rest as the suffix.
        m = re.match(r"^\s*\d+\s+(.+)$", title)
        if m:
            suffixes[m.group(1).strip().lower()] += 1
    # Anything that appears 3+ times with a leading number is almost certainly a running head.
    return {s for s, n in suffixes.items() if n >= 3}


def _is_skip_title(title: str) -> bool:
    return any(p.search(title) for p in SKIP_TITLE_PATTERNS)


def _looks_like_bibliography(body: str) -> bool:
    """True if a body is dominated by bibliography-shaped lines."""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    bib_hits = sum(1 for ln in lines if BIB_LINE_RE.search(ln))
    return bib_hits / len(lines) > 0.5


@dataclass
class Chapter:
    title: str
    body: str


def split_by_headers(markdown: str, *, min_body_chars: int = 400) -> list[Chapter]:
    """Split markdown into chapters, dropping non-content sections.

    Picks chapter boundaries by inspecting the markdown's actual header
    structure (marker's output is unreliable about H1 vs H3) and selecting
    the level(s) that look like top-level sections — usually those starting
    with section numbering ("1.", "2.1", etc.) or that share the most common
    "real header" level.
    """
    raw_headers = list(HEADER_RE.finditer(markdown))
    if not raw_headers:
        title = _guess_title(markdown) or "Paper"
        return [Chapter(title=title, body=markdown.strip())]

    headers = [
        (m.start(), m.end(), len(m.group(1)), _clean_title(m.group(2)), m.group(2))
        for m in raw_headers
    ]

    # Drop dense clusters of headers (5+ within ~5KB) — these are TOC pages.
    headers = _drop_toc_clusters(headers)

    if not headers:
        title = _guess_title(markdown) or "Paper"
        return [Chapter(title=title, body=markdown.strip())]

    # Drop running-head pollution: titles like "4 Transient Truths", "12 Transient Truths"
    # are page banners, not chapters. A repeated suffix with different leading numbers
    # is the signal.
    running_heads = _detect_running_heads([h[:4] for h in headers])
    if running_heads:
        def _stripped_suffix(title: str) -> str:
            m = re.match(r"^\s*\d+\s+(.+)$", title)
            return m.group(1).strip().lower() if m else ""
        headers = [h for h in headers if _stripped_suffix(h[3]) not in running_heads]

    if not headers:
        title = _guess_title(markdown) or "Paper"
        return [Chapter(title=title, body=markdown.strip())]

    # Drop tuple to original 4-element form for downstream code.
    headers = [h[:4] for h in headers]

    chapter_headers = _select_chapter_headers(headers, total_doc_chars=len(markdown))

    raw: list[Chapter] = []
    if chapter_headers:
        first_start = chapter_headers[0][0]
        preamble = markdown[:first_start].strip()
        if preamble and len(preamble) >= min_body_chars:
            raw.append(Chapter(title="Introduction", body=preamble))
        for i, (_, end, _level, title) in enumerate(chapter_headers):
            next_start = chapter_headers[i + 1][0] if i + 1 < len(chapter_headers) else len(markdown)
            body = markdown[end:next_start].strip()
            if body:
                raw.append(Chapter(title=title, body=body))
    else:
        raw.append(Chapter(title=_guess_title(markdown) or "Paper", body=markdown.strip()))

    chapters: list[Chapter] = []
    for chap in raw:
        if _is_skip_title(chap.title):
            continue
        if _looks_like_bibliography(chap.body):
            continue
        if chapters and len(chap.body) < min_body_chars:
            prev = chapters[-1]
            chapters[-1] = Chapter(title=prev.title, body=f"{prev.body}\n\n{chap.body}")
        else:
            chapters.append(chap)

    chapters = _merge_orphan_subsections(chapters)
    return chapters or [Chapter(title="Paper", body=markdown.strip())]


def split_by_pdf_toc(markdown: str, pdf_path: Path) -> list[Chapter] | None:
    """Split a marker-extracted markdown using the source PDF's outline.

    Books worth audiobook-ifying almost always carry a real TOC in the PDF
    bookmarks. Header-heuristic splitting (split_by_headers) is unreliable
    on these — marker promotes running heads, sub-numbered sections, and
    bare page numbers to the same level as real chapter titles, so the
    split-by-headers output for a 12-chapter book frequently degenerates
    into 4 mega-chapters. The TOC is the ground truth.

    How it works:
      - Read the PDF's outline (top-level entries only).
      - Marker emits per-page anchors as `<span id="page-{N}-{X}"></span>`
        in the markdown body, where N is the 0-indexed PDF page that
        matches `pdfium.OutlineItem.page_index`. We search forward by up
        to 5 pages so chapters whose start page lacks an anchor (figure
        page, section break) still split correctly.
      - The chapter slice runs from one anchor to the next.
      - Front matter, bibliography, and index entries are filtered using
        the same helpers split_by_headers uses.

    Returns None if the PDF has no outline or no body anchors are found
    (in which case the caller should fall back to split_by_headers).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        toc = list(pdf.get_toc())
    except Exception:
        return None
    if not toc:
        return None

    top = [item for item in toc if item.level == 0 and item.page_index is not None]
    if len(top) < 2:
        return None  # one or zero entries — nothing to split on

    # If the L0 entries are mostly book-part dividers ("Part I", "Book Two",
    # ...), the real chapters live one level deeper. Descend in that case.
    # Look at the non-front-matter L0 entries: if more than half are part
    # dividers and L1 entries exist below them, switch to L1.
    contentish_top = [it for it in top if not _is_front_matter(it.title)]
    part_count = sum(1 for it in contentish_top if PART_DIVIDER_RE.match(it.title))
    if contentish_top and part_count >= max(2, len(contentish_top) // 2 + 1):
        l1 = [item for item in toc if item.level == 1 and item.page_index is not None]
        if len(l1) >= 2:
            top = l1

    # Locate each TOC entry's start position in the markdown by searching
    # for the corresponding body anchor. Skip TOC pages, which contain
    # `(#page-N-...)` LINK references rather than `<span id=...>` anchors.
    body_anchor = re.compile(r'<span\s+id="page-(\d+)-\d+"></span>')
    anchor_pos: dict[int, int] = {}
    for m in body_anchor.finditer(markdown):
        page = int(m.group(1))
        if page not in anchor_pos:
            anchor_pos[page] = m.start()

    if not anchor_pos:
        return None

    # Build chapter boundaries: (start_pos, end_pos, title) tuples.
    boundaries: list[tuple[int, str]] = []
    for item in top:
        pi = item.page_index
        # Walk forward until we find an anchor; some pages (figures, blanks)
        # have no anchor at all.
        pos = None
        for offset in range(0, 6):
            pos = anchor_pos.get(pi + offset)
            if pos is not None:
                break
        if pos is None:
            continue
        boundaries.append((pos, item.title.strip()))

    if len(boundaries) < 2:
        return None

    # Sort by position to defend against weird outline orderings, then
    # collapse duplicates that landed on the same anchor (rare; happens
    # when a chapter title and its first subsection both point at the
    # same page).
    boundaries.sort(key=lambda b: b[0])
    deduped: list[tuple[int, str]] = []
    for pos, title in boundaries:
        if deduped and deduped[-1][0] == pos:
            continue
        deduped.append((pos, title))
    boundaries = deduped

    chapters: list[Chapter] = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(markdown)
        body = markdown[start:end].strip()
        if not body:
            continue
        # Apply the same content filters as split_by_headers so we don't
        # synthesize the index / bibliography / contents page.
        if _is_skip_title(title) or _is_front_matter(title):
            continue
        if _looks_like_bibliography(body):
            continue
        chapters.append(Chapter(title=title, body=body))

    return chapters or None


def _top_number(title: str) -> int | None:
    """Top-level number of a numbered title. '1.2.3 ...' -> 1, '2 Foo' -> 2.
    Returns None if the title isn't numbered."""
    m = re.match(r"^\s*(\d+)(?:\.\d+)*", title)
    return int(m.group(1)) if m else None


def _merge_orphan_subsections(chapters: list[Chapter]) -> list[Chapter]:
    """Marker often promotes numbered subsections (1.1, 1.1.1, 1.2, ...) to
    the same header level as their parent chapter title. When that happens
    they appear as sibling chapters, splitting what should be a single
    chapter into many tiny pieces.

    This pass merges runs of numbered sub-chapters back into their parent.
    Heuristic:
      - When a chapter's title starts with a multi-level number (1.x, 2.1.3,
        etc.), it's a subsection of its parent. Merge it into the previous
        chapter.
      - When consecutive chapters all share the same TOP-level number (all
        '1.x' for various x), and the previous non-numbered chapter exists,
        merge them all into that previous chapter.
    """
    if not chapters:
        return chapters
    out: list[Chapter] = []
    for chap in chapters:
        m = re.match(r"^\s*(\d+)\.(\d+)", chap.title)
        is_subsection = bool(m)
        if is_subsection and out:
            prev = out[-1]
            out[-1] = Chapter(
                title=prev.title,
                body=f"{prev.body}\n\n## {chap.title}\n\n{chap.body}",
            )
        else:
            out.append(chap)
    return out


def select_content_chapters(
    chapters: list[Chapter], *, min_body_chars: int = 2000,
) -> list[Chapter]:
    """Keep all substantive chapters, drop front/back matter.

    Filters: front matter (preface/contents/dedication/etc.), part dividers
    (which have no body of their own), copyright pages, guest-author bylines,
    block-quote headers (titles ending in ':'), and tiny stub sections below
    min_body_chars. Returns chapters in document order.
    """
    return [
        c for c in chapters
        if not _is_front_matter(c.title)
        and not _is_part_divider(c.title)
        and not _looks_like_copyright_page(c.body)
        and not _looks_like_person_byline(c.title, c.body)
        and not c.title.rstrip().endswith(":")
        and len(c.body) >= min_body_chars
    ]


def select_first_chapter(chapters: list[Chapter], *, min_body_chars: int = 2000) -> list[Chapter]:
    """Pick the first major chapter of a book — usually Chapter 1 or the
    Introduction, whichever comes first. Skips:
      - front matter (preface, foreword, contents, dedication, etc.)
      - "Part I" / "Book Two" dividers (which have no body of their own)
      - tiny stub sections (< min_body_chars)
    Returns a single-element list so the rest of the pipeline stays shape-stable.
    Falls back to the longest chapter if everything else is filtered out.
    """
    if not chapters:
        return chapters
    candidates = [
        c for c in chapters
        if not _is_front_matter(c.title)
        and not _is_part_divider(c.title)
        and not _looks_like_copyright_page(c.body)
        and not _looks_like_person_byline(c.title, c.body)
        and not c.title.rstrip().endswith(":")  # block-quote headers like "Another passage ends:"
        and len(c.body) >= min_body_chars
    ]
    if candidates:
        return [candidates[0]]
    # Fallback: longest body among non-junk candidates. We relax min_body_chars
    # but keep the title-shape filters — better to pick a short real chapter
    # than a 40K-char block-quote with a "...:" header.
    relaxed = [
        c for c in chapters
        if not _is_front_matter(c.title)
        and not _is_part_divider(c.title)
        and not _looks_like_copyright_page(c.body)
        and not _looks_like_person_byline(c.title, c.body)
        and not c.title.rstrip().endswith(":")
        and c.body.strip()
    ]
    if relaxed:
        return [max(relaxed, key=lambda c: len(c.body))]
    return [max(chapters, key=lambda c: len(c.body))]


def _select_chapter_headers(
    headers: list[tuple[int, int, int, str]],
    *,
    total_doc_chars: int | None = None,
) -> list[tuple[int, int, int, str]]:
    """Pick which headers are chapter-level.

    Books in marker output have wildly inconsistent header levels: real chapter
    titles can come out as H1, H2, H3, or H4 depending on font size; sub-section
    numbers ("1.1.1") often get promoted to the same level as real chapters.
    Picking by header level alone (or by numbering depth alone) leads to wrong
    answers — sub-sub-sections steal chapter status, or chapter-bearing levels
    get missed because chapters aren't H1.

    The robust signal is BODY SIZE between consecutive headers. Real chapters
    have many KB of text between them; sub-sub-sections have a few hundred
    chars at most. So:
      1. For each header level (1..4), compute the median body length between
         consecutive headers at that level (treating headers at deeper levels
         as part of the parent's body).
      2. Pick the level whose median body length is largest AND has at least
         3 candidates AND median is at least min_chapter_body chars.
      3. If no level qualifies (very short paper, scattered headers), fall
         back to all H1+H2 headers.
    """
    if not headers:
        return []

    # Quick path: only one header total → that's the document.
    if len(headers) == 1:
        return list(headers)

    # Try each level from 1..4. We're looking for the level whose typical
    # body size matches what real chapters look like (3000+ chars), with
    # at least 2 candidates — that's the chapter-bearing level.
    #
    # Tiebreaking: prefer the SHALLOWEST level among qualifying ones.
    # Sub-sub-section levels often have one or two large bodies (e.g. an
    # epigraph that spans many pages without another deep header) that
    # would otherwise inflate the median past the real chapter level.
    end_pos = total_doc_chars if total_doc_chars is not None else max(h[1] for h in headers) + 1

    def _bodies(level: int) -> list[int]:
        at_level = [h for h in headers if h[2] == level]
        same_or_shallower = sorted(
            [h for h in headers if h[2] <= level], key=lambda h: h[0],
        )
        bodies = []
        for h in at_level:
            nxt = end_pos
            for cand in same_or_shallower:
                if cand[0] > h[1]:
                    nxt = cand[0]
                    break
            bodies.append(nxt - h[1])
        return bodies

    # First pass: prefer levels with >=2 candidates and median body >= 3000.
    qualifying: list[tuple[int, int, int]] = []  # (level, median, good_titles)
    for level in (1, 2, 3, 4):
        at_level = [h for h in headers if h[2] == level]
        if len(at_level) < 2:
            continue
        bodies = sorted(_bodies(level))
        median = bodies[len(bodies) // 2]
        if median < 3000:
            continue
        good_titles = sum(
            1 for h in at_level
            if not _is_front_matter(h[3])
            and not _is_part_divider(h[3])
            and not h[3].rstrip().endswith(":")
        )
        if good_titles < max(1, len(at_level) // 2):
            continue
        qualifying.append((level, median, good_titles))

    if qualifying:
        # Shallowest wins; ties broken by larger median. EXCEPT: if a shallower
        # level has fewer than 2 good titles (just the doc title or similar),
        # skip to a deeper level with substantive content.
        qualifying.sort(key=lambda x: (x[0], -x[1]))
        for level, median, good_titles in qualifying:
            if good_titles >= 2:
                return [h for h in headers if h[2] == level]
        # Nothing had >= 2 good titles — fall through to looser pass.

    # Fallback: try levels with as few as 1 candidate, but still require the
    # title to look chapter-shaped (not "###" or a quoted-passage tail).
    for level in (1, 2, 3, 4):
        at_level = [h for h in headers if h[2] == level]
        if not at_level:
            continue
        good_titles = sum(
            1 for h in at_level
            if h[3].strip()
            and not _is_front_matter(h[3])
            and not _is_part_divider(h[3])
            and not h[3].rstrip().endswith(":")
            and not re.fullmatch(r"#+", h[3].strip())
        )
        if good_titles == 0:
            continue
        bodies = sorted(_bodies(level))
        median = bodies[len(bodies) // 2]
        if median >= 3000:
            return [h for h in headers if h[2] == level]

    # Fallback: keep H1+H2.
    top_levels = [h for h in headers if h[2] <= 2]
    if len(top_levels) >= 2:
        return top_levels
    return list(headers)


def _guess_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        s = line.strip()
        if s:
            return s.lstrip("# ").strip() or None
    return None


def spoken_chapter_text(chapter: Chapter, index: int, total: int) -> str:
    """Prepend a spoken chapter announcement to the chapter body.

    Index is 1-based. The very first chapter is treated as the title page (no
    "Chapter 1." prefix); every subsequent chapter gets one.
    """
    spoken_title = _spokenize_title(chapter.title)
    if index == 1:
        prefix = f"{spoken_title}."
    else:
        prefix = f"Chapter {index - 1}. {spoken_title}."
    if total == 1:
        return chapter.body
    return f"{prefix}\n\n{chapter.body}"


def _spokenize_title(title: str) -> str:
    """Strip leading numbering like "1.", "2.3", "Section 4:" from a title for speech."""
    t = title.strip().rstrip(".:")
    t = re.sub(r"^\s*(?:section|chapter|part)\s+\d+(\.\d+)*[.:]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*\d+(\.\d+)*[.:]?\s*", "", t)
    return t.strip() or title


def write_chapters_m4b(
    wav_segments: list[tuple[Chapter, np.ndarray]],
    *,
    sample_rate: int,
    out_path: Path,
    bitrate: str = "96k",
    metadata_title: str | None = None,
    metadata_author: str | None = None,
) -> Path:
    """Concatenate WAV segments and encode to M4B with chapter markers."""
    if not wav_segments:
        raise ValueError("no segments")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        combined_wav = tmp / "combined.wav"
        meta_file = tmp / "chapters.ffmeta"

        durations_ms: list[int] = []
        pieces: list[np.ndarray] = []
        for _, audio in wav_segments:
            pieces.append(audio)
            durations_ms.append(int(round(len(audio) / sample_rate * 1000)))
        full = np.concatenate(pieces)
        sf.write(combined_wav, full, sample_rate)

        lines = [";FFMETADATA1"]
        if metadata_title:
            lines.append(f"title={_escape(metadata_title)}")
            # `album` mirrors title — BookPlayer falls back to album for
            # listings if title is missing, and many audiobook players sort
            # by it.
            lines.append(f"album={_escape(metadata_title)}")
        if metadata_author:
            lines.append(f"artist={_escape(metadata_author)}")
            lines.append(f"album_artist={_escape(metadata_author)}")
        cursor_ms = 0
        for (chap, _), dur in zip(wav_segments, durations_ms):
            lines.append("[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={cursor_ms}")
            lines.append(f"END={cursor_ms + dur}")
            lines.append(f"title={_escape(chap.title)}")
            cursor_ms += dur
        meta_file.write_text("\n".join(lines) + "\n")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(combined_wav),
                "-i", str(meta_file),
                "-map_metadata", "1",
                "-codec:a", "aac", "-b:a", bitrate,
                "-movflags", "+faststart",
                "-f", "ipod",
                str(out_path),
            ],
            check=True,
        )
    return out_path


def _escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", "\\\n")
    )
