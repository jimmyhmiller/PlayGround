"""Online metadata enrichment for the corpus.

Given whatever we can scrape locally (title, author, and sometimes an ISBN or
DOI lifted out of the document text), look the work up online and pull back
richer, structured metadata — subjects/categories, a description, publication
year, publisher, canonical title/author. That richer metadata is what makes
the corpus useful for vector-search recommendations later.

Sources, in priority order:
  - **Open Library** (openlibrary.org) — free, no key. Great ISBN coverage and
    `subjects` tags, which are gold for recommendation clustering.
  - **Google Books** — free, no key (works keyless at low volume). Better
    descriptions and `categories`; good fallback when Open Library is thin.

Everything here is best-effort and network-tolerant: any lookup that fails or
times out returns None, and the caller keeps the locally-scraped metadata. We
never raise from a lookup — a flaky network shouldn't kill a corpus build.

NOTE on identifiers: this project's corpus is heavy on academic *papers*
(Gettier, Hoare, Dijkstra...) which have DOIs, not ISBNs. We detect and record
DOIs too; Crossref enrichment for DOIs is a clean future addition (see
`lookup_doi` stub) but Open Library / Google Books only cover books, so a
paper with only a DOI just keeps its locally-scraped title/author for now.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Identifier extraction (from raw document text)
# ---------------------------------------------------------------------------

# ISBN-13 (978/979 prefix) or ISBN-10, optionally hyphen/space separated, often
# introduced by "ISBN" on a copyright page. We capture the digit run and
# normalize afterward.
_ISBN_RE = re.compile(
    r"\bISBN(?:-1[03])?\s*:?\s*"
    r"((?:97[89][\s-]?)?(?:\d[\s-]?){9}[\dXx])",
    re.IGNORECASE,
)
# DOI: 10.NNNN/suffix. Stop at whitespace or characters that don't belong in a
# DOI. Trailing punctuation (period, paren, comma) is stripped afterward.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)


def _normalize_isbn(raw: str) -> str | None:
    """Strip separators and validate length (10 or 13 digits, X allowed at end-10)."""
    s = re.sub(r"[\s-]", "", raw).upper()
    if len(s) == 13 and s.isdigit():
        return s
    if len(s) == 10 and re.fullmatch(r"\d{9}[\dX]", s):
        return s
    return None


def extract_isbns(text: str) -> list[str]:
    """All distinct, valid ISBNs found in text (copyright pages list several)."""
    out: list[str] = []
    for m in _ISBN_RE.finditer(text):
        isbn = _normalize_isbn(m.group(1))
        if isbn and isbn not in out:
            out.append(isbn)
    return out


def extract_dois(text: str) -> list[str]:
    out: list[str] = []
    for m in _DOI_RE.finditer(text):
        doi = m.group(1).rstrip(".,);]")
        if doi not in out:
            out.append(doi)
    return out


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

_UA = "paper-audiobooks-corpus/0.1 (personal library indexer)"


def _get_json(url: str, *, timeout: float = 10.0) -> Any | None:
    """GET a URL and parse JSON. Returns None on any failure (never raises)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
        return json.loads(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class BookMetadata:
    """Enriched metadata for one work. All fields optional — we record whatever
    we found and where it came from."""
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)  # the recommendation signal
    description: str | None = None
    publish_year: int | None = None
    publisher: str | None = None
    isbns: list[str] = field(default_factory=list)
    doi: str | None = None
    language: str | None = None
    source: str | None = None  # "openlibrary" | "google_books" | "local" | "merged"
    raw: dict[str, Any] = field(default_factory=dict)  # provenance / debugging

    def is_empty(self) -> bool:
        return not (self.title or self.authors or self.subjects or self.description)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Open Library
# ---------------------------------------------------------------------------

def lookup_isbn_openlibrary(isbn: str, *, timeout: float = 10.0) -> BookMetadata | None:
    """Open Library lookup by ISBN.

    Uses the jscmd=data endpoint, which returns a flatter, friendlier shape
    (subjects as [{name, url}], authors as [{name}]) than the raw /isbn/ record.
    """
    url = (
        "https://openlibrary.org/api/books?"
        + urllib.parse.urlencode({
            "bibkeys": f"ISBN:{isbn}",
            "jscmd": "data",
            "format": "json",
        })
    )
    data = _get_json(url, timeout=timeout)
    if not data:
        return None
    rec = data.get(f"ISBN:{isbn}")
    if not rec:
        return None

    authors = [a.get("name") for a in rec.get("authors", []) if a.get("name")]
    subjects = [s.get("name") for s in rec.get("subjects", []) if s.get("name")]
    year = _year_from(rec.get("publish_date"))
    publishers = [p.get("name") for p in rec.get("publishers", []) if p.get("name")]

    return BookMetadata(
        title=rec.get("title"),
        authors=authors,
        subjects=subjects,
        description=_ol_description(rec),
        publish_year=year,
        publisher=publishers[0] if publishers else None,
        isbns=[isbn],
        source="openlibrary",
        raw={"openlibrary_key": rec.get("key")},
    )


# Library-cataloging boilerplate that shows up in Open Library `notes`. These
# are not real descriptions and are useless (worse: noise) for vector search.
_NOTE_BOILERPLATE = re.compile(
    r"^\s*(includes\s+(bibliographical\s+references|index|"
    r"bibliography)|"
    r"bibliography:|"
    r"originally\s+published|"
    r"reprint\.?\s|"
    r"translation\s+of)",
    re.IGNORECASE,
)


def _ol_description(rec: dict) -> str | None:
    # jscmd=data rarely includes a description; some records have `notes` or
    # `excerpts`. Try excerpts first (often a real blurb).
    excerpts = rec.get("excerpts") or []
    for ex in excerpts:
        text = ex.get("text")
        if text and len(text) > 40:
            return text
    notes = rec.get("notes")
    if isinstance(notes, dict):
        notes = notes.get("value")
    # Skip pure library-cataloging boilerplate ("Includes bibliographical
    # references and index.") — it's noise for recommendation embeddings.
    if isinstance(notes, str) and len(notes) > 40 and not _NOTE_BOILERPLATE.match(notes):
        return notes
    return None


# ---------------------------------------------------------------------------
# Google Books
# ---------------------------------------------------------------------------

def lookup_google_books(
    *, isbn: str | None = None, title: str | None = None,
    author: str | None = None, timeout: float = 10.0,
) -> BookMetadata | None:
    """Google Books volumes search. Prefer ISBN; fall back to title+author.

    Works keyless at low request volume. Returns the first matching volume.
    """
    if isbn:
        q = f"isbn:{isbn}"
    elif title:
        q = f'intitle:{title}'
        if author:
            q += f' inauthor:{author}'
    else:
        return None

    url = (
        "https://www.googleapis.com/books/v1/volumes?"
        + urllib.parse.urlencode({"q": q, "maxResults": 1})
    )
    data = _get_json(url, timeout=timeout)
    if not data or not data.get("items"):
        return None
    vol = data["items"][0].get("volumeInfo", {})

    industry = vol.get("industryIdentifiers", []) or []
    isbns = [
        i["identifier"] for i in industry
        if i.get("type", "").startswith("ISBN") and i.get("identifier")
    ]
    year = _year_from(vol.get("publishedDate"))

    return BookMetadata(
        title=vol.get("title"),
        authors=list(vol.get("authors", []) or []),
        subjects=list(vol.get("categories", []) or []),
        description=vol.get("description"),
        publish_year=year,
        publisher=vol.get("publisher"),
        isbns=isbns or ([isbn] if isbn else []),
        language=vol.get("language"),
        source="google_books",
        raw={"google_books_id": data["items"][0].get("id")},
    )


# ---------------------------------------------------------------------------
# DOI (papers) — Crossref stub for future expansion
# ---------------------------------------------------------------------------

def lookup_doi(doi: str, *, timeout: float = 10.0) -> BookMetadata | None:
    """Crossref lookup by DOI — covers academic papers (which lack ISBNs).

    Pulls title, authors, container (journal), year, and subject tags.
    """
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    data = _get_json(url, timeout=timeout)
    if not data or data.get("status") != "ok":
        return None
    msg = data.get("message", {})
    titles = msg.get("title") or []
    authors = [
        " ".join(p for p in (a.get("given"), a.get("family")) if p)
        for a in msg.get("author", []) or []
    ]
    authors = [a for a in authors if a]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (msg.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    container = msg.get("container-title") or []
    return BookMetadata(
        title=titles[0] if titles else None,
        authors=authors,
        subjects=list(msg.get("subject", []) or []),
        description=msg.get("abstract"),
        publish_year=year,
        publisher=container[0] if container else msg.get("publisher"),
        doi=doi,
        source="crossref",
        raw={"crossref_type": msg.get("type")},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _year_from(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(date_str))
    return int(m.group(1)) if m else None


def _merge(primary: BookMetadata, secondary: BookMetadata) -> BookMetadata:
    """Fill gaps in `primary` from `secondary` (primary wins on conflicts)."""
    merged = BookMetadata(
        title=primary.title or secondary.title,
        authors=primary.authors or secondary.authors,
        subjects=_dedup(primary.subjects + secondary.subjects),
        description=primary.description or secondary.description,
        publish_year=primary.publish_year or secondary.publish_year,
        publisher=primary.publisher or secondary.publisher,
        isbns=_dedup(primary.isbns + secondary.isbns),
        doi=primary.doi or secondary.doi,
        language=primary.language or secondary.language,
        source="merged",
        raw={**secondary.raw, **primary.raw},
    )
    return merged


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = it.strip().lower()
        if it and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out


# ---------------------------------------------------------------------------
# LLM fallback — read title/author out of the document when scraping fails
# ---------------------------------------------------------------------------

_LLM_TITLE_SYSTEM = (
    "You extract bibliographic metadata from the opening of a document. "
    "Given the first part of a paper or book (as markdown, possibly noisy from "
    "OCR), identify its real title and author(s). "
    "Respond with ONLY a JSON object: "
    '{"title": "<title or null>", "authors": ["<name>", ...]}. '
    "Use the work's actual title — not a running header, copyright line, "
    "publisher address, or section heading. If you genuinely cannot tell, use "
    "null for title and an empty list for authors. No prose, no markdown fences."
)


def llm_extract_title_author(
    head_text: str,
    *,
    base_url: str = "http://127.0.0.1:8080",
    model: str = "qwen3",
    timeout: float = 120.0,
) -> tuple[str | None, list[str]]:
    """Ask the local llama.cpp server to read the title/author from a document's
    opening. Returns (title, authors). Best-effort: any failure → (None, []).

    Only the first ~6 KB of the document is sent — the title page / abstract is
    always near the top, and this keeps the prompt small and fast.
    """
    import httpx

    head = head_text[:6000]
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_TITLE_SYSTEM},
            {"role": "user", "content": head},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None, []

    # The model should return bare JSON; tolerate ```json fences and stray prose.
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None, []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, []
    title = obj.get("title")
    if isinstance(title, str):
        title = title.strip() or None
    else:
        title = None
    authors = obj.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    authors = [a.strip() for a in authors if isinstance(a, str) and a.strip()]
    return title, authors


def looks_like_bad_title(title: str | None) -> bool:
    """Heuristic: does this scraped title look like garbage (a copyright line,
    running header, publisher address, OCR fragment) rather than a real title?

    Used to decide whether to spend an LLM call fixing it.
    """
    if not title:
        return True
    t = title.strip()
    if len(t) < 4:
        return True
    low = t.lower()
    bad_markers = (
        "copyright", "notice", "all rights reserved", "research institute",
        "menlo park", "university press", "downloaded from", "introduction",
    )
    if any(b in low for b in bad_markers):
        return True
    # OCR fragment signatures: "I. Int reduction", lone roman-numeral heads,
    # or a title that's mostly non-letters.
    if re.match(r"^[IVXLC]+\.\s", t) and len(t) < 25:
        return True
    letters = sum(c.isalpha() for c in t)
    if letters < len(t) * 0.5:
        return True
    return False


def enrich(
    *,
    title: str | None,
    author: str | None,
    isbns: list[str] | None = None,
    dois: list[str] | None = None,
    timeout: float = 10.0,
) -> BookMetadata | None:
    """Best-effort enrichment from all available signals.

    Strategy:
      1. If we have ISBNs, query Open Library + Google Books by the first ISBN
         and merge (Open Library primary — better subjects; Google Books fills
         description). Try subsequent ISBNs only if the first yields nothing.
      2. If no ISBN but we have a DOI, try Crossref (papers).
      3. If still nothing and we have a title, do a Google Books title+author
         search as a last resort.

    Returns None if every lookup came back empty (caller keeps local metadata).
    """
    isbns = isbns or []
    dois = dois or []

    # 1. ISBN path.
    for isbn in isbns:
        ol = lookup_isbn_openlibrary(isbn, timeout=timeout)
        gb = lookup_google_books(isbn=isbn, timeout=timeout)
        if ol and gb:
            return _merge(ol, gb)
        if ol:
            return ol
        if gb:
            return gb

    # 2. DOI path (papers).
    for doi in dois:
        cr = lookup_doi(doi, timeout=timeout)
        if cr and not cr.is_empty():
            return cr

    # 3. Title/author fallback via Google Books.
    if title:
        gb = lookup_google_books(title=title, author=author, timeout=timeout)
        if gb and not gb.is_empty():
            return gb

    return None
