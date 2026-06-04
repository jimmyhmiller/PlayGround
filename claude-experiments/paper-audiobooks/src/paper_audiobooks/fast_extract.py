"""Fast text-layer extraction for PDFs that already have embedded text.

~95% of real-world academic PDFs carry a clean embedded text layer. Running
marker's full GPU OCR pipeline on those is wasteful — PyMuPDF pulls the text
in milliseconds. This module:

  1. Probes whether a PDF has a usable embedded text layer.
  2. If so, extracts it with PyMuPDF, emitting the SAME per-page anchors marker
     does (`<span id="page-{N}-0"></span>`) so downstream PDF-outline
     chaptering (`split_by_pdf_toc`) works identically.

Scanned PDFs (no text layer) return None from `extract_text_pdf` so the caller
falls back to marker OCR.

The anchor format must match `chapters.split_by_pdf_toc`'s regex exactly:
    <span\\s+id="page-(\\d+)-\\d+"></span>
where N is the 0-indexed page number (matching pdfium OutlineItem.page_index).
"""
from __future__ import annotations

from pathlib import Path

# A page is considered to have a real text layer if it averages at least this
# many extractable chars over the sampled pages. Scanned pages yield ~0.
_MIN_CHARS_PER_PAGE = 100
_PROBE_PAGES = 5


def has_text_layer(pdf_path: Path) -> bool:
    """True if the PDF has a usable embedded text layer (vs. a scanned image)."""
    try:
        import pymupdf
    except ImportError:
        return False
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception:
        return False
    try:
        n = min(_PROBE_PAGES, len(doc))
        if n == 0:
            return False
        total = 0
        for i in range(n):
            total += len(doc[i].get_text("text"))
        return (total / n) >= _MIN_CHARS_PER_PAGE
    finally:
        doc.close()


def extract_text_pdf(pdf_path: Path, *, page_range: list[int] | None = None) -> str | None:
    """Extract a text-layer PDF to anchored plain text. Returns None if the PDF
    has no usable text layer (caller should fall back to marker OCR).

    Output mirrors marker's anchor convention so split_by_pdf_toc still slices
    chapters correctly: each page's text is prefixed with
    `<span id="page-{idx}-0"></span>` using the 0-indexed page number.
    """
    if not has_text_layer(pdf_path):
        return None
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    try:
        wanted = set(page_range) if page_range else None
        parts: list[str] = []
        for idx, page in enumerate(doc):
            if wanted is not None and idx not in wanted:
                continue
            text = page.get_text("text")
            # Always emit the anchor (even for a near-empty page) so outline
            # entries that point at sparse pages still resolve.
            parts.append(f'<span id="page-{idx}-0"></span>\n{text}')
        return "\n\n".join(parts).strip()
    finally:
        doc.close()


# Headings are detected by font size relative to the document's body size. A
# span is a heading if its size exceeds body_size * this factor. Bold spans get
# a small discount so a bold-but-same-size run still counts.
_HEADING_SIZE_FACTOR = 1.15
_BOLD_FLAG = 16  # pymupdf span flag bit for bold

# Long text PDFs without a PDF outline are the one case where font-based
# heading detection is unreliable (multi-chapter books, dense figures/footers
# mis-detected as headings). Route those to marker, which has a real layout
# model. Short docs (papers) and any doc with an outline take the fast path.
_FAST_PATH_MAX_PAGES_NO_OUTLINE = 50


def _has_outline(pdf_path: Path) -> bool:
    try:
        import pypdfium2 as pdfium
        pf = pdfium.PdfDocument(str(pdf_path))
        top = [t for t in pf.get_toc() if t.level == 0 and t.page_index is not None]
        return len(top) >= 2
    except Exception:
        return False


def _page_count(pdf_path: Path) -> int:
    try:
        import pymupdf
        d = pymupdf.open(str(pdf_path))
        try:
            return len(d)
        finally:
            d.close()
    except Exception:
        return 0


def can_fast_extract(pdf_path: Path) -> bool:
    """True if this PDF should take the fast (no-GPU) structured-text path.

    Criteria: has an embedded text layer AND (has a usable PDF outline OR is
    short enough that font-based heading detection is reliable). Scanned PDFs
    and long outline-less books fall through to marker OCR.
    """
    if pdf_path.suffix.lower() != ".pdf":
        return False
    if not has_text_layer(pdf_path):
        return False
    if _has_outline(pdf_path):
        return True
    return _page_count(pdf_path) <= _FAST_PATH_MAX_PAGES_NO_OUTLINE


def extract_structured_pdf(pdf_path: Path, *, page_range: list[int] | None = None) -> str | None:
    """Extract a text-layer PDF to *structured markdown* — recovering headings
    from font size/weight, so the heuristic chapter splitter and the title
    scraper (which both expect marker-style `#`/`##` headers) work unchanged.
    No OCR, no GPU.

    Returns None if the PDF has no usable text layer (caller falls back to
    marker OCR for scanned docs).

    How headings are found: compute the document's dominant ("body") font size,
    then any line whose largest span exceeds body * _HEADING_SIZE_FACTOR (or is
    bold and clearly larger) becomes a markdown header. The single largest line
    near the top becomes the `# Title`; other large lines become `##`. Page
    anchors are emitted so split_by_pdf_toc still works when an outline exists.
    """
    if not has_text_layer(pdf_path):
        return None
    import pymupdf
    from collections import Counter

    doc = pymupdf.open(str(pdf_path))
    try:
        wanted = set(page_range) if page_range else None

        # Pass 1: find the body font size (the most common rounded span size,
        # weighted by character count so a few huge title chars don't skew it).
        size_chars: Counter[float] = Counter()
        for idx, page in enumerate(doc):
            if wanted is not None and idx not in wanted:
                continue
            for blk in page.get_text("dict").get("blocks", []):
                for ln in blk.get("lines", []):
                    for sp in ln.get("spans", []):
                        t = sp["text"].strip()
                        if t:
                            size_chars[round(sp["size"], 1)] += len(t)
        if not size_chars:
            return None
        body_size = size_chars.most_common(1)[0][0]
        heading_threshold = body_size * _HEADING_SIZE_FACTOR

        # Pass 2: emit markdown. Track the biggest heading size seen so the
        # title (largest, near the top) gets `#` and the rest get `##`.
        out: list[str] = []
        title_emitted = False
        for idx, page in enumerate(doc):
            if wanted is not None and idx not in wanted:
                continue
            out.append(f'<span id="page-{idx}-0"></span>')
            for blk in page.get_text("dict").get("blocks", []):
                for ln in blk.get("lines", []):
                    spans = [sp for sp in ln.get("spans", []) if sp["text"].strip()]
                    if not spans:
                        continue
                    line_text = "".join(sp["text"] for sp in spans).strip()
                    max_size = max(round(sp["size"], 1) for sp in spans)
                    is_bold = any(sp.get("flags", 0) & _BOLD_FLAG for sp in spans)
                    is_heading = max_size >= heading_threshold or (
                        is_bold and max_size >= body_size + 1.0
                    )
                    # Heading lines are short — guard against a whole bold
                    # paragraph being treated as one giant header.
                    if is_heading and len(line_text) <= 200:
                        if not title_emitted and idx < 2 and max_size >= heading_threshold:
                            out.append(f"\n# {line_text}\n")
                            title_emitted = True
                        else:
                            out.append(f"\n## {line_text}\n")
                    else:
                        out.append(line_text)
        return "\n".join(out).strip() or None
    finally:
        doc.close()
