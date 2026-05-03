"""Source document -> layout-aware Markdown.

Dispatches by file extension:
  .pdf  -> marker-pdf (GPU, layout-aware OCR)
  .epub -> ebooklib + html2text (chapter structure is explicit; no OCR needed)
  .djvu -> ddjvu converts to PDF, then marker-pdf

For .epub the `page_range` parameter is reinterpreted as a chapter index
range (epubs don't have pages). For .djvu it acts on the converted PDF.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_markdown(path: Path, *, page_range: list[int] | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, page_range=page_range)
    if suffix == ".epub":
        return _extract_epub(path, chapter_range=page_range)
    if suffix == ".djvu":
        return _extract_djvu(path, page_range=page_range)
    raise ValueError(
        f"unsupported source format: {path.suffix} (path={path}). "
        "Supported: .pdf .epub .djvu"
    )


def _extract_pdf(pdf_path: Path, *, page_range: list[int] | None = None) -> str:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    config = {"page_range": page_range} if page_range else None
    converter = PdfConverter(artifact_dict=create_model_dict(), config=config)
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def _extract_epub(epub_path: Path, *, chapter_range: list[int] | None = None) -> str:
    """EPUB has explicit chapter structure (the spine). Walk it, render each
    chapter's HTML to markdown, prefix with an H1 of the chapter's title.
    `chapter_range`, if given, restricts to those spine indices (0-based).
    """
    from ebooklib import epub
    import html2text

    book = epub.read_epub(str(epub_path))
    h2t = html2text.HTML2Text()
    h2t.body_width = 0
    h2t.ignore_images = True
    h2t.ignore_emphasis = False

    # Build a map from spine item id -> nav title (for nicer chapter headers).
    nav_titles = _collect_epub_nav_titles(book)

    parts: list[str] = []
    spine_ids = [item_id for item_id, _ in book.spine]
    if chapter_range is not None:
        wanted = set(chapter_range)
        spine_ids = [sid for i, sid in enumerate(spine_ids) if i in wanted]

    for sid in spine_ids:
        item = book.get_item_with_id(sid)
        if item is None:
            continue
        html = item.get_content().decode("utf-8", errors="replace")
        body = h2t.handle(html).strip()
        if not body:
            continue
        title = nav_titles.get(sid) or _guess_html_title(html) or item.file_name
        parts.append(f"# {title}\n\n{body}")

    return "\n\n".join(parts)


def _collect_epub_nav_titles(book) -> dict[str, str]:
    """Map spine item id -> human-readable chapter title from the EPUB's
    table of contents. Falls back gracefully when the toc structure is weird.
    """
    titles: dict[str, str] = {}

    def _walk(node) -> None:
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
            return
        href = getattr(node, "href", None)
        title = getattr(node, "title", None)
        if href and title:
            # href can be "chap1.xhtml" or "chap1.xhtml#section". Strip fragment.
            base = href.split("#", 1)[0]
            for item in book.get_items():
                if item.file_name.endswith(base):
                    titles[item.id] = title
                    break

    _walk(book.toc)
    return titles


def _guess_html_title(html: str) -> str | None:
    import re
    for tag in ("h1", "h2", "title"):
        m = re.search(
            rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.IGNORECASE | re.DOTALL,
        )
        if m:
            text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if text:
                return text
    return None


def _extract_djvu(djvu_path: Path, *, page_range: list[int] | None = None) -> str:
    """Convert .djvu -> .pdf via ddjvu, then run the PDF extractor.
    Requires `ddjvu` from djvulibre-bin (apt install djvulibre-bin).
    """
    if shutil.which("ddjvu") is None:
        raise RuntimeError(
            "ddjvu not found on PATH. Install djvulibre-bin: "
            "`sudo apt install -y djvulibre-bin`"
        )
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / (djvu_path.stem + ".pdf")
        # -format=pdf renders the document to PDF preserving page layout.
        # Use -quality=85 for reasonable size; raise to 95 if OCR struggles.
        subprocess.run(
            ["ddjvu", "-format=pdf", "-quality=85", str(djvu_path), str(pdf_path)],
            check=True,
            capture_output=True,
        )
        return _extract_pdf(pdf_path, page_range=page_range)
