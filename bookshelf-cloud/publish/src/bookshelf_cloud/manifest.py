"""Generate AudiobookShelf-shaped JSON for the BookPlayer client.

The schemas here are derived from BookPlayer's Swift Codable models in
TortugaPower/BookPlayer (see DESIGN.md). Fields required by those models are
always included; optional fields are populated when we have the data.

CRITICAL: `mediaType: "book"` MUST sit at the top level of each item, not
inside `media`. BookPlayer silently drops items without it.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from .ffprobe import M4BMetadata

LIBRARY_ID = "main"
LIBRARY_NAME = "Audiobooks"


def slugify(name: str) -> str:
    """Stable, URL-safe id derived from a filename stem."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "-", s)


def ping() -> dict[str, Any]:
    return {"success": True}


def library() -> dict[str, Any]:
    """Single canonical library object — reused in /api/libraries and item.libraryId refs."""
    return {
        "id": LIBRARY_ID,
        "name": LIBRARY_NAME,
        "folders": [
            {"id": "fol-main", "fullPath": "/audiobooks", "libraryId": LIBRARY_ID}
        ],
        "displayOrder": 1,
        "icon": "audiobookshelf",
        "mediaType": "book",
        "provider": "audible",
    }


def libraries_response() -> dict[str, Any]:
    return {"libraries": [library()]}


def filterdata_response(items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the /api/libraries/{id}/filterdata response.

    BookPlayer's author/series/genre filter views read this. All six keys
    are required (empty arrays are accepted, but BookPlayer then shows
    nothing in the filter UI). Authors here are aggregated across every
    item in the library — pulled from each item's media.metadata.authors.
    """
    authors: dict[str, str] = {}  # id → name
    series: dict[str, str] = {}
    genres: set[str] = set()
    tags: set[str] = set()
    narrators: set[str] = set()
    languages: set[str] = set()

    for it in items or []:
        meta = (it.get("media") or {}).get("metadata") or {}
        for a in meta.get("authors") or []:
            if a.get("id") and a.get("name"):
                authors[a["id"]] = a["name"]
        for s in meta.get("series") or []:
            if isinstance(s, dict) and s.get("id") and s.get("name"):
                series[s["id"]] = s["name"]
        for g in meta.get("genres") or []:
            if g:
                genres.add(g)
        for t in (it.get("media") or {}).get("tags") or []:
            if t:
                tags.add(t)
        for n in meta.get("narrators") or []:
            if n:
                narrators.add(n)
        lang = meta.get("language")
        if lang:
            languages.add(lang)

    return {
        "authors": [{"id": i, "name": n} for i, n in sorted(authors.items(), key=lambda kv: kv[1].lower())],
        "genres": sorted(genres, key=str.lower),
        "tags": sorted(tags, key=str.lower),
        "series": [{"id": i, "name": n} for i, n in sorted(series.items(), key=lambda kv: kv[1].lower())],
        "narrators": sorted(narrators, key=str.lower),
        "languages": sorted(languages, key=str.lower),
    }


def empty_collections() -> dict[str, Any]:
    return {"results": []}


def empty_search() -> dict[str, Any]:
    return {"book": []}


def _metadata_block(meta: M4BMetadata) -> dict[str, Any]:
    block: dict[str, Any] = {"title": meta.title}
    if meta.author:
        block["authorName"] = meta.author
        block["authors"] = [{"id": f"aut-{slugify(meta.author)}", "name": meta.author}]
    if meta.narrator:
        block["narratorName"] = meta.narrator
        block["narrators"] = [meta.narrator]
    block["series"] = []
    block["genres"] = []
    return block


def list_item(book_id: str, meta: M4BMetadata) -> dict[str, Any]:
    """Compact shape returned in `/api/libraries/{id}/items` results."""
    now_ms = int(time.time() * 1000)
    item: dict[str, Any] = {
        "id": book_id,
        "libraryId": LIBRARY_ID,
        "mediaType": "book",
        "addedAt": now_ms,
        "updatedAt": now_ms,
        "size": meta.size,
        "media": {
            "coverPath": f"/api/items/{book_id}/cover",
            "duration": meta.duration,
            "metadata": _metadata_block(meta),
        },
    }
    return item


def items_response(books: list[tuple[str, M4BMetadata]]) -> dict[str, Any]:
    results = [list_item(bid, m) for bid, m in books]
    return {"results": results, "total": len(results)}


def item_detail(book_id: str, meta: M4BMetadata) -> dict[str, Any]:
    """Full shape returned in `/api/items/{id}?expanded=1`."""
    filename = f"{book_id}.m4b"
    file_meta = {
        "filename": filename,
        "ext": ".m4b",
        "path": f"/audiobooks/{filename}",
        "size": meta.size,
    }
    audio_file: dict[str, Any] = {
        "index": 1,
        "ino": book_id,
        "duration": meta.duration,
        "format": "m4b",
        "metadata": file_meta,
    }
    if meta.bit_rate:
        audio_file["bitRate"] = meta.bit_rate

    return {
        "id": book_id,
        "libraryId": LIBRARY_ID,
        "size": meta.size,
        "libraryFiles": [
            {
                "ino": book_id,
                "fileType": "audio",
                "metadata": file_meta,
            }
        ],
        "media": {
            "coverPath": f"/api/items/{book_id}/cover",
            "duration": meta.duration,
            "size": meta.size,
            "tags": [],
            "metadata": _metadata_block(meta),
            "audioFiles": [audio_file],
            "chapters": [
                {
                    "id": ch.id,
                    "start": ch.start,
                    "end": ch.end,
                    "title": ch.title,
                }
                for ch in meta.chapters
            ],
        },
    }
