"""Read m4b metadata via ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chapter:
    id: int
    start: float
    end: float
    title: str


@dataclass
class M4BMetadata:
    title: str
    author: str | None
    narrator: str | None
    duration: float  # seconds
    bit_rate: int | None
    size: int  # bytes
    chapters: list[Chapter]
    cover_jpeg: bytes | None


def _ffprobe_json(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_chapters",
            "-show_streams",
            str(path),
        ],
        capture_output=True, check=True, text=True,
    )
    return json.loads(out.stdout)


def _extract_cover(path: Path) -> bytes | None:
    """Pull embedded cover art out of the m4b as JPEG bytes, if present."""
    try:
        out = subprocess.run(
            [
                "ffmpeg",
                "-v", "quiet",
                "-i", str(path),
                "-an",
                "-vcodec", "copy",
                "-f", "image2pipe",
                "-",
            ],
            capture_output=True, check=True,
        )
        return out.stdout if out.stdout else None
    except subprocess.CalledProcessError:
        return None


def probe(path: Path) -> M4BMetadata:
    data = _ffprobe_json(path)
    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}

    duration = float(fmt.get("duration", 0.0))
    bit_rate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None
    size = path.stat().st_size

    title = tags.get("title") or path.stem
    author = tags.get("artist") or tags.get("album_artist") or tags.get("author")
    narrator = tags.get("composer") or tags.get("narrator")

    chapters = []
    for i, ch in enumerate(data.get("chapters") or []):
        ch_tags = {k.lower(): v for k, v in (ch.get("tags") or {}).items()}
        chapters.append(Chapter(
            id=i,
            start=float(ch["start_time"]),
            end=float(ch["end_time"]),
            title=ch_tags.get("title") or f"Chapter {i + 1}",
        ))

    cover = _extract_cover(path)

    return M4BMetadata(
        title=title,
        author=author,
        narrator=narrator,
        duration=duration,
        bit_rate=bit_rate,
        size=size,
        chapters=chapters,
        cover_jpeg=cover,
    )
