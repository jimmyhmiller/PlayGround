"""bookshelf-cloud CLI.

Usage:
    bookshelf-cloud publish <m4b>          # add or replace a book
    bookshelf-cloud bootstrap              # write static endpoints (run once)
    bookshelf-cloud list                   # show what's currently published
    bookshelf-cloud remove <book-id>       # delete a book and update manifest

Config is read from `~/.config/bookshelf-cloud/config.json`:
    {"bucket": "bookshelf-cloud-1234567890"}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from . import ffprobe, manifest
from .s3 import Publisher


CONFIG_PATH = Path(
    os.environ.get("BOOKSHELF_CLOUD_CONFIG")
    or Path.home() / ".config" / "bookshelf-cloud" / "config.json"
)


def _load_bucket() -> str:
    if not CONFIG_PATH.exists():
        click.echo(
            f"config not found at {CONFIG_PATH}\n"
            f"Create it with: mkdir -p {CONFIG_PATH.parent} && "
            f"echo '{{\"bucket\": \"...\"}}' > {CONFIG_PATH}",
            err=True,
        )
        sys.exit(2)
    cfg = json.loads(CONFIG_PATH.read_text())
    bucket = cfg.get("bucket")
    if not bucket:
        click.echo(f"`bucket` missing in {CONFIG_PATH}", err=True)
        sys.exit(2)
    return bucket


def _items_key() -> str:
    return f"api/libraries/{manifest.LIBRARY_ID}/items"


def _item_detail_key(book_id: str) -> str:
    return f"api/items/{book_id}"


def _audio_key(book_id: str) -> str:
    return f"books/{book_id}/audio.m4b"


def _cover_key(book_id: str) -> str:
    return f"books/{book_id}/cover.jpg"


def _load_items(pub: Publisher) -> list[dict]:
    body = pub.get_json(_items_key())
    if not body:
        return []
    return list(body.get("results") or [])


def _write_items(pub: Publisher, items: list[dict]) -> None:
    pub.put_json(_items_key(), {"results": items, "total": len(items)})


@click.group()
def main() -> None:
    """bookshelf-cloud CLI."""


@main.command()
def bootstrap() -> None:
    """Write the static endpoint responses that don't depend on books."""
    pub = Publisher(_load_bucket())
    pub.put_json("ping", manifest.ping())
    pub.put_json("api/libraries", manifest.libraries_response())
    pub.put_json(f"api/libraries/{manifest.LIBRARY_ID}/filterdata", manifest.filterdata_response())
    pub.put_json(f"api/libraries/{manifest.LIBRARY_ID}/collections", manifest.empty_collections())
    pub.put_json(f"api/libraries/{manifest.LIBRARY_ID}/search", manifest.empty_search())
    # Initialize empty items list if missing.
    if pub.get_json(_items_key()) is None:
        _write_items(pub, [])
    click.echo("bootstrap complete")


@main.command()
@click.argument("m4b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--id", "book_id", default=None, help="Override slug (defaults to filename stem).")
def publish(m4b: Path, book_id: str | None) -> None:
    """Upload an m4b and splice it into the library manifest."""
    pub = Publisher(_load_bucket())
    bid = book_id or manifest.slugify(m4b.stem)

    click.echo(f"probing {m4b.name}...")
    meta = ffprobe.probe(m4b)
    click.echo(f"  title: {meta.title}")
    click.echo(f"  author: {meta.author}")
    click.echo(f"  duration: {meta.duration:.0f}s ({len(meta.chapters)} chapters)")

    click.echo(f"uploading audio ({meta.size / 1_048_576:.0f} MB) to s3://{pub.bucket}/{_audio_key(bid)}")
    pub.put_file(_audio_key(bid), m4b, "audio/mp4")

    if meta.cover_jpeg:
        click.echo(f"uploading cover to s3://{pub.bucket}/{_cover_key(bid)}")
        pub.put_bytes(_cover_key(bid), meta.cover_jpeg, "image/jpeg")
    else:
        click.echo("  no embedded cover — skipping")

    click.echo(f"writing item detail to s3://{pub.bucket}/{_item_detail_key(bid)}")
    pub.put_json(_item_detail_key(bid), manifest.item_detail(bid, meta))

    items = _load_items(pub)
    items = [it for it in items if it.get("id") != bid]
    items.append(manifest.list_item(bid, meta))
    items.sort(key=lambda it: (it.get("media", {}).get("metadata", {}).get("title") or "").lower())
    _write_items(pub, items)
    _refresh_filterdata(pub, items)
    click.echo(f"library now has {len(items)} book(s). done: {bid}")


def _refresh_filterdata(pub: Publisher, items: list[dict]) -> None:
    """Rebuild the filterdata response from the current item list. This is
    what powers BookPlayer's authors / genres / series filter views."""
    pub.put_json(
        f"api/libraries/{manifest.LIBRARY_ID}/filterdata",
        manifest.filterdata_response(items),
    )


@main.command(name="refresh-filterdata")
def refresh_filterdata() -> None:
    """Rebuild /api/libraries/{id}/filterdata from the current items list.

    Useful if items predate the filterdata-aggregation logic, or if you
    want to fix the authors filter view without republishing every book."""
    pub = Publisher(_load_bucket())
    items = _load_items(pub)
    _refresh_filterdata(pub, items)
    fd = manifest.filterdata_response(items)
    click.echo(f"filterdata: {len(fd['authors'])} authors, "
               f"{len(fd['series'])} series, {len(fd['genres'])} genres, "
               f"{len(fd['narrators'])} narrators, {len(fd['languages'])} languages")


@main.command(name="list")
def list_books() -> None:
    """Print the current library contents."""
    pub = Publisher(_load_bucket())
    items = _load_items(pub)
    if not items:
        click.echo("(library empty)")
        return
    for it in items:
        meta = it.get("media", {}).get("metadata", {})
        click.echo(f"  {it['id']:40s}  {meta.get('title') or '?'}")


@main.command()
@click.argument("book_id")
def remove(book_id: str) -> None:
    """Delete a book's audio, cover, detail JSON, and library entry."""
    pub = Publisher(_load_bucket())
    s3 = pub.s3
    for key in (_audio_key(book_id), _cover_key(book_id), _item_detail_key(book_id)):
        try:
            s3.delete_object(Bucket=pub.bucket, Key=key)
            click.echo(f"deleted s3://{pub.bucket}/{key}")
        except Exception as e:
            click.echo(f"  (skip {key}: {e})")
    items = [it for it in _load_items(pub) if it.get("id") != book_id]
    _write_items(pub, items)
    _refresh_filterdata(pub, items)
    click.echo(f"library now has {len(items)} book(s)")


if __name__ == "__main__":
    main()
