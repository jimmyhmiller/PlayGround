#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27",
# ]
# ///
"""Recover broken PDF links from the PEL scrape via the Wayback Machine.

Reads ~/Documents/pel-papers/manifest.json. For every entry with an `error`
field set (i.e. the original URL 404'd or otherwise failed), query the
Wayback CDX API for archived copies of that URL with statuscode=200 and
mimetype=application/pdf. If found, download the original bytes via the
`id_` flag (raw capture, no HTML wrapper) and update the manifest in place.

Run with: ./wayback_recover.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path.home() / "Documents" / "pel-papers"
MANIFEST = ROOT / "manifest.json"
BY_EPISODE = ROOT / "by-episode"
LOG_PATH = ROOT / "wayback.log"

USER_AGENT = "bookshelf-cloud-pel-scraper/0.1 (jimmyhmiller@gmail.com)"
CDX_URL = "https://web.archive.org/cdx/search/cdx"
CDX_DELAY = 4.0   # ~15 req/min cap; play it safe
DL_DELAY = 0.5    # web.archive.org/web/* fetches
HTTP_TIMEOUT = 90.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def find_snapshot(client: httpx.Client, url: str) -> tuple[str, str] | None:
    """Query CDX for an archived PDF capture. Returns (timestamp, original_url)
    of the OLDEST 200-status PDF snapshot, or None if no match."""
    params = {
        "url": url,
        "output": "json",
        "filter": ["statuscode:200", "mimetype:application/pdf"],
        "limit": "1",          # oldest first (CDX default ordering)
    }
    try:
        r = client.get(CDX_URL, params=params)
    except Exception as e:
        log(f"  cdx error for {url}: {e}")
        return None
    if r.status_code == 429:
        log("  cdx 429 (rate limit) — backing off 30s")
        time.sleep(30)
        return find_snapshot(client, url)
    if r.status_code != 200:
        log(f"  cdx http {r.status_code} for {url}")
        return None
    try:
        rows = r.json()
    except Exception:
        return None
    # First row is header; second onward are results.
    if len(rows) < 2:
        return None
    row = rows[1]
    # [urlkey, timestamp, original, mimetype, statuscode, digest, length]
    return row[1], row[2]


def safe_filename(url: str) -> str:
    import re
    path = urllib.parse.urlparse(url).path
    name = urllib.parse.unquote(path.rsplit("/", 1)[-1])
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    if len(name) > 200:
        stem, _, _ = name.rpartition(".pdf")
        name = stem[:196] + ".pdf"
    return name


def download(client: httpx.Client, timestamp: str, orig_url: str, dest: Path) -> tuple[int, str]:
    snap = f"https://web.archive.org/web/{timestamp}id_/{orig_url}"
    with client.stream("GET", snap, follow_redirects=True) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256()
        total = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                sha.update(chunk)
                total += len(chunk)
        # Quick sanity check: PDFs start with %PDF
        with tmp.open("rb") as f:
            magic = f.read(4)
        if magic != b"%PDF":
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"not a PDF (magic={magic!r})")
        tmp.rename(dest)
    return total, sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = json.loads(MANIFEST.read_text())
    broken = [e for e in entries if e.get("error") and not e.get("local_path")]
    log(f"manifest has {len(entries)} entries; {len(broken)} are broken")

    if args.limit:
        broken = broken[:args.limit]
        log(f"  limited to {len(broken)} for this run")

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    recovered = 0
    still_broken = 0
    with httpx.Client(headers=headers, timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        last_cdx = 0.0
        for i, entry in enumerate(broken, 1):
            url = entry["url"]
            log(f"[{i}/{len(broken)}] {url}")

            # CDX politeness
            sleep = (last_cdx + CDX_DELAY) - time.time()
            if sleep > 0:
                time.sleep(sleep)
            last_cdx = time.time()
            snap = find_snapshot(client, url)
            if not snap:
                log("  no archived copy")
                still_broken += 1
                continue
            timestamp, orig = snap
            log(f"  snapshot found: {timestamp}")

            if args.dry_run:
                recovered += 1
                continue

            dest = BY_EPISODE / entry["slug"] / safe_filename(url)
            time.sleep(DL_DELAY)
            try:
                size, sha = download(client, timestamp, orig, dest)
                entry["local_path"] = str(dest)
                entry["bytes"] = size
                entry["sha256"] = sha
                entry["error"] = None
                entry["wayback_timestamp"] = timestamp
                recovered += 1
                log(f"  ok via wayback: {dest.name} ({size/1024:.0f} KB)")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                entry["wayback_error"] = err
                still_broken += 1
                log(f"  wayback dl failed: {err}")

            # Persist after each entry — resumable
            MANIFEST.write_text(json.dumps(entries, indent=2))

    log(f"DONE: recovered={recovered}  still_broken={still_broken}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
