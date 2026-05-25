#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27",
#   "beautifulsoup4>=4.12",
#   "lxml>=5",
# ]
# ///
"""Scrape Partially Examined Life main numbered episodes for linked PDFs.

Output layout under ~/Documents/pel-papers/:
    manifest.json          # {url, episode_url, slug, ep_num, host, tier, sha256, local_path}
    by-episode/<slug>/<filename.pdf>
    by-host/<host>/<filename.pdf>  # symlinks

Run with: ./scrape_pel.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

OUT_ROOT = Path.home() / "Documents" / "pel-papers"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
BY_EPISODE = OUT_ROOT / "by-episode"
LOG_PATH = OUT_ROOT / "scrape.log"

SITEMAP_INDEX = "https://partiallyexaminedlife.com/sitemap.xml"
POST_SITEMAP_RE = re.compile(r"https://partiallyexaminedlife\.com/post-sitemap\d*\.xml")
EPISODE_URL_RE = re.compile(r"^https://partiallyexaminedlife\.com/\d{4}/\d{2}/\d{2}/([^/]+)/?$")
# Main numbered episode slug patterns. Older episodes used "episode-N" or
# "topic-for-N"; the modern form is "ep-NNN-".
EPISODE_SLUG_RE = re.compile(
    r"^(?:ep-?\d+|episode-\d+|topic-for-\d+)\b",
    re.IGNORECASE,
)
EP_NUM_RE = re.compile(r"\b(?:ep-?|episode-|topic-for-)(\d+)", re.IGNORECASE)

USER_AGENT = "bookshelf-cloud-pel-scraper/0.1 (https://github.com/jimmyhmiller; personal use)"
DELAY_PEL_SECONDS = 1.0       # main site
DELAY_OTHER_SECONDS = 1.5     # everything else, per-host
HTTP_TIMEOUT = 60.0


@dataclass
class PdfEntry:
    url: str
    episode_url: str
    slug: str
    ep_num: int | None
    host: str
    tier: int
    local_path: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    error: str | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def host_of(url: str) -> str:
    return urllib.parse.urlparse(url).hostname or ""


def trust_tier(url: str) -> int:
    h = host_of(url).lower()
    if h == "partiallyexaminedlife.com" or h.endswith(".archive.org") or h == "archive.org":
        return 1
    if h.endswith(".edu") or h == "philpapers.org" or "press" in h:
        return 2
    return 3


def fetch_sitemap_index(client: httpx.Client) -> list[str]:
    r = client.get(SITEMAP_INDEX)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")
            if POST_SITEMAP_RE.match(loc.text.strip())]


def fetch_post_urls(client: httpx.Client, post_sitemap: str) -> list[str]:
    r = client.get(post_sitemap)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]


def is_main_episode(url: str) -> tuple[bool, str | None, int | None]:
    m = EPISODE_URL_RE.match(url)
    if not m:
        return False, None, None
    slug = m.group(1)
    if slug.endswith("-citizen"):
        return False, None, None
    if not EPISODE_SLUG_RE.match(slug):
        return False, None, None
    # Drop spinoffs
    if slug.startswith(("pmp", "pvi", "closereads-")):
        return False, None, None
    num_match = EP_NUM_RE.match(slug)
    ep_num = int(num_match.group(1)) if num_match else None
    return True, slug, ep_num


def extract_pdf_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    # Restrict to article body if present — skips header/footer/sidebar links.
    container = soup.find("article") or soup.find("main") or soup
    out: set[str] = set()
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        absurl = urllib.parse.urljoin(base_url, href)
        # Strip fragment; keep query string (some hosts use ?download=1).
        absurl = urllib.parse.urldefrag(absurl)[0]
        # Match .pdf in path (ignore query string for the match).
        path = urllib.parse.urlparse(absurl).path
        if path.lower().endswith(".pdf"):
            out.add(absurl)
    return out


def safe_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = urllib.parse.unquote(path.rsplit("/", 1)[-1])
    # Sanitize
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    # Cap length
    if len(name) > 200:
        stem, _, _ = name.rpartition(".pdf")
        name = stem[:196] + ".pdf"
    return name


def per_host_delay(last_seen: dict[str, float], host: str) -> None:
    delay = DELAY_PEL_SECONDS if host == "partiallyexaminedlife.com" else DELAY_OTHER_SECONDS
    last = last_seen.get(host, 0.0)
    sleep = (last + delay) - time.time()
    if sleep > 0:
        time.sleep(sleep)
    last_seen[host] = time.time()


def download(client: httpx.Client, url: str, dest: Path, last_seen: dict[str, float]) -> tuple[int, str]:
    per_host_delay(last_seen, host_of(url))
    with client.stream("GET", url, follow_redirects=True) as r:
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
        tmp.rename(dest)
    return total, sha.hexdigest()


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(entries: list[PdfEntry]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps([asdict(e) for e in entries], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many episodes (debug).")
    parser.add_argument("--list-only", action="store_true",
                        help="Discover PDFs but don't download.")
    parser.add_argument("--rediscover", action="store_true",
                        help="Re-fetch episode pages even if manifest exists.")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    BY_EPISODE.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    with httpx.Client(headers=headers, timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        last_seen: dict[str, float] = {}

        log(f"fetching sitemap index {SITEMAP_INDEX}")
        post_sitemaps = fetch_sitemap_index(client)
        log(f"  {len(post_sitemaps)} post sub-sitemaps")

        all_posts: list[str] = []
        for ps in post_sitemaps:
            per_host_delay(last_seen, "partiallyexaminedlife.com")
            urls = fetch_post_urls(client, ps)
            log(f"  {ps}: {len(urls)} URLs")
            all_posts.extend(urls)

        episodes: list[tuple[str, str, int | None]] = []
        for u in all_posts:
            ok, slug, ep_num = is_main_episode(u)
            if ok:
                episodes.append((u, slug, ep_num))
        # Dedupe by slug (some episodes posted in multiple URLs?)
        seen = set()
        deduped = []
        for u, s, n in episodes:
            if s in seen: continue
            seen.add(s)
            deduped.append((u, s, n))
        episodes = sorted(deduped, key=lambda t: (t[2] or 0))
        log(f"main numbered episodes discovered: {len(episodes)}")
        if args.limit:
            episodes = episodes[:args.limit]
            log(f"  (limited to {len(episodes)} for this run)")

        # Re-use previous manifest where possible
        existing = {e["url"]: e for e in load_manifest()} if not args.rediscover else {}
        entries: list[PdfEntry] = []

        for i, (ep_url, slug, ep_num) in enumerate(episodes, 1):
            per_host_delay(last_seen, "partiallyexaminedlife.com")
            try:
                r = client.get(ep_url)
                r.raise_for_status()
            except Exception as e:
                log(f"[{i}/{len(episodes)}] {slug}: FETCH FAIL {e}")
                continue
            pdfs = extract_pdf_links(r.text, ep_url)
            log(f"[{i}/{len(episodes)}] {slug}: {len(pdfs)} pdfs")
            for purl in sorted(pdfs):
                if purl in existing and existing[purl].get("local_path") and not args.rediscover:
                    # Reuse existing record (skip redownload)
                    entries.append(PdfEntry(**existing[purl]))
                    continue
                entry = PdfEntry(
                    url=purl,
                    episode_url=ep_url,
                    slug=slug,
                    ep_num=ep_num,
                    host=host_of(purl),
                    tier=trust_tier(purl),
                )
                entries.append(entry)

        log(f"total PDF candidates: {len(entries)} (unique URLs: {len({e.url for e in entries})})")

        if args.list_only:
            save_manifest(entries)
            log(f"list-only: wrote {MANIFEST_PATH}")
            return 0

        # Download pass
        dl_count = 0
        for entry in entries:
            if entry.local_path and Path(entry.local_path).is_file():
                continue
            dest = BY_EPISODE / entry.slug / safe_filename(entry.url)
            try:
                total, sha = download(client, entry.url, dest, last_seen)
                entry.local_path = str(dest)
                entry.bytes = total
                entry.sha256 = sha
                dl_count += 1
                log(f"  ok  {entry.url} -> {dest.name} ({total/1024:.0f} KB)")
            except Exception as e:
                entry.error = f"{type(e).__name__}: {e}"
                log(f"  err {entry.url}: {entry.error}")
            # Save manifest after each download — resumable
            save_manifest(entries)

        log(f"downloaded {dl_count} new PDFs; manifest at {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
