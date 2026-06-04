#!/usr/bin/env python
"""At-a-glance status of the S3 corpus build.

Reads the manifest + written records and reports finished / remaining / failed,
plus throughput and ETA. No dependence on the running process — it just inspects
the output tree, so it's safe to run anytime.

    uv run python scripts/corpus_status.py
    uv run python scripts/corpus_status.py --watch     # refresh every 10s
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

OUT_DIR = Path("output")
CORPUS_DIR = OUT_DIR / "corpus"
MANIFEST = CORPUS_DIR / "s3_manifest.json"
BUILD_LOG = CORPUS_DIR / "s3_build.log"
SUPPORTED = {".pdf", ".epub", ".djvu"}


def _proc_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "build_corpus_from_s3"],
                       capture_output=True, text=True)
    # Exclude this status process / pgrep itself.
    pids = [p for p in r.stdout.split() if p]
    return bool(pids)


def _slug_key(key: str) -> str:
    """Mirror build_corpus_from_s3.slug_key: key -> fs-safe relative path."""
    rel = Path(key).with_suffix("")
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_") or "_" for part in rel.parts]
    return str(Path(*parts))


def _manifest_doc_keys() -> list[str]:
    if not MANIFEST.exists():
        return []
    try:
        man = json.loads(MANIFEST.read_text())
    except Exception:
        return []
    return [m["key"] for m in man if Path(m["key"]).suffix.lower() in SUPPORTED]


def _manifest_total() -> int:
    return len(_manifest_doc_keys())


def _records_done() -> int:
    """Count ONLY records that correspond to a manifest key (the S3 run),
    not the older local-file pilot records sitting in corpus/."""
    if not CORPUS_DIR.exists():
        return 0
    return sum(
        1 for key in _manifest_doc_keys()
        if (CORPUS_DIR / (_slug_key(key) + ".json")).exists()
    )


def _log_stats() -> tuple[int, int, list[str], list[str]]:
    """(n_wrote, n_failed, recent_titles, failed_keys) parsed from the build log."""
    if not BUILD_LOG.exists():
        return 0, 0, [], []
    wrote = failed = 0
    titles: list[str] = []
    failed_keys: list[str] = []
    for line in BUILD_LOG.read_text(errors="replace").splitlines():
        if "] wrote " in line:
            wrote += 1
            m = re.search(r"title=('.*?'|\".*?\")", line)
            if m:
                titles.append(m.group(1))
        elif "] FAILED " in line:
            failed += 1
            m = re.search(r"FAILED (\S+)", line)
            if m:
                failed_keys.append(m.group(1))
    return wrote, failed, titles[-5:], failed_keys[-5:]


def _avg_seconds_per_doc() -> float | None:
    """Estimate s/doc from the last '~Ns/doc' rate the builder logged."""
    if not BUILD_LOG.exists():
        return None
    last = None
    for m in re.finditer(r"~(\d+)s/doc", BUILD_LOG.read_text(errors="replace")):
        last = int(m.group(1))
    return float(last) if last else None


def render() -> str:
    total = _manifest_total()
    done = _records_done()
    wrote, failed, titles, failed_keys = _log_stats()
    remaining = max(total - done, 0)
    running = _proc_running()
    rate = _avg_seconds_per_doc()

    lines = []
    pct = (100.0 * done / total) if total else 0.0
    status = "RUNNING" if running else "STOPPED"
    lines.append(f"corpus build: {status}")
    lines.append(f"  finished : {done:>5} / {total}  ({pct:4.1f}%)")
    lines.append(f"  remaining: {remaining:>5}")
    lines.append(f"  failed   : {failed:>5}" + (f"  (last: {failed_keys[-1]})" if failed_keys else ""))
    if rate and remaining:
        eta_h = rate * remaining / 3600.0
        lines.append(f"  ~{rate:.0f}s/doc -> ETA ~{eta_h:.1f}h for the rest")
    # progress bar
    width = 40
    filled = int(width * done / total) if total else 0
    lines.append("  [" + "#" * filled + "-" * (width - filled) + "]")
    if titles:
        lines.append("  recent:")
        for t in titles:
            lines.append(f"    + {t[:70]}")
    if not running and remaining:
        lines.append("")
        lines.append("  not running — resume with:")
        lines.append("    uv run python scripts/build_corpus_from_s3.py "
                     "--bucket jimmyhmiller-bucket --prefix pdfs/")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="refresh every N seconds")
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()
    if not args.watch:
        print(render())
        return 0
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear screen
            print(render())
            print(f"\n  (refreshing every {args.interval}s — Ctrl-C to stop)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
