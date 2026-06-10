#!/usr/bin/env python3
"""Export the full raw `vercel` dataset from Axiom to newline-delimited JSON.

Design: DISJOINT time windows, no overlap, no dedup, no unique-id assumption.

This data has no unique row id (neither request.id, _sysTime, nor full content
is guaranteed unique), so any dedup-by-key scheme risks silently merging real
events. Instead we walk the retention window one hour at a time with an
exclusive end bound ([start, end)), so every event is fetched exactly once.

The busiest hour holds ~2.4k events, well under a single tabular response's
row capacity, so no within-window paging is needed. After writing, we assert
the emitted line count equals Axiom's own count() of the same window — the
completeness check that makes "no unique id" safe.

Output: events.ndjson (one raw JSON event per line).
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

ORG = "jimmyhmiller-hcux"
DATASET = "vercel"
URL = "https://api.axiom.co/v1/datasets/_apl?format=tabular"
# Retention window. Start before the earliest retained block; Axiom clamps the
# lower bound to whatever it still holds.
START = datetime(2026, 5, 10, tzinfo=timezone.utc)
END = datetime(2026, 6, 11, tzinfo=timezone.utc)
WINDOW = timedelta(hours=1)   # one request per hour; busiest hour ~2.4k rows
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.ndjson")


def load_token():
    with open(os.path.expanduser("~/.zshrc")) as f:
        for line in f:
            m = re.search(r'AXIOM_API_KEY=["\']?([^"\'\s]+)', line)
            if m:
                return m.group(1)
    sys.exit("AXIOM_API_KEY not found in ~/.zshrc")


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def query(token, apl, start, end):
    body = json.dumps({"apl": apl, "startTime": start, "endTime": end}).encode()
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-AXIOM-ORG-ID", ORG)
    req.add_header("Content-Type", "application/json")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            sys.exit(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except urllib.error.URLError:
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
    sys.exit("exhausted retries")


def table_rows(resp):
    """Axiom response -> list of dict rows.

    Empty windows (e.g. before the retention horizon) come back in the legacy
    shape (`matches`, no `tables`); populated windows come back tabular
    (column-major). Handle both.
    """
    tables = resp.get("tables")
    if tables:
        t = tables[0]
        if not t.get("columns"):
            return []
        fields = [f["name"] for f in t["fields"]]
        cols = t["columns"]
        n = len(cols[0])
        return [{fields[j]: cols[j][i] for j in range(len(fields))} for i in range(n)]
    # Legacy shape: each match carries its event under `data`.
    return [m.get("data", m) for m in resp.get("matches", [])]


def window_count(token, start, end):
    resp = query(token, f"['{DATASET}'] | count", start, end)
    tables = resp.get("tables")
    if not tables or not tables[0].get("columns"):
        return 0
    return tables[0]["columns"][0][0]


def main():
    token = load_token()
    total = 0
    with open(OUT, "w") as out:
        cur = START
        while cur < END:
            nxt = min(cur + WINDOW, END)
            s, e = iso(cur), iso(nxt)  # [s, e): end is exclusive in Axiom
            rows = table_rows(query(token, f"['{DATASET}'] | sort by _time asc | limit 50000", s, e))
            for ev in rows:
                out.write(json.dumps(ev, separators=(",", ":")) + "\n")
            total += len(rows)
            # Per-window completeness: did we get every row Axiom counts here?
            expected = window_count(token, s, e)
            flag = "" if expected == len(rows) else f"  !! expected {expected}"
            if len(rows) >= 60000:
                flag += "  !! near row cap — shrink WINDOW"
            print(f"{s} .. {e}: {len(rows)} rows (total {total}){flag}",
                  file=sys.stderr)
            cur = nxt

    grand = window_count(token, iso(START), iso(END))
    status = "OK" if grand == total else "MISMATCH"
    print(f"DONE [{status}]: wrote {total}, Axiom count() = {grand} -> {OUT}",
          file=sys.stderr)
    if grand != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
