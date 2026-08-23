#!/usr/bin/env python3
"""Formatters for ox run records. Kept in one file so the shell never has to
embed Python, which is where every display bug so far came from.

    ox-format.py table          runs.jsonl lines on stdin -> the status table
    ox-format.py show <meta>    one meta.json -> a key/value block
    ox-format.py failed <json>  one .last-run.json -> a one-line why
"""

import json
import os
import sys

COLOR = os.environ.get("OX_COLOR") == "1"


def c(code):
    return code if COLOR else ""


RESET = c("\033[0m")
BOLD = c("\033[1m")
HUES = {"success": "\033[32m", "partial": "\033[33m", "no-changes": "\033[33m"}


def hue(status):
    return c(HUES.get(status, "\033[31m"))


def load_lines(stream):
    rows = []
    for lineno, line in enumerate(stream, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # A corrupt record is itself news; never drop it silently.
            rows.append({
                "run_id": f"<unparseable line {lineno}>",
                "status": "corrupt",
                "reason": str(exc),
            })
    return rows


def short_id(run_id):
    # Run ids are <YYYYMMDD>-<HHMMSS>-<slug>; WHEN already shows the date.
    if len(run_id) > 16 and run_id[8:9] == "-" and run_id[15:16] == "-":
        return run_id[16:]
    return run_id


def table():
    rows = load_lines(sys.stdin)
    print(f"{BOLD}{'WHEN':<17}{'RUN':<26}{'STATUS':<13}{'REPO':<10}"
          f"{'FILES':>5} {'TIME':>7}  BRANCH / REASON{RESET}")
    if not rows:
        print("(no records)")
        return
    for r in reversed(rows):
        status = str(r.get("status", "?"))
        tail = r.get("branch") if r.get("pushed") else ""
        if not tail:
            tail = r.get("reason") or r.get("title") or ""
        when = str(r.get("finished", ""))[:16].replace("T", " ")
        dur = f"{r.get('duration_seconds', 0)}s"
        print(f"{when:<17}{short_id(str(r.get('run_id', '?')))[:25]:<26}"
              f"{hue(status)}{status[:12]:<13}{RESET}"
              f"{str(r.get('repo', ''))[:9]:<10}"
              f"{r.get('files_changed', 0):>5} {dur:>7}  {str(tail)[:58]}")


SHOW_KEYS = [
    "title", "status", "reason", "repo", "base", "branch", "pushed", "commits",
    "files_changed", "insertions", "deletions", "attempt", "max_attempts",
    "duration_seconds", "model", "effort", "sandbox", "codex_exit",
    "started", "finished",
]


def show(path):
    with open(path) as fh:
        d = json.load(fh)
    width = max(len(k) for k in SHOW_KEYS) + 2
    for k in SHOW_KEYS:
        v = d.get(k)
        if v in (None, "") and k not in ("status", "pushed"):
            continue
        if k == "status":
            v = f"{hue(str(v))}{v}{RESET}"
        print(f"{k:<{width}}{v}")


def failed(path):
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    (no readable run record: {exc})")
        return
    print(f"    {d.get('status')}: {d.get('reason')}")
    print(f"    attempt {d.get('attempt')}/{d.get('max_attempts')}"
          f"   run {d.get('run_id')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    mode = sys.argv[1]
    if mode == "table":
        table()
    elif mode == "show":
        show(sys.argv[2])
    elif mode == "failed":
        failed(sys.argv[2])
    else:
        print(f"ox-format: unknown mode {mode!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
