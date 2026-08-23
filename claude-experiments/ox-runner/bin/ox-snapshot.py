#!/usr/bin/env python3
"""Emit one JSON snapshot of the runner's state.

Used by `ox snapshot` (once) and `ox stream` (on every change), which is what
the jim widget consumes. Everything a monitor needs is in one line: daemon
health, queue depth, in-flight runs with their latest log line, and the recent
finished runs.
"""

import json
import os
import subprocess
import sys
import time

OX_HOME = os.environ.get("OX_HOME", os.path.expanduser("~/ox-runner"))
RUNS = os.path.join(OX_HOME, "runs")
RUNSLOG = os.path.join(OX_HOME, "runs.jsonl")

RECENT_DEFAULT = 12


def count_md(d):
    try:
        return sum(1 for f in os.listdir(d) if f.endswith(".md"))
    except OSError:
        return 0


def daemon_state():
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", "ox-runner.service"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def last_log_line(run_dir):
    """The most recent meaningful line of a live run's rendered log."""
    path = os.path.join(run_dir, "codex.log")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return ""
    for line in reversed(tail):
        s = line.strip()
        # Skip command-output continuation lines; they are rarely informative
        # on their own and often just build spew.
        if not s or s.startswith("|"):
            continue
        return s[:160]
    return ""


def in_flight():
    """Run dirs with no meta.json yet — i.e. still executing."""
    out = []
    try:
        names = os.listdir(RUNS)
    except OSError:
        return out
    for name in names:
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d) or os.path.exists(os.path.join(d, "meta.json")):
            continue
        started = None
        try:
            started = os.path.getmtime(os.path.join(d, "prompt.txt"))
        except OSError:
            try:
                started = os.path.getmtime(d)
            except OSError:
                pass
        repo = title = ""
        task = os.path.join(d, "task.md")
        if os.path.exists(task):
            try:
                with open(task) as fh:
                    for line in fh:
                        if line.startswith("repo:"):
                            repo = line.split(":", 1)[1].strip()
                        elif line.startswith("title:"):
                            title = line.split(":", 1)[1].strip()
                        elif line.strip() == "---" and repo:
                            break
            except OSError:
                pass
        out.append({
            "run_id": name,
            "repo": repo,
            "title": title,
            "elapsed": int(time.time() - started) if started else 0,
            "last": last_log_line(d),
        })
    out.sort(key=lambda r: r["run_id"], reverse=True)
    return out


def queued():
    inbox = os.path.join(OX_HOME, "inbox")
    out = []
    try:
        names = sorted(f for f in os.listdir(inbox) if f.endswith(".md"))
    except OSError:
        return out
    for name in names:
        repo = ""
        try:
            with open(os.path.join(inbox, name)) as fh:
                for line in fh:
                    if line.startswith("repo:"):
                        repo = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        out.append({"task": name, "repo": repo})
    return out


def recent(limit):
    rows = []
    try:
        with open(RUNSLOG) as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({
            "run_id": d.get("run_id", ""),
            "title": d.get("title", ""),
            "status": d.get("status", ""),
            "reason": d.get("reason", ""),
            "repo": d.get("repo", ""),
            "branch": d.get("branch", ""),
            "pushed": bool(d.get("pushed")),
            "files": d.get("files_changed", 0),
            "insertions": d.get("insertions", 0),
            "deletions": d.get("deletions", 0),
            "duration": d.get("duration_seconds", 0),
            "attempt": d.get("attempt", 1),
            "finished": d.get("finished", ""),
        })
    rows.reverse()
    return rows


def snapshot(limit=RECENT_DEFAULT):
    return {
        "daemon": daemon_state(),
        "model": os.environ.get("OX_MODEL", "stealth/ox-alpha"),
        "concurrency": int(os.environ.get("OX_CONCURRENCY", "2") or 2),
        "counts": {
            "waiting": count_md(os.path.join(OX_HOME, "inbox")),
            "running": count_md(os.path.join(OX_HOME, "processing")),
            "done": count_md(os.path.join(OX_HOME, "done")),
            "failed": count_md(os.path.join(OX_HOME, "failed")),
        },
        "queued": queued(),
        "in_flight": in_flight(),
        "recent": recent(limit),
        "ts": int(time.time()),
    }


def main():
    args = sys.argv[1:]
    stream = "--stream" in args
    interval = 2.0
    for i, a in enumerate(args):
        if a == "--interval" and i + 1 < len(args):
            interval = float(args[i + 1])
        if a == "--limit" and i + 1 < len(args):
            globals()["RECENT_DEFAULT"] = int(args[i + 1])

    if not stream:
        print(json.dumps(snapshot(), separators=(",", ":")))
        return 0

    # Emit only on change, plus a heartbeat, so the consumer stays event-driven
    # and an idle runner costs nothing.
    prev = None
    last_beat = 0.0
    while True:
        snap = snapshot()
        # `ts` always differs; compare everything else.
        cmp = dict(snap)
        cmp.pop("ts")
        blob = json.dumps(cmp, sort_keys=True)
        now = time.time()
        if blob != prev or now - last_beat > 20:
            sys.stdout.write(json.dumps(snap, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            prev = blob
            last_beat = now
        time.sleep(interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(0)
