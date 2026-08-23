#!/usr/bin/env python3
"""Emit one run record as JSON, built from OX_M_* environment variables.

Every value reaches this program through the environment rather than through
shell interpolation, so no repo name, branch, or error message can ever be
parsed as code. Both ox-run (normal completion) and ox-daemon (sealing runs
that were killed before they could record themselves) use this.
"""

import json
import os
import sys

BOOL_KEYS = {"retryable", "pushed"}
INT_KEYS = {
    "attempt", "max_attempts", "commits", "files_changed",
    "insertions", "deletions", "duration_seconds",
}

# Stable ordering keeps runs.jsonl readable and diff-friendly.
ORDER = [
    "run_id", "task", "title", "status", "reason", "retryable", "attempt",
    "max_attempts", "repo", "repo_url", "base", "base_sha", "branch", "pushed",
    "commits", "files_changed", "insertions", "deletions", "codex_exit",
    "model", "effort", "sandbox", "started", "finished", "duration_seconds",
    "run_dir",
]

DEFAULTS = {
    "retryable": False, "pushed": False, "attempt": 1, "commits": 0,
    "files_changed": 0, "insertions": 0, "deletions": 0, "duration_seconds": 0,
}


def main():
    out = dict(DEFAULTS)
    for env_key, val in os.environ.items():
        if not env_key.startswith("OX_M_"):
            continue
        key = env_key[len("OX_M_"):]
        if key in BOOL_KEYS:
            out[key] = val == "true"
        elif key in INT_KEYS:
            try:
                out[key] = int(val)
            except ValueError:
                out[key] = 0
        else:
            out[key] = val

    if not out.get("run_id"):
        print("ox-meta: OX_M_run_id is required", file=sys.stderr)
        return 2

    ordered = {k: out[k] for k in ORDER if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    print(json.dumps(ordered, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
