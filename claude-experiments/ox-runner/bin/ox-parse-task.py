#!/usr/bin/env python3
"""Parse an ox task markdown file into shell-eval-able assignments.

A task file looks like:

    ---
    repo: coil
    base: master
    branch: ox/my-thing      # optional, auto-generated otherwise
    timeout: 1800            # optional, seconds
    effort: high             # optional: low | high | max
    attempts: 3              # optional, overrides max attempts
    title: Add a --json flag # optional, used for the branch slug
    ---

    Free-form markdown instructions for the agent.

Emits `TASK_<KEY>=<shell-quoted-value>` lines on stdout, plus writes the body
to the path given as the second argument. Exits 2 with a message on stderr if
the frontmatter is malformed or the required `repo` key is missing.
"""

import re
import shlex
import sys

ALLOWED = {
    "repo", "base", "branch", "timeout", "effort", "attempts",
    "title", "model", "profile", "sandbox", "push",
}

# Deliberately permissive on values but strict on keys, so a typo surfaces as a
# hard error instead of being silently ignored.
KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")


def fail(msg):
    print(f"ox-parse-task: {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) != 3:
        fail("usage: ox-parse-task.py <task.md> <body-out>")
    task_path, body_out = sys.argv[1], sys.argv[2]

    try:
        with open(task_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        fail(f"cannot read {task_path}: {exc}")

    meta = {}
    body = text

    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                end = i
                break
        if end is None:
            fail("frontmatter opened with --- but never closed")
        for raw in lines[1:end]:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = KEY_RE.match(stripped)
            if not m:
                fail(f"cannot parse frontmatter line: {raw!r}")
            key = m.group(1).strip().lower().replace("-", "_")
            val = m.group(2).strip().strip('"').strip("'")
            # Strip trailing ` # comment` only when there is whitespace before it.
            val = re.sub(r"\s+#.*$", "", val).strip()
            if key not in ALLOWED:
                fail(f"unknown frontmatter key {key!r}; allowed: {', '.join(sorted(ALLOWED))}")
            meta[key] = val
        body = "\n".join(lines[end + 1:]).strip() + "\n"

    if not meta.get("repo"):
        fail("task is missing the required `repo:` frontmatter key")
    if not body.strip():
        fail("task has an empty body; nothing to instruct the agent with")

    for numeric in ("timeout", "attempts"):
        if numeric in meta:
            if not meta[numeric].isdigit() or int(meta[numeric]) <= 0:
                fail(f"`{numeric}` must be a positive integer, got {meta[numeric]!r}")

    if "effort" in meta and meta["effort"] not in ("low", "high", "max"):
        fail(f"`effort` must be one of low|high|max, got {meta['effort']!r}")

    if "sandbox" in meta and meta["sandbox"] not in (
        "read-only", "workspace-write", "danger-full-access"
    ):
        fail(f"`sandbox` must be read-only|workspace-write|danger-full-access, got {meta['sandbox']!r}")

    if "push" in meta and meta["push"] not in ("true", "false"):
        fail(f"`push` must be true or false, got {meta['push']!r}")

    if not meta.get("title"):
        # First markdown heading, else first non-empty line.
        title = ""
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            title = s.lstrip("#").strip()
            break
        meta["title"] = title[:120] or "task"

    try:
        with open(body_out, "w", encoding="utf-8") as fh:
            fh.write(body)
    except OSError as exc:
        fail(f"cannot write body to {body_out}: {exc}")

    for key, val in sorted(meta.items()):
        print(f"TASK_{key.upper()}={shlex.quote(val)}")


if __name__ == "__main__":
    main()
