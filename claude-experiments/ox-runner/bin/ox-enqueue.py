#!/usr/bin/env python3
"""Write a queued ox task file from task text on stdin.

    ox enqueue --repo NAME [--base REF] [--title T] [--effort E] [--dry]
        ... task body on stdin ...

Reads until EOF, or until a line that is exactly `__OX_END__`. The sentinel
exists for the jim widget: it feeds stdin down an ssh pipe it has no way to
close, so it marks the end of the body explicitly. From a shell, plain EOF
works and the sentinel never appears.

Prints one JSON line describing what happened, so a UI can report it.
"""

import json
import os
import re
import sys
import time

OX_HOME = os.environ.get("OX_HOME", os.path.expanduser("~/ox-runner"))
REPOS_TSV = os.path.join(OX_HOME, "config", "repos.tsv")
INBOX = os.path.join(OX_HOME, "inbox")

SENTINEL = "__OX_END__"


def out(**kw):
    sys.stdout.write(json.dumps(kw, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def die(msg, **extra):
    out(ok=False, error=msg, **extra)
    sys.exit(1)


def slug(text, limit=48):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:limit] or "task"


def load_repos():
    """name -> default base branch, from the registry."""
    repos = {}
    try:
        with open(REPOS_TSV) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0]:
                    repos[parts[0]] = parts[2] if len(parts) > 2 and parts[2] else "main"
    except OSError:
        pass
    return repos


def read_body():
    lines = []
    for line in sys.stdin:
        if line.rstrip("\n") == SENTINEL:
            break
        lines.append(line)
    return "".join(lines)


def first_line_title(body):
    for line in body.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:100]
    return "task"


def main():
    args = sys.argv[1:]
    repo = base = title = effort = None
    dry = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--repo" and i + 1 < len(args):
            repo = args[i + 1]; i += 1
        elif a == "--base" and i + 1 < len(args):
            base = args[i + 1]; i += 1
        elif a == "--title" and i + 1 < len(args):
            title = args[i + 1]; i += 1
        elif a == "--effort" and i + 1 < len(args):
            effort = args[i + 1]; i += 1
        elif a == "--dry":
            dry = True
        else:
            die(f"unexpected argument {a!r}")
        i += 1

    repos = load_repos()
    if not repos:
        die("no repos registered; run `ox repo add <name> <url>` first")
    if not repo:
        die("--repo is required", candidates=sorted(repos))
    if repo not in repos:
        die(f"{repo!r} is not a registered project", candidates=sorted(repos))

    body = read_body()
    if not body.strip():
        die("empty task: nothing was typed")

    title = title or first_line_title(body)
    base = base or repos[repo]
    name = f"{slug(title)}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.md"

    front = [f"repo: {repo}", f"base: {base}", f"title: {title}"]
    if effort:
        front.append(f"effort: {effort}")
    doc = "---\n" + "\n".join(front) + "\n---\n\n" + body.strip() + "\n"

    if dry:
        out(ok=True, dry=True, repo=repo, base=base, title=title,
            task=name, document=doc)
        return 0

    os.makedirs(INBOX, exist_ok=True)
    dest = os.path.join(INBOX, name)
    tmp = dest + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(doc)
        # Atomic rename: the daemon can never observe a half-written task.
        os.replace(tmp, dest)
    except OSError as e:
        die(f"could not write task file: {e}")

    out(ok=True, repo=repo, base=base, title=title, task=name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
