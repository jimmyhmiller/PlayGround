#!/usr/bin/env python3
"""Render codex `exec --json` JSONL events into a readable running log.

Reads events on stdin, writes human-readable text on stdout line-buffered so
`ox tail` shows progress live. Unrecognised event types are printed verbatim
rather than dropped, so a codex upgrade can never silently hide activity.
"""

import json
import sys
import time


def ts():
    return time.strftime("%H:%M:%S")


def emit(line):
    sys.stdout.write(line.rstrip("\n") + "\n")
    sys.stdout.flush()


def clip(text, limit):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n  ... [{len(text) - limit} more chars, see codex.jsonl]"


def main():
    counts = {"commands": 0, "messages": 0, "patches": 0, "errors": 0}
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            emit(f"[{ts()}] NON-JSON OUTPUT: {raw[:2000]}")
            continue

        kind = ev.get("type", "?")

        if kind == "thread.started":
            emit(f"[{ts()}] thread {ev.get('thread_id')}")
        elif kind == "turn.started":
            emit(f"[{ts()}] --- turn start ---")
        elif kind == "turn.completed":
            usage = ev.get("usage") or {}
            emit(
                f"[{ts()}] --- turn complete --- "
                f"in={usage.get('input_tokens', '?')} "
                f"cached={usage.get('cached_input_tokens', '?')} "
                f"out={usage.get('output_tokens', '?')}"
            )
        elif kind == "turn.failed":
            counts["errors"] += 1
            emit(f"[{ts()}] TURN FAILED: {json.dumps(ev.get('error'))}")
        elif kind == "item.started":
            item = ev.get("item") or {}
            if item.get("type") == "command_execution":
                counts["commands"] += 1
                emit(f"[{ts()}] $ {item.get('command')}")
        elif kind == "item.completed":
            item = ev.get("item") or {}
            itype = item.get("type")
            if itype == "command_execution":
                code = item.get("exit_code")
                out = clip(item.get("aggregated_output"), 4000)
                emit(f"[{ts()}]   exit={code}")
                for line in out.splitlines():
                    emit(f"           | {line}")
            elif itype == "agent_message":
                counts["messages"] += 1
                emit(f"[{ts()}] agent:")
                for line in clip(item.get("text"), 8000).splitlines():
                    emit(f"           {line}")
            elif itype == "reasoning":
                text = clip(item.get("text"), 2000)
                if text.strip():
                    emit(f"[{ts()}] reasoning:")
                    for line in text.splitlines():
                        emit(f"           . {line}")
            elif itype == "file_change":
                counts["patches"] += 1
                emit(f"[{ts()}] file changes: {json.dumps(item.get('changes'))[:2000]}")
            elif itype == "todo_list":
                emit(f"[{ts()}] plan: {json.dumps(item.get('items'))[:2000]}")
            elif itype == "error":
                counts["errors"] += 1
                emit(f"[{ts()}] ITEM ERROR: {clip(item.get('message'), 4000)}")
            else:
                emit(f"[{ts()}] {itype}: {json.dumps(item)[:2000]}")
        elif kind == "error":
            counts["errors"] += 1
            emit(f"[{ts()}] ERROR: {json.dumps(ev)[:4000]}")
        else:
            emit(f"[{ts()}] {kind}: {json.dumps(ev)[:2000]}")

    emit(
        f"[{ts()}] === summary: {counts['commands']} commands, "
        f"{counts['messages']} agent messages, {counts['patches']} patch items, "
        f"{counts['errors']} errors ==="
    )


if __name__ == "__main__":
    main()
