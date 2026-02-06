---
name: scope-cli
description: Use the scope CLI for developer-first issue tracking in this repo. Trigger when the user asks to create/update/list/show issues, manage comments, conflicts, indexes, project mapping, or scope config files, or when demonstrating or scripting the scope CLI.
---

# Scope CLI

## Overview

Use the `scope` CLI in this repo to manage local-first issues stored as markdown snapshots plus JSONL event logs. Prefer CLI commands over manual file edits.

## Quick Start

1. Initialize a project:
   - `scope issues init --project <name>`
2. Create an issue:
   - `scope issues create --project <name> --title "..." [--priority p1] [--assignee ...] [--label ...]`
3. List and show:
   - `scope issues list --project <name>`
   - `scope issues show --project <name> <id>`

## Core Tasks

### Update and workflow

- Update fields: `scope issues update <id> --status in_progress --add-label planning`
- Close/reopen: `scope issues close <id>` / `scope issues reopen <id>`
- Edit body: `scope issues edit <id>`

### Comments

- Add: `scope issues comments add <id> --body "..."` or `--body-file ...`
- List: `scope issues comments list <id> [--json]`

### Index and rebuild

- `scope issues index status|rebuild|verify`
- `scope issues rebuild` to regenerate snapshots + index from events

### Projects and mappings

- `scope issues projects [--json]`
- `scope issues project show --project <name>`
- Map multiple working copies:
  - `scope issues project link --project <name> --path /path/to/repo`
  - `scope issues project links --project <name> [--json]`

### Conflicts

- `scope issues conflicts list`
- `scope issues conflicts show <id>`
- `scope issues conflicts resolve <id> --keep local|remote`

## References

Read these when you need the detailed spec or schema:

- `references/cli.md` for command surface and file layout
- `references/specs.md` for links to the full design docs in `docs/`
