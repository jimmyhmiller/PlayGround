# Scope Issue Tracking CLI Spec (Draft)

This document defines the initial CLI interface for issue tracking in `scope`.

Current decisions:

- `scope` is the CLI name
- Issues are markdown files with YAML frontmatter
- Config files use TOML
- Local index is required for fast search/list/filter
- Default storage root is `~/.scope/projects/<subdir>`
- Event log is JSON Lines (append-only) for full history
- Focus is issue tracking only (for now)

## 1) Goals

- Local-first workflow with fast commands
- Clear, composable CLI interactions
- Human-readable storage and agent-friendly automation
- Stable machine output with `--json`

## 2) Storage Layout

Global root (default):

- `~/.scope/projects/`

Each project lives in a subdirectory:

- `~/.scope/projects/<subdir>/issues/<id>.md` (source of truth)
- `~/.scope/projects/<subdir>/events/<id>.jsonl` (append-only event log)
- `~/.scope/projects/<subdir>/index/issues.json` (derived local index)
- `~/.scope/projects/<subdir>/project.toml` (project config)

Global config:

- `~/.scope/config.toml`

Config resolution order:

- CLI flags (highest precedence)
- Project config (`project.toml`)
- Global config (`~/.scope/config.toml`)
- Built-in defaults

Rules:

- Issue markdown files are canonical
- Index is rebuildable from markdown files
- Event logs are append-only and never rewritten
- Any command that changes issues also updates index
- `scope issues index rebuild` repairs index if needed
- Adhoc projects are supported by creating new `<subdir>` entries
- Projects can override sync settings independently

Reference specs:

- Config schema: `docs/config-schema.md`
- Event schema: `docs/event-schema.md`
- Index schema: `docs/index-schema.md`
- Conflict model: `docs/conflict-model.md`
- Conflict model: `docs/conflict-model.md`

## 3) Issue ID Format

Issue IDs are generated, human-readable, and unique:

- Pattern: `<prefix>:<adjective>-<adjective>-<animal>`
- Example: `SC:brisk-silent-otter`

Constraints:

- Lowercase words
- Hyphen-separated slug after `SC:`
- Must be unique in the repository
- Once assigned, ID never changes

Prefix customization:

- Default prefix is `SC`
- Projects can override prefix in `project.toml` under `[ids]`

Collision handling:

- Regenerate until unique
- Optional numeric suffix if dictionary is exhausted (rare): `SC:brisk-silent-otter-2`

## 4) Issue File Format

Each issue is a markdown file with YAML frontmatter.

```md
---
id: SC:brisk-silent-otter
title: Add filter for blocked issues
status: todo
priority: p2
assignee: jimmy
labels:
  - cli
  - triage
created_at: 2026-02-04T16:30:00Z
updated_at: 2026-02-04T16:30:00Z
---

## Summary
Add an explicit filter for blocked issues in list output.

## Acceptance Criteria
- `scope issues list --status blocked` returns only blocked issues
- Works in both table and `--json` output
```

## 5) Status Model

Initial statuses:

- `todo`
- `in_progress`
- `blocked`
- `done`

Allowed transitions:

- `todo -> in_progress`
- `in_progress -> blocked`
- `blocked -> in_progress`
- `in_progress -> done`
- `todo -> done` (allowed for trivial work)

## 6) CLI Commands

### 6.1 Project setup

```bash
scope issues init --project acme-api
scope issues init --project acme-api --sync-engine git
scope issues init --project lab-notes --sync-engine service
scope issues init
```

Behavior:

- Creates `~/.scope/` if missing
- Creates `~/.scope/projects/<project>/`
- Creates `issues/`, `index/`, and `project.toml`
- Initializes empty `index/issues.json`
- Registers project in global config if needed
- If `--project` is omitted, prompts to use the current directory name

Project targeting (all commands):

```bash
scope issues list --project acme-api
scope issues create --project acme-api --title "Add CLI smoke test"
```

Behavior:

- Commands run against selected `--project`
- If `--project` is omitted, use project from current directory mapping or default project in config
- `scope issues projects` lists known local projects

### 6.2 Create issues

```bash
scope issues create --title "Add blocked filter"
scope issues create --title "Index rebuild command" --priority p1 --assignee jimmy --label cli --label index
scope issues create --title "Document status transitions" --body-file ./tmp/issue.md
```

Behavior:

- Generates unique ID (ex: `SC:calm-rapid-falcon`)
- Writes markdown issue file
- Updates local index
- Prints created ID

### 6.3 List issues

```bash
scope issues list
scope issues list --status in_progress
scope issues list --assignee jimmy --label cli
scope issues list --priority p1 --limit 20
scope issues list --query "blocked filter"
scope issues list --json
```

Behavior:

- Reads from local index by default (fast path)
- Supports filtering, sorting, and paging
- `--json` returns stable machine-readable objects

### 6.4 Show issue

```bash
scope issues show SC:brisk-silent-otter
scope issues show SC:brisk-silent-otter --json
```

Behavior:

- Loads issue by ID
- Default prints formatted view
- `--json` returns full metadata + body

### 6.5 Update issue fields

```bash
scope issues update SC:brisk-silent-otter --title "Add blocked-only filter"
scope issues update SC:brisk-silent-otter --status in_progress
scope issues update SC:brisk-silent-otter --assignee sarah --priority p1
scope issues update SC:brisk-silent-otter --add-label ux --remove-label triage
```

Behavior:

- Validates transition rules for `--status`
- Updates frontmatter in issue file
- Bumps `updated_at`
- Updates local index

### 6.6 Edit issue body

```bash
scope issues edit SC:brisk-silent-otter
scope issues edit SC:brisk-silent-otter --editor nvim
```

Behavior:

- Opens markdown file in configured editor
- On save/exit, reindexes issue entry

### 6.7 Close and reopen shortcuts

```bash
scope issues close SC:brisk-silent-otter
scope issues reopen SC:brisk-silent-otter
```

Behavior:

- `close` sets `status=done`
- `reopen` sets `status=todo` (or `in_progress` with `--status`)
- Updates timestamp + index

### 6.8 Delete issue

```bash
scope issues delete SC:brisk-silent-otter
scope issues delete SC:brisk-silent-otter --force
```

Behavior:

- Soft default: prompt confirmation
- `--force` skips prompt
- Removes issue file and index record

### 6.9 Index commands

```bash
scope issues index status
scope issues index rebuild
scope issues index verify
scope issues rebuild
```

Behavior:

- `status`: shows index freshness and counts
- `rebuild`: full scan from markdown source of truth
- `verify`: checks for drift, duplicates, missing files
- `scope issues rebuild`: regenerate issue snapshots and index from events

### 6.10 Comments

```bash
scope issues comments add SC:brisk-silent-otter --body "Looks good to me."
scope issues comments add SC:brisk-silent-otter --body-file ./comment.md
scope issues comments list SC:brisk-silent-otter
scope issues comments list SC:brisk-silent-otter --json
```

Behavior:

- Appends an `issue.comment` event
- Renders comments into the issue snapshot
- `list` returns comments from the event log

### 6.11 Conflicts

```bash
scope issues conflicts list
scope issues conflicts show conf_001
scope issues conflicts resolve conf_001 --keep local
```

Behavior:

- `list` shows unresolved conflicts
- `show` prints conflict details
- `resolve` writes a resolution event and updates the issue snapshot

### 6.12 Project and sync management

```bash
scope issues projects
scope issues projects --json
scope issues project show --project acme-api
scope issues project config set --project acme-api sync.engine git
scope issues project config set --project acme-api sync.remote git@github.com:org/acme-scope.git
scope issues project links --project acme-api
scope issues project links --json
scope issues project link --project acme-api --path /path/to/repo
scope issues sync pull --project acme-api
scope issues sync push --project acme-api
scope issues sync status --project acme-api
scope skills install
scope skills install --codex
scope skills install --claude
```

Behavior:

- `projects`: list all local project subdirs under `~/.scope/projects/`
- `projects --json`: machine-readable project list
- `project show`: print resolved config and paths
- `project config set`: update per-project overrides in `project.toml`
- `project links`: list all paths mapped to projects (optionally filtered)
- `project link`: map a working directory to a project for cwd-based resolution
- `sync pull/push/status`: run selected sync engine for that project
- `skills install`: writes the `scope-cli` skill into `~/.codex/skills` and `~/.claude/skills`

## 7) History and Events

Events are the authoritative history and collaboration layer.

Storage:

- One JSON Lines file per issue: `events/<id>.jsonl`
- Each line is a JSON object (append-only)

Example event lines:

```json
{"id":"evt_0001","type":"issue.create","issue":"SC:brisk-silent-otter","author":"jimmy","ts":"2026-02-04T16:30:00Z","data":{"title":"Add filter for blocked issues","status":"todo","priority":"p2","labels":["cli","triage"]}}
{"id":"evt_0002","type":"issue.comment","issue":"SC:brisk-silent-otter","author":"sarah","ts":"2026-02-04T18:00:00Z","data":{"body":"Should we support `--json` output for this filter?"}}
{"id":"evt_0003","type":"issue.update","issue":"SC:brisk-silent-otter","author":"jimmy","ts":"2026-02-04T19:00:00Z","data":{"status":"in_progress"}}
```

Event rules:

- Events are append-only and immutable
- Snapshot (`issues/<id>.md`) is derived from events on write
- Any write command appends an event and updates the snapshot + index
- `scope issues rebuild` can regenerate snapshots and index from events (if needed)

Comment rendering:

- Comments are stored as events
- Snapshot markdown includes a `## Comments` section that renders comments in order

## 8) Conflict Detection and Resolution

Conflict handling is field-level with last-write-wins by default.

Rules:

- Updates touching different fields do not conflict
- If two updates touch the same field, the later timestamp wins
- Conflicts are detected if two updates to the same field occur within a configurable conflict window
- Conflicts are reported to the user with the losing value and event IDs
- Comments never conflict

User experience:

- `scope issues sync pull` reports detected conflicts
- `scope issues conflicts list` shows unresolved conflicts
- `scope issues conflicts resolve <id> --keep local|remote` writes a resolution event

## 9) Output Modes

Default output:

- Human-friendly table/list

Machine output:

- `--json` for all read commands
- Deterministic field names and ordering
- Errors can also be emitted as JSON with `--json`

## 10) Simple Interaction Examples

Create and start work:

```bash
scope issues create --title "Implement issue index"
scope issues update SC:swift-lucid-panda --status in_progress
scope issues list --status in_progress
```

Find blockers:

```bash
scope issues list --status blocked
scope issues list --status blocked --json | jq '.[].id'
```

Repair index drift:

```bash
scope issues index verify
scope issues index rebuild
```

## 11) Validation Rules (V0)

- `title` required on create
- `status` must be valid enum
- `priority` in `p0|p1|p2|p3`
- `labels` are lowercase slugs
- issue ID immutable
- any write command updates index

## 12) Out of Scope (For Now)

- Epics and milestones
- Chat/docs/standup commands

## 13) Next Build Steps

1. Implement `scope issues init`
2. Implement `scope issues create`
3. Implement `scope issues list` (index-backed)
4. Implement `scope issues show`
5. Implement `scope issues index rebuild` and `scope issues index verify`
