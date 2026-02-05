# Scope Index Schema (Draft)

The index is a derived cache for fast queries.

Location:

- `~/.scope/projects/<subdir>/index/issues.json`

## 1) File Shape

Top-level object:

```json
{
  "format_version": 1,
  "generated_at": "2026-02-04T20:00:00Z",
  "project": "acme-api",
  "issues": [
    {
      "id": "SC:brisk-silent-otter",
      "title": "Add filter for blocked issues",
      "status": "in_progress",
      "priority": "p2",
      "assignee": "jimmy",
      "labels": ["cli","triage"],
      "created_at": "2026-02-04T16:30:00Z",
      "updated_at": "2026-02-04T19:00:00Z",
      "path": "issues/SC:brisk-silent-otter.md",
      "events_path": "events/SC:brisk-silent-otter.jsonl"
    }
  ]
}
```

## 2) Index Fields

Per issue entry:

- `id` (string)
- `title` (string)
- `status` (string enum)
- `priority` (string enum)
- `assignee` (string, optional)
- `labels` (string array)
- `created_at` (timestamp)
- `updated_at` (timestamp)
- `path` (relative file path)
- `events_path` (relative file path)

## 3) Rebuild Rules

- Index is always rebuildable from issue snapshots or from events
- On `scope issues rebuild`, snapshots and index are generated from events
- On `scope issues index rebuild`, index is generated from snapshots

## 4) Usage

- All read commands use index by default
- Index is updated incrementally on every write
- Index is rebuilt when drift is detected or requested
