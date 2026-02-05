# Scope Conflict Model (Draft)

Conflicts are detected at the field level with last-write-wins by default.

## 1) Conflict Window

Definition:

- If two updates to the same field occur within `conflict_window_seconds`, it is a conflict.
- Default window is `300` seconds.

Config:

- `events.conflict_window_seconds` in `project.toml` or `~/.scope/config.toml`.

## 2) Conflict Detection

Conflicts are detected when:

- Two events update the same field
- The time delta between them is within the conflict window

Non-conflicts:

- Updates to different fields
- Comments (append-only)

## 3) Conflict Records

Conflicts are recorded as structured data (stored in index or a separate file).

Proposed shape:

```json
{
  "id": "conf_001",
  "issue": "SC:brisk-silent-otter",
  "field": "status",
  "local_event": "evt_0009",
  "remote_event": "evt_0010",
  "local_value": "in_progress",
  "remote_value": "blocked",
  "detected_at": "2026-02-04T20:10:00Z",
  "state": "unresolved"
}
```

## 4) Resolution

Resolutions are events:

- `issue.resolve_conflict` with `conflict_id` and chosen resolution

Resolution strategies:

- `keep_local`
- `keep_remote`
- `manual` (user edits the issue and records a resolution)

## 5) CLI UX

```bash
scope issues conflicts list
scope issues conflicts show conf_001
scope issues conflicts resolve conf_001 --keep local
scope issues conflicts resolve conf_001 --keep remote
```

Behavior:

- `list` shows unresolved conflicts by issue
- `show` prints details of a conflict
- `resolve` writes a resolution event and updates snapshot + index
