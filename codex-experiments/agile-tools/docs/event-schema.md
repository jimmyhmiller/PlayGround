# Scope Event Schema (Draft)

Events are JSON Lines records stored in `events/<issue_id>.jsonl`.

## 1) Base Event Fields

Every event includes:

```json
{
  "id": "evt_0001",
  "type": "issue.create",
  "issue": "SC:brisk-silent-otter",
  "author": "jimmy",
  "ts": "2026-02-04T16:30:00Z",
  "data": {}
}
```

Field definitions:

- `id`: unique event ID (globally unique within project)
- `type`: event type string
- `issue`: issue ID
- `author`: user or agent ID
- `ts`: ISO-8601 timestamp in UTC
- `data`: type-specific payload

## 2) Event IDs

Event IDs must be unique within a project:

- Recommended format: `evt_<ulid>` or `evt_<timestamp>_<rand>`
- Used for dedupe during sync

## 3) Event Types

### 3.1 `issue.create`

Creates the issue.

```json
{"type":"issue.create","data":{"title":"Add filter","status":"todo","priority":"p2","assignee":"jimmy","labels":["cli"]}}
```

### 3.2 `issue.update`

Updates one or more fields.

```json
{"type":"issue.update","data":{"status":"in_progress","assignee":"sarah"}}
```

### 3.3 `issue.comment`

Adds a comment to the issue.

```json
{"type":"issue.comment","data":{"body":"We should support --json output.","reply_to":null}}
```

### 3.4 `issue.close`

Marks issue as done (shortcut).

```json
{"type":"issue.close","data":{"status":"done"}}
```

### 3.5 `issue.reopen`

Reopens issue.

```json
{"type":"issue.reopen","data":{"status":"todo"}}
```

### 3.6 `issue.resolve_conflict`

Records a conflict resolution.

```json
{"type":"issue.resolve_conflict","data":{"conflict_id":"conf_001","resolution":"keep_local"}}
```

## 4) Allowed Fields in `data`

Fields allowed in `issue.update` and `issue.create`:

- `title` (string)
- `status` (`todo|in_progress|blocked|done`)
- `priority` (`p0|p1|p2|p3`)
- `assignee` (string)
- `labels` (string array)
- `body` (string, optional; if present, replaces body)

## 5) Replay Rules (Snapshot Generation)

- Start from `issue.create`
- Apply updates in chronological order
- Later updates overwrite earlier values for the same field
- Comments are appended to `## Comments` in snapshot

## 6) Validation

- `issue.create` must be the first event for an issue
- `issue.update` requires the issue to exist
- Unknown `type` should be preserved but ignored for snapshot building
