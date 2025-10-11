# Integration Test Scenarios

Each scenario is a JSON file that describes:
1. Initial git repository state
2. Actions to perform (add notes, make commits, modify code)
3. Expectations to verify (note positions, migration success, etc.)

## Scenario Format

```json
{
  "name": "Scenario name",
  "description": "What this scenario tests",
  "steps": [
    {
      "action": "create_file",
      "file": "path/to/file.rs",
      "content": "file contents"
    },
    {
      "action": "git_commit",
      "message": "Initial commit"
    },
    {
      "action": "add_note",
      "file": "path/to/file.rs",
      "line": 5,
      "column": 0,
      "content": "Note content",
      "author": "TestAuthor",
      "collection": "default"
    },
    {
      "action": "modify_file",
      "file": "path/to/file.rs",
      "content": "new file contents"
    },
    {
      "action": "git_commit",
      "message": "Refactored code"
    },
    {
      "action": "migrate_notes",
      "collection": "default"
    },
    {
      "action": "expect_note",
      "content": "Note content",
      "file": "path/to/file.rs",
      "line": 8,
      "is_orphaned": false
    }
  ]
}
```

## Action Types

- `create_file`: Create a file with content
- `modify_file`: Modify existing file
- `delete_file`: Delete a file
- `git_commit`: Make a git commit
- `add_note`: Add a note via CLI
- `migrate_notes`: Run migration
- `expect_note`: Assert note properties
- `expect_migration_success`: Assert migration statistics

## Running Tests

```bash
cargo test --test integration_tests
```
