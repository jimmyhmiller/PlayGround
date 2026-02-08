# Scope CLI Reference (Summary)

## Storage

- Projects root: `~/.scope/projects/<subdir>/`
- Issues: `issues/<id>.md`
- Events: `events/<id>.jsonl`
- Index: `index/issues.json`
- Conflicts: `index/conflicts.json`
- Project config: `project.toml`
- Global config: `~/.scope/config.toml`

## Common Commands

```bash
scope issues init --project <name>
scope issues create --project <name> --title "..." [--body "..."] [--body-file ./issue.md]
scope issues list --project <name> [--json]
scope issues show --project <name> <id>
scope issues update <id> [--status ...] [--priority ...] [--add-label ...]
scope issues close <id>
scope issues reopen <id> [--status ...]
scope issues delete <id> [--force]
scope issues restore <id>

scope issues comments add <id> --body "..."
scope issues comments list <id> [--json]

scope issues index status|rebuild|verify
scope issues rebuild

scope issues projects [--json]
scope issues project show --project <name>
scope issues project config set --project <name> <key> <value>
scope issues project link --project <name> --path /path/to/repo
scope issues project links --project <name> [--json]

scope issues conflicts list
scope issues conflicts show <conflict_id>
scope issues conflicts resolve <conflict_id> --keep local|remote
```

Note: avoid stdin/heredoc bodies (ex: `/dev/stdin`, `<<EOF`) in agent shells; prefer `--body`, a real file path, or `scope issues edit`. `scope issues delete` is a soft delete by default and prints an undo command; use `--force` for permanent removal.
