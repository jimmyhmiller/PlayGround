# Scope Config Schema (Draft)

This document defines the TOML configuration files for `scope`.

Files:

- Global config: `~/.scope/config.toml`
- Project config: `~/.scope/projects/<subdir>/project.toml`

Precedence:

1. CLI flags
2. Project config
3. Global config
4. Built-in defaults

## 1) Global Config (`~/.scope/config.toml`)

```toml
# Global defaults
default_project = "acme-api"
editor = "nvim"
timezone = "local"

[paths]
projects_root = "~/.scope/projects"

[ids]
prefix = "SC"
pattern = "{adj}-{adj}-{animal}"
wordlist_adjectives = "~/.scope/wordlists/adjectives.txt"
wordlist_animals = "~/.scope/wordlists/animals.txt"
max_attempts = 100

[index]
format_version = 1
auto_rebuild = true

[events]
format_version = 1
conflict_window_seconds = 300

[sync]
engine = "git"          # git | service | custom
remote = ""             # optional
```

## 2) Project Config (`project.toml`)

```toml
name = "acme-api"

[ids]
prefix = "ACME"

[paths]
# Optional override for this project only
root = "~/.scope/projects/acme-api"

[sync]
engine = "git"
remote = "git@github.com:org/acme-scope.git"

[events]
conflict_window_seconds = 300
```

## 3) Notes

- `editor` is used by `scope issues edit` if `--editor` is not specified.
- `timezone = "local"` means timestamps are stored in UTC but displayed in local time.
- `paths.projects_root` must be an absolute path after expansion.
- `ids.wordlist_*` files are optional; built-in lists are used if missing.
- Any key can be overridden by CLI flags (not fully enumerated here).
