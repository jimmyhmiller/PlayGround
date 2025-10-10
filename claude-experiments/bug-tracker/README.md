# Bug Tracker CLI

A simple command-line tool designed for LLM agents (like Claude Code) to record bugs discovered during development. When an LLM is implementing features and discovers issues, it can use this tool to track them for later resolution.

## Features

- Record bugs to a local `BUGS.md` file in markdown format
- Capture bug metadata: title, description, location (file/context), severity, tags, timestamp
- Easy integration with Claude Code through `claude.md`
- Simple CLI interface with helpful defaults

## Installation

### Build from source

```bash
cargo build --release
```

The binary will be available at `target/release/bug-tracker`

### Add to PATH (optional)

```bash
# Copy to a location in your PATH
cp target/release/bug-tracker /usr/local/bin/
```

## Usage

### Recording a Bug

Basic usage:

```bash
bug-tracker add --project /path/to/project --title "Bug title"
```

With all options:

```bash
bug-tracker add \
  --project /path/to/project \
  --title "Null pointer dereference in auth module" \
  --description "The authenticate() function doesn't check for null before dereferencing the user pointer" \
  --file "src/auth.rs" \
  --context "authenticate()" \
  --severity high \
  --tags "memory,safety,authentication"
```

### Options

- `--project, -p` (required): Project root directory where BUGS.md will be created/updated
- `--title, -t` (required): Short bug title/summary
- `--description, -d`: Detailed description of the bug
- `--file, -f`: File path where the bug was found
- `--context, -c`: Code context (function/class/module name) where the bug was found
- `--severity, -s`: Severity level (low, medium, high, critical) [default: medium]
- `--tags`: Comma-separated tags for categorization

### Installing to Claude Code

Make sure `bug-tracker` is in your PATH. From your project root, run:

```bash
bug-tracker install
```

This will automatically find `claude.md` in `.claude/claude.md` or `./claude.md` and append the tool definition. You may need to restart your Claude Code session for changes to take effect.

If your claude.md is in a different location, specify it:

```bash
bug-tracker install --claude-md /path/to/claude.md
```

## Example BUGS.md Output

```markdown
# Bugs

This file tracks bugs discovered during development.

## Null pointer dereference in auth module

**Timestamp:** 2025-10-10 14:32:15
**Severity:** high
**Location:** src/auth.rs (authenticate())
**Tags:** memory, safety, authentication

### Description

The authenticate() function doesn't check for null before dereferencing the user pointer

---

## Memory leak in connection pool

**Timestamp:** 2025-10-10 15:45:22
**Severity:** medium
**Location:** src/pool.rs (ConnectionPool::release)
**Tags:** memory, performance

### Description

Connection objects are not properly released back to the pool when errors occur during query execution

---
```

## Integration with Claude Code

Once installed to `claude.md`, Claude Code can use this tool during development:

```
User: Please implement the new authentication feature
Claude: I'll implement the authentication feature...
[discovers a potential issue]
Claude: I've found a potential security issue - let me record it using the bug tracker...
[calls bug-tracker tool]
Claude: I've recorded the bug to BUGS.md for later review. Continuing with the implementation...
```

## Use Cases

- **During Feature Implementation**: Record bugs found while implementing new features
- **Code Review**: Track issues discovered during code review
- **Refactoring**: Note technical debt or issues found during refactoring
- **Debugging**: Document bugs encountered while fixing other issues
- **Security Audits**: Track potential security vulnerabilities

## Design Philosophy

This tool is intentionally simple:
- Writes to a plain markdown file for easy reading and version control
- Minimal dependencies
- Command-line focused for LLM integration
- No database or server required
- Human-readable output

## License

MIT
