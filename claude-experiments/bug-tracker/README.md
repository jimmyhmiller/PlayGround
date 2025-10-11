# Bug Tracker CLI

A simple command-line tool designed for LLM agents (like Claude Code) to record bugs discovered during development. When an LLM is implementing features and discovers issues, it can use this tool to track them for later resolution.

## Features

- Record bugs to a local `BUGS.md` file in markdown format
- Capture comprehensive bug metadata: title, description, location (file/context), severity, tags, timestamp
- Unique bug IDs using goofy animal names (e.g., "curious-elephant") for easy reference
- Close/remove bugs by ID when fixed
- List all open bugs with their IDs and severity
- Optional AI-powered quality validation to ensure bug reports are actionable
- Support for reproduction steps, code snippets, and custom metadata
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

- `--project, -p`: Project root directory where BUGS.md will be created/updated (optional; defaults to searching current directory and parents)
- `--title, -t` (required): Short bug title/summary
- `--description, -d`: Detailed description of the bug
- `--file, -f`: File path where the bug was found
- `--context, -c`: Code context (function/class/module name) where the bug was found
- `--severity, -s`: Severity level (low, medium, high, critical) [default: medium]
- `--tags`: Comma-separated tags for categorization
- `--repro, -r`: Minimal reproducing case or steps to reproduce the bug
- `--code-snippet`: Code snippet demonstrating the bug
- `--metadata`: Additional metadata as JSON string (e.g., `'{"version":"1.2.3","platform":"linux"}'`)
- `--validate`: Enable AI-powered quality check validation before recording the bug

### Listing Bugs

To see all open bugs:

```bash
bug-tracker list
```

Each bug is assigned a unique ID (e.g., `curious-elephant`) when created, which is displayed in the output.

### Closing Bugs

When a bug is fixed, close it using its ID:

```bash
bug-tracker close curious-elephant
```

This removes the bug entry from BUGS.md.

### Installing to Claude Code

Make sure `bug-tracker` is in your PATH. From your project root, run:

```bash
bug-tracker install
```

This will automatically find `claude.md` in `.claude/claude.md` or `./claude.md` and add the tool definition. The installation includes both the `add` command for creating bugs and the `close` command for removing them by ID.

**Note**: If the Bug Tracker section already exists in claude.md, it will be **replaced** with the latest version instead of being duplicated. This ensures you always have the most up-to-date tool definition.

You may need to restart your Claude Code session for changes to take effect.

If your claude.md is in a different location, specify it:

```bash
bug-tracker install --claude-md /path/to/claude.md
```

## AI Quality Validation

The `--validate` flag enables AI-powered quality checking of bug reports before they're recorded. This feature uses Claude CLI to analyze the bug report and ensure it contains sufficient information to be actionable.

When validation is enabled:
- The bug report is analyzed for completeness
- You'll receive feedback on what information might be missing
- Reports that lack critical details will be rejected with specific guidance
- Well-formed reports are approved and recorded

Example with validation:

```bash
bug-tracker add \
  --title "Authentication fails silently" \
  --description "When invalid credentials are provided, the authenticate() function returns null without logging or error messages" \
  --file "src/auth.rs" \
  --context "authenticate()" \
  --severity medium \
  --validate
```

## Example BUGS.md Output

```markdown
# Bugs

This file tracks bugs discovered during development.

## Null pointer dereference in auth module [curious-elephant]

**ID:** curious-elephant
**Timestamp:** 2025-10-10 14:32:15
**Severity:** high
**Location:** src/auth.rs (authenticate())
**Tags:** memory, safety, authentication

### Description

The authenticate() function doesn't check for null before dereferencing the user pointer

### Minimal Reproducing Case

Call authenticate with null user_ptr parameter

### Code Snippet

```
if (!user_ptr) { /* missing check */ }
user_ptr->name = "admin";
```

---

## Memory leak in connection pool [gentle-wombat]

**ID:** gentle-wombat
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
[calls bug-tracker add with details]
Claude: I've recorded the bug with ID "curious-elephant" to BUGS.md for later review.
        Continuing with the implementation...

[later, after fixing the issue]
User: I've fixed the authentication issue
Claude: Great! Let me close that bug...
[calls bug-tracker close curious-elephant]
Claude: Bug 'curious-elephant' has been closed and removed from BUGS.md.
```

**Workflow example:**

1. **Discovery**: Claude finds a bug while implementing features
2. **Recording**: Uses `bug-tracker add` to record it with comprehensive details
3. **Tracking**: Bug gets a unique ID and is added to BUGS.md
4. **Review**: Developer or Claude can use `bug-tracker list` to see all open bugs
5. **Resolution**: When fixed, use `bug-tracker close <ID>` to remove it

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
