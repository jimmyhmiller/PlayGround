# Code Notes

A layered annotation system for code that persists across git commits and refactorings.

## Overview

Code Notes allows you to attach notes and annotations to specific parts of your codebase. Unlike traditional comments that live in the source code itself, Code Notes stores annotations separately, offering several advantages:

1. **Non-intrusive**: Notes don't clutter your source code
2. **Personal**: Each developer can maintain their own set of notes
3. **Layered**: Different note collections for different purposes (onboarding, documentation, debugging trails, etc.)
4. **Persistent**: Notes automatically migrate across git commits using tree-sitter AST analysis

## Key Features

- **AST-based anchoring**: Notes are attached to code using tree-sitter AST nodes, making them resilient to line number changes
- **Automatic migration**: Notes automatically follow code as it moves and changes across commits
- **Multiple collections**: Organize notes into collections for different purposes or audiences
- **Metadata support**: Attach arbitrary metadata to notes for extensions like trails, priority, etc.
- **Import/Export**: Share note collections as bundles
- **Orphan detection**: Identifies notes that can no longer find their anchored code

## Architecture

### Core Components

1. **Models** (`src/models/`):
   - `Note`: A note with content, author, timestamps, and metadata
   - `CodeAnchor`: Precisely locates code using AST path, node kind, and context
   - `NoteAnchor`: Combines primary anchor with alternatives for resilience
   - `NoteCollection`: Groups related notes

2. **Parsers** (`src/parsers/`):
   - `CodeParser`: Tree-sitter wrapper supporting Rust, Python, JavaScript, TypeScript
   - `AnchorBuilder`: Creates anchors from tree-sitter AST nodes
   - `AnchorMatcher`: Finds matching nodes across different code versions

3. **Git Integration** (`src/git/`):
   - `GitRepo`: Git repository operations wrapper
   - `NoteMigrator`: Migrates notes across commits with confidence scoring

4. **Storage** (`src/storage/`):
   - File-based storage in `.code-notes/` directory
   - JSON format for notes and collections
   - Import/export functionality

5. **CLI** (`src/cli/`):
   - Command-line interface for all operations

## Installation

```bash
cargo build --release
# The binary will be in target/release/code-notes
# Optionally, copy to your PATH:
cp target/release/code-notes /usr/local/bin/
```

## Usage

### Initialize in a repository

```bash
code-notes init
```

This registers the repository with code-notes. Notes are stored globally in `~/.code-notes/` with smart path-based organization to avoid conflicts.

### Add a note

```bash
code-notes add \
  --file src/main.rs \
  --line 42 \
  --column 5 \
  --content "This function handles authentication logic" \
  --author "Alice" \
  --collection default
```

**Note**: File paths use smart resolution - you can provide just a filename, a relative path, or a full path. The CLI will intelligently find the file in your repository.

### List notes

```bash
# List all notes from all collections
code-notes list

# List notes from a specific collection
code-notes list --collection default

# List notes for a specific file (across all collections)
# You can use just the filename - it will be found automatically
code-notes list sample_code.rs
code-notes list src/main.rs

# List notes for a file in a specific collection
code-notes list src/main.rs --collection default
```

**Smart File Resolution**: You don't need to provide the full path to a file. The CLI will:
1. Try the path relative to your current directory
2. Try the path relative to the repository root
3. Search the entire repository for files matching that name
4. Provide helpful suggestions if multiple files match

### View a specific note

```bash
code-notes view <note-id>
```

### Update a note

```bash
code-notes update <note-id> --content "Updated description"
```

### Delete a note

```bash
code-notes delete <note-id> --collection default
```

### Migrate notes after git commits

```bash
code-notes migrate --collection default
```

This attempts to update all note anchors to the current commit.

### Working with collections

```bash
# List all collections
code-notes collections

# Create a new collection
code-notes create-collection "onboarding" --description "Notes for new developers"

# Export a collection
code-notes export onboarding --output onboarding-notes.json

# Import a collection
code-notes import onboarding-notes.json

# View orphaned notes
code-notes orphaned --collection default
```

## How It Works

### Anchor Mechanism

When you add a note, Code Notes:

1. Parses the source file using tree-sitter
2. Finds the AST node at the specified position
3. Creates a `CodeAnchor` containing:
   - Node kind (e.g., "identifier", "function_definition")
   - Node text content
   - Line and column position
   - AST path from root to node
   - Context (parent node types)
4. Stores alternative anchors (parent and siblings) for redundancy

### Migration Algorithm

When migrating notes across commits:

1. For each note, traverse commits from the note's original commit to HEAD
2. For each commit where the file changed:
   - Parse the file at that commit
   - Search for matching AST nodes using:
     - Exact text match (strongest signal)
     - Node kind match (required)
     - AST path similarity
     - Context similarity
   - Calculate confidence score (0.0 to 1.0)
   - If confidence > 0.7, update the anchor
   - If confidence too low, mark note as orphaned
3. Track migration history with success/failure records

### Storage Format

Notes are stored globally in `~/.code-notes/` with a per-project directory structure:

```
~/.code-notes/
├── project_index.json              # Maps repos to storage directories
├── my-project/                     # Project-specific directory
│   ├── collections.json           # Collection metadata
│   └── notes/
│       ├── <uuid1>.json          # Individual note files
│       ├── <uuid2>.json
│       └── ...
└── other-project/                  # Another project
    ├── collections.json
    └── notes/
        └── ...
```

**Smart Path Resolution**: Project directories use the repository name by default (e.g., "my-project"). If there's a naming conflict, parent directory names are added incrementally (e.g., "parent_my-project") until the name is unique.

## Use Cases

### Personal Development Notes

Keep personal notes about complex code sections without cluttering the codebase:

```bash
code-notes add --file src/parser.rs --line 150 --column 0 \
  --content "TODO: Optimize this recursive descent - consider using iterative approach" \
  --author "Me" --collection personal
```

### Onboarding Documentation

Create layered documentation for new team members:

```bash
code-notes create-collection "onboarding"
code-notes add --file src/auth.rs --line 42 --column 0 \
  --content "This is our main authentication flow. See docs/auth.md for details." \
  --author "TeamLead" --collection onboarding
```

### Code Review Trails

Track code review comments and discussions:

```bash
code-notes add --file src/api.rs --line 75 --column 0 \
  --content "Reviewer: Consider adding rate limiting here" \
  --author "Reviewer" --collection code-review
```

### Debugging Context

Maintain debugging notes that persist across refactorings:

```bash
code-notes add --file src/cache.rs --line 200 --column 0 \
  --content "Bug #1234: Cache invalidation issue occurs when X and Y happen together" \
  --author "DebugTeam" --collection bugs
```

## Future Enhancements

Potential future features:

- **Git Hooks**: Automatically inject/remove notes as inline comments during checkout
- **IDE Integration**: VS Code extension to display notes inline
- **Rename Tracking**: Track identifier renames across commits
- **Rich Metadata**: Support for tags, priority, timestamps, attachments
- **Web Interface**: Browse notes through a web UI
- **Collaborative Sharing**: Sync notes across team members
- **More Languages**: Expand tree-sitter support to more languages

## Technical Details

### Supported Languages

Currently supports:
- Rust (.rs)
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)

Adding new languages requires adding the tree-sitter grammar dependency.

### Confidence Scoring

The anchor matching algorithm uses weighted scoring:
- Exact text match: 50% weight
- Node kind match: 25% weight (required)
- AST path similarity: 15% weight
- Context similarity: 10% weight

Minimum confidence threshold: 0.7

### Data Format

Notes use JSON serialization with serde. Example note structure:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "This handles authentication",
  "author": "Alice",
  "created_at": 1234567890,
  "updated_at": 1234567890,
  "anchor": {
    "primary": {
      "file_path": "src/main.rs",
      "node_kind": "identifier",
      "node_text": "authenticate_user",
      "line_number": 42,
      "column": 5,
      "ast_path": [...],
      "context": [...]
    },
    "alternatives": [...],
    "commit_hash": "abc123...",
    "is_valid": true,
    "migration_history": []
  },
  "metadata": {},
  "is_orphaned": false
}
```

## Contributing

This is an experimental project exploring layered code annotations. Feedback and contributions welcome!

## License

MIT License - see LICENSE file for details
