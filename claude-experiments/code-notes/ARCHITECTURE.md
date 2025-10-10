# Code Notes - Architecture Documentation

## System Overview

Code Notes is a layered annotation system that allows developers to attach notes to code that persist across git commits and refactorings. The system uses tree-sitter for AST-based code parsing and intelligent anchor matching.

## Core Design Principles

1. **Non-intrusive**: Notes are stored separately from source code
2. **Resilient**: Notes survive code movement and refactoring via AST anchoring
3. **Flexible**: Open metadata system for extensibility
4. **Personal & Shareable**: Individual collections can be private or shared

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│              CLI Interface (clap)               │
└─────────────────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌─────────┐    ┌──────────────┐    ┌─────────────┐
│  Models │    │   Parsers    │    │   Storage   │
│         │◄───│              │    │             │
│ - Note  │    │ - CodeParser │    │ - JSON      │
│ - Anchor│    │ - AnchorBuild│    │ - File I/O  │
└─────────┘    └──────────────┘    └─────────────┘
    │                  │
    └──────────┬───────┘
               ▼
      ┌────────────────┐
      │ Git Integration│
      │                │
      │ - Migration    │
      │ - Commit Track │
      └────────────────┘
               │
               ▼
      ┌────────────────┐
      │   Git Repo     │
      └────────────────┘
```

## Component Details

### 1. Models (`src/models/`)

#### CodeAnchor
Represents a precise location in code using AST properties:

```rust
pub struct CodeAnchor {
    file_path: String,           // Relative to repo root
    node_kind: String,            // AST node type
    node_text: String,            // Text content
    line_number: usize,           // Quick lookup
    column: usize,
    ast_path: Vec<(String, usize)>, // Path from root
    context: Vec<String>,         // Parent node kinds
}
```

**Key Methods**:
- `match_confidence()`: Calculates similarity score with another anchor (0.0-1.0)
- Uses weighted scoring: text (50%), kind (25%), AST path (15%), context (10%)

#### NoteAnchor
Combines primary anchor with alternatives for resilience:

```rust
pub struct NoteAnchor {
    primary: CodeAnchor,
    alternatives: Vec<CodeAnchor>,    // Parent & siblings
    commit_hash: String,               // Original commit
    is_valid: bool,
    migration_history: Vec<MigrationRecord>,
}
```

#### Note
The main annotation with metadata:

```rust
pub struct Note {
    id: Uuid,
    content: String,
    author: String,
    created_at: i64,
    updated_at: i64,
    anchor: NoteAnchor,
    metadata: HashMap<String, Value>,  // Extensible
    is_orphaned: bool,
}
```

#### NoteCollection
Groups related notes:

```rust
pub struct NoteCollection {
    name: String,
    description: Option<String>,
    notes: Vec<Note>,
    metadata: HashMap<String, Value>,
}
```

### 2. Parsers (`src/parsers/`)

#### CodeParser
Wrapper around tree-sitter parser:

```rust
pub struct CodeParser {
    parser: Parser,
    language: SupportedLanguage,
}
```

**Supported Languages**:
- Rust (.rs)
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)

#### AnchorBuilder
Creates anchors from tree-sitter nodes:

```rust
pub struct AnchorBuilder<'a> {
    source: &'a str,
    file_path: String,
}
```

**Key Methods**:
- `build_anchor(node)`: Create primary anchor
- `build_note_anchor(node, commit)`: Create anchor with alternatives
- `build_ast_path(node)`: Construct path from root to node

#### AnchorMatcher
Finds matching nodes in modified code:

```rust
pub struct AnchorMatcher<'a> {
    source: &'a str,
    tree: &'a Tree,
}
```

**Algorithm**:
1. Find all nodes of the same kind
2. Score each candidate using `match_confidence()`
3. Return best match above threshold (0.7)

### 3. Git Integration (`src/git/`)

#### GitRepo
Git operations wrapper using libgit2:

```rust
pub struct GitRepo {
    repo: Repository,
}
```

**Key Methods**:
- `current_commit_hash()`: Get HEAD
- `file_content_at_commit()`: Read file at specific commit
- `commits_between()`: List commits in range
- `file_changed_between()`: Check if file modified

#### NoteMigrator
Migrates notes across commits:

```rust
pub struct NoteMigrator {
    repo: GitRepo,
}
```

**Migration Algorithm**:
1. Get commits between note's commit and HEAD
2. For each commit where file changed:
   - Parse file at that commit
   - Find matching node using AnchorMatcher
   - Update anchor if confidence > 0.7
   - Mark orphaned if no match found
3. Record migration history

**Optimization**: Only re-parse when file actually changed

### 4. Storage (`src/storage/`)

#### NoteStorage
File-based persistence:

```rust
pub struct NoteStorage {
    root_path: PathBuf,  // .code-notes/
}
```

**Directory Structure**:
```
.code-notes/
├── collections.json      # Collection metadata (without notes)
└── notes/
    ├── uuid1.json       # Individual note files
    ├── uuid2.json
    └── ...
```

**Design Rationale**:
- Separate note files for efficient partial updates
- Collection metadata stored separately
- JSON for human readability and tool compatibility

### 5. CLI (`src/cli/`)

Command-line interface using clap:

```
Commands:
  init              Initialize in repository
  add               Add note
  list              List notes
  view              View specific note
  update            Update note content
  delete            Delete note
  migrate           Migrate notes to current commit
  collections       List collections
  create-collection Create new collection
  export            Export collection to bundle
  import            Import collection from bundle
  orphaned          Show orphaned notes
```

## Data Flow Examples

### Adding a Note

```
User -> CLI (add command)
  -> Parse file with CodeParser
  -> Find node at position
  -> AnchorBuilder creates CodeAnchor + NoteAnchor
  -> Create Note with anchor
  -> Load/create NoteCollection
  -> NoteStorage saves collection + note
```

### Migrating Notes

```
User -> CLI (migrate command)
  -> Load NoteCollection from NoteStorage
  -> NoteMigrator processes each note:
     -> Get commits between note.commit and HEAD
     -> For each commit:
        -> If file changed:
           -> GitRepo gets file content at commit
           -> CodeParser parses file
           -> AnchorMatcher finds best match
           -> Update anchor if confidence > 0.7
     -> Save migration history
  -> NoteStorage saves updated collection
```

### Viewing Notes for a File

```
User -> CLI (list command with file)
  -> NoteStorage loads collection
  -> Filter notes by file_path
  -> Display note metadata and content
```

## Extension Points

### 1. Metadata System
Notes have open `HashMap<String, Value>` for extensions:
- Tags: `{"tags": ["bug", "security"]}`
- Priority: `{"priority": 5}`
- Trail ID: `{"trail_id": "auth-flow"}`
- Linked Issues: `{"issues": ["GH-123"]}`

### 2. Custom Parsers
Add language support by:
1. Add tree-sitter-{lang} dependency
2. Add variant to `SupportedLanguage` enum
3. Implement `tree_sitter_language()` for the language

### 3. Storage Backends
Currently file-based, but interface allows:
- Database storage (SQLite, Postgres)
- Cloud storage (S3, Git LFS)
- Distributed systems (CRDT-based sync)

### 4. Alternative UIs
- VS Code extension (read from `.code-notes/`)
- Web interface (serve notes via HTTP)
- Git hooks (inject as comments on checkout)

## Performance Considerations

### Parsing Performance
- Tree-sitter is fast (milliseconds for typical files)
- Parsing only happens when needed (add, migrate)
- Could cache parsed trees for frequently accessed files

### Storage Performance
- Individual note files allow partial updates
- Collections.json is small (no note content)
- For large repos, could index by file path

### Migration Performance
- Only parses commits where file changed
- Could parallelize migration across notes
- Could checkpoint partial migrations

## Future Enhancements

### 1. Identifier Rename Tracking
Track variable/function renames:
- Monitor AST diffs for rename patterns
- Update anchor text automatically
- Maintain confidence in renamed nodes

### 2. Semantic Analysis
Enhanced matching using semantic information:
- Function signature matching
- Type information
- Control flow analysis

### 3. Collaborative Features
- Shared collections with conflict resolution
- Real-time sync between team members
- Review workflow for note changes

### 4. IDE Integration
- Inline display of notes in editors
- Quick add/edit UI
- Visual migration confidence indicators

### 5. Git Hooks
- Pre-commit: Verify note validity
- Post-checkout: Auto-migrate notes
- Pre-push: Bundle notes for sharing

## Security Considerations

### Personal Information
- Notes might contain sensitive info
- Default: keep `.code-notes/` out of version control
- Provide encryption option for local storage

### Malicious Input
- Validate all file paths (prevent directory traversal)
- Limit note content size
- Sanitize metadata values

### Repository Access
- Only read repository content (no writes)
- Use libgit2 safely (no shell injection)
- Respect git permissions

## Testing Strategy

### Unit Tests
- CodeAnchor matching algorithm
- AnchorBuilder path construction
- NoteMigrator confidence scoring

### Integration Tests
- Full add -> migrate -> view flow
- Cross-commit migration
- Export/import round-trip

### Real-world Testing
- Test on actual repositories with history
- Measure migration success rates
- Profile performance on large codebases

## Deployment

### Installation
```bash
cargo install code-notes  # Future: publish to crates.io
```

### Configuration
Optional `.code-notes/config.toml`:
```toml
[migration]
confidence_threshold = 0.7
auto_migrate = true

[storage]
backend = "file"
```

## Metrics & Observability

Track:
- Migration success rates
- Average confidence scores
- Parse times by language
- Orphan rate over time

Use for:
- Tuning confidence threshold
- Identifying problematic patterns
- Improving matching algorithm
