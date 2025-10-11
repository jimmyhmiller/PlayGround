# Code Notes - Claude Project Documentation

## Original Prompt

> Okay, we are making something a bit unique. The idea is that code comments are great, but the fact that they live with the source code is a little less great. There are a few problems with them. First they can make reading the code a bit hard. They take up space. Second, they are universal, I can't make my own notes. Third, they are singular, maybe I want different comments for different people (imagine a new developer getting up to speed, or an AI agent needing more context).
>
> We are going to be build a system that lets us layer on notes to code. But, we have some complications. First, the notes need to be able to work across git commits. Imagine that I have a function with a variable isAuthedProperly on line 42. In the next commit, it may be on line 50, but I still want that comment attached to the identifier. So we need a way to attach comments to particular items. Ideally, this works even if that identifier has been renamed, but we can punt on that for now.
>
> The general idea I'm thinking is that we might want to use tree sitter for parsing and referring to parts of our code so that we can attach notes to them. Notes need to know the git hash for which they were added. We need to be able to have a process that tries to process each commit since the note was added to make sure the note can be pushed forward. If it cannot, that is fine, we just consider this note orphaned.
>
> Notes also need some open metadata so we can layer things like trails on top of notes. Notes need to be stored with data as a major concern so we can display these notes in various ways. The first interface for this will probably be a command line interface where you can view and add notes. But we also are going to want a tool where as a git hook you can add and remove the notes after checkout and before commit.
>
> We will also need a way to bundle up these notes and share them.

## Project Overview

Code Notes is a layered annotation system for code that persists across git commits and refactorings. It addresses the fundamental limitations of traditional in-source comments by storing notes separately while maintaining precise linkage to code through AST-based anchoring.

## What We've Built

### Core Architecture

The system is implemented in Rust and consists of five main components:

1. **Models** (`src/models/`)
   - `Note`: Contains content, author, timestamps, anchor, and extensible metadata
   - `CodeAnchor`: Precisely locates code using tree-sitter AST properties
   - `NoteAnchor`: Combines primary anchor with alternatives for resilience
   - `NoteCollection`: Groups related notes together

2. **Parsers** (`src/parsers/`)
   - `CodeParser`: Wrapper around tree-sitter supporting Rust, Python, JavaScript, TypeScript
   - `AnchorBuilder`: Creates rich anchors from tree-sitter AST nodes
   - `AnchorMatcher`: Finds matching nodes across different code versions using confidence scoring

3. **Git Integration** (`src/git/`)
   - `GitRepo`: Handles git repository operations via libgit2
   - `NoteMigrator`: Migrates notes across commits with intelligent matching

4. **Storage** (`src/storage/`)
   - Centralized file-based storage in `~/.code-notes/`
   - Smart path-based conflict resolution for project directories
   - JSON format for human readability and tool compatibility
   - Project index maps canonical paths to storage locations

5. **CLI** (`src/cli/`)
   - Full-featured command-line interface
   - Commands: init, add, list, view, update, delete, migrate, collections, export, import, orphaned

### Key Design Decisions

#### 1. AST-Based Anchoring

Instead of using simple line numbers, notes are anchored using tree-sitter AST properties:
- **Node kind** (e.g., "identifier", "function_definition")
- **Node text content** (exact match is strongest signal)
- **AST path** from root to node (e.g., `[(module, 0), (function_definition, 2), (identifier, 1)]`)
- **Context** (parent node types)
- **Line/column** (for quick lookup and user reference)

This makes notes resilient to code movement and refactoring.

#### 2. Migration Algorithm

When code changes across commits, notes are automatically migrated:

1. Traverse commits from note's original commit to HEAD
2. For each commit where the file changed:
   - Parse the file at that commit using tree-sitter
   - Find all nodes of the same kind as the anchor
   - Score each candidate using weighted confidence:
     - Exact text match: 50% weight
     - Node kind match: 25% weight (required)
     - AST path similarity: 15% weight
     - Context similarity: 10% weight
   - Update anchor if confidence > 0.7 threshold
   - Mark as orphaned if no good match found
3. Record full migration history for debugging

This approach balances precision with flexibility.

#### 3. Centralized Storage

Notes are stored globally in `~/.code-notes/` rather than in each repository:

```
~/.code-notes/
├── project_index.json              # Maps repos to storage dirs
├── my-project/                     # Smart conflict resolution
│   ├── collections.json
│   └── notes/
│       ├── <uuid1>.json
│       └── <uuid2>.json
└── another-project/
    └── ...
```

**Benefits**:
- Single location to backup all notes
- No git conflicts from note changes
- Easy to manage across multiple machines
- Notes are truly personal/per-developer

**Smart Path Resolution**:
- Starts with project name (e.g., "code-notes")
- Adds parent directories if conflict (e.g., "claude-experiments_code-notes")
- Falls back to hash suffix if still conflicting

#### 4. Smart File Path Resolution

The CLI intelligently resolves file paths using a three-strategy approach in `src/cli/commands.rs:resolve_file_path()`:

1. **Try relative to current directory**: If you're in a subdirectory and specify a file, it checks there first
2. **Try relative to repository root**: Falls back to checking from the repo root
3. **Search entire repository**: If not found, searches for any file matching that name

If multiple files match, it provides helpful error messages with all matches:
```
Multiple files named 'config.rs' found:
  src/config.rs
  tests/config.rs
Please specify a more complete path.
```

This means you can use simple filenames like `code-notes list sample_code.rs` from anywhere in your project.

### 5. Open Metadata System

Notes have a `HashMap<String, serde_json::Value>` for extensibility:

```json
{
  "tags": ["bug", "security"],
  "priority": 5,
  "trail_id": "auth-flow",
  "linked_issues": ["GH-123"],
  "audience": "junior-devs"
}
```

This enables future features like:
- Learning trails (sequence of notes to understand a feature)
- Priority/urgency indicators
- Linking to external issues/docs
- Audience-specific note collections

## Implementation Details

### Tree-Sitter Integration

We use tree-sitter for robust code parsing:

```rust
pub struct CodeParser {
    parser: Parser,
    language: SupportedLanguage,
}

// Currently supports: Rust, Python, JavaScript, TypeScript
```

When creating an anchor:
1. Parse the file to get the AST
2. Find the node at the specified line/column
3. Build the AST path from root to node
4. Capture parent context
5. Store alternatives (parent and sibling nodes) for redundancy

### Confidence Scoring

The matching algorithm uses weighted scoring to find the best anchor location in modified code:

```rust
pub fn match_confidence(&self, candidate: &CodeAnchor) -> f64 {
    let mut score = 0.0;

    // Exact text match (strongest signal)
    if self.node_text == candidate.node_text {
        score += 10.0;  // 50% weight
    }

    // Node kind must match
    if self.node_kind == candidate.node_kind {
        score += 5.0;   // 25% weight
    } else {
        return 0.0;     // Hard requirement
    }

    // AST path similarity
    score += ast_path_similarity(...) * 3.0;  // 15% weight

    // Context similarity
    score += context_similarity(...) * 2.0;   // 10% weight

    score / 20.0  // Normalize to 0.0-1.0
}
```

Minimum threshold: 0.7 (configurable in future)

### Collections

Notes are organized into collections for different purposes:

- **personal**: Your own notes
- **onboarding**: Notes for new team members
- **code-review**: Review comments and discussions
- **bugs**: Debugging notes and context
- **architecture**: High-level design notes

Collections can be exported as JSON bundles and shared with team members.

## What's Working

✅ **Implemented**:
- AST-based anchoring with tree-sitter
- Cross-commit migration with confidence scoring
- Centralized storage with smart conflict resolution
- Full CLI interface (add, list, view, update, delete, migrate)
- Smart file path resolution (no need for full paths)
- Collection management (create, export, import)
- Orphan detection
- Git integration (commit tracking, file history)
- Support for 4 languages (Rust, Python, JS, TS)
- Comprehensive documentation

✅ **Tested**:
- Build succeeds with no errors
- Note creation and storage
- Migration algorithm
- Path conflict resolution
- Demo script works end-to-end

## What's Punted (For Now)

❌ **Not Yet Implemented**:
- **Identifier rename tracking**: Original prompt acknowledged this could be punted
- **Git hooks**: Mentioned in original prompt but not yet implemented
- **IDE integration**: Not part of initial scope
- **Collaborative sync**: No multi-user support yet
- **More languages**: Only 4 languages currently supported

## Usage Examples

### Basic Workflow

```bash
# Initialize in a repository
cd my-project
code-notes init
# Output shows where notes are stored: ~/.code-notes/my-project/

# Add a note to a function
code-notes add \
  --file src/auth.rs \
  --line 42 \
  --column 8 \
  --content "This handles JWT validation. Uses RS256 algorithm." \
  --author "Alice" \
  --collection onboarding

# List all notes
code-notes list --collection onboarding

# After making commits
code-notes migrate --collection onboarding
# Migration report shows success rate and orphaned notes
```

### Advanced Features

```bash
# Create a trail for understanding authentication
code-notes create-collection "auth-trail" \
  --description "Follow these notes to understand our auth system"

# Add notes with metadata (via JSON editing)
# Notes can have: tags, trail_id, priority, audience, etc.

# Export for sharing
code-notes export auth-trail --output auth-trail.json

# Teammate imports it
code-notes import auth-trail.json

# View orphaned notes (where migration failed)
code-notes orphaned --collection onboarding
```

## File Structure

```
code-notes/
├── src/
│   ├── models/
│   │   ├── mod.rs
│   │   ├── note.rs              # Note and NoteCollection
│   │   └── anchor.rs            # CodeAnchor and NoteAnchor
│   ├── parsers/
│   │   ├── mod.rs
│   │   ├── tree_sitter_parser.rs  # Tree-sitter wrapper
│   │   └── anchor_builder.rs      # Anchor creation and matching
│   ├── git/
│   │   ├── mod.rs
│   │   ├── repo.rs              # Git operations
│   │   └── migration.rs         # Note migration logic
│   ├── storage/
│   │   ├── mod.rs
│   │   └── file_storage.rs      # JSON file storage
│   ├── cli/
│   │   ├── mod.rs
│   │   └── commands.rs          # CLI implementation
│   └── main.rs
├── examples/
│   ├── sample_code.rs           # Demo code
│   └── demo.sh                  # Usage demonstration
├── Cargo.toml
├── README.md                    # User documentation
├── ARCHITECTURE.md              # Technical deep-dive
└── claude.md                    # This file

User's home directory:
~/.code-notes/
├── project_index.json
└── <project-dirs>/
    ├── collections.json
    └── notes/*.json
```

## Design Philosophy

### 1. Data-First Storage

Notes are stored as individual JSON files with human-readable structure. This enables:
- Easy backup and versioning
- Tool integration (editors, scripts, web UIs)
- Future database migration if needed
- Direct file inspection and debugging

### 2. Fail Gracefully

The migration algorithm doesn't require perfection:
- Notes become "orphaned" rather than lost
- Orphaned notes can still be viewed and searched
- Users can manually re-anchor orphaned notes
- Migration history provides debugging info

### 3. Extensibility Through Metadata

Rather than hardcoding features, the open metadata system allows:
- Custom categorization schemes
- Integration with external tools
- Experimentation with new features
- User-defined workflows

### 4. Progressive Enhancement

Start simple, add complexity as needed:
- Line/column for quick reference
- AST anchoring for resilience
- Migration for commit traversal
- Collections for organization
- Metadata for advanced features

## Future Enhancements

### Short Term
- [ ] Metadata editing via CLI
- [ ] Search by content/author/metadata
- [ ] Better orphan recovery (suggest re-anchoring)
- [ ] Configuration file for threshold tuning

### Medium Term
- [ ] Git hooks (post-checkout, pre-commit)
- [ ] VS Code extension for inline display
- [ ] Support for more languages (Go, C++, Java)
- [ ] Identifier rename tracking (using git diff patterns)

### Long Term
- [ ] Web UI for browsing notes
- [ ] Collaborative sync (CRDT-based)
- [ ] AI-assisted note generation
- [ ] Learning trails with guided navigation

## Lessons Learned

### What Worked Well

1. **Tree-sitter choice**: AST-based anchoring is much more robust than line numbers
2. **Centralized storage**: Avoids git conflicts and simplifies management
3. **Confidence scoring**: Weighted approach balances different signal strengths
4. **JSON format**: Human-readable storage aids debugging
5. **Modular design**: Clean separation between parsing, storage, git, and CLI

### Challenges Overcome

1. **Tree-sitter API changes**: Had to update from constants to functions
2. **Path conflict resolution**: Smart incremental approach avoids full path mirroring
3. **Migration complexity**: Balancing performance (only parse when needed) with correctness
4. **Lifetime management**: Rust tree-sitter nodes require careful lifetime handling

### What Would We Do Differently

1. **Earlier testing**: Should have built demo earlier to validate UX
2. **Configuration**: Should have added config file from start
3. **More languages**: Should have designed for easier language addition
4. **Metadata UI**: Need better way to set metadata than JSON editing

## Testing Strategy

### Current Testing

- [x] Manual testing via demo script
- [x] Basic functionality (add, list, view)
- [x] Storage path resolution
- [x] Git repository detection

### Needed Testing

- [ ] Unit tests for confidence scoring
- [ ] Integration tests for migration
- [ ] Edge cases (empty files, binary files)
- [ ] Performance tests (large repos, many notes)
- [ ] Cross-platform testing (Windows, Linux)

## Dependencies

Key external dependencies:

```toml
clap = "4.5"              # CLI framework
tree-sitter = "0.22"      # Code parsing
git2 = "0.19"             # Git integration
serde = "1.0"             # Serialization
serde_json = "1.0"        # JSON handling
uuid = "1.10"             # Note IDs
anyhow = "1.0"            # Error handling
```

Tree-sitter language grammars:
- tree-sitter-rust
- tree-sitter-python
- tree-sitter-javascript
- tree-sitter-typescript

## Contributing

This is an experimental project. Future contributors should:

1. Read ARCHITECTURE.md for technical details
2. Run the demo to understand UX
3. Add tests for new features
4. Update documentation
5. Consider backward compatibility for storage format

## Project Status

**Status**: Alpha / Proof of Concept

The core functionality is working and demonstrates the viability of AST-based note anchoring across commits. Ready for personal use and experimentation. Not yet production-ready for team collaboration.

**Next Steps**:
1. Real-world testing on actual development workflows
2. Performance profiling on large repositories
3. User feedback on CLI ergonomics
4. Metadata feature design and implementation
5. VS Code extension prototype

## Questions & Answers

### Why not use line numbers?
Line numbers change constantly. AST-based anchoring survives code movement, refactoring, and even modest code changes.

### Why centralized storage?
Keeps notes personal, avoids git conflicts, simplifies backup, enables multi-repository note management.

### What about performance?
Tree-sitter is fast (milliseconds per file). Migration only happens on demand. Could be optimized further with caching.

### Can notes handle major refactoring?
Yes, with caveats. The confidence scoring handles modest changes well. Major refactoring might orphan notes, but they're not lost—just need re-anchoring.

### How do I share notes with my team?
Export collections as JSON bundles. Could build a sync service in the future.

### What about security/privacy?
Notes are local files. Don't commit sensitive info. Could add encryption later.

## Bug Tracker

Use this tool to record bugs discovered during development. This helps track issues that need to be addressed later. Each bug gets a unique ID (goofy animal name like "curious-elephant") for easy reference and closing.

### Tool Definition

```json
{
  "name": "bug_tracker",
  "description": "Records bugs discovered during development to BUGS.md in the project root. Each bug gets a unique goofy animal name ID. Includes AI-powered quality validation.",
  "input_schema": {
    "type": "object",
    "properties": {
      "project": {
        "type": "string",
        "description": "Project root directory path"
      },
      "title": {
        "type": "string",
        "description": "Short bug title/summary"
      },
      "description": {
        "type": "string",
        "description": "Detailed description of the bug"
      },
      "file": {
        "type": "string",
        "description": "File path where bug was found"
      },
      "context": {
        "type": "string",
        "description": "Code context like function/class/module name where bug was found"
      },
      "severity": {
        "type": "string",
        "enum": ["low", "medium", "high", "critical"],
        "description": "Bug severity level"
      },
      "tags": {
        "type": "string",
        "description": "Comma-separated tags"
      },
      "repro": {
        "type": "string",
        "description": "Minimal reproducing case or steps to reproduce"
      },
      "code_snippet": {
        "type": "string",
        "description": "Code snippet demonstrating the bug"
      },
      "metadata": {
        "type": "string",
        "description": "Additional metadata as JSON string (e.g., version, platform)"
      }
    },
    "required": ["project", "title"]
  }
}
```

### Usage

Add a bug:
```bash
bug-tracker add --title <TITLE> [OPTIONS]
```

Close a bug:
```bash
bug-tracker close <BUG_ID>
```

List bugs:
```bash
bug-tracker list
```

### Examples

**Add a comprehensive bug report:**
```bash
bug-tracker add --title "Null pointer dereference" --description "Found potential null pointer access" --file "src/main.rs" --context "authenticate()" --severity high --tags "memory,safety" --repro "Call authenticate with null user_ptr" --code-snippet "if (!user_ptr) { /* missing check */ }"
```

**Close a bug by ID:**
```bash
bug-tracker close curious-elephant
```

**Enable AI quality validation:**
```bash
bug-tracker add --title "Bug title" --description "Bug details" --validate
```

The `--validate` flag triggers AI-powered quality checking to ensure bug reports contain sufficient information before recording.
