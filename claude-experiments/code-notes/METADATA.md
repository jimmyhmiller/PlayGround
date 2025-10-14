# Metadata Feature Guide

## Overview

Code Notes supports adding rich, extensible metadata to notes. Metadata is stored as a flexible JSON object, allowing you to tag, categorize, prioritize, and link notes in any way that fits your workflow.

## Adding Metadata

### With the `add` Command

```bash
code-notes add \
  --file src/auth.rs \
  --line 42 \
  --column 0 \
  --content "Security issue: Plaintext password comparison" \
  --author "Security Team" \
  --collection security \
  --metadata '{"tags":["bug","security"],"priority":9,"severity":"critical","linked_issues":["SEC-123"]}'
```

### With the `capture` Command

The `capture` command applies the same metadata to all captured `@note:` markers in a file:

```bash
code-notes capture src/config.rs \
  --author "Onboarding Team" \
  --collection onboarding \
  --metadata '{"tags":["onboarding","config"],"audience":"junior-devs","trail_id":"getting-started"}'
```

### With Inline `@meta:` Markers 🆕

**NEW:** You can now add metadata directly in your code using `@meta:` markers!

```rust
// @note: This function has a security vulnerability
// @meta: {"tags":["bug","security"],"priority":9,"severity":"critical"}
pub fn authenticate(username: &str, password: &str) -> bool {
    username == "admin" && password == "secret"
}
```

When you run `capture`, the metadata will be automatically extracted and attached to the note:

```bash
code-notes capture src/auth.rs --author "Security Team" --collection security
```

**Key Features:**
- ✅ Each note can have its own metadata
- ✅ `@meta:` line must come after `@note:` and before the code
- ✅ Uses standard JSON format
- ✅ Works with all supported languages (Rust, Python, JS, TS, etc.)

**Metadata Merging:**
- CLI `--metadata` provides defaults for ALL notes
- Inline `@meta:` overrides or adds to CLI metadata per-note
- If both specify the same key, inline wins

```rust
// Example with CLI defaults:
// $ code-notes capture ... --metadata '{"audience":"all-devs","priority":5}'

// @note: Critical bug
// @meta: {"priority":9}  ← This overrides priority to 9
// Result: {"audience":"all-devs","priority":9}

// @note: Minor enhancement
// (no @meta: line)
// Result: {"audience":"all-devs","priority":5}  ← Uses CLI default
```

## Metadata Schema

Metadata is completely flexible - you define the structure. Here are some common patterns:

### Tags
```json
{
  "tags": ["bug", "security", "high-priority"]
}
```

### Priority & Severity
```json
{
  "priority": 9,
  "severity": "critical"
}
```

### Learning Trails
```json
{
  "trail_id": "auth-flow",
  "step": 1,
  "difficulty": "intermediate",
  "audience": "junior-devs"
}
```

### Issue Linking
```json
{
  "linked_issues": ["BUG-456", "SEC-123"],
  "pr_url": "https://github.com/org/repo/pull/123"
}
```

### Code Review
```json
{
  "reviewer": "alice",
  "review_date": "2024-01-15",
  "status": "approved",
  "comments": "LGTM with minor suggestions"
}
```

### Technical Debt
```json
{
  "effort": "medium",
  "impact": "high",
  "category": "refactoring",
  "estimated_hours": 8
}
```

### Performance Notes
```json
{
  "baseline_ms": 150,
  "target_ms": 50,
  "profiled": true,
  "optimization_ideas": ["use_cache", "async_io"]
}
```

### Documentation
```json
{
  "category": "api",
  "audience": "all-devs",
  "related_docs": ["api-guide.md", "architecture.md"],
  "last_reviewed": "2024-01-10"
}
```

## Viewing Metadata

Use the `view` command to see all metadata for a note:

```bash
code-notes view <note-id>
```

Output includes:
```
Note ID: 12345678-1234-1234-1234-123456789abc
File: src/auth.rs:42
Author: Security Team
Collection: security
Created: 1704067200
Updated: 1704067200
Commit: abc123def

Content:
Security issue: Plaintext password comparison

Metadata:
{
  "linked_issues": [
    "SEC-123"
  ],
  "priority": 9,
  "severity": "critical",
  "tags": [
    "bug",
    "security"
  ]
}
```

## Use Cases

### 1. Bug Tracking
Track bugs with severity, priority, and linked issues:
```bash
--metadata '{"tags":["bug"],"priority":8,"severity":"high","linked_issues":["BUG-456"],"assigned_to":"alice"}'
```

### 2. Onboarding Trails
Create learning paths for new developers:
```bash
--metadata '{"trail_id":"getting-started","step":1,"audience":"junior-devs","difficulty":"beginner"}'
```

### 3. Code Review Comments
Document review feedback:
```bash
--metadata '{"tags":["code-review"],"reviewer":"bob","date":"2024-01-15","status":"needs-fixing","blocking":true}'
```

### 4. Architecture Decisions
Track architectural notes and decisions:
```bash
--metadata '{"tags":["architecture","decision"],"adr_number":"ADR-042","decision_date":"2024-01-10"}'
```

### 5. Performance Optimization
Document performance issues and optimizations:
```bash
--metadata '{"tags":["performance"],"baseline_ms":150,"target_ms":50,"profiled":true,"cpu_bound":true}'
```

### 6. Security Audits
Mark security-sensitive code:
```bash
--metadata '{"tags":["security","audit"],"audit_date":"2024-01-15","auditor":"security-team","approved":false}'
```

## Metadata Persistence

- ✅ Metadata is stored with each note in JSON format
- ✅ Metadata is preserved across git commits and migrations
- ✅ Metadata is included in collection exports
- ✅ Metadata is imported when restoring collections
- ✅ Metadata can be viewed individually or as part of note listings

## Advanced Patterns

### Combining Multiple Metadata Types

```json
{
  "tags": ["bug", "security", "urgent"],
  "priority": 9,
  "severity": "critical",
  "linked_issues": ["SEC-123", "BUG-456"],
  "assigned_to": "security-team",
  "due_date": "2024-01-20",
  "estimated_hours": 16,
  "impact": "high",
  "affected_users": 5000,
  "hotfix_required": true
}
```

### Team-Specific Metadata

```json
{
  "team": "backend",
  "sprint": "2024-Q1-S3",
  "story_points": 8,
  "epic": "EPIC-42",
  "component": "authentication",
  "requires_migration": true
}
```

### AI Agent Metadata

```json
{
  "ai_generated": true,
  "confidence": 0.95,
  "model": "claude-3.5",
  "suggested_action": "refactor",
  "auto_fixable": false
}
```

## Tips

1. **Be Consistent**: Establish team conventions for metadata keys
2. **Use Arrays for Tags**: `"tags": ["a", "b"]` is searchable and flexible
3. **Include Dates**: Add ISO 8601 dates for time-based filtering
4. **Link External Systems**: Include URLs, issue IDs, PR numbers
5. **Document Your Schema**: Maintain a metadata schema document for your team

## Inline `@meta:` Syntax

### Basic Syntax

```rust
// @note: Your note content here
// @meta: {"key":"value","tags":["tag1","tag2"]}
```

### Python Example
```python
# @note: This needs validation
# @meta: {"tags":["enhancement"],"priority":6}
def process_data(input):
    return input
```

### JavaScript Example
```javascript
// @note: Add error handling
// @meta: {"tags":["bug"],"severity":"medium","assigned_to":"alice"}
function handleRequest(req) {
    return req.process();
}
```

### Multiline Notes with Metadata
```rust
// @note: This function needs refactoring
// It's too complex and hard to maintain
// Consider breaking into smaller functions
// @meta: {"tags":["refactor","tech-debt"],"effort":"large","priority":7}
fn complex_function() {
    // ...
}
```

### Complex Metadata
```rust
// @note: Performance bottleneck identified
// @meta: {"tags":["performance"],"baseline_ms":150,"target_ms":50,"profiled":true,"linked_issues":["PERF-42"],"assigned_to":"performance-team"}
fn slow_operation() {
    // ...
}
```

## JSON Syntax Tips

### Valid JSON
```bash
--metadata '{"tags":["bug"],"priority":5}'  # ✓ Correct
```

### Common Mistakes
```bash
--metadata {"tags":["bug"],"priority":5}    # ✗ Missing quotes (shell will parse)
--metadata '{"tags": ["bug"], "priority": 5}'  # ✓ Spaces are fine
--metadata "{\"tags\":[\"bug\"]}"           # ✓ Escaped quotes also work
```

### Multi-line (in scripts)
```bash
METADATA=$(cat <<'EOF'
{
  "tags": ["bug", "security"],
  "priority": 9,
  "severity": "critical"
}
EOF
)

code-notes add ... --metadata "$METADATA"
```

## Future Enhancements

Planned features for metadata:
- [ ] Metadata search and filtering
- [ ] Metadata validation schemas
- [ ] Metadata templates
- [ ] Metadata editing via CLI
- [ ] Metadata-based reports and analytics
- [ ] Metadata migration utilities

## Quick Reference

### CLI Metadata
```bash
code-notes add ... --metadata '{"tags":["bug"],"priority":8}'
code-notes capture ... --metadata '{"audience":"junior-devs"}'
```

### Inline Metadata
```rust
// @note: Your note here
// @meta: {"tags":["bug"],"priority":8}
```

### Metadata Hierarchy
1. Inline `@meta:` (highest priority - per-note)
2. CLI `--metadata` (default for all notes)
3. No metadata (empty metadata object)

## See Also

- [README.md](README.md) - General usage guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
- [CLAUDE.md](CLAUDE.md) - Development notes
- [examples/metadata_demo.sh](examples/metadata_demo.sh) - CLI metadata demonstration
- [examples/inline_metadata_demo.sh](examples/inline_metadata_demo.sh) - Inline @meta: demonstration
