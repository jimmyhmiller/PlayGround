# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-10-31)

**Phase 3 - Expressions:** 50% Complete

✅ **Implemented:**
- All primitive expressions (numbers, strings, booleans, identifiers)
- Binary operators (15 operators, left-associative, NO precedence)
- Parenthesized expressions `(1 + 2)`
- Function application `f(x, y, z)`
- Chained calls `f(x)(y)`
- Whitespace-sensitive parsing: `f(x)` vs `f (x)`
- **Array expressions** `[1, 2, 3]` - NEW! ✨
- **Dot access** `obj.field`, `obj.field1.field2` - NEW! ✨
- **Chained postfix operators** `obj.foo().bar()` - NEW! ✨

🎯 **Next Tasks:** Bracket access, objects, tuples

## 🚀 Quick Start

```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2

# Run tests
cargo test

# Run parser tests only
cargo test --test parser_tests

# Debug tokens
DEBUG_TOKENS=1 cargo test test_name
```

## 📚 Essential Documentation

**Start here for next tasks:**
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Comprehensive guide with prioritized tasks, implementation templates, and examples
- **[README.md](README.md)** - Project overview and quick reference
- **[HANDOFF_CHECKLIST.md](HANDOFF_CHECKLIST.md)** - Verification and quick reference

**Implementation completed:**
- **[PHASE3_PARENS_AND_APPS_COMPLETE.md](PHASE3_PARENS_AND_APPS_COMPLETE.md)** - Latest work: parentheses & function application
- **[PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)** - Primitives and binary operators
- **[PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)** - Foundation

**Project planning:**
- **[PARSER_PLAN.md](PARSER_PLAN.md)** - Overall roadmap and phase breakdown
- **[OPERATOR_PRECEDENCE.md](OPERATOR_PRECEDENCE.md)** - Important: Pyret has NO precedence!

## 📁 Key Files

```
src/
├── parser.rs       (887 lines)   - Parser implementation
├── ast.rs          (1,350 lines) - All AST node types
├── tokenizer.rs    (1,346 lines) - Complete tokenizer
└── error.rs        (73 lines)    - Error types

tests/
└── parser_tests.rs (773 lines)   - 35 tests, all passing ✅
```

## 🔑 Key Concepts

**Whitespace Sensitivity:**
- `f(x)` → `ParenNoSpace` → Direct function call
- `f (x)` → `ParenSpace` → Function applied to parenthesized expr

**No Operator Precedence:**
- `2 + 3 * 4` = `(2 + 3) * 4` = `20` (NOT 14)
- All binary operators have equal precedence
- Strictly left-associative

**Implementation Pattern:**
1. Add `parse_foo()` method in `src/parser.rs` Section 6
2. Update `parse_prim_expr()` or `parse_binop_expr()`
3. Add location extraction for new expr type
4. Add tests in `tests/parser_tests.rs`
5. Run `cargo test`

## 🎯 Next Priority Tasks

See [NEXT_STEPS.md](NEXT_STEPS.md) for detailed guides:

1. **Object expressions** `{ field: value }` - 2-3 hours
2. **Array expressions** `[1, 2, 3]` - 1 hour
3. **Dot access** `obj.field` - 1-2 hours
4. **Bracket access** `arr[0]` - 1-2 hours
5. **Tuple expressions** `{1; 2; 3}` - 1-2 hours

## ✅ Tests Status

```
35/35 parser tests passing
46/46 total tests passing
```

## 💡 Quick Tips

### First Time Here?
1. Read [NEXT_STEPS.md](NEXT_STEPS.md) - Most comprehensive guide
2. Look at `tests/parser_tests.rs:203-512` - See test patterns
3. Look at `src/parser.rs:462-520` - See recent implementations
4. Pick a task from NEXT_STEPS.md and start!

### Debugging
```bash
# See what tokens are generated
DEBUG_TOKENS=1 cargo test test_name

# Run specific test
cargo test test_parse_simple_function_call

# Watch for changes (if you have cargo-watch)
cargo watch -x test
```

### Common Patterns

**Parse primary expression:**
```rust
fn parse_foo_expr(&mut self) -> ParseResult<Expr> {
    let start = self.expect(TokenType::FooStart)?;
    let contents = self.parse_expr()?;
    let end = self.expect(TokenType::FooEnd)?;

    Ok(Expr::SFoo {
        l: self.make_loc(&start, &end),
        contents: Box::new(contents),
    })
}
```

**Parse comma-separated list:**
```rust
let items = self.parse_comma_list(|p| p.parse_expr())?;
```

## 🚨 Important Reminders

1. **No operator precedence** - Pyret design choice, don't add it!
2. **Whitespace matters** - Trust the token types from tokenizer
3. **Update location extraction** - Add new Expr types to match statements
4. **Test edge cases** - Empty, single item, nested, mixed expressions
5. **Follow existing patterns** - Look at similar code for consistency

## 📞 Reference Materials

- **Pyret Grammar:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
- **AST Definitions:** `src/ast.rs:292-808`
- **Current Parser:** `src/parser.rs:188-520` (Section 6)
- **Test Examples:** `tests/parser_tests.rs`

## 🎉 Ready to Code!

The codebase is clean, well-tested, and ready for the next features. Start with [NEXT_STEPS.md](NEXT_STEPS.md) and pick your task!

---

**Last Updated:** 2025-10-31
**Tests:** 24/24 parser tests passing, 35/35 total
**Next Milestone:** Complete Phase 3 expression parsing

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

View a bug:
```bash
bug-tracker view <BUG_ID>
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

**View a bug by ID:**
```bash
bug-tracker view curious-elephant
```

**Enable AI quality validation:**
```bash
bug-tracker add --title "Bug title" --description "Bug details" --validate
```

The `--validate` flag triggers AI-powered quality checking to ensure bug reports contain sufficient information before recording.
