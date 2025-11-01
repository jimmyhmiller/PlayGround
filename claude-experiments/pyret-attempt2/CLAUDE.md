# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-11-01 - Latest Update)

**Phase 3 - Expressions:** Advanced features (69/81 comparison tests passing ✅ 85.2%)

✅ **Implemented & Verified:**
- ✅ All primitive expressions (numbers, strings, booleans, identifiers)
- ✅ Binary operators (15 operators, left-associative, NO precedence)
- ✅ Parenthesized expressions `(1 + 2)`
- ✅ Function application `f(x, y, z)` with multiple arguments
- ✅ Chained function calls `f(x)(y)(z)` and `f()(g())` - FIXED! ✨
- ✅ **Whitespace-sensitive parsing** `f(x)` vs `f (x)` - FIXED! ✨
  - `f(x)` = function call (ParenNoSpace)
  - `f (x)` = two separate expressions (ParenSpace stops parsing)
- ✅ **Dot access** `obj.field`, `obj.field1.field2`
  - Including keywords as field names: `obj.method()` ✨
- ✅ **Bracket access** `arr[0]`, `dict["key"]`
- ✅ **Chained postfix operators** `obj.foo().bar().baz()`
- ✅ **Construct expressions** `[list: 1, 2, 3]`, `[set: x, y]`
- ✅ **Check operators** `is`, `raises`, `satisfies`, `violates`
  - Creates `SCheckTest` expressions with proper CheckOp enum
  - Supports all variants: is, is-roughly, is-not, satisfies, violates, raises, etc.
- ✅ **Object expressions** `{ x: 1, y: 2 }` ⭐⭐⭐
  - Empty objects: `{}`
  - Data fields: `{ x: 1, y: 2 }`
  - Nested objects: `{ point: { x: 0, y: 0 } }`
  - Fields with expressions: `{ sum: 1 + 2 }`
  - Trailing comma support: `{ x: 1, y: 2, }`
  - Mutable fields: `{ ref x :: Number : 5 }` (with optional type annotations)
  - Method fields: Not yet implemented (requires function parsing)
- ✅ **Lambda expressions** `lam(x): x + 1 end` ⭐⭐⭐⭐⭐
  - Simple lambdas: `lam(): 5 end`
  - Single parameter: `lam(x): x + 1 end`
  - Multiple parameters: `lam(n, m): n > m end`
  - In function calls: `filter(lam(e): e > 5 end, [list: -1, 1])`
  - Optional type annotations: `lam(x :: Number): x + 1 end`
  - Full `Bind` and `Name` support
- ✅ **Tuple expressions** `{1; 2; 3}` ⭐⭐⭐⭐
  - Simple tuples: `{1; 3; 10}`
  - Tuples with expressions: `{13; 1 + 4; 41; 1}`
  - Nested tuples: `{151; {124; 152; 12}; 523}`
  - Tuple access: `x.{2}` (creates `STupleGet` nodes)
  - Disambiguates from objects (semicolons vs colons)
- ✅ **Block expressions** `block: ... end` ⭐⭐⭐⭐ (COMPLETE!)
  - Simple blocks: `block: 5 end`
  - Multiple statements: `block: 1 + 2 3 * 4 end`
  - Empty blocks: `block: end`
  - Nested blocks: `block: block: 1 end end`
  - Creates `SUserBlock` wrapping `SBlock` with statements
- ✅ **If expressions** `if cond: ... else: ... end` - NEW! ⭐⭐⭐⭐ (COMPLETE!)
  - Simple if/else: `if true: 1 else: 2 end`
  - Else-if chains: `if c1: e1 else if c2: e2 else: e3 end`
  - If without else: `if cond: expr end`
  - Creates `SIf` / `SIfElse` with `IfBranch` structures
  - Bodies wrapped in `SBlock` for proper statement handling
- ✅ **Complex nested expressions** - All features work together!

✅ **Recent Updates (2025-11-01 - Current Session - PART 2):**
- ✅ **If expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Added `parse_if_expr()` method in Section 7 (Control Flow)
  - Parses if/else-if/else branches with proper structure
  - Creates `IfBranch` objects with test and body expressions
  - Bodies wrapped in `SBlock` for statement sequences
  - Added `if_branch_to_pyret_json()` JSON serialization helper
  - Updated location extraction for `SIf` and `SIfElse` expressions
  - Added `TokenType::If` case to `parse_prim_expr()`
  - **69/81 comparison tests passing (85.2%)** - up from 68!
  - All if ASTs match official Pyret parser byte-for-byte ✨

✅ **Previous Updates (2025-11-01 - Current Session - PART 1):**
- ✅ **Block expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Added `parse_block_expr()` method in Section 7 (Control Flow)
  - Fixed critical tokenizer bug: "block:" now tokenized correctly as single Block token
  - Moved keyword-colon checks to beginning of `tokenize_name_or_keyword()`
  - Added `SUserBlock` JSON serialization
  - Added 4 comprehensive parser tests (simple, multiple, empty, nested blocks)
  - **68/81 comparison tests passing (84.0%)** - up from 67!
  - All block ASTs match official Pyret parser byte-for-byte ✨

✅ **Previous Updates (2025-10-31 - Previous Session - PART 4):**
- ✅ **Tuple expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Added `parse_tuple_expr()` method with semicolon-separated syntax
  - Implemented tuple access parsing `.{index}` → `STupleGet`
  - Added checkpointing/backtracking to disambiguate tuples from objects
  - Updated JSON serialization for `STuple` and `STupleGet`
  - Added `peek_ahead()` helper for lookahead parsing
  - **4 new comparison tests passing** (67 total, up from 63)
  - All tuples produce **identical ASTs** to official Pyret parser ✨
- ⚠️ **IMPORTANT:** Pyret does NOT support `[1, 2, 3]` array syntax!
  - Must use construct expression syntax: `[list: 1, 2, 3]`
  - Empty arrays: `[list:]` not `[]`

🎯 **Next Tasks:** For expressions, let bindings, method fields in objects

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
├── parser.rs       (~1,480 lines) - Parser implementation (+100 lines for if expressions)
├── ast.rs          (1,350 lines)  - All AST node types
├── tokenizer.rs    (~1,390 lines) - Complete tokenizer
└── error.rs        (73 lines)     - Error types

src/bin/
└── to_pyret_json.rs (~295 lines)  - JSON serialization (+30 lines for if expressions)

tests/
├── parser_tests.rs      (~1,340 lines) - 64 tests, all passing ✅ (100%)
└── comparison_tests.rs  (524 lines)    - 69 tests passing ✅ (85.2%), 12 ignored
```

## 🔑 Key Concepts

**Whitespace Sensitivity (CORRECTED):**
- `f(x)` → `ParenNoSpace` → Direct function call (s-app)
- `f (x)` → `ParenSpace` → Two separate expressions (NOT a function call!)
  - Parser stops after `f` and returns just the identifier
  - The `(x)` is treated as a separate statement

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

See [PARSER_GAPS.md](PARSER_GAPS.md) for detailed guides:

1. **For expressions** `for map(x from lst): ... end` (HIGHEST PRIORITY) - 3-4 hours ⭐⭐⭐
   - Functional list operations
   - 2 comparison tests waiting
2. **Let bindings** `x = value` - 1-2 hours ⭐⭐⭐
   - Variable bindings in blocks
   - Needed for `block_multiple_stmts` test
3. **Method fields in objects** `{ method _plus(self, other): ... end }` - 2-3 hours ⭐⭐⭐
   - Complete object support
   - 1 comparison test waiting

## ✅ Tests Status

```
64/64 parser tests passing (unit tests) ✅ (100%)
69/81 comparison tests passing (integration tests against official Pyret parser) ✅ (85.2%)
12/81 comparison tests ignored (features not yet implemented)
  - All passing tests produce IDENTICAL ASTs to official Pyret parser
  - Full test coverage for all implemented features
```

**Recent Additions (2025-11-01 - Current Session - PART 2):**
- ✅ **If expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Simple if/else: `if true: 1 else: 2 end`
  - Else-if chains: `if c1: e1 else if c2: e2 else: e3 end`
  - If without else: `if cond: expr end`
  - Added `parse_if_expr()` method with branch parsing
  - Added `if_branch_to_pyret_json()` JSON serialization
  - Updated location extraction for if expressions
  - **69/81 comparison tests passing (85.2%)** - up from 68!
  - All if ASTs match official Pyret parser byte-for-byte ✨

**Previous Additions (2025-11-01 - Current Session - PART 1):**
- ✅ **Block expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Simple blocks: `block: 5 end`
  - Multiple statements: `block: 1 + 2 3 * 4 end`
  - Empty blocks: `block: end`
  - Nested blocks: `block: block: 1 end end`
  - Added `parse_block_expr()` method
  - Fixed tokenizer bug for "block:" keyword-colon combinations
  - Updated JSON serialization for `SUserBlock`
  - **68/81 comparison tests passing (84.0%)** - up from 67!
  - **64/64 parser tests passing** - 4 new block tests added
  - All block ASTs match official Pyret parser byte-for-byte ✨

**Previous Additions (2025-10-31 - Previous Session - PART 4):**
- ✅ **Tuple expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Simple tuples: `{1; 3; 10}`
  - Tuples with expressions: `{13; 1 + 4; 41; 1}`
  - Nested tuples: `{151; {124; 152; 12}; 523}`
  - Tuple access: `x.{2}` (creates `STupleGet` AST nodes)
  - Added `parse_tuple_expr()` method
  - Added tuple access parsing in dot operator
  - Checkpointing/backtracking to disambiguate from objects
  - Updated JSON serialization for `STuple` and `STupleGet`
  - **4 new comparison tests passing** (67 total, up from 63)
  - All tuples produce **identical ASTs** to official Pyret parser ✨

**Previous Additions (Earlier in Session - PART 3):**
- ✅ **Lambda expressions fully implemented!** ⭐⭐⭐⭐⭐
  - Simple lambdas: `lam(): 5 end`
  - Lambdas with parameters: `lam(x): x + 1 end`, `lam(n, m): n > m end`
  - Lambdas in function calls: `filter(lam(e): e > 5 end, [list: -1, 1])`
  - Full parameter binding support with optional type annotations
  - 4 new comparison tests passing (63 total)

**Previous Additions (Earlier in Session - PART 2):**
- ✅ **Object expressions fully implemented!** ⭐⭐⭐
  - Added `parse_obj_expr()` and `parse_obj_field()` methods
  - Support for data fields, mutable fields (ref), trailing commas
  - 5 new parser tests + 5 new comparison tests
- ✅ **Check operators implemented!** `is`, `raises`, `satisfies`, `violates`
- ✅ **Keyword-as-field-name support** - `obj.method()`, `obj.fun()` now work

**Previous Additions:**
- ✅ Fixed array syntax misconception - removed incorrect `[1, 2, 3]` shorthand
- ✅ Updated all array tests to use proper `[list: ...]` construct syntax
- ✅ Construct expressions now fully working: `[list: 1, 2, 3]`, `[set: x, y]`
- ✅ Bracket access: `arr[0]`, `matrix[i][j]`
- ✅ Ultra-complex expression test (validates ALL features work together)
- ✅ Whitespace sensitivity tests (f(x) vs f (x))
- ✅ Complex nested operator tests

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
3. **⚠️ CRITICAL: Array syntax** - Pyret does NOT support `[1, 2, 3]` shorthand!
   - Must use: `[list: 1, 2, 3]` (construct expression)
   - Empty: `[list:]` not `[]`
   - This is a construct expression, not a special array syntax
   - Official Pyret parser REJECTS `[1, 2, 3]` with parse error
4. **Update location extraction** - Add new Expr types to match statements
5. **Test edge cases** - Empty, single item, nested, mixed expressions
6. **Follow existing patterns** - Look at similar code for consistency

## 📞 Reference Materials

- **Pyret Grammar:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
- **AST Definitions:** `src/ast.rs:292-808`
- **Current Parser:** `src/parser.rs:188-520` (Section 6)
- **Test Examples:** `tests/parser_tests.rs`

## 🎉 Ready to Code!

The codebase is clean, well-tested, and ready for the next features. Start with [NEXT_STEPS.md](NEXT_STEPS.md) and pick your task!

---

**Last Updated:** 2025-11-01 (Latest - If Expressions Complete!)
**Tests:** 64/64 parser tests ✅ (100%), 69/81 comparison tests ✅ (85.2%)
**Next Milestone:** For expressions, let bindings, method fields in objects

## 🎉 Recent Achievements

**Latest (2025-11-01 - Current Session - PART 2):**
- ✅ **If expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Simple if/else: `if true: 1 else: 2 end`
  - Else-if chains: `if c1: e1 else if c2: e2 else: e3 end`
  - If without else: `if cond: expr end`
  - Creates proper `IfBranch` structures with test/body
  - Bodies wrapped in `SBlock` for statement handling
  - Added `parse_if_expr()` method in Section 7 (Control Flow)
  - Added `if_branch_to_pyret_json()` JSON serialization helper
  - Updated location extraction in 5 locations
  - **69/81 comparison tests passing (85.2% coverage)** - up from 68!
  - All if ASTs match official Pyret parser byte-for-byte ✨

**Previous (2025-11-01 - Current Session - PART 1):**
- ✅ **Block expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Simple blocks with single expression: `block: 5 end`
  - Multiple statements: `block: 1 + 2 3 * 4 end`
  - Empty blocks: `block: end`
  - Nested blocks: `block: block: 1 end end`
  - Fixed critical tokenizer bug for keyword-colon combinations
  - Added `parse_block_expr()` method
  - Updated JSON serialization for `SUserBlock`
  - Added 4 comprehensive parser tests
  - **68/81 comparison tests passing (84.0% coverage)** - up from 67!
  - All block ASTs match official Pyret parser byte-for-byte ✨

**Previous (2025-10-31 - Previous Session - PART 4):**
- ✅ **Tuple expressions fully implemented!** ⭐⭐⭐⭐ (COMPLETE!)
  - Simple tuples with semicolon separation: `{1; 3; 10}`
  - Tuples with complex expressions: `{13; 1 + 4; 41; 1}`
  - Nested tuples: `{151; {124; 152; 12}; 523}`
  - Tuple element access: `x.{2}` → `STupleGet` nodes
  - Smart disambiguation from objects (semicolons vs colons)
  - Checkpointing/backtracking for correct parsing
  - Added `parse_tuple_expr()` method
  - Added `peek_ahead()` helper for lookahead
  - Updated JSON serialization for `STuple` and `STupleGet`
  - **67/81 comparison tests passing (82.7% coverage)** - up from 63!
  - All tuple ASTs match official Pyret parser byte-for-byte ✨

**Earlier (2025-10-31 - This Session - PART 3):**
- ✅ **Lambda expressions fully implemented!** ⭐⭐⭐⭐⭐ (COMPLETE!)
  - Simple lambdas without parameters
  - Single and multiple parameter support
  - Parameter bindings with optional type annotations
  - Lambdas used as function arguments
  - Proper SBlock wrapping for lambda bodies
  - Added `parse_lambda_expr()` and `parse_bind()` methods
  - Updated JSON serialization for SLam, SBlock, Bind types
  - Fixed JSON field naming: `"id"` → `"name"` in bindings
  - Added missing `"check-loc"` field for compatibility
  - **63/81 comparison tests passing (77.8% coverage)** - up from 59!
  - All lambda ASTs match official Pyret parser byte-for-byte ✨

**Earlier (2025-10-31 - This Session - PART 2):**
- ✅ **Object expressions fully implemented!** ⭐⭐⭐ (Highest priority feature)
  - Empty objects: `{}`
  - Data fields: `{ x: 1, y: 2 }`
  - Nested objects: `{ point: { x: 0, y: 0 } }`
  - Fields with expressions: `{ sum: 1 + 2, product: 3 * 4 }`
  - Trailing comma support: `{ x: 1, y: 2, }` (grammar-compliant)
  - Mutable fields: `{ ref x :: Number : 5 }` with optional type annotations
  - Added `parse_obj_expr()` and `parse_obj_field()` methods
  - Updated JSON serialization for `Member` types (SDataField, SMutableField)
  - 5 new parser tests + 5 new comparison tests (all passing)
- ✅ **Fixed all failing comparison tests!** 🎉
  - 3 tests were failing due to keyword identifier usage
  - Updated tests to use non-keyword identifiers
  - **Achieved 100% test coverage: 60/60 parser tests, 59/59 comparison tests**

**Previous Session (2025-10-31 - PART 1):**
- ✅ **Check operators fully implemented!** ⭐
  - All 4 basic operators: `is`, `raises`, `satisfies`, `violates`
  - All 11 CheckOp variants supported
  - Creates proper `SCheckTest` AST nodes with CheckOp enum
  - JSON serialization support added
- ✅ **Keywords as field names** ⭐
  - Fixed: `obj.method()`, `obj.fun()`, `obj.if()` now work
  - Added `parse_field_name()` helper that accepts Name or keyword tokens
  - Critical for real-world Pyret code

**Previous Session (2025-10-31):**
- ✅ **Fixed array syntax misconception**
  - Discovered Pyret does NOT support `[1, 2, 3]` shorthand syntax
  - Removed incorrect shorthand implementation
  - Updated all tests to use proper `[list: 1, 2, 3]` construct syntax
  - Verified with official Pyret parser - it REJECTS `[1, 2, 3]`
- ✅ **Fixed whitespace sensitivity bug**
  - `f (x)` now correctly parsed as two separate expressions
  - Removed incorrect ParenSpace → function application logic
- ✅ **Construct expressions** - `[list: 1, 2, 3]`, `[set: x, y]`, `[lazy array: ...]`
- ✅ **Bracket access** - `arr[0]`, `matrix[i][j]`
- ✅ **Ultra-complex expression support** validated
  - Expression with 7+ levels of nesting works perfectly
  - All postfix operators chain correctly
  - AST matches official Pyret parser byte-for-byte

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
