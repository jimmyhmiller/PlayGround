# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-11-03 - EARLY MORNING UPDATE)

**Test Results: 99/126 tests passing (78.6%)**
- ✅ **99 tests PASSING** (78.6%) - **+9 since last session!**
- ⏸️ **27 tests IGNORED** (features not yet implemented)
- ❌ **0 tests FAILING**
- 🗑️ **2 tests DELETED** (invalid Pyret syntax - unary operators)

**All passing tests produce byte-for-byte identical ASTs to the official Pyret parser!** ✨

### Latest Completions: Advanced Pattern Matching & Data Sharing! ✅

**Session achievements (9 new tests enabled):**
1. ✅ **Underscore wildcards** - `_` in pattern matching (`cases(List) x: | link(_, _) => ...`)
2. ✅ **Cases-else** - Default branches in cases expressions
3. ✅ **Nested cases** - Cases expressions inside cases branches
4. ✅ **Cases in functions** - Pattern matching in function bodies
5. ✅ **For-filter** - `for filter(x from list): predicate end`
6. ✅ **For-fold** - `for fold(acc from init, x from list): body end`
7. ✅ **For cartesian product** - Multiple generators `for map(x from l1, y from l2): ...`
8. ✅ **Nested for expressions** - For loops inside for loops
9. ✅ **Data sharing clauses** - `data Tree: ... sharing: method size(self): ... end`

### Fully Implemented Features

The following features were already implemented:
- ✅ **Function definitions** `fun f(x): x + 1 end`
- ✅ **Where clauses** `fun f(x): body where: f(1) is 2 end` **[JUST COMPLETED]**
- ✅ **When expressions** `when cond: body end`
- ✅ **Assignment expressions** `x := 5`
- ✅ **Data declarations** `data Box: | box(ref v) end`
- ✅ **Cases expressions** `cases(Either) e: | left(v) => v | right(v) => v end`
- ✅ **Import statements** `import equality as E`

## 🚀 Quick Start

```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2

# Run all tests
cargo test

# Run comparison tests only (99 passing, 27 ignored)
cargo test --test comparison_tests

# Run ignored tests to see what needs work
cargo test --test comparison_tests -- --ignored

# Compare specific code
./compare_parsers.sh "your pyret code here"
```

## 📚 Essential Documentation

**Start here:**
- **[TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md)** - Complete analysis of what's working and what's not ⭐⭐⭐
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Implementation guide for remaining features
- **[README.md](README.md)** - Project overview

**Implementation history:**
- **[PHASE3_PARENS_AND_APPS_COMPLETE.md](PHASE3_PARENS_AND_APPS_COMPLETE.md)** - Parentheses & function application
- **[PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)** - Primitives and binary operators
- **[PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)** - Foundation

**Reference:**
- **[OPERATOR_PRECEDENCE.md](OPERATOR_PRECEDENCE.md)** - Important: Pyret has NO precedence!

## 📁 Key Files

```
src/
├── parser.rs       (~2,000 lines) - Parser implementation
├── ast.rs          (~1,350 lines) - All AST node types
├── tokenizer.rs    (~1,390 lines) - Complete tokenizer
└── error.rs        (73 lines)     - Error types

src/bin/
└── to_pyret_json.rs (~400 lines) - JSON serialization

tests/
├── parser_tests.rs      (~1,540 lines) - 72 unit tests, all passing ✅
└── comparison_tests.rs  (~1,360 lines) - 126 integration tests
    ├── 99 passing (basic + working features) ✅
    └── 27 ignored (advanced features not yet implemented)
```

## ✅ Fully Implemented Features (All produce identical ASTs!)

### Core Expressions ✅
- Numbers, strings, booleans, identifiers
- Binary operators (15 operators, left-associative, NO precedence)
- Parenthesized expressions `(1 + 2)`
- Function calls `f(x, y)` with multiple arguments
- Chained calls `f(x)(y)(z)`
- Whitespace-sensitive parsing: `f(x)` vs `f (x)`

### Data Access ✅
- Dot access `obj.field.subfield`
- Bracket access `arr[0]`, `matrix[i][j]`
- Tuple access `x.{2}`
- Keywords as field names `obj.method()`

### Data Structures ✅
- Construct expressions `[list: 1, 2, 3]`, `[set: x, y]`
- Object expressions `{ x: 1, y: 2 }`
  - Data fields, mutable fields (`ref`), method fields
- Tuple expressions `{1; 2; 3}` (semicolon-separated)

### Control Flow ✅
- Block expressions `block: ... end`
- If expressions `if c: a else: b end` with else-if chains
- When expressions `when c: body end`
- For expressions:
  - ✅ `for map(x from lst): x + 1 end`
  - ✅ **For-filter** `for filter(x from lst): x > 2 end` ✨ **[NEW!]**
  - ✅ **For-fold** `for fold(acc from 0, x from lst): acc + x end` ✨ **[NEW!]**
  - ✅ **Multiple generators** `for map(x from l1, y from l2): {x; y} end` ✨ **[NEW!]**
  - ✅ **Nested for** ✨ **[NEW!]**
- Cases expressions:
  - ✅ `cases(T) e: | variant => body end`
  - ✅ **Cases-else** `cases(T) e: | v1 => a | else => b end` ✨ **[NEW!]**
  - ✅ **Underscore wildcards** `| link(_, _) => ...` ✨ **[NEW!]**
  - ✅ **Nested cases** ✨ **[NEW!]**

### Functions & Bindings ✅
- Lambda expressions `lam(x): x + 1 end`
- Function definitions `fun f(x): body end`
- Where clauses `fun f(x): body where: test end`
- Let bindings `x = 5`, `let x = 5`
- **Var bindings** `var x = 5` ✨ **[NEW!]**
- **Type annotations** `x :: Number = 42` ✨ **[NEW!]**
- Assignment expressions `x := 5`
- **Multi-statement blocks** ✨ **[NEW!]**
- **Nested blocks with shadowing** ✨ **[NEW!]**

### Data & Types ✅
- **Simple data declarations** `data Color: | red | green | blue end`
- **Data with typed fields** `data Point: | point(x :: Number, y) end`
- **Data with mutable fields** `data Box: | box(ref v) end`
- **Data with multiple variants** `data Either: | left(v) | right(v) end`
- **Data with sharing clauses** `sharing: method size(self): ... end` ✨ **[NEW!]**
- Data with where clauses
- Check operators `is`, `raises`, `satisfies`, `violates`

### Modules ✅
- Import statements `import mod as M`
- Provide statements `provide *`

### Advanced Features ✅
- Chained postfix operators `obj.foo().bar().baz()`
- Ultra-complex nested expressions
- Program structure with prelude and body

## 🔴 Features Not Yet Implemented (27 Ignored Tests)

### ⚠️ Important: Unary Operators DO NOT Exist in Pyret!
Pyret does **not** have unary operators like traditional languages:
- ❌ `not x` is invalid - use `not(x)` (function call)
- ❌ `-x` is invalid - use `0 - x` (binary operation)
- The `SUnaryOp` AST node exists but is never used by the Pyret parser
- 2 tests for unary operators were removed as they tested invalid syntax

### Advanced Function Features (3 tests)
- Rest parameters (`...args`)
- Complex recursive patterns
- Function-returning-function patterns

### Advanced Data Features (1 test)
- Parameterized/generic data types (`data List<T>`)

### Type System (3 tests)
- Function type annotations with arrow (`->`)
- Union types (`Number | String`)
- Generic type parameters

### String Features (2 tests)
- String interpolation (`` `Hello $(name)` ``)
- Complex expressions in interpolation

### Other Advanced Features (15 tests)
- Table expressions (2 tests)
- Check blocks (2 tests)
- Advanced import/export (4 tests)
- Object extension/refinement (3 tests)
- List comprehensions with guards (1 test)
- Spy expressions (1 test)
- Contracts (1 test)
- Gradual typing (1 test)

## 🎯 Next Priority Tasks

Based on the 27 remaining ignored tests, here are the highest-value features to implement:

### 🔥 Priority 1: High-Value Features (NOT Quick Wins)
⚠️ **Note:** The remaining features are more complex and require significant tokenizer/parser changes:

1. **String interpolation** - 2 tests (~4-6 hours)
   - ❌ Tokenizer does NOT support backtick strings yet
   - Requires: Tokenizer updates for `` `Hello $(expr)` `` syntax
   - Requires: Parser support for embedded expressions

2. **Rest parameters** - 1 test (~2-3 hours)
   - Requires: `...` token recognition
   - `fun f(x, rest ...): ...`

3. **Generic data types** - 1 test (~3-4 hours)
   - `data List<T>: ...`
   - Requires: Type parameter parsing

### Priority 2: Medium-Value Features (10-15 hours)
1. **Check blocks** - 2 tests, important for testing
2. **Advanced import/export** - 4 tests
3. **Advanced type annotations** - 3 tests (arrows `->`, unions `|`)

### Priority 3: Advanced Features (15+ hours)
1. **Table expressions** - 2 tests
2. **Object refinement** - 3 tests
3. **List comprehensions with guards** - 1 test
4. **Spy expressions** - 1 test
5. **Complex real-world patterns** - 2 tests (integration)

## 🔑 Key Concepts

**Whitespace Sensitivity:**
- `f(x)` → Direct function call (s-app)
- `f (x)` → Two separate expressions (f and (x))

**No Operator Precedence:**
- `2 + 3 * 4` = `(2 + 3) * 4` = `20` (NOT 14)
- All binary operators have equal precedence
- Strictly left-associative

**Implementation Pattern:**
1. Add `parse_foo()` method in `src/parser.rs`
2. Update `parse_prim_expr()` or appropriate section
3. Add location extraction for new expr/stmt type
4. Add JSON serialization in `src/bin/to_pyret_json.rs`
5. Add tests in `tests/parser_tests.rs`
6. Update comparison test (remove `#[ignore]`)
7. Run `cargo test` and `./compare_parsers.sh "code"`

## ✅ Tests Status

```bash
# Run all comparison tests
cargo test --test comparison_tests
# Result: 99 passed, 27 ignored, 0 failed

# See what needs implementation
cargo test --test comparison_tests -- --ignored --list

# Test specific feature
./compare_parsers.sh "fun f(x): x + 1 end"
```

**72/72 parser unit tests passing** ✅ (100%)
**99/126 comparison integration tests passing** ✅ (78.6%)

## 💡 Quick Tips

### First Time Here?
1. Read [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) - See exactly what's working
2. Read [NEXT_STEPS.md](NEXT_STEPS.md) - Implementation guides
3. Look at `tests/comparison_tests.rs` - See test patterns
4. Look at `src/parser.rs` - See recent implementations
5. **Recommended next:** Check remaining ignored tests - most "easy wins" are done!

### Debugging
```bash
# See what tokens are generated
DEBUG_TOKENS=1 cargo test test_name

# Run specific test
cargo test test_pyret_match_simple_fun

# Compare with official parser
./compare_parsers.sh "your code"
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
3. **⚠️ Array syntax** - Pyret does NOT support `[1, 2, 3]` shorthand!
   - Must use: `[list: 1, 2, 3]` (construct expression)
4. **Update location extraction** - Add new Expr/Stmt types to match statements
5. **Test edge cases** - Empty, single item, nested, mixed expressions
6. **Follow existing patterns** - Look at similar code for consistency

## 📞 Reference Materials

- **Pyret Grammar:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
- **AST Definitions:** `src/ast.rs:292-808`
- **Parser Implementation:** `src/parser.rs`
- **Test Examples:** `tests/comparison_tests.rs`
- **Comparison Tool:** `./compare_parsers.sh`

## 🎯 Parser Completion Status

**Core Language: ~95% Complete** ✅
- All basic expressions ✅
- All basic statements ✅
- Function definitions ✅
- **Data declarations (basic)** ✅ **[COMPLETED]**
- Pattern matching (basic) ✅
- Import/export (basic) ✅
- **Advanced blocks** ✅ **[COMPLETED]**
- **Type annotations** ✅ **[COMPLETED]**

**Advanced Features: ~65% Complete** ⚠️
- Where clauses ✅
- **Cases-else, wildcards, nesting** ✅ **[COMPLETED]**
- **For-filter, fold, cartesian, nesting** ✅ **[COMPLETED]**
- **Data sharing clauses** ✅ **[COMPLETED]**
- String interpolation (missing - requires tokenizer work)
- Rest parameters (missing)
- Generic types (missing)
- Table expressions (missing)
- Check blocks (missing)
- ⚠️ Unary operators (DO NOT EXIST in Pyret)

**Overall: 78.6% Complete** (99/126 tests passing)

## 🎉 Ready to Code!

The codebase is clean, well-tested, and ready for the next features:

1. Start with [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) to see the big picture
2. Look at the ignored tests in `tests/comparison_tests.rs`
3. Follow the implementation pattern from recent work
4. Run tests and validate with `./compare_parsers.sh`

**Best next steps (all require significant work):**
- **String interpolation:** 2 tests, ~4-6 hours (requires tokenizer changes)
- **Rest parameters:** 1 test, ~2-3 hours (`fun f(x, rest ...): ...`)
- **Generic data types:** 1 test, ~3-4 hours (`data List<T>: ...`)

⚠️ **Note:** Most "quick wins" have been completed! Remaining features require more complex changes.

---

**Last Updated:** 2025-11-03 (Early Morning)
**Tests:** 72/72 parser tests ✅ (100%), 99/126 comparison tests ✅ (78.6%)
**This Session Completed:**
- ✅ **Underscore wildcards** (`_` in pattern matching)
- ✅ **Cases-else** (default branches in cases)
- ✅ **Nested cases** (cases inside cases)
- ✅ **Cases in functions** (pattern matching in function bodies)
- ✅ **For-filter** (`for filter(x from list): predicate end`)
- ✅ **For-fold** (`for fold(acc from init, x from list): body end`)
- ✅ **For cartesian product** (multiple generators)
- ✅ **Nested for expressions**
- ✅ **Data sharing clauses** (`sharing: method name(self): ... end`)
**Progress:** +9 tests enabled (from 90 to 99), 27 tests remaining
**Next Session:** String interpolation, rest parameters, or generic types (all require tokenizer/parser updates)
