# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-11-02 - UPDATED)

**Test Results: 81/126 tests passing (64.3%)**
- ✅ **81 tests PASSING** (64.3%)
- ⏸️ **45 tests IGNORED** (features not yet implemented)
- ❌ **0 tests FAILING**
- 🗑️ **2 tests DELETED** (invalid Pyret syntax - unary operators)

**All passing tests produce byte-for-byte identical ASTs to the official Pyret parser!** ✨

### Recent Discovery: More Features Working Than Documented! 🎉

The following features were already implemented but not documented:
- ✅ **Function definitions** `fun f(x): x + 1 end`
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

# Run comparison tests only (81 passing, 45 ignored)
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
├── parser_tests.rs      (~1,540 lines) - 68 unit tests, all passing ✅
└── comparison_tests.rs  (~1,360 lines) - 126 integration tests
    ├── 81 passing (basic + working features) ✅
    └── 45 ignored (advanced features not yet implemented)
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
- For expressions `for map(x from lst): x + 1 end`
- Cases expressions `cases(T) e: | variant => body end`

### Functions & Bindings ✅
- Lambda expressions `lam(x): x + 1 end`
- Function definitions `fun f(x): body end`
- Let bindings `x = 5`, `let x = 5`
- Var bindings `var x = 5`
- Assignment expressions `x := 5`

### Data & Types ✅
- Data declarations `data T: | variant end`
- Check operators `is`, `raises`, `satisfies`, `violates`

### Modules ✅
- Import statements `import mod as M`
- Provide statements `provide *`

### Advanced Features ✅
- Chained postfix operators `obj.foo().bar().baz()`
- Ultra-complex nested expressions
- Program structure with prelude and body

## 🔴 Features Not Yet Implemented (45 Ignored Tests)

### ⚠️ Important: Unary Operators DO NOT Exist in Pyret!
Pyret does **not** have unary operators like traditional languages:
- ❌ `not x` is invalid - use `not(x)` (function call)
- ❌ `-x` is invalid - use `0 - x` (binary operation)
- The `SUnaryOp` AST node exists but is never used by the Pyret parser
- 2 tests for unary operators were removed as they tested invalid syntax

### Advanced Block Features (4 tests)
- Multi-statement blocks with multiple let bindings
- Complex scoping and shadowing rules
- Type annotations on let bindings in blocks

### Advanced Function Features (4 tests)
- **Where clauses** with multiple test cases (PARTIALLY IMPLEMENTED - needs refinement)
- Rest parameters (`...args`)
- Complex recursive patterns

### Advanced Data Features (6 tests)
- Data with sharing clauses (shared methods across variants)
- Parameterized/generic data types (`data List<T>`)
- Complex variant patterns

### Advanced Cases Features (4 tests)
- Cases with else branches
- Nested cases expressions
- Complex pattern matching

### Advanced For Features (4 tests)
- For with multiple generators (cartesian product)
- For fold with complex accumulators
- For filter variant
- Nested for expressions

### Type System (3 tests)
- Function type annotations with arrow (`->`)
- Union types (`Number | String`)
- Generic type parameters

### String Features (2 tests)
- String interpolation (`` `Hello $(name)` ``)
- Complex expressions in interpolation

### Other Advanced Features (18 tests)
- Table expressions (2 tests)
- Check blocks (2 tests)
- Advanced import/export (4 tests)
- Object extension/refinement (3 tests)
- List comprehensions with guards (1 test)
- Spy expressions (1 test)
- Contracts (1 test)
- Complex real-world patterns (2 tests)
- Gradual typing (1 test)

## 🎯 Next Priority Tasks

Based on the 45 ignored tests, here are the highest-value features to implement:

### 🔥 Priority 1: Where Clauses (RECOMMENDED NEXT)
**Status:** Partially implemented, needs refinement to match official parser
- Parser already recognizes WHERE keyword and populates check field
- AST support exists (SFun.check field)
- Official Pyret parser confirmed this is a real feature
- Used for testing functions inline

**Example:**
```pyret
fun factorial(n):
  if n == 0: 1
  else: n * factorial(n - 1)
  end
where:
  factorial(0) is 1
  factorial(5) is 120
end
```

**What's needed:**
- Minor refinements to match official parser output exactly
- Ensure check-test nodes are created correctly
- Test with all where clause test cases

### Priority 2: High-Value Quick Wins (5-8 hours)
1. **Advanced block features** - 4 tests, multi-statement blocks
2. **Type annotations on bindings** - 3 tests, improves type safety
3. **String interpolation** - 2 tests, very common in practice

### Priority 3: Medium-Value Features (10-15 hours)
1. **Advanced for variants** (filter, fold) - 4 tests
2. **Advanced data features** (sharing, generics) - 6 tests
3. **Advanced import/export** - 4 tests
4. **Advanced cases patterns** - 4 tests

### Priority 4: Advanced Features (20+ hours)
1. **Table expressions** - 2 tests
2. **Check blocks** - 2 tests
3. **Object refinement** - 3 tests
4. **Complex patterns** - remaining tests

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
# Result: 81 passed, 45 ignored, 0 failed

# See what needs implementation
cargo test --test comparison_tests -- --ignored --list

# Test specific feature
./compare_parsers.sh "fun f(x): x + 1 end"
```

**68/68 parser unit tests passing** ✅ (100%)
**81/126 comparison integration tests passing** ✅ (64.3%)

## 💡 Quick Tips

### First Time Here?
1. Read [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) - See exactly what's working
2. Read [NEXT_STEPS.md](NEXT_STEPS.md) - Implementation guides
3. Look at `tests/comparison_tests.rs` - See test patterns
4. Look at `src/parser.rs` - See recent implementations (where clauses partially done!)
5. **Recommended:** Start with where clauses (already 80% complete)

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

**Core Language: ~90% Complete** ✅
- All basic expressions ✅
- All basic statements ✅
- Function definitions ✅
- Data declarations (basic) ✅
- Pattern matching (basic) ✅
- Import/export (basic) ✅

**Advanced Features: ~40% Complete** ⚠️
- Type annotations (partial)
- Where clauses (PARTIAL - 80% done, needs refinement)
- String interpolation (missing)
- Generic types (missing)
- Table expressions (missing)
- Check blocks (missing)
- ⚠️ Unary operators (DO NOT EXIST in Pyret)

**Overall: 64.3% Complete** (81/126 tests passing)

## 🎉 Ready to Code!

The codebase is clean, well-tested, and ready for the next features:

1. Start with [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) to see the big picture
2. **RECOMMENDED:** Start with where clauses (80% complete, just needs refinement)
3. Look at the ignored test to understand what's needed
4. Follow the implementation pattern
5. Run tests and validate with `./compare_parsers.sh`

**Best next steps:**
- ⭐ **Where clauses:** Partially done, ~1-2 hours to complete
- Type annotations on bindings: 3 tests, ~2-3 hours
- Advanced blocks: 4 tests, ~3-4 hours
- String interpolation: 2 tests, ~3-4 hours

---

**Last Updated:** 2025-11-02 (Evening)
**Tests:** 68/68 parser tests ✅ (100%), 81/126 comparison tests ✅ (64.3%)
**Recent Change:**
- Investigated and deleted invalid unary operator tests (Pyret doesn't have unary operators!)
- Verified where clauses are real and partially implemented
- Updated priorities to focus on where clauses next
**Next Session:** Complete where clause implementation (see Priority 1 above)
