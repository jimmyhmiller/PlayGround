# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-11-02 - UPDATED)

**Test Results: 81/128 tests passing (63.3%)**
- ✅ **81 tests PASSING** (63.3%)
- ⏸️ **47 tests IGNORED** (features not yet implemented)
- ❌ **0 tests FAILING**

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

# Run comparison tests only (81 passing, 47 ignored)
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
└── comparison_tests.rs  (~1,360 lines) - 128 integration tests
    ├── 81 passing (basic + working features) ✅
    └── 47 ignored (advanced features not yet implemented)
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

## 🔴 Features Not Yet Implemented (47 Ignored Tests)

### Advanced Block Features (4 tests)
- Multi-statement blocks with multiple let bindings
- Complex scoping and shadowing rules
- Type annotations on let bindings in blocks

### Advanced Function Features (4 tests)
- Where clauses with multiple test cases
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

### Other Advanced Features (20 tests)
- Table expressions (2 tests)
- Check blocks (2 tests)
- Advanced import/export (4 tests)
- Unary operators (`not`, `-`) (3 tests)
- Object extension/refinement (3 tests)
- List comprehensions with guards (1 test)
- Spy expressions (1 test)
- Contracts (1 test)
- Complex real-world patterns (2 tests)
- Gradual typing (1 test)

## 🎯 Next Priority Tasks

Based on the 47 ignored tests, here are the highest-value features to implement:

### Priority 1: High-Value Quick Wins (5-8 hours)
1. **Unary operators** (`not`, `-`) - 3 tests, common in real code
2. **Advanced block features** - 4 tests, multi-statement blocks
3. **Where clauses** - 4 tests, testing infrastructure
4. **Type annotations on bindings** - 3 tests, improves type safety

### Priority 2: Medium-Value Features (10-15 hours)
1. **String interpolation** - 2 tests, very common in practice
2. **Advanced for variants** (filter, fold) - 4 tests
3. **Advanced data features** (sharing, generics) - 6 tests
4. **Advanced import/export** - 4 tests

### Priority 3: Advanced Features (20+ hours)
1. **Table expressions** - 2 tests
2. **Check blocks** - 2 tests
3. **Advanced cases patterns** - 4 tests
4. **Object refinement** - 3 tests
5. **Complex patterns** - remaining tests

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
# Result: 81 passed, 47 ignored, 0 failed

# See what needs implementation
cargo test --test comparison_tests -- --ignored --list

# Test specific feature
./compare_parsers.sh "fun f(x): x + 1 end"
```

**68/68 parser unit tests passing** ✅ (100%)
**81/128 comparison integration tests passing** ✅ (63.3%)

## 💡 Quick Tips

### First Time Here?
1. Read [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) - See exactly what's working
2. Read [NEXT_STEPS.md](NEXT_STEPS.md) - Implementation guides
3. Look at `tests/comparison_tests.rs` - See test patterns (lines 700-1364 show gap tests)
4. Look at `src/parser.rs` - See recent implementations
5. Pick a feature from the 47 ignored tests

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

**Advanced Features: ~35% Complete** ⚠️
- Type annotations (partial)
- Where clauses (missing)
- String interpolation (missing)
- Unary operators (missing)
- Generic types (missing)
- Table expressions (missing)
- Check blocks (missing)

**Overall: 63.3% Complete** (81/128 tests passing)

## 🎉 Ready to Code!

The codebase is clean, well-tested, and ready for the next features:

1. Start with [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) to see the big picture
2. Pick a feature from the Priority 1 list above
3. Look at the ignored test to understand what's needed
4. Follow the implementation pattern
5. Run tests and validate with `./compare_parsers.sh`

**Quick wins available:**
- Unary operators: 3 tests, ~2-3 hours
- Type annotations on bindings: 3 tests, ~2-3 hours
- Advanced blocks: 4 tests, ~3-4 hours

---

**Last Updated:** 2025-11-02
**Tests:** 68/68 parser tests ✅ (100%), 81/128 comparison tests ✅ (63.3%)
**Recent Change:** Merged comprehensive gap tests into comparison_tests.rs, discovered 6 undocumented working features
**Next Session:** Pick any feature from the 47 ignored tests - see TEST_STATUS_REPORT.md for recommendations
