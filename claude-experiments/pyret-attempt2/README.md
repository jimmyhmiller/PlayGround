# Pyret Parser in Rust

A hand-written recursive descent parser for the [Pyret programming language](https://www.pyret.org/) that generates JSON ASTs matching the reference JavaScript implementation.

## 🚀 Quick Start

```bash
# Run all tests
cargo test

# Run only comparison tests (298 passing, 1 failing)
cargo test --test comparison_tests

# Run with debug output
DEBUG_TOKENS=1 cargo test test_name

# Compare with official Pyret parser
./scripts/compare_parsers.sh "your pyret code here"

# Build
cargo build
```

## 📊 Current Status (Updated: 2025-11-09)

**Test Results: 298/299 tests passing (99.7%)** 🎉
- ✅ **298 tests PASSING** - All produce byte-for-byte identical ASTs!
- ❌ **1 test FAILING** - `test_full_file_benchmark_adding_ones_2000`

### Parser is 99.7% Complete!

**Latest achievement: Scientific Notation Normalization!**
- ✅ Fixed scientific notation formatting - positive exponents now use `e+` sign
- ✅ Fixed rough number threshold - small numbers (< 1e-6) use scientific notation
- ✅ Examples: `~1.5e308` → `~1.5e+308`, `~0.0000001` → `~1e-7`
- 📊 Improved from 93.9% to 99.7% (+5.8 percentage points!)
- 🎯 **+19 tests fixed in this session!**

**Core Language: 100% complete** ✅
- ✅ All primitive expressions and operators
- ✅ Function definitions with arrow types and generics
- ✅ Data declarations with sharing clauses and generics
- ✅ Pattern matching with wildcards and else clauses
- ✅ Control flow (if, when, for with map/filter/fold, blocks)
- ✅ Lambda expressions and higher-order functions
- ✅ Object expressions with methods
- ✅ Let/var bindings, assignments, type annotations
- ✅ Import/provide statements (all variants)
- ✅ Where clauses and check blocks with refinements
- ✅ Complete type system (Any, generics, parameterized types)

**Advanced Features: 100% complete** ✅
- ✅ Object extension - `obj.{ field: value }`
- ✅ Table expressions - `table: name, age row: "Alice", 30 end`
- ✅ Spy expressions - `spy "label": x end`
- ✅ Template dots - `lam(): ... end`
- ✅ Underscore partial application - `(_ + 2)`
- ✅ Arbitrary precision numbers - stored as strings
- ✅ Scientific notation - proper formatting with `e+` for positive exponents

**Overall: 99.7% complete - parser nearly done!** 🚀

## ✅ Fully Implemented Features

<details>
<summary><b>Click to expand complete feature list</b></summary>

### Core Expressions
- Numbers, strings, booleans, identifiers
- Binary operators (15 operators, left-associative, NO precedence)
- Parenthesized expressions `(1 + 2)`
- Function calls `f(x, y)` with multiple arguments
- Chained calls `f(x)(y)(z)`
- Whitespace-sensitive parsing: `f(x)` vs `f (x)`

### Data Access
- Dot access `obj.field.subfield`
- Bracket access `arr[0]`, `matrix[i][j]`
- Tuple access `x.{2}`
- Keywords as field names `obj.method()`

### Data Structures
- Construct expressions `[list: 1, 2, 3]`, `[set: x, y]`
- Object expressions `{ x: 1, y: 2 }`
  - Data fields, mutable fields (`ref`), method fields
- Tuple expressions `{1; 2; 3}` (semicolon-separated)

### Control Flow
- Block expressions `block: ... end`
- If expressions `if c: a else: b end` with else-if chains
- When expressions `when c: body end`
- For expressions `for map(x from lst): x + 1 end`
- Cases expressions `cases(T) e: | variant => body end` (pattern matching)

### Functions & Bindings
- Lambda expressions `lam(x): x + 1 end`
- Function definitions `fun f(x): body end`
- Let bindings `x = 5`, `let x = 5`
- Var bindings `var x = 5`
- Assignment expressions `x := 5`

### Data & Types
- Data declarations `data T: | variant end`
- Check operators `is`, `raises`, `satisfies`, `violates`

### Modules
- Import statements `import mod as M`
- Provide statements `provide *`

### Advanced
- Chained postfix operators `obj.foo().bar().baz()`
- Ultra-complex nested expressions
- Program structure with prelude and body

</details>

## 🔴 Features Not Yet Implemented

See [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) for detailed analysis.

**47 features still need implementation:**
- Advanced block features (multi-statement blocks with scoping)
- Where clauses for testing
- Rest parameters in functions
- Generic/parameterized types
- String interpolation
- Unary operators (`not`, `-`)
- Advanced for variants (filter, fold)
- Sharing clauses on data types
- Table expressions
- Check blocks
- Advanced import/export patterns
- Object extension/refinement
- And more...

## 📁 Project Structure

```
compare_parsers.sh       - Validate against official Pyret parser

src/
├── parser.rs       (~2,000 lines) - Parser implementation
├── ast.rs          (~1,350 lines) - All AST node types
├── tokenizer.rs    (~1,390 lines) - Complete tokenizer
├── error.rs        (73 lines)     - Error types
├── lib.rs                         - Library exports
└── bin/
    ├── to_json.rs                 - Output full AST as JSON
    └── to_pyret_json.rs (~400)    - Pyret-compatible JSON

tests/
├── parser_tests.rs      (~1,540 lines) - 68 unit tests, all passing ✅
└── comparison_tests.rs  (~1,360 lines) - 128 integration tests
    ├── 81 passing (basic + working features) ✅
    └── 47 ignored (advanced features not yet implemented) ⏸️

docs/
├── TEST_STATUS_REPORT.md               - 📊 Current status & analysis ⭐
├── NEXT_STEPS.md                       - Implementation guide
├── PARSER_PLAN.md                      - Overall project plan
├── PHASE1_COMPLETE.md                  - Foundation
├── PHASE2_COMPLETE.md                  - Primitives & operators
├── PHASE3_PARENS_AND_APPS_COMPLETE.md - Parens & function calls
└── OPERATOR_PRECEDENCE.md              - Pyret has NO precedence!
```

## 🎯 Next Steps: Implement Type System (11 Tests Remaining)

**Parser is 90.7% complete!** Only 11 features remaining, all validated against official Pyret.

### 🔥 **START HERE: Complete Type System (3 tests, ~5-6 hours)**

The type system is partially working (arrow types like `fun f(x) -> Number: ...` already pass!). Complete it by implementing:

1. **`Any` type annotation** (~1-2 hours) ✨ **EASIEST!**
   - `x :: Any = 42`
   - AST node already exists, just add keyword recognition

2. **Generic function type parameters** (~2-3 hours)
   - `fun identity<T>(x :: T) -> T: x end`
   - Parse `<T, U>` after function names

3. **Generic data types** (~2-3 hours)
   - `data List<T>: | empty | link(first :: T, rest) end`
   - Parse `<T, U>` after data names

**See [NEXT_STEPS.md](NEXT_STEPS.md) for detailed step-by-step implementation guide with code examples!**

### After Types (8 Remaining Tests)

4. **Object extension** (2 tests, ~3-4 hours) - `point.{ z: 0 }`
5. **Advanced imports** (4 tests, ~4-6 hours) - file imports, provide-types
6. **Table literals** (1 test, ~4-6 hours) - `table: name, age row: ... end`
7. **Spy expressions** (1 test, ~1-2 hours) - `spy: x end`

---

## 🎯 For Contributors

**New to this project?** Start here:

1. **Read [NEXT_STEPS.md](NEXT_STEPS.md)** - **Complete type implementation guide** ⭐⭐⭐
2. **Read [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md)** - See all remaining features
3. **Read [CLAUDE.md](CLAUDE.md)** - Comprehensive project status
4. **Check `src/parser.rs`** - See patterns for expression/statement parsing

**Quick contribution guide:**
```bash
# 1. See what's remaining (11 tests)
cargo test --test comparison_tests -- --ignored --list

# 2. Start with Any type (easiest!)
# See NEXT_STEPS.md for step-by-step guide

# 3. Add code to src/parser.rs
# 4. Remove #[ignore] from test
# 5. Run: cargo test test_any_type
# 6. Validate: ./scripts/compare_parsers.sh "x :: Any = 42"
```

## 🧪 Testing

```bash
# All tests
cargo test

# Comparison tests (107 passing, 11 ignored)
cargo test --test comparison_tests

# See what needs implementation
cargo test --test comparison_tests -- --ignored --list

# Unit tests
cargo test --test parser_tests

# Specific test
cargo test test_pyret_match_simple_fun

# With token debug output
DEBUG_TOKENS=1 cargo test test_name
```

### Test Suites

1. **`parser_tests.rs`** - 68 unit tests for parser components (all passing)
2. **`comparison_tests.rs`** - 128 integration tests comparing with official Pyret parser
   - 81 tests passing (basic features + core language)
   - 47 tests ignored (advanced features not yet implemented)

## 🔍 Parser Validation

We validate that our parser produces **identical ASTs** to the official Pyret parser:

```bash
# Compare any expression
./scripts/compare_parsers.sh "2 + 3"
./scripts/compare_parsers.sh "fun f(x): x + 1 end"
./scripts/compare_parsers.sh "data Box: | box(ref v) end"

# Output JSON from Rust parser
echo "2 + 3" | cargo run --bin to_pyret_json
```

**Status**: All 81 passing tests verified byte-for-byte identical! ✅

## 📚 Key Features

### Whitespace-Sensitive Parsing

Pyret distinguishes function calls from parenthesized expressions by whitespace:

```pyret
f(x)    // Function application: f called with argument x
f (x)   // Two separate expressions: f, then (x)
```

The tokenizer produces different tokens (`ParenNoSpace` vs `ParenSpace`) and the parser handles them correctly.

### No Operator Precedence

Pyret has **NO operator precedence**. All binary operators have equal precedence and are strictly left-associative:

```pyret
2 + 3 * 4    // Evaluates as (2 + 3) * 4 = 20
             // NOT as 2 + (3 * 4) = 14
```

Users must use explicit parentheses to control evaluation order. See [OPERATOR_PRECEDENCE.md](OPERATOR_PRECEDENCE.md).

### Exact AST Matching

The parser generates JSON that matches the reference Pyret implementation exactly:

```json
{
  "type": "s-app",
  "fun": { "type": "s-id", "id": { "type": "s-name", "name": "f" } },
  "args": [
    { "type": "s-num", "value": "42" }
  ]
}
```

## 🔗 References

- **Pyret Language:** https://www.pyret.org/
- **Reference Implementation:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang`
- **Grammar Spec:** `/pyret-lang/src/js/base/pyret-grammar.bnf`

## 📋 Implementation Progress

| Category | Status | Progress |
|----------|--------|----------|
| Core Expressions | ✅ Complete | 100% |
| Data Structures | ✅ Complete | 100% |
| Control Flow | ✅ Complete | 90% |
| Functions & Bindings | ✅ Complete | 90% |
| Data Definitions | ✅ Basic | 60% |
| Pattern Matching | ✅ Basic | 60% |
| Type System | ⚠️ Partial | 30% |
| Imports/Exports | ✅ Basic | 60% |
| Advanced Features | ⚠️ Partial | 35% |

**Overall: 63.3% complete** (81/128 tests passing)
**Core language: ~90% complete** ✅

## 💡 Examples

### Simple Expression
```pyret
1 + 2
```
```rust
Expr::SOp {
    op: "op+",
    left: Box::new(Expr::SNum { n: 1.0 }),
    right: Box::new(Expr::SNum { n: 2.0 })
}
```

### Function Definition
```pyret
fun f(x): x + 1 end
```
```rust
Stmt::SFun {
    name: "f",
    args: vec![Bind { name: "x", ann: ABlank }],
    body: SBlock { stmts: [...] }
}
```

### Pattern Matching
```pyret
cases(Either) e:
  | left(v) => v
  | right(v) => v
end
```
```rust
Expr::SCases {
    typ: AName { id: "Either" },
    val: SId { id: "e" },
    branches: vec![
        CasesBranch { name: "left", args: [...], body: [...] },
        CasesBranch { name: "right", args: [...], body: [...] }
    ]
}
```

## 🐛 Known Limitations

- Rational number parsing incomplete (stores as floats, not numerator/denominator)
- 47 advanced features not yet implemented (see TEST_STATUS_REPORT.md)
- Type annotations partially implemented
- Where clauses not yet implemented
- String interpolation not yet implemented
- Generic types not yet implemented

## 🛠️ Development

Built with:
- Rust 1.74+
- `serde` for JSON serialization
- `thiserror` for error handling

No external parser generators - pure hand-written recursive descent.

## 📄 License

[License details to be added]

---

**Last Updated:** 2025-11-02
**Current Progress:** 63.3% overall (81/128 tests), ~90% core language complete
**Contributing:** See [NEXT_STEPS.md](NEXT_STEPS.md) for detailed guidance on what to implement next

**Quick wins for next session:**
- Unary operators (`not`, `-`) - 3 tests, ~2-3 hours
- Type annotations on bindings - 3 tests, ~2-3 hours
- Advanced block features - 4 tests, ~3-4 hours
