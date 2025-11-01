# Pyret Parser in Rust

A hand-written recursive descent parser for the [Pyret programming language](https://www.pyret.org/) that generates JSON ASTs matching the reference JavaScript implementation.

## 🚀 Quick Start

```bash
# Run all tests
cargo test

# Run only parser tests
cargo test --test parser_tests

# Run with debug output
DEBUG_TOKENS=1 cargo test

# Build
cargo build
```

## 📊 Current Status

**Phase 3 - Expressions:** Advanced features (68/81 comparison tests passing - 84.0%)

✅ **Working & Verified:**
- ✅ All primitive expressions (numbers, strings, booleans, identifiers)
- ✅ Binary operators (15 operators, left-associative, NO precedence)
- ✅ Parenthesized expressions: `(1 + 2)`
- ✅ Function application: `f(x, y, z)` with multiple arguments
- ✅ Chained calls: `f(x)(y)(z)`, `f()(g())`
- ✅ **Whitespace-sensitive parsing** - FIXED! ✨
  - `f(x)` = function call
  - `f (x)` = two separate expressions (stops at `f`)
- ✅ **Dot access:** `obj.field`, `obj.field1.field2`
  - Including keywords as field names: `obj.method()` ✨
- ✅ **Bracket access:** `arr[0]`, `matrix[i][j]`
- ✅ **Construct expressions:** `[list: 1, 2, 3]`, `[set: x, y]`
- ✅ **Check operators:** `is`, `raises`, `satisfies`, `violates` ✨
  - All 11 variants: is, is-roughly, is-not, satisfies, violates, raises, etc.
- ✅ **Object expressions:** `{ x: 1, y: 2 }` ✨
  - Data fields, mutable fields (ref), trailing commas
- ✅ **Lambda expressions:** `lam(x): x + 1 end` ✨✨✨
  - Simple lambdas: `lam(): 5 end`
  - With parameters: `lam(x): x + 1 end`, `lam(n, m): n > m end`
  - In function calls: `filter(lam(e): e > 5 end, [list: -1, 1])`
  - Optional type annotations: `lam(x :: Number): x + 1 end`
- ✅ **Tuple expressions:** `{1; 2; 3}`, `x.{2}` ✨✨
  - Semicolon-separated tuples, tuple element access
- ✅ **Block expressions:** `block: ... end` - NEW! ✨✨✨✨
  - Simple blocks: `block: 5 end`
  - Multiple statements: `block: 1 + 2 3 * 4 end`
  - Empty and nested blocks
- ✅ **Chained postfix operators:** `obj.foo().bar().baz()`
- ✅ **Ultra-complex expressions:** All features work together perfectly!

⚠️ **Important Discovery:**
- Pyret does NOT support `[1, 2, 3]` array syntax!
- Must use construct expressions: `[list: 1, 2, 3]`
- Official Pyret parser rejects `[1, 2, 3]` with parse error

🎯 **Next Up (Priority Order):**
- If expressions `if cond: ... end` (1 test) - HIGHEST PRIORITY
- For expressions `for map(x from lst): ... end` (2 tests)
- Let bindings `x = value` (needed for block_multiple_stmts test)

## 📁 Project Structure

```
compare_parsers.sh - Validate against official Pyret parser

src/
├── ast.rs          (1,350 lines)  - All AST node types
├── parser.rs       (~1,380 lines) - Parser implementation (+30 for blocks)
├── tokenizer.rs    (~1,390 lines) - Complete tokenizer (+44 for keyword-colon fix)
├── error.rs        (73 lines)     - Error types
├── lib.rs          - Library exports
└── bin/
    ├── to_json.rs          - Output full AST as JSON
    └── to_pyret_json.rs    (~265 lines) - Pyret-compatible JSON (updated for blocks)

tests/
├── parser_tests.rs      (~1,340 lines) - 64 tests, all passing ✅
└── comparison_tests.rs  (524 lines)    - 68/81 passing ✅ (84.0%, 13 ignored)

docs/
├── PARSER_PLAN.md                      - Overall project plan
├── PHASE1_COMPLETE.md                  - Foundation complete
├── PHASE2_COMPLETE.md                  - Primitives & operators complete
├── PHASE3_PARENS_AND_APPS_COMPLETE.md - Latest work
├── NEXT_STEPS.md                       - 👈 START HERE for next tasks
├── OPERATOR_PRECEDENCE.md              - Important: Pyret has NO precedence!
└── PARSER_COMPARISON.md                - 🆕 Tools for validating against official parser
```

## 🎯 For Contributors

**New to this project?** Start here:
1. Read [NEXT_STEPS.md](NEXT_STEPS.md) - comprehensive guide for next tasks
2. Read [PHASE3_PARENS_AND_APPS_COMPLETE.md](PHASE3_PARENS_AND_APPS_COMPLETE.md) - latest changes
3. Look at `tests/parser_tests.rs` - see working examples
4. Check `src/parser.rs` Section 6 - expression parsing patterns

**Quick contribution guide:**
```bash
# 1. Pick a task from NEXT_STEPS.md
# 2. Add parser method in src/parser.rs Section 6
# 3. Update parse_prim_expr() or parse_binop_expr()
# 4. Add tests in tests/parser_tests.rs
# 5. Run tests: cargo test
```

## 🧪 Testing

```bash
# All tests (35 passing)
cargo test

# Just parser tests (24 passing)
cargo test --test parser_tests

# Specific test
cargo test test_parse_simple_function_call

# With token debug output
DEBUG_TOKENS=1 cargo test test_name
```

## 🔍 Parser Validation

We've built tools to verify our parser produces **identical ASTs** to the official Pyret parser:

```bash
# Compare any expression
./compare_parsers.sh "2 + 3"
./compare_parsers.sh "f(x).result"
./compare_parsers.sh "obj.foo.bar"

# Output JSON from Rust parser
echo "2 + 3" | cargo run --bin to_pyret_json
```

**Status**: All implemented syntax verified identical! ✅

See [PARSER_COMPARISON.md](PARSER_COMPARISON.md) for full documentation.

## 📚 Key Features

### Whitespace-Sensitive Parsing

Pyret distinguishes function calls from parenthesized expressions by whitespace:

```pyret
f(x)    // Function application: f called with argument x
f (x)   // Function f applied to parenthesized expression (x)
```

The tokenizer produces different tokens (`ParenNoSpace` vs `ParenSpace`) and the parser handles them correctly.

### No Operator Precedence

Pyret has **NO operator precedence**. All binary operators have equal precedence and are strictly left-associative:

```pyret
2 + 3 * 4    // Evaluates as (2 + 3) * 4 = 20
             // NOT as 2 + (3 * 4) = 14
```

Users must use explicit parentheses to control evaluation order.

### Exact AST Matching

The parser generates JSON that matches the reference Pyret implementation exactly:

```json
{
  "type": "s-app",
  "l": { ... },
  "_fun": { "type": "s-id", "id": { "type": "s-name", "s": "f" } },
  "args": [
    { "type": "s-num", "n": 42 }
  ]
}
```

## 🔗 References

- **Pyret Language:** https://www.pyret.org/
- **Reference Implementation:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang`
- **Grammar Spec:** `/pyret-lang/src/js/base/pyret-grammar.bnf`

## 📋 Implementation Progress

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Foundation | ✅ Complete | 100% |
| 2. Parser Core | ✅ Complete | 100% |
| 3. Expressions | 🚧 In Progress | 35% |
| 4. Control Flow | ⏳ Pending | 0% |
| 5. Functions & Bindings | ⏳ Pending | 0% |
| 6. Data Definitions | ⏳ Pending | 0% |
| 7. Type System | ⏳ Pending | 0% |
| 8. Imports/Exports | ⏳ Pending | 0% |
| 9. Tables | ⏳ Pending | 0% |
| 10. Testing & Reactors | ⏳ Pending | 0% |

See [PARSER_PLAN.md](PARSER_PLAN.md) for detailed roadmap.

## 💡 Examples

### Simple Expression
```pyret
1 + 2
```
```rust
SOp {
    op: "op+",
    left: SNum { n: 1.0 },
    right: SNum { n: 2.0 }
}
```

### Function Call
```pyret
f(x, y)
```
```rust
SApp {
    _fun: SId { id: SName { s: "f" } },
    args: [
        SId { id: SName { s: "x" } },
        SId { id: SName { s: "y" } }
    ]
}
```

### Chained Calls
```pyret
f(x)(y)
```
```rust
SApp {
    _fun: SApp {
        _fun: SId { s: "f" },
        args: [SId { s: "x" }]
    },
    args: [SId { s: "y" }]
}
```

## 🐛 Known Limitations

- Rational number parsing incomplete (numerator/denominator not extracted)
- Only basic expressions implemented so far
- No control flow yet (if, cases, for, when)
- No function definitions yet
- No type annotations yet

## 🛠️ Development

Built with:
- Rust 1.74+
- `serde` for JSON serialization
- `thiserror` for error handling

No external parser generators - pure hand-written recursive descent.

## 📄 License

[License details to be added]

---

**Last Updated:** 2025-10-31
**Maintainer:** [To be determined]
**Contributing:** See [NEXT_STEPS.md](NEXT_STEPS.md)
