# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-11-04 - LATEST UPDATE)

**Test Results: 131/133 tests passing (98.5%)** 🎉
- ✅ **131 tests PASSING** (98.5%) - **100% of non-ignored tests!**
- ⏸️ **2 tests IGNORED** (valid features: tables, spy)
- ❌ **0 tests FAILING**
- 🗑️ **11 tests DELETED/FIXED** (invalid Pyret syntax)

**All passing tests produce byte-for-byte identical ASTs to the official Pyret parser!** ✨

### 🏆 MAJOR MILESTONE: test-equality.arr Fully Parses with IDENTICAL AST! ✅

**The 364-line test-equality.arr file from the official Pyret test suite now produces a 100% IDENTICAL AST!**

### Latest Completion: Prelude Ordering Fix! ✅

**This session's achievements:**
- 🔧 **Fixed prelude statement ordering** - Parser now allows import/provide in any order ✨ **[NEW!]**
  - Modified `parse_prelude()` to loop through provide/import statements
  - Follows Pyret grammar: `(provide-stmt|import-stmt)*`
  - Previously required provides before imports, now properly interleaved
  - **1 new test passing**: `test_realistic_module_structure`
- 🐛 **Fixed test_realistic_module_structure** - Corrected invalid syntax
  - Changed `provide { Tree, make-tree } end` to `provide: Tree, make-tree end`
  - Moved provide to prelude (before code) per Pyret grammar requirements
  - Verified with official Pyret parser - `provide { ... }` is NOT valid syntax
- 📊 **Test coverage improved** - From 97.7% to 98.5% (+1 test!)
- 🎉 **131 tests now passing** - Up from 130 (131/133 total = 100% of non-ignored!)
- 🎯 **Only 2 tests remaining**: Table literals and Spy expressions (both valid features)

**Previous session achievements:**
- 🎯 **Implemented check operator refinements** - `is%(refinement)`, `is-not%(refinement)`, etc.
  - Syntax: `3 is%(within(1)) 4` allows custom equality checking
  - Parses `%` after check operators and captures refinement expression
  - Properly unwraps parentheses to match official parser AST structure
- 🔧 **Fixed binary operators in check test right-hand side**
  - Can now parse `BIG is%(within-rel(TOL)) BIG * (1 + TOL)`
  - Created `parse_binop_expr_no_check()` to handle full expressions on RHS
  - Check tests can now have complex expressions with `+`, `*`, `/`, etc.
- 🧹 **Fixed comment handling**
  - Comments and block comments now filtered out like official parser
  - Matches Pyret behavior: `ignore: new Set(["WS", "COMMENT"])`
- 📊 **Test coverage improved** - From 93.2% to 95.9% (+2.7 percentage points!)
- 🎉 **118 tests now passing** - Up from 110 (118/123 total = 100% of non-ignored!)

**Previous session achievements:**
- 🎯 **Implemented complete type system** - All 3 type features now working!
  1. ✅ **Any type annotation** - `x :: Any = 42`
  2. ✅ **Generic function type parameters** - `fun identity<T>(x :: T) -> T: x end`
  3. ✅ **Generic data type parameters** - `data List<T>: | empty | link(first :: T, rest :: List<T>) end`
  4. ✅ **Parameterized type application** - `List<T>`, `Map<K, V>` in type annotations
- 📊 **Test coverage improved** - From 90.7% to 93.2% (+2.5 percentage points!)
- 🎉 **110 tests passing** - Up from 107 (110/118 total)

**Previous session achievements:**
- 🧹 **Test cleanup** - Removed 8 invalid tests, enabled 7 passing tests
- 📊 **Test percentage** - From 79.4% to 90.7% (+11.3 percentage points!)

**Previous session achievements (9 tests):**
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

# Run comparison tests only (124 passing, 5 ignored)
cargo test --test comparison_tests

# Run ignored tests to see what needs work (5 tests)
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
└── comparison_tests.rs  (~1,360 lines) - 118 integration tests
    ├── 107 passing (90.7% coverage) ✅
    └── 11 ignored (advanced features: types, objects, imports, tables)
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
- **Object extension** `obj.{ field: value }` ✨ **[THIS SESSION]**
- **Object update** `obj.{ x: 10 }` (same syntax as extension) ✨ **[THIS SESSION]**
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
- **Data with sharing clauses** `sharing: method size(self): ... end` ✨ **[PREVIOUS SESSION]**
- Data with where clauses
- Check operators `is`, `raises`, `satisfies`, `violates`

### Testing ✅
- **Check blocks** `check: 1 + 1 is 2 end`
- **Check blocks with names** `check "test name": ... end`
- Check test statements with `is`, `raises`, `satisfies`, `violates`
- **Check operator variants** ✨ **[THIS SESSION]**
  - `is==`, `is=~`, `is<=>` (custom equality comparators)
  - `is-not==`, `is-not=~`, `is-not<=>` (negated variants)
- **Check operator refinements** `is%(within(1))`, `is-not%(refinement)`

### Modules ✅
- Import statements `import mod as M`
- Provide statements `provide *`

### Advanced Features ✅
- Chained postfix operators `obj.foo().bar().baz()`
- Ultra-complex nested expressions
- Program structure with prelude and body

## 🔴 Features Not Yet Implemented (2 Ignored Tests - All Valid!)

**All remaining ignored tests have been verified against the official Pyret parser.** These represent real features worth implementing.

**Parser is now 98.5% complete!** Only 2 tests remaining, both for valid Pyret features.

### ⚠️ Features That DO NOT Exist in Pyret (Removed!)
The following features were tested and **removed** as they don't exist in Pyret:
- ❌ **Unary operators** - `not x` or `-x` (use `not(x)` and `0 - x`)
- ❌ **String interpolation** - `` `Hello $(name)` `` (backticks are for multi-line strings only)
- ❌ **Rest parameters** - `fun f(x, rest ...): ...` (the `...` syntax doesn't exist)
- ❌ **Union type annotations** - `x :: (Number | String)` (the `|` syntax doesn't exist)
- ❌ **Contract syntax on functions** - `fun f(x) :: (Number -> Number): ...`
- ❌ **For-when guards** - `for map(x from list) when x > 2: ...` (use `for filter` instead)
- ❌ **Computed object properties** - `{ [key]: value }` (doesn't exist)
- ❌ **Check examples blocks** - `check: examples: | input | output | ...`

### ✅ Method Expressions (COMPLETED THIS SESSION!)
- ✅ Method expressions: `m = method(self): body end` ✨
- ✅ Method with arguments: `method(self, x, y): x + y end` ✨
- ✅ AST node: `s-method` with `args`, `body`, `name`, etc.
- ✅ Unblocked test-equality.arr! Now parses 100% with IDENTICAL AST!

### ✅ For Each Iterations (COMPLETED THIS SESSION!)
- ✅ `for each(x from list): body end` ✨
- ✅ `for each2(x from l1, y from l2): body end` ✨
- ✅ Complex bodies with multiple statements ✨

### ✅ If Block Syntax (COMPLETED THIS SESSION!)
- ✅ `if cond block: body end` syntax ✨
- ✅ Sets `blocky` field correctly to match official parser

### ✅ Object Extension (COMPLETED PREVIOUS SESSION!)
- ✅ Object extension: `point.{ z: 0 }` ✨
- ✅ Object update: `point.{ x: 10 }` ✨
- ✅ Distinguishes `.{number}` (tuple access) from `.{fields}` (extension)
- ✅ AST nodes: `SExtend` and `SUpdate` (both serialize as `s-extend`)

### ✅ Check Operator Variants (COMPLETED PREVIOUS SESSION!)
- ✅ `is==`, `is=~`, `is<=>` operators ✨
- ✅ `is-not==`, `is-not=~`, `is-not<=>` operators ✨
- ✅ Tokenizer support for multi-character operators with `=` and `<`
- ✅ Parser creates `SOpIsOp` and `SOpIsNotOp` with operator names

### ✅ Check Operator Refinements (COMPLETED PREVIOUS SESSION!)
- ✅ Refinement syntax: `is%(refinement-fn)`, `is-not%(refinement-fn)`
- ✅ Complex right-hand expressions: `BIG is%(within-rel(TOL)) BIG * (1 + TOL)`
- ✅ Comment filtering: Comments properly ignored during parsing

### ✅ Type System (COMPLETED PREVIOUS SESSION!)
- ✅ Function type annotations with arrow: `fun f(x) -> Number: ...`
- ✅ `Any` type annotation: `x :: Any = 42`
- ✅ Generic function type parameters: `fun identity<T>(x :: T) -> T: x end`
- ✅ Generic data type parameters: `data List<T>: | empty | link(first :: T, rest :: List<T>) end`
- ✅ Parameterized type application: `List<T>`, `Map<K, V>` in type annotations

### Advanced Import/Export (1 test) - ✅ VALID - 🔥 **NEXT PRIORITY**
- File imports: `import file("util.arr") as U`
- Provide with types: `provide-types *`
- Provide with specific exports: `provide { foo, bar } end`
- Module structures with multiple imports/exports

### Table Features (1 test) - ✅ VALID
- Table literals: `table: name, age row: "Alice", 30 end`
- Requires significant parser work

### Spy Expressions (1 test) - ✅ VALID
- Spy expressions: `spy: x end`
- May already parse, needs investigation

## 🎯 NEXT STEPS: Implement Remaining Features (2 Tests Remaining)

**Parser is 98.5% complete!** Only 2 tests remaining, both for valid Pyret features.

### Remaining Features (2 Tests):

1. **Table Literals** (1 test, ~4-6 hours)
   - `table: name, age row: "Alice", 30 end`
   - Requires significant parser work
   - Grammar: `table-expr: TABLE COLON column-names [ROW COLON values]* END`

2. **Spy Expressions** (1 test, ~1-2 hours)
   - `spy: x end`
   - Grammar: `spy-expr: SPY COLON expr END`
   - May require minimal work - stub already exists in parser

### 🔥 **RECOMMENDED NEXT STEPS:**

1. **Try Spy Expressions First** (~1-2 hours)
   - Quick win - may already be mostly implemented
   - Check if `parse_spy_stmt()` stub just needs to be connected
   - Add to `parse_prim_expr()` to handle `TokenType::Spy`

2. **Then Table Literals** (~4-6 hours)
   - More complex feature requiring multiple parsing functions
   - `parse_table_expr()` stub exists but needs full implementation
   - Need to parse column names and row data

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
**118/123 comparison integration tests passing** ✅ (95.9%)

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

**Advanced Features: ~70% Complete** ⚠️
- Where clauses ✅
- **Cases-else, wildcards, nesting** ✅ **[COMPLETED]**
- **For-filter, fold, cartesian, nesting** ✅ **[COMPLETED]**
- **Data sharing clauses** ✅ **[COMPLETED]**
- **Check blocks (basic)** ✅ **[COMPLETED THIS SESSION]**
- String interpolation (missing - requires tokenizer work)
- Rest parameters (missing)
- Generic types (missing)
- Table expressions (missing)
- Check blocks with examples (missing)
- ⚠️ Unary operators (DO NOT EXIST in Pyret)

**Overall: 96.1% Complete** (124/129 tests passing)

## 🎉 Ready to Code!

The codebase is clean, well-tested, and ready for the next features:

1. Start with [TEST_STATUS_REPORT.md](TEST_STATUS_REPORT.md) to see the big picture
2. Look at the ignored tests in `tests/comparison_tests.rs`
3. Follow the implementation pattern from recent work
4. Run tests and validate with `./compare_parsers.sh`

**🚀 RECOMMENDED NEXT STEPS - START HERE:**

1. **🔥 IMMEDIATE: Implement Method Expressions** (~3-4 hours)
   - Highest priority - blocks test-equality.arr at line 213 (58%)
   - Parse `method(self): body end` as standalone expressions
   - AST node already exists: `SMethod` in `src/ast.rs`
   - See NEXT_STEPS section above for details

2. **Advanced imports/exports** (~4-6 hours)
   - File imports, provide-types, etc.
   - Critical for real programs

3. **Table literals** (~4-6 hours)
   - `table:` expression support
   - Moderate complexity

4. **Spy expressions** (~1-2 hours)
   - May already parse, needs investigation
   - Quick win if it works

---

**Last Updated:** 2025-11-03 (Latest)
**Tests:** 72/72 parser tests ✅ (100%), 124/129 comparison tests ✅ (96.1%)
**This Session Completed:**
- 🎯 **Object Extension** - `obj.{ field: value, ... }` ✨ **[NEW!]**
  - ✅ Syntax: `point.{ z: 0 }` extends objects with new fields
  - ✅ Distinguishes `.{number}` (tuple access) from `.{fields}` (object extension)
  - ✅ Parser handles empty extensions, multiple fields, trailing commas
  - ✅ AST nodes: `SExtend` and `SUpdate` (both serialize as `s-extend`)
  - ✅ JSON serialization added to `src/bin/to_pyret_json.rs`
  - ✅ **2 new tests passing**: `test_object_extension`, `test_object_update_syntax`
- 🔧 **Check Operator Variants** - `is<op>`, `is-not<op>` ✨ **[NEW!]**
  - ✅ Added: `is==`, `is=~`, `is<=>` (custom equality comparators)
  - ✅ Added: `is-not==`, `is-not=~`, `is-not<=>` (negated variants)
  - ✅ Fixed tokenizer: special handling for multi-character operators with `=` and `<`
  - ✅ Parser creates `SOpIsOp` and `SOpIsNotOp` with operator names
  - ✅ **4 new tests passing**: check operator variant tests
- 📊 **Test coverage improved** - From 95.9% to 96.1% (+6 tests!)
- 🎉 **124 tests now passing** - Up from 118 (124/129 total = 100% of non-ignored!)
- ✨ **test-equality.arr progress** - Now parses to line 213/364 (58%, up from 36%!)
  - **Next blocker**: Method expressions (`method(self): body end`)
**Implementation Details:**
- **Object Extension**: Modified postfix expression parser (src/parser.rs:849-981)
  - Detects `.{` and checks if next token is `Number` (tuple) or field name (extension)
  - Parses object fields using existing `parse_obj_field()` function
  - Creates `Expr::SExtend` with `supe` (super object) and `fields`
- **Check Operators**: Fixed tokenizer (src/tokenizer.rs:694-737)
  - Added early checks for `is==`, `is=~`, `is<=>`, `is-not==`, etc. before identifier scanning
  - Needed because `=` and `<` aren't valid identifier characters
  - Parser recognizes new token types in `is_check_op()` and `parse_check_op()`
**Progress:** 124/129 passing (96.1%), only 5 tests remaining (100% of non-ignored!)
**Next Session:** Method expressions (2 tests) - **PRIORITY**, blocks test-equality.arr at line 213
