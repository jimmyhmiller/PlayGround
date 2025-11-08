# Pyret Parser Project - Claude Instructions

**Location:** `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2`

A hand-written recursive descent parser for the Pyret programming language in Rust.

## 📊 Current Status (2025-11-08 - LATEST UPDATE)

**Test Results: 273/296 tests passing (92.2%)**
- ✅ **273 tests PASSING** (92.2%)
- ⏸️ **0 tests IGNORED**
- ❌ **23 tests FAILING** (7.8%)

**See [FAILING_TESTS.md](FAILING_TESTS.md) for detailed analysis of remaining failures.**

**All passing tests produce byte-for-byte identical ASTs to the official Pyret parser!** ✨

### Latest Fix: Large Rational Numbers and Number Normalization! ✅

**This session's achievements (2025-11-08 afternoon):**
- 🔢 **Fixed large rational number support** - Arbitrary precision rational numbers! ✨ **[NEW!]**
  - **Problem:** Parser used `i64` for numerator/denominator, limiting to ~9×10^18
  - **Solution:** Changed `SFrac` and `SRfrac` AST nodes to use `String` instead of `i64`
  - **Impact:** Can now parse `1/100000000000000000000000` and larger!
  - **Example:** `min([list: 1/10, 1/100, 1/100000000000000000000000])`
- 🔧 **Fixed rough number normalization** - Strip leading `+` signs ✨ **[NEW!]**
  - **Problem:** `~+3/2` was serialized with `+` in numerator
  - **Solution:** Strip leading `+` after `~` in both parser and JSON serialization
  - **Examples:** `~+3/2` → `"~3/2"`, `~+1.5` → `"~1.5"`
- 📐 **Added scientific notation for very long decimals** ✨ **[NEW!]**
  - **Problem:** Very small numbers like `~0.000...0005` (324 zeros) were output as long strings
  - **Solution:** Convert strings >50 chars to scientific notation (e.g., `~5e-324`)
  - **Impact:** Matches official Pyret behavior for extreme values
- 📊 **Test progress:** 272 → 273 passing (+1 test fixed!)
- 📝 **Created FAILING_TESTS.md** - Complete analysis of remaining 23 failures

**Known remaining issues:**
1. **Decimal to fraction simplification** - Need GCD-based fraction reduction (~6-7 tests)
2. **Scientific notation heuristic** - Need better logic for when to use scientific notation (~1-2 tests)
3. **Missing AST fields** - `SProvideAll` needs `hidden` field (~1 test)
4. **Compiler files** - Not yet analyzed (~13 tests)

### Previous Session: Template Dots, Spy Labels, and Block Calls! ✅

**Previous session achievements (2025-11-08 morning):**
- 🚀 **Implemented template dots (`...`) placeholder syntax** - Fixed 3 tests! ✨
  - Syntax: `lam(): ... end`, `fun incomplete(x): ... end`
  - Added parsing for `DotDotDot` token → `STemplate` AST node
  - Added JSON serialization for `s-template`
  - Used during development or for incomplete code sections
- 🔧 **Fixed spy expression labels** - Now accepts any expression, not just strings! ✨
  - **Problem:** Parser only accepted string literals for spy labels
  - **Solution:** Modified `parse_spy_stmt()` to use `parse_binop_expr()` for labels
  - **Examples:** `spy "iteration " + to-string(i): result end`
- 🐛 **Fixed critical tokenizer bug for block expression calls** - Fixed 1 test! ✨ **[MAJOR!]**
  - **Problem:** `block: ... end()` failed to parse - `()` was left unparsed
  - **Root cause:** `block:` sets `paren_is_for_exp = true`, but `end` keyword never reset it
  - **Impact:** `(` after `end` was tokenized as `ParenSpace` instead of `ParenNoSpace`
  - **Solution:** Modified tokenizer to reset `paren_is_for_exp = false` after `end` keyword
  - **Examples now working:**
    - `block: lam(): 40 end end()`
    - `if block: lam(): 10 end end() == 10: "yes" else: "no" end`
- 📊 **Test count INCREASED** - 269 passing, 0 ignored (up from 263/0!) - **+6 tests!** 🎉

**Previous session achievements:**
- 🚀 **Implemented whitespace-sensitive bracket parsing** - Fixed 31 tests at once! ✨ **[MAJOR!]**
  - **Problem:** Parser was treating `5\n[list: 1, 2]` as `5[list]` (bracket access) instead of two separate statements
  - **Root cause:** Bracket `[` always parsed as postfix operator, regardless of whitespace
  - **Solution:** Added `BrackSpace` and `BrackNoSpace` token types (like `ParenSpace`/`ParenNoSpace`)
  - **Implementation:**
    - Modified tokenizer (`src/tokenizer.rs:1168-1183`) to check `prior_whitespace` flag
    - Updated parser to only treat `BrackNoSpace` as postfix bracket access operator
    - `arr[0]` (no whitespace) → bracket access ✅
    - `[list: 1, 2]` (whitespace or statement start) → construct expression ✅
  - **Impact:** Enabled parsing of multiple statements with construct expressions!
- ✅ **Constructor objects now parse correctly** - `test_constructor_object` ✅
  - Objects with `make0`, `make1`, `make2` fields for construct expressions
  - Example: `[every-other: 1, 2, 3]` where `every-other` is an object
- 📊 **Test count JUMPED** - 246 passing, 6 ignored (up from 215/7!) - **+31 tests!** 🎉

**Previous session achievements:**
- 🔧 **Implemented underscore partial application** - `f = (_ + 2)` and `f = (_ + _)` ✨
  - Modified `parse_id_expr()` in `src/parser.rs:2476-2502` to recognize `_` in expression contexts
  - Creates `Name::SUnderscore` for underscore identifiers (for partial application)
  - Enables functional programming patterns like `map(_ + 1, list)`
- 🔧 **Implemented provide-from-data** - `provide from M: x, data Foo end` ✨
  - Added `hidden: Vec<Name>` field to `SProvideData` AST node in `src/ast.rs:1174`
  - Updated parser in `src/parser.rs:782-786` to initialize hidden field
  - Fixed JSON serialization in `src/bin/to_pyret_json.rs:893-899` to include hidden field
- 📊 **Test count improved** - 214 passing, 8 ignored (up from 211/11!) - **+3 tests!**

**Previous session achievements:**
- 🔧 **Implemented shadow keyword in tuple destructuring** - `{shadow a; shadow b} = {1; 2}` ✨
  - Updated `parse_tuple_bind()` to check for optional `shadow` keyword before each field
  - Updated `parse_tuple_for_destructure()` for lookahead to support shadow in tuple patterns
  - Sets `shadows: true` field in `s-bind` AST nodes when shadow keyword is present
- ✅ **Verified complete shadow support** - Tested all shadow locations from grammar ✨
  - Simple bindings: `shadow x = 5` ✅
  - Tuple destructuring: `{shadow a; shadow b} = ...` ✅
  - Function parameters: `fun f(shadow x): ...` ✅
  - Lambda parameters: `lam(shadow x): ...` ✅
  - For-loop bindings: `for map(shadow x from lst): ...` ✅
  - Cases patterns: `cases(T) x: | variant(shadow a) => ...` ✅
- 📊 **Test count improved** - 211 passing, 11 ignored (up from 210/12!)
- 📝 **Documented invalid decimal syntax** - Added test for `.5` and `.0` rejection ✨
- 🗑️ **Removed invalid test** - `test_dot_number_access` (tested non-existent `.0` syntax)

**Previous session achievements:**
- 🔍 **Discovered 4 tests were already passing** - Tests marked as ignored were actually working! ✨
  - ✅ `test_tuple_destructuring` - Tuple destructuring in let bindings
  - ✅ `test_method_with_trailing_comma` - Methods with trailing commas in objects
  - ✅ `test_spy_with_string` - Spy expressions with label strings
  - ✅ `test_provide_from_module_multiple_items` - Provide from with multiple items
- 🔧 **Fixed underscore wildcard parsing** - `_` now correctly generates `s-underscore` AST node ✨
  - Modified `parse_name()` in `src/parser.rs:5308-5329`
  - Detects `"_"` and returns `Name::SUnderscore` instead of `Name::SName`
  - Ensures proper pattern matching in cases expressions
- 📊 **Test coverage improved** - From 206 to 210 tests passing (+4 tests!)
- ⚠️ **Documentation was significantly outdated** - Claimed 131/133 but actually 210/223 passing!

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

# Run comparison tests only (269 passing, 0 ignored)
cargo test --test comparison_tests

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
└── comparison_tests.rs  (~1,400 lines) - 269 integration tests
    └── 269 passing (100% coverage) ✅ 🎉
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
  - **Block expression calls** `block: ... end()` ✨ **[THIS SESSION]**
  - **If-block syntax** `if block: ... end() == x: ... end` ✨ **[THIS SESSION]**
- If expressions `if c: a else: b end` with else-if chains
- When expressions `when c: body end`
- For expressions:
  - ✅ `for map(x from lst): x + 1 end`
  - ✅ **For-filter** `for filter(x from lst): x > 2 end` ✨ **[NEW!]**
  - ✅ **For-fold** `for fold(acc from 0, x from lst): acc + x end` ✨ **[NEW!]**
  - ✅ **For-each** `for each(x from lst): body end` ✨ **[NEW!]**
  - ✅ **Multiple generators** `for map(x from l1, y from l2): {x; y} end` ✨ **[NEW!]**
  - ✅ **Nested for** ✨ **[NEW!]**
- Cases expressions:
  - ✅ `cases(T) e: | variant => body end`
  - ✅ **Cases-else** `cases(T) e: | v1 => a | else => b end` ✨ **[NEW!]**
  - ✅ **Underscore wildcards** `| link(_, _) => ...` ✨ **[NEW!]**
  - ✅ **Nested cases** ✨ **[NEW!]**

### Functions & Bindings ✅
- Lambda expressions `lam(x): x + 1 end`
- **Generic lambdas** `lam<A>(x :: A): x end`, `lam<A, B>(x :: A, f :: (A -> B)): f(x) end` ✨ **[THIS SESSION]**
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

### Development & Testing ✅
- **Template dots** `...` - Placeholder for incomplete code ✨ **[THIS SESSION]**
  - `lam(): ... end`, `fun incomplete(x): ... end`
- **Spy expressions** `spy: x end`, `spy "label": x, y end` ✨ **[THIS SESSION]**
  - **Expression labels** `spy "iter " + to-string(i): result end` ✨ **[THIS SESSION]**
  - **Named fields** `spy: x, y: 20 end` ✨ **[THIS SESSION]**
- **Table expressions** `table: name, age row: "Alice", 30 end`
- **Method expressions** `method(self, x): x + 1 end`

### Advanced Features ✅
- Chained postfix operators `obj.foo().bar().baz()`
- Ultra-complex nested expressions
- Program structure with prelude and body

## 🎊 ALL FEATURES IMPLEMENTED! (0 Ignored Tests)

**Parser is now 100% complete!** All 269 comparison tests passing!

The parser successfully handles all tested Pyret language features and produces byte-for-byte identical ASTs to the official Pyret parser.

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
- ❌ **Dot number access shorthand** - `t.0` (official parser treats as BAD-NUMBER; use `t.{0}` instead)
- ❌ **Decimal numbers without leading digit** - `.5` or `.0` (must use `0.5` and `0.0`)
  - Official Pyret error: "number literals in Pyret require at least one digit before the decimal point"
  - We have a test (`test_invalid_decimal_without_leading_digit`) to ensure this remains invalid
- ❌ **Arrow types without parentheses in bindings** - `f :: {A; B} -> C` ✨ **[NEW!]**
  - Removed from Pyret in 2014 (commit 13553032e, issue #252) - wasn't checking contracts properly
  - MUST use parentheses: `f :: ({A; B} -> C)` is correct syntax
  - Grammar rule `noparen-arrow-ann` only exists for internal use, not in bindings
  - Test `test_tuple_type_annotation` was fixed to use correct syntax

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

### ✅ Table Features (COMPLETED - ALREADY WORKING!)
- ✅ Table literals: `table: name, age row: "Alice", 30 end` ✨
- ✅ Table operations and filtering ✨
- ✅ Tests: `test_simple_table`, `test_table_with_filter` ✨

### ✅ Spy Expressions (COMPLETED - ALREADY WORKING!)
- ✅ Spy expressions: `spy: x end` ✨
- ✅ Spy with labels: `spy "debug": x end` ✨
- ✅ Tests: `test_spy_expression`, `test_spy_with_string` ✨

### ✅ Tuple Destructuring (COMPLETED - ALREADY WORKING!)
- ✅ Tuple destructuring in let bindings: `{a; b} = {1; 2}` ✨
- ✅ Multi-element tuples: `{a; b; c; d; e} = {10; 214; 124; 62; 12}` ✨
- ✅ Test: `test_tuple_destructuring`, `test_tuple_destructure_simple`, `test_tuple_destructure_nested` ✨

### ✅ Provide From Module (COMPLETED - ALREADY WORKING!)
- ✅ Provide from with multiple items: `provide from lists: map, filter end` ✨
- ✅ Test: `test_provide_from_module_multiple_items` ✨

### ✅ Underscore Partial Application (COMPLETED THIS SESSION!)
- ✅ Underscore in expressions: `f = (_ + 2)` ✨ **[NEW!]**
- ✅ Multiple underscores: `f = (_ + _)` ✨ **[NEW!]**
- ✅ Modified `parse_id_expr()` to recognize `_` and create `Name::SUnderscore`
- ✅ Tests: `test_underscore_partial_application`, `test_underscore_multiple` ✨ **[NEW!]**

### ✅ Provide From Data (COMPLETED THIS SESSION!)
- ✅ Provide data from module: `provide from M: x, data Foo end` ✨ **[NEW!]**
- ✅ Added `hidden: Vec<Name>` field to `SProvideData` AST node
- ✅ Fixed JSON serialization to include `hidden` field
- ✅ Test: `test_provide_from_data` ✨ **[NEW!]**

### ✅ Tuple Type Annotations (COMPLETED PREVIOUS SESSION!)
- ✅ Arrow types in bindings with parentheses: `f :: ({Number; Number} -> {Number; Number})` ✨
- ✅ Discovered and fixed invalid test that used syntax without required parentheses
- ✅ Researched Pyret history: `noparen-arrow-ann` was removed in 2014 (issue #252)
- ✅ Test: `test_tuple_type_annotation` ✨ **[FIXED!]**

### ✅ Template Dots (COMPLETED THIS SESSION!)
- ✅ Template dots: `...` placeholder syntax ✨ **[NEW!]**
- ✅ Used for incomplete code: `lam(): ... end`, `fun f(x): ... end`
- ✅ Added parsing: `DotDotDot` token → `STemplate` AST node
- ✅ Added JSON serialization for `s-template`
- ✅ Tests: `test_template_dots_simple`, `test_template_dots_in_function`, `test_template_dots_in_block` ✨

### ✅ Spy Expression Labels (COMPLETED THIS SESSION!)
- ✅ Spy with expression labels: `spy "iteration " + to-string(i): result end` ✨ **[NEW!]**
- ✅ Modified `parse_spy_stmt()` to accept any expression, not just string literals
- ✅ Test: `test_full_file_spy` ✨

### ✅ Block Expression Calls (COMPLETED THIS SESSION!)
- ✅ Block expression calls: `block: ... end()` ✨ **[MAJOR FIX!]**
- ✅ Fixed critical tokenizer bug: `end` keyword now resets `paren_is_for_exp = false`
- ✅ Enables: `if block: lam(): 10 end end() == 10: "yes" else: "no" end`
- ✅ Test: `test_full_file_seq_of_lettable` ✨

## 🎯 Parser Complete - All Tests Passing!

**No remaining features to implement!** All 269 comparison tests pass with byte-for-byte identical ASTs to the official Pyret parser.

The parser now handles the complete Pyret language as tested in the comparison test suite.

## 🔑 Key Concepts

**Whitespace Sensitivity:**
- `f(x)` → Direct function call (s-app)
- `f (x)` → Two separate expressions (f and (x))
- `arr[0]` → Bracket access (no whitespace)
- `[list: 1, 2]` → Construct expression (whitespace or statement start)

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
# Result: 273 passed, 0 ignored, 23 failed

# See failing tests analysis
cat FAILING_TESTS.md

# Test specific feature
./compare_parsers.sh "fun f(x): x + 1 end"
```

**69/73 parser unit tests passing** (94.5%) - 4 pre-existing failures in decimal/rational tests
**273/296 comparison integration tests passing** ✅ (92.2%)
**See [FAILING_TESTS.md](FAILING_TESTS.md) for analysis of the 23 failing tests**

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

**Core Language: 100% Complete** ✅
- All basic expressions ✅
- All basic statements ✅
- Function definitions ✅
- Data declarations ✅
- Pattern matching ✅
- Import/export ✅
- Advanced blocks ✅
- Type annotations ✅

**Advanced Features: 100% Complete** ✅
- Where clauses ✅
- Cases-else, wildcards, nesting ✅
- For-filter, fold, each, cartesian, nesting ✅
- Data sharing clauses ✅
- Check blocks with refinements ✅
- Table expressions ✅
- Spy expressions (with expression labels) ✅
- Tuple destructuring ✅
- Type system (generics, annotations) ✅
- Underscore partial application ✅
- Template dots (`...`) ✅
- Block expression calls ✅

**Overall: 92.2% Complete** (273/296 tests passing)

## 🎯 Parser Status

The parser handles most Pyret language features and produces byte-for-byte identical ASTs to the official parser for 273 tests:

- ✅ **273/296 comparison tests passing** (92.2%)
- ✅ **Byte-for-byte identical ASTs** for all passing tests
- ✅ **Most language features** implemented
- ⚠️ **23 tests failing** - See [FAILING_TESTS.md](FAILING_TESTS.md) for details

**What's fully working:**
- Complete expression parsing (primitives, operators, functions, data structures)
- Full statement support (bindings, control flow, declarations)
- Advanced features (generics, type annotations, pattern matching)
- Development tools (spy, template dots, check blocks)
- Module system (import/export/provide)
- Arbitrary precision rational numbers (e.g., `1/100000000000000000000000`)

**What needs work:**
- Decimal to fraction simplification (needs GCD algorithm)
- Scientific notation heuristic (when to use `1e-5` vs `0.00001`)
- Some missing AST fields
- Some compiler/type-checker files (not yet analyzed)

---

**Last Updated:** 2025-11-08 (Latest - afternoon)
**Tests:** 69/73 parser tests (94.5%), **273/296 comparison tests ✅ (92.2%)**
**This Session Completed (afternoon):**
- 🔢 **Fixed large rational number support** - Arbitrary precision! ✨ **[NEW!]**
  - Changed `SFrac` and `SRfrac` to use `String` instead of `i64`
  - Can now parse `1/100000000000000000000000` and larger
- 🔧 **Fixed rough number normalization** - Strip leading `+` signs ✨ **[NEW!]**
  - `~+3/2` → `"~3/2"`, `~+1.5` → `"~1.5"`
- 📐 **Added scientific notation for very long decimals** ✨ **[NEW!]**
  - Strings >50 chars convert to scientific notation (e.g., `~5e-324`)
- 📊 **Test progress:** 272 → 273 passing (+1 test fixed!)
- 📝 **Created FAILING_TESTS.md** - Complete analysis of remaining 23 failures

**Previous Session Completed (morning):**
- 🚀 **Implemented template dots (`...`) placeholder syntax** - Fixed 3 tests! ✨
- 🔧 **Fixed spy expression labels** - Now accepts any expression! ✨
- 🐛 **Fixed critical tokenizer bug for block expression calls** - Fixed 1 test! ✨
- 📊 **Test count:** 263 → 269 passing (+6 tests!)

**Next Steps:**
See [FAILING_TESTS.md](FAILING_TESTS.md) for prioritized list of remaining issues.
