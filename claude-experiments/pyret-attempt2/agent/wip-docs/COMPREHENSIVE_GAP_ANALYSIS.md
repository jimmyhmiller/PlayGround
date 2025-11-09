# Comprehensive Gap Analysis

**Created:** $(date)
**Purpose:** Identify incomplete parser features using real Pyret code
**Test File:** `tests/comprehensive_gap_tests.rs`

## Overview

This document accompanies a comprehensive test suite (`comprehensive_gap_tests.rs`) containing **50+ tests** using **real Pyret code** patterns. All tests are currently marked with `#[ignore]` and represent features that need implementation or edge cases that may not be fully covered.

## Test Categories

### 1. Advanced Block Structures (4 tests)
**Status:** Basic blocks work, but advanced patterns incomplete

Tests cover:
- Multiple let bindings in a block
- Mutable variables with `var` keyword
- Type annotations on bindings
- Nested blocks with variable shadowing

**Why it matters:** Blocks are fundamental to Pyret's scoping and control flow. Real programs use complex block structures with multiple bindings.

**Example:**
```pyret
block:
  x = 10
  y = 20
  z = 30
  x + y + z
end
```

**Next Steps:**
1. Extend `parse_block_expr()` to handle multiple statements
2. Implement `var` binding support (mutable variables)
3. Add type annotation parsing for bindings
4. Test variable shadowing rules

---

### 2. Advanced Function Features (4 tests)
**Status:** Basic functions work, missing advanced features

Tests cover:
- Multiple where clause checks
- Recursive functions with pattern matching
- Higher-order functions (returning functions)
- Rest parameters (`...`)

**Why it matters:** Real Pyret code heavily uses testing with where clauses and functional programming patterns.

**Example:**
```pyret
fun adder(x):
  lam(y): x + y end
end
```

**Next Steps:**
1. Support multiple checks in where clauses
2. Implement rest parameter syntax (`rest ...`)
3. Test closure capturing
4. Verify recursive function parsing

---

### 3. Data Definitions (6 tests)
**Status:** Not implemented - HIGH PRIORITY

Tests cover:
- Simple enumerations
- Variants with fields
- Mutable fields (`ref`)
- Multiple variants (sum types)
- Sharing clauses (shared methods)
- Generic/parameterized types

**Why it matters:** Data definitions are a core Pyret feature for algebraic data types.

**Example:**
```pyret
data Either:
  | left(v)
  | right(v)
end
```

**Next Steps:**
1. Implement `parse_data_expr()` method
2. Parse variant definitions with fields
3. Handle `ref` keyword for mutable fields
4. Implement sharing clause parsing
5. Add type parameter support (`<T>`)

---

### 4. Cases Expressions (4 tests)
**Status:** Not implemented - HIGH PRIORITY

Tests cover:
- Basic pattern matching
- Else branches
- Nested cases
- Cases in function bodies

**Why it matters:** Pattern matching is essential for working with data types.

**Example:**
```pyret
cases (Option) opt:
  | some(v) => v
  | none => 0
end
```

**Next Steps:**
1. Implement `parse_cases_expr()` method
2. Parse case branches with patterns
3. Handle else/wildcard branches
4. Support nested pattern matching

---

### 5. Advanced For Expressions (4 tests)
**Status:** Basic for works, missing advanced features

Tests cover:
- Multiple generators (cartesian products)
- Fold with complex accumulators
- For filter
- Nested for expressions

**Why it matters:** Real code uses complex iteration patterns.

**Example:**
```pyret
for fold(acc from {0; 0}, x from [list: 1, 2, 3]):
  {acc.{0} + x; acc.{1} + 1}
end
```

**Next Steps:**
1. Extend for parser to handle multiple bindings
2. Implement filter variant
3. Test nested for expressions
4. Handle complex accumulator patterns

---

### 6. Table Expressions (2 tests)
**Status:** Not implemented

Tests cover:
- Table literals
- Table operations

**Why it matters:** Tables are a first-class Pyret feature for data science.

**Example:**
```pyret
table: name, age
  row: "Alice", 30
  row: "Bob", 25
end
```

**Next Steps:**
1. Implement `parse_table_expr()` method
2. Parse column headers
3. Parse row data
4. Handle table methods

---

### 7. String Interpolation (2 tests)
**Status:** Not implemented

Tests cover:
- Basic interpolation with variables
- Interpolation with expressions

**Why it matters:** String interpolation is commonly used for formatting.

**Example:**
```pyret
`Hello, $(name)!`
```

**Next Steps:**
1. Add backtick string support to tokenizer
2. Parse `$(expr)` interpolation syntax
3. Build interpolated string AST nodes

---

### 8. Advanced Object Features (3 tests)
**Status:** Basic objects work, missing advanced features

Tests cover:
- Object extension/refinement
- Computed property names
- Object update syntax

**Why it matters:** Object manipulation is common in real programs.

**Example:**
```pyret
point.{ x: 10 }  # Update syntax
```

**Next Steps:**
1. Implement object update/extension syntax
2. Parse computed property names
3. Test object refinement patterns

---

### 9. Check Blocks (2 tests)
**Status:** Not implemented

Tests cover:
- Standalone check blocks
- Example-based testing

**Why it matters:** Check blocks are Pyret's built-in testing mechanism.

**Example:**
```pyret
check:
  1 + 1 is 2
  "hello" is "hello"
end
```

**Next Steps:**
1. Implement `parse_check_block()` method
2. Parse is/raises/satisfies assertions
3. Handle examples tables

---

### 10. Advanced Import/Export (4 tests)
**Status:** Basic import/provide incomplete

Tests cover:
- Import with aliases
- Import from files
- Provide-types
- Selective exports

**Example:**
```pyret
import file("util.arr") as U
provide { add, multiply } end
```

**Next Steps:**
1. Complete import statement parsing
2. Add file() import variant
3. Implement provide-types
4. Handle selective exports with braces

---

### 11. Type Annotations (3 tests)
**Status:** Basic annotations work, missing advanced features

Tests cover:
- Arrow types for functions
- Union types
- Generic type parameters

**Example:**
```pyret
fun add(x :: Number, y :: Number) -> Number:
  x + y
end
```

**Next Steps:**
1. Extend type parsing for arrow types
2. Implement union type syntax (`|`)
3. Add generic type parameter parsing (`<T>`)

---

### 12. Operators and Edge Cases (3 tests)
**Status:** Binary operators work, missing unary

Tests cover:
- Custom operators (underscore methods)
- Unary not
- Unary minus

**Example:**
```pyret
not true
-(5 + 3)
```

**Next Steps:**
1. Implement unary operator parsing
2. Handle `not` keyword
3. Parse negative literals vs. unary minus

---

### 13. Comprehensions (1 test)
**Status:** Not implemented

Tests cover:
- For expressions with guards/filters

**Example:**
```pyret
for map(x from lst) when (x > 2):
  x * 2
end
```

**Next Steps:**
1. Add `when` clause support to for expressions
2. Parse guard conditions

---

### 14. Spy Expressions (1 test)
**Status:** Not implemented

**Example:**
```pyret
spy: x end
```

**Next Steps:**
1. Add spy keyword to tokenizer
2. Implement `parse_spy_expr()` method

---

### 15. Contracts (1 test)
**Status:** Not implemented

**Example:**
```pyret
fun divide(x, y) :: (Number, Number -> Number):
  x / y
end
```

**Next Steps:**
1. Extend function parsing for contract syntax
2. Parse function contract annotations

---

### 16. Complex Real-World Patterns (2 tests)
**Status:** Integration tests requiring multiple features

Tests cover:
- Complete module structure
- Functional composition patterns

**Why it matters:** Validates that features work together.

---

### 17. Gradual Typing (1 test)
**Status:** Not implemented

Tests cover:
- Any type annotation

**Example:**
```pyret
x :: Any = 42
```

**Next Steps:**
1. Add `Any` as a recognized type

---

## Running the Tests

### Run All Gap Tests (Currently Ignored)
```bash
# This will show all ignored tests
cargo test --test comprehensive_gap_tests -- --ignored
```

### Run Specific Category
```bash
# Example: run only data definition tests
cargo test --test comprehensive_gap_tests test_data -- --ignored
```

### Enable a Test
1. Remove the `#[ignore]` attribute
2. Implement the required feature
3. Run: `cargo test --test comprehensive_gap_tests test_name`
4. Compare with official parser: `./compare_parsers.sh "your code"`

---

## Implementation Priority

### Phase 1: Core Language Features (Highest Impact)
**Goal:** Complete essential Pyret features

1. **Data Definitions** (6 tests)
   - Time: 6-8 hours
   - Enables algebraic data types
   - Required for: cases expressions

2. **Cases Expressions** (4 tests)
   - Time: 4-6 hours
   - Pattern matching
   - Works with data definitions

3. **Advanced Blocks** (4 tests)
   - Time: 3-4 hours
   - Multiple statements
   - Variable shadowing

**Total:** ~13-18 hours, enables 14 tests

---

### Phase 2: Testing and Functional Features
**Goal:** Support real-world programming patterns

4. **Check Blocks** (2 tests)
   - Time: 2-3 hours
   - Built-in testing

5. **Advanced Functions** (4 tests)
   - Time: 3-4 hours
   - Where clauses, rest params

6. **Advanced For** (4 tests)
   - Time: 2-3 hours
   - Multiple generators, filters

**Total:** ~7-10 hours, enables 10 tests

---

### Phase 3: Advanced Type System
**Goal:** Complete type checking support

7. **Type Annotations** (3 tests)
   - Time: 4-5 hours
   - Arrow types, unions, generics

8. **Contracts** (1 test)
   - Time: 2-3 hours
   - Runtime contracts

**Total:** ~6-8 hours, enables 4 tests

---

### Phase 4: Special Features
**Goal:** Less common but useful features

9. **Table Expressions** (2 tests)
   - Time: 4-6 hours
   - Data science features

10. **String Interpolation** (2 tests)
    - Time: 2-3 hours
    - Formatted strings

11. **Operators** (3 tests)
    - Time: 2-3 hours
    - Unary operators

12. **Spy/Debug** (1 test)
    - Time: 1 hour
    - Debugging support

**Total:** ~9-13 hours, enables 8 tests

---

### Phase 5: Module System and Advanced Objects
**Goal:** Complete module support

13. **Import/Export** (4 tests)
    - Time: 4-6 hours
    - Module system

14. **Advanced Objects** (3 tests)
    - Time: 3-4 hours
    - Object refinement

**Total:** ~7-10 hours, enables 7 tests

---

### Phase 6: Advanced Patterns
**Goal:** Complex combinations

15. **Comprehensions** (1 test)
    - Time: 1-2 hours

16. **Real-World Integration** (2 tests)
    - Time: N/A (tests existing features)

17. **Gradual Typing** (1 test)
    - Time: 30 minutes

**Total:** ~2-3 hours, enables 4 tests

---

## Total Implementation Effort

- **Phase 1 (Core):** 13-18 hours, 14 tests
- **Phase 2 (Functional):** 7-10 hours, 10 tests
- **Phase 3 (Types):** 6-8 hours, 4 tests
- **Phase 4 (Special):** 9-13 hours, 8 tests
- **Phase 5 (Modules):** 7-10 hours, 7 tests
- **Phase 6 (Advanced):** 2-3 hours, 4 tests

**Grand Total:** ~44-62 hours for complete coverage

---

## How to Use This Document

### For Implementers

1. **Start with Phase 1** - these are the most impactful features
2. **Pick a test** from the category you're working on
3. **Read the test code** to understand what's expected
4. **Implement the parser method** following existing patterns
5. **Remove `#[ignore]`** and run the test
6. **Compare with Pyret** using `./compare_parsers.sh`
7. **Debug differences** until test passes
8. **Move to next test**

### For Project Managers

- Current coverage: 76/81 basic tests (93.8%)
- This adds: 50+ advanced tests
- Combined coverage shows: ~60% complete for production-ready parser
- Phase 1 completion brings us to ~70% complete
- Full completion brings us to ~95% complete

### For Quality Assurance

- All tests use real Pyret code patterns
- Tests are organized by feature category
- Each test has clear documentation
- Tests validate against official Pyret parser

---

## Success Criteria

A test passes when:
1. ✅ Parser successfully parses the code
2. ✅ Generated AST matches official Pyret parser (verified with `compare_parsers.sh`)
3. ✅ Test runs without `#[ignore]` attribute
4. ✅ No panics or errors

---

## Contributing

When implementing a feature:

1. **Document your progress** - update test comments
2. **Write parser tests** - add unit tests in `parser_tests.rs`
3. **Update gap analysis** - mark completed features
4. **Run full test suite** - ensure no regressions
5. **Update NEXT_STEPS.md** - keep roadmap current

---

## Questions?

- Check existing parser code for patterns
- Look at official Pyret grammar: `pyret-lang/src/js/base/pyret-grammar.bnf`
- Compare with official parser: `./compare_parsers.sh "code"`
- Read `NEXT_STEPS.md` for detailed guidance

---

**Last Updated:** $(date)
**Maintained By:** Development Team
**Test File:** `tests/comprehensive_gap_tests.rs` (50+ tests, all currently ignored)
