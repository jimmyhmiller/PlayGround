# Remaining Parser Features

This document tracks the unimplemented features based on bulk test analysis and comparison tests.

## Test Summary

- **Total comparison tests**: ~150+
- **Passing tests**: ~135 (including letrec tests added this session)
- **Ignored tests**: 18
- **Test coverage**: ~90%

## Remaining Features (Prioritized by Impact)

### 1. Load Table Expressions (3 ignored tests, 3+ files affected)

**Syntax:**
```pyret
load-table: name, age
  source: "data.csv"
end

load-table: name, species, age
  source: url("https://example.com/data.csv")
  sanitize name: string-sanitizer
end
```

**Tests:**
- `test_load_table_simple`
- `test_load_table_with_url`
- `test_load_table_with_options`

**Implementation notes:**
- Part of table literal system
- Requires parsing `load-table:` keyword
- Handles `source:` and optional `sanitize` clauses

---

### 2. Rec Bindings (2 ignored tests, 1+ files affected)

**Syntax:**
```pyret
rec x = { foo: 1 }

rec random-matrix = {
  make: lam(): random-matrix end
}
```

**Tests:**
- `test_rec_simple`
- `test_rec_with_reference`

**Implementation notes:**
- Alternative to `letrec` for recursive object bindings
- Similar to `let` but allows self-reference
- Grammar: `REC binding EQUALS expr`

---

### 3. Tuple Destructuring (3 ignored tests, 2+ files affected)

**Syntax:**
```pyret
{a; b} = {1; 2}
{a; b; c; d; e} = {10; 214; 124; 62; 12}
```

**Tests:**
- `test_tuple_destructure_simple`
- `test_tuple_destructure_nested`  
- `test_tuple_destructure_in_let`

**Implementation notes:**
- Pattern matching for tuples in bindings
- Already have tuple expressions `{1; 2; 3}`
- Need tuple pattern support in `parse_bind()`

---

### 4. Reactor Expressions (2 ignored tests, 2 files affected)

**Syntax:**
```pyret
reactor:
  init: 0,
  on-tick: lam(state): state + 1 end,
  to-draw: lam(state): circle(state, "solid", "red") end
end
```

**Tests:**
- `test_reactor_simple`
- `test_reactor_complete`

**Implementation notes:**
- Animation/interactive program framework
- Object literal with specific field names
- Grammar: `REACTOR COLON reactor-fields END`

---

### 5. Examples Blocks (2 ignored tests, 2 files affected)

**Syntax:**
```pyret
examples:
  f(1) is 2
  f(2) is 3
end

fun f(x):
  x + 1
where:
  examples:
    f(1) is 2
  end
end
```

**Tests:**
- `test_examples_simple`
- `test_examples_with_function`

**Implementation notes:**
- Similar to `check` blocks but for examples
- Can appear in `where` clauses
- Grammar: `EXAMPLES COLON test-stmts END`

---

### 6. Include From with Type (2 ignored tests, 10+ files affected)

**Syntax:**
```pyret
include from M:
  foo,
  type Bar
end

include from M:
  *,
  type *
end
```

**Tests:**
- `test_include_from_with_type`
- `test_include_from_type_star`

**Implementation notes:**
- Extends existing `include from` to support `type` keyword
- Related to provide-with-type (already has tests, needs implementation)
- Grammar: `INCLUDE FROM module COLON include-items END`

---

### 7. Newtype Declarations (2 ignored tests, 1 file affected)

**Syntax:**
```pyret
newtype Foo as FooT
newtype Array as ArrayT
```

**Tests:**
- `test_newtype_simple`
- `test_newtype_with_code`

**Implementation notes:**
- Creates opaque type aliases
- Similar to `type` but creates distinct type
- Grammar: `NEWTYPE NAME AS NAME`

---

## Already Implemented (Have Passing Tests)

These features are fully working:

- ✅ **Letrec expressions** - Recursive bindings (implemented this session!)
- ✅ **Provide from** - Re-exporting from modules (working)

## Features with Tests But NOT IMPLEMENTED

These have comparison tests that may be passing simple cases, but bulk test failures show they're not fully working:

- ⚠️ **Ask expressions** - Pattern matching conditionals (NEEDS IMPLEMENTATION - 5+ files failing)
- ⚠️ **Provide with types** - Type exports in provide blocks (NEEDS IMPLEMENTATION - 10+ files failing)
- ⚠️ **Use context** - Context imports (NEEDS IMPLEMENTATION - 5+ files failing)
- ⚠️ **Triple-backtick doc strings** - Multi-line documentation (NEEDS IMPLEMENTATION - 6+ files failing)

---

## Implementation Priority Recommendation

1. **Tuple Destructuring** (3 tests, moderate complexity)
   - Extends existing tuple support
   - Useful for many patterns
   - Estimated: 2-3 hours

2. **Rec Bindings** (2 tests, low complexity)
   - Very similar to letrec
   - Quick win
   - Estimated: 1-2 hours

3. **Examples Blocks** (2 tests, low complexity)
   - Similar to check blocks
   - Estimated: 1-2 hours

4. **Newtype** (2 tests, low complexity)
   - Simple type declaration
   - Estimated: 1 hour

5. **Reactor** (2 tests, moderate complexity)
   - Specific object structure
   - Estimated: 2-3 hours

6. **Include From with Type** (2 tests, low-moderate complexity)
   - Extends existing include from
   - Estimated: 2 hours

7. **Load Table** (3 tests, high complexity)
   - Complex table DSL
   - Save for later
   - Estimated: 4-6 hours

---

## Files Still Failing (After Letrec Fix)

Based on bulk test analysis, approximately 150-170 files still fail, with the following patterns:

- **~10 files**: Missing provide-with-type implementation (tests exist, need implementation)
- **~10 files**: Missing ask expression implementation (tests exist, need implementation)
- **~6 files**: Triple-backtick doc strings (tokenizer work needed)
- **~5 files**: Use context statements (tests exist, need implementation)
- **~5 files**: Advanced type features (generics, newtype, etc.)
- **~3 files**: Load table expressions
- **~100+ files**: Various advanced features (shadows, special operators, etc.)

---

Last Updated: 2025-11-05
After implementing: letrec expressions
