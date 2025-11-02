# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-11-02
**Current Status:** 81/128 tests passing (63.3%)
**Core Language:** ~90% complete ✅
**Advanced Features:** ~35% complete ⚠️

---

## 🎉 What's Already Working!

Recent discoveries show the parser is more complete than documented:

### Fully Implemented ✅
- ✅ All core expressions (primitives, operators, calls, access)
- ✅ **Function definitions** `fun f(x): x + 1 end`
- ✅ **When expressions** `when cond: body end`
- ✅ **Assignment expressions** `x := 5`
- ✅ **Data declarations** `data Box: | box(ref v) end`
- ✅ **Cases expressions** `cases(Either) e: | left(v) => v end`
- ✅ **Import statements** `import mod as M`
- ✅ Lambda expressions `lam(x): x + 1 end`
- ✅ Object expressions with methods
- ✅ Let/var bindings, tuples, blocks, if, for
- ✅ Provide statements `provide *`

**All 81 passing tests produce byte-for-byte identical ASTs!** ✨

---

## 📋 REMAINING WORK - 47 Features to Implement

Based on the ignored tests in `tests/comparison_tests.rs`, here's what needs implementation:

## Priority 1: High-Value Quick Wins ⭐⭐⭐

### 1. Unary Operators (3 tests, ~2-3 hours)

**Tests:**
- `test_unary_not` - Logical negation: `not true`
- `test_unary_minus` - Numeric negation: `-(5 + 3)`
- `test_custom_binary_operator` - Method calls as operators: `x._plus(y)`

**Why High Priority:**
- Very common in real Pyret code
- Quick to implement (similar to binary operators)
- Enables more realistic code patterns

**Implementation Guide:**
```rust
// In src/parser.rs, update parse_prim_expr()
fn parse_unary_expr(&mut self) -> ParseResult<Expr> {
    match self.peek() {
        TokenType::Not => {
            let op = self.advance();
            let expr = self.parse_unary_expr()?;  // Right-recursive for chaining
            Ok(Expr::SUnaryOp {
                l: self.make_loc(&op, &self.prev()),
                op: "not",
                expr: Box::new(expr),
            })
        }
        TokenType::Minus if !self.is_after_expr() => {
            let op = self.advance();
            let expr = self.parse_unary_expr()?;
            Ok(Expr::SUnaryOp {
                l: self.make_loc(&op, &self.prev()),
                op: "-",
                expr: Box::new(expr),
            })
        }
        _ => self.parse_postfix_expr()
    }
}
```

**Files to Update:**
- `src/parser.rs` - Add unary operator parsing
- `src/ast.rs` - May need `SUnaryOp` variant if not exists
- `src/bin/to_pyret_json.rs` - Add JSON serialization
- `tests/comparison_tests.rs` - Remove `#[ignore]` from 3 tests

---

### 2. Type Annotations on Bindings (3 tests, ~2-3 hours)

**Tests:**
- `test_block_with_typed_bindings` - `x :: Number = 42`
- `test_union_type_annotation` - `x :: (Number | String) = 42`
- `test_any_type` - `x :: Any = 42`

**Why High Priority:**
- Type safety improvements
- Already have type annotation parsing infrastructure
- Just needs integration with let/var bindings

**Implementation Guide:**
```rust
// In parse_implicit_let_expr(), add optional type annotation
fn parse_implicit_let_expr(&mut self) -> ParseResult<Stmt> {
    let name = self.expect_name()?;

    // Check for type annotation
    let ann = if self.check(TokenType::ColonColon) {
        self.advance();
        self.parse_ann()?
    } else {
        Ann::ABlank
    };

    self.expect(TokenType::Equals)?;
    let value = self.parse_expr()?;

    Ok(Stmt::SLet {
        bind: Bind { name, ann },
        value: Box::new(value),
    })
}
```

**Files to Update:**
- `src/parser.rs` - Update let/var parsing to accept `::` annotations
- Type annotation parsing already exists (`parse_ann()`)
- `tests/comparison_tests.rs` - Remove `#[ignore]` from 3 tests

---

### 3. Advanced Block Features (4 tests, ~3-4 hours)

**Tests:**
- `test_block_with_multiple_let_bindings` - Multiple lets in sequence
- `test_block_with_var_binding` - `var` bindings in blocks
- `test_nested_blocks_with_shadowing` - Scoping rules
- `test_block_with_typed_bindings` - Combines with #2 above

**Why High Priority:**
- Blocks with statements are very common
- Most infrastructure already exists
- Enables realistic code patterns

**Current State:**
- ✅ `block: 5 end` works (single expression)
- ❌ `block: x = 5 y = 10 x + y end` needs implementation

**Implementation Guide:**
```rust
// Update parse_block_expr() to parse statements before final expression
fn parse_block_expr(&mut self) -> ParseResult<Expr> {
    self.expect(TokenType::Block)?;

    let mut stmts = Vec::new();

    // Parse statements until we hit END or final expression
    while !self.check(TokenType::End) {
        // Try to parse as statement (let, var, etc.)
        if self.is_statement_start() {
            stmts.push(self.parse_stmt()?);
        } else {
            // Last expression
            let expr = self.parse_expr()?;
            stmts.push(Stmt::SExpr(expr));
            break;
        }
    }

    self.expect(TokenType::End)?;

    Ok(Expr::SUserBlock {
        body: Box::new(Expr::SBlock { stmts }),
    })
}
```

**Files to Update:**
- `src/parser.rs` - Update `parse_block_expr()` for multi-statement support
- Statement parsing infrastructure already exists
- `tests/comparison_tests.rs` - Remove `#[ignore]` from 4 tests

---

### 4. Where Clauses (4 tests, ~3-4 hours)

**Tests:**
- `test_function_with_multiple_where_clauses`
- `test_recursive_function_with_cases`
- `test_function_returning_function`
- `test_contract_on_function`

**Why High Priority:**
- Testing infrastructure is essential for Pyret
- Grammar already defined
- Enables test-driven development in Pyret

**Example:**
```pyret
fun factorial(n):
  if n == 0:
    1
  else:
    n * factorial(n - 1)
  end
where:
  factorial(0) is 1
  factorial(5) is 120
end
```

**Implementation Guide:**
```rust
// In parse_fun_expr(), add optional where clause after body
fn parse_fun_expr(&mut self) -> ParseResult<Stmt> {
    // ... parse name, params, body ...

    let check = if self.check(TokenType::Where) {
        self.advance();
        self.expect(TokenType::Colon)?;
        Some(self.parse_check_block()?)
    } else {
        None
    };

    self.expect(TokenType::End)?;

    Ok(Stmt::SFun { /* ... */ check })
}

fn parse_check_block(&mut self) -> ParseResult<Expr> {
    // Parse check test statements until END
    let mut tests = Vec::new();
    while !self.check(TokenType::End) {
        let test = self.parse_check_test()?;
        tests.push(test);
    }
    Ok(Expr::SCheckBlock { tests })
}
```

**Files to Update:**
- `src/parser.rs` - Add where clause parsing to functions
- `src/ast.rs` - May need check block structures
- `src/bin/to_pyret_json.rs` - Add JSON serialization
- `tests/comparison_tests.rs` - Remove `#[ignore]` from 4 tests

---

## Priority 2: Medium-Value Features ⭐⭐

### 5. String Interpolation (2 tests, ~4-5 hours)

**Tests:**
- `test_string_with_interpolation` - `` `Hello, $(name)!` ``
- `test_string_with_complex_expression` - `` `Answer: $(x + 32)` ``

**Implementation:**
- Requires tokenizer updates to handle backtick strings
- Parse embedded expressions inside `$(...)
`
- Create interpolated string AST nodes

---

### 6. Advanced For Expressions (4 tests, ~4-6 hours)

**Tests:**
- `test_for_with_cartesian_product` - Multiple generators
- `test_for_fold_with_tuple_accumulator` - Fold variant
- `test_for_filter` - Filter variant
- `test_nested_for_expressions` - Nested for loops

**Current State:**
- ✅ `for map(x from lst): body end` works
- ❌ `for fold(acc from init, x from lst): body end` needs implementation
- ❌ `for filter(x from lst): predicate end` needs implementation

---

### 7. Advanced Data Features (6 tests, ~6-8 hours)

**Tests:**
- `test_simple_data_definition` - Multiple variants
- `test_data_with_fields` - Typed fields
- `test_data_with_multiple_variants` - Sum types
- `test_data_with_shared_methods` - Sharing clauses
- `test_data_with_mutable_fields` - Complex ref patterns
- `test_generic_data_definition` - Generic types `<T>`

**Current State:**
- ✅ Basic data declarations work
- ❌ Sharing clauses need implementation
- ❌ Generic type parameters need implementation

---

### 8. Advanced Import/Export (4 tests, ~3-4 hours)

**Tests:**
- `test_import_specific_names` - `import lists as L`
- `test_import_from_file` - `import file("util.arr") as U`
- `test_provide_with_types` - `provide-types *`
- `test_provide_specific_names` - `provide { add, multiply } end`

---

## Priority 3: Advanced Features ⭐

### 9. Advanced Cases Patterns (4 tests, ~4-5 hours)

**Tests:**
- `test_cases_with_wildcard` - Wildcard patterns
- `test_cases_with_else` - Default else branch
- `test_nested_cases` - Nested pattern matching
- `test_cases_in_function_body` - Complex patterns

---

### 10. Object Extension (3 tests, ~3-4 hours)

**Tests:**
- `test_object_extension` - `point.{ z: 0 }`
- `test_object_with_computed_field` - `{ [key]: 42 }`
- `test_object_update_syntax` - `point.{ x: 10 }`

---

### 11. Other Advanced Features (~20 tests, 20+ hours)

- **Table expressions** (2 tests) - SQL-like tables
- **Check blocks** (2 tests) - Standalone test blocks
- **Rest parameters** (1 test) - `fun f(first, rest ...)`
- **Generic functions** (1 test) - `fun identity<T>(x :: T)`
- **List comprehensions with guards** (1 test)
- **Spy expressions** (1 test) - Debugging
- **Complex real-world patterns** (2 tests) - Integration tests

---

## 🎯 Recommended Path Forward

### Session 1: Quick Wins (5-8 hours)
1. ✅ Unary operators (3 tests)
2. ✅ Type annotations on bindings (3 tests)
3. ✅ Advanced block features (4 tests)

**Result: 91/128 tests passing (71%)**

### Session 2: Core Features (8-12 hours)
1. ✅ Where clauses (4 tests)
2. ✅ String interpolation (2 tests)
3. ✅ Advanced for variants (4 tests)

**Result: 101/128 tests passing (79%)**

### Session 3: Advanced Features (10-15 hours)
1. ✅ Advanced data features (6 tests)
2. ✅ Advanced import/export (4 tests)
3. ✅ Advanced cases patterns (4 tests)

**Result: 115/128 tests passing (90%)**

### Session 4: Polish (variable, ~10 hours)
1. ✅ Object extension (3 tests)
2. ✅ Remaining advanced features (10 tests)

**Result: 128/128 tests passing (100%)** 🎉

---

## 📝 Implementation Checklist

For each feature:

1. **Read the test** in `tests/comparison_tests.rs` (lines 700-1364)
2. **Run official parser** to see expected AST:
   ```bash
   ./compare_parsers.sh "your test code"
   ```
3. **Add parser method** in `src/parser.rs`
4. **Add JSON serialization** in `src/bin/to_pyret_json.rs`
5. **Update location extraction** if new Expr/Stmt type
6. **Remove `#[ignore]`** from test
7. **Run test:**
   ```bash
   cargo test --test comparison_tests test_name
   ```
8. **Validate:**
   ```bash
   ./compare_parsers.sh "your test code"
   ```

---

## 🚀 Ready to Start!

**Easiest first feature: Unary operators**
- Only 3 tests
- Similar to existing binary operator code
- ~2-3 hours of work
- Enables more realistic code

**Start here:**
1. Read `tests/comparison_tests.rs:1212-1227` (unary operator tests)
2. Run `./compare_parsers.sh "not true"` to see expected AST
3. Update `src/parser.rs` with unary operator parsing
4. Test and validate!

---

**Last Updated:** 2025-11-02
**Next Priority:** Unary operators (3 tests, ~2-3 hours)
**Quick Wins:** Features 1-4 = 14 tests in ~10-14 hours
