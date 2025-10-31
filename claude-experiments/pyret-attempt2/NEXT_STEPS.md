# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-10-31
**Current Status:** Phase 3 - Expressions (50% complete)
**Tests Passing:** 35/35 parser tests, 46/46 total

---

## 🎯 Quick Start - What to Do Next

The parser successfully handles:
- ✅ All primitive expressions (numbers, strings, booleans, identifiers)
- ✅ Binary operators (15 operators, left-associative)
- ✅ Parenthesized expressions: `(1 + 2)`
- ✅ Function application: `f(x, y)`
- ✅ Chained calls: `f(x)(y)`
- ✅ Whitespace-sensitive parsing: `f(x)` vs `f (x)`
- ✅ **Array expressions:** `[1, 2, 3]` - COMPLETED! ✨
- ✅ **Dot access:** `obj.field`, `obj.field1.field2` - COMPLETED! ✨
- ✅ **Chained postfix operators:** `obj.foo().bar()` - COMPLETED! ✨

**Next priority:** Bracket access `arr[0]`, then objects, then tuples.

---

## 📋 Immediate Next Tasks (Priority Order)

### 1. Object Expressions (HIGHEST PRIORITY) ⭐⭐⭐
**Why:** Objects are fundamental in Pyret - used everywhere

**Syntax:**
```pyret
{
  field1: value1,
  field2: value2,
  method() -> return-type:
    body
  end
}
```

**AST Node:** `Expr::SObj { l, fields }`

**Implementation Guide:**
- **Location:** Add `parse_obj_expr()` in `src/parser.rs` Section 6
- **Token:** Starts with `TokenType::LBrace`
- **Grammar:** `LBRACE obj-fields RBRACE`
- **Fields:** Can be:
  - Data fields: `name: expr`
  - Method fields: `name(args) -> ann: body end`
- **Helper needed:** `parse_obj_field()` for each field type

**Test Cases to Add:**
```rust
// Empty object
parse_expr("{}")

// Simple fields
parse_expr("{ x: 1, y: 2 }")

// Nested objects
parse_expr("{ point: { x: 0, y: 0 } }")

// With methods
parse_expr("{ value: 5, double() -> Number: self.value * 2 end }")
```

**Estimated Time:** 2-3 hours

---

### 2. Array Expressions ⭐⭐⭐
**Why:** Arrays are fundamental data structures

**Syntax:** `[expr, expr, ...]`

**AST Node:** `Expr::SArray { l, values }`

**Implementation Guide:**
- **Location:** Add `parse_array_expr()` in `src/parser.rs` Section 6
- **Token:** Starts with `TokenType::LBrack`
- **Grammar:** `LBRACK (expr COMMA)* RBRACK`
- **Helper:** Use existing `parse_comma_list()` helper

**Test Cases:**
```rust
parse_expr("[]")           // Empty
parse_expr("[1, 2, 3]")    // Numbers
parse_expr("[x, y, z]")    // Identifiers
parse_expr("[[1, 2], [3, 4]]")  // Nested
```

**Estimated Time:** 1 hour

---

### 3. Dot Access ⭐⭐⭐
**Why:** Essential for object field access

**Syntax:** `obj.field`

**AST Node:** `Expr::SDot { l, obj, field }`

**Implementation Guide:**
- **Location:** Update `parse_binop_expr()` to check for `TokenType::Dot` as postfix
- **Token:** `TokenType::Dot`
- **Grammar:** `expr DOT NAME`
- **Chain support:** `obj.field1.field2`

**Important:** Dot has higher precedence than binary operators, parse it as postfix in the expression loop before checking binops.

**Test Cases:**
```rust
parse_expr("obj.field")
parse_expr("obj.field1.field2")
parse_expr("f(x).field")
parse_expr("obj.method()")  // Chaining with function calls
```

**Estimated Time:** 1-2 hours

---

### 4. Bracket Access ⭐⭐
**Why:** Array/dictionary indexing

**Syntax:** `obj[key]`

**AST Node:** `Expr::SBracket { l, obj, key }`

**Implementation Guide:**
- **Location:** Update `parse_binop_expr()` as postfix operator
- **Token:** `TokenType::LBrack` (in postfix position)
- **Grammar:** `expr LBRACK expr RBRACK`

**Note:** Must distinguish from array literal `[1, 2, 3]` vs bracket access `arr[0]`
- Array literal: `LBrack` as start of primary expression
- Bracket access: `LBrack` after an expression

**Test Cases:**
```rust
parse_expr("arr[0]")
parse_expr("dict[\"key\"]")
parse_expr("matrix[i][j]")
parse_expr("f(x)[0]")
```

**Estimated Time:** 1-2 hours

---

### 5. Tuple Expressions ⭐
**Why:** Less common but needed for completeness

**Syntax:** `{expr; expr; expr}`  (semicolon-separated, not comma!)

**AST Node:** `Expr::STuple { l, fields }`

**Implementation Guide:**
- **Location:** Add `parse_tuple_expr()` in Section 6
- **Token:** Starts with `TokenType::LBrace`
- **Grammar:** `LBRACE (expr SEMICOLON)+ RBRACE`
- **Challenge:** Distinguish from object `{x: 1}` - check first separator

**Test Cases:**
```rust
parse_expr("{1; 2; 3}")
parse_expr("{x; y; z}")
parse_expr("{{1; 2}; {3; 4}}")  // Nested
```

**Estimated Time:** 1-2 hours

---

## 🔧 Implementation Tips

### Working with the Parser

**File:** `src/parser.rs` (currently 708 lines)

**Structure:**
- Section 1: Core methods (peek, advance, expect, matches)
- Section 6: Expression parsing ← **You'll work here**
- Section 12: Helper methods (parse_comma_list, etc.)

**Adding a new expression type:**

1. **Add parsing method:**
```rust
impl Parser {
    /// your-expr: YOUR GRAMMAR HERE
    fn parse_your_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenType::YourStartToken)?;

        // Parse components
        let components = self.parse_comma_list(|p| p.parse_expr())?;

        let end = self.expect(TokenType::YourEndToken)?;

        Ok(Expr::SYourExpr {
            l: self.make_loc(&start, &end),
            components: components,
        })
    }
}
```

2. **Update `parse_prim_expr()` OR `parse_binop_expr()`:**
   - If it's a **primary expression** (starts with a token): Add case to `parse_prim_expr()`
   - If it's a **postfix operator** (comes after an expression): Add to `parse_binop_expr()` loops

3. **Update location extraction in `parse_binop_expr()`:**
   - Add your new expression type to the match statements at lines 301-322 and 313-320

4. **Add tests in `tests/parser_tests.rs`:**
```rust
#[test]
fn test_parse_your_expr() {
    let expr = parse_expr("your syntax here").expect("Failed to parse");

    match expr {
        Expr::SYourExpr { components, .. } => {
            assert_eq!(components.len(), expected_count);
        }
        _ => panic!("Expected SYourExpr, got {:?}", expr),
    }
}
```

### Using Helpers

**parse_comma_list:**
```rust
// Parse comma-separated items
let items = self.parse_comma_list(|p| p.parse_expr())?;
```

**parse_optional:**
```rust
// Parse optional component
let opt = self.parse_optional(|p| p.parse_ann())?;
```

**Location tracking:**
```rust
// From two tokens
let loc = self.make_loc(&start_token, &end_token);

// Current location
let loc = self.current_loc();
```

### Running Tests

```bash
# Run all tests
cargo test

# Run only parser tests
cargo test --test parser_tests

# Run specific test
cargo test test_parse_simple_function_call

# Run with debug output
DEBUG_TOKENS=1 cargo test test_name
```

---

## 📚 Reference Materials

### Essential Files

1. **AST Definitions:** `src/ast.rs`
   - All expression types defined here
   - Line 292-808: Expression enum
   - Shows exact JSON serialization format

2. **Current Parser:** `src/parser.rs`
   - Section 6 (lines 188-520): Expression parsing
   - Study existing patterns for consistency

3. **Tokenizer:** `src/tokenizer.rs`
   - Lines 125-185: Token types
   - Understanding tokens is crucial

4. **Pyret Grammar:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
   - Official grammar specification
   - Maps directly to parser functions

5. **Test Examples:** `tests/parser_tests.rs`
   - Current 24 tests show testing patterns
   - Good examples of assertion structure

### Documentation

- `PARSER_PLAN.md` - Overall project plan and phases
- `PHASE1_COMPLETE.md` - Foundation work summary
- `PHASE2_COMPLETE.md` - Primitives and operators summary
- `PHASE3_PARENS_AND_APPS_COMPLETE.md` - Latest work completed
- `OPERATOR_PRECEDENCE.md` - Important: Pyret has NO precedence!

---

## 🚨 Common Pitfalls

### 1. Precedence Confusion
**Problem:** Trying to add operator precedence
**Solution:** Pyret has NO precedence! All operators are equal and left-associative.

### 2. Whitespace Sensitivity
**Problem:** Not considering token types like `ParenSpace` vs `ParenNoSpace`
**Solution:** The tokenizer already handles this - trust the token types.

### 3. Location Tracking
**Problem:** Forgetting to add new expression types to location extraction matches
**Solution:** Search for all `match &left {` and `match &right {` patterns and add your type.

### 4. Postfix vs Prefix
**Problem:** Parsing postfix operators (like `.field`) as primary expressions
**Solution:** Postfix operators go in `parse_binop_expr()` loops, not `parse_prim_expr()`.

### 5. Testing
**Problem:** Not testing edge cases
**Solution:** Always test: empty cases, single items, nested, mixed with other expressions.

---

## 📈 Progress Tracking

### Phase 3 Checklist

**Completed (35%):**
- ✅ Parenthesized expressions
- ✅ Function application (direct and juxtaposition)
- ✅ Chained function calls

**In Progress (Next 5 tasks):**
- ⬜ Object expressions
- ⬜ Array expressions
- ⬜ Dot access
- ⬜ Bracket access
- ⬜ Tuple expressions

**Remaining Phase 3:**
- ⬜ Extended dot access (`obj!field`)
- ⬜ Update expressions (`obj.{field: new-value}`)
- ⬜ Extend expressions (`obj.{extra-field: value}`)
- ⬜ If expressions (may move to Phase 4)
- ⬜ Block expressions
- ⬜ Let expressions
- ⬜ Var expressions
- ⬜ Assign expressions

### Estimated Timeline

| Task | Time | Total % |
|------|------|---------|
| Object expressions | 2-3h | +15% |
| Array expressions | 1h | +10% |
| Dot access | 1-2h | +10% |
| Bracket access | 1-2h | +10% |
| Tuple expressions | 1-2h | +5% |

After these 5 tasks: **~85% of Phase 3 complete**

---

## 💡 Tips for Success

1. **Start Small:** Begin with the simplest expression (arrays) to get comfortable
2. **Follow Patterns:** Copy structure from existing parse methods
3. **Test Early:** Write tests as you implement, not after
4. **Read the Grammar:** When stuck, check the BNF file for exact syntax
5. **Check AST:** Verify your AST nodes match `src/ast.rs` exactly
6. **Use Debug Output:** `DEBUG_TOKENS=1` helps understand tokenization
7. **Ask for Help:** Look at similar existing parsers in the file

---

## 🎓 Learning Resources

### Understanding the Codebase

**Start here:**
1. Read `PHASE2_COMPLETE.md` to understand what's implemented
2. Read `PHASE3_PARENS_AND_APPS_COMPLETE.md` to understand latest changes
3. Look at `parse_prim_expr()` (line 346) to see primary expression dispatch
4. Look at `parse_app_expr()` (line 481) to see a complete parsing method
5. Look at tests (line 203+) to see expected behavior

**Key concepts:**
- **Recursive descent:** Each grammar rule = one function
- **Token lookahead:** Use `peek()` to check, `advance()` to consume
- **Location tracking:** Every node needs precise source location
- **Left-associativity:** Always build `(left op right)` then make it the new left

### Pyret Language Quirks

1. **No operator precedence** - Everything is left-to-right
2. **Whitespace matters** - `f(x)` ≠ `f (x)`
3. **Semicolons for tuples** - `{1; 2; 3}` not `{1, 2, 3}`
4. **Colons for objects** - `{x: 1}` not `{x = 1}`
5. **End keyword** - Most blocks end with `end` not `}`

---

## 🆘 Getting Unstuck

### Problem: "I don't know where to add my function"
→ Look at existing functions in Section 6. Add yours in alphabetical order.

### Problem: "Tests are failing with unexpected token"
→ Add `DEBUG_TOKENS=1` before `cargo test` to see tokenization.

### Problem: "Location tracking is wrong"
→ Use `self.make_loc(&start, &end)` where start/end are Token objects.

### Problem: "I need to distinguish two cases that start with the same token"
→ Use lookahead with `peek()` to check the second token before committing.

### Problem: "My AST doesn't serialize correctly"
→ Check that your struct matches `src/ast.rs` exactly, including field names.

---

## ✅ Before You Commit

**Checklist:**
- [ ] All existing tests still pass (`cargo test`)
- [ ] New tests added for your feature
- [ ] Code follows existing patterns and style
- [ ] Comments explain non-obvious logic
- [ ] No compiler warnings
- [ ] Location tracking is accurate
- [ ] AST matches reference implementation

**Commit message format:**
```
feat(parser): implement <feature-name>

- Add parse_<feature>() method
- Support <syntax>
- Add <number> tests
- All tests passing

Closes #<issue> (if applicable)
```

---

## 🎉 You're Ready!

Start with **object expressions** (highest priority) or **array expressions** (easiest).

The codebase is well-structured, tests are comprehensive, and you have all the patterns you need.

**Good luck! 🚀**

---

**Questions?** Check existing code first - the answer is probably there!
