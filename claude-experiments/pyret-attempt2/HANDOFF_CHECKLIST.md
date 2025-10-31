# Handoff Checklist - Pyret Parser Project

**Date:** 2025-10-31
**Status:** Ready for next developer
**Phase:** 3 - Expressions (35% complete)

---

## ✅ Completed Work Verification

### Implementation
- [x] Parenthesized expressions implemented and tested
- [x] Function application implemented and tested
- [x] Chained function calls working
- [x] Juxtaposition application (whitespace-sensitive) working
- [x] Tokenizer bug fixed (Name tokens set paren_is_for_exp = false)
- [x] All location tracking accurate
- [x] AST nodes match reference implementation

### Testing
- [x] 24 parser tests passing
- [x] 35 total tests passing
- [x] No compiler warnings
- [x] Test coverage for all implemented features
- [x] Edge cases covered (empty args, nested parens, chaining, etc.)

### Documentation
- [x] **README.md** - Project overview created
- [x] **NEXT_STEPS.md** - Comprehensive guide for next developer created
- [x] **PHASE3_PARENS_AND_APPS_COMPLETE.md** - Work completion summary created
- [x] **PARSER_PLAN.md** - Updated with current status
- [x] **CLAUDE.md** - Updated with project links and quick reference
- [x] Code comments added for complex logic
- [x] All new functions documented with doc comments

### Code Quality
- [x] Code follows existing patterns and style
- [x] No breaking changes to existing functionality
- [x] Clean git status (or documented uncommitted changes)
- [x] Build succeeds without errors: `cargo build`
- [x] All tests pass: `cargo test`

---

## 📚 Documentation Quick Reference

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **CLAUDE.md** | Quick reference | Every session start (in this directory) |
| **README.md** | Project overview | First time viewing project |
| **NEXT_STEPS.md** | Implementation guide | Starting next task |
| **HANDOFF_CHECKLIST.md** | This file | Quick verification and reference |
| **PHASE3_PARENS_AND_APPS_COMPLETE.md** | Latest work details | Understanding recent changes |
| **PHASE2_COMPLETE.md** | Primitives & operators | Understanding binop parsing |
| **PHASE1_COMPLETE.md** | Foundation | Understanding project setup |
| **PARSER_PLAN.md** | Full roadmap | Planning long-term work |
| **OPERATOR_PRECEDENCE.md** | Pyret quirks | Understanding why no precedence |

---

## 🎯 Next Developer - Start Here

### First Steps (5 minutes)
1. Read `CLAUDE.md` in this directory - Project quick reference
2. Run tests to verify everything works: `cargo test`
3. Open `NEXT_STEPS.md` and pick first task
4. Start coding!

### Understanding the Codebase (30 minutes)
1. Read `README.md` - Get project overview
2. Read `PHASE3_PARENS_AND_APPS_COMPLETE.md` - See what was just implemented
3. Look at `tests/parser_tests.rs` lines 203-512 - See test patterns
4. Look at `src/parser.rs` lines 462-520 - See parse_paren_expr and parse_app_expr
5. Look at `src/parser.rs` lines 199-344 - See parse_binop_expr flow

### Ready to Code (Start here)
1. Open `NEXT_STEPS.md`
2. Choose task (recommend: **Object expressions** or **Array expressions**)
3. Follow implementation guide in NEXT_STEPS.md
4. Write tests as you go
5. Run `cargo test` frequently

---

## 🔍 Project State

### What Works
```rust
// Primitives
parse_expr("42")           // ✅ Numbers
parse_expr("true")         // ✅ Booleans
parse_expr("\"hello\"")    // ✅ Strings
parse_expr("x")            // ✅ Identifiers

// Binary operators (15 total, left-associative)
parse_expr("1 + 2")        // ✅ Arithmetic
parse_expr("x < 10")       // ✅ Comparison
parse_expr("true and false") // ✅ Logical

// Parentheses
parse_expr("(1 + 2)")      // ✅ Grouping
parse_expr("((x))")        // ✅ Nested

// Function application
parse_expr("f(x)")         // ✅ Single arg
parse_expr("f(x, y, z)")   // ✅ Multiple args
parse_expr("f()")          // ✅ No args
parse_expr("f(x)(y)")      // ✅ Chaining
parse_expr("f(1 + 2, 3)")  // ✅ Expr args

// Whitespace sensitivity
parse_expr("f(x)")         // ✅ Direct call
parse_expr("f (x)")        // ✅ Juxtaposition

// Mixed expressions
parse_expr("f(x) + g(y)")  // ✅ Operators + calls
parse_expr("1 + (2 * 3)")  // ✅ Grouping + operators
```

### What's Next (Priority Order)
```rust
// TODO: Object expressions
parse_expr("{ x: 1, y: 2 }")

// TODO: Array expressions
parse_expr("[1, 2, 3]")

// TODO: Dot access
parse_expr("obj.field")

// TODO: Bracket access
parse_expr("arr[0]")

// TODO: Tuple expressions
parse_expr("{1; 2; 3}")
```

### File Locations
```
Implementation:
  src/parser.rs:199-520    - Expression parsing (Section 6)
  src/ast.rs:292-808       - Expression AST nodes

Tests:
  tests/parser_tests.rs    - All parser tests

Next work area:
  src/parser.rs Section 6  - Add new parse_* methods here
```

---

## 💡 Key Information

### Pyret Quirks (IMPORTANT!)

1. **No Operator Precedence**
   - `2 + 3 * 4` = `(2 + 3) * 4` = `20` (NOT 14!)
   - All binary operators are equal
   - Strictly left-associative
   - Don't try to add precedence - it's intentional!

2. **Whitespace Matters**
   - `f(x)` = function call
   - `f (x)` = function applied to parenthesized expression
   - Tokenizer handles this via `ParenNoSpace` vs `ParenSpace`

3. **Semicolons for Tuples**
   - `{1; 2; 3}` = tuple (semicolons!)
   - `{x: 1, y: 2}` = object (commas!)

### Common Patterns

**Parse a primary expression:**
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

**Parse a postfix operator:**
```rust
// In parse_binop_expr(), after parsing primary:
while self.matches(&TokenType::Dot) {
    self.advance(); // consume dot
    let field = self.expect(TokenType::Name)?;

    left = Expr::SDot {
        l: self.make_loc(&start_of_left, &field),
        obj: Box::new(left),
        field: field.value,
    };
}
```

**Parse comma-separated list:**
```rust
let items = self.parse_comma_list(|p| p.parse_expr())?;
```

### Useful Commands

```bash
# Test everything
cargo test

# Test just parser
cargo test --test parser_tests

# Test specific function
cargo test test_parse_simple_function_call

# Debug tokens
DEBUG_TOKENS=1 cargo test test_name

# Watch mode (if you have cargo-watch)
cargo watch -x test

# Format code
cargo fmt

# Check without building
cargo check
```

---

## 🚨 Important Notes

### Don't Forget
1. **Update location extraction** - When adding new Expr types, add them to match statements in parse_binop_expr (lines 301-322)
2. **Test edge cases** - Empty cases, single items, nested, mixed with other exprs
3. **Follow patterns** - Look at existing code for consistency
4. **Debug tokens** - Use `DEBUG_TOKENS=1` when things don't parse as expected
5. **Read the grammar** - Check `/pyret-lang/src/js/base/pyret-grammar.bnf` when unsure

### Watch Out For
1. **Postfix vs Prefix** - Dot and bracket are postfix, objects and arrays are prefix
2. **Token disambiguation** - `{` starts both objects and tuples - check following tokens
3. **Location spans** - Must cover entire expression from first to last token
4. **Boxed expressions** - Most Expr fields are `Box<Expr>` or `Vec<Box<Expr>>`
5. **Whitespace** - Already handled by tokenizer, trust the token types

---

## 📞 Getting Help

### If Tests Fail
1. Check error message - usually very clear
2. Add `DEBUG_TOKENS=1` to see what's being tokenized
3. Look at similar working tests for patterns
4. Check that your AST matches `src/ast.rs` exactly

### If Confused About Grammar
1. Check `NEXT_STEPS.md` - has implementation guides
2. Look at `/pyret-lang/src/js/base/pyret-grammar.bnf`
3. Look at existing similar parser methods
4. Check `PHASE3_PARENS_AND_APPS_COMPLETE.md` for examples

### If Stuck on Design
1. Look at how existing features are implemented
2. Check the grammar BNF for exact syntax
3. Look at reference implementation if needed
4. Start simple - get basic case working first

---

## ✅ Verification Before Committing

Run through this checklist:

```bash
# 1. All tests pass
cargo test

# 2. No warnings
cargo build

# 3. Code is formatted
cargo fmt

# 4. New tests added
# Check tests/parser_tests.rs has tests for your feature

# 5. Documentation updated
# Update NEXT_STEPS.md to mark task complete
# Add brief note to README.md if major feature
```

---

## 🎉 Ready to Go!

Everything is set up for the next developer:
- ✅ Clean, working codebase
- ✅ Comprehensive documentation
- ✅ Clear next steps with examples
- ✅ All tests passing
- ✅ Good patterns to follow

**Start with NEXT_STEPS.md and pick the first task!**

Good luck! 🚀

---

**Handoff Date:** 2025-10-31
**Last Test Run:** All passing (24 parser tests, 35 total)
**Next Milestone:** Complete Phase 3 (85% after next 5 tasks)
