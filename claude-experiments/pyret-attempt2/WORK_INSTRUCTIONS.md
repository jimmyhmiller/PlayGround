# Work Instructions - Gap Analysis Implementation

**Quick Start Guide for Implementing Missing Features**

---

## 🎯 Your Mission

You have a Pyret parser that's 93.8% complete for basic features, but missing many advanced features found in real Pyret programs. Your job is to implement these missing features by:

1. Enabling tests one at a time
2. Implementing the required parser logic
3. Validating against the official Pyret parser
4. Moving to the next feature

---

## 📋 Step-by-Step Instructions

### Step 1: See What Needs Work

```bash
# List all ignored tests (features waiting to be implemented)
cargo test --test comprehensive_gap_tests -- --ignored --list
```

You'll see 50+ tests, organized by category:
- `test_block_*` - Advanced block structures
- `test_data_*` - Data definitions
- `test_cases_*` - Pattern matching
- And more...

### Step 2: Pick Your First Feature

**Recommended order:**
1. Start with **Data Definitions** (`test_simple_data_definition`)
2. Then **Cases Expressions** (depends on data)
3. Then **Advanced Blocks** (common pattern)

**Why this order?** Data definitions are foundational, cases depend on them, and blocks are used everywhere.

### Step 3: Read the Test

Open `tests/comprehensive_gap_tests.rs` and find your test:

```rust
#[test]
#[ignore] // TODO: Data definitions not implemented
fn test_simple_data_definition() {
    // Real pattern: basic algebraic data type
    assert_matches_pyret(r#"
data Color:
  | red
  | green
  | blue
end
"#);
}
```

This tells you:
- What syntax to support
- Why it matters
- What the code should do

### Step 4: Research the Feature

Read the detailed guide:
```bash
# For data definitions
cat COMPREHENSIVE_GAP_ANALYSIS.md | grep -A 50 "Data Definitions"
```

This gives you:
- Grammar specification
- AST node structure
- Implementation notes
- Time estimate

### Step 5: Implement the Parser

**Example: Adding data definitions**

1. **Add AST node** (`src/ast.rs`):
```rust
pub struct SData {
    pub l: Loc,
    pub name: String,
    pub params: Vec<String>,
    pub variants: Vec<Variant>,
    pub shared: Vec<Member>,
}
```

2. **Add parser method** (`src/parser.rs`):
```rust
fn parse_data_expr(&mut self) -> Result<Expr> {
    self.expect(TokenType::Data)?;
    let name = self.expect_name()?;
    self.expect(TokenType::Colon)?;

    // Parse variants...

    self.expect(TokenType::End)?;
    Ok(Expr::SData { /* ... */ })
}
```

3. **Hook it up** (in `parse_prim_expr`):
```rust
match self.current_token() {
    TokenType::Data => self.parse_data_expr(),
    // ... other cases
}
```

4. **Add JSON serialization** (`src/bin/to_pyret_json.rs`):
```rust
Expr::SData { name, variants, .. } => json!({
    "type": "s-data",
    "name": name,
    "variants": variants,
    // ...
})
```

### Step 6: Enable the Test

In `tests/comprehensive_gap_tests.rs`, remove the `#[ignore]` attribute:

```rust
#[test]
// #[ignore] - REMOVED!
fn test_simple_data_definition() {
    assert_matches_pyret(r#"
data Color:
  | red
  | green
  | blue
end
"#);
}
```

### Step 7: Run the Test

```bash
cargo test --test comprehensive_gap_tests test_simple_data_definition
```

**Possible outcomes:**
- ✅ Test passes → Great! Move to next test
- ❌ Parser error → Fix parsing logic
- ❌ AST mismatch → Check JSON output

### Step 8: Validate Against Pyret

```bash
./compare_parsers.sh "data Color: | red | green | blue end"
```

This compares your parser output with the official Pyret parser.

**Expected output:**
```
✅ IDENTICAL - Parsers produce the same AST!
```

**If different:**
```
❌ DIFFERENT - Found differences:

Pyret AST:
{ "type": "s-data", "name": "Color", ... }

Rust AST:
{ "type": "s-data", "name": "Color", ... }
```

Fix your JSON serialization until they match.

### Step 9: Move to Next Test

Repeat steps 3-8 for the next test in the category.

---

## 🎓 Learning by Example

### Example 1: How Lambda Was Implemented

Look at `src/parser.rs`, search for `parse_lam_expr`:

```rust
fn parse_lam_expr(&mut self) -> Result<Expr> {
    let start_loc = self.current_location();
    self.expect(TokenType::Lam)?;

    // Parse parameters
    let args = self.parse_fun_args()?;

    // Parse return type (optional)
    let return_type = self.parse_optional_return_type()?;

    // Parse body
    self.expect(TokenType::Colon)?;
    let body = self.parse_expr()?;
    self.expect(TokenType::End)?;

    Ok(Expr::SLam { /* ... */ })
}
```

**Key patterns:**
- Use `expect()` for required tokens
- Use `parse_optional_*()` for optional parts
- Track location for error messages
- Return proper AST node

### Example 2: How Objects Were Implemented

Look at `src/parser.rs`, search for `parse_obj_expr`:

```rust
fn parse_obj_expr(&mut self) -> Result<Expr> {
    self.expect(TokenType::LBrace)?;

    let mut fields = Vec::new();

    // Parse fields
    while !self.check(TokenType::RBrace) {
        let field = self.parse_member()?;
        fields.push(field);

        if self.check(TokenType::Comma) {
            self.advance();
        }
    }

    self.expect(TokenType::RBrace)?;

    Ok(Expr::SObj { fields })
}
```

**Key patterns:**
- Use loops for repeated elements
- Check for delimiters (commas)
- Handle trailing commas
- Collect into vectors

---

## 🐛 Debugging Tips

### Problem: Parser Panics

```
thread 'test_simple_data_definition' panicked at 'Unexpected token'
```

**Solution:**
1. Add `DEBUG_TOKENS=1` to see token stream
2. Check if tokenizer recognizes your keywords
3. Verify token types in match statements

```bash
DEBUG_TOKENS=1 cargo test --test comprehensive_gap_tests test_simple_data_definition
```

### Problem: AST Doesn't Match

```
❌ DIFFERENT - Found differences
```

**Solution:**
1. Run `./compare_parsers.sh "your code"`
2. Compare field names (must match exactly)
3. Check field order (must match exactly)
4. Verify location info is present

### Problem: Test Hangs

**Solution:**
1. Check for infinite loops in parser
2. Verify `advance()` is called in loops
3. Add debug prints to see where it's stuck

---

## 📊 Tracking Progress

### See What's Done

```bash
# Count enabled tests
cargo test --test comprehensive_gap_tests 2>&1 | grep "test result"
```

### See What's Left

```bash
# Count ignored tests
cargo test --test comprehensive_gap_tests -- --ignored 2>&1 | grep "test result"
```

### Visual Progress

Update `COMPREHENSIVE_GAP_ANALYSIS.md` as you complete features:

```markdown
## Feature Checklist

- [x] Data definitions (6 tests)
  - [x] Simple enumerations ← YOU COMPLETED THIS!
  - [ ] Variants with fields
  - [ ] ...
```

---

## 🎯 Milestones

### Milestone 1: First Feature Complete (5-8 hours)
**Goal:** Implement simple data definitions
- ✅ Parse `data Color: | red | green | blue end`
- ✅ Generate correct AST
- ✅ Pass comparison test
- 🎓 **Learn:** Basic data definition parsing

### Milestone 2: Data Types Complete (10-15 hours)
**Goal:** Implement all data definition variants
- ✅ Simple enumerations
- ✅ Variants with fields
- ✅ Mutable fields
- ✅ Sharing clauses
- 🎓 **Learn:** Complex AST construction

### Milestone 3: Pattern Matching Complete (15-20 hours)
**Goal:** Implement cases expressions
- ✅ Basic pattern matching
- ✅ Else branches
- ✅ Nested cases
- 🎓 **Learn:** Pattern parsing

### Milestone 4: Core Complete (25-35 hours)
**Goal:** Complete Phase 1 features
- ✅ Data definitions
- ✅ Cases expressions
- ✅ Advanced blocks
- 🎓 **Achievement:** Parser reaches 70% completion

---

## 🚀 Quick Commands Reference

```bash
# See all missing features
cargo test --test comprehensive_gap_tests -- --ignored --list

# Work on specific category
cargo test --test comprehensive_gap_tests test_data -- --ignored

# Enable a test (edit file, remove #[ignore])
# Then run:
cargo test --test comprehensive_gap_tests test_simple_data_definition

# Validate against Pyret
./compare_parsers.sh "data Box: | box(v) end"

# Debug tokens
DEBUG_TOKENS=1 cargo test test_name

# Run all tests
cargo test
```

---

## 📚 Resources

### Start Here
1. `IMPLEMENTATION_ROADMAP.md` - What to build and when
2. `COMPREHENSIVE_GAP_ANALYSIS.md` - Detailed feature descriptions
3. `tests/comprehensive_gap_tests.rs` - Test code

### Reference
- `src/parser.rs` - Parser implementation (look for similar features)
- `src/ast.rs` - AST definitions
- `src/bin/to_pyret_json.rs` - JSON serialization
- Pyret Grammar: `/path/to/pyret-lang/src/js/base/pyret-grammar.bnf`

### Tools
- `./compare_parsers.sh` - Compare with official Pyret
- `cargo test` - Run tests
- `DEBUG_TOKENS=1` - Debug tokenizer

---

## ❓ FAQ

**Q: Where do I start?**
A: Read this file, then start with `test_simple_data_definition`

**Q: How long will this take?**
A: First feature: 5-8 hours. Phase 1 complete: 13-18 hours. Everything: 44-62 hours.

**Q: What if I get stuck?**
A: Look at similar features (e.g., `parse_lam_expr` for inspiration), check the grammar, or examine the official Pyret parser.

**Q: How do I know it's working?**
A: Test passes AND `./compare_parsers.sh` shows "✅ IDENTICAL"

**Q: Can I skip ahead?**
A: Yes, but some features depend on others (cases needs data definitions)

**Q: What's the goal?**
A: Enable all 50+ tests, reach 90%+ parser completion, have production-ready Pyret parser

---

## 🎉 When You're Done

After completing all features:

1. ✅ All 50+ tests enabled (no `#[ignore]`)
2. ✅ All tests passing
3. ✅ All comparisons identical to Pyret
4. ✅ 125+ total passing tests
5. ✅ 90%+ parser completion
6. ✅ Production-ready Pyret parser

**Congratulations! You've built a comprehensive Pyret parser! 🎊**

---

**Last Updated:** 2025-01-31
**Maintained By:** Development Team
**Questions?** Check `COMPREHENSIVE_GAP_ANALYSIS.md` or `IMPLEMENTATION_ROADMAP.md`
