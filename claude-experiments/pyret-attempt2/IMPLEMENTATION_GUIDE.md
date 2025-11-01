# Implementation Guide - Adding Missing Features

**Last Updated:** 2025-10-31
**For:** Next developer implementing missing Pyret parser features
**Status:** 59/81 tests passing (73% coverage)

---

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [How to Use This Guide](#how-to-use-this-guide)
3. [Understanding the Test Suite](#understanding-the-test-suite)
4. [Step-by-Step Implementation Process](#step-by-step-implementation-process)
5. [Feature-Specific Guides](#feature-specific-guides)
6. [Troubleshooting](#troubleshooting)
7. [Verification & Testing](#verification--testing)

---

## 🚀 Quick Start

### What You Need to Know

**Current State:**
- ✅ 59 comparison tests passing perfectly
- ⏸️ 22 comparison tests ignored (features not implemented)
- ❌ 0 tests failing (no regressions!)

**Your Goal:**
Implement the 22 missing features to get 100% test coverage.

**Time Required:**
- Phase 1 (Core): ~10-12 hours → 86% coverage
- Phase 2 (Advanced): ~9-12 hours → 91% coverage
- Phase 3 (Statements): ~11-17 hours → 100% coverage

### Essential Reading (In Order)

1. **Start Here:** `GAP_ANALYSIS_SUMMARY.md` - 5-minute overview
2. **Then Read:** `PARSER_GAPS.md` - Detailed feature breakdown with AST examples
3. **For Context:** `MISSING_FEATURES_EXAMPLES.md` - Real code examples that fail
4. **Implementation:** This document (IMPLEMENTATION_GUIDE.md)
5. **Reference:** `NEXT_STEPS.md` - Original implementation patterns

### Quick Commands

```bash
# See all missing features
cargo test --test comparison_tests -- --ignored

# Test a specific feature manually
./compare_parsers.sh "lam(x): x + 1 end"

# Implement a feature (example: lambdas)
# 1. Read the guide below
# 2. Implement in src/parser.rs
# 3. Remove #[ignore] from tests
# 4. Run tests:
cargo test --test comparison_tests test_pyret_match_simple_lambda

# See expected AST structure
./compare_parsers.sh "lam(): 5 end" 2>&1 | grep -A 30 "Pyret AST"
```

---

## 📚 How to Use This Guide

### For Each Feature You Want to Implement:

1. **Read the priority** in `PARSER_GAPS.md` (⭐⭐⭐⭐⭐ = highest)
2. **See real examples** in `MISSING_FEATURES_EXAMPLES.md`
3. **Follow the step-by-step guide** below
4. **Check the expected AST** using `./compare_parsers.sh`
5. **Implement the parser method** in `src/parser.rs`
6. **Update JSON serialization** in `src/bin/to_pyret_json.rs`
7. **Remove `#[ignore]`** from relevant tests
8. **Run tests** and verify

### Recommended Order

**Start with lambdas!** They're the highest priority and unlock most real Pyret code.

```
Priority 1 (Start Here):
  1. Lambda expressions (4 tests) - ⭐⭐⭐⭐⭐
  2. Tuple expressions (4 tests) - ⭐⭐⭐⭐
  3. Block expressions (2 tests) - ⭐⭐⭐⭐
  4. If expressions (1 test) - ⭐⭐⭐⭐

Priority 2 (After Phase 1):
  5. Method fields (1 test) - ⭐⭐⭐
  6. For expressions (2 tests) - ⭐⭐⭐
  7. Cases expressions (1 test) - ⭐⭐⭐

Priority 3 (Final Polish):
  8. Everything else (7 tests)
```

---

## 🧪 Understanding the Test Suite

### Test File Structure

**Location:** `tests/comparison_tests.rs`

**Line Numbers:**
- Lines 1-497: Existing passing tests (59 tests) ✅
- Lines 498-706: New tests for missing features (22 tests) ⏸️

### How Comparison Tests Work

Each test uses `assert_matches_pyret()` which:

1. **Parses with official Pyret parser** → Outputs JSON to `/tmp/pyret_output.json`
2. **Parses with our Rust parser** → Outputs JSON
3. **Compares the JSON** → Test passes if identical

**Example:**
```rust
#[test]
#[ignore] // Remove this when lambda parsing is implemented
fn test_pyret_match_simple_lambda() {
    // lam(): body end
    assert_matches_pyret("lam(): \"no-op\" end");
}
```

### Running Tests

```bash
# See all ignored tests (missing features)
cargo test --test comparison_tests -- --ignored --list

# Run a specific ignored test (will fail until implemented)
cargo test --test comparison_tests test_pyret_match_simple_lambda -- --ignored

# Run all comparison tests (59 should pass)
cargo test --test comparison_tests

# Run only passing tests
cargo test --test comparison_tests -- --skip ignored
```

### Test Categories

The test file is organized by feature:

```rust
// Lines 498-528: Lambda Expressions (4 tests)
test_pyret_match_simple_lambda
test_pyret_match_lambda_with_params
test_pyret_match_lambda_multiple_params
test_pyret_match_lambda_in_call

// Lines 530-560: Tuple Expressions (4 tests)
test_pyret_match_simple_tuple
test_pyret_match_tuple_with_exprs
test_pyret_match_nested_tuples
test_pyret_match_tuple_access

// Lines 562-578: Block Expressions (2 tests)
test_pyret_match_simple_block
test_pyret_match_block_multiple_stmts

// Lines 580-596: For Expressions (2 tests)
test_pyret_match_for_map
test_pyret_match_for_map2

// Lines 598-607: Method Fields (1 test)
test_pyret_match_object_with_method

// Lines 609-618: Cases Expressions (1 test)
test_pyret_match_simple_cases

// Lines 620-640: If/When/Assignment/etc. (8 tests)
test_pyret_match_simple_if
test_pyret_match_simple_when
test_pyret_match_simple_assign
test_pyret_match_simple_let
test_pyret_match_simple_data
test_pyret_match_simple_fun
test_pyret_match_simple_import
test_pyret_match_simple_provide
```

---

## 🔧 Step-by-Step Implementation Process

### Phase 0: Understand the Feature

Before writing any code:

1. **Read the feature docs:**
   ```bash
   # See what's missing and why it matters
   grep -A 30 "Lambda Expressions" PARSER_GAPS.md
   grep -A 20 "Lambda Expressions" MISSING_FEATURES_EXAMPLES.md
   ```

2. **See the expected AST:**
   ```bash
   # Run comparison script to see official Pyret's AST
   ./compare_parsers.sh "lam(): 5 end" 2>&1 | grep -A 50 "Pyret AST"
   ```

3. **Check the grammar:**
   ```bash
   # Look at official grammar file
   grep -A 10 "lam-expr" /Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf
   ```

4. **Find the AST definition:**
   ```bash
   # Find AST node in our codebase
   grep -n "pub enum Expr" src/ast.rs
   # Look for SLam, STuple, etc. around line 292-808
   ```

### Phase 1: Implement Parser Method

**Location:** `src/parser.rs` Section 6 (Expression Parsing)

**Template:**
```rust
impl Parser {
    /// Parse <feature-name> expression
    /// Grammar: <bnf-rule>
    /// Example: <example-code>
    fn parse_<feature>_expr(&mut self) -> ParseResult<Expr> {
        // 1. Save start location
        let start = self.expect(TokenType::<StartToken>)?;

        // 2. Parse components
        let params = self.parse_<params>()?;
        let body = self.parse_<body>()?;

        // 3. Expect end token
        let end = self.expect(TokenType::End)?;

        // 4. Build AST node
        Ok(Expr::<VariantName> {
            l: self.make_loc(&start, &end),
            params,
            body,
            // ... other fields
        })
    }
}
```

**Where to Add:**
- If **primary expression** (starts with unique token): Add case in `parse_prim_expr()` around line 346
- If **postfix operator** (comes after expression): Add case in `parse_binop_expr()` around line 260

### Phase 2: Update Expression Parsing

**For Primary Expressions:**
```rust
// In parse_prim_expr() around line 346
fn parse_prim_expr(&mut self) -> ParseResult<Expr> {
    match self.peek().typ {
        // ... existing cases ...
        TokenType::Lam => self.parse_lam_expr(),  // Add your case
        // ... rest of cases ...
    }
}
```

**For Postfix Operators:**
```rust
// In parse_binop_expr() around line 260
// Add to the postfix loop after checking for calls/dots/brackets
```

### Phase 3: Update Location Extraction

**Location:** `src/parser.rs` around lines 301-322 and 313-320

You need to add your new expression type to TWO match statements:

```rust
// First match (around line 301)
let left_loc = match &left {
    // ... existing cases ...
    Expr::<YourVariant> { l, .. } => l,  // Add this
    // ... rest of cases ...
};

// Second match (around line 313)
let right_loc = match &right {
    // ... existing cases ...
    Expr::<YourVariant> { l, .. } => l,  // Add this too
    // ... rest of cases ...
};
```

**Why?** The parser needs to know how to extract location info from your new expression type for binary operators.

### Phase 4: Update JSON Serialization

**Location:** `src/bin/to_pyret_json.rs`

Add a case for your expression type in the `expr_to_json()` function:

```rust
fn expr_to_json(expr: &Expr) -> serde_json::Value {
    match expr {
        // ... existing cases ...

        Expr::<YourVariant> { l, field1, field2, .. } => {
            json!({
                "type": "<pyret-type-name>",  // e.g., "s-lam"
                "field1": expr_to_json(field1),
                "field2": value_to_json(field2),
                // ... map all fields to JSON ...
            })
        }

        // ... rest of cases ...
    }
}
```

**Important:** The JSON structure must EXACTLY match the official Pyret parser's output. Use `./compare_parsers.sh` to verify!

### Phase 5: Add Unit Tests (Optional but Recommended)

**Location:** `tests/parser_tests.rs`

Add unit tests for edge cases:

```rust
#[test]
fn test_parse_<feature>_<case>() {
    let expr = parse_expr("<your-code>").expect("Failed to parse");

    match expr {
        Expr::<YourVariant> { field, .. } => {
            // Assert field values
            assert_eq!(field, expected_value);
        }
        _ => panic!("Expected <YourVariant>, got {:?}", expr),
    }
}
```

### Phase 6: Remove `#[ignore]` and Test

1. **Find your test** in `tests/comparison_tests.rs`
2. **Remove `#[ignore]` attribute**
3. **Run the test:**
   ```bash
   cargo test --test comparison_tests test_pyret_match_<your_test>
   ```

4. **If it fails:**
   - Compare ASTs: `./compare_parsers.sh "<your-code>"`
   - Check differences in JSON structure
   - Verify field names match exactly
   - Check types match ("s-lam" not "lam", etc.)

5. **Iterate** until test passes!

### Phase 7: Test Edge Cases

Don't just test the happy path! Test:
- Empty cases (e.g., `lam(): 5 end` with no params)
- Single items (e.g., `{1}` for tuples)
- Nested structures (e.g., `{1; {2; 3}}`)
- Complex expressions (e.g., `lam(x, y): x + y * 2 end`)
- Error cases (optional - make sure parser fails gracefully)

---

## 🎯 Feature-Specific Guides

### 1. Lambda Expressions (HIGHEST PRIORITY ⭐⭐⭐⭐⭐)

**Tests:** 4 tests (lines 502-528)
**Time:** 2-3 hours
**Difficulty:** Medium

#### What to Implement

```pyret
lam(): body end                    # No parameters
lam(x): x + 1 end                  # One parameter
lam(x, y): x + y end              # Multiple parameters
lam(x :: Number): x + 1 end       # With type annotation
```

#### Grammar

```
lam-expr: LAM fun-header COLON body END
fun-header: LPAREN [params] RPAREN [ARROW ann]
params: param (COMMA param)*
param: NAME [COLONCOLON ann]
body: block | expr
```

#### Expected AST Structure

```json
{
  "type": "s-lam",
  "params": [
    {"bind": {"type": "s-name", "name": "e"}, "ann": null}
  ],
  "args": [],
  "body": {
    "type": "s-block",
    "stmts": [...]
  },
  "check": null,
  "name": "",
  "blocky": false,
  "doc": ""
}
```

#### Implementation Steps

1. **Add `parse_lam_expr()` method:**
   ```rust
   fn parse_lam_expr(&mut self) -> ParseResult<Expr> {
       let start = self.expect(TokenType::Lam)?;

       // Parse parameters
       self.expect(TokenType::ParenNoSpace)?;
       let params = self.parse_comma_list_opt(|p| p.parse_param())?;
       self.expect(TokenType::RParen)?;

       // Parse optional return annotation
       let ret_ann = if self.matches(TokenType::Arrow) {
           Some(self.parse_ann()?)
       } else {
           None
       };

       // Parse body
       self.expect(TokenType::Colon)?;
       let body = self.parse_block()?;
       let end = self.expect(TokenType::End)?;

       Ok(Expr::SLam {
           l: self.make_loc(&start, &end),
           params,
           ret_ann,
           body: Box::new(body),
           // ... other fields
       })
   }
   ```

2. **Add `parse_param()` helper:**
   ```rust
   fn parse_param(&mut self) -> ParseResult<Bind> {
       let name = self.expect_name()?;

       let ann = if self.matches(TokenType::ColonColon) {
           Some(self.parse_ann()?)
       } else {
           None
       };

       Ok(Bind::SBind {
           name,
           ann,
       })
   }
   ```

3. **Add case to `parse_prim_expr()`:**
   ```rust
   TokenType::Lam => self.parse_lam_expr(),
   ```

4. **Update location extraction** (add `Expr::SLam { l, .. } => l`)

5. **Update JSON serialization** in `to_pyret_json.rs`

6. **Remove `#[ignore]`** from tests 502-528

7. **Run tests:**
   ```bash
   cargo test --test comparison_tests test_pyret_match_simple_lambda
   ```

#### Key Challenges

- **Block parsing:** Lambda body is often a block, which might not be implemented yet
  - **Solution:** Implement blocks first, or start with simple expression bodies
- **Parameter annotations:** Need to handle optional type annotations
- **Args vs Params:** The AST has both - check official parser output

#### Verification

```bash
# Test all lambda variants
./compare_parsers.sh "lam(): 5 end"
./compare_parsers.sh "lam(x): x + 1 end"
./compare_parsers.sh "lam(x, y): x + y end"
```

---

### 2. Tuple Expressions (⭐⭐⭐⭐)

**Tests:** 4 tests (lines 534-560)
**Time:** 2-3 hours
**Difficulty:** Medium (disambiguation challenge)

#### What to Implement

```pyret
{1; 3; 10}                        # Basic tuple
{13; 1 + 4; 41; 1}               # With expressions
{151; {124; 152; 12}; 523}       # Nested
x.{2}                             # Tuple field access
```

#### Grammar

```
tuple-expr: LBRACE (expr SEMICOLON)+ RBRACE
tuple-get: expr DOT LBRACE NUMBER RBRACE
```

#### Expected AST Structure

```json
{
  "type": "s-tuple",
  "fields": [
    {"type": "s-num", "value": "1"},
    {"type": "s-num", "value": "3"},
    {"type": "s-num", "value": "10"}
  ]
}
```

#### Implementation Steps

1. **Add `parse_tuple_expr()` method:**
   ```rust
   fn parse_tuple_expr(&mut self) -> ParseResult<Expr> {
       let start = self.expect(TokenType::LBrace)?;

       // Parse fields separated by semicolons
       let mut fields = Vec::new();
       loop {
           fields.push(self.parse_expr()?);

           if !self.matches(TokenType::Semicolon) {
               break;
           }

           // Optional trailing semicolon
           if self.peek().typ == TokenType::RBrace {
               break;
           }
       }

       let end = self.expect(TokenType::RBrace)?;

       Ok(Expr::STuple {
           l: self.make_loc(&start, &end),
           fields,
       })
   }
   ```

2. **Disambiguation challenge:** `{` can start tuple OR object!
   ```rust
   // In parse_prim_expr()
   TokenType::LBrace => {
       // Look ahead to distinguish
       // {x: ...} = object (colon first)
       // {x; ...} = tuple (semicolon first)
       // {} = empty object

       let checkpoint = self.current;

       // Try parsing first element
       self.advance(); // consume '{'

       if self.peek().typ == TokenType::RBrace {
           // Empty object
           return self.parse_obj_expr_from(checkpoint);
       }

       // Parse first element and check separator
       // (Implementation detail - may need backtracking)
   }
   ```

3. **Add tuple field access:**
   ```rust
   // In parse_binop_expr() postfix loop
   if self.peek().typ == TokenType::Dot {
       self.advance();

       if self.matches(TokenType::LBrace) {
           // Tuple access: x.{2}
           let index = self.expect_number()?;
           self.expect(TokenType::RBrace)?;

           left = Expr::STupleGet {
               l: self.make_loc(&left_loc, &self.prev_loc()),
               obj: Box::new(left),
               index,
           };
       } else {
           // Regular dot access
           // ... existing code ...
       }
   }
   ```

4. **Update location extraction** (add both `Expr::STuple` and `Expr::STupleGet`)

5. **Update JSON serialization**

6. **Remove `#[ignore]`** from tests 534-560

#### Key Challenges

- **Disambiguation:** `{1; 2}` (tuple) vs `{x: 1}` (object)
  - **Solution:** Parse first element, check separator token
  - May need backtracking or lookahead
- **Tuple access syntax:** `x.{2}` different from `x.field`
  - **Solution:** Check for `{` after dot in postfix loop

#### Verification

```bash
./compare_parsers.sh "{1; 3; 10}"
./compare_parsers.sh "{13; 1 + 4; 41; 1}"
./compare_parsers.sh "x.{2}"
```

---

### 3. Block Expressions (⭐⭐⭐⭐)

**Tests:** 2 tests (lines 566-578)
**Time:** 2-3 hours
**Difficulty:** Medium (requires statement parsing)

#### What to Implement

```pyret
block: expr end
block: stmt1 stmt2 expr end
```

#### Expected AST Structure

```json
{
  "type": "s-user-block",
  "body": {
    "type": "s-block",
    "stmts": [
      {"type": "s-num", "value": "5"}
    ]
  }
}
```

#### Implementation Steps

1. **Add `parse_block_expr()` method:**
   ```rust
   fn parse_block_expr(&mut self) -> ParseResult<Expr> {
       let start = self.expect(TokenType::Block)?;
       self.expect(TokenType::Colon)?;

       let body = self.parse_block()?;

       let end = self.expect(TokenType::End)?;

       Ok(Expr::SUserBlock {
           l: self.make_loc(&start, &end),
           body: Box::new(body),
       })
   }
   ```

2. **Add `parse_block()` helper:**
   ```rust
   fn parse_block(&mut self) -> ParseResult<Expr> {
       let mut stmts = Vec::new();

       while !self.check_end_block() {
           stmts.push(self.parse_stmt()?);
       }

       Ok(Expr::SBlock {
           stmts,
       })
   }
   ```

3. **Note:** This requires statement parsing infrastructure
   - May need to implement `parse_stmt()`
   - Or start with simple expression-only blocks

4. **Add case to `parse_prim_expr()`**

5. **Update location extraction**

6. **Update JSON serialization**

#### Key Challenges

- **Statement parsing:** Blocks contain statements, not just expressions
  - **Temporary solution:** Start with expression-only blocks
  - **Full solution:** Implement statement parsing infrastructure

#### Verification

```bash
./compare_parsers.sh "block: 5 end"
```

---

### 4. If Expressions (⭐⭐⭐⭐)

**Tests:** 1 test (line 625-629)
**Time:** 2-3 hours
**Difficulty:** Medium

#### What to Implement

```pyret
if cond: then-expr else: else-expr end
if cond: then-expr else if cond2: expr2 else: expr3 end
```

#### Implementation Steps

1. **Add `parse_if_expr()` method:**
   ```rust
   fn parse_if_expr(&mut self) -> ParseResult<Expr> {
       let start = self.expect(TokenType::If)?;

       let cond = self.parse_expr()?;
       self.expect(TokenType::Colon)?;
       let then_branch = self.parse_block()?;

       // Handle else-if chain
       let mut else_ifs = Vec::new();
       while self.matches(TokenType::ElseIf) {
           let else_cond = self.parse_expr()?;
           self.expect(TokenType::Colon)?;
           let else_body = self.parse_block()?;
           else_ifs.push((else_cond, else_body));
       }

       // Handle else
       let else_branch = if self.matches(TokenType::Else) {
           self.expect(TokenType::Colon)?;
           Some(self.parse_block()?)
       } else {
           None
       };

       let end = self.expect(TokenType::End)?;

       Ok(Expr::SIf {
           l: self.make_loc(&start, &end),
           cond: Box::new(cond),
           then_branch: Box::new(then_branch),
           else_ifs,
           else_branch: else_branch.map(Box::new),
       })
   }
   ```

2. **Add to `parse_prim_expr()`**

3. **Update location extraction and JSON serialization**

---

### 5. Method Fields in Objects (⭐⭐⭐)

**Tests:** 1 test (lines 602-607)
**Time:** 2-3 hours
**Difficulty:** Medium

#### What to Implement

```pyret
{
  x: 1,
  method foo(self): self.x end
}
```

#### Implementation Steps

This extends existing object parsing:

1. **Update `parse_obj_field()` in `src/parser.rs`:**
   ```rust
   fn parse_obj_field(&mut self) -> ParseResult<Member> {
       if self.matches(TokenType::Method) {
           // Method field
           let name = self.expect_name()?;

           // Parse function header
           self.expect(TokenType::ParenNoSpace)?;
           let params = self.parse_comma_list_opt(|p| p.parse_param())?;
           self.expect(TokenType::RParen)?;

           // Optional return annotation
           let ret_ann = if self.matches(TokenType::Arrow) {
               Some(self.parse_ann()?)
           } else {
               None
           };

           // Parse body
           self.expect(TokenType::Colon)?;
           let body = self.parse_block()?;
           let end = self.expect(TokenType::End)?;

           Ok(Member::SMethodField {
               name,
               params,
               ret_ann,
               body,
           })
       } else {
           // Existing data field code...
       }
   }
   ```

2. **Update JSON serialization** for `Member::SMethodField`

3. **Remove `#[ignore]`** from test

#### Key Challenges

- **Reuses lambda/function parsing logic**
- **Needs block parsing** for method bodies

---

### 6-14. Remaining Features

See `PARSER_GAPS.md` for detailed specs on:
- For expressions (lines 584-596)
- Cases expressions (lines 613-618)
- Assignment, When, Let/Var (lines 643-662)
- Data declarations (lines 668-673)
- Function declarations (lines 679-684)
- Import/Provide (lines 689-706)

Each follows similar pattern - consult the grammar and AST examples in `PARSER_GAPS.md`.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Test still fails after implementation"

**Check:**
- JSON field names match exactly ("type": "s-lam", not "s-lambda")
- All required fields are present in JSON output
- Field types match (arrays vs objects, numbers vs strings)

**Debug:**
```bash
# See the difference
./compare_parsers.sh "your-code" 2>&1 | less

# Compare JSON structures side-by-side
diff <(./compare_parsers.sh "code" 2>&1 | grep -A 100 "Pyret AST") \
     <(./compare_parsers.sh "code" 2>&1 | grep -A 100 "Rust AST")
```

#### 2. "Parser fails with unexpected token"

**Check:**
- Token type is defined in `src/tokenizer.rs`
- Tokenizer produces expected tokens for your syntax
- You're checking `peek()` not `current`

**Debug:**
```bash
# See what tokens are generated
DEBUG_TOKENS=1 cargo test test_your_feature
```

#### 3. "Location tracking is wrong"

**Check:**
- You added your Expr variant to BOTH location extraction matches
- You're using `self.make_loc(&start, &end)` correctly
- Start token is saved before parsing, end token after

**Location extraction is at:**
- `src/parser.rs:301-322` (first match)
- `src/parser.rs:313-320` (second match)

#### 4. "Compilation errors about missing fields"

**Check:**
- Your Expr variant in `src/ast.rs` matches the official AST structure
- All fields are present in struct definition
- Field types match (Box<Expr> vs Expr, Vec vs single item)

**Reference:**
```bash
# See official AST
./compare_parsers.sh "your-code" 2>&1 | grep -A 50 "Pyret AST"
```

#### 5. "Test passes but AST looks wrong"

**This shouldn't happen** - if test passes, AST is correct by definition!

But if you're concerned:
```bash
# Verify manually
./compare_parsers.sh "your-code"
# Output should say "Identical!"
```

### Getting Help

If stuck:

1. **Check existing similar features** - look for similar expressions already implemented
2. **Read the grammar** - official BNF file has exact syntax rules
3. **Compare ASTs** - use comparison script to see expected structure
4. **Look at real Pyret code** - see how feature is used in practice

**Key files for reference:**
- `src/parser.rs:346-520` - Existing expression parsing
- `tests/parser_tests.rs:203-1229` - Test patterns
- `PHASE2_COMPLETE.md` - Implementation examples
- `PHASE3_PARENS_AND_APPS_COMPLETE.md` - Recent implementation guide

---

## ✅ Verification & Testing

### Before Committing

**Checklist:**
- [ ] All related `#[ignore]` attributes removed
- [ ] Comparison tests pass: `cargo test --test comparison_tests`
- [ ] Parser tests pass: `cargo test --test parser_tests`
- [ ] No compiler warnings: `cargo clippy`
- [ ] Code formatted: `cargo fmt`
- [ ] Added unit tests for edge cases (optional but recommended)
- [ ] Updated location extraction matches (2 places!)
- [ ] Updated JSON serialization
- [ ] Tested manually with `./compare_parsers.sh`

### Test Commands

```bash
# Run all tests
cargo test

# Run only comparison tests
cargo test --test comparison_tests

# Run only parser tests
cargo test --test parser_tests

# Run specific test
cargo test test_pyret_match_simple_lambda

# Run with output
cargo test test_pyret_match_simple_lambda -- --nocapture

# Check for warnings
cargo clippy

# Format code
cargo fmt
```

### Verification Checklist Per Feature

For each feature you implement:

1. **Unit tests pass** (if you added any)
2. **Comparison tests pass** (AST matches official parser)
3. **Manual verification:**
   ```bash
   ./compare_parsers.sh "simple-case"
   ./compare_parsers.sh "complex-case"
   ./compare_parsers.sh "edge-case"
   ```
4. **No regressions** (all 59 existing tests still pass)
5. **Code quality** (no warnings, formatted, documented)

### Success Criteria

**Feature is complete when:**
- ✅ All related comparison tests pass (not ignored)
- ✅ AST matches official Pyret parser exactly
- ✅ No existing tests broken
- ✅ Edge cases handled
- ✅ Code is clean and documented

---

## 📈 Progress Tracking

### Current Status

```
Total: 81 tests
Passing: 59 tests (73%)
Ignored: 22 tests (27%)
Failing: 0 tests (0%)
```

### Milestone Targets

**Phase 1 Complete (11 tests):**
```
Total: 81 tests
Passing: 70 tests (86%)
Ignored: 11 tests (14%)
```

**Phase 2 Complete (4 tests):**
```
Total: 81 tests
Passing: 74 tests (91%)
Ignored: 7 tests (9%)
```

**Phase 3 Complete (7 tests):**
```
Total: 81 tests
Passing: 81 tests (100%)
Ignored: 0 tests (0%)
```

### Tracking Your Progress

After each feature:

1. **Count tests:**
   ```bash
   cargo test --test comparison_tests 2>&1 | grep "test result"
   ```

2. **Update documentation:**
   - Add checkmark in `PARSER_GAPS.md`
   - Update `CLAUDE.md` status section

3. **Commit with clear message:**
   ```bash
   git add .
   git commit -m "feat(parser): implement lambda expressions

   - Add parse_lam_expr() method
   - Support parameters with optional type annotations
   - Add tests for simple, single-param, and multi-param lambdas
   - All lambda comparison tests passing (4/4)

   Tests: 63/81 passing (78% coverage)"
   ```

---

## 🎓 Additional Resources

### Documentation Files

**Must Read:**
- `GAP_ANALYSIS_SUMMARY.md` - Quick overview
- `PARSER_GAPS.md` - Detailed feature breakdown
- `MISSING_FEATURES_EXAMPLES.md` - Real code examples
- This file (`IMPLEMENTATION_GUIDE.md`) - Step-by-step guide

**Reference:**
- `NEXT_STEPS.md` - Original implementation patterns
- `CLAUDE.md` - Project overview and status
- `README.md` - Quick reference
- `HANDOFF_CHECKLIST.md` - Quick verification

**Historical:**
- `PHASE1_COMPLETE.md` - Foundation work
- `PHASE2_COMPLETE.md` - Primitives and operators
- `PHASE3_PARENS_AND_APPS_COMPLETE.md` - Recent work

### Code References

**Parser Implementation:**
- `src/parser.rs:188-520` - Expression parsing (Section 6)
- `src/parser.rs:301-322` - Location extraction (UPDATE THIS!)
- `src/parser.rs:708-end` - Helper methods (Section 12)

**AST Definitions:**
- `src/ast.rs:292-808` - Expression enum with all variants

**JSON Serialization:**
- `src/bin/to_pyret_json.rs:1-190` - JSON output (UPDATE THIS!)

**Tokenizer:**
- `src/tokenizer.rs:125-185` - Token types

**Tests:**
- `tests/comparison_tests.rs:1-497` - Passing tests (examples)
- `tests/comparison_tests.rs:498-706` - Ignored tests (YOUR TARGET!)
- `tests/parser_tests.rs:203-1229` - Unit tests

### Official Pyret Resources

**Grammar:**
- `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`

**Real Code Examples:**
- `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/tests/pyret/tests/test-lists.arr`
- `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/tests/pyret/tests/test-tuple.arr`
- `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/tests/pyret/tests/test-binops.arr`
- And more in `tests/pyret/tests/` directory

**Official Parser:**
- `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-parser.js`

---

## 🚀 Ready to Implement!

You now have everything you need:

1. **22 tests** waiting to be implemented
2. **Clear priorities** (start with lambdas!)
3. **Step-by-step guides** for each feature
4. **Real examples** from actual Pyret code
5. **Verification tools** (`./compare_parsers.sh`)
6. **Comprehensive documentation**

**Start with lambda expressions** - they're the highest priority and unlock most real Pyret programs!

```bash
# Your first command
grep -A 30 "Lambda Expressions" PARSER_GAPS.md
```

**Good luck!** 🎉

---

**Questions?** All answers are in the documentation files listed above.

**Found a bug?** Use the bug tracker tool documented in `CLAUDE.md`.

**Need help?** Check the troubleshooting section above, then examine similar existing implementations.
