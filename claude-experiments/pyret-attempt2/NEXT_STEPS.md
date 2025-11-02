# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-11-02
**Current Status:** ✅ Method fields complete! Ready for function definitions.
**Tests Passing:** 68/68 parser tests ✅ (100%), 73/81 comparison tests ✅ (90.1%)

---

## ✅ COMPLETED - Method Fields in Objects

**MILESTONE ACHIEVED!** 🎉

Method fields are now fully working, bringing us to 73/81 comparison tests passing (90.1%)!

**What was completed:**
1. ✅ Implemented `parse_method_field()` - Parses method syntax in objects
2. ✅ Method parameter parsing with `Bind` structures
3. ✅ Correctly distinguishes `params` (type parameters) from `args` (function parameters)
4. ✅ Optional return type annotation support (`-> ann`)
5. ✅ Optional where clause support for tests
6. ✅ Added JSON serialization for `SMethodField` with correct field ordering
7. ✅ Added comprehensive parser test `test_parse_object_with_method`
8. ✅ Enabled comparison test `test_pyret_match_object_with_method`

All 73 passing comparison tests produce identical ASTs to the official Pyret parser!

---

## 📋 Next Priority Tasks (IN ORDER)

### 1. parse_fun_expr() - Function Definitions ⭐⭐⭐⭐ (HIGHEST PRIORITY)
**Why:** Required for 1 comparison test (`simple_fun`) and very similar to already-implemented lambdas/methods

**Grammar:**
```bnf
fun-expr: FUN NAME fun-header (BLOCK|COLON) doc-string block where-clause END
fun-header: ty-params args return-ann | ty-params bad-args return-ann
ty-params: [(LANGLE|LT) comma-names (RANGLE|GT)]
args: (PARENNOSPACE|PARENAFTERBRACE) [binding (COMMA binding)*] RPAREN
return-ann: [THINARROW ann]
where-clause: [WHERE block]
```

**Examples:**
- `fun f(x): x + 1 end`
- `fun add(a, b): a + b end`
- `fun identity<T>(x :: T) -> T: x end`

**AST Node:** `Expr::SFun { l, name, params, args, ann, doc, body, check_loc, check, blocky }`

**Implementation Steps:**

1. **Study the SFun AST in** `src/ast.rs`:
```rust
Expr::SFun {
    l: Loc,
    name: String,           // Function name (e.g., "f", "add")
    params: Vec<Name>,      // Type parameters (empty for now, like <T>)
    args: Vec<Bind>,        // Value parameters (e.g., x, a, b)
    ann: Ann,               // Return type annotation
    doc: String,            // Documentation string
    body: Box<Expr>,        // Function body (usually SBlock)
    check_loc: Option<Loc>, // Location of check block
    check: Option<Box<Expr>>, // Optional check/where block
    blocky: bool,           // true if uses 'block' keyword
}
```

**Key insight:** This is VERY similar to `parse_method_field()` (which you just implemented) and `parse_lambda_expr()`.
The main differences are:
- Starts with `FUN` keyword instead of `METHOD` or `LAM`
- Has a function name (like method fields, unlike lambdas)
- Is an `Expr::SFun` not a `Member::SMethodField`
- Otherwise identical structure!

2. **Add FUN case to parse_prim_expr():**

In `src/parser.rs`, find the `parse_prim_expr()` method and add:

```rust
TokenType::Fun => self.parse_fun_expr(),
```

(Look at how `TokenType::Lam` is handled - do the same for `Fun`)

3. **Implement parse_fun_expr() - Copy parse_method_field() and adapt:**

```rust
fn parse_fun_expr(&mut self) -> ParseResult<Expr> {
    let start = self.expect(TokenType::Fun)?;

    // Parse function name
    let name_token = self.expect(TokenType::Name)?;
    let name = name_token.value.clone();

    // Parse parameters - SAME AS METHOD FIELDS
    let paren_token = self.peek().clone();
    match paren_token.token_type {
        TokenType::LParen | TokenType::ParenSpace | TokenType::ParenNoSpace => {
            self.advance();
        }
        _ => {
            return Err(ParseError::expected(TokenType::LParen, paren_token));
        }
    }

    let args = if self.matches(&TokenType::RParen) {
        Vec::new()
    } else {
        self.parse_comma_list(|p| p.parse_bind())?
    };

    // params is for type parameters (e.g., <T>), not function parameters
    let params: Vec<Name> = Vec::new();

    self.expect(TokenType::RParen)?;

    // Optional return type annotation (-> ann)
    let ann = if self.matches(&TokenType::ThinArrow) {
        self.expect(TokenType::ThinArrow)?;
        self.parse_ann()?
    } else {
        Ann::ABlank
    };

    // Parse body separator (COLON or BLOCK)
    let blocky = if self.matches(&TokenType::Block) {
        self.advance();
        true
    } else {
        self.expect(TokenType::Colon)?;
        false
    };

    let doc = String::new();

    // Parse function body (statements until END or WHERE)
    let mut body_stmts = Vec::new();
    while !self.matches(&TokenType::End)
        && !self.matches(&TokenType::Where)
        && !self.is_at_end()
    {
        let stmt = self.parse_expr()?;
        body_stmts.push(Box::new(stmt));
    }

    // Parse where clause if present
    let check = if self.matches(&TokenType::Where) {
        self.advance();
        let mut where_stmts = Vec::new();
        while !self.matches(&TokenType::End) && !self.is_at_end() {
            let stmt = self.parse_expr()?;
            where_stmts.push(Box::new(stmt));
        }
        Some(Box::new(Expr::SBlock {
            l: self.current_loc(),
            stmts: where_stmts,
        }))
    } else {
        None
    };

    let end = self.expect(TokenType::End)?;

    let body = Box::new(Expr::SBlock {
        l: self.current_loc(),
        stmts: body_stmts,
    });

    let check_loc = check.as_ref().map(|c| match c.as_ref() {
        Expr::SBlock { l, .. } => l.clone(),
        _ => self.current_loc(),
    });

    Ok(Expr::SFun {  // Note: Expr::SFun not Member::SMethodField!
        l: self.make_loc(&start, &end),
        name,
        params,
        args,
        ann,
        doc,
        body,
        check_loc,
        check,
        blocky,
    })
}
```

4. **Add JSON serialization in to_pyret_json.rs:**

Find the `expr_to_pyret_json()` function and add `SFun` case (look at `SLam` for reference):

```rust
Expr::SFun { name, params, args, ann, doc, body, check, check_loc, blocky, .. } => {
    json!({
        "type": "s-fun",
        "name": name,
        "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
        "args": args.iter().map(|a| bind_to_pyret_json(a)).collect::<Vec<_>>(),
        "ann": ann_to_pyret_json(ann),
        "doc": doc,
        "body": expr_to_pyret_json(body),
        "check": check.as_ref().map(|c| expr_to_pyret_json(c)),
        "check-loc": check_loc,
        "blocky": blocky
    })
}
```

5. **Update location extraction for SFun:**

Search for all the `match` statements that extract locations (search for `Expr::SLam { l, .. } => l.clone()`) and add:

```rust
Expr::SFun { l, .. } => l.clone(),
```

right after each `SLam` case.

6. **Add parser tests:**

```rust
#[test]
fn test_parse_simple_function() {
    let expr = parse_expr("fun f(x): x + 1 end").expect("Failed to parse function");

    match expr {
        Expr::SFun { name, args, body, .. } => {
            assert_eq!(name, "f");
            assert_eq!(args.len(), 1);
            // Check body is a block with one statement
            match body.as_ref() {
                Expr::SBlock { stmts, .. } => {
                    assert_eq!(stmts.len(), 1);
                }
                _ => panic!("Expected SBlock for function body"),
            }
        }
        _ => panic!("Expected SFun, got {:?}", expr),
    }
}
```

7. **Enable comparison test:**

Remove `#[ignore]` from `test_pyret_match_simple_fun` in `tests/comparison_tests.rs`

8. **Run comparison to debug differences:**

```bash
./compare_parsers.sh "fun f(x): x + 1 end"
```

**Estimated Time:** 2-3 hours (mostly copy-paste from method fields!)

---

### 2. parse_data_expr() - Data Definitions ⭐⭐
**Why:** Required for 1 comparison test (`simple_data`)

**Example:** `data Box: | box(ref v) end`

**Estimated Time:** 3-4 hours

---

### 3. parse_cases_expr() - Cases Expressions ⭐⭐⭐
**Why:** Required for 1 comparison test (`simple_cases`)

**Example:** `cases (Either) e: | left(v) => v | right(v) => v end`

**Estimated Time:** 4-5 hours

---

### 4. parse_when_expr() - When Expressions ⭐⭐
**Why:** Required for 1 comparison test (`simple_when`)

**Example:** `when true: print("yes") end`

**Estimated Time:** 1-2 hours

---

### 5. parse_assign_expr() - Assignment Expressions ⭐⭐
**Why:** Required for 1 comparison test (`simple_assign`)

**Example:** `x := 5`

**Estimated Time:** 1-2 hours

---

## 🧪 Testing Strategy

When implementing a new feature:

1. **Read the comparison test** to see what syntax is expected
2. **Check the Pyret grammar** in `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
3. **Look at similar features** already implemented (lambdas, methods, etc.)
4. **Implement parsing** - copy-paste similar code and adapt
5. **Add JSON serialization** - look at similar AST nodes
6. **Update location extraction** - add new Expr types to all match statements
7. **Add parser tests** - test the basic functionality
8. **Enable comparison test** - remove `#[ignore]`
9. **Run comparison** - `./compare_parsers.sh "your code here"`
10. **Debug differences** - adjust JSON field names/order to match

---

## 📝 Key Insights

**Similarities between features:**
- `SFun`, `SLam`, and `SMethodField` are almost identical
- All use `params` (type parameters) and `args` (value parameters)
- All support optional return types, doc strings, where clauses
- Copy-paste is your friend!

**Important patterns:**
- `params` = type parameters (like `<T>`) - always empty for now
- `args` = value parameters (like `x, y, z`)
- `check` / `check_loc` = where clause for tests
- `blocky` = true if uses `block:` instead of `:`

---

## 🎯 Quick Summary for Next Session

**Current Status:**
- ✅ 73/81 comparison tests passing (90.1%)
- ✅ Method fields complete - all ASTs match Pyret parser
- ✅ 68/68 parser tests passing (100%)

**Next Feature: FUNCTION DEFINITIONS**

**What to do:**
1. Look at `parse_method_field()` in `src/parser.rs:1384-1509`
2. Copy it and rename to `parse_fun_expr()`
3. Change `TokenType::Method` → `TokenType::Fun`
4. Change return type `Member::SMethodField` → `Expr::SFun`
5. Add `TokenType::Fun => self.parse_fun_expr()` to `parse_prim_expr()`
6. Add JSON serialization (copy from `SLam`, adapt for `SFun`)
7. Add location extraction (`Expr::SFun { l, .. } => l.clone()`)
8. Add parser test
9. Enable comparison test (remove `#[ignore]`)
10. Run `./compare_parsers.sh "fun f(x): x + 1 end"`

**Estimated Time:** 2-3 hours (mostly copy-paste!)

**We're at 90.1% completion!** Only 8 more features to go! 🚀

