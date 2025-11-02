# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-11-01
**Current Status:** ✅ For expressions complete! Ready for method fields.
**Tests Passing:** 67/67 parser tests ✅ (100%), 72/81 comparison tests ✅ (88.9%)

---

## ✅ COMPLETED - For Expressions

**MILESTONE ACHIEVED!** 🎉

For expressions are now fully working, bringing us to 72/81 comparison tests passing (88.9%)!

**What was completed:**
1. ✅ Implemented `parse_for_expr()` - Parses for expressions with iterator and bindings
2. ✅ Iterator expression parsing with dot access support (`lists.map2`)
3. ✅ For-bindings with `FROM` keyword (`x from lst`)
4. ✅ Added `ForBind` structures with proper `Bind` and value expressions
5. ✅ Added JSON serialization for `SFor` and `ForBind`
6. ✅ Updated location extraction for `SFor` expressions (5 locations)
7. ✅ Added 2 comprehensive parser tests (simple, dot access)
8. ✅ Enabled 2 comparison tests (`test_pyret_match_for_map`, `test_pyret_match_for_map2`)

All 72 passing comparison tests produce identical ASTs to the official Pyret parser!

---

## 📋 Next Priority Tasks (IN ORDER)

### 1. parse_method_field() - Method Fields in Objects ⭐⭐⭐⭐ (HIGHEST PRIORITY)
**Why:** Required for 1 comparison test (`object_with_method`) and completes object expression support

**Grammar:**
```bnf
obj-field: NAME COLON expr                  # data field
         | REF NAME ann COLON expr          # mutable field
         | METHOD NAME params ann COLON doc body WHERE bindings END  # method field
```

**Examples:**
- `{ method _plus(self, other): self.x + other.x end }`
- `{ x: 5, method double(self): self.x * 2 end }`

**AST Node:** `Member::SMethodField { l, name, params, ann, doc, body, blocky }`

**Implementation Steps:**

1. **Study the Member AST in** `src/ast.rs`:
```rust
Member::SMethodField {
    l: Loc,
    name: String,       // Method name (e.g., "_plus", "double")
    params: Vec<Bind>,  // Parameters including 'self'
    ann: Ann,           // Return type annotation
    doc: String,        // Documentation string
    body: Box<Expr>,    // Method body (usually SBlock)
    _check: Option<Box<Expr>>,  // Optional check block
    blocky: bool,       // true if uses 'block' keyword
}
```

2. **Update parse_obj_field() to handle METHOD keyword:**
```rust
fn parse_obj_field(&mut self) -> ParseResult<Member> {
    if self.matches(&TokenType::Method) {
        return self.parse_method_field();
    }
    // ... existing ref/data field logic ...
}
```

3. **Implement parse_method_field():**
```rust
fn parse_method_field(&mut self) -> ParseResult<Member> {
    let start = self.expect(TokenType::Method)?;

    // Parse method name
    let name_token = self.expect(TokenType::Name)?;
    let name = name_token.value.clone();

    // Parse parameters (like lambda)
    self.expect(TokenType::ParenSpace)?;  // or LParen
    let params = if self.matches(&TokenType::RParen) {
        Vec::new()
    } else {
        self.parse_comma_list(|p| p.parse_bind())?
    };
    self.expect(TokenType::RParen)?;

    // Optional type annotation
    let ann = if self.matches(&TokenType::ColonColon) {
        self.advance();
        self.parse_ann()?  // Parse type annotation
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

    // Parse doc string (usually empty)
    let doc = String::new();

    // Parse method body (statements until END)
    let mut body_stmts = Vec::new();
    while !self.matches(&TokenType::End) && !self.is_at_end() {
        let stmt = self.parse_expr()?;
        body_stmts.push(Box::new(stmt));
    }

    let body = Box::new(Expr::SBlock {
        l: self.current_loc(),
        stmts: body_stmts,
    });

    let end = self.expect(TokenType::End)?;

    Ok(Member::SMethodField {
        l: self.make_loc(&start, &end),
        name,
        params,
        ann,
        doc,
        body,
        _check: None,
        blocky,
    })
}
```

4. **Add JSON serialization in to_pyret_json.rs:**
```rust
Member::SMethodField { name, params, ann, doc, body, blocky, .. } => {
    json!({
        "type": "s-method-field",
        "name": name,
        "params": params.iter().map(|p| bind_to_pyret_json(p)).collect::<Vec<_>>(),
        "ann": ann_to_pyret_json(ann),
        "doc": doc,
        "body": expr_to_pyret_json(body),
        "check": null,  // _check field
        "blocky": blocky
    })
}
```

**Estimated Time:** 2-3 hours

---

### 2. parse_fun_expr() - Function Definitions ⭐⭐⭐⭐ (HIGH PRIORITY)
**Why:** Required for 1 comparison test (`simple_fun`)

---

### 4. Fix compare_parsers.sh ⭐⭐⭐⭐⭐
**Why:** Remove the stmts[0] hack and compare full programs

**Implementation Steps:**

1. **Remove the extraction logic** (lines 32-42):
```bash
# DELETE THIS SECTION:
# Extract just the expression from Pyret's output (first statement in body)
python3 -c "
import json, sys
with open('$PYRET_JSON') as f:
    data = json.load(f)
if 'body' in data and 'stmts' in data['body'] and len(data['body']['stmts']) > 0:
    with open('$PYRET_EXPR', 'w') as out:
        json.dump(data['body']['stmts'][0], out, indent=2)
else:
    print('ERROR: No expression found in Pyret output', file=sys.stderr)
    sys.exit(1)
"
```

2. **Compare full programs directly:**
```bash
# Parse with Pyret's official parser
echo "=== Pyret Parser ==="
cd /Users/jimmyhmiller/Documents/Code/open-source/pyret-lang
node ast-to-json.jarr "$TEMP_FILE" "$PYRET_JSON" 2>&1 | grep "JSON written" || true
cat "$PYRET_JSON"

# Parse with our Rust parser
echo "=== Rust Parser ==="
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2
cargo run --bin to_pyret_json "$TEMP_FILE" 2>/dev/null > "$RUST_JSON"
cat "$RUST_JSON"

# Compare the two JSON outputs
echo "=== Comparison ==="
python3 << 'EOF'
import json
import sys

with open('/tmp/pyret_output.json') as f:
    pyret = json.load(f)

with open('/tmp/rust_output.json') as f:
    rust = json.load(f)

# Compare full programs
if normalize_json(pyret) == normalize_json(rust):
    print("✅ IDENTICAL")
    sys.exit(0)
else:
    print("❌ DIFFERENT")
    # Print diffs
    sys.exit(1)
EOF
```

**Estimated Time:** 30 minutes

---

## 🧪 Testing Strategy

1. **Start simple** - Test with single-expression programs:
   - `42`
   - `2 + 3`
   - `f(x)`

2. **Add multi-statement programs:**
   - `1\n2\n3` (three expressions)
   - Mixed statements

3. **Run comparison tests:**
```bash
./compare_parsers.sh "42"
./compare_parsers.sh "2 + 3"
```

4. **Update comparison_tests.rs** to stop using expression-only parsing

---

## 📝 Key Points

1. **parse_block() vs parse_block_expr():**
   - `parse_block_expr()` parses `block: ... end` (expression form)
   - `parse_block()` parses statement sequences (for program bodies)
   - These are DIFFERENT!

2. **Statement termination:**
   - Need to understand when one statement ends
   - Likely newline-based (check Pyret grammar)
   - May need to handle implicit statement separators

3. **Keep it simple first:**
   - Start with single-expression programs
   - Add multi-statement support incrementally
   - Don't implement all statement types at once

4. **The goal:**
   - Parse complete `.arr` files
   - Compare full Program ASTs
   - Remove the stmts[0] hack forever

---

## ⏭️ After Program Parsing Works

Once parse_program() and parse_block() are working:

1. **For expressions** - List comprehensions
2. **Let bindings** - Variable bindings
3. **Function definitions** - Top-level functions
4. **Data definitions** - Custom types
5. **Import/provide statements** - Module system

But FIRST: **GET PROGRAM PARSING WORKING!**

---

**Estimated Total Time:** 4-5 hours
**Priority:** 🚨 CRITICAL - Drop everything else and do this first!

---

## 🎯 Quick Summary for Next Session

**Current Status:**
- ✅ 72/81 comparison tests passing (88.9%)
- ✅ For expressions fully working
- ✅ All control flow and statement parsing infrastructure in place

**Next Feature to Implement: METHOD FIELDS IN OBJECTS**

**What to do:**
1. Check the ignored comparison tests to see what features are needed
2. Look at `test_pyret_match_object_with_method`
3. Study `Member::SMethodField` AST structure in `src/ast.rs`
4. Update `parse_obj_field()` to handle `METHOD` keyword
5. Implement `parse_method_field()` (similar to lambda parsing)
6. Add JSON serialization in `to_pyret_json.rs`
7. Add parser tests
8. Enable comparison test

**Reference:**
- Grammar: `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
- Look for: `obj-field` and `method-field` in the grammar
- Similar to: lambda expressions (already implemented)

**Estimated Time:** 2-3 hours for full implementation

**After that:**
- Function definitions (`fun f(x): ... end`) - 1 test waiting
- Cases expressions (`cases (Type) expr: ... end`) - 1 test waiting
- Data definitions (`data Point: point(x, y) end`) - 1 test waiting
- When expressions (`when expr: ... end`) - 1 test waiting

**We're at 88.9% test coverage!** 🚀 Only 9 more tests to go!

