# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-11-01
**Current Status:** ✅ Full program parsing complete! Ready for statements.
**Tests Passing:** 64/64 parser tests ✅, 69/81 comparison tests ✅ (all using full Program ASTs!)

---

## ✅ COMPLETED - Full Program Parsing

**MILESTONE ACHIEVED!** 🎉

We now parse complete Pyret programs and compare full Program ASTs with the official parser.

**What was completed:**
1. ✅ Implemented `parse_program()` - parses complete Pyret programs (src/parser.rs:193-234)
2. ✅ Implemented `parse_block()` - parses statement sequences (src/parser.rs:245-269)
3. ✅ Updated `to_pyret_json.rs` to output full Program AST with helpers
4. ✅ Fixed `compare_parsers.sh` and `compare_parsers_quiet.sh` (removed stmts[0] hack)

All 69 passing comparison tests now compare complete Program ASTs byte-for-byte with the official parser!

---

## 📋 Next Priority Tasks (IN ORDER)

### 1. parse_let_expr() - Let Bindings ⭐⭐⭐⭐ (HIGHEST PRIORITY)
**Why:** Required for `block_multiple_stmts` test and enables variable declarations

**Grammar:**
```bnf
let-expr: LET bind = expr
        | LET bind = expr BLOCK body END
var-expr: VAR bind := expr
```

**AST Node:** `Expr::SLetExpr { l, binds, body }` or `Expr::SVarExpr { l, bind, value }`

**Implementation Steps:**

1. **Study the AST structure** in `src/ast.rs`:
```rust
Expr::SLetExpr {
    l: Loc,
    binds: Vec<LetBind>,  // List of let bindings
    body: Box<Expr>,       // Body expression
}

// LetBind is defined in ast.rs
pub struct LetBind {
    pub b: Bind,     // The binding (name + optional type)
    pub value: Expr, // The value expression
}
```

2. **Update parse_block() to handle let statements:**
```rust
fn parse_block(&mut self) -> ParseResult<Expr> {
    let start = self.peek().clone();
    let mut stmts = Vec::new();

    while !self.is_at_end() {
        // Try to parse different statement types
        let stmt = if self.matches(&TokenType::Let) {
            self.parse_let_expr()?
        } else if self.matches(&TokenType::Var) {
            self.parse_var_expr()?
        } else {
            // Default: parse as expression
            match self.parse_expr() {
                Ok(expr) => expr,
                Err(_) => break,  // Stop if we can't parse
            }
        };
        stmts.push(Box::new(stmt));
    }

    let end = if self.current > 0 {
        self.tokens[self.current - 1].clone()
    } else {
        start.clone()
    };

    Ok(Expr::SBlock {
        l: self.make_loc(&start, &end),
        stmts,
    })
}
```

3. **Implement parse_let_expr():**
```rust
fn parse_let_expr(&mut self) -> ParseResult<Expr> {
    let start = self.expect(TokenType::Let)?;

    // Parse binding: name [:: type]
    let bind = self.parse_bind()?;

    // Expect =
    self.expect(TokenType::Equals)?;

    // Parse value expression
    let value = self.parse_expr()?;

    // Create LetBind
    let let_bind = LetBind {
        b: bind.clone(),
        value: value.clone(),
    };

    // For now, let expressions without explicit body just use the value
    let body = value.clone();  // TODO: Handle BLOCK ... END syntax

    let end = self.tokens[self.current - 1].clone();

    Ok(Expr::SLetExpr {
        l: self.make_loc(&start, &end),
        binds: vec![let_bind],
        body: Box::new(body),
    })
}
```

4. **Add JSON serialization in to_pyret_json.rs:**
```rust
Expr::SLetExpr { binds, body, .. } => {
    json!({
        "type": "s-let-expr",
        "binds": binds.iter().map(|b| let_bind_to_pyret_json(b)).collect::<Vec<_>>(),
        "body": expr_to_pyret_json(body)
    })
}

fn let_bind_to_pyret_json(lb: &LetBind) -> Value {
    json!({
        "type": "s-let-bind",
        "bind": bind_to_pyret_json(&lb.b),
        "value": expr_to_pyret_json(&lb.value)
    })
}
```

**Estimated Time:** 1-2 hours

---

### 2. parse_for_expr() - For Expressions ⭐⭐⭐ (HIGH PRIORITY)
**Why:** Required for 2 comparison tests (`for_map`, `for_map2`)
```

**Estimated Time:** 1 hour

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
