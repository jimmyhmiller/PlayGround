# Operator Precedence in Pyret

## Summary

**Pyret has NO operator precedence hierarchy.** All binary operators have equal precedence and are left-associative. This is a **direct property of the BNF grammar**, not a parser implementation detail.

## The Grammar Rule

```bnf
binop-expr: expr (binop expr)*

binop: PLUS | DASH | TIMES | SLASH | LEQ | GEQ | EQUALEQUAL | SPACESHIP | EQUALTILDE
     | NEQ  | LT  | GT | AND | OR | CARET
```

This grammar rule defines the behavior:
1. **Flat structure**: All operators are listed as alternatives with no hierarchy
2. **Left-associative**: The `expr (binop expr)*` pattern naturally produces left-associativity
3. **Equal precedence**: There's no nesting of expression types (like `term`, `factor`, etc.) that would create precedence levels

## What This Means

### Expression Parsing
```pyret
2 + 3 * 4       # Parses as: (2 + 3) * 4 = 20  (not 2 + (3 * 4) = 14)
1 + 2 + 3 + 4   # Parses as: (((1 + 2) + 3) + 4)
x < y and z     # Parses as: (x < y) and z
```

### Left-Associative Processing
Given `a op1 b op2 c op3 d`:
1. Parse `a` (first expr)
2. See `op1`, parse `b` → create `(a op1 b)`
3. See `op2`, parse `c` → create `((a op1 b) op2 c)`
4. See `op3`, parse `d` → create `(((a op1 b) op2 c) op3 d)`

### Comparison to Traditional Precedence

Most languages have precedence hierarchies:
```bnf
# Traditional approach (NOT Pyret!)
expr: term ((PLUS | DASH) term)*
term: factor ((TIMES | SLASH) factor)*
factor: NUMBER | LPAREN expr RPAREN
```

This creates precedence: `*` binds tighter than `+`.

**Pyret doesn't do this.** Its grammar is intentionally flat.

## Our Implementation

Our Rust parser correctly implements this behavior in `parse_binop_expr()`:

```rust
fn parse_binop_expr(&mut self) -> ParseResult<Expr> {
    let mut left = self.parse_prim_expr()?;

    // Left-associative: keep building from left
    while self.is_binop() {
        let op = self.parse_binop()?;
        let right = self.parse_prim_expr()?;

        left = Expr::SOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            // ...
        };
    }

    Ok(left)
}
```

This directly implements the `expr (binop expr)*` grammar pattern.

## Why Pyret Does This

Design philosophy:
1. **Simplicity** - No precedence rules to memorize
2. **Explicitness** - Users write what they mean with parentheses
3. **Consistency** - All operators behave the same way
4. **Pedagogical** - Good for teaching (no "magic" precedence rules)

## Practical Impact

Users must use parentheses for any non-left-to-right evaluation:
```pyret
2 + 3 * 4        # ERROR: probably not what you want! (= 20)
2 + (3 * 4)      # Correct: = 14
```

The Pyret compiler/linter may warn about missing parentheses in ambiguous cases.

## Testing

Our test suite confirms this behavior:

```rust
#[test]
fn test_parse_left_associative() {
    // 1 + 2 + 3 should parse as (1 + 2) + 3
    let expr = parse_expr("1 + 2 + 3").expect("Failed to parse");

    match expr {
        Expr::SOp { left, right, .. } => {
            // Right is just 3
            assert!(matches!(*right, Expr::SNum { n: 3.0, .. }));
            // Left is (1 + 2)
            assert!(matches!(*left, Expr::SOp { .. }));
        }
        _ => panic!("Expected SOp"),
    }
}
```

## References

- BNF Grammar: `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf` (lines 122-125)
- Our Parser: `src/parser.rs` (`parse_binop_expr()` function)
- Tests: `tests/parser_tests.rs` (`test_parse_left_associative`)

---

**Key Takeaway**: Pyret's operator precedence is **zero** (all equal). This is not a parser implementation choice—it's baked into the grammar definition itself. Our parser correctly implements this by following the grammar structure.
