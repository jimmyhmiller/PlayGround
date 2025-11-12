# Pyret Parser

A recursive descent parser for the [Pyret programming language](https://www.pyret.org/), vibe coded with Claude. The parser generates JSON ASTs that match the reference JavaScript implementation byte-for-byte.


## (HUMAN WRITTEN NOTE)

This was pretty much entirely vibe coded with claude. If there is some nonesense don't judge me.

## Modes

The parser supports four modes for working with Pyret code:

### 1. **Tokenize Mode** - See the token stream
```bash
cargo run -- --mode tokenize examples/factorial.arr
```

Breaks down Pyret source into tokens (keywords, operators, identifiers, etc.). Useful for understanding how the lexer sees your code.

### 2. **Parse Mode** - View the AST
```bash
cargo run -- --mode parse examples/factorial.arr
```

Parses Pyret code into a complete Abstract Syntax Tree. Outputs Rust's debug format showing the full structure.

### 3. **JSON Mode** (default) - Pyret-compatible JSON
```bash
cargo run -- --mode json examples/factorial.arr
# or just:
cargo run -- examples/factorial.arr
```

Generates JSON output that exactly matches the official Pyret parser. This is the primary output mode and what's validated in our 307 passing tests.

### 4. **Scheme Mode** - Compile to R4RS Scheme
```bash
cargo run -- --mode scheme examples/simple.arr
```

Experimental compiler that translates a small subset of Pyret to R4RS Scheme. This is a proof-of-concept demonstrating that the parser can be used as a compiler frontend.

**What Scheme mode supports:**
- Numbers and identifiers
- Binary operators: `+`, `-`, `*`, `/`, `<=`, `>=`, `<`, `>`, `==`
- Function definitions: `fun factorial(n): n end`
- Function calls: `factorial(5)`
- If-else expressions

**Example:**
```pyret
fun factorial(n):
  if n <= 1:
    1
  else:
    n * factorial(n - 1)
  end
end

factorial(5)
```

Compiles to:
```scheme
(define (factorial n)
  (if (<= n 1)
    1
    (* n (factorial (- n 1)))))

(factorial 5)
```

## Quick Start

```bash
# Parse a Pyret file to JSON
cargo run -- myfile.arr

# Try the scheme compiler
echo "fun double(x): x + x end" > double.arr
cargo run -- --mode scheme double.arr

# See tokens
cargo run -- --mode tokenize myfile.arr --pretty

# Run all tests (307 passing!)
cargo test
```

## Project Structure

```
src/
├── parser.rs       - Hand-written recursive descent parser
├── ast.rs          - All AST node types (matching Pyret's reference impl)
├── tokenizer.rs    - Lexer with whitespace-sensitive tokens
├── codegen.rs      - Experimental Scheme compiler
└── main.rs         - CLI with 4 modes

tests/
├── parser_tests.rs      - 75 unit tests ✅
└── comparison_tests.rs  - 307 integration tests validating against official Pyret ✅
```

## Key Features

**Whitespace-sensitive parsing** - `f(x)` is a function call, `f (x)` is two expressions

**No operator precedence** - All binary operators are equal and left-associative:
```pyret
2 + 3 * 4  // evaluates as (2 + 3) * 4 = 20, not 14!
```

**Arbitrary precision numbers** - Numbers stored as strings to preserve exact values

**Type-safe operators** - Uses Rust enums instead of strings for operators (refactored today!)

## What's Implemented

Nearly everything in Pyret:
- All primitive expressions (numbers, strings, booleans, etc.)
- All binary operators (15 total)
- Function definitions with generics and type annotations
- Lambda expressions
- Data declarations with variants and sharing clauses
- Pattern matching (cases expressions)
- Control flow (if, when, for-map/filter/fold/each, blocks)
- Object expressions with methods
- Import/export/provide statements
- Check blocks with refinements
- Table expressions
- Spy expressions
- Template dots (`...`)
- Underscore partial application (`_ + 2`)

See [CLAUDE.md](CLAUDE.md) for the complete feature list and implementation history.

## Testing

All 307 passing tests verify **byte-for-byte identical JSON** output compared to the official Pyret parser.

```bash
# Run all tests
cargo test

# Compare with official parser
./scripts/compare_parsers.sh "fun f(x): x + 1 end"
```

## Why?

This project demonstrates:
1. AI-assisted development for complex parsing tasks
2. Building parsers through conversation rather than traditional coding
3. The Pyret language and its grammar
4. Recursive descent parsers as an alternative to parser generators

## References

- **Pyret Language:** https://www.pyret.org/
- **Grammar Spec:** Based on pyret-lang/src/js/base/pyret-grammar.bnf
- **Reference Implementation:** pyret-lang (JavaScript)
