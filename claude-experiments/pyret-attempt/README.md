# Pyret Parser in Rust

A Rust implementation of a parser for the [Pyret programming language](https://www.pyret.org/), built using the [Pest](https://pest.rs/) parser generator.

## Overview

This project converts the Pyret BNF grammar into a working Pest grammar and provides idiomatic Rust AST types that mirror Pyret's AST structure. The goal is to make it easier to port Pyret tools and analysis to Rust while maintaining compatibility with the original AST design.

## Features

- **Complete Pest Grammar**: Converted from `pyret-grammar.bnf` with left-recursion eliminated
- **Rust AST Types**: Comprehensive AST that mirrors Pyret's structure from `src/arr/trove/ast.arr`
- **Idiomatic Rust**: Uses enums for sum types, proper ownership, and Rust best practices
- **Source Locations**: Tracks source location information for all AST nodes
- **Modular Design**: Separate grammar, AST, and parser modules

## Structure

```
pyret-attempt/
├── src/
│   ├── pyret.pest      # Pest grammar (PEG)
│   ├── ast.rs          # AST type definitions
│   ├── parser.rs       # Parser implementation
│   ├── lib.rs          # Library root
│   └── main.rs         # Example usage
├── examples/
│   └── simple.pyret    # Example Pyret program
└── README.md
```

## Grammar Conversion

The BNF grammar from Pyret was converted to Pest's PEG format with the following key changes:

1. **Left Recursion Elimination**: PEG parsers cannot handle left recursion. Rules like:
   ```
   dot_expr: expr DOT NAME
   ```
   Were converted to:
   ```
   expr = { primary_expr ~ postfix_op* }
   postfix_op = { (DOT ~ NAME) | ... }
   ```

2. **Expression Postfix Operations**: All postfix operations (dot access, function calls, array indexing, etc.) are now parsed uniformly as a sequence of postfix operators applied to a primary expression.

3. **Annotation Postfix**: Type annotations with postfix modifiers (like `T<U>` for type application) were similarly restructured.

## AST Design

The Rust AST closely mirrors Pyret's original AST:

### Key Types

- **`Name`**: Various kinds of names (regular, global, module-scoped, type-scoped, generated atoms)
- **`Expr`**: The main expression enum with 40+ variants covering all Pyret expression forms
- **`Ann`**: Type annotations (name, arrow, record, tuple, predicate, etc.)
- **`Binding`**: Variable bindings (name or tuple destructuring)
- **`Import/Provide`**: Module system declarations

### Location Tracking

Every AST node includes a `Loc` field with:
- Source file name
- Start/end line and column
- Start/end character offset

This matches Pyret's `Srcloc` type and enables precise error reporting.

### Rust Idioms vs Pyret

The AST uses idiomatic Rust patterns:
- `Option<T>` for optional fields (instead of Pyret's `Option<T>` or nullable fields)
- `Vec<T>` for lists (instead of Pyret's `List<T>`)
- `Box<T>` for recursive types to ensure fixed size
- Enums for sum types with pattern matching

But maintains Pyret's structure:
- Same variant names (e.g., `s-fun` becomes `Expr::Fun`)
- Same field names where possible
- Same semantic structure for easy porting

## Usage

```rust
use pyret_attempt::parse_program;

fn main() {
    let source = r#"
        fun square(x):
            x * x
        end

        square(5)
    "#;

    match parse_program(source) {
        Ok(program) => {
            println!("Successfully parsed!");
            println!("Location: {:?}", program.loc);
            // Work with the AST...
        },
        Err(e) => {
            eprintln!("Parse error: {}", e);
        }
    }
}
```

## Building and Testing

```bash
# Build the project
cargo build

# Run the example
cargo run

# Run with the example file
cargo run examples/simple.pyret
```

## Grammar Features

The parser supports the full Pyret language including:

- **Functions**: `fun`, `lam` (lambda), `method`
- **Data definitions**: `data` with variants, `newtype`
- **Control flow**: `if`/`else`, `cases`, `ask`/`otherwise`, `when`
- **Bindings**: `let`, `var`, `rec`, `letrec`
- **Objects and tuples**: Object literals, tuple syntax, field access
- **Operators**: All Pyret operators with proper precedence
- **Types**: Type annotations, arrow types, record types, tuple types
- **Module system**: `import`, `include`, `provide`, `use`
- **Tables**: Table literals and table operations
- **Testing**: `check` and `where` blocks
- **Special forms**: `for` expressions, reactors, constructors

## Limitations

This is a parser implementation focusing on syntax. It does **not** include:

- Type checking
- Evaluation/interpretation
- Well-formedness checking
- Desugaring
- Code generation

These would be separate phases that operate on the AST.

## Comparison with Pyret

### Similarities
- Same AST structure and node types
- Same expression forms and semantics
- Compatible location tracking

### Differences
- Uses Rust's type system and ownership
- No runtime (parsing only)
- PEG grammar instead of BNF (different parsing algorithm)
- Some simplified implementations in the parser (e.g., full variant/member parsing is stubbed)

## Future Work

- Complete all parser implementations (some are currently simplified)
- Add comprehensive tests
- Add pretty-printing (AST -> source)
- Add visitor pattern for AST traversal
- Add well-formedness checking
- Add desugaring passes

## References

- [Pyret Language](https://www.pyret.org/)
- [Pyret Grammar](https://github.com/brownplt/pyret-lang/blob/master/src/js/base/pyret-grammar.bnf)
- [Pyret AST](https://github.com/brownplt/pyret-lang/blob/master/src/arr/trove/ast.arr)
- [Pest Parser](https://pest.rs/)

## License

This project is for educational purposes and experimentation with parsing Pyret in Rust.
