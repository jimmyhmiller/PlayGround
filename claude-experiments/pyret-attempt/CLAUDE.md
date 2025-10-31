# Pyret Parser Project

This is a Pyret programming language parser implementation written in Rust using the Pest parsing library.

## What is Pyret?

Pyret is a programming language designed for education, with features for writing and testing programs. It uses `.arr` file extensions for source files.

## Project Structure

- **src/parser.rs** - Pest-based parser using grammar rules
- **src/ast.rs** - Abstract Syntax Tree definitions for Pyret programs
- **src/lib.rs** - Library exports
- **src/main.rs** - CLI for parsing individual .arr files

## Building and Running

```bash
# Build the parser
cargo build --release

# Parse a single file
./target/release/pyret-attempt path/to/file.arr

# Parse all .arr files in pyret-lang directory
bash parse_all.sh
```

## Testing

The parser is being tested against the official Pyret language repository at:
`/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang`

This repository contains 527+ `.arr` files that serve as a comprehensive test suite.

## Parse Results

When running `parse_all.sh`, results are saved to `./parse_results/`:
- **parse_log.txt** - Detailed output for each file
- **successful.txt** - List of successfully parsed files
- **failed.txt** - List of files that failed to parse
- **summary.txt** - Overall statistics

## Current Status

This is an experimental parser implementation for learning about Pyret's syntax and building parsing tools. The parser uses Pest grammar rules to handle Pyret's syntax including:

- Function definitions
- Data type declarations
- Import/provide statements
- Where clauses for testing
- Various expression types
- Type annotations
