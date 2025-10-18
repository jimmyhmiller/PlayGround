# paredit-like

A command-line tool for structured editing and refactoring of Clojure s-expressions.

## Features

- **Parinfer-style balancing**: Automatically balance parentheses based on indentation
- **Structural editing**: Slurp, barf, splice, raise, and wrap operations
- **Advanced refactorings**: Merge nested `let` forms
- **Dry-run mode**: Preview changes before applying them
- **Batch processing**: Apply refactorings to multiple files at once
- **LLM-friendly**: Designed for easy integration with AI code tools

## Installation

```bash
cargo install --path .
```

## Usage

### Balance Parentheses

Fix malformed Clojure code by balancing parentheses based on indentation:

```bash
# Print balanced code to stdout
paredit-like balance myfile.clj

# Modify file in-place
paredit-like balance myfile.clj --in-place

# Preview changes without modifying
paredit-like balance myfile.clj --dry-run
```

**Example:**

Input:
```clojure
(defn foo
  (+ 1 2
```

Output:
```clojure
(defn foo
  (+ 1 2))
```

### Slurp

Pull the next form into the current list:

```bash
# Slurp forward at line 5
paredit-like slurp myfile.clj --line 5

# Slurp backward
paredit-like slurp myfile.clj --line 5 --backward
```

**Example (forward):**

Before:
```clojure
(foo bar) baz
```

After:
```clojure
(foo bar baz)
```

### Barf

Push the last form out of the current list:

```bash
# Barf forward at line 5
paredit-like barf myfile.clj --line 5

# Barf backward
paredit-like barf myfile.clj --line 5 --backward
```

**Example (forward):**

Before:
```clojure
(foo bar baz)
```

After:
```clojure
(foo bar) baz
```

### Splice

Remove parentheses around a form while keeping its children:

```bash
paredit-like splice myfile.clj --line 5
```

**Example:**

Before:
```clojure
(foo (bar baz) qux)
```

After:
```clojure
(foo bar baz qux)
```

### Raise

Replace the parent form with the current form:

```bash
paredit-like raise myfile.clj --line 5
```

**Example:**

Before:
```clojure
(foo (bar baz))
```

After (with cursor on `bar`):
```clojure
(bar baz)
```

### Wrap

Wrap the current form with parentheses, brackets, or braces:

```bash
# Wrap with parentheses
paredit-like wrap myfile.clj --line 5

# Wrap with brackets
paredit-like wrap myfile.clj --line 5 --with "["

# Wrap with braces
paredit-like wrap myfile.clj --line 5 --with "{"
```

**Example:**

Before:
```clojure
foo bar
```

After (wrapping `foo`):
```clojure
(foo) bar
```

### Merge Let

Merge nested `let` forms into a single `let`:

```bash
paredit-like merge-let myfile.clj --line 5
```

**Example:**

Before:
```clojure
(let [x 1]
  (let [y 2]
    (+ x y)))
```

After:
```clojure
(let [x 1 y 2]
  (+ x y))
```

### Batch Processing

Apply refactorings to multiple files matching a glob pattern:

```bash
paredit-like batch "src/**/*.clj" --command balance --dry-run
```

## LLM Integration Tips

This tool is designed to be easily used by LLMs for cleaning up Clojure code:

1. **Use `--dry-run` first**: Always preview changes before applying them
2. **Specific line numbers**: Provide exact line numbers for targeted refactorings
3. **Balance as post-processing**: Run `balance` after making manual edits to fix parentheses
4. **Stdout output**: By default, output goes to stdout for easy piping

Example LLM workflow:

```bash
# 1. Check what balance would do
paredit-like balance messy.clj --dry-run

# 2. Apply balance
paredit-like balance messy.clj > fixed.clj

# 3. Apply specific refactoring
paredit-like merge-let fixed.clj --line 10 --in-place
```

## Test Cases

### Balance (Parinfer-style)

| Input | Output | Description |
|-------|--------|-------------|
| `(defn foo\n  (+ 1 2` | `(defn foo\n  (+ 1 2))` | Add missing closing parens |
| `(defn foo []))` | `(defn foo [])` | Remove extra closing parens |
| `(let [x 1\n      y 2\n  (+ x y` | `(let [x 1\n      y 2]\n  (+ x y))` | Balance based on indentation |

### Structural Editing

| Operation | Before | After |
|-----------|--------|-------|
| Slurp forward | `(foo bar) baz` | `(foo bar baz)` |
| Slurp backward | `foo (bar baz)` | `(foo bar baz)` |
| Barf forward | `(foo bar baz)` | `(foo bar) baz` |
| Barf backward | `(foo bar baz)` | `foo (bar baz)` |
| Splice | `(foo (bar baz) qux)` | `(foo bar baz qux)` |
| Raise | `(foo (bar baz))` | `(bar baz)` |
| Wrap | `foo bar` | `(foo) bar` |

### Advanced Refactorings

| Operation | Before | After |
|-----------|--------|-------|
| Merge let | `(let [x 1] (let [y 2] body))` | `(let [x 1 y 2] body)` |

## Architecture

- **Parser**: Uses tree-sitter for robust parsing of Clojure code (handles malformed input)
- **AST**: Custom s-expression representation with span information
- **Parinfer**: Indentation-based parenthesis balancing
- **Refactorings**: Offset-based string manipulation (preserves comments and formatting)
- **CLI**: Clap-based subcommand interface

## Development

Run tests:

```bash
cargo test
```

Run with debug output:

```bash
RUST_LOG=debug cargo run -- balance test.clj
```

## License

MIT
