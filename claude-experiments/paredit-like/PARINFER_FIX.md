# Parinfer Fix for LLM-Generated Brace Indentation

## Problem

LLMs often generate Lisp code with brace structures like this:

```lisp
         (attributes {
           :sym_name @test
           :function_type (!function (inputs) (results i32))
         })
```

The original `parinfer-rust` implementation was incorrectly balancing this to:

```lisp
         (attributes {}
           :sym_name @test
           :function_type (!function (inputs) (results i32)))
```

This happened because parinfer-rust's indentation algorithm was closing braces too eagerly when content inside was only indented by 2 spaces relative to the opening line.

## Solution

Created a new robust parinfer implementation in `src/parinfer_simple.rs` that correctly handles indentation-based balancing for all reasonable Clojure/Lisp code styles.

### Key Algorithm Changes

The new implementation uses a simpler, more predictable indentation rule:

**Close delimiters when:** `next_indent < current_indent AND delimiter_opened_at_indent > next_indent`

This means:
1. Only close delimiters when indentation decreases to the next line
2. Close delimiters that were opened at indentation levels *greater than* the next line's indent (not greater-or-equal)

### Why This is Robust

1. **Preserves well-formed code**: If code is already balanced correctly, it stays unchanged
2. **Handles all delimiter types**: Parentheses `()`, brackets `[]`, and braces `{}` all work consistently
3. **Respects strings and comments**: Delimiters inside strings or comments are ignored
4. **Clojure-compatible**: Works with defn, let, maps, vectors, threading macros, cond, etc.
5. **LLM-friendly**: Handles the 2-space relative indentation that LLMs commonly generate
6. **Auto-balancing**: Automatically closes missing delimiters based on indentation
7. **Removes extras**: Silently removes mismatched or extra closing delimiters

### Test Coverage

The implementation includes 19 comprehensive tests covering:
- Basic brace/bracket/paren balancing
- Nested structures
- Missing delimiters (auto-close)
- Extra delimiters (auto-remove)
- Strings with delimiters
- Comments with delimiters
- Clojure `defn`, `let`, maps, vectors
- Threading macros
- Complex nested structures
- LLM-style indentation patterns

All 114 tests in the full test suite pass.

### Files Changed

- `src/parinfer_simple.rs` - New robust parinfer implementation
- `src/lib.rs` - Export new module
- `src/main.rs` - Use new parinfer for balance command

### Migration Path

The original `parinfer-rust` integration is still available in `src/parinfer.rs` if needed. To switch back:

```rust
// In src/main.rs, change:
let parinfer = parinfer_simple::Parinfer::new(&source);
// back to:
let parinfer = parinfer::Parinfer::new(&source);
```

## Usage

```bash
# Balance LLM-generated code
paredit-like balance mlir_code.lisp --in-place

# Preview changes
paredit-like balance mlir_code.lisp --dry-run

# See diff
paredit-like balance mlir_code.lisp --diff
```

## Performance

The new implementation is lightweight and fast:
- Single-pass algorithm
- O(n) time complexity where n = input length
- Minimal memory overhead (only tracks delimiter stack)
- No external dependencies

## Future Considerations

If you need even more sophisticated balancing (e.g., structural editing with syntax awareness), consider:
1. Integrating with the existing tree-sitter parser for full AST-aware balancing
2. Adding configurable indentation rules
3. Supporting more language-specific features (reader conditionals, etc.)

For now, this implementation provides a robust, predictable balance command that works well with LLM-generated code.
