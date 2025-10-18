## paredit-like - Structured Editing for Clojure

This project has access to `paredit-like`, a command-line tool for structural editing
of Clojure s-expressions. Use this tool when you need to refactor Clojure code while
maintaining proper parenthesis balance and structure.

### When to Use

- **Balance parentheses**: Automatically fix missing or extra parentheses based on indentation
- **Slurp/Barf**: Move forms in/out of lists (paredit-style operations)
- **Structural refactoring**: Splice, raise, wrap, or merge forms
- **Batch operations**: Apply refactorings to multiple files at once

### Common Commands

```bash
# Balance parentheses in a file
paredit-like balance src/core.clj --in-place

# Preview changes before applying
paredit-like balance src/core.clj --dry-run

# Slurp next form into list at line 5
paredit-like slurp src/core.clj --line 5 -i

# Barf last form out of list at line 10
paredit-like barf src/core.clj --line 10 -i

# Wrap form at line 3 with square brackets
paredit-like wrap src/core.clj --line 3 --with "[" -i

# Merge nested let forms
paredit-like merge-let src/core.clj --line 5 -i
```

### Getting Help

Run `paredit-like --help` to see all available commands and options.
Run `paredit-like <command> --help` for detailed help on specific commands.

### Available Operations

- `balance` - Fix parentheses using indentation (Parinfer-style)
- `slurp` - Move next/previous form into current list
- `barf` - Move last/first form out of current list
- `splice` - Remove parentheses, keep children
- `raise` - Replace parent with current form
- `wrap` - Surround form with parens/brackets/braces
- `merge-let` - Flatten nested let forms
- `batch` - Apply operations to multiple files

