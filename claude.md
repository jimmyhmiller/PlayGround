## paredit-like - Structured Editing for Clojure and Lisp S-expressions

This project has access to `paredit-like`, a command-line tool for structural editing
of Clojure and Lisp s-expressions. **IMPORTANT: Always use this tool when working with
s-expression files (.lisp, .clj, etc.) to fix parenthesis issues.** Do not manually
count or add parentheses.

### When to Use

- **Balance parentheses**: Automatically fix missing or extra parentheses based on indentation (use this for ALL paren issues!)
- **After editing s-expressions**: Run `paredit-like balance <file> --in-place` after making changes to lisp files
- **Before testing**: Balance files before running to avoid parsing errors
- **Slurp/Barf**: Move forms in/out of lists (paredit-style operations)
- **Structural refactoring**: Splice, raise, wrap, or merge forms
- **Batch operations**: Apply refactorings to multiple files at once

### Common Commands

```bash
# Balance parentheses in a file (MOST IMPORTANT - use this after editing s-expressions!)
paredit-like balance src/core.clj --in-place
paredit-like balance examples/mlsp_dialect.lisp --in-place

# Preview changes before applying (useful to verify fixes)
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

### Important Notes

- **Always run `paredit-like balance <file> --in-place` after editing .lisp files**
- The `balance` command uses indentation to determine correct parenthesis structure
- Make sure your indentation is correct before running balance
- Works with all s-expression-based files (.lisp, .clj, .el, etc.)

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

