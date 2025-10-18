# LLM Usage Guide for paredit-like

This guide explains how to use `paredit-like` effectively when integrated with Large Language Models for cleaning up and refactoring Clojure code.

## Overview

`paredit-like` is designed to be LLM-friendly with:
- Simple command-line interface
- Stdout output for easy piping
- Dry-run mode for safety
- Clear error messages
- Diff output for previewing changes

## Common Workflows

### 1. Fix Malformed Lisp Code

When an LLM generates Clojure code with unbalanced parentheses:

```bash
# Always preview first
paredit-like balance generated.clj --dry-run

# Apply the fix
paredit-like balance generated.clj > fixed.clj

# Or modify in-place
paredit-like balance generated.clj --in-place
```

**Example:**
```clojure
# Input (malformed):
(defn calculate [x y]
  (let [sum (+ x y)
        diff (- x y
    {:sum sum :diff diff

# Command:
paredit-like balance input.clj

# Output (fixed):
(defn calculate [x y]
  (let [sum (+ x y)
        diff (- x y)]
    {:sum sum :diff diff}))
```

### 2. Refactor Nested Let Forms

When code has deeply nested `let` bindings:

```bash
# Target the outer let form by line number
paredit-like merge-let nested.clj --line 5 --dry-run

# Apply the refactoring
paredit-like merge-let nested.clj --line 5 --in-place
```

**Example:**
```clojure
# Input:
(defn process [data]
  (let [x (parse data)]
    (let [y (validate x)]
      (let [z (transform y)]
        (save z)))))

# After merge-let on line 2:
(defn process [data]
  (let [x (parse data)
        y (validate x)
        z (transform y)]
    (save z)))
```

### 3. Structural Code Editing

#### Slurp - Expand Forms

```bash
# Slurp the next form into the list at line 10
paredit-like slurp code.clj --line 10

# Slurp backward
paredit-like slurp code.clj --line 10 --backward
```

**Use cases:**
- Grouping related expressions
- Expanding function argument lists
- Combining sequential operations

#### Barf - Contract Forms

```bash
# Barf the last element out of the list at line 10
paredit-like barf code.clj --line 10
```

**Use cases:**
- Splitting overly large forms
- Extracting expressions from function calls
- Reducing scope of `let` bindings

#### Splice - Remove Wrapper

```bash
# Remove parens around form at line 10
paredit-like splice code.clj --line 10
```

**Use cases:**
- Removing unnecessary `do` blocks
- Flattening nested forms
- Unwrapping single-item collections

#### Wrap - Add Grouping

```bash
# Wrap form at line 10 with parens
paredit-like wrap code.clj --line 10

# Wrap with vector
paredit-like wrap code.clj --line 10 --with "["

# Wrap with map
paredit-like wrap code.clj --line 10 --with "{"
```

**Use cases:**
- Creating new collections
- Grouping expressions
- Adding function calls around values

## LLM Integration Patterns

### Pattern 1: Post-Generation Cleanup

After generating Clojure code:

```python
# Pseudo-code for LLM integration
def generate_and_fix_clojure(prompt):
    # 1. Generate code
    code = llm.generate(prompt)

    # 2. Write to temp file
    write_file("temp.clj", code)

    # 3. Balance parens
    run("paredit-like balance temp.clj --in-place")

    # 4. Read fixed code
    return read_file("temp.clj")
```

### Pattern 2: Iterative Refactoring

When applying multiple refactorings:

```python
def iterative_refactor(file_path, operations):
    for op in operations:
        # Preview change
        diff = run(f"paredit-like {op.command} {file_path} --line {op.line} --dry-run")

        # Confirm with user or LLM
        if should_apply(diff):
            run(f"paredit-like {op.command} {file_path} --line {op.line} --in-place")
```

### Pattern 3: Batch Processing

For cleaning up multiple files:

```bash
# Balance all Clojure files in src/
paredit-like batch "src/**/*.clj" --command balance --dry-run

# Apply if satisfied
for file in src/**/*.clj; do
    paredit-like balance "$file" --in-place
done
```

## Command Reference for LLMs

### Balance
```bash
paredit-like balance <file> [--in-place] [--dry-run]
```
- **When to use**: Malformed code, missing/extra parens
- **Safety**: Very safe, uses indentation as guide
- **Output**: Balanced s-expressions

### Slurp
```bash
paredit-like slurp <file> --line <n> [--backward] [--in-place] [--dry-run]
```
- **When to use**: Need to expand a form to include neighbors
- **Safety**: Safe, preserves structure
- **Direction**: `--backward` pulls previous form instead of next

### Barf
```bash
paredit-like barf <file> --line <n> [--backward] [--in-place] [--dry-run]
```
- **When to use**: Need to shrink a form
- **Safety**: Safe, preserves structure
- **Direction**: `--backward` ejects first element instead of last

### Splice
```bash
paredit-like splice <file> --line <n> [--in-place] [--dry-run]
```
- **When to use**: Remove wrapper without removing contents
- **Safety**: Safe, only removes outer parens
- **Warning**: Can break semantics (e.g., removing `do` changes meaning)

### Raise
```bash
paredit-like raise <file> --line <n> [--in-place] [--dry-run]
```
- **When to use**: Replace parent with child form
- **Safety**: Destructive, removes siblings
- **Warning**: Carefully choose which form to raise

### Wrap
```bash
paredit-like wrap <file> --line <n> [--with <paren>] [--in-place] [--dry-run]
```
- **When to use**: Add grouping around a form
- **Safety**: Very safe, only adds parens
- **Options**: `(`, `[`, or `{`

### Merge Let
```bash
paredit-like merge-let <file> --line <n> [--in-place] [--dry-run]
```
- **When to use**: Simplify nested let forms
- **Safety**: Safe for pure bindings
- **Warning**: Check for binding dependencies

## Error Handling

Common errors and solutions:

### "No list found at line X"
- **Cause**: Line doesn't contain or isn't inside a list
- **Solution**: Use different line number or different operation

### "Cannot barf from empty list"
- **Cause**: Trying to barf from a list with no elements
- **Solution**: Check list contents first

### "No form found after position X"
- **Cause**: Slurp has nothing to slurp
- **Solution**: Ensure there's a form after the target

### "Not a let form"
- **Cause**: merge-let applied to non-let form
- **Solution**: Verify the form is actually a `let`

## Best Practices

1. **Always use `--dry-run` first**: Preview changes before applying
2. **Work with version control**: Commit before batch operations
3. **One operation at a time**: Don't chain too many refactorings
4. **Test after refactoring**: Ensure semantics are preserved
5. **Use specific line numbers**: Target exact forms to avoid surprises

## Examples from Real Code

### Example 1: Fixing LLM-Generated Code

```clojure
# LLM generated (broken):
(defn process-users [users]
  (map (fn [user]
         (let [name (:name user)
               email (:email user
           {:display-name name
            :contact email)
       users

# Fix with balance:
$ paredit-like balance code.clj

# Result:
(defn process-users [users]
  (map (fn [user]
         (let [name (:name user)
               email (:email user)]
           {:display-name name
            :contact email}))
       users))
```

### Example 2: Simplifying Nested Logic

```clojure
# Original (nested lets):
(defn calculate [x y z]
  (let [a (+ x y)]
    (let [b (* a z)]
      (let [c (/ b 2)]
        c))))

# Merge lets:
$ paredit-like merge-let code.clj --line 2

# Result:
(defn calculate [x y z]
  (let [a (+ x y)
        b (* a z)
        c (/ b 2)]
    c))
```

### Example 3: Restructuring Function Calls

```clojure
# Original:
(println "Result:") (calculate x y)

# Wrap println in do:
$ paredit-like wrap code.clj --line 1
# Then slurp the calculate call:
$ paredit-like slurp code.clj --line 1

# Result:
(do (println "Result:") (calculate x y))
```

## Integration with Your LLM System

### Environment Variables

```bash
export PAREDIT_DEFAULT_INDENT=2  # Not implemented yet
export PAREDIT_PRESERVE_COMMENTS=1  # Not implemented yet
```

### Exit Codes

- `0`: Success
- `1`: Error (invalid arguments, file not found, parse error, etc.)

### Output Formats

**Stdout (default)**: Modified source code
**Stderr**: Status messages (e.g., "Modified file.clj")
**Dry-run**: Diff format showing changes

### Piping

```bash
# Pipe to another tool
paredit-like balance input.clj | clj-kondo --lint -

# Pipe from stdin (not currently supported)
# echo "(+ 1 2" | paredit-like balance -
```

## Limitations

Current limitations to be aware of:

1. **No stdin support**: Must use files
2. **Line-based targeting**: Operations target whole lines, not specific columns
3. **Single-file operations**: Batch is limited to same operation on all files
4. **No multi-file refactoring**: Can't rename across files
5. **Limited semantic understanding**: Works on structure, not meaning

## Future Enhancements

Planned features for better LLM integration:

- [ ] JSON output mode with structured change information
- [ ] Column-based targeting for precision
- [ ] Stdin/stdout piping support
- [ ] LSP server mode for editor integration
- [ ] Configuration files for project-specific rules
- [ ] Custom refactoring patterns
- [ ] Multi-file rename support
