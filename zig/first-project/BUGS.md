# Bugs

This file tracks bugs discovered during development.

## Unhelpful error message: 'ERROR writing return expression: UnsupportedExpression' [fortunate-sociable-mastodon]

**ID:** fortunate-sociable-mastodon
**Timestamp:** 2025-10-18 10:31:59
**Severity:** high
**Location:** src/simple_c_compiler.zig (Error reporting in code generation)
**Tags:** error-messages, diagnostics, codegen, ux

### Description

When code generation fails for certain expressions, the compiler outputs 'ERROR writing return expression: UnsupportedExpression' with no details about WHICH expression failed, WHERE in the source file, or WHY it's unsupported. This makes debugging nearly impossible for users. The error should include: 1) The source location (file:line), 2) The actual expression that failed, 3) What about it is unsupported, 4) Ideally a hint about how to work around it.

### Minimal Reproducing Case

Any code that triggers a codegen failure (e.g., deallocate in nested contexts) shows this unhelpful error

### Code Snippet

```
Current: 'ERROR writing return expression: UnsupportedExpression'
Should be: 'ERROR at example.lisp:42: Cannot generate code for (deallocate ptr) in this context. Hint: deallocate may not work in all expression positions.'
```

---

