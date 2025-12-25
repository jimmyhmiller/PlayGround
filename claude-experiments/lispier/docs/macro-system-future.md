# Macro System: Future Unification with `require`

## Current Design

The macro system currently uses two separate mechanisms:

- **`(require ["./file.lisp" :as alias])`** - Loads runtime code from another module
- **`(require-macros "./file.lisp")`** - Loads and compiles macros from another module

This separation exists because macros need to be compiled and available *before* macro expansion happens, while `require` is processed after parsing.

## Why Two Mechanisms?

The compilation pipeline is:

```
read Values → expand macros → parse to AST → generate IR → compile
                  ↑
                  macros must be registered here
```

Currently:
- `require-macros` is detected at the Value level (before expansion)
- `require` is detected at the AST level (after parsing)

This timing difference is why they're separate.

## Future: Unified `require`

We could unify these into a single `require` that handles both:

### Approach 1: Scan Before Expansion

Modify the module loading flow:

1. **Pre-scan phase**: Before any expansion, scan the source for ALL `require` forms at the Value level
2. **Recursive load**: Load all required modules (depth-first)
3. **Detect macros**: For each loaded module, check if it has `defmacro` declarations
4. **Compile macros**: JIT-compile any modules with `defmacro` and register the macros
5. **Expand**: Now expand with all macros available
6. **Parse**: Convert expanded Values to AST

### Implementation Changes Required

**In `module_loader.rs`:**

```rust
fn parse_source(&mut self, source: &str, path: &PathBuf, current_dir: &Path) -> Result<Vec<Node>, ModuleError> {
    // 1. Tokenize and read
    let values = self.read_source(source, path)?;

    // 2. Extract ALL require paths (not just require-macros)
    let require_paths = self.extract_require_paths(&values);

    // 3. Pre-load and compile any modules that have defmacro
    let registry = self.preload_macro_modules(&require_paths, current_dir, path)?;

    // 4. Expand with full registry
    let expander = MacroExpander::with_registry(registry);
    let expanded = expander.expand_all(&values)?;

    // 5. Parse
    Parser::new().parse(&expanded)
}

fn preload_macro_modules(&mut self, paths: &[String], ...) -> Result<MacroRegistry, ModuleError> {
    let mut registry = MacroRegistry::new();

    for path in paths {
        let resolved = self.resolve_path(path, current_dir)?;
        let source = fs::read_to_string(&resolved)?;

        // Quick scan: does this module have defmacro?
        if self.has_defmacro(&source) {
            // Compile and register
            let jit = self.macro_compiler.compile_and_register(&source, &mut registry)?;
            self.macro_jits.insert(resolved, jit);
        }
    }

    Ok(registry)
}

fn has_defmacro(&self, source: &str) -> bool {
    // Simple string scan or parse check
    source.contains("(defmacro ")
}
```

### Approach 2: Two-Phase Loading

Alternative: load everything twice - once for macros, once for runtime:

1. **Phase 1 (compile-time)**: Load all modules, compile those with `defmacro`, register macros
2. **Phase 2 (runtime)**: Load again with macros available, generate final output

This is similar to how Racket handles `require` with `for-syntax`.

### Considerations

**Pros of unification:**
- Simpler user experience - just use `require`
- No need to remember which form to use
- Matches Common Lisp behavior

**Cons of unification:**
- Implicit behavior - not obvious which modules get JIT-compiled
- Potential for slower loads (scanning all requires for defmacro)
- Circular dependency handling becomes more complex

**Keeping them separate (current design):**
- Explicit control over what gets compiled at macro-expansion time
- Matches ClojureScript's model
- Clearer mental model: compile-time vs runtime dependencies

## Migration Path

If we decide to unify:

1. Keep `require-macros` working for backwards compatibility
2. Add the pre-scan logic to detect `defmacro` in required modules
3. Deprecate `require-macros` with a warning
4. Eventually remove `require-macros` parsing

## Related: Macro-Defining Macros

A future consideration: if a macro module itself uses macros (macro-defining macros), we'd need recursive macro loading. The current system doesn't handle this - macro modules are compiled with only built-in macros. Unifying `require` would make this more natural to support.
