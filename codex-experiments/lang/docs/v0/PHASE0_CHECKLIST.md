# Phase 0 Checklist (Draft)

Phase 0 goal: a minimal bootstrap compiler in Rust that can parse, typecheck, and AOT compile a small subset of the language to native code using the GC runtime.

## 1. Language Surface (Must Have)

- [ ] `fn` declarations with explicit param/return types
- [ ] `let` bindings with optional type annotation
- [ ] `if` / `else`
- [ ] `while`
- [ ] `match` on enums
- [ ] `struct` and `enum`
- [ ] `RawPointer<T>`
- [ ] `extern fn` declarations (C ABI, optional varargs)

## 2. Parser and AST

- [ ] Lexer with tokens defined in `GRAMMAR.md`
- [ ] Pratt or precedence-based expression parser
- [ ] Error recovery nodes (do not stop on first error)
- [ ] AST includes locations (file, span)

## 3. Name Resolution

- [ ] Module path validation (`module a::b::c`)
- [ ] `use` imports
- [ ] Visibility rules (`pub` vs private)
- [ ] Stable item IDs

## 4. Typechecking

- [ ] Bidirectional typing
- [ ] Literal typing rules (integers/floats/bools/strings)
- [ ] Struct/enum construction and field access
- [ ] Function calls + extern calls
- [ ] Minimal trait rule enforcement (or explicitly disabled for phase 0)

## 5. Lowering and IR

- [ ] AST -> HIR lowering
- [ ] HIR contains only resolved names
- [ ] Typed IR (TIR) for codegen

## 6. Codegen

- [ ] Inkwell backend
- [ ] C ABI for generated functions
- [ ] Implicit `Thread*` arg0 for language functions
- [ ] GC allocation + safepoints (from runtime ABI)
- [ ] Correct root slot tracking

## 7. Runtime

- [ ] GC allocation entrypoints (`gc_allocate`, `gc_allocate_array`)
- [ ] Pollcheck slow path (`gc_pollcheck_slow`)
- [ ] Minimal I/O (`print_int`, `print_str`)
- [ ] Abort/panic (`abort`)

## 8. Build + Tooling

- [ ] CLI: `langc build main.lang`
- [ ] AOT pipeline: emit object file + link
- [ ] Clear error output

## 9. Smoke Tests

- [ ] `hello` program
- [ ] simple struct + function
- [ ] enum + match
- [ ] extern call (e.g., `printf`)
- [ ] GC allocation test (list or tree)

