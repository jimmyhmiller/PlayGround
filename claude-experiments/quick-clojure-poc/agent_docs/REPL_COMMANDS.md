# REPL Commands Reference

## Code Execution
- `(+ 1 2)` - Execute Clojure expressions
- `(def x 42)` - Define variables
- `(ns foo)` - Switch to namespace

## Debugging Commands
- `:ast (+ 1 2)` - Show AST for expression
- `:ir (+ 1 2)` - Show IR (intermediate representation)
- `:asm (+ 1 2)` - Show ARM64 machine code

## GC & Heap Inspection
- `:heap` - Show heap statistics and all allocated objects
- `:gc` - Manually trigger garbage collection
- `:namespaces` - List all namespaces with addresses and binding counts
- `:inspect <namespace>` - Show all bindings in a namespace

## Other
- `:help` - Show help
- `:quit` / `:exit` - Exit REPL

## Example Session

```clojure
user=> :heap
; Shows initial heap with just clojure.core and user namespaces

user=> (def x 42)
42
user=> :inspect user
; Shows x = 42 in user namespace

user=> :namespaces
; Lists all namespaces

user=> (ns math)
0
math=> (def pi 314)
314
math=> :inspect math
; Shows pi = 314 in math namespace

math=> user/x
42
math=> (+ user/x pi)
356

math=> :heap
; Shows all heap objects including namespace objects that grew as we added bindings

math=> :gc
✓ Garbage collection completed
```

## What You Can See

### Namespace Growth
Watch namespace objects grow as you add bindings:
- Empty namespace: 16 bytes
- With 1 binding: 32 bytes
- With 2 bindings: 48 bytes
- With 3 bindings: 64 bytes

Each binding adds 16 bytes (8 bytes for symbol name pointer + 8 bytes for value).

### Heap Layout
Every object shows:
- **Address**: Actual memory address on the heap
- **Type**: String or Namespace
- **Size**: Size in bytes
- **Marked**: GC mark bit (for mark-and-sweep)
- **Name**: The string content or namespace name

### Tagged Values
When you `:inspect` a namespace, values are shown in two forms:
- **Tagged**: `0x00000150` - The actual bits stored (value << 3)
- **Untagged**: `42` - The actual numeric value

The 3 low bits are used for type tags (000 = integer).
