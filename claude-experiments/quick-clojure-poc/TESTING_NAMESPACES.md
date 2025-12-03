# Testing GC-Allocated Namespaces

This guide shows how to test the new heap-allocated namespace implementation in the REPL.

## Quick Start

```bash
cargo run
```

Then try these commands interactively or paste from `namespace_demo.txt`.

## What to Look For

### 1. **Namespace Switching**
Watch the prompt change as you switch namespaces:
```clojure
user=> (ns math)
0
math=> (ns physics)
0
physics=> (ns user)
0
user=>
```

### 2. **Per-Namespace Variables**
Each namespace has its own variable bindings stored on the heap:
```clojure
user=> (def x 42)
42
user=> (ns other)
0
other=> (def x 100)
100
other=> x
100
other=> user/x
42
```

### 3. **Qualified Symbol Access**
Access variables from other namespaces:
```clojure
user=> (def pi 314)
314
user=> (ns math)
0
math=> user/pi
314
```

### 4. **Division Bug Fix**
The "/" operator now works correctly (it was being parsed as a namespace separator):
```clojure
user=> (/ 100 10)
10
user=> (def x 50)
50
user=> (/ x 5)
10
```

## Implementation Details You're Testing

When you run these commands, here's what happens under the hood:

1. **Namespace Creation** - `(ns foo)` allocates a namespace object on the heap with type_id=10
2. **Variable Definition** - `(def x 42)` adds a binding to the current namespace (may reallocate the namespace object)
3. **Variable Lookup** - `x` searches the current namespace's bindings using linear scan
4. **Qualified Lookup** - `user/x` looks up the "user" namespace pointer, then searches its bindings
5. **GC Roots** - All namespaces are registered as GC roots so they won't be collected

## Running the Full Demo

Basic namespace demo:
```bash
cat namespace_demo.txt | cargo run
```

GC and heap inspection demo:
```bash
cat gc_inspection_demo.txt | cargo run
```

Or run it step by step interactively:
```bash
cargo run
# Then paste commands from the demo files
```

## New Inspection Commands

### `:heap` - Show Heap Statistics
Shows all objects allocated on the heap with their addresses, types, sizes, and GC mark status:
```
user=> :heap

╔════════════════════ Heap Statistics ════════════════════╗
║ Heap Size:        1048576 bytes (1024.0 KB)              ║
║ Used:                 184 bytes (0.2 KB)              ║
║ Free:             1048392 bytes (1023.8 KB)              ║
║ Objects:                8                               ║
║ Namespaces:             2                               ║
╚═════════════════════════════════════════════════════════╝

  Address      Type         Size  Marked  Name
  ──────────────────────────────────────────────────────
  0xb25400000  String         24b          clojure.core
  0xb25400018  Namespace      16b          clojure.core
  0xb25400028  String         16b          user
  0xb25400038  Namespace      16b          user
  0xb25400048  String         16b          x
  0xb25400058  Namespace      32b          user
  0xb25400078  String         16b          y
  0xb25400088  Namespace      48b          user
```

Notice: The `user` namespace grows from 16b → 32b → 48b as we add bindings!

### `:namespaces` - List All Namespaces
Shows all namespaces with their heap addresses and binding counts:
```
user=> :namespaces

╔════════════════════ Namespaces ════════════════════╗
  Name                     Pointer      Bindings
  ─────────────────────────────────────────────────
  clojure.core          0x592a0000c6      0
  math                  0x592a0008c6      2
  user                  0x592a000446      2
╚════════════════════════════════════════════════════╝
```

### `:inspect <namespace>` - Inspect Namespace Bindings
Shows all variable bindings in a namespace with their tagged and untagged values:
```
user=> :inspect user

╔════════════ Namespace: user ════════════╗
  Symbol                   Tagged Value    Untagged
  ──────────────────────────────────────────────────
  x                     0x00000150      42
  y                     0x00000320      100
╚═══════════════════════════════════════════════════╝
```

The tagged value shows the actual bits stored (shifted left 3 for the tag bits).

### `:gc` - Run Garbage Collection
Manually triggers a garbage collection cycle:
```
user=> :gc
✓ Garbage collection completed
```

Currently all namespaces are GC roots, so nothing gets collected. But you can see the mark-and-sweep algorithm running!

## Verifying the Tests

All tests should pass:
```bash
cargo test
```

You should see:
- 26 lib tests passing (including 3 new GC runtime tests)
- 12 integration tests passing
- 0 failures

## What's Different From Before?

**Before (HashMap-based):**
- Namespaces were Rust `HashMap<String, isize>` in the compiler
- Not garbage collected
- Not first-class objects

**After (GC-based):**
- Namespaces are heap-allocated objects with type_id=10
- Garbage collected when unreferenced
- True first-class objects that can be introspected
- Foundation for future features like namespace metadata

## Memory Layout

Each namespace object on the heap looks like:
```
┌─────────────────────────────────┐
│ Header (8 bytes)                │
│  - type_id: 10 (Namespace)      │
│  - size: 1 + (num_bindings * 2) │
└─────────────────────────────────┘
│ Field 0: name (String pointer)  │ ──→ "user"
├─────────────────────────────────┤
│ Field 1: symbol_0_name          │ ──→ String pointer to "x"
│ Field 2: symbol_0_value         │ ──→ Tagged int: 42
├─────────────────────────────────┤
│ Field 3: symbol_1_name          │ ──→ String pointer to "y"
│ Field 4: symbol_1_value         │ ──→ Tagged int: 100
└─────────────────────────────────┘
```

Enjoy testing!
