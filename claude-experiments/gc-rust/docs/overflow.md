# Integer overflow semantics (frozen v1)

**Integer arithmetic wraps on overflow** (two's complement), and this is a
*defined, guaranteed* behavior — never undefined.

```rust
let big = 9223372036854775807;  // i64::MAX
big + 1                          // == -9223372036854775808 (i64::MIN), defined
```

## Why wrapping

gc-rust competes on performance — the thesis is matching or beating Rust on hot
numeric loops. Wrapping is **zero-cost**: it adds no per-operation branch. It
also matches every fast managed/systems language that competes on speed:

| Language | Overflow |
|---|---|
| Java | wraps (two's complement, mandated by the JVM spec) |
| Go | wraps at runtime (constant overflow is a compile error) |
| OCaml | wraps (native 63-bit int) |
| C (unsigned) | wraps |
| **gc-rust** | **wraps** |

Trapping languages (Swift, Rust debug builds) pay a branch on every `+`/`*`,
which directly taxes the loops gc-rust is built to win. We chose not to.

## Implementation

Codegen emits LLVM `add`/`sub`/`mul` with **no `nsw`/`nuw` flags**, which is
defined two's-complement wrapping. The `*_nsw`/`*_nuw` variants must never be
used — they make overflow undefined behavior and would break this guarantee.
See the comment in `codegen.rs::gen_bin`.

Division and remainder are the usual signed/unsigned LLVM ops; division by zero
and `INT_MIN / -1` follow LLVM/hardware semantics (a hardware trap), as in C and
Go — gc-rust does not insert software checks for these in v1.

## Opt-in checked arithmetic

For the rare case where overflow must be detected, the prelude provides checked
helpers that return `Option`:

```rust
checked_add_i64(a, b) -> Option<i64>   // None on overflow
checked_mul_i64(a, b) -> Option<i64>
```

These are ordinary library functions (a branch only where you ask for it), not a
language-wide cost. Wrapping helpers (`wrapping_add_i64`, etc.) are also provided
for symmetry / explicitness, though plain `+` already wraps.
