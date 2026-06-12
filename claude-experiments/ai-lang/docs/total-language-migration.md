# Fully-total language migration (in progress)

Jimmy's decision (2026-06-11): the language has NO divergence construct.

- `abort()` builtin: REMOVED from the surface language. (Runtime-internal
  Rust-side traps for compiler/runtime bugs — type confusion, heap
  exhaustion, non-exhaustive-match trap — remain; they are not API.)
- `*_trusted` accessors: REMOVED. One indexing API: checked, Result.
- Provably-impossible `Err` arms THREAD OUT: public stdlib signatures
  change (e.g. `smap_get -> Result<Option<T>, IndexError>`). No
  sentinels, ever.
- Uninitialized non-scalar slot reads: `Err(IndexError::Uninitialized(OobInfo))`.
  Prim-repr (scalar) arrays are zero-filled, so scalar reads return 0/0.0;
  boxed-repr uninit reads are the Err. SETs need no init check.
- Bytes are zero-filled: no Uninitialized case for bytes.

## Steps

1. [ ] stdlib types: add `Uninitialized(OobInfo)` variant to IndexError.
2. [ ] codegen: inline emitter for new internal builtin
       `core/array.is_init(a, i) -> Int` (prim → 1; boxed → slot != null;
       bounds are pre-checked by the expansion).
3. [ ] resolve: expansion gains the is_init branch for array GET
       (not set, not bytes); remove `abort` + `*_trusted` surface
       mappings (clear errors pointing at the checked API); typecheck
       drops core/abort signature.
4. [ ] stdlib migration (~120 accessor sites; signature cascade):
       HAMT/smap/imap, base64/crc32/zip, wire/http/json, svc_* (abort →
       Result), string/bytes helpers. Compile-error-driven via
       `add`-ing the stdlib.
5. [ ] Test snippets in stdlib.rs / codegen.rs / net.rs / evalrun.rs.
6. [ ] Examples (abort_demo deleted; array_demo/bytes_demo drop trusted
       mentions) + benchmarks unchanged (nbody already checked).
7. [ ] Full suite + benchmarks + README + memory + commit.

Perf note: the is_init branch must be INLINE or nbody regresses ~170ms.
