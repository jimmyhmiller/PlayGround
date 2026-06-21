# Phase-2 dogfood #4 — friction report (C interop)

Written after `examples/cinterop.coil` — a program that calls real **libc/libm**
across the C ABI (the FFI built in #5), to battle-test the systems-integration heart
of "as low as Zig/C": *living in the C ecosystem*. It exercises four distinct C-ABI
paths at once:

- **qsort with a Coil COMPARATOR CALLBACK** — C calls *back into* Coil. A Coil
  function (`i64-cmp`) is handed to `qsort` via `fnptr-of`, and libc invokes it as the
  comparator. This is the headline: Coil functions are callable *from* C.
- **libm `sqrt`/`pow`** — the float C ABI (`f64` args + return).
- **libc `div`** — a `div_t` struct returned **by value** across the ABI.
- **libc `strtol`** — a `(ptr i8)`/`c"…"` argument.

## Headline: Coil lives in the C ecosystem — the FFI holds at scale

All four worked together (the program returns 42 iff qsort's callback sorted
correctly *and* sqrt/pow/strtol/div all agree on 42):

- **The FFI CALLBACK is the validation that matters.** `(fnptr-of i64-cmp)` produced a
  C-ABI function pointer that libc's `qsort` called back into — Coil code running as a
  C callback, correctly receiving `const void*` args and returning `int`. A low-level
  language's value is largely in this; it works.
- **Float ABI, struct-by-value return, and `char*`/cstring args** all composed with no
  fuss — `sqrt(1764)=42`, `div(84,2).quot=42`, `strtol(c"42")=42`. The #5 C ABI
  (verified vs clang then) holds under a real mixed-call program.
- `c"…"` cstrings are exactly the right tool at the FFI boundary (strtol's arg),
  vindicating the distinct-cstring decision from #5.

## Friction surfaced

1. **No `void` return type for externs.** `qsort` returns `void`; Coil has no void/unit
   return, so it's declared `(-> i32)` and the (garbage) result ignored. The CALL
   sequence is identical (args passed correctly, qsort runs), so this is harmless in
   practice — but it's imprecise and a real gap: a true `void`/unit return type (or a
   `(-> )`/`(-> void)` spelling that emits an LLVM `void` return and forbids using the
   result) would make void C functions first-class. Small, core-appropriate (it's a
   type-system primitive, not a macro-able feature) — the next candidate if C-interop
   ergonomics matter.

(No other friction — `extern` with a `fnptr` param, `fnptr-of` a Coil fn for a C
callback, float/struct/cstring marshalling all worked.)

## Verdict
Coil is a systems language that lives in the C ecosystem: it calls libc/libm, marshals
floats/structs/strings across the ABI, and — the key one — is *callable from C* (the
qsort callback). One small gap (no `void` return). The "as low as Zig/C" claim now has
real FFI behind it, dogfooded end-to-end.
