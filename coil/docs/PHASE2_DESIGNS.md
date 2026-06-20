# Phase-2 design proposals: aggregate reference model + string type

DESIGN ONLY — for Leader + jimmyhmiller review before any implementation. These two
define Coil's identity (how aggregates are passed; what a string is), so they get a
design pass rather than a unilateral build. Each lists the goal, a recommended
design, the alternatives, and the decisions that need a ruling.

---

## #4 — Aggregate reference model

> **SHIPPED (conservative model, Jimmy's ruling).** Implemented in five commits:
> #4a rvalue-auto-spill at by-immutable-ref args (`Expr::SpillRef`, reusing the
> match-scrutinee spill); #4b uniform pass-by-reference for structs == sums ==
> arrays; #4c `(ref T)` syntax; #4d locked the read-ref-as-value auto-load of
> `(mut)`-bound aggregates (already covered by Phase-2 #1's `coerce`); #4e
> auto-borrow a *writable* place to `(ptr T)`. The metal `ptr` tier stays
> distinct (only `(mut)`/explicit/writable places auto-borrow), const-correctness
> is intact (immutable refs reject write-through; mutable-ref params reject
> rvalues), and the C ABI for by-value aggregate FFI is untouched (externs never
> route through `param_ref_type` — verified in IR). Decisions resolved below.
> **Open:** A5 (re-passing a `(mut)` param bare) — deferred as a character call.

### The friction (from the dogfood)
The rule "a by-value aggregate param is really an immutable reference" leaks, and
structs vs sums behave differently:
- struct params were by-ref, sum params by-value → `(store! p x)` (copy a param out)
  failed for structs, worked for sums. (Partly fixed in Phase-2 #1: `coerce` now reads
  a reference as its value.)
- `(mut x)` does not coerce to `(ptr T)` → forces `(alloc-stack T)` to get a pointer.
- re-passing a `(mut)` param needs explicit `(mut x)`.
- returning a `(mut)`-bound aggregate needs explicit `(load x)`.
- no `(ref T)` spelling for an immutable reference.

### Recommended design — one uniform rule
**"Aggregates pass/bind by reference; a reference reads as its value where a value is
needed; borrow a place to get a pointer."**

1. **Uniform pass-by-reference for ALL aggregates** (structs, sums, arrays): make
   `param_ref_type` wrap sums (and arrays) in `Ref` just like structs, so every
   aggregate param is an immutable reference by default, `(mut T)` opting into a
   mutable one. Combined with #1's read-a-ref-as-value `coerce`, they all read as
   values uniformly, and `(store! p param)` copies any of them out. This preserves
   the deliberate const-correctness default (params immutable unless `(mut)`).
   *Verified compatible:* `al-push!`'s `(store! (index d n) v)` with a sum `v` already
   works via the coerce-read-ref path, so unifying sums to by-ref keeps it working.
2. **Auto-borrow a place to `(ptr T)`**: when a callee expects `(ptr T)` and the arg is
   a place of type `T`/`(mut T)`/`(ref T)`, insert the address-of. Removes the
   `alloc-stack` dance for `call-ptr` args, parser cursors, hashmap key spills, etc.
   (A `(mut T)` already erases to `(ptr T)`, so this is mostly lifting the call-site
   restriction.) Only *places* auto-borrow; rvalues still don't.
3. **`(ref T)` syntax** for an explicit immutable-reference param/type (today only
   `(mut T)` exists; immutable is implicit via by-value-aggregate). Small reader add;
   documents intent and lets non-aggregate immutable-ref params be written.
4. **Auto-load `(mut)`-bound aggregates in value contexts** (return/args), so `lex`
   can end in `toks` not `(load toks)` — the same read-a-ref-as-value rule applied to
   let places.

### Alternatives
- *Pass all aggregates by value* (sums already are): would drop the const-correctness
  default and is worse for large structs (copies). Rejected.
- *Leave sums by-value, structs by-ref* (status quo + #1 patch): functional but the
  asymmetry remains a latent surprise. Rejected — uniformity is the goal.

### Decisions needed → RESOLVED
- OK to make sums (and arrays) pass by immutable reference like structs? **Yes (#4b).**
  External C sum-passing is unaffected — externs keep by-value params (they never go
  through `param_ref_type`); verified in IR (`declare i64 @consume_tag(%Tag)` stays
  by-value while a Coil fn becomes `define i64 @f(ptr %t)`).
- How far should auto-borrow-to-`(ptr T)` go — only `(mut)` places, or any place?
  **Only *writable* places (#4e):** `(mut T)`/`(ptr T)`/`(mut x)`. Immutable places do
  NOT auto-borrow (a `(ptr T)` is writable; that would breach const-correctness), which
  is exactly the line that keeps the metal `ptr` tier distinct.
- Add `(ref T)` syntax now, or keep immutable-ref implicit? **Added now (#4c).**

---

## #5 — String type (the biggest capability gap)

### The friction
Strings are C-`(ptr i8)` (NUL-terminated): no length, no content ops without
`strlen`/`strcmp` externs, and string map keys need a hand-written content-hash
`KeyOps`. The calc dogfood used single-char (`i64`) vars to dodge it.

### Recommended design — a string is a `(slice u8)`; only the literal lowering is core
1. **CORE (the only core piece): string literals lower to a `(slice u8)` value** — the
   reader/codegen emits a private `[N x i8]` global and constructs the `{ptr, len}`
   (len known at compile time). Everything else is library.
2. **`lib/str.coil` over `(slice u8)`**: `str-len` (=slice-len), `str-eq` (content),
   `str-hash` (content, FNV over the bytes), `substr` (=subslice), `char-at`, `find`,
   `starts-with`, concatenation (allocator-aware) — all library over slices + mem.
3. **String-keyed HashMap**: a `str-keyops` (a `KeyOps` whose hash/eq read the slice's
   bytes) wires `(slice u8)` keys into the existing HashMap — pure library on top.
   This closes the dogfood gap (real identifier/string keys).

This keeps the core minimal (one literal-lowering rule) with the whole string API as
library — consistent with the macros/library-first thesis.

### The hard decisions (why jimmyhmiller should weigh in — this defines Coil's character)
- **C interop / NUL-termination.** `(slice u8)` strings are NOT NUL-terminated, but C
  APIs need NUL. Options: keep a separate `c"…"` literal (→ `(ptr i8)`, NUL-terminated)
  for FFI; or a `to-cstr` that allocates a NUL copy; or store strings NUL-terminated
  *and* length-carrying. Pick the FFI story.
- **Migration of existing `"…"` literals.** Today `"…"` is `(ptr i8)` and is used by
  fmt/externs. Switching `"…"` to `(slice u8)` is a BREAKING change. Options:
  (a) migrate `"…"`→`(slice u8)` and add `c"…"` for cstrings (clean end state, touches
      existing code incl. fmt/io); (b) add a NEW slice-string literal syntax and leave
      `"…"` as the cstr (non-breaking, two string spellings); (c) interim: a library
      `Str` built from a cstr literal + `strlen` (no core change), native `(slice u8)`
      literals later.
- **Representation:** `(slice u8)` (reuse Slice) vs a dedicated `Str` struct (same shape,
  named). Recommend reusing `(slice u8)` so all slice ops apply; a `Str` alias is cosmetic.

### Recommendation
Target `(slice u8)` strings with core literal-lowering + a `lib/str.coil` + `str-keyops`.
For the migration, I lean toward **(a)** (migrate `"…"`→`(slice u8)`, add `c"…"` for FFI)
as the clean end state, but it's a breaking change touching fmt/io, so it's exactly the
identity call for jimmyhmiller. If a lower-risk first step is preferred, **(c)** (a library
`Str` from cstr+strlen now, native literals later) ships value without a breaking core
change.

---

## Suggested order once decided
#4 (reference model) first — it's pervasive and simplifies everything downstream
including the string library. Then #5 (strings) on top. Then #6 (nested-generic
inference, unsigned arithmetic).
