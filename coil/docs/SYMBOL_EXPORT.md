# Symbol export — Coil as a C library

**Status: implemented end to end.** `(export-c …)` makes Coil functions callable from C
by a stable C symbol — receiving and returning structs by value across AArch64 (AAPCS64)
and x86-64 (SysV) via the C-ABI thunk — plus the tooling to ship a library:

- `coil build <f> --lib`     → a static archive (`.a`)
- `coil build <f> --shared`  → a shared library (`.dylib`/`.so`)
- `coil cheader <f>` / `coil build <f> --emit-header <h>` → a C header (prototypes + struct
  typedefs) for the export set — the inverse of `coil cimport`.

Verified end to end: a C client `#include`s the generated header, links the generated
`.a`, and calls Coil (incl. struct by value) correctly (`tests/struct_abi.rs`,
`tests/linkage.rs`). The only deferred item is phase 5 (freestanding entry as an explicit
export, to internalize the rest of a bare-metal image).

## What's implemented

- `(export-c name…)` / `(export-c [name :as "sym"] …)` — parsed, resolved (the name is
  qualified like a callable), and validated.
- **Linkage**: external iff `main` or exported; otherwise internal (when there's an
  anchor — a `main` or any export). A library/freestanding image with **no** exports
  and no `main` keeps everything external (back-compat fallback).
- **C symbol**: the explicit `:as`, else the bare name with `-`→`_`; the exported LLVM
  function is named by that symbol while Coil call sites still resolve internally.
- **ABI checks at the boundary** (`check.rs`): the function must exist, be non-generic,
  not use a `:shim` convention, have C-representable parameter and return types, and
  have a symbol unique among exports.

### By-value struct parameters — implemented via the C-ABI thunk

An exported function may now both **return** and **take** a struct by value. The
internal function stays reference-model (it takes the struct by pointer, `internal`
linkage); a generated **thunk** under the C symbol receives the struct per the C ABI
and marshals it to the pointer the internal function expects (`emit_export_thunk` in
`codegen.rs`). On AArch64 a 16-byte struct arrives as `[2 x i64]` and the thunk scatters
it into a slot; a large struct arrives `byval` and is passed straight through. The
design below records how it works; the only remaining boundary rejections are types
with *no* C representation at all (slice, `defsum`, SIMD vec, `Code`, generic `App`).

## Design: by-value aggregate parameters (the C-ABI thunk)

**Problem.** A Coil function's body, under the reference model, accesses an aggregate
parameter through a *pointer* (`(field p x)` needs `p : ptr`), and the checker erases
the parameter to `(ptr T)` before codegen. So the defined LLVM function is
`@f(ptr %p)`. A C caller passing the struct **by value** puts it in registers (small,
AAPCS64/SysV "Direct") or in a caller-allocated copy passed by pointer (large,
`byval`/"Indirect") — neither matches `@f(ptr %p)`, hence the crash. A single symbol
can't satisfy both the by-value C ABI and the by-reference internal ABI.

**Approach: a generated C-ABI *thunk*.** Keep the internal function exactly as it is
(reference-model, pointer params, *internal* linkage, qualified name). Emit a separate
**thunk** with external linkage under the C symbol, whose signature is the *full* C ABI
(from `c_signature` over the original, un-erased types). The thunk translates the C ABI
into the reference model and calls the internal function. Because the internal function
is `internal`, LLVM inlines it into the thunk, so the thunk *is* the real body at -O3 —
no call overhead. This is the exact inverse of `emit_c_call` (the outbound Coil→C path),
reusing the same `abi.rs` classifier, so it is automatically target-correct (x86-64
SysV and AArch64 AAPCS64) and needs no new ABI theory.

**When a thunk is emitted.** Only when the C-ABI signature differs from the
reference-model one — i.e. the export has ≥1 **aggregate-by-value parameter**. Exports
with only scalar/pointer parameters (whatever the return, including a by-value struct
return) keep today's direct symbol-rename path; no thunk, no change.

**The thunk body**, walking the export's `CSig` (`sret`, `args: Vec<ArgAbi>`, `ret_direct`):

- **`sret` return** (large struct): the thunk's first parameter is the hidden result
  pointer. The internal function also returns via `sret` (its return type isn't erased),
  so forward the thunk's `sret` pointer straight into the internal call; the thunk
  itself returns void.
- **`ArgAbi::Scalar`**: forward the incoming scalar/pointer parameter unchanged.
- **`ArgAbi::Direct(sa)`** (struct in registers): the thunk receives N register
  parameters (the coerced eightbyte/ HFA slots). Allocate a `struct_slot`, **store**
  each incoming register into its eightbyte offset — the inverse of `emit_c_call`'s
  "load each coerced slot from memory" loop (codegen.rs ~1157) — then pass the slot
  *pointer* to the internal function (which wants the aggregate as a pointer).
- **`ArgAbi::Indirect(sa)`** (large struct, `byval`/pointer): the incoming pointer is a
  caller-allocated copy the callee owns and may mutate, so pass it directly as the
  internal function's pointer argument. (No inbound copy needed — the thunk *is* the
  callee; the C ABI already gave it a private copy. `(mut T)` params therefore get the
  correct C by-value semantics: mutations stay local.)
- **Direct (small) struct return**: forward the internal call's coerced return value.

This reuses `struct_slot`, `copy_struct`, `struct_arg_ptr`, and the eightbyte GEP/store
pattern already in `emit_c_call`; the only genuinely new code is the register→memory
*scatter* (the mirror of the existing memory→register *gather*). Factoring a shared
`scatter_slots`/`gather_slots` pair is the clean refactor.

**Checker change.** Drop the by-value-aggregate-**parameter** rejection: a struct
parameter on an exported function is now legal (the thunk handles it). Keep rejecting
the types with *no* C representation at all (slice fat pointers, `defsum` tagged unions,
SIMD vectors, `Code`) and the non-C cases (`:shim` convention, generics). So
`export_param_ok` becomes the same predicate as `export_ret_ok` minus `void`: scalars,
pointers, and C-layout structs (an `App` monomorphized struct) are OK.

**Where it lives.** A new emit step after function bodies (codegen step 2): for each
export whose `CSig` needs a thunk, emit the internal function under its qualified name
(internal linkage) and the thunk under the C symbol (external). The existing
coil-name→C-symbol export map already drives this; the only branch is "does this export
need a thunk."

**Edge cases (all handled by the existing classifier, not the thunk).** HFAs pass as
`[N x fT]` (a single Direct slot — the scatter stores the float array); nested structs
flatten to eightbytes; large structs are `Indirect`; x86-64 vs AArch64 differences live
entirely in `c_signature`/`classify`, so the thunk is target-agnostic. Variadic exports
are out of scope (an exported Coil function isn't variadic).

**Rejected alternative.** Making the exported function *natively* take by-value params
(no thunk, unpacking in `emit_func`) would force internal Coil callers — which pass
aggregates by reference — onto the C by-value ABI, since one symbol can't carry two
ABIs. The thunk cleanly separates the public C entry from the reference-model body and
leaves all internal call sites untouched.

**Testing.** Extend `tests/struct_abi.rs` so the C fixture passes a struct **by value**
to an exported Coil function (the `dist2(Point)` case that crashed), on both Host
(AAPCS64) and x86-64 under Rosetta (SysV); plus an IR test that the thunk exists under
the C symbol with the C-ABI signature while the internal function stays `internal`.

**Phasing.** (1) Direct (register) struct params — the common small-struct case;
(2) Indirect (`byval`) large-struct params; (3) drop the checker rejection. Returns
already work and are unchanged throughout.

## Deferred (still design)

Phases 3 (library output modes `--lib`/`--shared`), 4 (`--emit-header`), and 5
(freestanding entry as an export, so a bare-metal image internalizes its rest) are not
yet built. Today, `coil emit-obj` already produces an object whose exported symbols C
can link against (that's what the end-to-end test does); `--lib`/`--shared`/headers are
conveniences on top.

---

## Original design notes

The sections below are the original design; the parts above record what shipped.



## Why

Today Coil produces *executables*: `main` is the one external symbol; as of the
internal-linkage change (see `bench/README.md`), every other function in an executable
gets `internal` linkage so the optimizer can prune it — which is most of Coil's
compile-time parity with clang.

But internal linkage means those functions have **no stable C symbol**, so the dual of
`extern` — *Coil code that C calls* — isn't expressible. The stopgap is a heuristic:
a program with **no `main`** (a library, or a freestanding image whose entry is a
custom symbol like `start`) keeps *everything* external. That works but is blunt:
- a library can't internalize its private helpers (no compile-time win, larger surface);
- there's no control over which symbols are public, what they're named, or whether
  their signatures are actually C-callable;
- there's no generated header, so the C side hand-writes (and de-syncs) prototypes.

`extern` lets Coil **import** a C symbol. This feature is the mirror: **export** a Coil
function *as* a C symbol, with the same ABI rigor Coil already applies to `extern` and
struct-by-value (`src/abi.rs`, `check_c_abi_types`).

## The core rule (linkage)

Replace the current heuristic with an explicit set:

> A function gets **external** linkage iff it is `main`, or it is **exported**.
> Everything else is **internal**.

This unifies the three cases the stopgap handles separately:
- **Executable** (`coil build`): external = `{main} ∪ exports`. (Most programs export
  nothing → only `main`, exactly today's behavior.)
- **Library** (no `main`): external = `exports`. Private helpers finally internalize —
  smaller surface, faster optimize/emit, and the public API is *exactly* what's declared.
- **Freestanding**: the entry (`start`, a reset handler, …) is just an export. Then the
  rest of a bare-metal image can internalize too (a current missed win), and a
  zero-export image is a clear error ("nothing to enter / link").

The exported function's LLVM function is named by its **C symbol** (below), not its
qualified Coil name (`app.make-point`), and carries `external` linkage; the body is
unchanged.

## Surface syntax

A top-level `export-c` form, parallel to the existing `(export …)` (which controls
Coil-module visibility — orthogonal to this, which controls the C ABI surface):

```clojure
(module shapes)
(defstruct Point :c [(x :i64) (y :i64)])     ; :c layout — required for an exported sig

(defn make-point [(x :i64) (y :i64)] (-> Point)
  (let [p (alloc-stack Point)] (store! (field p x) x) (store! (field p y) y) (load p)))

(defn point-eq [(a Point) (b Point)] (-> bool) …)

;; Export two functions to C. Two ways the C symbol is chosen:
;;   - `:as "sym"`     — use exactly `sym` (the escape hatch; also for namespacing).
;;   - no `:as`        — the bare name with '-' → '_'  (point-eq → `point_eq`).
;; A name that still isn't a valid C identifier after conversion (e.g. it held `?`/`!`)
;; is a compile error pointing here, asking for an explicit `:as`.
(export-c
  [make-point :as "shapes_make_point"]   ; explicit C symbol
  point-eq)                              ; default naming -> C symbol `point_eq`
```

Why a top-level form rather than a `defn` attribute:
- it reads as the library's **public API manifest** in one place (what `--emit-header`
  walks);
- it keeps `defn` signatures clean and lets the same function be both internally used
  and exported without decorating its definition;
- it mirrors `extern`/`export`, so the model is "imports up top, exports up top."

(An attribute form `(defn foo … :export-c "foo")` is a possible sugar later; the
top-level form is the primitive.)

## ABI constraints (checked at the export boundary)

An exported function is a C ABI boundary, so the checker enforces — reusing the
existing `check_c_abi_types` / `src/abi.rs` machinery used for `extern` and
struct-by-value — that:

1. **Calling convention is C.** The function must use the `c` convention (or a `defcc`
   whose lowering is `:native` C). A `:shim`/custom-cc function can't be a C export
   (its ABI isn't the C one). Default-cc functions either must be declared `:cc c` or
   the export errors with a fix-it.
2. **Types are C-representable.** Parameters and return type must each be one of:
   scalars (`iN`/`f32`/`f64`/`bool`), a pointer (`(ptr T)`), a C-string (`(ptr i8)`),
   or a struct/array with a **C-compatible layout** (`:c`/`:packed`/`(align N)` /
   `:explicit`). **Rejected at the boundary:** `(slice T)` fat pointers, `defsum`
   tagged unions, closures, `Code`/comptime types, and **generics** (an exported
   function must be fully monomorphic — a `[T]` function has no single symbol). Each
   rejection is a located error ("`make-line` exports a `(slice u8)` parameter, which
   has no C representation; pass `(ptr u8)` + a length, or wrap it").
3. **Symbol uniqueness.** Two exports may not resolve to the same C symbol (after
   `:as` / kebab conversion) — a hard error, like a duplicate C definition.

This is the same posture as `extern` (which already type-checks the C side): the ABI is
verified at definition time, not discovered at link time.

## Symbol naming

Coil names are kebab-case and module-qualified (`shapes.make-point`); C symbols match
`[A-Za-z_][A-Za-z0-9_]*`. Resolution:
- `:as "name"` — use exactly `name` (the escape hatch; also how you get a C++-friendly
  or namespaced symbol like `shapes_make_point`).
- otherwise — the **bare** Coil name (module prefix dropped) with `-` → `_`. If the
  result isn't a valid C identifier (e.g. it contained `?`/`!`/`*`), that's an error
  asking for `:as`.
- The symbol is **not** otherwise mangled — C sees exactly what's declared (the point
  of the feature). Overload-style mangling is out of scope.

## Build modes (library output)

Exports are only useful if the output is something C can link. Add output modes to the
driver (today: executable from `coil build`, object from `coil emit-obj`):

- `coil build --lib libfoo.a` — a **static archive** of the object (no `main` required;
  exports are the public symbols).
- `coil build --shared libfoo.{dylib,so}` — a **shared library** (drives `cc -shared`;
  exported symbols are the dynamic surface; the rest stay internal/hidden, which also
  helps load time).
- `coil emit-obj` already emits an object; it simply honors exports now.
- `coil build` (executable) may also export (a plugin host whose symbols a `dlopen`'d
  module resolves, or a callback C resolves by name) — `main` plus the exports.

A library build with **zero** exports is a warning/error ("no public symbols — nothing
for C to call").

## Header generation

The export manifest is exactly the information a C header needs, so:

- `coil build --lib … --emit-header foo.h` (or a standalone `coil cheader file.coil`)
  walks the `export-c` set and emits:
  - include guards / `extern "C"` for C++;
  - a `typedef struct { … } Point;` for each `:c` struct/array reachable from an
    exported signature, at its real field layout (Coil already knows it — `LAYOUT.md`);
  - a prototype per exported function, with the C symbol and the C types, in the same
    order Coil lowers them.
- This is the inverse of `coil cimport` (which reads a C header → Coil `extern`s): here
  Coil *writes* the header. Round-tripping `cimport`(`cheader`(lib)) should reproduce
  the signatures.

Header generation is a separate increment from the linkage/ABI core, but it's the thing
that makes "Coil as a C library" pleasant rather than possible.

## Interactions / edge cases

- **Internal-linkage rule** changes from `name != "main"` to `name != "main" && !exported`.
  The "no `main` ⇒ all external" stopgap is removed once exports exist; a library now
  internalizes its privates.
- **`fnptr-of` to C callbacks** is unaffected — taking an internal function's address
  and handing the *pointer* to C already works (no symbol needed); export is only for C
  calling **by name**.
- **`main` is implicitly exported** as the symbol `main` (the C runtime's entry); it
  needn't be listed.
- **Cross-module**: `export-c` lives in the module that defines the function; the C
  symbol is global (flat C namespace), so cross-module symbol collisions are caught by
  the uniqueness check above.
- **LTO / visibility**: shared-library builds should additionally set
  `dso_local`/hidden visibility on internal functions so they don't bloat the dynamic
  symbol table; exported ones get default visibility.

## Phased plan

1. **Linkage + `export-c` core.** Parse `export-c`; thread an `exports` set into
   codegen; external iff `main` or exported; the LLVM function for an export is named by
   its C symbol. Replaces the no-`main` stopgap. (Smallest change that restores the
   `c_calls_coil_*` capability *selectively* and keeps the compile-time win for
   libraries.)
2. **ABI checking at the boundary.** Reuse `check_c_abi_types`; reject non-C types and
   non-C conventions with located errors; symbol-uniqueness check.
3. **Library output modes.** `--lib` (archive) and `--shared` (dylib), plus visibility.
4. **Header generation.** `--emit-header` / `coil cheader`, including the `:c` structs.
5. **Freestanding entry as an export.** Mark bare-metal entries with `export-c`,
   internalize the rest of the image, drop the freestanding linkage special-case.

## Open questions

- **Default convention for export.** Require an explicit `:cc c` on the `defn`, or let
  `export-c` *coerce* a default-cc function to the C ABI at the boundary (it already
  has C-compatible types)? Leaning: coerce, since the type check already proves it's
  C-callable — but a `:shim`/non-C cc is always an error.
- **Slices/sums at the boundary.** Always reject (forcing the author to lower to
  `ptr`+`len` / a `:c` tagged struct), or offer a blessed lowering (e.g. a slice
  exports as a `{ T* ptr; int64 len; }` C struct)? Start by rejecting; add blessed
  lowerings if dogfooding wants them.
- **Versioned/namespaced symbols.** Out of scope; `:as` covers manual prefixing.
- **`build.coil`** (future build system) is the natural place to declare a library
  target + its exports + header path, rather than CLI flags; the `export-c` form is the
  source-level primitive either way.
