# Coil — Layout Control (design sketch)

> Status: **design sketch, not implemented.** A proposal for total, explicit
> control over data layout, built as the structural twin of Coil's
> calling-convention support.

## Thesis: layout is the dual of calling convention

Coil already makes the *call boundary* fully explicit and part of the type — a
`defcc` says exactly which registers carry arguments, what's clobbered, the
stack discipline, and a lowering strategy (`:native` when LLVM can express it,
`:shim` when it can't). **Layout is the same idea pointed at memory instead of
registers:** a layout says exactly where every field lives (offset, padding,
alignment, bit position, byte order), it's part of the type, and it has a
lowering strategy (trust LLVM when it can express it, drop to an explicit
byte-addressed realization when it can't).

The symmetry is exact:

| Calling convention | Layout |
|---|---|
| where args live across a call (registers/stack) | where fields live in memory (offsets/bits) |
| `:native` → an LLVM calling convention | `:c`/`:packed` → an LLVM struct (auto/packed) |
| `:shim` → naked trampoline marshalling exact registers | `:explicit` → byte blob + manual GEP/mask/bswap |
| convention is part of the function/fnptr type | layout is part of the (nominal) struct type |
| `defcc` names a reusable convention | `deflayout` names a reusable layout policy |
| macros pick a convention per `target-arch` | macros pick a layout per `target-arch` |

Because structs are already **nominal**, "layout in the type" is free: two
structs with the same fields but different layouts are different types, and you
can't silently pass one where the other is expected.

## The control surface (what "total control" covers)

1. **Field order** in memory, independent of declaration order.
2. **Explicit offsets** — place a field at byte N.
3. **Padding** — explicit reserved bytes, or *no* padding (packed).
4. **Alignment** — per-field and whole-struct; over-align (cache lines) and
   under-align (packed).
5. **Size** — fix/assert the total size.
6. **Bitfields** — sub-byte fields with explicit bit position and width.
7. **Unions / overlap** — multiple fields at the same offset.
8. **Endianness** — per-field byte order (wire/file formats).
9. (extension) **Sum-type tag & niche placement** — where the discriminant
   goes, and niche encodings (`Option<ptr>` is pointer-sized).

## Two tiers, mirroring `:native` vs `:shim`

### Tier 1 — layout *strategies* (the common cases; the `:native` analog)

```lisp
(defstruct Point  :layout c        [(x :i64) (y :i64)])     ; target C ABI layout
(defstruct Blob   :layout packed   [(tag :i8) (val :i64)])  ; no padding
(defstruct Line   :layout (align 64) [(a :i64) (b :i64)])   ; cache-line aligned
```

- `:c` (default) — natural alignment + padding per the target C ABI. Lowers to a
  normal LLVM struct; the data layout *is* the C layout. Use for FFI.
- `:packed` — zero padding. Lowers to an LLVM packed struct `<{…}>`; field
  loads/stores use align-1.
- `(align N)` — force whole-struct alignment; applied at `alloc`/global and on
  loads/stores.

These compose with generics: the strategy is re-applied per instantiation.

### Tier 2 — `:explicit` (total control; the `:shim` analog)

Every field states its offset; the struct states size and alignment. The
compiler verifies and emits a realization that hits those bytes exactly.

```lisp
(defstruct Elf64Header :layout explicit :size 64 :align 8
  [(magic    (array i8 4) :at 0)
   (class    :i8          :at 4)
   (encoding :i8          :at 5)
   (version  :i16         :at 6)
   (entry    :i64         :at 24 :endian little)
   (phoff    :i64         :at 32)
   (reserved (array i8 8) :at 56)])     ; explicit padding / reserved
```

- `:at <n>` — exact byte offset. Omitted ⇒ packed right after the previous field.
- `:endian little|big` — byte order for this field on the wire.
- Two fields with the same `:at` ⇒ an intentional **union** (overlap is only
  allowed when offsets are explicit, so it's never accidental).
- `:size`/`:align` on the struct — asserted and padded to.

### Bitfields

```lisp
(defstruct ControlReg :layout bits :backing i32
  [(enable   :bits 1)      ; bit 0
   (mode     :bits 3)      ; bits 1–3
   (channel  :bits 4)      ; bits 4–7
   (reserved :bits 24)])   ; bits 8–31
```

- `:backing iN` (optional) — the integer the bits pack into; default = smallest
  covering int. `:bit-order lsb|msb` for total control.
- Bits aren't addressable, so bitfields are accessed **by value**, not by
  pointer (see "Access semantics").

### Reusable policies (`deflayout`) and per-target selection

```lisp
(deflayout wire :explicit :endian big :align 1)   ; reusable named policy
(defstruct Packet :layout wire [(len :i32 :at 0) (kind :i16 :at 4) ...])

;; per-target layout, exactly like per-arch defcc — macros branch on the target
(defmacro defword [name]
  (if (= target-pointer-width 64)
      `(defstruct ~name :layout c [(lo :i64) ...])
      `(defstruct ~name :layout c [(lo :i32) ...])))
```

## Access semantics (the one genuinely new rule)

For an ordinary byte-addressable field, `(field p x)` keeps its current meaning:
a **pointer** to the field (a GEP), used with `load`/`store!`. That covers `:c`,
`:packed`, `(align N)`, and the non-bit/non-endian fields of `:explicit`.

For fields that **aren't plain memory** — bitfields and `:endian`-swapped fields
— you can't hand out a stable pointer (a bit has no address; an endian field
needs a swap on every access). Those use by-value accessors:

```lisp
(get  p channel)     ; load backing int, shift+mask  (or load+bswap for :endian)
(set! p channel 3)   ; read-modify-write the bits     (or bswap+store)
```

So the rule is honest: `field` → pointer for real memory; `get`/`set!` →
value for encoded fields. The checker knows which fields are which.

## Lowering (the LLVM-honesty section, as with conventions)

Just as LLVM's calling conventions are a closed enum (forcing `:shim`), LLVM's
struct layout is mostly fixed by the data layout and member types. So:

- **`:c`** → a normal LLVM struct; trust the data layout (it *is* the C ABI).
- **`:packed`** → an LLVM packed struct.
- **`(align N)`** → struct as above; alignment set on `alloca`/global and memory ops.
- **`:explicit` / `:bits`** → the *shim equivalent*: the type lowers to a flat
  `[size x i8]` blob (a packed struct). Field access is a **byte-offset GEP**
  into the blob, reinterpreted to the field's type — we compute every offset
  ourselves and never trust LLVM's auto-layout. Bitfields → `load`/shift/mask
  (read) and read-modify-write (write) on the backing integer. `:endian` →
  `llvm.bswap.iN` around the load/store. Unions → overlapping GEPs into the same
  bytes.

This is the same move as the naked-asm trampoline: where the backend's automatic
behavior isn't enough, drop to a manual realization that hits the exact bytes.

## New primitives (small, natural additions)

- `(offsetof T field)` → compile-time byte offset (the dual of `sizeof`;
  lowers via a GEP-on-null constant expression, or the declared `:at`).
- `(alignof T)` → alignment.
- `(static-assert expr msg?)` → compile-time check, e.g.
  `(static-assert (= (sizeof Elf64Header) 64))`. This is what makes "total
  control" *safe*: layouts are explicit **and verified**, so drift is a compile
  error.

## Verification (checked total control)

The compiler checks, for `:explicit`/`bits` layouts:
- offsets are non-decreasing or, if overlapping, only at equal `:at` (intentional
  union);
- each field fits within `:size`; total size and `:align` are consistent;
- bitfields fit their backing integer; bit ranges don't collide;
- alignment of each field's offset is compatible with its access (or the field
  is flagged unaligned and gets align-1 ops).

Total control, but you can't accidentally produce an inconsistent layout — same
philosophy as the convention checker rejecting an unlowerable `defcc`.

## Generics interaction

- Strategy layouts (`:c`, `:packed`, `(align N)`) re-derive offsets per
  instantiation, so they're fully generic.
- `:explicit` byte offsets with a *generic* field are ill-defined (a later
  field's offset depends on `sizeof T`). Options: forbid `:at` in generic
  structs, or allow offsets to be `sizeof`/`offsetof` expressions. Start with
  the restriction; lift later if needed.

## Sum-type tag & niche control (extension)

The same machinery extends to `defsum`: `:layout` on a sum controls the
discriminant — its integer type and offset, whether it's a leading tag or a
**niche** (encode the discriminant in an unused bit pattern of a field, so
`Option<(ptr T)>` is pointer-sized with `None = null`). This is the layout
analog of niche optimization and is a natural follow-on once struct layouts
exist.

## Suggested phasing

1. ✅ DONE: `:packed` + `(align N)` strategies + `static-assert`/`offsetof`/`alignof`
   (small; immediately useful for FFI and asserting C layouts).
2. ✅ DONE: `:explicit` with `:at` offsets + unions (the byte-blob realization).
3. Bitfields (`:bits`, backing int, get/set accessors).
4. `:endian` fields (bswap on access).
5. `deflayout` reusable policies (mostly sugar once 1–4 exist).
6. Sum-type tag/niche control.

Each tier is independently shippable, and (1) alone already gives the
"explicit ability to deal with padding" you can build C-ABI and packed structs
against with compile-time size assertions.
