# 03 — Forwarding-Pointer Chase Helper

## Implementation status: ✅ Implemented

- **Code**: `pub const FORWARDING_BIT` and `pub unsafe fn follow_forwarding(ptr) -> *const u8` in [`crates/dynalloc/src/semi_space.rs`](../../crates/dynalloc/src/semi_space.rs); both re-exported from `dynalloc` top level.
- **Tests**: 4 tests in `semi_space::follow_forwarding_tests` (passthrough, follow, single-hop debug-assertion, constant value).
- **Migration**: dynlang's `prop_slow_thunk` and beagle's `decode_array_ptr` both replaced their inline forwarding chases with `dynalloc::follow_forwarding`. The local `FORWARDING_BIT: u64 = 1 << 63` const in beagle main.rs is gone.
- **Skipped**: `NanBox::decode_live_ptr` convenience on dynvalue (proposed in this doc). Not added because each call site has different fallback semantics; the inline `match NanBox::decode + follow_forwarding` is clear enough.

## Problem

When the dynalloc semispace collector moves an object, it stores a
forwarding pointer in the object header with the high bit set
(`FORWARDING_BIT = 1 << 63`, low 63 bits = to-space address). Any extern
thunk that decodes a NanBox pointer at runtime must check this bit and
follow it before reading the object — otherwise it reads stale data from
from-space.

Beagle has the same chase coded inline in three thunks:

- **`ext_prop_slow`** ([`main.rs:160-178`](../../crates/beagle/src/main.rs#L160))
- **`array_len_of` / `array_elem_at`** via `decode_array_ptr`
  ([`main.rs:372-388`](../../crates/beagle/src/main.rs#L372))

It also reproduces the constant in user code:

```rust
const FORWARDING_BIT: u64 = 1 << 63; // main.rs:151
```

That's a dynalloc internal — duplicating it in every embedder is begging for
drift if dynalloc ever changes the encoding (e.g. tag in low bits for
alignment, or a different bit for a generational write barrier).

The chase has subtle correctness invariants every embedder has to
re-derive: forwarding chains can occur briefly during collection;
`debug_assert_eq!(header & FORWARDING_BIT, 0)` after one hop catches the
"to-space object also forwarded" bug ([`main.rs:171-176`](../../crates/beagle/src/main.rs#L171))
— but only if you remember to write it.

## Proposed API

```rust
// In dynalloc::semi_space (or higher in dynalloc's public module tree):

/// High bit of an object header indicates a forwarding pointer; low 63 bits
/// hold the to-space address. Single-hop only — chains are a collector bug.
pub const FORWARDING_BIT: u64 = 1 << 63;

/// If `ptr`'s header is a forwarding entry, return the to-space pointer;
/// otherwise return `ptr` unchanged. Reads exactly one header word.
/// Asserts in debug builds that the result is itself not forwarded.
///
/// # Safety
/// `ptr` must point to a live or forwarded GC object header. Calling on
/// arbitrary memory is UB.
pub unsafe fn follow_forwarding(ptr: *const u8) -> *const u8;
```

Plus a NanBox-flavored convenience layered in `dynvalue` for the most common
use site (extern thunks decoding a NanBox arg):

```rust
// In dynvalue, gated behind the dynalloc dependency:
impl NanBox {
    /// Decode `bits` to a pointer (tag must be the configured ptr tag) and
    /// follow any forwarding entry. Returns `None` for non-pointer NanBoxes.
    pub unsafe fn decode_live_ptr(bits: u64, tags: &NanBoxTags) -> Option<*const u8>;
}
```

## Implementation plan

1. **Move `FORWARDING_BIT` into `dynalloc::semi_space` as `pub const`.**
   It's already defined privately there
   ([reference in main.rs comment, `lower.rs:151`](../../crates/beagle/src/main.rs#L151)).
   Just expose it.

2. **Add `follow_forwarding(ptr)`.** Five-line function. Document the
   single-hop invariant; debug-assert in the body.

3. **Add `NanBox::decode_live_ptr`.** Wraps `decode` + tag check +
   `follow_forwarding`. Returns `Option<*const u8>` so the caller doesn't
   panic on tag mismatches (contrast with beagle's current
   `panic!("non-object NanBox")` — that policy belongs to the embedder, not
   the helper).

4. **Beagle migration.** Replace inline chase code in `ext_prop_slow` and
   `decode_array_ptr` with `NanBox::decode_live_ptr`. Delete the local
   `FORWARDING_BIT` const. Estimated delta: −30 LOC and one fewer place
   where dynalloc's internal layout is replicated.

## Open questions / risks

- **Generational GC.** `dynalloc` ships generational mode
  (`GcConfig::generational`) — does the forwarding-bit convention apply
  uniformly to nursery-promote and major-cycle moves? If yes, the helper
  is universal. If no (e.g. nursery uses a different sentinel), the API
  should be `follow_forwarding(ptr, &GcConfig)` or live on a
  `Heap`-shaped trait.
- **Header layout assumption.** `follow_forwarding` assumes the forwarding
  word is at offset 0 in the header. If a future compact-header layout
  splits header bytes (e.g. low 16 bits = type_id, high 48 = forwarding
  payload), the helper still works because `dynobj::Compact` zero-pads
  type_id and the high bit is unused — but document this contract.
- **Decoded enum.** `NanBox::decode_live_ptr` is sugar; the lower-level
  `Decoded::Tagged { tag, payload }` form is still useful for thunks that
  want to handle multiple tags. Keep both.
