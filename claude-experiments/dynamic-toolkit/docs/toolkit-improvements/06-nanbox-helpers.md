# 06 — NanBox Embedder Helpers

## Implementation status: ✅ Implemented (constants + encoders only)

- **Code**: `NanBox::NIL`, `NanBox::from_int(i64) -> u64`, `NanBox::from_f64(f64) -> u64` as inherent methods in [`crates/dynvalue/src/nan_box.rs`](../../crates/dynvalue/src/nan_box.rs).
- **Tests**: 6 tests in `nan_box::inherent_tests` (NIL decodes, NIL constant value, from_int round-trip, lossless below 2^53, from_f64 passthrough, NaN canonicalization).
- **Migration**: beagle deleted local `nanbox_nil()`, `encode_f64_int(n)`, and the `TAG_PATTERN` const from main.rs. Two call sites now use `NanBox::NIL` and `NanBox::from_int`.
- **Not implemented**: `DecodedDisplay` formatter (proposed in this doc). Beagle's `print_value` still hand-rolls the tag-by-tag print logic — pending a real second consumer to justify the abstraction.

## Problem

`dynvalue::NanBox` exposes `decode` and `Decoded`, but several common
embedder needs aren't covered, so beagle reimplements them in main.rs:

- **`nanbox_nil()`** ([`main.rs:335-337`](../../crates/beagle/src/main.rs#L335))
  hand-rolls the nil bit pattern using a private constant
  `TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000`
  ([`main.rs:11`](../../crates/beagle/src/main.rs#L11)). This is the canonical
  NaN signaling pattern — internal to dynvalue, leaking to user code.
- **`encode_f64_int(n)`** ([`main.rs:339-341`](../../crates/beagle/src/main.rs#L339))
  is `(n as f64).to_bits()`. Common idiom for "boxed integer" — should be
  on `NanBox`.
- **`print_value`** ([`main.rs:390-416`](../../crates/beagle/src/main.rs#L390))
  switches on `Decoded` and emits human-readable output for floats, nil,
  bools, pointers, and string-tag-3. Every embedder will write this; the
  generic shape (everything except the string-pool resolver) belongs in
  dynvalue.
- **Tag-2 ptr decode** is repeated in `ext_prop_slow`, `array_len_of`,
  `array_elem_at`, `decode_array_ptr` ([covered also by doc 03](03-forwarding-pointer-helper.md)).
  The `match Decoded::Tagged { tag: 2, payload }` shape pretends `2` is
  a magic constant when it's actually `NanBoxTags::default().ptr`.

## Proposed API

Add to `dynvalue::NanBox`:

```rust
impl NanBox {
    /// Canonical nil bit pattern. Equivalent to NanBoxTags::default()'s
    /// nil tag with payload 0.
    pub const NIL: u64;

    /// Encode an integer as a NanBox float. Lossless for |n| < 2^53.
    pub fn from_int(n: i64) -> u64;

    /// Encode an `f64`. Just `f.to_bits()`, exposed for symmetry.
    pub fn from_f64(f: f64) -> u64;

    /// Encode `true`/`false` using the configured bool tag.
    pub fn from_bool(b: bool, tags: &NanBoxTags) -> u64;

    /// Encode a heap pointer (caller is responsible for tag policy).
    pub fn from_ptr(p: *const u8, tags: &NanBoxTags) -> u64;
}
```

And a generic display formatter on `Decoded`:

```rust
/// Human-readable formatting of a decoded NanBox. Embedder supplies a
/// callback for tags it owns (typically string interning, symbol tables,
/// custom type printers).
pub struct DecodedDisplay<'a, F: Fn(u32, u64) -> Option<String>> {
    pub decoded: Decoded,
    pub tags: &'a NanBoxTags,
    pub resolve_tag: F,
}

impl<F> std::fmt::Display for DecodedDisplay<'_, F>
where F: Fn(u32, u64) -> Option<String> { ... }
```

So beagle's `print_value` shrinks to:

```rust
fn print_value(bits: u64, newline: bool) {
    let dec = NanBox::decode(bits);
    let s = DecodedDisplay {
        decoded: dec,
        tags: &TAGS,
        resolve_tag: |tag, payload| {
            if tag == STRING_TAG {
                Some(host().strings.get(payload as u32).unwrap_or("<bad str>").into())
            } else {
                None
            }
        },
    };
    if newline { println!("{}", s); } else { print!("{}", s); }
}
```

## Implementation plan

1. **Constants on `NanBox`.** `NIL`, `TRUE`/`FALSE` (where the bool tag
   is fixed by `NanBoxTags::default`). Document that custom tag schemes
   need to call the `from_*(tags)` overloads.

2. **`from_int` / `from_f64` / `from_bool` / `from_ptr`.** Inverse of
   `decode`. Beagle today encodes inline; making these a method clarifies
   that `(n as f64).to_bits()` is *intentional* NanBox semantics, not
   bit-twiddling.

3. **`DecodedDisplay`.** Default branches: `Float` (with the integer-print
   shortcut beagle already does), `Tagged { tag: nil, .. }` → "null",
   `Tagged { tag: bool, .. }` → "true"/"false", `Tagged { tag: ptr, .. }`
   → "<ptr 0x…>", anything else → user `resolve_tag` callback or
   `<tagN 0x…>` fallback.

4. **Beagle migration.** Replace `nanbox_nil`, `encode_f64_int`,
   `TAG_PATTERN`, and `print_value`'s match body. Estimated delta:
   −30 LOC, and dynvalue internals stop leaking.

## Open questions / risks

- **`NanBox::NIL` vs `NanBox::nil(tags)`.** A `const` is convenient for
  use in static initializers but assumes `NanBoxTags::default`'s nil tag
  is `0`. If a future config moves the nil tag, embedders using `NIL`
  silently misbehave. Probably want both: `const NIL` for the default
  scheme + `pub fn nil(tags: &NanBoxTags) -> u64` for custom. Or only
  the latter to force the question.
- **Integer-print heuristic.** Beagle's `print_value` prints `f as i64`
  when `f.fract() == 0.0 && f.abs() < 1e16`. That heuristic isn't
  universal — Lua, JavaScript, and Lox all want it; Scheme probably
  doesn't. Make it a flag on `DecodedDisplay`.
- **No `Display`/`Debug` impl on `Decoded` directly.** The string-pool
  resolver is a free parameter, so a plain `impl Display for Decoded`
  isn't possible without a default of "`<tagN 0x…>` for everything
  unknown". Could ship that as a baseline impl and let
  `DecodedDisplay` override.
