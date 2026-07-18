# Coil issues found during the Native component port

## Primitive equality coverage

The bundled prelude did not implement the `Eq` bound used by `=` and `!=` for
integer widths other than `i64`:

```coil
(!= (load text) (cast i8 0))
```

The compiler reported `i8 does not implement Eq`. This was fixed in Coil by
adding `Eq` implementations for `i8`, `i16`, `i32`, `u8`, `u16`, `u32`, and
`u64`. The self-host fixpoint, full IR corpus, ARM64 runtime gate, and CLI gate
all pass. Flowline now uses the clean `!=` operator for null-terminator checks.

## Qualified exported constants

Constants exported by `src/raylib.coil` are not available through the module
alias (`raylib.MOUSE_BUTTON_LEFT`, `raylib.TEXTURE_FILTER_BILINEAR`), although
exported functions and structs are. The port currently uses the audited Raylib
numeric constants at the call site.

## Fields on returned structs

Field access directly on a struct-returning call is rejected:

```coil
(field (slider-frame) left)
```

The compiler reports that field access needs a pointer or reference even
though the same value works after binding or when passed to another function.
The port binds returned structs or uses the already-known layout coordinate.

## First-frame screenshot readback

Raylib screenshot readback immediately after the first frame produced a valid
PNG containing only the clear color. Rendering several warm-up frames before
`TakeScreenshot` is reliable. This may be platform/Raylib timing rather than a
Coil compiler defect, but the deterministic catalog harness preserves the
workaround.
