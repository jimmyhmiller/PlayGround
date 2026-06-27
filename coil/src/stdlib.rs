//! The standard library, compiled into the compiler. An `(import "control.coil")`
//! resolves by bare name from anywhere — no path, no install step — like Rust's
//! `std`. A real file relative to the importer still wins (so editing `lib/*.coil`
//! during development takes effect); the embedded copy is the fallback.
//!
//! This is what lets `coil.core` re-export `control` (one line in the prelude).
//! Only general, shippable, cross-platform modules belong here — platform-specific
//! or app-specific code (e.g. macOS objc glue, a particular app's VM) is vendored
//! into the app, not bundled.

/// Bundled module sources, keyed by file basename.
pub const BUNDLED: &[(&str, &str)] = &[
    ("alloc.coil", include_str!("../lib/alloc.coil")),
    ("arraylist.coil", include_str!("../lib/arraylist.coil")),
    ("atomic.coil", include_str!("../lib/atomic.coil")),
    ("closure.coil", include_str!("../lib/closure.coil")),
    ("control.coil", include_str!("../lib/control.coil")),
    ("derive.coil", include_str!("../lib/derive.coil")),
    ("fmt.coil", include_str!("../lib/fmt.coil")),
    ("hashmap.coil", include_str!("../lib/hashmap.coil")),
    ("io.coil", include_str!("../lib/io.coil")),
    ("match.coil", include_str!("../lib/match.coil")),
    ("mem.coil", include_str!("../lib/mem.coil")),
    ("mmio.coil", include_str!("../lib/mmio.coil")),
    ("result.coil", include_str!("../lib/result.coil")),
    ("sexp.coil", include_str!("../lib/sexp.coil")),
    ("simd.coil", include_str!("../lib/simd.coil")),
    ("slice.coil", include_str!("../lib/slice.coil")),
    ("str.coil", include_str!("../lib/str.coil")),
    ("thread.coil", include_str!("../lib/thread.coil")),
    ("try.coil", include_str!("../lib/try.coil")),
];

/// The embedded source for a stdlib module by file basename, if any.
pub fn lookup(basename: &str) -> Option<&'static str> {
    BUNDLED.iter().find(|(n, _)| *n == basename).map(|(_, s)| *s)
}
