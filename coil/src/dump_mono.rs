//! Canonical, lossless, deterministic dump of the **monomorphization** pass's
//! output — the differential-oracle target for the self-hosted monomorphizer
//! (`coil dump-mono`).
//!
//! The pass (`mono::monomorphize`) takes the checker's typed/elaborated/lowered
//! `Program` and produces a `Program` with NO remaining generics: each generic
//! function/struct/sum is stamped out per concrete type-arg set, calls are
//! rewritten to the specialized (mangled) names, and the generic templates are
//! dropped. Its output is a `Program`, so the canonical dump is exactly the
//! shared `dump_ast::dump_program` dumper — reused verbatim (the same byte-exact
//! encoder that gates the parser/resolver/checker output).
//!
//! A read / load / resolve / check / MONO error is dumped in the same canonical
//! shape as the other passes (`(error@<lo>:<hi> "msg")`). The front-end passes
//! produce a `Diag` (with a span); `mono::monomorphize` returns a plain `String`,
//! so a mono error is dumped as a spanless `(error@D:D "msg")` — the same shape a
//! `Diag` with a dummy span renders. This module is ADDITIVE: it changes no
//! existing compiler behavior.

use std::fmt::Write;

pub fn dump_mono(p: &crate::ast::Program) -> String {
    crate::dump_ast::dump_program(p)
}

// Diag-error dump reuses the resolved error dumper verbatim.
pub use crate::dump_resolved::dump_error;

/// Canonical dump of a `mono` (String) error: spanless, so `lo`/`hi` render as
/// `D` exactly like a `Diag` carrying `Span::DUMMY` (`u32::MAX`).
pub fn dump_str_error(msg: &str) -> String {
    let mut out = String::new();
    for &b in msg.as_bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(b as char),
            _ => write!(out, "\\x{b:02x}").unwrap(),
        }
    }
    format!("(error@D:D \"{out}\")")
}
