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


pub fn dump_mono(p: &crate::ast::Program) -> String {
    // `mono::monomorphize` builds its output funcs/structs/sums from
    // `HashMap::into_values()`, whose order is nondeterministic, so the raw dump is
    // not a stable oracle. Canonicalize by sorting these three sections by name
    // (mangled names are unique, so the order is total) before the shared dumper.
    // This sort lives ONLY in the mono dump path — the shared `dump_program` (used
    // by ast/resolved/checked, which are already deterministic) is untouched.
    let mut p = p.clone();
    p.funcs.sort_by(|a, b| a.name.cmp(&b.name));
    p.structs.sort_by(|a, b| a.name.cmp(&b.name));
    p.sums.sort_by(|a, b| a.name.cmp(&b.name));
    crate::dump_ast::dump_program(&p)
}

// Diag-error dump reuses the resolved error dumper verbatim.
pub use crate::dump_resolved::dump_error;

/// Canonical dump of a `mono` (String) error: spanless, so `lo`/`hi` render as
/// `D` exactly like a `Diag` carrying `Span::DUMMY` (`u32::MAX`).
pub fn dump_str_error(msg: &str) -> String {
    // A mono error is spanless — dump it as the reader's canonical DUMMY span so
    // the shape matches the other passes' error dumps (`@D:D:D:0`).
    crate::span::dump_diag_canonical(&crate::span::Diag::new(msg))
}
