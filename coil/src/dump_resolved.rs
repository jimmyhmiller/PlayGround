//! Canonical, lossless, deterministic dump of the **name-resolution** pass's
//! output — the differential-oracle target for the self-hosted resolver
//! (`coil dump-resolved`).
//!
//! The pass (`resolve::resolve_program`) takes the loader's module-tagged forms +
//! the import/export tables and produces one merged, name-resolved `Program`. Its
//! output is a `Program`, so the canonical dump is exactly the shared
//! `dump_ast::dump_program` dumper — reused verbatim, so parser-output parity and
//! resolver-output parity are gated by the same byte-exact encoder.
//!
//! A load OR resolve error is dumped in the same canonical shape as
//! `dump-read`/`dump-ast` (`(error@<lo>:<hi> "msg")`), so error-path parity is
//! gated too. This module is ADDITIVE: it changes no existing compiler behavior.


use crate::ast::Program;
use crate::span::Diag;

/// Dump a resolved program canonically — reuse the shared `Program` dumper.
pub fn dump_resolved(p: &Program) -> String {
    crate::dump_ast::dump_program(p)
}

/// Canonical dump of a load/resolve diagnostic, identical in shape to the
/// `dump-read` / `dump-ast` error path.
pub fn dump_error(d: &Diag) -> String {
    crate::span::dump_diag_canonical(d)
}
