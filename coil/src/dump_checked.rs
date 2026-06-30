//! Canonical, lossless, deterministic dump of the **type-check** pass's output —
//! the differential-oracle target for the self-hosted checker (`coil dump-checked`).
//!
//! The pass (`check::check_with`) takes the resolver's merged, name-resolved
//! `Program` + the import/export tables and produces a typed, elaborated, lowered
//! `Program`: it INTRODUCES the 6 ExprKind variants the parser never emits
//! (`Construct`, `TraitCall`, `DynDispatch`, `MakeDyn`, `SpillRef`, `StaticRef`)
//! and writes back inferred generic type-args. Its output is a `Program`, so the
//! canonical dump is exactly the shared `dump_ast::dump_program` dumper — reused
//! verbatim. That shared dumper is already exhaustive over the 48-variant
//! `ExprKind`, so it renders the 6 check-introduced variants with no change.
//!
//! A read / load / resolve / CHECK error is dumped in the same canonical shape as
//! `dump-read`/`dump-ast`/`dump-resolved` (`(error@<lo>:<hi> "msg")`), using the
//! FIRST diagnostic, so error-path parity is gated too. This module is ADDITIVE:
//! it changes no existing compiler behavior.

pub fn dump_checked(p: &crate::ast::Program) -> String {
    crate::dump_ast::dump_program(p)
}

// Error dump reuses the resolved error dumper verbatim.
pub use crate::dump_resolved::dump_error;
