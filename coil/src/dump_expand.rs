//! Canonical, lossless, deterministic dump of the macro **expander**'s output —
//! the differential-oracle target for the self-hosted Stage-3 macro expander
//! (`coil dump-expand`).
//!
//! The expander (`expand_stage3_macros`) runs the comptime macro functions over
//! the RAW (pre-macro) module-tagged form list and splices their generated syntax
//! to a fixpoint, returning a NEW module-tagged form list (the import/export tables
//! are passed through unchanged from the loader, so they are not re-dumped here —
//! the loader oracle already gates them). This module dumps the expanded form list
//! losslessly, in the SAME `(forms (tf MODTAG NODE) …)` shape as `dump_load`, so a
//! byte-diff gate over it is meaningful.
//!
//! Each tagged form's `Sexp` is dumped via the reader's own canonical node dumper,
//! so it is encoded exactly as `dump-read` would. ADDITIVE: no existing behavior
//! changes.

use crate::macros::TaggedForm;
use crate::reader::Sexp;

/// Per-byte canonical escape — identical to `reader::esc` / `dump_load::esc`.
fn esc(text: &str, out: &mut String) {
    use std::fmt::Write;
    for &b in text.as_bytes() {
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
}

fn dstr(s: &str, o: &mut String) {
    o.push('"');
    esc(s, o);
    o.push('"');
}

/// `(some "x")` / `(none)` for the optional module tag.
fn dopt_str(v: &Option<String>, o: &mut String) {
    match v {
        Some(s) => {
            o.push_str("(some ");
            dstr(s, o);
            o.push(')');
        }
        None => o.push_str("(none)"),
    }
}

/// A quoted `Sexp` — reuse the reader's canonical node dump (single form).
fn dump_sexp(s: &Sexp, o: &mut String) {
    o.push_str(&crate::reader::dump_canonical(std::slice::from_ref(s)));
}

/// Dump the expanded module-tagged form list canonically, in `Vec` order
/// (deterministic). Mirrors `dump_load`'s `(forms …)` section exactly.
pub fn dump_expanded(forms: &[TaggedForm]) -> String {
    let mut o = String::new();
    o.push_str("(forms");
    for (form, module) in forms {
        o.push_str(" (tf ");
        dopt_str(module, &mut o);
        o.push(' ');
        dump_sexp(form, &mut o);
        o.push(')');
    }
    o.push(')');
    o
}
