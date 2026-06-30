//! Canonical, lossless, deterministic dump of the module **loader**'s output —
//! the differential-oracle target for the self-hosted loader (`coil dump-load`).
//!
//! The loader (`macros::load_program`) turns a file's forms (plus the prelude and
//! everything it imports) into three outputs: a module-tagged form list, an
//! `ImportMap`, and an `ExportMap`. This module dumps all three losslessly.
//!
//! Discipline mirrors `reader::dump_canonical` / `dump_ast::dump_program`: numbers
//! explicit, strings via the reader's per-byte escape, spans `@lo:hi` (`@D:D` for a
//! dummy), and — crucially — the two `HashMap`s (`imports`, `exports`) and the
//! `HashSet`s/`HashMap`s inside them are emitted with keys and members SORTED, so
//! the dump is deterministic (a `HashMap`'s iteration order is not).
//!
//! Each tagged form's `Sexp` is dumped via the reader's own canonical node dumper,
//! so it is encoded exactly as `dump-read` would. This module is ADDITIVE: it does
//! not change any existing compiler behavior.

use crate::macros::{ExportMap, ImportMap, ModImports, TaggedForm, UseSpec};
use crate::reader::Sexp;

/// Per-byte canonical escape — identical to `reader::esc` / `dump_ast::esc`.
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

/// `["a" "b"]` — a vector of strings (caller decides ordering).
fn dstrs(v: &[String], o: &mut String) {
    o.push('[');
    for (i, s) in v.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        dstr(s, o);
    }
    o.push(']');
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

/// Dump the three loader outputs canonically. Sections appear in a fixed order;
/// the two `HashMap`s are sorted by key, and the inner `HashMap`/`HashSet` are
/// sorted too, for determinism. The form list is already a `Vec` (deterministic).
pub fn dump_loaded(forms: &[TaggedForm], imports: &ImportMap, exports: &ExportMap) -> String {
    let mut o = String::new();

    // ---- forms: (tf MODTAG NODE) in Vec order ----
    o.push_str("(forms");
    for (form, module) in forms {
        o.push_str(" (tf ");
        dopt_str(module, &mut o);
        o.push(' ');
        dump_sexp(form, &mut o);
        o.push(')');
    }
    o.push(')');

    // ---- imports: (mod NAME ALIASES USES REEXPORTS), keys sorted ----
    o.push_str("\n(imports");
    let mut inames: Vec<&String> = imports.keys().collect();
    inames.sort();
    for name in inames {
        o.push_str(" (mod ");
        dstr(name, &mut o);
        o.push(' ');
        dump_modimports(&imports[name], &mut o);
        o.push(')');
    }
    o.push(')');

    // ---- exports: (mod NAME NAMES), keys sorted, members sorted ----
    o.push_str("\n(exports");
    let mut enames: Vec<&String> = exports.keys().collect();
    enames.sort();
    for name in enames {
        o.push_str(" (mod ");
        dstr(name, &mut o);
        o.push(' ');
        let mut names: Vec<String> = exports[name].iter().cloned().collect();
        names.sort();
        dstrs(&names, &mut o);
        o.push(')');
    }
    o.push(')');

    o
}

/// `(aliases [(al "k" "v") …]) (uses [(use "target" SPEC) …]) (reexports [...])`.
/// Aliases (a `HashMap`) are sorted by key; uses + reexports are `Vec`s (kept in
/// insertion order, which is deterministic — `process_import` push order).
fn dump_modimports(m: &ModImports, o: &mut String) {
    // aliases — sorted by alias key
    o.push_str("(aliases [");
    let mut keys: Vec<&String> = m.aliases.keys().collect();
    keys.sort();
    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        o.push_str("(al ");
        dstr(k, o);
        o.push(' ');
        dstr(&m.aliases[*k], o);
        o.push(')');
    }
    o.push_str("]) (uses [");
    // uses — Vec order
    for (i, (target, spec)) in m.uses.iter().enumerate() {
        if i > 0 {
            o.push(' ');
        }
        o.push_str("(use ");
        dstr(target, o);
        o.push(' ');
        match spec {
            UseSpec::All => o.push_str("(all)"),
            UseSpec::Names(names) => {
                o.push_str("(names ");
                dstrs(names, o);
                o.push(')');
            }
        }
        o.push(')');
    }
    o.push_str("]) (reexports ");
    dstrs(&m.reexports, o);
    o.push(')');
}
