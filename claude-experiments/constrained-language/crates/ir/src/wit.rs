//! Generate the WIT world for a handler from a validated manifest.
//!
//! The contract: each handler's component is given exactly the imports
//! corresponding to its declared `read` / `write` / `emit` footprint, plus
//! the export `handle: func(event: <event-payload>)`. A body cannot
//! observe state or perform effects beyond what's in its world.
//!
//! v0.1 keeps the generator simple:
//! * All named schemas in the manifest are emitted into the world (no
//!   per-handler reachability trimming yet).
//! * Inline event payloads are synthesized as records named after the
//!   event.
//! * Read/write functions always take their key as an explicit argument
//!   (rather than computing it from the bound event-path); the runtime
//!   enforces that the key matches the declared binding at call time.
//! * Records and variants must be declared in `schemas` and referenced
//!   by name from cell/event/effect surfaces. The validator will be
//!   tightened later; for now the generator emits a clear marker if it
//!   sees one inline.

use std::fmt::Write as _;

use crate::access::AccessSegment;
use crate::manifest::{Handler, Manifest, StateDecl};
use crate::schema::{SchemaDef, SchemaRef};

/// Generate the full WIT source for one handler's world.
pub fn generate_world(handler: &Handler, manifest: &Manifest) -> String {
    let mut out = String::new();
    let pkg = kebab(&manifest.name);
    let _ = writeln!(out, "package program:{pkg};");
    let _ = writeln!(out);
    let _ = writeln!(out, "// Generated for handler `{}`.", handler.name);
    let _ = writeln!(out, "// The declared footprint in the manifest is the complete authoring surface.");
    let _ = writeln!(out);

    let _ = writeln!(out, "world {} {{", kebab(&handler.name));

    // ---- type declarations ----
    let _ = writeln!(out, "  // Types");
    for (name, def) in &manifest.schemas {
        emit_named_type(&mut out, name, def, 2);
    }
    // Synthesize a record for any inline event payload, named after the event.
    for (ename, edef) in &manifest.events {
        if let SchemaRef::Inline(def) = &edef.payload {
            emit_named_type(&mut out, ename, def, 2);
        }
    }
    let _ = writeln!(out);

    // ---- imports: reads ----
    if !handler.read.is_empty() {
        let _ = writeln!(out, "  // State reads (one function per declared read slice)");
        for path in &handler.read {
            emit_read_import(&mut out, path, manifest);
        }
        let _ = writeln!(out);
    }

    // ---- imports: writes ----
    if !handler.write.is_empty() {
        let _ = writeln!(out, "  // State writes (one function per writable slice; Map cells get put + delete)");
        for path in &handler.write {
            emit_write_imports(&mut out, path, manifest);
        }
        let _ = writeln!(out);
    }

    // ---- imports: effects ----
    if !handler.emit.is_empty() {
        let _ = writeln!(out, "  // Effects (one emit-<name> per declared effect type; returns an opaque emit id)");
        for effect_name in &handler.emit {
            if let Some(def) = manifest.effects.get(effect_name) {
                let fn_name = format!("emit-{}", kebab(effect_name));
                let req_ty = render_type_ref(&def.request);
                let _ = writeln!(out, "  import {fn_name}: func(req: {req_ty}) -> u64;");
            }
        }
        let _ = writeln!(out);
    }

    // ---- export ----
    let event_ty = event_type_ref(&handler.on, manifest);
    let _ = writeln!(out, "  // Handler entry point");
    let _ = writeln!(out, "  export handle: func(event: {event_ty});");
    let _ = writeln!(out, "}}");

    out
}

fn emit_read_import(out: &mut String, path: &crate::access::AccessPath, manifest: &Manifest) {
    let Some(cell) = manifest.state.get(&path.cell) else {
        return;
    };
    let cell_kb = kebab(&path.cell);
    match cell {
        StateDecl::Atom { schema } => {
            // Path may have field navigation, but reads always return the whole
            // cell value; navigation happens inside the body.
            let _ = writeln!(
                out,
                "  import get-{cell_kb}: func() -> {};",
                render_type_ref(schema)
            );
        }
        StateDecl::Map { key, value } => {
            let kty = render_type_ref(key);
            let vty = render_type_ref(value);
            match path.segments.first() {
                Some(AccessSegment::Key(_)) => {
                    let _ = writeln!(
                        out,
                        "  import get-{cell_kb}: func(key: {kty}) -> option<{vty}>;"
                    );
                }
                Some(AccessSegment::Wildcard) | None => {
                    let _ = writeln!(
                        out,
                        "  import list-{cell_kb}: func() -> list<tuple<{kty}, {vty}>>;"
                    );
                }
                Some(AccessSegment::Field(_)) => {
                    // validator catches this
                }
            }
        }
    }
}

fn emit_write_imports(out: &mut String, path: &crate::access::AccessPath, manifest: &Manifest) {
    let Some(cell) = manifest.state.get(&path.cell) else {
        return;
    };
    let cell_kb = kebab(&path.cell);
    match cell {
        StateDecl::Atom { schema } => {
            let _ = writeln!(
                out,
                "  import set-{cell_kb}: func(value: {});",
                render_type_ref(schema)
            );
        }
        StateDecl::Map { key, value } => {
            let kty = render_type_ref(key);
            let vty = render_type_ref(value);
            let _ = writeln!(out, "  import put-{cell_kb}: func(key: {kty}, value: {vty});");
            let _ = writeln!(out, "  import delete-{cell_kb}: func(key: {kty});");
        }
    }
}

fn event_type_ref(event_name: &str, manifest: &Manifest) -> String {
    match manifest.events.get(event_name) {
        Some(e) => match &e.payload {
            SchemaRef::Named(n) => {
                if SchemaDef::is_primitive_name(n) {
                    render_primitive(n).to_string()
                } else {
                    kebab_ident(n)
                }
            }
            SchemaRef::Inline(_) => kebab_ident(event_name),
        },
        None => "unit".to_string(),
    }
}

fn emit_named_type(out: &mut String, name: &str, def: &SchemaDef, indent: usize) {
    let kb = kebab_ident(name);
    let pad = " ".repeat(indent);
    match def {
        SchemaDef::Record { fields } => {
            let _ = writeln!(out, "{pad}record {kb} {{");
            for (fname, ftype) in fields {
                let _ = writeln!(
                    out,
                    "{pad}  {}: {},",
                    kebab_ident(fname),
                    render_type_ref(ftype)
                );
            }
            let _ = writeln!(out, "{pad}}}");
        }
        SchemaDef::Sum { variants } => {
            let _ = writeln!(out, "{pad}variant {kb} {{");
            for (vname, payload) in variants {
                match payload {
                    None => {
                        let _ = writeln!(out, "{pad}  {},", kebab_ident(vname));
                    }
                    Some(p) => {
                        let _ = writeln!(
                            out,
                            "{pad}  {}({}),",
                            kebab_ident(vname),
                            render_type_ref(p)
                        );
                    }
                }
            }
            let _ = writeln!(out, "{pad}}}");
        }
        primitive_or_alias => {
            let _ = writeln!(
                out,
                "{pad}type {kb} = {};",
                render_inline_def(primitive_or_alias)
            );
        }
    }
}

fn render_type_ref(sr: &SchemaRef) -> String {
    match sr {
        SchemaRef::Named(name) => {
            if SchemaDef::is_primitive_name(name) {
                render_primitive(name).to_string()
            } else {
                kebab_ident(name)
            }
        }
        SchemaRef::Inline(def) => render_inline_def(def),
    }
}

fn render_inline_def(def: &SchemaDef) -> String {
    match def {
        SchemaDef::Bool => "bool".to_string(),
        SchemaDef::U32 => "u32".to_string(),
        SchemaDef::U64 => "u64".to_string(),
        SchemaDef::I32 => "s32".to_string(),
        SchemaDef::I64 => "s64".to_string(),
        SchemaDef::F32 => "float32".to_string(),
        SchemaDef::F64 => "float64".to_string(),
        SchemaDef::String => "string".to_string(),
        SchemaDef::Bytes => "list<u8>".to_string(),
        SchemaDef::Timestamp => "u64".to_string(),
        SchemaDef::List { of } => format!("list<{}>", render_type_ref(of)),
        SchemaDef::Option { of } => format!("option<{}>", render_type_ref(of)),
        SchemaDef::Map { key, value } => format!(
            "list<tuple<{}, {}>>",
            render_type_ref(key),
            render_type_ref(value)
        ),
        SchemaDef::Record { .. } | SchemaDef::Sum { .. } => {
            // v0.1: records/variants must be declared in `schemas` and
            // referenced by name. Inline ones in non-event surfaces are not
            // supported. Validator will eventually reject these.
            "INLINE_RECORD_OR_VARIANT_NOT_SUPPORTED".to_string()
        }
    }
}

fn render_primitive(name: &str) -> &'static str {
    match name {
        "bool" => "bool",
        "u32" => "u32",
        "u64" => "u64",
        "i32" => "s32",
        "i64" => "s64",
        "f32" => "float32",
        "f64" => "float64",
        "string" => "string",
        "bytes" => "list<u8>",
        "timestamp" => "u64",
        _ => unreachable!("not a primitive name: {name}"),
    }
}

/// WIT reserved words. When a manifest names a field, case, or cell using
/// one of these (in kebab form), the WIT generator prefixes the emitted
/// identifier with `%` to escape it.
const WIT_KEYWORDS: &[&str] = &[
    "use", "type", "resource", "func", "record", "enum", "flags", "variant",
    "static", "interface", "world", "import", "export", "package", "include",
    "with", "as", "from", "list", "option", "result", "tuple", "future",
    "stream", "bool", "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64",
    "f32", "f64", "char", "string", "borrow", "own", "error-context",
];

fn is_wit_keyword(s: &str) -> bool {
    WIT_KEYWORDS.contains(&s)
}

/// Like `kebab`, but prefixes the result with `%` if it collides with a WIT
/// reserved word. Use this anywhere the identifier is *used*, not declared
/// as a type name (type names are also affected; see callers).
pub fn kebab_ident(s: &str) -> String {
    let k = kebab(s);
    if is_wit_keyword(&k) {
        format!("%{k}")
    } else {
        k
    }
}

/// Convert PascalCase / snake_case / camelCase identifiers to kebab-case
/// (WIT identifier convention).
pub fn kebab(s: &str) -> String {
    let mut out = String::new();
    let mut prev_was_lower_or_digit = false;
    let mut prev_was_upper = false;
    for (i, ch) in s.chars().enumerate() {
        if ch == '_' || ch == '-' || ch == ' ' || ch == '.' {
            if !out.is_empty() && !out.ends_with('-') {
                out.push('-');
            }
            prev_was_lower_or_digit = false;
            prev_was_upper = false;
        } else if ch.is_ascii_uppercase() {
            if i > 0 && prev_was_lower_or_digit && !out.ends_with('-') {
                out.push('-');
            }
            out.push(ch.to_ascii_lowercase());
            prev_was_lower_or_digit = false;
            prev_was_upper = true;
        } else {
            // If we just exited a run of uppercase chars (>=2), insert a
            // boundary before this lowercase: "HTTPRequest" -> "http-request".
            if prev_was_upper && i >= 2 && !out.ends_with('-') {
                let last = out.pop().unwrap();
                if !out.ends_with('-') {
                    out.push('-');
                }
                out.push(last);
            }
            out.push(ch);
            prev_was_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
            prev_was_upper = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::kebab;

    #[test]
    fn kebab_basic() {
        assert_eq!(kebab("DeployRecord"), "deploy-record");
        assert_eq!(kebab("RepoId"), "repo-id");
        assert_eq!(kebab("in_progress"), "in-progress");
        assert_eq!(kebab("MergeToMain"), "merge-to-main");
        assert_eq!(kebab("HTTPRequest"), "http-request");
        assert_eq!(kebab("kick_off_deploy"), "kick-off-deploy");
        assert_eq!(kebab("u32"), "u32");
    }
}
