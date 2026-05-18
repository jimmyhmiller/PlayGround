//! Pure rendering of IR data into human-readable text. No I/O, no clap; the
//! `main` binary owns those concerns.

use std::fmt::Write as _;

use ir::access::{AccessSegment, KeyBinding};
use ir::manifest::{EffectDef, EventDef, Handler, Manifest, StateDecl};
use ir::schema::{SchemaDef, SchemaRef};
use ir::validate;
use ir::AccessPath;

// ----------------------------------------------------------------------------
// Program map
// ----------------------------------------------------------------------------

/// The at-a-glance view: events, state, effects, handlers (with declared
/// footprints).
pub fn program_map(m: &Manifest) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Program: {} v{}", m.name, m.version);
    let _ = writeln!(out);

    section(&mut out, "EVENTS", m.events.len());
    for (name, ev) in &m.events {
        let _ = writeln!(out, "  {name:<28}  payload: {}", short_type(&ev.payload));
    }
    if m.events.is_empty() {
        let _ = writeln!(out, "  (none)");
    }
    let _ = writeln!(out);

    section(&mut out, "STATE", m.state.len());
    for (name, decl) in &m.state {
        let _ = writeln!(out, "  {name:<28}  {}", state_kind(decl));
    }
    if m.state.is_empty() {
        let _ = writeln!(out, "  (none)");
    }
    let _ = writeln!(out);

    section(&mut out, "EFFECTS", m.effects.len());
    for (name, ef) in &m.effects {
        let _ = writeln!(
            out,
            "  {name:<28}  {} -> {}{}",
            short_type(&ef.request),
            short_type(&ef.response),
            response_event_suffix(ef),
        );
    }
    if m.effects.is_empty() {
        let _ = writeln!(out, "  (none)");
    }
    let _ = writeln!(out);

    section(&mut out, "HANDLERS", m.handlers.len());
    for h in &m.handlers {
        let _ = writeln!(out, "  {}", h.name);
        let _ = writeln!(out, "    on:    {}", h.on);
        if !h.read.is_empty() {
            let _ = writeln!(out, "    read:  {}", format_paths(&h.read));
        }
        if !h.write.is_empty() {
            let _ = writeln!(out, "    write: {}", format_paths(&h.write));
        }
        if !h.emit.is_empty() {
            let _ = writeln!(out, "    emit:  {}", h.emit.join(", "));
        }
    }
    if m.handlers.is_empty() {
        let _ = writeln!(out, "  (none)");
    }

    out
}

// ----------------------------------------------------------------------------
// Handler card
// ----------------------------------------------------------------------------

/// Full declared footprint of a single handler plus its body pointer.
pub fn handler_card(m: &Manifest, name: &str) -> Option<String> {
    let h = m.handlers.iter().find(|h| h.name == name)?;
    let mut out = String::new();

    let _ = writeln!(out, "Handler: {}", h.name);
    let _ = writeln!(out, "  on:    {}", h.on);
    if let Some(ev) = m.events.get(&h.on) {
        let _ = writeln!(out, "         payload: {}", short_type(&ev.payload));
    }
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "  read   ({} path{})",
        h.read.len(),
        if h.read.len() == 1 { "" } else { "s" }
    );
    if h.read.is_empty() {
        let _ = writeln!(out, "    (none)");
    } else {
        for p in &h.read {
            let _ = writeln!(out, "    {}", p);
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "  write  ({} path{})",
        h.write.len(),
        if h.write.len() == 1 { "" } else { "s" }
    );
    if h.write.is_empty() {
        let _ = writeln!(out, "    (none)");
    } else {
        for p in &h.write {
            let _ = writeln!(out, "    {}", p);
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "  emit   ({} effect{})",
        h.emit.len(),
        if h.emit.len() == 1 { "" } else { "s" }
    );
    if h.emit.is_empty() {
        let _ = writeln!(out, "    (none)");
    } else {
        for e in &h.emit {
            let _ = writeln!(out, "    {}", e);
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "  body");
    let _ = writeln!(out, "    hash: {}", h.body.hash);
    let _ = writeln!(out, "    uri:  {}", h.body.uri);

    Some(out)
}

// ----------------------------------------------------------------------------
// State cell view
// ----------------------------------------------------------------------------

/// The reverse view: which handlers touch this cell, in which mode.
pub fn state_cell(m: &Manifest, cell: &str) -> Option<String> {
    let decl = m.state.get(cell)?;
    let mut out = String::new();

    let _ = writeln!(out, "State: {cell}");
    let _ = writeln!(out, "  kind: {}", state_kind(decl));
    let _ = writeln!(out);

    let mut readers: Vec<(&str, String)> = Vec::new();
    let mut writers: Vec<(&str, String)> = Vec::new();
    for h in &m.handlers {
        for p in &h.read {
            if p.cell == cell {
                readers.push((h.name.as_str(), key_shape(p)));
            }
        }
        for p in &h.write {
            if p.cell == cell {
                writers.push((h.name.as_str(), key_shape(p)));
            }
        }
    }

    let _ = writeln!(out, "  READERS ({})", readers.len());
    if readers.is_empty() {
        let _ = writeln!(out, "    (none)");
    } else {
        for (name, shape) in readers {
            let _ = writeln!(out, "    {name:<28}  {shape}");
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "  WRITERS ({})", writers.len());
    if writers.is_empty() {
        let _ = writeln!(out, "    (none)");
    } else {
        for (name, shape) in writers {
            let _ = writeln!(out, "    {name:<28}  {shape}");
        }
    }

    Some(out)
}

// ----------------------------------------------------------------------------
// Validate
// ----------------------------------------------------------------------------

/// `(ok, report)`. `ok` is false if the manifest has issues.
pub fn validate_report(m: &Manifest) -> (bool, String) {
    match validate::validate(m) {
        Ok(()) => (true, format!("{} v{}: OK\n", m.name, m.version)),
        Err(err) => {
            let mut out = String::new();
            let _ = writeln!(out, "{} v{}: INVALID ({} issue{})", m.name, m.version, err.issues().len(), if err.issues().len() == 1 { "" } else { "s" });
            for i in err.issues() {
                let _ = writeln!(out, "  - {i}");
            }
            (false, out)
        }
    }
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

fn section(out: &mut String, name: &str, count: usize) {
    let _ = writeln!(out, "{name} ({count})");
}

fn format_paths(paths: &[AccessPath]) -> String {
    paths
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn state_kind(decl: &StateDecl) -> String {
    match decl {
        StateDecl::Atom { schema } => format!("Atom<{}>", short_type(schema)),
        StateDecl::Map { key, value } => {
            format!("Map<{}, {}>", short_type(key), short_type(value))
        }
    }
}

fn short_type(sr: &SchemaRef) -> String {
    match sr {
        SchemaRef::Named(name) => name.clone(),
        SchemaRef::Inline(def) => short_inline(def),
    }
}

fn short_inline(def: &SchemaDef) -> String {
    match def {
        SchemaDef::Bool => "bool".into(),
        SchemaDef::U32 => "u32".into(),
        SchemaDef::U64 => "u64".into(),
        SchemaDef::I32 => "i32".into(),
        SchemaDef::I64 => "i64".into(),
        SchemaDef::F32 => "f32".into(),
        SchemaDef::F64 => "f64".into(),
        SchemaDef::String => "string".into(),
        SchemaDef::Bytes => "bytes".into(),
        SchemaDef::Timestamp => "timestamp".into(),
        SchemaDef::List { of } => format!("List<{}>", short_type(of)),
        SchemaDef::Option { of } => format!("Option<{}>", short_type(of)),
        SchemaDef::Map { key, value } => format!("Map<{}, {}>", short_type(key), short_type(value)),
        SchemaDef::Record { fields } => {
            let names: Vec<String> = fields.keys().cloned().collect();
            format!("{{ {} }}", names.join(", "))
        }
        SchemaDef::Sum { variants } => {
            let names: Vec<String> = variants.keys().cloned().collect();
            format!("<{}>", names.join(" | "))
        }
    }
}

fn key_shape(p: &AccessPath) -> String {
    match p.segments.first() {
        None => "whole-cell".into(),
        Some(AccessSegment::Field(_)) => p.to_string(),
        Some(AccessSegment::Wildcard) => "[*]".into(),
        Some(AccessSegment::Key(KeyBinding::Event(path))) => {
            format!("[$event.{}]", path.join("."))
        }
    }
}

fn response_event_suffix(ef: &EffectDef) -> String {
    ef.response_event
        .as_ref()
        .map(|n| format!("   (response_event: {n})"))
        .unwrap_or_default()
}

// Unused import suppression dance:
fn _refs(_: &EventDef, _: &Handler) {}
