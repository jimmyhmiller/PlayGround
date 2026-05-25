use std::collections::HashSet;

use thiserror::Error;

use crate::access::{AccessPath, AccessSegment, KeyBinding};
use crate::manifest::{Handler, Manifest, StateDecl};
use crate::schema::{SchemaDef, SchemaRef};

/// One problem found while validating a manifest. The validator collects
/// every issue rather than failing fast — easier to fix authoring errors in
/// bulk.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Issue {
    #[error("unresolved schema reference `{0}` (used in {1})")]
    UnresolvedSchema(String, String),
    #[error("user schema `{0}` shadows built-in primitive of the same name")]
    PrimitiveShadowed(String),
    #[error("handler `{handler}` references unknown event `{event}`")]
    UnknownEvent { handler: String, event: String },
    #[error("handler `{handler}` references unknown effect `{effect}` in `emit`")]
    UnknownEffect { handler: String, effect: String },
    #[error("access path `{path}` (in handler `{handler}`) names unknown state cell `{cell}`")]
    UnknownCell {
        handler: String,
        path: String,
        cell: String,
    },
    #[error(
        "access path `{path}` (in handler `{handler}`) uses a key segment on Atom cell `{cell}`; \
         Atom cells are addressed by field only"
    )]
    KeyOnAtom {
        handler: String,
        path: String,
        cell: String,
    },
    #[error(
        "access path `{path}` (in handler `{handler}`) starts with a field on Map cell `{cell}`; \
         Map cells must be addressed by `[<key>]` or `[*]` first"
    )]
    FieldOnMapRoot {
        handler: String,
        path: String,
        cell: String,
    },
    #[error(
        "access path `{path}` (in handler `{handler}`) binds `$event.{field}` but event `{event}` has no such field"
    )]
    EventFieldMissing {
        handler: String,
        path: String,
        event: String,
        field: String,
    },
    #[error(
        "access path `{path}` (in handler `{handler}`) navigates into a non-record type via `.{field}`"
    )]
    FieldOnNonRecord {
        handler: String,
        path: String,
        field: String,
    },
    #[error("duplicate handler name `{0}`")]
    DuplicateHandler(String),
    #[error("effect `{effect}` declares response_event `{event}` but that event is not defined")]
    EffectResponseEventMissing { effect: String, event: String },
    #[error("generator `{generator}` produces unknown event `{event}`")]
    GeneratorUnknownEvent { generator: String, event: String },
}

#[derive(Debug, Error)]
#[error("manifest is invalid ({} issue{})", .issues.len(), if .issues.len() == 1 { "" } else { "s" })]
pub struct ValidationError {
    pub issues: Vec<Issue>,
}

impl ValidationError {
    pub fn issues(&self) -> &[Issue] {
        &self.issues
    }
}

pub fn validate(manifest: &Manifest) -> Result<(), ValidationError> {
    let mut issues = Vec::new();

    // Schema names must not shadow primitives.
    for name in manifest.schemas.keys() {
        if SchemaDef::is_primitive_name(name) {
            issues.push(Issue::PrimitiveShadowed(name.clone()));
        }
    }

    // Every SchemaRef appearing anywhere must resolve.
    for (sname, sdef) in &manifest.schemas {
        check_schema_def(sdef, manifest, &format!("schema `{sname}`"), &mut issues);
    }
    for (ename, edef) in &manifest.events {
        check_schema_ref(
            &edef.payload,
            manifest,
            &format!("event `{ename}`"),
            &mut issues,
        );
    }
    for (cname, cdecl) in &manifest.state {
        match cdecl {
            StateDecl::Atom { schema } => check_schema_ref(
                schema,
                manifest,
                &format!("state `{cname}`"),
                &mut issues,
            ),
            StateDecl::Map { key, value } => {
                check_schema_ref(key, manifest, &format!("state `{cname}` key"), &mut issues);
                check_schema_ref(
                    value,
                    manifest,
                    &format!("state `{cname}` value"),
                    &mut issues,
                );
            }
        }
    }
    for (fxname, fxdef) in &manifest.effects {
        check_schema_ref(
            &fxdef.request,
            manifest,
            &format!("effect `{fxname}` request"),
            &mut issues,
        );
        check_schema_ref(
            &fxdef.response,
            manifest,
            &format!("effect `{fxname}` response"),
            &mut issues,
        );
        if let Some(ev) = &fxdef.response_event {
            if !manifest.events.contains_key(ev) {
                issues.push(Issue::EffectResponseEventMissing {
                    effect: fxname.clone(),
                    event: ev.clone(),
                });
            }
        }
    }

    // Handlers.
    let mut seen_handlers: HashSet<&str> = HashSet::new();
    for h in &manifest.handlers {
        if !seen_handlers.insert(h.name.as_str()) {
            issues.push(Issue::DuplicateHandler(h.name.clone()));
        }
        check_handler(h, manifest, &mut issues);
    }

    // Generators.
    for (gname, gdecl) in &manifest.generators {
        if !manifest.events.contains_key(&gdecl.event) {
            issues.push(Issue::GeneratorUnknownEvent {
                generator: gname.clone(),
                event: gdecl.event.clone(),
            });
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(ValidationError { issues })
    }
}

fn check_handler(h: &Handler, m: &Manifest, issues: &mut Vec<Issue>) {
    if !m.events.contains_key(&h.on) {
        issues.push(Issue::UnknownEvent {
            handler: h.name.clone(),
            event: h.on.clone(),
        });
    }
    for fx in &h.emit {
        if !m.effects.contains_key(fx) {
            issues.push(Issue::UnknownEffect {
                handler: h.name.clone(),
                effect: fx.clone(),
            });
        }
    }
    for p in &h.read {
        check_access_path(p, h, m, issues);
    }
    for p in &h.write {
        check_access_path(p, h, m, issues);
    }
}

fn check_access_path(p: &AccessPath, h: &Handler, m: &Manifest, issues: &mut Vec<Issue>) {
    let Some(cell) = m.state.get(&p.cell) else {
        issues.push(Issue::UnknownCell {
            handler: h.name.clone(),
            path: p.to_string(),
            cell: p.cell.clone(),
        });
        return;
    };

    let (current, mut remaining): (SchemaRef, &[AccessSegment]) = match cell {
        StateDecl::Atom { schema } => {
            if let Some(AccessSegment::Key(_) | AccessSegment::Wildcard) = p.segments.first() {
                issues.push(Issue::KeyOnAtom {
                    handler: h.name.clone(),
                    path: p.to_string(),
                    cell: p.cell.clone(),
                });
                return;
            }
            (schema.clone(), p.segments.as_slice())
        }
        StateDecl::Map { value, .. } => match p.segments.first() {
            Some(AccessSegment::Key(binding)) => {
                let KeyBinding::Event(field_path) = binding;
                let event = match m.events.get(&h.on) {
                    Some(e) => e,
                    None => return,
                };
                check_event_path(&event.payload, field_path, h, p, m, issues);
                (value.clone(), &p.segments[1..])
            }
            Some(AccessSegment::Wildcard) => (value.clone(), &p.segments[1..]),
            Some(AccessSegment::Field(_)) | None => {
                if matches!(p.segments.first(), Some(AccessSegment::Field(_))) {
                    issues.push(Issue::FieldOnMapRoot {
                        handler: h.name.clone(),
                        path: p.to_string(),
                        cell: p.cell.clone(),
                    });
                    return;
                }
                // No segments at all on a Map: treated as "the whole map", same as `[*]`.
                (value.clone(), &p.segments[..])
            }
        },
    };

    let mut cursor = current;
    while let Some((first, rest)) = remaining.split_first() {
        match first {
            AccessSegment::Field(name) => {
                let resolved = match resolve(&cursor, m) {
                    Some(d) => d,
                    None => return, // already reported as UnresolvedSchema
                };
                match resolved {
                    SchemaDef::Record { fields } => {
                        if let Some(next) = fields.get(name) {
                            cursor = next.clone();
                        } else {
                            issues.push(Issue::FieldOnNonRecord {
                                handler: h.name.clone(),
                                path: p.to_string(),
                                field: name.clone(),
                            });
                            return;
                        }
                    }
                    _ => {
                        issues.push(Issue::FieldOnNonRecord {
                            handler: h.name.clone(),
                            path: p.to_string(),
                            field: name.clone(),
                        });
                        return;
                    }
                }
            }
            AccessSegment::Key(_) | AccessSegment::Wildcard => {
                // Nested indexing not supported in v0.1.
                issues.push(Issue::KeyOnAtom {
                    handler: h.name.clone(),
                    path: p.to_string(),
                    cell: p.cell.clone(),
                });
                return;
            }
        }
        remaining = rest;
    }
}

fn check_event_path(
    event_payload: &SchemaRef,
    field_path: &[String],
    h: &Handler,
    p: &AccessPath,
    m: &Manifest,
    issues: &mut Vec<Issue>,
) {
    let mut cursor = event_payload.clone();
    for field in field_path {
        let resolved = match resolve(&cursor, m) {
            Some(d) => d,
            None => return,
        };
        match resolved {
            SchemaDef::Record { fields } => match fields.get(field) {
                Some(next) => cursor = next.clone(),
                None => {
                    issues.push(Issue::EventFieldMissing {
                        handler: h.name.clone(),
                        path: p.to_string(),
                        event: h.on.clone(),
                        field: field.clone(),
                    });
                    return;
                }
            },
            _ => {
                issues.push(Issue::EventFieldMissing {
                    handler: h.name.clone(),
                    path: p.to_string(),
                    event: h.on.clone(),
                    field: field.clone(),
                });
                return;
            }
        }
    }
}

fn check_schema_def(d: &SchemaDef, m: &Manifest, ctx: &str, issues: &mut Vec<Issue>) {
    match d {
        SchemaDef::Record { fields } => {
            for sr in fields.values() {
                check_schema_ref(sr, m, ctx, issues);
            }
        }
        SchemaDef::Sum { variants } => {
            for sr in variants.values().flatten() {
                check_schema_ref(sr, m, ctx, issues);
            }
        }
        SchemaDef::List { of }
        | SchemaDef::Option { of } => check_schema_ref(of, m, ctx, issues),
        SchemaDef::Map { key, value } => {
            check_schema_ref(key, m, ctx, issues);
            check_schema_ref(value, m, ctx, issues);
        }
        _ => {}
    }
}

fn check_schema_ref(sr: &SchemaRef, m: &Manifest, ctx: &str, issues: &mut Vec<Issue>) {
    match sr {
        SchemaRef::Named(name) => {
            if SchemaDef::is_primitive_name(name) {
                return;
            }
            if !m.schemas.contains_key(name) {
                issues.push(Issue::UnresolvedSchema(name.clone(), ctx.to_string()));
            }
        }
        SchemaRef::Inline(def) => check_schema_def(def, m, ctx, issues),
    }
}

/// Resolve a `SchemaRef` to its `SchemaDef`, following named references one
/// level (named-to-named chains are not allowed by the JSON shape since named
/// entries in `manifest.schemas` are `SchemaDef`, not `SchemaRef`).
fn resolve(sr: &SchemaRef, m: &Manifest) -> Option<SchemaDef> {
    match sr {
        SchemaRef::Named(name) => {
            if let Some(p) = SchemaDef::primitive_by_name(name) {
                return Some(p);
            }
            m.schemas.get(name).cloned()
        }
        SchemaRef::Inline(def) => Some((**def).clone()),
    }
}
