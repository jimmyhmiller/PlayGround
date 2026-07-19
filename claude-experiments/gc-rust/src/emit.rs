//! Structured IR dumps — the compiler "X-ray" taps.
//!
//! Every pipeline stage can be emitted as JSON so tooling and agents can
//! inspect *what the compiler actually produced* instead of reading 150KB of
//! `codegen.rs` and simulating the pipeline by hand. Wired to the CLI as
//! `gcr emit <stage> <file>`.
//!
//! Stages currently tapped:
//!   - `tokens`  — the raw lexer output (`Vec<Token>`)
//!   - `ast`     — the surface AST of the user's own module (no prelude)
//!   - `core`    — the full monomorphic Core IR (`CoreProgram`)
//!   - `layout`  — heap object layouts (GC shapes) + inline value layouts
//!   - `mono`    — the monomorphization table (every instance, grouped by base)
//!
//! The IR types derive `serde::Serialize`, so `core`/`tokens`/`ast` are direct
//! dumps; `layout`/`mono` are curated views built here for legibility.

use crate::core::CoreProgram;
use serde_json::{json, Value};

/// Pretty-print any serializable value as JSON (2-space indent).
pub fn pretty<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v)
        .unwrap_or_else(|e| format!("{{\"error\":\"serialize failed: {e}\"}}"))
}

/// The `layout` stage: every heap layout (its `type_id` is its index in the
/// runtime type table) with its pointer-fields-first shape, plus the inline
/// value-aggregate layouts. This is exactly what the GC traces and what a
/// heap-object box diagram would render.
pub fn layouts_view(prog: &CoreProgram) -> Value {
    let layouts: Vec<Value> = prog
        .layouts
        .iter()
        .enumerate()
        .map(|(id, l)| {
            json!({
                "type_id": id,
                "name": l.name,
                "ptr_fields": l.ptr_fields,
                "raw_bytes": l.raw_bytes,
                "varlen": format!("{:?}", l.varlen),
                "elem_stride": l.elem_stride,
                "field_map": serde_json::to_value(&l.field_map).unwrap_or(Value::Null),
            })
        })
        .collect();
    json!({
        "layout_count": prog.layouts.len(),
        "value_count": prog.values.len(),
        "layouts": layouts,
        "values": serde_json::to_value(&prog.values).unwrap_or(Value::Null),
    })
}

/// The `reflect` stage: the runtime reflection metadata table — exactly what is
/// installed into the heap (JIT) or baked into the executable (AOT) so that
/// heap-exploration tooling and in-language reflection can recover nominal
/// type/field information the GC type table omits. One entry per `type_id`,
/// built from each layout's [`crate::core::Layout::meta`].
pub fn reflect_view(prog: &CoreProgram) -> Value {
    use crate::gc::{FieldMeta, FieldTy, TypeKind, TypeMeta};

    fn field_json(f: &FieldMeta) -> Value {
        let ty = match f.ty {
            FieldTy::Ref(tid) => json!({ "kind": "ref", "type_id": tid }),
            FieldTy::Scalar(s) => json!({ "kind": "scalar", "scalar": s.as_str() }),
            FieldTy::Value(vid) => json!({ "kind": "value", "value_id": vid }),
        };
        json!({ "name": f.name, "offset": f.offset, "ty": ty })
    }

    fn meta_json(type_id: usize, m: &TypeMeta) -> Value {
        match &m.kind {
            TypeKind::Struct { fields } => json!({
                "type_id": type_id,
                "name": m.name,
                "kind": "struct",
                "fields": fields.iter().map(field_json).collect::<Vec<_>>(),
            }),
            TypeKind::Enum { tag_offset, variants } => json!({
                "type_id": type_id,
                "name": m.name,
                "kind": "enum",
                "tag_offset": tag_offset,
                "variants": variants.iter().map(|v| json!({
                    "name": v.name,
                    "tag": v.tag,
                    "fields": v.fields.iter().map(field_json).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            }),
            TypeKind::Opaque => json!({
                "type_id": type_id,
                "name": m.name,
                "kind": "opaque",
            }),
        }
    }

    let types: Vec<Value> = prog
        .layouts
        .iter()
        .enumerate()
        .map(|(id, l)| meta_json(id, &l.meta))
        .collect();
    // Value-aggregate table (inline `#[value]` structs/enums), with the same
    // per-entry shape, computed by the codegen bridge so it matches exactly what
    // the runtime installs.
    let value_metas = crate::codegen::layouts_to_value_meta(prog);
    let values: Vec<Value> = value_metas
        .iter()
        .map(|m| {
            let v = TypeMeta { type_id: m.value_id, name: m.name.clone(), kind: m.kind.clone() };
            meta_json(m.value_id as usize, &v)
        })
        .collect();
    json!({
        "type_count": prog.layouts.len(),
        "types": types,
        "value_count": value_metas.len(),
        "values": values,
    })
}

/// The `mono` stage: the monomorphization table. Every monomorphic function
/// instance, grouped by its generic *base* name (the mangling is `base$Args`),
/// so you can see at a glance which generics were instantiated and how many
/// times. Answers "what did `vec_map` get specialized to?" without reading the
/// lowerer.
pub fn mono_table(prog: &CoreProgram) -> Value {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (id, f) in prog.funcs.iter().enumerate() {
        let base = f.name.split('$').next().unwrap_or(&f.name).to_string();
        groups.entry(base).or_default().push(json!({
            "func_id": id,
            "mangled": f.name,
            "params": serde_json::to_value(&f.params).unwrap_or(Value::Null),
            "ret": serde_json::to_value(&f.ret).unwrap_or(Value::Null),
            "is_extern": f.is_extern,
            "is_closure": !f.closure_captures.is_empty(),
        }));
    }
    let multi = groups.values().filter(|v| v.len() > 1).count();
    let bases: Vec<Value> = groups
        .into_iter()
        .map(|(base, instances)| {
            json!({
                "base": base,
                "instance_count": instances.len(),
                "instances": instances,
            })
        })
        .collect();
    json!({
        "function_count": prog.funcs.len(),
        "base_count": bases.len(),
        "bases_with_multiple_instances": multi,
        "entry": prog.entry,
        "bases": bases,
    })
}
