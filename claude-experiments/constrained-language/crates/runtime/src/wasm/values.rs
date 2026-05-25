//! Schema-guided conversion between `serde_json::Value` and
//! `wasmtime::component::Val`.
//!
//! The two boundaries this crosses:
//! * On the way in (host → guest): convert a JSON value to a `Val` whose
//!   shape matches what the WIT-declared import expects. Field/case names
//!   are kebab-cased to match the WIT generator's output.
//! * On the way out (guest → host): convert a `Val` returned by a host
//!   function call (or by the body itself) back to JSON, schema-guided so
//!   we know which JSON shape to produce.
//!
//! Sum variants follow the convention used in the manifest fixtures:
//! `{ "tag": "name" }` for unit cases, `{ "tag": "name", "value": <payload> }`
//! for payloaded cases.
//!
//! Map values are exchanged as `list<tuple<K, V>>` (matching `ir::wit`).

use serde_json::Value;
use wasmtime::component::Val;

use ir::manifest::Manifest;
use ir::schema::{SchemaDef, SchemaRef};
use ir::wit::kebab;

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("schema reference `{0}` does not resolve to any known type")]
    UnresolvedSchema(String),
    #[error("expected {expected}, got JSON value `{got}`")]
    BadJson { expected: String, got: String },
    #[error("expected {expected}, got Val `{got:?}`")]
    BadVal { expected: String, got: Val },
    #[error("variant case `{0}` not declared in schema")]
    UnknownVariantCase(String),
    #[error("variant JSON is missing required `tag` field: `{0}`")]
    VariantMissingTag(String),
    #[error("u32/u64 value `{0}` out of range")]
    NumberOutOfRange(String),
}

/// Convert a JSON `Value` to a wasmtime component `Val`, using the schema
/// to disambiguate variant cases, record field order, etc.
pub fn json_to_val(
    json: &Value,
    schema: &SchemaRef,
    manifest: &Manifest,
) -> Result<Val, ConvertError> {
    let def = resolve(schema, manifest)?;
    json_to_val_def(json, &def, manifest)
}

fn json_to_val_def(
    json: &Value,
    def: &SchemaDef,
    manifest: &Manifest,
) -> Result<Val, ConvertError> {
    match def {
        SchemaDef::Bool => match json {
            Value::Bool(b) => Ok(Val::Bool(*b)),
            other => Err(ConvertError::BadJson {
                expected: "bool".into(),
                got: short(other),
            }),
        },
        SchemaDef::U32 => coerce_uint(json, "u32").and_then(|n| {
            u32::try_from(n)
                .map(Val::U32)
                .map_err(|_| ConvertError::NumberOutOfRange(n.to_string()))
        }),
        SchemaDef::U64 | SchemaDef::Timestamp => coerce_uint(json, "u64").map(Val::U64),
        SchemaDef::I32 => coerce_int(json, "i32").and_then(|n| {
            i32::try_from(n)
                .map(Val::S32)
                .map_err(|_| ConvertError::NumberOutOfRange(n.to_string()))
        }),
        SchemaDef::I64 => coerce_int(json, "i64").map(Val::S64),
        SchemaDef::F32 => coerce_float(json, "f32").map(|f| Val::Float32(f as f32)),
        SchemaDef::F64 => coerce_float(json, "f64").map(Val::Float64),
        SchemaDef::String => match json {
            Value::String(s) => Ok(Val::String(s.clone())),
            other => Err(ConvertError::BadJson {
                expected: "string".into(),
                got: short(other),
            }),
        },
        SchemaDef::Bytes => match json {
            Value::Array(arr) => {
                let mut bytes = Vec::with_capacity(arr.len());
                for v in arr {
                    let n = coerce_uint(v, "u8")?;
                    bytes.push(Val::U8(u8::try_from(n).map_err(|_| {
                        ConvertError::NumberOutOfRange(n.to_string())
                    })?));
                }
                Ok(Val::List(bytes))
            }
            other => Err(ConvertError::BadJson {
                expected: "list<u8>".into(),
                got: short(other),
            }),
        },
        SchemaDef::Record { fields } => {
            let obj = json.as_object().ok_or_else(|| ConvertError::BadJson {
                expected: "record (json object)".into(),
                got: short(json),
            })?;
            let mut out: Vec<(String, Val)> = Vec::with_capacity(fields.len());
            for (fname, ftype) in fields {
                let raw = obj.get(fname).cloned().unwrap_or(Value::Null);
                let val = json_to_val(&raw, ftype, manifest)?;
                out.push((kebab(fname), val));
            }
            Ok(Val::Record(out))
        }
        SchemaDef::Sum { variants } => {
            let obj = json.as_object().ok_or_else(|| ConvertError::BadJson {
                expected: "variant (json object with `tag`)".into(),
                got: short(json),
            })?;
            let tag = obj
                .get("tag")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ConvertError::VariantMissingTag(short(json)))?;
            let payload_schema = variants
                .get(tag)
                .ok_or_else(|| ConvertError::UnknownVariantCase(tag.to_string()))?;
            let case_kb = kebab(tag);
            match payload_schema {
                None => Ok(Val::Variant(case_kb, None)),
                Some(ps) => {
                    let raw = obj.get("value").cloned().unwrap_or(Value::Null);
                    let inner = json_to_val(&raw, ps, manifest)?;
                    Ok(Val::Variant(case_kb, Some(Box::new(inner))))
                }
            }
        }
        SchemaDef::List { of } => {
            let arr = json.as_array().ok_or_else(|| ConvertError::BadJson {
                expected: "list (json array)".into(),
                got: short(json),
            })?;
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                out.push(json_to_val(v, of, manifest)?);
            }
            Ok(Val::List(out))
        }
        SchemaDef::Option { of } => match json {
            Value::Null => Ok(Val::Option(None)),
            other => {
                let inner = json_to_val(other, of, manifest)?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
        },
        SchemaDef::Map { key, value } => {
            // Exchanged as list<tuple<K, V>>. JSON may be either an array of
            // [k, v] pairs (preserves arbitrary key types) or a string-keyed
            // object (only valid when K is string).
            let pairs: Vec<(Value, Value)> = match json {
                Value::Array(arr) => arr
                    .iter()
                    .map(|p| {
                        let a = p.as_array().ok_or_else(|| ConvertError::BadJson {
                            expected: "map entry [k, v]".into(),
                            got: short(p),
                        })?;
                        if a.len() != 2 {
                            return Err(ConvertError::BadJson {
                                expected: "map entry [k, v]".into(),
                                got: short(p),
                            });
                        }
                        Ok((a[0].clone(), a[1].clone()))
                    })
                    .collect::<Result<_, _>>()?,
                Value::Object(map) => map
                    .iter()
                    .map(|(k, v)| (Value::String(k.clone()), v.clone()))
                    .collect(),
                other => {
                    return Err(ConvertError::BadJson {
                        expected: "map (json array of pairs or object)".into(),
                        got: short(other),
                    })
                }
            };
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                let kv = json_to_val(&k, key, manifest)?;
                let vv = json_to_val(&v, value, manifest)?;
                out.push(Val::Tuple(vec![kv, vv]));
            }
            Ok(Val::List(out))
        }
    }
}

/// Convert a wasmtime `Val` back into a JSON `Value`, using the schema to
/// disambiguate field name un-kebabing, variant case un-kebabing, etc.
pub fn val_to_json(
    val: &Val,
    schema: &SchemaRef,
    manifest: &Manifest,
) -> Result<Value, ConvertError> {
    let def = resolve(schema, manifest)?;
    val_to_json_def(val, &def, manifest)
}

fn val_to_json_def(
    val: &Val,
    def: &SchemaDef,
    manifest: &Manifest,
) -> Result<Value, ConvertError> {
    match (def, val) {
        (SchemaDef::Bool, Val::Bool(b)) => Ok(Value::Bool(*b)),
        (SchemaDef::U32, Val::U32(n)) => Ok(Value::from(*n)),
        (SchemaDef::U64, Val::U64(n)) | (SchemaDef::Timestamp, Val::U64(n)) => {
            Ok(Value::from(*n))
        }
        (SchemaDef::I32, Val::S32(n)) => Ok(Value::from(*n)),
        (SchemaDef::I64, Val::S64(n)) => Ok(Value::from(*n)),
        (SchemaDef::F32, Val::Float32(f)) => Ok(serde_json::json!(*f)),
        (SchemaDef::F64, Val::Float64(f)) => Ok(serde_json::json!(*f)),
        (SchemaDef::String, Val::String(s)) => Ok(Value::String(s.clone())),
        (SchemaDef::Bytes, Val::List(items)) => {
            let mut bytes = Vec::with_capacity(items.len());
            for item in items {
                if let Val::U8(b) = item {
                    bytes.push(*b);
                } else {
                    return Err(ConvertError::BadVal {
                        expected: "list<u8>".into(),
                        got: item.clone(),
                    });
                }
            }
            // Match the JSON convention used by inputs: array of u8.
            Ok(Value::Array(
                bytes.into_iter().map(|b| Value::from(b)).collect(),
            ))
        }
        (SchemaDef::Record { fields }, Val::Record(items)) => {
            // The component returns kebab-cased field names; re-key by the
            // original schema names.
            let mut obj = serde_json::Map::with_capacity(fields.len());
            for (orig_name, ftype) in fields {
                let kb = kebab(orig_name);
                let found = items
                    .iter()
                    .find(|(n, _)| n == &kb)
                    .map(|(_, v)| v)
                    .ok_or_else(|| ConvertError::BadVal {
                        expected: format!("record with field `{kb}`"),
                        got: val.clone(),
                    })?;
                obj.insert(orig_name.clone(), val_to_json(found, ftype, manifest)?);
            }
            Ok(Value::Object(obj))
        }
        (SchemaDef::Sum { variants }, Val::Variant(case_kb, payload)) => {
            // Find the original case name whose kebab matches.
            let (orig_name, payload_schema) = variants
                .iter()
                .find(|(n, _)| kebab(n) == *case_kb)
                .map(|(n, p)| (n.clone(), p.clone()))
                .ok_or_else(|| ConvertError::UnknownVariantCase(case_kb.clone()))?;
            let mut obj = serde_json::Map::new();
            obj.insert("tag".to_string(), Value::String(orig_name));
            match (payload_schema, payload) {
                (Some(ps), Some(boxed)) => {
                    obj.insert("value".to_string(), val_to_json(boxed, &ps, manifest)?);
                }
                (None, None) => {}
                (Some(_), None) => {
                    return Err(ConvertError::BadVal {
                        expected: "variant case with payload".into(),
                        got: val.clone(),
                    })
                }
                (None, Some(_)) => {
                    return Err(ConvertError::BadVal {
                        expected: "unit variant case".into(),
                        got: val.clone(),
                    })
                }
            }
            Ok(Value::Object(obj))
        }
        (SchemaDef::List { of }, Val::List(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(val_to_json(it, of, manifest)?);
            }
            Ok(Value::Array(out))
        }
        (SchemaDef::Option { of }, Val::Option(inner)) => match inner {
            None => Ok(Value::Null),
            Some(boxed) => val_to_json(boxed, of, manifest),
        },
        (SchemaDef::Map { key, value }, Val::List(items)) => {
            // list<tuple<K, V>> back to JSON. We always emit array-of-pairs
            // form so non-string keys round-trip.
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                let Val::Tuple(tup) = it else {
                    return Err(ConvertError::BadVal {
                        expected: "tuple<k, v>".into(),
                        got: it.clone(),
                    });
                };
                if tup.len() != 2 {
                    return Err(ConvertError::BadVal {
                        expected: "tuple<k, v>".into(),
                        got: it.clone(),
                    });
                }
                let k = val_to_json(&tup[0], key, manifest)?;
                let v = val_to_json(&tup[1], value, manifest)?;
                out.push(Value::Array(vec![k, v]));
            }
            Ok(Value::Array(out))
        }
        (def, val) => Err(ConvertError::BadVal {
            expected: format!("{def:?}"),
            got: val.clone(),
        }),
    }
}

fn resolve(sr: &SchemaRef, m: &Manifest) -> Result<SchemaDef, ConvertError> {
    match sr {
        SchemaRef::Named(name) => {
            if let Some(p) = SchemaDef::primitive_by_name(name) {
                return Ok(p);
            }
            m.schemas
                .get(name)
                .cloned()
                .ok_or_else(|| ConvertError::UnresolvedSchema(name.clone()))
        }
        SchemaRef::Inline(def) => Ok((**def).clone()),
    }
}

fn coerce_uint(v: &Value, ctx: &str) -> Result<u64, ConvertError> {
    v.as_u64().ok_or_else(|| ConvertError::BadJson {
        expected: ctx.into(),
        got: short(v),
    })
}

fn coerce_int(v: &Value, ctx: &str) -> Result<i64, ConvertError> {
    v.as_i64().ok_or_else(|| ConvertError::BadJson {
        expected: ctx.into(),
        got: short(v),
    })
}

fn coerce_float(v: &Value, ctx: &str) -> Result<f64, ConvertError> {
    v.as_f64().ok_or_else(|| ConvertError::BadJson {
        expected: ctx.into(),
        got: short(v),
    })
}

fn short(v: &Value) -> String {
    let s = serde_json::to_string(v).unwrap_or_else(|_| "?".into());
    if s.len() > 80 {
        format!("{}…", &s[..80])
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use serde_json::json;

    fn empty_manifest() -> Manifest {
        Manifest {
            name: "t".into(),
            version: "0".into(),
            schemas: IndexMap::new(),
            events: IndexMap::new(),
            state: IndexMap::new(),
            effects: IndexMap::new(),
            handlers: Vec::new(),
            generators: IndexMap::new(),
        }
    }

    #[test]
    fn round_trip_record() {
        let mut fields = IndexMap::new();
        fields.insert("name".to_string(), SchemaRef::Named("string".into()));
        fields.insert("count".to_string(), SchemaRef::Named("u32".into()));
        let schema = SchemaRef::Inline(Box::new(SchemaDef::Record { fields }));
        let m = empty_manifest();

        let j = json!({ "name": "Alice", "count": 7 });
        let v = json_to_val(&j, &schema, &m).unwrap();
        // Field names are kebab-cased on the way in.
        if let Val::Record(items) = &v {
            assert_eq!(items[0].0, "name");
            assert_eq!(items[1].0, "count");
        } else {
            panic!("expected record, got {v:?}");
        }
        let back = val_to_json(&v, &schema, &m).unwrap();
        assert_eq!(back, j);
    }

    #[test]
    fn round_trip_unit_variant() {
        let mut variants = IndexMap::new();
        variants.insert("queued".to_string(), None);
        variants.insert("running".to_string(), None);
        let schema = SchemaRef::Inline(Box::new(SchemaDef::Sum { variants }));
        let m = empty_manifest();

        let j = json!({ "tag": "running" });
        let v = json_to_val(&j, &schema, &m).unwrap();
        assert_eq!(v, Val::Variant("running".into(), None));
        let back = val_to_json(&v, &schema, &m).unwrap();
        assert_eq!(back, j);
    }

    #[test]
    fn round_trip_payloaded_variant() {
        let mut variants = IndexMap::new();
        variants.insert("ok".to_string(), Some(SchemaRef::Named("u32".into())));
        variants.insert("err".to_string(), Some(SchemaRef::Named("string".into())));
        let schema = SchemaRef::Inline(Box::new(SchemaDef::Sum { variants }));
        let m = empty_manifest();

        let j = json!({ "tag": "ok", "value": 42 });
        let v = json_to_val(&j, &schema, &m).unwrap();
        assert_eq!(v, Val::Variant("ok".into(), Some(Box::new(Val::U32(42)))));
        let back = val_to_json(&v, &schema, &m).unwrap();
        assert_eq!(back, j);
    }

    #[test]
    fn option_and_list() {
        let m = empty_manifest();
        let schema = SchemaRef::Inline(Box::new(SchemaDef::Option {
            of: SchemaRef::Named("string".into()),
        }));
        assert_eq!(json_to_val(&json!(null), &schema, &m).unwrap(), Val::Option(None));
        assert_eq!(
            json_to_val(&json!("hi"), &schema, &m).unwrap(),
            Val::Option(Some(Box::new(Val::String("hi".into()))))
        );

        let schema = SchemaRef::Inline(Box::new(SchemaDef::List {
            of: SchemaRef::Named("u32".into()),
        }));
        let v = json_to_val(&json!([1, 2, 3]), &schema, &m).unwrap();
        assert_eq!(v, Val::List(vec![Val::U32(1), Val::U32(2), Val::U32(3)]));
        let back = val_to_json(&v, &schema, &m).unwrap();
        assert_eq!(back, json!([1, 2, 3]));
    }

    #[test]
    fn map_round_trip() {
        let m = empty_manifest();
        let schema = SchemaRef::Inline(Box::new(SchemaDef::Map {
            key: SchemaRef::Named("string".into()),
            value: SchemaRef::Named("u32".into()),
        }));
        let j = json!([["a", 1], ["b", 2]]);
        let v = json_to_val(&j, &schema, &m).unwrap();
        assert_eq!(
            v,
            Val::List(vec![
                Val::Tuple(vec![Val::String("a".into()), Val::U32(1)]),
                Val::Tuple(vec![Val::String("b".into()), Val::U32(2)]),
            ])
        );
        let back = val_to_json(&v, &schema, &m).unwrap();
        assert_eq!(back, j);
    }
}
