//! Typed-field accessors over the [`AstNode`] trait. Lowering is written against
//! these, so it never names a concrete AST type: any parser front-end that
//! implements [`AstNode`] is lowered by the same code.

use super::LowerResult;
use jsir_ast::{AstNode, Field};

/// A required field: errors if the node has no such field. An explicit `null`
/// is returned as [`Field::Null`] (callers that need a value will reject it via
/// the typed extractors below).
pub fn field<'a>(node: &'a dyn AstNode, key: &str) -> LowerResult<Field<'a>> {
    match node.field(key) {
        Field::Absent => Err(format!("{}: missing field {key}", node.node_type())),
        other => Ok(other),
    }
}

/// Field access on a helper record (`loc`, `extra`, ...). Records are `AstNode`s
/// too, so this is just [`field`] under a name that documents intent.
pub fn extra_field<'a>(e: &'a dyn AstNode, key: &str) -> LowerResult<Field<'a>> {
    field(e, key)
}

pub fn str_of(fv: Field<'_>) -> LowerResult<&str> {
    match fv {
        Field::Str(s) => Ok(s),
        other => Err(format!("expected string, got {}", kind(&other))),
    }
}

pub fn f64_of(fv: Field<'_>) -> LowerResult<f64> {
    match fv {
        Field::Float(f) => Ok(f),
        Field::Int(i) => Ok(i as f64),
        other => Err(format!("expected number, got {}", kind(&other))),
    }
}

pub fn i64_of(fv: Field<'_>) -> LowerResult<i64> {
    match fv {
        Field::Int(i) => Ok(i),
        other => Err(format!("expected int, got {}", kind(&other))),
    }
}

/// A node-or-record value. Helper records and real nodes are both `AstNode`.
pub fn node_of(fv: Field<'_>) -> LowerResult<&dyn AstNode> {
    match fv {
        Field::Node(n) => Ok(n),
        other => Err(format!("expected node, got {}", kind(&other))),
    }
}

/// Alias of [`node_of`]: helper records (`loc`, `extra`, ...) are `AstNode`s.
pub fn extra_of(fv: Field<'_>) -> LowerResult<&dyn AstNode> {
    node_of(fv)
}

pub fn bool_of(fv: Field<'_>) -> LowerResult<bool> {
    match fv {
        Field::Bool(b) => Ok(b),
        other => Err(format!("expected bool, got {}", kind(&other))),
    }
}

pub fn list_of(fv: Field<'_>) -> LowerResult<Vec<Field<'_>>> {
    match fv {
        Field::List(v) => Ok(v),
        other => Err(format!("expected array, got {}", kind(&other))),
    }
}

/// The first element of a list field, by value (lists are owned views).
pub fn first(list: Vec<Field<'_>>) -> LowerResult<Field<'_>> {
    list.into_iter()
        .next()
        .ok_or_else(|| "expected non-empty array".to_string())
}

/// A short tag for error messages (the old `{:?}` of `FieldValue`).
fn kind(f: &Field<'_>) -> &'static str {
    match f {
        Field::Absent => "absent",
        Field::Null => "null",
        Field::Bool(_) => "bool",
        Field::Int(_) => "int",
        Field::Float(_) => "float",
        Field::Str(_) => "string",
        Field::Node(_) => "node",
        Field::List(_) => "array",
    }
}
