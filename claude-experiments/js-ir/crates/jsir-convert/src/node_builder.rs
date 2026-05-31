//! `NodeBuilder`: the [`AstBuilder`] that builds the in-tree [`jsir_ast::Node`].
//! This keeps `hir2ast`'s byte-exact round-trip oracle (IR -> Node -> ast.json)
//! intact: it reconstructs base fields from trivia and assembles node-specific
//! fields in schema order, exactly as the old free-function `build_node` did.

use crate::ast_builder::AstBuilder;
use crate::LowerResult;
use jsir_ast::model::{ExtraVal, FieldValue, Presence};
use jsir_ast::schema_generated::{helper_schema, node_schema};
use jsir_ast::Node;
use jsir_ir::{Position, SourceLoc, SymbolId, Trivia};

/// The base (trivia-derived) field keys, handled from `trivia` rather than from
/// the node-specific fields.
const BASE_KEYS: &[&str] = &[
    "loc",
    "start",
    "end",
    "leadingCommentUids",
    "trailingCommentUids",
    "innerCommentUids",
    "scopeUid",
    "referencedSymbol",
    "definedSymbols",
];

pub struct NodeBuilder;

impl AstBuilder for NodeBuilder {
    type Out = FieldValue;

    fn node(
        &self,
        ty: &str,
        trivia: Option<&Trivia>,
        fields: Vec<(&'static str, FieldValue)>,
    ) -> LowerResult<FieldValue> {
        let schema = node_schema(ty);
        if schema.is_empty() {
            return Err(format!("NodeBuilder: unknown node type {ty}"));
        }
        let lookup = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone());
        let mut out = Vec::with_capacity(schema.len());
        for spec in schema {
            let fv = if BASE_KEYS.contains(&spec.key) {
                base_field(spec.key, trivia)
            } else {
                lookup(spec.key).unwrap_or_else(|| match spec.presence {
                    // A specific field not provided defaults to its "absent"
                    // form: omitted for MaybeUndef, explicit null otherwise.
                    Presence::MaybeUndef => FieldValue::Absent,
                    _ => FieldValue::Null,
                })
            };
            out.push(fv);
        }
        Ok(FieldValue::Node(Box::new(Node {
            ty: ty.to_string(),
            fields: out,
        })))
    }

    fn record(&self, name: &str, fields: Vec<(&'static str, FieldValue)>) -> FieldValue {
        let schema = helper_schema(name);
        let lookup = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone());
        let out = schema
            .iter()
            .map(|spec| lookup(spec.key).unwrap_or(FieldValue::Absent))
            .collect();
        FieldValue::Extra(Box::new(ExtraVal {
            name: name.to_string(),
            fields: out,
        }))
    }

    fn string(&self, s: String) -> FieldValue {
        FieldValue::Str(s)
    }
    fn int(&self, i: i64) -> FieldValue {
        FieldValue::Int(i)
    }
    fn float(&self, f: f64) -> FieldValue {
        FieldValue::Float(f)
    }
    fn boolean(&self, b: bool) -> FieldValue {
        FieldValue::Bool(b)
    }
    fn list(&self, items: Vec<FieldValue>) -> FieldValue {
        FieldValue::List(items)
    }
    fn null(&self) -> FieldValue {
        FieldValue::Null
    }
    fn absent(&self) -> FieldValue {
        FieldValue::Absent
    }
}

impl NodeBuilder {
    /// Extract the built `Node` from the top-level result.
    pub fn into_node(out: FieldValue) -> LowerResult<Node> {
        match out {
            FieldValue::Node(n) => Ok(*n),
            other => Err(format!("expected a node, built {other:?}")),
        }
    }
}

/// Reconstruct a base field's value from trivia (loc/start/end emit null when
/// absent; the rest are omitted when absent).
fn base_field(key: &str, trivia: Option<&Trivia>) -> FieldValue {
    let Some(t) = trivia else {
        return match key {
            "loc" | "start" | "end" => FieldValue::Null,
            _ => FieldValue::Absent,
        };
    };
    match key {
        "loc" => t.loc.as_ref().map(source_loc_to_fv).unwrap_or(FieldValue::Null),
        "start" => t.start.map(FieldValue::Int).unwrap_or(FieldValue::Null),
        "end" => t.end.map(FieldValue::Int).unwrap_or(FieldValue::Null),
        "leadingCommentUids" => uid_list(&t.leading_comment_uids),
        "trailingCommentUids" => uid_list(&t.trailing_comment_uids),
        "innerCommentUids" => uid_list(&t.inner_comment_uids),
        "scopeUid" => t.scope_uid.map(FieldValue::Int).unwrap_or(FieldValue::Absent),
        "referencedSymbol" => t
            .referenced_symbol
            .as_ref()
            .map(symbol_id_to_fv)
            .unwrap_or(FieldValue::Absent),
        "definedSymbols" => t
            .defined_symbols
            .as_ref()
            .map(|syms| FieldValue::List(syms.iter().map(symbol_id_to_fv).collect()))
            .unwrap_or(FieldValue::Absent),
        _ => FieldValue::Absent,
    }
}

fn uid_list(uids: &Option<Vec<i64>>) -> FieldValue {
    match uids {
        Some(v) => FieldValue::List(v.iter().map(|i| FieldValue::Int(*i)).collect()),
        None => FieldValue::Absent,
    }
}

fn source_loc_to_fv(loc: &SourceLoc) -> FieldValue {
    let pos = |p: &Position| {
        extra_node(
            "Position",
            vec![("line", FieldValue::Int(p.line)), ("column", FieldValue::Int(p.column))],
        )
    };
    let ident = match &loc.identifier_name {
        Some(s) => FieldValue::Str(s.clone()),
        None => FieldValue::Absent,
    };
    extra_node(
        "SourceLocation",
        vec![("start", pos(&loc.start)), ("end", pos(&loc.end)), ("identifierName", ident)],
    )
}

fn symbol_id_to_fv(sym: &SymbolId) -> FieldValue {
    let scope = sym.def_scope_uid.map(FieldValue::Int).unwrap_or(FieldValue::Absent);
    extra_node(
        "SymbolId",
        vec![("name", FieldValue::Str(sym.name.clone())), ("defScopeUid", scope)],
    )
}

/// Build a fixed helper struct value in its schema order (used for the trivia
/// records above; the public path is `AstBuilder::record`).
fn extra_node(name: &str, fields: Vec<(&str, FieldValue)>) -> FieldValue {
    let schema = helper_schema(name);
    let lookup = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone());
    let out = schema
        .iter()
        .map(|spec| lookup(spec.key).unwrap_or(FieldValue::Absent))
        .collect();
    FieldValue::Extra(Box::new(ExtraVal {
        name: name.to_string(),
        fields: out,
    }))
}
