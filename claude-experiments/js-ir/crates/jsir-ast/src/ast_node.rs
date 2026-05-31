//! The `AstNode` trait: the *only* surface `ast2hir` needs from an AST.
//!
//! Lowering reads an AST through exactly two operations: the node's type tag and
//! a named-field lookup. Capturing that as a trait means any parser's own tree
//! (the in-tree [`Node`], swc, oxc, ...) can be lowered to IR **directly**, with
//! no intermediate owned AST to materialize or keep in sync. A parser front-end
//! is just an `impl AstNode` over its native types.
//!
//! Field values are returned as a borrowed [`Field`] view, built on demand, so an
//! implementation never has to allocate a parallel tree: it answers each
//! `field(key)` by borrowing from whatever it already holds.

use crate::model::{ExtraVal, FieldValue, Node};

/// The read-only AST view that `ast2hir` lowers from.
///
/// `node_type` returns the Babel node-type tag (`"UnaryExpression"`,
/// `"Identifier"`, ...) and `field` looks up a child/attribute by its Babel JSON
/// key. Helper records (`loc`, `extra`, `referencedSymbol`, positions, ...) are
/// themselves `AstNode`s whose `node_type` is the record name; this keeps the
/// trait uniform so callers walk nodes and sub-records the same way.
pub trait AstNode {
    /// The Babel node-type tag for this node (or the record name, for helpers).
    fn node_type(&self) -> &str;

    /// The value of field `key`, or [`Field::Absent`] when this node has no such
    /// field. An explicit JSON `null` (e.g. an optional `id`) is [`Field::Null`].
    fn field(&self, key: &str) -> Field<'_>;
}

/// A borrowed view of one field value, produced on demand by [`AstNode::field`].
///
/// Sub-nodes and helper records both arrive as [`Field::Node`]; an
/// implementation borrows them from its own storage, so no field access ever
/// allocates an owned node.
pub enum Field<'a> {
    /// Field is not part of this node (Babel would omit it).
    Absent,
    /// Field is present but explicitly null (also used for array holes).
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'a str),
    Node(&'a dyn AstNode),
    List(Vec<Field<'a>>),
}

/// Borrow a [`FieldValue`] (the in-tree representation) as a [`Field`] view.
fn view(fv: &FieldValue) -> Field<'_> {
    match fv {
        FieldValue::Absent => Field::Absent,
        FieldValue::Null => Field::Null,
        FieldValue::Bool(b) => Field::Bool(*b),
        FieldValue::Int(i) => Field::Int(*i),
        FieldValue::Float(f) => Field::Float(*f),
        FieldValue::Str(s) => Field::Str(s),
        FieldValue::Node(n) => Field::Node(&**n),
        FieldValue::Extra(e) => Field::Node(&**e),
        FieldValue::List(v) => Field::List(v.iter().map(view).collect()),
    }
}

impl AstNode for Node {
    fn node_type(&self) -> &str {
        &self.ty
    }
    fn field(&self, key: &str) -> Field<'_> {
        match Node::field(self, key) {
            Some(fv) => view(fv),
            None => Field::Absent,
        }
    }
}

impl AstNode for ExtraVal {
    fn node_type(&self) -> &str {
        &self.name
    }
    fn field(&self, key: &str) -> Field<'_> {
        match ExtraVal::field(self, key) {
            Some(fv) => view(fv),
            None => Field::Absent,
        }
    }
}
