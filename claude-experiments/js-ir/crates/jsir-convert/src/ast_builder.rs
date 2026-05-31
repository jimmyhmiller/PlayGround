//! The `AstBuilder` trait: the *only* surface `hir2ast` needs to construct an
//! AST. It is the output-side dual of [`jsir_ast::AstNode`] (the input trait):
//! where lowering *reads* an AST through `AstNode`, lifting *builds* one through
//! `AstBuilder`. Any backend (the in-tree [`jsir_ast::Node`], swc, oxc, ...) can
//! be the build target, so the IR lifts straight into the backend's own tree
//! with no intermediate AST to materialize.
//!
//! Every value `hir2ast` produces is the builder's associated `Out` type, made
//! via `node`/`record`/`list`/scalar constructors. This unifies "a node" and "a
//! field value" into one type, so lifting never names a concrete AST node.

use crate::LowerResult;
use jsir_ir::Trivia;

/// Constructs AST values for a backend. `Out` is the backend's universal value
/// (a node, a list, or a scalar); `node`/`record` build composite values from
/// already-built children.
pub trait AstBuilder {
    /// A built value: a node, a helper record, a list, or a scalar.
    type Out: Clone;

    /// Build a node of Babel type `ty` carrying `trivia` (loc/offsets/scope/...,
    /// which a source-printing backend may ignore) and the given named fields.
    fn node(
        &self,
        ty: &str,
        trivia: Option<&Trivia>,
        fields: Vec<(&'static str, Self::Out)>,
    ) -> LowerResult<Self::Out>;

    /// Build a helper record (`SourceLocation`, `Position`, a literal's `extra`,
    /// a symbol id, ...). Backends that only print source can ignore these.
    fn record(&self, name: &str, fields: Vec<(&'static str, Self::Out)>) -> Self::Out;

    fn string(&self, s: String) -> Self::Out;
    fn int(&self, i: i64) -> Self::Out;
    fn float(&self, f: f64) -> Self::Out;
    fn boolean(&self, b: bool) -> Self::Out;
    fn list(&self, items: Vec<Self::Out>) -> Self::Out;
    /// An explicit `null` (e.g. an optional `id` that is absent but emitted).
    fn null(&self) -> Self::Out;
    /// An omitted field (Babel would not emit it).
    fn absent(&self) -> Self::Out;
}
