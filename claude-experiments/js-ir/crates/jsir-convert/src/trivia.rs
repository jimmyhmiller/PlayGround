//! Capturing the JSIR base fields (`loc`, offsets, comments, scope, symbols) as
//! [`jsir_ir::Trivia`], so they survive the AST -> IR -> AST round trip
//! (mirroring how upstream stores a `JsirTriviaAttr` as each op's location).
//!
//! Read through the [`AstNode`] trait, so trivia is captured identically no
//! matter which parser produced the node.

use crate::accessors::*;
use crate::LowerResult;
use jsir_ast::{AstNode, Field};
use jsir_ir::{Position, SourceLoc, SymbolId, Trivia};

/// Read every base field of an AST node into a [`Trivia`].
pub fn node_trivia(node: &dyn AstNode) -> LowerResult<Trivia> {
    Ok(Trivia {
        loc: opt_node(node, "loc").map(source_loc).transpose()?,
        start: opt_i64_field(node, "start"),
        end: opt_i64_field(node, "end"),
        leading_comment_uids: opt_i64_list(node, "leadingCommentUids"),
        trailing_comment_uids: opt_i64_list(node, "trailingCommentUids"),
        inner_comment_uids: opt_i64_list(node, "innerCommentUids"),
        scope_uid: opt_i64_field(node, "scopeUid"),
        referenced_symbol: opt_node(node, "referencedSymbol").map(symbol_id).transpose()?,
        defined_symbols: match node.field("definedSymbols") {
            Field::List(items) => Some(
                items
                    .into_iter()
                    .map(|fv| symbol_id(node_of(fv)?))
                    .collect::<LowerResult<Vec<_>>>()?,
            ),
            _ => None,
        },
    })
}

fn source_loc(loc: &dyn AstNode) -> LowerResult<SourceLoc> {
    Ok(SourceLoc {
        start: position(loc, "start")?,
        end: position(loc, "end")?,
        identifier_name: match loc.field("identifierName") {
            Field::Str(s) => Some(s.to_string()),
            _ => None,
        },
    })
}

fn position(loc: &dyn AstNode, key: &str) -> LowerResult<Position> {
    let p = extra_of(extra_field(loc, key)?)?;
    Ok(Position {
        line: i64_of(extra_field(p, "line")?)?,
        column: i64_of(extra_field(p, "column")?)?,
    })
}

fn symbol_id(sym: &dyn AstNode) -> LowerResult<SymbolId> {
    Ok(SymbolId {
        name: str_of(extra_field(sym, "name")?)?.to_string(),
        def_scope_uid: match sym.field("defScopeUid") {
            Field::Int(i) => Some(i),
            _ => None,
        },
    })
}

/// `Some(child)` when `node.key` holds a node/record value, else `None`.
fn opt_node<'a>(node: &'a dyn AstNode, key: &str) -> Option<&'a dyn AstNode> {
    match node.field(key) {
        Field::Node(n) => Some(n),
        _ => None,
    }
}

fn opt_i64_field(node: &dyn AstNode, key: &str) -> Option<i64> {
    match node.field(key) {
        Field::Int(i) => Some(i),
        _ => None,
    }
}

fn opt_i64_list(node: &dyn AstNode, key: &str) -> Option<Vec<i64>> {
    match node.field(key) {
        Field::List(items) => Some(
            items
                .into_iter()
                .filter_map(|fv| match fv {
                    Field::Int(i) => Some(i),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}
