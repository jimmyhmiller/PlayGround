//! MLIR-free IR model + a textual printer byte-compatible with MLIR's *generic*
//! operation format (the form upstream jsir's `ast2hir` emits).
//!
//! We model exactly the subset MLIR's generic printer needs: ops with operands,
//! an attribute dictionary, nested regions/blocks, and results. The printer
//! reproduces MLIR's SSA value numbering (block-level-first within a region,
//! then descend into child regions, with the counter scoped per region so
//! sibling regions reuse numbers) and its exact whitespace/attribute syntax.

pub mod attr;
pub mod print;

pub use attr::{
    Attr, CommentAttr, ExportSpecifierAttr, ForInOfDeclarationAttr, IdentifierAttr,
    ImportSpecKind, ImportSpecifierAttr, InterpreterDirectiveAttr, NumericLiteralKeyAttr,
    PrivateNameAttr, StringLiteralKeyAttr,
};

use std::collections::HashMap;

/// A unique id for an SSA value (an op result), assigned during IR building.
/// The printer maps these to the textual `%N` numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// An operation: `%results = "name"(operands) <{attrs}> (regions) : types`.
#[derive(Debug, Clone)]
pub struct Op {
    pub name: String,
    pub operands: Vec<ValueId>,
    /// Attribute dictionary. Stored in insertion order; the printer sorts by key
    /// (MLIR's `DictionaryAttr` is alphabetically ordered).
    pub attrs: Vec<(String, Attr)>,
    pub regions: Vec<Region>,
    /// Result value ids (0 or more; jshir ops have at most 1).
    pub results: Vec<ValueId>,
    /// AST trivia (loc/offsets/comments/scope/symbols) carried verbatim through
    /// the IR, exactly like MLIR attaches a `JsirTriviaAttr` as each op's
    /// Location. The textual printer IGNORES this (it is elided by upstream's
    /// default printer too), so it does not affect `ast2hir` byte-exactness; it
    /// exists so `hir2ast` can reconstruct each AST node's base fields.
    pub trivia: Option<Trivia>,
    /// A stable, monotonically-assigned origin id, minted during `ast2hir`
    /// (`None` for ops the textual builder constructs without a node and for
    /// any synthetically-created op). This is PURE INFRASTRUCTURE: the textual
    /// printer (`print.rs`) and the inverse `hir2ast` lowering MUST ignore it
    /// entirely — it never participates in byte-exactness or any structural
    /// decision. It exists so later IR->IR transforms can map an analyzed
    /// instruction back to the JSIR op (and the enclosing statement) it came
    /// from.
    pub node_id: Option<u32>,
}

/// A source position (1-based line, 0-based column), mirroring the JSIR `loc`.
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: i64,
    pub column: i64,
}

/// A JSIR `SourceLocation`: start/end positions plus an optional name.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLoc {
    pub start: Position,
    pub end: Position,
    pub identifier_name: Option<String>,
}

/// A JSIR symbol id `{name, defScopeUid?}`.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolId {
    pub name: String,
    pub def_scope_uid: Option<i64>,
}

/// The base fields every JSIR AST node carries. `loc`/`start`/`end` are always
/// emitted (null when `None`); the rest are omitted when `None`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trivia {
    pub loc: Option<SourceLoc>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub leading_comment_uids: Option<Vec<i64>>,
    pub trailing_comment_uids: Option<Vec<i64>>,
    pub inner_comment_uids: Option<Vec<i64>>,
    pub scope_uid: Option<i64>,
    pub referenced_symbol: Option<SymbolId>,
    pub defined_symbols: Option<Vec<SymbolId>>,
}

/// A region: an ordered list of blocks (jshir regions have a single block).
#[derive(Debug, Clone, Default)]
pub struct Region {
    pub blocks: Vec<Block>,
}

/// A block: optional arguments (none in jshir) plus an ordered list of ops.
#[derive(Debug, Clone, Default)]
pub struct Block {
    pub args: Vec<ValueId>,
    pub ops: Vec<Op>,
}

impl Op {
    pub fn new(name: impl Into<String>) -> Op {
        Op {
            name: name.into(),
            operands: Vec::new(),
            attrs: Vec::new(),
            regions: Vec::new(),
            results: Vec::new(),
            trivia: None,
            node_id: None,
        }
    }

    /// Print this op as a standalone top-level operation (indent 0), the way
    /// `jsir.file` is emitted. Returns the full text WITHOUT a trailing newline.
    pub fn print(&self) -> String {
        let mut numbering = HashMap::new();
        // Number the root op's regions starting at base 0 (the root op itself
        // has no results in jsir, but number them defensively).
        let mut counter = 0u32;
        for r in &self.results {
            numbering.insert(*r, counter);
            counter += 1;
        }
        for region in &self.regions {
            print::number_region(region, counter, &mut numbering);
        }
        let mut out = String::new();
        print::print_op(self, 0, &numbering, &mut out);
        out
    }
}

impl Region {
    pub fn with_block(block: Block) -> Region {
        Region {
            blocks: vec![block],
        }
    }
}

#[cfg(test)]
mod printer_tests {
    use super::*;

    fn vid(n: u32) -> ValueId {
        ValueId(n)
    }

    fn numlit(id: u32, raw: &str, v: f64) -> Op {
        let mut op = Op::new("jsir.numeric_literal");
        op.attrs.push((
            "extra".into(),
            Attr::NumericLiteralExtra {
                raw: raw.into(),
                value: v,
            },
        ));
        op.attrs.push(("value".into(), Attr::F64(v)));
        op.results.push(vid(id));
        op
    }

    fn binexpr(id: u32, l: u32, r: u32) -> Op {
        let mut op = Op::new("jsir.binary_expression");
        op.operands.push(vid(l));
        op.operands.push(vid(r));
        op.attrs.push(("operator_".into(), Attr::Str("+".into())));
        op.results.push(vid(id));
        op
    }

    /// Reconstruct the `binary_expression` fixture (`1 + 2 + 3;`) by hand and
    /// require the printer reproduces the golden `jshir.mlir` byte-for-byte.
    #[test]
    fn prints_binary_expression_fixture() {
        let mut expr_stmt = Op::new("jsir.expression_statement");
        expr_stmt.operands.push(vid(4));

        let body = Block {
            args: vec![],
            ops: vec![
                numlit(0, "1", 1.0),
                numlit(1, "2", 2.0),
                binexpr(2, 0, 1),
                numlit(3, "3", 3.0),
                binexpr(4, 2, 3),
                expr_stmt,
            ],
        };

        let mut program = Op::new("jsir.program");
        program
            .attrs
            .push(("source_type".into(), Attr::Str("script".into())));
        program.regions.push(Region::with_block(body));
        program.regions.push(Region::with_block(Block::default())); // empty directives

        let mut file = Op::new("jsir.file");
        file.attrs.push(("comments".into(), Attr::Array(vec![])));
        file.regions
            .push(Region::with_block(Block { args: vec![], ops: vec![program] }));

        let actual = file.print();
        let expected = jsir_oracle::list_fixtures()
            .into_iter()
            .find(|f| f.name == "binary_expression")
            .unwrap()
            .expected_jshir()
            .unwrap();

        if let Some(diff) = jsir_oracle::byte_diff(&expected, &actual) {
            panic!("printer diverged:\n{diff}\n--- actual ---\n{actual}");
        }
    }
}
