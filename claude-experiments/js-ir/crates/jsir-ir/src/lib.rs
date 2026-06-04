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
    Attr, CommentAttr, DialectPayload, ExportSpecifierAttr, ForInOfDeclarationAttr, IdentifierAttr,
    ImportSpecKind, ImportSpecifierAttr, InterpreterDirectiveAttr, NumericLiteralKeyAttr,
    OpaqueAttr, PrivateNameAttr, StringLiteralKeyAttr,
};

use std::collections::HashMap;

/// A unique id for an SSA value (an op result), assigned during IR building.
/// The printer maps these to the textual `%N` numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// A region-scoped block identity. Generic CFG infrastructure (MLIR's `^bbN`):
/// terminators name their successors by `BlockId`, and the printer labels each
/// block by it. Stable across edits, unlike a positional index — passes can hold
/// a `BlockId` while inserting/removing blocks. Carries no dialect meaning. The
/// single-block AST dialect leaves every block at the default (`BlockId(0)`) and
/// never references it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct BlockId(pub u32);

/// A CFG successor edge on a terminator op: a target block plus the values passed
/// as that block's arguments. This is MLIR's block-argument form of SSA — the
/// generic analog of phi operands (each predecessor supplies the args). Generic
/// infrastructure; the AST dialect has no terminators and leaves this empty.
#[derive(Debug, Clone, PartialEq)]
pub struct Successor {
    pub block: BlockId,
    pub args: Vec<ValueId>,
}

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
    /// CFG successor edges (MLIR-style). Empty for every non-terminator and for
    /// the entire AST dialect; a CFG-dialect terminator (`br`/`cond_br`/...) lists
    /// the blocks it can branch to, each with the values bound to that block's
    /// arguments. Generic infrastructure — the core attaches no meaning to it.
    pub successors: Vec<Successor>,
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

/// A block: a region-scoped identity, optional arguments (the CFG dialect's SSA
/// merge points; none in jshir), and an ordered list of ops. In a CFG dialect a
/// block ends with a terminator op carrying `successors`.
#[derive(Debug, Clone, Default)]
pub struct Block {
    /// Region-scoped identity (see [`BlockId`]). The AST dialect leaves it at 0.
    pub id: BlockId,
    pub args: Vec<ValueId>,
    pub ops: Vec<Op>,
}

impl Block {
    /// A single straight-line block with no arguments and the default id — the
    /// shape every AST-dialect region uses.
    pub fn leaf(ops: Vec<Op>) -> Block {
        Block { id: BlockId::default(), args: Vec::new(), ops }
    }
}

impl Op {
    pub fn new(name: impl Into<String>) -> Op {
        Op {
            name: name.into(),
            operands: Vec::new(),
            attrs: Vec::new(),
            regions: Vec::new(),
            results: Vec::new(),
            successors: Vec::new(),
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

        let body = Block::leaf(vec![
            numlit(0, "1", 1.0),
            numlit(1, "2", 2.0),
            binexpr(2, 0, 1),
            numlit(3, "3", 3.0),
            binexpr(4, 2, 3),
            expr_stmt,
        ]);

        let mut program = Op::new("jsir.program");
        program
            .attrs
            .push(("source_type".into(), Attr::Str("script".into())));
        program.regions.push(Region::with_block(body));
        program.regions.push(Region::with_block(Block::default())); // empty directives

        let mut file = Op::new("jsir.file");
        file.attrs.push(("comments".into(), Attr::Array(vec![])));
        file.regions
            .push(Region::with_block(Block::leaf(vec![program])));

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

    fn cst(id: u32) -> Op {
        let mut op = Op::new("jslir.const");
        op.results.push(vid(id));
        op
    }

    /// A CFG-dialect region: entry block (label elided), two arms, and a join
    /// block taking a block argument (SSA merge). Exercises block labels, block
    /// args, and terminator successors — none of which the AST dialect uses.
    #[test]
    fn prints_cfg_region() {
        let mut cond_br = Op::new("jslir.cond_br");
        cond_br.operands.push(vid(0));
        cond_br.successors.push(Successor { block: BlockId(1), args: vec![] });
        cond_br.successors.push(Successor { block: BlockId(2), args: vec![] });

        let mut br1 = Op::new("jslir.br");
        br1.successors.push(Successor { block: BlockId(3), args: vec![vid(1)] });
        let mut br2 = Op::new("jslir.br");
        br2.successors.push(Successor { block: BlockId(3), args: vec![vid(2)] });

        let mut ret = Op::new("jslir.return");
        ret.operands.push(vid(3));

        let region = Region {
            blocks: vec![
                Block { id: BlockId(0), args: vec![], ops: vec![cst(0), cond_br] },
                Block { id: BlockId(1), args: vec![], ops: vec![cst(1), br1] },
                Block { id: BlockId(2), args: vec![], ops: vec![cst(2), br2] },
                Block { id: BlockId(3), args: vec![vid(3)], ops: vec![ret] },
            ],
        };
        let mut func = Op::new("jslir.func");
        func.regions.push(region);
        let out = func.print();

        // Entry block label is elided (no args), like MLIR.
        assert!(!out.contains("^bb0"), "entry label should be elided:\n{out}");
        // Successor lists on terminators.
        assert!(out.contains("\"jslir.cond_br\"(%0)[^bb1, ^bb2]"), "{out}");
        assert!(out.contains("\"jslir.br\"()[^bb3(%1)]"), "{out}");
        assert!(out.contains("\"jslir.br\"()[^bb3(%2)]"), "{out}");
        // Plain block labels and a block-argument (phi) label.
        assert!(out.contains("\n^bb1:\n"), "{out}");
        assert!(out.contains("\n^bb2:\n"), "{out}");
        assert!(out.contains("\n^bb3(%3: !jsir.any):\n"), "{out}");
        assert!(out.contains("\"jslir.return\"(%3)"), "{out}");
    }

    /// A dialect attaches arbitrary state via the opaque escape hatch; the core
    /// renders, downcasts, and structurally compares it without knowing its type.
    #[test]
    fn opaque_dialect_attr() {
        #[derive(Debug, PartialEq)]
        struct MutRange {
            start: u32,
            end: u32,
        }
        impl DialectPayload for MutRange {
            fn render(&self) -> String {
                format!("#jslir<mut_range {}..{}>", self.start, self.end)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn dyn_eq(&self, other: &dyn DialectPayload) -> bool {
                other.as_any().downcast_ref::<MutRange>() == Some(self)
            }
        }

        let a = Attr::Opaque(OpaqueAttr::new(MutRange { start: 2, end: 9 }));
        assert_eq!(a.render(), "#jslir<mut_range 2..9>");

        // The owning dialect recovers its concrete type.
        if let Attr::Opaque(o) = &a {
            assert_eq!(o.downcast_ref::<MutRange>().unwrap().end, 9);
        } else {
            panic!("expected opaque");
        }

        // Structural equality flows through `dyn_eq` so `Attr: PartialEq` still holds.
        let b = Attr::Opaque(OpaqueAttr::new(MutRange { start: 2, end: 9 }));
        let c = Attr::Opaque(OpaqueAttr::new(MutRange { start: 2, end: 10 }));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
