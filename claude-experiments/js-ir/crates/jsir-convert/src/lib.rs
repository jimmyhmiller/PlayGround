//! `ast2hir`: lower the JSIR AST ([`jsir_ast::Node`]) into the JSIR/JSHIR IR
//! ([`jsir_ir::Op`]), reproducing upstream's post-order, l-value/r-value,
//! region-building lowering. The result, printed by `jsir-ir`, is byte-exact
//! with upstream's `ast2hir` output.

use jsir_ast::{AstNode, Field};
use jsir_ir::{
    Attr, Block, CommentAttr, ExportSpecifierAttr, ForInOfDeclarationAttr, IdentifierAttr,
    ImportSpecKind, ImportSpecifierAttr, InterpreterDirectiveAttr, NumericLiteralKeyAttr, Op,
    PrivateNameAttr, Region, StringLiteralKeyAttr, ValueId,
};

mod accessors;
pub mod ast_builder;
mod hir2ast;
mod node_builder;
mod trivia;
use accessors::*;
pub use ast_builder::AstBuilder;
pub use hir2ast::{hir2ast, hir2ast_with};
pub use node_builder::NodeBuilder;
use trivia::node_trivia;

/// Create an op carrying `node`'s trivia. This is the analog of upstream's
/// `CreateExpr`/`CreateStmt` op-construction helpers: every node-backed op is
/// built from its AST node, so the trivia (loc/offsets/comments/scope/symbols)
/// can never be forgotten. Region terminators / `jsir.none` use plain `Op::new`
/// (no node), matching upstream's `getUnknownLoc()` for those.
fn node_op(name: &str, node: &dyn AstNode) -> LowerResult<Op> {
    let mut op = Op::new(name);
    op.trivia = Some(node_trivia(node)?);
    Ok(op)
}

/// Error raised when lowering hits a construct not yet implemented.
pub type LowerResult<T> = Result<T, String>;

/// Assigns unique value ids; the printer renumbers them per MLIR's scheme.
struct Builder {
    next: u32,
}

impl Builder {
    fn fresh(&mut self) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        v
    }
}

/// Lower a `File` AST node to the top-level `jsir.file` op.
pub fn ast2hir(file: &dyn AstNode) -> LowerResult<Op> {
    if file.node_type() != "File" {
        return Err(format!("expected File, got {}", file.node_type()));
    }
    let mut b = Builder { next: 0 };
    let program = node_of(field(file, "program")?)?;
    let prog_op = lower_program(&mut b, program)?;

    let mut file_op = node_op("jsir.file", file)?;
    // (node_id assignment happens after the whole tree is built; see below.)
    // Comments: the `File.comments` list becomes a `comment_line`/`comment_block`
    // attribute array (empty when absent).
    let mut comment_attrs = Vec::new();
    if let Field::List(items) = file.field("comments") {
        for it in items {
            if let Field::Node(c) = it {
                comment_attrs.push(comment_attr(c)?);
            }
        }
    }
    file_op
        .attrs
        .push(("comments".into(), Attr::Array(comment_attrs)));
    file_op.regions.push(Region::with_block(Block::leaf(vec![prog_op])));

    // STEP 0 (Phase B): assign a stable, monotonically-increasing `node_id` to
    // every op in the built tree. We piggyback the same Builder counter used for
    // ValueIds (`b.next`) so the seed advances past the last value id; the walk
    // itself is a deterministic pre-order traversal (root, then each region's
    // blocks' ops in order, recursing). This is pure provenance: the textual
    // printer and `hir2ast` ignore `node_id`, so byte-exactness/round-trip are
    // unaffected.
    assign_node_ids(&mut file_op, &mut b);

    Ok(file_op)
}

/// Pre-order, deterministic assignment of `node_id` to every op in the tree,
/// drawing ids from the Builder's monotonic counter.
fn assign_node_ids(op: &mut Op, b: &mut Builder) {
    op.node_id = Some(b.next);
    b.next += 1;
    for region in &mut op.regions {
        for block in &mut region.blocks {
            for child in &mut block.ops {
                assign_node_ids(child, b);
            }
        }
    }
}

fn lower_program(b: &mut Builder, prog: &dyn AstNode) -> LowerResult<Op> {
    let source_type = str_of(field(prog, "sourceType")?)?;
    let mut body = Vec::new();
    for el in list_of(field(prog, "body")?)? {
        lower_stmt(b, &mut body, node_of(el)?)?;
    }
    let mut directives = Vec::new();
    if let Ok(dirs) = field(prog, "directives").and_then(list_of) {
        for el in dirs {
            lower_directive(b, &mut directives, node_of(el)?)?;
        }
    }
    let mut op = node_op("jsir.program", prog)?;
    if let Field::Node(interp) = prog.field("interpreter") {
        let loc = extra_of(field(interp, "loc")?)?;
        let (start_line, start_col) = position(loc, "start")?;
        let (end_line, end_col) = position(loc, "end")?;
        op.attrs.push((
            "interpreter".into(),
            Attr::InterpreterDirective(Box::new(InterpreterDirectiveAttr {
                start_line,
                start_col,
                end_line,
                end_col,
                start_index: i64_of(field(interp, "start")?)?,
                end_index: i64_of(field(interp, "end")?)?,
                value: str_of(field(interp, "value")?)?.to_string(),
            })),
        ));
    }
    op.attrs
        .push(("source_type".into(), Attr::Str(source_type.to_string())));
    op.regions.push(Region::with_block(Block::leaf(body)));
    op.regions.push(Region::with_block(Block::leaf(directives)));
    Ok(op)
}

fn lower_directive(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<()> {
    // Directive { value: DirectiveLiteral { value, extra } }.
    let dl = node_of(field(node, "value")?)?;
    let mut vop = node_op("jsir.directive_literal", dl)?;
    if let Field::Node(extra) = dl.field("extra") {
        vop.attrs.push((
            "extra".into(),
            Attr::DirectiveLiteralExtra {
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
                raw_value: str_of(extra_field(extra, "rawValue")?)?.to_string(),
            },
        ));
    }
    vop.attrs
        .push(("value".into(), Attr::Str(str_of(field(dl, "value")?)?.to_string())));
    let v = emit(ops, b, vop)?;
    let dir = unary_op("jsir.directive", node, v)?;
    ops.push(dir);
    Ok(())
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

fn lower_stmt(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<()> {
    match node.node_type() {
        "ExpressionStatement" => {
            let v = lower_expr(b, ops, node_of(field(node, "expression")?)?)?;
            ops.push(unary_op("jsir.expression_statement", node, v)?);
            Ok(())
        }
        "EmptyStatement" => {
            ops.push(node_op("jsir.empty_statement", node)?);
            Ok(())
        }
        "DebuggerStatement" => {
            ops.push(node_op("jsir.debugger_statement", node)?);
            Ok(())
        }
        "VariableDeclaration" => {
            ops.push(lower_variable_declaration(b, node)?);
            Ok(())
        }
        "FunctionDeclaration" => {
            let op = lower_function(b, node, "jsir.function_declaration")?;
            ops.push(op);
            Ok(())
        }
        "ClassDeclaration" => {
            let op = lower_class(b, ops, node, "jsir.class_declaration")?;
            ops.push(op);
            Ok(())
        }
        "BlockStatement" => {
            ops.push(lower_block_statement(b, node)?);
            Ok(())
        }
        "WhileStatement" => {
            let mut op = node_op("jshir.while_statement", node)?;
            op.regions
                .push(expr_region(b, node_of(field(node, "test")?)?, |bb, ops, n| {
                    lower_expr(bb, ops, n)
                })?);
            op.regions
                .push(stmt_region(b, node_of(field(node, "body")?)?)?);
            ops.push(op);
            Ok(())
        }
        "DoWhileStatement" => {
            let mut op = node_op("jshir.do_while_statement", node)?;
            op.regions
                .push(stmt_region(b, node_of(field(node, "body")?)?)?);
            op.regions
                .push(expr_region(b, node_of(field(node, "test")?)?, |bb, ops, n| {
                    lower_expr(bb, ops, n)
                })?);
            ops.push(op);
            Ok(())
        }
        "IfStatement" => {
            let test = lower_expr(b, ops, node_of(field(node, "test")?)?)?;
            let mut op = node_op("jshir.if_statement", node)?;
            op.operands.push(test);
            op.regions
                .push(stmt_region(b, node_of(field(node, "consequent")?)?)?);
            // Alternate region: present -> its block; absent -> zero-block region.
            match node.field("alternate") {
                Field::Node(alt) => {
                    op.regions.push(stmt_region(b, alt)?);
                }
                _ => op.regions.push(Region::default()),
            }
            ops.push(op);
            Ok(())
        }
        "ReturnStatement" => {
            let mut op = node_op("jsir.return_statement", node)?;
            if let Field::Node(arg) = node.field("argument") {
                let v = lower_expr(b, ops, arg)?;
                op.operands.push(v);
            }
            ops.push(op);
            Ok(())
        }
        "ThrowStatement" => {
            let v = lower_expr(b, ops, node_of(field(node, "argument")?)?)?;
            ops.push(unary_op("jsir.throw_statement", node, v)?);
            Ok(())
        }
        "WithStatement" => {
            let obj = lower_expr(b, ops, node_of(field(node, "object")?)?)?;
            let mut op = node_op("jshir.with_statement", node)?;
            op.operands.push(obj);
            op.regions
                .push(stmt_region(b, node_of(field(node, "body")?)?)?);
            ops.push(op);
            Ok(())
        }
        "LabeledStatement" => {
            let mut op = node_op("jshir.labeled_statement", node)?;
            op.attrs.push((
                "label".into(),
                identifier_attr(node_of(field(node, "label")?)?)?,
            ));
            op.regions
                .push(stmt_region(b, node_of(field(node, "body")?)?)?);
            ops.push(op);
            Ok(())
        }
        "BreakStatement" => {
            ops.push(break_continue("jshir.break_statement", node)?);
            Ok(())
        }
        "ContinueStatement" => {
            ops.push(break_continue("jshir.continue_statement", node)?);
            Ok(())
        }
        "ForStatement" => {
            let mut op = node_op("jshir.for_statement", node)?;
            // init: an UnknownRegion holding a var-decl statement, or an
            // expression terminated by `expr_region_end`; absent -> zero blocks.
            op.regions.push(match node.field("init") {
                Field::Node(init) => {
                    let mut rops = Vec::new();
                    if init.node_type() == "VariableDeclaration" {
                        rops.push(lower_variable_declaration(b, init)?);
                    } else {
                        let v = lower_expr(b, &mut rops, init)?;
                        let mut end = Op::new("jsir.expr_region_end");
                        end.operands.push(v);
                        rops.push(end);
                    }
                    Region::with_block(Block::leaf(rops))
                }
                _ => Region::default(),
            });
            op.regions.push(opt_expr_region(b, node.field("test"))?);
            op.regions.push(opt_expr_region(b, node.field("update"))?);
            op.regions
                .push(stmt_region(b, node_of(field(node, "body")?)?)?);
            ops.push(op);
            Ok(())
        }
        "ExportDefaultDeclaration" => {
            let decl = node_of(field(node, "declaration")?)?;
            let mut rops = Vec::new();
            if decl.node_type() == "FunctionDeclaration" || decl.node_type() == "ClassDeclaration" {
                lower_stmt(b, &mut rops, decl)?;
            } else {
                let v = lower_expr(b, &mut rops, decl)?;
                let mut end = Op::new("jsir.expr_region_end");
                end.operands.push(v);
                rops.push(end);
            }
            let mut op = node_op("jsir.export_default_declaration", node)?;
            op.regions
                .push(Region::with_block(Block::leaf(rops)));
            ops.push(op);
            Ok(())
        }
        "ImportDeclaration" => {
            let mut op = node_op("jsir.import_declaration", node)?;
            op.attrs.push((
                "source".into(),
                object_literal_key(node_of(field(node, "source")?)?)?,
            ));
            let mut specs = Vec::new();
            for s in list_of(field(node, "specifiers")?)? {
                specs.push(import_specifier_attr(node_of(s)?)?);
            }
            op.attrs.push(("specifiers".into(), Attr::Array(specs)));
            ops.push(op);
            Ok(())
        }
        "ExportAllDeclaration" => {
            let mut op = node_op("jsir.export_all_declaration", node)?;
            op.attrs.push((
                "source".into(),
                object_literal_key(node_of(field(node, "source")?)?)?,
            ));
            ops.push(op);
            Ok(())
        }
        "ExportNamedDeclaration" => {
            let mut op = node_op("jsir.export_named_declaration", node)?;
            if let Field::Node(src) = node.field("source") {
                op.attrs.push(("source".into(), object_literal_key(src)?));
            }
            let mut specs = Vec::new();
            for s in list_of(field(node, "specifiers")?)? {
                specs.push(export_specifier_attr(node_of(s)?)?);
            }
            op.attrs.push(("specifiers".into(), Attr::Array(specs)));
            // Region holds the inline declaration, or is empty (zero blocks).
            match node.field("declaration") {
                Field::Node(decl) => {
                    let mut rops = Vec::new();
                    lower_stmt(b, &mut rops, decl)?;
                    op.regions
                        .push(Region::with_block(Block::leaf(rops)));
                }
                _ => op.regions.push(Region::default()),
            }
            ops.push(op);
            Ok(())
        }
        "ForInStatement" => {
            lower_for_in_of(b, ops, node, "jshir.for_in_statement", false)
        }
        "ForOfStatement" => {
            lower_for_in_of(b, ops, node, "jshir.for_of_statement", true)
        }
        "TryStatement" => {
            let mut op = node_op("jshir.try_statement", node)?;
            op.regions
                .push(stmt_region(b, node_of(field(node, "block")?)?)?);
            // Handler region: lower the catch param as an l-value, then a
            // `catch_clause` op wrapping the catch block.
            op.regions.push(match node.field("handler") {
                Field::Node(handler) => {
                    let mut rops = Vec::new();
                    let mut cc = node_op("jshir.catch_clause", handler)?;
                    if let Field::Node(param) = handler.field("param") {
                        let p = lower_lval(b, &mut rops, param)?;
                        cc.operands.push(p);
                    }
                    cc.regions
                        .push(stmt_region(b, node_of(field(handler, "body")?)?)?);
                    rops.push(cc);
                    Region::with_block(Block::leaf(rops))
                }
                _ => Region::default(),
            });
            op.regions.push(match node.field("finalizer") {
                Field::Node(fin) => stmt_region(b, fin)?,
                _ => Region::default(),
            });
            ops.push(op);
            Ok(())
        }
        "SwitchStatement" => {
            let disc = lower_expr(b, ops, node_of(field(node, "discriminant")?)?)?;
            let mut op = node_op("jshir.switch_statement", node)?;
            op.operands.push(disc);
            let mut cases = Vec::new();
            for el in list_of(field(node, "cases")?)? {
                let case = node_of(el)?;
                let mut cop = node_op("jshir.switch_case", case)?;
                // test region: ExprRegion, or zero-block for `default:`.
                cop.regions.push(opt_expr_region(b, case.field("test"))?);
                // consequent region: a list of statements (StmtsRegion).
                let mut body = Vec::new();
                for s in list_of(field(case, "consequent")?)? {
                    lower_stmt(b, &mut body, node_of(s)?)?;
                }
                cop.regions
                    .push(Region::with_block(Block::leaf(body)));
                cases.push(cop);
            }
            op.regions
                .push(Region::with_block(Block::leaf(cases)));
            ops.push(op);
            Ok(())
        }
        other => Err(format!("unsupported statement: {other}")),
    }
}

fn lower_block_statement(b: &mut Builder, node: &dyn AstNode) -> LowerResult<Op> {
    let mut body = Vec::new();
    for el in list_of(field(node, "body")?)? {
        lower_stmt(b, &mut body, node_of(el)?)?;
    }
    let mut directives = Vec::new();
    if let Ok(dirs) = field(node, "directives").and_then(list_of) {
        for el in dirs {
            lower_directive(b, &mut directives, node_of(el)?)?;
        }
    }
    let mut op = node_op("jshir.block_statement", node)?;
    op.regions.push(Region::with_block(Block::leaf(body)));
    op.regions.push(Region::with_block(Block::leaf(directives)));
    Ok(op)
}

fn lower_variable_declaration(b: &mut Builder, node: &dyn AstNode) -> LowerResult<Op> {
    let kind = str_of(field(node, "kind")?)?;
    let mut region_ops = Vec::new();
    let mut declarator_results = Vec::new();
    for el in list_of(field(node, "declarations")?)? {
        let decl = node_of(el)?;
        let id_ref = lower_lval(b, &mut region_ops, node_of(field(decl, "id")?)?)?;
        let mut declr = node_op("jsir.variable_declarator", decl)?;
        declr.operands.push(id_ref);
        if let Field::Node(init) = decl.field("init") {
            let v = lower_expr(b, &mut region_ops, init)?;
            declr.operands.push(v);
        }
        let res = b.fresh();
        declr.results.push(res);
        declarator_results.push(res);
        region_ops.push(declr);
    }
    let mut end = Op::new("jsir.exprs_region_end");
    end.operands = declarator_results;
    region_ops.push(end);

    let mut op = node_op("jsir.variable_declaration", node)?;
    op.attrs.push(("kind".into(), Attr::Str(kind.to_string())));
    op.regions.push(Region::with_block(Block::leaf(region_ops)));
    Ok(op)
}

// ---------------------------------------------------------------------------
// Expressions (r-values)
// ---------------------------------------------------------------------------

fn lower_expr(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<ValueId> {
    match node.node_type() {
        "NumericLiteral" => emit(ops, b, numeric_literal(node)?),
        "StringLiteral" => emit(ops, b, string_literal(node)?),
        "BooleanLiteral" => {
            let mut op = node_op("jsir.boolean_literal", node)?;
            op.attrs
                .push(("value".into(), Attr::Bool(bool_of(field(node, "value")?)?)));
            emit(ops, b, op)
        }
        "NullLiteral" => emit(ops, b, node_op("jsir.null_literal", node)?),
        "BigIntLiteral" => emit(ops, b, big_int_literal(node)?),
        "RegExpLiteral" => emit(ops, b, reg_exp_literal(node)?),
        "Identifier" => {
            let mut op = node_op("jsir.identifier", node)?;
            op.attrs
                .push(("name".into(), Attr::Str(str_of(field(node, "name")?)?.to_string())));
            emit(ops, b, op)
        }
        "ThisExpression" => emit(ops, b, node_op("jsir.this_expression", node)?),
        "Super" => emit(ops, b, node_op("jsir.super", node)?),
        // `import(x)` callee: the `import` keyword as a value.
        "Import" => emit(ops, b, node_op("jsir.import", node)?),
        "BinaryExpression" => {
            let l = lower_expr(b, ops, node_of(field(node, "left")?)?)?;
            let r = lower_expr(b, ops, node_of(field(node, "right")?)?)?;
            let mut op = node_op("jsir.binary_expression", node)?;
            op.operands.push(l);
            op.operands.push(r);
            op.attrs.push((
                "operator_".into(),
                Attr::Str(str_of(field(node, "operator")?)?.to_string()),
            ));
            emit(ops, b, op)
        }
        "LogicalExpression" => {
            let l = lower_expr(b, ops, node_of(field(node, "left")?)?)?;
            let mut op = node_op("jshir.logical_expression", node)?;
            op.operands.push(l);
            op.attrs.push((
                "operator_".into(),
                Attr::Str(str_of(field(node, "operator")?)?.to_string()),
            ));
            op.regions
                .push(expr_region(b, node_of(field(node, "right")?)?, |bb, ops, n| {
                    lower_expr(bb, ops, n)
                })?);
            emit(ops, b, op)
        }
        "UnaryExpression" => {
            let a = lower_expr(b, ops, node_of(field(node, "argument")?)?)?;
            let mut op = node_op("jsir.unary_expression", node)?;
            op.operands.push(a);
            op.attrs.push((
                "operator_".into(),
                Attr::Str(str_of(field(node, "operator")?)?.to_string()),
            ));
            op.attrs
                .push(("prefix".into(), Attr::Bool(bool_of(field(node, "prefix")?)?)));
            emit(ops, b, op)
        }
        "UpdateExpression" => {
            // The argument is an l-value (it is read and written).
            let a = lower_lval(b, ops, node_of(field(node, "argument")?)?)?;
            let mut op = node_op("jsir.update_expression", node)?;
            op.operands.push(a);
            op.attrs.push((
                "operator_".into(),
                Attr::Str(str_of(field(node, "operator")?)?.to_string()),
            ));
            op.attrs
                .push(("prefix".into(), Attr::Bool(bool_of(field(node, "prefix")?)?)));
            emit(ops, b, op)
        }
        "AssignmentExpression" => {
            let l = lower_lval(b, ops, node_of(field(node, "left")?)?)?;
            let r = lower_expr(b, ops, node_of(field(node, "right")?)?)?;
            let mut op = node_op("jsir.assignment_expression", node)?;
            op.operands.push(l);
            op.operands.push(r);
            op.attrs.push((
                "operator_".into(),
                Attr::Str(str_of(field(node, "operator")?)?.to_string()),
            ));
            emit(ops, b, op)
        }
        "CallExpression" => lower_call(b, ops, node, "jsir.call_expression"),
        "NewExpression" => lower_call(b, ops, node, "jsir.new_expression"),
        "OptionalCallExpression" => {
            // `a?.(args)` — like a call, plus the `optional` flag the reversible
            // IR needs to reprint the `?.`. The short-circuit does not change the
            // memoization structure (the analysis treats it as a plain call), so
            // jsir-ssa lowers it the same as `jsir.call_expression`.
            let callee = lower_expr(b, ops, node_of(field(node, "callee")?)?)?;
            let mut op = node_op("jsir.optional_call_expression", node)?;
            op.operands.push(callee);
            for el in list_of(field(node, "arguments")?)? {
                op.operands.push(lower_expr(b, ops, node_of(el)?)?);
            }
            op.attrs
                .push(("optional".into(), Attr::Bool(bool_of(field(node, "optional")?)?)));
            emit(ops, b, op)
        }
        "MemberExpression" => lower_member(b, ops, node, "jsir.member_expression"),
        "OptionalMemberExpression" => {
            let object = lower_expr(b, ops, node_of(field(node, "object")?)?)?;
            let computed = bool_of(field(node, "computed")?)?;
            let mut op = node_op("jsir.optional_member_expression", node)?;
            op.operands.push(object);
            if computed {
                let p = lower_expr(b, ops, node_of(field(node, "property")?)?)?;
                op.operands.push(p);
            } else {
                op.attrs.push((
                    "literal_property".into(),
                    identifier_attr(node_of(field(node, "property")?)?)?,
                ));
            }
            op.attrs
                .push(("optional".into(), Attr::Bool(bool_of(field(node, "optional")?)?)));
            emit(ops, b, op)
        }
        "YieldExpression" => {
            let mut op = node_op("jsir.yield_expression", node)?;
            if let Field::Node(arg) = node.field("argument") {
                let v = lower_expr(b, ops, arg)?;
                op.operands.push(v);
            }
            op.attrs
                .push(("delegate".into(), Attr::Bool(bool_of(field(node, "delegate")?)?)));
            emit(ops, b, op)
        }
        "AwaitExpression" => {
            let a = lower_expr(b, ops, node_of(field(node, "argument")?)?)?;
            emit(ops, b, unary_op("jsir.await_expression", node, a)?)
        }
        "MetaProperty" => {
            let mut op = node_op("jsir.meta_property", node)?;
            op.attrs
                .push(("meta".into(), identifier_attr(node_of(field(node, "meta")?)?)?));
            op.attrs.push((
                "property".into(),
                identifier_attr(node_of(field(node, "property")?)?)?,
            ));
            emit(ops, b, op)
        }
        "FunctionExpression" => {
            let mut op = lower_function(b, node, "jsir.function_expression")?;
            let v = b.fresh();
            op.results.push(v);
            ops.push(op);
            Ok(v)
        }
        "SpreadElement" => {
            let a = lower_expr(b, ops, node_of(field(node, "argument")?)?)?;
            emit(ops, b, unary_op("jsir.spread_element", node, a)?)
        }
        "ParenthesizedExpression" => {
            let e = lower_expr(b, ops, node_of(field(node, "expression")?)?)?;
            emit(ops, b, unary_op("jsir.parenthesized_expression", node, e)?)
        }
        "ArrayExpression" => {
            let mut elem_values = Vec::new();
            for el in list_of(field(node, "elements")?)? {
                match el {
                    Field::Null => {
                        // A hole becomes a `jsir.none` op.
                        elem_values.push(emit(ops, b, Op::new("jsir.none"))?);
                    }
                    Field::Node(n) => {
                        elem_values.push(lower_expr(b, ops, n)?);
                    }
                    _ => return Err("unexpected array element".into()),
                }
            }
            let mut op = node_op("jsir.array_expression", node)?;
            op.operands = elem_values;
            emit(ops, b, op)
        }
        "SequenceExpression" => {
            // `a, b, c` -> all sub-expressions lowered in order, the op holds
            // each as an operand (the last is the sequence's value).
            let mut values = Vec::new();
            for el in list_of(field(node, "expressions")?)? {
                values.push(lower_expr(b, ops, node_of(el)?)?);
            }
            let mut op = node_op("jsir.sequence_expression", node)?;
            op.operands = values;
            emit(ops, b, op)
        }
        "ClassExpression" => {
            let mut op = lower_class(b, ops, node, "jsir.class_expression")?;
            let v = b.fresh();
            op.results.push(v);
            ops.push(op);
            Ok(v)
        }
        "ArrowFunctionExpression" => {
            // Operand segments: [id (0/1), params...]; arrows have no id.
            let mut operands = Vec::new();
            let nid = if let Field::Node(id) = node.field("id") {
                operands.push(lower_lval(b, ops, id)?);
                1
            } else {
                0
            };
            let mut nparams = 0i32;
            for el in list_of(field(node, "params")?)? {
                operands.push(lower_pattern_ref(b, ops, node_of(el)?)?);
                nparams += 1;
            }
            let mut op = node_op("jsir.arrow_function_expression", node)?;
            op.operands = operands;
            op.attrs
                .push(("generator".into(), Attr::Bool(bool_of(field(node, "generator")?)?)));
            op.attrs
                .push(("async".into(), Attr::Bool(bool_of(field(node, "async")?)?)));
            op.attrs
                .push(("operandSegmentSizes".into(), Attr::I32Array(vec![nid, nparams])));
            // Body region (UnknownRegion): a block statement, or an expression
            // terminated by `expr_region_end`.
            let body = node_of(field(node, "body")?)?;
            let mut rops = Vec::new();
            if body.node_type() == "BlockStatement" {
                lower_stmt(b, &mut rops, body)?;
            } else {
                let v = lower_expr(b, &mut rops, body)?;
                let mut end = Op::new("jsir.expr_region_end");
                end.operands.push(v);
                rops.push(end);
            }
            op.regions
                .push(Region::with_block(Block::leaf(rops)));
            emit(ops, b, op)
        }
        "ObjectExpression" => lower_object_expression(b, ops, node),
        "TemplateLiteral" => lower_template_literal(b, ops, node),
        "TaggedTemplateExpression" => {
            let tag = lower_expr(b, ops, node_of(field(node, "tag")?)?)?;
            let quasi = lower_template_literal(b, ops, node_of(field(node, "quasi")?)?)?;
            let mut op = node_op("jsir.tagged_template_expression", node)?;
            op.operands.push(tag);
            op.operands.push(quasi);
            emit(ops, b, op)
        }
        "ConditionalExpression" => {
            let test = lower_expr(b, ops, node_of(field(node, "test")?)?)?;
            let mut op = node_op("jshir.conditional_expression", node)?;
            op.operands.push(test);
            // Upstream emits the alternate region first, then the consequent.
            op.regions
                .push(expr_region(b, node_of(field(node, "alternate")?)?, |bb, ops, n| {
                    lower_expr(bb, ops, n)
                })?);
            op.regions
                .push(expr_region(b, node_of(field(node, "consequent")?)?, |bb, ops, n| {
                    lower_expr(bb, ops, n)
                })?);
            emit(ops, b, op)
        }
        other => Err(format!("unsupported expression: {other}")),
    }
}

/// Lower a `for-in` / `for-of` statement. The left target is lowered as an
/// l-value; when it is a `let`/`const`/`var` declaration, the declaration
/// metadata is attached as the `left_declaration` attribute.
fn lower_for_in_of(
    b: &mut Builder,
    ops: &mut Vec<Op>,
    node: &dyn AstNode,
    name: &str,
    is_of: bool,
) -> LowerResult<()> {
    let left_node = node_of(field(node, "left")?)?;
    let (left_val, decl_attr) = if left_node.node_type() == "VariableDeclaration" {
        let declarator = node_of(first(list_of(field(left_node, "declarations")?)?)?)?;
        let id = node_of(field(declarator, "id")?)?;
        let lv = lower_lval(b, ops, id)?;
        let kind = str_of(field(left_node, "kind")?)?.to_string();
        let attr = for_in_of_declaration_attr(left_node, declarator, kind)?;
        (lv, Some(attr))
    } else {
        (lower_lval(b, ops, left_node)?, None)
    };
    let right = lower_expr(b, ops, node_of(field(node, "right")?)?)?;
    let mut op = node_op(name, node)?;
    op.operands.push(left_val);
    op.operands.push(right);
    if is_of {
        op.attrs
            .push(("await".into(), Attr::Bool(bool_of(field(node, "await")?)?)));
    }
    if let Some(a) = decl_attr {
        op.attrs.push(("left_declaration".into(), a));
    }
    op.regions
        .push(stmt_region(b, node_of(field(node, "body")?)?)?);
    ops.push(op);
    Ok(())
}

/// Build an import specifier attribute from an Import{,Default,Namespace}
/// Specifier AST node. The specifier node carries the defined symbol; `local`
/// is rendered flattened and (for named) `imported` as a nested attribute.
fn import_specifier_attr(node: &dyn AstNode) -> LowerResult<Attr> {
    let loc = extra_of(field(node, "loc")?)?;
    let (start_line, start_col) = position(loc, "start")?;
    let (end_line, end_col) = position(loc, "end")?;
    let sym = extra_of(first(list_of(field(node, "definedSymbols")?)?)?)?;
    let local = build_identifier_attr(node_of(field(node, "local")?)?)?;
    let (kind, imported) = match node.node_type() {
        "ImportSpecifier" => (
            ImportSpecKind::Named,
            Some(object_literal_key(node_of(field(node, "imported")?)?)?),
        ),
        "ImportDefaultSpecifier" => (ImportSpecKind::Default, None),
        "ImportNamespaceSpecifier" => (ImportSpecKind::Namespace, None),
        other => return Err(format!("unsupported import specifier: {other}")),
    };
    Ok(Attr::ImportSpecifier(Box::new(ImportSpecifierAttr {
        kind,
        start_line,
        start_col,
        end_line,
        end_col,
        start_index: i64_of(field(node, "start")?)?,
        end_index: i64_of(field(node, "end")?)?,
        scope_uid: i64_of(field(node, "scopeUid")?)?,
        sym_name: str_of(extra_field(sym, "name")?)?.to_string(),
        sym_scope: i64_of(extra_field(sym, "defScopeUid")?)?,
        imported,
        local,
    })))
}

/// Build an `export_specifier` attribute from an ExportSpecifier AST node.
fn export_specifier_attr(node: &dyn AstNode) -> LowerResult<Attr> {
    let loc = extra_of(field(node, "loc")?)?;
    let (start_line, start_col) = position(loc, "start")?;
    let (end_line, end_col) = position(loc, "end")?;
    Ok(Attr::ExportSpecifier(Box::new(ExportSpecifierAttr {
        start_line,
        start_col,
        end_line,
        end_col,
        start_index: i64_of(field(node, "start")?)?,
        end_index: i64_of(field(node, "end")?)?,
        scope_uid: i64_of(field(node, "scopeUid")?)?,
        exported: object_literal_key(node_of(field(node, "exported")?)?)?,
        local: object_literal_key(node_of(field(node, "local")?)?)?,
    })))
}

/// Build the `left_declaration` attribute for `for (let x in/of y)`.
fn for_in_of_declaration_attr(decl: &dyn AstNode, declarator: &dyn AstNode, kind: String) -> LowerResult<Attr> {
    let dloc = extra_of(field(decl, "loc")?)?;
    let (d_start_line, d_start_col) = position(dloc, "start")?;
    let (d_end_line, d_end_col) = position(dloc, "end")?;
    let rloc = extra_of(field(declarator, "loc")?)?;
    let (r_start_line, r_start_col) = position(rloc, "start")?;
    let (r_end_line, r_end_col) = position(rloc, "end")?;
    let mut symbols = Vec::new();
    for s in list_of(field(declarator, "definedSymbols")?)? {
        let sym = extra_of(s)?;
        symbols.push((
            str_of(extra_field(sym, "name")?)?.to_string(),
            i64_of(extra_field(sym, "defScopeUid")?)?,
        ));
    }
    Ok(Attr::ForInOfDeclaration(Box::new(ForInOfDeclarationAttr {
        d_start_line,
        d_start_col,
        d_end_line,
        d_end_col,
        d_start_index: i64_of(field(decl, "start")?)?,
        d_end_index: i64_of(field(decl, "end")?)?,
        d_scope: i64_of(field(decl, "scopeUid")?)?,
        r_start_line,
        r_start_col,
        r_end_line,
        r_end_col,
        r_start_index: i64_of(field(declarator, "start")?)?,
        r_end_index: i64_of(field(declarator, "end")?)?,
        r_scope: i64_of(field(declarator, "scopeUid")?)?,
        symbols,
        kind,
    })))
}

/// Lower a member expression (`a.b` / `a[b]`) to `member_expression` or, in
/// l-value position, `member_expression_ref`. The object is always an r-value.
fn lower_member(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode, name: &str) -> LowerResult<ValueId> {
    let object = lower_expr(b, ops, node_of(field(node, "object")?)?)?;
    let computed = bool_of(field(node, "computed")?)?;
    let mut op = node_op(name, node)?;
    op.operands.push(object);
    if computed {
        let prop = lower_expr(b, ops, node_of(field(node, "property")?)?)?;
        op.operands.push(prop);
    } else {
        let prop = node_of(field(node, "property")?)?;
        // `obj.#priv` carries a PrivateName property; `obj.x` an Identifier.
        let attr = if prop.node_type() == "PrivateName" {
            private_name_attr(prop)?
        } else {
            identifier_attr(prop)?
        };
        op.attrs.push(("literal_property".into(), attr));
    }
    emit(ops, b, op)
}

/// Build a `JsirIdentifierAttr` value from an AST `Identifier` node.
fn build_identifier_attr(node: &dyn AstNode) -> LowerResult<IdentifierAttr> {
    if node.node_type() != "Identifier" {
        return Err(format!("identifier_attr: expected Identifier, got {}", node.node_type()));
    }
    let loc = extra_of(field(node, "loc")?)?;
    let (start_line, start_col) = position(loc, "start")?;
    let (end_line, end_col) = position(loc, "end")?;
    Ok(IdentifierAttr {
        start_line,
        start_col,
        end_line,
        end_col,
        identifier_name: str_of(extra_field(loc, "identifierName")?)?.to_string(),
        start_index: i64_of(field(node, "start")?)?,
        end_index: i64_of(field(node, "end")?)?,
        scope_uid: i64_of(field(node, "scopeUid")?)?,
        name: str_of(field(node, "name")?)?.to_string(),
    })
}

/// Build a `JsirIdentifierAttr` (used when an identifier appears as an op
/// attribute: member properties, ids, object keys).
fn identifier_attr(node: &dyn AstNode) -> LowerResult<Attr> {
    Ok(Attr::Identifier(Box::new(build_identifier_attr(node)?)))
}

/// Build a `JsirPrivateNameAttr` from a `PrivateName` AST node (`#x`).
fn private_name_attr(node: &dyn AstNode) -> LowerResult<Attr> {
    if node.node_type() != "PrivateName" {
        return Err(format!("private_name_attr: expected PrivateName, got {}", node.node_type()));
    }
    let loc = extra_of(field(node, "loc")?)?;
    let (start_line, start_col) = position(loc, "start")?;
    let (end_line, end_col) = position(loc, "end")?;
    Ok(Attr::PrivateName(Box::new(PrivateNameAttr {
        start_line,
        start_col,
        end_line,
        end_col,
        start_index: i64_of(field(node, "start")?)?,
        end_index: i64_of(field(node, "end")?)?,
        scope_uid: i64_of(field(node, "scopeUid")?)?,
        id: build_identifier_attr(node_of(field(node, "id")?)?)?,
    })))
}

/// Build a `comment_line`/`comment_block` attribute from a CommentLine/
/// CommentBlock AST node (the elements of `File.comments`).
fn comment_attr(node: &dyn AstNode) -> LowerResult<Attr> {
    let block = match node.node_type() {
        "CommentBlock" => true,
        "CommentLine" => false,
        other => return Err(format!("unexpected comment node: {other}")),
    };
    let loc = extra_of(field(node, "loc")?)?;
    let (start_line, start_col) = position(loc, "start")?;
    let (end_line, end_col) = position(loc, "end")?;
    Ok(Attr::Comment(Box::new(CommentAttr {
        block,
        start_line,
        start_col,
        end_line,
        end_col,
        start_index: i64_of(field(node, "start")?)?,
        end_index: i64_of(field(node, "end")?)?,
        value: str_of(field(node, "value")?)?.to_string(),
    })))
}

/// Read a `{line, column}` Position from a SourceLocation helper's field.
fn position(loc: &dyn AstNode, key: &str) -> LowerResult<(i64, i64)> {
    let p = extra_of(extra_field(loc, key)?)?;
    Ok((i64_of(extra_field(p, "line")?)?, i64_of(extra_field(p, "column")?)?))
}

/// Lower a class declaration/expression: optional superClass operand (`extends`)
/// + `id` attribute + a body region holding a single `class_body` op.
fn lower_class(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode, name: &str) -> LowerResult<Op> {
    // `extends X` evaluates the superclass expression before the class op and
    // passes its value as the class op's operand.
    let super_class = if let Field::Node(sc) = node.field("superClass") {
        Some(lower_expr(b, ops, sc)?)
    } else {
        None
    };
    let mut op = node_op(name, node)?;
    if let Some(sc) = super_class {
        op.operands.push(sc);
    }
    if let Field::Node(id) = node.field("id") {
        op.attrs.push(("id".into(), identifier_attr(id)?));
    }
    let class_body = lower_class_body(b, node_of(field(node, "body")?)?)?;
    op.regions.push(Region::with_block(Block::leaf(vec![class_body])));
    Ok(op)
}

/// Lower a `ClassBody` into a `class_body` op whose region holds the members.
fn lower_class_body(b: &mut Builder, node: &dyn AstNode) -> LowerResult<Op> {
    let mut members = Vec::new();
    for el in list_of(field(node, "body")?)? {
        lower_class_member(b, &mut members, node_of(el)?)?;
    }
    let mut op = node_op("jsir.class_body", node)?;
    op.regions
        .push(Region::with_block(Block::leaf(members)));
    Ok(op)
}

/// Lower one class member, pushing its ops (computed keys + the member op) into
/// the class body's op list.
fn lower_class_member(b: &mut Builder, ops: &mut Vec<Op>, m: &dyn AstNode) -> LowerResult<()> {
    match m.node_type() {
        "ClassProperty" => {
            let computed = bool_of(field(m, "computed")?)?;
            let mut op = node_op("jsir.class_property", m)?;
            if computed {
                let k = lower_expr(b, ops, node_of(field(m, "key")?)?)?;
                op.operands.push(k);
            } else {
                op.attrs
                    .push(("literal_key".into(), object_literal_key(node_of(field(m, "key")?)?)?));
            }
            op.attrs
                .push(("static_".into(), Attr::Bool(bool_of(field(m, "static")?)?)));
            op.regions.push(class_value_region(b, m)?);
            ops.push(op);
        }
        "ClassPrivateProperty" => {
            let mut op = node_op("jsir.class_private_property", m)?;
            op.attrs
                .push(("key".into(), private_name_attr(node_of(field(m, "key")?)?)?));
            op.attrs
                .push(("static_".into(), Attr::Bool(bool_of(field(m, "static")?)?)));
            op.regions.push(class_value_region(b, m)?);
            ops.push(op);
        }
        "ClassMethod" => {
            let computed = bool_of(field(m, "computed")?)?;
            // Operand segments are [params, computed_key] for class methods.
            let mut operands = Vec::new();
            let mut nparams = 0i32;
            for el in list_of(field(m, "params")?)? {
                operands.push(lower_pattern_ref(b, ops, node_of(el)?)?);
                nparams += 1;
            }
            let nkey = if computed {
                operands.push(lower_expr(b, ops, node_of(field(m, "key")?)?)?);
                1
            } else {
                0
            };
            let mut op = node_op("jsir.class_method", m)?;
            op.operands = operands;
            op.attrs
                .push(("generator".into(), Attr::Bool(bool_of(field(m, "generator")?)?)));
            op.attrs
                .push(("async".into(), Attr::Bool(bool_of(field(m, "async")?)?)));
            op.attrs
                .push(("kind".into(), Attr::Str(str_of(field(m, "kind")?)?.to_string())));
            if !computed {
                op.attrs
                    .push(("literal_key".into(), object_literal_key(node_of(field(m, "key")?)?)?));
            }
            op.attrs
                .push(("operandSegmentSizes".into(), Attr::I32Array(vec![nparams, nkey])));
            op.attrs
                .push(("static_".into(), Attr::Bool(bool_of(field(m, "static")?)?)));
            op.regions
                .push(stmt_region(b, node_of(field(m, "body")?)?)?);
            ops.push(op);
        }
        "ClassPrivateMethod" => {
            let mut operands = Vec::new();
            for el in list_of(field(m, "params")?)? {
                operands.push(lower_pattern_ref(b, ops, node_of(el)?)?);
            }
            let mut op = node_op("jsir.class_private_method", m)?;
            op.operands = operands;
            op.attrs
                .push(("generator".into(), Attr::Bool(bool_of(field(m, "generator")?)?)));
            op.attrs
                .push(("async".into(), Attr::Bool(bool_of(field(m, "async")?)?)));
            op.attrs
                .push(("key".into(), private_name_attr(node_of(field(m, "key")?)?)?));
            op.attrs
                .push(("kind".into(), Attr::Str(str_of(field(m, "kind")?)?.to_string())));
            op.attrs
                .push(("static_".into(), Attr::Bool(bool_of(field(m, "static")?)?)));
            op.regions
                .push(stmt_region(b, node_of(field(m, "body")?)?)?);
            ops.push(op);
        }
        other => return Err(format!("unsupported class member: {other}")),
    }
    Ok(())
}

/// The optional `value` ExprRegion of a class property (`= <expr>`), or a
/// zero-block region when there is no initializer.
fn class_value_region(b: &mut Builder, m: &dyn AstNode) -> LowerResult<Region> {
    match m.field("value") {
        Field::Node(val) => expr_region(b, val, |bb, ops, n| lower_expr(bb, ops, n)),
        _ => Ok(Region::default()),
    }
}

/// Lower a function/method-like node. `params` becomes an ExprsRegion of
/// pattern-refs ending in `exprs_region_end`; `body` (a BlockStatement) becomes
/// a StmtRegion. Caller assigns a result for expression forms.
fn lower_function(b: &mut Builder, node: &dyn AstNode, name: &str) -> LowerResult<Op> {
    let mut op = node_op(name, node)?;
    if let Field::Node(id) = node.field("id") {
        op.attrs.push(("id".into(), identifier_attr(id)?));
    }
    op.attrs
        .push(("generator".into(), Attr::Bool(bool_of(field(node, "generator")?)?)));
    op.attrs
        .push(("async".into(), Attr::Bool(bool_of(field(node, "async")?)?)));

    // Params region (ExprsRegion).
    let mut pops = Vec::new();
    let mut presults = Vec::new();
    for el in list_of(field(node, "params")?)? {
        presults.push(lower_pattern_ref(b, &mut pops, node_of(el)?)?);
    }
    let mut end = Op::new("jsir.exprs_region_end");
    end.operands = presults;
    pops.push(end);
    op.regions
        .push(Region::with_block(Block::leaf(pops)));

    // Body region (StmtRegion holding the function's BlockStatement).
    op.regions
        .push(stmt_region(b, node_of(field(node, "body")?)?)?);
    Ok(op)
}

/// Lower a binding pattern in reference (l-value) position.
fn lower_pattern_ref(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<ValueId> {
    match node.node_type() {
        "Identifier" | "MemberExpression" | "ParenthesizedExpression" => lower_lval(b, ops, node),
        "AssignmentPattern" => {
            let left = lower_pattern_ref(b, ops, node_of(field(node, "left")?)?)?;
            let right = lower_expr(b, ops, node_of(field(node, "right")?)?)?;
            let mut op = node_op("jsir.assignment_pattern_ref", node)?;
            op.operands.push(left);
            op.operands.push(right);
            emit(ops, b, op)
        }
        "RestElement" => {
            let arg = lower_pattern_ref(b, ops, node_of(field(node, "argument")?)?)?;
            emit(ops, b, unary_op("jsir.rest_element_ref", node, arg)?)
        }
        "ArrayPattern" => {
            let mut elems = Vec::new();
            for el in list_of(field(node, "elements")?)? {
                match el {
                    Field::Null => elems.push(emit(ops, b, Op::new("jsir.none"))?),
                    Field::Node(n) => elems.push(lower_pattern_ref(b, ops, n)?),
                    _ => return Err("unexpected array-pattern element".into()),
                }
            }
            let mut op = node_op("jsir.array_pattern_ref", node)?;
            op.operands = elems;
            emit(ops, b, op)
        }
        "ObjectPattern" => {
            let mut rops = Vec::new();
            let mut results = Vec::new();
            for el in list_of(field(node, "properties")?)? {
                let prop = node_of(el)?;
                if prop.node_type() == "RestElement" {
                    let arg = lower_pattern_ref(b, &mut rops, node_of(field(prop, "argument")?)?)?;
                    let rest = unary_op("jsir.rest_element_ref", prop, arg)?;
                    results.push(emit(&mut rops, b, rest)?);
                } else {
                    // ObjectProperty in pattern position: value is a pattern-ref.
                    let computed = bool_of(field(prop, "computed")?)?;
                    let shorthand = bool_of(field(prop, "shorthand")?)?;
                    let mut op = node_op("jsir.object_property_ref", prop)?;
                    if computed {
                        // Computed key is evaluated before the value.
                        let key = lower_expr(b, &mut rops, node_of(field(prop, "key")?)?)?;
                        let value = lower_pattern_ref(b, &mut rops, node_of(field(prop, "value")?)?)?;
                        op.operands.push(key);
                        op.operands.push(value);
                    } else {
                        let value = lower_pattern_ref(b, &mut rops, node_of(field(prop, "value")?)?)?;
                        op.operands.push(value);
                        op.attrs.push((
                            "literal_key".into(),
                            object_literal_key(node_of(field(prop, "key")?)?)?,
                        ));
                    }
                    op.attrs.push(("shorthand".into(), Attr::Bool(shorthand)));
                    results.push(emit(&mut rops, b, op)?);
                }
            }
            let mut end = Op::new("jsir.exprs_region_end");
            end.operands = results;
            rops.push(end);
            let mut op = node_op("jsir.object_pattern_ref", node)?;
            op.regions
                .push(Region::with_block(Block::leaf(rops)));
            emit(ops, b, op)
        }
        other => Err(format!("unsupported pattern: {other}")),
    }
}

/// Build the `literal_key` attribute for an object/class property whose key is
/// a static identifier, string literal, or numeric literal.
fn object_literal_key(node: &dyn AstNode) -> LowerResult<Attr> {
    match node.node_type() {
        "Identifier" => identifier_attr(node),
        "StringLiteral" => {
            let loc = extra_of(field(node, "loc")?)?;
            let (start_line, start_col) = position(loc, "start")?;
            let (end_line, end_col) = position(loc, "end")?;
            let extra = extra_of(field(node, "extra")?)?;
            Ok(Attr::StringLiteralKey(Box::new(StringLiteralKeyAttr {
                start_line,
                start_col,
                end_line,
                end_col,
                start_index: i64_of(field(node, "start")?)?,
                end_index: i64_of(field(node, "end")?)?,
                scope_uid: i64_of(field(node, "scopeUid")?)?,
                value: str_of(field(node, "value")?)?.to_string(),
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
                raw_value: str_of(extra_field(extra, "rawValue")?)?.to_string(),
            })))
        }
        "NumericLiteral" => {
            let loc = extra_of(field(node, "loc")?)?;
            let (start_line, start_col) = position(loc, "start")?;
            let (end_line, end_col) = position(loc, "end")?;
            let extra = extra_of(field(node, "extra")?)?;
            Ok(Attr::NumericLiteralKey(Box::new(NumericLiteralKeyAttr {
                start_line,
                start_col,
                end_line,
                end_col,
                start_index: i64_of(field(node, "start")?)?,
                end_index: i64_of(field(node, "end")?)?,
                scope_uid: i64_of(field(node, "scopeUid")?)?,
                value: f64_of(field(node, "value")?)?,
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
                raw_value: f64_of(extra_field(extra, "rawValue")?)?,
            })))
        }
        other => Err(format!("unsupported object key: {other}")),
    }
}

/// Lower an object literal: each property/method/spread is emitted into the
/// object's ExprsRegion, terminated by `exprs_region_end` over their results.
fn lower_object_expression(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<ValueId> {
    let mut rops = Vec::new();
    let mut results = Vec::new();
    for el in list_of(field(node, "properties")?)? {
        let p = node_of(el)?;
        match p.node_type() {
            "ObjectProperty" => {
                let computed = bool_of(field(p, "computed")?)?;
                let shorthand = bool_of(field(p, "shorthand")?)?;
                let mut op = node_op("jsir.object_property", p)?;
                if computed {
                    let key = lower_expr(b, &mut rops, node_of(field(p, "key")?)?)?;
                    let value = lower_expr(b, &mut rops, node_of(field(p, "value")?)?)?;
                    op.operands.push(key);
                    op.operands.push(value);
                } else {
                    let value = lower_expr(b, &mut rops, node_of(field(p, "value")?)?)?;
                    op.operands.push(value);
                    op.attrs
                        .push(("literal_key".into(), object_literal_key(node_of(field(p, "key")?)?)?));
                }
                op.attrs.push(("shorthand".into(), Attr::Bool(shorthand)));
                results.push(emit(&mut rops, b, op)?);
            }
            "ObjectMethod" => {
                results.push(lower_object_method(b, &mut rops, p)?);
            }
            "SpreadElement" => {
                results.push(lower_expr(b, &mut rops, p)?);
            }
            other => return Err(format!("unsupported object member: {other}")),
        }
    }
    let mut end = Op::new("jsir.exprs_region_end");
    end.operands = results;
    rops.push(end);
    let mut op = node_op("jsir.object_expression", node)?;
    op.regions
        .push(Region::with_block(Block::leaf(rops)));
    emit(ops, b, op)
}

/// Lower an object method. Its computed key (if any) and params are direct
/// operands segmented by `operandSegmentSizes = array<i32: nkey, nparams>`.
fn lower_object_method(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<ValueId> {
    let computed = bool_of(field(node, "computed")?)?;
    let mut operands = Vec::new();
    let nkey = if computed {
        operands.push(lower_expr(b, ops, node_of(field(node, "key")?)?)?);
        1
    } else {
        0
    };
    let mut nparams = 0i32;
    for el in list_of(field(node, "params")?)? {
        operands.push(lower_pattern_ref(b, ops, node_of(el)?)?);
        nparams += 1;
    }
    let mut op = node_op("jsir.object_method", node)?;
    op.operands = operands;
    op.attrs
        .push(("generator".into(), Attr::Bool(bool_of(field(node, "generator")?)?)));
    op.attrs
        .push(("async".into(), Attr::Bool(bool_of(field(node, "async")?)?)));
    op.attrs
        .push(("kind".into(), Attr::Str(str_of(field(node, "kind")?)?.to_string())));
    if !computed {
        op.attrs
            .push(("literal_key".into(), object_literal_key(node_of(field(node, "key")?)?)?));
    }
    op.attrs
        .push(("operandSegmentSizes".into(), Attr::I32Array(vec![nkey, nparams])));
    op.regions
        .push(stmt_region(b, node_of(field(node, "body")?)?)?);
    emit(ops, b, op)
}

/// Lower a template literal: quasis (each `template_element` wrapping a
/// `template_element_value`) come first, then the interpolated expressions, and
/// `template_literal` carries `operandSegmentSizes = array<i32: nquasis, nexprs>`.
fn lower_template_literal(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<ValueId> {
    let quasis = list_of(field(node, "quasis")?)?;
    let mut quasi_results = Vec::new();
    for q in quasis {
        let elem = node_of(q)?;
        let value = extra_of(field(elem, "value")?)?;
        let mut vop = Op::new("jsir.template_element_value");
        if let Field::Str(cooked) = value.field("cooked") {
            vop.attrs.push(("cooked".into(), Attr::Str(cooked.to_string())));
        }
        vop.attrs
            .push(("raw".into(), Attr::Str(str_of(extra_field(value, "raw")?)?.to_string())));
        let vres = emit(ops, b, vop)?;
        let mut te = node_op("jsir.template_element", elem)?;
        te.operands.push(vres);
        te.attrs
            .push(("tail".into(), Attr::Bool(bool_of(field(elem, "tail")?)?)));
        quasi_results.push(emit(ops, b, te)?);
    }
    let exprs = list_of(field(node, "expressions")?)?;
    let mut expr_results = Vec::new();
    for e in exprs {
        expr_results.push(lower_expr(b, ops, node_of(e)?)?);
    }
    let nquasis = quasi_results.len() as i32;
    let nexprs = expr_results.len() as i32;
    let mut op = node_op("jsir.template_literal", node)?;
    op.operands.extend(quasi_results);
    op.operands.extend(expr_results);
    op.attrs
        .push(("operandSegmentSizes".into(), Attr::I32Array(vec![nquasis, nexprs])));
    emit(ops, b, op)
}

fn lower_call(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode, name: &str) -> LowerResult<ValueId> {
    let callee = lower_expr(b, ops, node_of(field(node, "callee")?)?)?;
    let mut op = node_op(name, node)?;
    op.operands.push(callee);
    for el in list_of(field(node, "arguments")?)? {
        op.operands.push(lower_expr(b, ops, node_of(el)?)?);
    }
    emit(ops, b, op)
}

// ---------------------------------------------------------------------------
// L-values (assignment / declaration targets)
// ---------------------------------------------------------------------------

fn lower_lval(b: &mut Builder, ops: &mut Vec<Op>, node: &dyn AstNode) -> LowerResult<ValueId> {
    match node.node_type() {
        "Identifier" => {
            let mut op = node_op("jsir.identifier_ref", node)?;
            op.attrs
                .push(("name".into(), Attr::Str(str_of(field(node, "name")?)?.to_string())));
            emit(ops, b, op)
        }
        "MemberExpression" => lower_member(b, ops, node, "jsir.member_expression_ref"),
        "ParenthesizedExpression" => {
            let e = lower_lval(b, ops, node_of(field(node, "expression")?)?)?;
            emit(ops, b, unary_op("jsir.parenthesized_expression_ref", node, e)?)
        }
        // Binding patterns in l-value position.
        "ArrayPattern" | "AssignmentPattern" | "RestElement" | "ObjectPattern" => {
            lower_pattern_ref(b, ops, node)
        }
        other => Err(format!("unsupported l-value: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Region builders
// ---------------------------------------------------------------------------

/// Build an `ExprRegion`: lower an expression in a fresh block, ending with
/// `jsir.expr_region_end(%result)`.
fn expr_region(
    b: &mut Builder,
    node: &dyn AstNode,
    lower: impl Fn(&mut Builder, &mut Vec<Op>, &dyn AstNode) -> LowerResult<ValueId>,
) -> LowerResult<Region> {
    let mut ops = Vec::new();
    let v = lower(b, &mut ops, node)?;
    let mut end = Op::new("jsir.expr_region_end");
    end.operands.push(v);
    ops.push(end);
    Ok(Region::with_block(Block::leaf(ops)))
}

/// Build a `StmtRegion`: lower a single statement into a fresh block.
fn stmt_region(b: &mut Builder, node: &dyn AstNode) -> LowerResult<Region> {
    let mut ops = Vec::new();
    lower_stmt(b, &mut ops, node)?;
    Ok(Region::with_block(Block::leaf(ops)))
}

/// An optional `ExprRegion`: a block ending in `expr_region_end` when the field
/// holds a node, otherwise a zero-block region (prints as `{ }`).
fn opt_expr_region(b: &mut Builder, f: Field<'_>) -> LowerResult<Region> {
    match f {
        Field::Node(n) => expr_region(b, n, |bb, ops, n| lower_expr(bb, ops, n)),
        _ => Ok(Region::default()),
    }
}

/// Build a `break`/`continue` statement op with an optional label attribute.
fn break_continue(name: &str, node: &dyn AstNode) -> LowerResult<Op> {
    let mut op = node_op(name, node)?;
    if let Field::Node(label) = node.field("label") {
        op.attrs.push(("label".into(), identifier_attr(label)?));
    }
    Ok(op)
}

// ---------------------------------------------------------------------------
// Literal helpers
// ---------------------------------------------------------------------------

fn numeric_literal(node: &dyn AstNode) -> LowerResult<Op> {
    let value = f64_of(field(node, "value")?)?;
    let mut op = node_op("jsir.numeric_literal", node)?;
    if let Field::Node(extra) = node.field("extra") {
        op.attrs.push((
            "extra".into(),
            Attr::NumericLiteralExtra {
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
                value: f64_of(extra_field(extra, "rawValue")?)?,
            },
        ));
    }
    op.attrs.push(("value".into(), Attr::F64(value)));
    Ok(op)
}

fn string_literal(node: &dyn AstNode) -> LowerResult<Op> {
    let value = str_of(field(node, "value")?)?.to_string();
    let mut op = node_op("jsir.string_literal", node)?;
    if let Field::Node(extra) = node.field("extra") {
        op.attrs.push((
            "extra".into(),
            Attr::StringLiteralExtra {
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
                raw_value: str_of(extra_field(extra, "rawValue")?)?.to_string(),
            },
        ));
    }
    op.attrs.push(("value".into(), Attr::Str(value)));
    Ok(op)
}

fn big_int_literal(node: &dyn AstNode) -> LowerResult<Op> {
    let value = str_of(field(node, "value")?)?.to_string();
    let mut op = node_op("jsir.big_int_literal", node)?;
    if let Field::Node(extra) = node.field("extra") {
        op.attrs.push((
            "extra".into(),
            Attr::BigIntLiteralExtra {
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
                raw_value: str_of(extra_field(extra, "rawValue")?)?.to_string(),
            },
        ));
    }
    op.attrs.push(("value".into(), Attr::Str(value)));
    Ok(op)
}

fn reg_exp_literal(node: &dyn AstNode) -> LowerResult<Op> {
    let pattern = str_of(field(node, "pattern")?)?.to_string();
    let flags = str_of(field(node, "flags")?)?.to_string();
    let mut op = node_op("jsir.reg_exp_literal", node)?;
    if let Field::Node(extra) = node.field("extra") {
        op.attrs.push((
            "extra".into(),
            Attr::RegExpLiteralExtra {
                raw: str_of(extra_field(extra, "raw")?)?.to_string(),
            },
        ));
    }
    op.attrs.push(("flags".into(), Attr::Str(flags)));
    op.attrs.push(("pattern".into(), Attr::Str(pattern)));
    Ok(op)
}

// ---------------------------------------------------------------------------
// Op construction helpers
// ---------------------------------------------------------------------------

/// Append `op`, giving it a fresh result value, and return that value.
fn emit(ops: &mut Vec<Op>, b: &mut Builder, mut op: Op) -> LowerResult<ValueId> {
    let v = b.fresh();
    op.results.clear();
    op.results.push(v);
    ops.push(op);
    Ok(v)
}

fn unary_op(name: &str, node: &dyn AstNode, operand: ValueId) -> LowerResult<Op> {
    let mut op = node_op(name, node)?;
    op.operands.push(operand);
    Ok(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsir_ast::Node;

    fn lower_fixture(name: &str) -> Result<String, String> {
        let f = jsir_oracle::list_fixtures()
            .into_iter()
            .find(|f| f.name == name)
            .ok_or("fixture not found")?;
        let ast = f.expected_ast_json().ok_or("no ast.json")?;
        let value: serde_json::Value = serde_json::from_str(&ast).map_err(|e| e.to_string())?;
        let node = Node::from_json(&value).map_err(|e| e.to_string())?;
        Ok(ast2hir(&node)?.print())
    }

    /// Report which fixtures lower to byte-exact JSHIR. The straight-line and
    /// simple control-flow cases must pass; the rest are tracked but tolerated
    /// until their constructs are implemented.
    /// Round-trip: ast.json -> Node -> ast2hir -> Op -> hir2ast -> Node' and
    /// require Node' to reproduce ast.json (FileCheck-equivalent), proving
    /// `hir2ast` inverts `ast2hir`. Constructs whose `hir2ast` is not yet
    /// implemented are tracked, not failed.
    #[test]
    fn corpus_hir2ast_round_trip() {
        let mut byte_exact = Vec::new();
        let mut whitespace_only = Vec::new();
        let mut failed = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            let Some(expected) = f.expected_ast_json() else {
                continue;
            };
            let result = (|| -> Result<String, String> {
                let value: serde_json::Value =
                    serde_json::from_str(&expected).map_err(|e| e.to_string())?;
                let node = Node::from_json(&value).map_err(|e| e.to_string())?;
                let op = ast2hir(&node)?;
                let lifted = hir2ast(&op)?;
                Ok(lifted.to_json_string())
            })();
            match result {
                Ok(actual) if actual == expected => byte_exact.push(f.name.clone()),
                Ok(actual) if jsir_oracle::filecheck_equivalent(&expected, &actual) => {
                    whitespace_only.push(f.name.clone())
                }
                Ok(actual) => failed.push((
                    f.name.clone(),
                    jsir_oracle::byte_diff(&expected, &actual).unwrap_or_default(),
                )),
                Err(e) => failed.push((f.name.clone(), e)),
            }
        }
        let pass = byte_exact.len() + whitespace_only.len();
        eprintln!(
            "hir2ast round-trip: {pass}/{} pass ({} byte-exact). Not-yet: {:?}",
            pass + failed.len(),
            byte_exact.len(),
            failed.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        );
        if std::env::var("HIR2AST_DIFF").is_ok() {
            for (n, d) in &failed {
                eprintln!("\n##### {n} #####\n{}", d.lines().take(7).collect::<Vec<_>>().join("\n"));
            }
        }
        // Every fixture must round-trip ast.json -> IR -> ast.json.
        assert!(
            failed.is_empty(),
            "hir2ast round-trip regressions:\n{}",
            failed.iter().map(|(n, d)| format!("{n}:\n{d}")).collect::<Vec<_>>().join("\n\n")
        );
    }

    /// Lower every fixture and require it to pass upstream's verification.
    ///
    /// Upstream runs `FileCheck` without `--strict-whitespace`, so the bar for
    /// compatibility is FileCheck-equivalence (whitespace-canonical). We assert
    /// that bar for every fixture except a small set of not-yet-implemented
    /// constructs, and *separately* report which fixtures are additionally
    /// byte-for-byte identical so any whitespace-only divergence is visible.
    #[test]
    fn corpus_ast2hir_report() {
        // Constructs whose lowering is not implemented yet (each needs one of
        // the corpus's most complex attribute forms: private_name / class
        // members / import_specifier).
        let not_yet_implemented: [&str; 0] = [];

        let mut byte_exact = Vec::new();
        let mut whitespace_only = Vec::new();
        let mut failed = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            let Some(expected) = f.expected_jshir() else {
                continue;
            };
            match lower_fixture(&f.name) {
                Ok(actual) if actual == expected => byte_exact.push(f.name.clone()),
                Ok(actual) if jsir_oracle::filecheck_equivalent(&expected, &actual) => {
                    whitespace_only.push(f.name.clone())
                }
                Ok(actual) => failed.push((
                    f.name.clone(),
                    jsir_oracle::byte_diff(&expected, &actual).unwrap_or_default(),
                )),
                Err(e) => failed.push((f.name.clone(), e)),
            }
        }
        eprintln!(
            "ast2hir: {} byte-exact, {} FileCheck-equivalent-only {:?}, {} unimplemented {:?}",
            byte_exact.len(),
            whitespace_only.len(),
            whitespace_only,
            not_yet_implemented.len(),
            not_yet_implemented,
        );

        // Anything that fails FileCheck and is NOT a known-unimplemented
        // construct is a real regression.
        let unexpected: Vec<_> = failed
            .into_iter()
            .filter(|(n, _)| !not_yet_implemented.contains(&n.as_str()))
            .collect();
        assert!(
            unexpected.is_empty(),
            "fixtures failed upstream FileCheck verification:\n{}",
            unexpected
                .iter()
                .map(|(n, d)| format!("{n}:\n{d}"))
                .collect::<Vec<_>>()
                .join("\n\n")
        );
    }
}
