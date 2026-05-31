//! `hir2ast`: lift the JSIR/JSHIR IR ([`jsir_ir::Op`]) back to the JSIR AST
//! ([`jsir_ast::Node`]). This inverts `ast2hir`: node-specific fields are
//! recovered from the IR structure (operands/attrs/regions), and each node's
//! base fields (loc/offsets/comments/scope/symbols) come from the op's trivia
//! (carried verbatim through the IR, mirroring upstream's `op->getLoc()`).

use std::collections::HashMap;

use jsir_ast::Node;
use jsir_ir::{
    Attr, IdentifierAttr, ImportSpecKind, NumericLiteralKeyAttr, Op, Position, PrivateNameAttr,
    Region, SourceLoc, StringLiteralKeyAttr, SymbolId, Trivia, ValueId,
};

use crate::ast_builder::AstBuilder;
use crate::node_builder::NodeBuilder;
use crate::LowerResult;

/// Lift a top-level `jsir.file` op into the in-tree [`Node`] AST (keeps the
/// byte-exact round-trip oracle).
pub fn hir2ast(file: &Op) -> LowerResult<Node> {
    NodeBuilder::into_node(hir2ast_with(NodeBuilder, file)?)
}

/// Lift a `jsir.file` op into any backend's AST via its [`AstBuilder`].
pub fn hir2ast_with<B: AstBuilder>(builder: B, file: &Op) -> LowerResult<B::Out> {
    let mut lifter = Lifter { defs: std::collections::HashMap::new(), b: builder };
    lifter.index(file);
    lifter.file(file)
}

struct Lifter<'a, B: AstBuilder> {
    /// Maps each SSA result value to the op that defines it.
    defs: HashMap<u32, &'a Op>,
    /// The backend that constructs AST values.
    b: B,
}

impl<'a, B: AstBuilder> Lifter<'a, B> {
    /// Build the value -> defining-op index over the whole op tree.
    fn index(&mut self, op: &'a Op) {
        for r in &op.results {
            self.defs.insert(r.0, op);
        }
        for region in &op.regions {
            for block in &region.blocks {
                for inner in &block.ops {
                    self.index(inner);
                }
            }
        }
    }

    fn def(&self, v: ValueId) -> LowerResult<&'a Op> {
        self.defs
            .get(&v.0)
            .copied()
            .ok_or_else(|| format!("no defining op for value %{}", v.0))
    }

    // -- structure helpers --------------------------------------------------

    fn file(&self, op: &Op) -> LowerResult<B::Out> {
        let program = self.stmt_region_single(&op.regions[0])?;
        let program = self.stmt(program)?;
        let comments = match get_attr(op, "comments") {
            Some(Attr::Array(items)) => {
                self.b.list(items.iter().map(|a| comment_attr_to_node(&self.b, a)).collect::<LowerResult<_>>()?)
            }
            _ => self.b.list(vec![]),
        };
        self.b.node("File", op.trivia.as_ref(), vec![("program", program), ("comments", comments)])
    }

    /// The single op in a one-block region (e.g. file's program region).
    fn stmt_region_single(&self, region: &'a Region) -> LowerResult<&'a Op> {
        let block = region.blocks.first().ok_or("empty region")?;
        block.ops.last().ok_or_else(|| "region has no ops".to_string())
    }

    /// Reconstruct a list of statements from a statements region (the
    /// 0-result ops are the statement roots; 1-result ops are operands).
    fn stmts(&self, region: &'a Region) -> LowerResult<Vec<B::Out>> {
        let Some(block) = region.blocks.first() else {
            return Ok(vec![]);
        };
        let mut out = Vec::new();
        for op in &block.ops {
            if op.results.is_empty() {
                out.push(self.stmt(op)?);
            }
        }
        Ok(out)
    }

    /// Reconstruct a single statement region (one statement op).
    fn stmt_region(&self, region: &'a Region) -> LowerResult<B::Out> {
        let block = region.blocks.first().ok_or("empty stmt region")?;
        let op = block
            .ops
            .iter()
            .rfind(|o| o.results.is_empty())
            .ok_or("no statement op in region")?;
        self.stmt(op)
    }

    /// Reconstruct the value produced by an expression region (ends in
    /// `jsir.expr_region_end(%v)`).
    fn expr_region(&self, region: &'a Region) -> LowerResult<B::Out> {
        let block = region.blocks.first().ok_or("empty expr region")?;
        let end = block.ops.last().ok_or("empty expr region block")?;
        if end.name != "jsir.expr_region_end" {
            return Err(format!("expected expr_region_end, got {}", end.name));
        }
        self.expr(end.operands[0])
    }

    // -- statements ---------------------------------------------------------

    fn stmt(&self, op: &Op) -> LowerResult<B::Out> {
        let t = op.trivia.as_ref();
        match op.name.as_str() {
            "jsir.program" => {
                let body = self.stmts(&op.regions[0])?;
                let directives = self.stmts(&op.regions[1])?;
                let interpreter = match get_attr(op, "interpreter") {
                    Some(a) => interpreter_to_node(&self.b, a)?,
                    None => self.b.null(),
                };
                let source_type = str_attr(op, "source_type")?;
                self.b.node(
                    "Program",
                    t,
                    vec![
                        ("interpreter", interpreter),
                        ("sourceType", self.b.string(source_type)),
                        ("body", self.b.list(body)),
                        ("directives", self.b.list(directives)),
                    ],
                )
            }
            "jsir.expression_statement" => {
                let expr = self.expr(op.operands[0])?;
                self.b.node("ExpressionStatement", t, vec![("expression", expr)])
            }
            "jsir.empty_statement" => self.b.node("EmptyStatement", t, vec![]),
            "jsir.debugger_statement" => self.b.node("DebuggerStatement", t, vec![]),
            "jsir.return_statement" => {
                let arg = match op.operands.first() {
                    Some(v) => self.expr(*v)?,
                    None => self.b.null(),
                };
                self.b.node("ReturnStatement", t, vec![("argument", arg)])
            }
            "jsir.throw_statement" => {
                let arg = self.expr(op.operands[0])?;
                self.b.node("ThrowStatement", t, vec![("argument", arg)])
            }
            "jsir.variable_declaration" => {
                let kind = str_attr(op, "kind")?;
                let declarators = self.exprs_region(&op.regions[0])?;
                self.b.node(
                    "VariableDeclaration",
                    t,
                    vec![
                        ("declarations", self.b.list(declarators)),
                        ("kind", self.b.string(kind)),
                    ],
                )
            }
            "jshir.block_statement" => {
                let body = self.stmts(&op.regions[0])?;
                let directives = self.stmts(&op.regions[1])?;
                self.b.node(
                    "BlockStatement",
                    t,
                    vec![("body", self.b.list(body)), ("directives", self.b.list(directives))],
                )
            }
            "jshir.if_statement" => {
                let test = self.expr(op.operands[0])?;
                let consequent = self.stmt_region(&op.regions[0])?;
                let alternate = if region_has_ops(&op.regions[1]) {
                    self.stmt_region(&op.regions[1])?
                } else {
                    self.b.null()
                };
                self.b.node(
                    "IfStatement",
                    t,
                    vec![("test", test), ("consequent", consequent), ("alternate", alternate)],
                )
            }
            "jshir.while_statement" => {
                let test = self.expr_region(&op.regions[0])?;
                let body = self.stmt_region(&op.regions[1])?;
                self.b.node("WhileStatement", t, vec![("test", test), ("body", body)])
            }
            "jshir.do_while_statement" => {
                let body = self.stmt_region(&op.regions[0])?;
                let test = self.expr_region(&op.regions[1])?;
                self.b.node("DoWhileStatement", t, vec![("body", body), ("test", test)])
            }
            "jsir.directive" => {
                let value = self.expr(op.operands[0])?;
                self.b.node("Directive", t, vec![("value", value)])
            }
            "jshir.with_statement" => {
                let object = self.expr(op.operands[0])?;
                let body = self.stmt_region(&op.regions[0])?;
                self.b.node("WithStatement", t, vec![("object", object), ("body", body)])
            }
            "jshir.labeled_statement" => {
                let label = ident_attr_node(&self.b, op, "label")?;
                let body = self.stmt_region(&op.regions[0])?;
                self.b.node("LabeledStatement", t, vec![("label", label), ("body", body)])
            }
            "jshir.break_statement" => {
                self.b.node("BreakStatement", t, vec![("label", opt_ident_attr_node(&self.b, op, "label")?)])
            }
            "jshir.continue_statement" => {
                self.b.node("ContinueStatement", t, vec![("label", opt_ident_attr_node(&self.b, op, "label")?)])
            }
            "jshir.for_statement" => {
                let init = self.for_init(&op.regions[0])?;
                let test = self.opt_expr_region(&op.regions[1])?;
                let update = self.opt_expr_region(&op.regions[2])?;
                let body = self.stmt_region(&op.regions[3])?;
                self.b.node(
                    "ForStatement",
                    t,
                    vec![("init", init), ("test", test), ("update", update), ("body", body)],
                )
            }
            "jshir.for_in_statement" => self.for_in_of(op, "ForInStatement", false),
            "jshir.for_of_statement" => self.for_in_of(op, "ForOfStatement", true),
            "jshir.try_statement" => self.try_statement(op),
            "jshir.switch_statement" => self.switch_statement(op),
            "jsir.function_declaration" => self.function(op, "FunctionDeclaration"),
            "jsir.class_declaration" => self.class(op, "ClassDeclaration"),
            "jsir.import_declaration" => self.import_declaration(op),
            "jsir.export_default_declaration" => {
                let block = op.regions[0].blocks.first().ok_or("empty export default")?;
                let last = block.ops.last().ok_or("empty export default block")?;
                let decl = if last.name == "jsir.expr_region_end" {
                    self.expr(last.operands[0])?
                } else {
                    self.stmt(last)?
                };
                self.b.node("ExportDefaultDeclaration", t, vec![("declaration", decl)])
            }
            "jsir.export_all_declaration" => {
                let source = string_attr_node(&self.b, op, "source")?;
                self.b.node("ExportAllDeclaration", t, vec![("source", source)])
            }
            "jsir.export_named_declaration" => self.export_named(op),
            other => Err(format!("hir2ast: unsupported statement op {other}")),
        }
    }

    /// The optional value of a `for`-statement test/update region.
    fn opt_expr_region(&self, region: &'a Region) -> LowerResult<B::Out> {
        if region_has_ops(region) {
            Ok(self.expr_region(region)?)
        } else {
            Ok(self.b.null())
        }
    }

    /// The `init` clause of a `for` statement (a declaration, an expression, or
    /// null).
    fn for_init(&self, region: &'a Region) -> LowerResult<B::Out> {
        if !region_has_ops(region) {
            return Ok(self.b.null());
        }
        let block = region.blocks.first().ok_or("empty for-init")?;
        let last = block.ops.last().ok_or("empty for-init block")?;
        if last.name == "jsir.expr_region_end" {
            Ok(self.expr(last.operands[0])?)
        } else {
            Ok(self.stmt(last)?)
        }
    }

    /// Reconstruct values from an exprs region (ends in `exprs_region_end`).
    fn exprs_region(&self, region: &'a Region) -> LowerResult<Vec<B::Out>> {
        let block = region.blocks.first().ok_or("empty exprs region")?;
        let end = block.ops.last().ok_or("empty exprs region block")?;
        if end.name != "jsir.exprs_region_end" {
            return Err(format!("expected exprs_region_end, got {}", end.name));
        }
        end.operands.iter().map(|v| Ok(self.expr_from_def(*v)?)).collect()
    }

    /// Reconstruct an op that produces a value, where the value may be a
    /// declarator / element rather than a plain expression.
    fn expr_from_def(&self, v: ValueId) -> LowerResult<B::Out> {
        let op = self.def(v)?;
        if op.name == "jsir.variable_declarator" {
            let id = self.lval(op.operands[0])?;
            let init = match op.operands.get(1) {
                Some(v) => self.expr(*v)?,
                None => self.b.null(),
            };
            return self.b.node(
                "VariableDeclarator",
                op.trivia.as_ref(),
                vec![("id", id), ("init", init)],
            );
        }
        self.expr(v)
    }

    // -- expressions (r-values) --------------------------------------------

    fn expr(&self, v: ValueId) -> LowerResult<B::Out> {
        let op = self.def(v)?;
        let t = op.trivia.as_ref();
        match op.name.as_str() {
            "jsir.numeric_literal" => {
                let value = f64_attr(op, "value")?;
                let extra = match get_attr(op, "extra") {
                    Some(Attr::NumericLiteralExtra { raw, value }) => self.b.record(
                        "NumericLiteralExtra",
                        vec![("raw", self.b.string(raw.clone())), ("rawValue", self.b.float(*value))],
                    ),
                    _ => self.b.absent(),
                };
                self.b.node("NumericLiteral", t, vec![("value", self.b.float(value)), ("extra", extra)])
            }
            "jsir.string_literal" => {
                let value = str_attr(op, "value")?;
                let extra = match get_attr(op, "extra") {
                    Some(Attr::StringLiteralExtra { raw, raw_value }) => self.b.record(
                        "StringLiteralExtra",
                        vec![("raw", self.b.string(raw.clone())), ("rawValue", self.b.string(raw_value.clone()))],
                    ),
                    _ => self.b.absent(),
                };
                self.b.node("StringLiteral", t, vec![("value", self.b.string(value)), ("extra", extra)])
            }
            "jsir.boolean_literal" => {
                let value = bool_attr(op, "value")?;
                self.b.node("BooleanLiteral", t, vec![("value", self.b.boolean(value))])
            }
            "jsir.null_literal" => self.b.node("NullLiteral", t, vec![]),
            "jsir.big_int_literal" => {
                let value = str_attr(op, "value")?;
                let extra = match get_attr(op, "extra") {
                    Some(Attr::BigIntLiteralExtra { raw, raw_value }) => self.b.record(
                        "BigIntLiteralExtra",
                        vec![("raw", self.b.string(raw.clone())), ("rawValue", self.b.string(raw_value.clone()))],
                    ),
                    _ => self.b.absent(),
                };
                self.b.node("BigIntLiteral", t, vec![("value", self.b.string(value)), ("extra", extra)])
            }
            "jsir.reg_exp_literal" => {
                let pattern = str_attr(op, "pattern")?;
                let flags = str_attr(op, "flags")?;
                let extra = match get_attr(op, "extra") {
                    Some(Attr::RegExpLiteralExtra { raw }) => {
                        self.b.record("RegExpLiteralExtra", vec![("raw", self.b.string(raw.clone()))])
                    }
                    _ => self.b.absent(),
                };
                self.b.node(
                    "RegExpLiteral",
                    t,
                    vec![("extra", extra), ("pattern", self.b.string(pattern)), ("flags", self.b.string(flags))],
                )
            }
            "jsir.identifier" | "jsir.identifier_ref" => {
                let name = str_attr(op, "name")?;
                self.b.node("Identifier", t, vec![("name", self.b.string(name))])
            }
            "jsir.this_expression" => self.b.node("ThisExpression", t, vec![]),
            "jsir.super" => self.b.node("Super", t, vec![]),
            "jsir.import" => self.b.node("Import", t, vec![]),
            "jsir.binary_expression" => {
                let left = self.expr(op.operands[0])?;
                let right = self.expr(op.operands[1])?;
                self.b.node(
                    "BinaryExpression",
                    t,
                    vec![("operator", self.b.string(str_attr(op, "operator_")?)), ("left", left), ("right", right)],
                )
            }
            "jshir.logical_expression" => {
                let left = self.expr(op.operands[0])?;
                let right = self.expr_region(&op.regions[0])?;
                self.b.node(
                    "LogicalExpression",
                    t,
                    vec![("operator", self.b.string(str_attr(op, "operator_")?)), ("left", left), ("right", right)],
                )
            }
            "jsir.unary_expression" => {
                let arg = self.expr(op.operands[0])?;
                self.b.node(
                    "UnaryExpression",
                    t,
                    vec![
                        ("operator", self.b.string(str_attr(op, "operator_")?)),
                        ("prefix", self.b.boolean(bool_attr(op, "prefix")?)),
                        ("argument", arg),
                    ],
                )
            }
            "jsir.update_expression" => {
                let arg = self.lval(op.operands[0])?;
                self.b.node(
                    "UpdateExpression",
                    t,
                    vec![
                        ("operator", self.b.string(str_attr(op, "operator_")?)),
                        ("prefix", self.b.boolean(bool_attr(op, "prefix")?)),
                        ("argument", arg),
                    ],
                )
            }
            "jsir.assignment_expression" => {
                let left = self.lval(op.operands[0])?;
                let right = self.expr(op.operands[1])?;
                self.b.node(
                    "AssignmentExpression",
                    t,
                    vec![("operator", self.b.string(str_attr(op, "operator_")?)), ("left", left), ("right", right)],
                )
            }
            "jsir.call_expression" => self.call(op, "CallExpression"),
            "jsir.new_expression" => self.call(op, "NewExpression"),
            "jsir.member_expression" => self.member(op, "MemberExpression"),
            "jsir.spread_element" => {
                let arg = self.expr(op.operands[0])?;
                self.b.node("SpreadElement", t, vec![("argument", arg)])
            }
            "jsir.parenthesized_expression" => {
                let e = self.expr(op.operands[0])?;
                self.b.node("ParenthesizedExpression", t, vec![("expression", e)])
            }
            "jsir.array_expression" => {
                let mut elements = Vec::new();
                for v in &op.operands {
                    let def = self.def(*v)?;
                    if def.name == "jsir.none" {
                        elements.push(self.b.null());
                    } else {
                        elements.push(self.expr(*v)?);
                    }
                }
                self.b.node("ArrayExpression", t, vec![("elements", self.b.list(elements))])
            }
            "jsir.sequence_expression" => {
                let mut exprs = Vec::new();
                for v in &op.operands {
                    exprs.push(self.expr(*v)?);
                }
                self.b.node("SequenceExpression", t, vec![("expressions", self.b.list(exprs))])
            }
            "jshir.conditional_expression" => {
                let test = self.expr(op.operands[0])?;
                // Regions are [alternate, consequent].
                let alternate = self.expr_region(&op.regions[0])?;
                let consequent = self.expr_region(&op.regions[1])?;
                self.b.node(
                    "ConditionalExpression",
                    t,
                    vec![("test", test), ("consequent", consequent), ("alternate", alternate)],
                )
            }
            "jsir.optional_member_expression" => {
                let object = self.expr(op.operands[0])?;
                let (property, computed) = match get_attr(op, "literal_property") {
                    Some(Attr::Identifier(id)) => (identifier_attr_to_node(&self.b, id)?, false),
                    _ => (self.expr(op.operands[1])?, true),
                };
                self.b.node(
                    "OptionalMemberExpression",
                    t,
                    vec![
                        ("object", object),
                        ("property", property),
                        ("computed", self.b.boolean(computed)),
                        ("optional", self.b.boolean(bool_attr(op, "optional")?)),
                    ],
                )
            }
            "jsir.yield_expression" => {
                let arg = match op.operands.first() {
                    Some(v) => self.expr(*v)?,
                    None => self.b.null(),
                };
                self.b.node(
                    "YieldExpression",
                    t,
                    vec![("delegate", self.b.boolean(bool_attr(op, "delegate")?)), ("argument", arg)],
                )
            }
            "jsir.await_expression" => {
                let arg = self.expr(op.operands[0])?;
                self.b.node("AwaitExpression", t, vec![("argument", arg)])
            }
            "jsir.meta_property" => {
                let meta = ident_attr_node(&self.b, op, "meta")?;
                let property = ident_attr_node(&self.b, op, "property")?;
                self.b.node("MetaProperty", t, vec![("meta", meta), ("property", property)])
            }
            "jsir.directive_literal" => {
                let value = str_attr(op, "value")?;
                let extra = match get_attr(op, "extra") {
                    Some(Attr::DirectiveLiteralExtra { raw, raw_value }) => self.b.record(
                        "DirectiveLiteralExtra",
                        vec![("raw", self.b.string(raw.clone())), ("rawValue", self.b.string(raw_value.clone()))],
                    ),
                    _ => self.b.absent(),
                };
                self.b.node("DirectiveLiteral", t, vec![("value", self.b.string(value)), ("extra", extra)])
            }
            "jsir.function_expression" => self.function(op, "FunctionExpression"),
            "jsir.class_expression" => self.class(op, "ClassExpression"),
            "jsir.arrow_function_expression" => self.arrow(op),
            "jsir.object_expression" => self.object_expression(op),
            "jsir.template_literal" => self.template_literal(op),
            "jsir.tagged_template_expression" => {
                let tag = self.expr(op.operands[0])?;
                let quasi = self.expr(op.operands[1])?;
                self.b.node("TaggedTemplateExpression", t, vec![("tag", tag), ("quasi", quasi)])
            }
            other => Err(format!("hir2ast: unsupported expression op {other}")),
        }
    }

    fn call(&self, op: &Op, ty: &str) -> LowerResult<B::Out> {
        let callee = self.expr(op.operands[0])?;
        let mut args = Vec::new();
        for v in &op.operands[1..] {
            args.push(self.expr(*v)?);
        }
        self.b.node(ty, op.trivia.as_ref(), vec![("callee", callee), ("arguments", self.b.list(args))])
    }

    fn member(&self, op: &Op, ty: &str) -> LowerResult<B::Out> {
        let object = self.expr(op.operands[0])?;
        let (property, computed) = match get_attr(op, "literal_property") {
            Some(Attr::Identifier(id)) => (identifier_attr_to_node(&self.b, id)?, false),
            // `obj.#priv` carries a PrivateName, not a computed operand.
            Some(Attr::PrivateName(p)) => (private_name_attr_to_node(&self.b, p)?, false),
            _ => {
                let v = op.operands[1];
                (self.expr(v)?, true)
            }
        };
        self.b.node(
            ty,
            op.trivia.as_ref(),
            vec![("object", object), ("property", property), ("computed", self.b.boolean(computed))],
        )
    }

    // -- l-values -----------------------------------------------------------

    fn lval(&self, v: ValueId) -> LowerResult<B::Out> {
        let op = self.def(v)?;
        let t = op.trivia.as_ref();
        match op.name.as_str() {
            "jsir.identifier_ref" => self.expr(v),
            "jsir.member_expression_ref" => self.member(op, "MemberExpression"),
            "jsir.parenthesized_expression_ref" => {
                let e = self.lval(op.operands[0])?;
                self.b.node("ParenthesizedExpression", t, vec![("expression", e)])
            }
            "jsir.assignment_pattern_ref" => {
                let left = self.lval(op.operands[0])?;
                let right = self.expr(op.operands[1])?;
                self.b.node("AssignmentPattern", t, vec![("left", left), ("right", right)])
            }
            "jsir.rest_element_ref" => {
                let arg = self.lval(op.operands[0])?;
                self.b.node("RestElement", t, vec![("argument", arg)])
            }
            "jsir.array_pattern_ref" => {
                let mut elements = Vec::new();
                for v in &op.operands {
                    if self.def(*v)?.name == "jsir.none" {
                        elements.push(self.b.null());
                    } else {
                        elements.push(self.lval(*v)?);
                    }
                }
                self.b.node("ArrayPattern", t, vec![("elements", self.b.list(elements))])
            }
            "jsir.object_pattern_ref" => {
                let props = self.object_members(&op.regions[0], true)?;
                self.b.node("ObjectPattern", t, vec![("properties", self.b.list(props))])
            }
            _ => self.expr(v),
        }
    }

    // -- composite reconstructions -----------------------------------------

    fn for_in_of(&self, op: &Op, ty: &str, is_of: bool) -> LowerResult<B::Out> {
        let t = op.trivia.as_ref();
        let left = match get_attr(op, "left_declaration") {
            Some(Attr::ForInOfDeclaration(d)) => {
                // Rebuild the `let x` VariableDeclaration from the attr.
                let id = self.lval(op.operands[0])?;
                let declarator_trivia = Trivia {
                    loc: Some(loc(d.r_start_line, d.r_start_col, d.r_end_line, d.r_end_col)),
                    start: Some(d.r_start_index),
                    end: Some(d.r_end_index),
                    scope_uid: Some(d.r_scope),
                    defined_symbols: Some(
                        d.symbols
                            .iter()
                            .map(|(name, scope)| SymbolId {
                                name: name.clone(),
                                def_scope_uid: Some(*scope),
                            })
                            .collect(),
                    ),
                    ..Default::default()
                };
                let declarator = self.b.node(
                    "VariableDeclarator",
                    Some(&declarator_trivia),
                    vec![("id", id), ("init", self.b.null())],
                )?;
                let decl_trivia = Trivia {
                    loc: Some(loc(d.d_start_line, d.d_start_col, d.d_end_line, d.d_end_col)),
                    start: Some(d.d_start_index),
                    end: Some(d.d_end_index),
                    scope_uid: Some(d.d_scope),
                    ..Default::default()
                };
                self.b.node(
                    "VariableDeclaration",
                    Some(&decl_trivia),
                    vec![("declarations", self.b.list(vec![declarator])), ("kind", self.b.string(d.kind.clone()))],
                )?
            }
            _ => self.lval(op.operands[0])?,
        };
        let right = self.expr(op.operands[1])?;
        let body = self.stmt_region(&op.regions[0])?;
        let mut fields = vec![("left", left), ("right", right), ("body", body)];
        if is_of {
            fields.push(("await", self.b.boolean(bool_attr(op, "await")?)));
        }
        self.b.node(ty, t, fields)
    }

    fn try_statement(&self, op: &Op) -> LowerResult<B::Out> {
        let block = self.stmt_region(&op.regions[0])?;
        let handler = if region_has_ops(&op.regions[1]) {
            // Handler region: [identifier_ref?, catch_clause].
            let blk = op.regions[1].blocks.first().ok_or("empty handler")?;
            let cc = blk.ops.last().ok_or("no catch_clause")?;
            let param = match cc.operands.first() {
                Some(v) => self.lval(*v)?,
                None => self.b.null(),
            };
            let body = self.stmt_region(&cc.regions[0])?;
            self.b.node("CatchClause", cc.trivia.as_ref(), vec![("param", param), ("body", body)])?
        } else {
            self.b.null()
        };
        let finalizer = if region_has_ops(&op.regions[2]) {
            self.stmt_region(&op.regions[2])?
        } else {
            self.b.null()
        };
        self.b.node(
            "TryStatement",
            op.trivia.as_ref(),
            vec![("block", block), ("handler", handler), ("finalizer", finalizer)],
        )
    }

    fn switch_statement(&self, op: &Op) -> LowerResult<B::Out> {
        let disc = self.expr(op.operands[0])?;
        let block = op.regions[0].blocks.first().ok_or("empty switch")?;
        let mut cases = Vec::new();
        for case in &block.ops {
            let test = if region_has_ops(&case.regions[0]) {
                self.expr_region(&case.regions[0])?
            } else {
                self.b.null()
            };
            let consequent = self.stmts(&case.regions[1])?;
            cases.push(self.b.node(
                "SwitchCase",
                case.trivia.as_ref(),
                vec![("test", test), ("consequent", self.b.list(consequent))],
            )?);
        }
        self.b.node(
            "SwitchStatement",
            op.trivia.as_ref(),
            vec![("discriminant", disc), ("cases", self.b.list(cases))],
        )
    }

    fn function(&self, op: &Op, ty: &str) -> LowerResult<B::Out> {
        let id = opt_ident_attr_node(&self.b, op, "id")?;
        let params = self.params_region(&op.regions[0])?;
        let body = self.stmt_region(&op.regions[1])?;
        self.b.node(
            ty,
            op.trivia.as_ref(),
            vec![
                ("id", id),
                ("params", self.b.list(params)),
                ("generator", self.b.boolean(bool_attr(op, "generator")?)),
                ("async", self.b.boolean(bool_attr(op, "async")?)),
                ("body", body),
            ],
        )
    }

    fn arrow(&self, op: &Op) -> LowerResult<B::Out> {
        // Operand segments: [id (0/1), params...]; arrows have no id.
        let params = op.operands.iter().map(|v| Ok(self.lval(*v)?)).collect::<LowerResult<Vec<_>>>()?;
        let block = op.regions[0].blocks.first().ok_or("empty arrow body")?;
        let last = block.ops.last().ok_or("empty arrow body block")?;
        let body = if last.name == "jsir.expr_region_end" {
            self.expr(last.operands[0])?
        } else {
            self.stmt(last)?
        };
        self.b.node(
            "ArrowFunctionExpression",
            op.trivia.as_ref(),
            vec![
                ("params", self.b.list(params)),
                ("generator", self.b.boolean(bool_attr(op, "generator")?)),
                ("async", self.b.boolean(bool_attr(op, "async")?)),
                ("body", body),
            ],
        )
    }

    /// Reconstruct a parameter list (ExprsRegion of pattern-refs).
    fn params_region(&self, region: &'a Region) -> LowerResult<Vec<B::Out>> {
        let block = region.blocks.first().ok_or("empty params region")?;
        let end = block.ops.last().ok_or("empty params block")?;
        end.operands.iter().map(|v| Ok(self.lval(*v)?)).collect()
    }

    fn class(&self, op: &Op, ty: &str) -> LowerResult<B::Out> {
        let id = opt_ident_attr_node(&self.b, op, "id")?;
        // `extends X` is the class op's lone operand, if present.
        let super_class = match op.operands.first() {
            Some(v) => self.expr(*v)?,
            None => self.b.null(),
        };
        let class_body = self.stmt_region_single(&op.regions[0])?;
        let members = self.class_members(&class_body.regions[0])?;
        let body = self.b.node("ClassBody", class_body.trivia.as_ref(), vec![("body", self.b.list(members))])?;
        self.b.node(
            ty,
            op.trivia.as_ref(),
            vec![("id", id), ("superClass", super_class), ("body", body)],
        )
    }

    fn class_members(&self, region: &'a Region) -> LowerResult<Vec<B::Out>> {
        let Some(block) = region.blocks.first() else {
            return Ok(vec![]);
        };
        let mut out = Vec::new();
        for m in &block.ops {
            if m.results.is_empty() {
                out.push(self.class_member(m)?);
            }
        }
        Ok(out)
    }

    fn class_member(&self, m: &Op) -> LowerResult<B::Out> {
        let t = m.trivia.as_ref();
        let node = match m.name.as_str() {
            "jsir.class_property" => {
                let (key, computed) = self.key_of(m, m.operands.first().copied())?;
                let value = if region_has_ops(&m.regions[0]) {
                    self.expr_region(&m.regions[0])?
                } else {
                    self.b.null()
                };
                self.b.node(
                    "ClassProperty",
                    t,
                    vec![("key", key), ("value", value), ("static", self.b.boolean(bool_attr(m, "static_")?)), ("computed", self.b.boolean(computed))],
                )?
            }
            "jsir.class_private_property" => {
                let key = private_name_node(&self.b, m, "key")?;
                let value = if region_has_ops(&m.regions[0]) {
                    self.expr_region(&m.regions[0])?
                } else {
                    self.b.null()
                };
                self.b.node(
                    "ClassPrivateProperty",
                    t,
                    vec![("key", key), ("value", value), ("static", self.b.boolean(bool_attr(m, "static_")?))],
                )?
            }
            "jsir.class_method" => {
                // segments [params, computed_key].
                let computed = get_attr(m, "literal_key").is_none();
                let nparams = m.operands.len() - if computed { 1 } else { 0 };
                let params: Vec<_> = m.operands[..nparams].iter().map(|v| Ok(self.lval(*v)?)).collect::<LowerResult<_>>()?;
                let (key, _) = self.key_of(m, if computed { m.operands.get(nparams).copied() } else { None })?;
                let body = self.stmt_region(&m.regions[0])?;
                self.b.node(
                    "ClassMethod",
                    t,
                    vec![
                        ("params", self.b.list(params)),
                        ("generator", self.b.boolean(bool_attr(m, "generator")?)),
                        ("async", self.b.boolean(bool_attr(m, "async")?)),
                        ("body", body),
                        ("key", key),
                        ("kind", self.b.string(str_attr(m, "kind")?)),
                        ("computed", self.b.boolean(computed)),
                        ("static", self.b.boolean(bool_attr(m, "static_")?)),
                    ],
                )?
            }
            "jsir.class_private_method" => {
                let key = private_name_node(&self.b, m, "key")?;
                let params: Vec<_> = m.operands.iter().map(|v| Ok(self.lval(*v)?)).collect::<LowerResult<_>>()?;
                let body = self.stmt_region(&m.regions[0])?;
                self.b.node(
                    "ClassPrivateMethod",
                    t,
                    vec![
                        ("params", self.b.list(params)),
                        ("generator", self.b.boolean(bool_attr(m, "generator")?)),
                        ("async", self.b.boolean(bool_attr(m, "async")?)),
                        ("body", body),
                        ("key", key),
                        ("kind", self.b.string(str_attr(m, "kind")?)),
                        ("static", self.b.boolean(bool_attr(m, "static_")?)),
                    ],
                )?
            }
            other => return Err(format!("hir2ast: unsupported class member {other}")),
        };
        Ok(node)
    }

    fn object_expression(&self, op: &Op) -> LowerResult<B::Out> {
        let props = self.object_members(&op.regions[0], false)?;
        self.b.node("ObjectExpression", op.trivia.as_ref(), vec![("properties", self.b.list(props))])
    }

    /// Reconstruct object members (properties/methods/spread or, for patterns,
    /// property-refs/rest) from an ExprsRegion.
    fn object_members(&self, region: &'a Region, pattern: bool) -> LowerResult<Vec<B::Out>> {
        let block = region.blocks.first().ok_or("empty object region")?;
        let end = block.ops.last().ok_or("empty object block")?;
        let mut out = Vec::new();
        for v in &end.operands {
            let def = self.def(*v)?;
            let t = def.trivia.as_ref();
            let n = match def.name.as_str() {
                "jsir.object_property" | "jsir.object_property_ref" => {
                    let computed = get_attr(def, "literal_key").is_none();
                    let (key, value_v) = if computed {
                        (self.expr(def.operands[0])?, def.operands[1])
                    } else {
                        let key = literal_key_to_node(&self.b, get_attr(def, "literal_key").unwrap())?;
                        (key, def.operands[0])
                    };
                    let value = if pattern { self.lval(value_v)? } else { self.expr(value_v)? };
                    self.b.node(
                        "ObjectProperty",
                        t,
                        vec![
                            ("key", key),
                            ("value", value),
                            ("computed", self.b.boolean(computed)),
                            ("shorthand", self.b.boolean(bool_attr(def, "shorthand")?)),
                        ],
                    )?
                }
                "jsir.object_method" => self.object_method(def)?,
                "jsir.spread_element" => {
                    let arg = self.expr(def.operands[0])?;
                    self.b.node("SpreadElement", t, vec![("argument", arg)])?
                }
                "jsir.rest_element_ref" => {
                    let arg = self.lval(def.operands[0])?;
                    self.b.node("RestElement", t, vec![("argument", arg)])?
                }
                other => return Err(format!("hir2ast: unsupported object member {other}")),
            };
            out.push(n);
        }
        Ok(out)
    }

    fn object_method(&self, op: &Op) -> LowerResult<B::Out> {
        // segments [computed_key, params].
        let computed = get_attr(op, "literal_key").is_none();
        let (key, params_start) = if computed {
            (self.expr(op.operands[0])?, 1)
        } else {
            let (k, _) = self.key_of(op, None)?;
            (k, 0)
        };
        let params: Vec<_> = op.operands[params_start..].iter().map(|v| Ok(self.lval(*v)?)).collect::<LowerResult<_>>()?;
        let body = self.stmt_region(&op.regions[0])?;
        self.b.node(
            "ObjectMethod",
            op.trivia.as_ref(),
            vec![
                ("key", key),
                ("params", self.b.list(params)),
                ("generator", self.b.boolean(bool_attr(op, "generator")?)),
                ("async", self.b.boolean(bool_attr(op, "async")?)),
                ("computed", self.b.boolean(computed)),
                ("kind", self.b.string(str_attr(op, "kind")?)),
                ("body", body),
            ],
        )
    }

    /// Reconstruct an object/class key from a `literal_key` attribute, or null.
    fn key_of(&self, op: &Op, computed: Option<ValueId>) -> LowerResult<(B::Out, bool)> {
        if let Some(v) = computed {
            return Ok((self.expr(v)?, true));
        }
        match get_attr(op, "literal_key") {
            Some(a) => Ok((literal_key_to_node(&self.b, a)?, false)),
            None => Ok((self.b.null(), false)),
        }
    }

    fn template_literal(&self, op: &Op) -> LowerResult<B::Out> {
        let n = get_attr(op, "operandSegmentSizes");
        let nquasis = match n {
            Some(Attr::I32Array(v)) => v[0] as usize,
            _ => return Err("template_literal missing operandSegmentSizes".into()),
        };
        let mut quasis = Vec::new();
        for v in &op.operands[..nquasis] {
            let te = self.def(*v)?;
            let value_op = self.def(te.operands[0])?;
            let cooked = match get_attr(value_op, "cooked") {
                Some(Attr::Str(s)) => self.b.string(s.clone()),
                _ => self.b.null(),
            };
            let raw = str_attr(value_op, "raw")?;
            let value = self.b.record("TemplateElementValue", vec![("cooked", cooked), ("raw", self.b.string(raw))]);
            quasis.push(self.b.node(
                "TemplateElement",
                te.trivia.as_ref(),
                vec![("tail", self.b.boolean(bool_attr(te, "tail")?)), ("value", value)],
            )?);
        }
        let exprs: Vec<_> = op.operands[nquasis..].iter().map(|v| Ok(self.expr(*v)?)).collect::<LowerResult<_>>()?;
        self.b.node(
            "TemplateLiteral",
            op.trivia.as_ref(),
            vec![("quasis", self.b.list(quasis)), ("expressions", self.b.list(exprs))],
        )
    }

    fn import_declaration(&self, op: &Op) -> LowerResult<B::Out> {
        let source = string_attr_node(&self.b, op, "source")?;
        let specifiers = match get_attr(op, "specifiers") {
            Some(Attr::Array(items)) => items.iter().map(|a| import_specifier_node(&self.b, a)).collect::<LowerResult<_>>()?,
            _ => vec![],
        };
        self.b.node(
            "ImportDeclaration",
            op.trivia.as_ref(),
            vec![("specifiers", self.b.list(specifiers)), ("source", source)],
        )
    }

    fn export_named(&self, op: &Op) -> LowerResult<B::Out> {
        let source = match get_attr(op, "source") {
            Some(a) => literal_key_to_node(&self.b, a)?,
            None => self.b.null(),
        };
        let specifiers = match get_attr(op, "specifiers") {
            Some(Attr::Array(items)) => items.iter().map(|a| export_specifier_node(&self.b, a)).collect::<LowerResult<_>>()?,
            _ => vec![],
        };
        let declaration = if region_has_ops(&op.regions[0]) {
            self.stmt_region(&op.regions[0])?
        } else {
            self.b.null()
        };
        self.b.node(
            "ExportNamedDeclaration",
            op.trivia.as_ref(),
            vec![("declaration", declaration), ("specifiers", self.b.list(specifiers)), ("source", source)],
        )
    }
}


/// Build an `Identifier` node from a `JsirIdentifierAttr` (member properties,
/// object keys, etc.).
fn identifier_attr_to_node<B: AstBuilder>(b: &B, id: &IdentifierAttr) -> LowerResult<B::Out> {
    let trivia = Trivia {
        loc: Some(SourceLoc {
            start: jsir_ir::Position { line: id.start_line, column: id.start_col },
            end: jsir_ir::Position { line: id.end_line, column: id.end_col },
            identifier_name: Some(id.identifier_name.clone()),
        }),
        start: Some(id.start_index),
        end: Some(id.end_index),
        scope_uid: Some(id.scope_uid),
        ..Default::default()
    };
    b.node("Identifier", Some(&trivia), vec![("name", b.string(id.name.clone()))])
}

/// Reconstruct an InterpreterDirective node from the program's `interpreter` attr.
fn interpreter_to_node<B: AstBuilder>(b: &B, attr: &Attr) -> LowerResult<B::Out> {
    let Attr::InterpreterDirective(d) = attr else {
        return Err("expected interpreter_directive attr".into());
    };
    let trivia = Trivia {
        loc: Some(SourceLoc {
            start: jsir_ir::Position { line: d.start_line, column: d.start_col },
            end: jsir_ir::Position { line: d.end_line, column: d.end_col },
            identifier_name: None,
        }),
        start: Some(d.start_index),
        end: Some(d.end_index),
        ..Default::default()
    };
    b.node("InterpreterDirective", Some(&trivia), vec![("value", b.string(d.value.clone()))])
}

/// Reconstruct a CommentLine/CommentBlock node from a comment attr.
fn comment_attr_to_node<B: AstBuilder>(b: &B, attr: &Attr) -> LowerResult<B::Out> {
    let Attr::Comment(c) = attr else {
        return Err("expected comment attr in file comments".into());
    };
    let ty = if c.block { "CommentBlock" } else { "CommentLine" };
    let trivia = Trivia {
        loc: Some(SourceLoc {
            start: jsir_ir::Position { line: c.start_line, column: c.start_col },
            end: jsir_ir::Position { line: c.end_line, column: c.end_col },
            identifier_name: None,
        }),
        start: Some(c.start_index),
        end: Some(c.end_index),
        ..Default::default()
    };
    Ok(b.node(ty, Some(&trivia), vec![("value", b.string(c.value.clone()))])?)
}

// ---------------------------------------------------------------------------
// Attribute accessors
// ---------------------------------------------------------------------------

fn get_attr<'a>(op: &'a Op, key: &str) -> Option<&'a Attr> {
    op.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn str_attr(op: &Op, key: &str) -> LowerResult<String> {
    match get_attr(op, key) {
        Some(Attr::Str(s)) => Ok(s.clone()),
        _ => Err(format!("{}: missing string attr {key}", op.name)),
    }
}

fn f64_attr(op: &Op, key: &str) -> LowerResult<f64> {
    match get_attr(op, key) {
        Some(Attr::F64(f)) => Ok(*f),
        _ => Err(format!("{}: missing f64 attr {key}", op.name)),
    }
}

fn bool_attr(op: &Op, key: &str) -> LowerResult<bool> {
    match get_attr(op, key) {
        Some(Attr::Bool(b)) => Ok(*b),
        _ => Err(format!("{}: missing bool attr {key}", op.name)),
    }
}

fn region_has_ops(region: &Region) -> bool {
    region.blocks.iter().any(|b| !b.ops.is_empty())
}

fn loc(sl: i64, sc: i64, el: i64, ec: i64) -> SourceLoc {
    SourceLoc {
        start: Position { line: sl, column: sc },
        end: Position { line: el, column: ec },
        identifier_name: None,
    }
}

/// An `Identifier` node from a required identifier-valued attribute.
fn ident_attr_node<B: AstBuilder>(b: &B, op: &Op, key: &str) -> LowerResult<B::Out> {
    match get_attr(op, key) {
        Some(Attr::Identifier(id)) => identifier_attr_to_node(b, id),
        _ => Err(format!("{}: missing identifier attr {key}", op.name)),
    }
}

/// An `Identifier` node from an optional identifier attribute, else null.
fn opt_ident_attr_node<B: AstBuilder>(b: &B, op: &Op, key: &str) -> LowerResult<B::Out> {
    match get_attr(op, key) {
        Some(Attr::Identifier(id)) => Ok(identifier_attr_to_node(b, id)?),
        _ => Ok(b.null()),
    }
}

/// A `PrivateName` node from a `private_name` attribute.
fn private_name_node<B: AstBuilder>(b: &B, op: &Op, key: &str) -> LowerResult<B::Out> {
    let Some(Attr::PrivateName(p)) = get_attr(op, key) else {
        return Err(format!("{}: missing private_name attr {key}", op.name));
    };
    private_name_attr_to_node(b, p)
}

fn private_name_attr_to_node<B: AstBuilder>(b: &B, p: &PrivateNameAttr) -> LowerResult<B::Out> {
    let trivia = Trivia {
        loc: Some(loc(p.start_line, p.start_col, p.end_line, p.end_col)),
        start: Some(p.start_index),
        end: Some(p.end_index),
        scope_uid: Some(p.scope_uid),
        ..Default::default()
    };
    let id = identifier_attr_to_node(b, &p.id)?;
    b.node("PrivateName", Some(&trivia), vec![("id", id)])
}

/// A `StringLiteral` node from a `source`-style string-literal attribute.
fn string_attr_node<B: AstBuilder>(b: &B, op: &Op, key: &str) -> LowerResult<B::Out> {
    match get_attr(op, key) {
        Some(a) => literal_key_to_node(b, a),
        None => Err(format!("{}: missing string attr {key}", op.name)),
    }
}

/// Reconstruct an Identifier/StringLiteral/NumericLiteral node from a key/source
/// attribute (identifier or full literal attr form).
fn literal_key_to_node<B: AstBuilder>(b: &B, attr: &Attr) -> LowerResult<B::Out> {
    match attr {
        Attr::Identifier(id) => identifier_attr_to_node(b, id),
        Attr::StringLiteralKey(k) => string_literal_key_to_node(b, k),
        Attr::NumericLiteralKey(k) => numeric_literal_key_to_node(b, k),
        _ => Err("unexpected literal key attr".into()),
    }
}

fn string_literal_key_to_node<B: AstBuilder>(b: &B, k: &StringLiteralKeyAttr) -> LowerResult<B::Out> {
    let trivia = key_trivia(k.start_line, k.start_col, k.end_line, k.end_col, k.start_index, k.end_index, k.scope_uid);
    let extra = b.record(
        "StringLiteralExtra",
        vec![("raw", b.string(k.raw.clone())), ("rawValue", b.string(k.raw_value.clone()))],
    );
    b.node("StringLiteral", Some(&trivia), vec![("value", b.string(k.value.clone())), ("extra", extra)])
}

fn numeric_literal_key_to_node<B: AstBuilder>(b: &B, k: &NumericLiteralKeyAttr) -> LowerResult<B::Out> {
    let trivia = key_trivia(k.start_line, k.start_col, k.end_line, k.end_col, k.start_index, k.end_index, k.scope_uid);
    let extra = b.record(
        "NumericLiteralExtra",
        vec![("raw", b.string(k.raw.clone())), ("rawValue", b.float(k.raw_value))],
    );
    b.node("NumericLiteral", Some(&trivia), vec![("value", b.float(k.value)), ("extra", extra)])
}

#[allow(clippy::too_many_arguments)]
fn key_trivia(sl: i64, sc: i64, el: i64, ec: i64, si: i64, ei: i64, scope: i64) -> Trivia {
    Trivia {
        loc: Some(loc(sl, sc, el, ec)),
        start: Some(si),
        end: Some(ei),
        scope_uid: Some(scope),
        ..Default::default()
    }
}

/// Reconstruct an import specifier node from an `import_*_specifier` attr.
fn import_specifier_node<B: AstBuilder>(b: &B, attr: &Attr) -> LowerResult<B::Out> {
    let Attr::ImportSpecifier(s) = attr else {
        return Err("expected import specifier attr".into());
    };
    let trivia = Trivia {
        loc: Some(loc(s.start_line, s.start_col, s.end_line, s.end_col)),
        start: Some(s.start_index),
        end: Some(s.end_index),
        scope_uid: Some(s.scope_uid),
        defined_symbols: Some(vec![SymbolId { name: s.sym_name.clone(), def_scope_uid: Some(s.sym_scope) }]),
        ..Default::default()
    };
    let local = identifier_attr_to_node(b, &s.local)?;
    let n = match s.kind {
        ImportSpecKind::Named => {
            let imported = literal_key_to_node(b, s.imported.as_ref().ok_or("named import missing imported")?)?;
            b.node("ImportSpecifier", Some(&trivia), vec![("imported", imported), ("local", local)])?
        }
        ImportSpecKind::Default => b.node("ImportDefaultSpecifier", Some(&trivia), vec![("local", local)])?,
        ImportSpecKind::Namespace => b.node("ImportNamespaceSpecifier", Some(&trivia), vec![("local", local)])?,
    };
    Ok(n)
}

/// Reconstruct an `ExportSpecifier` node from an `export_specifier` attr.
fn export_specifier_node<B: AstBuilder>(b: &B, attr: &Attr) -> LowerResult<B::Out> {
    let Attr::ExportSpecifier(s) = attr else {
        return Err("expected export specifier attr".into());
    };
    let trivia = Trivia {
        loc: Some(loc(s.start_line, s.start_col, s.end_line, s.end_col)),
        start: Some(s.start_index),
        end: Some(s.end_index),
        scope_uid: Some(s.scope_uid),
        ..Default::default()
    };
    let exported = literal_key_to_node(b, &s.exported)?;
    let local = literal_key_to_node(b, &s.local)?;
    Ok(b.node("ExportSpecifier", Some(&trivia), vec![("exported", exported), ("local", local)])?)
}
