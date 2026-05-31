//! Convert our JSIR-schema AST (jsir_ast::Node) into an swc `Program`, so
//! swc's code generator can emit JS (`ast2source`). Spans are irrelevant to
//! codegen, so everything uses `DUMMY_SP`.

use jsir_ast::model::FieldValue;
use jsir_ast::Node;
use swc_common::comments::{Comment, CommentKind, Comments, SingleThreadedComments};
use swc_common::{BytePos, Span, DUMMY_SP};
use swc_ecma_ast as ast;

pub type Conv<T> = Result<T, String>;

/// Convert a `File` node into an swc `Program` (Script or Module).
pub fn to_program(file: &Node) -> Conv<ast::Program> {
    let program = nfield(file, "program")?;
    let body_nodes = list(program, "body")?;

    // The directive prologue (`'use strict'`) is emitted as leading string-literal
    // expression statements; re-parsing splits them back into directives.
    let mut directive_stmts: Vec<ast::Stmt> = Vec::new();
    if let Ok(dirs) = list(program, "directives") {
        for d in dirs {
            let lit = nfield(node_of(d)?, "value")?; // DirectiveLiteral
            directive_stmts.push(ast::Stmt::Expr(ast::ExprStmt {
                span: DUMMY_SP,
                expr: Box::new(ast::Expr::Lit(ast::Lit::Str(ast::Str {
                    span: DUMMY_SP,
                    value: str_field(lit, "value")?.into(),
                    raw: None,
                }))),
            }));
        }
    }
    // The interpreter directive (`#!...`) becomes the program shebang.
    let shebang = opt_node(program, "interpreter")?
        .map(|i| str_field(i, "value").map(|s| s.into()))
        .transpose()?;

    if str_field(program, "sourceType")? == "module" {
        let mut body: Vec<ast::ModuleItem> =
            directive_stmts.into_iter().map(ast::ModuleItem::Stmt).collect();
        for el in body_nodes {
            body.push(module_item(node_of(el)?)?);
        }
        Ok(ast::Program::Module(ast::Module { span: DUMMY_SP, body, shebang }))
    } else {
        let mut body = directive_stmts;
        for el in body_nodes {
            body.push(stmt(node_of(el)?)?);
        }
        Ok(ast::Program::Script(ast::Script { span: DUMMY_SP, body, shebang }))
    }
}

fn module_item(n: &Node) -> Conv<ast::ModuleItem> {
    if n.ty.starts_with("Import") || n.ty.starts_with("Export") {
        Ok(ast::ModuleItem::ModuleDecl(module_decl(n)?))
    } else {
        Ok(ast::ModuleItem::Stmt(stmt(n)?))
    }
}

fn module_decl(n: &Node) -> Conv<ast::ModuleDecl> {
    Ok(match n.ty.as_str() {
        "ImportDeclaration" => ast::ModuleDecl::Import(ast::ImportDecl {
            span: DUMMY_SP,
            specifiers: list(n, "specifiers")?.iter().map(|s| import_specifier(node_of(s)?)).collect::<Conv<_>>()?,
            src: Box::new(str_lit(nfield(n, "source")?)?),
            type_only: false,
            with: None,
            phase: Default::default(),
        }),
        "ExportAllDeclaration" => ast::ModuleDecl::ExportAll(ast::ExportAll {
            span: DUMMY_SP,
            src: Box::new(str_lit(nfield(n, "source")?)?),
            type_only: false,
            with: None,
        }),
        "ExportNamedDeclaration" => {
            if let Some(decl_node) = opt_node(n, "declaration")? {
                ast::ModuleDecl::ExportDecl(ast::ExportDecl { span: DUMMY_SP, decl: decl(decl_node)? })
            } else {
                ast::ModuleDecl::ExportNamed(ast::NamedExport {
                    span: DUMMY_SP,
                    specifiers: list(n, "specifiers")?.iter().map(|s| export_specifier(node_of(s)?)).collect::<Conv<_>>()?,
                    src: opt_node(n, "source")?.map(|s| str_lit(s).map(Box::new)).transpose()?,
                    type_only: false,
                    with: None,
                })
            }
        }
        "ExportDefaultDeclaration" => {
            let d = nfield(n, "declaration")?;
            match d.ty.as_str() {
                "FunctionDeclaration" => ast::ModuleDecl::ExportDefaultDecl(ast::ExportDefaultDecl {
                    span: DUMMY_SP,
                    decl: ast::DefaultDecl::Fn(ast::FnExpr {
                        ident: opt_node(d, "id")?.map(|i| Ok::<_, String>(ident(str_field(i, "name")?))).transpose()?,
                        function: function_obj(d)?,
                    }),
                }),
                "ClassDeclaration" => ast::ModuleDecl::ExportDefaultDecl(ast::ExportDefaultDecl {
                    span: DUMMY_SP,
                    decl: ast::DefaultDecl::Class(ast::ClassExpr {
                        ident: opt_node(d, "id")?.map(|i| Ok::<_, String>(ident(str_field(i, "name")?))).transpose()?,
                        class: class_obj(d)?,
                    }),
                }),
                _ => ast::ModuleDecl::ExportDefaultExpr(ast::ExportDefaultExpr {
                    span: DUMMY_SP,
                    expr: Box::new(expr(d)?),
                }),
            }
        }
        other => return Err(format!("module decl: {other}")),
    })
}

fn import_specifier(n: &Node) -> Conv<ast::ImportSpecifier> {
    Ok(match n.ty.as_str() {
        "ImportDefaultSpecifier" => ast::ImportSpecifier::Default(ast::ImportDefaultSpecifier {
            span: DUMMY_SP,
            local: ident(str_field(nfield(n, "local")?, "name")?),
        }),
        "ImportNamespaceSpecifier" => ast::ImportSpecifier::Namespace(ast::ImportStarAsSpecifier {
            span: DUMMY_SP,
            local: ident(str_field(nfield(n, "local")?, "name")?),
        }),
        "ImportSpecifier" => ast::ImportSpecifier::Named(ast::ImportNamedSpecifier {
            span: DUMMY_SP,
            local: ident(str_field(nfield(n, "local")?, "name")?),
            imported: Some(module_export_name(nfield(n, "imported")?)?),
            is_type_only: false,
        }),
        other => return Err(format!("import specifier: {other}")),
    })
}

fn export_specifier(n: &Node) -> Conv<ast::ExportSpecifier> {
    Ok(ast::ExportSpecifier::Named(ast::ExportNamedSpecifier {
        span: DUMMY_SP,
        orig: module_export_name(nfield(n, "local")?)?,
        exported: Some(module_export_name(nfield(n, "exported")?)?),
        is_type_only: false,
    }))
}

fn module_export_name(n: &Node) -> Conv<ast::ModuleExportName> {
    Ok(match n.ty.as_str() {
        "Identifier" => ast::ModuleExportName::Ident(ident(str_field(n, "name")?)),
        "StringLiteral" => ast::ModuleExportName::Str(str_lit(n)?),
        other => return Err(format!("module export name: {other}")),
    })
}

fn str_lit(n: &Node) -> Conv<ast::Str> {
    Ok(ast::Str { span: DUMMY_SP, value: str_field(n, "value")?.into(), raw: raw_of(n).map(Into::into) })
}

/// Convert a declaration statement node into an swc `Decl`.
fn decl(n: &Node) -> Conv<ast::Decl> {
    Ok(match n.ty.as_str() {
        "VariableDeclaration" => ast::Decl::Var(Box::new(var_decl(n)?)),
        "FunctionDeclaration" => ast::Decl::Fn(ast::FnDecl {
            ident: ident(str_field(nfield(n, "id")?, "name")?),
            declare: false,
            function: function_obj(n)?,
        }),
        "ClassDeclaration" => ast::Decl::Class(ast::ClassDecl {
            ident: ident(str_field(nfield(n, "id")?, "name")?),
            declare: false,
            class: class_obj(n)?,
        }),
        other => return Err(format!("decl: {other}")),
    })
}

fn stmt(n: &Node) -> Conv<ast::Stmt> {
    let mut s = stmt_inner(n)?;
    // Give the statement its original span so the comment store (keyed by
    // BytePos) can attach comments to it during codegen.
    if let Some(sp) = span_of(n) {
        set_stmt_span(&mut s, sp);
    }
    Ok(s)
}

fn stmt_inner(n: &Node) -> Conv<ast::Stmt> {
    Ok(match n.ty.as_str() {
        "ExpressionStatement" => ast::Stmt::Expr(ast::ExprStmt {
            span: DUMMY_SP,
            expr: Box::new(expr(nfield(n, "expression")?)?),
        }),
        "EmptyStatement" => ast::Stmt::Empty(ast::EmptyStmt { span: DUMMY_SP }),
        "DebuggerStatement" => ast::Stmt::Debugger(ast::DebuggerStmt { span: DUMMY_SP }),
        "BlockStatement" => ast::Stmt::Block(block_stmt(n)?),
        "ReturnStatement" => ast::Stmt::Return(ast::ReturnStmt {
            span: DUMMY_SP,
            arg: opt_node(n, "argument")?.map(|e| expr(e).map(Box::new)).transpose()?,
        }),
        "ThrowStatement" => ast::Stmt::Throw(ast::ThrowStmt {
            span: DUMMY_SP,
            arg: Box::new(expr(nfield(n, "argument")?)?),
        }),
        "IfStatement" => ast::Stmt::If(ast::IfStmt {
            span: DUMMY_SP,
            test: Box::new(expr(nfield(n, "test")?)?),
            cons: Box::new(stmt(nfield(n, "consequent")?)?),
            alt: opt_node(n, "alternate")?.map(|s| stmt(s).map(Box::new)).transpose()?,
        }),
        "WhileStatement" => ast::Stmt::While(ast::WhileStmt {
            span: DUMMY_SP,
            test: Box::new(expr(nfield(n, "test")?)?),
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "DoWhileStatement" => ast::Stmt::DoWhile(ast::DoWhileStmt {
            span: DUMMY_SP,
            test: Box::new(expr(nfield(n, "test")?)?),
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "VariableDeclaration" => ast::Stmt::Decl(ast::Decl::Var(Box::new(var_decl(n)?))),
        "FunctionDeclaration" => ast::Stmt::Decl(ast::Decl::Fn(ast::FnDecl {
            ident: ident(str_field(nfield(n, "id")?, "name")?),
            declare: false,
            function: function_obj(n)?,
        })),
        "ClassDeclaration" => ast::Stmt::Decl(ast::Decl::Class(ast::ClassDecl {
            ident: ident(str_field(nfield(n, "id")?, "name")?),
            declare: false,
            class: class_obj(n)?,
        })),
        "WithStatement" => ast::Stmt::With(ast::WithStmt {
            span: DUMMY_SP,
            obj: Box::new(expr(nfield(n, "object")?)?),
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "LabeledStatement" => ast::Stmt::Labeled(ast::LabeledStmt {
            span: DUMMY_SP,
            label: ident(str_field(nfield(n, "label")?, "name")?),
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "BreakStatement" => ast::Stmt::Break(ast::BreakStmt {
            span: DUMMY_SP,
            label: opt_node(n, "label")?.map(|l| Ok::<_, String>(ident(str_field(l, "name")?))).transpose()?,
        }),
        "ContinueStatement" => ast::Stmt::Continue(ast::ContinueStmt {
            span: DUMMY_SP,
            label: opt_node(n, "label")?.map(|l| Ok::<_, String>(ident(str_field(l, "name")?))).transpose()?,
        }),
        "ForStatement" => ast::Stmt::For(ast::ForStmt {
            span: DUMMY_SP,
            init: opt_node(n, "init")?.map(for_init).transpose()?,
            test: opt_node(n, "test")?.map(|e| expr(e).map(Box::new)).transpose()?,
            update: opt_node(n, "update")?.map(|e| expr(e).map(Box::new)).transpose()?,
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "ForInStatement" => ast::Stmt::ForIn(ast::ForInStmt {
            span: DUMMY_SP,
            left: for_head(nfield(n, "left")?)?,
            right: Box::new(expr(nfield(n, "right")?)?),
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "ForOfStatement" => ast::Stmt::ForOf(ast::ForOfStmt {
            span: DUMMY_SP,
            is_await: bool_field(n, "await").unwrap_or(false),
            left: for_head(nfield(n, "left")?)?,
            right: Box::new(expr(nfield(n, "right")?)?),
            body: Box::new(stmt(nfield(n, "body")?)?),
        }),
        "TryStatement" => ast::Stmt::Try(Box::new(ast::TryStmt {
            span: DUMMY_SP,
            block: block_stmt(nfield(n, "block")?)?,
            handler: opt_node(n, "handler")?.map(catch_clause).transpose()?,
            finalizer: opt_node(n, "finalizer")?.map(block_stmt).transpose()?,
        })),
        "SwitchStatement" => ast::Stmt::Switch(ast::SwitchStmt {
            span: DUMMY_SP,
            discriminant: Box::new(expr(nfield(n, "discriminant")?)?),
            cases: list(n, "cases")?.iter().map(|c| switch_case(node_of(c)?)).collect::<Conv<_>>()?,
        }),
        other => return Err(format!("to_swc stmt: unsupported {other}")),
    })
}

fn expr(n: &Node) -> Conv<ast::Expr> {
    Ok(match n.ty.as_str() {
        "NumericLiteral" => ast::Expr::Lit(ast::Lit::Num(ast::Number {
            span: DUMMY_SP,
            value: f64_field(n, "value")?,
            raw: raw_of(n).map(Into::into),
        })),
        "StringLiteral" => ast::Expr::Lit(ast::Lit::Str(ast::Str {
            span: DUMMY_SP,
            value: str_field(n, "value")?.into(),
            raw: raw_of(n).map(Into::into),
        })),
        "BooleanLiteral" => ast::Expr::Lit(ast::Lit::Bool(ast::Bool {
            span: DUMMY_SP,
            value: bool_field(n, "value")?,
        })),
        "NullLiteral" => ast::Expr::Lit(ast::Lit::Null(ast::Null { span: DUMMY_SP })),
        "Identifier" => ast::Expr::Ident(ident(str_field(n, "name")?)),
        "BinaryExpression" => ast::Expr::Bin(ast::BinExpr {
            span: DUMMY_SP,
            op: bin_op(str_field(n, "operator")?)?,
            left: Box::new(expr(nfield(n, "left")?)?),
            right: Box::new(expr(nfield(n, "right")?)?),
        }),
        "ParenthesizedExpression" => ast::Expr::Paren(ast::ParenExpr {
            span: DUMMY_SP,
            expr: Box::new(expr(nfield(n, "expression")?)?),
        }),
        "LogicalExpression" => ast::Expr::Bin(ast::BinExpr {
            span: DUMMY_SP,
            op: bin_op(str_field(n, "operator")?)?,
            left: Box::new(expr(nfield(n, "left")?)?),
            right: Box::new(expr(nfield(n, "right")?)?),
        }),
        "UnaryExpression" => ast::Expr::Unary(ast::UnaryExpr {
            span: DUMMY_SP,
            op: unary_op(str_field(n, "operator")?)?,
            arg: Box::new(expr(nfield(n, "argument")?)?),
        }),
        "UpdateExpression" => ast::Expr::Update(ast::UpdateExpr {
            span: DUMMY_SP,
            op: if str_field(n, "operator")? == "++" {
                ast::UpdateOp::PlusPlus
            } else {
                ast::UpdateOp::MinusMinus
            },
            prefix: bool_field(n, "prefix")?,
            arg: Box::new(expr(nfield(n, "argument")?)?),
        }),
        "ConditionalExpression" => ast::Expr::Cond(ast::CondExpr {
            span: DUMMY_SP,
            test: Box::new(expr(nfield(n, "test")?)?),
            cons: Box::new(expr(nfield(n, "consequent")?)?),
            alt: Box::new(expr(nfield(n, "alternate")?)?),
        }),
        "SequenceExpression" => ast::Expr::Seq(ast::SeqExpr {
            span: DUMMY_SP,
            exprs: list(n, "expressions")?
                .iter()
                .map(|e| expr(node_of(e)?).map(Box::new))
                .collect::<Conv<_>>()?,
        }),
        "ThisExpression" => ast::Expr::This(ast::ThisExpr { span: DUMMY_SP }),
        "MemberExpression" if is_super_member(n) => ast::Expr::SuperProp(super_prop_expr(n)?),
        "MemberExpression" => ast::Expr::Member(member_expr(n)?),
        "CallExpression" => {
            let callee_node = nfield(n, "callee")?;
            // `import(x)` and `super(...)` are non-expression callees.
            let callee = match callee_node.ty.as_str() {
                "Import" => ast::Callee::Import(ast::Import {
                    span: DUMMY_SP,
                    phase: Default::default(),
                }),
                "Super" => ast::Callee::Super(ast::Super { span: DUMMY_SP }),
                _ => ast::Callee::Expr(Box::new(expr(callee_node)?)),
            };
            ast::Expr::Call(ast::CallExpr {
                span: DUMMY_SP,
                ctxt: Default::default(),
                callee,
                args: args(n)?,
                type_args: None,
            })
        }
        "NewExpression" => ast::Expr::New(ast::NewExpr {
            span: DUMMY_SP,
            ctxt: Default::default(),
            callee: Box::new(expr(nfield(n, "callee")?)?),
            args: Some(args(n)?),
            type_args: None,
        }),
        "ArrayExpression" => ast::Expr::Array(ast::ArrayLit {
            span: DUMMY_SP,
            elems: list(n, "elements")?
                .iter()
                .map(|el| match el {
                    FieldValue::Null => Ok(None),
                    FieldValue::Node(node) => expr_or_spread(node).map(Some),
                    o => Err(format!("array element: {o:?}")),
                })
                .collect::<Conv<_>>()?,
        }),
        "AssignmentExpression" => ast::Expr::Assign(ast::AssignExpr {
            span: DUMMY_SP,
            op: assign_op(str_field(n, "operator")?)?,
            left: assign_target(nfield(n, "left")?)?,
            right: Box::new(expr(nfield(n, "right")?)?),
        }),
        "BigIntLiteral" => ast::Expr::Lit(ast::Lit::BigInt(ast::BigInt {
            span: DUMMY_SP,
            value: Box::new(
                str_field(n, "value")?
                    .parse::<num_bigint::BigInt>()
                    .map_err(|e| format!("bigint parse: {e}"))?,
            ),
            raw: None,
        })),
        "RegExpLiteral" => ast::Expr::Lit(ast::Lit::Regex(ast::Regex {
            span: DUMMY_SP,
            exp: str_field(n, "pattern")?.into(),
            flags: str_field(n, "flags")?.into(),
        })),
        "FunctionExpression" => ast::Expr::Fn(ast::FnExpr {
            ident: opt_node(n, "id")?.map(|i| Ok::<_, String>(ident(str_field(i, "name")?))).transpose()?,
            function: function_obj(n)?,
        }),
        "ClassExpression" => ast::Expr::Class(ast::ClassExpr {
            ident: opt_node(n, "id")?.map(|i| Ok::<_, String>(ident(str_field(i, "name")?))).transpose()?,
            class: class_obj(n)?,
        }),
        "ArrowFunctionExpression" => {
            let params = list(n, "params")?.iter().map(|p| pat(node_of(p)?)).collect::<Conv<_>>()?;
            let body_node = nfield(n, "body")?;
            let body = if body_node.ty == "BlockStatement" {
                ast::BlockStmtOrExpr::BlockStmt(block_stmt(body_node)?)
            } else {
                ast::BlockStmtOrExpr::Expr(Box::new(expr(body_node)?))
            };
            ast::Expr::Arrow(ast::ArrowExpr {
                span: DUMMY_SP,
                ctxt: Default::default(),
                params,
                body: Box::new(body),
                is_async: bool_field(n, "async")?,
                is_generator: bool_field(n, "generator")?,
                type_params: None,
                return_type: None,
            })
        }
        "ObjectExpression" => ast::Expr::Object(ast::ObjectLit {
            span: DUMMY_SP,
            props: list(n, "properties")?.iter().map(|p| prop_or_spread(node_of(p)?)).collect::<Conv<_>>()?,
        }),
        "TemplateLiteral" => ast::Expr::Tpl(tpl(n)?),
        "TaggedTemplateExpression" => ast::Expr::TaggedTpl(ast::TaggedTpl {
            span: DUMMY_SP,
            ctxt: Default::default(),
            tag: Box::new(expr(nfield(n, "tag")?)?),
            type_params: None,
            tpl: Box::new(tpl(nfield(n, "quasi")?)?),
        }),
        "YieldExpression" => ast::Expr::Yield(ast::YieldExpr {
            span: DUMMY_SP,
            arg: opt_node(n, "argument")?.map(|e| expr(e).map(Box::new)).transpose()?,
            delegate: bool_field(n, "delegate")?,
        }),
        "AwaitExpression" => ast::Expr::Await(ast::AwaitExpr {
            span: DUMMY_SP,
            arg: Box::new(expr(nfield(n, "argument")?)?),
        }),
        "MetaProperty" => {
            let meta = str_field(nfield(n, "meta")?, "name")?;
            let prop = str_field(nfield(n, "property")?, "name")?;
            let kind = match (meta, prop) {
                ("new", "target") => ast::MetaPropKind::NewTarget,
                ("import", "meta") => ast::MetaPropKind::ImportMeta,
                _ => return Err(format!("meta property {meta}.{prop}")),
            };
            ast::Expr::MetaProp(ast::MetaPropExpr { span: DUMMY_SP, kind })
        }
        "OptionalMemberExpression" => ast::Expr::OptChain(ast::OptChainExpr {
            span: DUMMY_SP,
            optional: bool_field(n, "optional")?,
            base: Box::new(ast::OptChainBase::Member(member_expr(n)?)),
        }),
        other => return Err(format!("to_swc expr: unsupported {other}")),
    })
}

fn ident(name: &str) -> ast::Ident {
    ast::Ident::new_no_ctxt(name.into(), DUMMY_SP)
}

fn binding_ident(name: &str) -> ast::BindingIdent {
    ast::BindingIdent { id: ident(name), type_ann: None }
}

fn block_stmt(n: &Node) -> Conv<ast::BlockStmt> {
    Ok(ast::BlockStmt {
        span: DUMMY_SP,
        ctxt: Default::default(),
        stmts: list(n, "body")?
            .iter()
            .map(|s| stmt(node_of(s)?))
            .collect::<Conv<_>>()?,
    })
}

fn var_decl(n: &Node) -> Conv<ast::VarDecl> {
    let kind = match str_field(n, "kind")? {
        "var" => ast::VarDeclKind::Var,
        "let" => ast::VarDeclKind::Let,
        "const" => ast::VarDeclKind::Const,
        k => return Err(format!("var kind {k}")),
    };
    let decls = list(n, "declarations")?
        .iter()
        .map(|d| {
            let d = node_of(d)?;
            Ok(ast::VarDeclarator {
                span: DUMMY_SP,
                name: pat(nfield(d, "id")?)?,
                init: opt_node(d, "init")?.map(|e| expr(e).map(Box::new)).transpose()?,
                definite: false,
            })
        })
        .collect::<Conv<_>>()?;
    Ok(ast::VarDecl { span: DUMMY_SP, ctxt: Default::default(), kind, declare: false, decls })
}

/// Convert a binding pattern.
fn pat(n: &Node) -> Conv<ast::Pat> {
    Ok(match n.ty.as_str() {
        "Identifier" => ast::Pat::Ident(binding_ident(str_field(n, "name")?)),
        "MemberExpression" if is_super_member(n) => {
            ast::Pat::Expr(Box::new(ast::Expr::SuperProp(super_prop_expr(n)?)))
        }
        "MemberExpression" => ast::Pat::Expr(Box::new(ast::Expr::Member(member_expr(n)?))),
        "ParenthesizedExpression" => pat(nfield(n, "expression")?)?,
        "AssignmentPattern" => ast::Pat::Assign(ast::AssignPat {
            span: DUMMY_SP,
            left: Box::new(pat(nfield(n, "left")?)?),
            right: Box::new(expr(nfield(n, "right")?)?),
        }),
        "RestElement" => ast::Pat::Rest(ast::RestPat {
            span: DUMMY_SP,
            dot3_token: DUMMY_SP,
            arg: Box::new(pat(nfield(n, "argument")?)?),
            type_ann: None,
        }),
        "ArrayPattern" => ast::Pat::Array(array_pat(n)?),
        "ObjectPattern" => ast::Pat::Object(object_pat(n)?),
        other => return Err(format!("to_swc pat: unsupported {other}")),
    })
}

fn array_pat(n: &Node) -> Conv<ast::ArrayPat> {
    Ok(ast::ArrayPat {
        span: DUMMY_SP,
        elems: list(n, "elements")?
            .iter()
            .map(|e| match e {
                FieldValue::Null => Ok(None),
                FieldValue::Node(node) => pat(node).map(Some),
                o => Err(format!("array pat elem: {o:?}")),
            })
            .collect::<Conv<_>>()?,
        optional: false,
        type_ann: None,
    })
}

fn object_pat(n: &Node) -> Conv<ast::ObjectPat> {
    Ok(ast::ObjectPat {
        span: DUMMY_SP,
        props: list(n, "properties")?
            .iter()
            .map(|p| obj_pat_prop(node_of(p)?))
            .collect::<Conv<_>>()?,
        optional: false,
        type_ann: None,
    })
}

fn obj_pat_prop(n: &Node) -> Conv<ast::ObjectPatProp> {
    if n.ty == "RestElement" {
        return Ok(ast::ObjectPatProp::Rest(ast::RestPat {
            span: DUMMY_SP,
            dot3_token: DUMMY_SP,
            arg: Box::new(pat(nfield(n, "argument")?)?),
            type_ann: None,
        }));
    }
    // ObjectProperty in a pattern.
    let value = nfield(n, "value")?;
    if bool_field(n, "shorthand")? {
        // `{a}` or `{a = default}`.
        let (key_name, default) = if value.ty == "AssignmentPattern" {
            (str_field(nfield(value, "left")?, "name")?, Some(Box::new(expr(nfield(value, "right")?)?)))
        } else {
            (str_field(value, "name")?, None)
        };
        Ok(ast::ObjectPatProp::Assign(ast::AssignPatProp {
            span: DUMMY_SP,
            key: binding_ident(key_name),
            value: default,
        }))
    } else {
        Ok(ast::ObjectPatProp::KeyValue(ast::KeyValuePatProp {
            key: prop_name(n)?,
            value: Box::new(pat(value)?),
        }))
    }
}

/// Build an swc `Function` from a node with params/body/generator/async.
fn function_obj(n: &Node) -> Conv<Box<ast::Function>> {
    let params = list(n, "params")?
        .iter()
        .map(|p| {
            Ok(ast::Param { span: DUMMY_SP, decorators: vec![], pat: pat(node_of(p)?)? })
        })
        .collect::<Conv<_>>()?;
    Ok(Box::new(ast::Function {
        params,
        decorators: vec![],
        span: DUMMY_SP,
        ctxt: Default::default(),
        body: Some(block_stmt(nfield(n, "body")?)?),
        is_generator: bool_field(n, "generator")?,
        is_async: bool_field(n, "async")?,
        type_params: None,
        return_type: None,
    }))
}

/// Convert an object/class key to an swc `PropName`.
fn prop_name(n: &Node) -> Conv<ast::PropName> {
    if bool_field(n, "computed").unwrap_or(false) {
        return Ok(ast::PropName::Computed(ast::ComputedPropName {
            span: DUMMY_SP,
            expr: Box::new(expr(nfield(n, "key")?)?),
        }));
    }
    let key = nfield(n, "key")?;
    Ok(match key.ty.as_str() {
        "Identifier" => ast::PropName::Ident(ast::IdentName::new(str_field(key, "name")?.into(), DUMMY_SP)),
        "StringLiteral" => ast::PropName::Str(ast::Str { span: DUMMY_SP, value: str_field(key, "value")?.into(), raw: raw_of(key).map(Into::into) }),
        "NumericLiteral" => ast::PropName::Num(ast::Number { span: DUMMY_SP, value: f64_field(key, "value")?, raw: raw_of(key).map(Into::into) }),
        other => return Err(format!("prop name: {other}")),
    })
}

fn member_expr(n: &Node) -> Conv<ast::MemberExpr> {
    let prop_node = nfield(n, "property")?;
    let prop = if bool_field(n, "computed")? {
        ast::MemberProp::Computed(ast::ComputedPropName {
            span: DUMMY_SP,
            expr: Box::new(expr(prop_node)?),
        })
    } else if prop_node.ty == "PrivateName" {
        // `obj.#priv`: the name lives on the inner `id` Identifier, sans `#`.
        let id = nfield(prop_node, "id")?;
        ast::MemberProp::PrivateName(ast::PrivateName {
            span: DUMMY_SP,
            name: str_field(id, "name")?.into(),
        })
    } else {
        ast::MemberProp::Ident(ast::IdentName::new(str_field(prop_node, "name")?.into(), DUMMY_SP))
    };
    Ok(ast::MemberExpr { span: DUMMY_SP, obj: Box::new(expr(nfield(n, "object")?)?), prop })
}

/// Whether a `MemberExpression` node's object is the `super` keyword. swc
/// models `super.x` as a distinct `SuperPropExpr`, not a `MemberExpr`.
fn is_super_member(n: &Node) -> bool {
    nfield(n, "object").map(|o| o.ty == "Super").unwrap_or(false)
}

fn super_prop_expr(n: &Node) -> Conv<ast::SuperPropExpr> {
    let prop_node = nfield(n, "property")?;
    let prop = if bool_field(n, "computed")? {
        ast::SuperProp::Computed(ast::ComputedPropName {
            span: DUMMY_SP,
            expr: Box::new(expr(prop_node)?),
        })
    } else {
        ast::SuperProp::Ident(ast::IdentName::new(str_field(prop_node, "name")?.into(), DUMMY_SP))
    };
    Ok(ast::SuperPropExpr { span: DUMMY_SP, obj: ast::Super { span: DUMMY_SP }, prop })
}

fn args(n: &Node) -> Conv<Vec<ast::ExprOrSpread>> {
    list(n, "arguments")?.iter().map(|a| expr_or_spread(node_of(a)?)).collect()
}

fn expr_or_spread(n: &Node) -> Conv<ast::ExprOrSpread> {
    if n.ty == "SpreadElement" {
        Ok(ast::ExprOrSpread { spread: Some(DUMMY_SP), expr: Box::new(expr(nfield(n, "argument")?)?) })
    } else {
        Ok(ast::ExprOrSpread { spread: None, expr: Box::new(expr(n)?) })
    }
}

fn assign_target(n: &Node) -> Conv<ast::AssignTarget> {
    use ast::{AssignTarget, SimpleAssignTarget};
    Ok(match n.ty.as_str() {
        "Identifier" => AssignTarget::Simple(SimpleAssignTarget::Ident(binding_ident(str_field(n, "name")?))),
        "MemberExpression" if is_super_member(n) => {
            AssignTarget::Simple(SimpleAssignTarget::SuperProp(super_prop_expr(n)?))
        }
        "MemberExpression" => AssignTarget::Simple(SimpleAssignTarget::Member(member_expr(n)?)),
        "ParenthesizedExpression" => assign_target(nfield(n, "expression")?)?,
        "ObjectPattern" => AssignTarget::Pat(ast::AssignTargetPat::Object(object_pat(n)?)),
        "ArrayPattern" => AssignTarget::Pat(ast::AssignTargetPat::Array(array_pat(n)?)),
        other => return Err(format!("to_swc assign target: unsupported {other}")),
    })
}

fn unary_op(op: &str) -> Conv<ast::UnaryOp> {
    use ast::UnaryOp::*;
    Ok(match op {
        "-" => Minus, "+" => Plus, "!" => Bang, "~" => Tilde,
        "typeof" => TypeOf, "void" => Void, "delete" => Delete,
        other => return Err(format!("unary op {other}")),
    })
}

fn assign_op(op: &str) -> Conv<ast::AssignOp> {
    use ast::AssignOp::*;
    Ok(match op {
        "=" => Assign, "+=" => AddAssign, "-=" => SubAssign, "*=" => MulAssign,
        "/=" => DivAssign, "%=" => ModAssign, "**=" => ExpAssign,
        "<<=" => LShiftAssign, ">>=" => RShiftAssign, ">>>=" => ZeroFillRShiftAssign,
        "|=" => BitOrAssign, "^=" => BitXorAssign, "&=" => BitAndAssign,
        "&&=" => AndAssign, "||=" => OrAssign, "??=" => NullishAssign,
        other => return Err(format!("assign op {other}")),
    })
}

fn opt_node<'a>(n: &'a Node, k: &str) -> Conv<Option<&'a Node>> {
    match n.field(k) {
        Some(FieldValue::Node(b)) => Ok(Some(b)),
        _ => Ok(None),
    }
}

fn for_init(n: &Node) -> Conv<ast::VarDeclOrExpr> {
    Ok(if n.ty == "VariableDeclaration" {
        ast::VarDeclOrExpr::VarDecl(Box::new(var_decl(n)?))
    } else {
        ast::VarDeclOrExpr::Expr(Box::new(expr(n)?))
    })
}

fn for_head(n: &Node) -> Conv<ast::ForHead> {
    Ok(if n.ty == "VariableDeclaration" {
        ast::ForHead::VarDecl(Box::new(var_decl(n)?))
    } else {
        ast::ForHead::Pat(Box::new(pat(n)?))
    })
}

fn catch_clause(n: &Node) -> Conv<ast::CatchClause> {
    Ok(ast::CatchClause {
        span: DUMMY_SP,
        param: opt_node(n, "param")?.map(pat).transpose()?,
        body: block_stmt(nfield(n, "body")?)?,
    })
}

fn switch_case(n: &Node) -> Conv<ast::SwitchCase> {
    Ok(ast::SwitchCase {
        span: DUMMY_SP,
        test: opt_node(n, "test")?.map(|e| expr(e).map(Box::new)).transpose()?,
        cons: list(n, "consequent")?.iter().map(|s| stmt(node_of(s)?)).collect::<Conv<_>>()?,
    })
}

fn prop_or_spread(n: &Node) -> Conv<ast::PropOrSpread> {
    if n.ty == "SpreadElement" {
        return Ok(ast::PropOrSpread::Spread(ast::SpreadElement {
            dot3_token: DUMMY_SP,
            expr: Box::new(expr(nfield(n, "argument")?)?),
        }));
    }
    if n.ty == "ObjectMethod" {
        let prop = match str_field(n, "kind")? {
            "get" => ast::Prop::Getter(ast::GetterProp {
                span: DUMMY_SP,
                key: prop_name(n)?,
                type_ann: None,
                body: Some(block_stmt(nfield(n, "body")?)?),
            }),
            "set" => ast::Prop::Setter(ast::SetterProp {
                span: DUMMY_SP,
                key: prop_name(n)?,
                this_param: None,
                param: Box::new(pat(node_of(&list(n, "params")?[0])?)?),
                body: Some(block_stmt(nfield(n, "body")?)?),
            }),
            _ => ast::Prop::Method(ast::MethodProp { key: prop_name(n)?, function: function_obj(n)? }),
        };
        return Ok(ast::PropOrSpread::Prop(Box::new(prop)));
    }
    // ObjectProperty
    let value = nfield(n, "value")?;
    if bool_field(n, "shorthand")? && value.ty == "Identifier" {
        return Ok(ast::PropOrSpread::Prop(Box::new(ast::Prop::Shorthand(ident(str_field(value, "name")?)))));
    }
    Ok(ast::PropOrSpread::Prop(Box::new(ast::Prop::KeyValue(ast::KeyValueProp {
        key: prop_name(n)?,
        value: Box::new(expr(value)?),
    }))))
}

fn tpl(n: &Node) -> Conv<ast::Tpl> {
    Ok(ast::Tpl {
        span: DUMMY_SP,
        exprs: list(n, "expressions")?.iter().map(|e| expr(node_of(e)?).map(Box::new)).collect::<Conv<_>>()?,
        quasis: list(n, "quasis")?.iter().map(|q| tpl_element(node_of(q)?)).collect::<Conv<_>>()?,
    })
}

fn tpl_element(n: &Node) -> Conv<ast::TplElement> {
    let value = match n.field("value") {
        Some(FieldValue::Extra(e)) => e,
        _ => return Err("template element missing value".into()),
    };
    let raw = match value.field("raw") {
        Some(FieldValue::Str(s)) => s.clone(),
        _ => return Err("template element missing raw".into()),
    };
    let cooked = match value.field("cooked") {
        Some(FieldValue::Str(s)) => Some(s.clone().into()),
        _ => None,
    };
    Ok(ast::TplElement { span: DUMMY_SP, tail: bool_field(n, "tail")?, cooked, raw: raw.into() })
}

fn private_name(n: &Node) -> Conv<ast::PrivateName> {
    // PrivateName { id: Identifier }
    Ok(ast::PrivateName { span: DUMMY_SP, name: str_field(nfield(n, "id")?, "name")?.into() })
}

fn class_obj(n: &Node) -> Conv<Box<ast::Class>> {
    let body_node = nfield(n, "body")?; // ClassBody
    let body = list(body_node, "body")?
        .iter()
        .map(|m| class_member(node_of(m)?))
        .collect::<Conv<_>>()?;
    Ok(Box::new(ast::Class {
        span: DUMMY_SP,
        ctxt: Default::default(),
        decorators: vec![],
        body,
        super_class: opt_node(n, "superClass")?.map(|e| expr(e).map(Box::new)).transpose()?,
        is_abstract: false,
        type_params: None,
        super_type_params: None,
        implements: vec![],
    }))
}

fn class_member(m: &Node) -> Conv<ast::ClassMember> {
    let is_static = bool_field(m, "static").unwrap_or(false);
    Ok(match m.ty.as_str() {
        "ClassMethod" => ast::ClassMember::Method(ast::ClassMethod {
            span: DUMMY_SP,
            key: prop_name(m)?,
            function: function_obj(m)?,
            kind: method_kind(str_field(m, "kind")?),
            is_static,
            accessibility: None,
            is_abstract: false,
            is_optional: false,
            is_override: false,
        }),
        "ClassPrivateMethod" => ast::ClassMember::PrivateMethod(ast::PrivateMethod {
            span: DUMMY_SP,
            key: private_name(nfield(m, "key")?)?,
            function: function_obj(m)?,
            kind: method_kind(str_field(m, "kind")?),
            is_static,
            accessibility: None,
            is_abstract: false,
            is_optional: false,
            is_override: false,
        }),
        "ClassProperty" => ast::ClassMember::ClassProp(ast::ClassProp {
            span: DUMMY_SP,
            key: prop_name(m)?,
            value: opt_node(m, "value")?.map(|e| expr(e).map(Box::new)).transpose()?,
            type_ann: None,
            is_static,
            decorators: vec![],
            accessibility: None,
            is_abstract: false,
            is_optional: false,
            is_override: false,
            readonly: false,
            declare: false,
            definite: false,
        }),
        "ClassPrivateProperty" => ast::ClassMember::PrivateProp(ast::PrivateProp {
            span: DUMMY_SP,
            ctxt: Default::default(),
            key: private_name(nfield(m, "key")?)?,
            value: opt_node(m, "value")?.map(|e| expr(e).map(Box::new)).transpose()?,
            type_ann: None,
            is_static,
            decorators: vec![],
            accessibility: None,
            is_optional: false,
            is_override: false,
            readonly: false,
            definite: false,
        }),
        other => return Err(format!("class member: {other}")),
    })
}

fn method_kind(kind: &str) -> ast::MethodKind {
    match kind {
        "get" => ast::MethodKind::Getter,
        "set" => ast::MethodKind::Setter,
        _ => ast::MethodKind::Method,
    }
}

fn bin_op(op: &str) -> Conv<ast::BinaryOp> {
    use ast::BinaryOp::*;
    Ok(match op {
        "==" => EqEq, "!=" => NotEq, "===" => EqEqEq, "!==" => NotEqEq,
        "<" => Lt, "<=" => LtEq, ">" => Gt, ">=" => GtEq,
        "<<" => LShift, ">>" => RShift, ">>>" => ZeroFillRShift,
        "+" => Add, "-" => Sub, "*" => Mul, "/" => Div, "%" => Mod, "**" => Exp,
        "|" => BitOr, "^" => BitXor, "&" => BitAnd,
        "in" => In, "instanceof" => InstanceOf,
        "&&" => LogicalAnd, "||" => LogicalOr, "??" => NullishCoalescing,
        other => return Err(format!("bin op {other}")),
    })
}

// -- accessors over jsir_ast::Node --
fn nfield<'a>(n: &'a Node, k: &str) -> Conv<&'a Node> {
    node_of(n.field(k).ok_or_else(|| format!("{}: missing {k}", n.ty))?)
}
fn node_of(fv: &FieldValue) -> Conv<&Node> {
    match fv { FieldValue::Node(b) => Ok(b), o => Err(format!("expected node, got {o:?}")) }
}
fn list<'a>(n: &'a Node, k: &str) -> Conv<&'a [FieldValue]> {
    match n.field(k) { Some(FieldValue::List(v)) => Ok(v), _ => Err(format!("{}: missing list {k}", n.ty)) }
}
fn str_field<'a>(n: &'a Node, k: &str) -> Conv<&'a str> {
    match n.field(k) { Some(FieldValue::Str(s)) => Ok(s), _ => Err(format!("{}: missing str {k}", n.ty)) }
}
fn f64_field(n: &Node, k: &str) -> Conv<f64> {
    match n.field(k) { Some(FieldValue::Float(f)) => Ok(*f), Some(FieldValue::Int(i)) => Ok(*i as f64), _ => Err(format!("{}: missing num {k}", n.ty)) }
}
fn bool_field(n: &Node, k: &str) -> Conv<bool> {
    match n.field(k) { Some(FieldValue::Bool(b)) => Ok(*b), _ => Err(format!("{}: missing bool {k}", n.ty)) }
}
/// The verbatim source token from a literal's `extra.raw` (preserves quote
/// style, numeric base, etc.); swc's codegen emits it verbatim when set.
fn raw_of(n: &Node) -> Option<String> {
    match n.field("extra") {
        Some(FieldValue::Extra(e)) => match e.field("raw") {
            Some(FieldValue::Str(s)) => Some(s.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn int_field(n: &Node, k: &str) -> Option<i64> {
    match n.field(k) {
        Some(FieldValue::Int(i)) => Some(*i),
        Some(FieldValue::Float(f)) => Some(*f as i64),
        _ => None,
    }
}

// -- comment re-emission --
//
// JSIR stores comments only as a flat `File.comments` array (per-node
// attachment is dropped in `ast2hir`). Like Babel's generator, we re-associate
// each comment to a node by source position: a comment becomes the *leading*
// comment of the next statement after it (or trailing of the last statement at
// EOF). swc's code generator emits comments from a `BytePos`-keyed store, so we
// give statements their original spans (`stmt`/`span_of`) and register each
// comment at the matching position. BytePos 0 is swc's "dummy", so all offsets
// are shifted by 1.

/// A statement's original span (shifted), or `None` if it has no position.
fn span_of(n: &Node) -> Option<Span> {
    let lo = int_field(n, "start")?;
    let hi = int_field(n, "end")?;
    Some(Span::new(BytePos(lo as u32 + 1), BytePos(hi as u32 + 1)))
}

/// True for the node types that `stmt` is called on (so their spans are set).
fn is_stmt_ty(ty: &str) -> bool {
    ty.ends_with("Statement") || ty.ends_with("Declaration")
}

/// Collect statement starts (the registerable comment-attachment positions) in
/// ascending order, by walking the lifted tree.
fn collect_stmt_starts(n: &Node, out: &mut Vec<i64>) {
    if is_stmt_ty(&n.ty) {
        if let Some(s) = int_field(n, "start") {
            out.push(s);
        }
    }
    for f in &n.fields {
        walk_field_stmt_starts(f, out);
    }
}
fn walk_field_stmt_starts(f: &FieldValue, out: &mut Vec<i64>) {
    match f {
        FieldValue::Node(b) => collect_stmt_starts(b, out),
        FieldValue::List(v) => {
            for e in v {
                walk_field_stmt_starts(e, out);
            }
        }
        _ => {}
    }
}

/// Build an swc comment store from the `File.comments` array, attaching each
/// comment to the next statement (by position) so codegen re-emits it.
pub fn build_comments(file: &Node) -> SingleThreadedComments {
    let comments = SingleThreadedComments::default();
    let Some(FieldValue::List(items)) = file.field("comments") else {
        return comments;
    };

    let program = match nfield(file, "program") {
        Ok(p) => p,
        Err(_) => return comments,
    };
    let mut starts: Vec<i64> = Vec::new();
    collect_stmt_starts(program, &mut starts);
    starts.sort_unstable();
    let last_end = starts.last().copied();

    for item in items {
        let Ok(c) = node_of(item) else { continue };
        let Some(start) = int_field(c, "start") else { continue };
        let text = c.field("value").and_then(|f| match f {
            FieldValue::Str(s) => Some(s.clone()),
            _ => None,
        });
        let Some(text) = text else { continue };
        let kind = if c.ty == "CommentBlock" {
            CommentKind::Block
        } else {
            CommentKind::Line
        };
        let comment = Comment { kind, span: DUMMY_SP, text: text.into() };
        // Attach as leading to the first statement starting after this comment.
        match starts.iter().find(|&&s| s > start) {
            Some(&node_start) => {
                comments.add_leading(BytePos(node_start as u32 + 1), comment);
            }
            None => {
                // After the last statement: trailing of the program's last node.
                if let Some(end) = last_end {
                    comments.add_trailing(BytePos(end as u32 + 1), comment);
                }
            }
        }
    }
    comments
}

/// Set a statement's span (used so the comment store can target it).
fn set_stmt_span(s: &mut ast::Stmt, span: Span) {
    use ast::Stmt::*;
    match s {
        Block(x) => x.span = span,
        Empty(x) => x.span = span,
        Debugger(x) => x.span = span,
        With(x) => x.span = span,
        Return(x) => x.span = span,
        Labeled(x) => x.span = span,
        Break(x) => x.span = span,
        Continue(x) => x.span = span,
        If(x) => x.span = span,
        Switch(x) => x.span = span,
        Throw(x) => x.span = span,
        Try(x) => x.span = span,
        While(x) => x.span = span,
        DoWhile(x) => x.span = span,
        For(x) => x.span = span,
        ForIn(x) => x.span = span,
        ForOf(x) => x.span = span,
        Expr(x) => x.span = span,
        Decl(d) => set_decl_span(d, span),
    }
}

fn set_decl_span(d: &mut ast::Decl, span: Span) {
    use ast::Decl::*;
    match d {
        Class(x) => x.class.span = span,
        Fn(x) => x.function.span = span,
        Var(x) => x.span = span,
        Using(x) => x.span = span,
        _ => {}
    }
}
