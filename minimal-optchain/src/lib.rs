//! Minimal, self-contained slice: one optional-chaining (`?.`) desugar written
//! generically over oxc **and** swc, plus the hand-written direct version for
//! each backend. Depends only on oxc/swc, not on the parent `oxs` crate.
//!
//! The transform lowers optional-member chains to a null-safe helper call:
//!
//! ```text
//!   a?.b       =>  __opt(a, "b")
//!   a?.b.c     =>  __opt(__opt(a, "b"), "c")   // trailing `.c` still short-circuits
//!   a.b?.c     =>  __opt(a.b, "c")             // plain `a.b` moved in, evaluated once
//!   a?.[k]     =>  __opt(a, k)
//! ```
//!
//! swc and oxc model `?.` in *opposite* ways — swc wraps each link in
//! `Expr::OptChain { optional, base }`; oxc marks the chain root with a
//! `Expression::ChainExpression` and hangs an `optional: bool` flag on each
//! member/call. The generic transform never sees that: a single *general*
//! mutable projection, [`Backend::expr_ref_mut`], reconciles both in one place.
//!
//! Module map:
//!   * this file — the [`Backend`] trait, the [`ExprRefMut`] view, and **both**
//!     backend impls ([`Oxc`], [`Swc`]).
//!   * [`generic`] — the one generic transform.
//!   * [`oxc`] / [`swc`] — the hand-written direct transform for each backend.

#![allow(deprecated)] // oxc's AstBuilder is mid-migration (oxc#23043)

pub mod generic;
pub mod import_map;
pub mod oxc;
pub mod swc;

pub use oxc_backend::Oxc;
pub use swc_backend::{Swc, SwcProgram};

// ===========================================================================
// Binding-identity layer: the second thing an abstraction over oxc and swc has
// to reconcile, and the one that unblocks real turbopack transforms.
//
// "Which binding does this identifier refer to?" is modeled in *opposite* ways:
//   * swc bakes hygiene into the AST — every `Ident` carries a `SyntaxContext`
//     stamped by a `resolver` pass, so `Id = (Atom, SyntaxContext)` names a
//     binding and a free/global reference is one whose ctxt is `unresolved`.
//   * oxc keeps hygiene *out* of the AST — a separate `Semantic` pass builds a
//     symbol table and fills `SymbolId`/`ReferenceId` side-cells; a free/global
//     reference is one that resolves to no `SymbolId`.
// [`Backend::BindingId`] + [`Backend::Semantics`] hide that split, so an
// analysis like [`import_map::ImportMap`] is written once against them.
// ===========================================================================

/// One imported binding surfaced by [`Backend::for_each_import`] — the shape
/// `import_analyzer`'s `ImportMap` is built from.
pub struct ImportBinding<Id> {
    /// The local binding the import introduces (`foo` in `import { bar as foo }`).
    pub local: Id,
    /// The module specifier (`"react"` in `import … from "react"`).
    pub source: String,
    /// What was imported under `local`.
    pub imported: Imported,
}

/// The three ways a local name can be bound by an `import`.
pub enum Imported {
    /// `import { orig as local }` / `import { orig }` — carries the *original*
    /// exported name (`orig`), which is what queries match against.
    Named(String),
    /// `import local from "…"` — the module's default export.
    Default,
    /// `import * as local from "…"`.
    Namespace,
}

// ===========================================================================
// The interface: a general mutable projection + construction, one trait.
// ===========================================================================

/// A **general** mutable projection of an expression — the write-path dual of a
/// read view. [`Backend::expr_ref_mut`] classifies *any* node into this; a
/// transform picks the arm it cares about (the `?.` desugar uses
/// [`OptChain`](Self::OptChain)). Children are exposed as `&mut` slots directly,
/// so disjoint field borrows just work. This is where the swc/oxc shape
/// differences (split logical ops, the `OptChain` wrapper vs `ChainExpression` +
/// `optional` flags) are reconciled — the *only* place either backend is named.
///
/// (A fuller view would also carry the binary operator; omitted here since no
/// transform in this crate needs it.)
pub enum ExprRefMut<'r, 'a, B: Backend + 'a> {
    /// `a <op> b` (`??` is surfaced separately as [`Nullish`](Self::Nullish)).
    Bin { left: &'r mut B::Expr<'a>, right: &'r mut B::Expr<'a> },
    /// `a ?? b`.
    Nullish { left: &'r mut B::Expr<'a>, right: &'r mut B::Expr<'a> },
    /// Plain member access (`a.b`, `a[k]`, `a.#f`).
    Member { object: &'r mut B::Expr<'a>, key: MemberKeyMut<'r, 'a, B> },
    /// One optional-chain link (`a?.b`, plus the non-optional links threaded
    /// through a chain). `optional` is this link's own `?.` bit.
    OptChain { optional: bool, base: OptChainBaseMut<'r, 'a, B> },
    /// `(a)`.
    Paren { inner: &'r mut B::Expr<'a> },
    /// Leaves and everything not projected here.
    Other,
}

/// The base a member / optional-member link operates on.
pub enum OptChainBaseMut<'r, 'a, B: Backend + 'a> {
    Member { object: &'r mut B::Expr<'a>, key: MemberKeyMut<'r, 'a, B> },
    /// `callee?.(...)` — reported but not decomposed.
    Call,
}

/// The property a member access reads.
pub enum MemberKeyMut<'r, 'a, B: Backend + 'a> {
    Static(&'r str),
    Computed(&'r mut B::Expr<'a>),
    Private(&'r str),
}

/// Everything a transform needs from a backend: parse/print, traverse, the
/// general mutable projection, and node construction.
pub trait Backend: Sized {
    type Arena: Default;
    type Ctx<'a>: Copy;
    type Program<'a>;
    type Expr<'a>;

    fn parse<'a>(arena: &'a Self::Arena, source: &'a str) -> Self::Program<'a>;
    fn codegen(program: &Self::Program<'_>) -> String;
    fn ctx(arena: &Self::Arena) -> Self::Ctx<'_>;

    /// Visit every expression, children before parents (post-order).
    fn visit_exprs_mut<'a>(program: &mut Self::Program<'a>, f: impl FnMut(&mut Self::Expr<'a>));

    /// The general mutable projection. Reconciles the swc/oxc shape differences
    /// once; every transform is written against this, not against a backend.
    fn expr_ref_mut<'r, 'a: 'r>(expr: &'r mut Self::Expr<'a>) -> ExprRefMut<'r, 'a, Self>;

    // Construction primitives.
    fn take_expr<'a>(cx: Self::Ctx<'a>, slot: &mut Self::Expr<'a>) -> Self::Expr<'a>;
    fn ident<'a>(cx: Self::Ctx<'a>, name: &str) -> Self::Expr<'a>;
    fn string<'a>(cx: Self::Ctx<'a>, value: &str) -> Self::Expr<'a>;
    fn call<'a>(cx: Self::Ctx<'a>, callee: Self::Expr<'a>, args: Vec<Self::Expr<'a>>) -> Self::Expr<'a>;

    // -- Binding-identity layer -------------------------------------------
    // The one place the swc `SyntaxContext` / oxc `Semantic` split is bridged.

    /// A binding's identity — swc `Id = (Atom, SyntaxContext)`, oxc `SymbolId`.
    /// Not `Copy` because swc's carries an `Atom`.
    type BindingId: Clone + Eq + std::hash::Hash;

    /// The resolution context a *reference* needs to find its binding. swc keeps
    /// almost everything on the ident itself, so this is just the ctxt that
    /// marks unresolved (global) refs; oxc keeps it all here, in `Semantic`.
    type Semantics<'a>;

    /// Run binding resolution over a parsed program and hand back the context
    /// the query methods below need. Call before any of them. (swc resolves at
    /// parse time and this just surfaces the marker; oxc builds `Semantic` here,
    /// which borrows `program` — hence the shared, tied-lifetime borrow.)
    fn build_semantics<'a>(program: &'a Self::Program<'a>) -> Self::Semantics<'a>;

    /// Invoke `f` once per bound `import` specifier, in source order.
    fn for_each_import<'a>(
        program: &Self::Program<'a>,
        sem: &Self::Semantics<'a>,
        f: impl FnMut(ImportBinding<Self::BindingId>),
    );

    /// Read-only post-order walk over every expression (the write-path dual is
    /// [`visit_exprs_mut`](Self::visit_exprs_mut); a `Semantics` borrow rules the
    /// mutable one out on oxc, so analyses need this).
    fn visit_exprs<'a>(program: &Self::Program<'a>, f: impl FnMut(&Self::Expr<'a>));

    /// If `expr` is a bare identifier *reference* that resolves to a binding,
    /// that binding's id. `None` for non-identifiers **and** for unresolved
    /// (global) references — on oxc those are indistinguishable here, which is
    /// exactly why [`is_free_ident`](Self::is_free_ident) is its own method.
    fn ident_binding<'a>(sem: &Self::Semantics<'a>, expr: &Self::Expr<'a>) -> Option<Self::BindingId>;

    /// The source name of `expr` if it is a bare identifier (bound or free).
    fn ident_name<'r, 'a: 'r>(expr: &'r Self::Expr<'a>) -> Option<&'r str>;

    /// `(object, property_name)` if `expr` is a static member access `obj.prop`.
    fn as_static_member<'r, 'a: 'r>(
        expr: &'r Self::Expr<'a>,
    ) -> Option<(&'r Self::Expr<'a>, &'r str)>;

    /// Whether `expr` is a free (unresolved / global) identifier reference. This
    /// can't be derived from [`ident_binding`](Self::ident_binding): on swc a
    /// global still has an `Id`, so freeness is `ctxt == unresolved`; on oxc a
    /// global has no `SymbolId`. The divergence lives here, resolved once.
    fn is_free_ident<'a>(sem: &Self::Semantics<'a>, expr: &Self::Expr<'a>) -> bool;
}

// ===========================================================================
// oxc backend. (The sibling `oxc` module — the direct transform — shadows the
// `oxc` crate, so we reach the crate through a leading `::oxc`.)
// ===========================================================================

mod oxc_backend {
    use ::oxc::allocator::{Allocator, Dummy, FromIn, GetAllocator, Vec as ArenaVec};
    use ::oxc::ast::ast::{
        Argument, ChainElement, Expression, ImportDeclarationSpecifier, Program, Statement,
    };
    use ::oxc::ast::AstBuilder;
    use ::oxc::ast::NONE;
    use ::oxc::ast_visit::{walk, walk_mut, Visit, VisitMut};
    use ::oxc::codegen::Codegen;
    use ::oxc::parser::Parser;
    use ::oxc::semantic::{Semantic, SemanticBuilder, SymbolId};
    use ::oxc::span::{SourceType, SPAN};
    use ::oxc::syntax::operator::LogicalOperator;

    use super::{Backend, ExprRefMut, ImportBinding, Imported, MemberKeyMut, OptChainBaseMut};

    pub struct Oxc;

    struct EveryExpr<F>(F);
    impl<'a, F: FnMut(&mut Expression<'a>)> VisitMut<'a> for EveryExpr<F> {
        fn visit_expression(&mut self, e: &mut Expression<'a>) {
            walk_mut::walk_expression(self, e); // post-order
            (self.0)(e);
        }
    }

    struct EveryExprRef<F>(F);
    impl<'a, F: FnMut(&Expression<'a>)> Visit<'a> for EveryExprRef<F> {
        fn visit_expression(&mut self, e: &Expression<'a>) {
            walk::walk_expression(self, e); // post-order
            (self.0)(e);
        }
    }

    /// A bare member is `Member`; a bare `optional` member is an inner
    /// `OptChain` link.
    #[inline]
    fn member<'r, 'a: 'r>(
        optional: bool,
        object: &'r mut Expression<'a>,
        key: MemberKeyMut<'r, 'a, Oxc>,
    ) -> ExprRefMut<'r, 'a, Oxc> {
        if optional {
            ExprRefMut::OptChain { optional: true, base: OptChainBaseMut::Member { object, key } }
        } else {
            ExprRefMut::Member { object, key }
        }
    }

    impl Backend for Oxc {
        type Arena = Allocator;
        type Ctx<'a> = AstBuilder<'a>;
        type Program<'a> = Program<'a>;
        type Expr<'a> = Expression<'a>;

        fn parse<'a>(arena: &'a Self::Arena, source: &'a str) -> Self::Program<'a> {
            Parser::new(arena, source, SourceType::mjs()).parse().program
        }

        fn codegen(program: &Self::Program<'_>) -> String {
            Codegen::new().build(program).code
        }

        fn ctx(arena: &Self::Arena) -> Self::Ctx<'_> {
            AstBuilder::new(arena)
        }

        #[inline]
        fn visit_exprs_mut<'a>(program: &mut Self::Program<'a>, f: impl FnMut(&mut Self::Expr<'a>)) {
            EveryExpr(f).visit_program(program);
        }

        #[inline]
        fn expr_ref_mut<'r, 'a: 'r>(e: &'r mut Self::Expr<'a>) -> ExprRefMut<'r, 'a, Self> {
            // Reborrow through the arena `Box` once so sibling field borrows split.
            match e {
                Expression::BinaryExpression(bin) => {
                    let bin = &mut **bin;
                    ExprRefMut::Bin { left: &mut bin.left, right: &mut bin.right }
                }
                Expression::LogicalExpression(l) => {
                    let l = &mut **l;
                    let (left, right) = (&mut l.left, &mut l.right);
                    if l.operator == LogicalOperator::Coalesce {
                        ExprRefMut::Nullish { left, right }
                    } else {
                        ExprRefMut::Bin { left, right }
                    }
                }
                // Bare member: `optional` marks a `?.` link reached inside a chain.
                Expression::StaticMemberExpression(m) => {
                    let m = &mut **m;
                    member(m.optional, &mut m.object, MemberKeyMut::Static(m.property.name.as_str()))
                }
                Expression::ComputedMemberExpression(m) => {
                    let m = &mut **m;
                    let optional = m.optional;
                    member(optional, &mut m.object, MemberKeyMut::Computed(&mut m.expression))
                }
                Expression::PrivateFieldExpression(m) => {
                    let m = &mut **m;
                    member(m.optional, &mut m.object, MemberKeyMut::Private(m.field.name.as_str()))
                }
                Expression::CallExpression(c) if c.optional => {
                    ExprRefMut::OptChain { optional: true, base: OptChainBaseMut::Call }
                }
                // Chain root: the outermost link, lowered even when its own flag
                // is false (the `.c` in `a?.b.c` still short-circuits).
                Expression::ChainExpression(chain) => match &mut chain.expression {
                    ChainElement::StaticMemberExpression(m) => {
                        let m = &mut **m;
                        ExprRefMut::OptChain {
                            optional: m.optional,
                            base: OptChainBaseMut::Member {
                                object: &mut m.object,
                                key: MemberKeyMut::Static(m.property.name.as_str()),
                            },
                        }
                    }
                    ChainElement::ComputedMemberExpression(m) => {
                        let m = &mut **m;
                        ExprRefMut::OptChain {
                            optional: m.optional,
                            base: OptChainBaseMut::Member {
                                object: &mut m.object,
                                key: MemberKeyMut::Computed(&mut m.expression),
                            },
                        }
                    }
                    ChainElement::PrivateFieldExpression(m) => {
                        let m = &mut **m;
                        ExprRefMut::OptChain {
                            optional: m.optional,
                            base: OptChainBaseMut::Member {
                                object: &mut m.object,
                                key: MemberKeyMut::Private(m.field.name.as_str()),
                            },
                        }
                    }
                    ChainElement::CallExpression(c) => {
                        ExprRefMut::OptChain { optional: c.optional, base: OptChainBaseMut::Call }
                    }
                    ChainElement::TSNonNullExpression(_) => ExprRefMut::Other,
                },
                Expression::ParenthesizedExpression(p) => ExprRefMut::Paren { inner: &mut p.expression },
                _ => ExprRefMut::Other,
            }
        }

        #[inline]
        fn take_expr<'a>(cx: Self::Ctx<'a>, slot: &mut Self::Expr<'a>) -> Self::Expr<'a> {
            std::mem::replace(slot, Expression::dummy(cx.allocator()))
        }

        #[inline]
        fn ident<'a>(cx: Self::Ctx<'a>, name: &str) -> Self::Expr<'a> {
            cx.expression_identifier(SPAN, oxc_str::Ident::from_in(name, cx.allocator()))
        }

        #[inline]
        fn string<'a>(cx: Self::Ctx<'a>, value: &str) -> Self::Expr<'a> {
            cx.expression_string_literal(SPAN, oxc_str::Str::from_in(value, cx.allocator()), None)
        }

        #[inline]
        fn call<'a>(cx: Self::Ctx<'a>, callee: Self::Expr<'a>, args: Vec<Self::Expr<'a>>) -> Self::Expr<'a> {
            let mut arguments: ArenaVec<'a, Argument<'a>> = ArenaVec::new_in(&cx);
            for a in args {
                arguments.push(Argument::from(a));
            }
            cx.expression_call(SPAN, callee, NONE, arguments, false)
        }

        // -- Binding-identity layer ---------------------------------------
        // oxc's model: a side-table (`Semantic`) filled by a dedicated pass,
        // storing `SymbolId`/`ReferenceId` in per-node cells. Hygiene is NOT on
        // the AST — resolution is a lookup through `Semantic`.

        type BindingId = SymbolId;
        type Semantics<'a> = Semantic<'a>;

        fn build_semantics<'a>(program: &'a Self::Program<'a>) -> Self::Semantics<'a> {
            // Fills each `IdentifierReference::reference_id` / `BindingIdentifier::symbol_id`
            // cell and builds the scope/symbol tables. Borrows `program` for `'a`.
            SemanticBuilder::new().build(program).semantic
        }

        fn for_each_import<'a>(
            program: &Self::Program<'a>,
            _sem: &Self::Semantics<'a>,
            mut f: impl FnMut(ImportBinding<Self::BindingId>),
        ) {
            // In oxc, module declarations live in `program.body` as statements.
            for stmt in &program.body {
                let Statement::ImportDeclaration(import) = stmt else { continue };
                let source = import.source.value.as_str().to_string();
                let Some(specifiers) = &import.specifiers else { continue };
                for spec in specifiers {
                    let (local, imported) = match spec {
                        ImportDeclarationSpecifier::ImportSpecifier(s) => {
                            (s.local.symbol_id(), Imported::Named(s.imported.name().as_str().to_string()))
                        }
                        ImportDeclarationSpecifier::ImportDefaultSpecifier(s) => {
                            (s.local.symbol_id(), Imported::Default)
                        }
                        ImportDeclarationSpecifier::ImportNamespaceSpecifier(s) => {
                            (s.local.symbol_id(), Imported::Namespace)
                        }
                    };
                    f(ImportBinding { local, source: source.clone(), imported });
                }
            }
        }

        fn visit_exprs<'a>(program: &Self::Program<'a>, f: impl FnMut(&Self::Expr<'a>)) {
            EveryExprRef(f).visit_program(program);
        }

        fn ident_binding<'a>(sem: &Self::Semantics<'a>, expr: &Self::Expr<'a>) -> Option<Self::BindingId> {
            let Expression::Identifier(r) = expr else { return None };
            // Reference → its resolved symbol, if any. `None` = free/global.
            sem.scoping().get_reference(r.reference_id.get()?).symbol_id()
        }

        fn ident_name<'r, 'a: 'r>(expr: &'r Self::Expr<'a>) -> Option<&'r str> {
            match expr {
                Expression::Identifier(r) => Some(r.name.as_str()),
                _ => None,
            }
        }

        fn as_static_member<'r, 'a: 'r>(
            expr: &'r Self::Expr<'a>,
        ) -> Option<(&'r Self::Expr<'a>, &'r str)> {
            match expr {
                Expression::StaticMemberExpression(m) => Some((&m.object, m.property.name.as_str())),
                _ => None,
            }
        }

        fn is_free_ident<'a>(sem: &Self::Semantics<'a>, expr: &Self::Expr<'a>) -> bool {
            let Expression::Identifier(r) = expr else { return false };
            match r.reference_id.get() {
                // Resolved to no symbol ⇒ free/global.
                Some(rid) => sem.scoping().get_reference(rid).symbol_id().is_none(),
                None => true,
            }
        }
    }
}

// ===========================================================================
// swc backend.
// ===========================================================================

mod swc_backend {
    use std::sync::Arc;

    use swc_common::sync::Lrc;
    use swc_common::util::take::Take;
    use swc_common::{FileName, Globals, Mark, SourceMap, SyntaxContext, DUMMY_SP, GLOBALS};
    use swc_ecma_ast::{
        BinaryOp, CallExpr, Callee, EsVersion, Expr, ExprOrSpread, Id, Ident, ImportSpecifier, Lit,
        MemberProp, ModuleDecl, ModuleItem, OptChainBase, Program, Str,
    };
    use swc_ecma_codegen::text_writer::JsWriter;
    use swc_ecma_codegen::{Emitter, Node};
    use swc_ecma_parser::{parse_file_as_program, EsSyntax, Syntax};
    use swc_ecma_transforms_base::resolver;
    use swc_ecma_visit::{Visit, VisitMut, VisitMutWith, VisitWith};

    use super::{Backend, ExprRefMut, ImportBinding, Imported, MemberKeyMut, OptChainBaseMut};

    pub struct Swc;

    /// swc's AST is owned; keep the `SourceMap` alongside it for codegen. Also
    /// carries the hygiene state the resolver produced: `globals` owns the
    /// `SyntaxContext` interner the AST's ctxts index into, and `unresolved_ctxt`
    /// is the ctxt the resolver stamps on free (global) references.
    pub struct SwcProgram {
        pub program: Program,
        cm: Lrc<SourceMap>,
        #[allow(dead_code)] // held only to keep the ctxt interner alive for the AST.
        globals: Arc<Globals>,
        unresolved_ctxt: SyntaxContext,
    }

    /// swc keeps binding info on the AST (in each ident's `SyntaxContext`), so
    /// the only side-context a reference query needs is which ctxt means "free".
    #[derive(Clone, Copy)]
    pub struct SwcSemantics {
        unresolved_ctxt: SyntaxContext,
    }

    struct EveryExpr<F>(F);
    impl<F: FnMut(&mut Expr)> VisitMut for EveryExpr<F> {
        fn visit_mut_expr(&mut self, e: &mut Expr) {
            e.visit_mut_children_with(self); // post-order
            (self.0)(e);
        }
    }

    struct EveryExprRef<F>(F);
    impl<F: FnMut(&Expr)> Visit for EveryExprRef<F> {
        fn visit_expr(&mut self, e: &Expr) {
            e.visit_children_with(self); // post-order
            (self.0)(e);
        }
    }

    #[inline]
    fn key<'r, 'a: 'r>(prop: &'r mut MemberProp) -> MemberKeyMut<'r, 'a, Swc> {
        match prop {
            MemberProp::Ident(id) => MemberKeyMut::Static(id.sym.as_str()),
            MemberProp::Computed(c) => MemberKeyMut::Computed(&mut *c.expr),
            MemberProp::PrivateName(p) => MemberKeyMut::Private(p.name.as_str()),
        }
    }

    impl Backend for Swc {
        type Arena = ();
        type Ctx<'a> = ();
        type Program<'a> = SwcProgram;
        type Expr<'a> = Expr;

        fn parse<'a>(_arena: &'a Self::Arena, source: &'a str) -> Self::Program<'a> {
            let cm: Lrc<SourceMap> = Default::default();
            let fm = cm.new_source_file(Lrc::new(FileName::Custom("in.js".into())), source.to_string());
            let mut recovered = Vec::new();
            let mut program = parse_file_as_program(
                &fm,
                Syntax::Es(EsSyntax::default()),
                EsVersion::latest(),
                None,
                &mut recovered,
            )
            .expect("parse failed");
            // swc's binding resolution: stamp every ident with a hygienic
            // `SyntaxContext` so `to_id()` names a binding and free references
            // carry `unresolved_ctxt`. Marks + resolver must run inside `GLOBALS`.
            let globals = Arc::new(Globals::new());
            let unresolved_ctxt = GLOBALS.set(&globals, || {
                let unresolved_mark = Mark::new();
                let top_level_mark = Mark::new();
                program.visit_mut_with(&mut resolver(unresolved_mark, top_level_mark, false));
                SyntaxContext::empty().apply_mark(unresolved_mark)
            });
            SwcProgram { program, cm, globals, unresolved_ctxt }
        }

        fn codegen(program: &Self::Program<'_>) -> String {
            let mut buf = Vec::new();
            {
                let mut emitter = Emitter {
                    cfg: Default::default(),
                    cm: program.cm.clone(),
                    comments: None,
                    wr: JsWriter::new(program.cm.clone(), "\n", &mut buf, None),
                };
                program.program.emit_with(&mut emitter).expect("emit failed");
            }
            String::from_utf8(buf).expect("valid utf-8")
        }

        fn ctx(_arena: &Self::Arena) -> Self::Ctx<'_> {}

        #[inline]
        fn visit_exprs_mut<'a>(program: &mut Self::Program<'a>, f: impl FnMut(&mut Self::Expr<'a>)) {
            program.program.visit_mut_with(&mut EveryExpr(f));
        }

        #[inline]
        fn expr_ref_mut<'r, 'a: 'r>(e: &'r mut Self::Expr<'a>) -> ExprRefMut<'r, 'a, Self> {
            match e {
                Expr::Bin(bin) => {
                    let (left, right) = (&mut *bin.left, &mut *bin.right);
                    if bin.op == BinaryOp::NullishCoalescing {
                        ExprRefMut::Nullish { left, right }
                    } else {
                        ExprRefMut::Bin { left, right }
                    }
                }
                Expr::Member(m) => ExprRefMut::Member { object: &mut *m.obj, key: key(&mut m.prop) },
                Expr::OptChain(oc) => ExprRefMut::OptChain {
                    optional: oc.optional,
                    base: match &mut *oc.base {
                        OptChainBase::Call(_) => OptChainBaseMut::Call,
                        OptChainBase::Member(m) => {
                            OptChainBaseMut::Member { object: &mut *m.obj, key: key(&mut m.prop) }
                        }
                    },
                },
                Expr::Paren(p) => ExprRefMut::Paren { inner: &mut *p.expr },
                _ => ExprRefMut::Other,
            }
        }

        #[inline]
        fn take_expr<'a>(_cx: Self::Ctx<'a>, slot: &mut Self::Expr<'a>) -> Self::Expr<'a> {
            slot.take()
        }

        #[inline]
        fn ident<'a>(_cx: Self::Ctx<'a>, name: &str) -> Self::Expr<'a> {
            Expr::Ident(Ident::new_no_ctxt(name.into(), DUMMY_SP))
        }

        #[inline]
        fn string<'a>(_cx: Self::Ctx<'a>, value: &str) -> Self::Expr<'a> {
            Expr::Lit(Lit::Str(Str { span: DUMMY_SP, value: value.into(), raw: None }))
        }

        #[inline]
        fn call<'a>(_cx: Self::Ctx<'a>, callee: Self::Expr<'a>, args: Vec<Self::Expr<'a>>) -> Self::Expr<'a> {
            Expr::Call(CallExpr {
                span: DUMMY_SP,
                ctxt: SyntaxContext::empty(),
                callee: Callee::Expr(Box::new(callee)),
                args: args.into_iter().map(|e| ExprOrSpread { spread: None, expr: Box::new(e) }).collect(),
                type_args: None,
            })
        }

        // -- Binding-identity layer ---------------------------------------
        // swc's model: hygiene is *on* the AST. `Id = (Atom, SyntaxContext)`
        // names a binding, and freeness is a ctxt comparison — no side-table.

        type BindingId = Id;
        type Semantics<'a> = SwcSemantics;

        fn build_semantics<'a>(program: &'a Self::Program<'a>) -> Self::Semantics<'a> {
            // Resolution already happened in `parse`; just surface the marker.
            SwcSemantics { unresolved_ctxt: program.unresolved_ctxt }
        }

        fn for_each_import<'a>(
            program: &Self::Program<'a>,
            _sem: &Self::Semantics<'a>,
            mut f: impl FnMut(ImportBinding<Self::BindingId>),
        ) {
            let Program::Module(module) = &program.program else { return };
            for item in &module.body {
                let ModuleItem::ModuleDecl(ModuleDecl::Import(import)) = item else { continue };
                let source = import.src.value.as_wtf8().to_string_lossy().into_owned();
                for spec in &import.specifiers {
                    let (local, imported) = match spec {
                        ImportSpecifier::Named(n) => {
                            let orig = match &n.imported {
                                Some(m) => m.atom().to_string(),
                                None => n.local.sym.to_string(),
                            };
                            (n.local.to_id(), Imported::Named(orig))
                        }
                        ImportSpecifier::Default(d) => (d.local.to_id(), Imported::Default),
                        ImportSpecifier::Namespace(ns) => (ns.local.to_id(), Imported::Namespace),
                    };
                    f(ImportBinding { local, source: source.clone(), imported });
                }
            }
        }

        fn visit_exprs<'a>(program: &Self::Program<'a>, f: impl FnMut(&Self::Expr<'a>)) {
            program.program.visit_with(&mut EveryExprRef(f));
        }

        fn ident_binding<'a>(_sem: &Self::Semantics<'a>, expr: &Self::Expr<'a>) -> Option<Self::BindingId> {
            match expr {
                // `to_id()` always succeeds — a free ident just has the
                // unresolved ctxt, so map lookups against it simply miss.
                Expr::Ident(i) => Some(i.to_id()),
                _ => None,
            }
        }

        fn ident_name<'r, 'a: 'r>(expr: &'r Self::Expr<'a>) -> Option<&'r str> {
            match expr {
                Expr::Ident(i) => Some(i.sym.as_str()),
                _ => None,
            }
        }

        fn as_static_member<'r, 'a: 'r>(
            expr: &'r Self::Expr<'a>,
        ) -> Option<(&'r Self::Expr<'a>, &'r str)> {
            match expr {
                Expr::Member(m) => match &m.prop {
                    MemberProp::Ident(p) => Some((&*m.obj, p.sym.as_str())),
                    _ => None,
                },
                _ => None,
            }
        }

        fn is_free_ident<'a>(sem: &Self::Semantics<'a>, expr: &Self::Expr<'a>) -> bool {
            matches!(expr, Expr::Ident(i) if i.ctxt == sem.unresolved_ctxt)
        }
    }
}
