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
pub mod oxc;
pub mod swc;

pub use oxc_backend::Oxc;
pub use swc_backend::{Swc, SwcProgram};

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
}

// ===========================================================================
// oxc backend. (The sibling `oxc` module — the direct transform — shadows the
// `oxc` crate, so we reach the crate through a leading `::oxc`.)
// ===========================================================================

mod oxc_backend {
    use ::oxc::allocator::{Allocator, Dummy, FromIn, GetAllocator, Vec as ArenaVec};
    use ::oxc::ast::ast::{Argument, ChainElement, Expression, Program};
    use ::oxc::ast::AstBuilder;
    use ::oxc::ast::NONE;
    use ::oxc::ast_visit::{walk_mut, VisitMut};
    use ::oxc::codegen::Codegen;
    use ::oxc::parser::Parser;
    use ::oxc::span::{SourceType, SPAN};
    use ::oxc::syntax::operator::LogicalOperator;

    use super::{Backend, ExprRefMut, MemberKeyMut, OptChainBaseMut};

    pub struct Oxc;

    struct EveryExpr<F>(F);
    impl<'a, F: FnMut(&mut Expression<'a>)> VisitMut<'a> for EveryExpr<F> {
        fn visit_expression(&mut self, e: &mut Expression<'a>) {
            walk_mut::walk_expression(self, e); // post-order
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
    }
}

// ===========================================================================
// swc backend.
// ===========================================================================

mod swc_backend {
    use swc_common::sync::Lrc;
    use swc_common::util::take::Take;
    use swc_common::{FileName, SourceMap, SyntaxContext, DUMMY_SP};
    use swc_ecma_ast::{
        BinaryOp, CallExpr, Callee, EsVersion, Expr, ExprOrSpread, Ident, Lit, MemberProp,
        OptChainBase, Program, Str,
    };
    use swc_ecma_codegen::text_writer::JsWriter;
    use swc_ecma_codegen::{Emitter, Node};
    use swc_ecma_parser::{parse_file_as_program, EsSyntax, Syntax};
    use swc_ecma_visit::{VisitMut, VisitMutWith};

    use super::{Backend, ExprRefMut, MemberKeyMut, OptChainBaseMut};

    pub struct Swc;

    /// swc's AST is owned; keep the `SourceMap` alongside it for codegen.
    pub struct SwcProgram {
        pub program: Program,
        cm: Lrc<SourceMap>,
    }

    struct EveryExpr<F>(F);
    impl<F: FnMut(&mut Expr)> VisitMut for EveryExpr<F> {
        fn visit_mut_expr(&mut self, e: &mut Expr) {
            e.visit_mut_children_with(self); // post-order
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
            let program = parse_file_as_program(
                &fm,
                Syntax::Es(EsSyntax::default()),
                EsVersion::latest(),
                None,
                &mut recovered,
            )
            .expect("parse failed");
            SwcProgram { program, cm }
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
    }
}
