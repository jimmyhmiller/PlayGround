//! ANF normalization for GC safety.
//!
//! A precise, moving GC can only find live pointers that are in *frame root
//! slots* — i.e. in declared locals (codegen roots every `Ref` local, and every
//! value-with-ref local via indirect roots). A GC value computed as an inline
//! *temporary* (an operand evaluated and held in a register while a later
//! operand or the enclosing allocation runs) is invisible to the collector, so a
//! collection there would dangle it. Example:
//!
//! ```text
//! Cons(makeA(), makeB())   // makeA()'s result is live across makeB()'s alloc
//! ```
//!
//! This pass rewrites every function body into administrative-normal form for GC
//! values: every non-atomic subexpression whose result is a GC value (a `Ref`,
//! or a flattened `#[value]` aggregate that transitively holds a ref) is
//! let-bound to a fresh local before use. After this, no GC value is ever live in
//! a temporary across a safepoint — they are all locals, and the prologue roots
//! them. Scalars are left alone.
//!
//! Evaluation order is preserved: hoisted bindings are emitted left-to-right in
//! the statement sequence that contains the expression, matching codegen's
//! left-to-right operand evaluation. Branch/loop bodies are normalized as their
//! own contexts (their bindings stay inside the branch), so nothing is hoisted
//! across a conditional and short-circuit/laziness is preserved.

use crate::core::*;

/// Normalize every function in the program for GC-temporary safety.
pub fn anf_program(prog: &mut CoreProgram) {
    // Precompute, per ValueId, whether the value aggregate transitively holds a
    // GC reference (so a flattened value temp also needs rooting). Done up front
    // to avoid borrowing `prog.values` while mutating `prog.funcs`.
    let gc_value: Vec<bool> = (0..prog.values.len() as u32)
        .map(|v| value_has_ref(&prog.values, v))
        .collect();

    for f in &mut prog.funcs {
        if f.is_extern {
            continue;
        }
        let body = std::mem::replace(&mut f.body, CoreBlock { stmts: Vec::new(), tail: None });
        let mut anf = Anf { gc_value: &gc_value, locals: &mut f.locals };
        f.body = anf.block(body);
    }
}

/// Whether value type `vid` (transitively) holds a GC reference.
fn value_has_ref(values: &[ValueLayout], vid: u32) -> bool {
    let vl = &values[vid as usize];
    let mut all: Vec<&Repr> = Vec::new();
    match &vl.variants {
        Some(variants) => variants.iter().for_each(|v| all.extend(v.fields.iter())),
        None => all.extend(vl.fields.iter()),
    }
    all.iter().any(|f| match f {
        Repr::Ref(_) => true,
        Repr::Value(s) => value_has_ref(values, *s),
        _ => false,
    })
}

struct Anf<'a> {
    gc_value: &'a [bool],
    locals: &'a mut Vec<Repr>,
}

impl<'a> Anf<'a> {
    fn fresh(&mut self, repr: Repr) -> LocalId {
        let id = self.locals.len() as LocalId;
        self.locals.push(repr);
        id
    }

    /// A GC value the collector must be able to find: a heap reference, or a
    /// flattened value aggregate holding one.
    fn is_gc(&self, repr: &Repr) -> bool {
        match repr {
            Repr::Ref(_) => true,
            Repr::Value(v) => self.gc_value[*v as usize],
            _ => false,
        }
    }

    /// An expression that needs no binding to be a safe operand: already a local,
    /// or a constant with no allocation. (`ConstStr` allocates a String, so it is
    /// NOT atomic.)
    fn is_atomic(e: &CoreExpr) -> bool {
        matches!(
            *e.kind,
            CoreExprKind::Local(_)
                | CoreExprKind::Unit
                | CoreExprKind::ConstInt(..)
                | CoreExprKind::ConstFloat(..)
                | CoreExprKind::ConstBool(_)
                | CoreExprKind::ConstChar(_)
                | CoreExprKind::ConstZero(_)
                | CoreExprKind::CallbackPtr(_)
        )
    }

    /// Normalize a block: each statement's hoisted bindings are flushed into the
    /// statement sequence just before it, preserving order. The block is its own
    /// evaluation context — nothing escapes it.
    fn block(&mut self, b: CoreBlock) -> CoreBlock {
        let mut out: Vec<CoreStmt> = Vec::with_capacity(b.stmts.len());
        for s in b.stmts {
            match s {
                CoreStmt::Let(id, e) => {
                    let mut pending = Vec::new();
                    let e2 = self.expr(e, &mut pending);
                    out.append(&mut pending);
                    out.push(CoreStmt::Let(id, e2));
                }
                CoreStmt::Expr(e) => {
                    let mut pending = Vec::new();
                    let e2 = self.expr(e, &mut pending);
                    out.append(&mut pending);
                    out.push(CoreStmt::Expr(e2));
                }
            }
        }
        let tail = b.tail.map(|t| {
            let mut pending = Vec::new();
            let t2 = self.expr(t, &mut pending);
            out.append(&mut pending);
            t2
        });
        CoreBlock { stmts: out, tail }
    }

    /// Normalize an expression that forms its own evaluation context (a match-arm
    /// body): any hoisted bindings are wrapped into a `Block` so they stay inside
    /// the arm rather than leaking to the enclosing sequence.
    fn context(&mut self, e: CoreExpr) -> CoreExpr {
        let mut pending = Vec::new();
        let e2 = self.expr(e, &mut pending);
        if pending.is_empty() {
            e2
        } else {
            let repr = e2.repr.clone();
            CoreExpr::new(
                CoreExprKind::Block(Box::new(CoreBlock { stmts: pending, tail: Some(e2) })),
                repr,
            )
        }
    }

    /// Normalize an eager operand and, if its result is a non-atomic GC value,
    /// hoist it to a fresh local (rooted by the prologue) so it survives any later
    /// safepoint. Scalars and atoms pass through unchanged.
    fn operand(&mut self, e: CoreExpr, pending: &mut Vec<CoreStmt>) -> CoreExpr {
        let e2 = self.expr(e, pending);
        if self.is_gc(&e2.repr) && !Self::is_atomic(&e2) {
            let repr = e2.repr.clone();
            let id = self.fresh(repr.clone());
            pending.push(CoreStmt::Let(id, e2));
            CoreExpr::new(CoreExprKind::Local(id), repr)
        } else {
            e2
        }
    }

    fn boxed(&mut self, e: Box<CoreExpr>, pending: &mut Vec<CoreStmt>) -> Box<CoreExpr> {
        Box::new(self.operand(*e, pending))
    }

    fn list(&mut self, es: Vec<CoreExpr>, pending: &mut Vec<CoreStmt>) -> Vec<CoreExpr> {
        es.into_iter().map(|e| self.operand(e, pending)).collect()
    }

    fn arm(&mut self, a: CoreArm) -> CoreArm {
        CoreArm { tag: a.tag, binds: a.binds, body: self.context(a.body) }
    }

    /// Normalize `e` in eager position: rebuild it with normalized children
    /// (operands atomized into `pending`), but do NOT hoist `e` itself — that is
    /// the caller's (`operand`) decision based on where `e` sits.
    fn expr(&mut self, e: CoreExpr, pending: &mut Vec<CoreStmt>) -> CoreExpr {
        let repr = e.repr.clone();
        let kind: CoreExprKind = *e.kind;
        let k = match kind {
            // Atoms — no children.
            k @ (CoreExprKind::ConstInt(..)
            | CoreExprKind::ConstFloat(..)
            | CoreExprKind::ConstBool(_)
            | CoreExprKind::ConstChar(_)
            | CoreExprKind::ConstZero(_)
            | CoreExprKind::ConstStr(_)
            | CoreExprKind::Unit
            | CoreExprKind::Local(_)
            | CoreExprKind::CallbackPtr(_)
            | CoreExprKind::ThreadYield
            | CoreExprKind::ThreadCurrentId
            | CoreExprKind::Continue) => k,

            // Single eager operand.
            CoreExprKind::ThreadSleep(a) => CoreExprKind::ThreadSleep(self.boxed(a, pending)),
            CoreExprKind::PtrReadI64(a) => CoreExprKind::PtrReadI64(self.boxed(a, pending)),
            CoreExprKind::Un(op, a) => CoreExprKind::Un(op, self.boxed(a, pending)),
            CoreExprKind::FloatIntrinsic(op, a) => {
                CoreExprKind::FloatIntrinsic(op, self.boxed(a, pending))
            }
            CoreExprKind::Print(a) => CoreExprKind::Print(self.boxed(a, pending)),
            CoreExprKind::PrintStr(a) => CoreExprKind::PrintStr(self.boxed(a, pending)),
            CoreExprKind::PrintStrRaw(a) => CoreExprKind::PrintStrRaw(self.boxed(a, pending)),
            CoreExprKind::StrLen(a) => CoreExprKind::StrLen(self.boxed(a, pending)),
            CoreExprKind::StrToFloat(a) => CoreExprKind::StrToFloat(self.boxed(a, pending)),
            CoreExprKind::FloatBits(a) => CoreExprKind::FloatBits(self.boxed(a, pending)),
            CoreExprKind::StrHash(a) => CoreExprKind::StrHash(self.boxed(a, pending)),
            CoreExprKind::TypeIdOf(a) => CoreExprKind::TypeIdOf(self.boxed(a, pending)),
            CoreExprKind::ArrayLen(a) => CoreExprKind::ArrayLen(self.boxed(a, pending)),
            CoreExprKind::ThreadSpawn(a) => CoreExprKind::ThreadSpawn(self.boxed(a, pending)),
            CoreExprKind::ThreadJoin(a) => CoreExprKind::ThreadJoin(self.boxed(a, pending)),
            CoreExprKind::EnumTag(a) => CoreExprKind::EnumTag(self.boxed(a, pending)),
            CoreExprKind::TypeNameOf { layout, obj } => {
                CoreExprKind::TypeNameOf { layout, obj: self.boxed(obj, pending) }
            }
            CoreExprKind::StrFromNum { layout, is_float, v } => {
                CoreExprKind::StrFromNum { layout, is_float, v: self.boxed(v, pending) }
            }
            CoreExprKind::StrFromChar { layout, cp } => {
                CoreExprKind::StrFromChar { layout, cp: self.boxed(cp, pending) }
            }
            CoreExprKind::ReadFile { layout, path } => {
                CoreExprKind::ReadFile { layout, path: self.boxed(path, pending) }
            }
            CoreExprKind::ArrayNew { layout, len, elem } => {
                CoreExprKind::ArrayNew { layout, len: self.boxed(len, pending), elem }
            }
            CoreExprKind::AtomLoad { atom, elem } => {
                CoreExprKind::AtomLoad { atom: self.boxed(atom, pending), elem }
            }
            CoreExprKind::AsCBytes { src, elem, is_string, copy_out } => {
                CoreExprKind::AsCBytes { src: self.boxed(src, pending), elem, is_string, copy_out }
            }
            CoreExprKind::EnumPayload { scrutinee, field, repr: r, payload_reprs } => {
                CoreExprKind::EnumPayload {
                    scrutinee: self.boxed(scrutinee, pending),
                    field,
                    repr: r,
                    payload_reprs,
                }
            }
            CoreExprKind::Field { base, loc } => {
                CoreExprKind::Field { base: self.boxed(base, pending), loc }
            }
            CoreExprKind::Cast { value, from, to } => {
                CoreExprKind::Cast { value: self.boxed(value, pending), from, to }
            }
            CoreExprKind::Assign { local, value } => {
                CoreExprKind::Assign { local, value: self.boxed(value, pending) }
            }

            // Two eager operands.
            CoreExprKind::Bin(op, a, b) => {
                let a = self.boxed(a, pending);
                let b = self.boxed(b, pending);
                CoreExprKind::Bin(op, a, b)
            }
            CoreExprKind::StrEq(a, b) => {
                let a = self.boxed(a, pending);
                let b = self.boxed(b, pending);
                CoreExprKind::StrEq(a, b)
            }
            CoreExprKind::StrGet(a, b) => {
                let a = self.boxed(a, pending);
                let b = self.boxed(b, pending);
                CoreExprKind::StrGet(a, b)
            }
            CoreExprKind::StrConcat { layout, a, b } => {
                let a = self.boxed(a, pending);
                let b = self.boxed(b, pending);
                CoreExprKind::StrConcat { layout, a, b }
            }
            CoreExprKind::ArrayGet { array, index, elem } => {
                let array = self.boxed(array, pending);
                let index = self.boxed(index, pending);
                CoreExprKind::ArrayGet { array, index, elem }
            }
            CoreExprKind::ChanRecv { buf, ctrl, elem } => {
                let buf = self.boxed(buf, pending);
                let ctrl = self.boxed(ctrl, pending);
                CoreExprKind::ChanRecv { buf, ctrl, elem }
            }
            CoreExprKind::SetField { base, loc, value } => {
                let base = self.boxed(base, pending);
                let value = self.boxed(value, pending);
                CoreExprKind::SetField { base, loc, value }
            }

            // Three eager operands.
            CoreExprKind::StrSubstring { layout, s, start, end } => {
                let s = self.boxed(s, pending);
                let start = self.boxed(start, pending);
                let end = self.boxed(end, pending);
                CoreExprKind::StrSubstring { layout, s, start, end }
            }
            CoreExprKind::ArraySet { array, index, value, elem } => {
                let array = self.boxed(array, pending);
                let index = self.boxed(index, pending);
                let value = self.boxed(value, pending);
                CoreExprKind::ArraySet { array, index, value, elem }
            }
            CoreExprKind::AtomCas { atom, old, new } => {
                let atom = self.boxed(atom, pending);
                let old = self.boxed(old, pending);
                let new = self.boxed(new, pending);
                CoreExprKind::AtomCas { atom, old, new }
            }
            CoreExprKind::ChanSend { buf, ctrl, value } => {
                let buf = self.boxed(buf, pending);
                let ctrl = self.boxed(ctrl, pending);
                let value = self.boxed(value, pending);
                CoreExprKind::ChanSend { buf, ctrl, value }
            }

            // Operand lists.
            CoreExprKind::Call(id, args) => CoreExprKind::Call(id, self.list(args, pending)),
            CoreExprKind::RuntimeCall { func, args, ret } => {
                CoreExprKind::RuntimeCall { func, args: self.list(args, pending), ret }
            }
            CoreExprKind::MakeClosure { code, env, captures } => {
                CoreExprKind::MakeClosure { code, env, captures: self.list(captures, pending) }
            }
            CoreExprKind::New { layout, fields } => {
                CoreExprKind::New { layout, fields: self.list(fields, pending) }
            }
            CoreExprKind::MakeValue { value, fields } => {
                CoreExprKind::MakeValue { value, fields: self.list(fields, pending) }
            }
            CoreExprKind::MakeVariant { layout, tag, fields } => {
                CoreExprKind::MakeVariant { layout, tag, fields: self.list(fields, pending) }
            }
            CoreExprKind::MakeValueVariant { value, tag, fields } => {
                CoreExprKind::MakeValueVariant { value, tag, fields: self.list(fields, pending) }
            }
            CoreExprKind::CallClosure { callee, args } => {
                let callee = self.boxed(callee, pending);
                let args = self.list(args, pending);
                CoreExprKind::CallClosure { callee, args }
            }

            // Operand + optional value.
            CoreExprKind::Return(v) => {
                CoreExprKind::Return(v.map(|e| self.boxed(e, pending)))
            }
            CoreExprKind::Break(v) => {
                CoreExprKind::Break(v.map(|e| self.boxed(e, pending)))
            }

            // Control-flow contexts: the scrutinee/condition is eager (and may be
            // hoisted by the caller), but branch/arm bodies are their own
            // contexts so nothing is hoisted across them.
            CoreExprKind::Match { scrutinee, arms } => {
                let scrutinee = self.boxed(scrutinee, pending);
                let arms = arms.into_iter().map(|a| self.arm(a)).collect();
                CoreExprKind::Match { scrutinee, arms }
            }
            CoreExprKind::ValueMatch { scrutinee, arms } => {
                let scrutinee = self.boxed(scrutinee, pending);
                let arms = arms.into_iter().map(|a| self.arm(a)).collect();
                CoreExprKind::ValueMatch { scrutinee, arms }
            }
            CoreExprKind::If(cond, then_b, else_b) => {
                let cond = self.boxed(cond, pending);
                let then_b = Box::new(self.block(*then_b));
                let else_b = Box::new(self.block(*else_b));
                CoreExprKind::If(cond, then_b, else_b)
            }
            CoreExprKind::Block(b) => CoreExprKind::Block(Box::new(self.block(*b))),
            CoreExprKind::Loop(b) => CoreExprKind::Loop(Box::new(self.block(*b))),
        };
        CoreExpr::new(k, repr)
    }
}
