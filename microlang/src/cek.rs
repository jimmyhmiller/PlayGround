//! A stackless CEK-style interpreter — the execution tier that supports FULL,
//! multi-shot continuations.
//!
//! The other tiers evaluate on the host (Rust) call stack, so "the current
//! continuation" is that native stack — uncapturable, and gone once a call
//! returns. This machine makes the continuation an explicit, `Arc`-linked,
//! heap-allocated data structure (`Kont`). Evaluation is a loop over
//! `Eval`/`Apply` states, no host recursion. `%callcc` reifies the current
//! `Kont` as a first-class value; invoking it re-installs that `Kont` — any
//! number of times, because it is immutable and shared. That is what makes
//! generators, coroutines, and backtracking possible, and it is why tail calls
//! are automatically iterative here.
//!
//! This is the deepest validation of the `CodeSpace` axis: multi-shot
//! continuations are not a mechanism bolted onto the existing evaluator; they
//! require a fundamentally DIFFERENT evaluation strategy. And it slots in as
//! just another `CodeSpace` over the SAME High IR.
//!
//! Scope: the functional core (const/var/if/do/let/lambda/call/prim/set!/callcc).
//! It does not integrate with the moving GC (no `(gc)` mid-run) or dispatch —
//! those error clearly. It is self-contained (does not compose via `top`),
//! because it is a machine, not a recursive evaluator.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::code::CodeSpace;
use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::Runtime;
use crate::value::{clone_slots, frame_get, frame_set, slot_load, Frame, Locals, Obj, Sym, Val};

/// The reified continuation: "what to do with the value of the current
/// subexpression." An `Arc`-linked stack of frames. Immutable, so re-installable
/// (multi-shot). Stores no value-model-specific data, so it is not generic.
pub enum Kont {
    Done,
    If { then_: Ir, else_: Ir, env: Locals, next: Arc<Kont> },
    Seq { rest: Vec<Ir>, env: Locals, next: Arc<Kont> },
    Def { name: Sym, next: Arc<Kont> },
    SetLoc { up: u16, idx: u16, env: Locals, next: Arc<Kont> },
    SetGlob { name: Sym, next: Arc<Kont> },
    // `done` holds already-evaluated argument values. They are raw heap pointers
    // the moving GC must be able to relocate WHILE this continuation is live or
    // captured, so — exactly like a lexical `Frame` slot — each is an `AtomicU64` the
    // collector rewrites in place. That is what lets a captured continuation
    // survive a collection (see `gc::walk_kont`).
    CallK { pending: Vec<Ir>, done: Vec<AtomicU64>, env: Locals, next: Arc<Kont> },
    PrimK { op: Prim, pending: Vec<Ir>, done: Vec<AtomicU64>, env: Locals, next: Arc<Kont> },
    CallCc { next: Arc<Kont> },
    LetSlot { inits: Vec<Ir>, i: usize, frame: Locals, body: Ir, next: Arc<Kont> },
    /// A continuation delimiter, installed by `%reset`. Marks how far a `%shift`
    /// capture reaches; when a value flows into it, the prompt is transparent
    /// (the value passes straight through to `next`).
    Prompt { next: Arc<Kont> },
    /// Awaiting the `%shift` receiver closure `f`; once it is a value, capture the
    /// slice up to the nearest `Prompt` and apply `f` to the reified continuation.
    ShiftK { next: Arc<Kont> },
}

/// Snapshot a `done` vector into fresh atomics (each `Kont` is immutable and may
/// be shared across multi-shot resumptions, so a step that extends `done` must
/// not mutate the shared prefix).
fn snapshot(done: &[AtomicU64]) -> Vec<AtomicU64> {
    clone_slots(done)
}

enum Step {
    Eval(Ir, Locals, Arc<Kont>),
    Apply(u64, Arc<Kont>),
}

pub struct CekMachine;

impl<M: ValueModel> CodeSpace<M> for CekMachine {
    fn eval_ir(&self, _top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64 {
        run(rt, Step::Eval(ir.clone(), locals.clone(), Arc::new(Kont::Done)))
    }
    fn invoke(&self, _top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        let step = apply_callable(rt, callee, args, Arc::new(Kont::Done));
        run(rt, step)
    }
}

fn run<M: ValueModel>(rt: &mut Runtime<M>, mut step: Step) -> u64 {
    loop {
        step = match step {
            Step::Eval(ir, env, k) => eval_step::<M>(rt, ir, env, k),
            Step::Apply(v, k) => {
                if matches!(&*k, Kont::Done) {
                    return v;
                }
                apply_step::<M>(rt, v, &k)
            }
        };
    }
}

fn eval_step<M: ValueModel>(rt: &mut Runtime<M>, ir: Ir, env: Locals, k: Arc<Kont>) -> Step {
    match ir {
        Ir::Const(id) | Ir::Quote(id) => Step::Apply(rt.get_const(id), k),
        Ir::Local { up, idx } => Step::Apply(frame_get(&env, up, idx), k),
        Ir::Global(s) => {
            let v = rt
                .global(s)
                .unwrap_or_else(|| panic!("Unable to resolve symbol: {}", rt.sym_name(s)));
            Step::Apply(v, k)
        }
        Ir::Lambda { nparams, variadic, body } => {
            let id = rt.alloc(Obj::Closure { nparams, variadic, body, env: env.clone() });
            Step::Apply(M::R::enc_ref(id), k)
        }
        Ir::If(c, t, e) => {
            Step::Eval(*c, env.clone(), Arc::new(Kont::If { then_: *t, else_: *e, env, next: k }))
        }
        Ir::Do(xs) => {
            let mut it = xs.into_iter();
            match it.next() {
                None => Step::Apply(M::R::enc_nil(), k),
                Some(first) => {
                    let rest: Vec<Ir> = it.collect();
                    Step::Eval(first, env.clone(), Arc::new(Kont::Seq { rest, env, next: k }))
                }
            }
        }
        Ir::Def { name, init } => {
            Step::Eval(*init, env, Arc::new(Kont::Def { name, next: k }))
        }
        Ir::SetLocal { up, idx, val } => {
            Step::Eval(*val, env.clone(), Arc::new(Kont::SetLoc { up, idx, env, next: k }))
        }
        Ir::SetGlobal { name, val } => {
            Step::Eval(*val, env, Arc::new(Kont::SetGlob { name, next: k }))
        }
        Ir::Call(f, args) => {
            Step::Eval(*f, env.clone(), Arc::new(Kont::CallK { pending: args, done: Vec::new(), env, next: k }))
        }
        Ir::Prim(Prim::CallCc, mut args) => {
            let f = args.remove(0);
            Step::Eval(f, env, Arc::new(Kont::CallCc { next: k }))
        }
        // `%reset body` — evaluate `body` under a fresh prompt delimiter.
        Ir::Prim(Prim::Reset, mut args) => {
            let body = args.remove(0);
            Step::Eval(body, env, Arc::new(Kont::Prompt { next: k }))
        }
        // `%shift f` — evaluate the receiver `f`; the capture happens once it is a
        // value (see the `ShiftK` arm of `apply_step`).
        Ir::Prim(Prim::Shift, mut args) => {
            let f = args.remove(0);
            Step::Eval(f, env, Arc::new(Kont::ShiftK { next: k }))
        }
        // A GC safepoint that composes with captured continuations: the live
        // continuation `k` (its `done` cells and captured frames) is rooted and
        // relocated alongside the environment. This is the fusion of the moving
        // GC and the full-continuation execution tier.
        Ir::Prim(Prim::Gc, _) => {
            rt.collect_cek(&env, &k);
            Step::Apply(M::R::enc_nil(), k)
        }
        Ir::Prim(op, args) => {
            let mut it = args.into_iter();
            match it.next() {
                None => Step::Apply(rt.prim(op, &[]), k),
                Some(first) => {
                    let rest: Vec<Ir> = it.collect();
                    Step::Eval(first, env.clone(), Arc::new(Kont::PrimK { op, pending: rest, done: Vec::new(), env, next: k }))
                }
            }
        }
        Ir::Let(inits, body) => {
            let n = inits.len();
            let frame: Locals = Some(Arc::new(Frame {
                slots: (0..n).map(|_| AtomicU64::new(M::R::enc_nil())).collect(),
                parent: env,
            }));
            if n == 0 {
                Step::Eval(*body, frame, k)
            } else {
                let first = inits[0].clone();
                Step::Eval(first, frame.clone(), Arc::new(Kont::LetSlot { inits, i: 0, frame, body: *body, next: k }))
            }
        }
        Ir::Dispatch { .. } | Ir::DefMethod { .. } | Ir::FieldGet { .. } => {
            panic!("CekMachine: dispatch not supported; run on the tree-walker")
        }
        Ir::Try { .. } => panic!("CekMachine: try/catch not supported; run on the tree-walker"),
    }
}

fn apply_step<M: ValueModel>(rt: &mut Runtime<M>, v: u64, k: &Arc<Kont>) -> Step {
    match &**k {
        Kont::Done => Step::Apply(v, k.clone()), // handled in run; unreachable
        Kont::If { then_, else_, env, next } => {
            let branch = if M::truthy(rt.decode(v)) { then_.clone() } else { else_.clone() };
            Step::Eval(branch, env.clone(), next.clone())
        }
        Kont::Seq { rest, env, next } => {
            if rest.is_empty() {
                Step::Apply(v, next.clone())
            } else {
                let first = rest[0].clone();
                let more = rest[1..].to_vec();
                Step::Eval(first, env.clone(), Arc::new(Kont::Seq { rest: more, env: env.clone(), next: next.clone() }))
            }
        }
        Kont::Def { name, next } => {
            rt.define_global(*name, v);
            Step::Apply(M::R::enc_sym(*name), next.clone())
        }
        Kont::SetLoc { up, idx, env, next } => {
            frame_set(env, *up, *idx, v);
            Step::Apply(v, next.clone())
        }
        Kont::SetGlob { name, next } => {
            if !rt.set_global_val(*name, v) {
                panic!("set!: unbound variable: {}", rt.sym_name(*name));
            }
            Step::Apply(v, next.clone())
        }
        Kont::CallK { pending, done, env, next } => {
            let mut done2 = snapshot(done);
            done2.push(AtomicU64::new(v));
            if pending.is_empty() {
                let vals: Vec<u64> = done2.iter().map(slot_load).collect();
                apply_callable(rt, vals[0], &vals[1..], next.clone())
            } else {
                let first = pending[0].clone();
                let more = pending[1..].to_vec();
                Step::Eval(first, env.clone(), Arc::new(Kont::CallK { pending: more, done: done2, env: env.clone(), next: next.clone() }))
            }
        }
        Kont::PrimK { op, pending, done, env, next } => {
            let mut done2 = snapshot(done);
            done2.push(AtomicU64::new(v));
            if pending.is_empty() {
                let vals: Vec<u64> = done2.iter().map(slot_load).collect();
                // `apply` invokes a closure, which the generic `rt.prim` cannot
                // do — flatten `(apply f a … lst)` and dispatch here instead.
                if let Prim::Apply = op {
                    let (f, rest) = (vals[0], &vals[1..]);
                    let mut flat: Vec<u64> = rest[..rest.len().saturating_sub(1)].to_vec();
                    if let Some(&last) = rest.last() {
                        flat.extend(rt.list_to_vec(last));
                    }
                    return apply_callable(rt, f, &flat, next.clone());
                }
                let r = rt.prim(*op, &vals);
                Step::Apply(r, next.clone())
            } else {
                let first = pending[0].clone();
                let more = pending[1..].to_vec();
                Step::Eval(first, env.clone(), Arc::new(Kont::PrimK { op: *op, pending: more, done: done2, env: env.clone(), next: next.clone() }))
            }
        }
        // A value reaching a prompt: the delimiter is transparent to normal
        // return — pass it straight through.
        Kont::Prompt { next } => Step::Apply(v, next.clone()),
        // `%shift`'s receiver is now the value `v`. Capture the continuation
        // slice up to the nearest enclosing prompt, reify it as a composable
        // `PartialCont`, and run `v` applied to it under a re-established prompt
        // (the outer `reset` persists around the shift body).
        Kont::ShiftK { next } => {
            let below = prompt_tail(next);
            let cont_id = rt.alloc(Obj::PartialCont(next.clone()));
            let cref = M::R::enc_ref(cont_id);
            apply_callable(rt, v, &[cref], Arc::new(Kont::Prompt { next: below }))
        }
        Kont::CallCc { next } => {
            // v is the closure `f`. Reify `next` as a first-class continuation
            // and apply f to it.
            let cont_id = rt.alloc(Obj::Cont(next.clone()));
            let cref = M::R::enc_ref(cont_id);
            apply_callable(rt, v, &[cref], next.clone())
        }
        Kont::LetSlot { inits, i, frame, body, next } => {
            if let Some(f) = frame {
                crate::value::slot_store(&f.slots[*i], v);
            }
            let ni = i + 1;
            if ni >= inits.len() {
                Step::Eval(body.clone(), frame.clone(), next.clone())
            } else {
                let init = inits[ni].clone();
                Step::Eval(init, frame.clone(), Arc::new(Kont::LetSlot { inits: inits.clone(), i: ni, frame: frame.clone(), body: body.clone(), next: next.clone() }))
            }
        }
    }
}

/// Apply a callable to args under continuation `next`. A closure evaluates its
/// body; a continuation RE-INSTALLS itself with the argument (the multi-shot
/// jump — it discards `next` and resumes the captured `Kont`).
fn apply_callable<M: ValueModel>(rt: &mut Runtime<M>, callee: u64, args: &[u64], next: Arc<Kont>) -> Step {
    let Val::Ref(id) = rt.decode(callee) else {
        panic!("value not callable: {}", rt.print(callee));
    };
    enum What {
        Closure(usize, bool, Arc<Ir>, Locals),
        Cont(Arc<Kont>),
        PartialCont(Arc<Kont>),
        Bad,
    }
    let what = match &rt.heap()[id as usize] {
        Obj::Closure { nparams, variadic, body, env } => {
            What::Closure(*nparams, *variadic, body.clone(), env.clone())
        }
        Obj::Cont(c) => What::Cont(c.clone()),
        Obj::PartialCont(c) => What::PartialCont(c.clone()),
        _ => What::Bad,
    };
    match what {
        What::Closure(nparams, variadic, body, env) => {
            let frame = rt.build_call_frame(nparams, variadic, args, env);
            Step::Eval((*body).clone(), frame, next)
        }
        What::Cont(captured) => {
            let v = args.first().copied().unwrap_or_else(M::R::enc_nil);
            Step::Apply(v, captured) // jump: re-install the captured continuation
        }
        What::PartialCont(captured) => {
            // Composable: graft the captured slice ONTO the caller's continuation
            // `next` under a fresh prompt (so it is re-delimited), then feed it the
            // argument. It returns to `next` when the slice finishes — it does not
            // abort. Immutable, so this works any number of times (multi-shot).
            let v = args.first().copied().unwrap_or_else(M::R::enc_nil);
            let grafted = regraft(&captured, next);
            Step::Apply(v, grafted)
        }
        What::Bad => panic!("value not callable: {}", rt.print(callee)),
    }
}

/// The continuation below the nearest enclosing prompt (the `%reset`'s own
/// continuation). Panics if there is no enclosing prompt (a `%shift` with no
/// dynamically-enclosing `%reset`).
fn prompt_tail(k: &Arc<Kont>) -> Arc<Kont> {
    let mut cur = k.clone();
    loop {
        let next = match &*cur {
            Kont::Prompt { next } => return next.clone(),
            Kont::Done => panic!("%shift: no enclosing %reset"),
            other => kont_next(other)
                .expect("Done handled above")
                .clone(),
        };
        cur = next;
    }
}

/// The `next` link of a frame (every frame but `Done` has one).
fn kont_next(k: &Kont) -> Option<&Arc<Kont>> {
    match k {
        Kont::Done => None,
        Kont::If { next, .. }
        | Kont::Seq { next, .. }
        | Kont::Def { next, .. }
        | Kont::SetLoc { next, .. }
        | Kont::SetGlob { next, .. }
        | Kont::CallK { next, .. }
        | Kont::PrimK { next, .. }
        | Kont::CallCc { next, .. }
        | Kont::LetSlot { next, .. }
        | Kont::Prompt { next, .. }
        | Kont::ShiftK { next, .. } => Some(next),
    }
}

/// Copy the frames of `k` above the nearest prompt, re-terminating the copy with
/// a fresh `Prompt { next: new_tail }`. This IS the composable graft: the
/// captured delimited slice, followed by a fresh delimiter, followed by the
/// caller's continuation. Frames are immutable `Arc`, so the original is untouched
/// and reusable (multi-shot).
fn regraft(k: &Arc<Kont>, new_tail: Arc<Kont>) -> Arc<Kont> {
    match &**k {
        Kont::Prompt { .. } => Arc::new(Kont::Prompt { next: new_tail }),
        Kont::Done => panic!("%shift: captured slice had no prompt boundary"),
        Kont::If { then_, else_, env, next } => Arc::new(Kont::If {
            then_: then_.clone(), else_: else_.clone(), env: env.clone(), next: regraft(next, new_tail),
        }),
        Kont::Seq { rest, env, next } => Arc::new(Kont::Seq {
            rest: rest.clone(), env: env.clone(), next: regraft(next, new_tail),
        }),
        Kont::Def { name, next } => Arc::new(Kont::Def {
            name: *name, next: regraft(next, new_tail),
        }),
        Kont::SetLoc { up, idx, env, next } => Arc::new(Kont::SetLoc {
            up: *up, idx: *idx, env: env.clone(), next: regraft(next, new_tail),
        }),
        Kont::SetGlob { name, next } => Arc::new(Kont::SetGlob {
            name: *name, next: regraft(next, new_tail),
        }),
        Kont::CallK { pending, done, env, next } => Arc::new(Kont::CallK {
            pending: pending.clone(), done: snapshot(done), env: env.clone(), next: regraft(next, new_tail),
        }),
        Kont::PrimK { op, pending, done, env, next } => Arc::new(Kont::PrimK {
            op: *op, pending: pending.clone(), done: snapshot(done), env: env.clone(), next: regraft(next, new_tail),
        }),
        Kont::CallCc { next } => Arc::new(Kont::CallCc { next: regraft(next, new_tail) }),
        Kont::ShiftK { next } => Arc::new(Kont::ShiftK { next: regraft(next, new_tail) }),
        Kont::LetSlot { inits, i, frame, body, next } => Arc::new(Kont::LetSlot {
            inits: inits.clone(), i: *i, frame: frame.clone(), body: body.clone(), next: regraft(next, new_tail),
        }),
    }
}
