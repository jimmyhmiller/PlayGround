//! The execution axis: `CodeSpace`.
//!
//! `CodeSpace` decouples *what your language means* (the `Ir` it consumes) from
//! *how it runs*. Two design choices make backends genuinely composable, not
//! just swappable:
//!
//!   1. The backend is a **value you pass**, not a type parameter baked into
//!      `Runtime`. `Runtime<M>` has no backend in its type, so one runtime can
//!      be handed to different backends, and a wrapping backend can hold
//!      another as a `Box<dyn CodeSpace<M>>`. Under the old `Runtime<M, C>`
//!      design that did not typecheck.
//!
//!   2. **Open recursion.** Every method also receives `top`, the OUTERMOST
//!      backend, and recurses through `top` rather than `self`. Without this a
//!      wrapper only sees the calls `Runtime` initiates; the inner backend's
//!      own nested calls bypass it. Threading `top` makes composition total —
//!      `Traced` below counts *every* call, including ones made deep inside
//!      `eval_ir`. This is the fixpoint that stops "almost composes" from
//!      rotting into a hardcode.
//!
//! `TreeWalk` is the interpreter tier: it exists from form zero and boots a
//! bootstrapping language. A JIT tier is another `impl` over the same `Ir` and
//! the same re-entrant contract; mutable JIT state lives behind interior
//! mutability so `&self` still holds.

use std::cell::Cell;
use std::rc::Rc;

use crate::ir::Ir;
use crate::model::{Repr, ValueModel};
use crate::runtime::{Runtime, Var};
use crate::value::{frame_get, Frame, Locals, Obj, Val};

pub trait CodeSpace<M: ValueModel> {
    /// Evaluate one IR node. `top` is the outermost backend to recurse through.
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals)
        -> u64;

    /// Call a callable value. For macros the args are unevaluated forms.
    /// Re-entrant: `Runtime::macroexpand` calls this mid-compilation.
    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64;
}

/// The interpreter tier.
pub struct TreeWalk;

impl<M: ValueModel> CodeSpace<M> for TreeWalk {
    fn eval_ir(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        ir: &Ir,
        locals: &Locals,
    ) -> u64 {
        match ir {
            Ir::Const(b) | Ir::Quote(b) => *b,
            Ir::Local { up, idx } => frame_get(locals, *up, *idx),
            Ir::Global(s) => match rt.globals.get(s) {
                Some(v) => v.val,
                None => panic!("Unable to resolve symbol: {}", rt.sym_name(*s)),
            },
            Ir::If(c, t, e) => {
                let cv = top.eval_ir(top, rt, c, locals);
                if M::truthy(rt.decode(cv)) {
                    top.eval_ir(top, rt, t, locals)
                } else {
                    top.eval_ir(top, rt, e, locals)
                }
            }
            Ir::Do(xs) => {
                let mut r = rt.encode(Val::Nil);
                for x in xs {
                    r = top.eval_ir(top, rt, x, locals);
                }
                r
            }
            Ir::Def {
                name,
                init,
                is_macro,
            } => {
                let v = top.eval_ir(top, rt, init, locals);
                rt.globals.insert(
                    *name,
                    Var {
                        val: v,
                        is_macro: *is_macro,
                    },
                );
                rt.encode(Val::Sym(*name))
            }
            Ir::Let(inits, body) => {
                // The let is one pushed frame (matching `analyze`), so even the
                // first init evaluates with the let frame already innermost.
                let mut slots: Vec<u64> = Vec::new();
                let mut cur: Locals = Some(Rc::new(Frame {
                    slots: Vec::new(),
                    parent: locals.clone(),
                }));
                for iexpr in inits {
                    let v = top.eval_ir(top, rt, iexpr, &cur);
                    slots.push(v);
                    cur = Some(Rc::new(Frame {
                        slots: slots.clone(),
                        parent: locals.clone(),
                    }));
                }
                top.eval_ir(top, rt, body, &cur)
            }
            Ir::Lambda {
                nparams,
                variadic,
                body,
            } => {
                let id = rt.alloc(Obj::Closure {
                    nparams: *nparams,
                    variadic: *variadic,
                    body: body.clone(),
                    env: locals.clone(),
                });
                M::R::enc_ref(id)
            }
            Ir::Call(f, args) => {
                let fv = top.eval_ir(top, rt, f, locals);
                let argv: Vec<u64> = args
                    .iter()
                    .map(|a| top.eval_ir(top, rt, a, locals))
                    .collect();
                top.invoke(top, rt, fv, &argv)
            }
            Ir::Prim(op, args) => {
                let argv: Vec<u64> = args
                    .iter()
                    .map(|a| top.eval_ir(top, rt, a, locals))
                    .collect();
                rt.prim(*op, &argv)
            }
        }
    }

    fn invoke(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        callee: u64,
        args: &[u64],
    ) -> u64 {
        let Val::Ref(id) = rt.decode(callee) else {
            panic!("value not callable: {}", rt.print(callee));
        };
        let (nparams, variadic, body, env) = match &rt.heap[id as usize] {
            Obj::Closure {
                nparams,
                variadic,
                body,
                env,
            } => (*nparams, *variadic, body.clone(), env.clone()),
            _ => panic!("value not callable: {}", rt.print(callee)),
        };
        let frame = rt.build_call_frame(nparams, variadic, args, env);
        top.eval_ir(top, rt, &body, &frame)
    }
}

/// A wrapping backend: instruments any inner `CodeSpace` and delegates. Because
/// it passes `top` (itself) down, the inner backend recurses back through it,
/// so it observes EVERY call — runtime and macro-expansion, top-level and
/// deeply nested. That totality is the proof the seam composes.
pub struct Traced<M: ValueModel> {
    inner: Box<dyn CodeSpace<M>>,
    invokes: Cell<u64>,
}

impl<M: ValueModel> Traced<M> {
    pub fn new(inner: impl CodeSpace<M> + 'static) -> Self {
        Traced {
            inner: Box::new(inner),
            invokes: Cell::new(0),
        }
    }
    /// How many calls (runtime + macro-expansion, at every depth) flowed
    /// through the wrapper.
    pub fn invoke_count(&self) -> u64 {
        self.invokes.get()
    }
}

impl<M: ValueModel> CodeSpace<M> for Traced<M> {
    fn eval_ir(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        ir: &Ir,
        locals: &Locals,
    ) -> u64 {
        self.inner.eval_ir(top, rt, ir, locals)
    }
    fn invoke(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        callee: u64,
        args: &[u64],
    ) -> u64 {
        self.invokes.set(self.invokes.get() + 1);
        self.inner.invoke(top, rt, callee, args)
    }
}
