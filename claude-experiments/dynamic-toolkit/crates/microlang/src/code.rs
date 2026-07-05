//! The execution axis: `CodeSpace`.
//!
//! `CodeSpace` is the seam that decouples *what your language means* (the `Ir`
//! it consumes) from *how it runs*. `TreeWalk` below is the interpreter tier:
//! it exists from form zero, needs no compiler backend, and is what boots a
//! bootstrapping language (you must be able to run macros while still
//! compiling). A JIT tier would be a second `impl CodeSpace` over the same
//! `Ir`, with the SAME `invoke` contract — the crucial hard part being that
//! its `define`/`invoke` must be incremental and support calling functions not
//! yet defined (late binding through Vars). Nothing above this trait changes
//! when you swap tiers.
//!
//! `invoke` is deliberately re-entrant: `Runtime::macroexpand` calls it in the
//! middle of `analyze`. In a tree-walker that is just a recursive Rust call;
//! in a JIT it is "enter native code, which may re-enter the compiler." Both
//! satisfy one contract.

use std::rc::Rc;

use crate::ir::Ir;
use crate::model::{Repr, ValueModel};
use crate::runtime::{Runtime, Var};
use crate::value::{frame_lookup, Frame, Locals, Obj, Val};

pub trait CodeSpace<M: ValueModel>: Sized {
    /// Evaluate one IR node under a lexical frame.
    fn eval_ir(rt: &mut Runtime<M, Self>, ir: &Ir, locals: &Locals) -> u64;

    /// Call a callable value with already-evaluated (or, for macros,
    /// unevaluated) argument bits. Re-entrant.
    fn invoke(rt: &mut Runtime<M, Self>, callee: u64, args: &[u64]) -> u64;
}

/// The interpreter tier.
pub struct TreeWalk;

impl<M: ValueModel> CodeSpace<M> for TreeWalk {
    fn eval_ir(rt: &mut Runtime<M, Self>, ir: &Ir, locals: &Locals) -> u64 {
        match ir {
            Ir::Const(b) | Ir::Quote(b) => *b,
            Ir::Var(s) => {
                if let Some(v) = frame_lookup(locals, *s) {
                    return v;
                }
                match rt.globals.get(s) {
                    Some(v) => v.val,
                    None => panic!("Unable to resolve symbol: {}", rt.sym_name(*s)),
                }
            }
            Ir::If(c, t, e) => {
                let cv = Self::eval_ir(rt, c, locals);
                if M::truthy(rt.decode(cv)) {
                    Self::eval_ir(rt, t, locals)
                } else {
                    Self::eval_ir(rt, e, locals)
                }
            }
            Ir::Do(xs) => {
                let mut r = rt.encode(Val::Nil);
                for x in xs {
                    r = Self::eval_ir(rt, x, locals);
                }
                r
            }
            Ir::Def {
                name,
                init,
                is_macro,
            } => {
                let v = Self::eval_ir(rt, init, locals);
                rt.globals.insert(
                    *name,
                    Var {
                        val: v,
                        is_macro: *is_macro,
                    },
                );
                rt.encode(Val::Sym(*name))
            }
            Ir::Let(binds, body) => {
                let mut vars: Vec<(u32, u64)> = Vec::new();
                let mut cur: Locals = locals.clone();
                for (s, iexpr) in binds {
                    let v = Self::eval_ir(rt, iexpr, &cur);
                    vars.push((*s, v));
                    cur = Some(Rc::new(Frame {
                        vars: vars.clone(),
                        parent: locals.clone(),
                    }));
                }
                Self::eval_ir(rt, body, &cur)
            }
            Ir::Lambda {
                params,
                variadic,
                body,
            } => {
                let id = rt.alloc(Obj::Closure {
                    params: params.clone(),
                    variadic: *variadic,
                    body: body.clone(),
                    env: locals.clone(),
                });
                M::R::enc_ref(id)
            }
            Ir::Call(f, args) => {
                let fv = Self::eval_ir(rt, f, locals);
                let argv: Vec<u64> = args.iter().map(|a| Self::eval_ir(rt, a, locals)).collect();
                Self::invoke(rt, fv, &argv)
            }
            Ir::Prim(op, args) => {
                let argv: Vec<u64> = args.iter().map(|a| Self::eval_ir(rt, a, locals)).collect();
                rt.prim(*op, &argv)
            }
        }
    }

    fn invoke(rt: &mut Runtime<M, Self>, callee: u64, args: &[u64]) -> u64 {
        let Val::Ref(id) = rt.decode(callee) else {
            panic!("value not callable: {}", rt.print(callee));
        };
        let (params, variadic, body, env) = match &rt.heap[id as usize] {
            Obj::Closure {
                params,
                variadic,
                body,
                env,
            } => (params.clone(), *variadic, body.clone(), env.clone()),
            _ => panic!("value not callable: {}", rt.print(callee)),
        };

        let mut vars: Vec<(u32, u64)> = Vec::new();
        match variadic {
            Some(rest) => {
                assert!(
                    args.len() >= params.len(),
                    "arity: expected at least {}, got {}",
                    params.len(),
                    args.len()
                );
                for (i, p) in params.iter().enumerate() {
                    vars.push((*p, args[i]));
                }
                let restlist = rt.vec_to_list(&args[params.len()..]);
                vars.push((rest, restlist));
            }
            None => {
                assert!(
                    args.len() == params.len(),
                    "arity: expected {}, got {}",
                    params.len(),
                    args.len()
                );
                for (i, p) in params.iter().enumerate() {
                    vars.push((*p, args[i]));
                }
            }
        }
        let frame = Some(Rc::new(Frame { vars, parent: env }));
        Self::eval_ir(rt, &body, &frame)
    }
}
