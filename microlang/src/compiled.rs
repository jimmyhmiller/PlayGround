//! A second, structurally different execution tier: a closure-generation
//! compiler.
//!
//! Where `TreeWalk` re-dispatches on the `Ir` enum every time a node runs,
//! `ClosureComp` compiles each `Ir` subtree ONCE into a Rust closure
//! (`Compiled`) and thereafter just calls it. Function bodies are compiled
//! lazily on first `invoke` and cached by body identity, so a function called a
//! million times is compiled once. This is the "one rung up the perf ladder"
//! story made real: a different strategy behind the SAME `CodeSpace` contract,
//! the SAME `Ir`, and the SAME re-entrant `invoke`.
//!
//! It validates two properties the contract promised:
//!
//!   * **Incremental / late-bound.** Compilation captures only a symbol for a
//!     global reference; resolution happens at call time through `rt.globals`.
//!     So a compiled function can call one defined *later* (mutual recursion
//!     with forward references works — see the tests).
//!
//!   * **Composition via `top`.** Compiled call sites invoke through `top`, not
//!     `self`, so `Traced(ClosureComp)` still observes every call. Node-level
//!     structure is legitimately erased by compilation; the contract intercepts
//!     at the semantic boundary (calls), which survives.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::code::CodeSpace;
use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::{Runtime, Var};
use crate::value::{frame_get, frame_set, Frame, Locals, Obj, Val};

/// A compiled expression: run it with the runtime, a lexical frame, and the
/// outermost backend (for re-entrant, composable calls). Captures only owned
/// data, so it is `'static`.
type Compiled<M> = Rc<dyn Fn(&mut Runtime<M>, &Locals, &dyn CodeSpace<M>) -> u64>;

pub struct ClosureComp<M: ValueModel> {
    /// Compiled function bodies, keyed by body identity (`Rc<Ir>` address).
    /// A real tier would key by a stable function id; the heap here never frees
    /// (leak GC), so the pointer is stable for the sketch.
    cache: RefCell<HashMap<*const Ir, Compiled<M>>>,
    compiles: Cell<u64>,
}

impl<M: ValueModel> ClosureComp<M> {
    pub fn new() -> Self {
        ClosureComp {
            cache: RefCell::new(HashMap::new()),
            compiles: Cell::new(0),
        }
    }

    /// Distinct function bodies compiled so far. Repeated calls do not grow it.
    pub fn compiled_bodies(&self) -> usize {
        self.cache.borrow().len()
    }
    /// Total compile actions (including one-shot top-level expressions).
    pub fn compile_count(&self) -> u64 {
        self.compiles.get()
    }

    fn compile(&self, ir: &Ir) -> Compiled<M> {
        self.compiles.set(self.compiles.get() + 1);
        match ir {
            Ir::Const(id) | Ir::Quote(id) => {
                let id = *id;
                Rc::new(move |rt, _env, _top| rt.get_const(id))
            }
            Ir::Local { up, idx } => {
                let (up, idx) = (*up, *idx);
                Rc::new(move |_rt, env, _top| frame_get(env, up, idx))
            }
            Ir::Global(s) => {
                let s = *s;
                // Late binding: resolved at call time, not compile time.
                Rc::new(move |rt, _env, _top| match rt.globals.get(&s) {
                    Some(v) => v.val,
                    None => panic!("Unable to resolve symbol: {}", rt.sym_name(s)),
                })
            }
            Ir::SetLocal { up, idx, val } => {
                let (up, idx) = (*up, *idx);
                let cv = self.compile(val);
                Rc::new(move |rt, env, top| {
                    let v = cv(rt, env, top);
                    frame_set(env, up, idx, v);
                    v
                })
            }
            Ir::SetGlobal { name, val } => {
                let name = *name;
                let cv = self.compile(val);
                Rc::new(move |rt, env, top| {
                    let v = cv(rt, env, top);
                    match rt.globals.get_mut(&name) {
                        Some(var) => var.val = v,
                        None => panic!("set!: unbound variable: {}", rt.sym_name(name)),
                    }
                    v
                })
            }
            Ir::If(c, t, e) => {
                let cc = self.compile(c);
                let ct = self.compile(t);
                let ce = self.compile(e);
                Rc::new(move |rt, env, top| {
                    let cv = cc(rt, env, top);
                    if M::truthy(rt.decode(cv)) {
                        ct(rt, env, top)
                    } else {
                        ce(rt, env, top)
                    }
                })
            }
            Ir::Do(xs) => {
                let cs: Vec<Compiled<M>> = xs.iter().map(|x| self.compile(x)).collect();
                Rc::new(move |rt, env, top| {
                    let mut r = rt.encode(Val::Nil);
                    for c in &cs {
                        r = c(rt, env, top);
                    }
                    r
                })
            }
            Ir::Def { name, init } => {
                let name = *name;
                let ci = self.compile(init);
                Rc::new(move |rt, env, top| {
                    let v = ci(rt, env, top);
                    rt.globals.insert(name, Var { val: v });
                    rt.encode(Val::Sym(name))
                })
            }
            Ir::Let(inits, body) => {
                let cinits: Vec<Compiled<M>> = inits.iter().map(|ie| self.compile(ie)).collect();
                let cbody = self.compile(body);
                Rc::new(move |rt, env, top| {
                    // Rebuild each frame from the previous frame's current cell
                    // values, so a GC during an init keeps earlier bindings sound.
                    let mut cur: Locals = Some(Rc::new(Frame {
                        slots: Vec::new(),
                        parent: env.clone(),
                    }));
                    for ci in &cinits {
                        let v = ci(rt, &cur, top);
                        let prev = cur.as_ref().unwrap();
                        let mut slots: Vec<Cell<u64>> =
                            prev.slots.iter().map(|c| Cell::new(c.get())).collect();
                        slots.push(Cell::new(v));
                        cur = Some(Rc::new(Frame {
                            slots,
                            parent: env.clone(),
                        }));
                    }
                    cbody(rt, &cur, top)
                })
            }
            Ir::Lambda {
                nparams,
                variadic,
                body,
            } => {
                let nparams = *nparams;
                let variadic = *variadic;
                let body = body.clone();
                Rc::new(move |rt, env, _top| {
                    let id = rt.alloc(Obj::Closure {
                        nparams,
                        variadic,
                        body: body.clone(),
                        env: env.clone(),
                    });
                    M::R::enc_ref(id)
                })
            }
            Ir::Call(f, args) => {
                let cf = self.compile(f);
                let cargs: Vec<Compiled<M>> = args.iter().map(|a| self.compile(a)).collect();
                Rc::new(move |rt, env, top| {
                    let callee = cf(rt, env, top);
                    let mut argv: Vec<u64> = Vec::with_capacity(cargs.len());
                    for c in &cargs {
                        argv.push(c(rt, env, top));
                    }
                    // through `top`: composable + late-bound dispatch
                    top.invoke(top, rt, callee, &argv)
                })
            }
            Ir::Prim(Prim::Gc, _) => {
                // Safepoint: pass the live env (this closure's `env`) to collect.
                Rc::new(move |rt: &mut Runtime<M>, env: &Locals, _top| {
                    rt.collect(env);
                    rt.encode(Val::Nil)
                })
            }
            Ir::Prim(op, args) => {
                let op = *op;
                let cargs: Vec<Compiled<M>> = args.iter().map(|a| self.compile(a)).collect();
                Rc::new(move |rt, env, top| {
                    let mut argv: Vec<u64> = Vec::with_capacity(cargs.len());
                    for c in &cargs {
                        argv.push(c(rt, env, top));
                    }
                    rt.prim(op, &argv)
                })
            }
            Ir::DefMethod { name, ty, imp } => {
                let (name, ty) = (*name, *ty);
                let ci = self.compile(imp);
                Rc::new(move |rt, env, top| {
                    let c = ci(rt, env, top);
                    rt.register_method(name, ty, c);
                    rt.encode(Val::Nil)
                })
            }
            Ir::Dispatch { site, method, args } => {
                let (site, method) = (*site, *method);
                let cargs: Vec<Compiled<M>> = args.iter().map(|a| self.compile(a)).collect();
                Rc::new(move |rt, env, top| {
                    let mut argv: Vec<u64> = Vec::with_capacity(cargs.len());
                    for c in &cargs {
                        argv.push(c(rt, env, top));
                    }
                    let ty = rt
                        .type_of(argv[0])
                        .unwrap_or_else(|| panic!("dispatch: receiver is not a record"));
                    let imp = rt.resolve_method(site, method, ty).unwrap_or_else(|| {
                        panic!(
                            "no method '{}' for type '{}'",
                            rt.sym_name(method),
                            rt.sym_name(ty)
                        )
                    });
                    top.invoke(top, rt, imp, &argv)
                })
            }
        }
    }

    fn compiled_body(&self, body: &Rc<Ir>) -> Compiled<M> {
        let key = Rc::as_ptr(body);
        if let Some(c) = self.cache.borrow().get(&key) {
            return c.clone();
        }
        let c = self.compile(body);
        self.cache.borrow_mut().insert(key, c.clone());
        c
    }
}

impl<M: ValueModel> CodeSpace<M> for ClosureComp<M> {
    fn eval_ir(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        ir: &Ir,
        locals: &Locals,
    ) -> u64 {
        // Top-level forms are one-shot: compile then run.
        let c = self.compile(ir);
        c(rt, locals, top)
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

        // Compile-once: cached by body identity across all calls.
        let compiled = self.compiled_body(&body);
        let frame = rt.build_call_frame(nparams, variadic, args, env);
        compiled(rt, &frame, top)
    }
}
