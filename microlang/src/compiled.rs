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
use std::sync::Arc;

use crate::code::CodeSpace;
use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::{ObjView, Runtime};
use crate::value::{build_caps, frame_cap, frame_get, frame_set, Locals, Obj, Val};

/// A compiled expression: run it with the runtime, a lexical frame, and the
/// outermost backend (for re-entrant, composable calls). Captures only owned
/// data, so it is `'static`.
type Compiled<M> = Arc<dyn Fn(&mut Runtime<M>, &Locals, &dyn CodeSpace<M>) -> u64>;

pub struct ClosureComp<M: ValueModel> {
    /// Compiled function bodies, keyed by body identity (`Arc<Ir>` address).
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
                Arc::new(move |rt, _env, _top| rt.get_const(id))
            }
            Ir::Local { up, idx } => {
                let (up, idx) = (*up, *idx);
                Arc::new(move |_rt, env, _top| frame_get(env, up, idx))
            }
            Ir::Capture(idx) => {
                let idx = *idx;
                Arc::new(move |_rt, env, _top| frame_cap::<M::R>(env, idx))
            }
            Ir::Global(s) => {
                let s = *s;
                // Late binding: resolved at call time, not compile time.
                Arc::new(move |rt, _env, _top| match rt.global(s) {
                    Some(v) => v,
                    None => panic!("Unable to resolve symbol: {}", rt.sym_name(s)),
                })
            }
            Ir::SetLocal { up, idx, val } => {
                let (up, idx) = (*up, *idx);
                let cv = self.compile(val);
                Arc::new(move |rt, env, top| {
                    let v = cv(rt, env, top);
                    frame_set(env, up, idx, v);
                    v
                })
            }
            Ir::SetGlobal { name, val } => {
                let name = *name;
                let cv = self.compile(val);
                Arc::new(move |rt, env, top| {
                    let v = cv(rt, env, top);
                    if !rt.set_global_val(name, v) {
                        panic!("set!: unbound variable: {}", rt.sym_name(name));
                    }
                    v
                })
            }
            Ir::If(c, t, e) => {
                let cc = self.compile(c);
                let ct = self.compile(t);
                let ce = self.compile(e);
                Arc::new(move |rt, env, top| {
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
                Arc::new(move |rt, env, top| {
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
                Arc::new(move |rt, env, top| {
                    let v = ci(rt, env, top);
                    rt.define_global(name, v);
                    rt.encode(Val::Sym(name))
                })
            }
            Ir::Let(..) => {
                panic!("unflattened Ir reached a tier: Let survives only before flatten::flatten")
            }
            Ir::Lambda { nparams, variadic, nslots, captures, body } => {
                let nparams = *nparams;
                let variadic = *variadic;
                let nslots = *nslots;
                let captures = captures.clone();
                let body = body.clone();
                Arc::new(move |rt, env, _top| {
                    let caps = build_caps::<M::R>(&captures, env);
                    let id = rt.alloc(Obj::Closure {
                        nparams,
                        variadic,
                        nslots,
                        body: body.clone(),
                        caps,
                    });
                    M::R::enc_ref(id)
                })
            }
            Ir::Call(f, args) => {
                let cf = self.compile(f);
                let cargs: Vec<Compiled<M>> = args.iter().map(|a| self.compile(a)).collect();
                Arc::new(move |rt, env, top| {
                    // Precise rooting, exactly as code.rs's `Call` arm: the callee is
                    // a bare `u64` while the args evaluate (a nested call in a later
                    // arg reaches a safepoint and would relocate it), and each earlier
                    // arg is bare while later ones evaluate. Publish each on the shadow
                    // stack as computed and re-read after; the collector rewrites them.
                    rt.safepoint(env);
                    let callee = cf(rt, env, top);
                    let base = rt.push_root(callee);
                    for c in &cargs {
                        let v = c(rt, env, top);
                        rt.push_root(v);
                    }
                    let calleer = rt.root_get(base);
                    let argv: Vec<u64> = (1..=cargs.len()).map(|i| rt.root_get(base + i)).collect();
                    rt.truncate_roots(base);
                    // Publish this frame's env so a collection deep inside the callee
                    // traces our still-live locals (mirrors code.rs). through `top`:
                    // composable + late-bound dispatch.
                    rt.env_stack.push(env.clone());
                    let r = top.invoke(top, rt, calleer, &argv);
                    rt.env_stack.pop();
                    r
                })
            }
            Ir::Prim(Prim::Gc, _) => {
                // Safepoint: pass the live env (this closure's `env`) to collect.
                Arc::new(move |rt: &mut Runtime<M>, env: &Locals, _top| {
                    rt.collect(env);
                    rt.encode(Val::Nil)
                })
            }
            Ir::Prim(op, args) => {
                let op = *op;
                let cargs: Vec<Compiled<M>> = args.iter().map(|a| self.compile(a)).collect();
                Arc::new(move |rt, env, top| {
                    // Precise rooting across arg evaluation (see code.rs's `Prim` arm):
                    // an already-evaluated earlier arg is a bare `u64` a collection
                    // fired at a LATER arg's nested-call safepoint would relocate — and
                    // `rt.prim` then stores them into a heap object (e.g. `cons`),
                    // where the next collection would trip over the stale bits.
                    let base = rt.root_depth();
                    for c in &cargs {
                        let v = c(rt, env, top);
                        rt.push_root(v);
                    }
                    let argv: Vec<u64> = (0..cargs.len()).map(|i| rt.root_get(base + i)).collect();
                    rt.truncate_roots(base);
                    rt.prim(op, &argv)
                })
            }
            Ir::DefMethod { name, ty, imp } => {
                let (name, ty) = (*name, *ty);
                let ci = self.compile(imp);
                Arc::new(move |rt, env, top| {
                    let c = ci(rt, env, top);
                    rt.register_method(name, ty, c);
                    rt.encode(Val::Nil)
                })
            }
            Ir::Dispatch { site, method, args } => {
                let (site, method) = (*site, *method);
                let cargs: Vec<Compiled<M>> = args.iter().map(|a| self.compile(a)).collect();
                Arc::new(move |rt, env, top| {
                    // Precise rooting across arg evaluation (see code.rs's `Dispatch`
                    // arm): earlier args are bare `u64`s a later arg's nested-call
                    // safepoint would relocate. Publish each as computed, re-read after.
                    let base = rt.root_depth();
                    for c in &cargs {
                        let v = c(rt, env, top);
                        rt.push_root(v);
                    }
                    let argv: Vec<u64> = (0..cargs.len()).map(|i| rt.root_get(base + i)).collect();
                    rt.truncate_roots(base);
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
                    // Publish this frame's env across the call (mirrors code.rs): a
                    // collection inside the impl must trace our still-live locals.
                    rt.env_stack.push(env.clone());
                    let r = top.invoke(top, rt, imp, &argv);
                    rt.env_stack.pop();
                    r
                })
            }
            Ir::FieldGet { site, field, obj } => {
                let (site, field) = (*site, *field);
                let cobj = self.compile(obj);
                Arc::new(move |rt, env, top| {
                    let o = cobj(rt, env, top);
                    rt.field_get(site, field, o)
                })
            }
            Ir::Try { .. } => panic!("try/catch is only supported on the TreeWalk tier"),
        }
    }

    fn compiled_body(&self, body: &Arc<Ir>) -> Compiled<M> {
        let key = Arc::as_ptr(body);
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
        let mut callee = callee;
        if let Some(sel) = rt.multifn_select(callee, args.len()) {
            if rt.pending() {
                return M::R::enc_nil();
            }
            callee = sel;
        }
        let Val::Ref(id) = rt.decode(callee) else {
            panic!("value not callable: {}", rt.print(callee));
        };
        let (nparams, variadic, nslots, body) = match rt.view_gc(id) {
            ObjView::Closure { nparams, variadic, nslots, template, .. } => {
                (nparams, variadic, nslots, rt.template(template).clone())
            }
            _ => panic!("value not callable: {}", rt.print(callee)),
        };

        // Compile-once: cached by body identity across all calls.
        let compiled = self.compiled_body(&body);
        let frame = rt.build_call_frame(nparams, variadic, nslots, args, callee);
        compiled(rt, &frame, top)
    }
}
