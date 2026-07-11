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
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::Runtime;
use crate::value::{clone_slots, frame_get, frame_set, Frame, Locals, Obj, Val};

pub trait CodeSpace<M: ValueModel> {
    /// Evaluate one IR node. `top` is the outermost backend to recurse through.
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals)
        -> u64;

    /// Call a callable value. For macros the args are unevaluated forms.
    /// Re-entrant: `Runtime::macroexpand` calls this mid-compilation.
    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64;
}

/// The panic payload carrying a non-local escape back to its `%callec`.
struct EscapeSignal {
    tag: u64,
    value: u64,
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
            Ir::Const(id) | Ir::Quote(id) => rt.get_const(*id),
            Ir::Local { up, idx } => frame_get(locals, *up, *idx),
            Ir::Global(s) => match rt.global(*s) {
                Some(v) => v,
                None => panic!("Unable to resolve symbol: {}", rt.sym_name(*s)),
            },
            Ir::SetLocal { up, idx, val } => {
                let v = top.eval_ir(top, rt, val, locals);
                frame_set(locals, *up, *idx, v);
                v
            }
            Ir::SetGlobal { name, val } => {
                let v = top.eval_ir(top, rt, val, locals);
                if !rt.set_global_val(*name, v) {
                    panic!("set!: unbound variable: {}", rt.sym_name(*name));
                }
                v
            }
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
            Ir::Def { name, init } => {
                let v = top.eval_ir(top, rt, init, locals);
                rt.define_global(*name, v);
                rt.encode(Val::Sym(*name))
            }
            Ir::Let(inits, body) => {
                // One growing frame (matching `analyze`), rebuilt each binding
                // from the PREVIOUS frame's current cell values (not a bare
                // Vec) so a GC during an init leaves the earlier bindings sound.
                let mut cur: Locals = Some(Arc::new(Frame {
                    slots: Vec::new(),
                    parent: locals.clone(),
                }));
                for iexpr in inits {
                    let v = top.eval_ir(top, rt, iexpr, &cur);
                    let prev = cur.as_ref().unwrap();
                    let mut slots = clone_slots(&prev.slots);
                    slots.push(AtomicU64::new(v));
                    cur = Some(Arc::new(Frame {
                        slots,
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
                // GC safepoint: a call boundary holds no `&Obj` borrow, so this is
                // a safe place to park for a stop-the-world collection on another
                // thread. `locals` is published so the collector traces our env.
                rt.safepoint(locals);
                // Precise rooting: an already-evaluated callee / earlier argument
                // is a bare `u64` that a collection (triggered by ANOTHER thread's
                // safepoint reached while we evaluate a LATER argument) would
                // relocate. Publish each on the shadow stack and re-read after.
                let fv = top.eval_ir(top, rt, f, locals);
                let base = rt.push_root(fv);
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    rt.push_root(v);
                }
                let fvr = rt.root_get(base);
                let argv: Vec<u64> = (1..=args.len()).map(|i| rt.root_get(base + i)).collect();
                rt.truncate_roots(base);
                // Record this caller's env on the dynamic env stack so a collection
                // that fires deep inside the callee traces THIS frame's live locals
                // too (a callee's lexical parent chain does not reach its caller).
                rt.env_stack.push(locals.clone());
                let r = top.invoke(top, rt, fvr, &argv);
                rt.env_stack.pop();
                r
            }
            // The GC safepoint: the backend holds the live env, so it passes it
            // to `collect`. Frames are `Rc` with `Cell` slots, so `locals` stays
            // valid across the move and variable reads see relocated addresses.
            Ir::Prim(Prim::Gc, _) => {
                rt.collect(locals);
                rt.encode(Val::Nil)
            }
            // `(%spawn f)` — run the thunk `f` on a fresh OS thread that SHARES
            // this runtime (heap/globals/interner via the `Arc<Shared>`), with its
            // own shadow stack. Returns a `Future`. The child runs on the base
            // `TreeWalk` tier (the only backend guaranteed `Send`). Rooting note:
            // the future's eventual result is NOT a GC root until awaited, so an
            // explicit `(gc)` between the worker finishing and `deref` could
            // reclaim it — fine under this toolkit's explicit-only GC; a safepoint
            // GC (next phase) must also scan future slots.
            Ir::Prim(Prim::Spawn, args) => {
                let f = top.eval_ir(top, rt, &args[0], locals);
                let child = rt.thread_handle();
                let slot = std::sync::Arc::new(std::sync::Mutex::new(
                    crate::value::FutureSlot { handle: None, result: None },
                ));
                let slot_worker = slot.clone();
                let slot_obj = slot.clone();
                let handle = std::thread::spawn(move || {
                    let cs = TreeWalk;
                    let mut crt = child;
                    let r = cs.invoke(&cs, &mut crt, f, &[]);
                    // Publish the result into the shared slot BEFORE the worker's
                    // handle drops (deregistering it), so it is GC-rooted via the
                    // reachable Future object (scan_obj forwards it).
                    slot_worker.lock().unwrap().result = Some(r);
                    r
                });
                slot.lock().unwrap().handle = Some(handle);
                let id = rt.alloc(Obj::Future(slot_obj));
                M::R::enc_ref(id)
            }
            // `(%await fut)` — backend-handled so it can publish this thread's
            // roots + park while blocked on the join (a concurrent collector can
            // then proceed). (The plain-`prim` `Await` is the single-thread path.)
            Ir::Prim(Prim::Await, args) => {
                let fut = top.eval_ir(top, rt, &args[0], locals);
                rt.await_future(fut, locals)
            }
            // `(%callec f)`: call f with a fresh escape continuation. Invoking
            // the continuation unwinds (a panic carrying the tag) back here; f
            // returning normally is the result. Escape-only (one-shot upward);
            // full multi-shot continuations would need CPS or stack copying.
            Ir::Prim(Prim::CallEc, args) => {
                let f = top.eval_ir(top, rt, &args[0], locals);
                let tag = rt.fresh_escape_tag();
                let kid = rt.alloc(Obj::Escape { tag });
                let kref = M::R::enc_ref(kid);
                let prev = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {})); // escapes are not errors
                let res = {
                    let rt2 = &mut *rt;
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        top.invoke(top, rt2, f, &[kref])
                    }))
                };
                std::panic::set_hook(prev);
                match res {
                    Ok(v) => v,
                    Err(payload) => match payload.downcast::<EscapeSignal>() {
                        Ok(sig) if sig.tag == tag => sig.value, // caught our escape
                        Ok(sig) => std::panic::resume_unwind(sig), // a different escape
                        Err(other) => std::panic::resume_unwind(other), // a real panic
                    },
                }
            }
            Ir::Try { body, catch, finally } => {
                let prev = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {})); // a caught throw is not an error
                let res = {
                    let rt2 = &mut *rt;
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        top.eval_ir(top, rt2, body, locals)
                    }))
                };
                std::panic::set_hook(prev);
                let outcome = match res {
                    Ok(v) => Ok(v),
                    Err(payload) => match payload.downcast::<crate::runtime::Thrown>() {
                        // Our throw: run the catch handler with the value bound in
                        // a fresh one-slot frame (Local{up:0,idx:0}).
                        Ok(t) => match catch {
                            Some(cbody) => {
                                let frame: Locals = Some(Arc::new(Frame {
                                    slots: vec![AtomicU64::new(t.value)],
                                    parent: locals.clone(),
                                }));
                                // Root the thrown value across the handler's evals.
                                Ok(top.eval_ir(top, rt, cbody, &frame))
                            }
                            None => Err(Box::new(*t) as Box<dyn std::any::Any + Send>),
                        },
                        // Some other unwind (a real panic, an escape signal): after
                        // running `finally`, re-raise it.
                        Err(other) => Err(other),
                    },
                };
                if let Some(fbody) = finally {
                    top.eval_ir(top, rt, fbody, locals);
                }
                match outcome {
                    Ok(v) => v,
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            Ir::FieldGet { site, field, obj } => {
                // No allocation between eval and the field read, so no rooting.
                let o = top.eval_ir(top, rt, obj, locals);
                rt.field_get(*site, *field, o)
            }
            Ir::Prim(Prim::Apply, args) => {
                // `(apply f a … lst)` — invoking a closure with a runtime-built
                // arg list, which `rt.prim` cannot do (it has no `top`). Evaluate
                // the args (rooted), flatten the leading args with the elements of
                // the final list, then invoke through `top` (so tail-calls, GC
                // safepoints, and dispatch all compose).
                let base = rt.root_depth();
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    rt.push_root(v);
                }
                let argv: Vec<u64> = (0..args.len()).map(|i| rt.root_get(base + i)).collect();
                rt.truncate_roots(base);
                let f = argv[0];
                let rest = &argv[1..];
                let mut flat: Vec<u64> = rest[..rest.len().saturating_sub(1)].to_vec();
                if let Some(&last) = rest.last() {
                    flat.extend(rt.list_to_vec(last));
                }
                top.invoke(top, rt, f, &flat)
            }
            Ir::Prim(op, args) => {
                // Precise rooting of already-evaluated args across the remaining
                // arg evals (which may safepoint) — see the `Call` arm.
                let base = rt.root_depth();
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    rt.push_root(v);
                }
                let argv: Vec<u64> = (0..args.len()).map(|i| rt.root_get(base + i)).collect();
                rt.truncate_roots(base);
                rt.prim(*op, &argv)
            }
            Ir::DefMethod { name, ty, imp } => {
                let c = top.eval_ir(top, rt, imp, locals);
                rt.register_method(*name, *ty, c);
                rt.encode(Val::Nil)
            }
            Ir::Dispatch { site, method, args } => {
                // Precise rooting of already-evaluated args (see the `Call` arm).
                let base = rt.root_depth();
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    rt.push_root(v);
                }
                let argv: Vec<u64> = (0..args.len()).map(|i| rt.root_get(base + i)).collect();
                rt.truncate_roots(base);
                let ty = rt.type_tag(argv[0]);
                let imp = rt.resolve_or_default(*site, *method, ty).unwrap_or_else(|| {
                    panic!(
                        "no method '{}' for type '{}'",
                        rt.sym_name(*method),
                        rt.sym_name(ty)
                    )
                });
                top.invoke(top, rt, imp, &argv)
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
        // Trampoline for proper tail calls: a tail call bounces back here and
        // loops, reusing this Rust frame, instead of recursing. Deep tail
        // recursion runs in O(1) native stack. Non-tail calls still recurse
        // (through `top`, so composition/GC-safepoints are preserved), and a
        // tail call's args are consumed into the new frame before the next
        // body runs, so no bare pointer is held across a collection.
        let mut callee = callee;
        let mut args: Vec<u64> = args.to_vec();
        loop {
            let Val::Ref(id) = rt.decode(callee) else {
                panic!("value not callable: {}", rt.print(callee));
            };
            // Callable-object hook: a non-closure record invoked with a registered
            // apply handler redirects to `(handler object args…)` (e.g. keywords).
            if let Obj::Record { .. } = &rt.heap()[id as usize] {
                if let Some(h) = rt.apply_handler() {
                    let mut new_args = vec![callee];
                    new_args.extend_from_slice(&args);
                    callee = h;
                    args = new_args;
                    continue;
                }
            }
            let (nparams, variadic, body, env) = match &rt.heap()[id as usize] {
                Obj::Closure {
                    nparams,
                    variadic,
                    body,
                    env,
                } => (*nparams, *variadic, body.clone(), env.clone()),
                Obj::Escape { tag } => {
                    // Invoking an escape continuation: unwind to its `%callec`.
                    let sig = EscapeSignal {
                        tag: *tag,
                        value: args.first().copied().unwrap_or_else(M::R::enc_nil),
                    };
                    std::panic::panic_any(sig);
                }
                _ => panic!("value not callable: {}", rt.print(callee)),
            };
            let frame = rt.build_call_frame(nparams, variadic, &args, env);
            match eval_tail(top, rt, &body, &frame) {
                Bounce::Done(v) => return v,
                Bounce::Tail(next, next_args) => {
                    callee = next;
                    args = next_args;
                }
            }
        }
    }
}

/// The result of evaluating an expression in TAIL position: either a finished
/// value, or a tail call the `invoke` trampoline should bounce to.
enum Bounce {
    Done(u64),
    Tail(u64, Vec<u64>),
}

/// Evaluate in tail position. Tail-transparent forms (`if`, `do`, `let`) recurse
/// structurally (bounded by static nesting), a tail `Call`/`Dispatch` returns a
/// `Tail` for the trampoline, and everything else is a finished value. Non-tail
/// subexpressions go through `top` so composition and GC safepoints hold.
fn eval_tail<M: ValueModel>(
    top: &dyn CodeSpace<M>,
    rt: &mut Runtime<M>,
    ir: &Ir,
    locals: &Locals,
) -> Bounce {
    match ir {
        Ir::If(c, t, e) => {
            let cv = top.eval_ir(top, rt, c, locals);
            if M::truthy(rt.decode(cv)) {
                eval_tail(top, rt, t, locals)
            } else {
                eval_tail(top, rt, e, locals)
            }
        }
        Ir::Do(xs) => match xs.split_last() {
            None => Bounce::Done(rt.encode(Val::Nil)),
            Some((last, init)) => {
                for x in init {
                    top.eval_ir(top, rt, x, locals);
                }
                eval_tail(top, rt, last, locals)
            }
        },
        Ir::Let(inits, body) => {
            let mut cur: Locals = Some(Arc::new(Frame {
                slots: Vec::new(),
                parent: locals.clone(),
            }));
            for iexpr in inits {
                let v = top.eval_ir(top, rt, iexpr, &cur);
                let prev = cur.as_ref().unwrap();
                let mut slots = clone_slots(&prev.slots);
                slots.push(AtomicU64::new(v));
                cur = Some(Arc::new(Frame {
                    slots,
                    parent: locals.clone(),
                }));
            }
            eval_tail(top, rt, body, &cur)
        }
        Ir::Call(f, args) => {
            // Precise rooting across arg evaluation, exactly as the eval_ir `Call`
            // arm: the already-evaluated callee/args are bare u64s a collection
            // (fired at a nested safepoint) would relocate — and they escape this
            // frame in the `Bounce::Tail`, so they must be re-read after.
            let callee = top.eval_ir(top, rt, f, locals);
            let base = rt.push_root(callee);
            for a in args {
                let v = top.eval_ir(top, rt, a, locals);
                rt.push_root(v);
            }
            let calleer = rt.root_get(base);
            let argv: Vec<u64> = (1..=args.len()).map(|i| rt.root_get(base + i)).collect();
            rt.truncate_roots(base);
            Bounce::Tail(calleer, argv)
        }
        Ir::Dispatch { site, method, args } => {
            let base = rt.root_depth();
            for a in args {
                let v = top.eval_ir(top, rt, a, locals);
                rt.push_root(v);
            }
            let argv: Vec<u64> = (0..args.len()).map(|i| rt.root_get(base + i)).collect();
            rt.truncate_roots(base);
            let ty = rt.type_tag(argv[0]);
            let imp = rt.resolve_or_default(*site, *method, ty).unwrap_or_else(|| {
                panic!(
                    "no method '{}' for type '{}'",
                    rt.sym_name(*method),
                    rt.sym_name(ty)
                )
            });
            Bounce::Tail(imp, argv)
        }
        _ => Bounce::Done(top.eval_ir(top, rt, ir, locals)),
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
