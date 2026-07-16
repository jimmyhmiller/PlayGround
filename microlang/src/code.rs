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

use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::{ObjView, Runtime};
use crate::value::{build_caps, frame_cap, frame_get, frame_set, Locals, Obj, Val};

pub trait CodeSpace<M: ValueModel> {
    /// Evaluate one IR node. `top` is the outermost backend to recurse through.
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals)
        -> u64;

    /// Call a callable value. For macros the args are unevaluated forms.
    /// Re-entrant: `Runtime::macroexpand` calls this mid-compilation.
    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64;
}


/// The interpreter tier.
#[derive(Clone, Copy)]
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
            Ir::Capture(idx) => frame_cap::<M::R>(locals, *idx),
            Ir::Global(s) => match rt.global(*s) {
                Some(v) => v,
                None => {
                    // An unresolved var is a CATCHABLE throw (Clojure's
                    // CompilerException), not an abort — a REPL/server must
                    // survive a typo'd reference.
                    let id = rt.alloc(Obj::Str(format!(
                        "Unable to resolve symbol: {}",
                        rt.sym_name(*s)
                    )));
                    rt.signal_throw(M::R::enc_ref(id));
                    M::R::enc_nil()
                }
            },
            Ir::SetLocal { up, idx, val } => {
                let v = top.eval_ir(top, rt, val, locals);
                if rt.pending() {
                    return v;
                }
                frame_set(locals, *up, *idx, v);
                v
            }
            Ir::SetGlobal { name, val } => {
                let v = top.eval_ir(top, rt, val, locals);
                if rt.pending() {
                    return v;
                }
                if !rt.set_global_val(*name, v) {
                    panic!("set!: unbound variable: {}", rt.sym_name(*name));
                }
                v
            }
            Ir::If(c, t, e) => {
                let cv = top.eval_ir(top, rt, c, locals);
                if rt.pending() {
                    return cv;
                }
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
                    if rt.pending() {
                        return r; // a throw/escape short-circuits the sequence
                    }
                }
                r
            }
            Ir::Def { name, init } => {
                let v = top.eval_ir(top, rt, init, locals);
                if rt.pending() {
                    return v;
                }
                rt.define_global(*name, v);
                rt.encode(Val::Sym(*name))
            }
            Ir::Let(..) => {
                panic!("unflattened Ir reached a tier: Let survives only before flatten::flatten")
            }
            Ir::Lambda { nparams, variadic, nslots, captures, body } => {
                let caps = build_caps::<M::R>(captures, locals);
                let id = rt.alloc(Obj::Closure {
                    nparams: *nparams,
                    variadic: *variadic,
                    nslots: *nslots,
                    body: body.clone(),
                    caps,
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
                if rt.pending() {
                    return fv;
                }
                let base = rt.push_root(fv);
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    if rt.pending() {
                        rt.truncate_roots(base);
                        return v;
                    }
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
                let mut child = rt.thread_handle();
                // Root the thunk in the CHILD's shadow before the thread exists
                // (Stage E): no collection can COMPLETE until the worker parks,
                // and parking publishes (and rewrites) this slot — so the
                // worker's re-read always sees the thunk's current address. A
                // bare moved `u64` would go stale under pressure GC.
                let f_slot = child.push_root(f);
                let slot = std::sync::Arc::new(std::sync::Mutex::new(
                    crate::value::FutureSlot { handle: None, result: None },
                ));
                let slot_worker = slot.clone();
                let slot_obj = slot.clone();
                let handle = std::thread::spawn(move || {
                    let cs = TreeWalk;
                    let mut crt = child;
                    let f = crt.root_get(f_slot);
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
                if rt.pending() {
                    return fut;
                }
                rt.await_future(fut, locals)
            }
            // `(%callec f)`: call f with a fresh escape continuation. Invoking the
            // continuation sets an ESCAPE signal carrying this call's tag; f
            // returning normally is the result. Escape-only (one-shot, upward).
            Ir::Prim(Prim::CallEc, args) => {
                let f = top.eval_ir(top, rt, &args[0], locals);
                if rt.pending() {
                    return f;
                }
                let tag = rt.fresh_escape_tag();
                let kid = rt.alloc(Obj::Escape { tag });
                let kref = M::R::enc_ref(kid);
                // Publish THIS frame's env across the call (see the `Call` arm):
                // a collection fired inside `f` must trace our still-live locals.
                rt.env_stack.push(locals.clone());
                let v = top.invoke(top, rt, f, &[kref]);
                rt.env_stack.pop();
                // Caught our escape? (kind 2 == escape, matching tag.)
                if rt.signal.kind == 2 && rt.signal_tag() == tag {
                    return rt.take_signal().value;
                }
                v // normal return, or a throw / other escape that keeps propagating
            }
            Ir::Try { body, catch, finally, cslot } => {
                let mut result = top.eval_ir(top, rt, body, locals);
                // Catch a THROW (kind 1) if there is a handler; escapes pass through.
                if rt.pending_throw() {
                    if let Some(cbody) = catch {
                        let thrown = rt.take_signal().value;
                        // The thrown value's binding was re-homed by `flatten` to
                        // a slot of THIS activation frame — no fresh frame.
                        frame_set(locals, 0, *cslot, thrown);
                        result = top.eval_ir(top, rt, cbody, locals); // may re-raise
                    }
                    // no catch: the throw signal stays pending and propagates below
                }
                // `finally` runs on every path; a signal IT raises supersedes, else
                // the suspended throw/escape (if any) is restored.
                if let Some(fbody) = finally {
                    let suspended = rt.take_signal();
                    // `finally` runs ARBITRARY code — safepoints included — and
                    // TWO bare heap pointers are live across it, named by nothing
                    // else: `result` (the body's/handler's value, which we return
                    // BELOW) and the suspended signal's payload (taken OUT of the
                    // signal, so not even the signal root covers it). Root both,
                    // re-read both.
                    let base = rt.push_root(result);
                    rt.push_root(suspended.value);
                    let fv = top.eval_ir(top, rt, fbody, locals);
                    result = rt.root_get(base);
                    let value = rt.root_get(base + 1);
                    rt.truncate_roots(base);
                    if rt.pending() {
                        return fv; // a signal raised BY `finally` supersedes
                    }
                    rt.signal = crate::runtime::Signal { value, ..suspended };
                }
                result
            }
            Ir::FieldGet { site, field, obj } => {
                // No allocation between eval and the field read, so no rooting.
                let o = top.eval_ir(top, rt, obj, locals);
                if rt.pending() {
                    return o;
                }
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
                    if rt.pending() {
                        rt.truncate_roots(base);
                        return v;
                    }
                    rt.push_root(v);
                }
                // EVERYTHING below is inside one safepoint-reaching region:
                // `seq_flatten` INVOKES the seq fn to force a lazy tail, which
                // RUNS USER CODE (`(apply str (map f xs))` calls `f` in there —
                // this is how core's `pr-str` is written). So:
                //  * the callee + leading args stay ROOTED across the flatten and
                //    are re-read after. Copying them out to bare locals first (as
                //    this arm used to) leaves them pointing into the collected
                //    space, and `invoke` then stores them into the callee's frame,
                //    where the next collection trips over them in `visit_env`.
                //  * THIS frame's env is published for the WHOLE region (see the
                //    `Call` arm) — not just around the invoke, since a collection
                //    can equally fire inside the flatten, and our still-live
                //    locals must be traced for it.
                rt.env_stack.push(locals.clone());
                let n = args.len();
                let mut flat: Vec<u64> = Vec::new();
                if n >= 2 {
                    let last_slot = base + n - 1;
                    let last = rt.root_get(last_slot);
                    let tail = rt.seq_flatten(top, last);
                    flat.extend((base + 1..last_slot).map(|i| rt.root_get(i)));
                    flat.extend(tail);
                }
                let f = rt.root_get(base);
                rt.truncate_roots(base);
                let r = top.invoke(top, rt, f, &flat);
                rt.env_stack.pop();
                r
            }
            Ir::Prim(op, args) => {
                // Precise rooting of already-evaluated args across the remaining
                // arg evals (which may safepoint) — see the `Call` arm.
                let base = rt.root_depth();
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    if rt.pending() {
                        rt.truncate_roots(base);
                        return v;
                    }
                    rt.push_root(v);
                }
                let argv: Vec<u64> = (0..args.len()).map(|i| rt.root_get(base + i)).collect();
                rt.truncate_roots(base);
                // `Throw` sets the signal here; the returned dummy propagates up.
                rt.prim(*op, &argv)
            }
            Ir::DefMethod { name, ty, imp } => {
                let c = top.eval_ir(top, rt, imp, locals);
                if rt.pending() {
                    return c;
                }
                rt.register_method(*name, *ty, c);
                rt.encode(Val::Nil)
            }
            Ir::Dispatch { site, method, args } => {
                // Precise rooting of already-evaluated args (see the `Call` arm).
                let base = rt.root_depth();
                for a in args {
                    let v = top.eval_ir(top, rt, a, locals);
                    if rt.pending() {
                        rt.truncate_roots(base);
                        return v;
                    }
                    rt.push_root(v);
                }
                let argv: Vec<u64> = (0..args.len()).map(|i| rt.root_get(base + i)).collect();
                rt.truncate_roots(base);
                let ty = rt.type_tag(argv[0]);
                let imp = match rt.resolve_or_default(*site, *method, ty) {
                    Some(imp) => imp,
                    None => {
                        // No protocol impl for this type — a CATCHABLE throw (like
                        // Clojure's "no implementation of method" / seq-on-non-seq),
                        // not a hard panic, so `try`/`catch` can handle it.
                        let msg = format!(
                            "no method '{}' for type '{}'",
                            rt.sym_name(*method),
                            rt.sym_name(ty)
                        );
                        let id = rt.alloc(Obj::Str(msg));
                        rt.signal_throw(M::R::enc_ref(id));
                        return M::R::enc_nil();
                    }
                };
                // Publish THIS frame's env across the call (see the `Call` arm).
                rt.env_stack.push(locals.clone());
                let r = top.invoke(top, rt, imp, &argv);
                rt.env_stack.pop();
                r
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
            if let ObjView::Record { .. } = rt.view_gc(id) {
                if let Some(h) = rt.apply_handler() {
                    let mut new_args = vec![callee];
                    new_args.extend_from_slice(&args);
                    callee = h;
                    args = new_args;
                    continue;
                }
            }
            // Multi-arity fn: select the clause serving this arg count and loop.
            if matches!(rt.view_gc(id), ObjView::MultiFn { .. }) {
                let sel = rt.multifn_select(callee, args.len()).expect("checked MultiFn");
                if rt.pending() {
                    return M::R::enc_nil(); // no matching clause: arity throw
                }
                callee = sel;
                continue;
            }
            let (nparams, variadic, nslots, body) = match rt.view_gc(id) {
                ObjView::Closure { nparams, variadic, nslots, template, .. } => {
                    (nparams, variadic, nslots, rt.template(template).clone())
                }
                ObjView::Escape { tag } => {
                    // Invoking an escape continuation: raise an ESCAPE signal to its
                    // `%callec`; the dummy return propagates up like any signal.
                    let v = args.first().copied().unwrap_or_else(M::R::enc_nil);
                    rt.signal_escape(tag, v);
                    return M::R::enc_nil();
                }
                _ => panic!("value not callable: {}", rt.print(callee)),
            };
            // Arity mismatch is a CATCHABLE throw (not a hard panic), so `try`/
            // `catch` can handle a wrong-arity call as it would in Clojure.
            let arity_ok = if variadic { args.len() >= nparams } else { args.len() == nparams };
            if !arity_ok {
                let msg = if variadic {
                    format!("arity: expected at least {}, got {}", nparams, args.len())
                } else {
                    format!("arity: expected {}, got {}", nparams, args.len())
                };
                let sid = rt.alloc(Obj::Str(msg));
                rt.signal_throw(M::R::enc_ref(sid));
                return M::R::enc_nil();
            }
            let frame = rt.build_call_frame(nparams, variadic, nslots, &args, callee);
            match eval_tail(top, rt, &body, &frame) {
                Bounce::Done(v) => return v,
                Bounce::Tail(next, next_args) => {
                    // A signal raised while evaluating the tail call's args: stop.
                    if rt.pending() {
                        return next;
                    }
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
            if rt.pending() {
                return Bounce::Done(cv);
            }
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
                    let v = top.eval_ir(top, rt, x, locals);
                    if rt.pending() {
                        return Bounce::Done(v);
                    }
                }
                eval_tail(top, rt, last, locals)
            }
        },
        Ir::Call(f, args) => {
            // Precise rooting across arg evaluation, exactly as the eval_ir `Call`
            // arm: the already-evaluated callee/args are bare u64s a collection
            // (fired at a nested safepoint) would relocate — and they escape this
            // frame in the `Bounce::Tail`, so they must be re-read after.
            let callee = top.eval_ir(top, rt, f, locals);
            if rt.pending() {
                return Bounce::Done(callee);
            }
            let base = rt.push_root(callee);
            for a in args {
                let v = top.eval_ir(top, rt, a, locals);
                if rt.pending() {
                    rt.truncate_roots(base);
                    return Bounce::Done(v);
                }
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
                if rt.pending() {
                    rt.truncate_roots(base);
                    return Bounce::Done(v);
                }
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
