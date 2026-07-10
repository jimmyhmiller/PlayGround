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

use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::{Runtime, Var};
use crate::value::{frame_get, frame_set, Frame, Locals, Obj, Val};

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
            Ir::Global(s) => match rt.globals.get(s) {
                Some(v) => v.val,
                None => panic!("Unable to resolve symbol: {}", rt.sym_name(*s)),
            },
            Ir::SetLocal { up, idx, val } => {
                let v = top.eval_ir(top, rt, val, locals);
                frame_set(locals, *up, *idx, v);
                v
            }
            Ir::SetGlobal { name, val } => {
                let v = top.eval_ir(top, rt, val, locals);
                match rt.globals.get_mut(name) {
                    Some(var) => var.val = v,
                    None => panic!("set!: unbound variable: {}", rt.sym_name(*name)),
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
                rt.globals.insert(*name, Var { val: v });
                rt.encode(Val::Sym(*name))
            }
            Ir::Let(inits, body) => {
                // One growing frame (matching `analyze`), rebuilt each binding
                // from the PREVIOUS frame's current cell values (not a bare
                // Vec) so a GC during an init leaves the earlier bindings sound.
                let mut cur: Locals = Some(Rc::new(Frame {
                    slots: Vec::new(),
                    parent: locals.clone(),
                }));
                for iexpr in inits {
                    let v = top.eval_ir(top, rt, iexpr, &cur);
                    let prev = cur.as_ref().unwrap();
                    let mut slots: Vec<Cell<u64>> =
                        prev.slots.iter().map(|c| Cell::new(c.get())).collect();
                    slots.push(Cell::new(v));
                    cur = Some(Rc::new(Frame {
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
                let fv = top.eval_ir(top, rt, f, locals);
                let argv: Vec<u64> = args
                    .iter()
                    .map(|a| top.eval_ir(top, rt, a, locals))
                    .collect();
                top.invoke(top, rt, fv, &argv)
            }
            // The GC safepoint: the backend holds the live env, so it passes it
            // to `collect`. Frames are `Rc` with `Cell` slots, so `locals` stays
            // valid across the move and variable reads see relocated addresses.
            Ir::Prim(Prim::Gc, _) => {
                rt.collect(locals);
                rt.encode(Val::Nil)
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
            Ir::Prim(op, args) => {
                let argv: Vec<u64> = args
                    .iter()
                    .map(|a| top.eval_ir(top, rt, a, locals))
                    .collect();
                rt.prim(*op, &argv)
            }
            Ir::DefMethod { name, ty, imp } => {
                let c = top.eval_ir(top, rt, imp, locals);
                rt.register_method(*name, *ty, c);
                rt.encode(Val::Nil)
            }
            Ir::Dispatch { site, method, args } => {
                let argv: Vec<u64> = args
                    .iter()
                    .map(|a| top.eval_ir(top, rt, a, locals))
                    .collect();
                let ty = rt
                    .type_of(argv[0])
                    .unwrap_or_else(|| panic!("dispatch: receiver is not a record"));
                let imp = rt.resolve_method(*site, *method, ty).unwrap_or_else(|| {
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
            if let Obj::Record { .. } = &rt.heap[id as usize] {
                if let Some(h) = rt.apply_handler() {
                    let mut new_args = vec![callee];
                    new_args.extend_from_slice(&args);
                    callee = h;
                    args = new_args;
                    continue;
                }
            }
            let (nparams, variadic, body, env) = match &rt.heap[id as usize] {
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
            let mut cur: Locals = Some(Rc::new(Frame {
                slots: Vec::new(),
                parent: locals.clone(),
            }));
            for iexpr in inits {
                let v = top.eval_ir(top, rt, iexpr, &cur);
                let prev = cur.as_ref().unwrap();
                let mut slots: Vec<Cell<u64>> =
                    prev.slots.iter().map(|c| Cell::new(c.get())).collect();
                slots.push(Cell::new(v));
                cur = Some(Rc::new(Frame {
                    slots,
                    parent: locals.clone(),
                }));
            }
            eval_tail(top, rt, body, &cur)
        }
        Ir::Call(f, args) => {
            let callee = top.eval_ir(top, rt, f, locals);
            let argv: Vec<u64> = args.iter().map(|a| top.eval_ir(top, rt, a, locals)).collect();
            Bounce::Tail(callee, argv)
        }
        Ir::Dispatch { site, method, args } => {
            let argv: Vec<u64> = args.iter().map(|a| top.eval_ir(top, rt, a, locals)).collect();
            let ty = rt
                .type_of(argv[0])
                .unwrap_or_else(|| panic!("dispatch: receiver is not a record"));
            let imp = rt.resolve_method(*site, *method, ty).unwrap_or_else(|| {
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
