//! A NATIVE emit tier: a Cranelift JIT.
//!
//! This is the machine-code analogue of `bytecode.rs`. Where the bytecode tier
//! lowers each `Ir` to a stack-VM op stream, this tier lowers the same `Ir` to
//! real Cranelift IR and compiles it to host machine code. It is the first
//! backend to make the CODEGEN_AXES "emit half" produce actual instructions.
//!
//! ## Per-model arithmetic, now with the numeric tower
//!
//! `bytecode.rs` introduced `ModelEmit`: the value model's EMIT half for `+ - * <`,
//! *differing by representation* (LowBit shifts to untag, HighBit does not, NanBox
//! boxes to a slow call). That recipe is the fixnum FAST PATH and it *wraps* on
//! overflow. This tier keeps the same per-model split via `ModelArithJit`
//! (`emit_both_int`/`emit_untag`/`emit_tag`) but wraps it in a guard: take the fast
//! path only when both operands are immediate fixnums AND the result fits fixnum
//! range; otherwise call the runtime's checked, promoting `prim`. So the JIT has
//! the FULL numeric tower — bignum promotion, floats, mixed — that the tree-walker
//! has, and is strictly more correct than the wrapping bytecode tier. That guard +
//! fallback is the emit half of codegen axis #2 (overflow policy).
//!
//! ## What is inline vs. what calls the runtime
//!
//! Emitted as native instructions (no call): guarded arithmetic, control flow
//! (`if`/`do`/`let`), constant loads (from the pool base cached in `JitCtx`),
//! `if` truthiness (two compares against the model's `nil`/`false` words), and
//! innermost-frame (`up == 0`) local reads/writes (a load/store through the cached
//! `cur_slots` pointer — every function param and hot-loop variable). What still
//! funnels back through `extern "C"` shims: globals, `up > 0` locals (the parent
//! chain walk), closure construction, non-arith prims, and the calling convention
//! (`shim_call`/`shim_tail_call` -> `invoke`). Those keep the value-model + GC +
//! dispatch contracts in one place; the hot straight-line path is compiled code.
//!
//! ## Calling convention: a frame pool
//!
//! Calls used to heap-allocate an `Arc<Frame>` per callee — the dominant cost. Now
//! a freed frame (uniquely owned, `strong_count == 1`, so provably not captured by
//! any closure) is returned to a pool and refilled via `Arc::get_mut` on the next
//! call. Deep recursion converges to ~depth frames; a tail loop reuses one; and
//! tail args reuse the trampoline's buffer. The `get_mut`/`strong_count` guard is
//! the exact test for "no other owner", so a captured frame can NEVER be recycled
//! (see `jit_frame_pool_never_recycles_a_captured_frame`). This closed most of the
//! per-call gap to a production compiler without touching the frame/GC model; the
//! rest (register args, a direct code pointer, native-stack frames) is the deeper
//! calling-convention overhaul still ahead.
//!
//! ## Scope
//!
//! Covered: constants, locals, globals, `if`, `do`, `def`, `let`, `set!`,
//! `fn`/closures, calls (with PROPER TAIL CALLS via a trampoline, so unbounded
//! tail recursion runs in O(1) stack), the arithmetic prims, and other prims via a
//! runtime escape. That is all of Scheme except first-class continuations. NOT
//! covered — each errors clearly, and `Tiered` routes it to the `CekMachine`:
//! `%callcc`/`%reset`/`%shift`/`%callec`, `apply`, records/`dispatch`, and the
//! `(gc)` safepoint. Like the bytecode tier this tier does not yet model the GC
//! safepoint (native temporaries are not yet roots — the frame/roots emit axis,
//! the next honest step).

use std::cell::{Cell, RefCell};
use std::sync::atomic::AtomicU64;
use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I128, I64, I8};
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::bytecode::ModelEmit;
use crate::code::CodeSpace;
use crate::ir::{CapSrc, Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::Runtime;
use crate::value::{frame_get, frame_set, no_caps, Caps, Frame, Locals, Obj, Sym, Val};

// ─────────────────────────────────────────────────────────────────────────
// The runtime-interaction ABI.
//
// A compiled body is `extern "C" fn(*mut JitCtx<M>) -> u64`. Everything it
// cannot do with a bare arithmetic/branch instruction it does by calling one of
// the shims below with that context pointer. This mirrors how the tree-walker
// and bytecode VM call back into `Runtime` — the difference is only that here
// the caller is machine code.
// ─────────────────────────────────────────────────────────────────────────

/// The per-invocation context handed to compiled code. Held on the Rust stack in
/// `run`; the compiled function only ever sees a `*mut` to it. Pointers (not
/// references) so the shims can re-borrow `&mut Runtime` at each call boundary
/// without fighting the borrow checker — the same single-threaded, reentrant
/// discipline the interpreter tiers already rely on.
///
/// `#[repr(C)]` so `cur_slots` and `consts_base` sit at stable offsets: compiled
/// code loads them directly (via `offset_of!`) to read innermost locals and
/// constants inline, without a shim call. See `emit_local_load` / the `Const` arm.
#[repr(C)]
pub struct JitCtx<'a, M: ValueModel> {
    /// Pointer to the "run context" — the outermost `JitCtx` of this call tree,
    /// which owns the fields that DON'T change per call (`rt`, `top`, the four
    /// base pointers, `direct`, `tail_args`). Every child copies just this pointer
    /// (plus its own `cur_slots` / `self_closure` / `tail_pending`), and reads the
    /// shared fields through it — so a native call builds a 4-store context instead
    /// of copying nine fields. The top context points at itself.
    rc: *const JitCtx<'a, M>,
    top: &'a dyn CodeSpace<M>,
    rt: *mut Runtime<M>,
    /// Raw base pointer of the INNERMOST frame's slot array (`AtomicU64` is
    /// transparent over `u64`). Compiled code reads `up == 0` locals as
    /// `*(cur_slots + idx)` — no call. Kept in sync with `cur` on entry and on
    /// every `let` enter/exit.
    cur_slots: Cell<*const u64>,
    /// Raw base pointer of the constant pool, for inline `Const`/`Quote` loads.
    consts_base: Cell<*const u64>,
    /// Raw base + length of the dense global mirror (`Runtime::global_slots`), for
    /// inline `Global` reads. A slot holding `GLOBAL_UNBOUND` means "not bound
    /// yet" and the compiled code falls back to the slow late-binding lookup.
    global_base: Cell<*const u64>,
    global_len: Cell<usize>,
    /// Base + length of the native fast-call table (`JitCranelift::fast_targets`),
    /// so a call site can resolve a callee to its native entry inline. Copied into
    /// each caller-built child context so nested fast calls keep resolving.
    fast_base: Cell<*const FastTarget>,
    fast_len: Cell<usize>,
    /// Is the native fast call path enabled? True only when this backend is the
    /// OUTERMOST one (`top == self`). When wrapped (e.g. by `Traced`, which must
    /// observe every call, or a future router), this is 0 and every call takes the
    /// shim path through `top` — so composition is preserved exactly.
    direct: Cell<u8>,
    /// The bits of the closure currently running (0 if none / top-level). A tail
    /// call to THIS same closure becomes an in-place native loop (redefine the
    /// SSA loop variables, branch back) — O(1) stack, no FFI. A tail call to
    /// anything else falls back to the shim + trampoline, so mutual tail
    /// recursion keeps full TCO.
    self_closure: Cell<u64>,
    /// Base pointer of the RUNNING closure's capture array (`Caps`), null when
    /// nothing is captured. `Capture(i)` compiles to `*(caps_base + i)`.
    caps_base: Cell<*const u64>,
    /// The activation frame, as an `Arc` handle — only for MEMORY-mode bodies
    /// (those containing `try`/`(gc)`/`%await`, whose locals must be GC roots /
    /// interpreter-visible). SSA-mode bodies never touch it.
    cur: RefCell<Locals>,
    /// The GLOBAL closure-template registry (owned by the backend, append-only),
    /// read through `rc` by `shim_make_closure`.
    tregistry: *const RefCell<Vec<ClosureTemplate>>,
    /// Proper-tail-call signalling. A tail `Call` stashes its (callee, args) here
    /// and returns; the `invoke` trampoline reuses this native frame for the next
    /// body instead of recursing — so a million-deep tail loop runs in O(1) stack,
    /// the JIT analogue of the tree-walker's `Bounce`/`eval_tail` trampoline.
    tail_pending: Cell<bool>,
    tail_callee: Cell<u64>,
    /// Raw pointer to the trampoline's reusable args buffer. A tail call writes
    /// its evaluated args here (reusing the capacity), so a tail loop allocates
    /// nothing per bounce.
    tail_args: *mut Vec<u64>,
}

/// Raw base pointer of a frame's slot array (null for the empty environment).
/// `AtomicU64` is `repr(transparent)` over `UnsafeCell<u64>` (same layout as `u64`), so this doubles as a `*const u64`
/// the compiled code reads and writes slots through.
fn slots_ptr(l: &Locals) -> *const u64 {
    match l {
        Some(f) => f.slots.as_ptr() as *const u64,
        None => std::ptr::null(),
    }
}

/// A minimal identity-ish hasher for the pointer-keyed compiled-body cache. The
/// default `SipHash` is overkill for a `*const Ir` looked up on every call; this
/// multiplies the pointer by a 64-bit odd constant (Fibonacci hashing) for good
/// bucket spread at a fraction of the cost.
#[derive(Default)]
struct PtrHasher(u64);
impl std::hash::Hasher for PtrHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // Fallback (not hit by pointer keys): fold bytes in.
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}
type PtrBuildHasher = std::hash::BuildHasherDefault<PtrHasher>;

/// A closure this body can construct: the arity/frame metadata + shared `Ir`
/// body + capture list, registered once at compile time so the emitted code
/// passes only a small integer (plus the capture VALUES, spilled at the site).
struct ClosureTemplate {
    nparams: usize,
    variadic: bool,
    nslots: u16,
    ncaps: u16,
    body: Arc<Ir>,
}

extern "C" fn shim_load_local<M: ValueModel>(ctx: *mut JitCtx<M>, up: u32, idx: u32) -> u64 {
    let ctx = unsafe { &*ctx };
    frame_get(&ctx.cur.borrow(), up as u16, idx as u16)
}

extern "C" fn shim_set_local<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    up: u32,
    idx: u32,
    val: u64,
) -> u64 {
    let ctx = unsafe { &*ctx };
    frame_set(&ctx.cur.borrow(), up as u16, idx as u16, val);
    val // `set!` evaluates to the assigned value (matches the tree-walker)
}

extern "C" fn shim_set_global<M: ValueModel>(ctx: *mut JitCtx<M>, sym: u32, val: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    if !rt.set_global_val(sym as Sym, val) {
        panic!("set!: unbound variable: {}", rt.sym_name(sym as Sym));
    }
    val
}

extern "C" fn shim_load_global<M: ValueModel>(ctx: *mut JitCtx<M>, sym: u32) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    match rt.global(sym as Sym) {
        Some(v) => v,
        None => {
            // Catchable, as on the tree-walk tier: signal a throw and yield nil
            // (the signal is observed at the next check point / try frame).
            let id = rt.alloc(Obj::Str(format!(
                "Unable to resolve symbol: {}",
                rt.sym_name(sym as Sym)
            )));
            rt.signal_throw(M::R::enc_ref(id));
            M::R::enc_nil()
        }
    }
}

extern "C" fn shim_def_global<M: ValueModel>(ctx: *mut JitCtx<M>, sym: u32, val: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.define_global(sym as Sym, val);
    // Match the tree-walker: `def` evaluates to the (encoded) defined symbol.
    rt.encode(Val::Sym(sym as Sym))
}

/// Allocate a closure: the caller has already RESOLVED the capture values (from
/// its SSA registers / frame slots / own captures, per the template's capture
/// list) and spilled them, in order, at `caps_ptr`. The shim just copies
/// `ncaps` words into a fresh capture array.
extern "C" fn shim_make_closure<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    template_id: u32,
    caps_ptr: *const u64,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let reg = unsafe { &*(*ctx.rc).tregistry };
    let (nparams, variadic, nslots, ncaps, body) = {
        let reg = reg.borrow();
        let t = &reg[template_id as usize];
        (t.nparams, t.variadic, t.nslots, t.ncaps as usize, t.body.clone())
    };
    let caps: Caps = if ncaps == 0 {
        no_caps()
    } else {
        let vals = unsafe { std::slice::from_raw_parts(caps_ptr, ncaps) };
        vals.iter().map(|&v| AtomicU64::new(v)).collect()
    };
    let id = rt.alloc(Obj::Closure { nparams, variadic, nslots, body, caps });
    M::R::enc_ref(id)
}

/// Finish a NON-self tail call left by a natively-called fast body: it set
/// `tail_pending` + `tail_callee` + `tail_args` (via `shim_tail_call`) and
/// returned, and its stack frame is already gone, so invoking the target here
/// through `top` preserves TCO (bounded stack) and composition. The common self
/// tail case never reaches this (it loops in place); this only handles a fast body
/// tail-calling a DIFFERENT function.
extern "C" fn shim_finish_tail<M: ValueModel>(ctx: *mut JitCtx<M>) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let top = unsafe { (*ctx.rc).top };
    let callee = ctx.tail_callee.get();
    let args = unsafe { (*(*ctx.rc).tail_args).clone() };
    ctx.tail_pending.set(false);
    top.invoke(top, rt, callee, &args)
}

extern "C" fn shim_call<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    callee: u64,
    args: *const u64,
    argc: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let top = unsafe { (*ctx.rc).top };
    let args: &[u64] = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args, argc as usize) }
    };
    // Recurse through `top` so composition / macro-reentrancy hold, exactly like
    // the other tiers' `invoke` call sites.
    top.invoke(top, rt, callee, args)
}

/// Record a tail call for the trampoline instead of recursing. Returns a dummy
/// value that flows to the function's `return`; the trampoline ignores it and
/// reads the stashed callee/args.
extern "C" fn shim_tail_call<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    callee: u64,
    args: *const u64,
    argc: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let buf = unsafe { &mut *(*ctx.rc).tail_args };
    buf.clear();
    if argc != 0 {
        buf.extend_from_slice(unsafe { std::slice::from_raw_parts(args, argc as usize) });
    }
    ctx.tail_callee.set(callee);
    ctx.tail_pending.set(true);
    0
}

extern "C" fn shim_prim<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    prim_tag: u32,
    args: *const u64,
    argc: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let args: &[u64] = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args, argc as usize) }
    };
    rt.prim(prim_from_tag(prim_tag), args)
}

/// `(.-field obj)` — the inline-cached field read, same as TreeWalk's `FieldGet`.
extern "C" fn shim_field_get<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    site: u32,
    field: u32,
    obj: u64,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.field_get(site as usize, field, obj)
}

/// A protocol/method dispatch: resolve the impl for the receiver's type (with the
/// `Object` default) and invoke it through `top` — identical to TreeWalk.
extern "C" fn shim_dispatch<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    site: u32,
    method: u32,
    args: *const u64,
    argc: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let top = unsafe { (*ctx.rc).top };
    let args: &[u64] =
        if argc == 0 { &[] } else { unsafe { std::slice::from_raw_parts(args, argc as usize) } };
    let ty = rt.type_tag(args[0]);
    let imp = match rt.resolve_or_default(site as usize, method, ty) {
        Some(imp) => imp,
        None => {
            // A CATCHABLE throw (matching TreeWalk), not an abort — the caller
            // checks `pending()` after the dispatch shim and bubbles to the try.
            let msg = format!("no method '{}' for type '{}'", rt.sym_name(method), rt.sym_name(ty));
            let id = rt.alloc(Obj::Str(msg));
            rt.signal_throw(M::R::enc_ref(id));
            return M::R::enc_nil();
        }
    };
    top.invoke(top, rt, imp, args)
}

/// Register a `deftype`/protocol method impl (type-indexed dispatch table).
extern "C" fn shim_def_method<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    name: u32,
    ty: u32,
    imp: u64,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.register_method(name, ty, imp);
    rt.encode(Val::Nil)
}

/// `(apply f a … lst)` — flatten the leading args with the final list and invoke
/// `f` through `top`, exactly as the TreeWalk `Prim::Apply` arm does.
extern "C" fn shim_apply<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    args: *const u64,
    argc: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let top = unsafe { (*ctx.rc).top };
    let argv: &[u64] =
        if argc == 0 { &[] } else { unsafe { std::slice::from_raw_parts(args, argc as usize) } };
    let f = argv[0];
    let rest = &argv[1..];
    let mut flat: Vec<u64> = rest[..rest.len().saturating_sub(1)].to_vec();
    if let Some(&last) = rest.last() {
        flat.extend(rt.list_to_vec(last));
    }
    top.invoke(top, rt, f, &flat)
}

/// `(%spawn thunk)` — spawn an OS thread that runs `thunk` on the JIT. The worker
/// builds its OWN `JitCranelift` in-thread (so the JIT need not be `Send`) and
/// shares this runtime's heap/globals via the thread handle; the closure and
/// everything it calls run NATIVE on the worker. Mirrors the TreeWalk `Spawn` arm
/// but with a JIT worker instead of a tree-walker.
extern "C" fn shim_spawn<M: ModelArithJit>(ctx: *mut JitCtx<M>, thunk: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let child = rt.thread_handle();
    let slot = std::sync::Arc::new(std::sync::Mutex::new(crate::value::FutureSlot {
        handle: None,
        result: None,
    }));
    let slot_worker = slot.clone();
    let slot_obj = slot.clone();
    let handle = std::thread::spawn(move || {
        let cs = JitCranelift::<M>::new();
        let mut crt = child;
        let r = cs.invoke(&cs, &mut crt, thunk, &[]);
        // Publish the result before the worker's handle drops (see TreeWalk Spawn).
        slot_worker.lock().unwrap().result = Some(r);
        r
    });
    slot.lock().unwrap().handle = Some(handle);
    let id = rt.alloc(Obj::Future(slot_obj));
    M::R::enc_ref(id)
}

/// `(%await fut)` — join the future's worker, PARKING this thread while blocked so
/// a concurrent stop-the-world collector can proceed (the reason this is backend-
/// handled and not a plain `rt.prim`). Uses `ctx.cur` for the roots to publish.
extern "C" fn shim_await<M: ValueModel>(ctx: *mut JitCtx<M>, fut: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let locals = ctx.cur.borrow().clone();
    rt.await_future(fut, &locals)
}

/// `(gc)` — a modeled safepoint for the JIT tier. `ctx.cur` is the live frame
/// chain; `collect` walks its whole parent chain (`update_env`), so every JIT
/// local is a root and survives the move (its slot is rewritten in place). The
/// only values NOT rooted are native-register temporaries mid-expression, but a
/// `(gc)` statement has no live operands, so there are none. Concurrent `(gc)`
/// calls rendezvous via `stw_collect` (losers park + participate).
extern "C" fn shim_gc<M: ValueModel>(ctx: *mut JitCtx<M>) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let locals = ctx.cur.borrow().clone();
    rt.collect(&locals);
    M::R::enc_nil()
}

/// `(try body (catch e handler) (finally f))` — the body/catch/finally Ir nodes
/// are passed by raw pointer (they outlive the compiled code) and evaluated via
/// `top`, so this mirrors TreeWalk's `Ir::Try` EXACTLY: catch_unwind the body, on
/// a `Thrown` bind the value in a fresh one-slot frame and run the handler, run
/// finally on every path, re-raise anything else. Requires unwind info on the
/// emitted frames so the panic can cross them (enabled in the ISA flags).
extern "C" fn shim_try<M: ValueModel>(
    ctx: *mut JitCtx<M>,
    body: *const Ir,
    catch: *const Ir,
    finally: *const Ir,
    cslot: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let top = unsafe { (*ctx.rc).top };
    let locals = ctx.cur.borrow().clone();
    let body = unsafe { &*body };

    // `throw` is a flag on the runtime now, not a panic — so this is the exact
    // signal-checking logic of TreeWalk's `Ir::Try`, no unwinding involved. The
    // body/handlers run on `top` (the JIT), which propagates the flag natively via
    // the per-call pending checks; this shim just checks + routes.
    let mut result = top.eval_ir(top, rt, body, &locals);
    if rt.pending_throw() && !catch.is_null() {
        let thrown = rt.take_signal().value;
        let cbody = unsafe { &*catch };
        // The catch binding was re-homed by `flatten` to a slot of THIS
        // activation frame (no fresh frame).
        frame_set(&locals, 0, cslot as u16, thrown);
        result = top.eval_ir(top, rt, cbody, &locals);
    }
    if !finally.is_null() {
        let fbody = unsafe { &*finally };
        let suspended = rt.take_signal();
        let fv = top.eval_ir(top, rt, fbody, &locals);
        if rt.pending() {
            return fv;
        }
        rt.signal = suspended;
    }
    result
}

// A stable integer tag so the emitted code can name one. (The `Prim`
// enum carries no `#[repr]`, so this is an explicit, total mapping.)
fn prim_tag(p: Prim) -> u32 {
    use Prim::*;
    match p {
        Add => 0,
        Sub => 1,
        Mul => 2,
        Lt => 3,
        Eq => 4,
        List => 5,
        Cons => 6,
        First => 7,
        Rest => 8,
        IsNil => 9,
        Println => 10,
        Print => 67,
        MethodTypes => 68,
        Record => 11,
        Field => 12,
        Identical => 13,
        StrLen => 14,
        CharToInt => 15,
        IntToChar => 16,
        Vector => 17,
        VectorRef => 18,
        VectorSet => 19,
        VectorLen => 20,
        Values => 21,
        ValuesToList => 22,
        TypeOf => 23,
        NFields => 24,
        Throw => 25,
        Await => 26,
        Spawn => 27,
        AtomNew => 28,
        AtomGet => 29,
        AtomSet => 30,
        AtomCas => 31,
        Quot => 32,
        Rem => 33,
        Mod => 34,
        Div => 64,
        StrCat => 35,
        StrOf => 36,
        MakeArray => 37,
        AClone => 38,
        BitAnd => 39,
        BitOr => 40,
        BitXor => 41,
        BitShl => 42,
        BitShr => 43,
        BitCount => 44,
        RegisterFields => 45,
        FieldByName => 46,
        Hash => 47,
        DynGet => 48,
        DynSet => 49,
        DynMark => 50,
        DynBind => 51,
        DynUnwind => 52,
        GlobalGet => 53,
        GlobalSet => 54,
        GlobalBound => 55,
        SymName => 56,
        SymNs => 57,
        VarFlags => 58,
        NsInterns => 59,
        AllNs => 60,
        SymbolOf => 61,
        VarArglists => 62,
        StrChars => 63,
        FieldNames => 65,
        MakeRecord => 66,
        ArrPush => 69,
        ArrShift => 70,
        ArrClear => 71,
        ReadString => 72,
        Eval => 73,
        MacroExpand1 => 74,
        Numerator => 75,
        Denominator => 76,
        BigIntP => 77,
        ToLong => 78,
        StrToBytes => 79,
        BytesToStr => 80,
        TcpListen => 81,
        TcpAccept => 82,
        TcpRead => 83,
        TcpWrite => 84,
        TcpClose => 85,
        TcpLocalPort => 86,
        ErrPrint => 87,
        CurrentNs => 88,
        PvConj => 89,
        PvNth => 90,
        PvAssoc => 91,
        LazyRealize => 92,
        RangeFill => 93,
        HamtAssoc => 94,
        HamtLookup => 95,
        HamtWithout => 96,
        StrJoinArr => 97,
        StrCmp => 98,
        PvConjChunk => 99,
        PvFromArray => 100,
        ApushChunk => 101,
        MultiFnNew => 102,
        // These require a backend the JIT tier does not model; rejected at
        // compile time, so they never reach a tag. Listed for totality.
        Gc | CallEc | Apply | CallCc | Reset | Shift => {
            panic!("prim {p:?} is not lowerable by the JIT tier")
        }
        // Optimizer-introduced ops the JIT lowers INLINE (never through a shim
        // tag): `Fx*` go via `emit_unguarded_arith` after `dechecked()`, and
        // `AllFixnum` via `emit_all_fixnum`. Listed for totality.
        FxAdd | FxSub | FxMul | FxLt | FxEq | AllFixnum => {
            panic!("prim {p:?} is lowered inline by the JIT, not via a shim tag")
        }
    }
}

fn prim_from_tag(tag: u32) -> Prim {
    use Prim::*;
    match tag {
        0 => Add,
        1 => Sub,
        2 => Mul,
        3 => Lt,
        4 => Eq,
        5 => List,
        6 => Cons,
        7 => First,
        8 => Rest,
        9 => IsNil,
        10 => Println,
        67 => Print,
        68 => MethodTypes,
        11 => Record,
        12 => Field,
        13 => Identical,
        14 => StrLen,
        15 => CharToInt,
        16 => IntToChar,
        17 => Vector,
        18 => VectorRef,
        19 => VectorSet,
        20 => VectorLen,
        21 => Values,
        22 => ValuesToList,
        23 => TypeOf,
        24 => NFields,
        25 => Throw,
        26 => Await,
        27 => Spawn,
        28 => AtomNew,
        29 => AtomGet,
        30 => AtomSet,
        31 => AtomCas,
        32 => Quot,
        33 => Rem,
        34 => Mod,
        35 => StrCat,
        36 => StrOf,
        37 => MakeArray,
        38 => AClone,
        39 => BitAnd,
        40 => BitOr,
        41 => BitXor,
        42 => BitShl,
        43 => BitShr,
        44 => BitCount,
        45 => RegisterFields,
        46 => FieldByName,
        47 => Hash,
        48 => DynGet,
        49 => DynSet,
        50 => DynMark,
        51 => DynBind,
        52 => DynUnwind,
        53 => GlobalGet,
        54 => GlobalSet,
        55 => GlobalBound,
        56 => SymName,
        57 => SymNs,
        58 => VarFlags,
        59 => NsInterns,
        60 => AllNs,
        61 => SymbolOf,
        62 => VarArglists,
        63 => StrChars,
        65 => FieldNames,
        66 => MakeRecord,
        69 => ArrPush,
        70 => ArrShift,
        71 => ArrClear,
        72 => ReadString,
        73 => Eval,
        74 => MacroExpand1,
        75 => Numerator,
        76 => Denominator,
        77 => BigIntP,
        78 => ToLong,
        79 => StrToBytes,
        80 => BytesToStr,
        81 => TcpListen,
        82 => TcpAccept,
        83 => TcpRead,
        84 => TcpWrite,
        85 => TcpClose,
        86 => TcpLocalPort,
        87 => ErrPrint,
        88 => CurrentNs,
        64 => Div,
        89 => PvConj,
        90 => PvNth,
        91 => PvAssoc,
        92 => LazyRealize,
        93 => RangeFill,
        94 => HamtAssoc,
        95 => HamtLookup,
        96 => HamtWithout,
        97 => StrJoinArr,
        98 => StrCmp,
        99 => PvConjChunk,
        100 => PvFromArray,
        101 => ApushChunk,
        102 => MultiFnNew,
        other => panic!("bad prim tag {other}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The imported-shim function ids, and the per-function `FuncRef`s.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Shims {
    load_local: FuncId,
    load_global: FuncId,
    def_global: FuncId,
    make_closure: FuncId,
    call: FuncId,
    tail_call: FuncId,
    finish_tail: FuncId,
    prim: FuncId,
    set_local: FuncId,
    set_global: FuncId,
    field_get: FuncId,
    dispatch: FuncId,
    def_method: FuncId,
    apply: FuncId,
    try_: FuncId,
    spawn: FuncId,
    await_: FuncId,
    gc: FuncId,
}

#[derive(Clone, Copy)]
struct ShimRefs {
    load_local: cranelift_codegen::ir::FuncRef,
    load_global: cranelift_codegen::ir::FuncRef,
    def_global: cranelift_codegen::ir::FuncRef,
    make_closure: cranelift_codegen::ir::FuncRef,
    call: cranelift_codegen::ir::FuncRef,
    tail_call: cranelift_codegen::ir::FuncRef,
    finish_tail: cranelift_codegen::ir::FuncRef,
    prim: cranelift_codegen::ir::FuncRef,
    set_local: cranelift_codegen::ir::FuncRef,
    set_global: cranelift_codegen::ir::FuncRef,
    field_get: cranelift_codegen::ir::FuncRef,
    dispatch: cranelift_codegen::ir::FuncRef,
    def_method: cranelift_codegen::ir::FuncRef,
    apply: cranelift_codegen::ir::FuncRef,
    try_: cranelift_codegen::ir::FuncRef,
    spawn: cranelift_codegen::ir::FuncRef,
    await_: cranelift_codegen::ir::FuncRef,
    gc: cranelift_codegen::ir::FuncRef,
}

/// A finished, runnable body.
struct Compiled {
    code: *const u8,
    /// `Some(k)` — a REGISTER-ARG entry `fn(*mut JitCtx, a0..a{k-1}) -> u64`
    /// (SSA-mode, non-variadic, k ≤ MAX_REG_ARGS): no activation frame is ever
    /// built for a call. `None` — the ctx-only entry `fn(*mut JitCtx) -> u64`:
    /// the caller builds a real heap frame (variadic / many params / memory
    /// mode) and the prologue reads it.
    entry_arity: Option<usize>,
    /// MEMORY mode: the body contains `try`/`(gc)`/`%await`, so its locals live
    /// in the heap activation frame (GC-rootable, interpreter-visible via
    /// `shim_try`) and `ctx.cur` must hold the frame handle. SSA mode keeps
    /// every local in a register.
    mem_mode: bool,
}

/// Is there a `Call`/`Dispatch` in TAIL position of `ir`? Such a body may set
/// `tail_pending` and return; native call sites finish it via
/// `shim_finish_tail`, the invoke trampoline by bouncing.
fn has_tail_call(ir: &Ir) -> bool {
    match ir {
        Ir::Call(..) | Ir::Dispatch { .. } => true,
        Ir::If(_, t, e) => has_tail_call(t) || has_tail_call(e),
        Ir::Do(xs) => xs.last().is_some_and(has_tail_call),
        _ => false,
    }
}

/// MEMORY mode? True iff the body (not descending into nested `Lambda` bodies —
/// those compile separately) contains a construct whose locals must be visible
/// outside the compiled code: `try` (the interpreter runs the catch against the
/// frame), `(gc)` / `%await` (the frame is the GC root set for the parked
/// thread). Everything else runs its locals in SSA registers.
fn body_mem_mode(ir: &Ir) -> bool {
    match ir {
        Ir::Try { .. } => true,
        Ir::Prim(Prim::Await | Prim::Gc, _) => true,
        Ir::Lambda { .. } => false,
        Ir::Const(_) | Ir::Quote(_) | Ir::Global(_) | Ir::Local { .. } | Ir::Capture(_) => false,
        Ir::SetLocal { val, .. } | Ir::SetGlobal { val, .. } => body_mem_mode(val),
        Ir::If(a, b, c) => body_mem_mode(a) || body_mem_mode(b) || body_mem_mode(c),
        Ir::Do(xs) | Ir::Prim(_, xs) => xs.iter().any(body_mem_mode),
        Ir::Call(f, args) => body_mem_mode(f) || args.iter().any(body_mem_mode),
        Ir::Def { init, .. } => body_mem_mode(init),
        Ir::DefMethod { imp, .. } => body_mem_mode(imp),
        Ir::Dispatch { args, .. } => args.iter().any(body_mem_mode),
        Ir::FieldGet { obj, .. } => body_mem_mode(obj),
        Ir::Let(..) => panic!("unflattened Ir reached the JIT: Let"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The numeric-overflow axis, emit half.
//
// The `ModelEmit` recipe is the fixnum FAST PATH: it wraps on overflow and
// assumes both operands are immediate ints, exactly like the bytecode tier. To
// match the tree-walker's numeric tower (promote to BigInt on overflow, handle
// floats/mixed), the JIT wraps that fast path in a guard: take it only when both
// operands are immediate fixnums AND the result fits fixnum range; otherwise call
// the runtime's checked+promoting `prim`. This trait supplies the model-specific
// pieces the guard needs (tag test, untag, retag). It is the emit form of
// codegen axis #2.
// ─────────────────────────────────────────────────────────────────────────

/// The model's contribution to guarded native arithmetic: how to test for /
/// unwrap / rewrap an immediate integer. Fixnum range is shared (±2^60).
pub trait ModelArithJit: ModelEmit {
    /// Emit: are BOTH `a` and `b` immediate integers of this model? Returns a
    /// branch-ready bool. Folding the two operand tests into one `(a|b)`-based
    /// check halves the guard's tag work on the hot path. (For a model that boxes
    /// integers this is constant-`false`, so the guard always takes the runtime
    /// slow path — correct, since there is no fast one.)
    fn emit_both_int(c: &mut Compiler, a: Value, b: Value) -> Value;
    /// Emit: the signed i64 VALUE of immediate int `v` (untagged).
    fn emit_untag(c: &mut Compiler, v: Value) -> Value;
    /// Emit: encode signed i64 `x` back into an immediate int word.
    fn emit_tag(c: &mut Compiler, x: Value) -> Value;
    /// Emit `(is_ref, heap_id)` for `v`: `is_ref` is a branch-ready bool, `id` is
    /// meaningful only when it holds. Used to resolve a native call target inline.
    /// A model may return a constant-false `is_ref` to opt out of native fast
    /// calls entirely (correct — the call site then always takes the shim path).
    fn emit_ref_id(c: &mut Compiler, v: Value) -> (Value, Value);
}

const FIXNUM_MIN: i64 = -(1 << 60);
const FIXNUM_MAX: i64 = (1 << 60) - 1;

impl ModelArithJit for crate::model::LowBitModel {
    fn emit_both_int(c: &mut Compiler, a: Value, b: Value) -> Value {
        // both have low 3 bits clear  <=>  (a|b) & 7 == 0
        let or = c.fb.ins().bor(a, b);
        let masked = c.fb.ins().band_imm(or, 0b111);
        c.fb.ins().icmp_imm(IntCC::Equal, masked, 0)
    }
    fn emit_untag(c: &mut Compiler, v: Value) -> Value {
        c.fb.ins().sshr_imm(v, 3) // arithmetic shift drops the tag, keeps sign
    }
    fn emit_tag(c: &mut Compiler, x: Value) -> Value {
        c.fb.ins().ishl_imm(x, 3) // tag bits are 0 for an int
    }
    fn emit_ref_id(c: &mut Compiler, v: Value) -> (Value, Value) {
        // LowBit ref: tag `LB_REF` = 0b001 in the low 3 bits, id in the rest.
        let tag = c.fb.ins().band_imm(v, 0b111);
        let is_ref = c.fb.ins().icmp_imm(IntCC::Equal, tag, 0b001);
        let id = c.fb.ins().ushr_imm(v, 3);
        (is_ref, id)
    }
}

impl ModelArithJit for crate::model::HighBitModel {
    fn emit_both_int(c: &mut Compiler, a: Value, b: Value) -> Value {
        // both have top 3 bits clear  <=>  (a|b) >> 61 == 0
        let or = c.fb.ins().bor(a, b);
        let hi = c.fb.ins().ushr_imm(or, 61);
        c.fb.ins().icmp_imm(IntCC::Equal, hi, 0)
    }
    fn emit_untag(c: &mut Compiler, v: Value) -> Value {
        // sign-extend the low 61 bits: (v << 3) >>signed 3
        let sh = c.fb.ins().ishl_imm(v, 3);
        c.fb.ins().sshr_imm(sh, 3)
    }
    fn emit_tag(c: &mut Compiler, x: Value) -> Value {
        // keep the low 61 bits (top 3 = tag 0), matching `enc_int`
        c.fb.ins().band_imm(x, (1i64 << 61) - 1)
    }
    fn emit_ref_id(c: &mut Compiler, _v: Value) -> (Value, Value) {
        // Opt out of native fast calls under HighBit (always take the shim path).
        let f = c.fb.ins().iconst(cranelift_codegen::ir::types::I8, 0);
        let z = c.fb.ins().iconst(I64, 0);
        (f, z)
    }
}

impl ModelArithJit for crate::model::NanBoxModel {
    fn emit_both_int(c: &mut Compiler, _a: Value, _b: Value) -> Value {
        // integers are boxed under NaN-boxing: never a fast path
        c.fb.ins().iconst(cranelift_codegen::ir::types::I8, 0)
    }
    fn emit_untag(_c: &mut Compiler, v: Value) -> Value {
        v // unreachable (guard is always false); kept well-typed
    }
    fn emit_tag(_c: &mut Compiler, x: Value) -> Value {
        x
    }
    fn emit_ref_id(c: &mut Compiler, _v: Value) -> (Value, Value) {
        // Opt out of native fast calls under NaN-boxing (always take the shim path).
        let f = c.fb.ins().iconst(cranelift_codegen::ir::types::I8, 0);
        let z = c.fb.ins().iconst(I64, 0);
        (f, z)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The backend.
// ─────────────────────────────────────────────────────────────────────────

pub struct JitCranelift<M: ModelArithJit> {
    module: RefCell<JITModule>,
    fbctx: RefCell<FunctionBuilderContext>,
    shims: Shims,
    /// Compiled closure bodies, keyed by the `Arc<Ir>` identity (compile-once,
    /// like the bytecode tier's chunk cache).
    cache: RefCell<HashMap<*const Ir, Arc<Compiled>, PtrBuildHasher>>,
    counter: Cell<u32>,
    /// A free list of `Arc<Frame>` whose call has returned without being captured
    /// (`strong_count == 1`). Refilled via `Arc::get_mut` on the next call instead
    /// of allocating — so deep recursion converges to ~depth frames and a tail
    /// loop reuses one, cutting the per-call heap traffic that was the gap to a
    /// production compiler. Bounded so it never holds unbounded memory.
    frame_pool: RefCell<Vec<Arc<Frame>>>,
    /// An 8-way direct-mapped inline cache of resolved callees, keyed on the
    /// callee's bits. Call sites are overwhelmingly monomorphic, but the shim
    /// path serves MANY sites (dispatch impls, variadic fns, trampoline
    /// bounces), so a single entry thrashes; eight direct-mapped ways keep the
    /// hot mix resident. A hit skips decode + heap lookup + compile-cache hash.
    call_ic: RefCell<[Option<CallTarget>; CALL_IC_WAYS]>,
    /// Dense, heap-id-indexed table of NATIVE fast-call entries: for a closure
    /// whose body is directly callable (see `FastTarget`), the call site jumps to
    /// its compiled code with `call_indirect`, passing args in registers — no
    /// FFI, no `invoke`, no frame. Sized once per top-level form (to
    /// `heap.len()`), so its base is stable for the whole run; filled lazily on
    /// the slow path. `code == null` (the default) means "not a fast target, use
    /// the shim path", which is what keeps continuations / wrappers composable.
    fast_targets: RefCell<Vec<FastTarget>>,
    /// The GLOBAL closure-template registry: every `Lambda` site of every
    /// compiled body appends here once, at compile time; `shim_make_closure`
    /// reads it through the run context. Append-only, bounded by code size.
    templates: Box<RefCell<Vec<ClosureTemplate>>>,
    _pd: std::marker::PhantomData<fn() -> M>,
}

/// One entry of the fast-call table, keyed by closure heap id. `code` is the
/// compiled body's REGISTER-ARG entry `fn(*mut JitCtx, a0..a{n-1}) -> u64`; the
/// call site builds a minimal 4-store `JitCtx` on its own stack, passes the args
/// in registers, and jumps to it. `caps` is THIS closure instance's capture
/// array base (stable — the `Arc<[AtomicU64]>` allocation never moves, and the
/// GC's to-space clone shares it), stored so the call site can hand the callee
/// its captures without touching the heap object. Eligible bodies are SSA-mode
/// (no try/(gc)/%await) and non-variadic with ≤ MAX_REG_ARGS params; everything
/// else keeps `code == null` and goes through the shim path.
#[repr(C)]
#[derive(Clone, Copy)]
struct FastTarget {
    code: *const u8,
    caps: *const u64,
    /// nparams; the call-site guard is one compare against the arg count.
    arity: u32,
    _pad: u32,
}

impl Default for FastTarget {
    fn default() -> Self {
        FastTarget { code: std::ptr::null(), caps: std::ptr::null(), arity: 0, _pad: 0 }
    }
}

/// Most args that ride in a register-arg entry signature. Bodies with more
/// params (rare) use the ctx-entry + frame-loads prologue instead.
const MAX_REG_ARGS: usize = 8;

/// Speculative-inlining limits: a callee body inlines only when it is at most
/// this many Ir nodes, and one compiled body stops inlining after this total
/// (which is also what terminates recursive inlining — each nested inline
/// strictly consumes budget).
const INLINE_MAX_CALLEE_NODES: usize = 64;
const INLINE_TOTAL_BUDGET: usize = 600;

/// Count Ir nodes up to `limit`; `None` if the tree is bigger. (Not descending
/// into nested `Lambda` bodies — they compile separately and only cost their
/// closure-creation site here.)
fn node_count_capped(ir: &Ir, limit: usize) -> Option<usize> {
    fn walk(ir: &Ir, left: &mut isize) -> bool {
        *left -= 1;
        if *left < 0 {
            return false;
        }
        match ir {
            Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Capture(_) | Ir::Global(_)
            | Ir::Lambda { .. } => true,
            Ir::SetLocal { val, .. } | Ir::SetGlobal { val, .. } => walk(val, left),
            Ir::If(a, b, c) => walk(a, left) && walk(b, left) && walk(c, left),
            Ir::Do(xs) | Ir::Prim(_, xs) => xs.iter().all(|x| walk(x, left)),
            Ir::Def { init, .. } => walk(init, left),
            Ir::Call(f, args) => walk(f, left) && args.iter().all(|x| walk(x, left)),
            Ir::DefMethod { imp, .. } => walk(imp, left),
            Ir::Dispatch { args, .. } => args.iter().all(|x| walk(x, left)),
            Ir::FieldGet { obj, .. } => walk(obj, left),
            Ir::Try { body, catch, finally, .. } => {
                walk(body, left)
                    && catch.as_ref().is_none_or(|c| walk(c, left))
                    && finally.as_ref().is_none_or(|f| walk(f, left))
            }
            Ir::Let(..) => false, // unflattened — never inline
        }
    }
    let mut left = limit as isize;
    if walk(ir, &mut left) {
        Some((limit as isize - left) as usize)
    } else {
        None
    }
}

/// What the speculative inliner needs to splice a known global callee into a
/// call site: the guard bits and the callee's (capture-free) body + frame size.
struct InlinePlan {
    bits: u64,
    nparams: usize,
    nslots: u16,
    body: Arc<Ir>,
}

/// Extra `FastTarget` slots reserved past the live heap size at each top-level
/// form's start, so heap objects THIS form allocates while running still land
/// in range (see the comment at the `eval_ir` resize site). 4M slots * 24 bytes
/// = 96MiB, paid once (the table only ever grows), amortized over the process's
/// whole lifetime — cheap next to what it unlocks (fast-calling a `map`/`filter`
/// callback allocated mid-form instead of every call going through `shim_call`).
const FAST_TARGET_SLACK: usize = 4_000_000;

/// A resolved call target (the payload of the monomorphic inline cache).
struct CallTarget {
    callee: u64,
    nparams: usize,
    variadic: bool,
    nslots: u16,
    compiled: Arc<Compiled>,
    caps: Caps,
}

/// Cap on pooled frames — plenty for realistic recursion depth, bounded memory.
const FRAME_POOL_CAP: usize = 1024;

/// Ways in the direct-mapped resolved-callee cache.
const CALL_IC_WAYS: usize = 8;

/// The cache way for a callee: heap ids are the high bits, so shift past the
/// tag and fold.
fn call_ic_way(callee: u64) -> usize {
    ((callee >> 3) as usize) & (CALL_IC_WAYS - 1)
}

impl<M: ModelArithJit> JitCranelift<M> {
    pub fn new() -> Self {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").unwrap();
        flags.set("is_pic", "false").unwrap();
        // Speed of compilation matters more than of the code for a first tier.
        // Bodies compile once and run many times, so optimize the emitted code
        // (register allocation, redundancy elimination) — worth the compile cost.
        flags.set("opt_level", "speed").unwrap();
        // The IR verifier is a development aid; it showed up in profiles once the
        // speculative inliner grew per-body code. The test suites are the
        // correctness gate — skip verification in the built product.
        flags.set("enable_verifier", "false").unwrap();
        let isa = cranelift_native::builder()
            .expect("host machine is not supported by Cranelift")
            .finish(settings::Flags::new(flags))
            .expect("failed to build Cranelift ISA");

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // Register the monomorphised shim addresses under stable names. Casting
        // each `extern "C"` fn pointer to a raw address is what lets the JIT link
        // the emitted `call` to concrete host code.
        let ll: extern "C" fn(*mut JitCtx<M>, u32, u32) -> u64 = shim_load_local::<M>;
        let lg: extern "C" fn(*mut JitCtx<M>, u32) -> u64 = shim_load_global::<M>;
        let dg: extern "C" fn(*mut JitCtx<M>, u32, u64) -> u64 = shim_def_global::<M>;
        let mk: extern "C" fn(*mut JitCtx<M>, u32, *const u64) -> u64 = shim_make_closure::<M>;
        let ca: extern "C" fn(*mut JitCtx<M>, u64, *const u64, u32) -> u64 = shim_call::<M>;
        let tc: extern "C" fn(*mut JitCtx<M>, u64, *const u64, u32) -> u64 = shim_tail_call::<M>;
        let ft: extern "C" fn(*mut JitCtx<M>) -> u64 = shim_finish_tail::<M>;
        let pr: extern "C" fn(*mut JitCtx<M>, u32, *const u64, u32) -> u64 = shim_prim::<M>;
        let sl: extern "C" fn(*mut JitCtx<M>, u32, u32, u64) -> u64 = shim_set_local::<M>;
        let sg: extern "C" fn(*mut JitCtx<M>, u32, u64) -> u64 = shim_set_global::<M>;
        let fget: extern "C" fn(*mut JitCtx<M>, u32, u32, u64) -> u64 = shim_field_get::<M>;
        let disp: extern "C" fn(*mut JitCtx<M>, u32, u32, *const u64, u32) -> u64 = shim_dispatch::<M>;
        let dmeth: extern "C" fn(*mut JitCtx<M>, u32, u32, u64) -> u64 = shim_def_method::<M>;
        let apl: extern "C" fn(*mut JitCtx<M>, *const u64, u32) -> u64 = shim_apply::<M>;
        let tryf: extern "C" fn(*mut JitCtx<M>, *const Ir, *const Ir, *const Ir, u32) -> u64 = shim_try::<M>;
        let spwn: extern "C" fn(*mut JitCtx<M>, u64) -> u64 = shim_spawn::<M>;
        let awt: extern "C" fn(*mut JitCtx<M>, u64) -> u64 = shim_await::<M>;
        let gcf: extern "C" fn(*mut JitCtx<M>) -> u64 = shim_gc::<M>;
        builder.symbol("ml_load_local", ll as *const u8);
        builder.symbol("ml_load_global", lg as *const u8);
        builder.symbol("ml_def_global", dg as *const u8);
        builder.symbol("ml_make_closure", mk as *const u8);
        builder.symbol("ml_call", ca as *const u8);
        builder.symbol("ml_tail_call", tc as *const u8);
        builder.symbol("ml_finish_tail", ft as *const u8);
        builder.symbol("ml_prim", pr as *const u8);
        builder.symbol("ml_set_local", sl as *const u8);
        builder.symbol("ml_set_global", sg as *const u8);
        builder.symbol("ml_field_get", fget as *const u8);
        builder.symbol("ml_dispatch", disp as *const u8);
        builder.symbol("ml_def_method", dmeth as *const u8);
        builder.symbol("ml_apply", apl as *const u8);
        builder.symbol("ml_try", tryf as *const u8);
        builder.symbol("ml_spawn", spwn as *const u8);
        builder.symbol("ml_await", awt as *const u8);
        builder.symbol("ml_gc", gcf as *const u8);

        let mut module = JITModule::new(builder);

        // Declare each shim once as an imported function with its ABI signature.
        let sig = |params: &[cranelift_codegen::ir::Type]| {
            let mut s = module.make_signature();
            for &p in params {
                s.params.push(AbiParam::new(p));
            }
            s.returns.push(AbiParam::new(I64));
            s
        };
        let ptr = module.target_config().pointer_type();
        let s_load_local = sig(&[ptr, cranelift_codegen::ir::types::I32, cranelift_codegen::ir::types::I32]);
        let s_load_global = sig(&[ptr, cranelift_codegen::ir::types::I32]);
        let s_def_global = sig(&[ptr, cranelift_codegen::ir::types::I32, I64]);
        let s_make_closure = sig(&[ptr, cranelift_codegen::ir::types::I32, ptr]);
        let s_call = sig(&[ptr, I64, ptr, cranelift_codegen::ir::types::I32]);
        let s_prim = sig(&[ptr, cranelift_codegen::ir::types::I32, ptr, cranelift_codegen::ir::types::I32]);
        let s_finish_tail = sig(&[ptr]);
        let s_set_local = sig(&[ptr, cranelift_codegen::ir::types::I32, cranelift_codegen::ir::types::I32, I64]);
        let s_set_global = sig(&[ptr, cranelift_codegen::ir::types::I32, I64]);
        let i32t = cranelift_codegen::ir::types::I32;
        let s_field_get = sig(&[ptr, i32t, i32t, I64]);
        let s_dispatch = sig(&[ptr, i32t, i32t, ptr, i32t]);
        let s_def_method = sig(&[ptr, i32t, i32t, I64]);
        let s_apply = sig(&[ptr, ptr, i32t]);
        let s_try = sig(&[ptr, ptr, ptr, ptr, i32t]);
        let s_spawn = sig(&[ptr, I64]);
        let s_await = sig(&[ptr, I64]);
        let s_gc = sig(&[ptr]);

        let decl = |module: &mut JITModule, name: &str, sig: &cranelift_codegen::ir::Signature| {
            module
                .declare_function(name, Linkage::Import, sig)
                .expect("declare shim import")
        };
        let shims = Shims {
            load_local: decl(&mut module, "ml_load_local", &s_load_local),
            load_global: decl(&mut module, "ml_load_global", &s_load_global),
            def_global: decl(&mut module, "ml_def_global", &s_def_global),
            make_closure: decl(&mut module, "ml_make_closure", &s_make_closure),
            call: decl(&mut module, "ml_call", &s_call),
            tail_call: decl(&mut module, "ml_tail_call", &s_call),
            finish_tail: decl(&mut module, "ml_finish_tail", &s_finish_tail),
            prim: decl(&mut module, "ml_prim", &s_prim),
            set_local: decl(&mut module, "ml_set_local", &s_set_local),
            set_global: decl(&mut module, "ml_set_global", &s_set_global),
            field_get: decl(&mut module, "ml_field_get", &s_field_get),
            dispatch: decl(&mut module, "ml_dispatch", &s_dispatch),
            def_method: decl(&mut module, "ml_def_method", &s_def_method),
            apply: decl(&mut module, "ml_apply", &s_apply),
            try_: decl(&mut module, "ml_try", &s_try),
            spawn: decl(&mut module, "ml_spawn", &s_spawn),
            await_: decl(&mut module, "ml_await", &s_await),
            gc: decl(&mut module, "ml_gc", &s_gc),
        };

        JitCranelift {
            module: RefCell::new(module),
            fbctx: RefCell::new(FunctionBuilderContext::new()),
            shims,
            cache: RefCell::new(HashMap::default()),
            frame_pool: RefCell::new(Vec::new()),
            call_ic: RefCell::new(std::array::from_fn(|_| None)),
            fast_targets: RefCell::new(Vec::new()),
            templates: Box::new(RefCell::new(Vec::new())),
            counter: Cell::new(0),
            _pd: std::marker::PhantomData,
        }
    }

    /// How many distinct bodies have been compiled to native code (a compile-once
    /// counter, like the bytecode tier's `compiled_bodies`).
    pub fn compiled_bodies(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Human-readable Cranelift IR for one expression — evidence the emitted code
    /// differs by value model, the native mirror of `BytecodeVm::disassemble`.
    pub fn dump_ir(&self, ir: &Ir) -> String {
        let mut module = self.module.borrow_mut();
        let mut fbctx = self.fbctx.borrow_mut();
        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(module.target_config().pointer_type()));
        ctx.func.signature.returns.push(AbiParam::new(I64));
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            let shape = BodyShape { entry_arity: None, nparams: 0, variadic: false, nslots: 0, mem_mode: true, tail_root: true };
            build_body::<M>(&mut module, &mut fb, self.shims, None, ir, &self.templates, shape);
            fb.finalize();
        }
        let out = ctx.func.display().to_string();
        module.clear_context(&mut ctx);
        out
    }

    fn compile(&self, rt: Option<&Runtime<M>>, ir: &Ir, shape: BodyShape) -> Arc<Compiled> {
        let mut module = self.module.borrow_mut();
        let mut fbctx = self.fbctx.borrow_mut();
        let mut ctx = module.make_context();
        let ptr = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr));
        for _ in 0..shape.entry_arity.unwrap_or(0) {
            ctx.func.signature.params.push(AbiParam::new(I64));
        }
        ctx.func.signature.returns.push(AbiParam::new(I64));

        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            build_body::<M>(&mut module, &mut fb, self.shims, rt, ir, &self.templates, shape);
            fb.finalize();
        }

        let n = self.counter.get();
        self.counter.set(n + 1);
        let name = format!("ml_body_{n}");
        let id = module
            .declare_function(&name, Linkage::Local, &ctx.func.signature)
            .expect("declare body");
        module.define_function(id, &mut ctx).expect("define body");
        module.clear_context(&mut ctx);
        module.finalize_definitions().expect("finalize");
        let code = module.get_finalized_function(id);
        Arc::new(Compiled { code, entry_arity: shape.entry_arity, mem_mode: shape.mem_mode })
    }

    /// The shape a Lambda body compiles to (see `Compiled`): register-arg SSA
    /// entry when possible, ctx entry (+ frame-load prologue) otherwise, memory
    /// mode when the body needs its locals visible outside compiled code.
    fn body_shape(body: &Ir, nparams: usize, variadic: bool, nslots: u16) -> BodyShape {
        let mem_mode = body_mem_mode(body);
        let entry_arity = if !mem_mode && !variadic && nparams <= MAX_REG_ARGS {
            Some(nparams)
        } else {
            None
        };
        BodyShape { entry_arity, nparams, variadic, nslots, mem_mode, tail_root: true }
    }

    fn compiled_body(&self, rt: &Runtime<M>, body: &Arc<Ir>, nparams: usize, variadic: bool, nslots: u16) -> Arc<Compiled> {
        let key = Arc::as_ptr(body);
        if let Some(c) = self.cache.borrow().get(&key) {
            return c.clone();
        }
        let c = self.compile(Some(rt), body, Self::body_shape(body, nparams, variadic, nslots));
        self.cache.borrow_mut().insert(key, c.clone());
        c
    }

    /// Resolve a callee through the monomorphic inline cache (the common repeat
    /// case skips decode + heap read + cache lookups entirely).
    fn resolve_call(&self, rt: &mut Runtime<M>, callee: u64) -> (usize, bool, u16, Arc<Compiled>, Caps) {
        let way = call_ic_way(callee);
        if let Some(t) = self.call_ic.borrow()[way].as_ref() {
            if t.callee == callee {
                return (t.nparams, t.variadic, t.nslots, t.compiled.clone(), t.caps.clone());
            }
        }
        let Val::Ref(id) = rt.decode(callee) else {
            panic!("value not callable: {}", rt.print(callee));
        };
        let (nparams, variadic, nslots, body, caps) = match &rt.heap()[id as usize] {
            Obj::Closure { nparams, variadic, nslots, body, caps } => {
                (*nparams, *variadic, *nslots, body.clone(), caps.clone())
            }
            _ => panic!("value not callable: {}", rt.print(callee)),
        };
        let compiled = self.compiled_body(rt, &body, nparams, variadic, nslots);
        // (MultiFn callees never reach here: `invoke`/the trampoline select the
        // arity clause BEFORE resolution, so `callee` is always a closure.)
        self.call_ic.borrow_mut()[way] = Some(CallTarget {
            callee,
            nparams,
            variadic,
            nslots,
            compiled: compiled.clone(),
            caps: caps.clone(),
        });
        (nparams, variadic, nslots, compiled, caps)
    }

    /// Build the per-run `JitCtx` for one body execution. `caps` is the running
    /// closure's capture array (empty for a top-level expression).
    #[allow(clippy::too_many_arguments)]
    fn make_ctx<'a>(
        &'a self,
        top: &'a dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        frame: &Locals,
        caps: &Caps,
        args_buf: &mut Vec<u64>,
        self_closure: u64,
        needs_cur: bool,
    ) -> JitCtx<'a, M> {
        let consts_base = rt.consts_ptr();
        let global_base = rt.global_slots_ptr();
        let global_len = rt.global_slots_len();
        let (fast_base, fast_len) = {
            let t = self.fast_targets.borrow();
            (t.as_ptr(), t.len())
        };
        // Native fast calls bypass `top`, so only enable them when we ARE the top
        // (no wrapper is observing calls).
        let direct = std::ptr::eq(
            top as *const dyn CodeSpace<M> as *const u8,
            self as *const JitCranelift<M> as *const u8,
        ) as u8;
        let cur = if needs_cur { frame.clone() } else { None };
        JitCtx {
            rc: std::ptr::null(),
            top,
            rt: rt as *mut Runtime<M>,
            cur_slots: Cell::new(slots_ptr(frame)),
            consts_base: Cell::new(consts_base),
            global_base: Cell::new(global_base),
            global_len: Cell::new(global_len),
            fast_base: Cell::new(fast_base),
            fast_len: Cell::new(fast_len),
            direct: Cell::new(direct),
            self_closure: Cell::new(self_closure),
            caps_base: Cell::new(if caps.is_empty() {
                std::ptr::null()
            } else {
                caps.as_ptr() as *const u64
            }),
            cur: RefCell::new(cur),
            tregistry: &*self.templates as *const RefCell<Vec<ClosureTemplate>>,
            tail_pending: Cell::new(false),
            tail_callee: Cell::new(0),
            tail_args: args_buf as *mut Vec<u64>,
        }
    }

    /// Run one CTX-ENTRY body (`fn(*mut JitCtx) -> u64`): variadic / many-param /
    /// memory-mode bodies, and top-level expressions. `Ok(value)` on a normal
    /// return; `Err(callee)` on a tail call, args left in `args_buf`.
    fn run_ctx_entry(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        compiled: &Compiled,
        frame: &Locals,
        caps: &Caps,
        args_buf: &mut Vec<u64>,
        self_closure: u64,
    ) -> Result<u64, u64> {
        let mut ctx =
            self.make_ctx(top, rt, frame, caps, args_buf, self_closure, compiled.mem_mode);
        ctx.rc = &ctx as *const JitCtx<M>;
        let f: extern "C" fn(*mut JitCtx<M>) -> u64 =
            unsafe { std::mem::transmute::<*const u8, _>(compiled.code) };
        let ret = f(&mut ctx as *mut JitCtx<M>);
        if ctx.tail_pending.get() {
            Err(ctx.tail_callee.get())
        } else {
            Ok(ret)
        }
    }

    /// Run one REGISTER-ARG body: args in registers, NO activation frame at all.
    fn run_reg_entry(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        compiled: &Compiled,
        arity: usize,
        args: &[u64; MAX_REG_ARGS],
        caps: &Caps,
        args_buf: &mut Vec<u64>,
        self_closure: u64,
    ) -> Result<u64, u64> {
        let none: Locals = None;
        let mut ctx = self.make_ctx(top, rt, &none, caps, args_buf, self_closure, false);
        ctx.rc = &ctx as *const JitCtx<M>;
        // Pass the context pointer as a plain word so the per-arity fn-pointer
        // types need no lifetime parameters.
        let cp = &mut ctx as *mut JitCtx<M> as u64;
        let code = compiled.code;
        let a = args;
        let ret = unsafe {
            match arity {
                0 => std::mem::transmute::<*const u8, extern "C" fn(u64) -> u64>(code)(cp),
                1 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64) -> u64>(code)(cp, a[0]),
                2 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64) -> u64>(code)(cp, a[0], a[1]),
                3 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2]),
                4 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3]),
                5 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4]),
                6 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5]),
                7 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5], a[6]),
                8 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]),
                _ => unreachable!("register entries are capped at MAX_REG_ARGS"),
            }
        };
                if ctx.tail_pending.get() {
            Err(ctx.tail_callee.get())
        } else {
            Ok(ret)
        }
    }

    /// Build a callee frame for a CTX-ENTRY body, reusing a pooled `Arc<Frame>`
    /// if one is free (no allocation) and otherwise falling back to
    /// `Runtime::build_call_frame`. The pool only ever holds uniquely-owned
    /// frames, so `Arc::get_mut` never fails.
    fn alloc_frame(
        &self,
        rt: &mut Runtime<M>,
        nparams: usize,
        variadic: bool,
        nslots: u16,
        args: &[u64],
        caps: Caps,
    ) -> Locals {
        let popped = self.frame_pool.borrow_mut().pop();
        let mut rc = match popped {
            Some(rc) => rc,
            None => return rt.build_call_frame(nparams, variadic, nslots, args, caps),
        };
        let f = Arc::get_mut(&mut rc).expect("pooled frame is uniquely owned");
        f.slots.clear();
        if variadic {
            assert!(args.len() >= nparams, "arity: expected at least {nparams}, got {}", args.len());
            f.slots.extend(args[..nparams].iter().map(|&a| AtomicU64::new(a)));
            let rest = rt.vec_to_list(&args[nparams..]);
            f.slots.push(AtomicU64::new(rest));
        } else {
            assert!(args.len() == nparams, "arity: expected {nparams}, got {}", args.len());
            f.slots.extend(args.iter().map(|&a| AtomicU64::new(a)));
        }
        let nil = M::R::enc_nil();
        while f.slots.len() < nslots as usize {
            f.slots.push(AtomicU64::new(nil));
        }
        f.caps = caps;
        Some(rc)
    }

    /// Return a frame to the pool IF its call did not capture it. With flat
    /// closures nothing captures frames anymore, but a CEK continuation (via a
    /// routed body) still can — `Arc::get_mut` succeeding is the exact, sound
    /// test for "no other owner".
    fn recycle(&self, frame: Locals) {
        if let Some(mut rc) = frame {
            if let Some(f) = Arc::get_mut(&mut rc) {
                f.slots.clear();
                f.caps = no_caps(); // don't pin a capture array alive in the pool
                let mut pool = self.frame_pool.borrow_mut();
                if pool.len() < FRAME_POOL_CAP {
                    pool.push(rc);
                }
            }
        }
    }

    /// Publish a closure's native fast entry so emitted call sites can jump to
    /// its compiled code directly (once; a filled slot is left as-is).
    fn publish_fast_target(&self, callee: u64, compiled: &Compiled, caps: &Caps) {
        let Some(arity) = compiled.entry_arity else { return };
        let id = M::R::as_ref(callee) as usize;
        if let Some(slot) = self.fast_targets.borrow_mut().get_mut(id) {
            if slot.code.is_null() {
                slot.code = compiled.code;
                slot.caps = if caps.is_empty() {
                    std::ptr::null()
                } else {
                    caps.as_ptr() as *const u64
                };
                slot.arity = arity as u32;
            }
        }
    }

    /// The proper-tail-call trampoline: run a body; if it tail-calls, resolve the
    /// callee and loop — a bounded native stack for unbounded tail recursion.
    /// Non-tail calls still recurse (through `top` or native fast calls), so
    /// composition and macro-reentrancy hold, exactly like the tree-walker.
    #[allow(clippy::too_many_arguments)]
    fn run_trampoline(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        mut compiled: Arc<Compiled>,
        mut frame: Locals,
        mut caps: Caps,
        mut cur_callee: u64,
        mut cur_nparams: usize,
        mut cur_variadic: bool,
        mut cur_nslots: u16,
        first_args: &[u64],
    ) -> u64 {
        let mut args_buf: Vec<u64> = first_args.to_vec();
        loop {
            let outcome = if let Some(k) = compiled.entry_arity {
                debug_assert_eq!(args_buf.len(), k);
                let mut regs = [0u64; MAX_REG_ARGS];
                regs[..k].copy_from_slice(&args_buf[..k]);
                self.run_reg_entry(top, rt, &compiled, k, &regs, &caps, &mut args_buf, cur_callee)
            } else {
                // Ctx entry: needs a real activation frame.
                if frame.is_none() {
                    frame = self.alloc_frame(
                        rt,
                        cur_nparams,
                        cur_variadic,
                        cur_nslots,
                        &args_buf,
                        caps.clone(),
                    );
                }
                self.run_ctx_entry(top, rt, &compiled, &frame, &caps, &mut args_buf, cur_callee)
            };
            match outcome {
                Ok(v) => {
                    self.recycle(frame);
                    return v;
                }
                Err(callee) => {
                    // A signal raised while evaluating the tail call's args: stop.
                    if rt.pending() {
                        self.recycle(frame);
                        return M::R::enc_nil();
                    }
                    self.recycle(std::mem::take(&mut frame));
                    // Callable-object hook in TAIL position too (keywords / maps /
                    // callable deftype records): route `(obj args…)` to
                    // `(handler obj args…)`, exactly as `invoke` does.
                    let callee = match rt.decode(callee) {
                        Val::Ref(id)
                            if matches!(&rt.heap()[id as usize], Obj::Record { .. })
                                && rt.apply_handler().is_some() =>
                        {
                            let h = rt.apply_handler().unwrap();
                            args_buf.insert(0, callee);
                            h
                        }
                        _ => callee,
                    };
                    let callee = match rt.multifn_select(callee, args_buf.len()) {
                        Some(sel) => {
                            if rt.pending() {
                                return M::R::enc_nil();
                            }
                            sel
                        }
                        None => callee,
                    };
                    let (nparams, variadic, nslots, comp, ncaps) = self.resolve_call(rt, callee);
                    let arity_ok = if variadic {
                        args_buf.len() >= nparams
                    } else {
                        args_buf.len() == nparams
                    };
                    if !arity_ok {
                        let msg = if variadic {
                            format!("arity: expected at least {}, got {}", nparams, args_buf.len())
                        } else {
                            format!("arity: expected {}, got {}", nparams, args_buf.len())
                        };
                        let sid = rt.alloc(Obj::Str(msg));
                        rt.signal_throw(M::R::enc_ref(sid));
                        return M::R::enc_nil();
                    }
                    self.publish_fast_target(callee, &comp, &ncaps);
                    compiled = comp;
                    caps = ncaps;
                    cur_callee = callee;
                    cur_nparams = nparams;
                    cur_variadic = variadic;
                    cur_nslots = nslots;
                }
            }
        }
    }
}

impl<M: ModelArithJit> CodeSpace<M> for JitCranelift<M> {
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64 {
        // A top-level expression: compile it as a standalone ctx-entry body and
        // run it. (Not cached — matches the bytecode tier's `eval_ir`.) NOT a
        // tail root: the outermost call of a top-level form must flow through
        // `top.invoke` so wrappers (Traced) observe it.
        let shape = BodyShape {
            entry_arity: None,
            nparams: 0,
            variadic: false,
            nslots: 0,
            mem_mode: true,
            tail_root: false,
        };
        let compiled = self.compile(Some(rt), ir, shape);
        // Size the fast-call table to cover every closure that already exists, so
        // its base is stable for this whole form's run (it is only ever written,
        // never grown, during execution). Doing it here (form start) means no
        // native frame is holding a stale base when it (re)allocates.
        //
        // SLACK: a single top-level form (e.g. one big `reduce`/`loop` expression)
        // can allocate hundreds of thousands of heap objects of its own while it
        // runs, all with ids past `heap.len()` at this sizing point. Padding the
        // table lets ids allocated during this run stay in range, so a closure
        // allocated mid-form (a `map` callback used across a whole collection)
        // still gets the "fill once, fast-path every call after" caching.
        {
            let n = rt.heap().len() + FAST_TARGET_SLACK;
            let mut t = self.fast_targets.borrow_mut();
            if t.len() < n {
                t.resize(n, FastTarget::default());
            }
        }
        // A top-level expr is not a closure body: no current callee (0), so the
        // self-tail-call fast path is inert here (it engages once inside a fn).
        self.run_trampoline(
            top,
            rt,
            compiled,
            locals.clone(),
            no_caps(),
            0,
            0,
            false,
            0,
            &[],
        )
    }

    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        // Callable-object hook (keywords / maps / vectors / sets / multimethods): a
        // non-closure record with a registered apply handler routes to
        // `(handler object args…)`, exactly like the TreeWalk `invoke`.
        let routed;
        let (callee, args) = match rt.decode(callee) {
            Val::Ref(id)
                if matches!(&rt.heap()[id as usize], Obj::Record { .. })
                    && rt.apply_handler().is_some() =>
            {
                let h = rt.apply_handler().unwrap();
                let mut v = Vec::with_capacity(args.len() + 1);
                v.push(callee);
                v.extend_from_slice(args);
                routed = v;
                (h, routed.as_slice())
            }
            _ => (callee, args),
        };
        // Multi-arity fn: select the clause serving this arg count. Remember the
        // MultiFn's own bits — emitted call sites see THOSE as the callee, so
        // its fast-table entry must carry the selected clause's entry.
        let orig_callee = callee;
        let callee = match rt.multifn_select(callee, args.len()) {
            Some(sel) => {
                if rt.pending() {
                    return M::R::enc_nil();
                }
                sel
            }
            None => callee,
        };
        // Resolve through the monomorphic inline cache (decode + heap + compiled +
        // caps all skipped on a repeat callee).
        let (nparams, variadic, nslots, compiled, caps) = self.resolve_call(rt, callee);
        // Publish the native fast entry so future call sites can jump to it
        // directly — under the value call sites actually see (the MultiFn id
        // when routing happened, the closure id otherwise).
        self.publish_fast_target(orig_callee, &compiled, &caps);
        // Arity mismatch is a CATCHABLE throw (matching TreeWalk / Clojure), not an
        // abort in this non-unwinding shim.
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
        self.run_trampoline(
            top, rt, compiled, None, caps, callee, nparams, variadic, nslots, args,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Lowering: `Ir` -> Cranelift IR.
// ─────────────────────────────────────────────────────────────────────────

/// The compile-time shape of one body: entry convention + frame layout.
#[derive(Clone, Copy)]
struct BodyShape {
    /// `Some(k)`: register-arg entry `fn(ctx, a0..a{k-1})`. `None`: ctx entry.
    entry_arity: Option<usize>,
    nparams: usize,
    variadic: bool,
    /// Total activation slots (params + rest + let/catch), from `flatten`.
    nslots: u16,
    /// Locals in the heap frame (try/(gc)/%await) instead of SSA registers.
    mem_mode: bool,
    tail_root: bool,
}

/// Compile `ir` as a whole function body into an already-prepared builder.
fn build_body<M: ModelArithJit>(
    module: &mut JITModule,
    fb: &mut FunctionBuilder,
    shims: Shims,
    rt: Option<&Runtime<M>>,
    ir: &Ir,
    tregistry: &RefCell<Vec<ClosureTemplate>>,
    shape: BodyShape,
) {
    let entry = fb.create_block();
    fb.append_block_params_for_function_params(entry);
    fb.switch_to_block(entry);
    fb.seal_block(entry);
    let ctx_val = fb.block_params(entry)[0];
    let param_vals: Vec<Value> = fb.block_params(entry)[1..].to_vec();
    // Load the run-context pointer once; all shared-field reads go through it, and
    // a native call copies just this pointer into the callee's context. `rc` and the
    // fields reached through it are set before the body runs and never mutated
    // during it, so the loads are `readonly` — Cranelift may dedup/hoist them.
    let off_rc = core::mem::offset_of!(JitCtx<'static, M>, rc) as i32;
    let rc_val = fb.ins().load(I64, MemFlagsData::trusted().with_readonly(), ctx_val, off_rc);

    let refs = ShimRefs {
        load_local: module.declare_func_in_func(shims.load_local, fb.func),
        load_global: module.declare_func_in_func(shims.load_global, fb.func),
        def_global: module.declare_func_in_func(shims.def_global, fb.func),
        make_closure: module.declare_func_in_func(shims.make_closure, fb.func),
        call: module.declare_func_in_func(shims.call, fb.func),
        tail_call: module.declare_func_in_func(shims.tail_call, fb.func),
        finish_tail: module.declare_func_in_func(shims.finish_tail, fb.func),
        prim: module.declare_func_in_func(shims.prim, fb.func),
        set_local: module.declare_func_in_func(shims.set_local, fb.func),
        set_global: module.declare_func_in_func(shims.set_global, fb.func),
        field_get: module.declare_func_in_func(shims.field_get, fb.func),
        dispatch: module.declare_func_in_func(shims.dispatch, fb.func),
        def_method: module.declare_func_in_func(shims.def_method, fb.func),
        apply: module.declare_func_in_func(shims.apply, fb.func),
        try_: module.declare_func_in_func(shims.try_, fb.func),
        spawn: module.declare_func_in_func(shims.spawn, fb.func),
        await_: module.declare_func_in_func(shims.await_, fb.func),
        gc: module.declare_func_in_func(shims.gc, fb.func),
    };

    let mut c = Compiler {
        fb,
        refs,
        ctx_val,
        tregistry,
        entry_sigs: HashMap::new(),
        // Type-erased (the `Compiler` struct is not generic over M; the
        // M-generic methods cast it back). Never outlives this call.
        rt_ptr: rt.map_or(std::ptr::null(), |r| r as *const Runtime<M> as *const ()),
        inline_budget: Cell::new(INLINE_TOTAL_BUDGET),
        mem_mode: shape.mem_mode,
        // Stable byte offsets of `JitCtx` fields (repr(C)) for inline reads and for
        // building a callee context on the stack at a native call site.
        off_cur_slots: core::mem::offset_of!(JitCtx<'static, M>, cur_slots) as i32,
        off_consts_base: core::mem::offset_of!(JitCtx<'static, M>, consts_base) as i32,
        off_global_base: core::mem::offset_of!(JitCtx<'static, M>, global_base) as i32,
        off_fast_base: core::mem::offset_of!(JitCtx<'static, M>, fast_base) as i32,
        off_fast_len: core::mem::offset_of!(JitCtx<'static, M>, fast_len) as i32,
        off_direct: core::mem::offset_of!(JitCtx<'static, M>, direct) as i32,
        off_self_closure: core::mem::offset_of!(JitCtx<'static, M>, self_closure) as i32,
        off_caps_base: core::mem::offset_of!(JitCtx<'static, M>, caps_base) as i32,
        off_tail_pending: core::mem::offset_of!(JitCtx<'static, M>, tail_pending) as i32,
        off_rc,
        off_rt: core::mem::offset_of!(JitCtx<'static, M>, rt) as i32,
        signal_kind_off: Runtime::<M>::signal_kind_offset() as i32,
        ctx_size: core::mem::size_of::<JitCtx<'static, M>>() as u32,
        loop_header: None,
        loop_nparams: shape.nparams,
        vars: Vec::new(),
        rc_val,
    };

    // SSA mode: every activation slot is a Cranelift variable (register-
    // allocated). Params come from the entry's register args (fast entries) or
    // a frame-load prologue (ctx entries: variadic / >MAX_REG_ARGS); let/catch
    // slots start nil. Memory mode leaves locals in the heap frame.
    if !shape.mem_mode {
        for _ in 0..shape.nslots {
            let var = c.fb.declare_var(I64);
            c.vars.push(var);
        }
        let nfixed = shape.nparams + shape.variadic as usize;
        match shape.entry_arity {
            Some(k) => {
                debug_assert_eq!(k, nfixed);
                for i in 0..k {
                    c.fb.def_var(c.vars[i], param_vals[i]);
                }
            }
            None => {
                let base = c.load_ctx_field(c.off_cur_slots);
                for i in 0..nfixed.min(shape.nslots as usize) {
                    let v = c.fb.ins().load(I64, MemFlagsData::trusted(), base, (i * 8) as i32);
                    c.fb.def_var(c.vars[i], v);
                }
            }
        }
        if (shape.nslots as usize) > nfixed {
            let nil = c.fb.ins().iconst(I64, M::R::enc_nil() as i64);
            for i in nfixed..shape.nslots as usize {
                c.fb.def_var(c.vars[i], nil);
            }
        }
    }

    // A self-tail-recursive SSA body gets a loop header: a tail call to the same
    // closure redefines the param variables in place and branches here (an
    // O(1)-stack native loop in REGISTERS), with the shim/trampoline as the
    // fallback for any other tail call.
    if !shape.mem_mode && shape.tail_root && !shape.variadic && has_tail_call(ir) {
        let header = c.fb.create_block();
        c.fb.ins().jump(header, &[]);
        c.fb.switch_to_block(header);
        c.loop_header = Some(header);
    }
    // The whole body is in TAIL position: a tail call in it either loops (self) or
    // trampolines (`shim_tail_call`). `compile` threads that through if / do.
    let result = c.compile::<M>(ir, shape.tail_root);
    c.fb.ins().return_(&[result]);
    if let Some(h) = c.loop_header {
        c.fb.seal_block(h);
    }
    // `finalize` consumes the owned `FunctionBuilder`; the caller does it after
    // this returns, since we only hold a `&mut` here.
}

pub struct Compiler<'a, 'b> {
    fb: &'a mut FunctionBuilder<'b>,
    refs: ShimRefs,
    ctx_val: Value,
    tregistry: &'a RefCell<Vec<ClosureTemplate>>,
    /// Imported `fn(ctx, a0..a{k-1}) -> u64` signatures for native fast calls,
    /// built lazily per arity.
    entry_sigs: HashMap<usize, cranelift_codegen::ir::SigRef>,
    /// The runtime at COMPILE time (type-erased; may be null when unavailable,
    /// e.g. `dump_ir`) — the var-guarded speculative inliner resolves `Global`
    /// callees through it.
    rt_ptr: *const (),
    /// Remaining inlined-node allowance for THIS body (bounds code growth and
    /// terminates recursive inlining).
    inline_budget: Cell<usize>,
    /// Locals live in the heap frame (via `cur_slots`) instead of SSA variables.
    mem_mode: bool,
    off_cur_slots: i32,
    off_consts_base: i32,
    off_global_base: i32,
    off_fast_base: i32,
    off_fast_len: i32,
    off_direct: i32,
    off_self_closure: i32,
    off_caps_base: i32,
    off_tail_pending: i32,
    off_rc: i32,
    /// Offset of the `rt` pointer within `JitCtx`, and of the throw/escape flag
    /// (`signal.kind`) within `Runtime` — for the per-call pending check.
    off_rt: i32,
    signal_kind_off: i32,
    ctx_size: u32,
    /// The run-context pointer, loaded once at entry; shared-field reads use it.
    rc_val: Value,
    /// For a self-tail-recursive SSA body: the block to branch back to on a tail
    /// call to the same closure (an in-place native register loop). `None`
    /// disables the self-loop (tail calls then use the shim/trampoline).
    loop_header: Option<cranelift_codegen::ir::Block>,
    loop_nparams: usize,
    /// SSA mode: one variable per activation slot (params first). Empty in
    /// memory mode.
    vars: Vec<cranelift_frontend::Variable>,
}

impl<'a, 'b> Compiler<'a, 'b> {
    fn call_shim(&mut self, f: cranelift_codegen::ir::FuncRef, args: &[Value]) -> Value {
        let inst = self.fb.ins().call(f, args);
        self.fb.inst_results(inst)[0]
    }

    /// After an operation that can raise a `throw`/escape, check the runtime's
    /// `signal.kind` flag and RETURN early (propagating the pending signal up) if
    /// set. The dummy `result` flows out; whoever eventually checks `pending()` (a
    /// `Try`, or the frontend top level) handles it. Same design as the TreeWalk
    /// `if rt.pending() { return v }` after every sub-eval — a plain load + branch.
    fn emit_pending_check(&mut self, result: Value) -> Value {
        let rt_ptr = self.load_rc_field(self.off_rt);
        let kind = self.fb.ins().load(
            cranelift_codegen::ir::types::I8,
            MemFlagsData::trusted(),
            rt_ptr,
            self.signal_kind_off,
        );
        let ret_b = self.fb.create_block();
        let cont_b = self.fb.create_block();
        // `result` is computed in the current (dominating) block, so it is live in
        // both successors — no block params needed.
        self.fb.ins().brif(kind, ret_b, &[], cont_b, &[]);
        self.fb.switch_to_block(ret_b);
        self.fb.seal_block(ret_b);
        self.fb.ins().return_(&[result]);
        self.fb.switch_to_block(cont_b);
        self.fb.seal_block(cont_b);
        result
    }

    /// A shim call that can raise, with the pending check emitted after it.
    fn call_shim_checked(&mut self, f: cranelift_codegen::ir::FuncRef, args: &[Value]) -> Value {
        let result = self.call_shim(f, args);
        self.emit_pending_check(result)
    }

    fn iconst(&mut self, v: u64) -> Value {
        self.fb.ins().iconst(I64, v as i64)
    }

    fn i32const(&mut self, v: u32) -> Value {
        self.fb.ins().iconst(cranelift_codegen::ir::types::I32, v as i64)
    }

    /// Load a pointer-sized field of the `JitCtx` at a stable offset.
    fn load_ctx_field(&mut self, offset: i32) -> Value {
        self.fb.ins().load(I64, MemFlagsData::trusted(), self.ctx_val, offset)
    }

    /// Load a SHARED field at `offset` through the run-context pointer (loaded once
    /// at entry). Same cost as a direct ctx read, but the per-call context need not
    /// carry the field. Marked `readonly`: these fields are fixed for the whole run,
    /// so Cranelift can hoist them out of loops.
    fn load_rc_field(&mut self, offset: i32) -> Value {
        self.fb.ins().load(I64, MemFlagsData::trusted().with_readonly(), self.rc_val, offset)
    }

    /// Inline read of an innermost-frame (`up == 0`) local: the compiled analogue
    /// of `frame_get(.., 0, idx)`, two loads and no call. The slot base is cached
    /// in `JitCtx::cur_slots` (a `*const u64` over the frame's `Cell<u64>` array).
    fn emit_local0_load(&mut self, idx: u16) -> Value {
        let base = self.load_ctx_field(self.off_cur_slots);
        self.fb.ins().load(I64, MemFlagsData::trusted(), base, (idx as i32) * 8)
    }

    fn emit_local0_store(&mut self, idx: u16, v: Value) {
        let base = self.load_ctx_field(self.off_cur_slots);
        self.fb.ins().store(MemFlagsData::trusted(), v, base, (idx as i32) * 8);
    }

    /// Can `(s args…)` be speculatively inlined here? Requires: a compile-time
    /// resolvable global bound to a NON-variadic, CAPTURE-FREE closure of
    /// matching arity whose body is SSA-eligible and small, in an SSA-mode
    /// caller, within budget. Consumes budget on success.
    fn try_inline_plan<M: ModelArithJit>(&self, s: Sym, argc: usize) -> Option<InlinePlan> {
        if self.mem_mode || self.rt_ptr.is_null() {
            return None;
        }
        let rt = unsafe { &*(self.rt_ptr as *const Runtime<M>) };
        let bits = rt.global(s)?;
        let Val::Ref(id) = rt.decode(bits) else { return None };
        // See through a multi-arity fn: this call site's arg count statically
        // selects one fixed clause. The guard still compares the GLOBAL's bits
        // (the MultiFn), so redefinition deopts as usual.
        let target = match &rt.heap()[id as usize] {
            Obj::MultiFn { fixed, .. } => {
                let f = fixed.get(argc).copied().unwrap_or(0);
                if f == 0 {
                    return None; // variadic / no such arity: not inlinable
                }
                f
            }
            _ => bits,
        };
        let Val::Ref(tid) = rt.decode(target) else { return None };
        let (nparams, nslots, body) = match &rt.heap()[tid as usize] {
            Obj::Closure { nparams, variadic: false, nslots, body, caps }
                if caps.is_empty() && *nparams == argc =>
            {
                (*nparams, *nslots, body.clone())
            }
            _ => return None,
        };
        if body_mem_mode(&body) {
            return None;
        }
        let cost = node_count_capped(&body, INLINE_MAX_CALLEE_NODES)?;
        let budget = self.inline_budget.get();
        if budget < cost {
            return None;
        }
        self.inline_budget.set(budget - cost);
        Some(InlinePlan { bits, nparams, nslots, body })
    }

    /// Splice an inlined callee body into the current function: fresh SSA vars
    /// for its activation slots (params seeded from the evaluated args, the
    /// rest nil), compiled in NON-tail position with no self-loop — its own
    /// tail calls become ordinary calls, which preserves semantics (and one
    /// inlined level never grows the stack unboundedly).
    fn emit_inlined_body<M: ModelArithJit>(&mut self, plan: &InlinePlan, argvals: &[Value]) -> Value {
        let saved_vars = std::mem::take(&mut self.vars);
        let saved_header = self.loop_header.take();
        let saved_nparams = self.loop_nparams;
        for i in 0..plan.nslots as usize {
            let var = self.fb.declare_var(I64);
            if i < plan.nparams {
                self.fb.def_var(var, argvals[i]);
            } else {
                let nil = self.fb.ins().iconst(I64, M::R::enc_nil() as i64);
                self.fb.def_var(var, nil);
            }
            self.vars.push(var);
        }
        self.loop_nparams = plan.nparams;
        let v = self.compile::<M>(&plan.body, false);
        self.vars = saved_vars;
        self.loop_header = saved_header;
        self.loop_nparams = saved_nparams;
        v
    }

    /// The imported signature for a native fast entry of arity `k`:
    /// `fn(ctx, a0..a{k-1}) -> u64`.
    fn entry_sig(&mut self, k: usize) -> cranelift_codegen::ir::SigRef {
        if let Some(&sig) = self.entry_sigs.get(&k) {
            return sig;
        }
        let mut s = cranelift_codegen::ir::Signature::new(
            cranelift_codegen::isa::CallConv::SystemV,
        );
        s.params.push(AbiParam::new(I64));
        for _ in 0..k {
            s.params.push(AbiParam::new(I64));
        }
        s.returns.push(AbiParam::new(I64));
        let sig = self.fb.import_signature(s);
        self.entry_sigs.insert(k, sig);
        sig
    }

    /// A non-tail call. Try the NATIVE fast path — resolve the callee to its
    /// compiled register-arg entry inline through the fast-call table, build a
    /// minimal 4-store context ON THIS STACK, and `call_indirect` with the args
    /// in registers (no FFI, no `invoke`, no frame). Fall back to the shim path
    /// (`top.invoke`) for anything ineligible, which is what preserves
    /// composition: the guard requires `direct` (so a wrapped backend never
    /// fast-paths) and a filled, arity-matching table entry (so continuation /
    /// variadic / memory-mode callees, whose entries stay null, go through `top`).
    fn emit_call<M: ModelArithJit>(&mut self, callee: Value, argvals: &[Value]) -> Value {
        let n = argvals.len();
        let ctx = self.ctx_val;
        let flags = MemFlagsData::trusted();
        if n > MAX_REG_ARGS {
            // No register entry exists at this arity — always the shim path.
            let (addr, count) = self.spill_args(argvals);
            let sr = self.call_shim(self.refs.call, &[ctx, callee, addr, count]);
            return self.emit_pending_check(sr);
        }

        // guard1 = value is a ref AND its id is in range AND direct calls enabled
        let (is_ref, id) = M::emit_ref_id(self, callee);
        let rc = self.rc_val;
        let ro = MemFlagsData::trusted().with_readonly();
        let direct = self.fb.ins().load(I8, ro, rc, self.off_direct);
        let flen = self.fb.ins().load(I64, ro, rc, self.off_fast_len);
        let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, id, flen);
        let g = self.fb.ins().band(is_ref, inb);
        let guard1 = self.fb.ins().band(g, direct);

        let result = self.fb.declare_var(I64);
        let checkb = self.fb.create_block();
        let fastb = self.fb.create_block();
        let slowb = self.fb.create_block();
        let merge = self.fb.create_block();
        self.fb.ins().brif(guard1, checkb, &[], slowb, &[]);

        // ── id is in range: load the entry, check code present + arity match ──
        self.fb.switch_to_block(checkb);
        self.fb.seal_block(checkb);
        let fbase = self.fb.ins().load(I64, ro, rc, self.off_fast_base);
        let idx = self.fb.ins().imul_imm(id, core::mem::size_of::<FastTarget>() as i64);
        let entry = self.fb.ins().iadd(fbase, idx);
        let code = self.fb.ins().load(I64, flags, entry, 0);
        let caps = self.fb.ins().load(I64, flags, entry, 8);
        let arity = self.fb.ins().load(cranelift_codegen::ir::types::I32, flags, entry, 16);
        let has_code = self.fb.ins().icmp_imm(IntCC::NotEqual, code, 0);
        let arity_ok = self.fb.ins().icmp_imm(IntCC::Equal, arity, n as i64);
        let guard2 = self.fb.ins().band(has_code, arity_ok);
        self.fb.ins().brif(guard2, fastb, &[], slowb, &[]);

        // ── native fast path: 4-store context, args in registers ──
        self.fb.switch_to_block(fastb);
        self.fb.seal_block(fastb);
        let words = self.ctx_size.div_ceil(8) as i32;
        let ctx_ss = self.fb.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (words * 8) as u32,
            3,
        ));
        // Four stores: the shared run-context pointer, the callee's own bits (for
        // its self-tail loop), its capture base, and a clear tail-pending flag.
        // The callee reads everything else through `rc`.
        self.fb.ins().stack_store(rc, ctx_ss, self.off_rc);
        self.fb.ins().stack_store(callee, ctx_ss, self.off_self_closure);
        self.fb.ins().stack_store(caps, ctx_ss, self.off_caps_base);
        let zero8 = self.fb.ins().iconst(I8, 0);
        self.fb.ins().stack_store(zero8, ctx_ss, self.off_tail_pending);
        let new_ctx = self.fb.ins().stack_addr(I64, ctx_ss, 0);
        let sig = self.entry_sig(n);
        let mut call_args = Vec::with_capacity(n + 1);
        call_args.push(new_ctx);
        call_args.extend_from_slice(argvals);
        let inst = self.fb.ins().call_indirect(sig, code, &call_args);
        let r0 = self.fb.inst_results(inst)[0];
        // If the callee ended in a NON-self tail call, `tail_pending` is set and its
        // frame is gone; finish that tail through `top` (bounded stack, composable).
        // The common self-tail case looped in place and never sets this.
        let tp = self.fb.ins().load(I8, flags, new_ctx, self.off_tail_pending);
        let finb = self.fb.create_block();
        let doneb = self.fb.create_block();
        self.fb.ins().brif(tp, finb, &[], doneb, &[]);
        self.fb.switch_to_block(finb);
        self.fb.seal_block(finb);
        let fr = self.call_shim(self.refs.finish_tail, &[new_ctx]);
        self.fb.def_var(result, fr);
        self.fb.ins().jump(merge, &[]);
        self.fb.switch_to_block(doneb);
        self.fb.seal_block(doneb);
        self.fb.def_var(result, r0);
        self.fb.ins().jump(merge, &[]);

        // ── shim fallback: through `top`, preserving CEK routing / Traced / etc. ──
        self.fb.switch_to_block(slowb);
        self.fb.seal_block(slowb);
        let (addr, count) = self.spill_args(argvals);
        let sr = self.call_shim(self.refs.call, &[ctx, callee, addr, count]);
        self.fb.def_var(result, sr);
        self.fb.ins().jump(merge, &[]);

        self.fb.switch_to_block(merge);
        self.fb.seal_block(merge);
        // The callee may have raised a throw/escape (on either path); propagate.
        let r = self.fb.use_var(result);
        self.emit_pending_check(r)
    }

    /// Spill a run of computed values into a fresh stack slot and return
    /// `(base_addr, count)` for a shim that takes `*const u64, u32`.
    fn spill_args(&mut self, args: &[Value]) -> (Value, Value) {
        let n = args.len();
        if n == 0 {
            let z = self.iconst(0);
            let c = self.i32const(0);
            return (z, c);
        }
        let slot = self.fb.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (n * 8) as u32,
            3, // 2^3 = 8-byte alignment
        ));
        for (i, &a) in args.iter().enumerate() {
            self.fb.ins().stack_store(a, slot, (i * 8) as i32);
        }
        let addr = self.fb.ins().stack_addr(I64, slot, 0);
        let count = self.i32const(n as u32);
        (addr, count)
    }

    fn compile<M: ModelArithJit>(&mut self, ir: &Ir, tail: bool) -> Value {
        match ir {
            // Inline load from the constant-pool base — no shim call.
            Ir::Const(id) | Ir::Quote(id) => {
                let base = self.load_rc_field(self.off_consts_base);
                // The constant pool is fixed during a run: readonly, so a constant
                // used in a loop is hoisted out of it.
                let ro = MemFlagsData::trusted().with_readonly();
                self.fb.ins().load(I64, ro, base, (*id as i32) * 8)
            }
            // Flat model: every local is a slot of THIS activation. SSA mode
            // reads the register variable; memory mode loads the frame slot.
            Ir::Local { up, idx } => {
                assert_eq!(*up, 0, "unflattened Ir reached the JIT: Local up={up}");
                if self.mem_mode {
                    self.emit_local0_load(*idx)
                } else {
                    self.fb.use_var(self.vars[*idx as usize])
                }
            }
            // A captured value: one load off the closure's capture array. The
            // base is a per-call constant; in SSA mode the CONTENTS are immutable
            // too (no GC can rewrite them mid-body), so the load is readonly and
            // hoistable out of loops.
            Ir::Capture(idx) => {
                let ro = MemFlagsData::trusted().with_readonly();
                let base = self.fb.ins().load(I64, ro, self.ctx_val, self.off_caps_base);
                let flags = if self.mem_mode { MemFlagsData::trusted() } else { ro };
                self.fb.ins().load(I64, flags, base, (*idx as i32) * 8)
            }
            // Inline global read: load the value straight from the dense mirror
            // (the symbol was interned, so its slot exists — no bounds check). If
            // the slot is still `GLOBAL_UNBOUND` (a forward reference not yet
            // defined), fall back to the slow late-binding lookup, which errors or
            // resolves per the runtime's contract.
            Ir::Global(s) => {
                let base = self.load_rc_field(self.off_global_base);
                let raw = self.fb.ins().load(I64, MemFlagsData::trusted(), base, (*s as i32) * 8);
                let unbound = self.iconst(crate::runtime::GLOBAL_UNBOUND);
                let is_unbound = self.fb.ins().icmp(IntCC::Equal, raw, unbound);

                let result = self.fb.declare_var(I64);
                let slow = self.fb.create_block();
                let ok = self.fb.create_block();
                let merge = self.fb.create_block();
                self.fb.ins().brif(is_unbound, slow, &[], ok, &[]);

                self.fb.switch_to_block(ok);
                self.fb.seal_block(ok);
                self.fb.def_var(result, raw);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sv = self.i32const(*s);
                let ctx = self.ctx_val;
                let v = self.call_shim(self.refs.load_global, &[ctx, sv]);
                self.fb.def_var(result, v);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // Tail-transparent: only the LAST expression is in tail position.
            Ir::Do(xs) => {
                if xs.is_empty() {
                    return self.iconst(M::R::enc_nil());
                }
                let last_i = xs.len() - 1;
                let mut last = self.iconst(M::R::enc_nil());
                for (i, x) in xs.iter().enumerate() {
                    last = self.compile::<M>(x, tail && i == last_i);
                }
                last
            }
            // Tail-transparent: the condition is not in tail position, but both
            // arms inherit it, so a tail call in either arm trampolines.
            Ir::If(cnd, then, els) => {
                // Peephole: an optimizer-inserted fixnum guard
                // `(if (%all-fixnum? a ..) t e)` lowers to a RAW combined tag
                // test as the branch condition — no boolean is materialized and
                // no truthiness is decoded, so the guard costs one bit test.
                let t = if let Ir::Prim(Prim::AllFixnum, gargs) = cnd.as_ref() {
                    let vs: Vec<Value> = gargs.iter().map(|a| self.compile::<M>(a, false)).collect();
                    self.emit_all_fixnum_raw::<M>(&vs)
                } else {
                    let cv = self.compile::<M>(cnd, false);
                    // Inline truthiness: only `nil` and `false` are falsey, and each
                    // is a single value word, so `cv != nil && cv != false` — two
                    // compares, no shim call. (Refs/ints/syms/floats/true are truthy.)
                    let nil = self.iconst(M::R::enc_nil());
                    let fal = self.iconst(M::R::enc_bool(false));
                    let not_nil = self.fb.ins().icmp(IntCC::NotEqual, cv, nil);
                    let not_fal = self.fb.ins().icmp(IntCC::NotEqual, cv, fal);
                    self.fb.ins().band(not_nil, not_fal)
                };

                let result = self.fb.declare_var(I64);
                let then_b = self.fb.create_block();
                let else_b = self.fb.create_block();
                let merge = self.fb.create_block();

                self.fb.ins().brif(t, then_b, &[], else_b, &[]);

                self.fb.switch_to_block(then_b);
                self.fb.seal_block(then_b);
                let tv = self.compile::<M>(then, tail);
                self.fb.def_var(result, tv);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(else_b);
                self.fb.seal_block(else_b);
                let ev = self.compile::<M>(els, tail);
                self.fb.def_var(result, ev);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            Ir::Def { name, init } => {
                let v = self.compile::<M>(init, false);
                let namev = self.i32const(*name);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.def_global, &[ctx, namev, v])
            }
            Ir::Lambda { nparams, variadic, nslots, captures, body } => {
                // Register the template once (compile time), then emit: resolve
                // each capture VALUE here (registers / frame slots / own caps),
                // spill them in order, and let the shim copy + allocate.
                let tid = {
                    let mut reg = self.tregistry.borrow_mut();
                    reg.push(ClosureTemplate {
                        nparams: *nparams,
                        variadic: *variadic,
                        nslots: *nslots,
                        ncaps: captures.len() as u16,
                        body: body.clone(),
                    });
                    (reg.len() - 1) as u32
                };
                let vals: Vec<Value> = captures
                    .iter()
                    .map(|c| match c {
                        CapSrc::Slot(i) => {
                            if self.mem_mode {
                                self.emit_local0_load(*i)
                            } else {
                                self.fb.use_var(self.vars[*i as usize])
                            }
                        }
                        CapSrc::Cap(j) => {
                            let ro = MemFlagsData::trusted().with_readonly();
                            let base =
                                self.fb.ins().load(I64, ro, self.ctx_val, self.off_caps_base);
                            let flags = if self.mem_mode { MemFlagsData::trusted() } else { ro };
                            self.fb.ins().load(I64, flags, base, (*j as i32) * 8)
                        }
                    })
                    .collect();
                let (addr, _count) = self.spill_args(&vals);
                let tidv = self.i32const(tid);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.make_closure, &[ctx, tidv, addr])
            }
            Ir::Call(f, args) => {
                // Var-guarded speculative inlining: a non-tail call of a global
                // bound (NOW, at compile time) to a small capture-free closure
                // splices the callee body inline behind a one-compare guard on
                // the global slot. Redefinition (or a GC move) fails the guard
                // and takes the general path — never a wrong answer.
                if !tail {
                    if let Ir::Global(gsym) = f.as_ref() {
                        if let Some(plan) = self.try_inline_plan::<M>(*gsym, args.len()) {
                            let argvals: Vec<Value> =
                                args.iter().map(|a| self.compile::<M>(a, false)).collect();
                            let base = self.load_rc_field(self.off_global_base);
                            let raw = self.fb.ins().load(
                                I64,
                                MemFlagsData::trusted(),
                                base,
                                (*gsym as i32) * 8,
                            );
                            let want = self.iconst(plan.bits);
                            let bits_ok = self.fb.ins().icmp(IntCC::Equal, raw, want);
                            // Composition: an inlined call is invisible to `top`,
                            // so it is only legal when no wrapper is observing
                            // calls — the same `direct` rule as the fast-call path.
                            let ro = MemFlagsData::trusted().with_readonly();
                            let rc = self.rc_val;
                            let direct = self.fb.ins().load(I8, ro, rc, self.off_direct);
                            let same = self.fb.ins().band(bits_ok, direct);
                            let result = self.fb.declare_var(I64);
                            let inlb = self.fb.create_block();
                            let slowb = self.fb.create_block();
                            let merge = self.fb.create_block();
                            self.fb.ins().brif(same, inlb, &[], slowb, &[]);

                            self.fb.switch_to_block(inlb);
                            self.fb.seal_block(inlb);
                            let iv = self.emit_inlined_body::<M>(&plan, &argvals);
                            self.fb.def_var(result, iv);
                            self.fb.ins().jump(merge, &[]);

                            // Deopt: the general callee compile (handles unbound
                            // too) + the ordinary call path.
                            self.fb.switch_to_block(slowb);
                            self.fb.seal_block(slowb);
                            let callee = self.compile::<M>(f, false);
                            let sv = self.emit_call::<M>(callee, &argvals);
                            self.fb.def_var(result, sv);
                            self.fb.ins().jump(merge, &[]);

                            self.fb.switch_to_block(merge);
                            self.fb.seal_block(merge);
                            return self.fb.use_var(result);
                        }
                    }
                }
                let callee = self.compile::<M>(f, false);
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                if tail {
                    let ctx = self.ctx_val;
                    // Self-tail-call becomes a native REGISTER loop: if the callee
                    // is THIS closure, redefine the param variables and branch to
                    // the header (no FFI, O(1) stack). Arity must match statically
                    // (same closure ⇒ same nparams); otherwise it is an arity
                    // error the shim path raises. Anything else falls back to the
                    // shim + trampoline, keeping full mutual-recursion TCO.
                    match self.loop_header {
                        Some(header) if argvals.len() == self.loop_nparams => {
                            let flags = MemFlagsData::trusted();
                            let sc = self.fb.ins().load(I64, flags, ctx, self.off_self_closure);
                            let is_self = self.fb.ins().icmp(IntCC::Equal, callee, sc);
                            let selfloop = self.fb.create_block();
                            let notself = self.fb.create_block();
                            self.fb.ins().brif(is_self, selfloop, &[], notself, &[]);

                            self.fb.switch_to_block(selfloop);
                            self.fb.seal_block(selfloop);
                            // Redefine the SSA vars (args were already computed
                            // from the OLD values, so no clobber hazard).
                            for (i, &a) in argvals.iter().enumerate() {
                                self.fb.def_var(self.vars[i], a);
                            }
                            self.fb.ins().jump(header, &[]);

                            // Non-self: the shim tail-call, whose result flows to
                            // the function return (the trampoline reads
                            // `tail_pending`).
                            self.fb.switch_to_block(notself);
                            self.fb.seal_block(notself);
                            let (addr, count) = self.spill_args(&argvals);
                            self.call_shim(self.refs.tail_call, &[ctx, callee, addr, count])
                        }
                        _ => {
                            let (addr, count) = self.spill_args(&argvals);
                            self.call_shim(self.refs.tail_call, &[ctx, callee, addr, count])
                        }
                    }
                } else {
                    self.emit_call::<M>(callee, &argvals)
                }
            }
            // Model-emitted arithmetic with a guarded fast path. The fast path
            // untags per the model's tag layout (the same per-model split the
            // `ModelEmit` recipe encodes: LowBit shifts, HighBit masks, NanBox has
            // no immediate-int fast path), computes natively, and — unlike the
            // wrapping bytecode recipe — checks the fixnum range, falling back to
            // the runtime's promoting arithmetic on overflow / non-fixnum operands.
            // That gives the JIT the full numeric tower (bignum promotion, floats,
            // mixed) the tree-walker has. This is the emit half of codegen axis #2.
            Ir::Prim(p @ (Prim::Add | Prim::Sub | Prim::Mul | Prim::Lt | Prim::Eq), args) => {
                let a = self.compile::<M>(&args[0], false);
                let b = self.compile::<M>(&args[1], false);
                self.emit_guarded_arith::<M>(*p, a, b)
            }
            // Fixnum-specialized ops (from the optimizer): the operands were
            // PROVEN immediate fixnums by a dominating `AllFixnum` guard, so we
            // skip the per-op tag check. Add/Sub/Mul keep the overflow → promote
            // path (two fixnums can still overflow to a bignum); Lt/Eq need
            // nothing but the compare.
            Ir::Prim(p @ (Prim::FxAdd | Prim::FxSub | Prim::FxMul | Prim::FxLt | Prim::FxEq), args) => {
                let a = self.compile::<M>(&args[0], false);
                let b = self.compile::<M>(&args[1], false);
                self.emit_unguarded_arith::<M>(p.dechecked(), a, b)
            }
            // The specializer's entry guard: true iff every operand is an
            // immediate fixnum, as ONE combined tag test.
            Ir::Prim(Prim::AllFixnum, args) => {
                let vs: Vec<Value> = args.iter().map(|a| self.compile::<M>(a, false)).collect();
                self.emit_all_fixnum::<M>(&vs)
            }
            // `(%spawn thunk)` — spawn a native-JIT worker thread; `(%await fut)`
            // — join it, parking during the wait. Both via shims that reach the
            // runtime + build a worker JIT, so threads run WHOLLY on the JIT.
            Ir::Prim(Prim::Spawn, args) => {
                let thunk = self.compile::<M>(&args[0], false);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.spawn, &[ctx, thunk])
            }
            Ir::Prim(Prim::Await, args) => {
                let fut = self.compile::<M>(&args[0], false);
                let ctx = self.ctx_val;
                self.call_shim_checked(self.refs.await_, &[ctx, fut])
            }
            Ir::Prim(Prim::Gc, _) => {
                // Modeled safepoint: collect with `ctx.cur` (the live frame chain)
                // as the root, so JIT locals survive the move (see `shim_gc`).
                let ctx = self.ctx_val;
                self.call_shim(self.refs.gc, &[ctx])
            }
            Ir::Prim(Prim::CallEc | Prim::CallCc | Prim::Reset | Prim::Shift, _) => panic!(
                "JIT tier: continuations not supported; run on the tree-walker / CEK"
            ),
            // `(apply f a … lst)` — flatten + invoke through `top` in the shim.
            Ir::Prim(Prim::Apply, args) => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let (addr, count) = self.spill_args(&argvals);
                let ctx = self.ctx_val;
                self.call_shim_checked(self.refs.apply, &[ctx, addr, count])
            }
            // Every other prim: compute args, escape to the runtime (the native
            // analogue of the bytecode tier's `Slow`).
            Ir::Prim(p, args) => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let (addr, count) = self.spill_args(&argvals);
                let tagv = self.i32const(prim_tag(*p));
                let ctx = self.ctx_val;
                self.call_shim_checked(self.refs.prim, &[ctx, tagv, addr, count])
            }
            Ir::Let(..) => {
                panic!("unflattened Ir reached the JIT: Let survives only before flatten::flatten")
            }
            Ir::SetLocal { up, idx, val } => {
                assert_eq!(*up, 0, "unflattened Ir reached the JIT: SetLocal up={up}");
                let v = self.compile::<M>(val, false);
                if self.mem_mode {
                    self.emit_local0_store(*idx, v);
                } else {
                    self.fb.def_var(self.vars[*idx as usize], v);
                }
                v // `set!` evaluates to the assigned value
            }
            Ir::SetGlobal { name, val } => {
                let v = self.compile::<M>(val, false);
                let namev = self.i32const(*name);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.set_global, &[ctx, namev, v])
            }
            // `(.-field obj)` — inline-cached field read via the shim.
            Ir::FieldGet { site, field, obj } => {
                let o = self.compile::<M>(obj, false);
                let sitev = self.i32const(*site as u32);
                let fieldv = self.i32const(*field);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.field_get, &[ctx, sitev, fieldv, o])
            }
            // Protocol/method dispatch: args spilled like a call, resolved + invoked
            // in the shim (which reaches `top`).
            Ir::Dispatch { site, method, args } => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let (addr, count) = self.spill_args(&argvals);
                let sitev = self.i32const(*site as u32);
                let methodv = self.i32const(*method);
                let ctx = self.ctx_val;
                self.call_shim_checked(self.refs.dispatch, &[ctx, sitev, methodv, addr, count])
            }
            // Register a deftype/protocol method impl.
            Ir::DefMethod { name, ty, imp } => {
                let impv = self.compile::<M>(imp, false);
                let namev = self.i32const(*name);
                let tyv = self.i32const(*ty);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.def_method, &[ctx, namev, tyv, impv])
            }
            // try/catch/finally: pass the body/catch/finally Ir by pointer (they
            // outlive the compiled code) and let the shim run them via `top` under
            // a catch_unwind — so the whole construct is one shim call.
            Ir::Try { body, catch, finally, cslot } => {
                let bp = (&**body as *const Ir) as i64;
                let cp = catch.as_deref().map_or(std::ptr::null::<Ir>(), |c| c as *const Ir) as i64;
                let fp = finally.as_deref().map_or(std::ptr::null::<Ir>(), |c| c as *const Ir) as i64;
                let bpv = self.fb.ins().iconst(I64, bp);
                let cpv = self.fb.ins().iconst(I64, cp);
                let fpv = self.fb.ins().iconst(I64, fp);
                let csv = self.i32const(*cslot as u32);
                let ctx = self.ctx_val;
                self.call_shim_checked(self.refs.try_, &[ctx, bpv, cpv, fpv, csv])
            }
        }
    }

    /// Call the runtime's `prim` (the checked, promoting arithmetic / any prim) —
    /// the native `Slow` escape, shared by the arithmetic fallback and non-fast prims.
    fn slow_prim(&mut self, op: Prim, args: &[Value]) -> Value {
        let (addr, count) = self.spill_args(args);
        let tagv = self.i32const(prim_tag(op));
        let ctx = self.ctx_val;
        self.call_shim_checked(self.refs.prim, &[ctx, tagv, addr, count])
    }

    /// Emit guarded arithmetic: the per-model fixnum fast path when both operands
    /// are immediate integers and the result fits fixnum range, else a call to the
    /// runtime's promoting `prim` (bignum / float / mixed). The emit half of the
    /// numeric-overflow axis — what makes the JIT match the tree-walker's tower.
    fn emit_guarded_arith<M: ModelArithJit>(&mut self, op: Prim, a: Value, b: Value) -> Value {
        let both = M::emit_both_int(self, a, b);

        let result = self.fb.declare_var(I64);
        let fast = self.fb.create_block();
        let slow = self.fb.create_block();
        let merge = self.fb.create_block();
        self.fb.ins().brif(both, fast, &[], slow, &[]);

        // ── fast path: both operands are immediate fixnums ──
        self.fb.switch_to_block(fast);
        self.fb.seal_block(fast);
        let x = M::emit_untag(self, a);
        let y = M::emit_untag(self, b);
        match op {
            Prim::Lt | Prim::Eq => {
                // `<` / `=` don't overflow. For two immediate fixnums, `=` is just
                // bit-equality of the untagged values (the slow `equal?` path is
                // only needed for non-fixnums, which the guard already routed away).
                let cc = if let Prim::Eq = op { IntCC::Equal } else { IntCC::SignedLessThan };
                let c = self.fb.ins().icmp(cc, x, y);
                let cw = self.fb.ins().uextend(I64, c);
                let t = self.iconst(M::R::enc_bool(true));
                let f = self.iconst(M::R::enc_bool(false));
                let res = self.fb.ins().select(cw, t, f);
                self.fb.def_var(result, res);
                self.fb.ins().jump(merge, &[]);
            }
            Prim::Add | Prim::Sub => {
                // Both operands are < 2^60, so the i64 op can't overflow i64; it
                // only needs a fixnum-range check.
                let r = if let Prim::Add = op {
                    self.fb.ins().iadd(x, y)
                } else {
                    self.fb.ins().isub(x, y)
                };
                self.emit_range_check_and_retag::<M>(r, result, slow, merge);
            }
            Prim::Mul => {
                // The product of two 61-bit values can exceed i64; widen to i128,
                // range-check, then narrow.
                let x128 = self.fb.ins().sextend(I128, x);
                let y128 = self.fb.ins().sextend(I128, y);
                let r128 = self.fb.ins().imul(x128, y128);
                let min64 = self.iconst_signed(FIXNUM_MIN);
                let max64 = self.iconst_signed(FIXNUM_MAX);
                let min = self.fb.ins().sextend(I128, min64);
                let max = self.fb.ins().sextend(I128, max64);
                let ge = self.fb.ins().icmp(IntCC::SignedGreaterThanOrEqual, r128, min);
                let le = self.fb.ins().icmp(IntCC::SignedLessThanOrEqual, r128, max);
                let fits = self.fb.ins().band(ge, le);
                let ok = self.fb.create_block();
                self.fb.ins().brif(fits, ok, &[], slow, &[]);
                self.fb.switch_to_block(ok);
                self.fb.seal_block(ok);
                let r64 = self.fb.ins().ireduce(I64, r128);
                let tagged = M::emit_tag(self, r64);
                self.fb.def_var(result, tagged);
                self.fb.ins().jump(merge, &[]);
            }
            _ => unreachable!("emit_guarded_arith only handles +,-,*,<"),
        }

        // ── slow path: promote / handle non-fixnum operands in the runtime ──
        // (Reached both when operands aren't fixnums and when the fast result
        // overflowed fixnum range; all predecessor edges are emitted above.)
        self.fb.switch_to_block(slow);
        self.fb.seal_block(slow);
        let sp = self.slow_prim(op, &[a, b]);
        self.fb.def_var(result, sp);
        self.fb.ins().jump(merge, &[]);

        self.fb.switch_to_block(merge);
        self.fb.seal_block(merge);
        self.fb.use_var(result)
    }

    /// Fixnum-range-check an i64 fast result `r`: if it fits, retag and jump to
    /// `merge`; otherwise fall through to `slow`. Shared by add/sub.
    fn emit_range_check_and_retag<M: ModelArithJit>(
        &mut self,
        r: Value,
        result: cranelift_frontend::Variable,
        slow: cranelift_codegen::ir::Block,
        merge: cranelift_codegen::ir::Block,
    ) {
        let ge = self.fb.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, r, FIXNUM_MIN);
        let le = self.fb.ins().icmp_imm(IntCC::SignedLessThanOrEqual, r, FIXNUM_MAX);
        let fits = self.fb.ins().band(ge, le);
        let ok = self.fb.create_block();
        self.fb.ins().brif(fits, ok, &[], slow, &[]);
        self.fb.switch_to_block(ok);
        self.fb.seal_block(ok);
        let tagged = M::emit_tag(self, r);
        self.fb.def_var(result, tagged);
        self.fb.ins().jump(merge, &[]);
    }

    /// Arithmetic on operands the optimizer PROVED are immediate fixnums: the
    /// guarded path minus the entry tag check. Add/Sub/Mul still range-check and
    /// fall back to the runtime's promoting op on overflow (fixnum → bignum);
    /// Lt/Eq are a bare compare (they cannot overflow and, given fixnum
    /// operands, `=` is bit-equality of the untagged values).
    fn emit_unguarded_arith<M: ModelArithJit>(&mut self, op: Prim, a: Value, b: Value) -> Value {
        let x = M::emit_untag(self, a);
        let y = M::emit_untag(self, b);
        match op {
            Prim::Lt | Prim::Eq => {
                let cc = if let Prim::Eq = op { IntCC::Equal } else { IntCC::SignedLessThan };
                let c = self.fb.ins().icmp(cc, x, y);
                let cw = self.fb.ins().uextend(I64, c);
                let t = self.iconst(M::R::enc_bool(true));
                let f = self.iconst(M::R::enc_bool(false));
                self.fb.ins().select(cw, t, f)
            }
            Prim::Add | Prim::Sub => {
                let result = self.fb.declare_var(I64);
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let r = if let Prim::Add = op {
                    self.fb.ins().iadd(x, y)
                } else {
                    self.fb.ins().isub(x, y)
                };
                self.emit_range_check_and_retag::<M>(r, result, slow, merge);
                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(op, &[a, b]);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);
                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            Prim::Mul => {
                let result = self.fb.declare_var(I64);
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let x128 = self.fb.ins().sextend(I128, x);
                let y128 = self.fb.ins().sextend(I128, y);
                let r128 = self.fb.ins().imul(x128, y128);
                let min64 = self.iconst_signed(FIXNUM_MIN);
                let max64 = self.iconst_signed(FIXNUM_MAX);
                let min = self.fb.ins().sextend(I128, min64);
                let max = self.fb.ins().sextend(I128, max64);
                let ge = self.fb.ins().icmp(IntCC::SignedGreaterThanOrEqual, r128, min);
                let le = self.fb.ins().icmp(IntCC::SignedLessThanOrEqual, r128, max);
                let fits = self.fb.ins().band(ge, le);
                let ok = self.fb.create_block();
                self.fb.ins().brif(fits, ok, &[], slow, &[]);
                self.fb.switch_to_block(ok);
                self.fb.seal_block(ok);
                let r64 = self.fb.ins().ireduce(I64, r128);
                let tagged = M::emit_tag(self, r64);
                self.fb.def_var(result, tagged);
                self.fb.ins().jump(merge, &[]);
                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(op, &[a, b]);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);
                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            _ => unreachable!("emit_unguarded_arith only handles +,-,*,<,="),
        }
    }

    /// `AllFixnum`: true iff every value is an immediate fixnum. "All have their
    /// tag bits clear" ⟺ "the OR of all of them has its tag bits clear", so this
    /// is one bor-reduction plus the model's single tag test (const-false for a
    /// model with no immediate ints — NaN-boxing — which correctly forces the
    /// slow body). Returns an encoded boolean.
    fn emit_all_fixnum<M: ModelArithJit>(&mut self, vs: &[Value]) -> Value {
        let test = self.emit_all_fixnum_raw::<M>(vs);
        let tw = self.fb.ins().uextend(I64, test);
        let t = self.iconst(M::R::enc_bool(true));
        let f = self.iconst(M::R::enc_bool(false));
        self.fb.ins().select(tw, t, f)
    }

    /// The raw `i8` predicate behind `AllFixnum` (before boolean
    /// materialization), so an `If` guard can branch on it directly.
    fn emit_all_fixnum_raw<M: ModelArithJit>(&mut self, vs: &[Value]) -> Value {
        match vs.iter().copied().reduce(|a, b| self.fb.ins().bor(a, b)) {
            Some(v) => M::emit_both_int(self, v, v),
            // No args ⇒ vacuously true.
            None => self.fb.ins().iconst(cranelift_codegen::ir::types::I8, 1),
        }
    }

    fn iconst_signed(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I64, v)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tiered: the native JIT with an automatic CEK fallback.
// ─────────────────────────────────────────────────────────────────────────

/// Can the JIT compile this node directly? True unless the *directly-compiled*
/// tree (NOT descending into `Lambda` bodies — those are compiled lazily at their
/// own invoke, and re-classified then) contains a construct the native tier does
/// not model: the continuation / `apply` / `gc` prims, or record dispatch.
pub fn jit_can_compile(ir: &Ir) -> bool {
    match ir {
        Ir::Prim(Prim::CallCc | Prim::Reset | Prim::Shift | Prim::CallEc | Prim::Gc, _) => false,
        Ir::Dispatch { args, .. } => args.iter().all(jit_can_compile),
        Ir::DefMethod { imp, .. } => jit_can_compile(imp),
        Ir::FieldGet { obj, .. } => jit_can_compile(obj),
        // try/catch is a shim call; the body/handlers run via `top`, so their
        // compilability is decided when the shim evaluates them (not here).
        Ir::Try { .. } => true,
        // A `Lambda` only makes a closure here; its body's compilability is
        // decided when that closure is invoked. So do NOT descend.
        Ir::Lambda { .. } => true,
        Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Capture(_) | Ir::Global(_) => true,
        Ir::If(a, b, c) => jit_can_compile(a) && jit_can_compile(b) && jit_can_compile(c),
        Ir::Do(xs) | Ir::Prim(_, xs) => xs.iter().all(jit_can_compile),
        Ir::Let(inits, body) => inits.iter().all(jit_can_compile) && jit_can_compile(body),
        Ir::Call(f, args) => jit_can_compile(f) && args.iter().all(jit_can_compile),
        Ir::Def { init, .. } => jit_can_compile(init),
        Ir::SetLocal { val, .. } | Ir::SetGlobal { val, .. } => jit_can_compile(val),
    }
}

/// A composed backend: run each body on the native JIT when it can be compiled,
/// and on the stackless `CekMachine` otherwise. This is the tiering the whole
/// `CodeSpace` design was built for — two real strategies behind one seam, chosen
/// per body, with `top` threaded so the choice re-applies to every nested call.
///
/// ## Contract (what "otherwise" safely covers)
///
/// The JIT is a host-stack tier: intermediate values live in native registers /
/// stack frames the CEK cannot see. So this fallback is correct for CEK-only
/// operations that do NOT capture a continuation across a JIT frame — namely
/// `apply` and `values`/`call-with-values`, which the JIT routes to the CEK
/// transparently. It is NOT correct to interleave JIT frames *inside* a
/// `call/cc` capture (a `reset` on the CEK with a `shift` reached through a JIT
/// frame would capture an incomplete continuation). A program using first-class
/// continuations must therefore run WHOLLY on the CEK — the same all-or-nothing
/// rule any host-stack tier faces. The Scheme harness enforces that by running
/// `call/cc` programs directly on `CekMachine`; everything else goes here and
/// runs native wherever possible.
pub struct Tiered<M: ModelArithJit> {
    jit: JitCranelift<M>,
    cek: crate::cek::CekMachine,
}

impl<M: ModelArithJit> Tiered<M> {
    pub fn new() -> Self {
        Tiered {
            jit: JitCranelift::new(),
            cek: crate::cek::CekMachine,
        }
    }

    /// How many bodies were compiled to native code (JIT compile-once counter).
    pub fn native_bodies(&self) -> usize {
        self.jit.compiled_bodies()
    }
}

impl<M: ModelArithJit> CodeSpace<M> for Tiered<M> {
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64 {
        if jit_can_compile(ir) {
            self.jit.eval_ir(top, rt, ir, locals)
        } else {
            self.cek.eval_ir(top, rt, ir, locals)
        }
    }

    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        // Route on the callee's own body: native if it compiles, CEK if not.
        let native = match rt.decode(callee) {
            Val::Ref(id) => match &rt.heap()[id as usize] {
                Obj::Closure { body, .. } => jit_can_compile(body),
                // Route a multi-arity fn on the clause this call selects.
                Obj::MultiFn { .. } => match rt.multifn_select(callee, args.len()) {
                    Some(sel) if !rt.pending() => match rt.decode(sel) {
                        Val::Ref(cid) => match &rt.heap()[cid as usize] {
                            Obj::Closure { body, .. } => jit_can_compile(body),
                            _ => true,
                        },
                        _ => true,
                    },
                    _ => true,
                },
                _ => true, // non-closure callables (escape conts) are the JIT's error path
            },
            _ => true,
        };
        if native {
            self.jit.invoke(top, rt, callee, args)
        } else {
            self.cek.invoke(top, rt, callee, args)
        }
    }
}
