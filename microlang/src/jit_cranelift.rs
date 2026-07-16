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
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::{I128, I64, I8};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::bytecode::ModelEmit;
use crate::code::CodeSpace;
use crate::heap::{
    kind, CLOSURE_CAPS_OFF, CLOSURE_CODE_OFF, CLOSURE_META_OFF, HEADER_SIZE, MULTIFN_FIXED_OFF,
    RECORD_FIELDS_OFF,
};
use crate::ir::{CapSrc, Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::{ObjView, Runtime};
use crate::value::{frame_get, frame_set, Frame, Locals, Obj, RawTag, Sym, Val};

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
    /// Address of the heap's `AllocWindow` (cursor ptr / base / limit at
    /// offsets 0/8/16 — ABI, see heap.rs), for the inline allocation fast path
    /// (D5). Emitted code re-reads the fields on every allocation: they change
    /// at space flips (under STW) and `limit == 0` is gc-stress mode.
    alloc_window: Cell<*const u8>,
    /// Address of `Shared.relocated` (the collection counter) — one half of the
    /// dispatch-IC epoch. Stable for the runtime's lifetime; mutated only under
    /// STW, so a plain load is sound.
    reloc_ptr: Cell<*const u64>,
    /// Address of `Shared.dispatch_version` (bumped per `register_method`) —
    /// the other epoch half, so a redefinition invalidates immediately.
    dispver_ptr: Cell<*const u64>,
    /// Address of the heap's one-byte safepoint poll word (`Heap::poll`) —
    /// emitted code polls it at body entry and loop back-edges (Stage E).
    poll_ptr: Cell<*const u8>,
    /// Is the native fast call path enabled? True only when this backend is the
    /// OUTERMOST one (`top == self`). When wrapped (e.g. by `Traced`, which must
    /// observe every call, or a future router), this is 0 and every call takes the
    /// shim path through `top` — so composition is preserved exactly.
    direct: Cell<u8>,
    /// The bits of the closure currently running (0 if none / top-level). A tail
    /// call to THIS same closure becomes an in-place native loop (redefine the
    /// SSA loop variables, branch back) — O(1) stack, no FFI. A tail call to
    /// anything else falls back to the shim + trampoline, so mutual tail
    /// recursion keeps full TCO. Emitted bodies read this ONCE at entry into a
    /// stack-mapped SSA variable; capture reads derive the capture base from
    /// it per read (Stage E — no cached derived pointer can go stale).
    self_closure: Cell<u64>,
    /// The activation frame, as an `Arc` handle — only for MEMORY-mode bodies
    /// (those containing `try`/`(gc)`/`%await`, whose locals must be GC roots /
    /// interpreter-visible). SSA-mode bodies never touch it.
    cur: RefCell<Locals>,
    /// The backend itself (type-erased `*const JitCranelift<M>`), so shims can
    /// consult its compiled-body cache / per-template compiled-code map. Null
    /// when the running backend is wrapped (non-direct) — callers must check.
    jit: *const (),
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

/// Allocate a closure: the caller (emitted `Ir::Lambda` code) has already
/// RESOLVED the capture values (from its SSA registers / frame slots / own
/// captures) and spilled them, in order, at `caps_ptr`; `nparams`/`variadic`/
/// `nslots` are the Lambda node's own (compile-time-constant) arity, baked in
/// as immediates. `template_id` is the STABLE, GLOBAL id `rt.register_template`
/// handed out for this Lambda's body at compile time — the SAME id
/// `ObjView::Closure::template` reports, and the index into `template_code`.
extern "C" fn shim_make_closure<M: ModelArithJit>(
    ctx: *mut JitCtx<M>,
    template_id: u32,
    nparams: u32,
    variadic: u32,
    nslots: u32,
    caps_ptr: *const u64,
    ncaps: u32,
) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let caps: &[u64] =
        if ncaps == 0 { &[] } else { unsafe { std::slice::from_raw_parts(caps_ptr, ncaps as usize) } };
    let variadic = variadic != 0;
    let g = rt.alloc_closure(nparams as usize, variadic, nslots as u16, template_id, caps);
    // Stamp the new closure's CODE word NOW when the body is already compiled:
    // a freshly-allocated closure that is called once — every lazy-seq thunk,
    // every per-element step fn — would otherwise take the slow invoke on its
    // only call. The object IS the fast-call table (see
    // docs/STAGE_D_MIGRATION.md), so this is a single conditional store.
    let jitp = unsafe { (*ctx.rc).jit };
    if !jitp.is_null() && !variadic && (nparams as usize) <= MAX_REG_ARGS {
        let jit = unsafe { &*(jitp as *const JitCranelift<M>) };
        let code = jit.template_code.borrow().get(template_id as usize).copied().unwrap_or(std::ptr::null());
        if !code.is_null() {
            unsafe { *(g.0.add(CLOSURE_CODE_OFF) as *mut u64) = code as u64 };
        }
    }
    M::R::enc_ref(g)
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
    finish_tail_fast::<M>(ctx.rc, rt, ctx)
}

/// Finish a pending NON-SELF tail chain on the register path. This is the
/// per-element shape of every transducer/protocol step fn (`(rf a (inc x))`
/// ends in a tail call to `rf`), so bounce it in a LOOP — bounded stack,
/// exactly like the trampoline's bounce but without the frame/resolve
/// machinery: select the arity clause, jump straight to stamped code.
/// Anything the register path can't take (rest-arity clause, unstamped code,
/// oversize arity, non-direct top) finishes through `top.invoke` as before.
///
/// `ctx` is reused for the bounced calls; only the fields the emitted
/// fast-call path initializes (`rc`, `self_closure`, `tail_pending`,
/// `tail_callee`) are touched — the 3-store stack contexts are valid here.
fn finish_tail_fast<M: ValueModel>(
    rc: *const JitCtx<M>,
    rt: &mut Runtime<M>,
    ctx: &JitCtx<M>,
) -> u64 {
    let rcr = unsafe { &*rc };
    let top = rcr.top;
    let cp = ctx as *const JitCtx<M> as u64;
    'bounce: loop {
        ctx.tail_pending.set(false);
        let next = ctx.tail_callee.get();
        let tbuf = unsafe { &*rcr.tail_args };
        'fast: {
            if rcr.direct.get() == 0 {
                break 'fast; // a wrapper is observing calls: through `top`
            }
            let tn = tbuf.len();
            if tn > MAX_REG_ARGS {
                break 'fast;
            }
            // Same guards as the register call sites: arity clause, real
            // closure, fixed arity match, stamped code.
            let sel = match rt.multifn_select_raw(next, tn) {
                Some(sel) => {
                    if rt.pending() {
                        return M::R::enc_nil();
                    }
                    sel
                }
                None => next,
            };
            if M::R::tag_of(sel) != RawTag::Ref {
                break 'fast;
            }
            let g = M::R::as_ref(sel);
            if unsafe { g.type_id() } != kind::CLOSURE {
                break 'fast;
            }
            let meta = unsafe { *(g.0.add(CLOSURE_META_OFF) as *const u64) };
            if crate::heap::meta_variadic(meta) || crate::heap::meta_nparams(meta) != tn {
                break 'fast;
            }
            let code = unsafe { *(g.0.add(CLOSURE_CODE_OFF) as *const u64) } as *const u8;
            if code.is_null() {
                break 'fast;
            }
            let mut a = [0u64; MAX_REG_ARGS];
            a[..tn].copy_from_slice(tbuf);
            ctx.self_closure.set(sel);
            let ret = unsafe {
                match tn {
                    0 => std::mem::transmute::<*const u8, extern "C" fn(u64) -> u64>(code)(cp),
                    1 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64) -> u64>(code)(cp, a[0]),
                    2 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64) -> u64>(code)(cp, a[0], a[1]),
                    3 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2]),
                    4 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3]),
                    5 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4]),
                    6 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5]),
                    7 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5], a[6]),
                    8 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]),
                    _ => unreachable!(),
                }
            };
            if !ctx.tail_pending.get() {
                return ret;
            }
            continue 'bounce;
        }
        // Slow finish: through `top`, preserving composition.
        let targs = tbuf.clone();
        return top.invoke(top, rt, next, &targs);
    }
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
    // Direct fast path (multifn arity selection + register entry) with the
    // caller's run context; falls back through `top` so composition /
    // macro-reentrancy hold, exactly like the other tiers' `invoke` call sites.
    if unsafe { (*ctx.rc).direct.get() } != 0 {
        if let Some(r) = shim_fast_invoke::<M>(ctx.rc, rt, callee, args) {
            return r;
        }
    }
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

// ── Stage F2: MONOMORPHIC register-arg shims for the hot collection prims ──
// The generic `shim_prim` costs an arg spill to a stack slot, a
// `prim_from_tag` round trip, and the giant prim match — per element in
// collection-building loops (the vecbuild/group-by profile). These call the
// runtime methods directly with register args, exactly the treatment the
// arithmetic slow paths got in Stage A3. They ALLOCATE, and are classified
// PARKING for the fence policy (plain `call_shim`; `body_pure_loop` treats
// them as impure), so loops around them use Cranelift's precise demotion.

extern "C" fn shim_pv_conj<M: ModelArithJit>(ctx: *mut JitCtx<M>, pv: u64, o: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.pv_conj(pv, o)
}

extern "C" fn shim_pv_nth<M: ModelArithJit>(ctx: *mut JitCtx<M>, pv: u64, i: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let i = rt.raw_i64(i, "pv-nth: index must be an int");
    rt.pv_nth(pv, i)
}

extern "C" fn shim_pv_assoc<M: ModelArithJit>(ctx: *mut JitCtx<M>, pv: u64, i: u64, val: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let i = rt.raw_i64(i, "pv-assoc: index must be an int");
    rt.pv_assoc(pv, i, val)
}

extern "C" fn shim_hamt_assoc<M: ModelArithJit>(ctx: *mut JitCtx<M>, root: u64, key: u64, val: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let (new_root, added) = rt.hamt_map_assoc(root, key, val);
    let addedb = rt.encode(Val::Bool(added));
    M::R::enc_ref(rt.alloc_vector(&[new_root, addedb]))
}

extern "C" fn shim_hamt_lookup<M: ModelArithJit>(ctx: *mut JitCtx<M>, root: u64, key: u64, nf: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.hamt_map_lookup(root, key, nf)
}

extern "C" fn shim_arr_push<M: ModelArithJit>(ctx: *mut JitCtx<M>, arr: u64, v: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let Some(g) = rt.raw_gc(arr) else { panic!("apush: not an array") };
    rt.arr_extend(g, &[v]);
    arr
}

extern "C" fn shim_tv_conj<M: ModelArithJit>(ctx: *mut JitCtx<M>, tv: u64, x: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.tv_conj(tv, x)
}

extern "C" fn shim_tam_assoc<M: ModelArithJit>(ctx: *mut JitCtx<M>, tam: u64, k: u64, v: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.tam_assoc(tam, k, v)
}

extern "C" fn shim_thm_assoc<M: ModelArithJit>(ctx: *mut JitCtx<M>, thm: u64, k: u64, v: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.thm_assoc(thm, k, v)
}

extern "C" fn shim_cons2<M: ModelArithJit>(ctx: *mut JitCtx<M>, h: u64, t: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.cons(h, t)
}

extern "C" fn shim_first1<M: ModelArithJit>(ctx: *mut JitCtx<M>, v: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    // Exactly `rt.prim(Prim::First, ..)`'s arm.
    if let Some((h, _)) = rt.as_cons(v) {
        h
    } else if let Some((arr, off, _, _)) = rt.as_chunked(v) {
        rt.arr_at_pub(arr, off as usize)
    } else {
        rt.enc_nil()
    }
}

extern "C" fn shim_rest1<M: ModelArithJit>(ctx: *mut JitCtx<M>, v: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    // Exactly `rt.prim(Prim::Rest, ..)`'s arm.
    if let Some((_, t)) = rt.as_cons(v) {
        t
    } else if let Some((arr, off, end, more)) = rt.as_chunked(v) {
        if off + 1 < end {
            rt.mk_chunked(arr, off + 1, end, more)
        } else {
            more
        }
    } else {
        rt.enc_nil()
    }
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
/// Invoke `callee` with `args` REUSING the caller's run context: multifn
/// arity selection, a fast-path read straight off the callee OBJECT (header
/// type_id, meta arity, code word — the SAME table the emitted call site
/// reads), a minimal child context, and a register-arg `call_indirect`-
/// equivalent — no `make_ctx`, no trampoline, no `resolve_call`. `None` when
/// the callee is not a compiled, matching-arity closure (the caller falls
/// back to `top.invoke`). Only sound when `direct` (checked by the caller):
/// it bypasses `top`.
fn shim_fast_invoke<M: ValueModel>(
    rc: *const JitCtx<M>,
    rt: &mut Runtime<M>,
    callee: u64,
    args: &[u64],
) -> Option<u64> {
    let rcr = unsafe { &*rc };
    let n = args.len();
    if n > MAX_REG_ARGS {
        return None;
    }
    // Multi-arity: select the clause serving this arg count (cheap heap read;
    // errors pend a signal and yield nil like every shim error path).
    let callee = match rt.multifn_select_raw(callee, n) {
        Some(sel) => {
            if rt.pending() {
                return Some(M::R::enc_nil());
            }
            sel
        }
        None => callee,
    };
    if M::R::tag_of(callee) != RawTag::Ref {
        return None;
    }
    let g = M::R::as_ref(callee);
    if unsafe { g.type_id() } != kind::CLOSURE {
        return None;
    }
    let meta = unsafe { *(g.0.add(CLOSURE_META_OFF) as *const u64) };
    if crate::heap::meta_variadic(meta) || crate::heap::meta_nparams(meta) != n {
        return None;
    }
    let code = unsafe { *(g.0.add(CLOSURE_CODE_OFF) as *const u64) } as *const u8;
    if code.is_null() {
        return None;
    }
    // Minimal child context: everything shared rides through `rc`.
    let mut ctx = JitCtx {
        rc,
        top: rcr.top,
        rt: rt as *mut Runtime<M>,
        cur_slots: Cell::new(std::ptr::null()),
        consts_base: Cell::new(std::ptr::null()),
        global_base: Cell::new(std::ptr::null()),
        global_len: Cell::new(0),
        alloc_window: Cell::new(std::ptr::null()),
        reloc_ptr: Cell::new(std::ptr::null()),
        dispver_ptr: Cell::new(std::ptr::null()),
        poll_ptr: Cell::new(std::ptr::null()),
        direct: Cell::new(1),
        self_closure: Cell::new(callee),
        cur: RefCell::new(None),
        jit: rcr.jit,
        tail_pending: Cell::new(false),
        tail_callee: Cell::new(0),
        tail_args: rcr.tail_args,
    };
    let cp = &mut ctx as *mut JitCtx<M> as u64;
    let a = args;
    let ret = unsafe {
        match n {
            0 => std::mem::transmute::<*const u8, extern "C" fn(u64) -> u64>(code)(cp),
            1 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64) -> u64>(code)(cp, a[0]),
            2 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64) -> u64>(code)(cp, a[0], a[1]),
            3 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2]),
            4 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3]),
            5 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4]),
            6 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5]),
            7 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5], a[6]),
            8 => std::mem::transmute::<*const u8, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(code)(cp, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]),
            _ => unreachable!(),
        }
    };
    if ctx.tail_pending.get() {
        // The callee ended in a NON-self tail call: bounce the chain on the
        // register path (see `finish_tail_fast`).
        return Some(finish_tail_fast::<M>(rc, rt, &ctx));
    }
    Some(ret)
}

extern "C" fn shim_dispatch<M: ModelArithJit>(
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
    // Refill this site's emitted 2-way IC — but only when the installed
    // dispatch strategy is a pure lookup (`thread_cacheable`): an OBSERVING
    // strategy (per-site ICs, speculation counters) must keep seeing every
    // repeat dispatch, exactly the rule `resolve_or_default` applies to the
    // runtime's own per-thread cache. A slot exists only for sites compiled
    // with the inline path (INLINE_OBJECTS models).
    let jitp = unsafe { (*ctx.rc).jit };
    if !jitp.is_null() {
        let jit = unsafe { &*(jitp as *const JitCranelift<M>) };
        let mut ics = jit.dispatch_ic.borrow_mut();
        if let Some(slot) = ics.get_mut(site as usize) {
            if rt.shared.tables.lock().unwrap().dispatch.thread_cacheable() {
                let reloc = rt.relocated();
                let ver = rt.shared.dispatch_version.load(Ordering::Relaxed);
                let way = (ty as usize) & (DISPATCH_IC_WAYS - 1);
                slot[way] =
                    DispatchIcEntry { epoch: dispatch_epoch(reloc, ver), ty: ty as u64, imp };
            }
        }
    }
    // Direct fast path: call the impl's register entry with the caller's run
    // context (no make_ctx / trampoline / resolve). Only when unwrapped.
    if unsafe { (*ctx.rc).direct.get() } != 0 {
        if let Some(r) = shim_fast_invoke::<M>(ctx.rc, rt, imp, args) {
            return r;
        }
    }
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

/// `(apply f a … lst)` — invoke `f` on the leading args + the elements of the
/// final sequence, flattened NATIVELY (`seq_flatten` walks cons/chunked spines
/// and forces lazy nodes through the registered `seq` fn). The rest arg is
/// deliberately re-materialized as a realized list by the invoke path: this
/// dialect's variadic bodies may walk their rest arg with raw `%first`/`%rest`
/// prims, so handing them a lazy-tailed seq would break them.
extern "C" fn shim_apply<M: ModelArithJit>(
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
    // F4 FAST PATH: a SHORT, fully REALIZED cons-list tail (the overwhelmingly
    // common shape — an `&`-rest list or a literal list) flattens into a stack
    // buffer and invokes directly: no Vec round trip, no seq forcing, and the
    // register-arity fast invoke when unwrapped. Anything else — chunked,
    // lazy, vectors, > 8 total args — takes the general seq_flatten path
    // unchanged. Semantics identical (a realized list stays realized; the
    // callee's rest-arg materialization in build_call_frame is the same).
    let lead = rest.len().saturating_sub(1);
    if lead < 8 {
        if let Some(&last) = rest.last() {
            let mut buf = [0u64; 8];
            buf[..lead].copy_from_slice(&rest[..lead]);
            let mut n = lead;
            let mut cur = last;
            let realized_short = loop {
                if M::R::tag_of(cur) == RawTag::Nil {
                    break true;
                }
                if let Some(g) = rt.raw_gc(cur) {
                    if unsafe { g.type_id() } == kind::EMPTY_LIST {
                        break true;
                    }
                }
                match rt.as_cons(cur) {
                    Some((h, t)) => {
                        if n >= 8 {
                            break false;
                        }
                        buf[n] = h;
                        n += 1;
                        cur = t;
                    }
                    None => break false,
                }
            };
            if realized_short {
                if unsafe { (*ctx.rc).direct.get() } != 0 {
                    if let Some(r) = shim_fast_invoke::<M>(ctx.rc, rt, f, &buf[..n]) {
                        return r;
                    }
                }
                return top.invoke(top, rt, f, &buf[..n]);
            }
        }
    }
    let mut flat: Vec<u64> = rest[..rest.len().saturating_sub(1)].to_vec();
    if let Some(&last) = rest.last() {
        flat.extend(rt.seq_flatten(top, last));
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
    let mut child = rt.thread_handle();
    // Root the thunk in the CHILD's shadow stack before the thread exists
    // (Stage E): a collection can fire before the worker's first safepoint
    // only after the worker PARKS — and by then the rooted slot has been
    // published and rewritten, so the re-read below always sees the current
    // address. A bare moved `u64` would go stale.
    let thunk_slot = child.push_root(thunk);
    let slot = std::sync::Arc::new(std::sync::Mutex::new(crate::value::FutureSlot {
        handle: None,
        result: None,
    }));
    let slot_worker = slot.clone();
    let slot_obj = slot.clone();
    let handle = std::thread::spawn(move || {
        let cs = JitCranelift::<M>::new();
        let mut crt = child;
        let thunk = crt.root_get(thunk_slot);
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
/// local is a root and survives the move (its slot is rewritten in place).
/// Native-register temporaries in FRAMES BELOW us are covered by the stack
/// maps + frame walker (Stage E). Concurrent `(gc)` calls rendezvous via
/// `stw_collect` (losers park + participate).
extern "C" fn shim_gc<M: ValueModel>(ctx: *mut JitCtx<M>) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let locals = ctx.cur.borrow().clone();
    rt.collect(&locals);
    M::R::enc_nil()
}

/// The emitted POLL's cold path (Stage E): a body's entry or a self-loop
/// back-edge saw a nonzero poll word. This is an ordinary call, so Cranelift
/// spilled every live stack-mapped value around it — parking here (or
/// collecting, on allocation pressure) walks those spill slots via the frame
/// walker and the collector rewrites them in place; the body reloads them
/// when this returns.
///
/// NO `ctx.cur` read here: a native fast call builds a MINIMAL 3-store child
/// context whose other fields (including `cur`) are uninitialized stack
/// memory. Fast bodies' roots live in the stack maps, and memory-mode bodies'
/// frames ride the dynamic env chain (`run_ctx_entry` pushes them), so the
/// safepoint needs no locals of its own.
extern "C" fn shim_safepoint<M: ValueModel>(ctx: *mut JitCtx<M>) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.safepoint(&None);
    0
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
        Nanos => 104,
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
        SortArr => 103,
        TvNew => 117,
        TvConj => 118,
        TvAssoc => 119,
        TvNth => 120,
        TvPersistent => 121,
        TamNew => 122,
        TamAssoc => 123,
        TamDissoc => 124,
        TamPersistent => 125,
        ThmNew => 126,
        ThmAssoc => 127,
        ThmDissoc => 128,
        ThmPersistent => 129,
        TvPop => 130,
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
        104 => Nanos,
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
        103 => SortArr,
        117 => TvNew,
        118 => TvConj,
        119 => TvAssoc,
        120 => TvNth,
        121 => TvPersistent,
        122 => TamNew,
        123 => TamAssoc,
        124 => TamDissoc,
        125 => TamPersistent,
        126 => ThmNew,
        127 => ThmAssoc,
        128 => ThmDissoc,
        129 => ThmPersistent,
        130 => TvPop,
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
    safepoint: FuncId,
    // Stage F2 monomorphic collection shims.
    pv_conj: FuncId,
    pv_nth: FuncId,
    pv_assoc: FuncId,
    hamt_assoc: FuncId,
    hamt_lookup: FuncId,
    arr_push: FuncId,
    cons2: FuncId,
    first1: FuncId,
    rest1: FuncId,
    tv_conj: FuncId,
    tam_assoc: FuncId,
    thm_assoc: FuncId,
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
    safepoint: cranelift_codegen::ir::FuncRef,
    pv_conj: cranelift_codegen::ir::FuncRef,
    pv_nth: cranelift_codegen::ir::FuncRef,
    pv_assoc: cranelift_codegen::ir::FuncRef,
    hamt_assoc: cranelift_codegen::ir::FuncRef,
    hamt_lookup: cranelift_codegen::ir::FuncRef,
    arr_push: cranelift_codegen::ir::FuncRef,
    cons2: cranelift_codegen::ir::FuncRef,
    first1: cranelift_codegen::ir::FuncRef,
    rest1: cranelift_codegen::ir::FuncRef,
    tv_conj: cranelift_codegen::ir::FuncRef,
    tam_assoc: cranelift_codegen::ir::FuncRef,
    thm_assoc: cranelift_codegen::ir::FuncRef,
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

/// Is this body a PURE self-loop candidate: every `Call`/`Dispatch` is in tail
/// position (so the only non-tail calls the body makes are non-parking shims —
/// arithmetic slow paths, prim escapes, the poll)? Such bodies profit from
/// FENCING those shims (`call_shim_fenced`): their loop variables never cross
/// an unfenced call and stay in registers. Bodies with non-tail calls demote
/// their live values at those calls anyway, so fencing would only add
/// merge/spill churn — they use plain calls throughout.
fn body_pure_loop(ir: &Ir, tail: bool) -> bool {
    match ir {
        // A call is pure-loop-compatible only in TAIL position (the self-loop
        // back-edge or a trampolined tail), with a simple callee reference and
        // call-free arguments — a call anywhere else parks.
        Ir::Call(f, args) => {
            tail
                && matches!(f.as_ref(), Ir::Global(_) | Ir::Local { .. } | Ir::Capture(_))
                && args.iter().all(|a| body_pure_loop(a, false))
        }
        Ir::Dispatch { .. } => false,
        Ir::Prim(Prim::Apply | Prim::CallCc | Prim::CallEc | Prim::Reset | Prim::Shift, _) => false,
        // The Stage F2 monomorphic collection shims are PARKING-classified
        // (plain calls) — a loop around them demotes its live values there
        // anyway, so fencing would only add churn.
        Ir::Prim(
            Prim::PvConj | Prim::PvNth | Prim::PvAssoc | Prim::HamtAssoc | Prim::HamtLookup
            | Prim::ArrPush | Prim::TvConj | Prim::TamAssoc | Prim::ThmAssoc,
            _,
        ) => false,
        Ir::If(c, t, e) => {
            body_pure_loop(c, false) && body_pure_loop(t, tail) && body_pure_loop(e, tail)
        }
        Ir::Do(xs) => match xs.split_last() {
            None => true,
            Some((last, init)) => {
                init.iter().all(|x| body_pure_loop(x, false)) && body_pure_loop(last, tail)
            }
        },
        Ir::SetLocal { val, .. } | Ir::SetGlobal { val, .. } => body_pure_loop(val, false),
        Ir::Def { init, .. } => body_pure_loop(init, false),
        Ir::DefMethod { imp, .. } => body_pure_loop(imp, false),
        Ir::FieldGet { obj, .. } => body_pure_loop(obj, false),
        Ir::Prim(_, xs) => xs.iter().all(|x| body_pure_loop(x, false)),
        Ir::Try { .. } => false,
        Ir::Lambda { .. } => true, // creation only; its body compiles separately
        Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Capture(_) | Ir::Global(_) => true,
        Ir::Let(..) => false,
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
    /// Emit: may `(%num-eq a b)` be decided by comparing the two words' BITS?
    /// True whenever NEITHER operand is a heap ref AND the model has no
    /// immediate float (immediates are canonical, so bit-equality is value
    /// equality — exactly `Runtime::equal`'s immediate fast path). The default
    /// is the both-int guard (always sound); a model widens it only when its
    /// non-ref immediates are all canonical (LowBit: int/bool/nil/sym, floats
    /// boxed).
    fn emit_eq_immediates(c: &mut Compiler, a: Value, b: Value) -> Value {
        Self::emit_both_int(c, a, b)
    }
    /// Emit: the signed i64 VALUE of immediate int `v` (untagged).
    fn emit_untag(c: &mut Compiler, v: Value) -> Value;
    /// Emit: encode signed i64 `x` back into an immediate int word.
    fn emit_tag(c: &mut Compiler, x: Value) -> Value;
    /// Emit `(is_ref, addr)` for `v`: `is_ref` is a branch-ready bool, `addr` is
    /// the REAL heap address (mask off the tag — Stage D refs ARE addresses),
    /// meaningful only when `is_ref` holds. Used to resolve a native call target
    /// inline by reading the callee OBJECT directly (header type_id, meta arity,
    /// code word — see `emit_call`). A model may return a constant-false
    /// `is_ref` to opt out of native fast calls entirely (correct — the call
    /// site then always takes the shim path); NanBox/HighBit do this today.
    fn emit_ref_addr(c: &mut Compiler, v: Value) -> (Value, Value);

    /// Emit the raw heap ADDRESS of `v`, which the caller has already PROVEN
    /// is a ref — a mask, no tag test. Unlike `emit_ref_addr` (whose opt-out
    /// only disables native fast CALLS), every model implements this: capture
    /// reads derive the running closure's capture base from its stack-mapped
    /// bits on every model (Stage E).
    fn emit_ref_addr_unchecked(c: &mut Compiler, v: Value) -> Value;

    /// May emitted code read, write, and ALLOCATE heap objects inline under
    /// this model (D5)? Requires a real `emit_ref_addr`, a working
    /// `emit_tag`/`emit_untag` (immediate ints), and `emit_enc_ref`. LowBit
    /// only today; the models that opt out keep every object op on the shims —
    /// correct, just not inline.
    const INLINE_OBJECTS: bool = false;
    /// Emit: encode a raw heap ADDRESS into this model's ref word. Only called
    /// when `INLINE_OBJECTS`; the default is a loud dead-end so a model can
    /// never silently flip the const on without supplying the encoder.
    fn emit_enc_ref(_c: &mut Compiler, _addr: Value) -> Value {
        panic!("emit_enc_ref: INLINE_OBJECTS is on but the model supplied no ref encoder")
    }
    /// Emit: encode a raw `Sym` word (an i64 holding the interner id) into this
    /// model's sym value. Only called when `INLINE_OBJECTS` (the inline
    /// `type-of` fast path reads a record's type sym straight off the object);
    /// same loud dead-end contract as `emit_enc_ref`.
    fn emit_enc_sym(_c: &mut Compiler, _sym: Value) -> Value {
        panic!("emit_enc_sym: INLINE_OBJECTS is on but the model supplied no sym encoder")
    }
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
    fn emit_eq_immediates(c: &mut Compiler, a: Value, b: Value) -> Value {
        // Neither is a ref (tag 0b001): int/bool/nil/sym are all canonical
        // immediates under LowBit (no immediate float), so bits decide `=`.
        let ta = c.fb.ins().band_imm(a, 0b111);
        let na = c.fb.ins().icmp_imm(IntCC::NotEqual, ta, 0b001);
        let tb = c.fb.ins().band_imm(b, 0b111);
        let nb = c.fb.ins().icmp_imm(IntCC::NotEqual, tb, 0b001);
        c.fb.ins().band(na, nb)
    }
    fn emit_untag(c: &mut Compiler, v: Value) -> Value {
        c.fb.ins().sshr_imm(v, 3) // arithmetic shift drops the tag, keeps sign
    }
    fn emit_tag(c: &mut Compiler, x: Value) -> Value {
        c.fb.ins().ishl_imm(x, 3) // tag bits are 0 for an int
    }
    fn emit_ref_addr(c: &mut Compiler, v: Value) -> (Value, Value) {
        // LowBit ref: tag `LB_REF` = 0b001 in the low 3 bits; objects are
        // 8-aligned, so the address IS the payload — mask the tag bits off.
        let tag = c.fb.ins().band_imm(v, 0b111);
        let is_ref = c.fb.ins().icmp_imm(IntCC::Equal, tag, 0b001);
        let addr = c.fb.ins().band_imm(v, !0b111i64);
        (is_ref, addr)
    }
    fn emit_ref_addr_unchecked(c: &mut Compiler, v: Value) -> Value {
        c.fb.ins().band_imm(v, !0b111i64)
    }

    const INLINE_OBJECTS: bool = true;
    fn emit_enc_ref(c: &mut Compiler, addr: Value) -> Value {
        c.fb.ins().bor_imm(addr, 0b001) // matches `LowBit::enc_ref`
    }
    fn emit_enc_sym(c: &mut Compiler, sym: Value) -> Value {
        // matches `LowBit::enc_sym`: id << 3 | LB_SYM
        let sh = c.fb.ins().ishl_imm(sym, 3);
        c.fb.ins().bor_imm(sh, 0b100)
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
    fn emit_ref_addr(c: &mut Compiler, _v: Value) -> (Value, Value) {
        // Opt out of native fast calls under HighBit (always take the shim path).
        let f = c.fb.ins().iconst(cranelift_codegen::ir::types::I8, 0);
        let z = c.fb.ins().iconst(I64, 0);
        (f, z)
    }
    fn emit_ref_addr_unchecked(c: &mut Compiler, v: Value) -> Value {
        // HighBit ref: address in the low 61 bits (matches `HighBit::as_ref`).
        c.fb.ins().band_imm(v, (1i64 << 61) - 1)
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
    fn emit_ref_addr(c: &mut Compiler, _v: Value) -> (Value, Value) {
        // Opt out of native fast calls under NaN-boxing (always take the shim path).
        let f = c.fb.ins().iconst(cranelift_codegen::ir::types::I8, 0);
        let z = c.fb.ins().iconst(I64, 0);
        (f, z)
    }
    fn emit_ref_addr_unchecked(c: &mut Compiler, v: Value) -> Value {
        // NaN-box ref: address in the 47-bit payload (matches `NanBox::as_ref`).
        c.fb.ins().band_imm(v, (1i64 << 47) - 1)
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
    /// Per-TEMPLATE compiled-code map, indexed by the STABLE, GLOBAL template id
    /// `Runtime::register_template` hands out (the same id `ObjView::Closure`
    /// reports). A freshly-created closure of an already-compiled template is
    /// stamped with its code word at creation (`shim_make_closure`); the object
    /// itself is the fast-call table from then on (see
    /// `docs/STAGE_D_MIGRATION.md`'s "Closure object ABI") — this map exists
    /// only to seed THAT stamp, never consulted by emitted call sites directly.
    /// `null` = not yet compiled. Reserved (`TEMPLATE_CODE_CAP`) so the buffer
    /// NEVER reallocates: emitted `Ir::Lambda` sites bake a slot's address as
    /// an immediate (the creation-time code stamp reads through it). Growth
    /// past the cap panics loudly.
    template_code: RefCell<Vec<*const u8>>,
    /// Per-`Ir::Dispatch`-site inline caches for emitted code: 2 ways of
    /// `(epoch, receiver type, impl)`, indexed by site id. Reserved
    /// (`DISPATCH_SITE_CAP`) so emitted sites can bake their entry's address;
    /// FILLED by `shim_dispatch` on the slow path (and only when the installed
    /// strategy is `thread_cacheable` — an observing strategy must see every
    /// repeat dispatch, exactly the runtime's own site-cache rule). The epoch
    /// folds the relocation count and the dispatch version, so a moved impl or
    /// a redefinition never false-hits.
    dispatch_ic: RefCell<Vec<DispatchSiteIc>>,
    _pd: std::marker::PhantomData<fn() -> M>,
}

/// One way of a dispatch-site IC: the folded epoch at fill time, the receiver
/// type `Sym` (as u64; `u64::MAX` = empty, never a real sym), and the impl's
/// bits. Layout is ABI: emitted code reads the three words at 0/8/16.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchIcEntry {
    epoch: u64,
    ty: u64,
    imp: u64,
}

const DISPATCH_IC_WAYS: usize = 2;
type DispatchSiteIc = [DispatchIcEntry; DISPATCH_IC_WAYS];

const DISPATCH_IC_EMPTY: DispatchIcEntry = DispatchIcEntry { epoch: 0, ty: u64::MAX, imp: 0 };

/// Reserved capacities for the two JIT-owned, address-baked tables (see the
/// field docs). Exceeding either is a loud panic, never a silent reallocation
/// under baked pointers.
const TEMPLATE_CODE_CAP: usize = 1 << 20;
const DISPATCH_SITE_CAP: usize = 1 << 17;

/// The dispatch-IC epoch: the same fold `Runtime::resolve_or_default` uses for
/// its own per-site cache (relocation count mixed with the registry version).
fn dispatch_epoch(reloc: u64, ver: u64) -> u64 {
    reloc.wrapping_mul(DISPATCH_EPOCH_MIX) ^ ver
}
const DISPATCH_EPOCH_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

// ─────────────────────────────────────────────────────────────────────────
// Stage E: the native STACK-MAP registry + frame walker.
//
// Every compiled body registers its Cranelift user stack maps here (process-
// global — JIT code is never freed, so the ranges stay valid for the process
// lifetime, across every backend instance and worker thread). At park time
// the walker runs on the CURRENT thread: it follows the frame-pointer chain
// (preserve_frame_pointers is set; Rust keeps FP on the supported targets),
// resolves each return address against the registry, and yields the address
// of every live stack-map spill slot so the collector can rewrite the tagged
// values in place — Cranelift's emitted code reloads them after its call
// resumes, which is exactly the moving-GC contract of
// `declare_value_needs_stack_map`.
// ─────────────────────────────────────────────────────────────────────────

/// One safepoint (= call site) of a compiled body: the return address as a
/// code offset, the frame's active size (SP-to-FP distance in bytes at that
/// PC), and the SP-relative byte offsets of the live root slots.
struct SiteMap {
    ret_off: u32,
    active_size: u32,
    slots: Box<[u32]>,
}

/// One compiled body's code range + its safepoint sites (sorted by `ret_off`,
/// which Cranelift guarantees on emission).
struct CodeMap {
    start: usize,
    end: usize,
    sites: Vec<SiteMap>,
}

/// All compiled bodies, sorted by start address (disjoint ranges — the JIT
/// memory is append-only).
static CODE_MAPS: std::sync::RwLock<Vec<CodeMap>> = std::sync::RwLock::new(Vec::new());

fn register_code_map(start: usize, size: usize, sites: Vec<SiteMap>) {
    if sites.is_empty() {
        return; // nothing to resolve at any PC in this body
    }
    let mut maps = CODE_MAPS.write().unwrap();
    let at = maps.partition_point(|m| m.start < start);
    maps.insert(at, CodeMap { start, end: start + size, sites });
}

/// The current frame pointer — the anchor of the walk. Supported targets keep
/// the FP chain by ABI (aarch64) or rustc default (x86_64 frame pointers).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn current_fp() -> usize {
    let fp: usize;
    unsafe { core::arch::asm!("mov {}, x29", out(reg) fp, options(nostack, nomem)) };
    fp
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn current_fp() -> usize {
    let fp: usize;
    unsafe { core::arch::asm!("mov {}, rbp", out(reg) fp, options(nostack, nomem)) };
    fp
}

/// Walk the CURRENT thread's frame-pointer chain and collect the addresses of
/// every live JIT stack-map slot (installed as `runtime::NATIVE_ROOT_WALKER`).
///
/// Chain shape (identical on aarch64 and x86_64): the FP register points at a
/// saved `(caller_fp, return_address)` pair; the return address identifies the
/// CALL SITE in the caller — precisely the key Cranelift's user stack maps are
/// recorded under — and the caller's frame pointer value gives that frame's
/// FP. A JIT frame's SP at the call is `fp - active_size` (pinned empirically:
/// the aarch64 prologue is `stp fp, lr; mov fp, sp; sub sp, #active`, and
/// `active_size == sp_to_fp` in Cranelift's FrameLayout), so each root slot
/// lives at `fp - active_size + slot_off`. Rust frames in between simply fail
/// to resolve and are skipped; the chain ends at the thread's base (fp 0) or
/// on any non-ascending/misaligned link.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn walk_native_roots() -> Vec<usize> {
    let maps = CODE_MAPS.read().unwrap();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut fp = current_fp();
    let mut hops = 0usize;
    while fp != 0 && fp & 7 == 0 && hops < (1 << 20) {
        let caller_fp = unsafe { *(fp as *const usize) };
        let ra = unsafe { *((fp + 8) as *const usize) };
        let i = maps.partition_point(|m| m.end <= ra);
        if let Some(m) = maps.get(i) {
            if ra >= m.start {
                let off = (ra - m.start) as u32;
                if let Ok(s) = m.sites.binary_search_by_key(&off, |s| s.ret_off) {
                    let site = &m.sites[s];
                    let sp = caller_fp.wrapping_sub(site.active_size as usize);
                    for &slot in site.slots.iter() {
                        out.push(sp + slot as usize);
                    }
                }
            }
        }
        if caller_fp <= fp {
            break; // the chain must ascend (stack grows down)
        }
        fp = caller_fp;
        hops += 1;
    }
    out
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

/// Stage G2 — the VALUES a first compile observes: the triggering invoke's
/// argument words plus the invoked closure's capture words. `(f …)` call
/// sites whose callee is a parameter/capture read look the actual closure up
/// here and specialize on its TEMPLATE (see `try_value_inline_plan`). Pure
/// compile-time observation — the words are only used to pick a plan; the
/// emitted guard re-checks the live value's meta word, so a stale
/// observation can only mean a failed guard, never a wrong answer.
struct SpecEnv {
    args: Vec<u64>,
    caps: Vec<u64>,
}

/// A value-observed inline plan (Stage G2): splice the observed callee's
/// clause body inline behind a TEMPLATE guard — one header + one meta-word
/// compare against the live value (plus the arity-clause load when the
/// observed value was a MultiFn). Guarding on the meta word (template id,
/// arity, nslots, non-variadic — all baked into one u64) instead of the
/// value's bits makes the specialization survive GC moves AND apply to every
/// closure of the template, and is immune to address reuse: a different
/// object at the same address fails the meta compare. Captures are NOT baked
/// — the inlined body reads them off the live guarded value.
struct ValuePlan {
    /// The clause's full meta word (`closure_meta(template, nparams, nslots,
    /// variadic=false)`) — the guard constant.
    meta: u64,
    /// The observed value was a MultiFn: guard selects `fixed[argc]` first.
    multifn: bool,
    nparams: usize,
    nslots: u16,
    body: Arc<Ir>,
    /// Snapshot of the CLAUSE closure's captures at plan time — used only to
    /// plan nested specializations (each nested guard re-checks its own live
    /// value), never emitted as constants.
    caps: Vec<u64>,
}

/// A resolved call target (the payload of the monomorphic inline cache).
/// `epoch` is the collection count (`Runtime::relocated`) at fill time: a
/// moving collection can relocate a closure AND recycle its old address for a
/// different object (semi-space flip — unlike the old append-only heap ids,
/// addresses DO get reused), so a hit also requires the epoch to match. Same
/// invalidation discipline as the runtime's per-site dispatch IC.
struct CallTarget {
    callee: u64,
    epoch: u64,
    nparams: usize,
    variadic: bool,
    nslots: u16,
    template: u32,
    compiled: Arc<Compiled>,
}

/// Cap on pooled frames — plenty for realistic recursion depth, bounded memory.
const FRAME_POOL_CAP: usize = 1024;

/// Ways in the direct-mapped resolved-callee cache.
const CALL_IC_WAYS: usize = 8;

/// The cache way for a callee: refs are 8-aligned heap ADDRESSES (Stage D), so
/// shift past the always-equal low tag/alignment bits and fold.
fn call_ic_way(callee: u64) -> usize {
    ((callee >> 3) as usize) & (CALL_IC_WAYS - 1)
}

impl<M: ModelArithJit> JitCranelift<M> {
    pub fn new() -> Self {
        // Stage E: install the native frame walker so the runtime can find
        // JIT stack-map roots at every park/collection. Without it, a
        // collection concurrent with native frames would corrupt them — on an
        // unsupported target that is exactly what would happen, so refuse
        // loudly instead.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        crate::runtime::NATIVE_ROOT_WALKER
            .store(walk_native_roots as *const () as usize as u64, std::sync::atomic::Ordering::Release);
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        panic!(
            "JitCranelift: Stage E native-frame GC walking is implemented for \
             aarch64/x86_64 only; this target has no walker, so JIT + moving GC \
             would be unsound"
        );

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
        // Stage E: the native frame WALKER resolves live GC roots by walking
        // the FP chain from a parked thread's Rust frames up through its JIT
        // frames — every emitted function must keep the frame-pointer linkage.
        flags.set("preserve_frame_pointers", "true").unwrap();
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
        let mk: extern "C" fn(*mut JitCtx<M>, u32, u32, u32, u32, *const u64, u32) -> u64 = shim_make_closure::<M>;
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
        let sfp: extern "C" fn(*mut JitCtx<M>) -> u64 = shim_safepoint::<M>;
        let pvc: extern "C" fn(*mut JitCtx<M>, u64, u64) -> u64 = shim_pv_conj::<M>;
        let pvn: extern "C" fn(*mut JitCtx<M>, u64, u64) -> u64 = shim_pv_nth::<M>;
        let pva: extern "C" fn(*mut JitCtx<M>, u64, u64, u64) -> u64 = shim_pv_assoc::<M>;
        let hma: extern "C" fn(*mut JitCtx<M>, u64, u64, u64) -> u64 = shim_hamt_assoc::<M>;
        let hml: extern "C" fn(*mut JitCtx<M>, u64, u64, u64) -> u64 = shim_hamt_lookup::<M>;
        let apu: extern "C" fn(*mut JitCtx<M>, u64, u64) -> u64 = shim_arr_push::<M>;
        let cn2: extern "C" fn(*mut JitCtx<M>, u64, u64) -> u64 = shim_cons2::<M>;
        let fs1: extern "C" fn(*mut JitCtx<M>, u64) -> u64 = shim_first1::<M>;
        let rs1: extern "C" fn(*mut JitCtx<M>, u64) -> u64 = shim_rest1::<M>;
        let tvc: extern "C" fn(*mut JitCtx<M>, u64, u64) -> u64 = shim_tv_conj::<M>;
        let tma: extern "C" fn(*mut JitCtx<M>, u64, u64, u64) -> u64 = shim_tam_assoc::<M>;
        let tha: extern "C" fn(*mut JitCtx<M>, u64, u64, u64) -> u64 = shim_thm_assoc::<M>;
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
        builder.symbol("ml_safepoint", sfp as *const u8);
        builder.symbol("ml_pv_conj", pvc as *const u8);
        builder.symbol("ml_pv_nth", pvn as *const u8);
        builder.symbol("ml_pv_assoc", pva as *const u8);
        builder.symbol("ml_hamt_assoc", hma as *const u8);
        builder.symbol("ml_hamt_lookup", hml as *const u8);
        builder.symbol("ml_arr_push", apu as *const u8);
        builder.symbol("ml_cons2", cn2 as *const u8);
        builder.symbol("ml_first1", fs1 as *const u8);
        builder.symbol("ml_rest1", rs1 as *const u8);
        builder.symbol("ml_tv_conj", tvc as *const u8);
        builder.symbol("ml_tam_assoc", tma as *const u8);
        builder.symbol("ml_thm_assoc", tha as *const u8);

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
        let i32t = cranelift_codegen::ir::types::I32;
        let s_load_local = sig(&[ptr, i32t, i32t]);
        let s_load_global = sig(&[ptr, i32t]);
        let s_def_global = sig(&[ptr, i32t, I64]);
        let s_make_closure = sig(&[ptr, i32t, i32t, i32t, i32t, ptr, i32t]);
        let s_call = sig(&[ptr, I64, ptr, i32t]);
        let s_prim = sig(&[ptr, i32t, ptr, i32t]);
        let s_finish_tail = sig(&[ptr]);
        let s_set_local = sig(&[ptr, i32t, i32t, I64]);
        let s_set_global = sig(&[ptr, i32t, I64]);
        let s_field_get = sig(&[ptr, i32t, i32t, I64]);
        let s_dispatch = sig(&[ptr, i32t, i32t, ptr, i32t]);
        let s_def_method = sig(&[ptr, i32t, i32t, I64]);
        let s_apply = sig(&[ptr, ptr, i32t]);
        let s_try = sig(&[ptr, ptr, ptr, ptr, i32t]);
        let s_spawn = sig(&[ptr, I64]);
        let s_await = sig(&[ptr, I64]);
        let s_gc = sig(&[ptr]);
        let s_v1 = sig(&[ptr, I64]);
        let s_v2 = sig(&[ptr, I64, I64]);
        let s_v3 = sig(&[ptr, I64, I64, I64]);

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
            safepoint: decl(&mut module, "ml_safepoint", &s_gc),
            pv_conj: decl(&mut module, "ml_pv_conj", &s_v2),
            pv_nth: decl(&mut module, "ml_pv_nth", &s_v2),
            pv_assoc: decl(&mut module, "ml_pv_assoc", &s_v3),
            hamt_assoc: decl(&mut module, "ml_hamt_assoc", &s_v3),
            hamt_lookup: decl(&mut module, "ml_hamt_lookup", &s_v3),
            arr_push: decl(&mut module, "ml_arr_push", &s_v2),
            cons2: decl(&mut module, "ml_cons2", &s_v2),
            first1: decl(&mut module, "ml_first1", &s_v1),
            rest1: decl(&mut module, "ml_rest1", &s_v1),
            tv_conj: decl(&mut module, "ml_tv_conj", &s_v2),
            tam_assoc: decl(&mut module, "ml_tam_assoc", &s_v3),
            thm_assoc: decl(&mut module, "ml_thm_assoc", &s_v3),
        };

        JitCranelift {
            module: RefCell::new(module),
            fbctx: RefCell::new(FunctionBuilderContext::new()),
            shims,
            cache: RefCell::new(HashMap::default()),
            frame_pool: RefCell::new(Vec::new()),
            call_ic: RefCell::new(std::array::from_fn(|_| None)),
            template_code: RefCell::new(Vec::with_capacity(TEMPLATE_CODE_CAP)),
            dispatch_ic: RefCell::new(Vec::with_capacity(DISPATCH_SITE_CAP)),
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
            build_body::<M>(&mut module, &mut fb, self.shims, None, self as *const Self as *const (), ir, shape, None);
            fb.finalize();
        }
        let out = ctx.func.display().to_string();
        module.clear_context(&mut ctx);
        out
    }

    fn compile(&self, rt: Option<&Runtime<M>>, ir: &Ir, shape: BodyShape) -> Arc<Compiled> {
        self.compile_spec(rt, ir, shape, None)
    }

    fn compile_spec(
        &self,
        rt: Option<&Runtime<M>>,
        ir: &Ir,
        shape: BodyShape,
        spec: Option<SpecEnv>,
    ) -> Arc<Compiled> {
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
            build_body::<M>(&mut module, &mut fb, self.shims, rt, self as *const Self as *const (), ir, shape, spec);
            fb.finalize();
        }

        let n = self.counter.get();
        self.counter.set(n + 1);
        if std::env::var_os("MICROLANG_JIT_TRACE").is_some() {
            eprintln!("[jit-compile] #{n} arity={:?} mem={} nparams={} var={} ir_ptr={:p}", shape.entry_arity, shape.mem_mode, shape.nparams, shape.variadic, ir as *const Ir);
        }
        let name = format!("ml_body_{n}");
        let id = module
            .declare_function(&name, Linkage::Local, &ctx.func.signature)
            .expect("declare body");
        module.define_function(id, &mut ctx).expect("define body");
        // Stage E: lift this body's user stack maps out of the compile context
        // BEFORE it is cleared — (return-address offset, frame active size,
        // SP-relative root slot offsets) per safepoint.
        let (sites, code_size) = {
            let cc = ctx.compiled_code().expect("compiled code available after define_function");
            let sites: Vec<SiteMap> = cc
                .buffer
                .user_stack_maps()
                .iter()
                .map(|(ret_off, active, map)| SiteMap {
                    ret_off: *ret_off,
                    active_size: *active,
                    slots: map.entries().map(|(_ty, off)| off).collect(),
                })
                .collect();
            (sites, cc.buffer.data().len())
        };
        module.clear_context(&mut ctx);
        module.finalize_definitions().expect("finalize");
        let code = module.get_finalized_function(id);
        register_code_map(code as usize, code_size, sites);
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

    fn compiled_body(
        &self,
        rt: &Runtime<M>,
        body: &Arc<Ir>,
        nparams: usize,
        variadic: bool,
        nslots: u16,
        spec: Option<SpecEnv>,
    ) -> Arc<Compiled> {
        let key = Arc::as_ptr(body);
        if let Some(c) = self.cache.borrow().get(&key) {
            return c.clone();
        }
        let c = self.compile_spec(Some(rt), body, Self::body_shape(body, nparams, variadic, nslots), spec);
        self.cache.borrow_mut().insert(key, c.clone());
        c
    }

    /// Resolve a callee through the monomorphic inline cache (the common repeat
    /// case skips decode + heap read + cache lookups entirely). `args` — the
    /// triggering invoke's argument values, when the caller has them — seeds
    /// PARAM-VALUE SPECIALIZATION (Stage G2): a first compile observes the
    /// actual closures sitting in the parameters and splices their bodies
    /// inline behind template guards.
    fn resolve_call(&self, rt: &mut Runtime<M>, callee: u64, args: Option<&[u64]>) -> (usize, bool, u16, u32, Arc<Compiled>) {
        let way = call_ic_way(callee);
        let epoch = rt.relocated();
        if let Some(t) = self.call_ic.borrow()[way].as_ref() {
            if t.callee == callee && t.epoch == epoch {
                return (t.nparams, t.variadic, t.nslots, t.template, t.compiled.clone());
            }
        }
        if M::R::tag_of(callee) != RawTag::Ref {
            panic!("value not callable: {}", rt.print(callee));
        }
        let (nparams, variadic, nslots, template) = match rt.view(callee) {
            ObjView::Closure { nparams, variadic, nslots, template, .. } => {
                (nparams, variadic, nslots, template)
            }
            _ => panic!("value not callable: {}", rt.print(callee)),
        };
        let body = rt.template(template).clone();
        // Spec env for a first compile: the invoke's arg values (only when the
        // arity actually matches — a mismatch is thrown right after resolution)
        // plus this closure's own capture values, so `(f …)` callees held in
        // params OR captures can be observed.
        let spec = args.filter(|a| !variadic && a.len() == nparams).map(|a| SpecEnv {
            args: a.to_vec(),
            caps: {
                // The capture array lives inline in the closure (Stage D):
                // aux = ncaps, values at CLOSURE_CAPS_OFF.
                let g = M::R::as_ref(callee);
                let n = unsafe { g.aux() } as usize;
                (0..n)
                    .map(|i| unsafe { *(g.0.add(CLOSURE_CAPS_OFF + i * 8) as *const u64) })
                    .collect()
            },
        });
        let compiled = self.compiled_body(rt, &body, nparams, variadic, nslots, spec);
        // (MultiFn callees never reach here: `invoke`/the trampoline select the
        // arity clause BEFORE resolution, so `callee` is always a closure.)
        self.call_ic.borrow_mut()[way] = Some(CallTarget {
            callee,
            epoch,
            nparams,
            variadic,
            nslots,
            template,
            compiled: compiled.clone(),
        });
        (nparams, variadic, nslots, template, compiled)
    }

    /// Build the per-run `JitCtx` for one body execution. `self_closure` is the
    /// running closure's bits (0 = none / top-level); `caps_base` — the inline
    /// capture array's address — is derived straight from it (captures live IN
    /// the closure object, Stage D), no separate `Caps` array to carry.
    #[allow(clippy::too_many_arguments)]
    fn make_ctx<'a>(
        &'a self,
        top: &'a dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        frame: &Locals,
        args_buf: &mut Vec<u64>,
        self_closure: u64,
        needs_cur: bool,
    ) -> JitCtx<'a, M> {
        let consts_base = rt.consts_ptr();
        let global_base = rt.global_slots_ptr();
        let global_len = rt.global_slots_len();
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
            alloc_window: Cell::new(&rt.heap().window as *const crate::heap::AllocWindow as *const u8),
            // AtomicU64 is repr(transparent) over u64; both counters are only
            // written under STW / the registry lock, so plain loads in emitted
            // code observe a coherent value.
            reloc_ptr: Cell::new(&rt.shared.relocated as *const AtomicU64 as *const u64),
            dispver_ptr: Cell::new(&rt.shared.dispatch_version as *const AtomicU64 as *const u64),
            poll_ptr: Cell::new(&rt.heap().poll as *const std::sync::atomic::AtomicU8 as *const u8),
            direct: Cell::new(direct),
            self_closure: Cell::new(self_closure),
            cur: RefCell::new(cur),
            jit: self as *const JitCranelift<M> as *const (),
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
        args_buf: &mut Vec<u64>,
        self_closure: u64,
    ) -> Result<u64, u64> {
        // Stage E: a collection triggered anywhere inside the body must trace
        // this frame (its slots are the body's locals in memory mode, and the
        // SSA prologue reads them at entry) — push it on the dynamic env
        // chain exactly like the interpreter tiers do around invoke.
        rt.env_stack.push(frame.clone());
        let mut ctx =
            self.make_ctx(top, rt, frame, args_buf, self_closure, compiled.mem_mode);
        ctx.rc = &ctx as *const JitCtx<M>;
        let f: extern "C" fn(*mut JitCtx<M>) -> u64 =
            unsafe { std::mem::transmute::<*const u8, _>(compiled.code) };
        let ret = f(&mut ctx as *mut JitCtx<M>);
        rt.env_stack.pop();
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
        args_buf: &mut Vec<u64>,
        self_closure: u64,
    ) -> Result<u64, u64> {
        let none: Locals = None;
        let mut ctx = self.make_ctx(top, rt, &none, args_buf, self_closure, false);
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
        callee_bits: u64,
    ) -> Locals {
        let popped = self.frame_pool.borrow_mut().pop();
        let mut rc = match popped {
            Some(rc) => rc,
            None => return rt.build_call_frame(nparams, variadic, nslots, args, callee_bits),
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
        // `caps_src` is the traced GC root the collector rewrites on a move and
        // `frame_cap`/other backends decode to reach this frame's captures — the
        // Stage D replacement for the old per-frame `Caps` array (see `value.rs`).
        f.caps_src.store(callee_bits, Ordering::Release);
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
                f.caps_src.store(0, Ordering::Release); // don't pin a closure alive in the pool
                let mut pool = self.frame_pool.borrow_mut();
                if pool.len() < FRAME_POOL_CAP {
                    pool.push(rc);
                }
            }
        }
    }

    /// Publish a closure's native fast entry: seed the per-TEMPLATE compiled-
    /// code map (so a LATER closure created from the same template is stamped
    /// at creation by `shim_make_closure`) and, when `callee` is itself a
    /// genuine CLOSURE object (a MultiFn's own bits reach here too — it has no
    /// code word, so it is left alone and always takes the shim path, which
    /// re-selects the arity clause anyway), stamp ITS code word — the object
    /// IS the fast-call table from then on (see docs/STAGE_D_MIGRATION.md).
    fn publish_fast_target(&self, callee: u64, template: u32, compiled: &Compiled) {
        if compiled.entry_arity.is_none() {
            return;
        }
        {
            let mut tc = self.template_code.borrow_mut();
            if tc.len() <= template as usize {
                // The buffer must NEVER reallocate: emitted Lambda sites bake
                // slot addresses (see `template_code_slot`).
                assert!(
                    (template as usize) < TEMPLATE_CODE_CAP,
                    "template_code overflow: template id {template} exceeds TEMPLATE_CODE_CAP — raise it"
                );
                tc.resize(template as usize + 1, std::ptr::null());
            }
            if tc[template as usize].is_null() {
                tc[template as usize] = compiled.code;
            }
        }
        if M::R::tag_of(callee) == RawTag::Ref {
            let g = M::R::as_ref(callee);
            if unsafe { g.type_id() } == kind::CLOSURE {
                unsafe {
                    let code_slot = g.0.add(CLOSURE_CODE_OFF) as *mut u64;
                    if *code_slot == 0 {
                        *code_slot = compiled.code as u64;
                    }
                }
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
        mut cur_callee: u64,
        mut cur_nparams: usize,
        mut cur_variadic: bool,
        mut cur_nslots: u16,
        first_args: &[u64],
    ) -> u64 {
        // The bounce buffer starts EMPTY (no allocation — `Vec::new` is
        // malloc-free): the first iteration reads the caller's slice, and the
        // buffer is only ever filled by a tail bounce (`shim_tail_call`). The
        // per-invoke `to_vec` this replaces was a malloc/free pair on EVERY
        // call through `invoke` — a top frame of the transduce/comp profile.
        let mut args_buf: Vec<u64> = Vec::new();
        let mut from_buf = false;
        loop {
            let outcome = if let Some(k) = compiled.entry_arity {
                let mut regs = [0u64; MAX_REG_ARGS];
                {
                    let src: &[u64] = if from_buf { &args_buf } else { first_args };
                    debug_assert_eq!(src.len(), k);
                    regs[..k].copy_from_slice(&src[..k]);
                }
                self.run_reg_entry(top, rt, &compiled, k, &regs, &mut args_buf, cur_callee)
            } else {
                // Ctx entry: needs a real activation frame.
                if frame.is_none() {
                    let src: &[u64] = if from_buf { &args_buf } else { first_args };
                    // (alloc_frame reads `src` before the entry runs; the
                    // buffer is handed out mutably only after this borrow ends.)
                    let built = self.alloc_frame(
                        rt,
                        cur_nparams,
                        cur_variadic,
                        cur_nslots,
                        src,
                        cur_callee,
                    );
                    frame = built;
                }
                self.run_ctx_entry(top, rt, &compiled, &frame, &mut args_buf, cur_callee)
            };
            match outcome {
                Ok(v) => {
                    self.recycle(frame);
                    return v;
                }
                Err(callee) => {
                    // The bounce wrote the next call's args into the buffer.
                    from_buf = true;
                    // A signal raised while evaluating the tail call's args: stop.
                    if rt.pending() {
                        self.recycle(frame);
                        return M::R::enc_nil();
                    }
                    self.recycle(std::mem::take(&mut frame));
                    // TRAMPOLINE SAFEPOINT (Stage E): this loop is the back-edge
                    // of every non-self tail recursion (mutual loops, variadic /
                    // memory-mode loops), so it must poll too. The in-flight
                    // callee + args are bare bits — root them across the park /
                    // collection and read the (possibly relocated) values back.
                    let callee = if rt.heap().poll.load(Ordering::Acquire) != 0 {
                        let base = rt.push_root(callee);
                        for &a in args_buf.iter() {
                            rt.push_root(a);
                        }
                        rt.safepoint(&None);
                        let c = rt.root_get(base);
                        for (i, a) in args_buf.iter_mut().enumerate() {
                            *a = rt.root_get(base + 1 + i);
                        }
                        rt.truncate_roots(base);
                        c
                    } else {
                        callee
                    };
                    // Callable-object hook in TAIL position too (keywords / maps /
                    // callable deftype records): route `(obj args…)` to
                    // `(handler obj args…)`, exactly as `invoke` does.
                    let callee = if is_record_ref::<M>(callee) && rt.apply_handler().is_some() {
                        let h = rt.apply_handler().unwrap();
                        args_buf.insert(0, callee);
                        h
                    } else {
                        callee
                    };
                    let callee = match rt.multifn_select_raw(callee, args_buf.len()) {
                        Some(sel) => {
                            if rt.pending() {
                                return M::R::enc_nil();
                            }
                            sel
                        }
                        None => callee,
                    };
                    let (nparams, variadic, nslots, template, comp) =
                        self.resolve_call(rt, callee, Some(&args_buf));
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
                    self.publish_fast_target(callee, template, &comp);
                    compiled = comp;
                    cur_callee = callee;
                    cur_nparams = nparams;
                    cur_variadic = variadic;
                    cur_nslots = nslots;
                }
            }
        }
    }
}

/// Does `bits` reference a user RECORD? The callable-object hook (keywords /
/// maps / vectors / sets / multimethods, all frontend records with a
/// registered apply handler) keys off exactly this check — read straight off
/// the object's own header, no runtime handle needed.
fn is_record_ref<M: ValueModel>(bits: u64) -> bool {
    M::R::tag_of(bits) == RawTag::Ref && unsafe { M::R::as_ref(bits).type_id() } == kind::RECORD
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
        // A top-level expr is not a closure body: no current callee (0), so the
        // self-tail-call fast path is inert here (it engages once inside a fn).
        self.run_trampoline(top, rt, compiled, locals.clone(), 0, 0, false, 0, &[])
    }

    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        // Callable-object hook (keywords / maps / vectors / sets / multimethods): a
        // non-closure record with a registered apply handler routes to
        // `(handler object args…)`, exactly like the TreeWalk `invoke`.
        let routed;
        let (callee, args) = if is_record_ref::<M>(callee) && rt.apply_handler().is_some() {
            let h = rt.apply_handler().unwrap();
            let mut v = Vec::with_capacity(args.len() + 1);
            v.push(callee);
            v.extend_from_slice(args);
            routed = v;
            (h, routed.as_slice())
        } else {
            (callee, args)
        };
        // Multi-arity fn: select the clause serving this arg count. Remember the
        // MultiFn's own bits — emitted call sites see THOSE as the callee, but a
        // MultiFn object has no code word (only genuine CLOSURE objects do), so
        // `publish_fast_target` below only ever stamps the per-template map for
        // this branch, never the MultiFn's own header.
        let orig_callee = callee;
        let callee = match rt.multifn_select_raw(callee, args.len()) {
            Some(sel) => {
                if rt.pending() {
                    return M::R::enc_nil();
                }
                sel
            }
            None => callee,
        };
        // Resolve through the monomorphic inline cache (decode + heap + compiled
        // all skipped on a repeat callee).
        let (nparams, variadic, nslots, template, compiled) = self.resolve_call(rt, callee, Some(args));
        // Publish the native fast entry so future call sites can jump to it
        // directly — under the value call sites actually see (the MultiFn id
        // when routing happened, the closure id otherwise). When routing DID
        // happen, ALSO stamp the selected CLAUSE object: `shim_fast_invoke`
        // re-selects per call and reads the clause's code word — leaving it
        // null sent every multifn-routed call (the `(xf rf)` step fns, `+`,
        // `conj`, `get`, …) down the resolve/trampoline slow path PER ELEMENT
        // (the Stage G1 finding).
        self.publish_fast_target(orig_callee, template, &compiled);
        if callee != orig_callee {
            self.publish_fast_target(callee, template, &compiled);
        }
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
        self.run_trampoline(top, rt, compiled, None, callee, nparams, variadic, nslots, args)
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
#[allow(clippy::too_many_arguments)]
fn build_body<M: ModelArithJit>(
    module: &mut JITModule,
    fb: &mut FunctionBuilder,
    shims: Shims,
    rt: Option<&Runtime<M>>,
    jit_ptr: *const (),
    ir: &Ir,
    shape: BodyShape,
    spec: Option<SpecEnv>,
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
        safepoint: module.declare_func_in_func(shims.safepoint, fb.func),
        pv_conj: module.declare_func_in_func(shims.pv_conj, fb.func),
        pv_nth: module.declare_func_in_func(shims.pv_nth, fb.func),
        pv_assoc: module.declare_func_in_func(shims.pv_assoc, fb.func),
        hamt_assoc: module.declare_func_in_func(shims.hamt_assoc, fb.func),
        hamt_lookup: module.declare_func_in_func(shims.hamt_lookup, fb.func),
        arr_push: module.declare_func_in_func(shims.arr_push, fb.func),
        cons2: module.declare_func_in_func(shims.cons2, fb.func),
        first1: module.declare_func_in_func(shims.first1, fb.func),
        rest1: module.declare_func_in_func(shims.rest1, fb.func),
        tv_conj: module.declare_func_in_func(shims.tv_conj, fb.func),
        tam_assoc: module.declare_func_in_func(shims.tam_assoc, fb.func),
        thm_assoc: module.declare_func_in_func(shims.thm_assoc, fb.func),
    };

    let mut c = Compiler {
        fb,
        refs,
        ctx_val,
        entry_sigs: HashMap::new(),
        // Type-erased (the `Compiler` struct is not generic over M; the
        // M-generic methods cast it back). Never outlives this call.
        rt_ptr: rt.map_or(std::ptr::null(), |r| r as *const Runtime<M> as *const ()),
        jit_ptr,
        inline_budget: Cell::new(INLINE_TOTAL_BUDGET),
        mem_mode: shape.mem_mode,
        // Stable byte offsets of `JitCtx` fields (repr(C)) for inline reads and for
        // building a callee context on the stack at a native call site.
        off_cur_slots: core::mem::offset_of!(JitCtx<'static, M>, cur_slots) as i32,
        off_consts_base: core::mem::offset_of!(JitCtx<'static, M>, consts_base) as i32,
        off_global_base: core::mem::offset_of!(JitCtx<'static, M>, global_base) as i32,
        off_alloc_window: core::mem::offset_of!(JitCtx<'static, M>, alloc_window) as i32,
        off_reloc_ptr: core::mem::offset_of!(JitCtx<'static, M>, reloc_ptr) as i32,
        off_dispver_ptr: core::mem::offset_of!(JitCtx<'static, M>, dispver_ptr) as i32,
        off_direct: core::mem::offset_of!(JitCtx<'static, M>, direct) as i32,
        off_self_closure: core::mem::offset_of!(JitCtx<'static, M>, self_closure) as i32,
        off_poll_ptr: core::mem::offset_of!(JitCtx<'static, M>, poll_ptr) as i32,
        off_tail_pending: core::mem::offset_of!(JitCtx<'static, M>, tail_pending) as i32,
        off_rc,
        off_rt: core::mem::offset_of!(JitCtx<'static, M>, rt) as i32,
        signal_kind_off: Runtime::<M>::signal_kind_offset() as i32,
        ctx_size: core::mem::size_of::<JitCtx<'static, M>>() as u32,
        loop_header: None,
        loop_nparams: shape.nparams,
        vars: Vec::new(),
        fence_shims: false, // set below once the loop shape is known
        inline_outer_vars: Vec::new(),
        inline_ctx: Vec::new(),
        spec,
        self_var: cranelift_frontend::Variable::from_u32(0), // placeholder; set right below
        poll_counter: cranelift_frontend::Variable::from_u32(0), // placeholder; set with the loop header
        rc_val,
    };

    // The RUNNING CLOSURE's bits, read ONCE into a tracked variable (Stage
    // E): its reads are stack-map-declared and the poll sites copy-spill it,
    // so a collection at any safepoint rewrites the mapped slot and capture
    // reads (which re-derive the capture base from this) always see the
    // object's CURRENT address. Top-level bodies carry 0 (no closure — never
    // a ref under any model, so the collector ignores it).
    c.self_var = c.declare_root_var();
    let sc0 = c.load_ctx_field(c.off_self_closure);
    c.fb.def_var(c.self_var, sc0);

    // SSA mode: every activation slot is a Cranelift variable (register-
    // allocated). Locals hold tagged value bits and survive moving
    // collections through TWO routes: any value READ across a real call is
    // stack-map-declared (the blanket declare in `compile`), and the poll
    // sites copy-spill every variable explicitly (`emit_poll`). Params come
    // from the entry's register args (fast entries) or a frame-load prologue
    // (ctx entries: variadic / >MAX_REG_ARGS); let/catch slots start nil.
    // Memory mode leaves locals in the heap frame (traced via env
    // publication).
    if !shape.mem_mode {
        for _ in 0..shape.nslots {
            let var = c.declare_root_var();
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

    // ENTRY SAFEPOINT POLL (Stage E): with the locals in place, check the poll
    // word; a pending request/pressure takes the cold shim (a call — the live
    // declared values get a stack map there). This is what bounds a native
    // thread's time-to-park.
    c.emit_poll();

    // A self-tail-recursive SSA body gets a loop header: a tail call to the same
    // closure redefines the param variables in place and branches here (an
    // O(1)-stack native loop in REGISTERS), with the shim/trampoline as the
    // fallback for any other tail call. Its back-edge polls are COARSENED
    // through a countdown (see `BACKEDGE_POLL_INTERVAL`), seeded here.
    if !shape.mem_mode && shape.tail_root && !shape.variadic && has_tail_call(ir) {
        c.fence_shims = body_pure_loop(ir, true);
        c.poll_counter = c.fb.declare_var(I64); // untagged: NOT stack-mapped
        let n = c.fb.ins().iconst(I64, BACKEDGE_POLL_INTERVAL);
        c.fb.def_var(c.poll_counter, n);
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
    /// Imported `fn(ctx, a0..a{k-1}) -> u64` signatures for native fast calls,
    /// built lazily per arity.
    entry_sigs: HashMap<usize, cranelift_codegen::ir::SigRef>,
    /// The runtime at COMPILE time (type-erased; may be null when unavailable,
    /// e.g. `dump_ir`) — the var-guarded speculative inliner resolves `Global`
    /// callees through it.
    rt_ptr: *const (),
    /// The owning backend at COMPILE time (type-erased `*const JitCranelift<M>`,
    /// never null) — `Ir::Lambda`/`Ir::Dispatch` sites size + bake addresses
    /// into its reserved `template_code` / `dispatch_ic` tables.
    jit_ptr: *const (),
    /// Remaining inlined-node allowance for THIS body (bounds code growth and
    /// terminates recursive inlining).
    inline_budget: Cell<usize>,
    /// Locals live in the heap frame (via `cur_slots`) instead of SSA variables.
    mem_mode: bool,
    off_cur_slots: i32,
    off_consts_base: i32,
    off_global_base: i32,
    off_alloc_window: i32,
    off_reloc_ptr: i32,
    off_dispver_ptr: i32,
    off_direct: i32,
    off_self_closure: i32,
    off_poll_ptr: i32,
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
    /// memory mode. Rooted via the blanket read-declares + poll copy-spills.
    vars: Vec<cranelift_frontend::Variable>,
    /// FENCE non-parking shim calls (`call_shim_fenced` actually fences only
    /// when set): true for pure self-loop bodies, whose variables then never
    /// cross an unfenced call and stay in registers. See `body_pure_loop`.
    fence_shims: bool,
    /// The ENCLOSING frames' variables while a speculative inline is being
    /// spliced (`emit_inlined_body` swaps `vars`): `spill_roots` must keep
    /// copy-spilling the outer locals around calls inside the inlined region,
    /// or a collection under an inlined deopt call would miss them.
    inline_outer_vars: Vec<Vec<cranelift_frontend::Variable>>,
    /// Stage G2 — one entry per active inline level: the guarded callee VALUE
    /// (`Ir::Capture` inside the inlined body reads THROUGH it — a live,
    /// stack-mapped SSA value, so capture reads always see the post-move
    /// object) and the plan-time capture snapshot (drives NESTED value plans;
    /// never emitted). The legacy capture-free global inliner pushes `None`s.
    inline_ctx: Vec<(Option<Value>, Option<Vec<u64>>)>,
    /// Stage G2 — the values the triggering invoke observed (args + the
    /// invoked closure's captures), when this body's first compile happened
    /// under a real call. `None` for top-level/eval bodies.
    spec: Option<SpecEnv>,
    /// The running closure's bits, loaded once at entry (stack-mapped): the
    /// self-tail-call compare and every capture read go through this, so both
    /// see the closure's CURRENT address after any safepoint (Stage E).
    self_var: cranelift_frontend::Variable,
    /// The back-edge poll COUNTDOWN (self-loop bodies only): the loop checks
    /// the poll word every `BACKEDGE_POLL_INTERVAL` iterations instead of
    /// every iteration — one register decrement + branch on the hot path.
    /// An untagged integer: deliberately NOT stack-mapped.
    poll_counter: cranelift_frontend::Variable,
}

/// Self-tail back-edges poll every N iterations. Time-to-safepoint for a
/// native loop is bounded by N iterations of straight-line arithmetic
/// (microseconds); entry polls still fire on EVERY call, so recursion and
/// call-heavy loops park immediately. 1024 makes the per-iteration cost one
/// dec+brif (~amortized nothing) while keeping gc-stress collections dense.
const BACKEDGE_POLL_INTERVAL: i64 = 1024;

impl<'a, 'b> Compiler<'a, 'b> {
    /// A PARKING shim call (invoke/dispatch/apply/try/await/gc/poll targets —
    /// anywhere a collection can actually happen): plain. The variables'
    /// values live across it are stack-mapped by Cranelift's own
    /// declare-based spilling (`declare_root_var`), with precise liveness.
    fn call_shim(&mut self, f: cranelift_codegen::ir::FuncRef, args: &[Value]) -> Value {
        let inst = self.fb.ins().call(f, args);
        self.fb.inst_results(inst)[0]
    }

    /// A NON-PARKING shim call, FENCED (Stage E): `rt.prim` and the other
    /// pure-runtime shims can never reach a safepoint (no `top`, no park), so
    /// no value can move across them — but Cranelift cannot know that, and a
    /// declared value live across ANY call is demoted to memory for its whole
    /// life. The fence: copy every local + the self bits into fresh values
    /// before the call and REDEFINE the variables after, so the hot values'
    /// live ranges END here — a tight loop whose only in-body calls are
    /// fenced (arithmetic slow paths, the safepoint poll's cold shim) keeps
    /// its variables in registers. The copies are themselves var-bound (and
    /// so mapped), which also makes fencing SOUND even at the one fenced
    /// target that does park: the poll's `shim_safepoint`.
    fn call_shim_fenced(&mut self, f: cranelift_codegen::ir::FuncRef, args: &[Value]) -> Value {
        if !self.fence_shims {
            return self.call_shim(f, args);
        }
        let (roots, copies) = self.spill_roots();
        let inst = self.fb.ins().call(f, args);
        let r = self.fb.inst_results(inst)[0];
        self.reload_roots(&roots, &copies);
        r
    }

    /// Fenced + the pending-signal check (the fenced counterpart of
    /// `call_shim_checked`).
    fn call_shim_fenced_checked(&mut self, f: cranelift_codegen::ir::FuncRef, args: &[Value]) -> Value {
        let result = self.call_shim_fenced(f, args);
        self.emit_pending_check(result)
    }

    /// The pre-call half of the root discipline: copy every variable-held
    /// root into a fresh, declared SSA value (a real instruction, so the
    /// safepoint pass can demote the copy in isolation).
    fn spill_roots(&mut self) -> (Vec<cranelift_frontend::Variable>, Vec<Value>) {
        let mut roots: Vec<cranelift_frontend::Variable> = self.vars.clone();
        for outer in &self.inline_outer_vars {
            roots.extend_from_slice(outer);
        }
        roots.push(self.self_var);
        let mut copies = Vec::with_capacity(roots.len());
        for var in &roots {
            let x = self.fb.use_var(*var);
            let x2 = self.fb.ins().iadd_imm(x, 0);
            self.fb.declare_value_needs_stack_map(x2);
            copies.push(x2);
        }
        (roots, copies)
    }

    /// The post-call half: the variables take the (possibly relocated)
    /// values back from the mapped copies.
    fn reload_roots(&mut self, roots: &[cranelift_frontend::Variable], copies: &[Value]) {
        for (var, x2) in roots.iter().zip(copies) {
            self.fb.def_var(*var, *x2);
        }
    }

    /// Declare an I64 variable whose bindings carry TAGGED VALUE BITS, marked
    /// for the stack maps: any of its values live across a PARKING call (a
    /// real invoke, where a collection can happen) is spilled with a map
    /// entry, rewritten by a moving collection, and reloaded — with
    /// Cranelift-precise liveness, so call-dense code pays only for what is
    /// actually live. The demotion this implies is kept OFF the hot loops by
    /// FENCING every non-parking call (`call_shim_fenced`): a value whose
    /// only crossings are fenced never demotes.
    fn declare_root_var(&mut self) -> cranelift_frontend::Variable {
        let var = self.fb.declare_var(I64);
        self.fb.declare_var_needs_stack_map(var);
        var
    }

    /// The SAFEPOINT POLL (Stage E): one load of the heap's poll byte and a
    /// branch to a cold `shim_safepoint` call when it is nonzero (a sibling
    /// requested a collection, or allocation pressure crossed the threshold).
    /// Emitted at body entry and (countdown-coarsened) at every self-tail
    /// back-edge, which bounds a native loop's time-to-park. Rooting comes
    /// from `call_shim`'s copy-spill discipline.
    fn emit_poll(&mut self) {
        let pp = self.load_rc_field(self.off_poll_ptr);
        let b = self.fb.ins().load(I8, MemFlagsData::trusted(), pp, 0);
        let cold = self.fb.create_block();
        let cont = self.fb.create_block();
        self.fb.set_cold_block(cold);
        self.fb.ins().brif(b, cold, &[], cont, &[]);
        self.fb.switch_to_block(cold);
        self.fb.seal_block(cold);
        let ctx = self.ctx_val;
        self.call_shim_fenced(self.refs.safepoint, &[ctx]);
        self.fb.ins().jump(cont, &[]);
        self.fb.switch_to_block(cont);
        self.fb.seal_block(cont);
    }

    /// The COARSENED self-loop back-edge poll: decrement the (untagged,
    /// unmapped) countdown; only every `BACKEDGE_POLL_INTERVAL`-th iteration
    /// checks the poll word (and reseeds the countdown). Hot path = one
    /// dec + one branch — this is what keeps a tight arithmetic loop at
    /// its pre-Stage-E speed while still bounding time-to-safepoint.
    fn emit_backedge_poll(&mut self) {
        let c = self.fb.use_var(self.poll_counter);
        let c1 = self.fb.ins().iadd_imm(c, -1);
        self.fb.def_var(self.poll_counter, c1);
        let cold = self.fb.create_block();
        let cont = self.fb.create_block();
        self.fb.set_cold_block(cold);
        // nonzero countdown → skip the poll entirely.
        self.fb.ins().brif(c1, cont, &[], cold, &[]);
        self.fb.switch_to_block(cold);
        self.fb.seal_block(cold);
        let n = self.fb.ins().iconst(I64, BACKEDGE_POLL_INTERVAL);
        self.fb.def_var(self.poll_counter, n);
        self.emit_poll();
        self.fb.ins().jump(cont, &[]);
        self.fb.switch_to_block(cont);
        self.fb.seal_block(cont);
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

    /// Read capture `idx` of the RUNNING closure: mask the stack-mapped self
    /// bits to the object's current address, load the inline capture word
    /// (Stage E — the derived pointer never crosses a safepoint; it is built
    /// and consumed within this straight-line sequence).
    ///
    /// Inside a value-observed inline (Stage G2), "the running closure" is the
    /// GUARDED CALLEE, not this function — read through the inline level's
    /// self value instead (also live + stack-mapped, so also post-move-safe).
    fn emit_capture<M: ModelArithJit>(&mut self, idx: u16) -> Value {
        let sc = match self.inline_ctx.last() {
            Some((Some(v), _)) => *v,
            Some((None, _)) => panic!(
                "emit_capture inside a capture-free inline: the global inliner \
                 rejects capture-carrying callees, so this body cannot contain \
                 Ir::Capture — planner bug"
            ),
            None => {
                let sc = self.fb.use_var(self.self_var);
                // Not a `compile()` result, so the blanket declare misses it:
                // mark the self bits here so a read AFTER a real call is in
                // that call's map.
                self.fb.declare_value_needs_stack_map(sc);
                sc
            }
        };
        let base = M::emit_ref_addr_unchecked(self, sc);
        let off = (CLOSURE_CAPS_OFF as i32) + (idx as i32) * 8;
        self.fb.ins().load(I64, MemFlagsData::trusted(), base, off)
    }

    /// Inline bump allocation through the heap's `AllocWindow` (D5): claim
    /// `size` bytes (a compile-time constant, 8-aligned) with one atomic
    /// fetch_add on the active space's cursor and write the header word.
    /// Branches to `slow` when the window is CLOSED (`limit == 0` — gc-stress
    /// mode) or the claim exceeds the limit; the shim slow path owns the
    /// loud-exhaustion panic, so falling through preserves it. An overshooting
    /// claim is NOT undone: `heap.rs` clamps `used()` to the space, so the
    /// dead partial claim is harmless (same discipline as the CAS-free plan in
    /// the AllocWindow docs). Allocation NEVER collects (explicit-GC-only
    /// runtime), so no safepoint is needed here. Returns the object's raw
    /// ADDRESS in a fresh, sealed block; the claimed memory is zeroed (bump
    /// spaces are re-zeroed on reset), so untouched fields read as zero.
    fn emit_alloc(&mut self, size: i64, header: u64, slow: cranelift_codegen::ir::Block) -> Value {
        debug_assert_eq!(size & 7, 0, "inline alloc size must be 8-aligned");
        let flags = MemFlagsData::trusted();
        let win = self.load_rc_field(self.off_alloc_window);
        let limit = self.fb.ins().load(I64, flags, win, 16);
        let openb = self.fb.create_block();
        let nz = self.fb.ins().icmp_imm(IntCC::NotEqual, limit, 0);
        self.fb.ins().brif(nz, openb, &[], slow, &[]);
        self.fb.switch_to_block(openb);
        self.fb.seal_block(openb);
        let cursor_ptr = self.fb.ins().load(I64, flags, win, 0);
        let sz = self.fb.ins().iconst(I64, size);
        let old = self.fb.ins().atomic_rmw(I64, flags, AtomicRmwOp::Add, cursor_ptr, sz);
        let end = self.fb.ins().iadd_imm(old, size);
        let fits = self.fb.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, limit);
        let okb = self.fb.create_block();
        self.fb.ins().brif(fits, okb, &[], slow, &[]);
        self.fb.switch_to_block(okb);
        self.fb.seal_block(okb);
        let base = self.fb.ins().load(I64, flags, win, 8);
        let addr = self.fb.ins().iadd(base, old);
        let hv = self.iconst(header);
        self.fb.ins().store(flags, hv, addr, 0);
        addr
    }

    /// Inline type guard (D5): `v` is a ref whose header `type_id == want`.
    /// Branches to `slow` otherwise; returns `(addr, header)` in a fresh,
    /// sealed continuation block. The header read doubles as the aux source
    /// (array lengths, capture counts) for the caller.
    fn emit_typed_addr<M: ModelArithJit>(
        &mut self,
        v: Value,
        want: u16,
        slow: cranelift_codegen::ir::Block,
    ) -> (Value, Value) {
        let (is_ref, addr) = M::emit_ref_addr(self, v);
        let chk = self.fb.create_block();
        self.fb.ins().brif(is_ref, chk, &[], slow, &[]);
        self.fb.switch_to_block(chk);
        self.fb.seal_block(chk);
        let hdr = self.fb.ins().load(I64, MemFlagsData::trusted(), addr, 0);
        let tid = self.fb.ins().band_imm(hdr, 0xffff);
        let ok = self.fb.ins().icmp_imm(IntCC::Equal, tid, want as i64);
        let okb = self.fb.create_block();
        self.fb.ins().brif(ok, okb, &[], slow, &[]);
        self.fb.switch_to_block(okb);
        self.fb.seal_block(okb);
        (addr, hdr)
    }

    /// The (stable) address of `template_code[tid]`, sizing the reserved table
    /// so the slot exists. Baked as an immediate by `Ir::Lambda` sites — the
    /// creation-time code stamp is one load through it.
    fn template_code_slot<M: ModelArithJit>(&self, tid: u32) -> *const *const u8 {
        let jit = unsafe { &*(self.jit_ptr as *const JitCranelift<M>) };
        let mut tc = jit.template_code.borrow_mut();
        if tc.len() <= tid as usize {
            assert!(
                (tid as usize) < TEMPLATE_CODE_CAP,
                "template_code overflow: template id {tid} exceeds TEMPLATE_CODE_CAP — raise it"
            );
            tc.resize(tid as usize + 1, std::ptr::null());
        }
        unsafe { tc.as_ptr().add(tid as usize) }
    }

    /// The (stable) address of dispatch site `site`'s 2-way IC, sizing the
    /// reserved table so the slot exists. Baked as an immediate by
    /// `Ir::Dispatch` sites; `shim_dispatch` fills the ways on the slow path.
    fn dispatch_ic_slot<M: ModelArithJit>(&self, site: usize) -> *const DispatchIcEntry {
        let jit = unsafe { &*(self.jit_ptr as *const JitCranelift<M>) };
        let mut ics = jit.dispatch_ic.borrow_mut();
        if ics.len() <= site {
            assert!(
                site < DISPATCH_SITE_CAP,
                "dispatch_ic overflow: site {site} exceeds DISPATCH_SITE_CAP — raise it"
            );
            ics.resize(site + 1, [DISPATCH_IC_EMPTY; DISPATCH_IC_WAYS]);
        }
        ics[site].as_ptr()
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
        if M::R::tag_of(bits) != RawTag::Ref {
            return None;
        }
        let g = M::R::as_ref(bits);
        // See through a multi-arity fn: this call site's arg count statically
        // selects one fixed clause. The guard still compares the GLOBAL's bits
        // (the MultiFn), so redefinition deopts as usual.
        let target = if unsafe { g.type_id() } == kind::MULTIFN {
            let ObjView::MultiFn { fixed, .. } = rt.view_gc(g) else { unreachable!() };
            let f = fixed.get(argc).copied().unwrap_or(0);
            if f == 0 {
                return None; // variadic / no such arity: not inlinable
            }
            f
        } else {
            bits
        };
        if M::R::tag_of(target) != RawTag::Ref {
            return None;
        }
        let tg = M::R::as_ref(target);
        if unsafe { tg.type_id() } != kind::CLOSURE {
            return None;
        }
        let (nparams, variadic, nslots, template) = match rt.view_gc(tg) {
            ObjView::Closure { nparams, variadic, nslots, template, .. } => (nparams, variadic, nslots, template),
            _ => unreachable!(),
        };
        // Capture-free (aux == ncaps == 0), non-variadic, matching arity.
        if variadic || nparams != argc || unsafe { tg.aux() } != 0 {
            return None;
        }
        let body = rt.template(template).clone();
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

    /// Stage G2 — the compile-time-OBSERVED value of a callee expression, when
    /// there is one: a parameter read resolves through the triggering invoke's
    /// args, a capture read through the invoked closure's captures; inside a
    /// value-observed inline level, through that level's capture snapshot.
    /// Observation only — the emitted guard re-validates against the live value.
    fn spec_value_of(&self, f: &Ir) -> Option<u64> {
        if self.mem_mode {
            return None;
        }
        match self.inline_ctx.last() {
            Some((_, caps)) => match f {
                Ir::Capture(j) => caps.as_ref()?.get(*j as usize).copied(),
                _ => None, // an inlinee's params/locals: values not observed
            },
            None => match f {
                Ir::Capture(j) => self.spec.as_ref()?.caps.get(*j as usize).copied(),
                // `args.len() == nparams` (resolve_call only builds the env on an
                // arity match), so an in-range idx IS a parameter slot.
                Ir::Local { up: 0, idx } => self.spec.as_ref()?.args.get(*idx as usize).copied(),
                _ => None,
            },
        }
    }

    /// Stage G2 — can a call of OBSERVED value `bits` at an `argc`-arg site be
    /// spliced inline? Like `try_inline_plan` but value-driven: multi-arity
    /// values select their fixed clause, CAPTURE-CARRYING closures are allowed
    /// (reads go through the guarded live value), and the guard is the clause's
    /// meta word rather than the value's bits. Consumes budget on success.
    fn try_value_inline_plan<M: ModelArithJit>(&self, bits: u64, argc: usize) -> Option<ValuePlan> {
        if self.mem_mode || self.rt_ptr.is_null() {
            return None;
        }
        let rt = unsafe { &*(self.rt_ptr as *const Runtime<M>) };
        if M::R::tag_of(bits) != RawTag::Ref {
            return None;
        }
        let g = M::R::as_ref(bits);
        let (target, multifn) = if unsafe { g.type_id() } == kind::MULTIFN {
            let nfixed = unsafe { g.aux() } as usize;
            if argc >= nfixed {
                return None;
            }
            let f = unsafe { *(g.0.add(MULTIFN_FIXED_OFF + argc * 8) as *const u64) };
            if f == 0 {
                return None; // variadic-only at this arity: not inlinable
            }
            (f, true)
        } else {
            (bits, false)
        };
        if M::R::tag_of(target) != RawTag::Ref {
            return None;
        }
        let tg = M::R::as_ref(target);
        if unsafe { tg.type_id() } != kind::CLOSURE {
            return None;
        }
        let meta = unsafe { *(tg.0.add(CLOSURE_META_OFF) as *const u64) };
        if crate::heap::meta_variadic(meta) || crate::heap::meta_nparams(meta) != argc {
            return None;
        }
        let nslots = crate::heap::meta_nslots(meta);
        let body = rt.template(crate::heap::meta_template(meta)).clone();
        if body_mem_mode(&body) {
            return None;
        }
        let cost = node_count_capped(&body, INLINE_MAX_CALLEE_NODES)?;
        let budget = self.inline_budget.get();
        if budget < cost {
            return None;
        }
        self.inline_budget.set(budget - cost);
        let ncaps = unsafe { tg.aux() } as usize;
        let caps = (0..ncaps)
            .map(|i| unsafe { *(tg.0.add(CLOSURE_CAPS_OFF + i * 8) as *const u64) })
            .collect();
        Some(ValuePlan { meta, multifn, nparams: argc, nslots, body, caps })
    }

    /// Splice an inlined callee body into the current function: fresh SSA vars
    /// for its activation slots (params seeded from the evaluated args, the
    /// rest nil), compiled in NON-tail position with no self-loop — its own
    /// tail calls become ordinary calls, which preserves semantics (and one
    /// inlined level never grows the stack unboundedly).
    ///
    /// `self_val`/`caps` (Stage G2): for a value-observed inlinee, the guarded
    /// callee VALUE its capture reads go through, and the plan-time capture
    /// snapshot for nested planning. The capture-free global inliner passes
    /// `None`s.
    fn emit_inlined_body<M: ModelArithJit>(
        &mut self,
        nparams: usize,
        nslots: u16,
        body: &Ir,
        argvals: &[Value],
        self_val: Option<Value>,
        caps: Option<Vec<u64>>,
    ) -> Value {
        let saved_vars = std::mem::take(&mut self.vars);
        self.inline_outer_vars.push(saved_vars.clone());
        self.inline_ctx.push((self_val, caps));
        let saved_header = self.loop_header.take();
        let saved_nparams = self.loop_nparams;
        for i in 0..nslots as usize {
            let var = self.declare_root_var();
            if i < nparams {
                self.fb.def_var(var, argvals[i]);
            } else {
                let nil = self.fb.ins().iconst(I64, M::R::enc_nil() as i64);
                self.fb.def_var(var, nil);
            }
            self.vars.push(var);
        }
        self.loop_nparams = nparams;
        let v = self.compile::<M>(body, false);
        self.inline_ctx.pop();
        self.inline_outer_vars.pop();
        self.vars = saved_vars;
        self.loop_header = saved_header;
        self.loop_nparams = saved_nparams;
        v
    }

    /// Stage G2 — emit a value-specialized call: the plan's clause body spliced
    /// inline behind a TEMPLATE guard on the live callee value (`fval`). Guard
    /// chain (each failure exits to the generic `emit_call` path — never a
    /// wrong answer): direct + ref, then either header == CLOSURE and meta ==
    /// plan.meta, or (multifn shape) header == MULTIFN, `fixed[argc]` present,
    /// and the CLAUSE's meta == plan.meta. Captures inside the inlined body
    /// read through the guarded value (declared for the stack maps), so a
    /// collection mid-body is safe.
    fn emit_value_specialized<M: ModelArithJit>(
        &mut self,
        fval: Value,
        argvals: &[Value],
        plan: &ValuePlan,
    ) -> Value {
        let flags = MemFlagsData::trusted();
        let ro = MemFlagsData::trusted().with_readonly();
        let result = self.declare_root_var();
        let inlb = self.fb.create_block();
        // The inlined body's capture reads go through the guarded CLAUSE value.
        self.fb.append_block_param(inlb, I64);
        let slowb = self.fb.create_block();
        let merge = self.fb.create_block();

        let (is_ref, addr) = M::emit_ref_addr(self, fval);
        let rc = self.rc_val;
        let direct = self.fb.ins().load(I8, ro, rc, self.off_direct);
        let g1 = self.fb.ins().band(is_ref, direct);
        let hdrb = self.fb.create_block();
        self.fb.ins().brif(g1, hdrb, &[], slowb, &[]);

        self.fb.switch_to_block(hdrb);
        self.fb.seal_block(hdrb);
        let hdr = self.fb.ins().load(I64, flags, addr, 0);
        let tid = self.fb.ins().band_imm(hdr, 0xffff);
        if plan.multifn {
            let mfl = self.fb.create_block();
            let metab = self.fb.create_block();
            let is_mf = self.fb.ins().icmp_imm(IntCC::Equal, tid, kind::MULTIFN as i64);
            let aux = self.fb.ins().ushr_imm(hdr, 32);
            let in_range =
                self.fb.ins().icmp_imm(IntCC::UnsignedGreaterThan, aux, plan.nparams as i64);
            let mf_ok = self.fb.ins().band(is_mf, in_range);
            self.fb.ins().brif(mf_ok, mfl, &[], slowb, &[]);
            self.fb.switch_to_block(mfl);
            self.fb.seal_block(mfl);
            let clause = self.fb.ins().load(
                I64,
                flags,
                addr,
                MULTIFN_FIXED_OFF as i32 + (plan.nparams as i32) * 8,
            );
            // The clause value feeds capture reads across safepoints inside the
            // inlined body: declare it so it is spilled + rewritten on a move.
            self.fb.declare_value_needs_stack_map(clause);
            let has_clause = self.fb.ins().icmp_imm(IntCC::NotEqual, clause, 0);
            self.fb.ins().brif(has_clause, metab, &[], slowb, &[]);
            self.fb.switch_to_block(metab);
            self.fb.seal_block(metab);
            let caddr = M::emit_ref_addr_unchecked(self, clause);
            let cmeta = self.fb.ins().load(I64, flags, caddr, CLOSURE_META_OFF as i32);
            let want = self.iconst(plan.meta);
            let meta_ok = self.fb.ins().icmp(IntCC::Equal, cmeta, want);
            self.fb.ins().brif(meta_ok, inlb, &[clause.into()], slowb, &[]);
        } else {
            let metab = self.fb.create_block();
            let is_cl = self.fb.ins().icmp_imm(IntCC::Equal, tid, kind::CLOSURE as i64);
            self.fb.ins().brif(is_cl, metab, &[], slowb, &[]);
            self.fb.switch_to_block(metab);
            self.fb.seal_block(metab);
            let meta = self.fb.ins().load(I64, flags, addr, CLOSURE_META_OFF as i32);
            let want = self.iconst(plan.meta);
            let meta_ok = self.fb.ins().icmp(IntCC::Equal, meta, want);
            self.fb.ins().brif(meta_ok, inlb, &[fval.into()], slowb, &[]);
        }

        self.fb.switch_to_block(inlb);
        self.fb.seal_block(inlb);
        let target = self.fb.block_params(inlb)[0];
        // A block param is not a `compile()` result: declare it so capture
        // reads after an inner safepoint see the post-move bits.
        self.fb.declare_value_needs_stack_map(target);
        let iv = self.emit_inlined_body::<M>(
            plan.nparams,
            plan.nslots,
            &plan.body,
            argvals,
            Some(target),
            Some(plan.caps.clone()),
        );
        self.fb.def_var(result, iv);
        self.fb.ins().jump(merge, &[]);

        // Deopt: the ordinary call path on the SAME live value.
        self.fb.switch_to_block(slowb);
        self.fb.seal_block(slowb);
        let sv = self.emit_call::<M>(fval, argvals);
        self.fb.def_var(result, sv);
        self.fb.ins().jump(merge, &[]);

        self.fb.switch_to_block(merge);
        self.fb.seal_block(merge);
        self.fb.use_var(result)
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

    /// A non-tail call. Try the NATIVE fast path — the callee bits ARE its heap
    /// address (Stage D), so the call site reads the CLOSURE OBJECT directly: a
    /// tag test, the header's type_id, the meta word's arity + variadic bit, and
    /// the code word — no side table at all (the object IS the fast-call table;
    /// see `docs/STAGE_D_MIGRATION.md`'s "Closure object ABI"). On a hit, build a
    /// minimal 4-store context ON THIS STACK and `call_indirect` with the args in
    /// registers (no FFI, no `invoke`, no frame). Fall back to the shim path
    /// (`top.invoke`) for anything ineligible, which is what preserves
    /// composition: the guard requires `direct` (so a wrapped backend never
    /// fast-paths) and a compiled, matching-arity, non-variadic closure (so
    /// continuation / variadic / memory-mode / not-yet-compiled callees, whose
    /// code word stays 0, go through `top`).
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

        // guard1 = value is a ref AND direct calls enabled.
        let (is_ref, addr) = M::emit_ref_addr(self, callee);
        let rc = self.rc_val;
        let ro = MemFlagsData::trusted().with_readonly();
        let direct = self.fb.ins().load(I8, ro, rc, self.off_direct);
        let guard1 = self.fb.ins().band(is_ref, direct);

        let result = self.declare_root_var();
        let checkb = self.fb.create_block();
        // `closb` re-runs the closure checks over EITHER the original callee or
        // an inline-selected multifn clause: params are (callee bits, address).
        let closb = self.fb.create_block();
        self.fb.append_block_param(closb, I64);
        self.fb.append_block_param(closb, I64);
        let mfb = self.fb.create_block();
        let mfload = self.fb.create_block();
        let clprep = self.fb.create_block();
        let fastb = self.fb.create_block();
        let slowb = self.fb.create_block();
        let merge = self.fb.create_block();
        self.fb.ins().brif(guard1, checkb, &[], slowb, &[]);

        // ── a ref: closure? straight to the checks. MultiFn? select the fixed
        // clause for this arity INLINE (one bounds check + one load — Stage G1:
        // value-called multi-arity fns like `+`, `conj`, transducer step fns
        // took the shim per element) and run the same checks on the clause. ──
        self.fb.switch_to_block(checkb);
        self.fb.seal_block(checkb);
        let hdr = self.fb.ins().load(I64, flags, addr, 0);
        let tid = self.fb.ins().band_imm(hdr, 0xffff);
        let is_closure = self.fb.ins().icmp_imm(IntCC::Equal, tid, kind::CLOSURE as i64);
        self.fb.ins().brif(is_closure, closb, &[callee.into(), addr.into()], mfb, &[]);

        self.fb.switch_to_block(mfb);
        self.fb.seal_block(mfb);
        let is_mf = self.fb.ins().icmp_imm(IntCC::Equal, tid, kind::MULTIFN as i64);
        let aux = self.fb.ins().ushr_imm(hdr, 32); // fixed-clause table length
        let in_range = self.fb.ins().icmp_imm(IntCC::UnsignedGreaterThan, aux, n as i64);
        let mf_ok = self.fb.ins().band(is_mf, in_range);
        self.fb.ins().brif(mf_ok, mfload, &[], slowb, &[]);

        // fixed[n] lives in the varlen tail: [hdr | variadic closure | raw8 vmin | clauses…].
        self.fb.switch_to_block(mfload);
        self.fb.seal_block(mfload);
        let clause = self.fb.ins().load(I64, flags, addr, MULTIFN_FIXED_OFF as i32 + (n as i32) * 8);
        let has_clause = self.fb.ins().icmp_imm(IntCC::NotEqual, clause, 0);
        self.fb.ins().brif(has_clause, clprep, &[], slowb, &[]);
        self.fb.switch_to_block(clprep);
        self.fb.seal_block(clprep);
        let caddr = M::emit_ref_addr_unchecked(self, clause);
        self.fb.ins().jump(closb, &[clause.into(), caddr.into()]);

        // ── the closure checks: meta arity matches + !variadic, code present ──
        // (`target`/`taddr` — NOT the original `callee`, which the slow path
        // below still needs: the fast path invokes the CLAUSE when routing
        // happened, and its bits become the callee's stack-mapped self.)
        self.fb.switch_to_block(closb);
        self.fb.seal_block(closb);
        let target = self.fb.block_params(closb)[0];
        let taddr = self.fb.block_params(closb)[1];
        let meta = self.fb.ins().load(I64, flags, taddr, CLOSURE_META_OFF as i32);
        let nparams_raw = self.fb.ins().ushr_imm(meta, 32);
        let nparams = self.fb.ins().band_imm(nparams_raw, 0xffff);
        let arity_ok = self.fb.ins().icmp_imm(IntCC::Equal, nparams, n as i64);
        let variadic_bit = self.fb.ins().band_imm(meta, crate::heap::META_VARIADIC_BIT as i64);
        let not_variadic = self.fb.ins().icmp_imm(IntCC::Equal, variadic_bit, 0);
        let meta_ok = self.fb.ins().band(arity_ok, not_variadic);
        let code = self.fb.ins().load(I64, flags, taddr, CLOSURE_CODE_OFF as i32);
        let has_code = self.fb.ins().icmp_imm(IntCC::NotEqual, code, 0);
        let guard2 = self.fb.ins().band(meta_ok, has_code);
        self.fb.ins().brif(guard2, fastb, &[], slowb, &[]);

        // ── native fast path: 3-store context, args in registers ──
        self.fb.switch_to_block(fastb);
        self.fb.seal_block(fastb);
        let words = self.ctx_size.div_ceil(8) as i32;
        let ctx_ss = self.fb.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (words * 8) as u32,
            3,
        ));
        // Three stores: the shared run-context pointer, the callee's own bits
        // (the callee loads them into its stack-mapped self variable — its
        // self-tail loop AND its capture reads derive from that, Stage E), and
        // a clear tail-pending flag. The callee reads everything else through
        // `rc`.
        self.fb.ins().stack_store(rc, ctx_ss, self.off_rc);
        self.fb.ins().stack_store(target, ctx_ss, self.off_self_closure);
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
        let (saddr, count) = self.spill_args(argvals);
        let sr = self.call_shim(self.refs.call, &[ctx, callee, saddr, count]);
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

    /// Compile one node and DECLARE its value for the stack maps (Stage E):
    /// every expression yields TAGGED VALUE BITS, so anything live across a
    /// safepoint (= any call — a later argument's nested call, a shim, a poll)
    /// is spilled with a map entry, rewritten by a moving collection, and
    /// reloaded. Untagged arithmetic intermediates and derived addresses stay
    /// inside their emitting arm and are never declared — exactly the
    /// tagged-bits-only contract.
    fn compile<M: ModelArithJit>(&mut self, ir: &Ir, tail: bool) -> Value {
        let v = self.compile_node::<M>(ir, tail);
        self.fb.declare_value_needs_stack_map(v);
        v
    }

    fn compile_node<M: ModelArithJit>(&mut self, ir: &Ir, tail: bool) -> Value {
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
            // A captured value: RE-DERIVE the capture base from the running
            // closure's (stack-mapped) bits on every read, then one load off
            // the object (Stage E). A safepoint anywhere reloads `self_var`
            // with the closure's post-move address, so this can never read
            // through a stale derived pointer — the carried caps_base gap is
            // gone. (Loop-hoisting the derivation can return later behind a
            // "no safepoint in loop body" analysis.)
            Ir::Capture(idx) => self.emit_capture::<M>(*idx),
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

                let result = self.declare_root_var();
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
                let v = self.call_shim_fenced(self.refs.load_global, &[ctx, sv]);
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

                let result = self.declare_root_var();
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
                self.call_shim_fenced(self.refs.def_global, &[ctx, namev, v])
            }
            Ir::Lambda { nparams, variadic, nslots, captures, body } => {
                // Register this Lambda's BODY once (compile time) in the runtime's
                // GLOBAL, stable template registry (dedup by `Arc<Ir>` pointer —
                // the SAME id `ObjView::Closure::template` reports and `alloc`
                // uses for `Obj::Closure`), then emit: resolve each capture VALUE
                // here (registers / frame slots / own caps), spill them in order,
                // and let the shim copy + allocate. `nparams`/`variadic`/`nslots`
                // are this Lambda node's own compile-time-constant arity — baked
                // in as immediates, no per-template metadata table needed.
                let tid = if self.rt_ptr.is_null() {
                    0 // dump_ir only: this body is never executed.
                } else {
                    let rt = unsafe { &*(self.rt_ptr as *const Runtime<M>) };
                    rt.register_template(body)
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
                        // Transitive capture: read our OWN capture, deriving
                        // the base from the stack-mapped self bits (Stage E).
                        CapSrc::Cap(j) => self.emit_capture::<M>(*j),
                    })
                    .collect();
                if M::INLINE_OBJECTS {
                    // Inline the whole closure allocation (D5): header, meta,
                    // the creation-time code stamp (one load through the baked
                    // `template_code` slot — 0 until the body compiles, exactly
                    // like the shim), and the capture values, all straight-line
                    // stores. Window closed/exhausted → the shim.
                    let flags = MemFlagsData::trusted();
                    let ncaps = vals.len();
                    let result = self.declare_root_var();
                    let slow = self.fb.create_block();
                    let merge = self.fb.create_block();
                    let size = (CLOSURE_CAPS_OFF + ncaps * 8) as i64;
                    let header = crate::heap::make_header(kind::CLOSURE, 0, ncaps as u32);
                    let addr = self.emit_alloc(size, header, slow);
                    let meta =
                        crate::heap::closure_meta(tid, *nparams as u16, *nslots, *variadic);
                    let metav = self.iconst(meta);
                    self.fb.ins().store(flags, metav, addr, CLOSURE_META_OFF as i32);
                    let slotp = self.template_code_slot::<M>(tid) as u64;
                    let sp = self.iconst(slotp);
                    // NOT readonly: the slot is stamped when the body compiles.
                    let code = self.fb.ins().load(I64, flags, sp, 0);
                    self.fb.ins().store(flags, code, addr, CLOSURE_CODE_OFF as i32);
                    for (i, &v) in vals.iter().enumerate() {
                        self.fb.ins().store(flags, v, addr, (CLOSURE_CAPS_OFF + i * 8) as i32);
                    }
                    let r = M::emit_enc_ref(self, addr);
                    self.fb.def_var(result, r);
                    self.fb.ins().jump(merge, &[]);

                    self.fb.switch_to_block(slow);
                    self.fb.seal_block(slow);
                    let (caddr, _count) = self.spill_args(&vals);
                    let tidv = self.i32const(tid);
                    let npv = self.i32const(*nparams as u32);
                    let varv = self.i32const(*variadic as u32);
                    let nsv = self.i32const(*nslots as u32);
                    let ncapsv = self.i32const(ncaps as u32);
                    let ctx = self.ctx_val;
                    let sv = self.call_shim_fenced(
                        self.refs.make_closure,
                        &[ctx, tidv, npv, varv, nsv, caddr, ncapsv],
                    );
                    self.fb.def_var(result, sv);
                    self.fb.ins().jump(merge, &[]);

                    self.fb.switch_to_block(merge);
                    self.fb.seal_block(merge);
                    self.fb.use_var(result)
                } else {
                    let (addr, _count) = self.spill_args(&vals);
                    let tidv = self.i32const(tid);
                    let npv = self.i32const(*nparams as u32);
                    let varv = self.i32const(*variadic as u32);
                    let nsv = self.i32const(*nslots as u32);
                    let ncapsv = self.i32const(captures.len() as u32);
                    let ctx = self.ctx_val;
                    self.call_shim_fenced(self.refs.make_closure, &[ctx, tidv, npv, varv, nsv, addr, ncapsv])
                }
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
                            let result = self.declare_root_var();
                            let inlb = self.fb.create_block();
                            let slowb = self.fb.create_block();
                            let merge = self.fb.create_block();
                            self.fb.ins().brif(same, inlb, &[], slowb, &[]);

                            self.fb.switch_to_block(inlb);
                            self.fb.seal_block(inlb);
                            let iv = self.emit_inlined_body::<M>(
                                plan.nparams,
                                plan.nslots,
                                &plan.body,
                                &argvals,
                                None,
                                None,
                            );
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
                    // Stage G2 — param-value specialization: a non-tail call
                    // through a PARAMETER or CAPTURE whose actual value this
                    // compile observed (the triggering invoke's args / the
                    // invoked closure's captures) splices the observed clause
                    // body inline behind a template guard on the live value.
                    // This is the per-element `(f acc x)` / `(rf a (inc x))`
                    // shape of every reduce/transducer step.
                    if let Some(vbits) = self.spec_value_of(f) {
                        if let Some(plan) = self.try_value_inline_plan::<M>(vbits, args.len()) {
                            let fval = self.compile::<M>(f, false);
                            let argvals: Vec<Value> =
                                args.iter().map(|a| self.compile::<M>(a, false)).collect();
                            return self.emit_value_specialized::<M>(fval, &argvals, &plan);
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
                            // Compare against the tracked self bits, so a
                            // collection mid-body (which moved this closure —
                            // and rewrote both copies) still recognizes itself.
                            let sc = self.fb.use_var(self.self_var);
                            self.fb.declare_value_needs_stack_map(sc); // not a compile() result
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
                            // BACK-EDGE SAFEPOINT POLL (Stage E): a native
                            // self-loop must come to a safepoint in bounded
                            // time — this is what lets another thread's
                            // collection (or our own allocation pressure)
                            // interrupt a pure-arithmetic loop. Coarsened to
                            // every `BACKEDGE_POLL_INTERVAL` iterations.
                            self.emit_backedge_poll();
                            self.fb.ins().jump(header, &[]);

                            // Non-self: the shim tail-call, whose result flows to
                            // the function return (the trampoline reads
                            // `tail_pending`).
                            self.fb.switch_to_block(notself);
                            self.fb.seal_block(notself);
                            let (addr, count) = self.spill_args(&argvals);
                            self.call_shim_fenced(self.refs.tail_call, &[ctx, callee, addr, count])
                        }
                        _ => {
                            let (addr, count) = self.spill_args(&argvals);
                            self.call_shim_fenced(self.refs.tail_call, &[ctx, callee, addr, count])
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
            // `(type-of x)` — inline the two shapes every dispatch predicate
            // hits per element (Stage G1: `reduced?` is `(%num-eq (type-of x)
            // 'Reduced)` inside every reduce loop): an immediate fixnum is
            // 'Long (a compile-time constant sym), a RECORD ref's type sym is
            // one load off the object. Everything else (bignum refs, builtin
            // containers, bools, nil, …) keeps the runtime's `type_tag`.
            Ir::Prim(Prim::TypeOf, args) if M::INLINE_OBJECTS && !self.rt_ptr.is_null() => {
                let v = self.compile::<M>(&args[0], false);
                let long_bits = {
                    let rt = unsafe { &*(self.rt_ptr as *const Runtime<M>) };
                    M::R::enc_sym(rt.intern("Long"))
                };
                let flags = MemFlagsData::trusted();
                let result = self.declare_root_var();
                let intb = self.fb.create_block();
                let refchk = self.fb.create_block();
                let hdrb = self.fb.create_block();
                let recb = self.fb.create_block();
                let slowb = self.fb.create_block();
                let merge = self.fb.create_block();
                // immediate fixnum → 'Long (exactly `type_tag`'s Val::Int arm;
                // promoted bignums are refs and take the slow path).
                let is_int = M::emit_both_int(self, v, v);
                self.fb.ins().brif(is_int, intb, &[], refchk, &[]);
                self.fb.switch_to_block(intb);
                self.fb.seal_block(intb);
                let lc = self.iconst(long_bits);
                self.fb.def_var(result, lc);
                self.fb.ins().jump(merge, &[]);
                // ref → RECORD? its type sym is the raw word right after the
                // header (RECORD layout: [hdr | raw8 sym | fields…]).
                self.fb.switch_to_block(refchk);
                self.fb.seal_block(refchk);
                let (is_ref, addr) = M::emit_ref_addr(self, v);
                self.fb.ins().brif(is_ref, hdrb, &[], slowb, &[]);
                self.fb.switch_to_block(hdrb);
                self.fb.seal_block(hdrb);
                let hdr = self.fb.ins().load(I64, flags, addr, 0);
                let tid = self.fb.ins().band_imm(hdr, 0xffff);
                let is_rec = self.fb.ins().icmp_imm(IntCC::Equal, tid, kind::RECORD as i64);
                self.fb.ins().brif(is_rec, recb, &[], slowb, &[]);
                self.fb.switch_to_block(recb);
                self.fb.seal_block(recb);
                let sym_w = self.fb.ins().load(I64, flags, addr, HEADER_SIZE as i32);
                let enc = M::emit_enc_sym(self, sym_w);
                self.fb.def_var(result, enc);
                self.fb.ins().jump(merge, &[]);
                self.fb.switch_to_block(slowb);
                self.fb.seal_block(slowb);
                let sp = self.slow_prim(Prim::TypeOf, &[v]);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);
                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // `(%cons h t)` — inline allocation (D5): normalize a `()` tail to
            // nil (exactly `Runtime::cons`), bump-allocate 24 bytes through the
            // AllocWindow, store head/tail, encode. Window closed/exhausted →
            // the prim shim (which also owns the loud-exhaustion panic).
            Ir::Prim(Prim::Cons, args) if M::INLINE_OBJECTS => {
                let head = self.compile::<M>(&args[0], false);
                let tail = self.compile::<M>(&args[1], false);
                let flags = MemFlagsData::trusted();
                // tail2 = tail, unless tail is the `()` object → nil.
                let tailv = self.declare_root_var();
                self.fb.def_var(tailv, tail);
                let (t_is_ref, t_addr) = M::emit_ref_addr(self, tail);
                let chk = self.fb.create_block();
                let cont = self.fb.create_block();
                self.fb.ins().brif(t_is_ref, chk, &[], cont, &[]);
                self.fb.switch_to_block(chk);
                self.fb.seal_block(chk);
                let thdr = self.fb.ins().load(I64, flags, t_addr, 0);
                let ttid = self.fb.ins().band_imm(thdr, 0xffff);
                let is_empty =
                    self.fb.ins().icmp_imm(IntCC::Equal, ttid, kind::EMPTY_LIST as i64);
                let nil = self.iconst(M::R::enc_nil());
                let norm = self.fb.ins().select(is_empty, nil, tail);
                self.fb.def_var(tailv, norm);
                self.fb.ins().jump(cont, &[]);
                self.fb.switch_to_block(cont);
                self.fb.seal_block(cont);
                let tail2 = self.fb.use_var(tailv);

                let result = self.declare_root_var();
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let header = crate::heap::make_header(kind::CONS, 0, 0);
                let addr = self.emit_alloc(24, header, slow);
                self.fb.ins().store(flags, head, addr, 8);
                self.fb.ins().store(flags, tail2, addr, 16);
                let r = M::emit_enc_ref(self, addr);
                self.fb.def_var(result, r);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(Prim::Cons, &[head, tail]);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // `(%first x)` / `(%rest x)` — for a genuine CONS, one type guard +
            // one field load (D5). Anything else (chunked seqs, nil, `()`, the
            // use-after-move panic) keeps the shim's exact semantics.
            Ir::Prim(p @ (Prim::First | Prim::Rest), args) if M::INLINE_OBJECTS => {
                let v = self.compile::<M>(&args[0], false);
                let result = self.declare_root_var();
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let (addr, _hdr) = self.emit_typed_addr::<M>(v, kind::CONS, slow);
                let off = if matches!(p, Prim::First) { 8 } else { 16 };
                let r = self.fb.ins().load(I64, MemFlagsData::trusted(), addr, off);
                self.fb.def_var(result, r);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(*p, &[v]);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // `(field r i)` — a RECORD guard, an immediate-int index guard, an
            // unsigned bounds check against the record's field count (its header
            // aux — a negative index untags huge and fails it), then one indexed
            // load off the varlen tail. Stage H: every deftype method body opens
            // with `(let [f0 (field this 0) …] …)`, so this prim ran through the
            // generic shim several times per PROTOCOL CALL — the top frame of
            // both the vecbuild and group-by profiles. Any failed guard takes the
            // shim, which owns the loud non-record/bad-index panic.
            Ir::Prim(Prim::Field, args) if M::INLINE_OBJECTS => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let flags = MemFlagsData::trusted();
                let result = self.declare_root_var();
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let (addr, hdr) = self.emit_typed_addr::<M>(argvals[0], kind::RECORD, slow);
                let is_int = M::emit_both_int(self, argvals[1], argvals[1]);
                let okb = self.fb.create_block();
                self.fb.ins().brif(is_int, okb, &[], slow, &[]);
                self.fb.switch_to_block(okb);
                self.fb.seal_block(okb);
                let i = M::emit_untag(self, argvals[1]);
                // RECORD's varlen count IS its field count (header aux; bit 63
                // is the forwarding bit and is clear on any live object).
                let n = self.fb.ins().ushr_imm(hdr, 32);
                let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, i, n);
                let okb2 = self.fb.create_block();
                self.fb.ins().brif(inb, okb2, &[], slow, &[]);
                self.fb.switch_to_block(okb2);
                self.fb.seal_block(okb2);
                // RECORD layout = [hdr | raw8 type sym | fields…]: the varlen
                // tail starts at `RECORD_FIELDS_OFF`.
                let byteoff = self.fb.ins().ishl_imm(i, 3);
                let slot = self.fb.ins().iadd(addr, byteoff);
                let r = self.fb.ins().load(I64, flags, slot, RECORD_FIELDS_OFF as i32);
                self.fb.def_var(result, r);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(Prim::Field, &argvals);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // `(%alength v)` — the logical length rides the ARRAY handle's
            // header aux: one type guard + a shift + retag (D5).
            Ir::Prim(Prim::VectorLen, args) if M::INLINE_OBJECTS => {
                let v = self.compile::<M>(&args[0], false);
                let result = self.declare_root_var();
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let (_addr, hdr) = self.emit_typed_addr::<M>(v, kind::ARRAY, slow);
                let len = self.fb.ins().ushr_imm(hdr, 32);
                let r = M::emit_tag(self, len);
                self.fb.def_var(result, r);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(Prim::VectorLen, &[v]);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // `(%aget v i)` — ARRAY handle guard, immediate-int index guard,
            // unsigned bounds check against the handle's logical length (a
            // negative index untags huge and fails it), then a direct indexed
            // load through the data blob (D5). Any failed guard — including
            // out-of-bounds — takes the shim, which owns the loud range panic.
            //
            // `%aset` USED TO SHARE THIS ARM and is deliberately not here: its
            // inline store puts a (possibly young) value into a (possibly
            // promoted) data blob, and emitted code cannot yet mark the card —
            // that is I3's `emit_card_mark` plus the RunCtx mirrors. An
            // unbarriered store is not a slow `%aset`, it is a lost old→young
            // edge, so until I3 lands `%aset` goes through `slow_prim` and the
            // barriered `arr_slice_mut` choke point. The guards below are I3's
            // starting point: re-add `Prim::VectorSet` here and emit the mark
            // on `blob_addr` right before the store.
            Ir::Prim(p @ Prim::VectorRef, args) if M::INLINE_OBJECTS => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let flags = MemFlagsData::trusted();
                let result = self.declare_root_var();
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();
                let (addr, hdr) = self.emit_typed_addr::<M>(argvals[0], kind::ARRAY, slow);
                let is_int = M::emit_both_int(self, argvals[1], argvals[1]);
                let okb = self.fb.create_block();
                self.fb.ins().brif(is_int, okb, &[], slow, &[]);
                self.fb.switch_to_block(okb);
                self.fb.seal_block(okb);
                let i = M::emit_untag(self, argvals[1]);
                let len = self.fb.ins().ushr_imm(hdr, 32);
                let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, i, len);
                let okb2 = self.fb.create_block();
                self.fb.ins().brif(inb, okb2, &[], slow, &[]);
                self.fb.switch_to_block(okb2);
                self.fb.seal_block(okb2);
                // handle field 0 = the encoded DATA blob ref; its varlen tail
                // starts right after the blob header (ARRAY_DATA has no fields).
                let blob_bits = self.fb.ins().load(I64, flags, addr, 8);
                let (_ir, blob_addr) = M::emit_ref_addr(self, blob_bits);
                let byteoff = self.fb.ins().ishl_imm(i, 3);
                let slot = self.fb.ins().iadd(blob_addr, byteoff);
                let r = self.fb.ins().load(I64, flags, slot, 8);
                self.fb.def_var(result, r);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let sp = self.slow_prim(*p, &argvals);
                self.fb.def_var(result, sp);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                self.fb.use_var(result)
            }
            // Stage F2: the hot COLLECTION prims get monomorphic register-arg
            // shims — no arg spill, no prim tag, no giant match. PARKING-
            // classified (plain call): their live-across values ride the maps.
            // (Cons/First/Rest here serve the models without inline object
            // paths and any site the inline guards reject — the arms above
            // take precedence under LowBit.)
            Ir::Prim(p @ (Prim::PvConj | Prim::PvNth | Prim::ArrPush | Prim::Cons | Prim::TvConj), args) => {
                let a = self.compile::<M>(&args[0], false);
                let b = self.compile::<M>(&args[1], false);
                let f = match p {
                    Prim::PvConj => self.refs.pv_conj,
                    Prim::PvNth => self.refs.pv_nth,
                    Prim::ArrPush => self.refs.arr_push,
                    Prim::Cons => self.refs.cons2,
                    Prim::TvConj => self.refs.tv_conj,
                    _ => unreachable!(),
                };
                let ctx = self.ctx_val;
                self.call_shim_checked(f, &[ctx, a, b])
            }
            Ir::Prim(
                p @ (Prim::PvAssoc | Prim::HamtAssoc | Prim::HamtLookup | Prim::TamAssoc
                | Prim::ThmAssoc),
                args,
            ) => {
                let a = self.compile::<M>(&args[0], false);
                let b = self.compile::<M>(&args[1], false);
                let c = self.compile::<M>(&args[2], false);
                let f = match p {
                    Prim::PvAssoc => self.refs.pv_assoc,
                    Prim::HamtAssoc => self.refs.hamt_assoc,
                    Prim::HamtLookup => self.refs.hamt_lookup,
                    Prim::TamAssoc => self.refs.tam_assoc,
                    Prim::ThmAssoc => self.refs.thm_assoc,
                    _ => unreachable!(),
                };
                let ctx = self.ctx_val;
                self.call_shim_checked(f, &[ctx, a, b, c])
            }
            Ir::Prim(p @ (Prim::First | Prim::Rest), args) => {
                let v = self.compile::<M>(&args[0], false);
                let f = if matches!(p, Prim::First) { self.refs.first1 } else { self.refs.rest1 };
                let ctx = self.ctx_val;
                self.call_shim_checked(f, &[ctx, v])
            }
            // Every other prim: compute args, escape to the runtime (the native
            // analogue of the bytecode tier's `Slow`).
            Ir::Prim(p, args) => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let (addr, count) = self.spill_args(&argvals);
                let tagv = self.i32const(prim_tag(*p));
                let ctx = self.ctx_val;
                self.call_shim_fenced_checked(self.refs.prim, &[ctx, tagv, addr, count])
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
                self.call_shim_fenced(self.refs.set_global, &[ctx, namev, v])
            }
            // `(.-field obj)` — inline-cached field read via the shim.
            Ir::FieldGet { site, field, obj } => {
                let o = self.compile::<M>(obj, false);
                let sitev = self.i32const(*site as u32);
                let fieldv = self.i32const(*field);
                let ctx = self.ctx_val;
                self.call_shim_fenced(self.refs.field_get, &[ctx, sitev, fieldv, o])
            }
            // Protocol/method dispatch. D5 fast path: when unwrapped (`direct`)
            // and the receiver is a RECORD, read its type sym straight off the
            // object and probe this site's baked 2-way inline cache (filled by
            // the shim; epoch = relocation count ⊕ dispatch version, so a moved
            // impl or a redefinition never false-hits). A hit calls the impl
            // through the ordinary native call sequence; every miss — non-record
            // receivers (built-in categories), wrapped backends, cold/invalidated
            // sites — takes the shim, which resolves AND refills.
            Ir::Dispatch { site, method, args } => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                if !M::INLINE_OBJECTS || argvals.is_empty() {
                    let (addr, count) = self.spill_args(&argvals);
                    let sitev = self.i32const(*site as u32);
                    let methodv = self.i32const(*method);
                    let ctx = self.ctx_val;
                    return self
                        .call_shim_checked(self.refs.dispatch, &[ctx, sitev, methodv, addr, count]);
                }
                let flags = MemFlagsData::trusted();
                let ro = MemFlagsData::trusted().with_readonly();
                let result = self.declare_root_var();
                let slow = self.fb.create_block();
                let merge = self.fb.create_block();

                let recv = argvals[0];
                let (is_ref, addr) = M::emit_ref_addr(self, recv);
                let rc = self.rc_val;
                let direct = self.fb.ins().load(I8, ro, rc, self.off_direct);
                let g1 = self.fb.ins().band(is_ref, direct);
                let chk = self.fb.create_block();
                self.fb.ins().brif(g1, chk, &[], slow, &[]);
                self.fb.switch_to_block(chk);
                self.fb.seal_block(chk);
                let hdr = self.fb.ins().load(I64, flags, addr, 0);
                let tid = self.fb.ins().band_imm(hdr, 0xffff);
                let is_rec = self.fb.ins().icmp_imm(IntCC::Equal, tid, kind::RECORD as i64);
                let chk2 = self.fb.create_block();
                self.fb.ins().brif(is_rec, chk2, &[], slow, &[]);
                self.fb.switch_to_block(chk2);
                self.fb.seal_block(chk2);
                // The record's type sym (its raw word) IS `type_tag(recv)`.
                let ty = self.fb.ins().load(I64, flags, addr, 8);
                let rp = self.load_rc_field(self.off_reloc_ptr);
                let reloc = self.fb.ins().load(I64, flags, rp, 0);
                let vp = self.load_rc_field(self.off_dispver_ptr);
                let ver = self.fb.ins().load(I64, flags, vp, 0);
                let mixed = self.fb.ins().imul_imm(reloc, DISPATCH_EPOCH_MIX as i64);
                let epoch = self.fb.ins().bxor(mixed, ver);
                let sitep = self.iconst(self.dispatch_ic_slot::<M>(*site) as u64);
                let probe = |c: &mut Self, way: i32, miss: cranelift_codegen::ir::Block| {
                    let base = way * 24;
                    let e = c.fb.ins().load(I64, flags, sitep, base);
                    let t = c.fb.ins().load(I64, flags, sitep, base + 8);
                    let imp = c.fb.ins().load(I64, flags, sitep, base + 16);
                    let e_ok = c.fb.ins().icmp(IntCC::Equal, e, epoch);
                    let t_ok = c.fb.ins().icmp(IntCC::Equal, t, ty);
                    let hit = c.fb.ins().band(e_ok, t_ok);
                    let hitb = c.fb.create_block();
                    c.fb.ins().brif(hit, hitb, &[], miss, &[]);
                    c.fb.switch_to_block(hitb);
                    c.fb.seal_block(hitb);
                    (imp, hitb)
                };
                let way1b = self.fb.create_block();
                let (imp0, _b0) = probe(self, 0, way1b);
                let r0 = self.emit_call::<M>(imp0, &argvals);
                self.fb.def_var(result, r0);
                self.fb.ins().jump(merge, &[]);
                self.fb.switch_to_block(way1b);
                self.fb.seal_block(way1b);
                let (imp1, _b1) = probe(self, 1, slow);
                let r1 = self.emit_call::<M>(imp1, &argvals);
                self.fb.def_var(result, r1);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(slow);
                self.fb.seal_block(slow);
                let (aaddr, count) = self.spill_args(&argvals);
                let sitev = self.i32const(*site as u32);
                let methodv = self.i32const(*method);
                let ctx = self.ctx_val;
                let sr = self.call_shim(self.refs.dispatch, &[ctx, sitev, methodv, aaddr, count]);
                self.fb.def_var(result, sr);
                self.fb.ins().jump(merge, &[]);

                self.fb.switch_to_block(merge);
                self.fb.seal_block(merge);
                // Either path may have raised (a hit's impl, a miss's resolve
                // failure); propagate exactly as the shim-only emission did.
                let r = self.fb.use_var(result);
                self.emit_pending_check(r)
            }
            // Register a deftype/protocol method impl.
            Ir::DefMethod { name, ty, imp } => {
                let impv = self.compile::<M>(imp, false);
                let namev = self.i32const(*name);
                let tyv = self.i32const(*ty);
                let ctx = self.ctx_val;
                self.call_shim_fenced(self.refs.def_method, &[ctx, namev, tyv, impv])
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
        self.call_shim_fenced_checked(self.refs.prim, &[ctx, tagv, addr, count])
    }

    /// Emit guarded arithmetic: the per-model fixnum fast path when both operands
    /// are immediate integers and the result fits fixnum range, else a call to the
    /// runtime's promoting `prim` (bignum / float / mixed). The emit half of the
    /// numeric-overflow axis — what makes the JIT match the tree-walker's tower.
    fn emit_guarded_arith<M: ModelArithJit>(&mut self, op: Prim, a: Value, b: Value) -> Value {
        // `=`'s guard is wider than the arithmetic ops': ANY two non-ref
        // immediates compare by bits (Stage G1 — `(%num-eq (type-of x) 'Reduced)`
        // runs per element in every reduce, and syms took the shim).
        let both = if let Prim::Eq = op {
            M::emit_eq_immediates(self, a, b)
        } else {
            M::emit_both_int(self, a, b)
        };

        let result = self.declare_root_var();
        let fast = self.fb.create_block();
        let slow = self.fb.create_block();
        let merge = self.fb.create_block();
        self.fb.ins().brif(both, fast, &[], slow, &[]);

        // ── fast path: both operands are immediate fixnums (Eq: any immediates) ──
        self.fb.switch_to_block(fast);
        self.fb.seal_block(fast);
        match op {
            Prim::Lt | Prim::Eq => {
                // `<` / `=` don't overflow. `=` on two immediates is bit-equality
                // of the WORDS (canonical encodings; the slow `equal?` path is
                // only needed for refs, which the guard already routed away).
                let c = if let Prim::Eq = op {
                    self.fb.ins().icmp(IntCC::Equal, a, b)
                } else {
                    let x = M::emit_untag(self, a);
                    let y = M::emit_untag(self, b);
                    self.fb.ins().icmp(IntCC::SignedLessThan, x, y)
                };
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
                let x = M::emit_untag(self, a);
                let y = M::emit_untag(self, b);
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
                let x = M::emit_untag(self, a);
                let y = M::emit_untag(self, b);
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
                let result = self.declare_root_var();
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
                let result = self.declare_root_var();
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
        let native = if M::R::tag_of(callee) == RawTag::Ref {
            match rt.view(callee) {
                ObjView::Closure { template, .. } => jit_can_compile(rt.template(template)),
                // Route a multi-arity fn on the clause this call selects.
                ObjView::MultiFn { .. } => match rt.multifn_select(callee, args.len()) {
                    Some(sel) if !rt.pending() => {
                        if M::R::tag_of(sel) == RawTag::Ref {
                            match rt.view(sel) {
                                ObjView::Closure { template, .. } => jit_can_compile(rt.template(template)),
                                _ => true,
                            }
                        } else {
                            true
                        }
                    }
                    _ => true,
                },
                _ => true, // non-closure callables (escape conts) are the JIT's error path
            }
        } else {
            true
        };
        if native {
            self.jit.invoke(top, rt, callee, args)
        } else {
            self.cek.invoke(top, rt, callee, args)
        }
    }
}
