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
use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::Runtime;
use crate::value::{frame_get, frame_set, Frame, Locals, Obj, Sym, Val};

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
    /// call to THIS same closure becomes an in-place native loop (refill the frame,
    /// branch back) — O(1) stack, no FFI. A tail call to anything else falls back
    /// to the shim + trampoline, so mutual tail recursion keeps full TCO.
    self_closure: Cell<u64>,
    /// The INNERMOST active lexical frame. Starts as the call frame; a `let`
    /// pushes a fresh child frame here and restores it on exit, so `up`/`idx`
    /// resolution (and closure capture) always walks from the current scope —
    /// the JIT analogue of how the tree-walker threads `locals` into `let`.
    cur: RefCell<Locals>,
    /// Frames saved by enclosing `let`s, restored on `let_exit` (a scope stack).
    saved: RefCell<Vec<Locals>>,
    templates: *const Vec<ClosureTemplate>,
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

/// A closure this body can construct: the arity + shared `Ir` body, captured by
/// index so the emitted code passes only a small integer.
struct ClosureTemplate {
    nparams: usize,
    variadic: bool,
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

/// Enter a `let`: push a fresh child frame with `nslots` nil slots and make it
/// the current scope. Its inits fill the slots sequentially via `shim_let_set`.
extern "C" fn shim_let_enter<M: ValueModel>(ctx: *mut JitCtx<M>, nslots: u32) -> u64 {
    let ctx = unsafe { &*ctx };
    let parent = ctx.cur.borrow().clone();
    let nil = M::R::enc_nil();
    let slots: Vec<AtomicU64> =
        (0..nslots).map(|_| AtomicU64::new(nil)).collect();
    let frame = Some(Arc::new(crate::value::Frame { slots, parent }));
    ctx.cur_slots.set(slots_ptr(&frame));
    ctx.saved.borrow_mut().push(ctx.cur.replace(frame));
    0
}

/// Set slot `idx` of the current `let` frame during its sequential init.
extern "C" fn shim_let_set<M: ValueModel>(ctx: *mut JitCtx<M>, idx: u32, val: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    ctx.cur.borrow().as_ref().unwrap().slots[idx as usize].store(val, std::sync::atomic::Ordering::Relaxed);
    val
}

/// Leave a `let`: restore the enclosing scope.
extern "C" fn shim_let_exit<M: ValueModel>(ctx: *mut JitCtx<M>, _unused: u32) -> u64 {
    let ctx = unsafe { &*ctx };
    let restored = ctx.saved.borrow_mut().pop().expect("let_exit without enter");
    ctx.cur_slots.set(slots_ptr(&restored));
    *ctx.cur.borrow_mut() = restored;
    0
}

extern "C" fn shim_load_global<M: ValueModel>(ctx: *mut JitCtx<M>, sym: u32) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &*(*ctx.rc).rt };
    match rt.global(sym as Sym) {
        Some(v) => v,
        None => panic!("Unable to resolve symbol: {}", rt.sym_name(sym as Sym)),
    }
}

extern "C" fn shim_def_global<M: ValueModel>(ctx: *mut JitCtx<M>, sym: u32, val: u64) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    rt.define_global(sym as Sym, val);
    // Match the tree-walker: `def` evaluates to the (encoded) defined symbol.
    rt.encode(Val::Sym(sym as Sym))
}

extern "C" fn shim_make_closure<M: ValueModel>(ctx: *mut JitCtx<M>, template_id: u32) -> u64 {
    let ctx = unsafe { &*ctx };
    let rt = unsafe { &mut *(*ctx.rc).rt };
    let templates = unsafe { &*ctx.templates };
    let t = &templates[template_id as usize];
    let env = ctx.cur.borrow().clone();
    let id = rt.alloc(Obj::Closure {
        nparams: t.nparams,
        variadic: t.variadic,
        body: t.body.clone(),
        env,
    });
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

// A stable integer tag per prim so the emitted code can name one. (The `Prim`
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
    let_enter: FuncId,
    let_set: FuncId,
    let_exit: FuncId,
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
    let_enter: cranelift_codegen::ir::FuncRef,
    let_set: cranelift_codegen::ir::FuncRef,
    let_exit: cranelift_codegen::ir::FuncRef,
}

/// A finished, runnable body: the host code pointer plus the closure templates
/// its `make_closure` sites index into.
struct Compiled {
    code: *const u8,
    templates: Vec<ClosureTemplate>,
    /// Does this body ever consult `JitCtx::cur` (the `Arc<Frame>` chain)? Only
    /// bodies with `let`, a closure (`Lambda`), or an `up > 0` local do. When
    /// false (the common leaf/loop shape — all locals `up == 0`), `run_once`
    /// skips cloning the frame into `cur`, saving an `Arc` bump per call.
    needs_cur: bool,
    /// Is this body callable via the NATIVE fast path (`call_indirect` with a
    /// caller-built stack frame)? True iff it does not need the heap frame chain
    /// (`!needs_cur`, so a stack frame is safe and it makes no closures) AND
    /// contains no tail call (so it never leaves a `tail_pending` signal for a
    /// caller that will not trampoline it). Callers still check arity + variadic
    /// at the (runtime) fill; everything ineligible keeps `code == null`.
    body_fast_ok: bool,
}

/// Is there a `Call`/`Dispatch` in TAIL position of `ir`? Such a body may set
/// `tail_pending` and return, which only the `invoke` trampoline handles — so a
/// body with one is not eligible for the caller-built-frame fast path (that path
/// uses a plain `call_indirect` and would drop the tail signal).
fn has_tail_call(ir: &Ir) -> bool {
    match ir {
        Ir::Call(..) | Ir::Dispatch { .. } => true,
        Ir::If(_, t, e) => has_tail_call(t) || has_tail_call(e),
        Ir::Do(xs) => xs.last().is_some_and(has_tail_call),
        Ir::Let(_, body) => has_tail_call(body),
        _ => false,
    }
}

/// Does compiling `ir` require the live `cur` frame chain at run time? True iff it
/// contains a `let`, a closure that captures the frame, or an `up > 0` local
/// reference/assignment. (Does not descend into `Lambda` bodies — those compile
/// separately.) `up == 0` locals go through the raw `cur_slots` pointer, not `cur`.
fn body_needs_cur(ir: &Ir) -> bool {
    match ir {
        Ir::Lambda { .. } | Ir::Let(..) => true,
        Ir::Local { up, .. } => *up > 0,
        Ir::SetLocal { up, val, .. } => *up > 0 || body_needs_cur(val),
        Ir::Const(_) | Ir::Quote(_) | Ir::Global(_) => false,
        Ir::If(a, b, c) => body_needs_cur(a) || body_needs_cur(b) || body_needs_cur(c),
        Ir::Do(xs) | Ir::Prim(_, xs) => xs.iter().any(body_needs_cur),
        Ir::Call(f, args) => body_needs_cur(f) || args.iter().any(body_needs_cur),
        Ir::Def { init, .. } | Ir::SetGlobal { val: init, .. } => body_needs_cur(init),
        // Not compiled by this tier (rejected earlier); be conservative.
        Ir::DefMethod { .. } | Ir::Dispatch { .. } | Ir::FieldGet { .. } | Ir::Try { .. } => true,
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
    /// A monomorphic inline cache for the LAST callee resolved. Call sites in the
    /// wild are overwhelmingly monomorphic (a given `(f …)` calls the same `f`
    /// every time), so caching the resolution — decode + heap lookup + compiled
    /// body + captured env, all keyed on the callee's bits — turns the steady
    /// state into a single compare + a couple of `Arc` bumps, skipping the hash
    /// lookups entirely. This is the dispatch axis's "resolve once, not per call".
    call_ic: RefCell<Option<CallTarget>>,
    /// Dense, heap-id-indexed table of NATIVE fast-call entries: for a closure
    /// whose body is directly callable (see `FastTarget`), the call site jumps to
    /// its compiled code with `call_indirect`, building the callee frame + context
    /// inline in emitted code — no FFI, no `invoke`. Sized once per top-level form
    /// (to `heap.len()`), so its base is stable for the whole run; filled lazily on
    /// the slow path. `code == null` (the default) means "not a fast target, use
    /// the shim path", which is what keeps continuations / wrappers composable.
    fast_targets: RefCell<Vec<FastTarget>>,
    _pd: std::marker::PhantomData<fn() -> M>,
}

/// One entry of the fast-call table. `code` is the compiled body's native entry
/// (the usual `fn(*mut JitCtx) -> u64`); the call site builds a `JitCtx` on its
/// own stack and jumps to it. Only bodies that are non-escaping, non-variadic, and
/// contain no tail call are eligible (so a stack frame is safe and there is no
/// lost tail-call signal); everything else keeps `code == null` and goes through
/// the shim path.
#[repr(C)]
#[derive(Clone, Copy)]
struct FastTarget {
    code: *const u8,
    nparams: u32,
}

impl Default for FastTarget {
    fn default() -> Self {
        FastTarget { code: std::ptr::null(), nparams: 0 }
    }
}

/// A resolved call target (the payload of the monomorphic inline cache).
struct CallTarget {
    callee: u64,
    nparams: usize,
    variadic: bool,
    compiled: Arc<Compiled>,
    env: Locals,
}

/// Cap on pooled frames — plenty for realistic recursion depth, bounded memory.
const FRAME_POOL_CAP: usize = 1024;

impl<M: ModelArithJit> JitCranelift<M> {
    pub fn new() -> Self {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").unwrap();
        flags.set("is_pic", "false").unwrap();
        // Speed of compilation matters more than of the code for a first tier.
        // Bodies compile once and run many times, so optimize the emitted code
        // (register allocation, redundancy elimination) — worth the compile cost.
        flags.set("opt_level", "speed").unwrap();
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
        let mk: extern "C" fn(*mut JitCtx<M>, u32) -> u64 = shim_make_closure::<M>;
        let ca: extern "C" fn(*mut JitCtx<M>, u64, *const u64, u32) -> u64 = shim_call::<M>;
        let tc: extern "C" fn(*mut JitCtx<M>, u64, *const u64, u32) -> u64 = shim_tail_call::<M>;
        let ft: extern "C" fn(*mut JitCtx<M>) -> u64 = shim_finish_tail::<M>;
        let pr: extern "C" fn(*mut JitCtx<M>, u32, *const u64, u32) -> u64 = shim_prim::<M>;
        let sl: extern "C" fn(*mut JitCtx<M>, u32, u32, u64) -> u64 = shim_set_local::<M>;
        let sg: extern "C" fn(*mut JitCtx<M>, u32, u64) -> u64 = shim_set_global::<M>;
        let le: extern "C" fn(*mut JitCtx<M>, u32) -> u64 = shim_let_enter::<M>;
        let ls: extern "C" fn(*mut JitCtx<M>, u32, u64) -> u64 = shim_let_set::<M>;
        let lx: extern "C" fn(*mut JitCtx<M>, u32) -> u64 = shim_let_exit::<M>;
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
        builder.symbol("ml_let_enter", le as *const u8);
        builder.symbol("ml_let_set", ls as *const u8);
        builder.symbol("ml_let_exit", lx as *const u8);

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
        let s_make_closure = sig(&[ptr, cranelift_codegen::ir::types::I32]);
        let s_call = sig(&[ptr, I64, ptr, cranelift_codegen::ir::types::I32]);
        let s_prim = sig(&[ptr, cranelift_codegen::ir::types::I32, ptr, cranelift_codegen::ir::types::I32]);
        let s_finish_tail = sig(&[ptr]);
        let s_set_local = sig(&[ptr, cranelift_codegen::ir::types::I32, cranelift_codegen::ir::types::I32, I64]);
        let s_set_global = sig(&[ptr, cranelift_codegen::ir::types::I32, I64]);
        let s_let_enter = sig(&[ptr, cranelift_codegen::ir::types::I32]);
        let s_let_set = sig(&[ptr, cranelift_codegen::ir::types::I32, I64]);
        let s_let_exit = sig(&[ptr, cranelift_codegen::ir::types::I32]);

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
            let_enter: decl(&mut module, "ml_let_enter", &s_let_enter),
            let_set: decl(&mut module, "ml_let_set", &s_let_set),
            let_exit: decl(&mut module, "ml_let_exit", &s_let_exit),
        };

        JitCranelift {
            module: RefCell::new(module),
            fbctx: RefCell::new(FunctionBuilderContext::new()),
            shims,
            cache: RefCell::new(HashMap::default()),
            frame_pool: RefCell::new(Vec::new()),
            call_ic: RefCell::new(None),
            fast_targets: RefCell::new(Vec::new()),
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
            let mut templates = Vec::new();
            build_body::<M>(&mut module, &mut fb, self.shims, ir, &mut templates, true, 0);
            fb.finalize();
        }
        let out = ctx.func.display().to_string();
        module.clear_context(&mut ctx);
        out
    }

    fn compile(&self, ir: &Ir, tail_root: bool, nparams: usize) -> Arc<Compiled> {
        let mut module = self.module.borrow_mut();
        let mut fbctx = self.fbctx.borrow_mut();
        let mut ctx = module.make_context();
        let ptr = module.target_config().pointer_type();
        ctx.func.signature.params.push(AbiParam::new(ptr));
        ctx.func.signature.returns.push(AbiParam::new(I64));

        let mut templates = Vec::new();
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            build_body::<M>(&mut module, &mut fb, self.shims, ir, &mut templates, tail_root, nparams);
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
        let needs_cur = body_needs_cur(ir);
        Arc::new(Compiled {
            code,
            templates,
            needs_cur,
            // Non-escaping bodies are fast-callable even when they contain tail
            // calls: a self-tail becomes an in-place loop, and a non-self tail sets
            // `tail_pending`, which the native call site finishes correctly (see
            // `emit_call`). So a stack frame is always safe and no signal is lost.
            body_fast_ok: !needs_cur,
        })
    }

    fn compiled_body(&self, body: &Arc<Ir>, nparams: usize) -> Arc<Compiled> {
        let key = Arc::as_ptr(body);
        if let Some(c) = self.cache.borrow().get(&key) {
            return c.clone();
        }
        let c = self.compile(body, true, nparams);
        self.cache.borrow_mut().insert(key, c.clone());
        c
    }

    /// Resolve a callee to `(nparams, variadic, compiled, env)`, using the
    /// monomorphic inline cache when the callee's bits match the last call — the
    /// common case — and otherwise decoding the closure, compiling its body, and
    /// refilling the cache.
    fn resolve_call(&self, rt: &mut Runtime<M>, callee: u64) -> (usize, bool, Arc<Compiled>, Locals) {
        if let Some(t) = self.call_ic.borrow().as_ref() {
            if t.callee == callee {
                return (t.nparams, t.variadic, t.compiled.clone(), t.env.clone());
            }
        }
        let Val::Ref(id) = rt.decode(callee) else {
            panic!("value not callable: {}", rt.print(callee));
        };
        let (nparams, variadic, body, env) = match &rt.heap()[id as usize] {
            Obj::Closure { nparams, variadic, body, env } => {
                (*nparams, *variadic, body.clone(), env.clone())
            }
            _ => panic!("value not callable: {}", rt.print(callee)),
        };
        let compiled = self.compiled_body(&body, nparams);
        *self.call_ic.borrow_mut() = Some(CallTarget {
            callee,
            nparams,
            variadic,
            compiled: compiled.clone(),
            env: env.clone(),
        });
        (nparams, variadic, compiled, env)
    }

    /// Run one compiled body once. Returns `Ok(value)` for a normal return, or
    /// `Err((callee, args))` when the body ended in a tail call the trampoline
    /// should bounce to (reusing this native frame instead of recursing).
    /// Run one body. `Ok(value)` on a normal return; `Err(callee)` on a tail call,
    /// with the tail args left in `args_buf` for the trampoline to consume.
    fn run_once(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        compiled: &Compiled,
        frame: &Locals,
        args_buf: &mut Vec<u64>,
        self_closure: u64,
    ) -> Result<u64, u64> {
        let consts_base = rt.consts_ptr();
        let global_base = rt.global_slots_ptr();
        let global_len = rt.global_slots_len();
        let (fast_base, fast_len) = {
            let t = self.fast_targets.borrow();
            (t.as_ptr(), t.len())
        };
        // Native fast calls bypass `top`, so only enable them when we ARE the top
        // (no wrapper is observing calls). Compare the trait object's data pointer
        // to `self`.
        let direct = std::ptr::eq(
            top as *const dyn CodeSpace<M> as *const u8,
            self as *const JitCranelift<M> as *const u8,
        ) as u8;
        let cur_slots = slots_ptr(frame);
        // `cur` holds a fresh handle to the frame (an `Arc` clone, not an alloc) —
        // but only for bodies that actually consult the frame chain (let / closures
        // / `up > 0`). Leaf/loop bodies read locals through `cur_slots` alone, so we
        // skip the clone (and keep the frame trivially uniquely-owned for the pool).
        let cur = if compiled.needs_cur { frame.clone() } else { None };
        let mut ctx = JitCtx {
            rc: std::ptr::null(),
            top,
            rt: rt as *mut Runtime<M>,
            cur_slots: Cell::new(cur_slots),
            consts_base: Cell::new(consts_base),
            global_base: Cell::new(global_base),
            global_len: Cell::new(global_len),
            fast_base: Cell::new(fast_base),
            fast_len: Cell::new(fast_len),
            direct: Cell::new(direct),
            self_closure: Cell::new(self_closure),
            cur: RefCell::new(cur),
            saved: RefCell::new(Vec::new()),
            templates: &compiled.templates as *const Vec<ClosureTemplate>,
            tail_pending: Cell::new(false),
            tail_callee: Cell::new(0),
            tail_args: args_buf as *mut Vec<u64>,
        };
        // The top of a native call tree is its own run context; children copy this
        // pointer and read the shared fields through it.
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

    /// Build a callee frame, reusing a pooled `Arc<Frame>` if one is free (no
    /// allocation) and otherwise falling back to `Runtime::build_call_frame`. The
    /// pool only ever holds uniquely-owned frames, so `Arc::get_mut` never fails.
    fn alloc_frame(
        &self,
        rt: &mut Runtime<M>,
        nparams: usize,
        variadic: bool,
        args: &[u64],
        env: Locals,
    ) -> Locals {
        let popped = self.frame_pool.borrow_mut().pop();
        let mut rc = match popped {
            Some(rc) => rc,
            None => return rt.build_call_frame(nparams, variadic, args, env),
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
        f.parent = env;
        Some(rc)
    }

    /// Return a frame to the pool IF its call did not capture it (a closure that
    /// escaped would keep `strong_count > 1`). `Arc::get_mut` succeeding is the
    /// exact, sound test for "no other owner" — a captured frame can never be
    /// recycled, so this cannot corrupt a live closure's environment.
    fn recycle(&self, frame: Locals) {
        if let Some(mut rc) = frame {
            if let Some(f) = Arc::get_mut(&mut rc) {
                f.slots.clear();
                f.parent = None; // don't pin the parent chain alive in the pool
                let mut pool = self.frame_pool.borrow_mut();
                if pool.len() < FRAME_POOL_CAP {
                    pool.push(rc);
                }
            }
            // else: still shared (captured by a live closure) — just drop it.
        }
    }

    /// The proper-tail-call trampoline: run a body; if it tail-calls, resolve the
    /// callee, build its frame, and loop — a bounded native stack for unbounded
    /// tail recursion. Non-tail calls still recurse (through `top`), so
    /// composition and macro-reentrancy hold, exactly like the tree-walker.
    fn run_trampoline(
        &self,
        top: &dyn CodeSpace<M>,
        rt: &mut Runtime<M>,
        mut compiled: Arc<Compiled>,
        mut frame: Locals,
        // The closure being run (0 = none, e.g. a top-level expr), so a tail call
        // back to the SAME closure can take the fast path below.
        mut cur_callee: u64,
        mut cur_nparams: usize,
        mut cur_variadic: bool,
    ) -> u64 {
        let mut args_buf: Vec<u64> = Vec::new();
        loop {
            let outcome = self.run_once(top, rt, &compiled, &frame, &mut args_buf, cur_callee);
            match outcome {
                Ok(v) => {
                    self.recycle(frame);
                    return v;
                }
                // Self-tail-call fast path: the loop calls ITSELF, its frame did
                // not escape (`!needs_cur` => no closure captured it, so it is
                // uniquely owned), and it is non-variadic. Refill the SAME frame's
                // slots in place and reuse the same compiled body — no decode, no
                // cache lookup, no pool traffic, no allocation. This is what makes
                // a tail loop cheap.
                Err(callee)
                    if callee == cur_callee
                        && !cur_variadic
                        && !compiled.needs_cur
                        && args_buf.len() == cur_nparams =>
                {
                    let f = frame
                        .as_mut()
                        .and_then(Arc::get_mut)
                        .expect("non-escaping frame is uniquely owned");
                    f.slots.clear();
                    f.slots.extend(args_buf.iter().map(|&a| AtomicU64::new(a)));
                    // parent (env) unchanged: same closure => same environment.
                    // `compiled`, `cur_*` unchanged. `cur_slots` is recomputed in
                    // the next `run_once`.
                }
                Err(callee) => {
                    // General tail call: reclaim the old frame, resolve the callee
                    // (through the inline cache).
                    self.recycle(frame);
                    let (nparams, variadic, comp, env) = self.resolve_call(rt, callee);
                    frame = self.alloc_frame(rt, nparams, variadic, &args_buf, env);
                    compiled = comp;
                    cur_callee = callee;
                    cur_nparams = nparams;
                    cur_variadic = variadic;
                }
            }
        }
    }
}

impl<M: ModelArithJit> CodeSpace<M> for JitCranelift<M> {
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64 {
        // A top-level expression: compile it as a standalone body and run it.
        // (Not cached — matches the bytecode tier's `eval_ir`.) A tail call in the
        // expression trampolines just like one inside a function body.
        let compiled = self.compile(ir, false, 0);
        // Size the fast-call table to cover every closure that already exists, so
        // its base is stable for this whole form's run (it is only ever written,
        // never grown, during execution). Doing it here (form start) means no
        // native frame is holding a stale base when it (re)allocates.
        {
            let n = rt.heap().len();
            let mut t = self.fast_targets.borrow_mut();
            if t.len() < n {
                t.resize(n, FastTarget::default());
            }
        }
        // A top-level expr is not a closure body: no current callee (0), so the
        // self-tail-call fast path is inert here (it engages once inside a fn).
        self.run_trampoline(top, rt, compiled, locals.clone(), 0, 0, false)
    }

    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        // Resolve through the monomorphic inline cache (decode + heap + compiled +
        // env all skipped on a repeat callee).
        let (nparams, variadic, compiled, env) = self.resolve_call(rt, callee);
        // Publish this closure's native fast entry so future call sites can jump to
        // it directly (once; a filled slot is left as-is).
        if compiled.body_fast_ok && !variadic {
            let id = M::R::as_ref(callee) as usize;
            if let Some(slot) = self.fast_targets.borrow_mut().get_mut(id) {
                if slot.code.is_null() {
                    slot.code = compiled.code;
                    slot.nparams = nparams as u32;
                }
            }
        }
        let frame = self.alloc_frame(rt, nparams, variadic, args, env);
        self.run_trampoline(top, rt, compiled, frame, callee, nparams, variadic)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Lowering: `Ir` -> Cranelift IR.
// ─────────────────────────────────────────────────────────────────────────

/// Compile `ir` as a whole function body into an already-prepared builder.
fn build_body<M: ModelArithJit>(
    module: &mut JITModule,
    fb: &mut FunctionBuilder,
    shims: Shims,
    ir: &Ir,
    templates: &mut Vec<ClosureTemplate>,
    tail_root: bool,
    nparams: usize,
) {
    let entry = fb.create_block();
    fb.append_block_params_for_function_params(entry);
    fb.switch_to_block(entry);
    fb.seal_block(entry);
    let ctx_val = fb.block_params(entry)[0];
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
        let_enter: module.declare_func_in_func(shims.let_enter, fb.func),
        let_set: module.declare_func_in_func(shims.let_set, fb.func),
        let_exit: module.declare_func_in_func(shims.let_exit, fb.func),
    };

    // A signature matching every compiled body — `fn(*mut JitCtx) -> u64` — for
    // the native `call_indirect` fast path.
    let body_sig = {
        let mut s = module.make_signature();
        s.params.push(AbiParam::new(module.target_config().pointer_type()));
        s.returns.push(AbiParam::new(I64));
        fb.import_signature(s)
    };

    let mut c = Compiler {
        fb,
        refs,
        ctx_val,
        templates,
        body_sig,
        // Stable byte offsets of `JitCtx` fields (repr(C)) for inline reads and for
        // building a callee context on the stack at a native call site.
        off_cur_slots: core::mem::offset_of!(JitCtx<'static, M>, cur_slots) as i32,
        off_consts_base: core::mem::offset_of!(JitCtx<'static, M>, consts_base) as i32,
        off_global_base: core::mem::offset_of!(JitCtx<'static, M>, global_base) as i32,
        off_fast_base: core::mem::offset_of!(JitCtx<'static, M>, fast_base) as i32,
        off_fast_len: core::mem::offset_of!(JitCtx<'static, M>, fast_len) as i32,
        off_direct: core::mem::offset_of!(JitCtx<'static, M>, direct) as i32,
        off_self_closure: core::mem::offset_of!(JitCtx<'static, M>, self_closure) as i32,
        off_tail_pending: core::mem::offset_of!(JitCtx<'static, M>, tail_pending) as i32,
        off_rc,
        ctx_size: core::mem::size_of::<JitCtx<'static, M>>() as u32,
        loop_header: None,
        loop_vars: Vec::new(),
        rc_val,
    };
    // A self-tail-recursive, non-escaping body gets a loop header: a tail call to
    // the same closure refills the frame in place and branches here (an O(1)-stack
    // native loop), with the shim/trampoline as the fallback for any other tail
    // call. Bodies without a tail call (or that need the heap frame chain) skip
    // this and compile straight through.
    if tail_root && !body_needs_cur(ir) && has_tail_call(ir) {
        // Lift the params into SSA variables, seeded from the incoming frame, so the
        // loop body reads/writes registers and a self-tail just redefines them.
        let base = c.load_ctx_field(c.off_cur_slots);
        for i in 0..nparams {
            let var = c.fb.declare_var(I64);
            let v = c.fb.ins().load(I64, MemFlagsData::trusted(), base, (i * 8) as i32);
            c.fb.def_var(var, v);
            c.loop_vars.push(var);
        }
        let header = c.fb.create_block();
        c.fb.ins().jump(header, &[]);
        c.fb.switch_to_block(header);
        c.loop_header = Some(header);
    }
    // The whole body is in TAIL position: a tail call in it either loops (self) or
    // trampolines (`shim_tail_call`). `compile` threads that through if / do / let.
    let result = c.compile::<M>(ir, tail_root);
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
    templates: &'a mut Vec<ClosureTemplate>,
    body_sig: cranelift_codegen::ir::SigRef,
    off_cur_slots: i32,
    off_consts_base: i32,
    off_global_base: i32,
    off_fast_base: i32,
    off_fast_len: i32,
    off_direct: i32,
    off_self_closure: i32,
    off_tail_pending: i32,
    off_rc: i32,
    ctx_size: u32,
    /// The run-context pointer, loaded once at entry; shared-field reads use it.
    rc_val: Value,
    /// For a self-tail-recursive body: the block to branch back to on a tail call
    /// to the same closure (an in-place native loop). `None` disables the self-loop
    /// (tail calls then use the shim/trampoline).
    loop_header: Option<cranelift_codegen::ir::Block>,
    /// For a self-loop body: the params held as SSA variables (Cranelift inserts
    /// the loop phis), so the loop runs in REGISTERS, not the memory frame. `up == 0`
    /// locals read these; a self-tail redefines them and branches. Empty otherwise.
    loop_vars: Vec<cranelift_frontend::Variable>,
}

impl<'a, 'b> Compiler<'a, 'b> {
    fn call_shim(&mut self, f: cranelift_codegen::ir::FuncRef, args: &[Value]) -> Value {
        let inst = self.fb.ins().call(f, args);
        self.fb.inst_results(inst)[0]
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

    /// A non-tail call. Try the NATIVE fast path — resolve the callee to its
    /// compiled entry inline through the fast-call table, build the callee's frame
    /// and context ON THIS STACK, and `call_indirect` to it (no FFI, no `invoke`).
    /// Fall back to the shim path (`top.invoke`) for anything ineligible, which is
    /// what preserves composition: the guard requires `direct` (so a wrapped
    /// backend never fast-paths) and a filled, arity-matching table entry (so
    /// continuation / non-JIT callees, whose entries stay null, go through `top`).
    fn emit_call<M: ModelArithJit>(&mut self, callee: Value, argvals: &[Value]) -> Value {
        let n = argvals.len();
        let ctx = self.ctx_val;
        let flags = MemFlagsData::trusted();

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
        let idx = self.fb.ins().imul_imm(id, 16); // sizeof FastTarget
        let entry = self.fb.ins().iadd(fbase, idx);
        let code = self.fb.ins().load(I64, flags, entry, 0);
        let np = self.fb.ins().load(cranelift_codegen::ir::types::I32, flags, entry, 8);
        let has_code = self.fb.ins().icmp_imm(IntCC::NotEqual, code, 0);
        let arity_ok = self.fb.ins().icmp_imm(IntCC::Equal, np, n as i64);
        let guard2 = self.fb.ins().band(has_code, arity_ok);
        self.fb.ins().brif(guard2, fastb, &[], slowb, &[]);

        // ── native fast path: build frame + context, jump to the compiled body ──
        self.fb.switch_to_block(fastb);
        self.fb.seal_block(fastb);
        // The callee's frame: a stack array of its args (it is non-escaping, so a
        // stack frame is safe; it never outlives this call).
        let fsize = (n.max(1) * 8) as u32;
        let frame_ss = self.fb.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            fsize,
            3,
        ));
        for (i, &a) in argvals.iter().enumerate() {
            self.fb.ins().stack_store(a, frame_ss, (i * 8) as i32);
        }
        let frame_addr = self.fb.ins().stack_addr(I64, frame_ss, 0);
        // The callee's context. A non-escaping fast body only ever reads the fields
        // set below (it never touches `cur`/`saved`/`templates`, and `tail_callee`
        // is written by `shim_tail_call` before any read), so we skip zeroing the
        // rest and just set what is live — roughly halving the per-call stores.
        let words = self.ctx_size.div_ceil(8) as i32;
        let ctx_ss = self.fb.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (words * 8) as u32,
            3,
        ));
        // Just four stores: the shared run-context pointer, the frame, the callee's
        // own bits (for its self-tail loop), and a clear tail-pending flag. The
        // callee reads everything else through `rc`.
        self.fb.ins().stack_store(rc, ctx_ss, self.off_rc);
        self.fb.ins().stack_store(frame_addr, ctx_ss, self.off_cur_slots);
        self.fb.ins().stack_store(callee, ctx_ss, self.off_self_closure);
        let zero8 = self.fb.ins().iconst(I8, 0);
        self.fb.ins().stack_store(zero8, ctx_ss, self.off_tail_pending);
        let new_ctx = self.fb.ins().stack_addr(I64, ctx_ss, 0);
        let inst = self.fb.ins().call_indirect(self.body_sig, code, &[new_ctx]);
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
        self.fb.use_var(result)
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
            // Innermost locals (`up == 0`, the common case — all function params
            // and every hot-loop variable) load inline; deeper frames need the
            // parent-chain walk, which stays on the shim.
            Ir::Local { up, idx } => {
                if *up == 0 && (*idx as usize) < self.loop_vars.len() {
                    // A self-loop param: read the SSA register, not the frame.
                    self.fb.use_var(self.loop_vars[*idx as usize])
                } else if *up == 0 {
                    self.emit_local0_load(*idx)
                } else {
                    let upv = self.i32const(*up as u32);
                    let idxv = self.i32const(*idx as u32);
                    let ctx = self.ctx_val;
                    self.call_shim(self.refs.load_local, &[ctx, upv, idxv])
                }
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
            Ir::Lambda {
                nparams,
                variadic,
                body,
            } => {
                let tid = self.templates.len() as u32;
                self.templates.push(ClosureTemplate {
                    nparams: *nparams,
                    variadic: *variadic,
                    body: body.clone(),
                });
                let tidv = self.i32const(tid);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.make_closure, &[ctx, tidv])
            }
            Ir::Call(f, args) => {
                let callee = self.compile::<M>(f, false);
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                if tail {
                    let ctx = self.ctx_val;
                    if let Some(header) = self.loop_header {
                        // Self-tail-call becomes a native loop: if the callee is
                        // THIS closure, refill the frame in place and branch to the
                        // header (no FFI, O(1) stack). Otherwise fall through to the
                        // shim + trampoline, which handles mutual tail recursion
                        // with full TCO.
                        let flags = MemFlagsData::trusted();
                        let sc = self.fb.ins().load(I64, flags, ctx, self.off_self_closure);
                        let is_self = self.fb.ins().icmp(IntCC::Equal, callee, sc);
                        let selfloop = self.fb.create_block();
                        let notself = self.fb.create_block();
                        self.fb.ins().brif(is_self, selfloop, &[], notself, &[]);

                        self.fb.switch_to_block(selfloop);
                        self.fb.seal_block(selfloop);
                        if self.loop_vars.is_empty() {
                            // Memory frame: refill the slots in place.
                            let base = self.fb.ins().load(I64, flags, ctx, self.off_cur_slots);
                            for (i, &a) in argvals.iter().enumerate() {
                                self.fb.ins().store(flags, a, base, (i * 8) as i32);
                            }
                        } else {
                            // Register loop: redefine the SSA vars (args were already
                            // computed from the OLD values, so no clobber hazard).
                            for (i, &a) in argvals.iter().enumerate() {
                                if i < self.loop_vars.len() {
                                    self.fb.def_var(self.loop_vars[i], a);
                                }
                            }
                        }
                        self.fb.ins().jump(header, &[]);

                        // Non-self: the shim tail-call, whose result flows to the
                        // function return (the trampoline reads `tail_pending`).
                        self.fb.switch_to_block(notself);
                        self.fb.seal_block(notself);
                        let (addr, count) = self.spill_args(&argvals);
                        self.call_shim(self.refs.tail_call, &[ctx, callee, addr, count])
                    } else {
                        let (addr, count) = self.spill_args(&argvals);
                        self.call_shim(self.refs.tail_call, &[ctx, callee, addr, count])
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
            Ir::Prim(Prim::Gc, _) => panic!(
                "JIT tier: (gc) is a safepoint not modeled here; run it on the tree-walker / CEK"
            ),
            Ir::Prim(Prim::CallEc | Prim::CallCc | Prim::Reset | Prim::Shift, _) => panic!(
                "JIT tier: continuations not supported; run on the tree-walker / CEK"
            ),
            Ir::Prim(Prim::Apply, _) => {
                panic!("JIT tier: apply requires a backend that intercepts it (CekMachine)")
            }
            // Every other prim: compute args, escape to the runtime (the native
            // analogue of the bytecode tier's `Slow`).
            Ir::Prim(p, args) => {
                let argvals: Vec<Value> =
                    args.iter().map(|a| self.compile::<M>(a, false)).collect();
                let (addr, count) = self.spill_args(&argvals);
                let tagv = self.i32const(prim_tag(*p));
                let ctx = self.ctx_val;
                self.call_shim(self.refs.prim, &[ctx, tagv, addr, count])
            }
            // Sequential `let`: push a fresh frame, fill its slots in order (each
            // init can see the earlier ones), run the body in it, then restore.
            // The body inherits tail position. When the whole `let` is in tail
            // position we SKIP `let_exit`: the body either tail-calls (this frame
            // is abandoned to the trampoline) or returns a value (the function
            // returns) — either way the `JitCtx`, and its scope stack, is dropped,
            // so restoring the frame would be dead work.
            Ir::Let(inits, body) => {
                let n = inits.len() as u32;
                let ctx = self.ctx_val;
                let nv = self.i32const(n);
                self.call_shim(self.refs.let_enter, &[ctx, nv]);
                for (k, init) in inits.iter().enumerate() {
                    let v = self.compile::<M>(init, false);
                    let idxv = self.i32const(k as u32);
                    let ctx = self.ctx_val;
                    self.call_shim(self.refs.let_set, &[ctx, idxv, v]);
                }
                let result = self.compile::<M>(body, tail);
                if !tail {
                    let ctx = self.ctx_val;
                    let z = self.i32const(0);
                    self.call_shim(self.refs.let_exit, &[ctx, z]);
                }
                result
            }
            Ir::SetLocal { up, idx, val } => {
                let v = self.compile::<M>(val, false);
                if *up == 0 && (*idx as usize) < self.loop_vars.len() {
                    self.fb.def_var(self.loop_vars[*idx as usize], v);
                    v
                } else if *up == 0 {
                    self.emit_local0_store(*idx, v);
                    v // `set!` evaluates to the assigned value
                } else {
                    let upv = self.i32const(*up as u32);
                    let idxv = self.i32const(*idx as u32);
                    let ctx = self.ctx_val;
                    self.call_shim(self.refs.set_local, &[ctx, upv, idxv, v])
                }
            }
            Ir::SetGlobal { name, val } => {
                let v = self.compile::<M>(val, false);
                let namev = self.i32const(*name);
                let ctx = self.ctx_val;
                self.call_shim(self.refs.set_global, &[ctx, namev, v])
            }
            Ir::DefMethod { .. } | Ir::Dispatch { .. } | Ir::FieldGet { .. } => {
                panic!("JIT tier: dispatch not supported; run on the tree-walker")
            }
            Ir::Try { .. } => panic!("JIT tier: try/catch not supported; run on the tree-walker"),
        }
    }

    /// Call the runtime's `prim` (the checked, promoting arithmetic / any prim) —
    /// the native `Slow` escape, shared by the arithmetic fallback and non-fast prims.
    fn slow_prim(&mut self, op: Prim, args: &[Value]) -> Value {
        let (addr, count) = self.spill_args(args);
        let tagv = self.i32const(prim_tag(op));
        let ctx = self.ctx_val;
        self.call_shim(self.refs.prim, &[ctx, tagv, addr, count])
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
        Ir::Prim(Prim::CallCc | Prim::Reset | Prim::Shift | Prim::CallEc | Prim::Gc | Prim::Apply | Prim::Spawn, _) => false,
        Ir::Dispatch { .. } | Ir::DefMethod { .. } | Ir::FieldGet { .. } => false,
        // try/catch unwinds the native stack; only the TreeWalk tier models it.
        Ir::Try { .. } => false,
        // A `Lambda` only makes a closure here; its body's compilability is
        // decided when that closure is invoked. So do NOT descend.
        Ir::Lambda { .. } => true,
        Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Global(_) => true,
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
