//! THE executor. One engine, one actor loop, every configuration.
//!
//! An [`Engine`] is the runtime the whole system runs on: the thread-safe
//! [`Shared`] world/heap/GC plus a tiering policy. Every actor — on the calling
//! thread or on worker threads — runs the same loop over a mixed stack of
//! interpreted frames (`Value` registers, driven by [`step_instruction`]) and
//! native frames ([`RawSlot`] registers, driven by compiled `step` functions).
//! A per-`(function, version)` call counter promotes hot functions to native
//! code; where compiled code comes from is abstracted behind [`TierSource`], so
//! this crate never links LLVM: the `livetype` crate supplies the compiler, and
//! a [`NoJit`] engine (used under Miri and as the differential-testing oracle)
//! simply never promotes.
//!
//! "Interpreted vs JIT" and "single- vs multi-threaded" are therefore
//! *configuration of one code path*, not separate executors:
//! - interpreter-only = `Engine::new(NoJit, …)`
//! - always-JIT       = a compiling source with `threshold` 0
//! - auto-tiering     = a compiling source with a positive `threshold`
//! - concurrency      = the same [`Engine::run`] loop on more threads
//!   ([`Engine::run_threads`]), with the same live-edit and STW-GC behavior.
//!
//! Live edits land through [`Shared`]'s world lock; a running frame keeps its
//! pinned version, the next call re-resolves the current one, and a version
//! that gets hot recompiles on demand. Trap-and-repair holds across tiers: a
//! paused actor keeps its mixed stack, and [`Engine::resume`] /
//! [`Engine::resume_with`] continue it whichever kind of frame is on top.
//!
//! ## The fast paths (what keeps this loop cheap)
//!
//! - The interpreter runs *batches* of instructions under one world read guard
//!   with the current function resolved once per frame change, instead of a
//!   lock + map lookup + instruction clone per instruction. A batch ends at a
//!   `Yield`, a block, a stop, a tier switch, or [`INTERP_BATCH`] instructions
//!   — so a pending edit or stop-the-world GC still gets in within a bounded
//!   window (and immediately at every explicit safe point).
//! - A native frame caches its compiled entry address and declared result type
//!   at push time, so a steady-state native turn (a loop iteration hitting its
//!   `Yield`, a call, a return) takes **no locks at all**. This is sound
//!   because a function *version* is immutable and compiled engines are never
//!   deallocated: an address, once resolved, stays valid forever.
//! - Compiled-address lookups go through a per-actor cache of the
//!   [`TierSource`]'s map, invalidated by [`Shared::code_epoch`] (an atomic
//!   mirror of the world epoch) — resolving a callee's tier at call time
//!   doesn't re-enter the compiler's lock unless an edit actually landed.

use crate::mt::{Outcome, Shared};
use crate::native::{
    NativeHost, OUT_CALL, OUT_CONDITION, OUT_RETURN, OUT_TYPE_ERROR, OUT_YIELD, RawFrame, RawSlot,
    TAG_BOOL, TAG_FOREIGN, TAG_I64, TAG_KIND_SHIFT, TAG_MASK, TAG_REF, TAG_STR, TAG_UNIT,
    step_at,
};
use crate::runtime::{ForeignFn, InstallError, ResumePlan, operand_type_error, resume_shape};
use crate::{
    ActorStatus, Condition, DefId, Flow, ForeignCall, ForeignFnId, Frame, Function, FunctionState,
    GlobalRead, Heap, Instruction, Machine, Migration, ObjectId, RecvResult, Schema, Type, Value,
    Version, World, step_instruction,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The compiled-address map a [`TierSource`] hands out: every Ready function
/// version's native `step` entry, for one world epoch.
pub type NativeMap = Arc<HashMap<(DefId, Version), usize>>;

/// Where compiled `step` code comes from. The engine asks for the whole map
/// (and caches it per actor, keyed by [`Shared::code_epoch`]) so steady-state
/// execution never re-enters the source's locks; implementations compile on
/// demand and cache by world epoch. `None` means "nothing ever compiles" —
/// the interpreter-only configuration.
pub trait TierSource: Send + Sync {
    fn native_map(&self, world: &World) -> Option<NativeMap>;
}

/// The no-compiler source: every function runs interpreted, forever. This is
/// the Miri-safe configuration and the differential-testing oracle.
pub struct NoJit;
impl TierSource for NoJit {
    fn native_map(&self, _world: &World) -> Option<NativeMap> {
        None
    }
}

/// One native frame: flat `RawSlot` registers a compiled `step` operates on
/// directly. The boxed slice gives native code (and the GC) a stable pointer.
pub struct JitFrame {
    pub func_id: DefId,
    pub version: Version,
    pub pc: usize,
    pub regs: Box<[RawSlot]>,
    /// Caller register to receive the result, or `None` for the entry frame.
    pub return_to: Option<usize>,
    /// The compiled `step` entry for this frame's pinned version, resolved at
    /// push time. A version's code is immutable and compiled engines are never
    /// torn down, so this never goes stale — native turns take no locks.
    addr: usize,
    /// The pinned version's declared result type, so the return-boundary
    /// soundness check needs no world lookup.
    result_ty: Type,
}

/// A frame in an actor's stack: interpreted or native. Both carry the caller
/// register their result returns to; calls and returns marshal values at the
/// boundary, so any mix of tiers composes.
pub enum TierFrame {
    Interp(Frame),
    Jit(JitFrame),
}

impl TierFrame {
    pub fn function(&self) -> (DefId, Version) {
        match self {
            TierFrame::Interp(f) => f.function,
            TierFrame::Jit(f) => (f.func_id, f.version),
        }
    }
    pub fn pc(&self) -> usize {
        match self {
            TierFrame::Interp(f) => f.pc,
            TierFrame::Jit(f) => f.pc,
        }
    }
    fn return_to(&self) -> Option<usize> {
        match self {
            TierFrame::Interp(f) => f.return_to,
            TierFrame::Jit(f) => f.return_to,
        }
    }
}

/// One running (or paused, or finished) computation: a mixed-tier frame stack
/// plus its status and mailbox. Owned by the caller so a paused actor's whole
/// stack survives for inspection, repair, and resume.
pub struct Actor {
    /// Identifies this actor to the GC (its published-roots slot) and names its
    /// mailbox (the `Send` target id).
    pub tid: usize,
    pub stack: Vec<TierFrame>,
    pub status: ActorStatus,
    /// Created lazily on the first `Recv` (or up front by
    /// [`Engine::run_threads`]) so short-lived actors — trampoline calls,
    /// `letonce` initializers — don't grow the mailbox table.
    mailbox: Option<Arc<Mutex<VecDeque<Value>>>>,
    /// This actor's snapshot of the compiled-address map: `(code epoch, map)`.
    /// Refreshed from the [`TierSource`] only when [`Shared::code_epoch`]
    /// advances past it.
    code: Option<(u64, Option<NativeMap>)>,
    /// Function versions this actor already knows are promoted. Hotness is
    /// monotonic per version, so this local cache lets the steady-state call
    /// path skip the global promotion counter's lock entirely.
    hot_local: HashSet<(DefId, Version)>,
}

impl Actor {
    /// Every live [`ObjectId`] in this actor's frame slots — the precise root
    /// set, identical in contract for interpreted and native frames.
    pub fn roots(&self) -> Vec<ObjectId> {
        let mut roots = Vec::new();
        for frame in &self.stack {
            match frame {
                TierFrame::Interp(f) => {
                    for value in f.registers.iter().flatten() {
                        if let Value::Ref(id) = value {
                            roots.push(*id);
                        }
                    }
                }
                TierFrame::Jit(f) => {
                    for slot in f.regs.iter() {
                        if slot.tag == TAG_REF {
                            roots.push(slot.payload as ObjectId);
                        }
                    }
                }
            }
        }
        roots
    }

    /// Is the top frame currently native code? (Tests assert promotion.)
    pub fn top_is_native(&self) -> bool {
        matches!(self.stack.last(), Some(TierFrame::Jit(_)))
    }
}

/// The result of one engine turn — one interpreted instruction, or one native
/// `step` call (which runs until it hands control back).
#[derive(Debug, PartialEq, Eq)]
pub enum Turn {
    /// Work happened; the actor is still runnable.
    Progress,
    /// The actor crossed a `Yield` safe point (still runnable). Hosts that
    /// interleave (a REPL landing edits between safe points) stop here;
    /// [`Engine::run`] just keeps going.
    Yielded,
    /// A blocking `Recv` found no message; retry after a safepoint.
    Blocked,
    /// The actor completed; its status holds the value.
    Done,
    /// The actor trapped; its status holds the condition.
    Paused,
}

/// Auto-spawned actors (trampolines, `letonce` initializers) take tids from
/// here so they can never collide with the dense `0..n` tids
/// [`Engine::run_threads`] hands its workers (which tests use as `Send`
/// targets).
const AUTO_TID_BASE: usize = 1 << 32;

/// How many interpreted instructions may run under one world read guard before
/// the engine lets pending edits and stop-the-world GC in. Explicit safe points
/// (`Yield`), calls into native frames, blocks, and stops end a batch early.
const INTERP_BATCH: usize = 64;

#[derive(Default)]
struct Tiering {
    counts: HashMap<(DefId, Version), u64>,
    hot: HashSet<(DefId, Version)>,
}

/// The one runtime+executor. See the module docs.
pub struct Engine {
    shared: Arc<Shared>,
    source: Arc<dyn TierSource>,
    /// A function version's activation count at which it is promoted to native
    /// code. `0` promotes on first entry; `u64::MAX` (or a [`NoJit`] source)
    /// never promotes.
    threshold: u64,
    tiering: Mutex<Tiering>,
    next_auto_tid: AtomicUsize,
}

impl Engine {
    pub fn new(source: Arc<dyn TierSource>, threshold: u64) -> Arc<Engine> {
        Arc::new(Engine {
            shared: Shared::new(),
            source,
            threshold,
            tiering: Mutex::new(Tiering::default()),
            next_auto_tid: AtomicUsize::new(AUTO_TID_BASE),
        })
    }

    /// An interpreter-only engine — the oracle configuration.
    pub fn interp() -> Arc<Engine> {
        Engine::new(Arc::new(NoJit), u64::MAX)
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// How many function versions have been promoted to native code so far.
    /// (With `threshold` 0 everything is native from first entry and nothing is
    /// counted; this reports promotions the counter performed.)
    pub fn promoted(&self) -> usize {
        self.tiering.lock().unwrap().hot.len()
    }
    /// Was `(func, version)` promoted by the activation counter?
    pub fn is_hot(&self, func: DefId, version: Version) -> bool {
        self.tiering.lock().unwrap().hot.contains(&(func, version))
    }

    // ── the world: installs land live, whichever threads are running ─────────

    pub fn install_schema(&self, schema: Schema) -> Result<(), InstallError> {
        self.shared.install_schema(schema)
    }
    pub fn install_function(&self, function: Function) -> Result<(), InstallError> {
        self.shared.install_function(function)
    }
    pub fn install_verified_function(
        &self,
        function: Function,
        deps: std::collections::BTreeSet<DefId>,
    ) -> Result<(), InstallError> {
        self.shared.install_verified_function(function, deps)
    }
    pub fn install_migration(&self, migration: Migration) -> Result<(), InstallError> {
        self.shared.install_migration(migration)
    }
    pub fn register_foreign(&self, id: ForeignFnId, f: ForeignFn) {
        self.shared.register_foreign(id, f);
    }
    pub fn set_global(&self, id: DefId, value: Value) {
        self.shared.set_global(id, value);
    }
    pub fn global(&self, id: DefId) -> Option<Value> {
        self.shared.global(id)
    }
    pub fn output(&self) -> Vec<Value> {
        self.shared.output()
    }
    pub fn with_world<R>(&self, f: impl FnOnce(&World) -> R) -> R {
        self.shared.with_world(f)
    }

    // ── running ───────────────────────────────────────────────────────────────

    /// Spawn an actor at `func`'s current version. The entry frame goes through
    /// the same tiering decision as every call, so an always-JIT engine
    /// (`threshold` 0) natively executes single-function programs too.
    pub fn spawn(&self, func: DefId, args: Vec<Value>) -> Result<Actor, Condition> {
        let tid = self.next_auto_tid.fetch_add(1, Ordering::Relaxed);
        self.spawn_with_tid(tid, func, args)
    }

    /// Spawn with an explicit tid — [`Engine::run_threads`] uses dense `0..n`
    /// tids so hand-built `Send` targets stay meaningful.
    pub fn spawn_with_tid(
        &self,
        tid: usize,
        func: DefId,
        args: Vec<Value>,
    ) -> Result<Actor, Condition> {
        let world = self.shared.world_read();
        let version = *world.current_functions.get(&func).ok_or_else(|| {
            Condition::BrokenFunction {
                function: func,
                diagnostics: vec!["unknown function".into()],
            }
        })?;
        let registers_len = match &world.functions[&(func, version)] {
            FunctionState::Ready(f) => f.registers,
            FunctionState::Broken { diagnostics, .. } => {
                return Err(Condition::BrokenFunction {
                    function: func,
                    diagnostics: diagnostics.clone(),
                });
            }
        };
        let mut registers = vec![None; registers_len];
        for (slot, value) in args.into_iter().enumerate() {
            registers[slot] = Some(value);
        }
        let mut actor = Actor {
            tid,
            stack: Vec::new(),
            status: ActorStatus::Runnable,
            mailbox: None,
            code: None,
            hot_local: HashSet::new(),
        };
        self.push_callee(&world, &mut actor, func, version, registers, None);
        Ok(actor)
    }

    /// Run an actor to completion or a trap on the calling thread. Registers as
    /// a GC mutator and hits a safepoint between turns, so a preemptive
    /// stop-the-world collection (from any thread) can pause it; a blocked
    /// `Recv` keeps hitting safepoints while it polls, so it stays collectable.
    /// Live edits land between turns/batches through the world lock.
    pub fn run(&self, actor: &mut Actor) -> Outcome {
        struct Active<'a>(&'a Shared, usize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.unregister(self.1);
            }
        }
        self.shared.register();
        let _active = Active(&self.shared, actor.tid);
        loop {
            match &actor.status {
                ActorStatus::Complete(v) => return Outcome::Complete(*v),
                ActorStatus::Paused(c) => return Outcome::Paused(c.clone()),
                ActorStatus::Runnable => {}
            }
            if self.shared.gc_pending() {
                self.shared.safepoint_roots(actor.tid, actor.roots());
            }
            let turn = match actor.stack.last() {
                Some(TierFrame::Interp(_)) => self.interp_batch(actor, INTERP_BATCH),
                _ => self.step(actor),
            };
            if matches!(turn, Turn::Blocked) {
                std::thread::yield_now();
            }
        }
    }

    /// Spawn `func(args)` and run it to a stop — the one-shot form (the
    /// trampoline, `letonce` initializers, tests).
    pub fn run_call(&self, func: DefId, args: Vec<Value>) -> Outcome {
        match self.spawn(func, args) {
            Ok(mut actor) => self.run(&mut actor),
            Err(c) => Outcome::Paused(c),
        }
    }

    /// One OS thread per actor over this engine, tids `0..n`, mailboxes created
    /// up front so a `Send` can never race its recipient into existence.
    /// Returns each actor's [`Outcome`] in order.
    pub fn run_threads(self: &Arc<Self>, actors: Vec<(DefId, Vec<Value>)>) -> Vec<Outcome> {
        for tid in 0..actors.len() {
            self.shared.mailbox(tid);
        }
        let handles: Vec<_> = actors
            .into_iter()
            .enumerate()
            .map(|(tid, (func, args))| {
                let engine = Arc::clone(self);
                std::thread::spawn(move || match engine.spawn_with_tid(tid, func, args) {
                    Ok(mut actor) => engine.run(&mut actor),
                    Err(c) => Outcome::Paused(c),
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    /// Continue a paused actor after a repair (a redefined function, an
    /// installed migration). The stack is intact from the pause and the top
    /// frame's pc still points at the trapping instruction, so re-driving
    /// re-checks it against the repaired world — and re-traps cleanly if the
    /// repair didn't take. Works with any mix of tiers on the stack.
    pub fn resume(&self, actor: &mut Actor) -> Outcome {
        self.thaw(actor);
        self.run(actor)
    }

    /// Mark a paused actor runnable again without running it — for hosts that
    /// drive at [`Engine::step`] granularity. The next step re-executes the
    /// trapping instruction (and re-traps cleanly if nothing was repaired).
    pub fn thaw(&self, actor: &mut Actor) {
        if matches!(actor.status, ActorStatus::Paused(_)) && !actor.stack.is_empty() {
            actor.status = ActorStatus::Runnable;
        }
    }

    /// The type of value a paused actor's con-freeness trap expects, so a
    /// developer knows what to hand back. `None` if the actor isn't paused on a
    /// value-resumable type trap.
    pub fn pause_expected(&self, actor: &Actor) -> Option<Type> {
        if !matches!(
            actor.status,
            ActorStatus::Paused(Condition::RuntimeTypeError { .. })
        ) {
            return None;
        }
        let frame = actor.stack.last()?;
        let world = self.shared.world_read();
        let FunctionState::Ready(f) = &world.functions[&frame.function()] else {
            return None;
        };
        resume_shape(&f.code[frame.pc()], &f.result, &world)
            .ok()
            .map(|(ty, _)| ty)
    }

    /// Resume a con-freeness trap by supplying a value — the delimited-
    /// continuation repair. The frozen instruction "produces" `value` and the
    /// frame continues; the value must satisfy the trap's expected type, so a
    /// repair can never reintroduce an ill-typed value. Leaves the actor
    /// `Runnable` (or `Complete`); call [`Engine::run`] to continue it.
    pub fn resume_with(&self, actor: &mut Actor, value: Value) -> Result<(), String> {
        if !matches!(
            actor.status,
            ActorStatus::Paused(Condition::RuntimeTypeError { .. })
        ) {
            return Err("actor is not paused on a resumable type trap".into());
        }
        let (expected, plan) = {
            let world = self.shared.world_read();
            let frame = actor.stack.last().ok_or("paused actor has no frame")?;
            let FunctionState::Ready(f) = &world.functions[&frame.function()] else {
                return Err("frame pins non-ready code".into());
            };
            resume_shape(&f.code[frame.pc()], &f.result, &world)?
        };
        if !self.shared.value_ok(&value, &expected) {
            return Err(format!(
                "supplied value does not have the expected type {expected:?}"
            ));
        }
        match plan {
            ResumePlan::SetAdvance(dst) => {
                match actor.stack.last_mut().unwrap() {
                    TierFrame::Interp(f) => {
                        f.registers[dst] = Some(value);
                        f.pc += 1;
                    }
                    TierFrame::Jit(f) => {
                        f.regs[dst] = RawSlot::from_value(&value);
                        f.pc += 1;
                    }
                }
                actor.status = ActorStatus::Runnable;
            }
            ResumePlan::Branch(then_pc, else_pc) => {
                let target = if matches!(value, Value::Bool(true)) {
                    then_pc
                } else {
                    else_pc
                };
                match actor.stack.last_mut().unwrap() {
                    TierFrame::Interp(f) => f.pc = target,
                    TierFrame::Jit(f) => f.pc = target,
                }
                actor.status = ActorStatus::Runnable;
            }
            ResumePlan::ReturnValue => {
                actor.status = ActorStatus::Runnable;
                self.deliver_return(actor, value);
            }
        }
        Ok(())
    }

    /// Host-driven precise collection at a quiescent point: no actor may be
    /// mid-turn. The roots are the given actors' frames plus globals (the
    /// collector adds those itself). For collecting *while* actors run on other
    /// threads, use [`Shared::request_gc`] — the preemptive stop-the-world path.
    pub fn collect(&self, actors: &[&Actor]) -> usize {
        let roots: Vec<ObjectId> = actors.iter().flat_map(|a| a.roots()).collect();
        self.shared.collect(&roots)
    }

    // ── one turn ─────────────────────────────────────────────────────────────

    /// Advance an actor by one turn: one interpreted instruction, or one native
    /// `step` call (which runs until it hands control back — its next call,
    /// return, yield, or trap). Does *not* register with the GC; hosts that
    /// interleave turns with other work collect via [`Engine::collect`].
    pub fn step(&self, actor: &mut Actor) -> Turn {
        if !matches!(actor.status, ActorStatus::Runnable) {
            return match actor.status {
                ActorStatus::Complete(_) => Turn::Done,
                _ => Turn::Paused,
            };
        }
        match actor.stack.last() {
            None => {
                // A runnable actor always has a frame; an empty stack means the
                // last return already delivered a completion.
                actor.status = ActorStatus::Complete(Value::Unit);
                Turn::Done
            }
            Some(TierFrame::Interp(_)) => self.interp_batch(actor, 1),
            Some(TierFrame::Jit(_)) => self.jit_turn(actor),
        }
    }

    /// Execute up to `budget` interpreted instructions under ONE world read
    /// guard, with the current function resolved once per frame change. Ends
    /// early at a `Yield` (the explicit safe point), a block, a stop, or when
    /// the top frame becomes native. This is the cold tier's fast path: the
    /// per-instruction cost is the dispatch itself, not lock/lookup overhead —
    /// while the bounded budget keeps live edits and stop-the-world GC at most
    /// one batch away.
    fn interp_batch(&self, actor: &mut Actor, budget: usize) -> Turn {
        let world = self.shared.world_read();
        let mut cached: Option<((DefId, Version), &Function)> = None;
        for _ in 0..budget {
            let (key, pc) = match actor.stack.last() {
                None => {
                    actor.status = ActorStatus::Complete(Value::Unit);
                    return Turn::Done;
                }
                Some(TierFrame::Jit(_)) => return Turn::Progress, // tier switch
                Some(TierFrame::Interp(f)) => (f.function, f.pc),
            };
            let function = match cached {
                Some((k, f)) if k == key => f,
                _ => {
                    let FunctionState::Ready(f) = &world.functions[&key] else {
                        unreachable!("a frame only pins ready code");
                    };
                    cached = Some((key, f));
                    f
                }
            };
            let instruction = &function.code[pc];
            let yielded = matches!(instruction, Instruction::Yield);
            let mut machine = EngineMachine {
                engine: self,
                world: &world,
                actor,
            };
            match step_instruction(&mut machine, instruction) {
                Ok(Flow::Stepped) => {
                    if matches!(actor.status, ActorStatus::Complete(_)) {
                        return Turn::Done;
                    }
                    if yielded {
                        return Turn::Yielded;
                    }
                }
                Ok(Flow::Blocked) => return Turn::Blocked,
                Err(condition) => {
                    actor.status = ActorStatus::Paused(condition);
                    return Turn::Paused;
                }
            }
        }
        Turn::Progress
    }

    fn jit_turn(&self, actor: &mut Actor) -> Turn {
        // The frame's compiled address was resolved at push time and stays
        // valid forever (immutable version, never-torn-down code) — a native
        // turn starts with no locks. Compiled code reaches the runtime only
        // through the externs, which lock what they need.
        let (outcome, pending, scratch, key) = {
            let TierFrame::Jit(frame) = actor.stack.last_mut().unwrap() else {
                unreachable!()
            };
            let step = step_at(frame.addr);
            let mut raw = RawFrame {
                func_id: frame.func_id as i64,
                version: frame.version.0 as i64,
                pc: frame.pc as i64,
                n_regs: frame.regs.len() as i64,
                regs: frame.regs.as_mut_ptr(),
                scratch: RawSlot::EMPTY,
                return_reg: frame.return_to.map_or(-1, |r| r as i64),
            };
            let mut host = NativeHost::new(&self.shared);
            let out = unsafe { step(&mut raw, &mut host as *mut NativeHost) };
            frame.pc = raw.pc as usize;
            (
                out,
                host.take_pending(),
                raw.scratch,
                (frame.func_id, frame.version),
            )
        };

        match outcome {
            OUT_RETURN => {
                // Check the result against the returning frame's declared type
                // (cached at push) before it leaves the frame — the
                // con-freeness trap for a pinned old function returning a
                // since-migrated value. Done at the slot level, so a native →
                // native return never materializes a `Value` and takes no
                // locks (a reference still consults the heap for its nominal
                // type, exactly like `value_ok`).
                {
                    let TierFrame::Jit(f) = actor.stack.last().unwrap() else {
                        unreachable!()
                    };
                    if !self.slot_ok(scratch, &f.result_ty) {
                        let (fid, ret_pc, ty) = (f.func_id, f.pc, f.result_ty.clone());
                        actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                            function: fid,
                            pc: ret_pc,
                            message: format!(
                                "return value: expected {ty:?}, found a value of another type"
                            ),
                        });
                        return Turn::Paused;
                    }
                }
                self.deliver_return_slot(actor, scratch);
                match actor.status {
                    ActorStatus::Complete(_) => Turn::Done,
                    _ => Turn::Progress,
                }
            }
            OUT_CALL => self.jit_handle_call(actor),
            OUT_YIELD => Turn::Yielded,
            OUT_CONDITION => {
                actor.status = ActorStatus::Paused(
                    pending.expect("CONDITION outcome without a stashed condition"),
                );
                Turn::Paused
            }
            OUT_TYPE_ERROR => {
                let trap_pc = {
                    let TierFrame::Jit(f) = actor.stack.last().unwrap() else {
                        unreachable!()
                    };
                    f.pc
                };
                // Rebuild the exact condition from the trapping instruction so
                // it matches the interpreter's byte for byte (rare path — the
                // world lookup is fine here).
                let instruction = self.shared.with_world(|w| match &w.functions[&key] {
                    FunctionState::Ready(f) => f.code[trap_pc].clone(),
                    _ => unreachable!("a frame only pins ready code"),
                });
                actor.status =
                    ActorStatus::Paused(operand_type_error(key.0, trap_pc, &instruction));
                Turn::Paused
            }
            other => {
                actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                    function: key.0,
                    pc: 0,
                    message: format!("unknown step outcome {other}"),
                });
                Turn::Paused
            }
        }
    }

    /// A `Call` hand-back from a native frame: read the call site from the IR,
    /// gather + type-check arguments from its slots, and push the callee
    /// through the same tiering decision as every other call. One world read
    /// guard, no clones of code or types.
    fn jit_handle_call(&self, actor: &mut Actor) -> Turn {
        let world = self.shared.world_read();
        let (caller_key, call_pc) = {
            let TierFrame::Jit(f) = actor.stack.last().unwrap() else {
                unreachable!()
            };
            ((f.func_id, f.version), f.pc)
        };
        let FunctionState::Ready(caller) = &world.functions[&caller_key] else {
            unreachable!("a frame only pins ready code");
        };
        let Instruction::Call {
            dst,
            function: callee,
            args,
        } = &caller.code[call_pc]
        else {
            actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                function: caller_key.0,
                pc: call_pc,
                message: "CALL outcome at a non-call pc".into(),
            });
            return Turn::Paused;
        };
        let (dst, callee) = (*dst, *callee);

        // Resolve the callee's *current* version — late binding is what makes
        // a live edit take effect at the next call.
        let Some(&version) = world.current_functions.get(&callee) else {
            actor.status = ActorStatus::Paused(Condition::BrokenFunction {
                function: callee,
                diagnostics: vec!["unknown function".to_string()],
            });
            return Turn::Paused;
        };
        let key = (callee, version);
        let function = match &world.functions[&key] {
            FunctionState::Ready(f) => f,
            FunctionState::Broken { diagnostics, .. } => {
                actor.status = ActorStatus::Paused(Condition::BrokenFunction {
                    function: callee,
                    diagnostics: diagnostics.clone(),
                });
                return Turn::Paused;
            }
        };

        // Decide the callee's tier first, then marshal in the representation
        // the callee will actually run in. Argument type checks happen either
        // way — a pinned old caller passing a since-migrated value traps here,
        // exactly as the interpreter's Call arm does.
        let native = if self.decide_hot(actor, key) {
            self.resolve_native(&world, actor, key)
        } else {
            None
        };
        if let Some(addr) = native {
            // Native → native: slots copy straight across, checked at the slot
            // level — no `Value` materialization, no intermediate vec.
            let mut regs = vec![RawSlot::EMPTY; function.registers].into_boxed_slice();
            {
                let TierFrame::Jit(frame) = actor.stack.last().unwrap() else {
                    unreachable!()
                };
                for (slot, (reg, expected)) in args.iter().zip(&function.params).enumerate() {
                    let s = frame.regs[*reg];
                    if !self.slot_ok(s, expected) {
                        actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                            function: callee,
                            pc: call_pc,
                            message: "call argument: expected a value of another type".into(),
                        });
                        return Turn::Paused;
                    }
                    regs[slot] = s;
                }
            }
            let result_ty = function.result.clone();
            {
                let TierFrame::Jit(frame) = actor.stack.last_mut().unwrap() else {
                    unreachable!()
                };
                frame.pc = call_pc + 1;
            }
            actor.stack.push(TierFrame::Jit(JitFrame {
                func_id: callee,
                version,
                pc: 0,
                regs,
                return_to: Some(dst),
                addr,
                result_ty,
            }));
            return Turn::Progress;
        }

        // Native → interpreted: marshal to `Value` registers.
        let mut registers = vec![None; function.registers];
        {
            let TierFrame::Jit(frame) = actor.stack.last().unwrap() else {
                unreachable!()
            };
            for (slot, (reg, expected)) in args.iter().zip(&function.params).enumerate() {
                let value = frame.regs[*reg].to_value();
                if !self.shared.value_ok(&value, expected) {
                    actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                        function: callee,
                        pc: call_pc,
                        message: "call argument: expected a value of another type".into(),
                    });
                    return Turn::Paused;
                }
                registers[slot] = Some(value);
            }
        }
        {
            let TierFrame::Jit(frame) = actor.stack.last_mut().unwrap() else {
                unreachable!()
            };
            frame.pc = call_pc + 1;
        }
        actor.stack.push(TierFrame::Interp(Frame {
            function: key,
            pc: 0,
            registers,
            return_to: Some(dst),
        }));
        Turn::Progress
    }

    /// Does a raw slot satisfy `expected`? Mirrors [`Heap::value_ok`]
    /// (`shallow_type(value) == expected`) without building the `Value`:
    /// scalars are decided by tag alone; a reference still asks the heap for
    /// its nominal type. An empty slot satisfies nothing.
    fn slot_ok(&self, slot: RawSlot, expected: &Type) -> bool {
        match expected {
            Type::I64 => slot.tag == TAG_I64,
            Type::Bool => slot.tag == TAG_BOOL,
            Type::Unit => slot.tag == TAG_UNIT,
            Type::Str => slot.tag == TAG_STR,
            Type::Foreign(kind) => {
                slot.tag & TAG_MASK == TAG_FOREIGN
                    && (slot.tag >> TAG_KIND_SHIFT) as u32 == *kind
            }
            Type::Ref(_) => {
                slot.tag == TAG_REF && self.shared.value_ok(&slot.to_value(), expected)
            }
        }
    }

    /// The tier decision: count this activation and report whether `key` is
    /// (now) promoted. Thresholds 0 (always) and `u64::MAX` (never) skip the
    /// counter lock entirely; a version this actor already saw promoted skips
    /// it too (hotness is monotonic per version, so the local cache is sound).
    fn decide_hot(&self, actor: &mut Actor, key: (DefId, Version)) -> bool {
        match self.threshold {
            0 => true,
            u64::MAX => false,
            threshold => {
                if actor.hot_local.contains(&key) {
                    return true;
                }
                let hot = {
                    let mut t = self.tiering.lock().unwrap();
                    let count = t.counts.entry(key).or_insert(0);
                    *count += 1;
                    if *count >= threshold {
                        t.hot.insert(key);
                    }
                    t.hot.contains(&key)
                };
                if hot {
                    actor.hot_local.insert(key);
                }
                hot
            }
        }
    }

    /// This actor's compiled-step address for `key`, through its epoch-keyed
    /// snapshot of the source's map — the compiler's own lock is entered only
    /// when an edit has actually landed since the snapshot.
    fn resolve_native(&self, world: &World, actor: &mut Actor, key: (DefId, Version)) -> Option<usize> {
        let epoch = self.shared.code_epoch();
        if actor.code.as_ref().map(|(e, _)| *e) != Some(epoch) {
            actor.code = Some((epoch, self.source.native_map(world)));
        }
        actor.code.as_ref().unwrap().1.as_ref()?.get(&key).copied()
    }

    /// Push a frame for `(callee, version)`, deciding its tier: count the
    /// activation, promote past the threshold, and enter native code if the
    /// source has it compiled (a source that can't compile this function —
    /// or a [`NoJit`] source — keeps it interpreted). This is the ONE place a
    /// tier is chosen; spawn, interpreted calls, and native calls all land
    /// here (the native caller's fast path inlines the same decision through
    /// [`Engine::is_hot`] + [`Engine::resolve_native`] to marshal slots
    /// directly).
    fn push_callee(
        &self,
        world: &World,
        actor: &mut Actor,
        callee: DefId,
        version: Version,
        registers: Vec<Option<Value>>,
        return_to: Option<usize>,
    ) {
        let key = (callee, version);
        if self.decide_hot(actor, key) {
            if let Some(addr) = self.resolve_native(world, actor, key) {
                let FunctionState::Ready(f) = &world.functions[&key] else {
                    unreachable!("callers resolve Ready code under this guard");
                };
                let regs: Vec<RawSlot> = registers
                    .iter()
                    .map(|o| o.as_ref().map_or(RawSlot::EMPTY, RawSlot::from_value))
                    .collect();
                actor.stack.push(TierFrame::Jit(JitFrame {
                    func_id: callee,
                    version,
                    pc: 0,
                    regs: regs.into_boxed_slice(),
                    return_to,
                    addr,
                    result_ty: f.result.clone(),
                }));
                return;
            }
        }
        actor.stack.push(TierFrame::Interp(Frame {
            function: key,
            pc: 0,
            registers,
            return_to,
        }));
    }

    /// [`Engine::deliver_return`], starting from a raw slot: a native caller
    /// receives the slot verbatim; only an interpreted caller (or completion)
    /// materializes the `Value`.
    fn deliver_return_slot(&self, actor: &mut Actor, slot: RawSlot) {
        let done = actor.stack.pop().unwrap();
        let return_to = done.return_to();
        match actor.stack.last_mut() {
            None => actor.status = ActorStatus::Complete(slot.to_value()),
            Some(TierFrame::Interp(f)) => {
                if let Some(r) = return_to {
                    f.registers[r] = Some(slot.to_value());
                }
            }
            Some(TierFrame::Jit(f)) => {
                if let Some(r) = return_to {
                    f.regs[r] = slot;
                }
            }
        }
    }

    /// Pop the returning frame and deliver `value` into the caller's register
    /// in the caller's representation; if the stack empties, the actor
    /// completes. The other half of the tier-and-marshal seam.
    fn deliver_return(&self, actor: &mut Actor, value: Value) {
        let done = actor.stack.pop().unwrap();
        let return_to = done.return_to();
        match actor.stack.last_mut() {
            None => actor.status = ActorStatus::Complete(value),
            Some(TierFrame::Interp(f)) => {
                if let Some(r) = return_to {
                    f.registers[r] = Some(value);
                }
            }
            Some(TierFrame::Jit(f)) => {
                if let Some(r) = return_to {
                    f.regs[r] = RawSlot::from_value(&value);
                }
            }
        }
    }
}

/// The [`Machine`] for interpreted frames: the shared step semantics run over
/// the actor's top frame, with effects going to the one [`Shared`] runtime.
/// `Call` and `Return` route through the engine's tiering policy
/// (`push_callee`/`deliver_return`), so an interpreted caller can enter a
/// native callee and vice versa. Nothing is unsupported here — FFI, globals,
/// and message passing all work on the one engine.
struct EngineMachine<'a> {
    engine: &'a Engine,
    world: &'a World,
    actor: &'a mut Actor,
}

impl EngineMachine<'_> {
    fn top(&self) -> &Frame {
        match self.actor.stack.last().unwrap() {
            TierFrame::Interp(f) => f,
            TierFrame::Jit(_) => unreachable!("interp step over a native frame"),
        }
    }
    fn top_mut(&mut self) -> &mut Frame {
        match self.actor.stack.last_mut().unwrap() {
            TierFrame::Interp(f) => f,
            TierFrame::Jit(_) => unreachable!("interp step over a native frame"),
        }
    }
}

impl Machine for EngineMachine<'_> {
    fn world(&self) -> &World {
        self.world
    }
    fn heap(&self) -> &Heap {
        self.engine.shared.heap()
    }
    fn current(&self) -> (DefId, Version) {
        self.top().function
    }
    fn pc(&self) -> usize {
        self.top().pc
    }
    fn reg(&self, i: usize) -> Option<Value> {
        self.top().registers.get(i).copied().flatten()
    }
    fn set_reg(&mut self, dst: usize, value: Value) {
        self.top_mut().registers[dst] = Some(value);
    }
    fn set_pc(&mut self, pc: usize) {
        self.top_mut().pc = pc;
    }
    fn advance(&mut self) {
        self.top_mut().pc += 1;
    }
    fn emit(&mut self, value: Value) {
        self.engine.shared.emit(value);
    }
    fn global(&self, id: DefId) -> GlobalRead {
        match self.engine.shared.global(id) {
            Some(v) => GlobalRead::Value(v),
            None => GlobalRead::Missing,
        }
    }
    fn call_foreign(&mut self, id: ForeignFnId, args: &[Value]) -> ForeignCall {
        match self.engine.shared.call_foreign(id, args) {
            Some(v) => ForeignCall::Done(Ok(v)),
            None => ForeignCall::Done(Err(format!(
                "foreign fn {id} has no registered implementation"
            ))),
        }
    }
    fn push_call(
        &mut self,
        callee: DefId,
        version: Version,
        registers: Vec<Option<Value>>,
        return_reg: usize,
    ) {
        self.engine
            .push_callee(self.world, self.actor, callee, version, registers, Some(return_reg));
    }
    fn do_return(&mut self, value: Value) {
        self.engine.deliver_return(self.actor, value);
    }
    fn send(&mut self, target: usize, value: Value) -> Option<bool> {
        Some(self.engine.shared.deliver(target, value))
    }
    fn recv(&mut self) -> RecvResult {
        // The mailbox is created on first use so short-lived actors never
        // touch the mailbox table.
        let mailbox = self
            .actor
            .mailbox
            .get_or_insert_with(|| self.engine.shared.mailbox(self.actor.tid));
        match mailbox.lock().unwrap().pop_front() {
            Some(v) => RecvResult::Got(v),
            None => RecvResult::WouldBlock,
        }
    }
}
