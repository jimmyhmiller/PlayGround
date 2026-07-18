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
//! Live edits land through [`Shared`]'s world lock between any two turns of any
//! actor; a running frame keeps its pinned version, the next call re-resolves
//! the current one, and a version that gets hot recompiles on demand (the
//! source re-answers per epoch). Trap-and-repair holds across tiers: a paused
//! actor keeps its mixed stack, and [`Engine::resume`] /
//! [`Engine::resume_with`] continue it whichever kind of frame is on top.

use crate::mt::{Outcome, Shared};
use crate::native::{
    NativeHost, OUT_CALL, OUT_CONDITION, OUT_RETURN, OUT_TYPE_ERROR, OUT_YIELD, RawFrame, RawSlot,
    TAG_REF, step_at,
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

/// Where compiled `step` code comes from. The engine asks at frame-push time
/// (to decide a hot callee's tier) and per native turn (so a live edit's new
/// epoch is picked up by recompile-on-demand). `None` means "run interpreted" —
/// either nothing is compiled, or this function can't be (e.g. it contains
/// message-passing ops the codegen doesn't cover).
pub trait TierSource: Send + Sync {
    fn native_step(&self, world: &World, key: (DefId, Version)) -> Option<usize>;
}

/// The no-compiler source: every function runs interpreted, forever. This is
/// the Miri-safe configuration and the differential-testing oracle.
pub struct NoJit;
impl TierSource for NoJit {
    fn native_step(&self, _world: &World, _key: (DefId, Version)) -> Option<usize> {
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
    mailbox: Arc<Mutex<VecDeque<Value>>>,
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
    pub fn promoted(&self) -> usize {
        self.tiering.lock().unwrap().hot.len()
    }
    /// Was `(func, version)` promoted?
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
            mailbox: self.shared.mailbox(tid),
        };
        self.push_callee(&world, &mut actor, func, version, registers, None);
        Ok(actor)
    }

    /// Run an actor to completion or a trap on the calling thread. Registers as
    /// a GC mutator and hits a safepoint every turn, so a preemptive
    /// stop-the-world collection (from any thread) can pause it; a blocked
    /// `Recv` keeps hitting safepoints while it polls, so it stays collectable.
    /// Live edits land between turns through the world lock.
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
                ActorStatus::Complete(v) => return Outcome::Complete(v.clone()),
                ActorStatus::Paused(c) => return Outcome::Paused(c.clone()),
                ActorStatus::Runnable => {}
            }
            if self.shared.gc_pending() {
                self.shared.safepoint_roots(actor.tid, actor.roots());
            }
            match self.step(actor) {
                Turn::Progress | Turn::Yielded | Turn::Done | Turn::Paused => {}
                Turn::Blocked => std::thread::yield_now(),
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
            Some(TierFrame::Interp(_)) => self.interp_turn(actor),
            Some(TierFrame::Jit(_)) => self.jit_turn(actor),
        }
    }

    fn interp_turn(&self, actor: &mut Actor) -> Turn {
        // A read guard is held for exactly one instruction: a live edit takes
        // the write lock between turns, and the next `Call` re-resolves.
        let world = self.shared.world_read();
        let (key, pc) = {
            let TierFrame::Interp(f) = actor.stack.last().unwrap() else {
                unreachable!()
            };
            (f.function, f.pc)
        };
        let instruction = match &world.functions[&key] {
            FunctionState::Ready(f) => f.code[pc].clone(),
            _ => unreachable!("a frame only pins ready code"),
        };
        let yielded = matches!(instruction, Instruction::Yield);
        let mailbox = Arc::clone(&actor.mailbox);
        let mut machine = EngineMachine {
            engine: self,
            world: &world,
            mailbox: &mailbox,
            actor,
        };
        match step_instruction(&mut machine, &instruction) {
            Ok(Flow::Stepped) => match actor.status {
                ActorStatus::Complete(_) => Turn::Done,
                _ if yielded => Turn::Yielded,
                _ => Turn::Progress,
            },
            Ok(Flow::Blocked) => Turn::Blocked,
            Err(condition) => {
                actor.status = ActorStatus::Paused(condition);
                Turn::Paused
            }
        }
    }

    fn jit_turn(&self, actor: &mut Actor) -> Turn {
        let key = {
            let TierFrame::Jit(f) = actor.stack.last().unwrap() else {
                unreachable!()
            };
            (f.func_id, f.version)
        };
        // Resolve compiled code for this frame's pinned version. A live edit
        // bumps the world epoch; the source recompiles on demand, so the
        // pinned (still-Ready) version stays executable across edits.
        let addr = {
            let world = self.shared.world_read();
            self.source.native_step(&world, key)
        };
        let Some(addr) = addr else {
            actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                function: key.0,
                pc: 0,
                message: format!("no compiled step for {}@{}", key.0, key.1.0),
            });
            return Turn::Paused;
        };
        let step = step_at(addr);
        // No world guard is held across the native call: compiled code reaches
        // the runtime only through the externs, which lock what they need.
        let (outcome, pending, scratch) = {
            let TierFrame::Jit(frame) = actor.stack.last_mut().unwrap() else {
                unreachable!()
            };
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
            (out, host.take_pending(), raw.scratch)
        };

        match outcome {
            OUT_RETURN => {
                let result = scratch.to_value();
                let (fid, ret_pc) = {
                    let TierFrame::Jit(f) = actor.stack.last().unwrap() else {
                        unreachable!()
                    };
                    (f.func_id, f.pc)
                };
                // Check the result against the returning frame's declared type
                // before it leaves the frame — the con-freeness trap for a
                // pinned old function returning a since-migrated value.
                let result_ty = self.shared.with_world(|w| match &w.functions[&key] {
                    FunctionState::Ready(f) => Some(f.result.clone()),
                    _ => None,
                });
                if let Some(ty) = result_ty {
                    if !self.shared.value_ok(&result, &ty) {
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
                self.deliver_return(actor, result);
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
                // it matches the interpreter's byte for byte.
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
    /// through the same tiering decision as every other call.
    fn jit_handle_call(&self, actor: &mut Actor) -> Turn {
        let (caller_key, call_pc) = {
            let TierFrame::Jit(f) = actor.stack.last().unwrap() else {
                unreachable!()
            };
            ((f.func_id, f.version), f.pc)
        };
        let world = self.shared.world_read();
        let call = match &world.functions[&caller_key] {
            FunctionState::Ready(f) => f.code[call_pc].clone(),
            _ => unreachable!("a frame only pins ready code"),
        };
        let Instruction::Call {
            dst,
            function: callee,
            args,
        } = call
        else {
            actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                function: caller_key.0,
                pc: call_pc,
                message: "CALL outcome at a non-call pc".into(),
            });
            return Turn::Paused;
        };

        // Resolve the callee's *current* version — late binding is what makes
        // a live edit take effect at the next call.
        let resolved = match world.current_functions.get(&callee) {
            None => Err(vec!["unknown function".to_string()]),
            Some(v) => match &world.functions[&(callee, *v)] {
                FunctionState::Ready(f) => Ok((*v, f.params.clone(), f.registers)),
                FunctionState::Broken { diagnostics, .. } => Err(diagnostics.clone()),
            },
        };
        let (version, params, registers_len) = match resolved {
            Ok(x) => x,
            Err(diagnostics) => {
                actor.status = ActorStatus::Paused(Condition::BrokenFunction {
                    function: callee,
                    diagnostics,
                });
                return Turn::Paused;
            }
        };

        let arg_values: Vec<Value> = {
            let TierFrame::Jit(frame) = actor.stack.last().unwrap() else {
                unreachable!()
            };
            args.iter().map(|r| frame.regs[*r].to_value()).collect()
        };
        // Check each argument against the callee's parameter type before the
        // frame is pushed — a pinned old caller passing a since-migrated value
        // traps here, exactly as the interpreter's Call arm does.
        for (value, expected) in arg_values.iter().zip(&params) {
            if !self.shared.value_ok(value, expected) {
                actor.status = ActorStatus::Paused(Condition::RuntimeTypeError {
                    function: callee,
                    pc: call_pc,
                    message: "call argument: expected a value of another type".into(),
                });
                return Turn::Paused;
            }
        }
        let mut registers = vec![None; registers_len];
        for (slot, value) in arg_values.into_iter().enumerate() {
            registers[slot] = Some(value);
        }
        {
            let TierFrame::Jit(frame) = actor.stack.last_mut().unwrap() else {
                unreachable!()
            };
            frame.pc = call_pc + 1;
        }
        self.push_callee(&world, actor, callee, version, registers, Some(dst));
        Turn::Progress
    }

    /// Push a frame for `(callee, version)`, deciding its tier: count the
    /// activation, promote past the threshold, and enter native code if the
    /// source has it compiled (a source that can't compile this function —
    /// or a [`NoJit`] source — keeps it interpreted). This is the ONE place a
    /// tier is chosen; spawn, interpreted calls, and native calls all land
    /// here.
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
        let hot = {
            let mut t = self.tiering.lock().unwrap();
            let count = t.counts.entry(key).or_insert(0);
            *count += 1;
            if *count >= self.threshold {
                t.hot.insert(key);
            }
            t.hot.contains(&key)
        };
        if hot && self.source.native_step(world, key).is_some() {
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
            }));
        } else {
            actor.stack.push(TierFrame::Interp(Frame {
                function: key,
                pc: 0,
                registers,
                return_to,
            }));
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
    mailbox: &'a Arc<Mutex<VecDeque<Value>>>,
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
        self.top().registers.get(i).cloned().flatten()
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
        match self.mailbox.lock().unwrap().pop_front() {
            Some(v) => RecvResult::Got(v),
            None => RecvResult::WouldBlock,
        }
    }
}
