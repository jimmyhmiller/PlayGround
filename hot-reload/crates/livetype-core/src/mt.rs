//! A thread-safe runtime tier: many actors on real OS threads over one shared
//! heap. This is the "properly thread-safe runtime" direction — the object
//! model is non-moving handles over atomically-swappable bodies, so a migration
//! one thread performs is visible to the others without tearing, and old bodies
//! are reclaimed by refcount (no hazard pointers).
//!
//! Setup (schemas, functions, auto-derived migrations, verification) is done
//! through the ordinary [`Runtime`] and then frozen via [`Runtime::into_parts`];
//! only the *executor* is reimplemented here to be thread-safe. During a
//! concurrent run the world is immutable (updates land at quiescent points), so
//! it needs no lock; the heap and effects do. Actors are single-frame here — the
//! shared-heap race the design asks about is about objects, not call stacks — so
//! `Call` is intentionally unsupported in this tier.
//!
//! Not yet built (documented follow-ons): the JIT under threads, mid-run
//! installs while threads run, and a *preemptive* stop-the-world GC (today
//! [`Shared::collect`] runs at a quiescent point).

use crate::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// One call frame of a concurrent actor: the pinned function version it runs,
/// its program counter and registers, and where its result goes in the caller.
struct MtFrame {
    func: DefId,
    version: Version,
    pc: usize,
    regs: Vec<Option<Value>>,
    return_to: Option<usize>,
}

/// The outcome of running one actor to a stop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Complete(Value),
    Paused(Condition),
}

/// Stop-the-world GC rendezvous. Actor threads increment `active` while
/// running and, when a collection is requested, publish their live roots and
/// park (bumping `parked`) at their next safepoint; the collector proceeds once
/// `parked == active`, sweeps, then bumps `generation` to release them.
#[derive(Default)]
struct GcCoord {
    active: usize,
    parked: usize,
    generation: u64,
    roots: BTreeMap<usize, Vec<ObjectId>>,
    collections: u64,
}

/// The shared, thread-safe runtime.
pub struct Shared {
    world: World,
    /// The one [`Heap`] — the *same* type the single-threaded interpreter uses,
    /// shared here behind the `Arc<Shared>`. Migration, allocation, and the
    /// soundness predicate all live on it, so this tier can no longer drift from
    /// the interpreter on object semantics.
    heap: Heap,
    output: Mutex<Vec<Value>>,
    /// One mailbox per actor id, created before threads spawn so a `Send` never
    /// races the recipient into existence. A `Recv` polls its own mailbox,
    /// hitting a GC safepoint each spin so a waiting actor stays collectable.
    mailboxes: Mutex<BTreeMap<usize, Arc<Mutex<std::collections::VecDeque<Value>>>>>,
    /// Fast-path poll flag: threads check this at every safepoint and only take
    /// the coordination lock when a collection is actually pending.
    gc_pending: AtomicBool,
    gc: Mutex<GcCoord>,
    gc_cv: Condvar,
}

impl Shared {
    /// Freeze a set-up [`Runtime`] into a shareable runtime — the heap moves in
    /// as-is (same `Heap`, now shared behind the `Arc`), no rebuild.
    pub fn from_runtime(rt: Runtime) -> Arc<Shared> {
        let (world, heap) = rt.into_parts();
        Arc::new(Shared {
            world,
            heap,
            output: Mutex::new(Vec::new()),
            mailboxes: Mutex::new(BTreeMap::new()),
            gc_pending: AtomicBool::new(false),
            gc: Mutex::new(GcCoord::default()),
            gc_cv: Condvar::new(),
        })
    }

    pub fn output(&self) -> Vec<Value> {
        self.output.lock().unwrap().clone()
    }

    /// Current schema version of an object (for tests/inspection).
    pub fn object_schema(&self, id: ObjectId) -> Option<Version> {
        self.heap.schema_version(id)
    }

    pub fn object_count(&self) -> usize {
        self.heap.len()
    }

    fn emit(&self, value: Value) {
        self.output.lock().unwrap().push(value);
    }

    fn mailbox(&self, tid: usize) -> Arc<Mutex<std::collections::VecDeque<Value>>> {
        self.mailboxes
            .lock()
            .unwrap()
            .entry(tid)
            .or_insert_with(|| Arc::new(Mutex::new(std::collections::VecDeque::new())))
            .clone()
    }

    /// Push a value into actor `target`'s mailbox. False if no such actor.
    fn deliver(&self, target: usize, value: Value) -> bool {
        let Some(mb) = self.mailboxes.lock().unwrap().get(&target).cloned() else {
            return false;
        };
        mb.lock().unwrap().push_back(value);
        true
    }

    /// Run one actor — a full call stack — to completion or a trap. `tid`
    /// identifies it to the GC (its published-roots slot) and names its mailbox.
    /// It counts as an active mutator and hits a safepoint every step, so a
    /// preemptive collection can pause it; a blocked `Recv` keeps hitting
    /// safepoints while it polls, so it stays collectable too.
    pub fn run_actor(self: &Arc<Self>, tid: usize, function: DefId, args: Vec<Value>) -> Outcome {
        struct Active<'a>(&'a Shared, usize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.unregister(self.1);
            }
        }
        self.register();
        let _active = Active(self, tid);
        let mailbox = self.mailbox(tid);

        let version = self.world.current_functions[&function];
        let registers = match &self.world.functions[&(function, version)] {
            FunctionState::Ready(f) => f.registers,
            FunctionState::Broken { diagnostics, .. } => {
                return Outcome::Paused(Condition::BrokenFunction {
                    function,
                    diagnostics: diagnostics.clone(),
                });
            }
        };
        let mut regs = vec![None; registers];
        for (i, v) in args.into_iter().enumerate() {
            regs[i] = Some(v);
        }
        let frames = vec![MtFrame {
            func: function,
            version,
            pc: 0,
            regs,
            return_to: None,
        }];

        // Drive the *one* step semantics (shared with the interpreter) over a
        // thread-local frame stack. A safepoint at the top of each turn keeps
        // this actor parkable for a stop-the-world collection; a `Recv` with no
        // message yet comes back as `Flow::Blocked`, so we spin (still hitting
        // the safepoint) until it arrives.
        let mut machine = MtMachine {
            shared: self,
            mailbox,
            frames,
            done: None,
        };
        loop {
            machine.shared.safepoint(tid, &machine.frames);
            let (func, version, pc) = {
                let t = machine.frames.last().unwrap();
                (t.func, t.version, t.pc)
            };
            let instruction = match &machine.shared.world.functions[&(func, version)] {
                FunctionState::Ready(f) => f.code[pc].clone(),
                _ => unreachable!("a frame only pins ready code"),
            };
            match step_instruction(&mut machine, &instruction) {
                Ok(Flow::Stepped) => {
                    if let Some(value) = machine.done.take() {
                        return Outcome::Complete(value);
                    }
                }
                Ok(Flow::Blocked) => std::thread::yield_now(),
                Err(condition) => return Outcome::Paused(condition),
            }
        }
    }

    /// Spawn one OS thread per actor over the shared runtime and join them,
    /// returning outcomes in the actors' order. Mailboxes for every actor id
    /// are created up front, so a `Send` can never race its recipient into
    /// existence. All threads share this `Shared`'s heap.
    pub fn run_threads(self: &Arc<Self>, actors: Vec<(DefId, Vec<Value>)>) -> Vec<Outcome> {
        for tid in 0..actors.len() {
            self.mailbox(tid);
        }
        let handles: Vec<_> = actors
            .into_iter()
            .enumerate()
            .map(|(tid, (func, args))| {
                let shared = Arc::clone(self);
                std::thread::spawn(move || shared.run_actor(tid, func, args))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    fn register(&self) {
        self.gc.lock().unwrap().active += 1;
    }

    fn unregister(&self, tid: usize) {
        let mut c = self.gc.lock().unwrap();
        c.active -= 1;
        c.roots.remove(&tid);
        // A collector may be waiting for `parked == active`; a finished thread
        // lowers `active`, so nudge it to re-check.
        self.gc_cv.notify_all();
    }

    /// A safepoint: if a collection is pending, publish this actor's live roots
    /// (every reference across its whole call stack) and park until the
    /// collector releases us. Called every instruction, so preemption latency is
    /// one step.
    fn safepoint(&self, tid: usize, frames: &[MtFrame]) {
        if !self.gc_pending.load(Ordering::Acquire) {
            return;
        }
        let live: Vec<ObjectId> = frames
            .iter()
            .flat_map(|f| f.regs.iter().flatten())
            .filter_map(|v| match v {
                Value::Ref(id) => Some(*id),
                _ => None,
            })
            .collect();
        let mut c = self.gc.lock().unwrap();
        c.roots.insert(tid, live);
        c.parked += 1;
        let generation = c.generation;
        self.gc_cv.notify_all();
        while c.generation == generation {
            c = self.gc_cv.wait(c).unwrap();
        }
    }

    /// Preemptive stop-the-world collection: request a pause, wait until every
    /// running actor has parked at a safepoint, sweep from the union of their
    /// published roots, then release them. Safe to call from any thread while
    /// actors run. Returns the number of objects reclaimed.
    pub fn request_gc(&self) -> usize {
        self.gc_pending.store(true, Ordering::Release);
        let mut c = self.gc.lock().unwrap();
        self.gc_cv.notify_all();
        while c.parked != c.active {
            c = self.gc_cv.wait(c).unwrap();
        }
        let roots: Vec<ObjectId> = c.roots.values().flatten().copied().collect();
        drop(c); // mutators stay parked (generation unchanged); free the lock to sweep
        let reclaimed = self.collect(&roots);
        let mut c = self.gc.lock().unwrap();
        c.parked = 0;
        c.roots.clear();
        c.generation += 1;
        c.collections += 1;
        self.gc_pending.store(false, Ordering::Release);
        self.gc_cv.notify_all();
        reclaimed
    }

    /// How many preemptive collections have completed.
    pub fn collections(&self) -> u64 {
        self.gc.lock().unwrap().collections
    }

    /// Stop-the-world precise collection over caller-supplied roots (the live
    /// references of every actor). Must be called at a quiescent point — no
    /// actor mid-step — which the caller guarantees today by joining threads
    /// first. Returns the number of objects reclaimed.
    pub fn collect(&self, roots: &[ObjectId]) -> usize {
        let mut work: Vec<ObjectId> = roots.to_vec();
        let mut live = std::collections::BTreeSet::new();
        while let Some(id) = work.pop() {
            if !live.insert(id) {
                continue;
            }
            for child in self.heap.child_refs(id) {
                work.push(child);
            }
        }
        self.heap.retain(&live)
    }
}

/// A concurrent worker's [`Machine`]: the shared step semantics run over a
/// thread-local frame stack layered on the shared runtime. Effects go to the
/// shared heap/output/mailboxes (all internally synchronized). This tier has no
/// FFI or globals (`Unsupported`); `Recv` polls the mailbox and reports
/// `WouldBlock` so the driver can spin at a safepoint.
struct MtMachine<'a> {
    shared: &'a Arc<Shared>,
    mailbox: Arc<Mutex<std::collections::VecDeque<Value>>>,
    frames: Vec<MtFrame>,
    /// Set when the top frame returns — the driver turns it into `Complete`.
    done: Option<Value>,
}

impl MtMachine<'_> {
    fn top(&self) -> &MtFrame {
        self.frames.last().unwrap()
    }
    fn top_mut(&mut self) -> &mut MtFrame {
        self.frames.last_mut().unwrap()
    }
}

impl Machine for MtMachine<'_> {
    fn world(&self) -> &World {
        &self.shared.world
    }
    fn heap(&self) -> &Heap {
        &self.shared.heap
    }
    fn current(&self) -> (DefId, Version) {
        let t = self.top();
        (t.func, t.version)
    }
    fn pc(&self) -> usize {
        self.top().pc
    }
    fn reg(&self, i: usize) -> Option<Value> {
        self.top().regs.get(i).cloned().flatten()
    }
    fn set_reg(&mut self, dst: usize, value: Value) {
        self.top_mut().regs[dst] = Some(value);
    }
    fn set_pc(&mut self, pc: usize) {
        self.top_mut().pc = pc;
    }
    fn advance(&mut self) {
        self.top_mut().pc += 1;
    }
    fn emit(&mut self, value: Value) {
        self.shared.emit(value);
    }
    fn global(&self, _id: DefId) -> GlobalRead {
        GlobalRead::Unsupported
    }
    fn call_foreign(&mut self, _id: ForeignFnId, _args: &[Value]) -> ForeignCall {
        ForeignCall::Unsupported
    }
    fn push_call(
        &mut self,
        callee: DefId,
        version: Version,
        registers: Vec<Option<Value>>,
        return_reg: usize,
    ) {
        self.frames.push(MtFrame {
            func: callee,
            version,
            pc: 0,
            regs: registers,
            return_to: Some(return_reg),
        });
    }
    fn do_return(&mut self, value: Value) {
        let done = self.frames.pop().unwrap();
        match done.return_to {
            Some(dst) => self.top_mut().regs[dst] = Some(value),
            None => self.done = Some(value),
        }
    }
    fn send(&mut self, target: usize, value: Value) -> Option<bool> {
        Some(self.shared.deliver(target, value))
    }
    fn recv(&mut self) -> RecvResult {
        match self.mailbox.lock().unwrap().pop_front() {
            Some(v) => RecvResult::Got(v),
            None => RecvResult::WouldBlock,
        }
    }
}
