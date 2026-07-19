//! The one thread-safe runtime: the world (behind a lock live edits take), the
//! shared [`Heap`], persistent globals, native `foreign fn` implementations,
//! actor mailboxes, effects, and the preemptive stop-the-world GC rendezvous.
//! The object model is non-moving handles over atomically-swappable bodies, so
//! a migration one thread performs is visible to the others without tearing,
//! and old bodies are reclaimed by refcount (no hazard pointers).
//!
//! [`Shared`] holds no execution loop — the [`crate::Engine`] drives actors
//! (interpreted or native, one thread or many) over this state. It is
//! thread-safe even when used from a single thread, so "single-threaded" is a
//! deployment choice, not a separate runtime.

use crate::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, RwLockReadGuard};


fn rt_type_error(message: &str) -> Condition {
    Condition::RuntimeTypeError {
        function: 0,
        pc: 0,
        message: message.into(),
    }
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
    /// Behind an `RwLock` so live edits can land while worker threads run: a
    /// worker holds a read guard for one step; an edit takes the write lock
    /// (which waits out the in-flight steps), installs new versions, and the
    /// next `Call` on each worker re-resolves the current one.
    world: RwLock<World>,
    /// The one [`Heap`] — the *same* type the single-threaded interpreter uses,
    /// shared here behind the `Arc<Shared>`. Migration, allocation, and the
    /// soundness predicate all live on it, so this tier can no longer drift from
    /// the interpreter on object semantics.
    heap: Heap,
    /// Persistent `letonce` globals, behind an `RwLock` like the world so an
    /// edit can add one while workers read them. A `Ref` global is a GC root.
    globals: RwLock<BTreeMap<DefId, Value>>,
    /// Native `foreign fn` implementations, one `Mutex` per fn so concurrent
    /// calls to *different* foreign fns run in parallel while calls to the *same*
    /// one serialize (native code is rarely reentrant-safe). This is what lets
    /// FFI run on the concurrent tier, not just the single-threaded one.
    foreign_registry: Mutex<BTreeMap<ForeignFnId, Arc<Mutex<ForeignFn>>>>,
    output: Mutex<Vec<Value>>,
    /// One mailbox per actor id, created before threads spawn so a `Send` never
    /// races the recipient into existence. A `Recv` polls its own mailbox,
    /// hitting a GC safepoint each spin so a waiting actor stays collectable.
    mailboxes: Mutex<BTreeMap<usize, Arc<Mutex<std::collections::VecDeque<Value>>>>>,
    /// Mirror of `world.epoch`, readable without the world lock. The engine
    /// uses it to cheaply detect that installed code changed (per-actor code
    /// caches, native-frame address reuse) without taking a read guard per
    /// turn. Updated under the world write lock, so it never runs ahead of
    /// what a subsequent read guard will observe.
    code_epoch: AtomicU64,
    /// Fast-path poll flag: threads check this at every safepoint and only take
    /// the coordination lock when a collection is actually pending.
    gc_pending: AtomicBool,
    gc: Mutex<GcCoord>,
    gc_cv: Condvar,
}

impl Shared {
    /// An empty runtime. Definitions arrive through the live install path —
    /// there is no separate "setup" runtime to freeze.
    pub fn new() -> Arc<Shared> {
        Arc::new(Shared {
            world: RwLock::new(World::default()),
            heap: Heap::default(),
            globals: RwLock::new(BTreeMap::new()),
            foreign_registry: Mutex::new(BTreeMap::new()),
            output: Mutex::new(Vec::new()),
            mailboxes: Mutex::new(BTreeMap::new()),
            code_epoch: AtomicU64::new(0),
            gc_pending: AtomicBool::new(false),
            gc: Mutex::new(GcCoord::default()),
            gc_cv: Condvar::new(),
        })
    }

    /// A read guard on the world — the engine holds one per interpreted
    /// instruction (and for native code/version lookups), so a live edit's
    /// write lock waits out in-flight turns and lands between them.
    pub(crate) fn world_read(&self) -> RwLockReadGuard<'_, World> {
        self.world.read().unwrap()
    }

    /// The one heap — public for inspection/differential testing; mutation goes
    /// through the engine and the install/extern paths.
    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    /// Publish the frontend's declared `foreign fn` signatures and `letonce`
    /// global types (accumulated across evals) so verification and the
    /// `CallForeign`/`LoadGlobal` runtime checks see them.
    pub fn set_declared_interface(
        &self,
        foreign_sigs: BTreeMap<ForeignFnId, (Vec<Type>, Type)>,
        global_types: BTreeMap<DefId, Type>,
    ) {
        let mut world = self.world.write().unwrap();
        world.foreign_sigs = foreign_sigs;
        world.global_types = global_types;
    }

    pub fn output(&self) -> Vec<Value> {
        self.output.lock().unwrap().clone()
    }

    /// Current schema version of an object (for tests/inspection).
    pub fn object_schema(&self, id: ObjectId) -> Option<Version> {
        self.heap.schema_version(id)
    }

    /// An object's current body snapshot (for tests/inspection).
    pub fn object_body(&self, id: ObjectId) -> Option<Arc<Body>> {
        self.heap.body(id)
    }

    pub fn object_count(&self) -> usize {
        self.heap.len()
    }

    // ── live editing while worker threads run ────────────────────────────────
    // Each takes the world write lock (waiting out any in-flight steps) and
    // applies the *same* `World` install path the single-threaded runtime uses.
    // A running frame keeps its pinned version; the next `Call` picks up the new
    // one. These take `&self` because the editor holds an `Arc<Shared>` clone.

    pub fn install_schema(&self, schema: Schema) -> Result<(), InstallError> {
        let mut world = self.world.write().unwrap();
        let result = world.install_schema(schema);
        self.code_epoch.store(world.epoch, Ordering::Release);
        result
    }
    pub fn install_function(&self, function: Function) -> Result<(), InstallError> {
        let mut world = self.world.write().unwrap();
        let result = world.install_function(function);
        self.code_epoch.store(world.epoch, Ordering::Release);
        result
    }
    pub fn install_verified_function(
        &self,
        function: Function,
        deps: std::collections::BTreeSet<DefId>,
    ) -> Result<(), InstallError> {
        let mut world = self.world.write().unwrap();
        let result = world.install_verified_function(function, deps);
        self.code_epoch.store(world.epoch, Ordering::Release);
        result
    }
    pub fn install_migration(&self, migration: Migration) -> Result<(), InstallError> {
        self.world
            .write()
            .unwrap()
            .install_migration(migration, &self.heap)
    }

    /// The current code epoch (mirrors `world.epoch`) without taking the world
    /// lock — the cheap staleness check for cached compiled-code lookups.
    pub fn code_epoch(&self) -> u64 {
        self.code_epoch.load(Ordering::Acquire)
    }

    /// Register (or replace) a native `foreign fn` on the running tier — the
    /// live-edit counterpart to installing a new function version.
    pub fn register_foreign(&self, id: ForeignFnId, f: ForeignFn) {
        self.foreign_registry
            .lock()
            .unwrap()
            .insert(id, Arc::new(Mutex::new(f)));
    }
    /// Set a global's value directly (seed a native handle, publish a `letonce`).
    pub fn set_global(&self, id: DefId, value: Value) {
        self.globals.write().unwrap().insert(id, value);
    }

    /// Invoke a registered foreign fn. The per-fn `Mutex` is held only for the
    /// call, so different foreign fns run concurrently; `None` = not registered.
    pub(crate) fn call_foreign(&self, id: ForeignFnId, args: &[Value]) -> Option<Value> {
        let cell = self.foreign_registry.lock().unwrap().get(&id).cloned()?;
        let mut f = cell.lock().unwrap();
        Some(f(args))
    }
    pub(crate) fn global(&self, id: DefId) -> Option<Value> {
        self.globals.read().unwrap().get(&id).cloned()
    }

    /// Pre-create actor `tid`'s mailbox so an external `send_to` can't race its
    /// creation. For hosts/tests coordinating with a running actor.
    pub fn ensure_mailbox(&self, tid: usize) {
        self.mailbox(tid);
    }
    /// Deliver a value into actor `tid`'s mailbox from outside the thread pool.
    pub fn send_to(&self, tid: usize, value: Value) -> bool {
        self.deliver(tid, value)
    }

    pub(crate) fn emit(&self, value: Value) {
        self.output.lock().unwrap().push(value);
    }

    pub(crate) fn mailbox(&self, tid: usize) -> Arc<Mutex<std::collections::VecDeque<Value>>> {
        self.mailboxes
            .lock()
            .unwrap()
            .entry(tid)
            .or_insert_with(|| Arc::new(Mutex::new(std::collections::VecDeque::new())))
            .clone()
    }

    /// Push a value into actor `target`'s mailbox. False if no such actor.
    pub(crate) fn deliver(&self, target: usize, value: Value) -> bool {
        let Some(mb) = self.mailboxes.lock().unwrap().get(&target).cloned() else {
            return false;
        };
        mb.lock().unwrap().push_back(value);
        true
    }

    /// Enter as an active mutator (a thread that touches the shared heap). Also
    /// used by the JIT tier's worker threads (in the `livetype` crate), so it is
    /// public. Pair with [`Shared::unregister`].
    pub fn register(&self) {
        self.gc.lock().unwrap().active += 1;
    }

    pub fn unregister(&self, tid: usize) {
        let mut c = self.gc.lock().unwrap();
        c.active -= 1;
        c.roots.remove(&tid);
        // A collector may be waiting for `parked == active`; a finished thread
        // lowers `active`, so nudge it to re-check.
        self.gc_cv.notify_all();
    }

    /// The root-carrying core of a safepoint, for callers that compute their own
    /// root set — the JIT worker threads read theirs out of native frame slots.
    /// Assumes a collection is pending.
    pub fn safepoint_roots(&self, tid: usize, roots: Vec<ObjectId>) {
        let mut c = self.gc.lock().unwrap();
        c.roots.insert(tid, roots);
        c.parked += 1;
        let generation = c.generation;
        self.gc_cv.notify_all();
        while c.generation == generation {
            c = self.gc_cv.wait(c).unwrap();
        }
    }

    /// Is a collection pending? JIT workers poll this to decide whether to park.
    pub fn gc_pending(&self) -> bool {
        self.gc_pending.load(Ordering::Acquire)
    }

    // ── JIT bridge (called by the `livetype` crate's externs/driver) ─────────
    /// Allocate at the current schema (thread-safe) — the JIT's `lt_new`.
    pub fn jit_new(
        &self,
        type_id: DefId,
        supplied: &[(FieldId, Value)],
    ) -> Result<ObjectId, Condition> {
        let world = self.world.read().unwrap();
        self.heap.new_object(type_id, supplied, &world)
    }
    /// Migrate + read a field (thread-safe) — the JIT's `lt_get_field`.
    pub fn jit_get_field(&self, id: ObjectId, field: FieldId) -> Result<Value, Condition> {
        let world = self.world.read().unwrap();
        self.heap.get_field(id, field, &world)
    }
    /// Construct an enum variant at the current schema — the JIT's
    /// `lt_new_variant`.
    pub fn jit_new_variant(
        &self,
        type_id: DefId,
        variant: VariantId,
        supplied: &[(FieldId, Value)],
    ) -> Result<ObjectId, Condition> {
        let world = self.world.read().unwrap();
        self.heap.new_variant(type_id, variant, supplied, &world)
    }
    /// The `match` barrier — the JIT's `lt_case_variant` (see
    /// [`Heap::variant_case`]).
    pub fn jit_case_variant(
        &self,
        id: ObjectId,
        arms: &[(VariantId, usize)],
    ) -> Result<usize, Condition> {
        let world = self.world.read().unwrap();
        self.heap.variant_case(id, arms, &world)
    }
    /// Commit an observation — the JIT's `lt_emit`.
    pub fn jit_emit(&self, value: Value) {
        self.emit(value);
    }
    pub fn value_ok(&self, value: &Value, expected: &Type) -> bool {
        self.heap.value_ok(value, expected)
    }
    /// Run a closure against a read guard on the world — for the JIT driver's
    /// function/version lookups.
    pub fn with_world<R>(&self, f: impl FnOnce(&World) -> R) -> R {
        f(&self.world.read().unwrap())
    }

    /// The full `CallForeign` semantics for the JIT extern (concurrent tier).
    pub fn jit_call_foreign(
        &self,
        foreign: ForeignFnId,
        args: &[Value],
    ) -> Result<Value, Condition> {
        let result_ty = self
            .world
            .read()
            .unwrap()
            .foreign_sigs
            .get(&foreign)
            .map(|(_, r)| r.clone())
            .ok_or_else(|| rt_type_error("call to unknown foreign fn"))?;
        let result = self.call_foreign(foreign, args).ok_or_else(|| {
            rt_type_error(&format!("foreign fn {foreign} has no registered implementation"))
        })?;
        if !self.value_ok(&result, &result_ty) {
            return Err(rt_type_error(&format!(
                "foreign result: expected {result_ty:?}, found a value of another type"
            )));
        }
        Ok(result)
    }

    /// The `LoadGlobal` semantics for the JIT extern (concurrent tier).
    pub fn jit_load_global(&self, id: DefId) -> Result<Value, Condition> {
        self.global(id)
            .ok_or_else(|| rt_type_error("global read before initialization"))
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
        // Globals are roots too (same as the single-threaded collector).
        for value in self.globals.read().unwrap().values() {
            if let Value::Ref(id) = value {
                work.push(*id);
            }
        }
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
