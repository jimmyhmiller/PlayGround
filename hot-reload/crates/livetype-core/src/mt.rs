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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// One heap slot: a stable handle whose body can be atomically swapped. The
/// `Mutex<Arc<Body>>` gives a reader a consistent whole body (clone the `Arc`,
/// then read) and a migrator an atomic swap; the old `Arc` lives until its last
/// reader drops it.
struct ObjCell {
    body: Mutex<Arc<Body>>,
}

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
    objects: Mutex<BTreeMap<ObjectId, Arc<ObjCell>>>,
    output: Mutex<Vec<Value>>,
    next_object: AtomicU64,
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
    /// Freeze a set-up [`Runtime`] into a shareable runtime.
    pub fn from_runtime(rt: Runtime) -> Arc<Shared> {
        let (world, heap, next_object) = rt.into_parts();
        let objects = heap
            .into_iter()
            .map(|(id, obj)| {
                (
                    id,
                    Arc::new(ObjCell {
                        body: Mutex::new(obj.body),
                    }),
                )
            })
            .collect();
        Arc::new(Shared {
            world,
            objects: Mutex::new(objects),
            output: Mutex::new(Vec::new()),
            next_object: AtomicU64::new(next_object),
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
        let cell = self.objects.lock().unwrap().get(&id).cloned()?;
        let body = cell.body.lock().unwrap();
        Some(body.schema)
    }

    pub fn object_count(&self) -> usize {
        self.objects.lock().unwrap().len()
    }

    fn cell(&self, id: ObjectId) -> Arc<ObjCell> {
        self.objects.lock().unwrap().get(&id).unwrap().clone()
    }

    fn body_of(&self, id: ObjectId) -> Type {
        let cell = self.cell(id);
        let tid = cell.body.lock().unwrap().type_id;
        Type::Ref(tid)
    }

    fn value_ok(&self, value: &Value, expected: &Type) -> bool {
        let actual = match value {
            Value::Unit => Type::Unit,
            Value::I64(_) => Type::I64,
            Value::Bool(_) => Type::Bool,
            Value::Ref(id) => self.body_of(*id),
        };
        actual == *expected
    }

    fn alloc(&self, type_id: DefId, schema: Version, fields: BTreeMap<FieldId, Value>) -> ObjectId {
        let id = self.next_object.fetch_add(1, Ordering::Relaxed) + 1;
        let cell = Arc::new(ObjCell {
            body: Mutex::new(Arc::new(Body {
                type_id,
                schema,
                fields,
            })),
        });
        self.objects.lock().unwrap().insert(id, cell);
        id
    }

    /// Construct an object at the type's current schema, checking each field.
    fn new_object(&self, type_id: DefId, supplied: &[(FieldId, Value)]) -> Result<ObjectId, Condition> {
        let version = self.world.current_schemas[&type_id];
        let schema = &self.world.schemas[&(type_id, version)];
        let mut values = BTreeMap::new();
        for field in &schema.fields {
            let value = supplied
                .iter()
                .find(|(id, _)| *id == field.id)
                .map(|(_, v)| v.clone())
                .or_else(|| field.default.clone())
                .expect("verified constructor");
            if !self.value_ok(&value, &field.ty) {
                return Err(Condition::RuntimeTypeError {
                    function: 0,
                    pc: 0,
                    message: format!("field '{}' has the wrong type", field.name),
                });
            }
            values.insert(field.id, value);
        }
        Ok(self.alloc(type_id, version, values))
    }

    /// Migrate an object up to its current schema (concurrently safe) and read a
    /// field. Migration builds each replacement body *without* holding the
    /// object's lock (so allocating wrapper objects can't deadlock against the
    /// heap-table lock), then swaps it in under the lock with a double-check —
    /// if another thread already advanced this object, the freshly built body is
    /// discarded (its wrapper allocations become garbage the GC reclaims).
    fn read_field(&self, id: ObjectId, field: FieldId) -> Result<Value, Condition> {
        let cell = self.cell(id);
        loop {
            let body = cell.body.lock().unwrap().clone();
            let current = self.world.current_schemas[&body.type_id];
            if body.schema == current {
                return body.fields.get(&field).cloned().ok_or_else(|| {
                    Condition::RuntimeTypeError {
                        function: 0,
                        pc: 0,
                        message: "field is absent after migration".into(),
                    }
                });
            }
            let Some(plan) = self.world.migrations.get(&(body.type_id, body.schema)).cloned() else {
                return Err(Condition::MissingMigration {
                    object: id,
                    type_id: body.type_id,
                    from: body.schema,
                    to: Version(body.schema.0 + 1),
                });
            };
            // Build the next body (allocating wrappers) without the cell lock.
            let mut fields = BTreeMap::new();
            for (target, source) in &plan.fields {
                let value = match source {
                    MigrationSource::Copy(s) => body.fields[s].clone(),
                    MigrationSource::Value(v) => v.clone(),
                    MigrationSource::Wrap {
                        type_id,
                        field,
                        source,
                    } => {
                        let v = self.world.current_schemas[type_id];
                        let wid = self.alloc(
                            *type_id,
                            v,
                            BTreeMap::from([(*field, body.fields[source].clone())]),
                        );
                        Value::Ref(wid)
                    }
                };
                fields.insert(*target, value);
            }
            let next = Arc::new(Body {
                type_id: body.type_id,
                schema: plan.to,
                fields,
            });
            // Swap under the lock, only if nobody migrated this step already.
            {
                let mut slot = cell.body.lock().unwrap();
                if slot.schema == body.schema {
                    *slot = next;
                }
            }
            // Re-read: continue the chain or read the now-current field.
        }
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
        let mut frames = vec![MtFrame {
            func: function,
            version,
            pc: 0,
            regs,
            return_to: None,
        }];

        loop {
            self.safepoint(tid, &frames);
            let (func, version, pc) = {
                let t = frames.last().unwrap();
                (t.func, t.version, t.pc)
            };
            let (instruction, result_ty) = match &self.world.functions[&(func, version)] {
                FunctionState::Ready(f) => (f.code[pc].clone(), f.result.clone()),
                _ => unreachable!("a frame only pins ready code"),
            };
            let err = |msg: &str| {
                Outcome::Paused(Condition::RuntimeTypeError {
                    function: func,
                    pc,
                    message: msg.into(),
                })
            };
            let read = |frames: &Vec<MtFrame>, i: usize| -> Value {
                frames.last().unwrap().regs[i].clone().expect("verified register read")
            };

            match instruction {
                Instruction::Const { dst, value } => {
                    let t = frames.last_mut().unwrap();
                    t.regs[dst] = Some(value);
                    t.pc += 1;
                }
                Instruction::New {
                    dst,
                    type_id,
                    fields,
                } => {
                    let supplied: Vec<(FieldId, Value)> =
                        fields.iter().map(|(f, r)| (*f, read(&frames, *r))).collect();
                    match self.new_object(type_id, &supplied) {
                        Ok(oid) => {
                            let t = frames.last_mut().unwrap();
                            t.regs[dst] = Some(Value::Ref(oid));
                            t.pc += 1;
                        }
                        Err(c) => return Outcome::Paused(c),
                    }
                }
                Instruction::GetField { dst, object, field } => {
                    let Value::Ref(oid) = read(&frames, object) else {
                        return err("field access on non-reference");
                    };
                    match self.read_field(oid, field) {
                        Ok(v) => {
                            let t = frames.last_mut().unwrap();
                            t.regs[dst] = Some(v);
                            t.pc += 1;
                        }
                        Err(c) => return Outcome::Paused(c),
                    }
                }
                Instruction::Copy { dst, src } => {
                    let v = read(&frames, src);
                    let t = frames.last_mut().unwrap();
                    t.regs[dst] = Some(v);
                    t.pc += 1;
                }
                Instruction::AddI64 { dst, left, right } => {
                    let (Value::I64(a), Value::I64(b)) = (read(&frames, left), read(&frames, right))
                    else {
                        return err(crate::runtime::ERR_ADD_NON_I64);
                    };
                    let t = frames.last_mut().unwrap();
                    t.regs[dst] = Some(Value::I64(a + b));
                    t.pc += 1;
                }
                Instruction::SubI64 { dst, left, right } => {
                    let (Value::I64(a), Value::I64(b)) = (read(&frames, left), read(&frames, right))
                    else {
                        return err(crate::runtime::ERR_SUB_NON_I64);
                    };
                    let t = frames.last_mut().unwrap();
                    t.regs[dst] = Some(Value::I64(a - b));
                    t.pc += 1;
                }
                Instruction::LtI64 { dst, left, right } => {
                    let (Value::I64(a), Value::I64(b)) = (read(&frames, left), read(&frames, right))
                    else {
                        return err(crate::runtime::ERR_LT_NON_I64);
                    };
                    let t = frames.last_mut().unwrap();
                    t.regs[dst] = Some(Value::Bool(a < b));
                    t.pc += 1;
                }
                Instruction::Branch {
                    cond,
                    then_pc,
                    else_pc,
                } => {
                    let Value::Bool(taken) = read(&frames, cond) else {
                        return err(crate::runtime::ERR_BRANCH_NON_BOOL);
                    };
                    frames.last_mut().unwrap().pc = if taken { then_pc } else { else_pc };
                }
                Instruction::Jump { target } => frames.last_mut().unwrap().pc = target,
                Instruction::Yield => frames.last_mut().unwrap().pc += 1,
                Instruction::Emit { value } => {
                    self.emit(read(&frames, value));
                    frames.last_mut().unwrap().pc += 1;
                }
                Instruction::Send { target, value } => {
                    let Value::I64(id) = read(&frames, target) else {
                        return err("send target must be an actor id");
                    };
                    let payload = read(&frames, value);
                    if !self.deliver(id as usize, payload) {
                        return err("send to an unknown actor");
                    }
                    frames.last_mut().unwrap().pc += 1;
                }
                Instruction::Recv { dst, ty } => {
                    // Poll our mailbox, hitting a GC safepoint each spin so a
                    // waiting actor is still parkable and collectable.
                    let message = loop {
                        self.safepoint(tid, &frames);
                        if let Some(v) = mailbox.lock().unwrap().pop_front() {
                            break v;
                        }
                        std::thread::yield_now();
                    };
                    if !self.value_ok(&message, &ty) {
                        return err("received message has the wrong type");
                    }
                    let t = frames.last_mut().unwrap();
                    t.regs[dst] = Some(message);
                    t.pc += 1;
                }
                Instruction::Call {
                    dst,
                    function: callee,
                    args,
                } => {
                    let callee_version = self.world.current_functions[&callee];
                    let callee_regs = match &self.world.functions[&(callee, callee_version)] {
                        FunctionState::Ready(f) => {
                            for (arg, expected) in args.iter().zip(&f.params) {
                                if !self.value_ok(&read(&frames, *arg), expected) {
                                    return err("call argument has the wrong type");
                                }
                            }
                            f.registers
                        }
                        FunctionState::Broken { diagnostics, .. } => {
                            return Outcome::Paused(Condition::BrokenFunction {
                                function: callee,
                                diagnostics: diagnostics.clone(),
                            });
                        }
                    };
                    let mut regs = vec![None; callee_regs];
                    for (slot, arg) in args.iter().enumerate() {
                        regs[slot] = Some(read(&frames, *arg));
                    }
                    frames.last_mut().unwrap().pc += 1;
                    frames.push(MtFrame {
                        func: callee,
                        version: callee_version,
                        pc: 0,
                        regs,
                        return_to: Some(dst),
                    });
                }
                Instruction::Return { value } => {
                    let result = read(&frames, value);
                    if !self.value_ok(&result, &result_ty) {
                        return err("return value has the wrong type");
                    }
                    let done = frames.pop().unwrap();
                    match done.return_to {
                        Some(dst) => frames.last_mut().unwrap().regs[dst] = Some(result),
                        None => return Outcome::Complete(result),
                    }
                }
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
            let Some(cell) = self.objects.lock().unwrap().get(&id).cloned() else {
                continue;
            };
            let body = cell.body.lock().unwrap();
            for v in body.fields.values() {
                if let Value::Ref(child) = v {
                    work.push(*child);
                }
            }
        }
        let mut objects = self.objects.lock().unwrap();
        let before = objects.len();
        objects.retain(|id, _| live.contains(id));
        before - objects.len()
    }
}
