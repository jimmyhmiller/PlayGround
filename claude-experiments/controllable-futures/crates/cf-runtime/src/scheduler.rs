use crate::control::{Controller, GateDecision, SchedulerMode};
use crate::event::{EventKind, EventLog};
use crate::task::{TaskId, TaskMeta, TaskState};
use async_task::Runnable;
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use parking_lot::{Condvar, Mutex, RwLock};
use slab::Slab;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// What a worker thread is currently doing. Updated by the worker as it
/// loops; readable by the UI for the worker-state lane. Encoded as a u8
/// so we can use lock-free atomics.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum WorkerStatus {
    /// Worker is asleep on the park condvar — no work to do.
    Parked = 0,
    /// Worker has popped a task and is searching for queues.
    Searching = 1,
    /// Worker is blocked at the controller's pause/step gate.
    GateBlocked = 2,
    /// Worker is currently inside `Runnable::run()` — i.e. polling a task.
    Running = 3,
    /// Worker thread has exited.
    Exited = 4,
}

impl WorkerStatus {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Parked,
            1 => Self::Searching,
            2 => Self::GateBlocked,
            3 => Self::Running,
            _ => Self::Exited,
        }
    }
}

#[derive(Default)]
pub struct WorkerStatusBoard {
    slots: Vec<AtomicU8>,
}

impl WorkerStatusBoard {
    pub fn new(n: usize) -> Self {
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(AtomicU8::new(WorkerStatus::Parked as u8));
        }
        Self { slots }
    }

    pub fn set(&self, idx: usize, s: WorkerStatus) {
        if idx < self.slots.len() {
            self.slots[idx].store(s as u8, Ordering::Relaxed);
        }
    }

    pub fn get(&self, idx: usize) -> WorkerStatus {
        if idx < self.slots.len() {
            WorkerStatus::from_u8(self.slots[idx].load(Ordering::Relaxed))
        } else {
            WorkerStatus::Exited
        }
    }

    pub fn snapshot(&self) -> Vec<WorkerStatus> {
        self.slots
            .iter()
            .map(|s| WorkerStatus::from_u8(s.load(Ordering::Relaxed)))
            .collect()
    }
}

/// A scheduled unit of work. The `Runnable` is the `async-task` handle that
/// will drive a single `poll` of the underlying future when invoked. We carry
/// the `TaskId` alongside so we can apply gates / metadata updates without
/// downcasting through the Runnable.
pub struct ScheduledTask {
    pub id: TaskId,
    pub runnable: Runnable,
}

/// Storage for live task metadata. Keyed by `TaskId` — we use a HashMap rather
/// than slab so that ids stay stable across removals (UI users care about
/// stable ids more than tight indexing).
pub struct TaskRegistry {
    inner: RwLock<HashMap<TaskId, Arc<Mutex<TaskMeta>>>>,
    /// Insertion-order ids, for the UI list. Append-only; no removals so the
    /// UI sees completed tasks too. `inner` may have entries removed if we
    /// later add eviction; for now nothing removes.
    order: RwLock<Vec<TaskId>>,
}

impl TaskRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(HashMap::new()),
            order: RwLock::new(Vec::new()),
        })
    }

    pub fn insert(&self, meta: TaskMeta) -> Arc<Mutex<TaskMeta>> {
        let id = meta.id;
        let entry = Arc::new(Mutex::new(meta));
        self.inner.write().insert(id, entry.clone());
        self.order.write().push(id);
        entry
    }

    pub fn get(&self, id: TaskId) -> Option<Arc<Mutex<TaskMeta>>> {
        self.inner.read().get(&id).cloned()
    }

    pub fn ids(&self) -> Vec<TaskId> {
        self.order.read().clone()
    }

    pub fn snapshot(&self) -> Vec<TaskMetaSnapshot> {
        let order = self.order.read().clone();
        let map = self.inner.read();
        order
            .into_iter()
            .filter_map(|id| {
                let meta = map.get(&id)?.lock();
                Some(TaskMetaSnapshot::from(&*meta))
            })
            .collect()
    }
}

/// Cheap, owned snapshot of `TaskMeta` for the UI thread to render without
/// holding any locks.
#[derive(Clone, Debug)]
pub struct TaskMetaSnapshot {
    pub id: TaskId,
    pub name: String,
    pub parent: Option<TaskId>,
    pub state: TaskState,
    pub poll_count: u64,
    pub busy_nanos: u128,
    pub last_worker: Option<usize>,
    pub wake_pending: bool,
    pub age_nanos: u128,
    pub wait_reason: Option<crate::task::WaitReason>,
    /// Nanoseconds since the task last entered Suspended. `None` if the
    /// task is in any other state. Useful for "stuck task" detection.
    pub suspended_for_nanos: Option<u128>,
    pub future_size_bytes: usize,
    /// Recent poll durations in nanoseconds. UI uses these for p50/p95/p99.
    pub recent_poll_nanos: Vec<u64>,
}

impl From<&TaskMeta> for TaskMetaSnapshot {
    fn from(m: &TaskMeta) -> Self {
        Self {
            id: m.id,
            name: m.name.clone(),
            parent: m.parent,
            state: m.state,
            poll_count: m.poll_count,
            busy_nanos: m.busy_nanos,
            last_worker: m.last_worker,
            wake_pending: m.wake_pending,
            age_nanos: m.created_at.elapsed().as_nanos(),
            wait_reason: m.wait_reason.clone(),
            suspended_for_nanos: m.suspended_at.map(|t| t.elapsed().as_nanos()),
            future_size_bytes: m.future_size_bytes,
            recent_poll_nanos: m.recent_poll_nanos.iter().copied().collect(),
        }
    }
}

/// Multi-producer, multi-consumer scheduler. Each worker owns a deque, plus
/// there's a global injector for tasks that are scheduled from non-worker
/// threads (the I/O reactor, the UI's manual-run button, etc.). Stealing is
/// only attempted in Auto mode.
pub struct Scheduler {
    /// Default work queue. Used in Auto mode for any task that doesn't have
    /// a specific worker affinity.
    injector: Injector<ScheduledTask>,
    /// Tasks the controller has explicitly released from the manual queue.
    /// Workers ALWAYS check this queue, even in Manual mode — that's how
    /// "click run" actually causes the task to run when no other queue is
    /// being consulted.
    released: Injector<ScheduledTask>,
    /// One per worker. Held in an Arc so the worker thread can pop locally
    /// while other workers steal from it.
    workers: Vec<WorkerHandle>,
    controller: Arc<Controller>,
    registry: Arc<TaskRegistry>,
    log: Arc<EventLog>,
    /// Pending manual queue: in `Manual` mode the injector is bypassed, and
    /// scheduled runnables wait here until the controller pops one. We use
    /// a separate slab so the UI can list everything that *would* run next.
    manual_pending: Mutex<Slab<ScheduledTask>>,
    /// Condvar to wake idle workers when new work arrives.
    park: (Mutex<()>, Condvar),
    /// Live per-worker status, updated by the worker loop.
    pub worker_status: WorkerStatusBoard,
}

struct WorkerHandle {
    /// Stealer half of the worker's local deque. Held by the scheduler so
    /// other workers can steal from it.
    stealer: Stealer<ScheduledTask>,
    /// Local push handle. Held in a Mutex because we need to push from the
    /// scheduling closure (any thread), not just the owning worker. A real
    /// implementation would prefer per-worker MPSC, but we expect very low
    /// contention here in practice.
    local: Mutex<Option<Worker<ScheduledTask>>>,
}

impl Scheduler {
    pub fn new(
        n_workers: usize,
        controller: Arc<Controller>,
        registry: Arc<TaskRegistry>,
        log: Arc<EventLog>,
    ) -> Arc<Self> {
        let mut workers = Vec::with_capacity(n_workers);
        let mut local_owners = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let w = Worker::new_fifo();
            let s = w.stealer();
            local_owners.push(w);
            workers.push(WorkerHandle {
                stealer: s,
                local: Mutex::new(None),
            });
        }
        // Hand the local Worker halves out under their slot's mutex so worker
        // threads can `take()` them on startup.
        for (slot, w) in workers.iter().zip(local_owners.into_iter()) {
            *slot.local.lock() = Some(w);
        }
        Arc::new(Self {
            injector: Injector::new(),
            released: Injector::new(),
            workers,
            controller,
            registry,
            log,
            manual_pending: Mutex::new(Slab::new()),
            park: (Mutex::new(()), Condvar::new()),
            worker_status: WorkerStatusBoard::new(n_workers),
        })
    }

    pub fn n_workers(&self) -> usize {
        self.workers.len()
    }

    /// Claim a worker's local Worker half. Called once at worker thread
    /// startup. Panics if called twice for the same index.
    pub fn take_worker_local(&self, idx: usize) -> Worker<ScheduledTask> {
        self.workers[idx]
            .local
            .lock()
            .take()
            .expect("worker local already claimed")
    }

    /// Schedule a runnable. Decides which queue based on mode. Notifies one
    /// parked worker so it can pick the task up.
    pub fn schedule(&self, task: ScheduledTask) {
        let mode = self.controller.mode();
        match mode {
            SchedulerMode::Auto => {
                self.injector.push(task);
                self.park.1.notify_one();
            }
            SchedulerMode::Manual => {
                self.manual_pending.lock().insert(task);
                // Don't notify workers — in manual mode, only `manual_run`
                // releases tasks.
            }
        }
        // Update task state to Runnable. Done after queueing so the UI never
        // sees "Runnable" for a task that isn't actually queued.
        // Note: we look up in the registry; if metadata was already taken
        // (terminal state), this is a no-op.
        // Skipped here because we don't know the task id without unpacking.
    }

    /// Worker poll loop primitive. Returns the next task to run, blocking the
    /// thread (parking) until either work is available or `should_exit`
    /// returns true.
    ///
    /// Queue order:
    ///   1. `released` (always — that's what makes "click run" in the UI's
    ///      manual queue actually trigger work, even when mode == Manual)
    ///   2. local deque (Auto only)
    ///   3. global injector (Auto only)
    ///   4. steal from peers (Auto only)
    pub fn next_for_worker(
        &self,
        idx: usize,
        local: &Worker<ScheduledTask>,
        should_exit: &impl Fn() -> bool,
    ) -> Option<ScheduledTask> {
        loop {
            if should_exit() {
                self.worker_status.set(idx, WorkerStatus::Exited);
                return None;
            }
            self.worker_status.set(idx, WorkerStatus::Searching);
            // 1. Released queue — checked in every mode.
            loop {
                match self.released.steal_batch_and_pop(local) {
                    Steal::Success(t) => return Some(t),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
            if matches!(self.controller.mode(), SchedulerMode::Auto) {
                // 2. local
                if let Some(t) = local.pop() {
                    return Some(t);
                }
                // 3. global injector
                loop {
                    match self.injector.steal_batch_and_pop(local) {
                        Steal::Success(t) => return Some(t),
                        Steal::Retry => continue,
                        Steal::Empty => break,
                    }
                }
                // 4. steal from peers
                for (i, peer) in self.workers.iter().enumerate() {
                    if i == idx {
                        continue;
                    }
                    loop {
                        match peer.stealer.steal_batch_and_pop(local) {
                            Steal::Success(t) => return Some(t),
                            Steal::Retry => continue,
                            Steal::Empty => break,
                        }
                    }
                }
            }
            // No work. Park until notified.
            let (m, cv) = &self.park;
            let mut g = m.lock();
            self.worker_status.set(idx, WorkerStatus::Parked);
            // Re-check before parking to avoid lost wakeup: in Manual mode,
            // released_queue is the only queue we care about; in Auto mode,
            // the injector is the canonical "is there anything to do" probe.
            if !self.released.is_empty() {
                continue;
            }
            if matches!(self.controller.mode(), SchedulerMode::Auto)
                && !self.injector.is_empty()
            {
                continue;
            }
            cv.wait(&mut g);
        }
    }

    /// Wake all parked workers. Call when shutting down so they observe the
    /// `should_exit` flag.
    pub fn unpark_all(&self) {
        self.park.1.notify_all();
    }

    /// Inspect the manual-mode pending queue. Returns (slab key, task id) so
    /// the UI can render the queue and request a specific task to run.
    pub fn manual_queue(&self) -> Vec<(usize, TaskId)> {
        self.manual_pending
            .lock()
            .iter()
            .map(|(k, t)| (k, t.id))
            .collect()
    }

    /// Pop a specific entry from the manual-mode pending list and put it
    /// on the `released` queue so a worker picks it up regardless of mode.
    /// Returns false if the slab key is no longer valid.
    pub fn manual_run(&self, slab_key: usize) -> bool {
        let task = match self.manual_pending.lock().try_remove(slab_key) {
            Some(t) => t,
            None => return false,
        };
        self.log.push(
            Some(task.id),
            None,
            crate::event::EventKind::Control(format!("manual-run #{}", task.id.0)),
        );
        self.released.push(task);
        self.park.1.notify_one();
        true
    }

    /// Move every currently-pending non-released task into manual_pending,
    /// and put every task already in the global injector there too. Tasks
    /// already in worker locals can't be migrated without crossing thread
    /// boundaries — they'll drain naturally if we then switch back to Auto,
    /// or wait inert if we stay in Manual. Called when the controller flips
    /// Auto → Manual so the UI immediately reflects the actual queued work.
    pub fn migrate_to_manual(&self) {
        loop {
            match self.injector.steal() {
                Steal::Success(t) => {
                    self.manual_pending.lock().insert(t);
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
    }

    pub fn registry(&self) -> &Arc<TaskRegistry> {
        &self.registry
    }

    pub fn controller(&self) -> &Arc<Controller> {
        &self.controller
    }

    pub fn log(&self) -> &Arc<EventLog> {
        &self.log
    }
}

/// Run a single poll of `task`, applying the controller gate, updating
/// metadata, and recording events. The Runnable is consumed; the
/// `async-task` machinery decides whether to reschedule based on the future's
/// return value (it'll call our scheduling closure again on wake).
pub fn execute_one(scheduler: &Scheduler, worker_idx: usize, task: ScheduledTask) {
    let id = task.id;
    let registry = scheduler.registry().clone();
    let log = scheduler.log().clone();
    let controller = scheduler.controller();

    // Apply the gate. If task is user-paused, requeue it under a still-paused
    // marker. We don't actually run it — the scheduler will poll it again
    // once the user resumes.
    let was_paused_before_gate = controller.is_paused();
    if was_paused_before_gate {
        log.push(
            Some(id),
            Some(worker_idx),
            EventKind::Control(format!(
                "gate-blocked w{} #{}",
                worker_idx, id.0
            )),
        );
        scheduler.worker_status.set(worker_idx, WorkerStatus::GateBlocked);
    }
    let gate = controller.gate_before_poll(id);
    match gate {
        GateDecision::TaskPaused => {
            // Drop the runnable without running. async-task will keep the
            // future alive; when the user resumes the task, they need a way
            // to re-trigger scheduling. The simplest mechanism: the
            // controller's resume_task pokes the wake fn manually.
            //
            // For now, requeue immediately so the worker doesn't busy-loop.
            // The user resume logic in `Controller::resume_task` is expected
            // to clear the pause first; if we got here it's a transient race
            // and we'll lose a tick.
            //
            // To avoid the busy-loop in pathological cases we re-push to the
            // injector — workers will visit other tasks first.
            scheduler.injector.push(task);
            scheduler.park.1.notify_one();
            return;
        }
        GateDecision::Proceed { stepped } => {
            if stepped {
                log.push(
                    Some(id),
                    Some(worker_idx),
                    EventKind::Control("step".into()),
                );
            }
        }
    }

    // Update metadata to Running.
    if let Some(meta) = registry.get(id) {
        let mut m = meta.lock();
        let from = m.state;
        m.state = TaskState::Running;
        m.last_worker = Some(worker_idx);
        m.last_poll_started = Some(Instant::now());
        m.wake_pending = false;
        if from != TaskState::Running {
            log.push(
                Some(id),
                Some(worker_idx),
                EventKind::StateChanged {
                    from,
                    to: TaskState::Running,
                },
            );
        }
    }
    log.push(Some(id), Some(worker_idx), EventKind::PollStart);

    // Run the actual poll. `run()` returns `true` if the task was woken
    // during poll (i.e. will be rescheduled immediately), not whether it
    // returned Ready. The wrapped future emits Completed itself.
    scheduler.worker_status.set(worker_idx, WorkerStatus::Running);
    let started = Instant::now();
    let rescheduled = task.runnable.run();
    let elapsed = started.elapsed();
    scheduler.worker_status.set(worker_idx, WorkerStatus::Searching);

    log.push(
        Some(id),
        Some(worker_idx),
        EventKind::PollEnd {
            rescheduled,
            duration_nanos: elapsed.as_nanos() as u64,
        },
    );

    if let Some(meta) = registry.get(id) {
        let mut m = meta.lock();
        m.poll_count += 1;
        m.busy_nanos = m.busy_nanos.saturating_add(elapsed.as_nanos());
        m.last_poll_started = None;
        if m.recent_poll_nanos.len() == 64 {
            m.recent_poll_nanos.pop_front();
        }
        m.recent_poll_nanos.push_back(elapsed.as_nanos() as u64);
        // If the wrapped future already transitioned us to Completed inside
        // the poll, leave that alone. Otherwise: rescheduled→Runnable,
        // not-rescheduled→Suspended.
        if !matches!(m.state, TaskState::Completed | TaskState::Aborted) {
            let new_state = if rescheduled {
                TaskState::Runnable
            } else {
                TaskState::Suspended
            };
            if m.state != new_state {
                let from = m.state;
                m.state = new_state;
                if matches!(new_state, TaskState::Suspended) {
                    m.suspended_at = Some(Instant::now());
                } else {
                    m.suspended_at = None;
                }
                log.push(
                    Some(id),
                    Some(worker_idx),
                    EventKind::StateChanged {
                        from,
                        to: new_state,
                    },
                );
            }
        }
    }
}
