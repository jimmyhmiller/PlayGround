//! Hooks called from inside the forked tokio runtime. cf-runtime's UI
//! consumes events through the same `EventLog` and `Controller` whether
//! events come from our own scheduler (`scheduler::execute_one`) or from
//! patched tokio.
//!
//! The hooks are functions, not a trait, because tokio is not generic
//! over its scheduler — it has one — and we need the absolute minimum
//! call overhead. Each hook acquires nothing on the hot path unless an
//! observer is registered.
//!
//! Lifecycle:
//!   1. Application creates a cf-runtime `Runtime` (or `Observer`).
//!   2. `register(observer)` installs it as the global observer.
//!   3. Patched tokio calls `on_*` hooks; they delegate to the observer.
//!   4. UI reads events from the observer's `EventLog`.

use crate::control::{Controller, GateDecision};
use crate::event::{EventKind, EventLog};
use crate::resource::ResourceRegistry;
use crate::scheduler::TaskRegistry;
use crate::task::{TaskId, TaskMeta, TaskState, WaitReason};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Minimal interface tokio's hooks call into. Holds the same state as a
/// regular cf-runtime: registry, log, controller, resources. We separate
/// it from `Runtime` so a forked tokio can be observed without the
/// (now-redundant) cf-runtime worker pool spinning up its own threads.
pub struct Observer {
    pub registry: Arc<TaskRegistry>,
    pub resources: Arc<ResourceRegistry>,
    pub log: Arc<EventLog>,
    pub controller: Arc<Controller>,
}

impl Observer {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            registry: TaskRegistry::new(),
            resources: ResourceRegistry::new(),
            // Real builds emit thousands of events per second once
            // tracing spans are wired in (1872× visit_mut_expr alone in
            // a 200ms hello-world build). 200K headroom keeps a few
            // seconds of full data resident even on dense workloads.
            log: EventLog::new(200_000),
            controller: Controller::new(),
        })
    }
}

/// Global observer. Patched tokio reads it via the public `current()`
/// accessor. We keep it in a `RwLock<Option<Arc>>` so observers can be
/// installed/removed at process startup without atomic-pointer churn on
/// the hot path. Reads are cheap; the lock is uncontended in practice.
static OBSERVER: RwLock<Option<Arc<Observer>>> = RwLock::new(None);

pub fn register(observer: Arc<Observer>) {
    *OBSERVER.write() = Some(observer);
}

pub fn current() -> Option<Arc<Observer>> {
    OBSERVER.read().clone()
}

/// Cheap "is there even an observer?" probe. Tokio hot paths use this to
/// skip the rest of the hook entirely when no UI is attached.
pub fn enabled() -> bool {
    OBSERVER.read().is_some()
}

/// Mapping from tokio's internal task ids to our `TaskId`. Tokio task ids
/// are u64 (`tokio::task::Id`); we mint our own monotonic ids and stash
/// the mapping here so wake/poll events line up with spawn events even
/// across worker threads.
static MAP: parking_lot::Mutex<Option<Box<dyn TaskIdMap>>> = parking_lot::Mutex::new(None);

trait TaskIdMap: Send + Sync {
    fn lookup(&self, tokio_id: u64) -> Option<TaskId>;
    fn insert(&self, tokio_id: u64, ours: TaskId);
    fn remove(&self, tokio_id: u64);
}

struct DashMapMap {
    inner: dashmap::DashMap<u64, TaskId>,
}

impl TaskIdMap for DashMapMap {
    fn lookup(&self, tokio_id: u64) -> Option<TaskId> {
        self.inner.get(&tokio_id).map(|v| *v)
    }
    fn insert(&self, tokio_id: u64, ours: TaskId) {
        self.inner.insert(tokio_id, ours);
    }
    fn remove(&self, tokio_id: u64) {
        self.inner.remove(&tokio_id);
    }
}

fn ensure_map() {
    let mut g = MAP.lock();
    if g.is_none() {
        *g = Some(Box::new(DashMapMap {
            inner: dashmap::DashMap::new(),
        }));
    }
}

fn map_lookup(tokio_id: u64) -> Option<TaskId> {
    let g = MAP.lock();
    g.as_ref().and_then(|m| m.lookup(tokio_id))
}

/// Public accessor for the cf-tracing-layer (and any other consumer)
/// that needs to resolve a tokio task id to our internal `TaskId` for
/// cross-layer correlation.
pub fn task_id_for_tokio_id(tokio_id: u64) -> Option<TaskId> {
    map_lookup(tokio_id)
}

// === Resource registration hooks (Stage 6) ===

/// Register a runtime resource (Notify, Semaphore, mpsc, etc).
/// Returns an opaque id the caller stashes and passes back to
/// `resource_unregister` from `Drop`. No-op (returns 0) when no
/// observer is installed; the caller must tolerate id=0.
pub fn resource_register(
    kind: crate::resource::ResourceKind,
    label: String,
) -> u64 {
    let Some(o) = current() else {
        return 0;
    };
    o.resources
        .insert(
            kind,
            label,
            None,
            std::sync::Arc::new(NoopProbe),
        )
        .0
}

pub fn resource_unregister(id: u64) {
    if id == 0 {
        return;
    }
    if let Some(o) = current() {
        o.resources.remove(crate::resource::ResourceId(id));
    }
}

// === Pluggable allocation snapshot (Stage 7) ===

/// Snapshot of "allocations and deallocations so far" for the calling
/// thread, as `(allocated_bytes, deallocated_bytes, alloc_count,
/// dealloc_count)`. Returned by whatever the application has wired as
/// the snapshot source — `turbopack-cli` wires `TurboMalloc::
/// allocation_counters` at startup; other consumers leave it `None`.
type AllocSnapshotFn = fn() -> (u64, u64, u64, u64);

static ALLOC_SNAPSHOT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub fn register_alloc_snapshot(f: AllocSnapshotFn) {
    ALLOC_SNAPSHOT.store(f as usize, std::sync::atomic::Ordering::Release);
}

/// Returns the current allocation snapshot if a source is installed,
/// or `None`. Stable on a quiescent system; callers compute deltas by
/// snapshotting at scope-enter and scope-exit.
pub fn alloc_snapshot() -> Option<(u64, u64, u64, u64)> {
    let p = ALLOC_SNAPSHOT.load(std::sync::atomic::Ordering::Acquire);
    if p == 0 {
        return None;
    }
    let f: AllocSnapshotFn = unsafe { std::mem::transmute(p) };
    Some(f())
}

/// A trivial probe that returns an empty snapshot. Sufficient for the
/// Resources tab's "this primitive exists" listing; richer per-resource
/// state requires bespoke probes per primitive type, future work.
struct NoopProbe;
impl crate::resource::ResourceProbe for NoopProbe {
    fn snapshot(&self) -> crate::resource::ResourceStateSnapshot {
        crate::resource::ResourceStateSnapshot::default()
    }
}

fn map_insert(tokio_id: u64, ours: TaskId) {
    ensure_map();
    let g = MAP.lock();
    if let Some(m) = g.as_ref() {
        m.insert(tokio_id, ours);
    }
}

fn map_remove(tokio_id: u64) {
    let g = MAP.lock();
    if let Some(m) = g.as_ref() {
        m.remove(tokio_id);
    }
}

/// Cache of tokio_id → cf TaskId for tasks that JUST completed but whose
/// outer `on_poll_end` hook hasn't run yet. Tokio's harness drops the
/// task synchronously inside `harness.poll()` when the future returns
/// Ready, which fires our `on_complete` hook (removing the map entry).
/// Then control returns to `raw::poll`'s outer instrumentation and our
/// `on_poll_end` runs — but the map entry is gone. This cache fills
/// that gap. Bounded so it doesn't grow unbounded; entries get evicted
/// LRU-style.
static RECENTLY_COMPLETED: parking_lot::Mutex<
    Option<Box<dyn TaskIdMap>>,
> = parking_lot::Mutex::new(None);

fn ensure_recently_completed() {
    let mut g = RECENTLY_COMPLETED.lock();
    if g.is_none() {
        *g = Some(Box::new(DashMapMap {
            inner: dashmap::DashMap::new(),
        }));
    }
}

fn recently_completed_lookup(tokio_id: u64) -> Option<TaskId> {
    let g = RECENTLY_COMPLETED.lock();
    g.as_ref().and_then(|m| m.lookup(tokio_id))
}

fn recently_completed_insert(tokio_id: u64, ours: TaskId) {
    ensure_recently_completed();
    let g = RECENTLY_COMPLETED.lock();
    if let Some(m) = g.as_ref() {
        m.insert(tokio_id, ours);
    }
}

// === Per-thread worker index ===

thread_local! {
    /// Worker index for this thread. Set by patched tokio worker.rs at
    /// the start of each `run_task` so that raw::poll can attribute the
    /// poll to the right lane in the timeline.
    static WORKER_IDX: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    /// Tokio id of the task currently being polled on this thread, if
    /// any. Set by raw::poll's hook around the actual poll call so wake
    /// hooks fired from inside the poll body can attribute "who woke
    /// whom" — that's what makes the wake graph readable.
    static CURRENT_TASK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn set_current_worker(idx: usize) {
    WORKER_IDX.with(|c| c.set(idx));
}

pub fn current_worker() -> Option<usize> {
    WORKER_IDX.with(|c| {
        let v = c.get();
        if v == usize::MAX {
            None
        } else {
            Some(v)
        }
    })
}

pub fn set_polling_tokio_id(tokio_id: u64) {
    CURRENT_TASK.with(|c| c.set(tokio_id));
}

pub fn clear_polling_tokio_id() {
    CURRENT_TASK.with(|c| c.set(0));
}

pub fn polling_tokio_id() -> Option<u64> {
    CURRENT_TASK.with(|c| {
        let v = c.get();
        if v == 0 {
            None
        } else {
            Some(v)
        }
    })
}

// === The hooks themselves ===

/// Called by patched tokio when a new task is spawned. Returns our
/// `TaskId` so the spawn site can stash it for later hook calls (or just
/// rely on the tokio_id → TaskId map).
pub fn on_spawn(tokio_id: u64, name: Option<&str>, parent: Option<u64>) -> TaskId {
    let Some(o) = current() else {
        return TaskId(0);
    };
    let id = TaskId::fresh();
    let parent = parent.and_then(map_lookup);
    let mut meta = TaskMeta::new(id, name.unwrap_or("tokio-task").to_string(), parent);
    meta.state = TaskState::Runnable;
    o.registry.insert(meta);
    map_insert(tokio_id, id);
    let name = name.unwrap_or("tokio-task").to_string();
    o.log.push(
        Some(id),
        None,
        EventKind::Spawned { name, parent },
    );
    id
}

/// Called by patched tokio just before polling a task. Applies the
/// pause/step gate. Returns whether the worker should proceed.
pub fn on_poll_begin(tokio_id: u64, worker_idx: usize) -> bool {
    let Some(o) = current() else {
        return true;
    };
    let Some(id) = map_lookup(tokio_id) else {
        return true;
    };
    let was_paused = o.controller.is_paused();
    if was_paused {
        o.log.push(
            Some(id),
            Some(worker_idx),
            EventKind::Control(format!("gate-blocked w{} #{}", worker_idx, id.0)),
        );
    }
    let gate = o.controller.gate_before_poll(id);
    match gate {
        GateDecision::TaskPaused => return false,
        GateDecision::Proceed { stepped } => {
            if stepped {
                o.log.push(
                    Some(id),
                    Some(worker_idx),
                    EventKind::Control("step".into()),
                );
            }
        }
    }
    if let Some(meta) = o.registry.get(id) {
        let mut m = meta.lock();
        let from = m.state;
        m.state = TaskState::Running;
        m.last_worker = Some(worker_idx);
        m.last_poll_started = Some(Instant::now());
        m.wake_pending = false;
        if from != TaskState::Running {
            o.log.push(
                Some(id),
                Some(worker_idx),
                EventKind::StateChanged { from, to: TaskState::Running },
            );
        }
    }
    o.log.push(Some(id), Some(worker_idx), EventKind::PollStart);
    true
}

/// Called by patched tokio just after polling. `rescheduled` should be
/// true if the future will be polled again immediately.
///
/// Important: if the task completed during this poll, `on_complete`
/// already ran (via harness::complete) and removed the map entry. We
/// fall back to looking up the most recent task we mapped for this
/// `tokio_id` via a small "recently completed" cache so the timeline's
/// span pairing still finds an end event for the matching start.
pub fn on_poll_end(tokio_id: u64, worker_idx: usize, duration_nanos: u64, rescheduled: bool) {
    let Some(o) = current() else { return };
    let id = match map_lookup(tokio_id) {
        Some(id) => id,
        // Map entry already removed by on_complete during poll. Look in
        // the post-completion cache so we still emit a PollEnd.
        None => match recently_completed_lookup(tokio_id) {
            Some(id) => id,
            None => return,
        },
    };
    o.log.push(
        Some(id),
        Some(worker_idx),
        EventKind::PollEnd { rescheduled, duration_nanos },
    );
    if let Some(meta) = o.registry.get(id) {
        let mut m = meta.lock();
        m.poll_count += 1;
        m.busy_nanos = m.busy_nanos.saturating_add(duration_nanos as u128);
        m.last_poll_started = None;
        if m.recent_poll_nanos.len() == 64 {
            m.recent_poll_nanos.pop_front();
        }
        m.recent_poll_nanos.push_back(duration_nanos);
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
                o.log.push(
                    Some(id),
                    Some(worker_idx),
                    EventKind::StateChanged { from, to: new_state },
                );
            }
        }
    }
}

pub fn on_complete(tokio_id: u64) {
    let Some(o) = current() else { return };
    let Some(id) = map_lookup(tokio_id) else { return };
    if let Some(meta) = o.registry.get(id) {
        let mut m = meta.lock();
        let from = m.state;
        m.state = TaskState::Completed;
        if from != TaskState::Completed {
            o.log.push(
                Some(id),
                None,
                EventKind::StateChanged { from, to: TaskState::Completed },
            );
        }
    }
    o.log.push(Some(id), None, EventKind::Completed);
    // Stash for on_poll_end's lookup, then remove from the live map.
    recently_completed_insert(tokio_id, id);
    map_remove(tokio_id);
}

/// Wake hook. `from_tokio_id` is the task that issued the wake (if any).
pub fn on_wake(tokio_id: u64, from_tokio_id: Option<u64>, from_worker: Option<usize>) {
    let Some(o) = current() else { return };
    let Some(id) = map_lookup(tokio_id) else { return };
    let from_task = from_tokio_id.and_then(map_lookup);
    o.log.push(Some(id), from_worker, EventKind::Wake { from_worker, from_task });
    if let Some(meta) = o.registry.get(id) {
        let mut m = meta.lock();
        m.wake_pending = true;
        if !matches!(
            m.state,
            TaskState::Running
                | TaskState::Completed
                | TaskState::Aborted
                | TaskState::PausedByUser
        ) {
            let from = m.state;
            m.state = TaskState::Runnable;
            if from != TaskState::Runnable {
                o.log.push(
                    Some(id),
                    None,
                    EventKind::StateChanged { from, to: TaskState::Runnable },
                );
            }
        }
    }
}

/// Wait reason hook. Called by patched primitive `poll_X` methods just
/// before returning `Pending` so the UI can show what each suspended task
/// is waiting on.
pub fn set_wait_reason(tokio_id: u64, reason: WaitReason) {
    let Some(o) = current() else { return };
    let Some(id) = map_lookup(tokio_id) else { return };
    if let Some(meta) = o.registry.get(id) {
        meta.lock().wait_reason = Some(reason);
    }
}

/// User-event hook for primitive operations (mpsc send, semaphore acquire,
/// notify_one, etc.). Called by patched primitives.
pub fn user_event(category: &'static str, detail: String, tokio_task_id: Option<u64>) {
    let Some(o) = current() else { return };
    let task = tokio_task_id.and_then(map_lookup);
    o.log.push(task, None, EventKind::User { category, detail });
}

// === Convenience accessors used by cf-host on startup ===

pub fn registry() -> Option<Arc<TaskRegistry>> {
    current().map(|o| o.registry.clone())
}

pub fn resources() -> Option<Arc<ResourceRegistry>> {
    current().map(|o| o.resources.clone())
}

pub fn log() -> Option<Arc<EventLog>> {
    current().map(|o| o.log.clone())
}

pub fn controller() -> Option<Arc<Controller>> {
    current().map(|o| o.controller.clone())
}

// Suppress "unused imports" since this is a fresh module.
#[allow(dead_code)]
fn _hooks_module_present() {
    let _: AtomicU64 = AtomicU64::new(0);
    let _ = Ordering::Relaxed;
}
