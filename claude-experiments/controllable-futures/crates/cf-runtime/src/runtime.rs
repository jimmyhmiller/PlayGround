use crate::control::Controller;
use crate::event::{EventKind, EventLog};
use crate::resource::ResourceRegistry;
use crate::scheduler::{execute_one, ScheduledTask, Scheduler, TaskRegistry};
use crate::task::{JoinHandle, TaskId, TaskMeta, TaskState, WaitReason};
use parking_lot::Mutex;
use std::cell::RefCell;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle as ThreadHandle};

/// A handle that crosses thread boundaries cheaply. The `Runtime` itself owns
/// the worker threads; a `RuntimeHandle` is just the shared state needed to
/// spawn into it from anywhere.
#[derive(Clone)]
pub struct RuntimeHandle {
    inner: Arc<RuntimeInner>,
}

pub(crate) struct RuntimeInner {
    pub scheduler: Arc<Scheduler>,
    pub registry: Arc<TaskRegistry>,
    pub resources: Arc<ResourceRegistry>,
    pub controller: Arc<Controller>,
    pub log: Arc<EventLog>,
    pub shutdown: AtomicBool,
}

thread_local! {
    /// The handle of the runtime owning the current thread, if any. Set by
    /// worker threads during startup; cleared on exit. Used by the tokio
    /// shim's `spawn` to find "the" runtime.
    static CURRENT: RefCell<Option<RuntimeHandle>> = const { RefCell::new(None) };
    /// The id of the task currently being polled on this thread, if any. Set
    /// by `execute_one` (via the scheduler's run loop). Drives parent linkage
    /// for nested spawns.
    static CURRENT_TASK: RefCell<Option<TaskId>> = const { RefCell::new(None) };
    /// The worker index of this thread, if it's a runtime worker.
    static WORKER_IDX: RefCell<Option<usize>> = const { RefCell::new(None) };
}

/// Returns the runtime handle for the current thread, panicking if there is
/// none. Mirrors `tokio::runtime::Handle::current()`.
pub fn current() -> RuntimeHandle {
    try_current().expect("called outside of a cf-runtime context")
}

pub fn try_current() -> Option<RuntimeHandle> {
    CURRENT.with(|c| c.borrow().clone())
}

pub fn current_task() -> Option<TaskId> {
    CURRENT_TASK.with(|c| *c.borrow())
}

pub struct Runtime {
    handle: RuntimeHandle,
    workers: Mutex<Vec<ThreadHandle<()>>>,
}

impl Runtime {
    /// Build a runtime with `n_workers` worker threads. The runtime starts in
    /// `Auto` scheduler mode and is not paused.
    pub fn new(n_workers: usize) -> Self {
        assert!(n_workers >= 1, "need at least one worker");
        let controller = Controller::new();
        let registry = TaskRegistry::new();
        let resources = ResourceRegistry::new();
        let log = EventLog::new(8192);
        let scheduler = Scheduler::new(
            n_workers,
            controller.clone(),
            registry.clone(),
            log.clone(),
        );
        let inner = Arc::new(RuntimeInner {
            scheduler,
            registry,
            resources,
            controller,
            log,
            shutdown: AtomicBool::new(false),
        });
        let handle = RuntimeHandle { inner };

        // Spawn worker threads. Each worker claims its local deque and loops
        // through the scheduler.
        let mut threads = Vec::with_capacity(n_workers);
        for idx in 0..n_workers {
            let h = handle.clone();
            let t = thread::Builder::new()
                .name(format!("cf-worker-{idx}"))
                .spawn(move || worker_main(idx, h))
                .expect("spawn worker");
            threads.push(t);
        }
        Self {
            handle,
            workers: Mutex::new(threads),
        }
    }

    pub fn handle(&self) -> RuntimeHandle {
        self.handle.clone()
    }

    /// Spawn a future. The caller can use the returned `JoinHandle` to await
    /// completion or abort.
    pub fn spawn<F>(&self, name: impl Into<String>, fut: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.handle.spawn(name, fut)
    }

    /// Run the future to completion on the current thread, parking it on a
    /// condvar between wakes. The future itself is *not* spawned onto the
    /// worker pool — block_on is single-threaded by design — but futures it
    /// `spawn`s are scheduled on the workers.
    pub fn block_on<F: Future>(&self, fut: F) -> F::Output {
        // Make sure we have CURRENT set on the calling thread so that
        // spawn-from-future works.
        let _guard = enter(self.handle.clone());
        futures_lite::future::block_on(fut)
    }

    /// Stop all worker threads. Tasks currently in queues are dropped.
    pub fn shutdown(self) {
        self.handle.inner.shutdown.store(true, Ordering::Release);
        self.handle.inner.scheduler.unpark_all();
        let threads = std::mem::take(&mut *self.workers.lock());
        for t in threads {
            let _ = t.join();
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.handle.inner.shutdown.store(true, Ordering::Release);
        self.handle.inner.scheduler.unpark_all();
        let threads = std::mem::take(&mut *self.workers.lock());
        for t in threads {
            let _ = t.join();
        }
    }
}

impl RuntimeHandle {
    pub fn spawn<F>(&self, name: impl Into<String>, fut: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let id = TaskId::fresh();
        let parent = current_task();
        let mut meta = TaskMeta::new(id, name.into(), parent);
        meta.state = TaskState::Runnable;
        meta.future_size_bytes = std::mem::size_of_val(&fut);
        let name_for_log = meta.name.clone();
        self.inner.registry.insert(meta);
        self.inner.log.push(
            Some(id),
            None,
            EventKind::Spawned {
                name: name_for_log,
                parent,
            },
        );

        // Wrap the user's future so we observe the moment it returns Ready.
        // This is the only reliable hook for "completed" state because
        // async-task's `Runnable::run()` returns whether the task was woken
        // during run, not whether it completed.
        let registry_done = self.inner.registry.clone();
        let log_done = self.inner.log.clone();
        let wrapped = async move {
            let r = fut.await;
            if let Some(meta) = registry_done.get(id) {
                let mut m = meta.lock();
                let from = m.state;
                m.state = TaskState::Completed;
                if from != TaskState::Completed {
                    log_done.push(
                        Some(id),
                        None,
                        EventKind::StateChanged {
                            from,
                            to: TaskState::Completed,
                        },
                    );
                }
            }
            log_done.push(Some(id), None, EventKind::Completed);
            r
        };

        let scheduler = self.inner.scheduler.clone();
        let registry = self.inner.registry.clone();
        let log = self.inner.log.clone();
        let controller = self.inner.controller.clone();
        let schedule = move |runnable: async_task::Runnable| {
            // Look up current state to decide whether this is the initial
            // schedule or a wake. Initial schedule happens while the task is
            // Fresh; everything after is a wake.
            let is_initial = registry
                .get(id)
                .map(|m| matches!(m.lock().state, TaskState::Fresh))
                .unwrap_or(false);
            if !is_initial {
                log.push(
                    Some(id),
                    worker_idx_now(),
                    EventKind::Wake {
                        from_worker: worker_idx_now(),
                        from_task: current_task(),
                    },
                );
            }
            if let Some(meta) = registry.get(id) {
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
                        log.push(
                            Some(id),
                            None,
                            EventKind::StateChanged {
                                from,
                                to: TaskState::Runnable,
                            },
                        );
                    }
                }
            }
            // Honor task-level pause: if the user paused this task, refuse to
            // schedule. The Runnable is dropped — but `async-task` retains
            // the future, and a subsequent wake (after resume) will create a
            // fresh Runnable.
            if controller.is_task_paused(id) {
                return;
            }
            scheduler.schedule(ScheduledTask { id, runnable });
        };

        let (runnable, task) = async_task::spawn(wrapped, schedule);
        runnable.schedule();
        JoinHandle::from_task(task)
    }

    /// Post a free-form named event into the runtime log. The event is
    /// associated with the currently-running task (if any) so the timeline
    /// can render it alongside that task's lane. Cheap; intended for use by
    /// the cf-tokio shim's sync primitives and by application code that
    /// wants annotations.
    pub fn log_user_event(&self, category: &'static str, detail: impl Into<String>) {
        self.inner.log.push(
            current_task(),
            worker_idx_now(),
            EventKind::User {
                category,
                detail: detail.into(),
            },
        );
    }

    /// Record what the current task is about to start waiting on. Called
    /// from cf-tokio shim primitives just before they return `Pending`.
    /// Cheap (one mutex acquire on the per-task meta). No-op if not in a
    /// task context.
    pub fn set_wait_reason(&self, reason: WaitReason) {
        let Some(id) = current_task() else { return };
        let Some(meta) = self.inner.registry.get(id) else {
            return;
        };
        meta.lock().wait_reason = Some(reason);
    }

    pub fn clear_wait_reason(&self) {
        let Some(id) = current_task() else { return };
        let Some(meta) = self.inner.registry.get(id) else {
            return;
        };
        meta.lock().wait_reason = None;
    }

    pub fn registry(&self) -> Arc<TaskRegistry> {
        self.inner.registry.clone()
    }

    pub fn resources(&self) -> Arc<ResourceRegistry> {
        self.inner.resources.clone()
    }

    pub fn controller(&self) -> Arc<Controller> {
        self.inner.controller.clone()
    }

    pub fn log(&self) -> Arc<EventLog> {
        self.inner.log.clone()
    }

    pub fn scheduler(&self) -> Arc<Scheduler> {
        self.inner.scheduler.clone()
    }
}

fn worker_idx_now() -> Option<usize> {
    WORKER_IDX.with(|c| *c.borrow())
}

/// RAII guard that installs a runtime handle on the current thread and
/// removes it on drop.
pub struct EnterGuard {
    prev: Option<RuntimeHandle>,
}

impl Drop for EnterGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.prev.take());
    }
}

pub fn enter(handle: RuntimeHandle) -> EnterGuard {
    let prev = CURRENT.with(|c| c.replace(Some(handle)));
    EnterGuard { prev }
}

fn worker_main(idx: usize, handle: RuntimeHandle) {
    CURRENT.with(|c| *c.borrow_mut() = Some(handle.clone()));
    WORKER_IDX.with(|c| *c.borrow_mut() = Some(idx));
    let local = handle.inner.scheduler.take_worker_local(idx);
    let scheduler = handle.inner.scheduler.clone();
    let should_exit = || handle.inner.shutdown.load(Ordering::Acquire);
    while let Some(task) = scheduler.next_for_worker(idx, &local, &should_exit) {
        if should_exit() {
            break;
        }
        let id = task.id;
        // Track the running task so spawn-from-future can record a parent.
        CURRENT_TASK.with(|c| *c.borrow_mut() = Some(id));
        execute_one(&scheduler, idx, task);
        CURRENT_TASK.with(|c| *c.borrow_mut() = None);
    }
    CURRENT.with(|c| c.borrow_mut().take());
    WORKER_IDX.with(|c| c.borrow_mut().take());
}
