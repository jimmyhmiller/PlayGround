use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Why a task is currently in `Suspended` state — i.e. what it was waiting
/// on when its last poll returned `Pending`. Set by cf-tokio shim primitives
/// just before returning `Poll::Pending`; the most recently set value
/// "wins" if a single poll touched several primitives. Carries enough
/// information to identify the resource (fd, address, depth) without
/// promising it's still accurate after a wake.
#[derive(Clone, Debug)]
pub enum WaitReason {
    TcpAccept { local_addr: String },
    TcpRead { peer: String },
    TcpWrite { peer: String },
    TcpFlush { peer: String },
    MpscSend { depth: usize, capacity: usize },
    MpscRecv { depth: usize },
    OneshotRecv,
    BroadcastRecv,
    BroadcastSend,
    SemAcquire { permits_requested: usize },
    NotifyWait,
    Sleep { remaining_ms: u64 },
    /// Application code or shim site that hasn't been categorized yet.
    Other(&'static str),
}

impl std::fmt::Display for WaitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaitReason::TcpAccept { local_addr } => write!(f, "TCP accept on {local_addr}"),
            WaitReason::TcpRead { peer } => write!(f, "TCP read from {peer}"),
            WaitReason::TcpWrite { peer } => write!(f, "TCP write to {peer}"),
            WaitReason::TcpFlush { peer } => write!(f, "TCP flush to {peer}"),
            WaitReason::MpscSend { depth, capacity } => {
                write!(f, "mpsc::send (queue {depth}/{capacity} full)")
            }
            WaitReason::MpscRecv { depth } => {
                write!(f, "mpsc::recv (queue depth {depth})")
            }
            WaitReason::OneshotRecv => write!(f, "oneshot::recv"),
            WaitReason::BroadcastRecv => write!(f, "broadcast::recv"),
            WaitReason::BroadcastSend => write!(f, "broadcast::send"),
            WaitReason::SemAcquire { permits_requested } => {
                write!(f, "Semaphore::acquire ({permits_requested} permit(s))")
            }
            WaitReason::NotifyWait => write!(f, "Notify::notified"),
            WaitReason::Sleep { remaining_ms } => write!(f, "sleep ({remaining_ms}ms left)"),
            WaitReason::Other(s) => f.write_str(s),
        }
    }
}

/// Stable identifier for a task. Allocated monotonically; never reused.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TaskId(pub u64);

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

impl TaskId {
    pub fn fresh() -> Self {
        TaskId(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum TaskState {
    /// Newly spawned, never polled.
    Fresh,
    /// Sitting in a worker's run-queue, waiting for a worker to pick it up.
    Runnable,
    /// Currently inside `poll`, executing on a worker thread.
    Running,
    /// Returned `Pending` from its last poll; no waker has fired since.
    Suspended,
    /// Manually paused by the controller; cannot be scheduled until resumed.
    PausedByUser,
    /// Future returned `Ready`; will not be polled again.
    Completed,
    /// Aborted via `JoinHandle::abort` or controller-initiated kill.
    Aborted,
}

impl TaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, TaskState::Completed | TaskState::Aborted)
    }
}

/// Metadata that the runtime keeps for every live task. The future itself is
/// owned by `async-task` machinery; this struct is the inspectable shadow.
#[derive(Debug)]
pub struct TaskMeta {
    pub id: TaskId,
    pub name: String,
    /// Optional file:line where the task was spawned. Captured via `caller!` in
    /// the spawn API; useful for the UI but not load-bearing.
    pub spawn_location: Option<&'static str>,
    /// The task that called `spawn` to create this one, if any. Lets the UI
    /// build a parent/child tree.
    pub parent: Option<TaskId>,
    pub state: TaskState,
    /// Number of times `poll` has returned (Ready or Pending).
    pub poll_count: u64,
    /// Wall-clock instant of the most recent poll start, for the "current"
    /// duration display.
    pub last_poll_started: Option<Instant>,
    /// Cumulative time spent inside `poll`. Updated when each poll returns.
    pub busy_nanos: u128,
    /// Worker that ran the most recent poll, if any.
    pub last_worker: Option<usize>,
    /// True if a wake event has arrived since the last `Pending` return.
    /// Cleared when the task is next polled.
    pub wake_pending: bool,
    pub created_at: Instant,
    /// What the task was waiting on at its last `Pending` return. None
    /// while running, on initial state, or for tasks whose suspending
    /// future doesn't go through an instrumented shim primitive.
    pub wait_reason: Option<WaitReason>,
    /// When the task most recently entered `Suspended` state. Lets the UI
    /// compute "time-since-wake" without scanning the event log.
    pub suspended_at: Option<Instant>,
    /// Compile-time size of the future the task was spawned with — i.e.
    /// the size of the async fn's state machine. Captured once at spawn
    /// via `mem::size_of_val`.
    pub future_size_bytes: usize,
    /// Most recent poll durations (nanoseconds). Bounded ring buffer; the
    /// UI computes p50/p95/p99 over this. Capacity is small on purpose —
    /// 64 samples is enough to characterize "typical poll cost" for a
    /// task without accumulating unbounded memory per long-lived task.
    pub recent_poll_nanos: std::collections::VecDeque<u64>,
}

impl TaskMeta {
    pub fn new(id: TaskId, name: String, parent: Option<TaskId>) -> Self {
        Self {
            id,
            name,
            spawn_location: None,
            parent,
            state: TaskState::Fresh,
            poll_count: 0,
            last_poll_started: None,
            busy_nanos: 0,
            last_worker: None,
            wake_pending: false,
            created_at: Instant::now(),
            wait_reason: None,
            suspended_at: None,
            future_size_bytes: 0,
            recent_poll_nanos: std::collections::VecDeque::with_capacity(64),
        }
    }
}

/// Handle to a spawned task. Mirrors tokio's `JoinHandle` semantics:
/// dropping the handle does NOT cancel the task — it detaches it so it runs
/// to completion in the background. To cancel, call `abort`.
pub struct JoinHandle<T> {
    inner: Option<async_task::Task<T>>,
}

impl<T> JoinHandle<T> {
    pub(crate) fn from_task(t: async_task::Task<T>) -> Self {
        Self { inner: Some(t) }
    }

    /// Cancel the task. The future will be dropped at the next poll
    /// boundary; awaiting the handle after abort returns the cancellation
    /// (which surfaces as a panic from the underlying `Task` future — for
    /// our purposes mini-redis doesn't await on aborted handles, so we
    /// don't need to expose a richer JoinError type yet).
    pub fn abort(mut self) {
        if let Some(t) = self.inner.take() {
            // Dropping cancels.
            drop(t);
        }
    }

    pub fn is_finished(&self) -> bool {
        self.inner
            .as_ref()
            .map(|t| t.is_finished())
            .unwrap_or(true)
    }
}

impl<T> Drop for JoinHandle<T> {
    fn drop(&mut self) {
        if let Some(t) = self.inner.take() {
            t.detach();
        }
    }
}

impl<T> std::future::Future for JoinHandle<T> {
    type Output = T;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        let t = self
            .inner
            .as_mut()
            .expect("JoinHandle polled after abort");
        std::pin::Pin::new(t).poll(cx)
    }
}
