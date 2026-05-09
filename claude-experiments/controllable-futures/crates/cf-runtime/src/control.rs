use crate::task::TaskId;
use parking_lot::{Condvar, Mutex};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SchedulerMode {
    /// Workers freely pull tasks from their local queue / steal from peers.
    Auto,
    /// Workers idle. Tasks only run when the controller explicitly enqueues
    /// them via `Controller::manual_run`.
    Manual,
}

/// Shared control plane. The runtime workers and the UI both hold an `Arc`.
/// All fields are guarded by a single mutex on purpose: this is a low-traffic
/// coordination point, not a hot path, and keeping it simple avoids subtle
/// races between (mode, paused, step_budget).
pub struct Controller {
    inner: Mutex<ControllerInner>,
    cv: Condvar,
}

struct ControllerInner {
    mode: SchedulerMode,
    /// When `paused` is true, workers block before each poll. The condvar
    /// `cv` is signaled when the controller resumes them or grants a step.
    paused: bool,
    /// Each step grant lets exactly one worker proceed past the gate, then
    /// auto-pauses the runtime again. Decremented as workers consume them.
    step_budget: u32,
    /// Tasks the user has explicitly paused. The scheduler refuses to run
    /// them even when they're otherwise runnable.
    paused_tasks: HashSet<TaskId>,
    /// Break before the next poll of any of these tasks. The runtime sets
    /// `paused = true` and clears the entry once hit.
    poll_breakpoints: HashSet<TaskId>,
}

impl Controller {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(ControllerInner {
                mode: SchedulerMode::Auto,
                paused: false,
                step_budget: 0,
                paused_tasks: HashSet::new(),
                poll_breakpoints: HashSet::new(),
            }),
            cv: Condvar::new(),
        })
    }

    pub fn mode(&self) -> SchedulerMode {
        self.inner.lock().mode
    }

    pub fn set_mode(&self, mode: SchedulerMode) {
        let mut g = self.inner.lock();
        g.mode = mode;
        // Switching mode wakes workers in case they were blocked waiting on
        // the gate; they'll re-evaluate.
        self.cv.notify_all();
    }

    pub fn pause(&self) {
        let mut g = self.inner.lock();
        g.paused = true;
    }

    pub fn resume(&self) {
        let mut g = self.inner.lock();
        g.paused = false;
        self.cv.notify_all();
    }

    pub fn is_paused(&self) -> bool {
        self.inner.lock().paused
    }

    /// Allow `n` more polls to occur, then auto-pause again. If the runtime
    /// was already paused, this releases up to `n` workers; if not, it has
    /// the side-effect of pausing.
    pub fn step(&self, n: u32) {
        let mut g = self.inner.lock();
        g.paused = true;
        g.step_budget = g.step_budget.saturating_add(n);
        self.cv.notify_all();
    }

    pub fn pause_task(&self, id: TaskId) {
        self.inner.lock().paused_tasks.insert(id);
    }

    pub fn resume_task(&self, id: TaskId) {
        self.inner.lock().paused_tasks.remove(&id);
        self.cv.notify_all();
    }

    pub fn is_task_paused(&self, id: TaskId) -> bool {
        self.inner.lock().paused_tasks.contains(&id)
    }

    pub fn add_breakpoint(&self, id: TaskId) {
        self.inner.lock().poll_breakpoints.insert(id);
    }

    pub fn remove_breakpoint(&self, id: TaskId) {
        self.inner.lock().poll_breakpoints.remove(&id);
    }

    /// Called by a worker just before polling. If the runtime is paused, this
    /// blocks until either:
    ///   - the controller resumes (returns Ok)
    ///   - a step grant is available for this worker to consume (returns Ok)
    ///   - this specific task is paused (returns Err — caller should put the
    ///     task back into its run-queue and try a different one)
    /// Returns whether this poll consumed a step grant.
    pub fn gate_before_poll(&self, task: TaskId) -> GateDecision {
        let mut g = self.inner.lock();
        if g.paused_tasks.contains(&task) {
            return GateDecision::TaskPaused;
        }
        if g.poll_breakpoints.remove(&task) {
            // Hit a breakpoint: enter paused state and surface the event.
            // The worker reports it via the event bus.
            g.paused = true;
        }
        loop {
            if !g.paused {
                return GateDecision::Proceed { stepped: false };
            }
            if g.step_budget > 0 {
                g.step_budget -= 1;
                return GateDecision::Proceed { stepped: true };
            }
            self.cv.wait(&mut g);
            // Re-check paused_tasks: the user may have paused this task while
            // we were blocked waiting at the gate.
            if g.paused_tasks.contains(&task) {
                return GateDecision::TaskPaused;
            }
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum GateDecision {
    /// Worker may run this poll. `stepped` indicates whether a step grant was
    /// consumed (purely informational, for event log labeling).
    Proceed { stepped: bool },
    /// This task is user-paused. Worker should requeue it (if applicable) and
    /// pick a different task.
    TaskPaused,
}
