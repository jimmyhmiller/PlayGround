//! cf-runtime: a controllable async runtime.
//!
//! Provides a multi-threaded executor with full visibility and control:
//!  - every task carries metadata (name, source, parent, poll counts)
//!  - the scheduler can run in Auto (work-stealing) or Manual (UI-driven) mode
//!  - all spawn / poll-start / poll-end / wake / complete events are recorded
//!  - the controller can pause workers, step a single poll, or pick which task runs next
//!
//! Public surface is intentionally small. The two crates that consume it are
//! `cf-tokio` (compatibility shim) and `cf-ui` (egui inspector).

pub mod control;
pub mod event;
pub mod hooks;
pub mod resource;
pub mod runtime;
pub mod scheduler;
pub mod task;
pub mod time;

pub use runtime::{current, try_current, Runtime, RuntimeHandle};
pub use task::{JoinHandle, TaskId, TaskMeta, TaskState, WaitReason};
pub use event::{Event, EventKind, EventLog};
pub use control::{Controller, SchedulerMode};
