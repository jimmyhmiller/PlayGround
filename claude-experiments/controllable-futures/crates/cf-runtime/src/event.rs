use crate::task::{TaskId, TaskState, WaitReason};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use std::time::SystemTime;

/// Optional anchor: if set, recorded events stamp themselves both with
/// `Instant` (for live computation) and `SystemTime` (for serialization).
/// Set once at runtime startup so every recorded event maps to a stable
/// timestamp on disk.
#[allow(dead_code)]
fn epoch_offset(at: Instant, anchor_instant: Instant, anchor_sys: SystemTime) -> SystemTime {
    if at >= anchor_instant {
        anchor_sys + (at - anchor_instant)
    } else {
        anchor_sys - (anchor_instant - at)
    }
}

/// Lightweight serializable event for trace export. Mirrors `Event`
/// but uses `SystemTime` (or u64 ns from a fixed epoch) so the trace
/// can be reopened in a different process.
pub mod export {
    use super::*;
    use std::io::{self, Write};

    /// Serialize an event log slice as JSONL (one JSON object per
    /// line) to the given writer. Caller picks the writer (file,
    /// gzip, etc).
    pub fn write_jsonl<W: Write>(events: &[Event], anchor_instant: Instant, anchor_sys: SystemTime, w: &mut W) -> io::Result<()> {
        for e in events {
            let ts_ns = e
                .at
                .saturating_duration_since(anchor_instant)
                .as_nanos() as u64;
            let ts_sys = anchor_sys + std::time::Duration::from_nanos(ts_ns);
            let unix_ns = ts_sys
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            // Manual JSON construction — avoids serde dep on the
            // critical event type and keeps the format stable for
            // anyone who wants to parse outside Rust.
            let mut line = String::new();
            line.push_str("{\"seq\":");
            line.push_str(&e.seq.to_string());
            line.push_str(",\"ts_ns\":");
            line.push_str(&unix_ns.to_string());
            if let Some(t) = e.task {
                line.push_str(",\"task\":");
                line.push_str(&t.0.to_string());
            }
            if let Some(wid) = e.worker {
                line.push_str(",\"worker\":");
                line.push_str(&wid.to_string());
            }
            line.push_str(",\"kind\":");
            kind_to_json(&e.kind, &mut line);
            line.push('}');
            line.push('\n');
            w.write_all(line.as_bytes())?;
        }
        Ok(())
    }

    fn kind_to_json(k: &EventKind, out: &mut String) {
        match k {
            EventKind::Spawned { name, parent } => {
                out.push_str("{\"k\":\"spawn\",\"name\":");
                push_str_json(name, out);
                if let Some(p) = parent {
                    out.push_str(",\"parent\":");
                    out.push_str(&p.0.to_string());
                }
                out.push('}');
            }
            EventKind::PollStart => out.push_str("{\"k\":\"poll_start\"}"),
            EventKind::PollEnd { rescheduled, duration_nanos } => {
                out.push_str("{\"k\":\"poll_end\",\"resched\":");
                out.push_str(if *rescheduled { "true" } else { "false" });
                out.push_str(",\"d_ns\":");
                out.push_str(&duration_nanos.to_string());
                out.push('}');
            }
            EventKind::Wake { from_worker, from_task } => {
                out.push_str("{\"k\":\"wake\"");
                if let Some(w) = from_worker {
                    out.push_str(",\"from_worker\":");
                    out.push_str(&w.to_string());
                }
                if let Some(t) = from_task {
                    out.push_str(",\"from_task\":");
                    out.push_str(&t.0.to_string());
                }
                out.push('}');
            }
            EventKind::StateChanged { from, to } => {
                out.push_str("{\"k\":\"state\",\"from\":\"");
                out.push_str(state_name(*from));
                out.push_str("\",\"to\":\"");
                out.push_str(state_name(*to));
                out.push_str("\"}");
            }
            EventKind::Completed => out.push_str("{\"k\":\"completed\"}"),
            EventKind::Aborted => out.push_str("{\"k\":\"aborted\"}"),
            EventKind::Control(s) => {
                out.push_str("{\"k\":\"control\",\"msg\":");
                push_str_json(s, out);
                out.push('}');
            }
            EventKind::User { category, detail } => {
                out.push_str("{\"k\":\"user\",\"cat\":");
                push_str_json(category, out);
                out.push_str(",\"detail\":");
                push_str_json(detail, out);
                out.push('}');
            }
            EventKind::SpanEnter { span_id, name, target, parent_id, fields } => {
                out.push_str("{\"k\":\"span_enter\",\"id\":");
                out.push_str(&span_id.to_string());
                out.push_str(",\"name\":");
                push_str_json(name, out);
                out.push_str(",\"target\":");
                push_str_json(target, out);
                if let Some(p) = parent_id {
                    out.push_str(",\"parent\":");
                    out.push_str(&p.to_string());
                }
                if !fields.is_empty() {
                    out.push_str(",\"fields\":");
                    push_str_json(fields, out);
                }
                out.push('}');
            }
            EventKind::SpanExit { span_id } => {
                out.push_str("{\"k\":\"span_exit\",\"id\":");
                out.push_str(&span_id.to_string());
                out.push('}');
            }
            EventKind::SpanClose { span_id } => {
                out.push_str("{\"k\":\"span_close\",\"id\":");
                out.push_str(&span_id.to_string());
                out.push('}');
            }
            EventKind::SpanEvent { target, level, message, in_span } => {
                out.push_str("{\"k\":\"span_event\",\"target\":");
                push_str_json(target, out);
                out.push_str(",\"level\":");
                push_str_json(level, out);
                out.push_str(",\"msg\":");
                push_str_json(message, out);
                if let Some(s) = in_span {
                    out.push_str(",\"in_span\":");
                    out.push_str(&s.to_string());
                }
                out.push('}');
            }
            EventKind::SpanAllocs { span_id, bytes_delta, count_delta } => {
                out.push_str("{\"k\":\"span_allocs\",\"id\":");
                out.push_str(&span_id.to_string());
                out.push_str(",\"bytes\":");
                out.push_str(&bytes_delta.to_string());
                out.push_str(",\"count\":");
                out.push_str(&count_delta.to_string());
                out.push('}');
            }
        }
    }

    fn push_str_json(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    fn state_name(s: TaskState) -> &'static str {
        match s {
            TaskState::Fresh => "fresh",
            TaskState::Runnable => "runnable",
            TaskState::Running => "running",
            TaskState::Suspended => "suspended",
            TaskState::PausedByUser => "paused",
            TaskState::Completed => "completed",
            TaskState::Aborted => "aborted",
        }
    }

    // WaitReason intentionally not serialized in this minimal v1 — it's
    // a per-task field, not an event payload. Leaving the import live
    // for future expansion when we add WaitReason events.
    #[allow(dead_code)]
    fn _wait_reason_used(r: WaitReason) -> WaitReason { r }
}

/// Everything that happens inside the runtime is published as an `Event`.
/// The UI subscribes via `EventLog` to render history; the controller can
/// react synchronously by inspecting the same log.
#[derive(Clone, Debug)]
pub struct Event {
    pub seq: u64,
    pub at: Instant,
    pub task: Option<TaskId>,
    pub worker: Option<usize>,
    pub kind: EventKind,
}

#[derive(Clone, Debug)]
pub enum EventKind {
    Spawned {
        name: String,
        parent: Option<TaskId>,
    },
    PollStart,
    /// Emitted when a poll returns. `rescheduled` means the task woke itself
    /// during poll and will be polled again immediately. Whether the future
    /// returned `Ready` is recorded separately as a `Completed` event (we
    /// can't infer Ready from `Runnable::run()` alone — it returns
    /// `rescheduled`, not "ready").
    PollEnd {
        rescheduled: bool,
        duration_nanos: u64,
    },
    /// Wake of a task. `from_task` is set when the wake happened from
    /// within another task's poll (so the timeline can draw edges); `None`
    /// means the wake came from a non-task context (timer, I/O reactor).
    Wake {
        from_worker: Option<usize>,
        from_task: Option<TaskId>,
    },
    StateChanged {
        from: TaskState,
        to: TaskState,
    },
    Completed,
    Aborted,
    /// A user-driven event from the controller (pause, step, manual-pick).
    /// Useful for rendering a timeline of operator actions alongside runtime
    /// events.
    Control(String),
    /// Free-form named event. Emitted by instrumentation in the cf-tokio
    /// shim (channel send, semaphore acquire, notify wake) and by user
    /// code that calls `RuntimeHandle::log_user_event`. Optional payload
    /// for human-readable detail.
    User {
        category: &'static str,
        detail: String,
    },
    /// A `tracing` span entered the current thread. Fields are the
    /// recorded `tracing` span attributes flattened to "k=v" lines.
    /// Span ids are tracing's own; the cf-runtime UI builds the tree
    /// from `parent_id` links. `task` is the tokio task id polling at
    /// the moment the span was entered, captured from our thread-local
    /// — this is the cross-layer link between traced spans and
    /// runtime polls.
    SpanEnter {
        span_id: u64,
        name: &'static str,
        target: &'static str,
        parent_id: Option<u64>,
        fields: String,
    },
    /// `tracing` span exited the thread (popped from the stack). The
    /// span may still be alive elsewhere; "Closed" is when the last
    /// reference drops.
    SpanExit {
        span_id: u64,
    },
    /// `tracing` span was destroyed.
    SpanClose {
        span_id: u64,
    },
    /// `tracing::event!()` invocation — a one-shot record (debug log,
    /// metric, etc) with no enter/exit. Attached to the span that's
    /// currently on top of the thread's stack, if any.
    SpanEvent {
        target: &'static str,
        level: &'static str,
        message: String,
        in_span: Option<u64>,
    },
    /// Memory delta over the lifetime of a span. Emitted by the
    /// tracing layer at span exit when an allocation snapshot hook is
    /// installed (e.g. via turbo-tasks-malloc). Negative values mean
    /// the span freed more than it allocated.
    SpanAllocs {
        span_id: u64,
        bytes_delta: i64,
        count_delta: i64,
    },
}

/// Bounded ring-buffer log. We keep the most recent N events for the UI; older
/// events are dropped. Cheap to push from worker threads, cheap to snapshot
/// from the UI thread.
pub struct EventLog {
    inner: Mutex<EventLogInner>,
    capacity: usize,
}

struct EventLogInner {
    events: VecDeque<Event>,
    next_seq: u64,
}

impl EventLog {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(EventLogInner {
                events: VecDeque::with_capacity(capacity),
                next_seq: 0,
            }),
            capacity,
        })
    }

    pub fn push(&self, task: Option<TaskId>, worker: Option<usize>, kind: EventKind) -> u64 {
        let mut g = self.inner.lock();
        let seq = g.next_seq;
        g.next_seq += 1;
        let evt = Event {
            seq,
            at: Instant::now(),
            task,
            worker,
            kind,
        };
        if g.events.len() == self.capacity {
            g.events.pop_front();
        }
        g.events.push_back(evt);
        seq
    }

    /// Snapshot the current contents. Allocates; intended for the UI thread
    /// at frame rate, not hot paths.
    pub fn snapshot(&self) -> Vec<Event> {
        let g = self.inner.lock();
        g.events.iter().cloned().collect()
    }

    /// Returns the most recent `n` events.
    pub fn tail(&self, n: usize) -> Vec<Event> {
        let g = self.inner.lock();
        g.events.iter().rev().take(n).rev().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().events.len()
    }
}
