//! Backend visualizer instrumentation.
//!
//! When the `visualizer` feature is enabled and `TURBO_TASKS_VIZ=1` is set,
//! records semantic backend events to a SQLite database for offline analysis.

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use parking_lot::Mutex;

use crossbeam_channel::{Sender, TrySendError, bounded};
use rusqlite::{Connection, params};
use tokio::time::Instant;

/// Maximum number of events buffered in the channel before dropping.
const CHANNEL_CAPACITY: usize = 64 * 1024;

/// Flush to SQLite after this many events or after the timeout, whichever comes first.
const BATCH_SIZE: usize = 10_000;

/// Maximum time between SQLite flushes.
const FLUSH_INTERVAL: Duration = Duration::from_millis(100);

// Event kind constants stored in the `events.kind` and used for filtering.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum EventKind {
    TaskCreated = 0,
    TaskScheduled = 1,
    TaskStarted = 2,
    TaskCompleted = 3,
    TaskInvalidated = 4,
    CellUpdated = 5,
    ChildConnected = 6,
    DependencyAdded = 7,
}

impl EventKind {
    pub fn name(&self) -> &'static str {
        match self {
            EventKind::TaskCreated => "TaskCreated",
            EventKind::TaskScheduled => "TaskScheduled",
            EventKind::TaskStarted => "TaskStarted",
            EventKind::TaskCompleted => "TaskCompleted",
            EventKind::TaskInvalidated => "TaskInvalidated",
            EventKind::CellUpdated => "CellUpdated",
            EventKind::ChildConnected => "ChildConnected",
            EventKind::DependencyAdded => "DependencyAdded",
        }
    }
}

// Edge type constants stored in `edges.edge_type`.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum EdgeType {
    Child = 0,
    OutputDep = 1,
    CellDep = 2,
}

// Task state constants stored in `task_states.state`.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum TaskState {
    Created = 0,
    Scheduled = 1,
    Started = 2,
    Completed = 3,
    Invalidated = 4,
}

/// A single instrumentation event emitted from the backend hot path.
#[derive(Debug)]
pub struct VizEvent {
    pub seq: u64,
    pub timestamp_us: u64,
    pub kind: EventKind,
    pub task_id: u64,
    pub data: VizEventData,
}

/// Kind-specific payload for a VizEvent.
#[derive(Debug)]
pub enum VizEventData {
    TaskCreated {
        name: String,
        is_transient: bool,
    },
    TaskScheduled {
        reason: String,
    },
    TaskStarted,
    TaskCompleted {
        stale: bool,
    },
    TaskInvalidated,
    CellUpdated {
        cell_type_id: u32,
        cell_index: u32,
    },
    ChildConnected {
        child_task_id: u64,
    },
    DependencyAdded {
        target_task_id: u64,
        dep_type: EdgeType,
    },
}

/// Format a human-readable detail string for a VizEventData.
pub fn format_viz_detail(data: &VizEventData) -> String {
    match data {
        VizEventData::TaskCreated { name, is_transient } => {
            if *is_transient {
                format!("{name} (transient)")
            } else {
                name.clone()
            }
        }
        VizEventData::TaskScheduled { reason } => format!("reason: {reason}"),
        VizEventData::TaskStarted => String::new(),
        VizEventData::TaskCompleted { stale } => {
            if *stale {
                "completed (stale)".to_string()
            } else {
                "completed".to_string()
            }
        }
        VizEventData::TaskInvalidated => String::new(),
        VizEventData::CellUpdated {
            cell_type_id,
            cell_index,
        } => format!("type={cell_type_id} [cell {cell_index}]"),
        VizEventData::ChildConnected { child_task_id } => format!("child #{child_task_id}"),
        VizEventData::DependencyAdded {
            target_task_id,
            dep_type,
        } => {
            let dep_name = match dep_type {
                EdgeType::Child => "child",
                EdgeType::OutputDep => "output dep",
                EdgeType::CellDep => "cell dep",
            };
            format!("-> #{target_task_id} ({dep_name})")
        }
    }
}

/// Collects visualization events and sends them to a background writer thread.
pub struct VizCollector {
    sender: Sender<VizEvent>,
    seq: AtomicU64,
    start_time: Instant,
}

impl VizCollector {
    /// Returns `true` if the visualizer is enabled via env var.
    pub fn is_enabled() -> bool {
        std::env::var("TURBO_TASKS_VIZ").ok().as_deref() == Some("1")
    }

    /// Creates a new VizCollector (without Arc) and spawns the background writer thread.
    ///
    /// Returns `None` if `TURBO_TASKS_VIZ` is not set to `1`.
    fn try_create(start_time: Instant) -> Option<Self> {
        if !Self::is_enabled() {
            return None;
        }

        let db_path = std::env::var("TURBO_TASKS_VIZ_PATH")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".next/viz.db"));

        let (sender, receiver) = bounded(CHANNEL_CAPACITY);

        // Spawn background writer thread
        thread::Builder::new()
            .name("viz-writer".to_string())
            .spawn(move || {
                writer_thread(receiver, db_path);
            })
            .expect("Failed to spawn viz writer thread");

        Some(Self {
            sender,
            seq: AtomicU64::new(0),
            start_time,
        })
    }

    /// Creates a new VizCollector and spawns the background writer thread.
    ///
    /// Returns `None` if `TURBO_TASKS_VIZ` is not set to `1`.
    pub fn try_new(start_time: Instant) -> Option<Arc<Self>> {
        Self::try_create(start_time).map(Arc::new)
    }

    /// Returns a monotonically increasing sequence number.
    #[inline]
    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Returns microseconds elapsed since the backend start time.
    #[inline]
    pub fn timestamp_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    /// Emit an event. Lossy — drops the event if the channel is full.
    #[inline]
    pub fn emit(&self, kind: EventKind, task_id: u64, data: VizEventData) {
        let event = VizEvent {
            seq: self.next_seq(),
            timestamp_us: self.timestamp_us(),
            kind,
            task_id,
            data,
        };
        // Lossy send — we prefer not blocking the hot path
        match self.sender.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // Channel full, drop the event silently
            }
            Err(TrySendError::Disconnected(_)) => {
                // Writer thread has exited, nothing we can do
            }
        }
    }
}

/// Background thread that drains events from the channel and writes them to SQLite.
fn writer_thread(receiver: crossbeam_channel::Receiver<VizEvent>, db_path: PathBuf) {
    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[turbo-tasks-viz] Failed to open SQLite database at {db_path:?}: {e}");
            return;
        }
    };

    if let Err(e) = init_schema(&conn) {
        eprintln!("[turbo-tasks-viz] Failed to initialize schema: {e}");
        return;
    }

    eprintln!(
        "[turbo-tasks-viz] Recording events to {}",
        db_path.display()
    );

    let mut batch: Vec<VizEvent> = Vec::with_capacity(BATCH_SIZE);

    loop {
        // Block until at least one event arrives, or timeout
        match receiver.recv_timeout(FLUSH_INTERVAL) {
            Ok(event) => {
                batch.push(event);
                // Drain as many as we can without blocking, up to BATCH_SIZE
                while batch.len() < BATCH_SIZE {
                    match receiver.try_recv() {
                        Ok(event) => batch.push(event),
                        Err(_) => break,
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Nothing to do, but flush whatever we have
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // Channel closed — flush remaining and exit
                while let Ok(event) = receiver.try_recv() {
                    batch.push(event);
                }
                if !batch.is_empty() {
                    if let Err(e) = flush_batch(&conn, &batch) {
                        eprintln!("[turbo-tasks-viz] Final flush error: {e}");
                    }
                }
                eprintln!("[turbo-tasks-viz] Writer thread exiting");
                return;
            }
        }

        if !batch.is_empty() {
            if let Err(e) = flush_batch(&conn, &batch) {
                eprintln!("[turbo-tasks-viz] Flush error: {e}");
            }
            batch.clear();
        }
    }
}

fn init_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch("PRAGMA journal_mode = WAL;")?;
    conn.execute_batch("PRAGMA synchronous = NORMAL;")?;

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            is_transient INTEGER NOT NULL DEFAULT 0,
            created_seq INTEGER NOT NULL,
            created_us INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY,
            timestamp_us INTEGER NOT NULL,
            kind INTEGER NOT NULL,
            task_id INTEGER NOT NULL,
            data TEXT
        );

        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY,
            source_task INTEGER NOT NULL,
            target_task INTEGER NOT NULL,
            edge_type INTEGER NOT NULL,
            created_seq INTEGER NOT NULL,
            removed_seq INTEGER
        );

        CREATE TABLE IF NOT EXISTS task_states (
            id INTEGER PRIMARY KEY,
            task_id INTEGER NOT NULL,
            state INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            timestamp_us INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp_us);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_task);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_task);
        CREATE INDEX IF NOT EXISTS idx_task_states_task ON task_states(task_id);
        ",
    )?;

    Ok(())
}

fn flush_batch(conn: &Connection, batch: &[VizEvent]) -> rusqlite::Result<()> {
    let tx = conn.unchecked_transaction()?;

    {
        let mut insert_event = tx.prepare_cached(
            "INSERT OR IGNORE INTO events (seq, timestamp_us, kind, task_id, data) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        let mut insert_task = tx.prepare_cached(
            "INSERT OR IGNORE INTO tasks (task_id, name, is_transient, created_seq, created_us) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        let mut insert_edge = tx.prepare_cached(
            "INSERT INTO edges (source_task, target_task, edge_type, created_seq) VALUES (?1, ?2, ?3, ?4)",
        )?;
        let mut insert_state = tx.prepare_cached(
            "INSERT INTO task_states (task_id, state, seq, timestamp_us) VALUES (?1, ?2, ?3, ?4)",
        )?;

        for event in batch {
            // Build JSON data string for event-specific fields
            let data_json = match &event.data {
                VizEventData::TaskCreated { name, is_transient } => {
                    Some(format!(
                        r#"{{"name":{},"is_transient":{}}}"#,
                        serde_json::to_string(name).unwrap_or_else(|_| "\"?\"".to_string()),
                        is_transient
                    ))
                }
                VizEventData::TaskScheduled { reason } => {
                    Some(format!(
                        r#"{{"reason":{}}}"#,
                        serde_json::to_string(reason).unwrap_or_else(|_| "\"?\"".to_string())
                    ))
                }
                VizEventData::TaskStarted => None,
                VizEventData::TaskCompleted { stale } => {
                    Some(format!(r#"{{"stale":{stale}}}"#))
                }
                VizEventData::TaskInvalidated => None,
                VizEventData::CellUpdated {
                    cell_type_id,
                    cell_index,
                } => Some(format!(
                    r#"{{"cell_type_id":{cell_type_id},"cell_index":{cell_index}}}"#
                )),
                VizEventData::ChildConnected { child_task_id } => {
                    Some(format!(r#"{{"child_task_id":{child_task_id}}}"#))
                }
                VizEventData::DependencyAdded {
                    target_task_id,
                    dep_type,
                } => Some(format!(
                    r#"{{"target_task_id":{target_task_id},"dep_type":{}}}"#,
                    *dep_type as u8
                )),
            };

            insert_event.execute(params![
                event.seq,
                event.timestamp_us,
                event.kind as u8,
                event.task_id,
                data_json,
            ])?;

            // Maintain derived tables
            match &event.data {
                VizEventData::TaskCreated { name, is_transient } => {
                    insert_task.execute(params![
                        event.task_id,
                        name,
                        *is_transient as i32,
                        event.seq,
                        event.timestamp_us,
                    ])?;
                    insert_state.execute(params![
                        event.task_id,
                        TaskState::Created as u8,
                        event.seq,
                        event.timestamp_us,
                    ])?;
                }
                VizEventData::TaskScheduled { .. } => {
                    insert_state.execute(params![
                        event.task_id,
                        TaskState::Scheduled as u8,
                        event.seq,
                        event.timestamp_us,
                    ])?;
                }
                VizEventData::TaskStarted => {
                    insert_state.execute(params![
                        event.task_id,
                        TaskState::Started as u8,
                        event.seq,
                        event.timestamp_us,
                    ])?;
                }
                VizEventData::TaskCompleted { .. } => {
                    insert_state.execute(params![
                        event.task_id,
                        TaskState::Completed as u8,
                        event.seq,
                        event.timestamp_us,
                    ])?;
                }
                VizEventData::TaskInvalidated => {
                    insert_state.execute(params![
                        event.task_id,
                        TaskState::Invalidated as u8,
                        event.seq,
                        event.timestamp_us,
                    ])?;
                }
                VizEventData::ChildConnected { child_task_id } => {
                    insert_edge.execute(params![
                        event.task_id,
                        child_task_id,
                        EdgeType::Child as u8,
                        event.seq,
                    ])?;
                }
                VizEventData::DependencyAdded {
                    target_task_id,
                    dep_type,
                } => {
                    insert_edge.execute(params![
                        event.task_id,
                        target_task_id,
                        *dep_type as u8,
                        event.seq,
                    ])?;
                }
                VizEventData::CellUpdated { .. } => {
                    // No derived table updates needed for cell updates
                }
            }
        }
    }

    tx.commit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Task Debugger — pause/step/breakpoints for the control center
// ---------------------------------------------------------------------------

/// A breakpoint condition that can pause scheduling when matched.
pub struct Breakpoint {
    pub id: u64,
    pub pattern: String,
    pub enabled: bool,
}

/// A task that was intercepted and is waiting to be released.
pub struct PendingTask {
    pub task_id: u64,
    pub name: String,
    /// Which breakpoint caused this task to be held, if any.
    pub hit_breakpoint: Option<u64>,
}

/// Maximum number of debug events kept in the ring buffer.
const EVENT_LOG_CAPACITY: usize = 2000;

/// A lightweight event for the control center event log.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DebugEvent {
    pub seq: u64,
    pub kind: u8,
    pub kind_name: &'static str,
    pub task_id: u64,
    pub detail: String,
    pub timestamp_us: u64,
}

/// Controls task scheduling — supports pause, step, and conditional breakpoints.
pub struct TaskDebugger {
    paused: AtomicBool,
    breakpoints: Mutex<Vec<Breakpoint>>,
    next_bp_id: AtomicU64,
    pending_queue: Mutex<VecDeque<PendingTask>>,
    /// Closure that schedules a task by its id. Set once during startup.
    schedule_fn: OnceLock<Box<dyn Fn(u64) + Send + Sync>>,
    /// Ring buffer of recent debug events for the control center.
    event_log: Mutex<VecDeque<DebugEvent>>,
    /// Monotonically increasing sequence number for debug events.
    event_seq: AtomicU64,
    /// Tracking recently active tasks for the overview grid.
    active_tasks: Mutex<HashMap<u64, ActiveTask>>,
    /// Monotonically increasing step counter for step markers.
    step_seq: AtomicU64,
}

impl TaskDebugger {
    pub fn new() -> Self {
        Self {
            paused: AtomicBool::new(false),
            breakpoints: Mutex::new(Vec::new()),
            next_bp_id: AtomicU64::new(1),
            pending_queue: Mutex::new(VecDeque::new()),
            schedule_fn: OnceLock::new(),
            event_log: Mutex::new(VecDeque::with_capacity(EVENT_LOG_CAPACITY)),
            event_seq: AtomicU64::new(0),
            active_tasks: Mutex::new(HashMap::new()),
            step_seq: AtomicU64::new(0),
        }
    }

    /// Push a debug event into the ring buffer.
    pub fn push_event(
        &self,
        kind: u8,
        kind_name: &'static str,
        task_id: u64,
        task_name: String,
        detail: String,
        timestamp_us: u64,
    ) {
        let seq = self.event_seq.fetch_add(1, Ordering::Relaxed);
        let event = DebugEvent {
            seq,
            kind,
            kind_name,
            task_id,
            detail,
            timestamp_us,
        };
        let mut log = self.event_log.lock();
        if log.len() >= EVENT_LOG_CAPACITY {
            log.pop_front();
        }
        log.push_back(event);
        drop(log);

        // Track active tasks for the overview grid
        if task_id > 0 {
            let state = match kind {
                0 => "created",
                1 => "scheduled",
                2 => "in_progress",
                3 => "completed",
                4 => "dirty",
                5 | 6 | 7 => "in_progress",
                _ => return, // Don't track marker events
            };
            let mut active = self.active_tasks.lock();
            if let Some(existing) = active.get_mut(&task_id) {
                existing.state = state;
                existing.last_event_seq = seq;
                if !task_name.is_empty() {
                    existing.name = task_name;
                }
            } else {
                let name = if task_name.is_empty() {
                    format!("#{}", task_id)
                } else {
                    task_name
                };
                active.insert(task_id, ActiveTask {
                    task_id,
                    name,
                    state,
                    last_event_seq: seq,
                });
                // Cap at 500 entries — remove the one with lowest seq
                if active.len() > 500 {
                    if let Some((&oldest_id, _)) =
                        active.iter().min_by_key(|(_, t)| t.last_event_seq)
                    {
                        active.remove(&oldest_id);
                    }
                }
            }
        }
    }

    /// Return all events with seq > since_seq.
    pub fn events_since(&self, since_seq: u64) -> Vec<DebugEvent> {
        let log = self.event_log.lock();
        log.iter()
            .filter(|e| e.seq > since_seq)
            .cloned()
            .collect()
    }

    /// Store the schedule function. Called once during backend startup.
    pub fn set_schedule_fn(&self, f: Box<dyn Fn(u64) + Send + Sync>) {
        let _ = self.schedule_fn.set(f);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
    }

    /// Check if this task should be held back from scheduling.
    ///
    /// Returns `true` if the task was enqueued (meaning: don't schedule it).
    /// Returns `false` if the task should proceed to schedule normally.
    pub fn check_and_enqueue(&self, task_id: u64, name: String) -> bool {
        // Check breakpoints first (even when not paused)
        let hit_bp = {
            let bps = self.breakpoints.lock();
            bps.iter().find_map(|bp| {
                if bp.enabled && name.contains(&bp.pattern) {
                    Some(bp.id)
                } else {
                    None
                }
            })
        };

        if let Some(bp_id) = hit_bp {
            // Auto-pause on breakpoint hit
            self.paused.store(true, Ordering::Release);
            self.pending_queue.lock().push_back(PendingTask {
                task_id,
                name,
                hit_breakpoint: Some(bp_id),
            });
            return true;
        }

        if self.paused.load(Ordering::Acquire) {
            self.pending_queue.lock().push_back(PendingTask {
                task_id,
                name,
                hit_breakpoint: None,
            });
            return true;
        }

        false
    }

    pub fn add_breakpoint(&self, pattern: String) -> u64 {
        let id = self.next_bp_id.fetch_add(1, Ordering::Relaxed);
        self.breakpoints.lock().push(Breakpoint {
            id,
            pattern,
            enabled: true,
        });
        id
    }

    pub fn remove_breakpoint(&self, id: u64) -> bool {
        let mut bps = self.breakpoints.lock();
        let len_before = bps.len();
        bps.retain(|bp| bp.id != id);
        bps.len() < len_before
    }

    pub fn toggle_breakpoint(&self, id: u64, enabled: bool) -> bool {
        let mut bps = self.breakpoints.lock();
        if let Some(bp) = bps.iter_mut().find(|bp| bp.id == id) {
            bp.enabled = enabled;
            true
        } else {
            false
        }
    }

    pub fn list_breakpoints(&self) -> Vec<(u64, String, bool)> {
        self.breakpoints
            .lock()
            .iter()
            .map(|bp| (bp.id, bp.pattern.clone(), bp.enabled))
            .collect()
    }

    /// Release one pending task. Returns the task_id that was released, if any.
    pub fn release_one(&self) -> Option<u64> {
        let task = self.pending_queue.lock().pop_front()?;
        let task_id = task.task_id;
        let task_name = task.name.clone();
        self.do_schedule(task_id);
        let step_num = self.step_seq.fetch_add(1, Ordering::Relaxed) + 1;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.push_event(
            255,
            "StepMarker",
            task_id,
            task_name,
            format!("Step #{}: released task #{}", step_num, task_id),
            ts,
        );
        Some(task_id)
    }

    /// Release a specific pending task by id.
    pub fn release_specific(&self, task_id: u64) -> bool {
        let mut queue = self.pending_queue.lock();
        if let Some(pos) = queue.iter().position(|t| t.task_id == task_id) {
            let task = queue.remove(pos).unwrap();
            let task_name = task.name.clone();
            drop(queue);
            self.do_schedule(task_id);
            let step_num = self.step_seq.fetch_add(1, Ordering::Relaxed) + 1;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            self.push_event(
                255,
                "StepMarker",
                task_id,
                task_name,
                format!("Step #{}: released task #{}", step_num, task_id),
                ts,
            );
            true
        } else {
            false
        }
    }

    /// Release up to `count` pending tasks.
    pub fn release_count(&self, count: usize) -> usize {
        let to_release: Vec<_> = {
            let mut queue = self.pending_queue.lock();
            let n = count.min(queue.len());
            queue.drain(..n).collect()
        };
        let released = to_release.len();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        for task in to_release {
            self.do_schedule(task.task_id);
            let step_num = self.step_seq.fetch_add(1, Ordering::Relaxed) + 1;
            self.push_event(
                255,
                "StepMarker",
                task.task_id,
                task.name,
                format!("Step #{}: released task #{}", step_num, task.task_id),
                ts,
            );
        }
        released
    }

    /// Resume: unpause and flush all pending tasks.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
        let mut queue = self.pending_queue.lock();
        let count = queue.len();
        let tasks: Vec<_> = queue.drain(..).collect();
        drop(queue);
        for task in tasks {
            self.do_schedule(task.task_id);
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.push_event(
            254,
            "ResumeMarker",
            0,
            String::new(),
            format!("Resumed: released {} pending tasks", count),
            ts,
        );
    }

    /// Release pending tasks one at a time until the queue is idle, up to max.
    pub fn release_until_idle(&self, max: usize) -> usize {
        let mut total_released = 0;
        for _ in 0..max {
            let task = self.pending_queue.lock().pop_front();
            if let Some(task) = task {
                let task_id = task.task_id;
                let task_name = task.name.clone();
                self.do_schedule(task_id);
                let step_num = self.step_seq.fetch_add(1, Ordering::Relaxed) + 1;
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                self.push_event(
                    255,
                    "StepMarker",
                    task_id,
                    task_name,
                    format!("Step #{}: released task #{}", step_num, task_id),
                    ts,
                );
                total_released += 1;
                // Small sleep to allow the released task to execute and
                // potentially produce new pending tasks
                std::thread::sleep(std::time::Duration::from_millis(10));
            } else {
                break;
            }
        }
        total_released
    }

    pub fn pending_list(&self) -> Vec<(u64, String, Option<u64>)> {
        self.pending_queue
            .lock()
            .iter()
            .map(|t| (t.task_id, t.name.clone(), t.hit_breakpoint))
            .collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_queue.lock().len()
    }

    /// Return all recently active tasks, sorted by last event sequence.
    pub fn active_task_list(&self) -> Vec<ActiveTask> {
        let active = self.active_tasks.lock();
        let mut tasks: Vec<ActiveTask> = active.values().cloned().collect();
        tasks.sort_by_key(|t| t.last_event_seq);
        tasks
    }

    fn do_schedule(&self, task_id: u64) {
        if let Some(f) = self.schedule_fn.get() {
            f(task_id);
        }
    }
}

// ---------------------------------------------------------------------------
// VizController — bundles the collector and debugger
// ---------------------------------------------------------------------------

/// Bundles the event collector and task debugger together.
pub struct VizController {
    pub collector: VizCollector,
    pub debugger: TaskDebugger,
}

impl VizController {
    pub fn try_new(start_time: Instant) -> Option<Arc<Self>> {
        let collector = VizCollector::try_create(start_time)?;
        Some(Arc::new(Self {
            collector,
            debugger: TaskDebugger::new(),
        }))
    }
}

// ---------------------------------------------------------------------------
// VizBackendAccess — type-erased trait for live task inspection
// ---------------------------------------------------------------------------

/// Information about a single cell in a task.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CellInfo {
    pub type_name: String,
    pub cell_index: u32,
    pub has_data: bool,
}

/// Information about a task's dependencies.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskDepsInfo {
    pub output_deps: Vec<(u64, String)>,
    pub cell_deps: Vec<(u64, String, u32)>,
    pub dependents: Vec<(u64, String)>,
}

/// A node in a task relationship graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphNode {
    pub task_id: u64,
    pub name: String,
    pub state: &'static str,
}

/// An edge in a task relationship graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphEdge {
    pub source: u64,
    pub target: u64,
    pub edge_type: &'static str,
    pub label: Option<String>,
}

/// A task's neighborhood graph: nodes, edges, and the root task.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub root: u64,
}

/// Detailed information about a single cell's content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CellDetail {
    pub type_name: String,
    pub cell_index: u32,
    pub has_data: bool,
    pub data_preview: Option<String>,
    pub data_size_bytes: Option<usize>,
}

/// Comprehensive state information for a task.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskStateInfo {
    pub task_id: u64,
    pub name: Option<String>,
    pub state: &'static str,
    pub is_dirty: bool,
    pub is_in_progress: bool,
    pub has_output: bool,
    pub output_description: Option<String>,
    pub cell_count: usize,
    pub child_count: usize,
    pub output_dep_count: usize,
    pub cell_dep_count: usize,
    pub dependent_count: usize,
    pub is_stateful: bool,
    pub is_immutable: bool,
}

/// A recently active task for the overview grid.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActiveTask {
    pub task_id: u64,
    pub name: String,
    pub state: &'static str,
    pub last_event_seq: u64,
}

/// Type-erased trait for inspecting live task state from the backend.
/// Keeps viz.rs and viz_server.rs non-generic over `B: BackingStorage`.
pub trait VizBackendAccess: Send + Sync + 'static {
    fn get_task_description(&self, task_id: u64) -> Option<String>;
    fn list_task_cells(&self, task_id: u64) -> Vec<CellInfo>;
    fn list_task_children(&self, task_id: u64) -> Vec<(u64, String)>;
    fn list_task_dependencies(&self, task_id: u64) -> TaskDepsInfo;
    fn search_tasks(&self, pattern: &str, limit: usize) -> Vec<(u64, String)>;
    fn get_task_state(&self, task_id: u64) -> Option<&'static str>;
    fn get_task_graph(&self, task_id: u64, depth: usize) -> TaskGraph;
    fn get_task_state_info(&self, task_id: u64) -> Option<TaskStateInfo>;
    fn get_cell_detail(&self, task_id: u64, cell_index: u32) -> Option<CellDetail>;
}
