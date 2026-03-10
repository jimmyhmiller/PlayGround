//! Statemap trace instrumentation for GC safepoint visualization.
//!
//! Emits [statemap](https://github.com/bcantrill/statemap)-compatible
//! trace data capturing thread states and GC phases. Enable with the
//! `statemap` feature flag.
//!
//! Usage:
//! ```rust,ignore
//! let tracer = StatemapTracer::new();
//! // ... pass to Heap, run workload ...
//! tracer.write_to_file("gc_trace.out").unwrap();
//! // Then: statemap gc_trace.out > gc_trace.svg
//! ```

use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

// ─── States ──────────────────────────────────────────────────────

/// Thread/GC entity states for the statemap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TraceState {
    /// Mutator thread running normally.
    Running = 0,
    /// Mutator thread suspended at a safepoint.
    AtSafepoint = 1,
    /// Mutator thread blocked waiting for GC lock (in trigger_gc).
    WaitingForGc = 2,
    /// GC thread idle (between collections).
    GcIdle = 3,
    /// GC thread in STW pause (scanning roots, etc.).
    GcStw = 4,
    /// GC thread in concurrent copy phase.
    GcConcurrent = 5,
    /// GC thread waiting for threads to reach safepoints.
    GcWaitingForSafepoints = 6,
    /// GC thread resuming threads.
    GcResuming = 7,
}

impl TraceState {
    fn name(self) -> &'static str {
        match self {
            TraceState::Running => "running",
            TraceState::AtSafepoint => "at-safepoint",
            TraceState::WaitingForGc => "waiting-for-gc",
            TraceState::GcIdle => "gc-idle",
            TraceState::GcStw => "gc-stw",
            TraceState::GcConcurrent => "gc-concurrent",
            TraceState::GcWaitingForSafepoints => "gc-waiting-for-safepoints",
            TraceState::GcResuming => "gc-resuming",
        }
    }

    fn color(self) -> &'static str {
        match self {
            TraceState::Running => "#57ab5a",             // green
            TraceState::AtSafepoint => "#e5534b",         // red
            TraceState::WaitingForGc => "#c69026",        // yellow/orange
            TraceState::GcIdle => "#444c56",              // dark grey
            TraceState::GcStw => "#e5534b",               // red
            TraceState::GcConcurrent => "#6cb6ff",        // blue
            TraceState::GcWaitingForSafepoints => "#c69026", // orange
            TraceState::GcResuming => "#986ee2",          // purple
        }
    }

    fn value(self) -> u8 {
        self as u8
    }
}

const ALL_STATES: [TraceState; 8] = [
    TraceState::Running,
    TraceState::AtSafepoint,
    TraceState::WaitingForGc,
    TraceState::GcIdle,
    TraceState::GcStw,
    TraceState::GcConcurrent,
    TraceState::GcWaitingForSafepoints,
    TraceState::GcResuming,
];

// ─── Event record ────────────────────────────────────────────────

#[derive(Clone)]
struct TraceEvent {
    entity: String,
    time_ns: u64,
    state: TraceState,
}

// ─── Tracer ──────────────────────────────────────────────────────

/// Accumulates statemap trace events and writes them out.
///
/// Thread-safe: all mutation goes through a `Mutex<Vec<TraceEvent>>`.
/// The lock is held only briefly per event (one push).
pub struct StatemapTracer {
    epoch: Instant,
    events: Mutex<Vec<TraceEvent>>,
}

impl StatemapTracer {
    /// Create a new tracer. The epoch is set to `Instant::now()`.
    pub fn new() -> Self {
        StatemapTracer {
            epoch: Instant::now(),
            events: Mutex::new(Vec::with_capacity(64 * 1024)),
        }
    }

    /// Record a state transition.
    pub fn record(&self, entity: &str, state: TraceState) {
        let elapsed = self.epoch.elapsed();
        let time_ns = elapsed.as_nanos() as u64;
        let event = TraceEvent {
            entity: entity.to_string(),
            time_ns,
            state,
        };
        self.events.lock().unwrap().push(event);
    }

    /// Record a state transition for a thread by ID.
    pub fn record_thread(&self, thread_id: usize, state: TraceState) {
        let entity = format!("thread{}", thread_id);
        self.record(&entity, state);
    }

    /// Record a GC state transition.
    pub fn record_gc(&self, state: TraceState) {
        self.record("gc", state);
    }

    /// Write the statemap data to a file.
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.write_to(&mut f)
    }

    /// Write the statemap data to any writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Metadata
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let secs = now.as_secs();
        let nanos = now.subsec_nanos();

        write!(w, "{{")?;
        write!(w, "\"title\":\"dynalloc GC trace\",")?;
        write!(w, "\"start\":[{},{}],", secs, nanos)?;
        write!(w, "\"states\":{{")?;
        for (i, state) in ALL_STATES.iter().enumerate() {
            if i > 0 {
                write!(w, ",")?;
            }
            write!(
                w,
                "\"{}\":{{\"value\":{},\"color\":\"{}\"}}",
                state.name(),
                state.value(),
                state.color()
            )?;
        }
        write!(w, "}}")?;
        writeln!(w, "}}")?;

        // Data records — time must be a string per statemap format.
        // Sort by time since events come from multiple threads.
        let mut events = self.events.lock().unwrap().clone();
        events.sort_by_key(|e| e.time_ns);
        for event in events.iter() {
            writeln!(
                w,
                "{{\"time\":\"{}\",\"entity\":\"{}\",\"state\":{}}}",
                event.time_ns,
                event.entity,
                event.state.value()
            )?;
        }

        Ok(())
    }

    /// Number of events recorded so far.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}
