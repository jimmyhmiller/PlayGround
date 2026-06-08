//! Per-frame profiler accumulator.
//!
//! A process-global, thread-safe collector that any pane crate can write
//! into without threading a Bevy resource through its queries (the
//! instrumented systems live across `pane-bevy`, `editor-bevy`,
//! `widget-bevy`, and `terminal-bevy`, and only `pane-bevy` is shared by
//! all of them — so the accumulator lives here as a global).
//!
//! Two flavours of span, both RAII guards that record elapsed wall time
//! on drop:
//!   - [`pane_span`] — work attributable to a specific pane Entity (the
//!     terminal cell sync, a widget's render, an editor's text sync). The
//!     overlay later resolves the Entity to a human label (title/project)
//!     via the Bevy `World`.
//!   - [`sys_span`] — a shared subsystem not tied to one pane (input,
//!     layout, chrome, shader uniforms).
//!
//! Cost when disabled is a single relaxed atomic load: [`enabled`] gates
//! every span, so leaving the guards in the hot paths is free in normal
//! runs. The collector is reset-on-read: [`take_frame`] swaps out the
//! accumulated map, so the span total between two reads is exactly one
//! frame's worth (read once per frame from the `Last` schedule).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Global enable flag. Flipped by the overlay toggle (Cmd+Shift+F) and by
/// `TBPROF` at startup. A relaxed load is all an idle span pays.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// The live accumulator. Locked only briefly when a guard drops (to add
/// its duration) — contention is a handful of systems per frame, so the
/// mutex never shows up in its own measurements.
static ACCUM: Mutex<Option<FrameAccum>> = Mutex::new(None);

/// One pane's accumulated cost this frame.
#[derive(Clone, Copy, Debug)]
pub struct PaneTime {
    /// Entity bits, so the overlay can re-derive the `Entity` and look up
    /// its title/project without us storing strings on the hot path.
    pub entity_bits: u64,
    /// Static pane-kind tag: "terminal" | "editor" | "widget".
    pub kind: &'static str,
    pub total: Duration,
    /// How many spans rolled into `total` (e.g. per-line editor work).
    pub hits: u32,
}

/// One subsystem's accumulated cost this frame.
#[derive(Clone, Copy, Debug)]
pub struct SysTime {
    pub name: &'static str,
    pub total: Duration,
    pub hits: u32,
    /// True when this span runs *inside* a pane span (e.g. taffy layout
    /// inside a widget render). Nested time is already part of some pane's
    /// total, so the overlay shows it for detail but must NOT add it into
    /// the disjoint frame budget — doing so would double-count.
    pub nested: bool,
}

/// A whole frame's breakdown, handed to the overlay.
#[derive(Default, Clone)]
pub struct FrameData {
    pub panes: Vec<PaneTime>,
    pub subsystems: Vec<SysTime>,
}

#[derive(Default)]
struct FrameAccum {
    panes: HashMap<u64, PaneTime>,
    subsystems: HashMap<&'static str, (Duration, u32, bool)>,
}

/// Is profiling on? Instrumented sites check this before doing any timing
/// so the guards are free when the profiler is idle.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Turn the collector on/off. Off also drops any partially accumulated
/// frame so re-enabling starts clean.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    if let Ok(mut g) = ACCUM.lock() {
        if on {
            *g = Some(FrameAccum::default());
        } else {
            *g = None;
        }
    }
}

/// Snapshot the spans accumulated since the last call and reset. Returns
/// `None` when profiling is off. Called once per frame from `Last`.
pub fn take_frame() -> Option<FrameData> {
    if !enabled() {
        return None;
    }
    let mut g = ACCUM.lock().ok()?;
    let accum = g.get_or_insert_with(FrameAccum::default);
    let mut panes: Vec<PaneTime> = accum.panes.values().copied().collect();
    panes.sort_by(|a, b| b.total.cmp(&a.total));
    let mut subsystems: Vec<SysTime> = accum
        .subsystems
        .iter()
        .map(|(name, (total, hits, nested))| SysTime {
            name,
            total: *total,
            hits: *hits,
            nested: *nested,
        })
        .collect();
    subsystems.sort_by(|a, b| b.total.cmp(&a.total));
    *accum = FrameAccum::default();
    Some(FrameData { panes, subsystems })
}

fn record_pane(entity_bits: u64, kind: &'static str, dur: Duration) {
    if let Ok(mut g) = ACCUM.lock() {
        let accum = g.get_or_insert_with(FrameAccum::default);
        let e = accum.panes.entry(entity_bits).or_insert(PaneTime {
            entity_bits,
            kind,
            total: Duration::ZERO,
            hits: 0,
        });
        e.total += dur;
        e.hits += 1;
        // First non-empty kind wins; keeps a stable label if a pane is
        // touched by more than one instrumented system.
        if e.kind.is_empty() {
            e.kind = kind;
        }
    }
}

fn record_sys(name: &'static str, dur: Duration, nested: bool) {
    if let Ok(mut g) = ACCUM.lock() {
        let accum = g.get_or_insert_with(FrameAccum::default);
        let e = accum
            .subsystems
            .entry(name)
            .or_insert((Duration::ZERO, 0, nested));
        e.0 += dur;
        e.1 += 1;
        // Once nested, always nested (a span used both ways is rare, but
        // counting it as nested keeps the budget conservative).
        e.2 |= nested;
    }
}

/// RAII timing guard. Records its elapsed lifetime into the accumulator on
/// drop. Created disabled (a no-op) when profiling is off.
#[must_use = "the span times until it is dropped; bind it to a local"]
pub struct Span {
    start: Option<Instant>,
    target: Target,
}

enum Target {
    Pane { entity_bits: u64, kind: &'static str },
    Sys { name: &'static str, nested: bool },
}

impl Drop for Span {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let dur = start.elapsed();
        match self.target {
            Target::Pane { entity_bits, kind } => record_pane(entity_bits, kind, dur),
            Target::Sys { name, nested } => record_sys(name, dur, nested),
        }
    }
}

/// Time work attributable to a pane `Entity`. `kind` is a static tag
/// ("terminal"/"editor"/"widget"). No-op (and does not even read the
/// clock) when profiling is disabled.
#[inline]
pub fn pane_span(entity_bits: u64, kind: &'static str) -> Span {
    Span {
        start: if enabled() { Some(Instant::now()) } else { None },
        target: Target::Pane { entity_bits, kind },
    }
}

/// Time a shared subsystem (not tied to a single pane, not inside a pane
/// span). Counted in the disjoint frame budget.
#[inline]
pub fn sys_span(name: &'static str) -> Span {
    Span {
        start: if enabled() { Some(Instant::now()) } else { None },
        target: Target::Sys {
            name,
            nested: false,
        },
    }
}

/// Time a subsystem that runs *inside* a pane span (e.g. taffy layout
/// inside a widget render). Shown for detail but excluded from the
/// disjoint frame budget so it isn't double-counted against the pane.
#[inline]
pub fn sys_span_nested(name: &'static str) -> Span {
    Span {
        start: if enabled() { Some(Instant::now()) } else { None },
        target: Target::Sys { name, nested: true },
    }
}
