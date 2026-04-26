//! Lightweight, program-specific frame-time tracing.
//!
//! Bevy's built-in `FrameTimeDiagnosticsPlugin` measures total frame
//! time. Sampler-based profilers (samply, cargo flamegraph) measure CPU
//! breakdowns at the function level. Neither tells us "how much time
//! did *the packet cloud update* take this frame, vs *the binary slot
//! sync*, vs *the visual timeline ingest*?" — which is what we need to
//! decide where to optimise next.
//!
//! This module provides that. A [`PhaseTimings`] resource accumulates
//! per-frame samples for each named phase, and the [`time_phase!`]
//! macro wraps a block of code so the elapsed wall time gets recorded
//! into the resource keyed by phase name. At end-of-bench (or any
//! other moment), [`PhaseTimings::report`] returns per-phase
//! mean/p50/p95/p99/max stats.
//!
//! Cost is one `Instant::now()` pair plus a hash-map insert per phase
//! per frame — well below the timing resolution of anything we'd
//! measure with it.
//!
//! Usage:
//! ```ignore
//! use crate::perf::{time_phase, PhaseTimings};
//!
//! fn my_system(mut perf: ResMut<PhaseTimings>, /* ... */) {
//!     time_phase!(perf, "my_system", {
//!         // ... work ...
//!     });
//! }
//! ```

use std::collections::HashMap;

use bevy::prelude::*;

/// Accumulator for per-phase frame-time samples. Insert via the
/// [`time_phase!`] macro; read via [`PhaseTimings::report`] at exit.
///
/// Samples are stored in microseconds (f64) — finer than the
/// frame-time resolution of any relevant phase, coarse enough that
/// `f64` arithmetic stays exact.
#[derive(Resource, Default)]
pub struct PhaseTimings {
    samples: HashMap<&'static str, Vec<f64>>,
}

impl PhaseTimings {
    /// Record a sample for `phase` in microseconds. Prefer the
    /// [`time_phase!`] macro which calls this with the right value.
    pub fn record_us(&mut self, phase: &'static str, us: f64) {
        self.samples.entry(phase).or_default().push(us);
    }

    /// Compute summary stats for every phase. Returns one row per
    /// phase, sorted by name for stable reporting.
    pub fn report(&self) -> Vec<PhaseReport> {
        let mut rows: Vec<PhaseReport> = self
            .samples
            .iter()
            .map(|(name, raw)| {
                let mut sorted = raw.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let mean = sorted.iter().sum::<f64>() / n as f64;
                let q = |p: f64| sorted[((n - 1) as f64 * p) as usize];
                PhaseReport {
                    name,
                    samples: n,
                    mean_us: mean,
                    p50_us: q(0.50),
                    p95_us: q(0.95),
                    p99_us: q(0.99),
                    max_us: sorted[n - 1],
                }
            })
            .collect();
        rows.sort_by_key(|r| r.name);
        rows
    }
}

/// One row of [`PhaseTimings::report`]. All durations in microseconds.
pub struct PhaseReport {
    pub name: &'static str,
    pub samples: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

/// Time a block of code and record the elapsed microseconds into
/// [`PhaseTimings`] under the given static phase name.
///
/// The block can return a value — the macro forwards it.
#[macro_export]
macro_rules! time_phase {
    ($timings:expr, $name:literal, $body:block) => {{
        let __start = ::std::time::Instant::now();
        let __result = $body;
        let __us = __start.elapsed().as_secs_f64() * 1_000_000.0;
        $timings.record_us($name, __us);
        __result
    }};
}

/// Plugin that just registers the [`PhaseTimings`] resource. Systems
/// that want to record into it pull it as `ResMut<PhaseTimings>` and
/// invoke [`time_phase!`].
pub struct PerfPlugin;
impl Plugin for PerfPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhaseTimings>();
    }
}

/// Print a phase report to stdout in the same table style as
/// `fps_bench`'s frame-time report.
pub fn print_report(timings: &PhaseTimings) {
    let rows = timings.report();
    if rows.is_empty() {
        println!("(no phase samples recorded)");
        return;
    }
    println!();
    println!("=== per-phase timings ===");
    println!(
        "  {:<28} {:>8}  {:>10} {:>10} {:>10} {:>10} {:>10}",
        "phase", "frames", "mean", "p50", "p95", "p99", "max",
    );
    for r in rows {
        println!(
            "  {:<28} {:>8}  {:>9.1}µ {:>9.1}µ {:>9.1}µ {:>9.1}µ {:>9.1}µ",
            r.name, r.samples, r.mean_us, r.p50_us, r.p95_us, r.p99_us, r.max_us,
        );
    }
}
