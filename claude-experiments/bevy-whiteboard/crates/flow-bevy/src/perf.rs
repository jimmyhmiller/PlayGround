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

    /// For each phase, sum any samples recorded since the previous
    /// call (tracked per-phase in `cursors`). Returns one entry per
    /// phase that produced new samples this slice.
    ///
    /// Used by the bench to slice `PhaseTimings` per frame: call
    /// once at end-of-frame, get exactly that frame's contribution.
    pub fn delta_since(
        &self,
        cursors: &mut HashMap<&'static str, usize>,
    ) -> Vec<(&'static str, f64)> {
        let mut out = Vec::new();
        for (name, vec) in self.samples.iter() {
            let cursor = cursors.entry(*name).or_insert(0);
            if vec.len() > *cursor {
                let sum: f64 = vec[*cursor..].iter().sum();
                out.push((*name, sum));
                *cursor = vec.len();
            }
        }
        out
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

/// Companion bucket for samples that originate on the **sim worker
/// thread**, not the main thread. The worker drops perf samples
/// (sim.run_until.total, sim.fire_rules, sim.deliver_packets, etc.)
/// into the snapshot it publishes; the bridge drains them here.
///
/// Crucially these samples are *not* on the main-thread frame's
/// critical path under worker mode — keeping them out of
/// `PhaseTimings` avoids the misleading impression that they
/// contribute to frame time.
#[derive(Resource, Default)]
pub struct WorkerPerf(pub PhaseTimings);

/// Plugin that registers [`PhaseTimings`] and [`WorkerPerf`]. Systems
/// that want to record into the main-thread bucket pull it as
/// `ResMut<PhaseTimings>` and invoke [`time_phase!`].
///
/// Also installs:
///  * `sched.main_world_total` — start of `First` to end of `Last`.
///    Reliable because `First` and `Last` are ordered by Bevy's
///    `MainScheduleOrder`. The complement (`frame_ms − main_world_total`)
///    is "render-world + wgpu submit + present + OS".
///  * `sched.transform_propagate` — wraps Bevy's
///    `TransformSystem::TransformPropagate` set. With many entities
///    + a deep hierarchy this often dominates `PostUpdate`.
///  * `sched.visibility_propagate` — wraps Bevy's
///    `VisibilitySystems::VisibilityPropagate` set.
///
/// Per-schedule (sched.update/sched.post_update/etc.) wrappers were
/// removed because they couldn't be ordered against arbitrary
/// foreign systems in the same schedule, so the recorded duration
/// was effectively "marker-to-marker with parallel systems missed",
/// not the schedule's wall time.
pub struct PerfPlugin;
impl Plugin for PerfPlugin {
    fn build(&self, app: &mut App) {
        use bevy::app::{First, Last, PostUpdate};
        use bevy::transform::TransformSystems;
        use bevy::camera::visibility::VisibilitySystems;
        app.init_resource::<PhaseTimings>()
            .init_resource::<WorkerPerf>()
            .init_resource::<ScheduleClock>()
            // Frame-wide bookend.
            .add_systems(First, mark_first_start)
            .add_systems(Last, finish_main_world)
            // Transform propagation bookend.
            .add_systems(
                PostUpdate,
                mark_transform_start.before(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                mark_transform_end.after(TransformSystems::Propagate),
            )
            // Visibility propagation bookend.
            .add_systems(
                PostUpdate,
                mark_visibility_start.before(VisibilitySystems::VisibilityPropagate),
            )
            .add_systems(
                PostUpdate,
                mark_visibility_end.after(VisibilitySystems::VisibilityPropagate),
            );
    }
}

/// Cached start `Instant`s for the named bookend pairs.
#[derive(Resource, Default)]
pub struct ScheduleClock {
    first_start: Option<std::time::Instant>,
    transform_start: Option<std::time::Instant>,
    visibility_start: Option<std::time::Instant>,
}

fn mark_first_start(mut clock: ResMut<ScheduleClock>) {
    clock.first_start = Some(std::time::Instant::now());
}

/// Records `sched.main_world_total`. Public so callers (the bench)
/// can `.after(finish_main_world)` to ensure their per-frame capture
/// runs after this writes the over-arching span.
pub fn finish_main_world(mut clock: ResMut<ScheduleClock>, mut perf: ResMut<PhaseTimings>) {
    if let Some(start) = clock.first_start.take() {
        let us = start.elapsed().as_secs_f64() * 1_000_000.0;
        perf.record_us("sched.main_world_total", us);
    }
}

fn mark_transform_start(mut clock: ResMut<ScheduleClock>) {
    clock.transform_start = Some(std::time::Instant::now());
}
fn mark_transform_end(mut clock: ResMut<ScheduleClock>, mut perf: ResMut<PhaseTimings>) {
    if let Some(start) = clock.transform_start.take() {
        let us = start.elapsed().as_secs_f64() * 1_000_000.0;
        perf.record_us("sched.transform_propagate", us);
    }
}
fn mark_visibility_start(mut clock: ResMut<ScheduleClock>) {
    clock.visibility_start = Some(std::time::Instant::now());
}
fn mark_visibility_end(mut clock: ResMut<ScheduleClock>, mut perf: ResMut<PhaseTimings>) {
    if let Some(start) = clock.visibility_start.take() {
        let us = start.elapsed().as_secs_f64() * 1_000_000.0;
        perf.record_us("sched.visibility_propagate", us);
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
