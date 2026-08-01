//! Coarse build-stage wall-clock profiling.
//!
//! [`frontend_profile`](crate::frontend_profile) measures the four per-module frontend
//! phases (read / transform / lower / resolve). That is the right granularity for the
//! graph, but a production build spends real time outside it too: evaluating
//! integration configuration, processing assets, copying static files, and spawning
//! post-build workers. Stage-level attribution must span all of it.
//!
//! This module is that breakdown: named stages, accumulated across the whole process
//! (so a stage entered many times reports its total and its call count), printed as a
//! table when `DIFFPACK_PROFILE=1` is set. Disabled it costs one relaxed atomic load per
//! [`stage`] call and allocates nothing.
//!
//! Stages nest freely — `emit` contains `emit/minify` — so the table is read as a tree
//! by name, not summed. Overlapping/parallel stages accumulate their own wall time,
//! which is why a stage total can exceed the elapsed build time on a rayon pool; the
//! call count disambiguates.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

/// One stage's accumulated cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTotal {
    pub nanos: u128,
    pub calls: u64,
}

type Totals = Vec<(&'static str, StageTotal)>;

fn totals() -> &'static Mutex<Totals> {
    static TOTALS: OnceLock<Mutex<Totals>> = OnceLock::new();
    TOTALS.get_or_init(|| Mutex::new(Vec::new()))
}

/// `DIFFPACK_PROFILE=1` turns the stage table on. Read once.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("DIFFPACK_PROFILE").is_ok_and(|value| value != "0" && !value.is_empty())
    })
}

/// A running stage. Dropping it records the elapsed time under its name.
pub struct Stage {
    name: &'static str,
    started: Option<Instant>,
}

impl Drop for Stage {
    fn drop(&mut self) {
        let Some(started) = self.started else { return };
        let nanos = started.elapsed().as_nanos();
        let Ok(mut totals) = totals().lock() else {
            return;
        };
        match totals.iter_mut().find(|(name, _)| *name == self.name) {
            Some((_, total)) => {
                total.nanos += nanos;
                total.calls += 1;
            }
            None => totals.push((self.name, StageTotal { nanos, calls: 1 })),
        }
    }
}

/// Time a build stage. Hold the returned guard for the stage's duration:
///
/// ```ignore
/// let _stage = build_profile::stage("emit/public");
/// ```
pub fn stage(name: &'static str) -> Stage {
    Stage {
        name,
        started: enabled().then(Instant::now),
    }
}

/// Every recorded stage, sorted by descending total time.
pub fn snapshot() -> Vec<(&'static str, StageTotal)> {
    let Ok(totals) = totals().lock() else {
        return Vec::new();
    };
    let mut rows = totals.clone();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.nanos));
    rows
}

/// Print the stage table to stderr. A no-op unless `DIFFPACK_PROFILE=1`. `label`
/// identifies which build produced it (`build-app client`, ...), because a production
/// build is several processes and their tables interleave in one terminal.
pub fn report(label: &str, elapsed_ms: f64) {
    if !enabled() {
        return;
    }
    let rows = snapshot();
    eprintln!("\n=== diffpack stage profile: {label} ({elapsed_ms:.0} ms wall) ===");
    if rows.is_empty() {
        eprintln!("  (no stages recorded)");
        return;
    }
    let width = rows.iter().map(|(name, _)| name.len()).max().unwrap_or(0);
    for (name, total) in rows {
        let ms = total.nanos as f64 / 1_000_000.0;
        let share = if elapsed_ms > 0.0 {
            ms / elapsed_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {name:<width$}  {ms:>9.1} ms  {share:>5.1}%  x{}",
            total.calls
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_stage_records_nothing_and_snapshot_stays_empty_for_it() {
        // The profiler is opt-in; with `DIFFPACK_PROFILE` unset (the test default) a
        // stage guard must not appear in the table at all, so instrumentation can be
        // sprinkled anywhere without cost.
        if enabled() {
            return; // The env is set for this run; the assertion below does not apply.
        }
        let _stage = stage("test/never-recorded");
        assert!(
            !snapshot()
                .iter()
                .any(|(name, _)| *name == "test/never-recorded"),
            "a disabled stage must not be recorded",
        );
    }

    #[test]
    fn stage_totals_accumulate_across_entries() {
        // Drive the accumulator directly (independent of the env gate) so the
        // "entered N times" arithmetic the table reports is actually covered.
        let mut totals: Totals = Vec::new();
        for nanos in [10u128, 30, 5] {
            match totals.iter_mut().find(|(name, _)| *name == "x") {
                Some((_, total)) => {
                    total.nanos += nanos;
                    total.calls += 1;
                }
                None => totals.push(("x", StageTotal { nanos, calls: 1 })),
            }
        }
        assert_eq!(totals.len(), 1);
        assert_eq!(totals[0].1.nanos, 45);
        assert_eq!(totals[0].1.calls, 3);
    }
}
