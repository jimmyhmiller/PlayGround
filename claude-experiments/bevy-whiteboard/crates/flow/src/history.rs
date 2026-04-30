//! Snapshot + rewind.
//!
//! Every piece of simulation state lives on `Sim`: nodes, edges,
//! in-flight, clock, seeded RNG, templates, pending scenario actions,
//! event log. Because the RNG is seeded and `Sim: Clone`, a snapshot
//! taken at time T and replayed forward will produce identical
//! subsequent events (modulo new external scenario input).
//!
//! This module provides a bounded ring of snapshots keyed by the
//! event log's cumulative counter (`total_recorded`), so callers can
//! scrub to "roughly K events ago" without keeping the entire history.
use std::collections::VecDeque;

use crate::sim::Sim;

/// One saved snapshot with the event count at which it was taken.
#[derive(Clone)]
pub struct Snapshot {
    /// Cumulative `EventLog::total_recorded` at snapshot time.
    pub event_index: u64,
    pub sim_now_ns: u64,
    pub sim: Sim,
}

/// Cadence policy for `SnapshotRing::auto_capture`. The ring captures
/// when *either* condition crosses the threshold since the last capture.
/// Both default to "never trigger on this axis" so callers can opt into
/// time-based, event-based, or both.
#[derive(Clone, Copy, Debug)]
pub struct CapturePolicy {
    /// Minimum sim-ns between captures, or `u64::MAX` to disable the
    /// time axis.
    pub min_interval_ns: u64,
    /// Minimum events between captures, or `u64::MAX` to disable the
    /// event axis.
    pub min_event_delta: u64,
}

impl CapturePolicy {
    /// 250ms between captures, no event-axis trigger. Reasonable
    /// default for interactive rewind in the host UI.
    pub const DEFAULT: Self = Self {
        min_interval_ns: 250_000_000,
        min_event_delta: u64::MAX,
    };
}

impl Default for CapturePolicy {
    fn default() -> Self { Self::DEFAULT }
}

/// Ring of snapshots with a sticky anchor at the *first* capture
/// (typically t=0). Capacity-bounded queue evicts oldest entries; the
/// anchor is never evicted, so rewind can always reach the very
/// beginning even if the ring has rolled forward many times.
pub struct SnapshotRing {
    pub cap: usize,
    pub entries: VecDeque<Snapshot>,
    /// First snapshot ever taken. Sticky — never evicted, never
    /// overwritten. Returned by `latest_before*` when no entry in the
    /// ring is older than the query target.
    pub anchor: Option<Snapshot>,
}

impl SnapshotRing {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0);
        Self { cap, entries: VecDeque::with_capacity(cap), anchor: None }
    }

    /// Take a snapshot unconditionally. Sets the anchor on first call.
    pub fn capture(&mut self, sim: &Sim) {
        let snap = Snapshot {
            event_index: sim.log.total_recorded,
            sim_now_ns: sim.now_ns,
            sim: sim.clone(),
        };
        if self.anchor.is_none() {
            self.anchor = Some(snap.clone());
        }
        if self.entries.len() == self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(snap);
    }

    /// Capture only if the policy thresholds have been crossed since
    /// the last entry. Returns `true` if a snapshot was taken. Always
    /// captures the very first time (so the anchor gets set even if
    /// `now_ns` is 0 and intervals haven't elapsed).
    pub fn auto_capture(&mut self, sim: &Sim, policy: CapturePolicy) -> bool {
        let last = self.entries.back();
        let should = match last {
            None => true,
            Some(last) => {
                let dt = sim.now_ns.saturating_sub(last.sim_now_ns);
                let de = sim.log.total_recorded.saturating_sub(last.event_index);
                dt >= policy.min_interval_ns || de >= policy.min_event_delta
            }
        };
        if should {
            self.capture(sim);
        }
        should
    }

    /// Most-recent snapshot taken at or before the given event index.
    /// Falls back to the sticky anchor if the ring is empty or all
    /// entries are past the target. Returns `None` only when no
    /// snapshots have ever been taken.
    pub fn latest_before(&self, event_index: u64) -> Option<&Snapshot> {
        self.entries
            .iter()
            .rev()
            .find(|s| s.event_index <= event_index)
            .or_else(|| {
                self.anchor
                    .as_ref()
                    .filter(|a| a.event_index <= event_index)
            })
    }

    /// Most-recent snapshot taken at or before the given sim time.
    /// Same anchor-fallback semantics as [`Self::latest_before`].
    pub fn latest_before_ns(&self, sim_ns: u64) -> Option<&Snapshot> {
        self.entries
            .iter()
            .rev()
            .find(|s| s.sim_now_ns <= sim_ns)
            .or_else(|| {
                self.anchor
                    .as_ref()
                    .filter(|a| a.sim_now_ns <= sim_ns)
            })
    }

    /// Sim-ns of every snapshot the UI can scrub to, anchor first.
    /// Useful for rendering scrub-strip markers.
    pub fn marker_times_ns(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.entries.len() + 1);
        if let Some(a) = &self.anchor {
            out.push(a.sim_now_ns);
        }
        for e in &self.entries {
            // Anchor is also the first entry on first capture; dedupe.
            if out.last().copied() != Some(e.sim_now_ns) {
                out.push(e.sim_now_ns);
            }
        }
        out
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

impl Sim {
    /// Take a snapshot that you can hand to `restore_from` later.
    /// Equivalent to `self.clone()` but more self-documenting.
    pub fn snapshot(&self) -> Sim { self.clone() }

    /// Rewind by overwriting `self` from a previous snapshot. After
    /// this, you can call `run_until` to replay forward identically
    /// (given the seeded RNG was captured in the snapshot).
    pub fn restore_from(&mut self, snapshot: Sim) {
        *self = snapshot;
    }
}
