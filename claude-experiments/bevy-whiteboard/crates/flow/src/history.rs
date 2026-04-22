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

/// Ring of snapshots. Oldest snapshots are evicted when capacity is
/// exceeded. The *newest* snapshot's sim is what you'd roll back to
/// for a rewind.
pub struct SnapshotRing {
    pub cap: usize,
    pub entries: VecDeque<Snapshot>,
}

impl SnapshotRing {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0);
        Self { cap, entries: VecDeque::with_capacity(cap) }
    }

    /// Take a snapshot of the current sim and push it into the ring.
    /// Call periodically — every K events or every T ns — to bound
    /// how far forward you'd need to replay on a rewind.
    pub fn capture(&mut self, sim: &Sim) {
        if self.entries.len() == self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(Snapshot {
            event_index: sim.log.total_recorded,
            sim_now_ns: sim.now_ns,
            sim: sim.clone(),
        });
    }

    /// Most-recent snapshot taken at or before the given event index.
    /// Returns `None` if no such snapshot exists.
    pub fn latest_before(&self, event_index: u64) -> Option<&Snapshot> {
        self.entries.iter().rev().find(|s| s.event_index <= event_index)
    }

    /// Most-recent snapshot taken at or before the given sim time.
    pub fn latest_before_ns(&self, sim_ns: u64) -> Option<&Snapshot> {
        self.entries.iter().rev().find(|s| s.sim_now_ns <= sim_ns)
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
