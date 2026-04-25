//! Externalized scenario events. Plays the same role as
//! `scenario::Scenario` but is editable at run time and visible to
//! the host UI (which renders a strip showing scheduled events).
//!
//! Lives on [`Sim`] as `sim.timeline`. The engine's main loop
//! ([`Sim::run_until`]) fires due events as part of normal sim time
//! advancement; nothing about the timeline mechanism involves the
//! host UI framework (it sits in the `flow` crate, not `flow-bevy`).
//! Tests hitting `Sim::run_until` directly get timeline behavior with
//! no extra setup.
//!
//! Compound events: an event is a `(at_ns, Vec<TimelineAction>)` pair.
//! All actions in one event fire at the same instant, atomically —
//! "at t=2s, set A.x := 1 AND set B.y := 2" is one event with two
//! actions, not two adjacent events. This keeps the UI marker count
//! aligned with the user's mental model of "one moment, multiple
//! changes" and matches how the scenario DSL groups things.
//!
//! Storage shape: a time-sorted `Vec<TimelineEvent>` rather than a
//! `BinaryHeap` because we need three operations heaps don't give us:
//!   - remove an event by id
//!   - render the entire pending + fired list for the strip
//!   - mark events as fired without losing them, so the UI can keep
//!     them visible as history
//!
//! Type-check on fire: a scheduled write whose value type doesn't
//! match the slot's current type is silently skipped (per-action).
//! The containing event still completes and is marked `fired`.

use serde::{Deserialize, Serialize};

use crate::sim::{NodeId, Time};
use crate::value::Value;

/// One slot write inside a [`TimelineEvent`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineAction {
    pub node: NodeId,
    pub slot: String,
    pub value: Value,
}

/// One scheduled moment carrying a list of slot writes that fire
/// atomically at the same `at_ns`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub id: u64,
    pub at_ns: Time,
    pub actions: Vec<TimelineAction>,
    pub fired: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timeline {
    pub events: Vec<TimelineEvent>,
    next_id: u64,
}

impl Timeline {
    pub fn new() -> Self { Self::default() }

    /// Convenience: schedule a single-action event. Returns the event
    /// id so callers can later [`Timeline::remove`] it.
    pub fn schedule(
        &mut self,
        at_ns: Time,
        node: NodeId,
        slot: String,
        value: Value,
    ) -> u64 {
        self.schedule_compound(at_ns, vec![TimelineAction { node, slot, value }])
    }

    /// Schedule a compound event — a list of slot writes that fire
    /// atomically at `at_ns`.
    pub fn schedule_compound(&mut self, at_ns: Time, actions: Vec<TimelineAction>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(TimelineEvent { id, at_ns, actions, fired: false });
        self.events.sort_by_key(|e| e.at_ns);
        id
    }

    /// Add an action to an existing event. Useful for the UI's "add
    /// another change to this moment" affordance — keeps the event
    /// count and visual marker count constant.
    pub fn append_action(&mut self, id: u64, action: TimelineAction) -> bool {
        for e in &mut self.events {
            if e.id == id {
                e.actions.push(action);
                return true;
            }
        }
        false
    }

    /// Remove an event regardless of fired status. Returns true on
    /// hit. Removing a fired event is allowed — the user might want
    /// to clear history rows.
    pub fn remove(&mut self, id: u64) -> bool {
        let len = self.events.len();
        self.events.retain(|e| e.id != id);
        len != self.events.len()
    }

    /// Number of events not yet fired.
    pub fn pending(&self) -> usize {
        self.events.iter().filter(|e| !e.fired).count()
    }

    /// Earliest `at_ns` among unfired events, or `None` if every
    /// event has fired (or there are none). The engine reads this to
    /// fold timeline events into its `next_event_ns` computation.
    pub fn next_pending_at_ns(&self) -> Option<Time> {
        self.events.iter().find(|e| !e.fired).map(|e| e.at_ns)
    }

    /// Latest event time. UI strips use this to decide their visible
    /// time range. `None` when there are no events at all.
    pub fn last_at_ns(&self) -> Option<Time> {
        self.events.iter().map(|e| e.at_ns).max()
    }
}
