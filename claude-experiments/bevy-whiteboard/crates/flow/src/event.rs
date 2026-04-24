use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::sim::{NodeId, PacketId, Time};
use crate::value::Value;

/// Everything observable that happens in the simulation. The log is
/// the source of truth for scrubbing, plotting, and — eventually —
/// rewind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Simulation clock advanced (monotonic).
    ClockAdvanced { from_ns: Time, to_ns: Time },

    /// A rule was selected and fired. Deterministic given the log
    /// prefix plus the seeded RNG.
    RuleFired { node: NodeId, rule: String, at_ns: Time },

    /// A slot value was written.
    SlotWritten { node: NodeId, slot: String, value: Value, at_ns: Time },

    /// A packet was emitted onto an edge; delivery scheduled at `arrives_at_ns`.
    PacketEmitted {
        packet: PacketId,
        from: NodeId,
        to: NodeId,
        at_ns: Time,
        arrives_at_ns: Time,
        /// The packet's payload, carried so visualizers can label and
        /// color packets by variant tag without re-simulating. Small.
        payload: Value,
    },

    /// A scheduled packet was delivered to a node's inbox.
    PacketDelivered { packet: PacketId, to: NodeId, at_ns: Time },

    /// A packet was consumed by a rule firing.
    PacketConsumed { packet: PacketId, by: NodeId, rule: String, at_ns: Time },

    /// A metric value was recorded.
    MetricRecorded { node: NodeId, name: String, value: Value, at_ns: Time },

    /// A node was spawned from a template. `parent` is the spawner.
    NodeSpawned { node: NodeId, template: String, parent: Option<NodeId>, at_ns: Time },

    /// A node was despawned. Its edges are already gone; any in-flight
    /// packets to/from it will be silently dropped on delivery.
    NodeDespawned { node: NodeId, at_ns: Time },

    /// A runtime error happened during rule execution. The engine never
    /// panics on user-configurable failure modes (missing edge, type
    /// mismatch, empty return-path pop); it records one of these and
    /// drops the effect on the floor. Paired with `Sim.error_counts`
    /// for aggregate visibility.
    ///
    /// `kind` is a short stable string key (e.g. `"emit_no_edge"`,
    /// `"return_path_empty_pop"`, `"slot_type_mismatch"`) so downstream
    /// tools can classify without scraping the detail message.
    RuntimeError {
        kind: String,
        node: Option<NodeId>,
        rule: Option<String>,
        detail: String,
        at_ns: Time,
    },
}

/// Ring-bounded event log. Always on; the bound caps memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLog {
    pub cap: usize,
    pub events: VecDeque<Event>,
    /// Total events ever recorded, even those evicted from the ring.
    /// Useful for stable indexing by "event N" even after eviction.
    pub total_recorded: u64,
}

impl EventLog {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0);
        Self { cap, events: VecDeque::with_capacity(cap), total_recorded: 0 }
    }

    pub fn push(&mut self, e: Event) {
        if self.events.len() == self.cap {
            self.events.pop_front();
        }
        self.events.push_back(e);
        self.total_recorded += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }

    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }
}
