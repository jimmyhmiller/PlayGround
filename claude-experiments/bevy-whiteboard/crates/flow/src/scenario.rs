//! Scripted external interventions scheduled for specific sim times.
//!
//! A scenario is just a timed list of actions that fire alongside the
//! normal rule-firing + packet-delivery loop. Use it for:
//!   - Generating workloads ("fire a request every 5 ms for 2 s")
//!   - Injecting faults ("set edge E to 10 s latency between t=100
//!     and t=200 ms")
//!   - Probing state ("at t=500 ms, snapshot the sim")
//!   - Killing nodes ("take Worker_3 down at t=400 ms")
//!
//! Scenarios are separate from rules. A scenario *is not* part of the
//! system being modeled; it's the external perturber.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::expr::Expr;
use crate::sim::{EdgeId, NodeId, Time};
use crate::value::Value;

/// One scheduled intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Deliver a packet directly into a node's inbox (bypassing edges).
    /// `metadata` and `return_path` default to empty when absent; pass
    /// non-empty values to seed a request that expects a response from
    /// the graph.
    Inject {
        node: NodeId,
        payload: Value,
        metadata: BTreeMap<String, Value>,
        return_path: Vec<NodeId>,
    },
    /// Overwrite a slot's value.
    SetSlot {
        node: NodeId,
        slot: String,
        value: Value,
    },
    /// Swap an edge's latency expression. Use to simulate partitions,
    /// slowdowns, congestion. Setting it extremely high effectively
    /// stalls new emissions on that edge.
    SetEdgeLatency {
        edge: EdgeId,
        latency: Expr,
    },
    /// Remove a node and its edges. In-flight packets targeting it
    /// are dropped on delivery.
    KillNode { node: NodeId },
    /// Rebind a live parameter. Takes effect on the next expression
    /// evaluation that references it.
    SetParam { name: String, value: Expr },
}

/// A builder for a list of timed actions.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub(crate) entries: Vec<(Time, Action)>,
}

impl Scenario {
    pub fn new() -> Self { Self::default() }
    pub fn at(mut self, t: Time, action: Action) -> Self {
        self.entries.push((t, action));
        self
    }
    /// Convenience: schedule a burst of injections at regular intervals.
    pub fn periodic_inject(
        mut self,
        start: Time,
        period: Time,
        count: usize,
        node: NodeId,
        payload: Value,
    ) -> Self {
        for i in 0..count {
            self.entries.push((
                start + (i as Time) * period,
                Action::Inject {
                    node,
                    payload: payload.clone(),
                    metadata: BTreeMap::new(),
                    return_path: Vec::new(),
                },
            ));
        }
        self
    }
}
