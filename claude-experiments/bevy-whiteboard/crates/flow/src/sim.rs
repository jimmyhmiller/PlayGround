use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::event::{Event, EventLog};
use crate::expr::Expr;
use crate::rule::Rule;
use crate::scenario::{Action, Scenario};
use crate::template::Template;
use crate::value::Value;

pub type Time = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PacketId(pub u64);

/// A packet flowing between nodes.
///
/// `from_edge` records which edge delivered this packet to the current
/// node's inbox — used for `When::Input { from: ... }` filtering and
/// for reply addressing. `reply_to` records the packet's requester; a
/// `Respond` effect sends back to that node.
#[derive(Debug, Clone)]
pub struct Packet {
    pub id: PacketId,
    pub payload: Value,
    pub from_edge: Option<EdgeId>,
    pub reply_to: Option<NodeId>,
    pub emitted_at_ns: Time,
}

/// A simulation node.
///
/// Nodes are either **leaves** (typical case — slots + rules + inbox
/// drive behavior) or **compounds** (containers whose behavior is
/// given by inner nodes wired via ports). A compound has empty
/// `slots`/`rules`/`inbox` and populates `compound` instead.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,

    // --- Leaf state (unused / empty for compounds) ---
    pub slots: BTreeMap<String, Value>,
    pub rules: Vec<Rule>,
    /// Packets delivered to this node, waiting to be matched by a rule.
    /// In-order FIFO. Unbounded in principle; if this grows without
    /// bound, you have modeled the system correctly and it is overloaded.
    pub inbox: VecDeque<Packet>,

    // --- Compound state (Some only for compounds) ---
    pub compound: Option<CompoundBody>,
}

impl Node {
    /// `true` iff this node is a compound (container).
    pub fn is_compound(&self) -> bool { self.compound.is_some() }
}

/// The container-specific body of a compound node. Maps port names to
/// inner nodes. Ports are the only interface between a compound's
/// internals and the outside world: external edges attach to ports,
/// inner leaves emit through ports using `EmitTo::ToOutPort`.
#[derive(Debug, Clone)]
pub struct CompoundBody {
    /// Port name → which inner node receives packets arriving on this port.
    pub in_ports: BTreeMap<String, NodeId>,
    /// Port name → which inner node is allowed to emit on this port.
    /// Emits on an out port are fanned over the compound's outgoing
    /// edges whose `from_port` matches.
    pub out_ports: BTreeMap<String, NodeId>,
}

/// A directed edge with a latency expression evaluated at emit time.
#[derive(Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    /// If `from` is a compound, the out-port name this edge attaches to.
    /// Inner-node emits on this port are fanned over edges with a
    /// matching `from_port`.
    pub from_port: Option<String>,
    pub to: NodeId,
    /// If `to` is a compound, the in-port name; delivery resolves via
    /// the compound's `in_ports` map to the inner node.
    pub to_port: Option<String>,
    /// Evaluated with bindings for the emitted packet's payload as `packet`,
    /// plus the source node's slots. Returns Int nanoseconds.
    pub latency_ns: Expr,
}

#[derive(Debug, Clone)]
pub(crate) struct Scheduled {
    pub arrives_at_ns: Time,
    pub packet: Packet,
    /// The edge this travel is associated with (for observability and
    /// `from_edge` on delivery). For request-response, the reply
    /// travels on the same edge but in reverse; `deliver_to` overrides
    /// `edge.to` so we don't need synthetic reverse edges.
    pub edge: EdgeId,
    pub deliver_to: NodeId,
}

impl PartialEq for Scheduled {
    fn eq(&self, other: &Self) -> bool { self.arrives_at_ns == other.arrives_at_ns }
}
impl Eq for Scheduled {}
impl PartialOrd for Scheduled {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for Scheduled {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Earlier-arrival compares "greater" so BinaryHeap (max-heap) with Reverse
        // pops the earliest first — but we avoid that dance by reversing here:
        self.arrives_at_ns.cmp(&other.arrives_at_ns)
    }
}

#[derive(Clone)]
pub struct Sim {
    pub nodes: BTreeMap<NodeId, Node>,
    pub edges: BTreeMap<EdgeId, Edge>,
    pub now_ns: Time,
    pub rng: StdRng,
    pub log: EventLog,
    pub(crate) in_flight: BinaryHeap<Reverse<Scheduled>>,
    pub(crate) next_node_id: u64,
    pub(crate) next_edge_id: u64,
    pub(crate) next_packet_id: u64,
    pub(crate) next_instance_seq: u64,
    pub(crate) next_scenario_seq: u64,
    pub templates: HashMap<String, Template>,
    /// Live, re-evaluated parameter namespace. Expressions can
    /// reference these via `Expr::Param(name)`; changing the binding
    /// here updates every reference on next evaluation.
    pub params: HashMap<String, Expr>,
    /// Pending scenario actions keyed by (time, insertion-seq).
    /// Min-heap: earliest time pops first; ties break by insertion order.
    pub(crate) pending_actions: BinaryHeap<Reverse<PendingAction>>,
    /// Safety cap to prevent infinite-loop rule firings at a single
    /// instant. A well-written model won't hit this; if it does, we
    /// want a loud error, not a hang.
    pub max_steps_per_instant: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PendingAction {
    pub at_ns: Time,
    pub seq: u64,
    pub action: Action,
}
impl PartialEq for PendingAction {
    fn eq(&self, o: &Self) -> bool { self.at_ns == o.at_ns && self.seq == o.seq }
}
impl Eq for PendingAction {}
impl PartialOrd for PendingAction {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for PendingAction {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.at_ns.cmp(&o.at_ns).then(self.seq.cmp(&o.seq))
    }
}

impl Sim {
    pub fn new(seed: u64) -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            now_ns: 0,
            rng: StdRng::seed_from_u64(seed),
            log: EventLog::new(100_000),
            in_flight: BinaryHeap::new(),
            next_node_id: 1,
            next_edge_id: 1,
            next_packet_id: 1,
            next_instance_seq: 1,
            next_scenario_seq: 1,
            templates: HashMap::new(),
            params: HashMap::new(),
            pending_actions: BinaryHeap::new(),
            max_steps_per_instant: 10_000,
        }
    }

    /// Bind or rebind a live parameter. Takes effect on the next
    /// expression evaluation that references it.
    pub fn set_param(&mut self, name: impl Into<String>, value: Expr) {
        self.params.insert(name.into(), value);
    }

    /// Schedule one action at the given sim time. Actions with the same
    /// time fire in insertion order.
    pub fn schedule_action(&mut self, at_ns: Time, action: Action) {
        let seq = self.next_scenario_seq;
        self.next_scenario_seq += 1;
        self.pending_actions.push(Reverse(PendingAction { at_ns, seq, action }));
    }

    /// Load all actions from a scenario onto the pending-actions heap.
    pub fn load_scenario(&mut self, s: Scenario) {
        for (t, a) in s.entries {
            self.schedule_action(t, a);
        }
    }

    pub fn register_template(&mut self, t: Template) {
        let key = t.name.clone();
        self.templates.insert(key, t);
    }

    pub fn add_node(&mut self, name: impl Into<String>, slots: BTreeMap<String, Value>, rules: Vec<Rule>) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = Node {
            id,
            name: name.into(),
            slots,
            rules,
            inbox: VecDeque::new(),
            parent: None,
            compound: None,
        };
        self.nodes.insert(id, node);
        id
    }

    /// Create a compound (container) node. Compound nodes have no
    /// rules of their own — their behavior is what their inner nodes
    /// do. `in_ports` and `out_ports` declare the interface.
    pub fn add_compound(
        &mut self,
        name: impl Into<String>,
        in_ports: BTreeMap<String, NodeId>,
        out_ports: BTreeMap<String, NodeId>,
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = Node {
            id,
            name: name.into(),
            slots: BTreeMap::new(),
            rules: Vec::new(),
            inbox: VecDeque::new(),
            parent: None,
            compound: Some(CompoundBody { in_ports: in_ports.clone(), out_ports: out_ports.clone() }),
        };
        self.nodes.insert(id, node);
        // Parent the ports' inner nodes to this compound (so they're contained).
        for inner in in_ports.values().chain(out_ports.values()) {
            if let Some(n) = self.nodes.get_mut(inner) {
                n.parent = Some(id);
            }
        }
        id
    }

    /// Re-parent a node to a compound. Any node referenced by a
    /// compound's port maps is automatically parented there at
    /// `add_compound`, but you can also nest arbitrary sibling
    /// nodes by calling this directly.
    pub fn reparent(&mut self, child: NodeId, parent: Option<NodeId>) {
        if let Some(n) = self.nodes.get_mut(&child) {
            n.parent = parent;
        }
    }

    /// Children of `nid` (nodes whose `parent` is `nid`), in id order.
    pub fn children_of(&self, nid: NodeId) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.values()
            .filter(|n| n.parent == Some(nid))
            .map(|n| n.id)
            .collect();
        ids.sort_by_key(|n| n.0);
        ids
    }

    /// Despawn a node and all edges touching it. In-flight packets to
    /// or from this node will be dropped at delivery time (the
    /// scheduler checks for missing nodes).
    ///
    /// Also scrubs any `Value::NodeRef(nid)` from the *other* nodes'
    /// slots — users like `pending_pull` / `upstream` / `downstream`
    /// hold weak references and would otherwise become stale and route
    /// emits into the void. Replacing them with `Nil` lets subsequent
    /// `ToTargetExpr(slot(...))` calls fall through the silent-drop
    /// path instead of chasing a ghost.
    pub fn despawn_node(&mut self, nid: NodeId) {
        let Some(_) = self.nodes.remove(&nid) else { return; };
        self.edges.retain(|_, e| e.from != nid && e.to != nid);
        for node in self.nodes.values_mut() {
            for (_, v) in node.slots.iter_mut() {
                if matches!(v, Value::NodeRef(r) if *r == nid) {
                    *v = Value::Nil;
                }
            }
        }
        self.log.push(Event::NodeDespawned { node: nid, at_ns: self.now_ns });
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId, latency_ns: Expr) -> EdgeId {
        self.add_edge_ports(from, None, to, None, latency_ns)
    }

    /// Create an edge whose endpoints may reference compound ports.
    /// Use `from_port = Some("foo")` if `from` is a compound emitting
    /// on its "foo" out-port, and similarly `to_port` for a compound's
    /// in-port.
    pub fn add_edge_ports(
        &mut self,
        from: NodeId,
        from_port: Option<String>,
        to: NodeId,
        to_port: Option<String>,
        latency_ns: Expr,
    ) -> EdgeId {
        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;
        self.edges.insert(id, Edge { id, from, from_port, to, to_port, latency_ns });
        id
    }

    pub fn node_by_name(&self, name: &str) -> Option<NodeId> {
        self.nodes.values().find(|n| n.name == name).map(|n| n.id)
    }

    pub(crate) fn next_packet_id(&mut self) -> PacketId {
        let id = PacketId(self.next_packet_id);
        self.next_packet_id += 1;
        id
    }

    /// Inject a packet directly into a node's inbox at the current clock.
    /// Useful for scripted scenarios.
    pub fn inject(&mut self, to: NodeId, payload: Value, reply_to: Option<NodeId>) -> PacketId {
        let id = self.next_packet_id();
        let pkt = Packet {
            id,
            payload,
            from_edge: None,
            reply_to,
            emitted_at_ns: self.now_ns,
        };
        self.log.push(Event::PacketDelivered { packet: id, to, at_ns: self.now_ns });
        self.nodes.get_mut(&to).expect("inject: node not found").inbox.push_back(pkt);
        id
    }

    /// Outbound edges from a node, in the order they were added.
    pub fn outbound(&self, n: NodeId) -> Vec<EdgeId> {
        let mut v: Vec<EdgeId> = self.edges.values().filter(|e| e.from == n).map(|e| e.id).collect();
        v.sort_by_key(|e| e.0);
        v
    }

    /// The earliest scheduled event time across in-flight deliveries
    /// and pending scenario actions, or `None` if the sim is
    /// quiescent. Useful for "step to next event" UX.
    pub fn next_event_time_ns(&self) -> Option<Time> {
        let a = self.in_flight.peek().map(|r| r.0.arrives_at_ns);
        let b = self.pending_actions.peek().map(|r| r.0.at_ns);
        match (a, b) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        }
    }
}
