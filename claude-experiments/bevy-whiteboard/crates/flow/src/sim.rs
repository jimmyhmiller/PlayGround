use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;

use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// The RNG type used throughout the sim. A thin alias for
/// `rand_chacha::ChaCha12Rng` — the same RNG `rand::rngs::StdRng`
/// wraps, but named directly so it picks up `rand_chacha`'s `serde1`
/// feature (the wrapper type doesn't expose it). Snapshot fidelity
/// requires the RNG state to round-trip, so we use the concrete type.
pub type SimRng = rand_chacha::ChaCha12Rng;

use crate::event::{Event, EventLog};
use crate::expr::Expr;
use crate::rule::Rule;
use crate::scenario::{Action, Scenario};
use crate::template::{Probe, ProbePart, Template};
use crate::value::Value;

pub type Time = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PacketId(pub u64);

/// A packet flowing between nodes.
///
/// `from_edge` records which edge delivered this packet to the current
/// node's inbox — used for `When::Input { from: ... }` filtering.
///
/// `metadata` and `return_path` are **always preserved** across hops:
/// when a rule fires on a consumed packet, emits inherit both fields
/// by default. Rules opt in to mutation explicitly (push/pop/set/remove)
/// — there is no engine-level overwriting. Empty `Vec` and empty
/// `BTreeMap` are zero-heap defaults, so packets that don't use either
/// mechanism pay no allocation cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Packet {
    pub id: PacketId,
    pub payload: Value,
    pub from_edge: Option<EdgeId>,
    /// Arbitrary string-keyed metadata. Rule authors decide what keys
    /// mean (trace ids, deadlines, auth tokens, …). Preserved through
    /// all emits that inherit from a consumed packet.
    pub metadata: BTreeMap<String, Value>,
    /// Stack of nodes that want to receive the response. Head is the
    /// innermost caller; `popping` an emit removes the head. Empty for
    /// packets that are fire-and-forget. Preserved through all emits
    /// that don't explicitly push/pop/replace.
    pub return_path: Vec<NodeId>,
    pub emitted_at_ns: Time,
}

/// A simulation node.
///
/// Nodes are either **leaves** (typical case — slots + rules + inbox
/// drive behavior) or **compounds** (containers whose behavior is
/// given by inner nodes wired via ports). A compound has empty
/// `slots`/`rules`/`inbox` and populates `compound` instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub parent: Option<NodeId>,
    /// Name of the class (template) this node was instantiated from,
    /// or `None` for nodes built via the pre-DSL imperative API or for
    /// compounds (which aren't instantiated from templates). Loaders
    /// use this to map sim nodes back to their visual kind — the
    /// instance name alone is unreliable once a class can be spawned
    /// multiple times (suffix `_N`).
    #[serde(default)]
    pub class: Option<String>,

    // --- Leaf state (unused / empty for compounds) ---
    pub slots: BTreeMap<String, Value>,
    pub rules: Vec<Rule>,
    /// Packets delivered to this node, waiting to be matched by a rule.
    /// In-order FIFO. Unbounded in principle; if this grows without
    /// bound, you have modeled the system correctly and it is overloaded.
    pub inbox: VecDeque<Packet>,
    /// Probes declared on this node's class. Copied from the template
    /// at spawn time. Empty for compounds and for nodes built by the
    /// pre-DSL imperative API.
    pub probes: Vec<Probe>,

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundBody {
    /// Port name → which inner node receives packets arriving on this port.
    pub in_ports: BTreeMap<String, NodeId>,
    /// Port name → which inner node is allowed to emit on this port.
    /// Emits on an out port are fanned over the compound's outgoing
    /// edges whose `from_port` matches.
    pub out_ports: BTreeMap<String, NodeId>,
}

/// A directed edge with a latency expression evaluated at emit time.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Engine-managed monotonic sequence number of the most recent
    /// forward-direction emit on this edge. Incremented from a single
    /// sim-wide counter so every emission gets a unique value — sim
    /// time alone can't disambiguate multiple emits in the same tick.
    /// `None` until first traversal; higher values mean "sent more
    /// recently", so LRU / round-robin picks the edge with the
    /// smallest value (treating `None` as "smaller than any sent").
    /// Reverse-route replies do not tick this — it tracks *forward*
    /// load only.
    pub last_sent_seq: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
pub struct Sim {
    pub nodes: BTreeMap<NodeId, Node>,
    pub edges: BTreeMap<EdgeId, Edge>,
    pub now_ns: Time,
    pub rng: SimRng,
    pub log: EventLog,
    #[serde(with = "reverse_heap_serde")]
    pub(crate) in_flight: BinaryHeap<Reverse<Scheduled>>,
    pub(crate) next_node_id: u64,
    pub(crate) next_edge_id: u64,
    pub(crate) next_packet_id: u64,
    /// Monotonic counter ticked per forward-direction emit; written to
    /// `Edge.last_sent_seq` so LRU / round-robin routing can break
    /// same-tick ties. `1` is the first emission so `0` can act as a
    /// "definitely older than any real send" sentinel in read paths
    /// that don't want an `Option`.
    pub(crate) next_emit_seq: u64,
    pub(crate) next_instance_seq: u64,
    pub(crate) next_scenario_seq: u64,
    pub templates: HashMap<String, Template>,
    /// Named scenario library. Populated by the DSL lowerer so callers
    /// can pick which scenario to run (via [`Sim::run_scenario`])
    /// instead of all scenarios firing at load. The DSL's single
    /// unnamed `scenario { … }` block is stored under the key "main".
    pub scenarios: HashMap<String, crate::scenario::Scenario>,
    /// Live, re-evaluated parameter namespace. Expressions can
    /// reference these via `Expr::Param(name)`; changing the binding
    /// here updates every reference on next evaluation.
    pub params: HashMap<String, Expr>,
    /// Pending scenario actions keyed by (time, insertion-seq).
    /// Min-heap: earliest time pops first; ties break by insertion order.
    #[serde(with = "reverse_heap_serde")]
    pub(crate) pending_actions: BinaryHeap<Reverse<PendingAction>>,
    /// Safety cap to prevent infinite-loop rule firings at a single
    /// instant. A well-written model won't hit this; if it does, we
    /// want a loud error, not a hang.
    pub max_steps_per_instant: usize,
    /// Running counts of runtime errors, keyed by `kind` string.
    /// Incremented alongside `Event::RuntimeError` emission. Lets the
    /// UI / tests surface aggregate error rates without scraping the
    /// event log.
    pub error_counts: BTreeMap<String, u64>,
    /// User-editable scenario timeline. Plays the same role as a
    /// pre-loaded `Scenario`, but lives editable on the sim so a UI
    /// can add / remove events at run time. The engine's `run_until`
    /// fires due timeline events as part of normal time advancement;
    /// nothing about the timeline mechanism involves the host UI.
    #[serde(default)]
    pub timeline: crate::timeline::Timeline,
}

/// Serde adapter for `BinaryHeap<Reverse<T>>` — serializes as a flat
/// `Vec<T>` (losing the Reverse-wrap and the heap order, both of which
/// are reconstructed on deserialize). The engine's heap invariant (min
/// by `T::cmp`) is reestablished on load because `BinaryHeap::from(Vec)`
/// heapifies. We drop the heap order on purpose — it's redundant with
/// `T::cmp` and storing it would be inviting skew.
mod reverse_heap_serde {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, T>(
        heap: &BinaryHeap<Reverse<T>>,
        ser: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Ord,
    {
        let v: Vec<&T> = heap.iter().map(|r| &r.0).collect();
        v.serialize(ser)
    }

    pub fn deserialize<'de, D, T>(
        de: D,
    ) -> Result<BinaryHeap<Reverse<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de> + Ord,
    {
        let v: Vec<T> = Vec::deserialize(de)?;
        Ok(v.into_iter().map(Reverse).collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
            rng: SimRng::seed_from_u64(seed),
            log: EventLog::new(100_000),
            in_flight: BinaryHeap::new(),
            next_node_id: 1,
            next_edge_id: 1,
            next_packet_id: 1,
            next_emit_seq: 1,
            next_instance_seq: 1,
            next_scenario_seq: 1,
            templates: HashMap::new(),
            scenarios: HashMap::new(),
            params: HashMap::new(),
            pending_actions: BinaryHeap::new(),
            max_steps_per_instant: 10_000,
            error_counts: BTreeMap::new(),
            timeline: crate::timeline::Timeline::new(),
        }
    }

    /// Record a runtime error: increments `error_counts[kind]` and
    /// emits a `RuntimeError` event. Call this instead of panicking
    /// when the failure is user-configurable (bad expression type,
    /// missing edge, empty return-path pop, etc.) — the sim keeps
    /// running and the caller silently drops the offending effect.
    pub(crate) fn record_error(
        &mut self,
        kind: &str,
        node: Option<NodeId>,
        rule: Option<&str>,
        detail: impl Into<String>,
    ) {
        *self.error_counts.entry(kind.to_string()).or_insert(0) += 1;
        self.log.push(Event::RuntimeError {
            kind: kind.to_string(),
            node,
            rule: rule.map(|s| s.to_string()),
            detail: detail.into(),
            at_ns: self.now_ns,
        });
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

    /// Load all actions from a scenario. `SetSlot` actions are routed
    /// into `self.timeline` so they appear as user-visible events on
    /// the UI strip; everything else goes onto the pending-actions
    /// heap. Multiple `SetSlot`s sharing an `at_ns` are grouped into a
    /// single compound timeline event — that's what the user sees as
    /// "at t=2s, change A.x AND B.y."
    pub fn load_scenario(&mut self, s: Scenario) {
        use std::collections::BTreeMap;
        let mut grouped: BTreeMap<Time, Vec<crate::timeline::TimelineAction>> = BTreeMap::new();
        for (t, a) in s.entries {
            match a {
                Action::SetSlot { node, slot, value } => {
                    grouped.entry(t).or_default().push(
                        crate::timeline::TimelineAction { node, slot, value }
                    );
                }
                other => self.schedule_action(t, other),
            }
        }
        for (at_ns, actions) in grouped {
            self.timeline.schedule_compound(at_ns, actions);
        }
    }

    /// Schedule every action from a named scenario in the sim's
    /// scenario library. Returns `Err` if no such scenario is registered.
    /// Use this when loading a canvas to start from a specific scenario
    /// instead of the default auto-scheduled one.
    pub fn run_scenario(&mut self, name: &str) -> Result<(), String> {
        let sc = self.scenarios.get(name).cloned()
            .ok_or_else(|| format!("no scenario named `{}`", name))?;
        self.load_scenario(sc);
        Ok(())
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
            probes: Vec::new(),
            class: None,
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
            probes: Vec::new(),
            class: None,
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
        self.edges.insert(id, Edge {
            id, from, from_port, to, to_port, latency_ns,
            last_sent_seq: None,
        });
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
    /// Useful for scripted scenarios. `metadata` and `return_path` both
    /// default to empty — most injections are fire-and-forget.
    pub fn inject(&mut self, to: NodeId, payload: Value) -> PacketId {
        self.inject_with(to, payload, BTreeMap::new(), Vec::new())
    }

    /// Inject a packet carrying specific metadata and/or return_path.
    /// Use this when seeding a request that expects a response from the
    /// graph, or pre-populating metadata for a scenario-driven test.
    pub fn inject_with(
        &mut self,
        to: NodeId,
        payload: Value,
        metadata: BTreeMap<String, Value>,
        return_path: Vec<NodeId>,
    ) -> PacketId {
        let id = self.next_packet_id();
        let pkt = Packet {
            id,
            payload,
            from_edge: None,
            metadata,
            return_path,
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

    /// Probe labels declared on a node's class, in declaration order.
    /// Empty for compounds and for nodes without declared probes.
    pub fn probe_labels(&self, nid: NodeId) -> Vec<String> {
        self.nodes.get(&nid)
            .map(|n| n.probes.iter().map(|p| p.label.clone()).collect())
            .unwrap_or_default()
    }

    /// Evaluate a single probe on a node and return its formatted string.
    /// Returns `None` if the node or probe label doesn't exist.
    ///
    /// Evaluation reads slots, params, and other nodes' state but never
    /// mutates sim state — a private RNG clone absorbs any distribution
    /// samples so probe reads don't perturb the sim's seeded sequence.
    pub fn probe_reading(&self, nid: NodeId, label: &str) -> Option<String> {
        let node = self.nodes.get(&nid)?;
        let probe = node.probes.iter().find(|p| p.label == label)?;
        Some(self.eval_probe_parts(nid, &probe.parts))
    }

    /// Evaluate every probe on a node. Stable order matches declaration.
    pub fn probe_readings(&self, nid: NodeId) -> Vec<(String, String)> {
        let Some(node) = self.nodes.get(&nid) else { return Vec::new(); };
        node.probes.iter()
            .map(|p| (p.label.clone(), self.eval_probe_parts(nid, &p.parts)))
            .collect()
    }

    fn eval_probe_parts(&self, nid: NodeId, parts: &[ProbePart]) -> String {
        let slots = self.nodes[&nid].slots.clone();
        let bindings = crate::value::Bindings::new();
        let mut pstack: Vec<String> = Vec::new();
        let mut rng = self.rng.clone();
        let mut out = String::new();
        for part in parts {
            match part {
                ProbePart::Literal(s) => out.push_str(s),
                ProbePart::Hole(e) => {
                    let mut ctx = crate::expr::EvalCtx {
                        bindings: &bindings,
                        slots: &slots,
                        now_ns: self.now_ns,
                        rng: &mut rng,
                        current_node: Some(nid),
                        params: &self.params,
                        param_stack: &mut pstack,
                        nodes: &self.nodes,
                        edges: &self.edges,
                        packet: None,
                    };
                    let v = e.eval(&mut ctx);
                    out.push_str(&format_probe_value(&v));
                }
            }
        }
        out
    }
}

/// Default formatting for a probe hole value. Keeps output compact and
/// human-readable without needing per-probe format hints. Rule of thumb:
/// ints print as-is; floats drop trailing zeros; strings print raw;
/// Nil becomes an em-dash placeholder.
fn format_probe_value(v: &Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => {
            if !f.is_finite() { return "—".into(); }
            if f.abs() >= 10.0 { format!("{:.0}", f) }
            else { format!("{:.1}", f) }
        }
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => s.clone(),
        Value::Nil => "—".into(),
        other => format!("{:?}", other),
    }
}
