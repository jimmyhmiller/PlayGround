use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
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

/// Index into `Sim.templates`. Keys nodes to their class without a
/// per-fire string lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TemplateId(pub u32);

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
    /// Index of the class (template) this node was instantiated from,
    /// or `None` for compounds and rules-less leaves. The string name
    /// is reachable via [`Sim::class_name`] when needed (logs, UI,
    /// snapshot serialization); the hot path uses the integer.
    #[serde(default)]
    pub class: Option<TemplateId>,

    // --- Leaf state (unused / empty for compounds) ---
    pub slots: BTreeMap<String, Value>,
    /// Edges where this node is the source, in `EdgeId` order. Owned
    /// here so routing primitives (`out_neighbors()`, `EmitTo::*`)
    /// don't have to scan the entire `Sim.edges` map. Maintained by
    /// `add_edge_ports` (push) and `despawn_node` (drain + scrub
    /// references in other nodes' lists).
    #[serde(default)]
    pub outbound: Vec<EdgeId>,
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
pub struct Scheduled {
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
    pub in_flight: BinaryHeap<Reverse<Scheduled>>,
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
    /// Templates indexed by [`TemplateId`]. `Node.class` carries the
    /// id; runtime template lookups go straight through this Vec
    /// (one bounds check) instead of the previous string-hashed
    /// HashMap.
    pub templates: Vec<Template>,
    /// Name → id index, populated alongside `templates` by
    /// [`Sim::register_template`]. Only used at canvas-load time
    /// (`instantiate(class_name, ...)`) and for the few external
    /// "does this canvas have class X" checks. Hot path doesn't
    /// touch this map.
    #[serde(default)]
    pub template_by_name: HashMap<String, TemplateId>,
    /// Compound-class library. A `compound Foo { ... }` block registered
    /// via [`crate::dsl::register_classes`] is stored here unexpanded;
    /// [`Sim::instantiate`] expands the recipe on demand into the live
    /// sim, with the instance name as the per-instance prefix for all
    /// inner nodes. Lets the palette spawn composites the same way it
    /// spawns leaf classes.
    ///
    /// Skipped during serde because [`crate::dsl::ast::CompoundDecl`]
    /// isn't (de)serializable today — restoring a snapshot loses the
    /// compound registry, which is fine because snapshots reload the
    /// DSL source separately during canvas re-hydration. Tests that
    /// round-trip through snapshot need to re-register if they
    /// instantiate compounds afterwards.
    #[serde(skip)]
    pub compound_templates: HashMap<String, crate::dsl::ast::CompoundDecl>,
    /// `instance NodeId → compound class name` map for shim nodes
    /// produced by [`Sim::instantiate_compound`]. Lets external code
    /// (palette UX, has_color_slot, inspector) recover which compound
    /// recipe spawned a given shim without re-deriving from its inner
    /// nodes. Empty for any compound created via the inline DSL path
    /// (those don't track an originating class — their interior was
    /// authored directly).
    #[serde(skip)]
    pub compound_class_of: HashMap<NodeId, String>,
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
    /// instant. Set high enough that a legitimately large board
    /// (Life 100×100 = 10k cells, each firing once per tick) never
    /// trips it; only a genuine zero-latency cycle should. `usize::MAX`
    /// effectively disables the check.
    pub max_steps_per_instant: usize,
    /// Scratch buffer used by the fire loop to snapshot node ids each
    /// time it scans the world. Reusing the allocation avoids a fresh
    /// `Vec` per `try_fire_one` call (8000+ calls per generation on a
    /// 30×30 grid).
    #[serde(skip)]
    pub(crate) fire_iter_buf: Vec<NodeId>,
    /// Worklist of nodes that currently have packets in their inbox or
    /// whose templates have a source-style rule (must always be
    /// rechecked). The fire loop iterates this set instead of the
    /// full `nodes` map — turns the per-firing scan from O(N) to
    /// O(active). Sorted by `NodeId` so the firing order is identical
    /// to the previous "iterate all by id" behavior.
    #[serde(skip)]
    pub(crate) fireable: BTreeSet<NodeId>,
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
    /// Per-phase wall-clock samples accumulated during `run_until`.
    /// Each entry is `(phase_name, microseconds)`. The host drains
    /// this after each `run_until` call (see `drain_perf_samples`)
    /// to feed its own perf-tracking layer; the sim itself only
    /// pushes — never reads or trims. Drained vec stays at its
    /// cap allocation so steady-state runs don't reallocate.
    #[serde(skip)]
    pub perf_samples: Vec<(&'static str, f64)>,
    /// Global multiplier applied to every edge latency at emit time.
    /// Used by the host to "slow down packet transit" without slowing
    /// the rest of the sim (rule firings, scenario actions). 1.0 is
    /// neutral. Changes also rescale the remaining travel time of
    /// every in-flight packet — see `Sim::set_edge_latency_scale` —
    /// so the visible effect is uniform across already-emitted and
    /// future packets.
    #[serde(default = "default_edge_latency_scale")]
    pub edge_latency_scale: f64,
}

fn default_edge_latency_scale() -> f64 { 1.0 }

/// Per-step floor (ns) added to the edge-latency scale. At
/// `edge_latency_scale = S > 1`, every edge's effective latency is at
/// least `(S - 1) * EDGE_LATENCY_SCALE_FLOOR_NS`. This makes
/// 0-latency edges (broadcast fan-outs, "wires" with no declared
/// time cost) still visibly slow down when the user cranks the
/// scale; without it `0 * S = 0` would leave them instant. At
/// `S = 1` the floor is 0 and behavior is unchanged.
pub const EDGE_LATENCY_SCALE_FLOOR_NS: u64 = 50_000_000;

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
            templates: Vec::new(),
            template_by_name: HashMap::new(),
            compound_templates: HashMap::new(),
            compound_class_of: HashMap::new(),
            scenarios: HashMap::new(),
            params: HashMap::new(),
            pending_actions: BinaryHeap::new(),
            max_steps_per_instant: usize::MAX,
            fire_iter_buf: Vec::new(),
            fireable: BTreeSet::new(),
            error_counts: BTreeMap::new(),
            timeline: crate::timeline::Timeline::new(),
            perf_samples: Vec::new(),
            edge_latency_scale: 1.0,
        }
    }

    /// Set the edge-latency multiplier and rescale every in-flight
    /// packet's remaining travel time so the change feels uniform.
    /// `new = now + (arrives - now) * (new_scale / old_scale)`.
    /// No-op if the scale is unchanged or non-positive (we refuse to
    /// invert / zero out time).
    pub fn set_edge_latency_scale(&mut self, new_scale: f64) {
        if !(new_scale.is_finite() && new_scale > 0.0) {
            return;
        }
        let old = self.edge_latency_scale;
        if (old - new_scale).abs() < f64::EPSILON {
            return;
        }
        let factor = new_scale / old;
        let now = self.now_ns;
        let drained: Vec<Scheduled> = self.in_flight.drain().map(|r| r.0).collect();
        let mut rescaled: Vec<Reverse<Scheduled>> = Vec::with_capacity(drained.len());
        for mut s in drained {
            let remaining = s.arrives_at_ns.saturating_sub(now);
            let new_rem = (remaining as f64 * factor).round() as u64;
            s.arrives_at_ns = now.saturating_add(new_rem);
            rescaled.push(Reverse(s));
        }
        self.in_flight = BinaryHeap::from(rescaled);
        self.edge_latency_scale = new_scale;
    }

    /// Drain the per-phase timing samples accumulated during the most
    /// recent `run_until` call. Each entry is `(phase_name, micros)`.
    /// Called by the host once per frame so it can roll the samples
    /// into its own diagnostics layer; clears the buffer in place so
    /// the next `run_until` starts fresh.
    pub fn drain_perf_samples(&mut self) -> std::vec::Drain<'_, (&'static str, f64)> {
        self.perf_samples.drain(..)
    }

    /// Record one phase sample. Inline so non-debug builds don't pay
    /// for an indirect call. `Instant::elapsed` is a couple of
    /// `clock_gettime` syscalls — fine at the granularity we're
    /// timing (whole phases of `run_until`, not per-rule firings).
    #[inline]
    pub(crate) fn record_phase_us(&mut self, name: &'static str, start: std::time::Instant) {
        let us = start.elapsed().as_secs_f64() * 1_000_000.0;
        self.perf_samples.push((name, us));
    }

    /// Mark `nid` as needing a fire-scan. Idempotent. Called whenever
    /// a packet is injected or delivered, and at node-spawn time for
    /// templates with a source-style rule.
    pub(crate) fn mark_fireable(&mut self, nid: NodeId) {
        self.fireable.insert(nid);
    }

    /// Recompute whether `nid` should still be in `fireable`: keep it
    /// if its inbox has a packet OR its template carries a source
    /// rule. Otherwise drop it. Called after a fire consumes a packet
    /// and we need to know if more work remains.
    pub(crate) fn refresh_fireable(&mut self, nid: NodeId) {
        let keep = match self.nodes.get(&nid) {
            Some(node) => {
                let has_inbox = !node.inbox.is_empty();
                let has_source = node
                    .class
                    .and_then(|tid| self.templates.get(tid.0 as usize))
                    .map(|t| t.has_source_rule)
                    .unwrap_or(false);
                has_inbox || has_source
            }
            None => false,
        };
        if keep {
            self.fireable.insert(nid);
        } else {
            self.fireable.remove(&nid);
        }
    }

    /// Resolve a node's class name (the human-readable label registered
    /// in the DSL). Slow path; use [`Node.class`] directly when you
    /// have a `TemplateId` already.
    pub fn class_name(&self, nid: NodeId) -> Option<&str> {
        let tid = self.nodes.get(&nid)?.class?;
        Some(self.templates.get(tid.0 as usize)?.name.as_str())
    }

    /// Look up a class by its DSL name. Returns the template id, or
    /// `None` if no such class is registered.
    pub fn template_id_by_name(&self, name: &str) -> Option<TemplateId> {
        self.template_by_name.get(name).copied()
    }

    /// Convenience: did we register a class (leaf template OR compound)
    /// with this DSL name?
    pub fn has_class(&self, name: &str) -> bool {
        self.template_by_name.contains_key(name)
            || self.compound_templates.contains_key(name)
    }

    /// Did we register a *compound* class with this name? Distinct from
    /// `has_class` so callers can branch on which expansion path to use
    /// (the leaf path goes through `templates`; compounds go through
    /// on-demand expansion into the existing sim).
    pub fn has_compound_class(&self, name: &str) -> bool {
        self.compound_templates.contains_key(name)
    }

    /// Walk a NodeId up the compound-parent chain to find the outermost
    /// ancestor that's still itself a compound (or the node itself if
    /// it isn't inside a compound). Lets tests pattern-match
    /// `PacketEmitted { from, .. }` events against a compound shim id
    /// even though the actual emit comes from one of the shim's inner
    /// nodes. Returns `nid` unchanged for monolithic / unparented
    /// nodes.
    pub fn compound_outermost(&self, nid: NodeId) -> NodeId {
        let mut cur = nid;
        while let Some(parent) = self.nodes.get(&cur).and_then(|n| n.parent) {
            cur = parent;
        }
        cur
    }

    /// Read a slot from `nid` or, if `nid` is a compound shim, walk into
    /// its prefixed inner nodes (`<shim.name>::*`) and return the first
    /// inner that declares the slot. Used by tests and inspector code
    /// that want the equivalent of "the node named X's `period_ns`"
    /// regardless of whether X is a monolithic leaf or a composite —
    /// for composites the slot actually lives on an inner like
    /// `client::T`.
    pub fn read_slot_resolved<'a>(&'a self, nid: NodeId, slot: &str) -> Option<&'a Value> {
        if let Some(n) = self.nodes.get(&nid) {
            if let Some(v) = n.slots.get(slot) { return Some(v); }
            let prefix = format!("{}::", n.name);
            for inner in self.nodes.values() {
                if inner.name.starts_with(&prefix) {
                    if let Some(v) = inner.slots.get(slot) { return Some(v); }
                }
            }
        }
        None
    }

    /// Write a slot on `nid` or its compound children. Same resolution
    /// rules as `read_slot_resolved`: direct hit wins, otherwise
    /// propagate to every prefixed inner that declares the slot.
    /// Returns the number of nodes mutated (0 = no slot of that name
    /// reachable). Tests use this to set e.g. a composite Worker's
    /// `up = 0` without knowing which inner owns the slot. Named
    /// `_resolved` to distinguish from the engine-internal `write_slot`
    /// rule-firing path which writes one specific node unconditionally.
    pub fn write_slot_resolved(&mut self, nid: NodeId, slot: &str, value: Value) -> usize {
        if let Some(n) = self.nodes.get_mut(&nid) {
            if n.slots.contains_key(slot) {
                n.slots.insert(slot.to_string(), value);
                return 1;
            }
        }
        let prefix = self.nodes.get(&nid).map(|n| format!("{}::", n.name));
        let Some(prefix) = prefix else { return 0 };
        let targets: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.name.starts_with(&prefix) && n.slots.contains_key(slot))
            .map(|(id, _)| *id)
            .collect();
        let count = targets.len();
        for id in targets {
            if let Some(n) = self.nodes.get_mut(&id) {
                n.slots.insert(slot.to_string(), value.clone());
            }
        }
        count
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

    /// User-side slot edit. Writes the slot AND emits a
    /// `UserSlotEdit` boundary event so visual layers can drop their
    /// future-queued backlog (same effect as a fired timeline event).
    /// UI code should prefer this over `nodes[id].slots.insert(...)`
    /// for human-driven changes — toggles, slider drags, etc.
    /// Tests that just want to seed initial state can still poke
    /// `slots` directly to avoid logging spurious boundaries.
    pub fn user_edit_slot(&mut self, node: NodeId, slot: impl Into<String>, value: Value) {
        let slot_name = slot.into();
        let at_ns = self.now_ns;
        // For compound shims the slot lives on an inner node (e.g.
        // `period_ns` is on `<gen>::T`). Use write_slot_resolved to
        // walk into prefixed children. If the slot is on the node
        // itself, that path is also a 1-node write.
        let touched = self.write_slot_resolved(node, &slot_name, value.clone());
        if touched == 0 {
            // No slot of that name anywhere — preserve the legacy
            // behaviour of creating it on the target node so a fresh
            // user-edit doesn't silently disappear.
            if let Some(n) = self.nodes.get_mut(&node) {
                n.slots.insert(slot_name.clone(), value.clone());
            }
        }
        self.log.push(crate::event::Event::SlotWritten {
            node, slot: slot_name.clone(), value: value.clone(), at_ns,
        });
        self.log.push(crate::event::Event::UserSlotEdit {
            node, slot: slot_name, value, at_ns,
        });
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

    pub fn register_template(&mut self, mut t: Template) -> TemplateId {
        // Templates from outside (DSL lowering, hand-built builders)
        // may not have run the cached-flag computation. Recompute
        // here so the fire loop's fast-skip is always correct.
        t.refresh_has_source_rule();
        let name = t.name.clone();
        // Re-registering a class with the same name overwrites the
        // existing entry in place — same id, fresh body. Matches the
        // previous HashMap semantics where `insert(key, t)` replaced
        // the value.
        if let Some(&existing) = self.template_by_name.get(&name) {
            self.templates[existing.0 as usize] = t;
            return existing;
        }
        let id = TemplateId(self.templates.len() as u32);
        self.templates.push(t);
        self.template_by_name.insert(name, id);
        id
    }

    /// Append a rule to the class that backs this node. If the node has
    /// a shared class, every instance picks up the new rule — that's
    /// the cost of rules-on-templates and matches how a builder
    /// modifying a real class would behave. Returns `Err` if the node
    /// has no class (compound or rules-less leaf).
    pub fn add_rule_to_node(&mut self, nid: NodeId, rule: Rule) -> Result<(), String> {
        let tid = self
            .nodes
            .get(&nid)
            .and_then(|n| n.class)
            .ok_or_else(|| format!("node {:?} has no class to attach a rule to", nid))?;
        let template = self
            .templates
            .get_mut(tid.0 as usize)
            .ok_or_else(|| format!("template id {:?} for node {:?} is missing", tid, nid))?;
        if !crate::template::rule_has_input(&rule) {
            template.has_source_rule = true;
        }
        template.rules.push(rule);
        Ok(())
    }

    pub fn add_node(&mut self, name: impl Into<String>, slots: BTreeMap<String, Value>, rules: Vec<Rule>) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        // Rules now live on templates, not on instances. Inline rules
        // get a private synthetic template named `__inline_<id>` so
        // there's still a single place to look them up at fire time.
        // Compound nodes and any node passing an empty rules vec end
        // up with `class: None` and never enter the rule-fire path.
        let class = if rules.is_empty() {
            None
        } else {
            let class_name = format!("__inline_{}", id.0);
            let mut tpl = Template {
                name: class_name.clone(),
                node_name_prefix: class_name.clone(),
                slots: BTreeMap::new(),
                rules,
                edges: Vec::new(),
                initial_packets: Vec::new(),
                probes: Vec::new(),
                has_source_rule: false,
            };
            tpl.refresh_has_source_rule();
            Some(self.register_template(tpl))
        };
        let node = Node {
            id,
            name: name.into(),
            slots,
            outbound: Vec::new(),
            inbox: VecDeque::new(),
            probes: Vec::new(),
            class,
            parent: None,
            compound: None,
        };
        self.nodes.insert(id, node);
        // Source-rule templates need to be in the worklist from spawn —
        // they fire even with an empty inbox.
        let has_source = class
            .and_then(|tid| self.templates.get(tid.0 as usize))
            .map(|t| t.has_source_rule)
            .unwrap_or(false);
        if has_source {
            self.fireable.insert(id);
        }
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
            outbound: Vec::new(),
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
    /// Tear down every node whose name is prefixed by `<compound_name>::`,
    /// plus their edges, plus optionally the compound body itself.
    /// Returns the set of removed `NodeId`s in deterministic order so
    /// the visual layer can mirror the demolition (despawn matching
    /// Bevy entities, drop traveling-packet rows on those edges, etc.).
    ///
    /// Used by the live-edit path: the canvas captures this before
    /// re-lowering a fresh interior, and the same set drives Bevy
    /// entity cleanup.
    pub fn despawn_compound_interior(
        &mut self,
        compound_name: &str,
        include_self: bool,
    ) -> Vec<NodeId> {
        let prefix = format!("{}::", compound_name);
        let mut targets: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| {
                n.name.starts_with(&prefix)
                    || (include_self && n.name == compound_name)
            })
            .map(|(id, _)| *id)
            .collect();
        // Deterministic — `nodes` is a BTreeMap so the iter order is
        // already stable; collecting NodeIds preserves it.
        targets.sort();
        for nid in &targets {
            self.despawn_node(*nid);
        }
        targets
    }

    pub fn despawn_node(&mut self, nid: NodeId) {
        let Some(_) = self.nodes.remove(&nid) else { return; };
        self.fireable.remove(&nid);
        // Drop edges incident to the despawned node and remember which
        // ones we removed so we can scrub them from other nodes'
        // outbound lists in one pass.
        let mut removed_edges: Vec<EdgeId> = Vec::new();
        self.edges.retain(|eid, e| {
            let keep = e.from != nid && e.to != nid;
            if !keep {
                removed_edges.push(*eid);
            }
            keep
        });
        if !removed_edges.is_empty() {
            for node in self.nodes.values_mut() {
                node.outbound.retain(|eid| !removed_edges.contains(eid));
            }
        }
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
        // Maintain the from-node's outbound list. Edge ids are
        // monotonically increasing, so push preserves ascending
        // order — `outbound()` doesn't need to sort.
        if let Some(n) = self.nodes.get_mut(&from) {
            n.outbound.push(id);
        }
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
        self.mark_fireable(to);
        id
    }

    /// Outbound edges from a node, in the order they were added (which
    /// is `EdgeId` order, since ids are assigned monotonically).
    /// Now reads `Node.outbound` directly — O(1) instead of an O(E)
    /// scan over the whole edge map. Returns an empty slice for
    /// missing nodes.
    pub fn outbound(&self, n: NodeId) -> &[EdgeId] {
        match self.nodes.get(&n) {
            Some(node) => &node.outbound,
            None => &[],
        }
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
