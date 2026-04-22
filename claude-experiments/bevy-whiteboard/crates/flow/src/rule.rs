use crate::expr::Expr;
use crate::value::Pattern;

/// A rule's left-hand side: what must match for it to fire.
///
/// A rule may have at most one `Input` pattern in v1 (single packet
/// consumption per firing). `SlotMatch` patterns are non-consuming:
/// they only read slot values and bind variables.
#[derive(Debug, Clone)]
pub enum When {
    /// Matches an inbound packet's payload against `pattern`.
    /// Binds via the pattern. Optionally restricts to packets that
    /// arrived via an edge whose source matches `from`.
    Input {
        pattern: Pattern,
        /// If `Some("NodeName")`, only match packets that arrived from
        /// an edge whose source node has that name. `None` = any.
        from: Option<String>,
    },
    /// Non-consuming slot match.
    SlotMatch { slot: String, pattern: Pattern },
}

impl When {
    pub fn input(pattern: Pattern) -> Self {
        When::Input { pattern, from: None }
    }
    pub fn input_from(pattern: Pattern, from: impl Into<String>) -> Self {
        When::Input { pattern, from: Some(from.into()) }
    }
    pub fn slot(slot: impl Into<String>, pattern: Pattern) -> Self {
        When::SlotMatch { slot: slot.into(), pattern }
    }
}

/// Where an `Emit` should go. Keep this small; routing strategies can
/// be added later.
#[derive(Debug, Clone)]
pub enum EmitTo {
    /// Emit on the first outbound edge. Panics if the node has none.
    DefaultOut,
    /// Emit on the outbound edge whose target node has this name.
    /// Panics if no such edge exists.
    ToTarget(String),
    /// Dynamic target: evaluate `Expr` at emit time. The result must be
    /// either a `Value::Str(name)` (looked up by node name) or a
    /// `Value::NodeRef(id)` (direct id). Panics if no outbound edge
    /// from the emitter reaches the resolved node.
    ToTargetExpr(Expr),
    /// Emit through a named out-port of the emitter's enclosing
    /// compound. The engine walks up the `parent` chain to find the
    /// nearest compound with this out-port, verifies the port is
    /// mapped to the emitting leaf, and fans the packet onto every
    /// outgoing edge from that compound whose `from_port` matches.
    ToOutPort(String),
}

/// What a rule does when it fires. Effects run in order, atomically —
/// no simulation time passes between them.
#[derive(Debug, Clone)]
pub enum Effect {
    /// Write a value to a slot.
    SetSlot { slot: String, value: Expr },
    /// Push a value onto a `Samples` slot's ring.
    SamplesPush { slot: String, value: Expr },
    /// Pop oldest from a `Samples` slot, bind to `into_var` for use in later effects.
    SamplesPopOldestInto { slot: String, into_var: String },
    /// Drop oldest N samples. `n` is an expression evaluated at fire time.
    SamplesDropOldest { slot: String, n: Expr },
    /// Emit a packet.
    Emit { payload: Expr, to: EmitTo },
    /// Emit the same payload to every NodeRef in `targets`. The
    /// `targets` expression must yield `Value::List(NodeRef…)` —
    /// typically `Expr::OutNeighbors` or a filtered/mapped variant of
    /// it. Each delivery uses the outbound edge from this node to that
    /// target's latency expression. Targets without an outbound edge
    /// are silently skipped (same contract as `EmitTo::ToTargetExpr`).
    EmitToEach { payload: Expr, targets: Expr },
    /// Reply to the original requester via the inbound packet's reply address.
    /// Requires an Input pattern in the rule's `when`.
    Respond { payload: Expr },
    /// Record a metric (name, value) in the event log. Observation only.
    RecordMetric { name: String, value: Expr },
    /// Instantiate a registered template. The firing node becomes the
    /// new instance's parent. If `into_var` is set, the new node's
    /// `NodeRef` is bound for subsequent effects in this rule firing.
    Spawn { template: String, into_var: Option<String> },
    /// Despawn a node. Its edges are removed. In-flight packets
    /// targeting or originating from this node are dropped on
    /// delivery.
    Despawn { node: Expr },
}

/// A rule = pattern + guard + effects. Fires atomically.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub when: Vec<When>,
    pub guard: Option<Expr>,
    pub effects: Vec<Effect>,
}

impl Rule {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), when: Vec::new(), guard: None, effects: Vec::new() }
    }
    pub fn when(mut self, w: When) -> Self {
        self.when.push(w);
        self
    }
    pub fn guard(mut self, g: Expr) -> Self {
        self.guard = Some(g);
        self
    }
    pub fn do_(mut self, e: Effect) -> Self {
        self.effects.push(e);
        self
    }
}
