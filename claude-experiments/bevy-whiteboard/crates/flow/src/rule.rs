use serde::{Deserialize, Serialize};

use crate::expr::Expr;
use crate::value::Pattern;

/// A rule's left-hand side: what must match for it to fire.
///
/// A rule may have at most one `Input` pattern in v1 (single packet
/// consumption per firing). `SlotMatch` patterns are non-consuming:
/// they only read slot values and bind variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Modification to a packet's `metadata` map during an emit. Applied
/// in order after the consumed packet's metadata has been inherited.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaOp {
    /// Insert or overwrite `key` with the evaluated `value`.
    Set { key: String, value: Expr },
    /// Remove `key` from the emitted packet's metadata. No-op if
    /// `key` isn't present.
    Remove { key: String },
}

/// Modification to a packet's `return_path` stack during an emit.
/// Rules opt in explicitly; the default is `Inherit`, which just
/// copies the consumed packet's return_path (or empty for source
/// emits). At most one of Push/Pop/Replace applies per emit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReturnPathOp {
    /// Default: new packet's return_path = consumed packet's (or empty).
    Inherit,
    /// Prepend `expr` (must eval to `NodeRef`) to the inherited path.
    /// Typical for clients making a request they expect a response to.
    Push(Expr),
    /// Drop the head of the inherited path. Typical for responders
    /// sending a reply back to the head (combine with
    /// `EmitTo::ToTargetExpr(head(return_path))`). If the inherited
    /// path is empty, the emit records a runtime error and is skipped.
    Pop,
    /// Replace the return_path entirely. `expr` must eval to
    /// `Value::List(NodeRef…)`.
    Replace(Expr),
}

impl Default for ReturnPathOp {
    fn default() -> Self { ReturnPathOp::Inherit }
}

/// What a rule does when it fires. Effects run in order, atomically —
/// no simulation time passes between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effect {
    /// Write a value to a slot.
    SetSlot { slot: String, value: Expr },
    /// Push a value onto a `Samples` slot's ring.
    SamplesPush { slot: String, value: Expr },
    /// Pop oldest from a `Samples` slot, bind to `into_var` for use in later effects.
    SamplesPopOldestInto { slot: String, into_var: String },
    /// Drop oldest N samples. `n` is an expression evaluated at fire time.
    SamplesDropOldest { slot: String, n: Expr },
    /// Emit a packet. `meta_ops` and `return_path_op` default to
    /// no-op/inherit — omit them to forward the consumed packet's
    /// metadata and return_path unchanged.
    Emit {
        payload: Expr,
        to: EmitTo,
        meta_ops: Vec<MetaOp>,
        return_path_op: ReturnPathOp,
    },
    /// Emit the same payload to every NodeRef in `targets`. The
    /// `targets` expression must yield `Value::List(NodeRef…)` —
    /// typically `Expr::OutNeighbors` or a filtered/mapped variant of
    /// it. Each delivery uses the outbound edge from this node to that
    /// target's latency expression. Targets without an outbound edge
    /// are silently skipped (same contract as `EmitTo::ToTargetExpr`).
    /// `meta_ops` and `return_path_op` apply identically to every
    /// emitted copy.
    EmitToEach {
        payload: Expr,
        targets: Expr,
        meta_ops: Vec<MetaOp>,
        return_path_op: ReturnPathOp,
    },
    /// Record a metric (name, value) in the event log. Observation only.
    RecordMetric { name: String, value: Expr },
    /// Record a user-triggered runtime error. Increments
    /// `Sim.error_counts[kind]` and emits a `RuntimeError` event,
    /// identical to the internal `record_error` path the engine
    /// uses for its own bad-input cases. Use this in DSL rules to
    /// signal domain-level violations (color mismatch, bad
    /// preconditions, etc.) that should surface on the error panel.
    RecordError { kind: String, detail: Expr },
    /// Instantiate a registered template. The firing node becomes the
    /// new instance's parent. If `into_var` is set, the new node's
    /// `NodeRef` is bound for subsequent effects in this rule firing.
    Spawn { template: String, into_var: Option<String> },
    /// Despawn a node. Its edges are removed. In-flight packets
    /// targeting or originating from this node are dropped on
    /// delivery.
    Despawn { node: Expr },
}

impl Effect {
    /// Convenience constructor for a plain emit — no metadata/return_path
    /// modifications. Equivalent to setting `meta_ops = []` and
    /// `return_path_op = ReturnPathOp::Inherit`.
    pub fn emit(payload: Expr, to: EmitTo) -> Self {
        Effect::Emit { payload, to, meta_ops: Vec::new(), return_path_op: ReturnPathOp::Inherit }
    }

    /// Convenience constructor for a plain emit-to-each.
    pub fn emit_to_each(payload: Expr, targets: Expr) -> Self {
        Effect::EmitToEach { payload, targets, meta_ops: Vec::new(), return_path_op: ReturnPathOp::Inherit }
    }

    /// Convenience: emit `payload` to `head(return_path)` with
    /// `ReturnPathOp::Pop` — i.e. reply to the innermost caller and
    /// drop that frame. Equivalent to the old `Effect::Respond` when
    /// the caller pushed itself on request.
    ///
    /// Requires an outbound edge from the firing node to
    /// `head(return_path)`. If the edge is missing or the path is
    /// empty, the engine records an error and drops the emit — it
    /// never panics.
    pub fn respond(payload: Expr) -> Self {
        Effect::Emit {
            payload,
            to: EmitTo::ToTargetExpr(Expr::head(Expr::return_path())),
            meta_ops: Vec::new(),
            return_path_op: ReturnPathOp::Pop,
        }
    }
}

/// A rule = pattern + guard + effects. Fires atomically.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
