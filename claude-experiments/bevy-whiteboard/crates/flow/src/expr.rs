use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Exp};
use serde::{Deserialize, Serialize};

use crate::sim::SimRng;

use std::collections::BTreeMap;

use crate::samples::Samples;
use crate::sim::{Edge, EdgeId, Node, NodeId, Packet};
use crate::value::{Bindings, Value};

/// Binary operator over two evaluated expressions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Lt, Gt, Le, Ge, Eq, Neq,
    And, Or,
}

/// The expression language used for:
///  - guards (Bool-valued),
///  - effect arguments (any Value),
///  - edge latency (Int nanoseconds).
///
/// Keep this small. Below `Expr` we don't decompose further into
/// "nodes and wires" — it's the textual leaf of the substrate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// Constant value.
    Lit(Value),
    /// Variable bound by a pattern match in the current rule firing.
    Var(String),
    /// Read current slot value from the firing node.
    Slot(String),
    /// Field access on a record (or payload of a variant, via pattern bindings).
    Field { of: Box<Expr>, field: String },
    /// Current simulation clock in nanoseconds.
    Now,
    /// The node on which the currently-firing rule lives. Evaluates to
    /// `Value::NodeRef(self)`. Lets a rule refer to its own node
    /// without needing a bootstrap packet or a dedicated slot.
    SelfRef,
    /// Read a live sim parameter by name. Parameters are a flat
    /// namespace of named expressions, editable at runtime via
    /// `Sim::set_param`. Each reference re-evaluates on access, so
    /// changing a param propagates to every use on the next evaluation.
    Param(String),
    /// Construct a record from field-expression pairs.
    /// The "tagged envelope" shape used by rules is just a Record:
    /// `Expr::variant(tag, payload)` builds `Record({kind: Lit(tag), value: payload})`.
    Record(Vec<(String, Expr)>),
    /// Binary op.
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    /// Boolean negation.
    Not(Box<Expr>),
    /// if / then / else.
    If { cond: Box<Expr>, then: Box<Expr>, els: Box<Expr> },
    /// len of a `Samples` slot.
    SamplesLen(String),
    /// mean of a `Samples` slot (as f64 Float).
    SamplesMean(String),
    /// Count of samples satisfying a predicate `x -> Bool` where `x`
    /// binds the sample value. The predicate is an `Expr` that uses
    /// the variable whose name is `bind`.
    SamplesCountWhere { slot: String, bind: String, pred: Box<Expr> },

    // ---- Distributions. All sampled via the seeded RNG on Sim. ----
    /// Exponential distribution with given rate (1/mean).
    /// Returns an Int nanoseconds.
    ExpDist { mean_ns: Box<Expr> },
    /// Uniform Int in [lo, hi] inclusive.
    UniformInt { lo: Box<Expr>, hi: Box<Expr> },
    /// Bernoulli(p). Returns Bool.
    Bernoulli { p: Box<Expr> },

    // ---- Routing-strategy primitives. ----------------------------------
    /// `Value::List(Vec<Value::NodeRef>)` of the firing node's outbound
    /// neighbours, in edge-id order. Excludes self-loops. Foundation for
    /// every routing strategy that has to look across multiple downstreams.
    OutNeighbors,
    /// Read a slot off another node by `NodeRef`. The `node` expression
    /// must evaluate to `Value::NodeRef(_)`. Panics if the target node
    /// doesn't exist or the slot isn't defined.
    SlotOf { node: Box<Expr>, slot: String },

    // ---- List operators. ------------------------------------------------
    /// Number of items in a `Value::List` or `Value::Samples`.
    Length(Box<Expr>),
    /// Nth item of a `Value::List` (0-indexed). Wraps with `% len`, so
    /// round-robin counters can keep growing without explicit mod.
    Index { list: Box<Expr>, i: Box<Expr> },
    /// `[expr(item) for item in list if pred(item)]` — but as two ops:
    /// see also `Map`. `Filter` returns a `Value::List` containing the
    /// items for which `pred` evaluates to true. `bind` is the variable
    /// name introduced into `pred`'s scope.
    Filter { list: Box<Expr>, bind: String, pred: Box<Expr> },
    /// `[expr(item) for item in list]`. `bind` is the variable name
    /// introduced into `expr`'s scope.
    Map { list: Box<Expr>, bind: String, expr: Box<Expr> },
    /// Left fold. `init` evaluated first; then for each item:
    ///   acc = expr(acc=acc, item=bind)
    /// Result is the final accumulator. Use to express min/max/sum/etc.
    Reduce {
        list: Box<Expr>,
        bind: String,
        acc: String,
        init: Box<Expr>,
        expr: Box<Expr>,
    },
    /// Return the item of `list` that minimises `expr(item)`. `Value::Nil`
    /// when the list is empty. Ties broken by first-seen (list order). The
    /// keying expression must yield `Value::Int`.
    Argmin { list: Box<Expr>, bind: String, expr: Box<Expr> },

    // ---- Edge-scoped reads. --------------------------------------------
    /// Sim-time (Int ns) of the most recent forward-direction emit on the
    /// outbound edge from the current node to `to` (a `NodeRef`). Returns
    /// `0` if the edge has never been traversed, or if no such outbound
    /// edge exists. Used by routing strategies like LRU / round-robin.
    EdgeLastSent { to: Box<Expr> },

    // ---- Packet introspection (read the consumed inbound packet). ------
    /// Look up a key in the consumed packet's `metadata`. Returns
    /// `Value::Nil` if the key is absent or no packet was consumed
    /// (e.g. source-rule fire). Never errors — metadata absence is a
    /// normal state, not a type mismatch.
    Meta(String),
    /// The consumed packet's `return_path`, materialized as
    /// `Value::List(NodeRef…)` with head at index 0. `Value::Nil` if
    /// no packet was consumed.
    ReturnPath,
    /// First element of a `Value::List`, or `Value::Nil` if empty.
    /// Complements `Tail` so you can pattern a stack without indexing.
    Head(Box<Expr>),
    /// All-but-first of a `Value::List`, or empty list if the list is
    /// empty or has one element.
    Tail(Box<Expr>),
}

/// Helpers for building expressions concisely.
impl Expr {
    pub fn lit(v: Value) -> Self { Expr::Lit(v) }
    pub fn int(n: i64) -> Self { Expr::Lit(Value::Int(n)) }
    pub fn float(f: f64) -> Self { Expr::Lit(Value::Float(f)) }
    pub fn bool(b: bool) -> Self { Expr::Lit(Value::Bool(b)) }
    pub fn var(name: impl Into<String>) -> Self { Expr::Var(name.into()) }
    pub fn slot(name: impl Into<String>) -> Self { Expr::Slot(name.into()) }
    pub fn now() -> Self { Expr::Now }
    pub fn self_ref() -> Self { Expr::SelfRef }
    pub fn param(name: impl Into<String>) -> Self { Expr::Param(name.into()) }

    pub fn add(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b)) }
    pub fn sub(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b)) }
    pub fn mul(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b)) }
    pub fn div(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Div, Box::new(a), Box::new(b)) }
    pub fn pow(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Pow, Box::new(a), Box::new(b)) }
    pub fn lt(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Lt, Box::new(a), Box::new(b)) }
    pub fn gt(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Gt, Box::new(a), Box::new(b)) }
    pub fn le(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Le, Box::new(a), Box::new(b)) }
    pub fn ge(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Ge, Box::new(a), Box::new(b)) }
    pub fn eq(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Eq, Box::new(a), Box::new(b)) }
    pub fn and(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::And, Box::new(a), Box::new(b)) }
    pub fn or(a: Expr, b: Expr) -> Self { Expr::BinOp(BinOp::Or, Box::new(a), Box::new(b)) }
    pub fn not(a: Expr) -> Self { Expr::Not(Box::new(a)) }
    pub fn if_(cond: Expr, then: Expr, els: Expr) -> Self {
        Expr::If { cond: Box::new(cond), then: Box::new(then), els: Box::new(els) }
    }
    pub fn field(of: Expr, field: impl Into<String>) -> Self {
        Expr::Field { of: Box::new(of), field: field.into() }
    }
    /// Construct the conventional "tagged envelope" record:
    /// `{ kind: Lit(Str(tag)), value: payload }`. Used by the DSL when
    /// lowering `emit packet(p)` / `req(p)` / etc. — there is no
    /// separate Variant variant on `Expr`; envelopes are just records.
    pub fn variant(tag: impl Into<String>, payload: Expr) -> Self {
        Expr::record([
            (crate::value::KIND_FIELD.to_string(), Expr::Lit(crate::value::Value::Str(tag.into()))),
            (crate::value::VALUE_FIELD.to_string(), payload),
        ])
    }
    pub fn record<I, K>(fields: I) -> Self
    where I: IntoIterator<Item = (K, Expr)>, K: Into<String>,
    {
        Expr::Record(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    pub fn exp_dist(mean_ns: Expr) -> Self { Expr::ExpDist { mean_ns: Box::new(mean_ns) } }
    pub fn uniform_int(lo: Expr, hi: Expr) -> Self {
        Expr::UniformInt { lo: Box::new(lo), hi: Box::new(hi) }
    }
    pub fn bernoulli(p: Expr) -> Self { Expr::Bernoulli { p: Box::new(p) } }
    pub fn samples_len(slot: impl Into<String>) -> Self { Expr::SamplesLen(slot.into()) }
    pub fn samples_mean(slot: impl Into<String>) -> Self { Expr::SamplesMean(slot.into()) }
    pub fn samples_count_where(slot: impl Into<String>, bind: impl Into<String>, pred: Expr) -> Self {
        Expr::SamplesCountWhere { slot: slot.into(), bind: bind.into(), pred: Box::new(pred) }
    }

    pub fn out_neighbors() -> Self { Expr::OutNeighbors }
    pub fn slot_of(node: Expr, slot: impl Into<String>) -> Self {
        Expr::SlotOf { node: Box::new(node), slot: slot.into() }
    }
    pub fn length(list: Expr) -> Self { Expr::Length(Box::new(list)) }
    pub fn index(list: Expr, i: Expr) -> Self {
        Expr::Index { list: Box::new(list), i: Box::new(i) }
    }
    pub fn meta(key: impl Into<String>) -> Self { Expr::Meta(key.into()) }
    pub fn return_path() -> Self { Expr::ReturnPath }
    pub fn head(list: Expr) -> Self { Expr::Head(Box::new(list)) }
    pub fn tail(list: Expr) -> Self { Expr::Tail(Box::new(list)) }
    pub fn filter(list: Expr, bind: impl Into<String>, pred: Expr) -> Self {
        Expr::Filter { list: Box::new(list), bind: bind.into(), pred: Box::new(pred) }
    }
    pub fn map(list: Expr, bind: impl Into<String>, expr: Expr) -> Self {
        Expr::Map { list: Box::new(list), bind: bind.into(), expr: Box::new(expr) }
    }
    pub fn reduce(list: Expr, bind: impl Into<String>, acc: impl Into<String>, init: Expr, expr: Expr) -> Self {
        Expr::Reduce {
            list: Box::new(list),
            bind: bind.into(),
            acc: acc.into(),
            init: Box::new(init),
            expr: Box::new(expr),
        }
    }
    pub fn argmin(list: Expr, bind: impl Into<String>, expr: Expr) -> Self {
        Expr::Argmin { list: Box::new(list), bind: bind.into(), expr: Box::new(expr) }
    }
    pub fn edge_last_sent(to: Expr) -> Self {
        Expr::EdgeLastSent { to: Box::new(to) }
    }
}

/// Context for evaluating an expression. Borrows from the simulator
/// so evaluation can read slots, the clock, the RNG, and live params.
pub struct EvalCtx<'a> {
    pub bindings: &'a Bindings,
    pub slots: &'a BTreeMap<String, Value>,
    pub now_ns: u64,
    pub rng: &'a mut SimRng,
    /// The node on which the firing rule lives. `Expr::SelfRef` reads
    /// this. `None` is only valid if the expression doesn't reference
    /// `SelfRef` — in practice, all engine evaluations supply it.
    pub current_node: Option<NodeId>,
    /// Live param namespace. `Expr::Param(name)` looks up here and
    /// recursively evaluates the bound expression. May be empty.
    pub params: &'a std::collections::HashMap<String, Expr>,
    /// Stack of currently-evaluating param names, for cycle detection.
    /// Reset between top-level evaluations; pushed/popped within
    /// `Expr::Param` lookups.
    pub param_stack: &'a mut Vec<String>,
    /// Read-only references to the sim's node and edge maps. Required
    /// for the routing-strategy primitives (`OutNeighbors`, `SlotOf`)
    /// to enumerate neighbours and look up cross-node slots.
    pub nodes: &'a BTreeMap<NodeId, Node>,
    pub edges: &'a BTreeMap<EdgeId, Edge>,
    /// The packet the current rule firing is consuming, if any. Read
    /// by `Expr::Meta` and `Expr::ReturnPath`. `None` for source-rule
    /// fires (no `Input` pattern) and for any evaluation outside a
    /// rule firing — both treat the absent packet as empty metadata /
    /// empty return_path.
    pub packet: Option<&'a Packet>,
}

impl Expr {
    /// Evaluate this expression in the given context.
    ///
    /// Panics on type errors — modeling mistakes should be loud.
    pub fn eval(&self, ctx: &mut EvalCtx) -> Value {
        match self {
            Expr::Lit(v) => v.clone(),
            Expr::Var(name) => ctx.bindings.get(name)
                .unwrap_or_else(|| panic!("Expr::Var: unbound variable `{}`", name))
                .clone(),
            Expr::Slot(name) => ctx.slots.get(name)
                .unwrap_or_else(|| panic!("Expr::Slot: no slot `{}` on this node", name))
                .clone(),
            Expr::Now => Value::Int(ctx.now_ns as i64),
            Expr::SelfRef => {
                let nid = ctx.current_node
                    .expect("Expr::SelfRef evaluated without current_node in context");
                Value::NodeRef(nid)
            }
            Expr::Param(name) => {
                if ctx.param_stack.iter().any(|n| n == name) {
                    panic!("Param cycle detected: {:?} → {}",
                        ctx.param_stack, name);
                }
                let e = ctx.params.get(name)
                    .unwrap_or_else(|| panic!("Expr::Param: no param `{}` defined", name))
                    .clone();
                ctx.param_stack.push(name.clone());
                let v = e.eval(ctx);
                ctx.param_stack.pop();
                v
            }
            Expr::Record(fields) => {
                let mut m = std::collections::BTreeMap::new();
                for (k, e) in fields {
                    m.insert(k.clone(), e.eval(ctx));
                }
                Value::Record(m)
            }
            Expr::Field { of, field } => {
                let v = of.eval(ctx);
                match v {
                    Value::Record(map) => map.get(field)
                        .unwrap_or_else(|| panic!("Expr::Field: no field `{}`", field))
                        .clone(),
                    other => panic!("Expr::Field: not a record: {:?}", other),
                }
            }
            Expr::Not(e) => {
                let v = e.eval(ctx);
                Value::Bool(!v.as_bool().expect("Not: expected Bool"))
            }
            Expr::If { cond, then, els } => {
                if cond.eval(ctx).as_bool().expect("If: cond must be Bool") {
                    then.eval(ctx)
                } else {
                    els.eval(ctx)
                }
            }
            Expr::BinOp(op, a, b) => {
                let av = a.eval(ctx);
                let bv = b.eval(ctx);
                eval_binop(*op, &av, &bv)
            }
            Expr::SamplesLen(slot) => {
                let v = ctx.slots.get(slot)
                    .unwrap_or_else(|| panic!("SamplesLen: no slot `{}`", slot));
                match v {
                    Value::Samples(s) => Value::Int(s.len() as i64),
                    _ => panic!("SamplesLen: slot `{}` is not a Samples", slot),
                }
            }
            Expr::SamplesMean(slot) => {
                let v = ctx.slots.get(slot)
                    .unwrap_or_else(|| panic!("SamplesMean: no slot `{}`", slot));
                match v {
                    Value::Samples(s) => Value::Float(s.mean_f64()),
                    _ => panic!("SamplesMean: slot `{}` is not a Samples", slot),
                }
            }
            Expr::SamplesCountWhere { slot, bind, pred } => {
                let v = ctx.slots.get(slot)
                    .unwrap_or_else(|| panic!("SamplesCountWhere: no slot `{}`", slot))
                    .clone();
                let items = match v {
                    Value::Samples(Samples { items, .. }) => items,
                    _ => panic!("SamplesCountWhere: slot `{}` is not a Samples", slot),
                };
                let mut count: i64 = 0;
                let mut local = ctx.bindings.clone();
                for item in items {
                    local.insert(bind.clone(), item);
                    let mut sub = EvalCtx {
                        bindings: &local,
                        slots: ctx.slots,
                        now_ns: ctx.now_ns,
                        rng: ctx.rng,
                        current_node: ctx.current_node,
                        params: ctx.params,
                        param_stack: ctx.param_stack,
                        nodes: ctx.nodes,
                        edges: ctx.edges,
                        packet: ctx.packet,
                    };
                    if pred.eval(&mut sub).as_bool().expect("SamplesCountWhere: pred must be Bool") {
                        count += 1;
                    }
                }
                Value::Int(count)
            }
            Expr::ExpDist { mean_ns } => {
                let mean = mean_ns.eval(ctx).as_float().expect("ExpDist: mean must be numeric");
                if mean <= 0.0 {
                    panic!("ExpDist: mean must be > 0, got {}", mean);
                }
                let dist = Exp::new(1.0 / mean).expect("Exp::new failed");
                let sample: f64 = dist.sample(ctx.rng);
                Value::Int(sample as i64)
            }
            Expr::UniformInt { lo, hi } => {
                let l = lo.eval(ctx).as_int().expect("UniformInt: lo must be Int");
                let h = hi.eval(ctx).as_int().expect("UniformInt: hi must be Int");
                assert!(l <= h, "UniformInt: lo {} > hi {}", l, h);
                Value::Int(ctx.rng.gen_range(l..=h))
            }
            Expr::Bernoulli { p } => {
                let pv = p.eval(ctx).as_float().expect("Bernoulli: p must be numeric");
                let dist = Bernoulli::new(pv).expect("Bernoulli::new failed");
                Value::Bool(dist.sample(ctx.rng))
            }

            // ---- Routing primitives ------------------------------------
            Expr::OutNeighbors => {
                let nid = ctx.current_node
                    .expect("OutNeighbors: needs current_node");
                // Edge-id-ordered, self-loops excluded — same convention
                // as `EmitTo::DefaultOut`'s preference. Reads the
                // node's owned outbound list (already in `EdgeId`
                // order) instead of scanning the global edge map.
                let items: Vec<Value> = ctx
                    .nodes
                    .get(&nid)
                    .map(|n| n.outbound.as_slice())
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|eid| ctx.edges.get(eid))
                    .filter(|e| e.to != nid)
                    .map(|e| Value::NodeRef(e.to))
                    .collect();
                Value::List(items)
            }
            Expr::SlotOf { node, slot } => {
                let v = node.eval(ctx);
                let nid = match v {
                    Value::NodeRef(id) => id,
                    other => panic!(
                        "SlotOf: first arg must yield NodeRef, got {:?}", other
                    ),
                };
                ctx.nodes.get(&nid)
                    .unwrap_or_else(|| panic!("SlotOf: no node {:?}", nid))
                    .slots.get(slot)
                    .cloned()
                    .unwrap_or_else(|| panic!(
                        "SlotOf: node {:?} has no slot `{}`", nid, slot
                    ))
            }

            // ---- List operators -----------------------------------------
            Expr::Length(e) => {
                let v = e.eval(ctx);
                match v {
                    Value::List(items) => Value::Int(items.len() as i64),
                    Value::Samples(s) => Value::Int(s.len() as i64),
                    other => panic!("Length: expected List or Samples, got {:?}", other),
                }
            }
            Expr::Index { list, i } => {
                let lv = list.eval(ctx);
                let items = match lv {
                    Value::List(v) => v,
                    other => panic!("Index: first arg must be a List, got {:?}", other),
                };
                if items.is_empty() {
                    panic!("Index: list is empty");
                }
                let raw = i.eval(ctx).as_int()
                    .expect("Index: index must be Int");
                // Wrap to handle round-robin counters that grow forever.
                let n = items.len() as i64;
                let idx = ((raw % n) + n) % n;
                items[idx as usize].clone()
            }
            Expr::Filter { list, bind, pred } => {
                let lv = list.eval(ctx);
                let items = match lv {
                    Value::List(v) => v,
                    other => panic!("Filter: first arg must be a List, got {:?}", other),
                };
                let mut out: Vec<Value> = Vec::new();
                let mut local = ctx.bindings.clone();
                for item in items {
                    local.insert(bind.clone(), item.clone());
                    let mut sub = EvalCtx {
                        bindings: &local,
                        slots: ctx.slots,
                        now_ns: ctx.now_ns,
                        rng: ctx.rng,
                        current_node: ctx.current_node,
                        params: ctx.params,
                        param_stack: ctx.param_stack,
                        nodes: ctx.nodes,
                        edges: ctx.edges,
                        packet: ctx.packet,
                    };
                    if pred.eval(&mut sub).as_bool().expect("Filter: pred must be Bool") {
                        out.push(item);
                    }
                }
                Value::List(out)
            }
            Expr::Map { list, bind, expr } => {
                let lv = list.eval(ctx);
                let items = match lv {
                    Value::List(v) => v,
                    other => panic!("Map: first arg must be a List, got {:?}", other),
                };
                let mut out: Vec<Value> = Vec::with_capacity(items.len());
                let mut local = ctx.bindings.clone();
                for item in items {
                    local.insert(bind.clone(), item);
                    let mut sub = EvalCtx {
                        bindings: &local,
                        slots: ctx.slots,
                        now_ns: ctx.now_ns,
                        rng: ctx.rng,
                        current_node: ctx.current_node,
                        params: ctx.params,
                        param_stack: ctx.param_stack,
                        nodes: ctx.nodes,
                        edges: ctx.edges,
                        packet: ctx.packet,
                    };
                    out.push(expr.eval(&mut sub));
                }
                Value::List(out)
            }
            Expr::Reduce { list, bind, acc, init, expr } => {
                let lv = list.eval(ctx);
                let items = match lv {
                    Value::List(v) => v,
                    other => panic!("Reduce: first arg must be a List, got {:?}", other),
                };
                let mut accumulator = init.eval(ctx);
                let mut local = ctx.bindings.clone();
                for item in items {
                    local.insert(bind.clone(), item);
                    local.insert(acc.clone(), accumulator.clone());
                    let mut sub = EvalCtx {
                        bindings: &local,
                        slots: ctx.slots,
                        now_ns: ctx.now_ns,
                        rng: ctx.rng,
                        current_node: ctx.current_node,
                        params: ctx.params,
                        param_stack: ctx.param_stack,
                        nodes: ctx.nodes,
                        edges: ctx.edges,
                        packet: ctx.packet,
                    };
                    accumulator = expr.eval(&mut sub);
                }
                accumulator
            }

            // ---- Packet introspection. Absence of a packet is NOT an
            // error; it produces the "empty" reading. Rule authors often
            // read meta/return_path on fallthrough guards where no packet
            // is available, and we don't want to panic them.
            Expr::Meta(key) => match ctx.packet {
                Some(p) => p.metadata.get(key).cloned().unwrap_or(Value::Nil),
                None => Value::Nil,
            },
            Expr::ReturnPath => match ctx.packet {
                Some(p) => Value::List(p.return_path.iter().copied().map(Value::NodeRef).collect()),
                None => Value::Nil,
            },
            Expr::Head(list_expr) => {
                let v = list_expr.eval(ctx);
                match v {
                    Value::List(items) => items.into_iter().next().unwrap_or(Value::Nil),
                    Value::Nil => Value::Nil,
                    other => panic!("Head: expected List or Nil, got {:?}", other),
                }
            }
            Expr::Tail(list_expr) => {
                let v = list_expr.eval(ctx);
                match v {
                    Value::List(items) => {
                        if items.is_empty() { Value::List(Vec::new()) }
                        else { Value::List(items.into_iter().skip(1).collect()) }
                    }
                    Value::Nil => Value::List(Vec::new()),
                    other => panic!("Tail: expected List or Nil, got {:?}", other),
                }
            }
            Expr::Argmin { list, bind, expr } => {
                let lv = list.eval(ctx);
                let items = match lv {
                    Value::List(v) => v,
                    other => panic!("Argmin: first arg must be a List, got {:?}", other),
                };
                let mut best: Option<(i64, Value)> = None;
                let mut local = ctx.bindings.clone();
                for item in items {
                    local.insert(bind.clone(), item.clone());
                    let mut sub = EvalCtx {
                        bindings: &local,
                        slots: ctx.slots,
                        now_ns: ctx.now_ns,
                        rng: ctx.rng,
                        current_node: ctx.current_node,
                        params: ctx.params,
                        param_stack: ctx.param_stack,
                        nodes: ctx.nodes,
                        edges: ctx.edges,
                        packet: ctx.packet,
                    };
                    let key = expr.eval(&mut sub).as_int()
                        .expect("Argmin: key expression must yield Int");
                    match &best {
                        Some((k, _)) if *k <= key => {} // keep earlier on ties
                        _ => best = Some((key, item)),
                    }
                }
                match best {
                    Some((_, v)) => v,
                    None => Value::Nil,
                }
            }
            Expr::EdgeLastSent { to } => {
                let v = to.eval(ctx);
                let target = match v {
                    Value::NodeRef(id) => id,
                    Value::Nil => return Value::Int(0),
                    other => panic!(
                        "EdgeLastSent: arg must yield NodeRef or Nil, got {:?}", other
                    ),
                };
                let nid = ctx.current_node
                    .expect("EdgeLastSent: needs current_node");
                // Forward edge from self to target; `0` if no edge or
                // never traversed. The emit-seq counter starts at 1,
                // so never-sent (0) always sorts strictly before any
                // real emission under argmin — new neighbours absorb
                // load first, matching LRU semantics.
                let seq = ctx.edges.values()
                    .find(|e| e.from == nid && e.to == target)
                    .and_then(|e| e.last_sent_seq)
                    .unwrap_or(0);
                Value::Int(seq as i64)
            }
        }
    }
}

fn eval_binop(op: BinOp, a: &Value, b: &Value) -> Value {
    use BinOp::*;
    // Arithmetic: prefer Int when both are Int, else promote to Float.
    let both_int = matches!(a, Value::Int(_)) && matches!(b, Value::Int(_));
    match op {
        Add | Sub | Mul | Div | Mod | Pow => {
            if both_int {
                let x = a.as_int().unwrap();
                let y = b.as_int().unwrap();
                let r = match op {
                    Add => x + y,
                    Sub => x - y,
                    Mul => x * y,
                    Div => { assert!(y != 0, "Div by zero"); x / y },
                    Mod => { assert!(y != 0, "Mod by zero"); x % y },
                    Pow => {
                        assert!(y >= 0, "Pow: negative exponent on Int");
                        // NB: saturating_pow clips instead of wrapping — retries of
                        // 2^63 produce i64::MAX, not a panic or wrap-around.
                        x.saturating_pow(y as u32)
                    }
                    _ => unreachable!(),
                };
                Value::Int(r)
            } else {
                let x = a.as_float().expect("arith: expected numeric");
                let y = b.as_float().expect("arith: expected numeric");
                let r = match op {
                    Add => x + y,
                    Sub => x - y,
                    Mul => x * y,
                    Div => { assert!(y != 0.0, "Div by zero"); x / y },
                    Mod => x % y,
                    Pow => x.powf(y),
                    _ => unreachable!(),
                };
                Value::Float(r)
            }
        }
        Lt | Gt | Le | Ge => {
            let x = a.as_float().expect("cmp: expected numeric");
            let y = b.as_float().expect("cmp: expected numeric");
            Value::Bool(match op {
                Lt => x < y,
                Gt => x > y,
                Le => x <= y,
                Ge => x >= y,
                _ => unreachable!(),
            })
        }
        Eq => Value::Bool(a == b),
        Neq => Value::Bool(a != b),
        And => {
            let x = a.as_bool().expect("And: expected Bool");
            let y = b.as_bool().expect("And: expected Bool");
            Value::Bool(x && y)
        }
        Or => {
            let x = a.as_bool().expect("Or: expected Bool");
            let y = b.as_bool().expect("Or: expected Bool");
            Value::Bool(x || y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn empty_nodes() -> &'static BTreeMap<NodeId, Node> {
        use std::sync::OnceLock;
        static M: OnceLock<BTreeMap<NodeId, Node>> = OnceLock::new();
        M.get_or_init(BTreeMap::new)
    }
    fn empty_edges() -> &'static BTreeMap<EdgeId, Edge> {
        use std::sync::OnceLock;
        static M: OnceLock<BTreeMap<EdgeId, Edge>> = OnceLock::new();
        M.get_or_init(BTreeMap::new)
    }

    fn ctx<'a>(
        bindings: &'a Bindings,
        slots: &'a std::collections::BTreeMap<String, Value>,
        rng: &'a mut SimRng,
        params: &'a std::collections::HashMap<String, Expr>,
        param_stack: &'a mut Vec<String>,
    ) -> EvalCtx<'a> {
        EvalCtx {
            bindings, slots, now_ns: 0, rng,
            current_node: None, params, param_stack,
            nodes: empty_nodes(), edges: empty_edges(),
            packet: None,
        }
    }

    #[test]
    fn arith_int() {
        let b = Bindings::new();
        let s = std::collections::BTreeMap::new();
        let p = std::collections::HashMap::new();
        let mut ps = Vec::new();
        let mut rng = SimRng::seed_from_u64(0);
        let mut c = ctx(&b, &s, &mut rng, &p, &mut ps);
        let e = Expr::mul(Expr::int(100), Expr::pow(Expr::int(2), Expr::int(3)));
        assert_eq!(e.eval(&mut c), Value::Int(800));
    }

    #[test]
    fn exp_dist_seeded_reproducible() {
        let b = Bindings::new();
        let s = std::collections::BTreeMap::new();
        let p = std::collections::HashMap::new();
        let mut ps1 = Vec::new();
        let mut ps2 = Vec::new();

        let mut rng1 = SimRng::seed_from_u64(42);
        let mut rng2 = SimRng::seed_from_u64(42);
        let e = Expr::exp_dist(Expr::float(10_000_000.0));

        let mut c1 = ctx(&b, &s, &mut rng1, &p, &mut ps1);
        let v1 = e.eval(&mut c1);
        let mut c2 = ctx(&b, &s, &mut rng2, &p, &mut ps2);
        let v2 = e.eval(&mut c2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn param_basic_and_live_update() {
        let b = Bindings::new();
        let s = std::collections::BTreeMap::new();
        let mut p = std::collections::HashMap::new();
        p.insert("base".to_string(), Expr::int(100));
        p.insert("x".to_string(), Expr::add(Expr::param("base"), Expr::int(1)));
        let mut ps = Vec::new();
        let mut rng = SimRng::seed_from_u64(0);
        let e = Expr::param("x");

        {
            let mut c = ctx(&b, &s, &mut rng, &p, &mut ps);
            assert_eq!(e.eval(&mut c), Value::Int(101));
        }

        // "Live update": rebind base, same expression yields new value.
        p.insert("base".to_string(), Expr::int(500));
        {
            let mut c = ctx(&b, &s, &mut rng, &p, &mut ps);
            assert_eq!(e.eval(&mut c), Value::Int(501));
        }
    }

    #[test]
    #[should_panic(expected = "Param cycle")]
    fn param_cycle_detected() {
        let b = Bindings::new();
        let s = std::collections::BTreeMap::new();
        let mut p = std::collections::HashMap::new();
        p.insert("a".to_string(), Expr::param("b"));
        p.insert("b".to_string(), Expr::param("a"));
        let mut ps = Vec::new();
        let mut rng = SimRng::seed_from_u64(0);
        let mut c = ctx(&b, &s, &mut rng, &p, &mut ps);
        let _ = Expr::param("a").eval(&mut c);
    }
}
