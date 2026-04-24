//! AST for the Flow DSL.
//!
//! This is a thin surface over `flow::Expr` / `flow::Rule` — the
//! lowering step in `lower.rs` maps most things 1:1 and resolves
//! context-dependent identifiers (slot vs variable vs param).

#[derive(Debug, Clone)]
pub struct File {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Params(Vec<ParamDecl>),
    Node(NodeDecl),
    Compound(CompoundDecl),
    Edges(Vec<EdgeDecl>),
    Scenario(ScenarioDecl),
}

#[derive(Debug, Clone)]
pub struct ScenarioDecl {
    /// `None` for an unnamed `scenario { … }` block, which is treated as
    /// the default scenario (stored under the name "main" in the lowered
    /// sim's scenario library).
    pub name: Option<String>,
    pub stmts: Vec<SceneStmt>,
}

#[derive(Debug, Clone)]
pub struct ParamDecl { pub name: String, pub value: Expr }

#[derive(Debug, Clone)]
pub struct NodeDecl {
    pub name: String,
    pub slots: Vec<SlotDecl>,
    pub rules: Vec<RuleDecl>,
    /// Bootstrap wiring run every time an instance of this class is
    /// created — self-edges the instance needs to tick, and initial
    /// packets that seed its inbox. Empty for classes that are pure
    /// behaviour (no self-driven activity).
    pub on_spawn: Vec<OnSpawnStmt>,
    /// Per-class probe declarations. Each probe is a labeled format
    /// string with `{expr}` interpolation holes evaluated against the
    /// node's slots at read time.
    pub probes: Vec<ProbeDecl>,
}

/// A probe is a short named readout the UI can display next to a node.
/// Format is a sequence of literal text and expression holes that
/// evaluate at read time; concatenating their string forms yields the
/// displayed value.
#[derive(Debug, Clone)]
pub struct ProbeDecl {
    pub label: String,
    pub parts: Vec<ProbePart>,
}

#[derive(Debug, Clone)]
pub enum ProbePart {
    Literal(String),
    Hole(Expr),
}

/// One statement inside a `node`'s `on_spawn { ... }` block.
///
/// These run per-instance at instantiation time, giving each instance a
/// working self-loop / bootstrap without requiring the caller to wire
/// them up externally.
#[derive(Debug, Clone)]
pub enum OnSpawnStmt {
    /// `self -> self : LATENCY_EXPR` — create a self-edge on the new
    /// instance with the given latency expression (re-evaluated per
    /// emission, so live slot writes propagate).
    SelfEdge { latency: Expr },
    /// `inject TAG(PAYLOAD?)` — deliver a `Variant(tag, payload)`
    /// packet to the new instance's inbox at sim-now. Payload defaults
    /// to `Nil` when omitted.
    Inject { tag: String, payload: Option<Expr> },
}

#[derive(Debug, Clone)]
pub struct CompoundDecl {
    pub name: String,
    pub in_ports: Vec<PortDecl>,
    pub out_ports: Vec<PortDecl>,
}

#[derive(Debug, Clone)]
pub struct PortDecl { pub port: String, pub inner: String }

#[derive(Debug, Clone)]
pub struct SlotDecl {
    pub name: String,
    pub ty: SlotType,
    pub init: Option<Expr>,
}

#[derive(Debug, Clone)]
pub enum SlotType {
    Int, Float, Bool, Str, Nil, Any,
    Samples(u32),   // capacity
}

#[derive(Debug, Clone)]
pub struct RuleDecl {
    pub name: String,
    pub ons: Vec<Pattern>,
    pub when: Option<Expr>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wild,
    Var(String),
    Lit(Expr),                   // must be a literal at lowering time
    Variant(String, Vec<Pattern>),
}

#[derive(Debug, Clone)]
pub enum Stmt {
    SlotSet { slot: String, value: Expr },
    Push    { slot: String, value: Expr },
    Pop     { slot: String, into: String },
    DropN   { slot: String, n: Expr },
    Emit    {
        payload: Expr,
        target: EmitTarget,
        meta_ops: Vec<MetaOp>,
        rp_op: ReturnPathOp,
    },
    EmitEach{
        payload: Expr,
        targets: Expr,
        meta_ops: Vec<MetaOp>,
        rp_op: ReturnPathOp,
    },
    Record  { name: String, value: Expr },
    Spawn   { template: String, into: String },
    /// `error "kind" detail_expr` — record a runtime error in
    /// `Sim.error_counts[kind]` and emit a `RuntimeError` event.
    /// `kind` must be a string literal at parse time; detail can be
    /// any expression (string literal typical).
    Error   { kind: String, detail: Expr },
}

/// Surface-level metadata modification. Lowered to `rule::MetaOp`.
#[derive(Debug, Clone)]
pub enum MetaOp {
    Set { key: String, value: Expr },
    Remove { key: String },
}

/// Surface-level return_path modification. Lowered to `rule::ReturnPathOp`.
#[derive(Debug, Clone)]
pub enum ReturnPathOp {
    Inherit,
    Push(Expr),
    Pop,
    Replace(Expr),
}

#[derive(Debug, Clone)]
pub enum EmitTarget {
    Default,
    Self_,
    OutPort(String),
    Target(String),              // static node name
    Dynamic(Expr),               // evaluates to Str or NodeRef
}

#[derive(Debug, Clone)]
pub struct EdgeDecl {
    pub from: EdgeEndpoint,
    pub to: EdgeEndpoint,
    pub latency: Expr,
}

#[derive(Debug, Clone)]
pub struct EdgeEndpoint {
    pub node: String,
    pub port: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SceneStmt {
    pub at_ns: u64,
    pub action: SceneAction,
}

#[derive(Debug, Clone)]
pub enum SceneAction {
    Inject { node: String, tag: String, payload: Option<Expr> },
    SetParam { name: String, value: Expr },
    SetSlot { node: String, slot: String, value: Expr },
    Kill { node: String },
}

// -----------------------------------------------------------------------------

/// Surface expressions. Shapes closely track `flow::Expr` — we just
/// carry IDs for deferred resolution: `Name(x)` could be a slot, a
/// variable, or something else until we know the scope.
#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Nil,
    Now,
    SelfRef,
    Name(String),                            // slot | var | ??? (resolved in lower.rs)
    Param(String),                           // explicit param(name) … or just uses Name heuristic
    Variant(String, Option<Box<Expr>>),      // Tag(payload?)
    FnCall(String, Vec<Expr>),               // Exp(…), Uniform(…), Bernoulli(…), len/mean etc.
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    If { cond: Box<Expr>, then_: Box<Expr>, else_: Box<Expr> },
    /// `meta(key)` — look up the consumed packet's metadata value. Lowers
    /// to `flow::Expr::Meta`. Key must be a string literal at parse time.
    Meta(String),
    /// `return_path` bare — the consumed packet's return_path as a List
    /// of NodeRefs. Lowers to `flow::Expr::ReturnPath`.
    ReturnPath,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, NEq, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp { Neg, Not }
