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
    Scenario(Vec<SceneStmt>),
}

#[derive(Debug, Clone)]
pub struct ParamDecl { pub name: String, pub value: Expr }

#[derive(Debug, Clone)]
pub struct NodeDecl {
    pub name: String,
    pub slots: Vec<SlotDecl>,
    pub rules: Vec<RuleDecl>,
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
    Emit    { payload: Expr, target: EmitTarget },
    EmitEach{ payload: Expr, targets: Expr },
    Respond { payload: Expr },
    Record  { name: String, value: Expr },
    Spawn   { template: String, into: String },
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, NEq, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp { Neg, Not }
