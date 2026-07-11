//! The surface AST — a Rust-flavored syntax over the runtime's IR.

/// A written-down type: `i64`, `bool`, `()`, or `&StructName` (a reference to a
/// user-defined struct; all struct values are references).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeExpr {
    I64,
    Bool,
    Unit,
    Ref(String),
}

#[derive(Clone, Debug)]
pub struct FieldDef {
    pub name: String,
    pub ty: TypeExpr,
    /// `= <literal>` — enables construction without the field and auto-derived
    /// migration when the field is added later.
    pub default: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
}

#[derive(Clone, Debug)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: TypeExpr,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub enum Item {
    Struct(StructDef),
    Fn(FnDef),
}

#[derive(Clone, Debug)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Lt,
    Gt,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let { name: String, value: Expr },
    Assign { name: String, value: Expr },
    Return(Expr),
    If { cond: Expr, then: Vec<Stmt>, els: Vec<Stmt> },
    While { cond: Expr, body: Vec<Stmt> },
    Emit(Expr),
    Yield,
    /// A bare expression evaluated for effect (e.g. a call).
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Unit,
    Var(String),
    Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Field { object: Box<Expr>, field: String },
    StructLit { name: String, fields: Vec<(String, Expr)> },
    Call { name: String, args: Vec<Expr> },
}
