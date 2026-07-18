//! The surface AST — a Rust-flavored syntax over the runtime's IR.

/// A written-down type: `i64`, `bool`, `()`, or `&StructName` (a reference to a
/// user-defined struct; all struct values are references).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeExpr {
    I64,
    F64,
    Bool,
    Str,
    Unit,
    /// `[T]` — a mutable, growable array.
    Array(Box<TypeExpr>),
    /// `fn(T, U) -> R` — a first-class function value.
    Fn(Vec<TypeExpr>, Box<TypeExpr>),
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

/// `enum Shape { Circle { r: i64 }, Point }` — a sum type; each variant has its
/// own (possibly empty) fields.
#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<VariantDef>,
}

#[derive(Clone, Debug)]
pub struct VariantDef {
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

/// A native function declaration: `foreign fn draw(w: Window, n: i64);`. Only
/// the signature is written here; the implementation is registered on the
/// runtime by the host. This is the managed → native boundary.
#[derive(Clone, Debug)]
pub struct ForeignFnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: TypeExpr,
}

/// A persistent top-level binding: `letonce win = open_window();`. Its
/// initializer runs once and the value survives hot edits — where native
/// resources live so a reload changes code, not the running world.
#[derive(Clone, Debug)]
pub struct GlobalDef {
    pub name: String,
    pub init: Expr,
}

#[derive(Clone, Debug)]
pub enum Item {
    Struct(StructDef),
    Enum(EnumDef),
    Fn(FnDef),
    /// `foreign type Window;` — declares an opaque native resource type.
    ForeignType(String),
    ForeignFn(ForeignFnDef),
    Global(GlobalDef),
}

#[derive(Clone, Debug)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let x = e;` or `let x: T = e;` — the annotation is optional and only
    /// *required* where nothing else pins the type (an empty array literal).
    Let { name: String, ty: Option<TypeExpr>, value: Expr },
    Assign { name: String, value: Expr },
    Return(Expr),
    If { cond: Expr, then: Vec<Stmt>, els: Vec<Stmt> },
    While { cond: Expr, body: Vec<Stmt> },
    Emit(Expr),
    Yield,
    /// `match e { Circle { r } => { … } Point => { … } }` — arms bind the
    /// variant's fields by name. Exhaustiveness is enforced by the verifier
    /// against the enum's *current* version, so adding a variant invalidates
    /// stale matches instead of surprising them at runtime.
    Match {
        scrutinee: Expr,
        arms: Vec<MatchArm>,
    },
    /// `a[i] = e;` — element assignment (checked against the element type).
    IndexAssign { array: Expr, index: Expr, value: Expr },
    /// A bare expression evaluated for effect (e.g. a call).
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub variant: String,
    /// Field names bound as locals inside the arm body.
    pub bindings: Vec<String>,
    pub body: Vec<Stmt>,
}

/// One arm of an expression `match`: `Variant { a, b } => expr`.
#[derive(Clone, Debug)]
pub struct ExprArm {
    pub variant: String,
    pub bindings: Vec<String>,
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// A string literal (interned at lowering).
    Str(String),
    Unit,
    Var(String),
    Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Not(Box<Expr>),
    /// Unary minus — type-directed sugar for `0 - x` / `0.0 - x`.
    Neg(Box<Expr>),
    /// `[e1, e2, …]`; the element type is the first item's. `[]` needs a
    /// `let` annotation.
    ArrayLit(Vec<Expr>),
    /// `a[i]`.
    Index { array: Box<Expr>, index: Box<Expr> },
    /// `match` in expression position: every arm yields a value of one type.
    Match { scrutinee: Box<Expr>, arms: Vec<ExprArm> },
    Field { object: Box<Expr>, field: String },
    StructLit { name: String, fields: Vec<(String, Expr)> },
    /// `Shape::Circle { r: 5 }` (or a bare fieldless `Shape::Point`).
    VariantLit { enum_name: String, variant: String, fields: Vec<(String, Expr)> },
    Call { name: String, args: Vec<Expr> },
}
