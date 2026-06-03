//! Surface AST — what the parser produces.
//!
//! Unlike the canonical AST (which is content-addressed and stripped of
//! names and positions), the surface AST keeps source-level information:
//!
//! - Identifiers as strings (parameters, variables, function names).
//! - Byte spans for every node, used for error messages.
//!
//! Resolution is the pass that turns this into a canonical AST: locals
//! become de Bruijn indices, top-level names become content hashes, and
//! spans are dropped.

use crate::lexer::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub defs: Vec<SurfaceDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceDef {
    pub name: String,
    pub span: Span,
    pub kind: SurfaceDefKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceDefKind {
    Fn {
        is_local: bool,
        /// Type parameter names declared by this def, e.g. `<T, U>`.
        /// Scoped to this def's body; references inside become
        /// `Type::TypeVar(de_bruijn)` in canonical form. Empty for
        /// non-generic defs.
        type_params: Vec<String>,
        params: Vec<(String, SurfaceType)>,
        ret: SurfaceType,
        body: SurfaceExpr,
    },
    Struct {
        type_params: Vec<String>,
        fields: Vec<(String, SurfaceType)>,
    },
    Enum {
        type_params: Vec<String>,
        /// Each variant is `(name, payload_type)`. `None` = nullary.
        /// v1 restricts to 0 or 1 payload type per variant; multi-
        /// payload variants are expressible as a struct payload.
        variants: Vec<(String, Option<SurfaceType>)>,
    },
    /// `extern fn name(params) -> ret` — a declaration that binds
    /// `name` to a runtime-provided function. Calls lower to a
    /// `BuiltinRef("ext/<name>")` call. Externs are NOT part of the
    /// canonical AST and are not content-addressed — they carry only the
    /// surface signature.
    ///
    /// `library` distinguishes the two binding mechanisms:
    /// - `None` — a host (Rust) function from the process's extern
    ///   registry (e.g. `println`). Called with a leading `Thread*`.
    /// - `Some(lib)` — a real C symbol resolved from the shared library
    ///   `lib` via `dlopen`/`dlsym` (e.g. `getenv` from "c",
    ///   `curl_easy_init` from "curl"). Called with the plain C ABI, no
    ///   `Thread*`. Written `extern "C" lib "<lib>" { fn ... }`.
    Extern {
        params: Vec<(String, SurfaceType)>,
        ret: SurfaceType,
        library: Option<String>,
        /// `true` if the C declaration ended with `...` (a variadic
        /// function such as `curl_easy_setopt`). `params` then holds only
        /// the fixed leading parameters; each call site may pass
        /// additional trailing C-scalar arguments. Always `false` for a
        /// host (Rust) extern.
        variadic: bool,
    },

    /// `state NAME: TYPE = INIT` — a node-resident singleton binding.
    /// `INIT` is evaluated exactly once per node (at boot / first install)
    /// and its result is stored in the node's state table keyed by the
    /// binding's content hash; every reference resolves to that one live
    /// cell. Distinct from a `def`: a `def` is re-runnable; a `state` has
    /// node identity and is never re-evaluated. Typically `TYPE` is an
    /// `Atom<T>`, making the cell a shared mutable reference.
    State {
        ty: SurfaceType,
        init: SurfaceExpr,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceType {
    /// A bare type name: `Int`, `Bool`, `MyStruct`, ... Resolution decides
    /// whether it's a builtin or a user type.
    Named { name: String, span: Span },

    /// A generic type application: `Result<Int, Failure>`. `name` is
    /// the head type's surface name; `args` are the instantiation
    /// arguments in declaration order.
    Applied {
        name: String,
        name_span: Span,
        args: Vec<SurfaceType>,
        span: Span,
    },

    /// `fn(T1, T2, ..) -> R` — a function type. Used as the type of
    /// closure values.
    FnType {
        params: Vec<SurfaceType>,
        ret: Box<SurfaceType>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceExpr {
    IntLit {
        value: i64,
        span: Span,
    },
    FloatLit {
        value: f64,
        span: Span,
    },
    BoolLit {
        value: bool,
        span: Span,
    },
    StringLit {
        value: String,
        span: Span,
    },
    /// A bare identifier. Resolution determines whether it refers to a
    /// local, a top-level def, or a builtin.
    Var {
        name: String,
        span: Span,
    },
    /// `Enum::Variant` — a qualified enum-variant reference. Variant
    /// construction MUST be qualified (bare `Variant` is rejected by the
    /// resolver). In nullary position this stands alone; with a payload it
    /// is the callee of a `Call`.
    VariantRef {
        enum_name: String,
        variant_name: String,
        span: Span,
    },
    /// `callee(arg1, arg2, ...)` or `callee::<T, ...>(arg1, ...)`.
    /// `type_args` holds explicit type arguments (turbofish); empty when
    /// none were written.
    Call {
        callee: Box<SurfaceExpr>,
        args: Vec<SurfaceExpr>,
        type_args: Vec<SurfaceType>,
        span: Span,
    },
    BinOp {
        op: BinOp,
        left: Box<SurfaceExpr>,
        right: Box<SurfaceExpr>,
        span: Span,
    },
    /// Unary prefix: `-x`, `!x`.
    UnaryOp {
        op: UnaryOp,
        operand: Box<SurfaceExpr>,
        span: Span,
    },
    /// `{ stmt; stmt; tail_expr }`. The block evaluates to `tail`. Each
    /// `let` statement extends the lexical environment for the rest of
    /// the block (including `tail`).
    Block {
        stmts: Vec<SurfaceStmt>,
        tail: Box<SurfaceExpr>,
        span: Span,
    },
    /// `|x: T1, y: T2| body` — a closure literal. Parameter types are
    /// required in v1 (no inference yet). Captures are implicit; the
    /// resolver and codegen recover them from the body's free vars.
    Lambda {
        params: Vec<(String, SurfaceType)>,
        body: Box<SurfaceExpr>,
        span: Span,
    },
    /// `Point { x: 1, y: 2 }` — struct literal. Field assignments may
    /// be in any order at the surface; the resolver reorders them to
    /// declaration order in the canonical form.
    StructLit {
        type_name: String,
        type_name_span: Span,
        fields: Vec<(String, SurfaceExpr)>,
        span: Span,
    },
    /// `expr.field` — read a named field from a struct value.
    FieldAccess {
        base: Box<SurfaceExpr>,
        field_name: String,
        field_span: Span,
        span: Span,
    },
    /// `expr?` — the try operator. `expr` must be a `Result<T, E>`
    /// (a 2-variant enum `Ok(T) | Err(E)`). Evaluates to `T` when the
    /// value is `Ok`, and early-returns the `Err` value from the
    /// enclosing function otherwise.
    Try {
        expr: Box<SurfaceExpr>,
        span: Span,
    },
    /// `match scrutinee { pat => expr, … }`.
    Match {
        scrutinee: Box<SurfaceExpr>,
        arms: Vec<SurfaceMatchArm>,
        span: Span,
    },
    /// `if cond { then_branch } else { else_branch }`. Both branches
    /// must produce the same type. `cond` is `Int` (truthy iff != 0);
    /// we don't have first-class `Bool` lowering yet, so this matches
    /// the existing convention used by comparison builtins.
    If {
        cond: Box<SurfaceExpr>,
        then_branch: Box<SurfaceExpr>,
        else_branch: Box<SurfaceExpr>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMatchArm {
    pub pattern: SurfacePattern,
    pub body: SurfaceExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfacePattern {
    /// `_` — wildcard.
    Wildcard { span: Span },
    /// `ident` — binds a name. Note: this is also the surface form for
    /// nullary constructor patterns. The resolver disambiguates by
    /// looking up the identifier in scope (variant → constructor
    /// pattern; otherwise binding).
    Ident { name: String, span: Span },
    /// `Name(pat)` — constructor pattern with one payload sub-pattern.
    Ctor {
        name: String,
        payload: Box<SurfacePattern>,
        span: Span,
    },
    /// `Enum::Variant` or `Enum::Variant(pat)` — a qualified variant
    /// pattern. Variant patterns MUST be qualified; bare ones are rejected
    /// by the resolver (a bare ident is always a binding now).
    QualifiedCtor {
        enum_name: String,
        variant_name: String,
        payload: Option<Box<SurfacePattern>>,
        span: Span,
    },
}

impl SurfacePattern {
    pub fn span(&self) -> Span {
        match self {
            SurfacePattern::Wildcard { span }
            | SurfacePattern::Ident { span, .. }
            | SurfacePattern::Ctor { span, .. }
            | SurfacePattern::QualifiedCtor { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceStmt {
    /// `let ident = expr;` — type annotations not yet supported.
    Let {
        name: String,
        value: SurfaceExpr,
        span: Span,
    },
    /// `defer expr;` — register a deterministic cleanup. `expr` runs for
    /// its side effect when the enclosing block exits, on normal
    /// completion and on any `?` early-return that unwinds through it,
    /// in LIFO order. Adds no binding. See [`crate::ast::Expr::Defer`].
    Defer { expr: SurfaceExpr, span: Span },
}

impl SurfaceExpr {
    pub fn span(&self) -> Span {
        match self {
            SurfaceExpr::IntLit { span, .. }
            | SurfaceExpr::FloatLit { span, .. }
            | SurfaceExpr::BoolLit { span, .. }
            | SurfaceExpr::StringLit { span, .. }
            | SurfaceExpr::Var { span, .. }
            | SurfaceExpr::VariantRef { span, .. }
            | SurfaceExpr::Call { span, .. }
            | SurfaceExpr::BinOp { span, .. }
            | SurfaceExpr::UnaryOp { span, .. }
            | SurfaceExpr::Block { span, .. }
            | SurfaceExpr::Lambda { span, .. }
            | SurfaceExpr::StructLit { span, .. }
            | SurfaceExpr::FieldAccess { span, .. }
            | SurfaceExpr::Try { span, .. }
            | SurfaceExpr::Match { span, .. }
            | SurfaceExpr::If { span, .. } => *span,
        }
    }
}

impl SurfaceType {
    pub fn span(&self) -> Span {
        match self {
            SurfaceType::Named { span, .. }
            | SurfaceType::Applied { span, .. }
            | SurfaceType::FnType { span, .. } => *span,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}
