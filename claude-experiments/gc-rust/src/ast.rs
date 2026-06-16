//! Surface AST for gc-rust — the output of the parser and the input to name
//! resolution + type checking. Carries source spans throughout for diagnostics.
//!
//! This is the *surface* tree (names as written, sugar intact). Later phases
//! lower it to a typed, monomorphic core IR (see `src/core.rs`, Phase 2).

use crate::lexer::{NumSuffix, Span};

#[derive(Clone, Debug)]
pub struct Module {
    pub items: Vec<Item>,
}

// ============================================================================
// Items
// ============================================================================

#[derive(Clone, Debug)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ItemKind {
    Fn(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplBlock),
    TypeAlias(TypeAlias),
    Const(ConstDef),
    Mod(ModDef),
    Use(UsePath),
}

#[derive(Clone, Debug)]
pub struct FnDef {
    pub vis: bool, // pub?
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    /// `self` receiver, if this is a method (`self` param). gc-rust passes self
    /// by value (GC reference under the hood) — there is no `&self`/`&mut self`.
    pub has_self: bool,
    /// Was the receiver declared `mut self`? Mutating the receiver (or assigning
    /// through it) is only allowed when this is true. See `docs/mutability.md`.
    pub self_is_mut: bool,
    pub ret: Option<Type>,
    pub body: Block,
    /// `true` for an `extern "C" fn name(...) -> ret;` declaration: a foreign C
    /// function with no body. Its `body` is empty and must not be lowered; the
    /// `name` is the unmangled C symbol. See `docs/ffi.md`.
    pub is_extern: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub is_mut: bool,
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub vis: bool,
    pub is_value: bool,
    pub name: String,
    pub generics: Generics,
    pub body: StructBody,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StructBody {
    Named(Vec<FieldDef>),
    Tuple(Vec<Type>),
    Unit,
}

#[derive(Clone, Debug)]
pub struct FieldDef {
    pub vis: bool,
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct EnumDef {
    pub vis: bool,
    pub is_value: bool,
    pub name: String,
    pub generics: Generics,
    pub variants: Vec<VariantDef>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct VariantDef {
    pub name: String,
    pub payload: VariantPayload,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum VariantPayload {
    None,
    Tuple(Vec<Type>),
    Named(Vec<FieldDef>),
}

#[derive(Clone, Debug)]
pub struct TraitDef {
    pub vis: bool,
    pub name: String,
    pub generics: Generics,
    pub supertraits: Vec<TraitRef>,
    pub items: Vec<TraitItem>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TraitItem {
    /// A required method signature (no body).
    Required(FnSig),
    /// A provided method (default body).
    Provided(FnDef),
    /// An associated type declaration.
    AssocType(String),
}

#[derive(Clone, Debug)]
pub struct FnSig {
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub has_self: bool,
    pub self_is_mut: bool,
    pub ret: Option<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ImplBlock {
    pub generics: Generics,
    /// `Some(trait)` for a trait impl, `None` for an inherent impl.
    pub trait_ref: Option<TraitRef>,
    pub self_ty: Type,
    pub items: Vec<FnDef>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TypeAlias {
    pub vis: bool,
    pub name: String,
    pub generics: Generics,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ConstDef {
    pub vis: bool,
    pub name: String,
    pub ty: Type,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ModDef {
    pub vis: bool,
    pub name: String,
    pub items: Vec<Item>,
    /// `true` for an inline `mod foo { ... }`; `false` for `mod foo;`, whose
    /// items are loaded from a sibling file (`foo.gcr` or `foo/mod.gcr`) by the
    /// compile pipeline before resolution.
    pub inline: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct UsePath {
    pub segments: Vec<String>,
    pub span: Span,
}

// ============================================================================
// Generics + trait bounds
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct Generics {
    pub params: Vec<TypeParam>,
    /// `where` clause predicates (in addition to inline `T: Bound`).
    pub where_clauses: Vec<WherePredicate>,
}

#[derive(Clone, Debug)]
pub struct TypeParam {
    pub name: String,
    pub bounds: Vec<TraitRef>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct WherePredicate {
    pub ty: Type,
    pub bounds: Vec<TraitRef>,
}

#[derive(Clone, Debug)]
pub struct TraitRef {
    pub path: Path,
    pub args: Vec<Type>,
    pub span: Span,
}

// ============================================================================
// Types
// ============================================================================

#[derive(Clone, Debug)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
    /// A named (possibly generic) type or a type variable: `i64`, `Vec<T>`,
    /// `geometry::Shape`, `T`, `Self`.
    Path(Path, Vec<Type>),
    /// `(A, B, ...)`; the empty tuple is unit `()`.
    Tuple(Vec<Type>),
    /// `[T; N]` fixed-size array.
    Array(Box<Type>, Box<Expr>),
    /// `fn(A, B) -> R`.
    Fn(Vec<Type>, Option<Box<Type>>),
    /// `extern fn(A, B) -> R` — a C function-pointer (callback) type. Distinct
    /// from `Fn` (a gc-rust closure): passing a named gc-rust function where this
    /// is expected synthesizes a C-ABI trampoline. See `docs/ffi.md`.
    ExternFn(Vec<Type>, Option<Box<Type>>),
    /// The `Self` type inside a trait/impl.
    SelfType,
}

/// A `::`-separated path, e.g. `Option::Some`, `geometry::area`.
#[derive(Clone, Debug)]
pub struct Path {
    pub segments: Vec<String>,
    pub span: Span,
}

impl Path {
    pub fn single(name: String, span: Span) -> Self {
        Path { segments: vec![name], span }
    }
    pub fn is_single(&self) -> bool {
        self.segments.len() == 1
    }
    pub fn last(&self) -> &str {
        self.segments.last().map(|s| s.as_str()).unwrap_or("")
    }
}

// ============================================================================
// Expressions
// ============================================================================

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: Box<ExprKind>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    // literals
    Int(u64, NumSuffix),
    Float(f64, NumSuffix),
    Str(String),
    Char(char),
    Bool(bool),
    Unit,

    /// A value path: a local var, a function, a unit/tuple enum variant, a
    /// const. Resolution decides which.
    Path(Path),

    /// Function/method/constructor application: `f(a, b)`.
    Call(Expr, Vec<Expr>),
    /// `recv.method(args)` — kept distinct from `Call` so resolution can do
    /// trait method lookup on the receiver type.
    MethodCall { recv: Expr, method: String, args: Vec<Expr>, span: Span },
    /// `base.field` or `base.0` (tuple index).
    Field { base: Expr, field: FieldAccess },
    /// `base[index]`.
    Index { base: Expr, index: Expr },

    Unary(UnOp, Expr),
    Binary(BinOp, Expr, Expr),
    /// `lhs op= rhs` (compound assignment) and plain `lhs = rhs`.
    Assign { target: Expr, op: Option<BinOp>, value: Expr },
    /// `e as T`.
    Cast(Expr, Type),

    /// `Name { field: expr, ... }` struct/variant literal.
    StructLit { path: Path, fields: Vec<FieldInit>, span: Span },
    /// `(a, b, ...)` tuple (len != 1; a 1-tuple is just a parenthesized expr).
    Tuple(Vec<Expr>),
    /// `[a, b, c]` or `[x; n]`.
    Array(ArrayLit),

    Block(Block),
    If { cond: Expr, then_branch: Block, else_branch: Option<Expr> },
    Match { scrutinee: Expr, arms: Vec<MatchArm> },
    While { cond: Expr, body: Block },
    Loop { body: Block },
    /// `for pat in iter { body }`.
    For { pat: Pattern, iter: Expr, body: Block },

    Closure { params: Vec<ClosureParam>, ret: Option<Type>, body: Expr },

    Return(Option<Expr>),
    Break(Option<Expr>),
    Continue,
    /// `expr?`.
    Try(Expr),
    /// `lo..hi` / `lo..=hi`.
    Range { lo: Option<Expr>, hi: Option<Expr>, inclusive: bool },
}

#[derive(Clone, Debug)]
pub enum FieldAccess {
    Named(String),
    Tuple(u32),
}

#[derive(Clone, Debug)]
pub struct FieldInit {
    pub name: String,
    /// `None` for field shorthand `Point { x }`.
    pub value: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ArrayLit {
    /// `[a, b, c]`
    Elems(Vec<Expr>),
    /// `[value; count]`
    Repeat(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct ClosureParam {
    pub name: String,
    pub ty: Option<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp { Neg, Not }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
}

// ============================================================================
// Statements + blocks
// ============================================================================

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// Trailing expression (the block's value), if any.
    pub tail: Option<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let {
        pattern: Pattern,
        ty: Option<Type>,
        init: Option<Expr>,
        span: Span,
    },
    Expr(Expr),
    Item(Item),
}

// ============================================================================
// Patterns
// ============================================================================

#[derive(Clone, Debug)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum PatternKind {
    Wildcard,
    /// A binding `x` or `mut x`. Could also be a unit variant — resolution
    /// disambiguates a bare uppercase path from a binding.
    Binding { is_mut: bool, name: String },
    Literal(LitPattern),
    /// `Enum::Variant(p, ...)` or a unit variant path.
    Variant { path: Path, payload: Vec<Pattern> },
    /// `Struct { field: pat, ... }`.
    Struct { path: Path, fields: Vec<(String, Pattern)>, rest: bool },
    Tuple(Vec<Pattern>),
}

#[derive(Clone, Debug)]
pub enum LitPattern {
    Int(u64),
    Bool(bool),
    Char(char),
    Str(String),
}
