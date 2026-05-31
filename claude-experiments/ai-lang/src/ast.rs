//! Canonical AST.
//!
//! This is the on-disk and over-wire representation of code. It is the input
//! to the content-address hasher: two ASTs that encode to the same bytes have
//! the same hash and are the same definition.
//!
//! Invariants:
//!
//! - No source positions, no identifiers for locals, no original names of
//!   anything that isn't load-bearing for behaviour. Local variables are
//!   de Bruijn indices.
//! - Top-level references are encoded as 32-byte content hashes
//!   (`TopRef`) or as `(component_index, member_index)` for members of the
//!   currently-being-hashed mutually-recursive group (`SelfRef`). The
//!   resolver substitutes `SelfRef` placeholders for `TopRef` after the
//!   component is finalised.
//! - Builtins are referred to by stable string IDs ("core/i64.mul",
//!   "core/println", "net/at"). The set of valid builtins is a runtime
//!   contract negotiated at node handshake.

use crate::hash::Hash;

/// A canonical expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),

    /// De Bruijn index into the lexical environment. 0 is the innermost
    /// binder (rightmost parameter of the enclosing lambda or the most
    /// recently introduced `let`).
    LocalVar(u32),

    /// Reference to another `def` by its content hash.
    TopRef(Hash),

    /// Reference to a member of the currently-being-hashed mutually-recursive
    /// component, by member index. After the component is hashed, the
    /// resolver replaces these with `TopRef`s in stored copies; they only
    /// appear during hashing.
    SelfRef(u32),

    /// Reference to a builtin by stable string id.
    BuiltinRef(String),

    /// Function application.
    Call(Box<Expr>, Vec<Expr>),

    /// Lambda with explicitly-typed parameters. The body's de Bruijn
    /// environment is extended by `params.len()` (param 0 is index
    /// N-1, param N-1 is index 0). The lambda implicitly captures all
    /// free variables in its body; captures are recovered by walking
    /// the body and listing the `LocalVar(i)` with `i >= params.len()`.
    ///
    /// Parameter types are part of the canonical form (and therefore
    /// part of the hash) — `|x: Int| x` and `|x: fn(Int)->Int| x` are
    /// different definitions even though their bodies match. Param
    /// types also let codegen classify captures as pointer vs. value
    /// so the GC can trace closures precisely.
    Lambda {
        params: Vec<Type>,
        body: Box<Expr>,
    },

    /// `let x = value in body`. The body's environment has one additional
    /// binder (the new local at index 0).
    Let { value: Box<Expr>, body: Box<Expr> },

    /// `defer cleanup; body` — register a deterministic cleanup. `body`
    /// (the rest of the enclosing block) is evaluated and its value is
    /// the value of the whole `Defer`; `cleanup` runs for its side effect
    /// AFTER `body` finishes, on every way out: normal completion and any
    /// `?` early-return that unwinds through this point. Cleanups run in
    /// LIFO order. `cleanup` does NOT add a binder; it is evaluated in the
    /// same environment as `body` (its free variables are the locals in
    /// scope where the `defer` appears). The deferred value is discarded,
    /// so `cleanup` is typically a `free`/`fclose`-style call returning
    /// Int. This is the deterministic alternative to GC finalizers for
    /// releasing C-FFI resources.
    Defer { cleanup: Box<Expr>, body: Box<Expr> },

    /// Construct a struct value: allocate on the heap, populate fields
    /// in declaration order.
    ///
    /// `struct_ref` is the content hash of the `Def::Struct` this
    /// constructs. `fields` are in the order declared by that struct
    /// (not in the order written at the call site).
    StructNew {
        struct_ref: Hash,
        fields: Vec<Expr>,
    },

    /// Read a field from a struct value. `index` is the field's
    /// position in the struct's declaration. `struct_ref` is the hash
    /// of the struct definition the access is against — embedded so
    /// codegen knows the field's offset and type without runtime
    /// dispatch, and so the canonical form is complete (the field's
    /// meaning is unambiguous in isolation).
    Field {
        base: Box<Expr>,
        struct_ref: Hash,
        index: u32,
    },

    /// Construct an enum value: allocate a per-variant heap object,
    /// store the discriminant tag, then the payload (if any).
    ///
    /// `payload` is `None` for nullary variants and `Some(expr)` for
    /// variants declared to carry one. (v1 restricts enums to 0-or-1
    /// payload per variant; multi-payload encodes as a struct payload.)
    EnumNew {
        enum_ref: Hash,
        variant_index: u32,
        payload: Option<Box<Expr>>,
    },

    /// Pattern-match on a scrutinee. Arms are tried in order; v1 codegen
    /// assumes arms are EXHAUSTIVE — non-exhaustive matches will fall
    /// through to a trap. A future typechecker pass will enforce this.
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// `if cond { then_branch } else { else_branch }`. `cond` is Int
    /// (truthy iff non-zero — same convention as comparison
    /// builtins which return 0/1 widened to i64). Both branches must
    /// produce the same type.
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },

    /// `expr?` — the try operator. `expr` evaluates to a `Result<T, E>`
    /// (the 2-variant enum identified by `enum_ref`, with `Ok` at
    /// `ok_index` carrying `T` and `Err` at `err_index` carrying `E`).
    /// Evaluates to the `Ok` payload (`T`) on success; on `Err` it
    /// early-returns the whole `Result` value from the enclosing
    /// function (whose return type must be a `Result<_, E>`).
    Try {
        expr: Box<Expr>,
        enum_ref: Hash,
        ok_index: u32,
        err_index: u32,
    },
}

/// A `match` arm: a pattern and the expression that runs when the
/// pattern matches.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
}

/// Canonical pattern. Variable bindings have no name (de Bruijn handles
/// that — the binding's index is determined by the pattern's position
/// in the lexical environment of the arm body).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    /// `_` — matches anything, binds nothing.
    Wildcard,
    /// Identifier — matches anything, binds one local (innermost).
    Var,
    /// `Variant(sub_pattern)` or `Variant`. `enum_ref` + `variant_index`
    /// identify which variant; `payload` is None for nullary and
    /// `Some(sub_pattern)` for variants with a payload.
    Enum {
        enum_ref: Hash,
        variant_index: u32,
        payload: Option<Box<Pattern>>,
    },
}

/// A canonical type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Built-in type: "Int", "Bool", "String", "Float", "Bytes", ...
    Builtin(String),

    /// Reference to a user-defined struct/enum by content hash.
    TypeRef(Hash),

    /// Type-parameter de Bruijn index, scoped to the enclosing `Def`.
    TypeVar(u32),

    /// Type application: `List<Int>` is `Apply(TypeRef(list_hash), [Builtin("Int")])`.
    Apply(Box<Type>, Vec<Type>),

    /// `fn(A, B) -> R`
    FnType { params: Vec<Type>, ret: Box<Type> },

    /// Reference to a member of the currently-being-hashed type SCC,
    /// by member index. Analogous to `Expr::SelfRef`. After an SCC
    /// of struct/enum defs is hashed, the resolver rewrites each
    /// `SelfRef(i)` in the *stored* canonical AST to `TopRef(hash_of_member_i)`;
    /// `SelfRef` only ever appears in the bytes that get hashed, not
    /// in stored / typechecked / codegen'd ASTs.
    SelfRef(u32),
}

/// A canonical top-level definition.
#[derive(Debug, Clone, PartialEq)]
pub enum Def {
    /// Function definition. Parameter names are dropped; only types remain.
    Fn {
        /// `true` for `def local` — per-node singleton. Hash-distinguishing.
        is_local: bool,
        /// Number of type parameters introduced for this def. Type-parameter
        /// references inside `params`/`ret`/`body` use `TypeVar(de_bruijn)`.
        type_params: u32,
        params: Vec<Type>,
        ret: Type,
        body: Expr,
    },

    /// Struct (product) type. Field names ARE part of the canonical
    /// form and the hash — renaming a field produces a new, different
    /// struct. Fields are listed in declaration order; expressions
    /// reference them by positional index (`Expr::Field`).
    Struct {
        type_params: u32,
        fields: Vec<(String, Type)>,
    },

    /// Enum (sum) type. Variants are listed in declaration order;
    /// expressions reference them by positional index. Variant names
    /// ARE part of the hash (renaming a variant produces a new enum,
    /// just like a struct field rename).
    ///
    /// Each variant has 0 or 1 payload type in v1. Multi-payload
    /// variants can be expressed by composing a struct payload.
    Enum {
        type_params: u32,
        variants: Vec<(String, Option<Type>)>,
    },
}
