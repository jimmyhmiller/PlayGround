//! Abstract syntax + types for lambda-Tally (v0.2: a type-directed core).
//!
//! The L3 split is now in the *types*:
//!   * `Own<S>` — a LINEAR owning capability to a heap cell of struct `S`
//!     (must be consumed exactly once: free, or moved). Dereferenceable.
//!   * `Ptr<S>` — an UNRESTRICTED (copyable) bare address into such a cell;
//!     carries no permission, so it CANNOT be dereferenced.
//! Field types must be copyable (never `Own`), which is exactly the invariant
//! "you cannot fabricate a capability by reading memory".

use std::fmt;

/// A type-level natural-number index, in normal form: a literal `k`, or a
/// variable plus an offset `n + k`. Definitional equality is structural on this
/// form, which is exactly enough for the length arithmetic of `Vec<n>`.
#[derive(Clone, PartialEq, Debug)]
pub enum Idx {
    Lit(u32),
    Var(String, u32),
}

impl Idx {
    pub fn succ(&self) -> Idx {
        match self {
            Idx::Lit(k) => Idx::Lit(k + 1),
            Idx::Var(n, k) => Idx::Var(n.clone(), k + 1),
        }
    }
    /// is this index provably ≥ 1 (so a Vec of this length is non-empty)?
    pub fn nonempty(&self) -> bool {
        match self {
            Idx::Lit(k) => *k >= 1,
            Idx::Var(_, k) => *k >= 1,
        }
    }
    pub fn pred(&self) -> Idx {
        match self {
            Idx::Lit(k) => Idx::Lit(k.saturating_sub(1)),
            Idx::Var(n, k) => Idx::Var(n.clone(), k.saturating_sub(1)),
        }
    }
    pub fn is_zero(&self) -> bool {
        matches!(self, Idx::Lit(0))
    }
}

impl fmt::Display for Idx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Idx::Lit(k) => write!(f, "{k}"),
            Idx::Var(n, 0) => write!(f, "{n}"),
            Idx::Var(n, k) => write!(f, "{n}+{k}"),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Ty {
    Unit,
    Int,
    /// a type-level natural number (only ever used erased, at multiplicity 0)
    Nat,
    /// linear owning capability to a cell of the named struct
    Own(String),
    /// unrestricted bare address into a cell of the named struct
    Ptr(String),
    /// a LINEAR length-indexed vector (stack); the index is erased at runtime
    Vec(Idx),
}

impl Ty {
    pub fn is_linear(&self) -> bool {
        matches!(self, Ty::Own(_) | Ty::Vec(_))
    }
    pub fn is_copyable(&self) -> bool {
        !self.is_linear()
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Unit => write!(f, "Unit"),
            Ty::Int => write!(f, "Int"),
            Ty::Nat => write!(f, "Nat"),
            Ty::Own(s) => write!(f, "Own<{s}>"),
            Ty::Ptr(s) => write!(f, "Ptr<{s}>"),
            Ty::Vec(i) => write!(f, "Vec<{i}>"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
}

use crate::mult::Mult;

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Null,
    Unit,
    Var(String),
    /// `alloc S { f: e, ... }` — allocate a cell of struct `S`; yields `Own<S>`.
    Alloc(String, Vec<(String, Expr)>),
    /// `addr(x)` — the copyable `Ptr<S>` of an owned `x : Own<S>` (no consume).
    AddrOf(String),
    /// `e.f` — read field `f` (the base must be `Own<S>`; yields the field type).
    Field(Box<Expr>, String),
    /// `f(a1, ..., an)` — call a top-level function.
    Call(String, Vec<Expr>),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let name (: Ty)? = rhs;`
    Let(String, Option<Ty>, Expr),
    /// `base.fld = rhs;`  (base must be `Own<S>`)
    Write(Expr, String, Expr),
    /// `free name;`  (consume the `Own`)
    Free(String),
    Expr(Expr),
}

/// A function parameter: an optional multiplicity budget, a name, and a type.
/// If `mult` is `None`, the budget defaults from the type (`Own` → 1, else ω).
#[derive(Clone, Debug)]
pub struct Param {
    pub mult: Option<Mult>,
    pub name: String,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Ty,
    pub body: Vec<Stmt>,
    /// the trailing return expression (absent ⇒ returns `Unit`).
    pub tail: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub structs: Vec<StructDecl>,
    pub funcs: Vec<Func>,
}
