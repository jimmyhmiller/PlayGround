//! Abstract syntax for the lambda-Tally surface language (v0 core).
//!
//! v0 is the L3 address/permission core: `alloc` / read / write / `free`, with
//! `Addr` (unrestricted, copyable) split from the linear `Perm` (erased).

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Null,
    Unit,
    Var(String),
    /// allocate a cell with the given fields; yields a linear owning handle.
    Alloc(Vec<(String, Expr)>),
    /// the COPYABLE address of an owned cell; does NOT spend its Perm.
    AddrOf(String),
    /// read field `fld` of `obj` (needs `obj`'s Perm).
    Field(Box<Expr>, String),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let name = rhs;`
    Let(String, Expr),
    /// `base.fld = rhs;`  (write through the base's Perm)
    Write(Expr, String, Expr),
    /// `free name;`  (consume the Perm, reclaim the cell)
    Free(String),
    /// expression statement
    Expr(Expr),
}

pub type Program = Vec<Stmt>;
