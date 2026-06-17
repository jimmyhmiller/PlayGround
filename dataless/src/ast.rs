//! Syntax trees for programs (.dl) and declarations (.decl).

use crate::repr::{Rep, Type};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Text(String),
    Bool(bool),
    Nil,
    /// A local program variable.
    Var(String),
    /// The current member of the innermost active iteration (FOR EACH/generator).
    It,
    /// The single canonical reference form: `name(index)`. Whether `name` is a
    /// stored field or a computed function is resolved against the declarations,
    /// not the program — this is the core of Dataless Programming.
    Ref(String, Box<Expr>),
    /// `size(collection)`.
    Size(String),
    /// `member(collection, position)` — positional lookup; returns a handle.
    MemberAt(String, Box<Expr>),
    /// `insert collection` — appends a member, returns its handle.
    Insert(String),
    /// `insert after <handle-expr> in collection` — returns the new handle.
    InsertAfter(String, Box<Expr>),
    /// Search expression: `there exists <var> in <coll> such that (cond)`.
    Exists {
        var: String,
        coll: String,
        cond: Box<Expr>,
    },
    /// `next <generator>` — advance a generator; yields the new handle or nil.
    Next(String),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let(String, Expr),
    AssignVar(String, Expr),
    /// `name(index) = value` — store into a field (ignored if `name` is computed).
    /// A bare `name = value` where `name` is a field is resolved at runtime to
    /// implied qualification (store into the current member).
    AssignRef(String, Expr, Expr),
    Print(Vec<Expr>),
    ExprStmt(Expr),
    /// `delete <handle>` — the handle carries its collection.
    Delete(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    /// `repeat <n> with <var> { body }` — var runs 1..=n.
    Repeat(Expr, String, Vec<Stmt>),
    /// `for each <coll> [such that (cond)] { body }`.
    ForEach {
        coll: String,
        cond: Option<Expr>,
        body: Vec<Stmt>,
    },
    /// `generate <g> over <coll> such that (cond)` — set up a lazy cursor.
    Generate {
        gen: String,
        coll: String,
        cond: Expr,
    },
    /// `whenever (cond) { body }` — a STATE monitor; fires on false→true.
    Whenever(Expr, Vec<Stmt>),
}

pub type Program = Vec<Stmt>;

// ----- declarations ------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FieldDecl {
    pub name: String,
    pub ty: Type,
    pub init: Option<Expr>,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Collection {
        name: String,
        rep: Rep,
        fields: Vec<FieldDecl>,
    },
    /// `computed <name>(<param>) = <expr>` — defines `name` as a function that
    /// is referenced exactly like a stored field: `name(handle)`.
    Computed {
        name: String,
        param: String,
        body: Expr,
    },
}

pub type Decls = Vec<Decl>;
