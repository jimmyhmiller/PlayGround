//! Core AST. M0/M1 is deliberately i64-only; the `Type` enum exists so the
//! type checker and codegen are shaped for richer types later.

use std::collections::HashMap;

use crate::convention::Convention;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I64,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Var(String),
    Let {
        binds: Vec<(String, Expr)>,
        body: Vec<Expr>,
    },
    Bin {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Cmp {
        op: CmpOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
    },
    Do(Vec<Expr>),
    Call {
        func: String,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: String,
    /// Name of the calling convention this function uses.
    pub cc: String,
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub conventions: HashMap<String, Convention>,
    pub funcs: Vec<Func>,
}

impl Program {
    pub fn func(&self, name: &str) -> Option<&Func> {
        self.funcs.iter().find(|f| f.name == name)
    }
}
