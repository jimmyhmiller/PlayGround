//! Core AST. M0/M1 is deliberately i64-only; the `Type` enum exists so the
//! type checker and codegen are shaped for richer types later.

use std::collections::HashMap;

use crate::convention::Convention;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I64,
    /// A pointer whose *region* (where the pointee lives) is part of the type.
    /// Pointees are i64 in M3; the interesting axis is the region.
    Ptr(Region),
}

/// Where allocated memory lives. Part of a pointer's type, so `Ptr(Heap)` and
/// `Ptr(Static)` are distinct types and (e.g.) `free` only accepts `Ptr(Heap)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Current stack frame (`alloca`). May not cross a function boundary.
    Frame,
    /// Module-static storage (a global). Lives forever.
    Static,
    /// libc heap (`malloc`/`free`). Must be freed.
    Heap,
}

impl Region {
    pub fn parse(s: &str) -> Option<Region> {
        match s {
            "frame" => Some(Region::Frame),
            "static" => Some(Region::Static),
            "heap" => Some(Region::Heap),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Region::Frame => "frame",
            Region::Static => "static",
            Region::Heap => "heap",
        }
    }
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
    /// Allocate one i64 slot in `region`, yielding `Ptr(region)`.
    Alloc {
        region: Region,
    },
    /// Read the i64 a pointer points at.
    Load(Box<Expr>),
    /// Write an i64 through a pointer; evaluates to the stored value.
    Store {
        ptr: Box<Expr>,
        val: Box<Expr>,
    },
    /// Release a heap pointer; evaluates to 0.
    Free(Box<Expr>),
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
