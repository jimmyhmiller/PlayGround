//! Core AST. M0/M1 is deliberately i64-only; the `Type` enum exists so the
//! type checker and codegen are shaped for richer types later.

use std::collections::HashMap;

use crate::convention::Convention;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Integer of the given bit width (8, 16, 32, 64).
    Int(u32),
    /// A pointer whose *region* and *pointee type* are both part of the type.
    /// `Ptr(C, i8)` is a C `char*`.
    Ptr(Region, Box<Type>),
    /// A named struct (defined by `defstruct`).
    Struct(String),
    /// A fixed-size array of `len` elements.
    Array(Box<Type>, u32),
}

impl Type {
    pub fn i64() -> Type {
        Type::Int(64)
    }
}

/// Where allocated memory lives. Part of a pointer's type, so `Ptr(Heap, _)`
/// and `Ptr(Static, _)` are distinct types and `free` only accepts `Ptr(Heap)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Current stack frame (`alloca`). May not cross a function boundary.
    Frame,
    /// Module-static storage (a global). Lives forever.
    Static,
    /// libc heap (`malloc`/`free`). Must be freed.
    Heap,
    /// A raw/foreign pointer (from C, the OS, hand-written asm). Not owned by
    /// Coil's allocators: no `free`, no escape rules — just a machine pointer.
    C,
}

impl Region {
    pub fn parse(s: &str) -> Option<Region> {
        match s {
            "frame" => Some(Region::Frame),
            "static" => Some(Region::Static),
            "heap" => Some(Region::Heap),
            "c" => Some(Region::C),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Region::Frame => "frame",
            Region::Static => "static",
            Region::Heap => "heap",
            Region::C => "c",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
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
    /// Allocate one value of `ty` in `region`, yielding `Ptr(region, ty)`.
    Alloc {
        region: Region,
        ty: Type,
    },
    /// Pointer to a struct field: `Ptr(R, Struct S)` and a field name →
    /// `Ptr(R, fieldtype)`.
    Field {
        ptr: Box<Expr>,
        field: String,
    },
    /// Read the value a pointer points at (type = the pointee type).
    Load(Box<Expr>),
    /// Write a value through a pointer; evaluates to the stored value.
    Store {
        ptr: Box<Expr>,
        val: Box<Expr>,
    },
    /// Pointer + index (in elements): `Ptr(R,T)` and an i64 → `Ptr(R,T)`.
    Index {
        ptr: Box<Expr>,
        idx: Box<Expr>,
    },
    /// Integer width conversion (sign-extend / truncate).
    Cast {
        ty: Type,
        expr: Box<Expr>,
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

/// A foreign function declaration: a name, a calling convention, and a typed
/// signature, with no body. Calls are checked against it; it lowers to an
/// external LLVM declaration the linker resolves (libc, hand-written asm, ...).
#[derive(Debug, Clone)]
pub struct Extern {
    pub name: String,
    pub cc: String,
    pub params: Vec<Type>,
    pub ret: Type,
}

/// A named struct definition: ordered (field-name, field-type) pairs.
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub conventions: HashMap<String, Convention>,
    pub structs: Vec<StructDef>,
    pub externs: Vec<Extern>,
    pub funcs: Vec<Func>,
}

impl Program {
    pub fn func(&self, name: &str) -> Option<&Func> {
        self.funcs.iter().find(|f| f.name == name)
    }
}
