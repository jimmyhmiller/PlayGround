//! Core AST. M0/M1 is deliberately i64-only; the `Type` enum exists so the
//! type checker and codegen are shaped for richer types later.

use std::collections::HashMap;

use crate::convention::Convention;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Integer of an arbitrary bit width, signed or unsigned (Zig-style: `u2`,
    /// `i7`, `u23`, ...). LLVM has native `iN`; signedness drives the operations.
    Int(u32, bool), // (bits, signed)
    /// IEEE floating point: `f32` or `f64` (bits = 32 or 64).
    Float(u32),
    /// A pointer to a value of the pointee type. Pointers are region-less (a
    /// pointer is a pointer, à la Zig/C); where the memory came from is the
    /// `alloc` operation's concern, not the type's.
    Ptr(Box<Type>),
    /// A reference to a place holding the pointee type — the everyday tier above
    /// `Ptr`. `mutable` is opt-in: an immutable reference may be read but not
    /// written through. Same machine representation as `Ptr`; the checker erases
    /// it to `Ptr` (after the const-correctness check) before codegen, so this
    /// never reaches monomorphization or codegen.
    Ref(bool, Box<Type>), // (mutable, pointee)
    /// A named struct (defined by `defstruct`).
    Struct(String),
    /// A fixed-size array of `len` elements.
    Array(Box<Type>, u32),
    /// A function pointer: calling convention, parameter types, return type.
    Fn(String, Vec<Type>, Box<Type>),
    /// A generic type application, e.g. `(Pair i64 i64)` -> `App("Pair", [i64,i64])`.
    /// Removed by monomorphization before the checker runs.
    App(String, Vec<Type>),
}

impl Type {
    pub fn i64() -> Type {
        Type::Int(64, true)
    }
    /// True for any integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, Type::Int(..))
    }
}

/// How an `alloc` operation obtains memory. This is *not* part of the pointer
/// type — every `alloc` yields a plain `(ptr T)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// Current stack frame (`alloca`).
    Stack,
    /// A module global.
    Static,
    /// libc `malloc` (the inline convenience; the controllable path is an
    /// allocator value — see `lib/alloc.coil`).
    Heap,
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
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
    /// A string literal. Lowers to a private, NUL-terminated `[N x i8]` global;
    /// its value is a `(ptr i8)` to the first byte (C-string compatible).
    Str(String),
    Var(String),
    /// The zero value of a type (`(zeroed T)`) — used to initialize a fresh
    /// `let`-bound stack local. Codegen lowers it to LLVM's `zeroinitializer`.
    Zeroed(Type),
    /// Borrow a place as a reference: `(mut x)` (mutable) or the implicit
    /// immutable borrow inserted when a place is used where a reference is
    /// expected. The checker erases it to the underlying pointer.
    Borrow {
        mutable: bool,
        place: Box<Expr>,
    },
    Let {
        /// Each binding is (name, mutable, value). A `mutable` binding becomes a
        /// stack place (its name is a reference you can write through); a plain
        /// binding is an ordinary immutable value as before.
        binds: Vec<(String, bool, Expr)>,
        body: Vec<Expr>,
    },
    Bin {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Bitwise complement `(inot x)` (`~x`).
    Not(Box<Expr>),
    /// A floating-point literal (defaults to `f64`, adopts `f32` from context).
    Float(f64),
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
        /// Explicit type arguments for a generic call: `(id [i64] 5)`. Empty for
        /// ordinary calls.
        type_args: Vec<Type>,
        args: Vec<Expr>,
    },
    /// Allocate one value of `ty` using `storage`, yielding `(ptr ty)`.
    Alloc {
        storage: Storage,
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
    /// Integer width conversion (sign-extend / truncate), or pointer reinterpret.
    Cast {
        ty: Type,
        expr: Box<Expr>,
    },
    /// Size in bytes of a type's layout, as an i64 constant.
    SizeOf(Type),
    /// Alignment in bytes of a type, as an i64 constant.
    AlignOf(Type),
    /// Byte offset of a field within a struct, as an i64 constant.
    OffsetOf(Type, String),
    /// Read a bitfield by value (yields the field's `uN` type).
    BitGet {
        ptr: Box<Expr>,
        field: String,
    },
    /// Write a bitfield (read-modify-write the backing integer).
    BitSet {
        ptr: Box<Expr>,
        field: String,
        val: Box<Expr>,
    },
    /// Release a heap pointer; evaluates to 0.
    Free(Box<Expr>),
    /// Construct a sum-type variant. Produced by monomorphization from a call to
    /// a variant name; `sum` is the concrete (post-mono) sum type name.
    Construct {
        sum: String,
        variant: String,
        args: Vec<Expr>,
    },
    /// Pattern match on a sum value: one arm per variant, binding its fields.
    Match {
        scrut: Box<Expr>,
        arms: Vec<Arm>,
    },
    /// The address of a named function/extern, as a function pointer value.
    FnPtrOf(String),
    /// Indirect call through a function pointer value.
    CallPtr {
        fp: Box<Expr>,
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
    /// Generic type parameters (empty for an ordinary function).
    pub type_params: Vec<String>,
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

/// How a struct is laid out in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    /// Target C ABI: natural alignment + padding (the default; for FFI).
    C,
    /// No padding between fields.
    Packed,
    /// C layout but with the whole struct force-aligned to N bytes.
    Aligned(u32),
    /// Total control: each field placed at an explicit byte offset (parallel to
    /// the field list). Realized as a byte blob; overlapping offsets = a union.
    Explicit(ExplicitLayout),
    /// Bitfields: every field is a sub-byte run of bits packed (LSB-first) into a
    /// single backing integer. Fields are accessed by value via `get`/`set!`.
    Bits(BitsLayout),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitsLayout {
    /// Backing integer width in bits.
    pub backing: u32,
    /// Starting bit offset of each field, parallel to `StructDef.fields`.
    pub offsets: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitLayout {
    /// Byte offset of each field, parallel to `StructDef.fields`.
    pub offsets: Vec<u64>,
    /// Fixed total size in bytes (the struct is padded to this); None = computed.
    pub size: Option<u64>,
    /// Whole-struct alignment (0 = natural/1).
    pub align: u32,
}

/// A named struct definition: ordered (field-name, field-type) pairs.
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    /// Generic type parameters (empty for an ordinary struct).
    pub type_params: Vec<String>,
    pub layout: Layout,
    pub fields: Vec<(String, Type)>,
}

/// A match arm: a variant, names bound to its fields, and a body.
#[derive(Debug, Clone)]
pub struct Arm {
    pub variant: String,
    pub binds: Vec<String>,
    pub body: Expr,
}

/// A sum type (tagged union): an ordered list of variants, each with fields.
#[derive(Debug, Clone)]
pub struct SumDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub variants: Vec<SumVariant>,
}

#[derive(Debug, Clone)]
pub struct SumVariant {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

/// A compile-time assertion: `cond` must evaluate (at compile time) to nonzero.
#[derive(Debug, Clone)]
pub struct StaticAssert {
    pub cond: Expr,
    pub msg: String,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub conventions: HashMap<String, Convention>,
    pub structs: Vec<StructDef>,
    pub sums: Vec<SumDef>,
    pub externs: Vec<Extern>,
    pub funcs: Vec<Func>,
    pub asserts: Vec<StaticAssert>,
}

impl Program {
    pub fn func(&self, name: &str) -> Option<&Func> {
        self.funcs.iter().find(|f| f.name == name)
    }
}
