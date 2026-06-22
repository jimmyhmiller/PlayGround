//! Core AST. M0/M1 is deliberately i64-only; the `Type` enum exists so the
//! type checker and codegen are shaped for richer types later.

use std::collections::HashMap;

use crate::convention::Convention;
use crate::span::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// Integer of an arbitrary bit width, signed or unsigned (Zig-style: `u2`,
    /// `i7`, `u23`, ...). LLVM has native `iN`; signedness drives the operations.
    Int(u32, bool), // (bits, signed)
    /// IEEE floating point: `f32` or `f64` (bits = 32 or 64).
    Float(u32),
    /// A boolean (`i1`). The result of comparisons; the condition of `if`.
    Bool,
    /// The unit/`void` return type: a function that yields NO value (an LLVM
    /// `void` return). Only valid as a return type; a call to a `(-> void)`
    /// function produces a void value that CANNOT be used where a value is needed
    /// (the checker hard-errors on use). For void C functions (e.g. `qsort`) and
    /// Coil procedures run for effect.
    Void,
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
    /// A slice: a fat pointer `{ data: (ptr T), len: i64 }` — a *view* into
    /// contiguous elements (it borrows, it does not own). The everyday string
    /// type is `(slice u8)`. This is the only core support slices need: the
    /// fat-pointer type plus the string-literal lowering. Every slice/string
    /// *operation* is a library over `(llvm-ir …)` (lib/slice.coil, lib/str.coil),
    /// exactly as SIMD is a library over `(vec T N)`.
    Slice(Box<Type>),
    /// A fixed-width SIMD vector of `lanes` elements (LLVM `<N x T>`). The
    /// element type must be a scalar (int or float). Distinct from `Array`: it
    /// lowers to an LLVM vector, so `(llvm-ir ...)` and the existing arithmetic
    /// operate on it lane-wise.
    Vec(Box<Type>, u32),
    /// A function pointer: calling convention, parameter types, return type.
    Fn(String, Vec<Type>, Box<Type>),
    /// A generic type application, e.g. `(Pair i64 i64)` -> `App("Pair", [i64,i64])`.
    /// Removed by monomorphization before the checker runs.
    App(String, Vec<Type>),
    /// The bottom type: the type of an expression that never yields a value
    /// (`break`/`continue`/`return-from`). It is *not* user-writable; the checker
    /// produces it, and it unifies with any type (a `Never` branch adopts the
    /// other branch's type) so `(if c (do …T…) (break))` type-checks without a
    /// dummy value. It has no runtime representation (a `Never` value is never
    /// materialized — the expression diverges first).
    Never,
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
    /// Division: `idiv` (signed) / `fdiv` (float), dispatched by operand type for
    /// integers (a `uN` operand divides unsigned).
    Div,
    /// Remainder: `irem` / `frem`, dispatched by operand type like `Div`.
    Rem,
    /// Forced UNSIGNED division (`udiv`) — interprets the operand bits as
    /// unsigned regardless of their type, so an `i64` with the high bit set
    /// divides as the large positive value (for hex/uint printing, hashing, …).
    UDiv,
    /// Forced unsigned remainder (`urem`) — the `UDiv` counterpart.
    URem,
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
    /// A string literal `"…"` — a `(slice u8)` VIEW over a private `[N x i8]`
    /// global (length known at compile time). No allocator, no copy: it borrows
    /// the static bytes. Content/ops live in lib/str.coil over `(slice u8)`.
    Str(String),
    /// A C-string literal `c"…"` — a `(ptr i8)` to a private, NUL-terminated
    /// `[N+1 x i8]` global, for FFI. The DISTINCT FFI spelling (not conflated
    /// with the length-carrying `(slice u8)` string).
    CStr(String),
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
    /// Spill an rvalue to a fresh stack slot and yield a pointer to it. The
    /// checker inserts this when an aggregate (or any) rvalue is passed to a
    /// by-immutable-reference parameter — a temporary needs a place to borrow.
    /// Codegen lowers it to `alloca` + `store` + the slot pointer (the same
    /// spill the match scrutinee uses). Not user-writable.
    SpillRef(Box<Expr>),
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
    /// A boolean literal (`true` / `false`).
    Bool(bool),
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
    /// An unconditional loop: evaluate `body` repeatedly until a `Break` exits
    /// it. The loop's value is the value carried by the `break` that fires
    /// (unified across all break sites); a loop with no reachable `break` is
    /// divergent. `label` (optional) names the loop so a nested `break`/`continue`
    /// can target it. This is the one structured-control-flow PRIMITIVE: `while`,
    /// `for`, `block`/`return-from` are implemented as macros over it (see
    /// `lib/control.coil`); `loop`/`recur` and `defer` are planned macros.
    Loop {
        label: Option<String>,
        body: Vec<Expr>,
    },
    /// Exit the enclosing loop (or the one named `label`), with an optional value
    /// (defaults to the i64 `0`). The loop expression evaluates to this value.
    Break {
        label: Option<String>,
        value: Option<Box<Expr>>,
    },
    /// Jump to the next iteration of the enclosing loop (or the one named
    /// `label`) — i.e. restart its body.
    Continue {
        label: Option<String>,
    },
    /// Raw LLVM IR escape hatch: `(llvm-ir RESULT [operands…] "BODY")`. The body
    /// is LLVM IR text for an inlined helper function: `$ret` / `$tN` expand to
    /// the result / operand LLVM type strings and `$N` to the operand SSA names
    /// (`%0`, `%1`, …). Lines beginning `declare` are hoisted to module scope.
    /// The whole form has the declared `result` type (the checker trusts it; the
    /// LLVM verifier checks the body). This is what makes every LLVM
    /// instruction/intrinsic reachable without per-opcode compiler support.
    LlvmIr {
        result: Type,
        args: Vec<Expr>,
        body: String,
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
    /// Source location of the `(defn …)` form, for debug info (DWARF line). A
    /// function read from an included/imported file (whose offsets are in another
    /// source) carries `Span::DUMMY` — it just gets no line info, never wrong info.
    pub span: Span,
}

/// A foreign function declaration: a name, a calling convention, and a typed
/// signature, with no body. Calls are checked against it; it lowers to an
/// external LLVM declaration the linker resolves (libc, hand-written asm, ...).
#[derive(Debug, Clone)]
pub struct Extern {
    pub name: String,
    pub cc: String,
    pub params: Vec<Type>,
    /// A C variadic function (`...`): `params` are the fixed prefix; calls may
    /// pass additional trailing arguments.
    pub variadic: bool,
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

/// A named compile-time constant — `(const NAME VALUE)` or `(const NAME TYPE
/// VALUE)`. A reference to it elaborates to the literal `value` inline (zero
/// runtime overhead): so an *untyped* const behaves exactly like writing the
/// literal at the use site, re-entering integer-width inference; an explicit
/// `ty` pins the reported type and fit-checks the value at definition. Consts
/// live in a flat global namespace (referenced bare, never module-renamed — the
/// same rule `extern` uses), matching C enum constants and `#define`s.
#[derive(Debug, Clone)]
pub struct Const {
    pub name: String,
    pub ty: Option<Type>,
    pub value: ConstLit,
}

/// The literal value of a `const`. Restricted to scalars that have both a
/// compile-time and a runtime representation (the overlap between the two
/// value universes — see the macro/runtime split).
#[derive(Debug, Clone, Copy)]
pub enum ConstLit {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub conventions: HashMap<String, Convention>,
    pub structs: Vec<StructDef>,
    pub sums: Vec<SumDef>,
    pub externs: Vec<Extern>,
    pub funcs: Vec<Func>,
    pub asserts: Vec<StaticAssert>,
    pub consts: Vec<Const>,
}

impl Program {
    pub fn func(&self, name: &str) -> Option<&Func> {
        self.funcs.iter().find(|f| f.name == name)
    }
}
