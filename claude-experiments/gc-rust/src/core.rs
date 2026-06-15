//! Core IR: the monomorphic, typed, desugared program that codegen consumes.
//!
//! See `docs/core-ir.md` for the rationale. Everything here is concrete: no
//! type variables, no generics, no trait bounds. Each value has an explicit
//! [`Repr`]; each heap shape has an explicit [`Layout`] that maps 1:1 onto a
//! `gc::TypeInfo`.

use crate::ast::{BinOp, UnOp};

/// How a value is represented at the machine level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Repr {
    /// Zero-size, no storage.
    Unit,
    /// An unboxed primitive scalar.
    Scalar(ScalarRepr),
    /// An inline value aggregate (a `value struct`/`value enum`), stored flat.
    Value(ValueId),
    /// A GC pointer to a heap object whose layout is `LayoutId`.
    Ref(LayoutId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarRepr {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F32, F64,
    Bool, Char,
}

impl ScalarRepr {
    /// Bit width of the scalar.
    pub fn bits(self) -> u32 {
        match self {
            ScalarRepr::I8 | ScalarRepr::U8 => 8,
            ScalarRepr::I16 | ScalarRepr::U16 => 16,
            ScalarRepr::I32 | ScalarRepr::U32 | ScalarRepr::F32 | ScalarRepr::Char => 32,
            ScalarRepr::I64 | ScalarRepr::U64 | ScalarRepr::F64 => 64,
            ScalarRepr::Bool => 1,
        }
    }
    pub fn is_float(self) -> bool {
        matches!(self, ScalarRepr::F32 | ScalarRepr::F64)
    }
    /// Signed integer? (`char`/`bool` are unsigned; floats are neither.)
    pub fn is_signed(self) -> bool {
        matches!(self, ScalarRepr::I8 | ScalarRepr::I16 | ScalarRepr::I32 | ScalarRepr::I64)
    }
    pub fn is_int(self) -> bool {
        !self.is_float() && !matches!(self, ScalarRepr::Bool)
    }
}

pub type ValueId = u32;
pub type LayoutId = u32;
pub type FuncId = u32;
pub type LocalId = u32;

// ============================================================================
// Heap object layout → gc::TypeInfo
// ============================================================================

#[derive(Clone, Debug)]
pub struct Layout {
    /// Number of leading 8-byte GC pointer slots (traced). Must come first.
    pub ptr_fields: u16,
    /// Untraced scalar bytes after the pointer slots.
    pub raw_bytes: u16,
    pub varlen: VarLen,
    /// Source-level field order → location, for field access codegen.
    pub field_map: Vec<FieldLoc>,
    /// Human name for FrameOrigin / debugging / mangling.
    pub name: String,
    /// For array layouts: the element stride in bytes (codegen-only; does not
    /// affect the GC `TypeInfo`). 0 for non-array layouts.
    pub elem_stride: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarLen { None, Values, Bytes }

#[derive(Clone, Copy, Debug)]
pub enum FieldLoc {
    /// A GC pointer slot at pointer-index `idx` (byte offset = header + idx*8).
    Ptr { idx: u16 },
    /// A raw scalar at byte `offset` within the raw section.
    Raw { offset: u16, repr: ScalarRepr },
    /// A field of an inline value aggregate, by its index in the LLVM struct.
    ValueField { index: u32 },
}

// ============================================================================
// Inline value aggregates (unboxed)
// ============================================================================

#[derive(Clone, Debug)]
pub struct ValueLayout {
    pub name: String,
    /// `None` for a value struct; `Some` for a value enum (tagged union).
    pub variants: Option<Vec<ValueVariant>>,
    /// For a value struct: its fields in order.
    pub fields: Vec<Repr>,
    pub size: u32,
    pub align: u32,
}

#[derive(Clone, Debug)]
pub struct ValueVariant {
    pub name: String,
    pub fields: Vec<Repr>,
}

// ============================================================================
// Program / functions
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct CoreProgram {
    pub funcs: Vec<CoreFn>,
    /// Index = `type_id` in the runtime type table.
    pub layouts: Vec<Layout>,
    pub values: Vec<ValueLayout>,
    pub entry: Option<FuncId>,
}

#[derive(Clone, Debug)]
pub struct CoreFn {
    /// Mangled, globally-unique name.
    pub name: String,
    pub params: Vec<Repr>,
    pub ret: Repr,
    /// Total local slots (params first, then `let`/temp locals).
    pub locals: Vec<Repr>,
    pub body: CoreBlock,
    /// If this is a lifted closure function, it takes an extra *leading* `env`
    /// pointer parameter (before `params`), and its first `closure_captures.len()`
    /// locals are initialized by loading the corresponding field from `env`.
    /// Empty for ordinary functions.
    pub closure_captures: Vec<ClosureCapture>,
}

/// How to initialize one capture local from the closure env object.
#[derive(Clone, Copy, Debug)]
pub struct ClosureCapture {
    /// The local slot (in `CoreFn.locals`) this capture initializes.
    pub local: LocalId,
    /// Absolute byte offset of this capture within the env object (past the
    /// `Full` header), where codegen loads it from.
    pub offset: u64,
}

#[derive(Clone, Debug)]
pub struct CoreBlock {
    pub stmts: Vec<CoreStmt>,
    pub tail: Option<CoreExpr>,
}

#[derive(Clone, Debug)]
pub enum CoreStmt {
    Let(LocalId, CoreExpr),
    Expr(CoreExpr),
}

// ============================================================================
// Expressions
// ============================================================================

#[derive(Clone, Debug)]
pub struct CoreExpr {
    pub kind: Box<CoreExprKind>,
    pub repr: Repr,
}

#[derive(Clone, Debug)]
pub enum CoreExprKind {
    ConstInt(u64, ScalarRepr),
    ConstFloat(f64, ScalarRepr),
    ConstBool(bool),
    ConstChar(char),
    /// A string literal → a `Ref` to a `String` heap object.
    ConstStr(String),
    Unit,

    Local(LocalId),

    Bin(BinOp, Box<CoreExpr>, Box<CoreExpr>),
    Un(UnOp, Box<CoreExpr>),
    /// A float intrinsic call (`sqrt`, `abs`, `floor`, …) on a single operand.
    FloatIntrinsic(FloatIntrinsic, Box<CoreExpr>),
    /// A runtime print of a scalar (`print_int`/`print_float`). Yields i64 0.
    Print(Box<CoreExpr>),

    /// Allocate a fresh varlen array of `len` elements of the given layout
    /// (a reference object whose varlen tail holds the elements). Elements are
    /// zero-initialized.
    ArrayNew { layout: LayoutId, len: Box<CoreExpr>, elem: Repr },
    /// Number of elements in an array (its varlen count). Yields i64.
    ArrayLen(Box<CoreExpr>),
    /// Read element `index` from an array. Repr is the element repr.
    ArrayGet { array: Box<CoreExpr>, index: Box<CoreExpr>, elem: Repr },
    /// Write `value` to element `index` of an array. Yields i64 0.
    ArraySet { array: Box<CoreExpr>, index: Box<CoreExpr>, value: Box<CoreExpr>, elem: Repr },
    /// Numeric conversion: trunc / sext / zext / fp<->int / fp resize.
    Cast { value: Box<CoreExpr>, from: Repr, to: Repr },

    /// Direct call to a known monomorphic function.
    Call(FuncId, Vec<CoreExpr>),
    /// Indirect call through a closure value.
    CallClosure { callee: Box<CoreExpr>, args: Vec<CoreExpr> },
    /// Build a closure: env layout + code fn + captures.
    MakeClosure { code: FuncId, env: LayoutId, captures: Vec<CoreExpr> },

    /// Reference-type construction → heap alloc + field stores.
    New { layout: LayoutId, fields: Vec<CoreExpr> },
    /// Inline value-type construction.
    MakeValue { value: ValueId, fields: Vec<CoreExpr> },
    /// Build a reference enum variant on the heap.
    MakeVariant { layout: LayoutId, tag: u32, fields: Vec<CoreExpr> },
    /// Build an inline value-enum variant: `{ i32 tag, payload }`. v0 supports
    /// payload-less variants (C-style enums); the payload bytes are zeroed.
    MakeValueVariant { value: ValueId, tag: u32, fields: Vec<CoreExpr> },
    /// Match on an inline value enum (tag is field 0 of the aggregate).
    ValueMatch { scrutinee: Box<CoreExpr>, arms: Vec<CoreArm> },

    /// Field access by resolved location.
    Field { base: Box<CoreExpr>, loc: FieldLoc },
    /// Store `value` into a reference object's field. Yields i64 0. (Only
    /// reference-struct fields; value structs are immutable SSA aggregates.)
    SetField { base: Box<CoreExpr>, loc: FieldLoc, value: Box<CoreExpr> },

    Match { scrutinee: Box<CoreExpr>, arms: Vec<CoreArm> },
    If(Box<CoreExpr>, Box<CoreBlock>, Box<CoreBlock>),
    Block(Box<CoreBlock>),

    Loop(Box<CoreBlock>),
    Break(Option<Box<CoreExpr>>),
    Continue,
    Return(Option<Box<CoreExpr>>),
    Assign { local: LocalId, value: Box<CoreExpr> },
}

#[derive(Clone, Debug)]
pub struct CoreArm {
    /// Variant tag this arm matches.
    pub tag: u32,
    /// Locals bound to the payload fields (in field order).
    pub binds: Vec<LocalId>,
    pub body: CoreExpr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatIntrinsic { Sqrt, Abs, Floor, Ceil }

impl CoreExpr {
    pub fn new(kind: CoreExprKind, repr: Repr) -> Self {
        CoreExpr { kind: Box::new(kind), repr }
    }
}
