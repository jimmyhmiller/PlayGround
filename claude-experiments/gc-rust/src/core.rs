//! Core IR: the monomorphic, typed, desugared program that codegen consumes.
//!
//! See `docs/core-ir.md` for the rationale. Everything here is concrete: no
//! type variables, no generics, no trait bounds. Each value has an explicit
//! [`Repr`]; each heap shape has an explicit [`Layout`] that maps 1:1 onto a
//! `gc::TypeInfo`.

use crate::ast::{BinOp, UnOp};
use crate::gc::TypeMeta;
use serde::Serialize;

/// How a value is represented at the machine level.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ScalarRepr {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F32, F64,
    Bool, Char,
    /// An opaque pointer-sized non-GC value (FFI `RawPtr`). Lowers to LLVM
    /// `ptr`. Never participates in arithmetic — only crosses the FFI boundary.
    Ptr,
}

impl ScalarRepr {
    /// Bit width of the scalar.
    pub fn bits(self) -> u32 {
        match self {
            ScalarRepr::I8 | ScalarRepr::U8 => 8,
            ScalarRepr::I16 | ScalarRepr::U16 => 16,
            ScalarRepr::I32 | ScalarRepr::U32 | ScalarRepr::F32 | ScalarRepr::Char => 32,
            ScalarRepr::I64 | ScalarRepr::U64 | ScalarRepr::F64 | ScalarRepr::Ptr => 64,
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
        !self.is_float() && !matches!(self, ScalarRepr::Bool | ScalarRepr::Ptr)
    }
}

pub type ValueId = u32;
pub type LayoutId = u32;
pub type FuncId = u32;
pub type LocalId = u32;

// ============================================================================
// Heap object layout → gc::TypeInfo
// ============================================================================

#[derive(Clone, Debug, Serialize)]
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
    /// Absolute byte offsets (header included) of GC pointers embedded in the raw
    /// region — i.e. references inside flattened `#[value]` fields. These become
    /// `gc::TypeInfo::interior_ptrs` so the collector traces them. Empty for the
    /// common case (no value-with-ref fields).
    pub interior_ptrs: Vec<u16>,
    /// Runtime reflection metadata (nominal type/field names + field types) for
    /// this layout. Travels 1:1 with the layout so the JIT/AOT paths and the
    /// `emit reflect` view all read the same source of truth. Skipped in the
    /// `core` JSON dump (it has a dedicated `reflect` view) and not consumed by
    /// the GC, which only needs the structural fields above.
    #[serde(skip)]
    pub meta: TypeMeta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VarLen { None, Values, Bytes }

#[derive(Clone, Copy, Debug, Serialize)]
pub enum FieldLoc {
    /// A GC pointer slot at pointer-index `idx` (byte offset = header + idx*8).
    Ptr { idx: u16 },
    /// A raw scalar at byte `offset` within the raw section.
    Raw { offset: u16, repr: ScalarRepr },
    /// A flattened `#[value]` aggregate stored inline at byte `offset` within the
    /// raw section. `value` is its [`ValueId`]; codegen stores/loads it as the
    /// whole LLVM aggregate (not a scalar). Distinct from `ValueField`, which
    /// addresses a sub-field of an already-loaded value aggregate.
    ValueAt { offset: u16, value: ValueId },
    /// A field of an inline value aggregate, by its index in the LLVM struct.
    ValueField { index: u32 },
}

// ============================================================================
// Inline value aggregates (unboxed)
// ============================================================================

#[derive(Clone, Debug, Serialize)]
pub struct ValueLayout {
    pub name: String,
    /// `None` for a value struct; `Some` for a value enum (tagged union).
    pub variants: Option<Vec<ValueVariant>>,
    /// For a value struct: its fields in order.
    pub fields: Vec<Repr>,
    /// Source field names for a value struct (`"0"`,`"1"`,… for tuples), for
    /// reflection. Empty for value enums and tuples-without-names.
    pub field_names: Vec<String>,
    pub size: u32,
    pub align: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ValueVariant {
    pub name: String,
    pub fields: Vec<Repr>,
}

/// The number of leading GC-pointer slots a flattened value enum reserves —
/// the max over its variants of their `Ref` payload count. Pointer payloads of
/// every variant share these leading slots (at fixed offsets `0, 8, …`), exactly
/// like a reference enum's heap layout, so an embedded ref's offset does not
/// depend on the runtime tag. The single source of truth for layout, codegen,
/// and GC interior-pointer offsets. `0` means no ref payloads (the value enum
/// keeps its compact `{ tag, raw }` form).
pub fn value_enum_max_ptrs(variants: &[ValueVariant]) -> u16 {
    variants
        .iter()
        .map(|v| v.fields.iter().filter(|f| matches!(f, Repr::Ref(_))).count() as u16)
        .max()
        .unwrap_or(0)
}

// ============================================================================
// Program / functions
// ============================================================================

#[derive(Clone, Debug, Default, Serialize)]
pub struct CoreProgram {
    pub funcs: Vec<CoreFn>,
    /// Index = `type_id` in the runtime type table.
    pub layouts: Vec<Layout>,
    pub values: Vec<ValueLayout>,
    pub entry: Option<FuncId>,
}

#[derive(Clone, Debug, Serialize)]
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
    /// `true` for a foreign `extern "C"` function. Such a function has no body
    /// (`body` is empty and must not be defined), takes no leading `Thread*`,
    /// and `name` is the unmangled C symbol resolved at link/JIT time. Calls to
    /// it omit the `Thread*` argument. See `docs/ffi.md`.
    pub is_extern: bool,
}

/// How to initialize one capture local from the closure env object.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ClosureCapture {
    /// The local slot (in `CoreFn.locals`) this capture initializes.
    pub local: LocalId,
    /// Absolute byte offset of this capture within the env object (past the
    /// `Full` header), where codegen loads it from.
    pub offset: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CoreBlock {
    pub stmts: Vec<CoreStmt>,
    pub tail: Option<CoreExpr>,
}

#[derive(Clone, Debug, Serialize)]
pub enum CoreStmt {
    Let(LocalId, CoreExpr),
    Expr(CoreExpr),
}

// ============================================================================
// Expressions
// ============================================================================

#[derive(Clone, Debug, Serialize)]
pub struct CoreExpr {
    pub kind: Box<CoreExprKind>,
    pub repr: Repr,
}

#[derive(Clone, Debug, Serialize)]
pub enum CoreExprKind {
    ConstInt(u64, ScalarRepr),
    ConstFloat(f64, ScalarRepr),
    ConstBool(bool),
    ConstChar(char),
    /// The zero/null value of a repr (scalar 0, ref null). Used to define a
    /// deferred-initialization `let mut x: T;` slot before its first assignment.
    ConstZero(Repr),
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

    /// String primitives. Operands are `Ref`s to `String` heap objects.
    /// `print_str(s)` → i64 0.
    PrintStr(Box<CoreExpr>),
    /// `print(s)` → i64 0. Like `PrintStr` but NO trailing newline; flushes.
    PrintStrRaw(Box<CoreExpr>),
    /// `str_len(s)` → i64 byte length.
    StrLen(Box<CoreExpr>),
    /// `str_eq(a, b)` → bool.
    StrEq(Box<CoreExpr>, Box<CoreExpr>),
    /// `str_concat(a, b)` → a fresh `String`. `layout` is the String layout id
    /// (passed to the allocating runtime call).
    StrConcat { layout: LayoutId, a: Box<CoreExpr>, b: Box<CoreExpr> },
    /// `str_get(s, i)` → i64 byte (or -1 out of range).
    StrGet(Box<CoreExpr>, Box<CoreExpr>),
    /// `str_substring(s, start, end)` → a fresh `String`.
    StrSubstring { layout: LayoutId, s: Box<CoreExpr>, start: Box<CoreExpr>, end: Box<CoreExpr> },
    /// `to_string(v)` → a fresh `String` (int or float rendering). `is_float`
    /// selects the runtime entry point.
    StrFromNum { layout: LayoutId, is_float: bool, v: Box<CoreExpr> },
    /// `char_to_str(cp)` → a fresh 1–4 byte `String`, the UTF-8 encoding of the
    /// Unicode scalar `cp` (invalid scalars become U+FFFD).
    StrFromChar { layout: LayoutId, cp: Box<CoreExpr> },
    /// `read_file(path)` → a fresh `String` with the file's bytes (empty if the
    /// file can't be read). Lets a self-hosted compiler driver read source files.
    ReadFile { layout: LayoutId, path: Box<CoreExpr> },
    /// `str_to_float(s)` → f64 (0.0 on malformed input).
    StrToFloat(Box<CoreExpr>),
    /// `float_bits(f)` → i64 reinterpreting the f64's bit pattern (bitcast). Lets a
    /// self-hosted compiler get a float literal's raw bits for uniform i64 storage.
    FloatBits(Box<CoreExpr>),
    /// `str_hash(s)` → i64 non-negative hash.
    StrHash(Box<CoreExpr>),
    /// `type_id_of(x)` → i64: the runtime `type_id` from a heap object's header.
    /// `x` must be a `Ref` (heap) value. Reflection primitive.
    TypeIdOf(Box<CoreExpr>),
    /// `type_name_of(x)` → String: the source type name from the reflection
    /// metadata table, allocated as a heap String (`layout` is the String
    /// layout). `obj` must be a `Ref` value. Reflection primitive.
    TypeNameOf { layout: LayoutId, obj: Box<CoreExpr> },
    /// FFI `as_c_bytes(x)` → a `RawPtr` (`Scalar(Ptr)`) to a stack-resident copy
    /// of a `String`'s or scalar `Array`'s contents. Codegen copies the heap
    /// bytes into a fresh stack alloca for the duration of the enclosing extern
    /// call (the only place this node may appear). Because the copy lives on the
    /// non-moving native stack, the pointer is stable across the call with no
    /// pinning. See `docs/ffi.md`.
    ///
    /// * `src` is the `Ref` to the String/Array object.
    /// * `elem` is the element scalar (`U8` for a `String`'s bytes) — its byte
    ///   size is the stride; total bytes = `array_len * stride`.
    /// * `is_string` adds a trailing NUL (C-string convention) and sizes the
    ///   buffer from the String's byte count.
    /// * `copy_out` (a `mut` array argument) means: after the enclosing extern
    ///   call returns, copy the stack buffer back into the heap object, so writes
    ///   the C function made are visible to the caller (e.g. `read(fd, buf, n)`).
    AsCBytes { src: Box<CoreExpr>, elem: ScalarRepr, is_string: bool, copy_out: bool },

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
    /// `thread_spawn(closure)` → spawn an OS thread running the no-arg closure
    /// `closure` (a `Ref` to a closure env). Yields a thread handle as a `RawPtr`
    /// (opaque, non-GC — it wraps a runtime-owned join handle). Codegen extracts
    /// the env + code pointer and calls `ai_thread_spawn`. See `docs/threads.md`.
    ThreadSpawn(Box<CoreExpr>),
    /// `thread_join(handle)` → block until the thread finishes; yields its i64
    /// result. Consumes the `RawPtr` handle.
    ThreadJoin(Box<CoreExpr>),
    /// Clojure-style atom (single mutable cell holding an immutable value).
    /// `AtomLoad` atomically loads the atom's value field (a GC pointer slot).
    /// The atom is an ordinary heap object `{ value: T }`; the field is at
    /// `HEADER + 0`. See `docs/threads.md`.
    AtomLoad { atom: Box<CoreExpr>, elem: Repr },
    /// `AtomCas` atomically compare-and-swaps the atom's value field: if it
    /// currently equals `old`, install `new` (with write barrier) and yield
    /// true, else yield false. The retry loop of `swap!` is written in-language
    /// over this (so old/new/atom are ordinary frame roots — GC-safe for free).
    AtomCas { atom: Box<CoreExpr>, old: Box<CoreExpr>, new: Box<CoreExpr> },
    /// Channel `send`: store `value` (a GC pointer) into the channel's on-heap
    /// `buf` at the control block `ctrl`'s tail slot, blocking while full. The
    /// queued value is traced by the GC via `buf`. Yields i64 0.
    ChanSend { buf: Box<CoreExpr>, ctrl: Box<CoreExpr>, value: Box<CoreExpr> },
    /// Channel `recv`: pop the head element of `buf` (blocking while empty);
    /// yields the element as `elem` (a GC pointer, or null when closed+drained —
    /// the prelude wraps the result as `Option<T>`).
    ChanRecv { buf: Box<CoreExpr>, ctrl: Box<CoreExpr>, elem: Repr },
    /// A direct call to a named runtime extern with already-lowered scalar/ptr
    /// args, threading the `Thread*` as the leading argument. Used for the
    /// shared-memory primitives (atomics) and similar runtime intrinsics that
    /// don't warrant a bespoke node. `ret` is the result repr.
    RuntimeCall { func: &'static str, args: Vec<CoreExpr>, ret: Repr },
    /// `thread_sleep(ms)` → sleep this thread; yields i64 0.
    ThreadSleep(Box<CoreExpr>),
    /// `thread_yield()` → yield this thread's timeslice; yields i64 0.
    ThreadYield,
    /// `thread_current_id()` → a stable per-thread id (i64).
    ThreadCurrentId,
    /// FFI `ptr_read_i64(p)` → the i64 at `RawPtr` `p` (a plain load). Lets a
    /// callback read through the C pointers it is handed (e.g. a `qsort`
    /// comparator). See `docs/ffi.md`.
    PtrReadI64(Box<CoreExpr>),
    /// FFI callback: a `RawPtr` (`Scalar(Ptr)`) to a C-ABI trampoline that
    /// invokes gc-rust function `FuncId`. Codegen synthesizes one trampoline per
    /// referenced function (re-entering managed state, calling the function with
    /// the ambient `Thread*`, restoring native state) and yields its address.
    /// Used when a named gc-rust function is passed where an `extern fn(..)` C
    /// callback is expected. See `docs/ffi.md`.
    CallbackPtr(FuncId),
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
    /// Read the discriminant tag (i32) of a reference enum object. Used by the
    /// if-chain match lowering (guards / literals) where there's no tag-switch.
    EnumTag(Box<CoreExpr>),
    /// Read payload field `field` of a reference enum object, as `repr`. The
    /// caller knows the variant matches (a preceding `EnumTag` test guards it).
    /// `payload_reprs` is the full ordered payload-field repr list (needed to
    /// compute the physical slot of `field`, since ptr and raw payloads are laid
    /// out in separate regions just like the tag-switch binding path).
    EnumPayload { scrutinee: Box<CoreExpr>, field: u32, repr: Repr, payload_reprs: Vec<Repr> },
    If(Box<CoreExpr>, Box<CoreBlock>, Box<CoreBlock>),
    Block(Box<CoreBlock>),

    Loop(Box<CoreBlock>),
    Break(Option<Box<CoreExpr>>),
    Continue,
    Return(Option<Box<CoreExpr>>),
    Assign { local: LocalId, value: Box<CoreExpr> },
}

#[derive(Clone, Debug, Serialize)]
pub struct CoreArm {
    /// Variant tag this arm matches.
    pub tag: u32,
    /// Locals bound to the payload fields (in field order).
    pub binds: Vec<LocalId>,
    pub body: CoreExpr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FloatIntrinsic { Sqrt, Abs, Floor, Ceil }

impl CoreExpr {
    pub fn new(kind: CoreExprKind, repr: Repr) -> Self {
        CoreExpr { kind: Box::new(kind), repr }
    }
}
