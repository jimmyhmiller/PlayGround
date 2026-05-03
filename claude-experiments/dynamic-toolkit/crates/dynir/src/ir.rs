use crate::types::{Signature, Type};

/// SSA value reference. Globally unique within a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub(crate) u32);

impl Value {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Basic block reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub(crate) u32);

impl BlockId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Reference to a function-level stack slot.
///
/// Stack slots are declared at function scope (not in any block) via
/// `FunctionBuilder::create_stack_slot`. Use `Inst::StackAddr` to get
/// a `Ptr` to the slot's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StackSlot(pub(crate) u32);

impl StackSlot {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Metadata for a function-level stack slot.
#[derive(Debug, Clone)]
pub struct StackSlotData {
    /// Size in bytes.
    pub size: u32,
    /// Whether this slot may hold GC pointers and must be scanned during collection.
    pub is_gc_root: bool,
}

/// Reference to a prompt boundary within a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptId(pub(crate) u32);

impl PromptId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn index_u32(self) -> u32 {
        self.0
    }

    pub fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Reference to an externally-declared function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncRef(pub(crate) u32);

impl FuncRef {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Construct a `FuncRef` from a raw index. Caller is responsible for
    /// ensuring the index points at a valid module slot.
    pub fn from_u32(idx: u32) -> Self {
        FuncRef(idx)
    }
}

/// Reference to an entry in a JitModule's literal pool.
///
/// The pool is a pointer-stable, GC-traced array of `u64` (NanBox-encoded)
/// slots. Quote-style literals whose payload is a heap pointer go through
/// the pool so a moving collector can rewrite slots in place — emitted code
/// reads the current slot value on each access.
///
/// Numbers, nil/true/false, symbols, and other immortal/non-pointer values
/// can stay as inline `Iconst`s; only literals that point at the GC heap
/// need the indirection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiteralRef(pub u32);

impl LiteralRef {
    pub fn index(self) -> usize { self.0 as usize }
    pub fn as_u32(self) -> u32 { self.0 }
    pub fn from_u32(idx: u32) -> Self { LiteralRef(idx) }
}

/// Reference to deoptimization metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeoptId(pub(crate) u32);

impl DeoptId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Deoptimization metadata: describes how to resume in the interpreter.
#[derive(Debug, Clone)]
pub struct DeoptInfo {
    /// Opaque identifier (e.g. bytecode offset) for the interpreter to resume at.
    pub resume_point: u64,
    /// Human-readable description (optional, for debugging).
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

/// Overflow-checked arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverflowOp {
    SAdd,
    SSub,
    SMul,
    UAdd,
    USub,
    UMul,
}

/// An IR instruction. Most produce a single result value.
#[derive(Debug, Clone)]
pub enum Inst {
    // -- Constants --
    Iconst(Type, i64),
    F64Const(f64),
    /// Load a NanBox-encoded GC-managed literal from the JitModule's
    /// literal pool. Result type is `I64`. Lowered to a base+index load
    /// against a pointer-stable pool whose slots the GC traces and may
    /// rewrite during collection.
    GcLiteral(LiteralRef),

    // -- Integer arithmetic (result type = operand type) --
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    SDiv(Value, Value),
    UDiv(Value, Value),

    // -- Float arithmetic (result type = F64) --
    FAdd(Value, Value),
    FSub(Value, Value),
    FMul(Value, Value),
    FDiv(Value, Value),

    // -- Bitwise (result type = operand type) --
    And(Value, Value),
    Or(Value, Value),
    Xor(Value, Value),
    Shl(Value, Value),
    LShr(Value, Value),
    AShr(Value, Value),

    // -- Unary --
    Neg(Value),  // integer negate (0 - x), result type = operand type
    FNeg(Value), // float negate, result type = F64
    Not(Value),  // bitwise not (xor with -1), result type = operand type

    // -- Comparison (result type = I8) --
    Icmp(CmpOp, Value, Value),
    Fcmp(CmpOp, Value, Value),

    // -- Conversions --
    Sext(Value, Type),
    Zext(Value, Type),
    Trunc(Value, Type),
    IntToFloat(Value),
    FloatToInt(Value),
    Bitcast(Value, Type),

    // -- Memory --
    /// Get the address of a function-level stack slot.
    /// Returns a Ptr. The slot must have been declared via `create_stack_slot`.
    StackAddr(StackSlot),
    Load(Type, Value, i32),
    Store(Value, Value, i32), // (value, addr, offset) — no result

    // -- Tagged values --
    /// Extract the tag from a tagged value. The tagging scheme is determined
    /// by the interpreter/codegen's `TagScheme` type parameter.
    TagOf(Value), // -> I32
    /// Extract the payload from a tagged value.
    Payload(Value), // -> I64
    /// Construct a tagged value from a tag constant and payload.
    MakeTagged(u32, Value), // (tag, payload) -> I64
    /// Check whether a value has a specific tag.
    IsTag(Value, u32), // -> I8

    // -- Select --
    Select(Value, Value, Value), // (cond, if_true, if_false)

    // -- Overflow checking (result type = I8) --
    OverflowCheck(OverflowOp, Value, Value),

    // -- Guard / deoptimization --
    /// Guard that a condition is true; if false, deoptimize.
    /// The `live` values are captured for the deopt frame.
    /// This is a void instruction (side-effect only).
    Guard(Value, DeoptId, Vec<Value>),

    // -- Delimited frame slices --
    /// Install a prompt boundary in the current dynamic extent.
    /// `handler` is the block to jump to when `abort_to_prompt` targets this
    /// prompt. The handler block must take exactly one parameter — the abort
    /// value. This is the clean merge point for normal-flow and abort-flow.
    PushPrompt(PromptId, BlockId),
    /// Pop a prompt boundary when leaving its extent. Void instruction.
    PopPrompt(PromptId),
    /// Capture the current delimited suffix of frames up to `prompt`.
    /// `live` values are explicitly reified into the captured slice.
    CaptureSlice(PromptId, Vec<Value>),
    /// Clone a captured frame slice to enable multi-shot resume.
    CloneSlice(Value),

    // -- Calls --
    Call(FuncRef, Vec<Value>),
    CallIndirect(Value, Vec<Value>, Option<Type>),

    // -- Dynamic dispatch --
    /// Invoke a method/field on a receiver using inline-cached dispatch.
    ///
    /// `receiver`: the object to dispatch on
    /// `symbol`: the method/field name (compile-time Symbol)
    /// `args`: arguments to pass (NOT including receiver — added automatically)
    /// `cache_id`: index into the module's InlineCacheArray
    ///
    /// The lowerer emits a fast path that checks the receiver's class against
    /// the cached class_id. On hit, it calls the cached target directly.
    /// On miss, it calls a slow lookup and updates the cache.
    ///
    /// Result type is I64 (NaN-boxed return value).
    InvokeDynamic {
        receiver: Value,
        symbol: dynsym::Symbol,
        args: Vec<Value>,
        cache_id: u32,
    },

    // -- GC safepoint --
    /// Explicit safepoint (e.g. at loop backedges). The lowering must emit
    /// a stack map here so the GC can trace and update live GcPtr values.
    /// The `live` values are the GcPtr values that the compiler has determined
    /// are live across this point. This is a void instruction.
    Safepoint(Vec<Value>),
}

impl Inst {
    /// Compute the result type of this instruction, or `None` for void (Store).
    pub fn result_type(
        &self,
        val_ty: impl Fn(Value) -> Type,
        extern_sigs: &[ExternFunc],
    ) -> Option<Type> {
        match self {
            Inst::Iconst(ty, _) => Some(*ty),
            Inst::F64Const(_) => Some(Type::F64),
            Inst::GcLiteral(_) => Some(Type::I64),

            Inst::Add(a, b)
            | Inst::Sub(a, b)
            | Inst::Mul(a, b)
            | Inst::SDiv(a, b)
            | Inst::UDiv(a, b) => {
                // Pointer arithmetic: if either operand is a pointer, result is that pointer type
                let ta = val_ty(*a);
                let tb = val_ty(*b);
                if ta == Type::GcPtr || tb == Type::GcPtr {
                    Some(Type::GcPtr)
                } else if ta == Type::Ptr || tb == Type::Ptr {
                    Some(Type::Ptr)
                } else {
                    Some(ta)
                }
            }

            Inst::FAdd(_, _) | Inst::FSub(_, _) | Inst::FMul(_, _) | Inst::FDiv(_, _) => {
                Some(Type::F64)
            }

            Inst::And(a, _)
            | Inst::Or(a, _)
            | Inst::Xor(a, _)
            | Inst::Shl(a, _)
            | Inst::LShr(a, _)
            | Inst::AShr(a, _) => Some(val_ty(*a)),

            Inst::Neg(v) | Inst::Not(v) => Some(val_ty(*v)),
            Inst::FNeg(_) => Some(Type::F64),

            Inst::Icmp(_, _, _) | Inst::Fcmp(_, _, _) => Some(Type::I8),

            Inst::Sext(_, to) | Inst::Zext(_, to) | Inst::Trunc(_, to) | Inst::Bitcast(_, to) => {
                Some(*to)
            }
            Inst::IntToFloat(_) => Some(Type::F64),
            Inst::FloatToInt(_) => Some(Type::I64),

            Inst::StackAddr(_) => Some(Type::Ptr),
            Inst::Load(ty, _, _) => Some(*ty),
            Inst::Store(_, _, _) => None,

            Inst::TagOf(_) => Some(Type::I32),
            Inst::Payload(_) => Some(Type::I64),
            Inst::MakeTagged(_, _) => Some(Type::I64),
            Inst::IsTag(_, _) => Some(Type::I8),

            Inst::Select(_, t, _) => Some(val_ty(*t)),

            Inst::OverflowCheck(_, _, _) => Some(Type::I8),
            Inst::Guard(_, _, _) => None,
            Inst::PushPrompt(_, _) | Inst::PopPrompt(_) => None,
            Inst::CaptureSlice(_, _) | Inst::CloneSlice(_) => Some(Type::FrameSlice),

            Inst::Call(fref, _) => extern_sigs[fref.index()].sig.ret,
            Inst::CallIndirect(_, _, ret_ty) => *ret_ty,
            Inst::InvokeDynamic { .. } => Some(Type::I64),
            Inst::Safepoint(_) => None,
        }
    }

    /// Call `f` on every Value operand.
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            Inst::Iconst(_, _) | Inst::F64Const(_) | Inst::GcLiteral(_) | Inst::StackAddr(_) => {}

            Inst::Add(a, b)
            | Inst::Sub(a, b)
            | Inst::Mul(a, b)
            | Inst::SDiv(a, b)
            | Inst::UDiv(a, b)
            | Inst::FAdd(a, b)
            | Inst::FSub(a, b)
            | Inst::FMul(a, b)
            | Inst::FDiv(a, b)
            | Inst::And(a, b)
            | Inst::Or(a, b)
            | Inst::Xor(a, b)
            | Inst::Shl(a, b)
            | Inst::LShr(a, b)
            | Inst::AShr(a, b) => {
                f(*a);
                f(*b);
            }

            Inst::Neg(v) | Inst::FNeg(v) | Inst::Not(v) => f(*v),

            Inst::Icmp(_, a, b) | Inst::Fcmp(_, a, b) => {
                f(*a);
                f(*b);
            }

            Inst::Sext(v, _)
            | Inst::Zext(v, _)
            | Inst::Trunc(v, _)
            | Inst::IntToFloat(v)
            | Inst::FloatToInt(v)
            | Inst::Bitcast(v, _) => f(*v),

            Inst::Load(_, addr, _) => f(*addr),
            Inst::Store(val, addr, _) => {
                f(*val);
                f(*addr);
            }

            Inst::TagOf(v) | Inst::Payload(v) | Inst::MakeTagged(_, v) | Inst::IsTag(v, _) => f(*v),

            Inst::Select(c, t, e) => {
                f(*c);
                f(*t);
                f(*e);
            }

            Inst::OverflowCheck(_, a, b) => {
                f(*a);
                f(*b);
            }

            Inst::Guard(cond, _, live) => {
                f(*cond);
                live.iter().for_each(|v| f(*v));
            }

            Inst::PushPrompt(_, _) | Inst::PopPrompt(_) => {}
            Inst::CaptureSlice(_, live) => live.iter().for_each(|v| f(*v)),
            Inst::CloneSlice(v) => f(*v),

            Inst::Call(_, args) => args.iter().for_each(|v| f(*v)),
            Inst::CallIndirect(callee, args, _) => {
                f(*callee);
                args.iter().for_each(|v| f(*v));
            }
            Inst::InvokeDynamic { receiver, args, .. } => {
                f(*receiver);
                args.iter().for_each(|v| f(*v));
            }
            Inst::Safepoint(live) => live.iter().for_each(|v| f(*v)),
        }
    }

    /// Call `f` on every Value operand (mutable).
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            Inst::Iconst(_, _) | Inst::F64Const(_) | Inst::GcLiteral(_) | Inst::StackAddr(_) => {}

            Inst::Add(a, b)
            | Inst::Sub(a, b)
            | Inst::Mul(a, b)
            | Inst::SDiv(a, b)
            | Inst::UDiv(a, b)
            | Inst::FAdd(a, b)
            | Inst::FSub(a, b)
            | Inst::FMul(a, b)
            | Inst::FDiv(a, b)
            | Inst::And(a, b)
            | Inst::Or(a, b)
            | Inst::Xor(a, b)
            | Inst::Shl(a, b)
            | Inst::LShr(a, b)
            | Inst::AShr(a, b) => {
                f(a);
                f(b);
            }

            Inst::Neg(v) | Inst::FNeg(v) | Inst::Not(v) => f(v),

            Inst::Icmp(_, a, b) | Inst::Fcmp(_, a, b) => {
                f(a);
                f(b);
            }

            Inst::Sext(v, _)
            | Inst::Zext(v, _)
            | Inst::Trunc(v, _)
            | Inst::IntToFloat(v)
            | Inst::FloatToInt(v)
            | Inst::Bitcast(v, _) => f(v),

            Inst::Load(_, addr, _) => f(addr),
            Inst::Store(val, addr, _) => {
                f(val);
                f(addr);
            }

            Inst::TagOf(v) | Inst::Payload(v) | Inst::MakeTagged(_, v) | Inst::IsTag(v, _) => f(v),

            Inst::Select(c, t, e) => {
                f(c);
                f(t);
                f(e);
            }

            Inst::OverflowCheck(_, a, b) => {
                f(a);
                f(b);
            }

            Inst::Guard(cond, _, live) => {
                f(cond);
                live.iter_mut().for_each(|v| f(v));
            }

            Inst::PushPrompt(_, _) | Inst::PopPrompt(_) => {}
            Inst::CaptureSlice(_, live) => live.iter_mut().for_each(|v| f(v)),
            Inst::CloneSlice(v) => f(v),

            Inst::Call(_, args) => args.iter_mut().for_each(|v| f(v)),
            Inst::CallIndirect(callee, args, _) => {
                f(callee);
                args.iter_mut().for_each(|v| f(v));
            }
            Inst::InvokeDynamic { receiver, args, .. } => {
                f(receiver);
                args.iter_mut().for_each(|v| f(v));
            }
            Inst::Safepoint(live) => live.iter_mut().for_each(|v| f(v)),
        }
    }
}

/// Block terminator — every block ends with exactly one.
#[derive(Debug, Clone)]
pub enum Terminator {
    Ret(Value),
    RetVoid,
    Jump(BlockId, Vec<Value>),
    BrIf {
        cond: Value,
        then_block: BlockId,
        then_args: Vec<Value>,
        else_block: BlockId,
        else_args: Vec<Value>,
    },
    /// Multi-way branch on an integer value.
    /// Each case maps a constant to a (block, args) pair.
    /// Falls through to `default_block` if no case matches.
    Switch {
        val: Value,
        cases: Vec<(i64, BlockId, Vec<Value>)>,
        default_block: BlockId,
        default_args: Vec<Value>,
    },
    /// Call that may throw: normal return jumps to `normal`, exception jumps to `exception`.
    /// The normal block's first param receives the return value (if any).
    Invoke {
        func: FuncRef,
        args: Vec<Value>,
        normal: BlockId,
        normal_args: Vec<Value>,
        exception: BlockId,
        exception_args: Vec<Value>,
    },
    /// Indirect call that may throw.
    InvokeIndirect {
        callee: Value,
        args: Vec<Value>,
        ret_ty: Option<Type>,
        normal: BlockId,
        normal_args: Vec<Value>,
        exception: BlockId,
        exception_args: Vec<Value>,
    },
    /// Resume a previously captured frame slice. Splices the captured frames
    /// on top of the current stack; when the captured computation eventually
    /// completes (via its reset returning, or abort-to-prompt + handler Ret),
    /// control lands at `return_block` with the produced value supplied as
    /// `return_block`'s first block param.
    ResumeSlice {
        slice: Value,
        args: Vec<Value>,
        return_block: BlockId,
        return_args: Vec<Value>,
    },
    /// Capture the current continuation up to `prompt`, then take one of two
    /// paths. At capture time, control transfers to `handler_block` with the
    /// freshly produced continuation handle as its first block param
    /// (FrameSlice-typed). On later resume, the captured frame's PC is
    /// restored to the entry of `resume_block` with the value passed to
    /// `resume` delivered as `resume_block`'s first block param (I64-typed).
    /// This is the Racket-style `shift k in body` form: the handler binds
    /// the continuation as a separate name from the resume value, so the
    /// two never collide in the type system.
    CaptureSlice {
        prompt: PromptId,
        handler_block: BlockId,
        resume_block: BlockId,
    },
    /// Abort directly to the nearest matching prompt.
    AbortToPrompt {
        prompt: PromptId,
        args: Vec<Value>,
    },
    Unreachable,
}

impl Terminator {
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            Terminator::Ret(v) => f(*v),
            Terminator::RetVoid | Terminator::Unreachable => {}
            Terminator::CaptureSlice { .. } => {}
            Terminator::AbortToPrompt { args, .. } => args.iter().for_each(|v| f(*v)),
            Terminator::ResumeSlice { slice, args, return_args, .. } => {
                f(*slice);
                args.iter().for_each(|v| f(*v));
                return_args.iter().for_each(|v| f(*v));
            }
            Terminator::Jump(_, args) => args.iter().for_each(|v| f(*v)),
            Terminator::BrIf {
                cond,
                then_args,
                else_args,
                ..
            } => {
                f(*cond);
                then_args.iter().for_each(|v| f(*v));
                else_args.iter().for_each(|v| f(*v));
            }
            Terminator::Switch {
                val,
                cases,
                default_args,
                ..
            } => {
                f(*val);
                for (_, _, args) in cases {
                    args.iter().for_each(|v| f(*v));
                }
                default_args.iter().for_each(|v| f(*v));
            }
            Terminator::Invoke {
                args,
                normal_args,
                exception_args,
                ..
            } => {
                args.iter().for_each(|v| f(*v));
                normal_args.iter().for_each(|v| f(*v));
                exception_args.iter().for_each(|v| f(*v));
            }
            Terminator::InvokeIndirect {
                callee,
                args,
                normal_args,
                exception_args,
                ..
            } => {
                f(*callee);
                args.iter().for_each(|v| f(*v));
                normal_args.iter().for_each(|v| f(*v));
                exception_args.iter().for_each(|v| f(*v));
            }
        }
    }

    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            Terminator::Ret(v) => f(v),
            Terminator::RetVoid | Terminator::Unreachable => {}
            Terminator::CaptureSlice { .. } => {}
            Terminator::AbortToPrompt { args, .. } => args.iter_mut().for_each(|v| f(v)),
            Terminator::ResumeSlice { slice, args, return_args, .. } => {
                f(slice);
                args.iter_mut().for_each(|v| f(v));
                return_args.iter_mut().for_each(|v| f(v));
            }
            Terminator::Jump(_, args) => args.iter_mut().for_each(|v| f(v)),
            Terminator::BrIf {
                cond,
                then_args,
                else_args,
                ..
            } => {
                f(cond);
                then_args.iter_mut().for_each(|v| f(v));
                else_args.iter_mut().for_each(|v| f(v));
            }
            Terminator::Switch {
                val,
                cases,
                default_args,
                ..
            } => {
                f(val);
                for (_, _, args) in cases {
                    args.iter_mut().for_each(|v| f(v));
                }
                default_args.iter_mut().for_each(|v| f(v));
            }
            Terminator::Invoke {
                args,
                normal_args,
                exception_args,
                ..
            } => {
                args.iter_mut().for_each(|v| f(v));
                normal_args.iter_mut().for_each(|v| f(v));
                exception_args.iter_mut().for_each(|v| f(v));
            }
            Terminator::InvokeIndirect {
                callee,
                args,
                normal_args,
                exception_args,
                ..
            } => {
                f(callee);
                args.iter_mut().for_each(|v| f(v));
                normal_args.iter_mut().for_each(|v| f(v));
                exception_args.iter_mut().for_each(|v| f(v));
            }
        }
    }

    /// Call `f` for each (target_block, args) pair in this terminator.
    ///
    /// NOTE: `CaptureSlice`'s successors (`handler_block`, `resume_block`)
    /// are intentionally omitted: their first block_param is supplied by
    /// the runtime (handle / resume value), and there are no static args
    /// to mutate. SSA transforms needing to see every successor should use
    /// `successors()` and handle CaptureSlice as a special case.
    pub fn for_each_successor_args_mut(&mut self, mut f: impl FnMut(BlockId, &mut Vec<Value>)) {
        match self {
            Terminator::Ret(_)
            | Terminator::RetVoid
            | Terminator::AbortToPrompt { .. }
            | Terminator::Unreachable
            | Terminator::CaptureSlice { .. } => {}
            Terminator::ResumeSlice { return_block, return_args, .. } => f(*return_block, return_args),
            Terminator::Jump(target, args) => f(*target, args),
            Terminator::BrIf {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                f(*then_block, then_args);
                f(*else_block, else_args);
            }
            Terminator::Switch {
                cases,
                default_block,
                default_args,
                ..
            } => {
                for (_, block, args) in cases {
                    f(*block, args);
                }
                f(*default_block, default_args);
            }
            Terminator::Invoke {
                normal,
                normal_args,
                exception,
                exception_args,
                ..
            } => {
                f(*normal, normal_args);
                f(*exception, exception_args);
            }
            Terminator::InvokeIndirect {
                normal,
                normal_args,
                exception,
                exception_args,
                ..
            } => {
                f(*normal, normal_args);
                f(*exception, exception_args);
            }
        }
    }

    /// Call `f` for each (target_block, args) pair in this terminator (immutable).
    pub fn for_each_successor_args(&self, mut f: impl FnMut(BlockId, &[Value])) {
        match self {
            Terminator::Ret(_)
            | Terminator::RetVoid
            | Terminator::AbortToPrompt { .. }
            | Terminator::Unreachable => {}
            Terminator::CaptureSlice { handler_block, resume_block, .. } => {
                f(*handler_block, &[]);
                f(*resume_block, &[]);
            }
            Terminator::ResumeSlice { return_block, return_args, .. } => f(*return_block, return_args),
            Terminator::Jump(target, args) => f(*target, args),
            Terminator::BrIf {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => {
                f(*then_block, then_args);
                f(*else_block, else_args);
            }
            Terminator::Switch {
                cases,
                default_block,
                default_args,
                ..
            } => {
                for (_, block, args) in cases {
                    f(*block, args);
                }
                f(*default_block, default_args);
            }
            Terminator::Invoke {
                normal,
                normal_args,
                exception,
                exception_args,
                ..
            } => {
                f(*normal, normal_args);
                f(*exception, exception_args);
            }
            Terminator::InvokeIndirect {
                normal,
                normal_args,
                exception,
                exception_args,
                ..
            } => {
                f(*normal, normal_args);
                f(*exception, exception_args);
            }
        }
    }

    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Ret(_)
            | Terminator::RetVoid
            | Terminator::AbortToPrompt { .. }
            | Terminator::Unreachable => vec![],
            Terminator::CaptureSlice { handler_block, resume_block, .. } => vec![*handler_block, *resume_block],
            Terminator::ResumeSlice { return_block, .. } => vec![*return_block],
            Terminator::Jump(target, _) => vec![*target],
            Terminator::BrIf {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
            Terminator::Switch {
                cases,
                default_block,
                ..
            } => {
                let mut succs: Vec<BlockId> = cases.iter().map(|(_, b, _)| *b).collect();
                succs.push(*default_block);
                succs
            }
            Terminator::Invoke {
                normal, exception, ..
            }
            | Terminator::InvokeIndirect {
                normal, exception, ..
            } => {
                vec![*normal, *exception]
            }
        }
    }
}

/// Instruction with its optional result value.
#[derive(Debug, Clone)]
pub struct InstNode {
    pub value: Option<Value>,
    pub inst: Inst,
}

/// A basic block.
#[derive(Debug, Clone)]
pub struct Block {
    pub params: Vec<(Value, Type)>,
    pub insts: Vec<InstNode>,
    pub terminator: Terminator,
}

/// An externally-declared function.
#[derive(Debug, Clone)]
pub struct ExternFunc {
    pub name: String,
    pub sig: Signature,
}

/// A complete function in the IR.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub sig: Signature,
    pub blocks: Vec<Block>,
    pub value_types: Vec<Type>,
    pub extern_funcs: Vec<ExternFunc>,
    pub deopt_info: Vec<DeoptInfo>,
    pub prompt_count: u32,
    /// Function-level stack slot declarations.
    pub stack_slots: Vec<StackSlotData>,
}

impl Function {
    pub fn value_type(&self, v: Value) -> Type {
        self.value_types[v.index()]
    }

    pub fn entry_block(&self) -> &Block {
        &self.blocks[0]
    }

    /// Total number of instructions across all blocks (for fixpoint detection).
    pub fn inst_count(&self) -> usize {
        self.blocks.iter().map(|b| b.insts.len()).sum()
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.index()]
    }

    /// Compute predecessors for each block.
    pub fn predecessors(&self) -> Vec<Vec<BlockId>> {
        let mut preds = vec![vec![]; self.blocks.len()];
        for (i, block) in self.blocks.iter().enumerate() {
            let src = BlockId(i as u32);
            for succ in block.terminator.successors() {
                preds[succ.index()].push(src);
            }
        }
        // ── Abort-to-prompt implicit edges ──────────────────────
        //
        // `Terminator::AbortToPrompt(P)` has no static successors —
        // at runtime it walks up the stack to the prompt-owning frame
        // and then jumps to whichever block was registered as the
        // handler via `Inst::PushPrompt(P, h)`. To make CFG analyses
        // (dominators, dead-block detection) see this as a real
        // control-flow edge, we look up each abort_to_prompt and add
        // the corresponding PushPrompt's handler block as an
        // implicit predecessor.
        //
        // This is what makes lowerings that route a reset's normal
        // exit through `abort_to_prompt(P, [val])` rather than an
        // explicit `pop_prompt + jump` still type-check: the
        // verifier sees the abort site as a predecessor of the
        // handler block and SSA dominance works out.
        let mut prompt_handlers: std::collections::HashMap<u32, BlockId> =
            std::collections::HashMap::new();
        for block in &self.blocks {
            for node in &block.insts {
                if let Inst::PushPrompt(p, h) = &node.inst {
                    prompt_handlers.insert(p.index_u32(), *h);
                }
            }
        }
        for (i, block) in self.blocks.iter().enumerate() {
            if let Terminator::AbortToPrompt { prompt, .. } = &block.terminator {
                if let Some(&handler) = prompt_handlers.get(&prompt.index_u32()) {
                    preds[handler.index()].push(BlockId(i as u32));
                }
            }
        }
        preds
    }
}

// ─── Module ────────────────────────────────────────────────────────

/// A function definition in a module: either an internal IR function or an extern.
#[derive(Debug, Clone)]
pub enum FuncDef {
    /// Index into `Module::functions`.
    Internal(usize),
    /// Host-provided extern function.
    Extern(ExternFunc),
}

/// A collection of functions with a unified function table.
///
/// `FuncRef` values index into `func_table`, which maps to either internal
/// IR functions or extern declarations.
#[derive(Debug, Clone)]
pub struct Module {
    pub functions: Vec<Function>,
    pub func_table: Vec<FuncDef>,
}

impl Module {
    /// Wrap a single [`Function`] into a [`Module`].
    ///
    /// The function's extern declarations are preserved. The function itself
    /// becomes the only internal function, referenced by the returned [`FuncRef`].
    pub fn from_function(func: Function) -> (Self, FuncRef) {
        let mut func_table: Vec<FuncDef> = Vec::new();
        // Externs first (matching FunctionBuilder's FuncRef indices)
        for ef in &func.extern_funcs {
            func_table.push(FuncDef::Extern(ef.clone()));
        }
        // Then the internal function
        let entry_ref = FuncRef(func_table.len() as u32);
        func_table.push(FuncDef::Internal(0));
        let module = Module {
            functions: vec![func],
            func_table,
        };
        (module, entry_ref)
    }

    pub fn func_sig(&self, fref: FuncRef) -> &Signature {
        match &self.func_table[fref.index()] {
            FuncDef::Internal(idx) => &self.functions[*idx].sig,
            FuncDef::Extern(ef) => &ef.sig,
        }
    }

    /// Verify the safepoint contract for a module compiled under a moving GC.
    ///
    /// For every function in the module, every call to an allocating extern
    /// (anything in `allocator_frefs`) must be **immediately preceded** by an
    /// `Inst::Safepoint` in the same basic block. Without this discipline, a
    /// collection that fires inside the allocator would have no stack map for
    /// the calling frame's live values — they'd be invisible to the GC and
    /// would either get freed (use-after-free) or, with a moving collector,
    /// would point at relocated memory and silently corrupt.
    ///
    /// This is a static check on the IR, before machine code emission. The
    /// JIT calls it automatically when compiled with a safepoint handler;
    /// frontends shipping with `CallMode::ControlAware { safepoint_handler }`
    /// can rely on a missing safepoint being caught at compile time, not at
    /// the next collection.
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message naming the offending function,
    /// block, and instruction position. Panic rather than `Result` because a
    /// missing safepoint is a frontend bug, not a recoverable error — there's
    /// nothing reasonable for the caller to do but fix the compiler.
    pub fn validate_safepoints(&self, allocator_frefs: &[FuncRef]) {
        if allocator_frefs.is_empty() {
            return;
        }
        let allocators: std::collections::HashSet<u32> =
            allocator_frefs.iter().map(|f| f.0).collect();
        for func in &self.functions {
            for (block_idx, block) in func.blocks.iter().enumerate() {
                for (inst_idx, node) in block.insts.iter().enumerate() {
                    let calls_allocator = match &node.inst {
                        Inst::Call(fref, _) => allocators.contains(&fref.0),
                        // CallIndirect can't be statically resolved; we
                        // require an immediately-preceding safepoint for
                        // *every* indirect call in a control-aware module.
                        Inst::CallIndirect(_, _, _) => true,
                        Inst::InvokeDynamic { .. } => true,
                        _ => false,
                    };
                    if !calls_allocator {
                        continue;
                    }
                    let preceded_by_safepoint = inst_idx > 0
                        && matches!(block.insts[inst_idx - 1].inst, Inst::Safepoint(_));
                    assert!(
                        preceded_by_safepoint,
                        "validate_safepoints: function `{}` block {} inst {} \
                         calls an allocating function (or indirect/dynamic \
                         dispatch) without an immediately-preceding \
                         `Inst::Safepoint`. \
                         A moving GC running inside the callee would corrupt \
                         the caller's live values.",
                        func.name, block_idx, inst_idx,
                    );
                }
            }
        }
    }

    pub fn func_name(&self, fref: FuncRef) -> &str {
        match &self.func_table[fref.index()] {
            FuncDef::Internal(idx) => &self.functions[*idx].name,
            FuncDef::Extern(ef) => &ef.name,
        }
    }
}
