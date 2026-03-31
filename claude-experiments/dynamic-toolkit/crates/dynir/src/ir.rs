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
    PushPrompt(PromptId),
    /// Pop a prompt boundary when leaving its extent.
    PopPrompt(PromptId),
    /// Capture the current delimited suffix of frames up to `prompt`.
    /// `live` values are explicitly reified into the captured slice.
    CaptureSlice(PromptId, Vec<Value>),
    /// Clone a captured frame slice to enable multi-shot resume.
    CloneSlice(Value),

    // -- Calls --
    Call(FuncRef, Vec<Value>),
    CallIndirect(Value, Vec<Value>, Option<Type>),

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
            Inst::PushPrompt(_) | Inst::PopPrompt(_) => None,
            Inst::CaptureSlice(_, _) | Inst::CloneSlice(_) => Some(Type::FrameSlice),

            Inst::Call(fref, _) => extern_sigs[fref.index()].sig.ret,
            Inst::CallIndirect(_, _, ret_ty) => *ret_ty,
            Inst::Safepoint(_) => None,
        }
    }

    /// Call `f` on every Value operand.
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            Inst::Iconst(_, _) | Inst::F64Const(_) | Inst::StackAddr(_) => {}

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

            Inst::PushPrompt(_) | Inst::PopPrompt(_) => {}
            Inst::CaptureSlice(_, live) => live.iter().for_each(|v| f(*v)),
            Inst::CloneSlice(v) => f(*v),

            Inst::Call(_, args) => args.iter().for_each(|v| f(*v)),
            Inst::CallIndirect(callee, args, _) => {
                f(*callee);
                args.iter().for_each(|v| f(*v));
            }
            Inst::Safepoint(live) => live.iter().for_each(|v| f(*v)),
        }
    }

    /// Call `f` on every Value operand (mutable).
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            Inst::Iconst(_, _) | Inst::F64Const(_) | Inst::StackAddr(_) => {}

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

            Inst::PushPrompt(_) | Inst::PopPrompt(_) => {}
            Inst::CaptureSlice(_, live) => live.iter_mut().for_each(|v| f(v)),
            Inst::CloneSlice(v) => f(v),

            Inst::Call(_, args) => args.iter_mut().for_each(|v| f(v)),
            Inst::CallIndirect(callee, args, _) => {
                f(callee);
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
    /// Resume a previously captured frame slice. This is a non-local control transfer.
    ResumeSlice {
        slice: Value,
        args: Vec<Value>,
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
            Terminator::AbortToPrompt { args, .. } => args.iter().for_each(|v| f(*v)),
            Terminator::ResumeSlice { slice, args } => {
                f(*slice);
                args.iter().for_each(|v| f(*v));
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
            Terminator::AbortToPrompt { args, .. } => args.iter_mut().for_each(|v| f(v)),
            Terminator::ResumeSlice { slice, args } => {
                f(slice);
                args.iter_mut().for_each(|v| f(v));
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
    pub fn for_each_successor_args_mut(&mut self, mut f: impl FnMut(BlockId, &mut Vec<Value>)) {
        match self {
            Terminator::Ret(_)
            | Terminator::RetVoid
            | Terminator::ResumeSlice { .. }
            | Terminator::AbortToPrompt { .. }
            | Terminator::Unreachable => {}
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
            | Terminator::ResumeSlice { .. }
            | Terminator::AbortToPrompt { .. }
            | Terminator::Unreachable => vec![],
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

    pub fn func_name(&self, fref: FuncRef) -> &str {
        match &self.func_table[fref.index()] {
            FuncDef::Internal(idx) => &self.functions[*idx].name,
            FuncDef::Extern(ef) => &ef.name,
        }
    }
}
