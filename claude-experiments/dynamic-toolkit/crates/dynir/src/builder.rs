use crate::ir::*;
use crate::types::{Signature, Type};

// ─── ModuleBuilder ─────────────────────────────────────────────────

enum ModuleEntryKind {
    Extern,
    Internal(usize), // index into internal_funcs
}

struct ModuleEntry {
    name: String,
    sig: Signature,
    kind: ModuleEntryKind,
}

/// Builder for constructing a [`Module`] containing multiple functions.
///
/// Usage: declare all functions and externs first, then define each internal
/// function using [`define_func`], and finally call [`build`].
///
/// ```ignore
/// let mut mb = ModuleBuilder::new();
/// let ext = mb.declare_extern("print", sig);
/// let f1 = mb.declare_func("main", &[Type::I64], Some(Type::I64));
/// let f2 = mb.declare_func("helper", &[Type::I64], Some(Type::I64));
///
/// let mut fb = mb.define_func(f1);
/// // ... build function body using fb ...
/// mb.finish_func(f1, fb);
///
/// let mut fb = mb.define_func(f2);
/// // ... build function body ...
/// mb.finish_func(f2, fb);
///
/// let module = mb.build();
/// ```
pub struct ModuleBuilder {
    entries: Vec<ModuleEntry>,
    internal_funcs: Vec<Option<Function>>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        ModuleBuilder {
            entries: Vec::new(),
            internal_funcs: Vec::new(),
        }
    }

    /// Declare an extern (host-provided) function. Returns a module-global FuncRef.
    pub fn declare_extern(&mut self, name: &str, sig: Signature) -> FuncRef {
        let fref = FuncRef(self.entries.len() as u32);
        self.entries.push(ModuleEntry {
            name: name.to_string(),
            sig,
            kind: ModuleEntryKind::Extern,
        });
        fref
    }

    /// Declare an internal function. Returns a module-global FuncRef.
    /// Call [`define_func`] later to provide the function body.
    pub fn declare_func(&mut self, name: &str, params: &[Type], ret: Option<Type>) -> FuncRef {
        let fref = FuncRef(self.entries.len() as u32);
        let internal_idx = self.internal_funcs.len();
        self.internal_funcs.push(None);
        self.entries.push(ModuleEntry {
            name: name.to_string(),
            sig: Signature {
                params: params.to_vec(),
                ret,
            },
            kind: ModuleEntryKind::Internal(internal_idx),
        });
        fref
    }

    /// Create a [`FunctionBuilder`] for a previously declared internal function.
    ///
    /// All declarations (both extern and internal) must be done before calling this,
    /// so that the builder can validate calls to any module-level function.
    pub fn define_func(&self, fref: FuncRef) -> FunctionBuilder {
        let entry = &self.entries[fref.index()];
        match entry.kind {
            ModuleEntryKind::Internal(_) => {}
            ModuleEntryKind::Extern => panic!("cannot define extern function '{}'", entry.name),
        }

        // Create a FunctionBuilder with all module-level signatures as extern_funcs,
        // so that Call/Invoke can validate arguments against any FuncRef.
        let mut fb = FunctionBuilder::new(&entry.name, &entry.sig.params, entry.sig.ret);
        fb.extern_funcs = self
            .entries
            .iter()
            .map(|e| ExternFunc {
                name: e.name.clone(),
                sig: e.sig.clone(),
            })
            .collect();
        fb
    }

    /// Finish defining an internal function by providing the built FunctionBuilder.
    pub fn finish_func(&mut self, fref: FuncRef, fb: FunctionBuilder) {
        let entry = &self.entries[fref.index()];
        let internal_idx = match entry.kind {
            ModuleEntryKind::Internal(idx) => idx,
            ModuleEntryKind::Extern => panic!("cannot finish extern function '{}'", entry.name),
        };
        assert!(
            self.internal_funcs[internal_idx].is_none(),
            "function '{}' already defined",
            entry.name
        );
        self.internal_funcs[internal_idx] = Some(fb.build());
    }

    /// Build the module. Panics if any declared internal function was not defined.
    pub fn build(self) -> Module {
        for (i, func_opt) in self.internal_funcs.iter().enumerate() {
            assert!(
                func_opt.is_some(),
                "internal function at index {} was declared but not defined",
                i
            );
        }

        let functions: Vec<Function> = self
            .internal_funcs
            .into_iter()
            .map(|f| f.unwrap())
            .collect();

        let func_table: Vec<FuncDef> = self
            .entries
            .into_iter()
            .map(|entry| match entry.kind {
                ModuleEntryKind::Internal(idx) => FuncDef::Internal(idx),
                ModuleEntryKind::Extern => FuncDef::Extern(ExternFunc {
                    name: entry.name,
                    sig: entry.sig,
                }),
            })
            .collect();

        Module {
            functions,
            func_table,
        }
    }
}

/// Internal mutable block data during construction.
struct BlockData {
    params: Vec<(Value, Type)>,
    insts: Vec<InstNode>,
    terminator: Option<Terminator>,
}

/// Builder for constructing IR functions.
///
/// The builder tracks value types and validates operations eagerly,
/// panicking on type mismatches so bugs are caught at construction time.
pub struct FunctionBuilder {
    name: String,
    sig: Signature,
    blocks: Vec<BlockData>,
    value_types: Vec<Type>,
    next_value: u32,
    current_block: Option<BlockId>,
    extern_funcs: Vec<ExternFunc>,
    deopt_info: Vec<DeoptInfo>,
    next_prompt: u32,
    stack_slots: Vec<StackSlotData>,
}

impl FunctionBuilder {
    /// Create a new function builder. An entry block (bb0) is created
    /// automatically with parameters matching the function signature.
    pub fn new(name: &str, params: &[Type], ret: Option<Type>) -> Self {
        let sig = Signature {
            params: params.to_vec(),
            ret,
        };
        let mut b = FunctionBuilder {
            name: name.to_string(),
            sig,
            blocks: Vec::new(),
            value_types: Vec::new(),
            next_value: 0,
            current_block: None,
            extern_funcs: Vec::new(),
            deopt_info: Vec::new(),
            next_prompt: 0,
            stack_slots: Vec::new(),
        };
        let entry = b.create_block(params);
        b.switch_to_block(entry);
        b
    }

    /// The entry block (always bb0).
    pub fn entry_block(&self) -> BlockId {
        BlockId(0)
    }

    /// Create a new basic block with the given parameter types.
    pub fn create_block(&mut self, param_types: &[Type]) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        let params: Vec<(Value, Type)> = param_types
            .iter()
            .map(|&ty| {
                let v = self.alloc_value(ty);
                (v, ty)
            })
            .collect();
        self.blocks.push(BlockData {
            params,
            insts: Vec::new(),
            terminator: None,
        });
        id
    }

    /// Switch emission to the given block.
    pub fn switch_to_block(&mut self, block: BlockId) {
        if let Some(cur) = self.current_block {
            assert!(
                self.blocks[cur.index()].terminator.is_some(),
                "current block bb{} must be terminated before switching",
                cur.index()
            );
        }
        assert!(
            self.blocks[block.index()].terminator.is_none(),
            "cannot switch to bb{} which is already terminated",
            block.index()
        );
        self.current_block = Some(block);
    }

    /// Get the Value for a block parameter.
    pub fn block_param(&self, block: BlockId, index: usize) -> Value {
        self.blocks[block.index()].params[index].0
    }

    /// Declare an external function that can be called.
    pub fn declare_func(&mut self, name: &str, sig: Signature) -> FuncRef {
        let id = FuncRef(self.extern_funcs.len() as u32);
        self.extern_funcs.push(ExternFunc {
            name: name.to_string(),
            sig,
        });
        id
    }

    /// Get the type of a value.
    pub fn value_type(&self, v: Value) -> Type {
        self.value_types[v.index()]
    }

    // ── Constants ──────────────────────────────────────────────

    /// Emit a safepoint: a GC-safe point where live GcPtr values are recorded.
    /// The lowering will emit a stack map here so the collector can trace roots.
    pub fn safepoint(&mut self, live: &[Value]) {
        for &v in live {
            assert!(
                self.value_type(v).is_gc(),
                "safepoint live value {} must be GcPtr, got {}",
                v,
                self.value_type(v)
            );
        }
        self.push_void_inst(Inst::Safepoint(live.to_vec()));
    }

    pub fn iconst(&mut self, ty: Type, val: i64) -> Value {
        assert!(
            ty.is_int() || ty.is_ptr(),
            "iconst requires int or ptr type"
        );
        self.push_inst(ty, Inst::Iconst(ty, val))
    }

    pub fn f64const(&mut self, val: f64) -> Value {
        self.push_inst(Type::F64, Inst::F64Const(val))
    }

    // ── Integer arithmetic ─────────────────────────────────────

    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::Add, a, b)
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::Sub, a, b)
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::Mul, a, b)
    }

    pub fn sdiv(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::SDiv, a, b)
    }

    pub fn udiv(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::UDiv, a, b)
    }

    // ── Float arithmetic ───────────────────────────────────────

    pub fn fadd(&mut self, a: Value, b: Value) -> Value {
        self.float_binop(Inst::FAdd, a, b)
    }

    pub fn fsub(&mut self, a: Value, b: Value) -> Value {
        self.float_binop(Inst::FSub, a, b)
    }

    pub fn fmul(&mut self, a: Value, b: Value) -> Value {
        self.float_binop(Inst::FMul, a, b)
    }

    pub fn fdiv(&mut self, a: Value, b: Value) -> Value {
        self.float_binop(Inst::FDiv, a, b)
    }

    // ── Bitwise ────────────────────────────────────────────────

    pub fn and(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::And, a, b)
    }

    pub fn or(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::Or, a, b)
    }

    pub fn xor(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::Xor, a, b)
    }

    pub fn shl(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::Shl, a, b)
    }

    pub fn lshr(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::LShr, a, b)
    }

    pub fn ashr(&mut self, a: Value, b: Value) -> Value {
        self.int_binop(Inst::AShr, a, b)
    }

    // ── Unary ──────────────────────────────────────────────────

    pub fn neg(&mut self, v: Value) -> Value {
        let ty = self.value_type(v);
        assert!(ty.is_int(), "neg requires int type, got {ty}");
        self.push_inst(ty, Inst::Neg(v))
    }

    pub fn fneg(&mut self, v: Value) -> Value {
        assert_eq!(self.value_type(v), Type::F64, "fneg requires f64");
        self.push_inst(Type::F64, Inst::FNeg(v))
    }

    pub fn not(&mut self, v: Value) -> Value {
        let ty = self.value_type(v);
        assert!(ty.is_int(), "not requires int type, got {ty}");
        self.push_inst(ty, Inst::Not(v))
    }

    // ── Comparison ─────────────────────────────────────────────

    pub fn icmp(&mut self, op: CmpOp, a: Value, b: Value) -> Value {
        let ta = self.value_type(a);
        let tb = self.value_type(b);
        assert!(
            ta == tb && ta.is_int(),
            "icmp requires matching int types, got {ta} and {tb}"
        );
        self.push_inst(Type::I8, Inst::Icmp(op, a, b))
    }

    pub fn fcmp(&mut self, op: CmpOp, a: Value, b: Value) -> Value {
        let ta = self.value_type(a);
        let tb = self.value_type(b);
        assert!(
            ta == tb && ta.is_float(),
            "fcmp requires matching float types, got {ta} and {tb}"
        );
        self.push_inst(Type::I8, Inst::Fcmp(op, a, b))
    }

    // ── Conversions ────────────────────────────────────────────

    pub fn sext(&mut self, v: Value, to: Type) -> Value {
        assert!(
            self.value_type(v).is_int() && to.is_int(),
            "sext requires int types"
        );
        self.push_inst(to, Inst::Sext(v, to))
    }

    pub fn zext(&mut self, v: Value, to: Type) -> Value {
        assert!(
            self.value_type(v).is_int() && to.is_int(),
            "zext requires int types"
        );
        self.push_inst(to, Inst::Zext(v, to))
    }

    pub fn trunc(&mut self, v: Value, to: Type) -> Value {
        assert!(
            self.value_type(v).is_int() && to.is_int(),
            "trunc requires int types"
        );
        self.push_inst(to, Inst::Trunc(v, to))
    }

    pub fn int_to_float(&mut self, v: Value) -> Value {
        assert!(
            self.value_type(v).is_int(),
            "int_to_float requires int input"
        );
        self.push_inst(Type::F64, Inst::IntToFloat(v))
    }

    pub fn float_to_int(&mut self, v: Value) -> Value {
        assert!(
            self.value_type(v).is_float(),
            "float_to_int requires float input"
        );
        self.push_inst(Type::I64, Inst::FloatToInt(v))
    }

    pub fn bitcast(&mut self, v: Value, to: Type) -> Value {
        let from = self.value_type(v);
        assert_eq!(
            from.size_bytes(),
            to.size_bytes(),
            "bitcast requires same-size types, got {from} -> {to}"
        );
        self.push_inst(to, Inst::Bitcast(v, to))
    }

    // ── Stack slots ────────────────────────────────────────────

    /// Declare a function-level stack slot. Returns a handle that can be
    /// used with `stack_addr` to get a pointer to the slot's memory.
    pub fn create_stack_slot(&mut self, size: u32, is_gc_root: bool) -> StackSlot {
        let idx = self.stack_slots.len();
        self.stack_slots.push(StackSlotData { size, is_gc_root });
        StackSlot(idx as u32)
    }

    /// Get the address of a declared stack slot. Returns a `Ptr`.
    pub fn stack_addr(&mut self, slot: StackSlot) -> Value {
        assert!(
            (slot.index()) < self.stack_slots.len(),
            "stack_addr: slot index {} out of range (have {})",
            slot.index(),
            self.stack_slots.len()
        );
        self.push_inst(Type::Ptr, Inst::StackAddr(slot))
    }

    // ── Memory ─────────────────────────────────────────────────

    pub fn load(&mut self, ty: Type, addr: Value, offset: i32) -> Value {
        let at = self.value_type(addr);
        assert!(
            at.is_ptr() || at == Type::I64,
            "load addr must be ptr or i64, got {at}"
        );
        self.push_inst(ty, Inst::Load(ty, addr, offset))
    }

    pub fn store(&mut self, value: Value, addr: Value, offset: i32) {
        let at = self.value_type(addr);
        assert!(
            at.is_ptr() || at == Type::I64,
            "store addr must be ptr or i64, got {at}"
        );
        self.push_void_inst(Inst::Store(value, addr, offset));
    }

    // ── Tagged values ──────────────────────────────────────────

    pub fn tag_of(&mut self, v: Value) -> Value {
        self.push_inst(Type::I32, Inst::TagOf(v))
    }

    pub fn payload(&mut self, v: Value) -> Value {
        self.push_inst(Type::I64, Inst::Payload(v))
    }

    pub fn make_tagged(&mut self, tag: u32, payload: Value) -> Value {
        self.push_inst(Type::I64, Inst::MakeTagged(tag, payload))
    }

    pub fn is_tag(&mut self, v: Value, tag: u32) -> Value {
        self.push_inst(Type::I8, Inst::IsTag(v, tag))
    }

    // ── Select ─────────────────────────────────────────────────

    pub fn select(&mut self, cond: Value, if_true: Value, if_false: Value) -> Value {
        assert_eq!(self.value_type(cond), Type::I8, "select cond must be i8");
        let tt = self.value_type(if_true);
        let tf = self.value_type(if_false);
        assert_eq!(tt, tf, "select arms must have same type, got {tt} and {tf}");
        self.push_inst(tt, Inst::Select(cond, if_true, if_false))
    }

    // ── Overflow checking ────────────────────────────────────────

    pub fn overflow_check(&mut self, op: OverflowOp, a: Value, b: Value) -> Value {
        let ta = self.value_type(a);
        let tb = self.value_type(b);
        assert!(
            ta == tb && ta.is_int(),
            "overflow_check requires matching int types, got {ta} and {tb}"
        );
        self.push_inst(Type::I8, Inst::OverflowCheck(op, a, b))
    }

    // ── Guard / deoptimization ─────────────────────────────────

    /// Create a deoptimization point. Returns a DeoptId that can be used with `guard()`.
    pub fn create_deopt(&mut self, resume_point: u64, description: &str) -> DeoptId {
        let id = DeoptId(self.deopt_info.len() as u32);
        self.deopt_info.push(DeoptInfo {
            resume_point,
            description: description.to_string(),
        });
        id
    }

    /// Emit a guard: if `cond` is false, deoptimize with the given metadata.
    /// `live` values are captured in the deopt frame.
    pub fn guard(&mut self, cond: Value, deopt_id: DeoptId, live: &[Value]) {
        assert_eq!(self.value_type(cond), Type::I8, "guard cond must be i8");
        assert!(deopt_id.index() < self.deopt_info.len(), "invalid deopt_id");
        self.push_void_inst(Inst::Guard(cond, deopt_id, live.to_vec()));
    }

    // ── Delimited frame slices ─────────────────────────────────

    pub fn create_prompt(&mut self) -> PromptId {
        let id = PromptId(self.next_prompt);
        self.next_prompt += 1;
        id
    }

    pub fn push_prompt(&mut self, prompt: PromptId) {
        assert!(prompt.index() < self.next_prompt as usize, "invalid prompt id");
        self.push_void_inst(Inst::PushPrompt(prompt));
    }

    pub fn pop_prompt(&mut self, prompt: PromptId) {
        assert!(prompt.index() < self.next_prompt as usize, "invalid prompt id");
        self.push_void_inst(Inst::PopPrompt(prompt));
    }

    pub fn capture_slice(&mut self, prompt: PromptId, live: &[Value]) -> Value {
        assert!(prompt.index() < self.next_prompt as usize, "invalid prompt id");
        self.push_inst(Type::FrameSlice, Inst::CaptureSlice(prompt, live.to_vec()))
    }

    pub fn clone_slice(&mut self, slice: Value) -> Value {
        assert_eq!(
            self.value_type(slice),
            Type::FrameSlice,
            "clone_slice requires frameslice, got {}",
            self.value_type(slice)
        );
        self.push_inst(Type::FrameSlice, Inst::CloneSlice(slice))
    }

    // ── Calls ──────────────────────────────────────────────────

    pub fn call(&mut self, func: FuncRef, args: &[Value]) -> Option<Value> {
        let sig = &self.extern_funcs[func.index()].sig;
        assert_eq!(
            sig.params.len(),
            args.len(),
            "call arg count mismatch: expected {}, got {}",
            sig.params.len(),
            args.len()
        );
        for (i, (&param_ty, &arg)) in sig.params.iter().zip(args.iter()).enumerate() {
            let at = self.value_type(arg);
            assert_eq!(
                at, param_ty,
                "call arg {i} type mismatch: expected {param_ty}, got {at}"
            );
        }
        let ret_ty = sig.ret;
        let inst = Inst::Call(func, args.to_vec());
        if let Some(ty) = ret_ty {
            Some(self.push_inst(ty, inst))
        } else {
            self.push_void_inst(inst);
            None
        }
    }

    pub fn call_indirect(
        &mut self,
        callee: Value,
        args: &[Value],
        ret_ty: Option<Type>,
    ) -> Option<Value> {
        let inst = Inst::CallIndirect(callee, args.to_vec(), ret_ty);
        if let Some(ty) = ret_ty {
            Some(self.push_inst(ty, inst))
        } else {
            self.push_void_inst(inst);
            None
        }
    }

    // ── Terminators ────────────────────────────────────────────

    pub fn ret(&mut self, v: Value) {
        assert_eq!(
            self.sig.ret,
            Some(self.value_type(v)),
            "return type mismatch"
        );
        self.set_terminator(Terminator::Ret(v));
    }

    pub fn ret_void(&mut self) {
        assert_eq!(self.sig.ret, None, "function has a return type, use ret()");
        self.set_terminator(Terminator::RetVoid);
    }

    pub fn jump(&mut self, target: BlockId, args: &[Value]) {
        self.check_branch_args(target, args);
        self.set_terminator(Terminator::Jump(target, args.to_vec()));
    }

    pub fn br_if(
        &mut self,
        cond: Value,
        then_block: BlockId,
        then_args: &[Value],
        else_block: BlockId,
        else_args: &[Value],
    ) {
        assert_eq!(self.value_type(cond), Type::I8, "br_if cond must be i8");
        self.check_branch_args(then_block, then_args);
        self.check_branch_args(else_block, else_args);
        self.set_terminator(Terminator::BrIf {
            cond,
            then_block,
            then_args: then_args.to_vec(),
            else_block,
            else_args: else_args.to_vec(),
        });
    }

    pub fn switch(
        &mut self,
        val: Value,
        cases: &[(i64, BlockId, &[Value])],
        default_block: BlockId,
        default_args: &[Value],
    ) {
        let vt = self.value_type(val);
        assert!(vt.is_int(), "switch value must be int, got {vt}");
        self.check_branch_args(default_block, default_args);
        let cases_vec: Vec<(i64, BlockId, Vec<Value>)> = cases
            .iter()
            .map(|(c, block, args)| {
                self.check_branch_args(*block, args);
                (*c, *block, args.to_vec())
            })
            .collect();
        self.set_terminator(Terminator::Switch {
            val,
            cases: cases_vec,
            default_block,
            default_args: default_args.to_vec(),
        });
    }

    pub fn invoke(
        &mut self,
        func: FuncRef,
        args: &[Value],
        normal: BlockId,
        normal_args: &[Value],
        exception: BlockId,
        exception_args: &[Value],
    ) {
        let sig = &self.extern_funcs[func.index()].sig;
        assert_eq!(
            sig.params.len(),
            args.len(),
            "invoke arg count mismatch: expected {}, got {}",
            sig.params.len(),
            args.len()
        );
        for (i, (&param_ty, &arg)) in sig.params.iter().zip(args.iter()).enumerate() {
            let at = self.value_type(arg);
            assert_eq!(
                at, param_ty,
                "invoke arg {i} type mismatch: expected {param_ty}, got {at}"
            );
        }
        // Normal block: first param is the return value (if any), rest must match normal_args
        self.check_invoke_normal_args(sig.ret, normal, normal_args);
        self.check_branch_args(exception, exception_args);
        self.set_terminator(Terminator::Invoke {
            func,
            args: args.to_vec(),
            normal,
            normal_args: normal_args.to_vec(),
            exception,
            exception_args: exception_args.to_vec(),
        });
    }

    pub fn invoke_indirect(
        &mut self,
        callee: Value,
        args: &[Value],
        ret_ty: Option<Type>,
        normal: BlockId,
        normal_args: &[Value],
        exception: BlockId,
        exception_args: &[Value],
    ) {
        self.check_invoke_normal_args(ret_ty, normal, normal_args);
        self.check_branch_args(exception, exception_args);
        self.set_terminator(Terminator::InvokeIndirect {
            callee,
            args: args.to_vec(),
            ret_ty,
            normal,
            normal_args: normal_args.to_vec(),
            exception,
            exception_args: exception_args.to_vec(),
        });
    }

    pub fn unreachable(&mut self) {
        self.set_terminator(Terminator::Unreachable);
    }

    pub fn resume_slice(&mut self, slice: Value, args: &[Value]) {
        assert_eq!(
            self.value_type(slice),
            Type::FrameSlice,
            "resume_slice requires frameslice, got {}",
            self.value_type(slice)
        );
        self.set_terminator(Terminator::ResumeSlice {
            slice,
            args: args.to_vec(),
        });
    }

    pub fn abort_to_prompt(&mut self, prompt: PromptId, args: &[Value]) {
        assert!(prompt.index() < self.next_prompt as usize, "invalid prompt id");
        self.set_terminator(Terminator::AbortToPrompt {
            prompt,
            args: args.to_vec(),
        });
    }

    // ── Build ──────────────────────────────────────────────────

    /// Finalize and produce the Function.
    /// Panics if any block is unterminated.
    pub fn build(self) -> Function {
        for (i, block) in self.blocks.iter().enumerate() {
            assert!(block.terminator.is_some(), "block bb{i} is not terminated");
        }
        let blocks = self
            .blocks
            .into_iter()
            .map(|b| Block {
                params: b.params,
                insts: b.insts,
                terminator: b.terminator.unwrap(),
            })
            .collect();
        Function {
            name: self.name,
            sig: self.sig,
            blocks,
            value_types: self.value_types,
            extern_funcs: self.extern_funcs,
            deopt_info: self.deopt_info,
            prompt_count: self.next_prompt,
            stack_slots: self.stack_slots,
        }
    }

    // ── Internal helpers ───────────────────────────────────────

    fn alloc_value(&mut self, ty: Type) -> Value {
        let v = Value(self.next_value);
        self.next_value += 1;
        self.value_types.push(ty);
        v
    }

    fn cur_block(&mut self) -> &mut BlockData {
        let id = self
            .current_block
            .expect("no current block — call switch_to_block first");
        &mut self.blocks[id.index()]
    }

    fn push_inst(&mut self, ty: Type, inst: Inst) -> Value {
        let v = self.alloc_value(ty);
        self.cur_block().insts.push(InstNode {
            value: Some(v),
            inst,
        });
        v
    }

    fn push_void_inst(&mut self, inst: Inst) {
        self.cur_block().insts.push(InstNode { value: None, inst });
    }

    fn set_terminator(&mut self, term: Terminator) {
        let block = self.cur_block();
        assert!(block.terminator.is_none(), "block already has a terminator");
        block.terminator = Some(term);
    }

    fn check_branch_args(&self, target: BlockId, args: &[Value]) {
        let params = &self.blocks[target.index()].params;
        assert_eq!(
            params.len(),
            args.len(),
            "branch to bb{} arg count mismatch: expected {}, got {}",
            target.index(),
            params.len(),
            args.len()
        );
        for (i, (&(_, pty), &arg)) in params.iter().zip(args.iter()).enumerate() {
            let at = self.value_type(arg);
            assert_eq!(
                at,
                pty,
                "branch to bb{} arg {i} type mismatch: expected {pty}, got {at}",
                target.index()
            );
        }
    }

    /// For invoke: the normal block's params = [return_value_param] + normal_args_params.
    /// The return value (if any) is implicitly passed as the first block param.
    fn check_invoke_normal_args(&self, ret_ty: Option<Type>, target: BlockId, args: &[Value]) {
        let params = &self.blocks[target.index()].params;
        let ret_param_count = if ret_ty.is_some() { 1 } else { 0 };
        let expected_args = params.len() - ret_param_count;
        assert_eq!(
            expected_args,
            args.len(),
            "invoke normal branch to bb{} arg count mismatch: expected {expected_args} (block has {} params, {} is return value), got {}",
            target.index(),
            params.len(),
            ret_param_count,
            args.len()
        );
        // Check return type matches first param
        if let Some(rty) = ret_ty {
            assert_eq!(
                params[0].1,
                rty,
                "invoke normal bb{} first param must match return type: expected {rty}, got {}",
                target.index(),
                params[0].1
            );
        }
        // Check remaining args match remaining params
        for (i, (&(_, pty), &arg)) in params[ret_param_count..]
            .iter()
            .zip(args.iter())
            .enumerate()
        {
            let at = self.value_type(arg);
            assert_eq!(
                at,
                pty,
                "invoke normal branch to bb{} arg {i} type mismatch: expected {pty}, got {at}",
                target.index()
            );
        }
    }

    fn int_binop(&mut self, ctor: fn(Value, Value) -> Inst, a: Value, b: Value) -> Value {
        let ta = self.value_type(a);
        let tb = self.value_type(b);

        // Allow pointer arithmetic: Ptr/GcPtr + I64 -> Ptr/GcPtr, I64 + Ptr/GcPtr -> Ptr/GcPtr
        if (ta.is_ptr() && tb == Type::I64) || (ta == Type::I64 && tb.is_ptr()) {
            let result_ty = if ta.is_ptr() { ta } else { tb };
            return self.push_inst(result_ty, ctor(a, b));
        }

        assert!(
            ta == tb && (ta.is_int() || ta.is_ptr()),
            "integer binop requires matching int/ptr types, got {ta} and {tb}"
        );
        self.push_inst(ta, ctor(a, b))
    }

    fn float_binop(&mut self, ctor: fn(Value, Value) -> Inst, a: Value, b: Value) -> Value {
        let ta = self.value_type(a);
        let tb = self.value_type(b);
        assert!(
            ta == tb && ta.is_float(),
            "float binop requires F64 types, got {ta} and {tb}"
        );
        self.push_inst(Type::F64, ctor(a, b))
    }
}
