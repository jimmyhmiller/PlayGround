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

/// Opaque marker for [`ModuleBuilder::checkpoint`]/[`ModuleBuilder::rollback`].
#[derive(Clone, Copy, Debug)]
pub struct ModuleBuilderCheckpoint {
    entries: usize,
    internal_funcs: usize,
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

    /// Snapshot the currently-defined functions and externs into a [`Module`],
    /// without consuming the builder. The builder remains live and can be
    /// extended with additional `declare_*` / `define_func` / `finish_func`
    /// calls; later snapshots will include the new functions.
    ///
    /// Panics if any declared internal function was not yet defined — only a
    /// fully-defined builder snapshots cleanly.
    pub fn snapshot(&self) -> Module {
        for (i, func_opt) in self.internal_funcs.iter().enumerate() {
            assert!(
                func_opt.is_some(),
                "internal function at index {} was declared but not defined",
                i
            );
        }

        let functions: Vec<Function> = self
            .internal_funcs
            .iter()
            .map(|f| f.as_ref().unwrap().clone())
            .collect();

        let func_table: Vec<FuncDef> = self
            .entries
            .iter()
            .map(|entry| match entry.kind {
                ModuleEntryKind::Internal(idx) => FuncDef::Internal(idx),
                ModuleEntryKind::Extern => FuncDef::Extern(ExternFunc {
                    name: entry.name.clone(),
                    sig: entry.sig.clone(),
                }),
            })
            .collect();

        Module {
            functions,
            func_table,
        }
    }

    /// Capture the current builder size so a failed/abandoned batch of
    /// `declare_func`/`finish_func` calls can be rolled back with
    /// [`rollback`]. Cheap: just records the two vector lengths.
    ///
    /// This is what makes per-form compilation transactional: a frontend
    /// that declares functions and then panics mid-definition (leaving an
    /// internal function "declared but not defined") would otherwise poison
    /// every later [`snapshot`] — `checkpoint`/`rollback` discards the
    /// half-built functions instead.
    pub fn checkpoint(&self) -> ModuleBuilderCheckpoint {
        ModuleBuilderCheckpoint {
            entries: self.entries.len(),
            internal_funcs: self.internal_funcs.len(),
        }
    }

    /// Discard every entry and internal function added since `cp` was taken.
    /// Safe because functions are only referenced by FuncRef indices, and
    /// anything added after `cp` can only be referenced by other things
    /// added after `cp` (which are also being discarded).
    pub fn rollback(&mut self, cp: ModuleBuilderCheckpoint) {
        debug_assert!(self.entries.len() >= cp.entries);
        debug_assert!(self.internal_funcs.len() >= cp.internal_funcs);
        self.entries.truncate(cp.entries);
        self.internal_funcs.truncate(cp.internal_funcs);
    }

    /// Number of currently-declared functions (extern + internal).
    pub fn func_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of currently-defined internal functions (those in `Module::functions`).
    pub fn internal_func_count(&self) -> usize {
        self.internal_funcs.len()
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

    /// True iff the current block already has a terminator (the next
    /// `switch_to_block` is required). Useful when a sub-expression
    /// might emit a non-returning terminator (recur, throw, abort)
    /// and the caller wants to skip subsequent cleanup that would
    /// land in an unreachable block with the wrong prompt stack.
    pub fn current_block_is_terminated(&self) -> bool {
        match self.current_block {
            Some(b) => self.blocks[b.index()].terminator.is_some(),
            None => true,
        }
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

    /// Inform this `FunctionBuilder` about a function declared on the
    /// owning `ModuleBuilder` *after* this builder was created (e.g.
    /// a helper function synthesized while compiling the body of the
    /// current function). Without this, `fb.call(fref)` /
    /// `fb.invoke(fref)` would index past the snapshot taken at
    /// `define_func` time and panic.
    ///
    /// `fref` must equal the one returned by `mb.declare_func`. The
    /// internal table is grown with placeholder entries up to
    /// `fref.index()` if needed; the slot at `fref.index()` is then
    /// set to `(name, sig)`.
    pub fn import_module_func(&mut self, fref: FuncRef, name: &str, sig: Signature) {
        let placeholder = || ExternFunc {
            name: String::new(),
            sig: Signature {
                params: Vec::new(),
                ret: None,
            },
        };
        while self.extern_funcs.len() < fref.index() {
            self.extern_funcs.push(placeholder());
        }
        let entry = ExternFunc {
            name: name.to_string(),
            sig,
        };
        if self.extern_funcs.len() == fref.index() {
            self.extern_funcs.push(entry);
        } else {
            self.extern_funcs[fref.index()] = entry;
        }
    }

    /// Get the type of a value.
    pub fn value_type(&self, v: Value) -> Type {
        self.value_types[v.index()]
    }

    // ── Constants ──────────────────────────────────────────────

    /// Emit a safepoint: a GC-safe point where live values that may hold
    /// heap pointers are recorded.
    ///
    /// Each live value must be one of:
    /// - `GcPtr` — typed root, always traced.
    /// - `I64` — used by NanBox-style frontends. The runtime's `PtrPolicy`
    ///   filters non-pointer payloads at scan time.
    /// - `Ptr` — raw pointer that the GC's policy may also choose to
    ///   trace.
    /// - `FrameSlice` — captured continuation frame slice.
    ///
    /// Values of other types (`I8`/`I32`/`F64`) cannot hold heap pointers
    /// and are rejected — listing them is almost certainly a bug.
    pub fn safepoint(&mut self, live: &[Value]) {
        for &v in live {
            let ty = self.value_type(v);
            let acceptable = ty.is_gc() || matches!(ty, Type::I64 | Type::Ptr);
            assert!(
                acceptable,
                "safepoint live value {} has type {} — only GcPtr, I64, \
                 Ptr, and FrameSlice can hold heap pointers",
                v, ty
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

    /// Load a NanBox-encoded GC-managed literal from the JitModule's
    /// literal pool by index. The pool slot must be populated (via the
    /// JitModule's `literal_pool().push(...)`) before the emitted code runs.
    pub fn gc_literal(&mut self, lit: LiteralRef) -> Value {
        self.push_inst(Type::I64, Inst::GcLiteral(lit))
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

    pub fn push_prompt(&mut self, prompt: PromptId, handler_block: BlockId) {
        assert!(
            prompt.index() < self.next_prompt as usize,
            "invalid prompt id"
        );
        self.push_void_inst(Inst::PushPrompt(prompt, handler_block));
    }

    pub fn pop_prompt(&mut self, prompt: PromptId) {
        assert!(
            prompt.index() < self.next_prompt as usize,
            "invalid prompt id"
        );
        self.push_void_inst(Inst::PopPrompt(prompt));
    }

    pub fn capture_slice(&mut self, prompt: PromptId, live: &[Value]) -> Value {
        assert!(
            prompt.index() < self.next_prompt as usize,
            "invalid prompt id"
        );
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

    // ── Exception handlers ─────────────────────────────────────
    //
    // The same-shape sibling of `push_prompt`/`pop_prompt`, but for the
    // exception channel instead of abort-to-prompt control flow. See the
    // doc comments on `Inst::PushHandler` / `Terminator::Raise` for the
    // semantics (LLVM invoke/landingpad cost model).

    /// Install an exception handler. Any `raise(v)` in this push's
    /// static scope lowers to a local jump to `handler_block` with `v`
    /// as the handler's first block param. Plain `Call`s in scope are
    /// implicitly invokes against this handler too.
    ///
    /// `handler_block` must have exactly one I64 block param.
    pub fn push_handler(&mut self, handler_block: BlockId) {
        self.push_void_inst(Inst::PushHandler(handler_block));
    }

    /// Pop the most-recently-pushed exception handler.
    pub fn pop_handler(&mut self) {
        self.push_void_inst(Inst::PopHandler);
    }

    /// Raise an exception. Lowers to a jump to the nearest active
    /// `push_handler` block (with `value` as its first block param), or
    /// — if no handler is in scope — to a return with outcome kind set
    /// to `Exception` and `value` as payload.
    pub fn raise(&mut self, value: Value) {
        self.set_terminator(Terminator::Raise(value));
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

    /// Indirect call resolved at runtime through a JIT call table.
    ///
    /// Treats the table as `[u64; N]` of code pointers indexed by
    /// FuncRef value. Emits the canonical lookup sequence
    /// `code_ptr = *(table_base + fr * 8)` and a `call_indirect`.
    ///
    /// `table_base` is typically `JitModule::call_table_base_addr()`
    /// — baked in as a constant at codegen time.
    /// `fr_value` is a runtime I64 holding the FuncRef index.
    ///
    /// Frontends that store FuncRefs inside their own heap shapes
    /// should wrap this with a one-liner that loads the FuncRef from
    /// the receiver and forwards everything else through.
    pub fn call_via_func_ref(
        &mut self,
        table_base: u64,
        fr_value: Value,
        args: &[Value],
        ret_ty: Option<Type>,
    ) -> Option<Value> {
        let base = self.iconst(Type::I64, table_base as i64);
        let three = self.iconst(Type::I64, 3);
        let off = self.shl(fr_value, three);
        let addr = self.add(base, off);
        let code_ptr = self.load(Type::I64, addr, 0);
        self.call_indirect(code_ptr, args, ret_ty)
    }

    /// Emit an inline-cached dynamic dispatch.
    ///
    /// Dispatches `symbol` on `receiver` with the given `args`.
    /// Returns the result value (I64, NaN-boxed).
    /// `cache_id` is the index into the module's InlineCacheArray.
    pub fn invoke_dynamic(
        &mut self,
        receiver: Value,
        symbol: dynsym::Symbol,
        args: &[Value],
        cache_id: u32,
    ) -> Value {
        let inst = Inst::InvokeDynamic {
            receiver,
            symbol,
            args: args.to_vec(),
            cache_id,
        };
        self.push_inst(Type::I64, inst)
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
        // Exception block: first param (if any) is the thrown value
        // (always I64), rest must match exception_args.
        self.check_invoke_exception_args(exception, exception_args);
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
        self.check_invoke_exception_args(exception, exception_args);
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

    pub fn resume_slice(
        &mut self,
        slice: Value,
        args: &[Value],
        return_block: BlockId,
        return_args: &[Value],
    ) {
        assert_eq!(
            self.value_type(slice),
            Type::FrameSlice,
            "resume_slice requires frameslice, got {}",
            self.value_type(slice)
        );
        self.set_terminator(Terminator::ResumeSlice {
            slice,
            args: args.to_vec(),
            return_block,
            return_args: return_args.to_vec(),
        });
    }

    /// Emit a CaptureSlice terminator. `handler_block` must have exactly
    /// one FrameSlice param (the handle), and `resume_block` must have
    /// exactly one I64 param (the resume value).
    pub fn capture_slice_term(
        &mut self,
        prompt: PromptId,
        handler_block: BlockId,
        resume_block: BlockId,
    ) {
        assert!(
            prompt.index() < self.next_prompt as usize,
            "invalid prompt id"
        );
        self.set_terminator(Terminator::CaptureSlice {
            prompt,
            handler_block,
            resume_block,
        });
    }

    pub fn abort_to_prompt(&mut self, prompt: PromptId, args: &[Value]) {
        assert!(
            prompt.index() < self.next_prompt as usize,
            "invalid prompt id"
        );
        self.set_terminator(Terminator::AbortToPrompt {
            prompt,
            args: args.to_vec(),
        });
    }

    // ── Exceptions ─────────────────────────────────────────────
    //
    // Exceptions are a degenerate use of the prompt machinery: a try
    // block pushes a prompt, the matching throw aborts to it, the
    // handler block runs. None of the multi-shot continuation ops
    // (`capture_slice`, `clone_slice`, `resume_slice`) are involved.
    //
    // Lowering cost (per `crates/dynlower/src/lib.rs`):
    //   - `push_prompt`/`pop_prompt` emit ZERO machine instructions —
    //     they only update the lowerer's compile-time active-prompt
    //     stack.
    //   - `abort_to_prompt` emits a static jump-to-runtime carrying a
    //     precomputed record_idx; no runtime stack walk.
    // This is the LLVM `invoke`/`landingpad` cost model — zero
    // overhead on the happy path, cost paid only at throw sites.

    /// Open a try region for exception handling. Allocates a fresh
    /// prompt id, pushes it with `handler_block` as the catch target,
    /// and returns the id so the caller can close the region with
    /// `pop_prompt(id)` and target it from `throw_to(id, value)`.
    ///
    /// `handler_block` must take exactly one I64 block parameter
    /// (the thrown value). Equivalent to:
    /// ```ignore
    /// let p = fb.create_prompt();
    /// fb.push_prompt(p, handler_block);
    /// ```
    pub fn try_scope(&mut self, handler_block: BlockId) -> PromptId {
        let p = self.create_prompt();
        self.push_prompt(p, handler_block);
        p
    }

    /// Throw `value` to the given try-scope prompt. The handler block
    /// associated with `prompt` (from `try_scope` / `push_prompt`)
    /// receives `value` as its single block parameter.
    ///
    /// Aliases `abort_to_prompt(prompt, &[value])` with single-value
    /// semantics. If no enclosing prompt with this id is on the
    /// active stack at runtime, the JIT exits with
    /// `JitOutcome::AbortToPrompt` and the surrounding runtime is
    /// expected to propagate or convert it to `JitOutcome::Exception`.
    pub fn throw_to(&mut self, prompt: PromptId, value: Value) {
        self.abort_to_prompt(prompt, &[value]);
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

    /// For invoke: if the exception block has any params, the first
    /// one is implicitly the runtime exception value (always I64).
    /// Remaining params match `exception_args`.
    fn check_invoke_exception_args(&self, target: BlockId, args: &[Value]) {
        let params = &self.blocks[target.index()].params;
        let exc_param_count = if params.is_empty() { 0 } else { 1 };
        let expected_args = params.len() - exc_param_count;
        assert_eq!(
            expected_args,
            args.len(),
            "invoke exception branch to bb{} arg count mismatch: \
             expected {expected_args} (block has {} params, {} is exception value), got {}",
            target.index(),
            params.len(),
            exc_param_count,
            args.len()
        );
        if exc_param_count == 1 {
            assert_eq!(
                params[0].1,
                Type::I64,
                "invoke exception bb{} first param must be I64 (the thrown value), got {}",
                target.index(),
                params[0].1
            );
        }
        for (i, (&(_, pty), &arg)) in params[exc_param_count..]
            .iter()
            .zip(args.iter())
            .enumerate()
        {
            let at = self.value_type(arg);
            assert_eq!(
                at,
                pty,
                "invoke exception bb{} arg {} type mismatch: expected {pty}, got {at}",
                target.index(),
                i + exc_param_count
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

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    fn define_const(mb: &mut ModuleBuilder, fref: FuncRef, val: i64) {
        let mut fb = mb.define_func(fref);
        let v = fb.iconst(Type::I64, val);
        fb.ret(v);
        mb.finish_func(fref, fb);
    }

    /// A function declared after a checkpoint but never defined would make
    /// `snapshot()` panic ("declared but not defined"). `rollback` must
    /// discard it so the builder snapshots cleanly — this is the property
    /// that keeps a persistent builder usable after a failed compile.
    #[test]
    fn rollback_discards_undefined_func() {
        let mut mb = ModuleBuilder::new();
        let f0 = mb.declare_func("good", &[], Some(Type::I64));
        define_const(&mut mb, f0, 1);

        let cp = mb.checkpoint();
        // Simulate a form that declares a function and then aborts before
        // defining it (leaving a half-built entry in the builder).
        let _f1 = mb.declare_func("half_built", &[], Some(Type::I64));
        assert_eq!(mb.func_count(), 2);

        mb.rollback(cp);
        assert_eq!(mb.func_count(), 1, "rolled-back func must be gone");

        // Snapshot would have panicked on the undefined `half_built`; after
        // rollback it succeeds and contains only the good function.
        let module = mb.snapshot();
        assert_eq!(module.functions.len(), 1);
    }

    /// After a rollback the builder must remain extendable: new declarations
    /// reuse the freed FuncRef indices and snapshot cleanly.
    #[test]
    fn builder_reusable_after_rollback() {
        let mut mb = ModuleBuilder::new();
        let f0 = mb.declare_func("a", &[], Some(Type::I64));
        define_const(&mut mb, f0, 10);

        let cp = mb.checkpoint();
        let _bad = mb.declare_func("bad", &[], Some(Type::I64));
        mb.rollback(cp);

        // Re-declare at the freed index and define it.
        let f1 = mb.declare_func("b", &[], Some(Type::I64));
        define_const(&mut mb, f1, 20);

        let module = mb.snapshot();
        assert_eq!(module.functions.len(), 2);
    }
}
