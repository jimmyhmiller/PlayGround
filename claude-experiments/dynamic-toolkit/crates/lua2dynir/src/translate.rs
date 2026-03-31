/// Lua 5.1 bytecode → DynIR translation.
///
/// Lua registers are stored in alloca'd stack slots. Each register R[i] has a
/// corresponding slot; reads and writes go through load_reg / store_reg.
/// All values are I64 (NanBox-encoded u64).
///
/// Callable signature:
/// `fn(closure, arg0..argN, vararg_count, vararg_ptr) -> I64`
///
/// The entry block allocates slots, stores ABI parameters, and jumps to the
/// first bytecode block.
use std::collections::{BTreeSet, HashMap};

use dynir::builder::FunctionBuilder;
use dynir::ir::{BlockId, CmpOp, FuncRef, Function, StackSlot, Value};
use dynir::types::{Signature, Type};

use crate::bytecode::*;

enum FloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Translate a Lua 5.1 prototype into a DynIR function.
pub fn translate(proto: &Proto) -> TranslatedFunction {
    let mut t = Translator::new(proto);
    t.translate();
    t.finish()
}

/// Result of translation: the IR function plus metadata needed for runtime binding.
pub struct TranslatedFunction {
    pub function: Function,
    pub extern_names: Vec<String>,
}

struct Translator<'a> {
    proto: &'a Proto,
    builder: FunctionBuilder,
    num_regs: usize,
    num_params: usize,
    abi_entry: BlockId,

    // Extern function refs
    extern_names: Vec<String>,

    // Block mapping: bytecode PC → DynIR block
    pc_to_block: HashMap<usize, BlockId>,

    // Which PCs are block leaders
    block_leaders: BTreeSet<usize>,

    // Stack slots for Lua registers and metadata (declared in new, addressed in emit_abi_entry)
    reg_slots: Vec<StackSlot>,      // reg_slots[i] = stack slot for R[i]
    closure_slot: StackSlot,        // stack slot for closure
    vararg_count_slot: StackSlot,   // stack slot for vararg_count
    vararg_ptr_slot: StackSlot,     // stack slot for vararg_ptr
}

impl<'a> Translator<'a> {
    fn new(proto: &'a Proto) -> Self {
        let num_regs = proto.max_stack_size as usize;
        let num_params = proto.num_params as usize;
        let mut param_types: Vec<Type> = Vec::with_capacity(num_params + 3);
        param_types.push(Type::I64); // closure
        param_types.extend((0..num_params).map(|_| Type::I64));
        param_types.push(Type::I64); // vararg_count
        param_types.push(Type::Ptr); // vararg_ptr
        let mut builder = FunctionBuilder::new("lua_func", &param_types, Some(Type::I64));

        let abi_entry = builder.entry_block();

        // The first bytecode block (PC 0) is a separate block — the entry
        // block stores ABI params into slots and jumps to it.
        let exec_entry = builder.create_block(&[]);

        let mut pc_to_block = HashMap::new();
        pc_to_block.insert(0, exec_entry);

        let mut block_leaders = BTreeSet::new();
        block_leaders.insert(0);

        // Declare function-level stack slots for Lua registers and metadata.
        // All slots are GC roots because NanBox values may be GC pointers.
        let closure_slot = builder.create_stack_slot(8, true);
        let vararg_count_slot = builder.create_stack_slot(8, true);
        let vararg_ptr_slot = builder.create_stack_slot(8, false); // Ptr, not a GC value
        let reg_slots: Vec<StackSlot> = (0..num_regs)
            .map(|_| builder.create_stack_slot(8, true))
            .collect();

        Translator {
            proto,
            builder,
            num_regs,
            num_params,
            abi_entry,
            extern_names: Vec::new(),
            pc_to_block,
            block_leaders,
            reg_slots,
            closure_slot,
            vararg_count_slot,
            vararg_ptr_slot,
        }
    }

    fn declare_extern(&mut self, name: &str, params: &[Type], ret: Option<Type>) -> FuncRef {
        let fref = self.builder.declare_func(
            name,
            Signature {
                params: params.to_vec(),
                ret,
            },
        );
        self.extern_names.push(name.to_string());
        fref
    }

    fn translate(&mut self) {
        // Phase 1: Find all block leaders (jump targets)
        self.find_block_leaders();

        // Phase 2: Create all blocks
        self.create_blocks();

        // Phase 3: Lower the callable ABI into register-threaded internal state.
        self.emit_abi_entry();

        // Phase 4: Declare extern functions
        let lua_add = self.declare_extern("lua_add", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_sub = self.declare_extern("lua_sub", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_mul = self.declare_extern("lua_mul", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_div = self.declare_extern("lua_div", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_mod = self.declare_extern("lua_mod", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_pow = self.declare_extern("lua_pow", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_unm = self.declare_extern("lua_unm", &[Type::I64], Some(Type::I64));
        let lua_not = self.declare_extern("lua_not", &[Type::I64], Some(Type::I64));
        let lua_len = self.declare_extern("lua_len", &[Type::I64], Some(Type::I64));
        let lua_eq = self.declare_extern("lua_eq", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_lt = self.declare_extern("lua_lt", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_le = self.declare_extern("lua_le", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_concat =
            self.declare_extern("lua_concat", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_getglobal = self.declare_extern("lua_getglobal", &[Type::I64], Some(Type::I64));
        let lua_setglobal = self.declare_extern("lua_setglobal", &[Type::I64, Type::I64], None);
        let lua_newtable = self.declare_extern("lua_newtable", &[], Some(Type::I64));
        let lua_gettable =
            self.declare_extern("lua_gettable", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_settable =
            self.declare_extern("lua_settable", &[Type::I64, Type::I64, Type::I64], None);
        let lua_call = self.declare_extern(
            "lua_call",
            &[Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_setlist = self.declare_extern(
            "lua_setlist",
            &[Type::I64, Type::I64, Type::I64, Type::I64],
            None,
        );
        let lua_forprep = self.declare_extern(
            "lua_forprep",
            &[Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_forloop = self.declare_extern(
            "lua_forloop",
            &[Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_is_nil = self.declare_extern("lua_is_nil", &[Type::I64], Some(Type::I64));
        let lua_self = self.declare_extern("lua_self", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_store_reg = self.declare_extern("lua_store_reg", &[Type::I64, Type::I64], None);
        // Closure creation externs for 0-4 upvalues
        let lua_make_closure_0 =
            self.declare_extern("lua_make_closure", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_make_closure_1 = self.declare_extern(
            "lua_make_closure_1",
            &[Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_make_closure_2 = self.declare_extern(
            "lua_make_closure_2",
            &[Type::I64, Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_make_closure_3 = self.declare_extern(
            "lua_make_closure_3",
            &[Type::I64, Type::I64, Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_make_closure_4 = self.declare_extern(
            "lua_make_closure_4",
            &[
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
                Type::I64,
            ],
            Some(Type::I64),
        );

        let externs = ExternFuncs {
            lua_add,
            lua_sub,
            lua_mul,
            lua_div,
            lua_mod,
            lua_pow,
            lua_unm,
            lua_not,
            lua_len,
            lua_eq,
            lua_lt,
            lua_le,
            lua_concat,
            lua_getglobal,
            lua_setglobal,
            lua_newtable,
            lua_gettable,
            lua_settable,
            lua_call,
            lua_setlist,
            lua_forprep,
            lua_forloop,
            lua_is_nil,
            lua_self,
            lua_store_reg,
            lua_make_closure_0,
            lua_make_closure_1,
            lua_make_closure_2,
            lua_make_closure_3,
            lua_make_closure_4,
        };

        // Phase 5: Emit IR for each block
        self.emit_blocks(&externs);
    }

    fn find_block_leaders(&mut self) {
        let code = &self.proto.code;
        for pc in 0..code.len() {
            let inst = code[pc];
            let op = OpCode::from_u8(opcode(inst));
            match op {
                Some(OpCode::Jmp) => {
                    let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                    self.block_leaders.insert(target);
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                }
                Some(OpCode::ForLoop) => {
                    let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                    self.block_leaders.insert(target);
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                }
                Some(OpCode::ForPrep) => {
                    let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                    self.block_leaders.insert(target);
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                }
                Some(OpCode::Eq | OpCode::Lt | OpCode::Le | OpCode::Test | OpCode::TestSet) => {
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                    if pc + 2 < code.len() {
                        self.block_leaders.insert(pc + 2);
                    }
                    if pc + 1 < code.len() {
                        let jmp_inst = code[pc + 1];
                        if OpCode::from_u8(opcode(jmp_inst)) == Some(OpCode::Jmp) {
                            let target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                            self.block_leaders.insert(target);
                        }
                    }
                }
                Some(OpCode::LoadBool) => {
                    let c = field_c(inst);
                    if c != 0 && pc + 2 < code.len() {
                        self.block_leaders.insert(pc + 2);
                    }
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                }
                Some(OpCode::Return) => {
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                }
                Some(OpCode::TForLoop) => {
                    if pc + 1 < code.len() {
                        self.block_leaders.insert(pc + 1);
                    }
                    if pc + 2 < code.len() {
                        self.block_leaders.insert(pc + 2);
                    }
                    if pc + 1 < code.len() {
                        let jmp_inst = code[pc + 1];
                        if OpCode::from_u8(opcode(jmp_inst)) == Some(OpCode::Jmp) {
                            let target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                            self.block_leaders.insert(target);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn create_blocks(&mut self) {
        // Blocks have zero parameters — registers live in alloca'd slots.
        for &pc in &self.block_leaders.clone() {
            if pc == 0 {
                continue; // entry block already created
            }
            let block = self.builder.create_block(&[]);
            self.pc_to_block.insert(pc, block);
        }
    }

    fn abi_closure(&self) -> Value {
        self.builder.block_param(self.abi_entry, 0)
    }

    fn abi_arg(&self, idx: usize) -> Value {
        self.builder.block_param(self.abi_entry, idx + 1)
    }

    fn abi_vararg_count(&self) -> Value {
        self.builder
            .block_param(self.abi_entry, self.num_params + 1)
    }

    fn abi_vararg_ptr(&self) -> Value {
        self.builder
            .block_param(self.abi_entry, self.num_params + 2)
    }

    /// Load the current value of Lua register R[i] from its stack slot.
    fn load_reg(&mut self, i: usize) -> Value {
        let addr = self.builder.stack_addr(self.reg_slots[i]);
        self.builder.load(Type::I64, addr, 0)
    }

    /// Store a value into Lua register R[i].
    fn store_reg(&mut self, i: usize, val: Value) {
        let addr = self.builder.stack_addr(self.reg_slots[i]);
        self.builder.store(val, addr, 0);
    }

    /// Load closure from its slot.
    fn load_closure(&mut self) -> Value {
        let addr = self.builder.stack_addr(self.closure_slot);
        self.builder.load(Type::I64, addr, 0)
    }

    /// Load vararg_count from its slot.
    fn load_vararg_count(&mut self) -> Value {
        let addr = self.builder.stack_addr(self.vararg_count_slot);
        self.builder.load(Type::I64, addr, 0)
    }

    /// Load vararg_ptr from its slot.
    fn load_vararg_ptr(&mut self) -> Value {
        let addr = self.builder.stack_addr(self.vararg_ptr_slot);
        self.builder.load(Type::Ptr, addr, 0)
    }

    fn emit_abi_entry(&mut self) {
        // Store ABI params into their stack slots
        let closure = self.abi_closure();
        let closure_addr = self.builder.stack_addr(self.closure_slot);
        self.builder.store(closure, closure_addr, 0);

        let vararg_count = self.abi_vararg_count();
        let vc_addr = self.builder.stack_addr(self.vararg_count_slot);
        self.builder.store(vararg_count, vc_addr, 0);

        let vararg_ptr = self.abi_vararg_ptr();
        let vp_addr = self.builder.stack_addr(self.vararg_ptr_slot);
        self.builder.store(vararg_ptr, vp_addr, 0);

        // Initialize registers: params get ABI values, rest get nil
        let nil = self.nil_const();
        for i in 0..self.num_regs {
            let addr = self.builder.stack_addr(self.reg_slots[i]);
            if i < self.num_params {
                let arg = self.abi_arg(i);
                self.builder.store(arg, addr, 0);
            } else {
                self.builder.store(nil, addr, 0);
            }
        }

        // Jump to first bytecode block
        let entry_block = self.pc_to_block[&0];
        self.builder.jump(entry_block, &[]);
    }

    /// Make a NanBox-encoded nil constant.
    fn nil_const(&mut self) -> Value {
        self.builder
            .iconst(Type::I64, 0x7FFD_0000_0000_0000u64 as i64)
    }

    /// Make a NanBox-encoded boolean constant.
    fn bool_const(&mut self, val: bool) -> Value {
        let bits = 0x7FFE_0000_0000_0000u64 | (val as u64);
        self.builder.iconst(Type::I64, bits as i64)
    }

    /// Make a NanBox-encoded number constant.
    fn number_const(&mut self, val: f64) -> Value {
        self.builder.iconst(Type::I64, val.to_bits() as i64)
    }

    /// Load a constant from the proto's constant table.
    fn load_constant(&mut self, idx: usize) -> Value {
        match &self.proto.constants[idx] {
            Constant::Nil => self.nil_const(),
            Constant::Bool(b) => self.bool_const(*b),
            Constant::Number(n) => self.number_const(*n),
            Constant::String(_) => {
                let bits = 0x7FFF_0000_0000_0000u64 | (idx as u64);
                self.builder.iconst(Type::I64, bits as i64)
            }
        }
    }

    /// Get the RK value: if bit 8 is set, load constant; otherwise load from register slot.
    fn rk_value_from_slot(&mut self, field: u16) -> Value {
        if is_constant(field) {
            self.load_constant(constant_index(field))
        } else {
            self.load_reg(field as usize)
        }
    }

    /// Jump to a target block, passing closure, vararg metadata, and current register state.
    fn jump_to(&mut self, target_pc: usize) {
        let target_block = self.pc_to_block[&target_pc];
        self.builder.jump(target_block, &[]);
    }

    fn br_if_to(&mut self, cond: Value, then_pc: usize, else_pc: usize) {
        let then_block = self.pc_to_block[&then_pc];
        let else_block = self.pc_to_block[&else_pc];
        self.builder.br_if(cond, then_block, &[], else_block, &[]);
    }

    fn emit_blocks(&mut self, ext: &ExternFuncs) {
        let leaders: Vec<usize> = self.block_leaders.iter().copied().collect();
        let code = self.proto.code.clone();

        for (leader_idx, &start_pc) in leaders.iter().enumerate() {
            let block = self.pc_to_block[&start_pc];
            self.builder.switch_to_block(block);

            let end_pc = if leader_idx + 1 < leaders.len() {
                leaders[leader_idx + 1]
            } else {
                code.len()
            };

            let mut pc = start_pc;
            let mut terminated = false;
            // Track the "top" register for variable-return CALL (C=0).
            // After CALL A B 0, top = A + 1 (we always return 1 value).
            // When next CALL/TAILCALL has B=0, nargs = top - (A + 1).
            let mut var_return_top: Option<usize> = None;

            while pc < end_pc && !terminated {
                let inst = code[pc];
                let op = OpCode::from_u8(opcode(inst));

                match op {
                    Some(OpCode::Move) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let val = self.load_reg(b);
                        self.store_reg(a, val);
                    }
                    Some(OpCode::LoadK) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        let val = self.load_constant(bx);
                        self.store_reg(a, val);
                    }
                    Some(OpCode::LoadBool) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);
                        let c = field_c(inst);
                        let val = self.bool_const(b != 0);
                        self.store_reg(a, val);
                        if c != 0 {
                            let target = pc + 2;
                            self.jump_to(target);
                            terminated = true;
                        }
                    }
                    Some(OpCode::LoadNil) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let nil = self.nil_const();
                        for i in a..=b {
                            self.store_reg(i, nil);
                        }
                    }
                    Some(OpCode::Add) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.emit_float_binop(b, c, FloatBinOp::Add, ext.lua_add);
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Sub) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.emit_float_binop(b, c, FloatBinOp::Sub, ext.lua_sub);
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Mul) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.emit_float_binop(b, c, FloatBinOp::Mul, ext.lua_mul);
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Div) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.emit_float_binop(b, c, FloatBinOp::Div, ext.lua_div);
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Mod) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.builder.call(ext.lua_mod, &[b, c]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Pow) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.builder.call(ext.lua_pow, &[b, c]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Unm) => {
                        let a = field_a(inst) as usize;
                        let b = self.load_reg(field_b(inst) as usize);
                        let result = self.builder.call(ext.lua_unm, &[b]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Not) => {
                        let a = field_a(inst) as usize;
                        let b = self.load_reg(field_b(inst) as usize);
                        let result = self.builder.call(ext.lua_not, &[b]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Len) => {
                        let a = field_a(inst) as usize;
                        let b = self.load_reg(field_b(inst) as usize);
                        let result = self.builder.call(ext.lua_len, &[b]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Concat) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let c = field_c(inst) as usize;
                        let mut result = self.load_reg(b);
                        for i in (b + 1)..=c {
                            let reg_i = self.load_reg(i);
                            result = self
                                .builder
                                .call(ext.lua_concat, &[result, reg_i])
                                .unwrap();
                        }
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Jmp) => {
                        let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                        // GC safepoint at backward jumps (loop back-edges)
                        if target <= pc {
                            self.builder.safepoint(&[]);
                        }
                        self.jump_to(target);
                        terminated = true;
                    }
                    Some(OpCode::Eq) => {
                        let a = field_a(inst);
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.builder.call(ext.lua_eq, &[b, c]).unwrap();
                        let true_val = self.bool_const(true);
                        let cmp = self.builder.icmp(CmpOp::Eq, result, true_val);

                        let cond = if a != 0 {
                            cmp
                        } else {
                            let one = self.builder.iconst(Type::I8, 1);
                            self.builder.xor(cmp, one)
                        };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;

                        self.br_if_to(cond, jmp_target, fall_through);
                        terminated = true;
                    }
                    Some(OpCode::Lt) => {
                        let a = field_a(inst);
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.builder.call(ext.lua_lt, &[b, c]).unwrap();
                        let true_val = self.bool_const(true);
                        let cmp = self.builder.icmp(CmpOp::Eq, result, true_val);

                        let cond = if a != 0 {
                            cmp
                        } else {
                            let one = self.builder.iconst(Type::I8, 1);
                            self.builder.xor(cmp, one)
                        };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;
                        self.br_if_to(cond, jmp_target, fall_through);
                        terminated = true;
                    }
                    Some(OpCode::Le) => {
                        let a = field_a(inst);
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.builder.call(ext.lua_le, &[b, c]).unwrap();
                        let true_val = self.bool_const(true);
                        let cmp = self.builder.icmp(CmpOp::Eq, result, true_val);

                        let cond = if a != 0 {
                            cmp
                        } else {
                            let one = self.builder.iconst(Type::I8, 1);
                            self.builder.xor(cmp, one)
                        };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;
                        self.br_if_to(cond, jmp_target, fall_through);
                        terminated = true;
                    }
                    Some(OpCode::Test) => {
                        let a = field_a(inst) as usize;
                        let c = field_c(inst);
                        let reg_a = self.load_reg(a);
                        let nil = self.nil_const();
                        let false_val = self.bool_const(false);
                        let is_nil = self.builder.icmp(CmpOp::Eq, reg_a, nil);
                        let is_false = self.builder.icmp(CmpOp::Eq, reg_a, false_val);
                        let is_falsy = self.builder.or(is_nil, is_false);

                        let one = self.builder.iconst(Type::I8, 1);
                        let is_truthy = self.builder.xor(is_falsy, one);

                        let cond = if c == 0 { is_falsy } else { is_truthy };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;
                        self.br_if_to(cond, jmp_target, fall_through);
                        terminated = true;
                    }
                    Some(OpCode::TestSet) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let c = field_c(inst);

                        let reg_b = self.load_reg(b);
                        let nil = self.nil_const();
                        let false_val = self.bool_const(false);
                        let is_nil = self.builder.icmp(CmpOp::Eq, reg_b, nil);
                        let is_false = self.builder.icmp(CmpOp::Eq, reg_b, false_val);
                        let is_falsy = self.builder.or(is_nil, is_false);
                        let one = self.builder.iconst(Type::I8, 1);
                        let is_truthy = self.builder.xor(is_falsy, one);

                        let cond = if c == 0 { is_falsy } else { is_truthy };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;

                        // TestSet: on the "taken" path, R(A) = R(B) before jumping.
                        // Create a helper block that stores R(A)=R(B) then jumps to jmp_target.
                        let store_block = self.builder.create_block(&[]);
                        let jmp_target_block = self.pc_to_block[&jmp_target];
                        let fall_through_block = self.pc_to_block[&fall_through];

                        self.builder.br_if(
                            cond,
                            store_block,
                            &[],
                            fall_through_block,
                            &[],
                        );

                        self.builder.switch_to_block(store_block);
                        self.store_reg(a, reg_b);
                        self.builder.jump(jmp_target_block, &[]);

                        terminated = true;
                    }
                    Some(OpCode::GetGlobal) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        let name_val = self.load_constant(bx);
                        let result = self.builder.call(ext.lua_getglobal, &[name_val]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::SetGlobal) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        let name_val = self.load_constant(bx);
                        let reg_a = self.load_reg(a);
                        self.builder.call(ext.lua_setglobal, &[name_val, reg_a]);
                    }
                    Some(OpCode::NewTable) => {
                        let a = field_a(inst) as usize;
                        let result = self.builder.call(ext.lua_newtable, &[]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::GetTable) => {
                        let a = field_a(inst) as usize;
                        let b = self.load_reg(field_b(inst) as usize);
                        let c = self.rk_value_from_slot(field_c(inst));
                        let result = self.builder.call(ext.lua_gettable, &[b, c]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::SetTable) => {
                        let a = field_a(inst) as usize;
                        let reg_a = self.load_reg(a);
                        let b = self.rk_value_from_slot(field_b(inst));
                        let c = self.rk_value_from_slot(field_c(inst));
                        self.builder.call(ext.lua_settable, &[reg_a, b, c]);
                    }
                    Some(OpCode::Self_) => {
                        let a = field_a(inst) as usize;
                        let b = self.load_reg(field_b(inst) as usize);
                        let c = self.rk_value_from_slot(field_c(inst));
                        // R(A+1) = R(B) (the object), then R(A) = R(B)[RK(C)]
                        self.store_reg(a + 1, b);
                        let result = self.builder.call(ext.lua_gettable, &[b, c]).unwrap();
                        self.store_reg(a, result);
                    }
                    Some(OpCode::Call) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);
                        let c = field_c(inst);
                        let nargs = if b == 0 {
                            // Variable args: from R(A+1) to top (set by previous C=0 call)
                            var_return_top.unwrap_or(a + 1).saturating_sub(a + 1)
                        } else {
                            (b - 1) as usize
                        };

                        // Store argument registers into the runtime's register file
                        for i in 0..nargs {
                            let idx_val = self.builder.iconst(Type::I64, i as i64);
                            let arg = self.load_reg(a + 1 + i);
                            self.builder.call(ext.lua_store_reg, &[idx_val, arg]);
                        }

                        let func_val = self.load_reg(a);
                        let base_idx = self.builder.iconst(Type::I64, 0i64);
                        let nargs_val = self.builder.iconst(Type::I64, nargs as i64);
                        let result = self
                            .builder
                            .call(ext.lua_call, &[func_val, base_idx, nargs_val])
                            .unwrap();

                        if c == 0 {
                            // Variable returns: top = A + 1 (we always return 1 value)
                            self.store_reg(a, result);
                            var_return_top = Some(a + 1);
                        } else if c != 1 {
                            self.store_reg(a, result);
                            var_return_top = None;
                        } else {
                            var_return_top = None;
                        }
                    }
                    Some(OpCode::TailCall) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);
                        let nargs = if b == 0 {
                            var_return_top.unwrap_or(a + 1).saturating_sub(a + 1)
                        } else {
                            (b - 1) as usize
                        };

                        for i in 0..nargs {
                            let idx_val = self.builder.iconst(Type::I64, i as i64);
                            let arg = self.load_reg(a + 1 + i);
                            self.builder.call(ext.lua_store_reg, &[idx_val, arg]);
                        }

                        let func_val = self.load_reg(a);
                        let base_idx = self.builder.iconst(Type::I64, 0i64);
                        let nargs_val = self.builder.iconst(Type::I64, nargs as i64);
                        let result = self
                            .builder
                            .call(ext.lua_call, &[func_val, base_idx, nargs_val])
                            .unwrap();
                        self.builder.ret(result);
                        terminated = true;
                    }
                    Some(OpCode::Return) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);

                        if b == 1 {
                            let nil = self.nil_const();
                            self.builder.ret(nil);
                        } else {
                            let reg_a = self.load_reg(a);
                            self.builder.ret(reg_a);
                        }
                        terminated = true;
                    }
                    Some(OpCode::ForPrep) => {
                        let a = field_a(inst) as usize;
                        let target = (pc as i32 + 1 + field_sbx(inst)) as usize;

                        let reg_a = self.load_reg(a);
                        let reg_a1 = self.load_reg(a + 1);
                        let reg_a2 = self.load_reg(a + 2);
                        let result = self
                            .builder
                            .call(ext.lua_forprep, &[reg_a, reg_a1, reg_a2])
                            .unwrap();
                        self.store_reg(a, result);
                        self.jump_to(target);
                        terminated = true;
                    }
                    Some(OpCode::ForLoop) => {
                        let a = field_a(inst) as usize;
                        let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                        let fall_through = pc + 1;

                        // GC safepoint at loop back-edge
                        self.builder.safepoint(&[]);

                        let reg_a = self.load_reg(a);
                        let reg_a1 = self.load_reg(a + 1);
                        let reg_a2 = self.load_reg(a + 2);
                        let new_idx = self
                            .builder
                            .call(ext.lua_forloop, &[reg_a, reg_a1, reg_a2])
                            .unwrap();

                        let nil = self.nil_const();
                        let is_nil = self.builder.icmp(CmpOp::Eq, new_idx, nil);

                        // If nil (loop done), fall through without modifying regs.
                        // If not nil (loop continues), store new index and jump to target.
                        let continue_block = self.builder.create_block(&[]);
                        let exit_block = self.pc_to_block[&fall_through];

                        self.builder
                            .br_if(is_nil, exit_block, &[], continue_block, &[]);

                        self.builder.switch_to_block(continue_block);
                        self.store_reg(a, new_idx);
                        self.store_reg(a + 3, new_idx);
                        self.jump_to(target);

                        terminated = true;
                    }
                    Some(OpCode::SetList) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let c = field_c(inst) as usize;

                        let count = if b == 0 { 0usize } else { b };
                        let offset = ((c - 1) * 50) as u64;

                        // Store the values into register_file so the runtime can read them
                        for i in 0..count {
                            let idx_val = self.builder.iconst(Type::I64, (a + 1 + i) as i64);
                            let val = self.load_reg(a + 1 + i);
                            self.builder.call(ext.lua_store_reg, &[idx_val, val]);
                        }

                        let table = self.load_reg(a);
                        let base_idx = self.builder.iconst(Type::I64, (a + 1) as i64);
                        let offset_val = self.builder.iconst(Type::I64, offset as i64);
                        let count_val = self.builder.iconst(Type::I64, count as i64);

                        self.builder
                            .call(ext.lua_setlist, &[table, base_idx, offset_val, count_val]);
                    }
                    Some(OpCode::Closure) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        let num_upvalues = self.proto.protos[bx].num_upvalues as usize;

                        // Read pseudo-instructions for upvalue capture
                        let mut upval_values = Vec::new();
                        let mut self_ref_indices: Vec<usize> = Vec::new();

                        for j in 0..num_upvalues {
                            let pseudo_pc = pc + 1 + j;
                            let pseudo_inst = code[pseudo_pc];
                            let pseudo_op = opcode(pseudo_inst);
                            let pseudo_b = field_b(pseudo_inst) as usize;

                            if pseudo_op == OpCode::Move as u8 {
                                // Capture register B of current function
                                let val = self.load_reg(pseudo_b);
                                upval_values.push(val);
                                if pseudo_b == a {
                                    self_ref_indices.push(j);
                                }
                            } else if pseudo_op == OpCode::GetUpval as u8 {
                                // Capture upvalue B from enclosing closure
                                let closure = self.load_closure();
                                let raw_ptr = self.builder.payload(closure);
                                let upval = self.builder.load(
                                    Type::I64,
                                    raw_ptr,
                                    (24 + pseudo_b * 8) as i32,
                                );
                                upval_values.push(upval);
                            } else {
                                upval_values.push(self.nil_const());
                            }
                        }

                        let proto_idx_val = self.builder.iconst(Type::I64, bx as i64);
                        let num_upval_val = self.builder.iconst(Type::I64, num_upvalues as i64);

                        let closure_val = match num_upvalues {
                            0 => self
                                .builder
                                .call(ext.lua_make_closure_0, &[proto_idx_val, num_upval_val])
                                .unwrap(),
                            1 => self
                                .builder
                                .call(
                                    ext.lua_make_closure_1,
                                    &[proto_idx_val, num_upval_val, upval_values[0]],
                                )
                                .unwrap(),
                            2 => self
                                .builder
                                .call(
                                    ext.lua_make_closure_2,
                                    &[
                                        proto_idx_val,
                                        num_upval_val,
                                        upval_values[0],
                                        upval_values[1],
                                    ],
                                )
                                .unwrap(),
                            3 => self
                                .builder
                                .call(
                                    ext.lua_make_closure_3,
                                    &[
                                        proto_idx_val,
                                        num_upval_val,
                                        upval_values[0],
                                        upval_values[1],
                                        upval_values[2],
                                    ],
                                )
                                .unwrap(),
                            4 => self
                                .builder
                                .call(
                                    ext.lua_make_closure_4,
                                    &[
                                        proto_idx_val,
                                        num_upval_val,
                                        upval_values[0],
                                        upval_values[1],
                                        upval_values[2],
                                        upval_values[3],
                                    ],
                                )
                                .unwrap(),
                            _ => {
                                panic!("closures with {} upvalues not yet supported", num_upvalues)
                            }
                        };

                        // Store closure into R(A) first (before fixing self-refs)
                        self.store_reg(a, closure_val);

                        // Skip the pseudo-instructions
                        pc += num_upvalues;

                        // Fix self-references: the closure was created with the OLD R(A),
                        // but Lua's VM sets R(A) first. Store the closure into its own upvalue slots.
                        for &j in &self_ref_indices {
                            let raw_ptr = self.builder.payload(closure_val);
                            self.builder.store(closure_val, raw_ptr, (24 + j * 8) as i32);
                        }
                    }
                    Some(OpCode::GetUpval) => {
                        // GETUPVAL A B: R(A) := UpValue[B]
                        // Load directly from the closure struct passed as param 0.
                        // Closure layout: [marker:u64, func_id:u64, num_upvals:u64, upval0, ...]
                        // Upvalue B is at offset 24 + B*8
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let closure = self.load_closure();
                        let raw_ptr = self.builder.payload(closure);
                        let val = self.builder.load(Type::I64, raw_ptr, (24 + b * 8) as i32);
                        self.store_reg(a, val);
                    }
                    Some(OpCode::SetUpval) => {
                        // SETUPVAL A B: UpValue[B] := R(A)
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let reg_a = self.load_reg(a);
                        let closure = self.load_closure();
                        let raw_ptr = self.builder.payload(closure);
                        self.builder.store(reg_a, raw_ptr, (24 + b * 8) as i32);
                    }
                    Some(OpCode::Close) => {
                        // Close upvalues - no-op for flat closures
                    }
                    Some(OpCode::VarArg) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);
                        if b >= 2 {
                            let nil = self.nil_const();
                            let vararg_count = self.load_vararg_count();
                            let vararg_ptr = self.load_vararg_ptr();
                            let vararg_slots = (b as usize) - 1;
                            for i in 0..vararg_slots {
                                let reg_idx = a + i;
                                if reg_idx < self.num_regs {
                                    let idx = self.builder.iconst(Type::I64, i as i64);
                                    let present =
                                        self.builder.icmp(CmpOp::Ult, idx, vararg_count);
                                    let loaded =
                                        self.builder.load(Type::I64, vararg_ptr, (i * 8) as i32);
                                    let val = self.builder.select(present, loaded, nil);
                                    self.store_reg(reg_idx, val);
                                }
                            }
                        }
                    }
                    Some(OpCode::TForLoop) => {
                        let a = field_a(inst) as usize;
                        let c = field_c(inst) as usize;

                        // Store args R(A+1) and R(A+2) to register_file for the call
                        let idx0 = self.builder.iconst(Type::I64, 0i64);
                        let reg_a1 = self.load_reg(a + 1);
                        self.builder.call(ext.lua_store_reg, &[idx0, reg_a1]);
                        let idx1 = self.builder.iconst(Type::I64, 1i64);
                        let reg_a2 = self.load_reg(a + 2);
                        self.builder.call(ext.lua_store_reg, &[idx1, reg_a2]);

                        let func_val = self.load_reg(a);
                        let base_idx = self.builder.iconst(Type::I64, 0i64);
                        let nargs_val = self.builder.iconst(Type::I64, 2i64);
                        let result = self
                            .builder
                            .call(ext.lua_call, &[func_val, base_idx, nargs_val])
                            .unwrap();
                        self.store_reg(a + 3, result);
                        if c > 1 {
                            // Additional return values would need support
                        }

                        let nil = self.nil_const();
                        let is_nil = self.builder.icmp(CmpOp::Eq, result, nil);

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;

                        // If not nil, set R(A+2) = R(A+3) and jump to loop body
                        let continue_block = self.builder.create_block(&[]);
                        let exit_block = self.pc_to_block[&fall_through];

                        self.builder
                            .br_if(is_nil, exit_block, &[], continue_block, &[]);

                        self.builder.switch_to_block(continue_block);
                        self.store_reg(a + 2, result);
                        self.jump_to(jmp_target);

                        terminated = true;
                    }
                    None => {
                        panic!("unknown opcode {} at pc {}", opcode(inst), pc);
                    }
                }

                pc += 1;

                // If we hit a comparison/test op, we consumed the next JMP too
                match op {
                    Some(
                        OpCode::Eq
                        | OpCode::Lt
                        | OpCode::Le
                        | OpCode::Test
                        | OpCode::TestSet
                        | OpCode::TForLoop,
                    ) => {
                        pc += 1;
                    }
                    _ => {}
                }
            }

            // If block wasn't terminated by a jump/return, fall through to next block
            if !terminated {
                if let Some(&next_pc) = leaders.get(leader_idx + 1) {
                    self.jump_to(next_pc);
                }
            }
        }
    }

    /// NanBox float detection: `(bits & 0xFFFC_0000_0000_0000) != 0x7FFC_0000_0000_0000`
    fn is_nanbox_float(&mut self, v: Value) -> Value {
        let mask = self.builder.iconst(Type::I64, 0xFFFC_0000_0000_0000u64 as i64);
        let pattern = self.builder.iconst(Type::I64, 0x7FFC_0000_0000_0000u64 as i64);
        let masked = self.builder.and(v, mask);
        self.builder.icmp(CmpOp::Ne, masked, pattern)
    }

    /// Emit inline NanBox binary float op with extern fallback.
    /// Uses branch to avoid the extern call overhead on the fast path.
    fn emit_float_binop(
        &mut self,
        a: Value,
        b: Value,
        op: FloatBinOp,
        fallback: FuncRef,
    ) -> Value {
        let a_is_float = self.is_nanbox_float(a);
        let b_is_float = self.is_nanbox_float(b);
        let both = self.builder.and(a_is_float, b_is_float);

        // Speculatively compute the float result (always safe — worst case
        // it's garbage that we discard). This avoids creating extra blocks
        // for the common case.
        let a_f64 = self.builder.bitcast(a, Type::F64);
        let b_f64 = self.builder.bitcast(b, Type::F64);
        let fast_f64 = match op {
            FloatBinOp::Add => self.builder.fadd(a_f64, b_f64),
            FloatBinOp::Sub => self.builder.fsub(a_f64, b_f64),
            FloatBinOp::Mul => self.builder.fmul(a_f64, b_f64),
            FloatBinOp::Div => self.builder.fdiv(a_f64, b_f64),
        };
        let fast_i64 = self.builder.bitcast(fast_f64, Type::I64);

        // For the slow path, we still need a branch to avoid the call
        let slow_block = self.builder.create_block(&[]);
        let merge_block = self.builder.create_block(&[Type::I64]);

        self.builder
            .br_if(both, merge_block, &[fast_i64], slow_block, &[]);

        self.builder.switch_to_block(slow_block);
        let slow_result = self.builder.call(fallback, &[a, b]).unwrap();
        self.builder.jump(merge_block, &[slow_result]);

        self.builder.switch_to_block(merge_block);
        self.builder.block_param(merge_block, 0)
    }

    /// Emit inline NanBox ForPrep: idx = init - step (Lua 5.1 semantics).
    fn emit_inline_forprep(
        &mut self,
        init: Value,
        limit: Value,
        step: Value,
        fallback: FuncRef,
    ) -> Value {
        let init_is_float = self.is_nanbox_float(init);
        let step_is_float = self.is_nanbox_float(step);
        let both = self.builder.and(init_is_float, step_is_float);

        let fast_block = self.builder.create_block(&[]);
        let slow_block = self.builder.create_block(&[]);
        let merge_block = self.builder.create_block(&[Type::I64]);

        self.builder
            .br_if(both, fast_block, &[], slow_block, &[]);

        self.builder.switch_to_block(fast_block);
        let init_f = self.builder.bitcast(init, Type::F64);
        let step_f = self.builder.bitcast(step, Type::F64);
        let result_f = self.builder.fsub(init_f, step_f);
        let result_i = self.builder.bitcast(result_f, Type::I64);
        self.builder.jump(merge_block, &[result_i]);

        self.builder.switch_to_block(slow_block);
        let slow = self
            .builder
            .call(fallback, &[init, limit, step])
            .unwrap();
        self.builder.jump(merge_block, &[slow]);

        self.builder.switch_to_block(merge_block);
        self.builder.block_param(merge_block, 0)
    }

    /// Emit inline NanBox ForLoop: idx += step, check against limit.
    /// Returns new idx if loop continues, or nil if done.
    fn emit_inline_forloop(
        &mut self,
        idx: Value,
        limit: Value,
        step: Value,
        fallback: FuncRef,
    ) -> Value {
        let idx_f = self.is_nanbox_float(idx);
        let lim_f = self.is_nanbox_float(limit);
        let step_f = self.is_nanbox_float(step);
        let tmp = self.builder.and(idx_f, lim_f);
        let all_float = self.builder.and(tmp, step_f);

        let fast_block = self.builder.create_block(&[]);
        let slow_block = self.builder.create_block(&[]);
        let merge_block = self.builder.create_block(&[Type::I64]);

        self.builder
            .br_if(all_float, fast_block, &[], slow_block, &[]);

        self.builder.switch_to_block(fast_block);
        let idx_fp = self.builder.bitcast(idx, Type::F64);
        let step_fp = self.builder.bitcast(step, Type::F64);
        let limit_fp = self.builder.bitcast(limit, Type::F64);
        let new_idx = self.builder.fadd(idx_fp, step_fp);
        let zero = self.builder.f64const(0.0);
        let step_positive = self.builder.fcmp(CmpOp::Sgt, step_fp, zero);

        // if step > 0: continue if new_idx <= limit
        // if step <= 0: continue if new_idx >= limit
        let le_limit = self.builder.fcmp(CmpOp::Sle, new_idx, limit_fp);
        let ge_limit = self.builder.fcmp(CmpOp::Sge, new_idx, limit_fp);
        let in_range = self.builder.select(step_positive, le_limit, ge_limit);

        let new_idx_i64 = self.builder.bitcast(new_idx, Type::I64);
        let nil = self.nil_const();

        let result = self.builder.select(in_range, new_idx_i64, nil);
        self.builder.jump(merge_block, &[result]);

        self.builder.switch_to_block(slow_block);
        let slow = self.builder.call(fallback, &[idx, limit, step]).unwrap();
        self.builder.jump(merge_block, &[slow]);

        self.builder.switch_to_block(merge_block);
        self.builder.block_param(merge_block, 0)
    }

    /// Emit inline GETTABLE with array fast path.
    fn emit_inline_gettable(
        &mut self,
        table: Value,
        key: Value,
        fallback: FuncRef,
    ) -> Value {
        // First check: key is a NanBox float (not a tagged value).
        // We must branch before doing float_to_int, because converting
        // a NaN (tagged value) to int is undefined on ARM64.
        let key_is_float = self.is_nanbox_float(key);

        let check_block = self.builder.create_block(&[]);
        let slow_block = self.builder.create_block(&[]);
        let fast_block = self.builder.create_block(&[]);
        let merge_block = self.builder.create_block(&[Type::I64]);

        self.builder
            .br_if(key_is_float, check_block, &[], slow_block, &[]);

        // Check block: key is a float — now safe to convert to int
        self.builder.switch_to_block(check_block);
        let key_f64 = self.builder.bitcast(key, Type::F64);
        let key_i64 = self.builder.float_to_int(key_f64);
        let key_back = self.builder.int_to_float(key_i64);
        let is_integral = self.builder.fcmp(CmpOp::Eq, key_f64, key_back);
        let one_i64 = self.builder.iconst(Type::I64, 1);
        let is_positive = self.builder.icmp(CmpOp::Sge, key_i64, one_i64);

        let table_raw = self.builder.payload(table);
        let array_len = self.builder.load(Type::I64, table_raw, 40);
        let in_bounds = self.builder.icmp(CmpOp::Ule, key_i64, array_len);

        let array_nanbox = self.builder.load(Type::I64, table_raw, 16);
        let array_payload = self.builder.payload(array_nanbox);
        let zero = self.builder.iconst(Type::I64, 0);
        let has_array = self.builder.icmp(CmpOp::Ne, array_payload, zero);

        let c1 = self.builder.and(is_integral, is_positive);
        let c2 = self.builder.and(c1, in_bounds);
        let all_ok = self.builder.and(c2, has_array);

        self.builder
            .br_if(all_ok, fast_block, &[], slow_block, &[]);

        // Fast path: load element from array
        self.builder.switch_to_block(fast_block);
        let idx_zero = self.builder.sub(key_i64, one_i64);
        let eight = self.builder.iconst(Type::I64, 8);
        let byte_offset = self.builder.mul(idx_zero, eight);
        let elem_addr = self.builder.add(array_payload, byte_offset);
        // Array elements start at offset 16 from array pointer
        let value = self.builder.load(Type::I64, elem_addr, 16);
        self.builder.jump(merge_block, &[value]);

        // Slow path: call extern
        self.builder.switch_to_block(slow_block);
        let slow = self.builder.call(fallback, &[table, key]).unwrap();
        self.builder.jump(merge_block, &[slow]);

        self.builder.switch_to_block(merge_block);
        self.builder.block_param(merge_block, 0)
    }

    /// Emit inline SETTABLE with array fast path.
    fn emit_inline_settable(
        &mut self,
        table: Value,
        key: Value,
        val: Value,
        fallback: FuncRef,
    ) {
        let key_is_float = self.is_nanbox_float(key);

        let check_block = self.builder.create_block(&[]);
        let slow_block = self.builder.create_block(&[]);
        let fast_block = self.builder.create_block(&[]);
        let merge_block = self.builder.create_block(&[]);

        self.builder
            .br_if(key_is_float, check_block, &[], slow_block, &[]);

        self.builder.switch_to_block(check_block);
        let key_f64 = self.builder.bitcast(key, Type::F64);
        let key_i64 = self.builder.float_to_int(key_f64);
        let key_back = self.builder.int_to_float(key_i64);
        let is_integral = self.builder.fcmp(CmpOp::Eq, key_f64, key_back);
        let one_i64 = self.builder.iconst(Type::I64, 1);
        let is_positive = self.builder.icmp(CmpOp::Sge, key_i64, one_i64);

        let table_raw = self.builder.payload(table);
        let array_len = self.builder.load(Type::I64, table_raw, 40);
        let in_bounds = self.builder.icmp(CmpOp::Ule, key_i64, array_len);

        let array_nanbox = self.builder.load(Type::I64, table_raw, 16);
        let array_payload = self.builder.payload(array_nanbox);
        let zero = self.builder.iconst(Type::I64, 0);
        let has_array = self.builder.icmp(CmpOp::Ne, array_payload, zero);

        let c1 = self.builder.and(is_integral, is_positive);
        let c2 = self.builder.and(c1, in_bounds);
        let all_ok = self.builder.and(c2, has_array);

        self.builder
            .br_if(all_ok, fast_block, &[], slow_block, &[]);

        // Fast path: store element into array
        self.builder.switch_to_block(fast_block);
        let idx_zero = self.builder.sub(key_i64, one_i64);
        let eight = self.builder.iconst(Type::I64, 8);
        let byte_offset = self.builder.mul(idx_zero, eight);
        let elem_addr = self.builder.add(array_payload, byte_offset);
        self.builder.store(val, elem_addr, 16);
        self.builder.jump(merge_block, &[]);

        // Slow path: call extern
        self.builder.switch_to_block(slow_block);
        self.builder.call(fallback, &[table, key, val]);
        self.builder.jump(merge_block, &[]);

        self.builder.switch_to_block(merge_block);
    }

    fn finish(self) -> TranslatedFunction {
        TranslatedFunction {
            function: self.builder.build(),
            extern_names: self.extern_names,
        }
    }
}

struct ExternFuncs {
    lua_add: FuncRef,
    lua_sub: FuncRef,
    lua_mul: FuncRef,
    lua_div: FuncRef,
    lua_mod: FuncRef,
    lua_pow: FuncRef,
    lua_unm: FuncRef,
    lua_not: FuncRef,
    lua_len: FuncRef,
    lua_eq: FuncRef,
    lua_lt: FuncRef,
    lua_le: FuncRef,
    lua_concat: FuncRef,
    lua_getglobal: FuncRef,
    lua_setglobal: FuncRef,
    lua_newtable: FuncRef,
    lua_gettable: FuncRef,
    lua_settable: FuncRef,
    lua_call: FuncRef,
    lua_setlist: FuncRef,
    lua_forprep: FuncRef,
    lua_forloop: FuncRef,
    #[allow(dead_code)]
    lua_is_nil: FuncRef,
    #[allow(dead_code)]
    lua_self: FuncRef,
    lua_store_reg: FuncRef,
    lua_make_closure_0: FuncRef,
    lua_make_closure_1: FuncRef,
    #[allow(dead_code)]
    lua_make_closure_2: FuncRef,
    #[allow(dead_code)]
    lua_make_closure_3: FuncRef,
    #[allow(dead_code)]
    lua_make_closure_4: FuncRef,
}
