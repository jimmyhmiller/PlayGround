/// Lua 5.1 bytecode → DynIR translation.
///
/// Each Lua register becomes a block parameter threaded through control flow.
/// All values are I64 (NanBox-encoded u64).
///
/// Function signature: fn(closure: I64, r0: I64, ..., rN: I64) -> I64
///   - closure (param 0) = the closure struct pointer (nil for main chunk)
///   - r0..rN = Lua registers (first num_params are arguments)

use std::collections::{BTreeSet, HashMap};

use dynir::builder::FunctionBuilder;
use dynir::ir::{BlockId, CmpOp, FuncRef, Function, Value};
use dynir::types::{Signature, Type};

use crate::bytecode::*;

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

    // Extern function refs
    extern_names: Vec<String>,

    // Block mapping: bytecode PC → DynIR block
    pc_to_block: HashMap<usize, BlockId>,

    // Which PCs are block leaders
    block_leaders: BTreeSet<usize>,
}

impl<'a> Translator<'a> {
    fn new(proto: &'a Proto) -> Self {
        let num_regs = proto.max_stack_size as usize;
        // closure + num_regs registers
        let param_types: Vec<Type> = vec![Type::I64; num_regs + 1];
        let builder = FunctionBuilder::new("lua_func", &param_types, Some(Type::I64));

        let entry = builder.entry_block();

        let mut pc_to_block = HashMap::new();
        pc_to_block.insert(0, entry);

        let mut block_leaders = BTreeSet::new();
        block_leaders.insert(0);

        Translator {
            proto,
            builder,
            num_regs,
            extern_names: Vec::new(),
            pc_to_block,
            block_leaders,
        }
    }

    fn declare_extern(&mut self, name: &str, params: &[Type], ret: Option<Type>) -> FuncRef {
        let fref = self.builder.declare_func(
            name,
            Signature { params: params.to_vec(), ret },
        );
        self.extern_names.push(name.to_string());
        fref
    }

    fn translate(&mut self) {
        // Phase 1: Find all block leaders (jump targets)
        self.find_block_leaders();

        // Phase 2: Create all blocks
        self.create_blocks();

        // Phase 3: Declare extern functions
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
        let lua_concat = self.declare_extern("lua_concat", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_getglobal = self.declare_extern("lua_getglobal", &[Type::I64], Some(Type::I64));
        let lua_setglobal = self.declare_extern("lua_setglobal", &[Type::I64, Type::I64], None);
        let lua_newtable = self.declare_extern("lua_newtable", &[], Some(Type::I64));
        let lua_gettable = self.declare_extern("lua_gettable", &[Type::I64, Type::I64], Some(Type::I64));
        let lua_settable = self.declare_extern("lua_settable", &[Type::I64, Type::I64, Type::I64], None);
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
        let lua_self = self.declare_extern(
            "lua_self",
            &[Type::I64, Type::I64],
            Some(Type::I64),
        );
        let lua_store_reg = self.declare_extern(
            "lua_store_reg",
            &[Type::I64, Type::I64],
            None,
        );
        // Closure creation externs for 0-4 upvalues
        let lua_make_closure_0 = self.declare_extern(
            "lua_make_closure",
            &[Type::I64, Type::I64],
            Some(Type::I64),
        );
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
            &[Type::I64, Type::I64, Type::I64, Type::I64, Type::I64, Type::I64],
            Some(Type::I64),
        );

        let externs = ExternFuncs {
            lua_add, lua_sub, lua_mul, lua_div, lua_mod, lua_pow,
            lua_unm, lua_not, lua_len, lua_eq, lua_lt, lua_le,
            lua_concat, lua_getglobal, lua_setglobal,
            lua_newtable, lua_gettable, lua_settable,
            lua_call, lua_setlist, lua_forprep, lua_forloop,
            lua_is_nil, lua_self, lua_store_reg,
            lua_make_closure_0, lua_make_closure_1,
            lua_make_closure_2, lua_make_closure_3, lua_make_closure_4,
        };

        // Phase 4: Emit IR for each block
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
        // All blocks have closure + num_regs parameters
        let reg_types: Vec<Type> = vec![Type::I64; self.num_regs + 1];
        for &pc in &self.block_leaders.clone() {
            if pc == 0 {
                continue; // entry block already created
            }
            let block = self.builder.create_block(&reg_types);
            self.pc_to_block.insert(pc, block);
        }
    }

    /// Get the register values (block params 1..num_regs+1, skipping closure at 0).
    fn regs_at(&self, block: BlockId) -> Vec<Value> {
        (0..self.num_regs)
            .map(|i| self.builder.block_param(block, i + 1))
            .collect()
    }

    /// Get the closure value (block param 0).
    fn closure_at(&self, block: BlockId) -> Value {
        self.builder.block_param(block, 0)
    }

    /// Make a NanBox-encoded nil constant.
    fn nil_const(&mut self) -> Value {
        self.builder.iconst(Type::I64, 0x7FFD_0000_0000_0000u64 as i64)
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

    /// Get the RK value: if bit 8 is set, load constant; otherwise read register.
    fn rk_value(&mut self, field: u16, regs: &[Value]) -> Value {
        if is_constant(field) {
            self.load_constant(constant_index(field))
        } else {
            regs[field as usize]
        }
    }

    /// Jump to a target block, passing closure + current register state.
    fn jump_to(&mut self, target_pc: usize, closure: Value, regs: &[Value]) {
        let target_block = self.pc_to_block[&target_pc];
        let mut args = Vec::with_capacity(regs.len() + 1);
        args.push(closure);
        args.extend_from_slice(regs);
        self.builder.jump(target_block, &args);
    }

    /// Emit a conditional branch with closure threaded through.
    fn br_if_to(&mut self, cond: Value, closure: Value,
                then_pc: usize, then_regs: &[Value],
                else_pc: usize, else_regs: &[Value]) {
        let then_block = self.pc_to_block[&then_pc];
        let else_block = self.pc_to_block[&else_pc];
        let mut then_args = Vec::with_capacity(then_regs.len() + 1);
        then_args.push(closure);
        then_args.extend_from_slice(then_regs);
        let mut else_args = Vec::with_capacity(else_regs.len() + 1);
        else_args.push(closure);
        else_args.extend_from_slice(else_regs);
        self.builder.br_if(cond, then_block, &then_args, else_block, &else_args);
    }

    fn emit_blocks(&mut self, ext: &ExternFuncs) {
        let leaders: Vec<usize> = self.block_leaders.iter().copied().collect();
        let code = self.proto.code.clone();

        for (leader_idx, &start_pc) in leaders.iter().enumerate() {
            let block = self.pc_to_block[&start_pc];
            if start_pc != 0 {
                self.builder.switch_to_block(block);
            }

            let end_pc = if leader_idx + 1 < leaders.len() {
                leaders[leader_idx + 1]
            } else {
                code.len()
            };

            let mut regs = self.regs_at(block);
            let closure = self.closure_at(block);
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
                        regs[a] = regs[b];
                    }
                    Some(OpCode::LoadK) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        regs[a] = self.load_constant(bx);
                    }
                    Some(OpCode::LoadBool) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);
                        let c = field_c(inst);
                        regs[a] = self.bool_const(b != 0);
                        if c != 0 {
                            let target = pc + 2;
                            self.jump_to(target, closure, &regs);
                            terminated = true;
                        }
                    }
                    Some(OpCode::LoadNil) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let nil = self.nil_const();
                        for i in a..=b {
                            regs[i] = nil;
                        }
                    }
                    Some(OpCode::Add) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_add, &[b, c]).unwrap();
                    }
                    Some(OpCode::Sub) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_sub, &[b, c]).unwrap();
                    }
                    Some(OpCode::Mul) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_mul, &[b, c]).unwrap();
                    }
                    Some(OpCode::Div) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_div, &[b, c]).unwrap();
                    }
                    Some(OpCode::Mod) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_mod, &[b, c]).unwrap();
                    }
                    Some(OpCode::Pow) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_pow, &[b, c]).unwrap();
                    }
                    Some(OpCode::Unm) => {
                        let a = field_a(inst) as usize;
                        let b = regs[field_b(inst) as usize];
                        regs[a] = self.builder.call(ext.lua_unm, &[b]).unwrap();
                    }
                    Some(OpCode::Not) => {
                        let a = field_a(inst) as usize;
                        let b = regs[field_b(inst) as usize];
                        regs[a] = self.builder.call(ext.lua_not, &[b]).unwrap();
                    }
                    Some(OpCode::Len) => {
                        let a = field_a(inst) as usize;
                        let b = regs[field_b(inst) as usize];
                        regs[a] = self.builder.call(ext.lua_len, &[b]).unwrap();
                    }
                    Some(OpCode::Concat) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let c = field_c(inst) as usize;
                        let mut result = regs[b];
                        for i in (b + 1)..=c {
                            result = self.builder.call(ext.lua_concat, &[result, regs[i]]).unwrap();
                        }
                        regs[a] = result;
                    }
                    Some(OpCode::Jmp) => {
                        let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                        self.jump_to(target, closure, &regs);
                        terminated = true;
                    }
                    Some(OpCode::Eq) => {
                        let a = field_a(inst);
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
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

                        self.br_if_to(cond, closure, jmp_target, &regs, fall_through, &regs);
                        terminated = true;
                    }
                    Some(OpCode::Lt) => {
                        let a = field_a(inst);
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        let result = self.builder.call(ext.lua_lt, &[b, c]).unwrap();
                        let true_val = self.bool_const(true);
                        let cmp = self.builder.icmp(CmpOp::Eq, result, true_val);

                        let cond = if a != 0 { cmp } else {
                            let one = self.builder.iconst(Type::I8, 1);
                            self.builder.xor(cmp, one)
                        };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;
                        self.br_if_to(cond, closure, jmp_target, &regs, fall_through, &regs);
                        terminated = true;
                    }
                    Some(OpCode::Le) => {
                        let a = field_a(inst);
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        let result = self.builder.call(ext.lua_le, &[b, c]).unwrap();
                        let true_val = self.bool_const(true);
                        let cmp = self.builder.icmp(CmpOp::Eq, result, true_val);

                        let cond = if a != 0 { cmp } else {
                            let one = self.builder.iconst(Type::I8, 1);
                            self.builder.xor(cmp, one)
                        };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;
                        self.br_if_to(cond, closure, jmp_target, &regs, fall_through, &regs);
                        terminated = true;
                    }
                    Some(OpCode::Test) => {
                        let a = field_a(inst) as usize;
                        let c = field_c(inst);
                        let nil = self.nil_const();
                        let false_val = self.bool_const(false);
                        let is_nil = self.builder.icmp(CmpOp::Eq, regs[a], nil);
                        let is_false = self.builder.icmp(CmpOp::Eq, regs[a], false_val);
                        let is_falsy = self.builder.or(is_nil, is_false);

                        let one = self.builder.iconst(Type::I8, 1);
                        let is_truthy = self.builder.xor(is_falsy, one);

                        let cond = if c == 0 {
                            is_falsy
                        } else {
                            is_truthy
                        };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;
                        self.br_if_to(cond, closure, jmp_target, &regs, fall_through, &regs);
                        terminated = true;
                    }
                    Some(OpCode::TestSet) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let c = field_c(inst);

                        let nil = self.nil_const();
                        let false_val = self.bool_const(false);
                        let is_nil = self.builder.icmp(CmpOp::Eq, regs[b], nil);
                        let is_false = self.builder.icmp(CmpOp::Eq, regs[b], false_val);
                        let is_falsy = self.builder.or(is_nil, is_false);
                        let one = self.builder.iconst(Type::I8, 1);
                        let is_truthy = self.builder.xor(is_falsy, one);

                        let cond = if c == 0 { is_falsy } else { is_truthy };

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;

                        let mut regs_with_set = regs.clone();
                        regs_with_set[a] = regs[b];

                        self.br_if_to(cond, closure, jmp_target, &regs_with_set, fall_through, &regs);
                        terminated = true;
                    }
                    Some(OpCode::GetGlobal) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        let name_val = self.load_constant(bx);
                        regs[a] = self.builder.call(ext.lua_getglobal, &[name_val]).unwrap();
                    }
                    Some(OpCode::SetGlobal) => {
                        let a = field_a(inst) as usize;
                        let bx = field_bx(inst) as usize;
                        let name_val = self.load_constant(bx);
                        self.builder.call(ext.lua_setglobal, &[name_val, regs[a]]);
                    }
                    Some(OpCode::NewTable) => {
                        let a = field_a(inst) as usize;
                        regs[a] = self.builder.call(ext.lua_newtable, &[]).unwrap();
                    }
                    Some(OpCode::GetTable) => {
                        let a = field_a(inst) as usize;
                        let b = regs[field_b(inst) as usize];
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a] = self.builder.call(ext.lua_gettable, &[b, c]).unwrap();
                    }
                    Some(OpCode::SetTable) => {
                        let a = field_a(inst) as usize;
                        let b = self.rk_value(field_b(inst), &regs);
                        let c = self.rk_value(field_c(inst), &regs);
                        self.builder.call(ext.lua_settable, &[regs[a], b, c]);
                    }
                    Some(OpCode::Self_) => {
                        let a = field_a(inst) as usize;
                        let b = regs[field_b(inst) as usize];
                        let c = self.rk_value(field_c(inst), &regs);
                        regs[a + 1] = b;
                        regs[a] = self.builder.call(ext.lua_gettable, &[b, c]).unwrap();
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
                            self.builder.call(ext.lua_store_reg, &[idx_val, regs[a + 1 + i]]);
                        }

                        let func_val = regs[a];
                        let base_idx = self.builder.iconst(Type::I64, 0i64);
                        let nargs_val = self.builder.iconst(Type::I64, nargs as i64);
                        let result = self.builder.call(ext.lua_call, &[func_val, base_idx, nargs_val]).unwrap();

                        if c == 0 {
                            // Variable returns: top = A + 1 (we always return 1 value)
                            regs[a] = result;
                            var_return_top = Some(a + 1);
                        } else if c != 1 {
                            regs[a] = result;
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
                            self.builder.call(ext.lua_store_reg, &[idx_val, regs[a + 1 + i]]);
                        }

                        let func_val = regs[a];
                        let base_idx = self.builder.iconst(Type::I64, 0i64);
                        let nargs_val = self.builder.iconst(Type::I64, nargs as i64);
                        let result = self.builder.call(ext.lua_call, &[func_val, base_idx, nargs_val]).unwrap();
                        self.builder.ret(result);
                        terminated = true;
                    }
                    Some(OpCode::Return) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);

                        if b == 1 {
                            let nil = self.nil_const();
                            self.builder.ret(nil);
                        } else if b == 2 {
                            self.builder.ret(regs[a]);
                        } else if b >= 3 {
                            self.builder.ret(regs[a]);
                        } else {
                            self.builder.ret(regs[a]);
                        }
                        terminated = true;
                    }
                    Some(OpCode::ForPrep) => {
                        let a = field_a(inst) as usize;
                        let target = (pc as i32 + 1 + field_sbx(inst)) as usize;

                        let result = self.builder.call(ext.lua_forprep, &[regs[a], regs[a + 1], regs[a + 2]]).unwrap();
                        regs[a] = result;
                        self.jump_to(target, closure, &regs);
                        terminated = true;
                    }
                    Some(OpCode::ForLoop) => {
                        let a = field_a(inst) as usize;
                        let target = (pc as i32 + 1 + field_sbx(inst)) as usize;
                        let fall_through = pc + 1;

                        let new_idx = self.builder.call(ext.lua_forloop, &[regs[a], regs[a + 1], regs[a + 2]]).unwrap();

                        let nil = self.nil_const();
                        let is_nil = self.builder.icmp(CmpOp::Eq, new_idx, nil);

                        let mut loop_regs = regs.clone();
                        loop_regs[a] = new_idx;
                        loop_regs[a + 3] = new_idx;

                        self.br_if_to(is_nil, closure, fall_through, &regs, target, &loop_regs);
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
                            self.builder.call(ext.lua_store_reg, &[idx_val, regs[a + 1 + i]]);
                        }

                        let table = regs[a];
                        let base_idx = self.builder.iconst(Type::I64, (a + 1) as i64);
                        let offset_val = self.builder.iconst(Type::I64, offset as i64);
                        let count_val = self.builder.iconst(Type::I64, count as i64);

                        self.builder.call(ext.lua_setlist, &[table, base_idx, offset_val, count_val]);
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
                                upval_values.push(regs[pseudo_b]);
                                if pseudo_b == a {
                                    self_ref_indices.push(j);
                                }
                            } else if pseudo_op == OpCode::GetUpval as u8 {
                                // Capture upvalue B from enclosing closure
                                let raw_ptr = self.builder.payload(closure);
                                let upval = self.builder.load(Type::I64, raw_ptr, (24 + pseudo_b * 8) as i32);
                                upval_values.push(upval);
                            } else {
                                upval_values.push(self.nil_const());
                            }
                        }

                        let proto_idx_val = self.builder.iconst(Type::I64, bx as i64);
                        let num_upval_val = self.builder.iconst(Type::I64, num_upvalues as i64);

                        regs[a] = match num_upvalues {
                            0 => self.builder.call(
                                ext.lua_make_closure_0,
                                &[proto_idx_val, num_upval_val],
                            ).unwrap(),
                            1 => self.builder.call(
                                ext.lua_make_closure_1,
                                &[proto_idx_val, num_upval_val, upval_values[0]],
                            ).unwrap(),
                            2 => self.builder.call(
                                ext.lua_make_closure_2,
                                &[proto_idx_val, num_upval_val, upval_values[0], upval_values[1]],
                            ).unwrap(),
                            3 => self.builder.call(
                                ext.lua_make_closure_3,
                                &[proto_idx_val, num_upval_val, upval_values[0], upval_values[1], upval_values[2]],
                            ).unwrap(),
                            4 => self.builder.call(
                                ext.lua_make_closure_4,
                                &[proto_idx_val, num_upval_val, upval_values[0], upval_values[1], upval_values[2], upval_values[3]],
                            ).unwrap(),
                            _ => panic!("closures with {} upvalues not yet supported", num_upvalues),
                        };

                        // Skip the pseudo-instructions
                        pc += num_upvalues;

                        // Fix self-references: the closure was created with the OLD R(A),
                        // but Lua's VM sets R(A) first. Store the closure into its own upvalue slots.
                        for &j in &self_ref_indices {
                            let raw_ptr = self.builder.payload(regs[a]);
                            self.builder.store(regs[a], raw_ptr, (24 + j * 8) as i32);
                        }
                    }
                    Some(OpCode::GetUpval) => {
                        // GETUPVAL A B: R(A) := UpValue[B]
                        // Load directly from the closure struct passed as param 0.
                        // Closure layout: [marker:u64, func_id:u64, num_upvals:u64, upval0, ...]
                        // Upvalue B is at offset 24 + B*8
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let raw_ptr = self.builder.payload(closure);
                        regs[a] = self.builder.load(Type::I64, raw_ptr, (24 + b * 8) as i32);
                    }
                    Some(OpCode::SetUpval) => {
                        // SETUPVAL A B: UpValue[B] := R(A)
                        let a = field_a(inst) as usize;
                        let b = field_b(inst) as usize;
                        let raw_ptr = self.builder.payload(closure);
                        self.builder.store(regs[a], raw_ptr, (24 + b * 8) as i32);
                    }
                    Some(OpCode::Close) => {
                        // Close upvalues - no-op for flat closures
                    }
                    Some(OpCode::VarArg) => {
                        let a = field_a(inst) as usize;
                        let b = field_b(inst);
                        if b >= 2 {
                            let nil = self.nil_const();
                            for i in a..(a + (b as usize) - 1) {
                                if i < self.num_regs {
                                    regs[i] = nil;
                                }
                            }
                        }
                    }
                    Some(OpCode::TForLoop) => {
                        let a = field_a(inst) as usize;
                        let c = field_c(inst) as usize;

                        // Store args R(A+1) and R(A+2) to register_file for the call
                        let idx0 = self.builder.iconst(Type::I64, 0i64);
                        self.builder.call(ext.lua_store_reg, &[idx0, regs[a + 1]]);
                        let idx1 = self.builder.iconst(Type::I64, 1i64);
                        self.builder.call(ext.lua_store_reg, &[idx1, regs[a + 2]]);

                        let func_val = regs[a];
                        let base_idx = self.builder.iconst(Type::I64, 0i64);
                        let nargs_val = self.builder.iconst(Type::I64, 2i64);
                        let result = self.builder.call(ext.lua_call, &[func_val, base_idx, nargs_val]).unwrap();
                        regs[a + 3] = result;
                        if c > 1 {
                            // Additional return values would need support
                        }

                        let nil = self.nil_const();
                        let is_nil = self.builder.icmp(CmpOp::Eq, regs[a + 3], nil);

                        let mut continue_regs = regs.clone();
                        continue_regs[a + 2] = regs[a + 3];

                        let jmp_inst = code[pc + 1];
                        let jmp_target = (pc as i32 + 2 + field_sbx(jmp_inst)) as usize;
                        let fall_through = pc + 2;

                        self.br_if_to(is_nil, closure, fall_through, &regs, jmp_target, &continue_regs);
                        terminated = true;
                    }
                    None => {
                        panic!("unknown opcode {} at pc {}", opcode(inst), pc);
                    }
                }

                pc += 1;

                // If we hit a comparison/test op, we consumed the next JMP too
                match op {
                    Some(OpCode::Eq | OpCode::Lt | OpCode::Le | OpCode::Test | OpCode::TestSet | OpCode::TForLoop) => {
                        pc += 1;
                    }
                    _ => {}
                }
            }

            // If block wasn't terminated by a jump/return, fall through to next block
            if !terminated {
                if let Some(&next_pc) = leaders.get(leader_idx + 1) {
                    self.jump_to(next_pc, closure, &regs);
                }
            }
        }
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
