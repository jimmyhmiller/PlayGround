/// Translate compiled Lox bytecodes → dynir Module.
///
/// Each Lox function becomes a dynir function. All values are NanBox-encoded I64.
/// Dynamic operations (add, call, property access, etc.) are delegated to extern
/// runtime functions.
///
/// Translation strategy: stack slots model the Lox operand stack and locals.
/// The operand stack pointer (sp) is tracked at compile time. At basic block
/// boundaries, sp must be consistent across all predecessors (guaranteed by Lox).
use std::collections::{BTreeSet, HashMap};

use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{BlockId, CmpOp, FuncRef, Module, StackSlot, Value};
use dynir::types::{Signature, Type};

use crate::chunk::OpCode;
use crate::object::ObjFunction;
use crate::value;

/// Result of translating a Lox program to dynir.
pub struct TranslatedModule {
    pub module: Module,
    pub entry: FuncRef,
    /// Maps Lox func_id → dynir FuncRef
    pub func_refs: Vec<FuncRef>,
}

/// All extern function references used by translated code.
struct Externs {
    lox_add: FuncRef,
    lox_sub: FuncRef,
    lox_mul: FuncRef,
    lox_div: FuncRef,
    lox_negate: FuncRef,
    lox_not: FuncRef,
    lox_equal: FuncRef,
    lox_greater: FuncRef,
    lox_less: FuncRef,
    lox_is_falsey: FuncRef,
    lox_print: FuncRef,
    lox_clock: FuncRef,
    lox_get_global: FuncRef,
    lox_set_global: FuncRef,
    lox_define_global: FuncRef,
    lox_get_property: FuncRef,
    lox_set_property: FuncRef,
    lox_get_super: FuncRef,
    lox_push_arg: FuncRef,
    lox_call: FuncRef,
    lox_invoke: FuncRef,
    lox_super_invoke: FuncRef,
    lox_get_upvalue: FuncRef,
    lox_set_upvalue: FuncRef,
    lox_make_closure: FuncRef,
    lox_make_class: FuncRef,
    lox_inherit: FuncRef,
    lox_define_method: FuncRef,
    lox_close_upvalue: FuncRef,
}

fn sig(params: &[Type], ret: Option<Type>) -> Signature {
    Signature {
        params: params.to_vec(),
        ret,
    }
}

fn declare_externs(mb: &mut ModuleBuilder) -> Externs {
    let i64_2 = [Type::I64, Type::I64];
    let i64_1 = [Type::I64];
    let i64_3 = [Type::I64, Type::I64, Type::I64];
    let i64_4 = [Type::I64, Type::I64, Type::I64, Type::I64];

    Externs {
        lox_add: mb.declare_extern("lox_add", sig(&i64_2, Some(Type::I64))),
        lox_sub: mb.declare_extern("lox_sub", sig(&i64_2, Some(Type::I64))),
        lox_mul: mb.declare_extern("lox_mul", sig(&i64_2, Some(Type::I64))),
        lox_div: mb.declare_extern("lox_div", sig(&i64_2, Some(Type::I64))),
        lox_negate: mb.declare_extern("lox_negate", sig(&i64_1, Some(Type::I64))),
        lox_not: mb.declare_extern("lox_not", sig(&i64_1, Some(Type::I64))),
        lox_equal: mb.declare_extern("lox_equal", sig(&i64_2, Some(Type::I64))),
        lox_greater: mb.declare_extern("lox_greater", sig(&i64_2, Some(Type::I64))),
        lox_less: mb.declare_extern("lox_less", sig(&i64_2, Some(Type::I64))),
        lox_is_falsey: mb.declare_extern("lox_is_falsey", sig(&i64_1, Some(Type::I64))),
        lox_print: mb.declare_extern("lox_print", sig(&i64_1, None)),
        lox_clock: mb.declare_extern("lox_clock", sig(&[], Some(Type::I64))),
        lox_get_global: mb.declare_extern("lox_get_global", sig(&i64_1, Some(Type::I64))),
        lox_set_global: mb.declare_extern("lox_set_global", sig(&i64_2, Some(Type::I64))),
        lox_define_global: mb.declare_extern("lox_define_global", sig(&i64_2, None)),
        lox_get_property: mb.declare_extern("lox_get_property", sig(&i64_2, Some(Type::I64))),
        lox_set_property: mb.declare_extern("lox_set_property", sig(&i64_3, Some(Type::I64))),
        lox_get_super: mb.declare_extern("lox_get_super", sig(&i64_3, Some(Type::I64))),
        lox_push_arg: mb.declare_extern("lox_push_arg", sig(&i64_2, None)),
        lox_call: mb.declare_extern("lox_call", sig(&i64_2, Some(Type::I64))),
        lox_invoke: mb.declare_extern("lox_invoke", sig(&i64_3, Some(Type::I64))),
        lox_super_invoke: mb.declare_extern("lox_super_invoke", sig(&i64_4, Some(Type::I64))),
        lox_get_upvalue: mb.declare_extern("lox_get_upvalue", sig(&i64_2, Some(Type::I64))),
        lox_set_upvalue: mb.declare_extern("lox_set_upvalue", sig(&i64_3, None)),
        lox_make_closure: mb.declare_extern("lox_make_closure", sig(&i64_3, Some(Type::I64))),
        lox_make_class: mb.declare_extern("lox_make_class", sig(&i64_1, Some(Type::I64))),
        lox_inherit: mb.declare_extern("lox_inherit", sig(&i64_2, Some(Type::I64))),
        lox_define_method: mb.declare_extern("lox_define_method", sig(&i64_3, None)),
        lox_close_upvalue: mb.declare_extern("lox_close_upvalue", sig(&i64_1, None)),
    }
}

/// Collect all ObjFunction pointers from a compiled Lox script.
/// Returns them in order: [script, func0, func1, ...] with func_id assigned.
pub fn collect_functions(script: *mut ObjFunction) -> Vec<*mut ObjFunction> {
    let mut funcs = Vec::new();
    collect_functions_recursive(script, &mut funcs);
    // Assign func_ids
    for (i, &f) in funcs.iter().enumerate() {
        unsafe { (*f).func_id = i };
    }
    funcs
}

fn collect_functions_recursive(func: *mut ObjFunction, out: &mut Vec<*mut ObjFunction>) {
    out.push(func);
    // Scan constants for nested functions
    let chunk = unsafe { &(*func).chunk };
    for &constant in &chunk.constants {
        if value::is_obj(constant) {
            let obj = value::as_obj(constant);
            if unsafe { (*obj).obj_type } == crate::object::ObjType::Function {
                collect_functions_recursive(obj as *mut ObjFunction, out);
            }
        }
    }
}

/// Translate all Lox functions into a dynir Module.
pub fn translate(functions: &[*mut ObjFunction]) -> TranslatedModule {
    let mut mb = ModuleBuilder::new();
    let externs = declare_externs(&mut mb);

    // Declare all internal functions
    let mut func_refs = Vec::with_capacity(functions.len());
    for (i, &func) in functions.iter().enumerate() {
        let arity = unsafe { (*func).arity } as usize;
        // Params: first is closure/this, then arity args
        let mut params = Vec::with_capacity(1 + arity);
        for _ in 0..1 + arity {
            params.push(Type::I64);
        }
        let name = if i == 0 {
            "lox_script".to_string()
        } else {
            let n = unsafe {
                if (*func).name.is_null() {
                    format!("lox_fn_{}", i)
                } else {
                    format!("lox_{}", (*(*func).name).chars)
                }
            };
            n
        };
        let fref = mb.declare_func(&name, &params, Some(Type::I64));
        func_refs.push(fref);
    }

    // Define each function
    for (i, &func) in functions.iter().enumerate() {
        let fb = mb.define_func(func_refs[i]);
        let finished = translate_function(func, fb, &externs, &func_refs);
        mb.finish_func(func_refs[i], finished);
    }

    let entry = func_refs[0];
    TranslatedModule {
        module: mb.build(),
        entry,
        func_refs,
    }
}

/// Compute max stack depth by simulating bytecodes.
fn compute_max_stack(func: *mut ObjFunction) -> usize {
    let chunk = unsafe { &(*func).chunk };
    let code = &chunk.code;
    let mut max_depth: usize = 0;
    let mut depth: i32 = 0;
    let mut ip = 0;

    while ip < code.len() {
        let op: OpCode = code[ip].into();
        ip += 1;

        match op {
            OpCode::True | OpCode::False | OpCode::Nil => {
                depth += 1;
            }
            OpCode::Constant | OpCode::GetLocal | OpCode::GetGlobal
            | OpCode::GetUpvalue | OpCode::GetProperty | OpCode::GetSuper => {
                depth += 1;
                ip += 1; // 1-byte operand
            }
            OpCode::Closure => {
                depth += 1;
                let const_idx = code[ip] as usize;
                ip += 1;
                // Read upvalue entries
                let func_val = chunk.constants[const_idx];
                let inner_func = value::as_obj(func_val) as *mut ObjFunction;
                let uv_count = unsafe { (*inner_func).upvalue_count } as usize;
                ip += uv_count * 2;
            }
            OpCode::SetLocal | OpCode::SetGlobal | OpCode::SetUpvalue => {
                ip += 1; // operand
                // peek, don't pop, net 0
            }
            OpCode::SetProperty => {
                ip += 1;
                depth -= 1; // pops instance, keeps value
            }
            OpCode::Pop | OpCode::Print | OpCode::CloseUpvalue => {
                depth -= 1;
            }
            OpCode::Negate | OpCode::Not => {
                // pop 1, push 1 = net 0
            }
            OpCode::Equal | OpCode::Greater | OpCode::Less
            | OpCode::Add | OpCode::Subtract | OpCode::Multiply | OpCode::Divide => {
                depth -= 1; // pop 2, push 1
            }
            OpCode::DefineGlobal => {
                ip += 1;
                depth -= 1;
            }
            OpCode::Jump | OpCode::JumpIfFalse | OpCode::Loop => {
                ip += 2; // 2-byte offset
            }
            OpCode::Call => {
                let arg_count = code[ip] as i32;
                ip += 1;
                depth -= arg_count; // pops args + callee, pushes result
            }
            OpCode::Invoke | OpCode::SuperInvoke => {
                ip += 1; // name constant
                let arg_count = code[ip] as i32;
                ip += 1;
                depth -= arg_count; // net: receiver + args → result
            }
            OpCode::Return => {
                depth -= 1;
                if depth < 0 { depth = 0; }
            }
            OpCode::Class => {
                ip += 1;
                depth += 1;
            }
            OpCode::Inherit => {
                depth -= 1;
            }
            OpCode::Method => {
                ip += 1;
                depth -= 1;
            }
        }
        if depth as usize > max_depth {
            max_depth = depth as usize;
        }
    }
    max_depth.max(1)
}

/// Identify bytecode offsets that are basic block leaders.
fn find_block_leaders(func: *mut ObjFunction) -> BTreeSet<usize> {
    let chunk = unsafe { &(*func).chunk };
    let code = &chunk.code;
    let mut leaders = BTreeSet::new();
    leaders.insert(0);

    let mut ip = 0;
    while ip < code.len() {
        let op: OpCode = code[ip].into();
        ip += 1;

        match op {
            OpCode::Jump | OpCode::Loop => {
                let offset = ((code[ip] as u16) << 8 | code[ip + 1] as u16) as usize;
                ip += 2;
                let target = if matches!(op, OpCode::Loop) {
                    ip - offset
                } else {
                    ip + offset
                };
                leaders.insert(target);
                leaders.insert(ip); // fall-through
            }
            OpCode::JumpIfFalse => {
                let offset = ((code[ip] as u16) << 8 | code[ip + 1] as u16) as usize;
                ip += 2;
                let target = ip + offset;
                leaders.insert(target);
                leaders.insert(ip);
            }
            OpCode::Return => {
                if ip < code.len() {
                    leaders.insert(ip);
                }
            }
            OpCode::Constant | OpCode::GetLocal | OpCode::SetLocal
            | OpCode::GetGlobal | OpCode::DefineGlobal | OpCode::SetGlobal
            | OpCode::GetUpvalue | OpCode::SetUpvalue
            | OpCode::GetProperty | OpCode::SetProperty | OpCode::GetSuper
            | OpCode::Class | OpCode::Method => {
                ip += 1;
            }
            OpCode::Call => {
                ip += 1;
            }
            OpCode::Invoke | OpCode::SuperInvoke => {
                ip += 2;
            }
            OpCode::Closure => {
                let const_idx = code[ip] as usize;
                ip += 1;
                let func_val = chunk.constants[const_idx];
                let inner_func = value::as_obj(func_val) as *mut ObjFunction;
                let uv_count = unsafe { (*inner_func).upvalue_count } as usize;
                ip += uv_count * 2;
            }
            _ => {}
        }
    }
    leaders
}

/// Translate a single Lox function into dynir.
fn translate_function(
    func: *mut ObjFunction,
    mut fb: FunctionBuilder,
    externs: &Externs,
    func_refs: &[FuncRef],
) -> FunctionBuilder {
    let chunk = unsafe { &(*func).chunk };
    let code = &chunk.code;
    let arity = unsafe { (*func).arity } as usize;
    let num_locals = count_locals(func);
    let max_stack = compute_max_stack(func);

    // Create stack slots for locals + operand stack.
    // Slot layout: [closure/this, param0, ..., paramN, local0, ..., localM, stack0, ..., stackK]
    let total_slots = num_locals + max_stack + 16; // padding
    let slots: Vec<StackSlot> = (0..total_slots)
        .map(|_| fb.create_stack_slot(8, false))
        .collect();

    // Entry block: store params into slots
    let entry = fb.entry_block();
    for i in 0..1 + arity {
        let param = fb.block_param(entry, i);
        let addr = fb.stack_addr(slots[i]);
        fb.store(param, addr, 0);
    }
    // Initialize remaining locals to nil
    let nil = fb.iconst(Type::I64, value::nil_val() as i64);
    for i in (1 + arity)..num_locals {
        let addr = fb.stack_addr(slots[i]);
        fb.store(nil, addr, 0);
    }

    // Identify block leaders and create blocks
    let leaders = find_block_leaders(func);
    let mut pc_to_block: HashMap<usize, BlockId> = HashMap::new();
    for &pc in &leaders {
        if pc == 0 {
            // Entry block handles setup; create a separate execution block for PC 0
            let blk = fb.create_block(&[]);
            pc_to_block.insert(0, blk);
        } else {
            let blk = fb.create_block(&[]);
            pc_to_block.insert(pc, blk);
        }
    }

    // Jump from entry to PC 0
    let exec_start = pc_to_block[&0];
    fb.jump(exec_start, &[]);

    // Translate each basic block
    let sorted_leaders: Vec<usize> = leaders.iter().copied().collect();
    let mut sp: usize = 0; // operand stack pointer (compile-time)

    for (leader_idx, &start_pc) in sorted_leaders.iter().enumerate() {
        let block = pc_to_block[&start_pc];
        fb.switch_to_block(block);

        let end_pc = if leader_idx + 1 < sorted_leaders.len() {
            sorted_leaders[leader_idx + 1]
        } else {
            code.len()
        };

        let mut ip = start_pc;
        let mut terminated = false;

        while ip < end_pc && !terminated {
            let op: OpCode = code[ip].into();
            ip += 1;

            match op {
                OpCode::Constant => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let constant = chunk.constants[idx];
                    let val = fb.iconst(Type::I64, constant as i64);
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::Nil => {
                    let val = fb.iconst(Type::I64, value::nil_val() as i64);
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::True => {
                    let val = fb.iconst(Type::I64, value::bool_val(true) as i64);
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::False => {
                    let val = fb.iconst(Type::I64, value::bool_val(false) as i64);
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::Pop => {
                    sp -= 1;
                }
                OpCode::GetLocal => {
                    let slot = code[ip] as usize;
                    ip += 1;
                    let val = load_local(&mut fb, &slots, slot);
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::SetLocal => {
                    let slot = code[ip] as usize;
                    ip += 1;
                    let val = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    store_local(&mut fb, &slots, slot, val);
                }
                OpCode::GetGlobal => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let val = fb.call(externs.lox_get_global, &[name]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::DefineGlobal => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    sp -= 1;
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let val = load_stack(&mut fb, &slots, num_locals, sp);
                    fb.call(externs.lox_define_global, &[name, val]);
                }
                OpCode::SetGlobal => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let val = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    fb.call(externs.lox_set_global, &[name, val]);
                }
                OpCode::GetUpvalue => {
                    let slot = code[ip] as usize;
                    ip += 1;
                    let closure = load_local(&mut fb, &slots, 0);
                    let idx = fb.iconst(Type::I64, slot as i64);
                    let val = fb.call(externs.lox_get_upvalue, &[closure, idx]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::SetUpvalue => {
                    let slot = code[ip] as usize;
                    ip += 1;
                    let closure = load_local(&mut fb, &slots, 0);
                    let idx = fb.iconst(Type::I64, slot as i64);
                    let val = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    fb.call(externs.lox_set_upvalue, &[closure, idx, val]);
                }
                OpCode::GetProperty => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    sp -= 1;
                    let instance = load_stack(&mut fb, &slots, num_locals, sp);
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let val = fb.call(externs.lox_get_property, &[instance, name]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::SetProperty => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let value = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let instance = load_stack(&mut fb, &slots, num_locals, sp - 2);
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let result = fb.call(externs.lox_set_property, &[instance, name, value]).unwrap();
                    sp -= 2;
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::GetSuper => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let superclass = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    sp -= 1;
                    let receiver = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let val = fb.call(externs.lox_get_super, &[receiver, name, superclass]).unwrap();
                    sp -= 1;
                    store_stack(&mut fb, &slots, num_locals, sp, val);
                    sp += 1;
                }
                OpCode::Equal => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_equal, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Greater => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_greater, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Less => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_less, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Add => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_add, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Subtract => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_sub, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Multiply => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_mul, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Divide => {
                    sp -= 1;
                    let b = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1;
                    let a = load_stack(&mut fb, &slots, num_locals, sp);
                    let result = fb.call(externs.lox_div, &[a, b]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Not => {
                    let val = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let result = fb.call(externs.lox_not, &[val]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp - 1, result);
                }
                OpCode::Negate => {
                    let val = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let result = fb.call(externs.lox_negate, &[val]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp - 1, result);
                }
                OpCode::Print => {
                    sp -= 1;
                    let val = load_stack(&mut fb, &slots, num_locals, sp);
                    fb.call(externs.lox_print, &[val]);
                }
                OpCode::Jump => {
                    let offset = ((code[ip] as u16) << 8 | code[ip + 1] as u16) as usize;
                    ip += 2;
                    let target = ip + offset;
                    let target_block = pc_to_block[&target];
                    fb.jump(target_block, &[]);
                    terminated = true;
                }
                OpCode::JumpIfFalse => {
                    let offset = ((code[ip] as u16) << 8 | code[ip + 1] as u16) as usize;
                    ip += 2;
                    let target = ip + offset;
                    let cond = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let is_falsey = fb.call(externs.lox_is_falsey, &[cond]).unwrap();
                    let cond_i8 = fb.trunc(is_falsey, Type::I8);
                    let target_block = pc_to_block[&target];
                    let fallthrough = pc_to_block[&ip];
                    fb.br_if(cond_i8, target_block, &[], fallthrough, &[]);
                    terminated = true;
                }
                OpCode::Loop => {
                    let offset = ((code[ip] as u16) << 8 | code[ip + 1] as u16) as usize;
                    ip += 2;
                    let target = ip - offset;
                    let target_block = pc_to_block[&target];
                    fb.jump(target_block, &[]);
                    terminated = true;
                }
                OpCode::Call => {
                    let arg_count = code[ip] as usize;
                    ip += 1;
                    // Push args to runtime arg buffer
                    for i in 0..arg_count {
                        let arg = load_stack(&mut fb, &slots, num_locals, sp - arg_count + i);
                        let idx = fb.iconst(Type::I64, i as i64);
                        fb.call(externs.lox_push_arg, &[idx, arg]);
                    }
                    sp -= arg_count;
                    sp -= 1; // callee
                    let callee = load_stack(&mut fb, &slots, num_locals, sp);
                    let argc = fb.iconst(Type::I64, arg_count as i64);
                    let result = fb.call(externs.lox_call, &[callee, argc]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Invoke => {
                    let name_idx = code[ip] as usize;
                    ip += 1;
                    let arg_count = code[ip] as usize;
                    ip += 1;
                    // Push args
                    for i in 0..arg_count {
                        let arg = load_stack(&mut fb, &slots, num_locals, sp - arg_count + i);
                        let idx = fb.iconst(Type::I64, i as i64);
                        fb.call(externs.lox_push_arg, &[idx, arg]);
                    }
                    sp -= arg_count;
                    sp -= 1; // receiver
                    let receiver = load_stack(&mut fb, &slots, num_locals, sp);
                    let name = fb.iconst(Type::I64, chunk.constants[name_idx] as i64);
                    let argc = fb.iconst(Type::I64, arg_count as i64);
                    let result = fb.call(externs.lox_invoke, &[receiver, name, argc]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::SuperInvoke => {
                    let name_idx = code[ip] as usize;
                    ip += 1;
                    let arg_count = code[ip] as usize;
                    ip += 1;
                    // Push args
                    for i in 0..arg_count {
                        let arg = load_stack(&mut fb, &slots, num_locals, sp - arg_count + i);
                        let idx = fb.iconst(Type::I64, i as i64);
                        fb.call(externs.lox_push_arg, &[idx, arg]);
                    }
                    sp -= arg_count;
                    sp -= 1; // superclass
                    let superclass = load_stack(&mut fb, &slots, num_locals, sp);
                    sp -= 1; // receiver (this)
                    let receiver = load_stack(&mut fb, &slots, num_locals, sp);
                    let name = fb.iconst(Type::I64, chunk.constants[name_idx] as i64);
                    let argc = fb.iconst(Type::I64, arg_count as i64);
                    let result = fb.call(externs.lox_super_invoke, &[receiver, name, superclass, argc]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, result);
                    sp += 1;
                }
                OpCode::Closure => {
                    let const_idx = code[ip] as usize;
                    ip += 1;
                    let func_val = chunk.constants[const_idx];
                    let inner_func = value::as_obj(func_val) as *mut ObjFunction;
                    let inner_func_id = unsafe { (*inner_func).func_id };
                    let uv_count = unsafe { (*inner_func).upvalue_count } as usize;

                    let func_id_val = fb.iconst(Type::I64, inner_func_id as i64);
                    let uv_count_val = fb.iconst(Type::I64, uv_count as i64);
                    // For simplicity, pass func constant (which contains the ObjFunction ptr)
                    let func_const = fb.iconst(Type::I64, func_val as i64);
                    let closure = fb.call(externs.lox_make_closure, &[func_const, func_id_val, uv_count_val]).unwrap();

                    // TODO: handle upvalue capture properly
                    // For now skip upvalue bytes
                    ip += uv_count * 2;

                    store_stack(&mut fb, &slots, num_locals, sp, closure);
                    sp += 1;
                }
                OpCode::CloseUpvalue => {
                    sp -= 1;
                    let val = load_stack(&mut fb, &slots, num_locals, sp);
                    fb.call(externs.lox_close_upvalue, &[val]);
                }
                OpCode::Return => {
                    let result = if sp > 0 {
                        sp -= 1;
                        load_stack(&mut fb, &slots, num_locals, sp)
                    } else {
                        fb.iconst(Type::I64, value::nil_val() as i64)
                    };
                    fb.ret(result);
                    terminated = true;
                }
                OpCode::Class => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    let class = fb.call(externs.lox_make_class, &[name]).unwrap();
                    store_stack(&mut fb, &slots, num_locals, sp, class);
                    sp += 1;
                }
                OpCode::Inherit => {
                    // peek(0)=subclass, peek(1)=superclass
                    let subclass = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let superclass = load_stack(&mut fb, &slots, num_locals, sp - 2);
                    fb.call(externs.lox_inherit, &[subclass, superclass]);
                    sp -= 1; // pop subclass
                }
                OpCode::Method => {
                    let idx = code[ip] as usize;
                    ip += 1;
                    let method = load_stack(&mut fb, &slots, num_locals, sp - 1);
                    let class = load_stack(&mut fb, &slots, num_locals, sp - 2);
                    let name = fb.iconst(Type::I64, chunk.constants[idx] as i64);
                    fb.call(externs.lox_define_method, &[class, name, method]);
                    sp -= 1;
                }
            }
        }

        // If block wasn't terminated, fall through to next block
        if !terminated && ip < code.len() {
            if let Some(&next_block) = pc_to_block.get(&ip) {
                fb.jump(next_block, &[]);
            }
        }
    }

    fb
}

/// Count the number of local variable slots needed.
/// In clox, locals use the bottom of the stack. We need to find the max
/// local slot index used by GetLocal/SetLocal.
fn count_locals(func: *mut ObjFunction) -> usize {
    let chunk = unsafe { &(*func).chunk };
    let code = &chunk.code;
    let mut max_local: usize = 0;
    let arity = unsafe { (*func).arity } as usize;

    let mut ip = 0;
    while ip < code.len() {
        let op: OpCode = code[ip].into();
        ip += 1;
        match op {
            OpCode::GetLocal | OpCode::SetLocal => {
                let slot = code[ip] as usize;
                ip += 1;
                if slot + 1 > max_local {
                    max_local = slot + 1;
                }
            }
            OpCode::Constant | OpCode::GetGlobal | OpCode::DefineGlobal
            | OpCode::SetGlobal | OpCode::GetUpvalue | OpCode::SetUpvalue
            | OpCode::GetProperty | OpCode::SetProperty | OpCode::GetSuper
            | OpCode::Class | OpCode::Method | OpCode::Call => {
                ip += 1;
            }
            OpCode::Jump | OpCode::JumpIfFalse | OpCode::Loop
            | OpCode::Invoke | OpCode::SuperInvoke => {
                ip += 2;
            }
            OpCode::Closure => {
                let const_idx = code[ip] as usize;
                ip += 1;
                let func_val = chunk.constants[const_idx];
                let inner_func = value::as_obj(func_val) as *mut ObjFunction;
                let uv_count = unsafe { (*inner_func).upvalue_count } as usize;
                ip += uv_count * 2;
            }
            _ => {}
        }
    }

    max_local.max(1 + arity)
}

// ── Helpers ─────────────────────────────────────────────────────

#[inline]
fn load_local(fb: &mut FunctionBuilder, slots: &[StackSlot], slot: usize) -> Value {
    let addr = fb.stack_addr(slots[slot]);
    fb.load(Type::I64, addr, 0)
}

#[inline]
fn store_local(fb: &mut FunctionBuilder, slots: &[StackSlot], slot: usize, val: Value) {
    let addr = fb.stack_addr(slots[slot]);
    fb.store(val, addr, 0);
}

#[inline]
fn load_stack(fb: &mut FunctionBuilder, slots: &[StackSlot], num_locals: usize, sp: usize) -> Value {
    let addr = fb.stack_addr(slots[num_locals + sp]);
    fb.load(Type::I64, addr, 0)
}

#[inline]
fn store_stack(fb: &mut FunctionBuilder, slots: &[StackSlot], num_locals: usize, sp: usize, val: Value) {
    let addr = fb.stack_addr(slots[num_locals + sp]);
    fb.store(val, addr, 0);
}
