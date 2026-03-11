use crate::ir::*;
use crate::types::Type;

/// Successful execution result.
#[derive(Debug)]
pub enum InterpResult {
    Value(u64),
    Void,
    Deopt {
        deopt_id: DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
}

/// Errors (bugs/traps, not normal control flow).
#[derive(Debug)]
pub enum InterpError {
    Unreachable,
    UncaughtException(u64),
    UnknownExternFunc(String),
    DivideByZero,
}

/// What extern callbacks return.
pub enum ExternCallResult {
    Value(Option<u64>),
    Exception(u64),
}

pub struct Interpreter<'a> {
    func: &'a Function,
    externs: Vec<Option<Box<dyn Fn(&[u64]) -> ExternCallResult>>>,
    indirect_handler: Option<Box<dyn Fn(u64, &[u64]) -> ExternCallResult>>,
}

impl<'a> Interpreter<'a> {
    pub fn new(func: &'a Function) -> Self {
        let externs = (0..func.extern_funcs.len()).map(|_| None).collect();
        Interpreter {
            func,
            externs,
            indirect_handler: None,
        }
    }

    /// Bind a closure to a declared extern function by FuncRef.
    pub fn bind(&mut self, fref: FuncRef, f: impl Fn(&[u64]) -> ExternCallResult + 'static) {
        self.externs[fref.index()] = Some(Box::new(f));
    }

    /// Bind a closure to a declared extern function by name.
    pub fn bind_by_name(&mut self, name: &str, f: impl Fn(&[u64]) -> ExternCallResult + 'static) {
        for (i, ef) in self.func.extern_funcs.iter().enumerate() {
            if ef.name == name {
                self.externs[i] = Some(Box::new(f));
                return;
            }
        }
        panic!("no extern func named '{}'", name);
    }

    /// Bind a handler for indirect calls.
    pub fn bind_indirect(
        &mut self,
        handler: impl Fn(u64, &[u64]) -> ExternCallResult + 'static,
    ) {
        self.indirect_handler = Some(Box::new(handler));
    }

    pub fn run(&self, args: &[u64]) -> Result<InterpResult, InterpError> {
        let func = self.func;
        let mut vals: Vec<u64> = vec![0; func.value_types.len()];

        // Initialize entry block params from args.
        let entry = &func.blocks[0];
        for (i, (v, _ty)) in entry.params.iter().enumerate() {
            vals[v.index()] = args[i];
        }

        let mut block_idx: usize = 0;

        loop {
            let block = &func.blocks[block_idx];

            // Execute instructions.
            for node in &block.insts {
                let result = self.exec_inst(&node.inst, &vals)?;
                if let Some(r) = result {
                    match r {
                        InstResult::Val(v) => {
                            if let Some(dest) = node.value {
                                vals[dest.index()] = v;
                            }
                        }
                        InstResult::Deopt {
                            deopt_id,
                            resume_point,
                            live_values,
                        } => {
                            return Ok(InterpResult::Deopt {
                                deopt_id,
                                resume_point,
                                live_values,
                            });
                        }
                    }
                }
            }

            // Execute terminator.
            match &block.terminator {
                Terminator::Ret(v) => {
                    return Ok(InterpResult::Value(vals[v.index()]));
                }
                Terminator::RetVoid => {
                    return Ok(InterpResult::Void);
                }
                Terminator::Jump(target, jump_args) => {
                    transfer_args(&mut vals, *target, jump_args, func);
                    block_idx = target.index();
                }
                Terminator::BrIf {
                    cond,
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                } => {
                    if vals[cond.index()] != 0 {
                        transfer_args(&mut vals, *then_block, then_args, func);
                        block_idx = then_block.index();
                    } else {
                        transfer_args(&mut vals, *else_block, else_args, func);
                        block_idx = else_block.index();
                    }
                }
                Terminator::Switch {
                    val,
                    cases,
                    default_block,
                    default_args,
                } => {
                    let v = vals[val.index()] as i64;
                    let mut matched = false;
                    for (case_val, target, case_args) in cases {
                        if v == *case_val {
                            transfer_args(&mut vals, *target, case_args, func);
                            block_idx = target.index();
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        transfer_args(&mut vals, *default_block, default_args, func);
                        block_idx = default_block.index();
                    }
                }
                Terminator::Invoke {
                    func: fref,
                    args: call_args,
                    normal,
                    normal_args,
                    exception,
                    exception_args,
                } => {
                    let arg_vals: Vec<u64> =
                        call_args.iter().map(|v| vals[v.index()]).collect();
                    let result = self.call_extern(*fref, &arg_vals)?;
                    match result {
                        ExternCallResult::Value(ret) => {
                            // Normal path: first param of normal block gets return value (if any),
                            // then normal_args fill remaining params.
                            let target_block = &func.blocks[normal.index()];
                            let mut param_idx = 0;
                            if let Some(ret_val) = ret {
                                if !target_block.params.is_empty() {
                                    vals[target_block.params[0].0.index()] = ret_val;
                                    param_idx = 1;
                                }
                            }
                            let extra: Vec<u64> =
                                normal_args.iter().map(|v| vals[v.index()]).collect();
                            for (i, val) in extra.iter().enumerate() {
                                vals[target_block.params[param_idx + i].0.index()] = *val;
                            }
                            block_idx = normal.index();
                        }
                        ExternCallResult::Exception(_exc) => {
                            transfer_args(&mut vals, *exception, exception_args, func);
                            block_idx = exception.index();
                        }
                    }
                }
                Terminator::InvokeIndirect {
                    callee,
                    args: call_args,
                    ret_ty: _,
                    normal,
                    normal_args,
                    exception,
                    exception_args,
                } => {
                    let callee_val = vals[callee.index()];
                    let arg_vals: Vec<u64> =
                        call_args.iter().map(|v| vals[v.index()]).collect();
                    let handler = self.indirect_handler.as_ref().ok_or_else(|| {
                        InterpError::UnknownExternFunc("(indirect)".to_string())
                    })?;
                    let result = handler(callee_val, &arg_vals);
                    match result {
                        ExternCallResult::Value(ret) => {
                            let target_block = &func.blocks[normal.index()];
                            let mut param_idx = 0;
                            if let Some(ret_val) = ret {
                                if !target_block.params.is_empty() {
                                    vals[target_block.params[0].0.index()] = ret_val;
                                    param_idx = 1;
                                }
                            }
                            let extra: Vec<u64> =
                                normal_args.iter().map(|v| vals[v.index()]).collect();
                            for (i, val) in extra.iter().enumerate() {
                                vals[target_block.params[param_idx + i].0.index()] = *val;
                            }
                            block_idx = normal.index();
                        }
                        ExternCallResult::Exception(_exc) => {
                            transfer_args(&mut vals, *exception, exception_args, func);
                            block_idx = exception.index();
                        }
                    }
                }
                Terminator::Unreachable => {
                    return Err(InterpError::Unreachable);
                }
            }
        }
    }

    fn call_extern(
        &self,
        fref: FuncRef,
        args: &[u64],
    ) -> Result<ExternCallResult, InterpError> {
        match &self.externs[fref.index()] {
            Some(f) => Ok(f(args)),
            None => Err(InterpError::UnknownExternFunc(
                self.func.extern_funcs[fref.index()].name.clone(),
            )),
        }
    }

    fn exec_inst(
        &self,
        inst: &Inst,
        vals: &[u64],
    ) -> Result<Option<InstResult>, InterpError> {
        let v = |val: &Value| vals[val.index()];
        let ty = |val: &Value| self.func.value_type(*val);

        match inst {
            // Constants
            Inst::Iconst(t, imm) => Ok(Some(InstResult::Val(mask(*imm as u64, *t)))),
            Inst::F64Const(f) => Ok(Some(InstResult::Val(f.to_bits()))),

            // Integer arithmetic
            Inst::Add(a, b) => {
                let res_ty = arith_result_type(ty(a), ty(b));
                Ok(Some(InstResult::Val(mask(v(a).wrapping_add(v(b)), res_ty))))
            }
            Inst::Sub(a, b) => {
                let res_ty = arith_result_type(ty(a), ty(b));
                Ok(Some(InstResult::Val(mask(v(a).wrapping_sub(v(b)), res_ty))))
            }
            Inst::Mul(a, b) => {
                let res_ty = arith_result_type(ty(a), ty(b));
                Ok(Some(InstResult::Val(mask(v(a).wrapping_mul(v(b)), res_ty))))
            }
            Inst::SDiv(a, b) => {
                let vb = v(b);
                if vb == 0 {
                    return Err(InterpError::DivideByZero);
                }
                let res_ty = arith_result_type(ty(a), ty(b));
                let sa = sign_extend(v(a), res_ty);
                let sb = sign_extend(vb, res_ty);
                Ok(Some(InstResult::Val(mask(
                    (sa.wrapping_div(sb)) as u64,
                    res_ty,
                ))))
            }
            Inst::UDiv(a, b) => {
                let vb = v(b);
                if vb == 0 {
                    return Err(InterpError::DivideByZero);
                }
                let res_ty = arith_result_type(ty(a), ty(b));
                Ok(Some(InstResult::Val(mask(v(a).wrapping_div(vb), res_ty))))
            }

            // Float arithmetic
            Inst::FAdd(a, b) => {
                let fa = f64::from_bits(v(a));
                let fb = f64::from_bits(v(b));
                Ok(Some(InstResult::Val((fa + fb).to_bits())))
            }
            Inst::FSub(a, b) => {
                let fa = f64::from_bits(v(a));
                let fb = f64::from_bits(v(b));
                Ok(Some(InstResult::Val((fa - fb).to_bits())))
            }
            Inst::FMul(a, b) => {
                let fa = f64::from_bits(v(a));
                let fb = f64::from_bits(v(b));
                Ok(Some(InstResult::Val((fa * fb).to_bits())))
            }
            Inst::FDiv(a, b) => {
                let fa = f64::from_bits(v(a));
                let fb = f64::from_bits(v(b));
                Ok(Some(InstResult::Val((fa / fb).to_bits())))
            }

            // Bitwise
            Inst::And(a, b) => Ok(Some(InstResult::Val(v(a) & v(b)))),
            Inst::Or(a, b) => Ok(Some(InstResult::Val(v(a) | v(b)))),
            Inst::Xor(a, b) => Ok(Some(InstResult::Val(mask(v(a) ^ v(b), ty(a))))),
            Inst::Shl(a, b) => Ok(Some(InstResult::Val(mask(v(a) << (v(b) & 63), ty(a))))),
            Inst::LShr(a, b) => Ok(Some(InstResult::Val(mask(v(a) >> (v(b) & 63), ty(a))))),
            Inst::AShr(a, b) => {
                let t = ty(a);
                let sa = sign_extend(v(a), t);
                Ok(Some(InstResult::Val(mask((sa >> (v(b) & 63)) as u64, t))))
            }

            // Unary
            Inst::Neg(val) => {
                let t = ty(val);
                Ok(Some(InstResult::Val(mask(0u64.wrapping_sub(v(val)), t))))
            }
            Inst::FNeg(val) => {
                let f = f64::from_bits(v(val));
                Ok(Some(InstResult::Val((-f).to_bits())))
            }
            Inst::Not(val) => {
                let t = ty(val);
                Ok(Some(InstResult::Val(mask(!v(val), t))))
            }

            // Comparison
            Inst::Icmp(op, a, b) => {
                let t = ty(a);
                let result = match op {
                    CmpOp::Eq => v(a) == v(b),
                    CmpOp::Ne => v(a) != v(b),
                    CmpOp::Slt => sign_extend(v(a), t) < sign_extend(v(b), t),
                    CmpOp::Sle => sign_extend(v(a), t) <= sign_extend(v(b), t),
                    CmpOp::Sgt => sign_extend(v(a), t) > sign_extend(v(b), t),
                    CmpOp::Sge => sign_extend(v(a), t) >= sign_extend(v(b), t),
                    CmpOp::Ult => v(a) < v(b),
                    CmpOp::Ule => v(a) <= v(b),
                    CmpOp::Ugt => v(a) > v(b),
                    CmpOp::Uge => v(a) >= v(b),
                };
                Ok(Some(InstResult::Val(result as u64)))
            }
            Inst::Fcmp(op, a, b) => {
                let fa = f64::from_bits(v(a));
                let fb = f64::from_bits(v(b));
                let result = match op {
                    CmpOp::Eq => fa == fb,
                    CmpOp::Ne => fa != fb,
                    CmpOp::Slt | CmpOp::Ult => fa < fb,
                    CmpOp::Sle | CmpOp::Ule => fa <= fb,
                    CmpOp::Sgt | CmpOp::Ugt => fa > fb,
                    CmpOp::Sge | CmpOp::Uge => fa >= fb,
                };
                Ok(Some(InstResult::Val(result as u64)))
            }

            // Conversions
            Inst::Sext(val, to) => {
                let from_ty = ty(val);
                let extended = sign_extend(v(val), from_ty) as u64;
                Ok(Some(InstResult::Val(mask(extended, *to))))
            }
            Inst::Zext(val, _to) => {
                // Value is already zero-extended in u64 representation.
                Ok(Some(InstResult::Val(v(val))))
            }
            Inst::Trunc(val, to) => Ok(Some(InstResult::Val(mask(v(val), *to)))),
            Inst::IntToFloat(val) => {
                let i = v(val) as i64;
                Ok(Some(InstResult::Val((i as f64).to_bits())))
            }
            Inst::FloatToInt(val) => {
                let f = f64::from_bits(v(val));
                Ok(Some(InstResult::Val(f as i64 as u64)))
            }
            Inst::Bitcast(val, _to) => Ok(Some(InstResult::Val(v(val)))),

            // Memory
            Inst::Load(load_ty, addr, offset) => {
                let ptr = (v(addr) as isize + *offset as isize) as *const u8;
                let result = unsafe {
                    match load_ty.size_bytes() {
                        1 => std::ptr::read_unaligned(ptr) as u64,
                        4 => std::ptr::read_unaligned(ptr as *const u32) as u64,
                        8 => std::ptr::read_unaligned(ptr as *const u64),
                        _ => unreachable!(),
                    }
                };
                Ok(Some(InstResult::Val(result)))
            }
            Inst::Store(val, addr, offset) => {
                let ptr = (v(addr) as isize + *offset as isize) as *mut u8;
                unsafe {
                    match self.func.value_type(*val).size_bytes() {
                        1 => std::ptr::write_unaligned(ptr, v(val) as u8),
                        4 => std::ptr::write_unaligned(ptr as *mut u32, v(val) as u32),
                        8 => std::ptr::write_unaligned(ptr as *mut u64, v(val)),
                        _ => unreachable!(),
                    }
                }
                Ok(None)
            }

            // Tagged values
            Inst::TagOf(val) => Ok(Some(InstResult::Val((v(val) >> 48) as u64))),
            Inst::Payload(val) => Ok(Some(InstResult::Val(v(val) & ((1u64 << 48) - 1)))),
            Inst::MakeTagged(tag, payload) => {
                let tagged = ((*tag as u64) << 48) | (v(payload) & ((1u64 << 48) - 1));
                Ok(Some(InstResult::Val(tagged)))
            }
            Inst::IsTag(val, tag) => {
                let actual_tag = (v(val) >> 48) as u32;
                Ok(Some(InstResult::Val((actual_tag == *tag) as u64)))
            }

            // Select
            Inst::Select(cond, t, f) => {
                Ok(Some(InstResult::Val(if v(cond) != 0 { v(t) } else { v(f) })))
            }

            // Overflow checking
            Inst::OverflowCheck(op, a, b) => {
                let va = v(a);
                let vb = v(b);
                let t = ty(a);
                let overflowed = match (op, t) {
                    (OverflowOp::SAdd, Type::I8) => {
                        (va as u8 as i8).overflowing_add(vb as u8 as i8).1
                    }
                    (OverflowOp::SAdd, Type::I32) => {
                        (va as u32 as i32).overflowing_add(vb as u32 as i32).1
                    }
                    (OverflowOp::SAdd, _) => (va as i64).overflowing_add(vb as i64).1,
                    (OverflowOp::SSub, Type::I8) => {
                        (va as u8 as i8).overflowing_sub(vb as u8 as i8).1
                    }
                    (OverflowOp::SSub, Type::I32) => {
                        (va as u32 as i32).overflowing_sub(vb as u32 as i32).1
                    }
                    (OverflowOp::SSub, _) => (va as i64).overflowing_sub(vb as i64).1,
                    (OverflowOp::SMul, Type::I8) => {
                        (va as u8 as i8).overflowing_mul(vb as u8 as i8).1
                    }
                    (OverflowOp::SMul, Type::I32) => {
                        (va as u32 as i32).overflowing_mul(vb as u32 as i32).1
                    }
                    (OverflowOp::SMul, _) => (va as i64).overflowing_mul(vb as i64).1,
                    (OverflowOp::UAdd, Type::I8) => (va as u8).overflowing_add(vb as u8).1,
                    (OverflowOp::UAdd, Type::I32) => {
                        (va as u32).overflowing_add(vb as u32).1
                    }
                    (OverflowOp::UAdd, _) => va.overflowing_add(vb).1,
                    (OverflowOp::USub, Type::I8) => (va as u8).overflowing_sub(vb as u8).1,
                    (OverflowOp::USub, Type::I32) => {
                        (va as u32).overflowing_sub(vb as u32).1
                    }
                    (OverflowOp::USub, _) => va.overflowing_sub(vb).1,
                    (OverflowOp::UMul, Type::I8) => (va as u8).overflowing_mul(vb as u8).1,
                    (OverflowOp::UMul, Type::I32) => {
                        (va as u32).overflowing_mul(vb as u32).1
                    }
                    (OverflowOp::UMul, _) => va.overflowing_mul(vb).1,
                };
                Ok(Some(InstResult::Val(overflowed as u64)))
            }

            // Guard
            Inst::Guard(cond, deopt_id, live) => {
                if v(cond) != 0 {
                    // Guard passes — continue.
                    Ok(None)
                } else {
                    let info = &self.func.deopt_info[deopt_id.index()];
                    let live_values: Vec<u64> = live.iter().map(|val| v(val)).collect();
                    Ok(Some(InstResult::Deopt {
                        deopt_id: *deopt_id,
                        resume_point: info.resume_point,
                        live_values,
                    }))
                }
            }

            // Calls
            Inst::Call(fref, args) => {
                let arg_vals: Vec<u64> = args.iter().map(|val| v(val)).collect();
                match self.call_extern(*fref, &arg_vals)? {
                    ExternCallResult::Value(ret) => {
                        Ok(ret.map(InstResult::Val))
                    }
                    ExternCallResult::Exception(exc) => {
                        Err(InterpError::UncaughtException(exc))
                    }
                }
            }
            // Safepoint — no-op in the interpreter (no GC to run).
            Inst::Safepoint(_) => Ok(None),

            Inst::CallIndirect(callee, args, _ret_ty) => {
                let callee_val = v(callee);
                let arg_vals: Vec<u64> = args.iter().map(|val| v(val)).collect();
                let handler = self.indirect_handler.as_ref().ok_or_else(|| {
                    InterpError::UnknownExternFunc("(indirect)".to_string())
                })?;
                match handler(callee_val, &arg_vals) {
                    ExternCallResult::Value(ret) => {
                        Ok(ret.map(InstResult::Val))
                    }
                    ExternCallResult::Exception(exc) => {
                        Err(InterpError::UncaughtException(exc))
                    }
                }
            }
        }
    }
}

enum InstResult {
    Val(u64),
    Deopt {
        deopt_id: DeoptId,
        resume_point: u64,
        live_values: Vec<u64>,
    },
}

/// Transfer block arguments: collect values, then write to target block params.
fn transfer_args(vals: &mut [u64], target: BlockId, args: &[Value], func: &Function) {
    let arg_vals: Vec<u64> = args.iter().map(|v| vals[v.index()]).collect();
    let target_block = &func.blocks[target.index()];
    for (i, val) in arg_vals.iter().enumerate() {
        vals[target_block.params[i].0.index()] = *val;
    }
}

/// Mask a u64 to the correct width for the given type.
fn mask(val: u64, ty: Type) -> u64 {
    match ty {
        Type::I8 => val & 0xFF,
        Type::I32 => val & 0xFFFF_FFFF,
        Type::I64 | Type::Ptr | Type::GcPtr | Type::F64 => val,
    }
}

/// Sign-extend a u64 based on source type.
fn sign_extend(val: u64, ty: Type) -> i64 {
    match ty {
        Type::I8 => val as u8 as i8 as i64,
        Type::I32 => val as u32 as i32 as i64,
        Type::I64 | Type::Ptr | Type::GcPtr => val as i64,
        Type::F64 => val as i64,
    }
}

/// Compute the result type for integer arithmetic (GcPtr > Ptr > int).
fn arith_result_type(a: Type, b: Type) -> Type {
    if a == Type::GcPtr || b == Type::GcPtr {
        Type::GcPtr
    } else if a == Type::Ptr || b == Type::Ptr {
        Type::Ptr
    } else {
        a
    }
}
