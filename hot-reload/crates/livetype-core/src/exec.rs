//! The one step semantics. Every instruction's behavior is defined exactly once
//! here, against the [`Machine`] interface, so the single-threaded interpreter
//! and the concurrent worker can no longer drift on what an instruction *means*.
//!
//! A `Machine` is "the current actor's execution context plus the shared world"
//! — register/pc access over the current frame, a frame stack to push/pop, the
//! shared [`Heap`] and [`World`], and the effect channels (emit, foreign calls,
//! globals, message passing). The interpreter implements it over `&mut Runtime`
//! + an actor id; each concurrent worker implements it over a thread-local frame
//! stack layered on the shared runtime. Tier differences (does this tier have
//! FFI? message passing?) are expressed by the `Machine` impl answering
//! "unsupported", which [`step_instruction`] turns into a clear trap — exactly
//! the seam Phase 6 later removes.

use crate::runtime::{
    ERR_ADD_NON_I64, ERR_BRANCH_NON_BOOL, ERR_EQ_NON_I64, ERR_LT_NON_I64, ERR_MUL_NON_I64,
    ERR_NOT_NON_BOOL, ERR_SUB_NON_I64,
};
use crate::{Condition, DefId, ForeignFnId, Heap, Instruction, Value, Version, World};

/// The outcome of executing one instruction.
pub enum Flow {
    /// The instruction ran; the machine's pc now points at the next work.
    Stepped,
    /// A blocking op (`Recv`) had nothing available yet. The driver should hit a
    /// safepoint and re-execute the same instruction. The interpreter, which has
    /// no message passing, never yields this.
    Blocked,
}

/// A message-receive attempt's result.
pub enum RecvResult {
    /// This tier has no message passing.
    Unsupported,
    /// No message available yet — the driver should retry after a safepoint.
    WouldBlock,
    Got(Value),
}

/// A foreign-call attempt's result.
pub enum ForeignCall {
    /// This tier has no FFI.
    Unsupported,
    Done(Result<Value, String>),
}

/// A global-read attempt's result.
pub enum GlobalRead {
    /// This tier has no globals.
    Unsupported,
    /// A global with this id exists nowhere / is not yet initialized.
    Missing,
    Value(Value),
}

/// The execution environment one instruction step operates against.
pub trait Machine {
    fn world(&self) -> &World;
    fn heap(&self) -> &Heap;
    /// The `(id, version)` of the function the current frame runs.
    fn current(&self) -> (DefId, Version);
    fn pc(&self) -> usize;
    /// The current frame's register `i` (`None` if never assigned).
    fn reg(&self, i: usize) -> Option<Value>;
    fn set_reg(&mut self, dst: usize, value: Value);
    fn set_pc(&mut self, pc: usize);
    /// Advance the current frame's pc by one.
    fn advance(&mut self);
    fn emit(&mut self, value: Value);
    fn global(&self, id: DefId) -> GlobalRead;
    fn call_foreign(&mut self, id: ForeignFnId, args: &[Value]) -> ForeignCall;
    /// Push a callee frame; the caller's pc has already been advanced past the
    /// `Call`. `return_reg` names the caller register the result lands in.
    fn push_call(
        &mut self,
        callee: DefId,
        version: Version,
        registers: Vec<Option<Value>>,
        return_reg: usize,
    );
    /// Pop the current frame, delivering `value` to the caller's return register,
    /// or completing the actor if this was the top frame.
    fn do_return(&mut self, value: Value);
    /// Deliver to another actor's mailbox: `None` = this tier has no message
    /// passing; `Some(false)` = no such actor; `Some(true)` = delivered.
    fn send(&mut self, target: usize, value: Value) -> Option<bool>;
    /// Try to receive a message for this actor.
    fn recv(&mut self) -> RecvResult;
}

fn type_err(function: DefId, pc: usize, message: &str) -> Condition {
    Condition::RuntimeTypeError {
        function,
        pc,
        message: message.into(),
    }
}

/// Execute one instruction against `m`. This is the single source of truth for
/// instruction semantics shared by every executor.
pub fn step_instruction<M: Machine>(m: &mut M, instr: &Instruction) -> Result<Flow, Condition> {
    let (function, _version) = m.current();
    let pc = m.pc();
    // Read a register or trap exactly as the reference interpreter does.
    let rd = |m: &M, i: usize| -> Result<Value, Condition> {
        m.reg(i)
            .ok_or_else(|| type_err(function, pc, &format!("empty r{i}")))
    };

    match instr {
        Instruction::Const { dst, value } => {
            m.set_reg(*dst, value.clone());
            m.advance();
        }
        Instruction::New {
            dst,
            type_id,
            fields,
        } => {
            let supplied: Vec<(crate::FieldId, Value)> = fields
                .iter()
                .map(|(id, reg)| Ok((*id, rd(m, *reg)?)))
                .collect::<Result<_, Condition>>()?;
            let id = m.heap().new_object(*type_id, &supplied, m.world())?;
            m.set_reg(*dst, Value::Ref(id));
            m.advance();
        }
        Instruction::GetField { dst, object, field } => {
            let Value::Ref(id) = rd(m, *object)? else {
                return Err(type_err(function, pc, "field access on non-reference"));
            };
            let value = m.heap().get_field(id, *field, m.world())?;
            m.set_reg(*dst, value);
            m.advance();
        }
        Instruction::Copy { dst, src } => {
            let value = rd(m, *src)?;
            m.set_reg(*dst, value);
            m.advance();
        }
        Instruction::AddI64 { dst, left, right } => {
            let (Value::I64(a), Value::I64(b)) = (rd(m, *left)?, rd(m, *right)?) else {
                return Err(type_err(function, pc, ERR_ADD_NON_I64));
            };
            m.set_reg(*dst, Value::I64(a + b));
            m.advance();
        }
        Instruction::SubI64 { dst, left, right } => {
            let (Value::I64(a), Value::I64(b)) = (rd(m, *left)?, rd(m, *right)?) else {
                return Err(type_err(function, pc, ERR_SUB_NON_I64));
            };
            m.set_reg(*dst, Value::I64(a - b));
            m.advance();
        }
        Instruction::MulI64 { dst, left, right } => {
            let (Value::I64(a), Value::I64(b)) = (rd(m, *left)?, rd(m, *right)?) else {
                return Err(type_err(function, pc, ERR_MUL_NON_I64));
            };
            m.set_reg(*dst, Value::I64(a * b));
            m.advance();
        }
        Instruction::LtI64 { dst, left, right } => {
            let (Value::I64(a), Value::I64(b)) = (rd(m, *left)?, rd(m, *right)?) else {
                return Err(type_err(function, pc, ERR_LT_NON_I64));
            };
            m.set_reg(*dst, Value::Bool(a < b));
            m.advance();
        }
        Instruction::EqI64 { dst, left, right } => {
            let (Value::I64(a), Value::I64(b)) = (rd(m, *left)?, rd(m, *right)?) else {
                return Err(type_err(function, pc, ERR_EQ_NON_I64));
            };
            m.set_reg(*dst, Value::Bool(a == b));
            m.advance();
        }
        Instruction::Not { dst, src } => {
            let Value::Bool(b) = rd(m, *src)? else {
                return Err(type_err(function, pc, ERR_NOT_NON_BOOL));
            };
            m.set_reg(*dst, Value::Bool(!b));
            m.advance();
        }
        Instruction::Jump { target } => m.set_pc(*target),
        Instruction::Branch {
            cond,
            then_pc,
            else_pc,
        } => {
            let Value::Bool(taken) = rd(m, *cond)? else {
                return Err(type_err(function, pc, ERR_BRANCH_NON_BOOL));
            };
            m.set_pc(if taken { *then_pc } else { *else_pc });
        }
        Instruction::Yield => m.advance(),
        Instruction::Call {
            dst,
            function: callee,
            args,
        } => {
            let version = m.world().current_functions[callee];
            let (params, registers_len) = match &m.world().functions[&(*callee, version)] {
                crate::FunctionState::Ready(f) => (f.params.clone(), f.registers),
                crate::FunctionState::Broken { diagnostics, .. } => {
                    return Err(Condition::BrokenFunction {
                        function: *callee,
                        diagnostics: diagnostics.clone(),
                    });
                }
            };
            let values: Vec<Value> = args.iter().map(|r| rd(m, *r)).collect::<Result<_, _>>()?;
            for (value, expected) in values.iter().zip(&params) {
                if !m.heap().value_ok(value, expected) {
                    return Err(type_err(
                        function,
                        pc,
                        &format!("call argument: expected {expected:?}, found a value of another type"),
                    ));
                }
            }
            let mut registers = vec![None; registers_len];
            for (slot, value) in values.into_iter().enumerate() {
                registers[slot] = Some(value);
            }
            m.advance();
            m.push_call(*callee, version, registers, *dst);
        }
        Instruction::CallForeign { dst, foreign, args } => {
            let result_ty = match m.world().foreign_sigs.get(foreign) {
                Some((_, r)) => r.clone(),
                None => return Err(type_err(function, pc, "call to unknown foreign fn")),
            };
            let values: Vec<Value> = args.iter().map(|r| rd(m, *r)).collect::<Result<_, _>>()?;
            let result = match m.call_foreign(*foreign, &values) {
                ForeignCall::Unsupported => {
                    return Err(type_err(
                        function,
                        pc,
                        "foreign calls are only available in the single-threaded (live-edit) tier",
                    ));
                }
                ForeignCall::Done(Ok(v)) => v,
                ForeignCall::Done(Err(m)) => return Err(type_err(function, pc, &m)),
            };
            if !m.heap().value_ok(&result, &result_ty) {
                return Err(type_err(
                    function,
                    pc,
                    &format!("foreign result: expected {result_ty:?}, found a value of another type"),
                ));
            }
            m.set_reg(*dst, result);
            m.advance();
        }
        Instruction::LoadGlobal { dst, global } => {
            let value = match m.global(*global) {
                GlobalRead::Value(v) => v,
                GlobalRead::Missing => {
                    return Err(type_err(function, pc, "global read before initialization"));
                }
                GlobalRead::Unsupported => {
                    return Err(type_err(
                        function,
                        pc,
                        "globals are only available in the single-threaded (live-edit) tier",
                    ));
                }
            };
            m.set_reg(*dst, value);
            m.advance();
        }
        Instruction::Emit { value } => {
            let value = rd(m, *value)?;
            m.emit(value);
            m.advance();
        }
        Instruction::Send { target, value } => {
            let Value::I64(id) = rd(m, *target)? else {
                return Err(type_err(function, pc, "send target must be an actor id"));
            };
            let payload = rd(m, *value)?;
            match m.send(id as usize, payload) {
                None => {
                    return Err(type_err(
                        function,
                        pc,
                        "message passing is only available in the concurrent runtime",
                    ));
                }
                Some(false) => return Err(type_err(function, pc, "send to an unknown actor")),
                Some(true) => m.advance(),
            }
        }
        Instruction::Recv { dst, ty } => match m.recv() {
            RecvResult::Unsupported => {
                return Err(type_err(
                    function,
                    pc,
                    "message passing is only available in the concurrent runtime",
                ));
            }
            RecvResult::WouldBlock => return Ok(Flow::Blocked),
            RecvResult::Got(message) => {
                if !m.heap().value_ok(&message, ty) {
                    return Err(type_err(function, pc, "received message has the wrong type"));
                }
                m.set_reg(*dst, message);
                m.advance();
            }
        },
        Instruction::Return { value } => {
            let result = rd(m, *value)?;
            let (function_id, version) = m.current();
            if let crate::FunctionState::Ready(f) = &m.world().functions[&(function_id, version)] {
                let result_ty = f.result.clone();
                if !m.heap().value_ok(&result, &result_ty) {
                    return Err(type_err(
                        function,
                        pc,
                        &format!("return value: expected {result_ty:?}, found a value of another type"),
                    ));
                }
            }
            m.do_return(result);
        }
    }
    Ok(Flow::Stepped)
}
