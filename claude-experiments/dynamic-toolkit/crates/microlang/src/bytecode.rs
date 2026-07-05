//! The capstone: a real EMIT tier. A bytecode compiler + stack VM — a distinct
//! compile phase producing an instruction stream that a dispatch loop executes,
//! genuinely different from the tree-walker and the closure-compiler.
//!
//! Its purpose is to make the "emit half" concrete. Until now every axis
//! supplied only a COMPUTE form (run now on a value). Here the **value model
//! supplies an EMIT form** (`ModelEmit`): given an op buffer, it appends the
//! bytecode for `+`, `-`, `*`, `<`. And the emitted code DIFFERS by
//! representation, from one source:
//!
//!   * `LowBit`  — tag in the low 3 bits, so a fixnum is `value << 3`. Two of
//!     them add directly (`AddRaw`); multiply untags one first (`Sar 3; MulRaw`).
//!   * `HighBit` — tag in the top bits, value unshifted in the low 61, so even
//!     multiply is a bare `MulRaw` (no shift). The simplest emission.
//!   * `NanBox`  — integers are boxed, so integer arithmetic emits a slow-path
//!     runtime call (`Slow`), mirroring the boxing the `calc` example measured,
//!     now visible in the generated code.
//!
//! The compiler is generic over `M: ModelEmit`; swapping the representation
//! changes the bytecode and nothing else. This is the emit-interface boundary
//! the codegen-axes doc promised, realized on the foundational value axis. A
//! machine-code tier is the same shape with a real ISA instead of these ops;
//! dispatch/GC/speculation each extend it with their own emit forms.
//!
//! Scope: this tier covers arithmetic, control flow, calls/recursion, closures,
//! globals, and list prims (via a `Slow` runtime escape). It does not cover
//! `let`, records/dispatch, or `(gc)`; those error clearly and run on the
//! tree-walker. It is a focused demonstration of the emit interface, not a
//! complete backend.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::code::CodeSpace;
use crate::ir::{ConstId, Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::{Runtime, Var};
use crate::value::{frame_get, Locals, Obj, Sym, Val};

/// A bytecode instruction. The operand stack holds tagged values; the
/// model-emitted arithmetic ops (`AddRaw`, `Sar`, ...) operate on the raw bits
/// with the tag layout baked into the emitted SEQUENCE, not into the op.
#[derive(Clone)]
pub enum Op {
    Const(ConstId),
    Nil,
    LoadLocal(u16, u16),
    LoadGlobal(Sym),
    MakeClosure {
        nparams: usize,
        variadic: bool,
        body: Rc<Ir>,
    },
    DefGlobal(Sym, bool),
    Pop,
    Jump(usize),
    BrFalse(usize),
    Truthy,  // pop tagged -> push raw 0/1
    EncBool, // pop raw 0/1 -> push tagged bool
    Call(u8),
    Ret,
    // ── model-emitted arithmetic (the star of the capstone) ──
    AddRaw,
    SubRaw,
    MulRaw,
    Sar(u32),   // arithmetic shift right by a constant (untag)
    CmpLtRaw,   // pop2 i64 -> push raw 0/1
    // ── runtime escape for everything else ──
    Slow(Prim, u8),
}

pub struct Chunk {
    pub ops: Vec<Op>,
}

/// The value model's EMIT half: append the bytecode for a primitive op. This is
/// the interface a JIT would implement with real machine instructions.
pub trait ModelEmit: ValueModel {
    fn emit_add(ops: &mut Vec<Op>);
    fn emit_sub(ops: &mut Vec<Op>);
    fn emit_mul(ops: &mut Vec<Op>);
    /// Must leave a TAGGED bool on the stack (uniform with every expression).
    fn emit_lt(ops: &mut Vec<Op>);
}

impl ModelEmit for crate::model::LowBitModel {
    fn emit_add(ops: &mut Vec<Op>) {
        ops.push(Op::AddRaw); // (x<<3)+(y<<3) == (x+y)<<3
    }
    fn emit_sub(ops: &mut Vec<Op>) {
        ops.push(Op::SubRaw);
    }
    fn emit_mul(ops: &mut Vec<Op>) {
        ops.push(Op::Sar(3)); // untag the top operand
        ops.push(Op::MulRaw); // (x<<3)*y == (x*y)<<3
    }
    fn emit_lt(ops: &mut Vec<Op>) {
        ops.push(Op::CmpLtRaw);
        ops.push(Op::EncBool);
    }
}

impl ModelEmit for crate::model::HighBitModel {
    fn emit_add(ops: &mut Vec<Op>) {
        ops.push(Op::AddRaw); // value unshifted in low bits, tag 0 in high bits
    }
    fn emit_sub(ops: &mut Vec<Op>) {
        ops.push(Op::SubRaw);
    }
    fn emit_mul(ops: &mut Vec<Op>) {
        ops.push(Op::MulRaw); // no shift needed — the simplest emission
    }
    fn emit_lt(ops: &mut Vec<Op>) {
        ops.push(Op::CmpLtRaw);
        ops.push(Op::EncBool);
    }
}

impl ModelEmit for crate::model::NanBoxModel {
    fn emit_add(ops: &mut Vec<Op>) {
        ops.push(Op::Slow(Prim::Add, 2)); // integers are boxed -> slow path
    }
    fn emit_sub(ops: &mut Vec<Op>) {
        ops.push(Op::Slow(Prim::Sub, 2));
    }
    fn emit_mul(ops: &mut Vec<Op>) {
        ops.push(Op::Slow(Prim::Mul, 2));
    }
    fn emit_lt(ops: &mut Vec<Op>) {
        ops.push(Op::Slow(Prim::Lt, 2)); // already a tagged bool
    }
}

/// The bytecode execution tier.
pub struct BytecodeVm<M: ModelEmit> {
    chunks: RefCell<HashMap<*const Ir, Rc<Chunk>>>,
    _pd: std::marker::PhantomData<fn() -> M>,
}

impl<M: ModelEmit> BytecodeVm<M> {
    pub fn new() -> Self {
        BytecodeVm {
            chunks: RefCell::new(HashMap::new()),
            _pd: std::marker::PhantomData,
        }
    }

    /// Compile one expression tree into a standalone, runnable chunk.
    pub fn compile_expr_chunk(&self, ir: &Ir) -> Chunk {
        let mut ops = Vec::new();
        Self::compile(ir, &mut ops);
        ops.push(Op::Ret);
        Chunk { ops }
    }

    /// Human-readable listing, for showing that the emitted code differs by model.
    pub fn disassemble(ir: &Ir) -> Vec<String> {
        let mut ops = Vec::new();
        Self::compile(ir, &mut ops);
        ops.push(Op::Ret);
        ops.iter().map(op_name).collect()
    }

    fn compiled_body(&self, body: &Rc<Ir>) -> Rc<Chunk> {
        let key = Rc::as_ptr(body);
        if let Some(c) = self.chunks.borrow().get(&key) {
            return c.clone();
        }
        let c = Rc::new(self.compile_expr_chunk(body));
        self.chunks.borrow_mut().insert(key, c.clone());
        c
    }

    fn compile(ir: &Ir, ops: &mut Vec<Op>) {
        match ir {
            Ir::Const(id) | Ir::Quote(id) => ops.push(Op::Const(*id)),
            Ir::Local { up, idx } => ops.push(Op::LoadLocal(*up, *idx)),
            Ir::Global(s) => ops.push(Op::LoadGlobal(*s)),
            Ir::Do(xs) => {
                if xs.is_empty() {
                    ops.push(Op::Nil);
                } else {
                    for (i, x) in xs.iter().enumerate() {
                        Self::compile(x, ops);
                        if i + 1 < xs.len() {
                            ops.push(Op::Pop);
                        }
                    }
                }
            }
            Ir::If(c, t, e) => {
                Self::compile(c, ops);
                ops.push(Op::Truthy);
                let bf = ops.len();
                ops.push(Op::BrFalse(0));
                Self::compile(t, ops);
                let jmp = ops.len();
                ops.push(Op::Jump(0));
                let else_addr = ops.len();
                ops[bf] = Op::BrFalse(else_addr);
                Self::compile(e, ops);
                let end = ops.len();
                ops[jmp] = Op::Jump(end);
            }
            Ir::Def { name, init, is_macro } => {
                Self::compile(init, ops);
                ops.push(Op::DefGlobal(*name, *is_macro));
            }
            Ir::Lambda { nparams, variadic, body } => ops.push(Op::MakeClosure {
                nparams: *nparams,
                variadic: *variadic,
                body: body.clone(),
            }),
            Ir::Call(f, args) => {
                Self::compile(f, ops);
                for a in args {
                    Self::compile(a, ops);
                }
                ops.push(Op::Call(args.len() as u8));
            }
            Ir::Prim(Prim::Add, a) => Self::binop(a, ops, M::emit_add),
            Ir::Prim(Prim::Sub, a) => Self::binop(a, ops, M::emit_sub),
            Ir::Prim(Prim::Mul, a) => Self::binop(a, ops, M::emit_mul),
            Ir::Prim(Prim::Lt, a) => Self::binop(a, ops, M::emit_lt),
            Ir::Prim(Prim::Gc, _) => {
                panic!("bytecode tier: (gc) is a safepoint not modeled here; run it on the tree-walker")
            }
            Ir::Prim(Prim::CallEc, _) => {
                panic!("bytecode tier: escape continuations not supported; run on the tree-walker")
            }
            Ir::Prim(p, args) => {
                for a in args {
                    Self::compile(a, ops);
                }
                ops.push(Op::Slow(*p, args.len() as u8));
            }
            Ir::Let(..) => panic!("bytecode tier: `let` not supported; run on the tree-walker"),
            Ir::SetLocal { .. } | Ir::SetGlobal { .. } => {
                panic!("bytecode tier: `set!` not supported; run on the tree-walker")
            }
            Ir::DefMethod { .. } | Ir::Dispatch { .. } => {
                panic!("bytecode tier: dispatch not supported; run on the tree-walker")
            }
        }
    }

    fn binop(args: &[Ir], ops: &mut Vec<Op>, emit: fn(&mut Vec<Op>)) {
        Self::compile(&args[0], ops);
        Self::compile(&args[1], ops);
        emit(ops);
    }

    fn run(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, chunk: &Chunk, locals: &Locals) -> u64 {
        let mut stack: Vec<u64> = Vec::new();
        let mut pc = 0usize;
        loop {
            match &chunk.ops[pc] {
                Op::Const(id) => stack.push(rt.get_const(*id)),
                Op::Nil => stack.push(M::R::enc_nil()),
                Op::LoadLocal(up, idx) => stack.push(frame_get(locals, *up, *idx)),
                Op::LoadGlobal(s) => {
                    let v = rt
                        .globals
                        .get(s)
                        .unwrap_or_else(|| panic!("Unable to resolve symbol: {}", rt.sym_name(*s)))
                        .val;
                    stack.push(v);
                }
                Op::MakeClosure { nparams, variadic, body } => {
                    let id = rt.alloc(Obj::Closure {
                        nparams: *nparams,
                        variadic: *variadic,
                        body: body.clone(),
                        env: locals.clone(),
                    });
                    stack.push(M::R::enc_ref(id));
                }
                Op::DefGlobal(name, is_macro) => {
                    let v = *stack.last().expect("def: empty stack");
                    rt.globals.insert(*name, Var { val: v, is_macro: *is_macro });
                }
                Op::Pop => {
                    stack.pop();
                }
                Op::Jump(t) => {
                    pc = *t;
                    continue;
                }
                Op::BrFalse(t) => {
                    if stack.pop().expect("brfalse: empty stack") == 0 {
                        pc = *t;
                        continue;
                    }
                }
                Op::Truthy => {
                    let v = stack.pop().unwrap();
                    stack.push(if M::truthy(rt.decode(v)) { 1 } else { 0 });
                }
                Op::EncBool => {
                    let r = stack.pop().unwrap();
                    stack.push(M::R::enc_bool(r != 0));
                }
                Op::Call(argc) => {
                    let n = *argc as usize;
                    let args = stack.split_off(stack.len() - n);
                    let callee = stack.pop().expect("call: empty stack");
                    let r = top.invoke(top, rt, callee, &args);
                    stack.push(r);
                }
                Op::Ret => return stack.pop().unwrap_or_else(M::R::enc_nil),
                Op::AddRaw => {
                    let b = stack.pop().unwrap() as i64;
                    let a = stack.pop().unwrap() as i64;
                    stack.push(a.wrapping_add(b) as u64);
                }
                Op::SubRaw => {
                    let b = stack.pop().unwrap() as i64;
                    let a = stack.pop().unwrap() as i64;
                    stack.push(a.wrapping_sub(b) as u64);
                }
                Op::MulRaw => {
                    let b = stack.pop().unwrap() as i64;
                    let a = stack.pop().unwrap() as i64;
                    stack.push(a.wrapping_mul(b) as u64);
                }
                Op::Sar(n) => {
                    let x = stack.pop().unwrap() as i64;
                    stack.push((x >> *n) as u64);
                }
                Op::CmpLtRaw => {
                    let b = stack.pop().unwrap() as i64;
                    let a = stack.pop().unwrap() as i64;
                    stack.push((a < b) as u64);
                }
                Op::Slow(prim, argc) => {
                    let n = *argc as usize;
                    let args = stack.split_off(stack.len() - n);
                    let r = rt.prim(*prim, &args);
                    stack.push(r);
                }
            }
            pc += 1;
        }
    }
}

impl<M: ModelEmit> CodeSpace<M> for BytecodeVm<M> {
    fn eval_ir(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64 {
        let chunk = self.compile_expr_chunk(ir);
        self.run(top, rt, &chunk, locals)
    }

    fn invoke(&self, top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        let Val::Ref(id) = rt.decode(callee) else {
            panic!("value not callable: {}", rt.print(callee));
        };
        let (nparams, variadic, body, env) = match &rt.heap[id as usize] {
            Obj::Closure { nparams, variadic, body, env } => {
                (*nparams, *variadic, body.clone(), env.clone())
            }
            _ => panic!("value not callable: {}", rt.print(callee)),
        };
        let frame = rt.build_call_frame(nparams, variadic, args, env);
        let chunk = self.compiled_body(&body);
        self.run(top, rt, &chunk, &frame)
    }
}

fn op_name(op: &Op) -> String {
    match op {
        Op::Const(id) => format!("Const #{id}"),
        Op::Nil => "Nil".into(),
        Op::LoadLocal(u, i) => format!("LoadLocal {u},{i}"),
        Op::LoadGlobal(s) => format!("LoadGlobal ${s}"),
        Op::MakeClosure { nparams, .. } => format!("MakeClosure/{nparams}"),
        Op::DefGlobal(s, _) => format!("DefGlobal ${s}"),
        Op::Pop => "Pop".into(),
        Op::Jump(t) => format!("Jump {t}"),
        Op::BrFalse(t) => format!("BrFalse {t}"),
        Op::Truthy => "Truthy".into(),
        Op::EncBool => "EncBool".into(),
        Op::Call(n) => format!("Call {n}"),
        Op::Ret => "Ret".into(),
        Op::AddRaw => "AddRaw".into(),
        Op::SubRaw => "SubRaw".into(),
        Op::MulRaw => "MulRaw".into(),
        Op::Sar(n) => format!("Sar {n}"),
        Op::CmpLtRaw => "CmpLtRaw".into(),
        Op::Slow(p, n) => format!("Slow({p:?},{n})"),
    }
}
