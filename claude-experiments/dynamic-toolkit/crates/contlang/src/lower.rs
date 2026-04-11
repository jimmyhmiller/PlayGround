use std::collections::HashMap;

use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{FuncRef, Module, PromptId, Value};
use dynir::types::Type;

use crate::parser::{BinOp, Expr, Program, Stmt, ValType};

pub struct LoweredProgram {
    pub module: Module,
    pub entry: FuncRef,
    pub func_refs: HashMap<String, FuncRef>,
}

struct FuncLowerer<'a> {
    fb: FunctionBuilder,
    vars: Vec<HashMap<String, Value>>,
    func_refs: &'a HashMap<String, FuncRef>,
    prompt_stack: Vec<PromptId>,
    foreign_prompt: Option<PromptId>,
    dead: bool,
    /// Return type of the enclosing function (for generating typed dummies).
    ret_ty: Type,
}

impl<'a> FuncLowerer<'a> {
    fn new(fb: FunctionBuilder, func_refs: &'a HashMap<String, FuncRef>, ret_ty: Type) -> Self {
        FuncLowerer {
            fb,
            vars: vec![HashMap::new()],
            func_refs,
            prompt_stack: Vec::new(),
            foreign_prompt: None,
            dead: false,
            ret_ty,
        }
    }

    fn push_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.vars.pop();
    }

    fn def_var(&mut self, name: &str, val: Value) {
        self.vars.last_mut().unwrap().insert(name.to_string(), val);
    }

    fn lookup_var(&self, name: &str) -> Value {
        for scope in self.vars.iter().rev() {
            if let Some(&v) = scope.get(name) {
                return v;
            }
        }
        panic!("undefined variable: {}", name);
    }

    fn set_var(&mut self, name: &str, val: Value) {
        for scope in self.vars.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return;
            }
        }
        panic!("undefined variable for assignment: {}", name);
    }

    fn current_prompt(&mut self) -> PromptId {
        if let Some(&prompt) = self.prompt_stack.last() {
            return prompt;
        }
        if self.foreign_prompt.is_none() {
            self.foreign_prompt = Some(self.fb.create_prompt());
        }
        self.foreign_prompt.unwrap()
    }

    fn enter_dead_block(&mut self) -> Value {
        self.dead = true;
        let dead_bb = self.fb.create_block(&[]);
        self.fb.switch_to_block(dead_bb);
        self.fb.iconst(Type::I64, 0)
    }

    fn lower_expr(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::Int(n) => self.fb.iconst(Type::I64, *n),
            Expr::Bool(b) => self.fb.iconst(Type::I64, if *b { 1 } else { 0 }),
            Expr::Var(name) => self.lookup_var(name),
            Expr::UnaryNeg(e) => {
                let v = self.lower_expr(e);
                let zero = self.fb.iconst(Type::I64, 0);
                self.fb.sub(zero, v)
            }
            Expr::BinOp(lhs, op, rhs) => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match op {
                    BinOp::Add => self.fb.add(l, r),
                    BinOp::Sub => self.fb.sub(l, r),
                    BinOp::Mul => self.fb.mul(l, r),
                    BinOp::Div => self.fb.sdiv(l, r),
                    BinOp::Mod => self.fb.sdiv(l, r), // TODO: srem
                    BinOp::Eq => {
                        let cmp = self.fb.icmp(dynir::ir::CmpOp::Eq, l, r);
                        self.fb.zext(cmp, Type::I64)
                    }
                    BinOp::Ne => {
                        let cmp = self.fb.icmp(dynir::ir::CmpOp::Ne, l, r);
                        self.fb.zext(cmp, Type::I64)
                    }
                    BinOp::Lt => {
                        let cmp = self.fb.icmp(dynir::ir::CmpOp::Slt, l, r);
                        self.fb.zext(cmp, Type::I64)
                    }
                    BinOp::Le => {
                        let cmp = self.fb.icmp(dynir::ir::CmpOp::Sle, l, r);
                        self.fb.zext(cmp, Type::I64)
                    }
                    BinOp::Gt => {
                        let cmp = self.fb.icmp(dynir::ir::CmpOp::Sgt, l, r);
                        self.fb.zext(cmp, Type::I64)
                    }
                    BinOp::Ge => {
                        let cmp = self.fb.icmp(dynir::ir::CmpOp::Sge, l, r);
                        self.fb.zext(cmp, Type::I64)
                    }
                }
            }
            Expr::Call(name, args) => {
                if name == "abort" {
                    assert_eq!(args.len(), 1, "abort() takes exactly 1 argument");
                    let val = self.lower_expr(&args[0]);
                    let prompt = self.current_prompt();
                    self.fb.abort_to_prompt(prompt, &[val]);
                    return self.enter_dead_block();
                }

                let func_ref = *self.func_refs.get(name.as_str())
                    .unwrap_or_else(|| panic!("undefined function: {}", name));
                let arg_vals: Vec<Value> = args.iter().map(|a| self.lower_expr(a)).collect();
                self.fb.call(func_ref, &arg_vals).unwrap()
            }
            Expr::If(cond, then_body, else_body) => {
                let c = self.lower_expr(cond);
                let cond_i8 = self.fb.trunc(c, Type::I8);

                let then_bb = self.fb.create_block(&[]);
                let else_bb = self.fb.create_block(&[]);
                let merge_bb = self.fb.create_block(&[Type::I64]);

                self.fb.br_if(cond_i8, then_bb, &[], else_bb, &[]);

                self.fb.switch_to_block(then_bb);
                self.dead = false;
                self.push_scope();
                let then_val = self.lower_expr(then_body);
                self.pop_scope();
                let then_dead = self.dead;
                if !then_dead {
                    self.fb.jump(merge_bb, &[then_val]);
                } else {
                    self.fb.unreachable();
                }

                self.fb.switch_to_block(else_bb);
                self.dead = false;
                let else_val = if let Some(eb) = else_body {
                    self.push_scope();
                    let v = self.lower_expr(eb);
                    self.pop_scope();
                    v
                } else {
                    self.fb.iconst(Type::I64, 0)
                };
                let else_dead = self.dead;
                if !else_dead {
                    self.fb.jump(merge_bb, &[else_val]);
                } else {
                    self.fb.unreachable();
                }

                self.dead = then_dead && else_dead;
                self.fb.switch_to_block(merge_bb);
                self.fb.block_param(merge_bb, 0)
            }
            Expr::While(cond, body) => {
                let header_bb = self.fb.create_block(&[]);
                let body_bb = self.fb.create_block(&[]);
                let exit_bb = self.fb.create_block(&[]);

                self.fb.jump(header_bb, &[]);

                self.fb.switch_to_block(header_bb);
                let c = self.lower_expr(cond);
                let c8 = self.fb.trunc(c, Type::I8);
                self.fb.br_if(c8, body_bb, &[], exit_bb, &[]);

                self.fb.switch_to_block(body_bb);
                self.push_scope();
                let _ = self.lower_expr(body);
                self.pop_scope();
                self.fb.jump(header_bb, &[]);

                self.fb.switch_to_block(exit_bb);
                self.fb.iconst(Type::I64, 0)
            }
            Expr::Block(stmts, tail) => {
                self.push_scope();
                for stmt in stmts {
                    self.lower_stmt(stmt);
                }
                let result = if let Some(t) = tail {
                    self.lower_expr(t)
                } else {
                    self.fb.iconst(Type::I64, 0)
                };
                self.pop_scope();
                result
            }
            Expr::Reset(body) => {
                // Create handler block for abort landing
                let handler_bb = self.fb.create_block(&[self.ret_ty]);
                let prompt = self.fb.create_prompt();
                self.fb.push_prompt(prompt, handler_bb);
                self.prompt_stack.push(prompt);

                let result = self.lower_expr(body);

                self.prompt_stack.pop();
                if !self.dead {
                    self.fb.pop_prompt(prompt);
                    // Normal path: jump to handler block with body result
                    self.fb.jump(handler_bb, &[result]);
                } else {
                    // Dead path: unreachable (abort will jump to handler_bb directly)
                    self.fb.unreachable();
                }

                // Handler block: merge point for normal flow and abort flow
                self.fb.switch_to_block(handler_bb);
                self.dead = false;
                self.fb.block_param(handler_bb, 0)
            }
            Expr::Capture => {
                let prompt = self.current_prompt();
                let live_vals: Vec<Value> = self.vars.iter()
                    .flat_map(|scope| scope.values().copied())
                    .collect();
                self.fb.capture_slice(prompt, &live_vals)
            }
            Expr::ShiftBind { binder, handler } => {
                // shift |k| { handler_body } — Racket-style delimited control.
                //
                // Lowered as a two-successor terminator:
                //   - handler_bb (takes [FrameSlice]): runs at capture time
                //     with `binder` bound to the fresh handle. Expected to
                //     abort out of the enclosing reset.
                //   - resume_bb (takes [I64]): runs at resume time. The
                //     block param *is* the value of the whole shift
                //     expression.
                let prompt = self.current_prompt();
                let handler_bb = self.fb.create_block(&[Type::FrameSlice]);
                let resume_bb = self.fb.create_block(&[Type::I64]);
                self.fb.capture_slice_term(prompt, handler_bb, resume_bb);

                // Capture-time path: bind the handle and lower the handler.
                self.fb.switch_to_block(handler_bb);
                let k_val = self.fb.block_param(handler_bb, 0);
                self.push_scope();
                self.def_var(binder, k_val);
                let body_result = self.lower_expr(handler);
                if self.dead {
                    // Handler body already aborted or otherwise terminated;
                    // we're in a dead block. Emit `unreachable` to give the
                    // block a terminator so the builder is happy.
                    self.fb.unreachable();
                } else {
                    // Handler body returned normally. Its value becomes
                    // the value of the whole reset expression — semantics
                    // match Racket: `(reset (+ e (shift k body)))` where
                    // body doesn't invoke k evaluates to body. Route via
                    // abort_to_prompt so the value lands at the reset's
                    // handler block.
                    self.fb.abort_to_prompt(prompt, &[body_result]);
                }
                self.pop_scope();
                self.dead = false;

                // Resume-time path: switch to resume_bb; the block param is
                // the shift expression's value.
                self.fb.switch_to_block(resume_bb);
                self.fb.block_param(resume_bb, 0)
            }
            Expr::Abort(val) => {
                let v = self.lower_expr(val);
                let prompt = self.current_prompt();
                self.fb.abort_to_prompt(prompt, &[v]);
                self.enter_dead_block()
            }
            Expr::Resume(cont, val) => {
                let k = self.lower_expr(cont);
                let v = self.lower_expr(val);
                // The captured computation will eventually return a value
                // (either by the reset's body returning normally or by
                // abort-to-prompt + handler Ret). Control lands in
                // `cont_bb` with that value as its first block param.
                let cont_bb = self.fb.create_block(&[Type::I64]);
                self.fb.resume_slice(k, &[v], cont_bb, &[]);
                self.fb.switch_to_block(cont_bb);
                self.fb.block_param(cont_bb, 0)
            }
            Expr::CloneCont(cont) => {
                let k = self.lower_expr(cont);
                self.fb.clone_slice(k)
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(name, expr) => {
                let val = self.lower_expr(expr);
                self.def_var(name, val);
            }
            Stmt::Assign(name, expr) => {
                let val = self.lower_expr(expr);
                self.set_var(name, val);
            }
            Stmt::Expr(expr) => {
                let _ = self.lower_expr(expr);
            }
            Stmt::Return(expr) => {
                let val = self.lower_expr(expr);
                self.fb.ret(val);
                self.enter_dead_block();
            }
        }
    }
}

fn val_type_to_ir(ty: ValType) -> Type {
    match ty {
        ValType::Int => Type::I64,
        ValType::Cont => Type::FrameSlice,
    }
}

pub fn lower_program(program: &Program) -> LoweredProgram {
    let mut mb = ModuleBuilder::new();

    // Phase 1: Declare user functions
    let mut func_refs = HashMap::new();
    for decl in &program.decls {
        let params: Vec<Type> = decl.params.iter().map(|p| val_type_to_ir(p.ty)).collect();
        let ret_ty = Some(val_type_to_ir(decl.ret_ty));
        let fref = mb.declare_func(&decl.name, &params, ret_ty);
        func_refs.insert(decl.name.clone(), fref);
    }

    // Phase 2: Define functions
    for decl in &program.decls {
        let fref = func_refs[&decl.name];
        let fb = mb.define_func(fref);

        let ret_ty = val_type_to_ir(decl.ret_ty);
        let mut lowerer = FuncLowerer::new(fb, &func_refs, ret_ty);

        let entry = lowerer.fb.entry_block();
        for (i, param) in decl.params.iter().enumerate() {
            let val = lowerer.fb.block_param(entry, i);
            lowerer.def_var(&param.name, val);
        }

        let result = lowerer.lower_expr(&decl.body);
        if lowerer.dead {
            lowerer.fb.unreachable();
        } else {
            lowerer.fb.ret(result);
        }

        mb.finish_func(fref, lowerer.fb);
    }

    let module = mb.build();
    let entry = *func_refs.get(&program.decls.last().unwrap().name)
        .expect("no functions in program");

    LoweredProgram { module, entry, func_refs }
}
