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

    /// Coerce a value to the target IR type.
    ///
    /// If the types already match, returns `v` unchanged. Otherwise
    /// emits a `bitcast` — which at runtime is a no-op but lets the
    /// IR builder see a type-consistent value. Used at merge points
    /// like reset's handler_bb (which is always typed `I64` internally
    /// so the builder's type checker is happy regardless of whether
    /// the reset body fell through as an int or aborted with a
    /// pointer) and at consumers of reset expressions where the
    /// surrounding context knows the "real" surface type.
    ///
    /// Bitcast only works for same-size types. Since every
    /// contlang-visible value is 64 bits wide (i64, cont, bytes, ptr
    /// all map to 8 bytes), this is always valid.
    fn coerce_to(&mut self, v: Value, target: Type) -> Value {
        let from = self.fb.value_type(v);
        if from == target {
            v
        } else {
            self.fb.bitcast(v, target)
        }
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
                    // Reset's handler_bb is uniformly typed I64;
                    // coerce the aborted value to match.
                    let val_i64 = self.coerce_to(val, Type::I64);
                    let prompt = self.current_prompt();
                    self.fb.abort_to_prompt(prompt, &[val_i64]);
                    return self.enter_dead_block();
                }

                let func_ref = *self
                    .func_refs
                    .get(name.as_str())
                    .unwrap_or_else(|| panic!("undefined function: {}", name));
                let arg_vals: Vec<Value> = args.iter().map(|a| self.lower_expr(a)).collect();
                // Void calls (e.g. `bytes_set`) return None from
                // `fb.call`; in expression context we produce a
                // dummy i64 0 so the caller has something to discard.
                match self.fb.call(func_ref, &arg_vals) {
                    Some(v) => v,
                    None => self.fb.iconst(Type::I64, 0),
                }
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
                // SSA-correct while lowering: every variable visible
                // in the enclosing scope at loop entry becomes a
                // header block param. On each back-edge from the body
                // end to the header, the current (possibly updated)
                // value is passed in. After the loop exits, those
                // variables are bound to the header's block params
                // (which dominate exit_bb).
                //
                // This handles loop-carried values like `let i = 0;
                // while i < 10 { i = i + 1 }` correctly — previously
                // this triggered DominanceViolation because i was
                // re-defined inside the body and its new value was
                // still referenced from the header by name.

                // Snapshot the variables visible at loop entry, in a
                // stable order. Uses an inner scope to avoid
                // duplicates when the same name is shadowed.
                let snapshot: Vec<(String, Value)> = {
                    let mut seen = HashMap::<String, Value>::new();
                    // Iterate scopes outer→inner so inner shadows win.
                    for scope in self.vars.iter() {
                        for (name, &val) in scope {
                            seen.insert(name.clone(), val);
                        }
                    }
                    seen.into_iter().collect()
                };
                let header_param_tys: Vec<Type> = snapshot
                    .iter()
                    .map(|(_, v)| self.fb.value_type(*v))
                    .collect();
                let header_bb = self.fb.create_block(&header_param_tys);
                let body_bb = self.fb.create_block(&[]);
                let exit_bb = self.fb.create_block(&[]);

                // Jump from current block to header with initial values.
                let initial_args: Vec<Value> = snapshot.iter().map(|(_, v)| *v).collect();
                self.fb.jump(header_bb, &initial_args);

                // Header: replace scope mappings with the block params.
                self.fb.switch_to_block(header_bb);
                let header_params: Vec<Value> = (0..snapshot.len())
                    .map(|i| self.fb.block_param(header_bb, i))
                    .collect();
                for (i, (name, _)) in snapshot.iter().enumerate() {
                    self.set_var(name, header_params[i]);
                }

                let c = self.lower_expr(cond);
                let c8 = self.fb.trunc(c, Type::I8);
                self.fb.br_if(c8, body_bb, &[], exit_bb, &[]);

                // Body: any `set_var` inside updates the outer scope
                // mapping for these names, which we then read back on
                // the back-edge.
                self.fb.switch_to_block(body_bb);
                self.push_scope();
                let _ = self.lower_expr(body);
                self.pop_scope();
                let back_args: Vec<Value> = snapshot
                    .iter()
                    .map(|(name, _)| self.lookup_var(name))
                    .collect();
                self.fb.jump(header_bb, &back_args);

                // Exit: the scope mappings need to point at the
                // header's block params (which dominate exit_bb),
                // NOT at whatever the last body iteration set them to
                // (which lives in body_bb).
                self.fb.switch_to_block(exit_bb);
                for (i, (name, _)) in snapshot.iter().enumerate() {
                    self.set_var(name, header_params[i]);
                }
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
                // The reset's handler block is a MERGE POINT: control
                // arrives here from two unrelated paths, the abort
                // path (whatever value was passed to abort_to_prompt
                // for this prompt) and the normal-fallthrough path
                // (whatever the reset body evaluated to). Those two
                // paths can have different IR types — e.g. the body
                // is `shift |k| { abort(k) }` where the abort path
                // contributes a FrameSlice and the fallthrough path
                // contributes the shift expression's I64 result.
                //
                // To keep the IR's static type system happy at this
                // merge, we type the handler block param uniformly
                // as `I64`. Both incoming edges coerce to I64 via
                // bitcast (a no-op at runtime — every contlang value
                // is 8 bytes wide). The reset expression's "real"
                // type for the consuming context is whatever the
                // consumer expects; consumers (let bindings with
                // type annotations, function returns, etc.) emit a
                // bitcast back to their target type if needed.
                let handler_bb = self.fb.create_block(&[Type::I64]);
                let prompt = self.fb.create_prompt();
                self.fb.push_prompt(prompt, handler_bb);
                self.prompt_stack.push(prompt);

                let result = self.lower_expr(body);

                self.prompt_stack.pop();
                if !self.dead {
                    // Normal-fallthrough path: route through
                    // abort_to_prompt instead of pop_prompt + jump.
                    // This makes BOTH paths (explicit abort AND
                    // synthetic exit) hit the interpreter's
                    // resumed-frame trampoline check, so when a
                    // captured frame's body completes naturally, it
                    // returns to the resumer instead of falling
                    // through into the resumer's IR.
                    //
                    // At capture time, the synthetic abort_to_prompt
                    // pops the prompt, finds the handler block (= bb1
                    // / `handler_bb` here, via find_handler on the
                    // PushPrompt instruction), and jumps with the
                    // value — semantically identical to the old
                    // pop_prompt + jump.
                    let result_i64 = self.coerce_to(result, Type::I64);
                    self.fb.abort_to_prompt(prompt, &[result_i64]);
                } else {
                    // Dead path: unreachable (abort will jump to handler_bb directly)
                    self.fb.unreachable();
                }

                // Handler block: merge point for normal flow and abort flow.
                // Returns the I64 block param. Consumers cast as needed.
                self.fb.switch_to_block(handler_bb);
                self.dead = false;
                self.fb.block_param(handler_bb, 0)
            }
            Expr::Capture => {
                let prompt = self.current_prompt();
                let live_vals: Vec<Value> = self
                    .vars
                    .iter()
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
                    // abort_to_prompt; the abort lowering coerces to I64.
                    let body_i64 = self.coerce_to(body_result, Type::I64);
                    self.fb.abort_to_prompt(prompt, &[body_i64]);
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
                // Reset's handler_bb is always typed I64; coerce
                // the aborted value before the merge.
                let v_i64 = self.coerce_to(v, Type::I64);
                let prompt = self.current_prompt();
                self.fb.abort_to_prompt(prompt, &[v_i64]);
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
            Stmt::Let(name, ty_ann, expr) => {
                let val = self.lower_expr(expr);
                // If the let has a type annotation, coerce the value
                // to the matching IR type. This is what makes
                // `let k: cont = reset { ... }` work even when the
                // reset expression's IR result is I64 — the cast
                // produces a FrameSlice-typed value, which goes
                // into a GC-rooted frame slot.
                let val = if let Some(ty) = ty_ann {
                    self.coerce_to(val, val_type_to_ir(*ty))
                } else {
                    val
                };
                self.def_var(name, val);
            }
            Stmt::Assign(name, expr) => {
                let val = self.lower_expr(expr);
                // Match the existing slot's type to keep IR types stable.
                let existing = self.lookup_var(name);
                let target = self.fb.value_type(existing);
                let val = self.coerce_to(val, target);
                self.set_var(name, val);
            }
            Stmt::Expr(expr) => {
                let _ = self.lower_expr(expr);
            }
            Stmt::Return(expr) => {
                let val = self.lower_expr(expr);
                let val = self.coerce_to(val, self.ret_ty);
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
        ValType::Bytes => Type::GcPtr,
    }
}

/// Names and signatures of the built-in heap externs that every
/// contlang program can call without declaring. The interpreter
/// harness must `bind` closures for each of these before running.
pub const BYTES_ALLOC: &str = "bytes_alloc";
pub const BYTES_GET: &str = "bytes_get";
pub const BYTES_SET: &str = "bytes_set";
pub const BYTES_LEN: &str = "bytes_len";

pub fn lower_program(program: &Program) -> LoweredProgram {
    use dynir::types::Signature;

    let mut mb = ModuleBuilder::new();
    let mut func_refs = HashMap::new();

    // Phase 0: Pre-declare built-in heap externs.
    //
    //   bytes_alloc(len: i64) -> bytes
    //   bytes_get(p: bytes, i: i64) -> i64
    //   bytes_set(p: bytes, i: i64, b: i64) -> void
    //   bytes_len(p: bytes) -> i64
    //
    // These are extern functions whose closures are supplied at
    // interpreter setup time (see the contlang test harness).
    let f_alloc = mb.declare_extern(
        BYTES_ALLOC,
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::GcPtr),
        },
    );
    func_refs.insert(BYTES_ALLOC.to_string(), f_alloc);

    let f_get = mb.declare_extern(
        BYTES_GET,
        Signature {
            params: vec![Type::GcPtr, Type::I64],
            ret: Some(Type::I64),
        },
    );
    func_refs.insert(BYTES_GET.to_string(), f_get);

    let f_set = mb.declare_extern(
        BYTES_SET,
        Signature {
            params: vec![Type::GcPtr, Type::I64, Type::I64],
            ret: None,
        },
    );
    func_refs.insert(BYTES_SET.to_string(), f_set);

    let f_len = mb.declare_extern(
        BYTES_LEN,
        Signature {
            params: vec![Type::GcPtr],
            ret: Some(Type::I64),
        },
    );
    func_refs.insert(BYTES_LEN.to_string(), f_len);

    // Phase 1: Declare user functions
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
            // Coerce the body's result to the function's declared
            // return type. This makes patterns like
            //   fn get_cont() -> cont { reset { ... } }
            // work even when the reset's IR result is I64 (because
            // reset's handler block is type-uniformly I64) — the
            // bitcast at the Ret restores the correct surface type.
            let result = lowerer.coerce_to(result, ret_ty);
            lowerer.fb.ret(result);
        }

        mb.finish_func(fref, lowerer.fb);
    }

    let module = mb.build();
    let entry = *func_refs
        .get(&program.decls.last().unwrap().name)
        .expect("no functions in program");

    LoweredProgram {
        module,
        entry,
        func_refs,
    }
}
