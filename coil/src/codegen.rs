//! LLVM lowering via inkwell.
//!
//! M1 lowers every function to an LLVM function whose **calling convention is
//! set from its `defcc`** (the `native_id`), and sets the same convention on
//! every call site. So a `:native fast` convention really does emit a `fastcc`
//! function and `call fastcc` — conventions are not cosmetic.

use std::collections::HashMap;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{BasicValue, FunctionValue, IntValue};
use inkwell::IntPredicate;

use crate::ast::*;

struct Cg<'ctx> {
    ctx: &'ctx Context,
    builder: Builder<'ctx>,
    funcs: HashMap<String, FunctionValue<'ctx>>,
}

pub fn compile<'ctx>(
    ctx: &'ctx Context,
    program: &Program,
) -> Result<Module<'ctx>, String> {
    let module = ctx.create_module("coil");
    let builder = ctx.create_builder();
    let mut cg = Cg {
        ctx,
        builder,
        funcs: HashMap::new(),
    };

    // 1. declare all functions first (so mutual recursion resolves), applying
    //    each function's calling convention.
    let i64t = ctx.i64_type();
    for f in &program.funcs {
        let param_types: Vec<_> = f.params.iter().map(|_| i64t.into()).collect();
        let fn_ty = i64t.fn_type(&param_types, false);
        let function = module.add_function(&f.name, fn_ty, None);
        let cc_id = program
            .conventions
            .get(&f.cc)
            .and_then(|c| c.native_id())
            .ok_or_else(|| format!("codegen: convention '{}' not lowerable", f.cc))?;
        function.set_call_conventions(cc_id);
        cg.funcs.insert(f.name.clone(), function);
    }

    // 2. emit bodies
    for f in &program.funcs {
        cg.emit_func(f)?;
    }

    module
        .verify()
        .map_err(|e| format!("LLVM module verification failed:\n{}", e.to_string()))?;
    Ok(module)
}

impl<'ctx> Cg<'ctx> {
    fn emit_func(&self, f: &Func) -> Result<(), String> {
        let function = self.funcs[&f.name];
        let entry = self.ctx.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let mut scope: HashMap<String, IntValue<'ctx>> = HashMap::new();
        for (i, p) in f.params.iter().enumerate() {
            let v = function
                .get_nth_param(i as u32)
                .ok_or("codegen: missing param")?
                .into_int_value();
            v.set_name(&p.name);
            scope.insert(p.name.clone(), v);
        }

        let mut last = self.ctx.i64_type().const_zero();
        for e in &f.body {
            last = self.emit_expr(e, &scope)?;
        }
        self.builder
            .build_return(Some(&last))
            .map_err(le)?;
        Ok(())
    }

    fn emit_expr(
        &self,
        e: &Expr,
        scope: &HashMap<String, IntValue<'ctx>>,
    ) -> Result<IntValue<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        match e {
            Expr::Int(n) => Ok(i64t.const_int(*n as u64, true)),
            Expr::Var(name) => scope
                .get(name)
                .copied()
                .ok_or_else(|| format!("codegen: unbound '{name}'")),
            Expr::Bin { op, lhs, rhs } => {
                let l = self.emit_expr(lhs, scope)?;
                let r = self.emit_expr(rhs, scope)?;
                let v = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "add"),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "sub"),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "mul"),
                    BinOp::Div => self.builder.build_int_signed_div(l, r, "div"),
                };
                v.map_err(le)
            }
            Expr::Cmp { op, lhs, rhs } => {
                let l = self.emit_expr(lhs, scope)?;
                let r = self.emit_expr(rhs, scope)?;
                let pred = match op {
                    CmpOp::Lt => IntPredicate::SLT,
                    CmpOp::Le => IntPredicate::SLE,
                    CmpOp::Gt => IntPredicate::SGT,
                    CmpOp::Ge => IntPredicate::SGE,
                    CmpOp::Eq => IntPredicate::EQ,
                    CmpOp::Ne => IntPredicate::NE,
                };
                let b = self.builder.build_int_compare(pred, l, r, "cmp").map_err(le)?;
                // widen i1 -> i64 so the language is uniformly i64-valued
                self.builder
                    .build_int_z_extend(b, i64t, "cmp64")
                    .map_err(le)
            }
            Expr::Do(es) => {
                let mut last = i64t.const_zero();
                for e in es {
                    last = self.emit_expr(e, scope)?;
                }
                Ok(last)
            }
            Expr::Let { binds, body } => {
                let mut child = scope.clone();
                for (name, val) in binds {
                    let v = self.emit_expr(val, &child)?;
                    child.insert(name.clone(), v);
                }
                let mut last = i64t.const_zero();
                for e in body {
                    last = self.emit_expr(e, &child)?;
                }
                Ok(last)
            }
            Expr::If { cond, then, els } => self.emit_if(cond, then, els, scope),
            Expr::Call { func, args } => {
                let callee = *self
                    .funcs
                    .get(func)
                    .ok_or_else(|| format!("codegen: call to undefined '{func}'"))?;
                let argvals: Vec<_> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope).map(|v| v.into()))
                    .collect::<Result<_, _>>()?;
                let cs = self
                    .builder
                    .build_call(callee, &argvals, "call")
                    .map_err(le)?;
                // call site convention must match the callee's
                cs.set_call_convention(callee.get_call_conventions());
                cs.try_as_basic_value()
                    .left()
                    .ok_or_else(|| "codegen: call returned void".to_string())
                    .map(|v| v.into_int_value())
            }
        }
    }

    fn emit_if(
        &self,
        cond: &Expr,
        then: &Expr,
        els: &Expr,
        scope: &HashMap<String, IntValue<'ctx>>,
    ) -> Result<IntValue<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;

        let c = self.emit_expr(cond, scope)?;
        let cmp = self
            .builder
            .build_int_compare(IntPredicate::NE, c, i64t.const_zero(), "ifc")
            .map_err(le)?;

        let then_bb = self.ctx.append_basic_block(function, "then");
        let else_bb = self.ctx.append_basic_block(function, "else");
        let merge_bb = self.ctx.append_basic_block(function, "ifcont");

        self.builder
            .build_conditional_branch(cmp, then_bb, else_bb)
            .map_err(le)?;

        self.builder.position_at_end(then_bb);
        let tv = self.emit_expr(then, scope)?;
        let then_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).map_err(le)?;

        self.builder.position_at_end(else_bb);
        let ev = self.emit_expr(els, scope)?;
        let else_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).map_err(le)?;

        self.builder.position_at_end(merge_bb);
        let phi = self.builder.build_phi(i64t, "ifval").map_err(le)?;
        phi.add_incoming(&[
            (&tv as &dyn BasicValue, then_end),
            (&ev as &dyn BasicValue, else_end),
        ]);
        Ok(phi.as_basic_value().into_int_value())
    }
}

fn le<E: std::fmt::Display>(e: E) -> String {
    format!("llvm: {e}")
}
