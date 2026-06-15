//! LLVM lowering via inkwell.
//!
//! Two lowering paths, chosen per function's calling convention:
//!
//! * **`:native`** — the convention coincides with a built-in LLVM calling
//!   convention. Emit one LLVM function with that convention and call it
//!   normally. Full optimization, zero overhead.
//!
//! * **`:shim`** — the convention's register layout is something LLVM's closed
//!   CC enum cannot express. We emit *two* symbols: a `ccc` `<name>__impl`
//!   holding the real body, and a `naked` trampoline `<name>` that marshals the
//!   convention's argument registers into the SysV argument registers, calls the
//!   impl (realigning the stack), and leaves the result in the convention's
//!   return register. Call sites use register-constrained inline asm so each
//!   argument is genuinely pinned to the requested register before `call`.
//!
//! The upshot: a function can have a calling convention LLVM doesn't know, and
//! still be defined and called correctly.

use std::collections::{HashMap, HashSet};

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{BasicValue, FunctionValue, IntValue};
use inkwell::{InlineAsmDialect, IntPredicate};

use crate::ast::*;
use crate::convention::Lowering;

/// SysV x86-64 integer argument registers, in order. The `ccc` impl receives
/// its arguments here; the trampoline moves the convention's registers into
/// these before the call.
const SYSV_ARG: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

/// Everything needed to lower calls to (and the trampoline of) a shim function.
struct ShimInfo<'ctx> {
    impl_fn: FunctionValue<'ctx>,
    param_regs: Vec<String>,
    ret_reg: String,
    clobber: Vec<String>,
}

struct Cg<'ctx> {
    ctx: &'ctx Context,
    builder: Builder<'ctx>,
    /// The callable symbol for each Coil function: the function itself for
    /// `:native`, or the trampoline for `:shim`.
    funcs: HashMap<String, FunctionValue<'ctx>>,
    /// Present iff the function uses a shim convention.
    shims: HashMap<String, ShimInfo<'ctx>>,
}

pub fn compile<'ctx>(ctx: &'ctx Context, program: &Program) -> Result<Module<'ctx>, String> {
    let module = ctx.create_module("coil");
    let builder = ctx.create_builder();
    let mut cg = Cg {
        ctx,
        builder,
        funcs: HashMap::new(),
        shims: HashMap::new(),
    };
    let i64t = ctx.i64_type();

    // 1. declare everything first (so mutual recursion resolves).
    for f in &program.funcs {
        let conv = program
            .conventions
            .get(&f.cc)
            .ok_or_else(|| format!("codegen: unknown convention '{}'", f.cc))?;
        let n = f.params.len();
        let fn_ty = i64t.fn_type(&vec![i64t.into(); n], false);

        match &conv.lowering {
            Lowering::Native(cc) => {
                let fv = module.add_function(&f.name, fn_ty, None);
                fv.set_call_conventions(cc.id());
                cg.funcs.insert(f.name.clone(), fv);
            }
            Lowering::Shim => {
                let ret_reg = conv
                    .ret
                    .clone()
                    .ok_or_else(|| format!("shim convention '{}' needs :ret", conv.name))?;
                if conv.params.len() < n {
                    return Err(format!(
                        "convention '{}' has {} param registers, function '{}' needs {}",
                        conv.name,
                        conv.params.len(),
                        f.name,
                        n
                    ));
                }
                // ccc body
                let impl_fn = module.add_function(&format!("{}__impl", f.name), fn_ty, None);
                // naked trampoline exposing the exotic ABI
                let void_ty = ctx.void_type().fn_type(&[], false);
                let tramp = module.add_function(&f.name, void_ty, None);
                for kind in ["naked", "noinline"] {
                    let attr = ctx.create_enum_attribute(Attribute::get_named_enum_kind_id(kind), 0);
                    tramp.add_attribute(AttributeLoc::Function, attr);
                }
                cg.funcs.insert(f.name.clone(), tramp);
                cg.shims.insert(
                    f.name.clone(),
                    ShimInfo {
                        impl_fn,
                        param_regs: conv.params.clone(),
                        ret_reg,
                        clobber: conv.clobber.clone(),
                    },
                );
            }
        }
    }

    // 2. emit bodies (+ trampolines).
    for f in &program.funcs {
        if let Some(shim) = cg.shims.get(&f.name) {
            cg.emit_func(f, shim.impl_fn)?;
            let tramp = cg.funcs[&f.name];
            cg.emit_trampoline(tramp, shim, f.params.len())?;
        } else {
            let fv = cg.funcs[&f.name];
            cg.emit_func(f, fv)?;
        }
    }

    module
        .verify()
        .map_err(|e| format!("LLVM module verification failed:\n{}", e.to_string()))?;
    Ok(module)
}

impl<'ctx> Cg<'ctx> {
    fn emit_func(&self, f: &Func, function: FunctionValue<'ctx>) -> Result<(), String> {
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
        self.builder.build_return(Some(&last)).map_err(le)?;
        Ok(())
    }

    /// Emit the naked trampoline for a shim function: marshal the convention's
    /// argument registers into the SysV registers, call the `ccc` impl, and move
    /// the result into the convention's return register.
    fn emit_trampoline(
        &self,
        tramp: FunctionValue<'ctx>,
        shim: &ShimInfo<'ctx>,
        n: usize,
    ) -> Result<(), String> {
        let entry = self.ctx.append_basic_block(tramp, "entry");
        self.builder.position_at_end(entry);

        // NOTE: sequential moves assume the convention's arg registers don't
        // collide with the SysV registers in a way that needs a parallel move.
        // A real implementation computes a safe move schedule; fine for M2.
        let mut asm = String::new();
        for i in 0..n {
            asm += &format!("movq %{}, %{}\n", shim.param_regs[i], SYSV_ARG[i]);
        }
        asm += "subq $$8, %rsp\n"; // realign to 16 before the ccc call
        let impl_name = shim
            .impl_fn
            .get_name()
            .to_str()
            .map_err(|_| "codegen: bad impl name")?;
        asm += &format!("call {}\n", impl_name);
        asm += "addq $$8, %rsp\n";
        if shim.ret_reg != "rax" {
            asm += &format!("movq %rax, %{}\n", shim.ret_reg);
        }
        asm += "ret\n";

        let void_ty = self.ctx.void_type().fn_type(&[], false);
        let asm_ptr = self.ctx.create_inline_asm(
            void_ty,
            asm,
            "~{memory}".to_string(),
            true,  // side effects
            false, // align stack (n/a inside a naked function)
            Some(InlineAsmDialect::ATT),
            false,
        );
        self.builder
            .build_indirect_call(void_ty, asm_ptr, &[], "")
            .map_err(le)?;
        self.builder.build_unreachable().map_err(le)?;
        Ok(())
    }

    /// A call to a shim function: pin each argument to the convention's register
    /// via inline-asm constraints, `call` the trampoline, read the result from
    /// the convention's return register.
    fn emit_shim_call(
        &self,
        name: &str,
        shim: &ShimInfo<'ctx>,
        args: &[IntValue<'ctx>],
    ) -> Result<IntValue<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        let n = args.len();

        // constraints: output register, then one input register per argument,
        // then the convention's clobbers (excluding operand registers).
        let mut cons = format!("={{{}}}", shim.ret_reg);
        for r in &shim.param_regs[..n] {
            cons += &format!(",{{{}}}", r);
        }
        let mut used: HashSet<&str> = shim.param_regs[..n].iter().map(String::as_str).collect();
        used.insert(shim.ret_reg.as_str());
        for c in &shim.clobber {
            if !used.contains(c.as_str()) {
                cons += &format!(",~{{{}}}", c);
            }
        }
        cons += ",~{memory}";

        let fn_ty = i64t.fn_type(&vec![i64t.into(); n], false);
        let asm_ptr = self.ctx.create_inline_asm(
            fn_ty,
            format!("call {}", name),
            cons,
            true, // side effects
            true, // align stack: guarantee 16-alignment at the `call`
            Some(InlineAsmDialect::ATT),
            false,
        );
        let argvals: Vec<_> = args.iter().map(|v| (*v).into()).collect();
        let cs = self
            .builder
            .build_indirect_call(fn_ty, asm_ptr, &argvals, "shimcall")
            .map_err(le)?;
        cs.try_as_basic_value()
            .left()
            .ok_or_else(|| "codegen: shim call returned void".to_string())
            .map(|v| v.into_int_value())
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
                self.builder.build_int_z_extend(b, i64t, "cmp64").map_err(le)
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
                let argvals: Vec<IntValue<'ctx>> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope))
                    .collect::<Result<_, _>>()?;
                if let Some(shim) = self.shims.get(func) {
                    return self.emit_shim_call(func, shim, &argvals);
                }
                let callee = *self
                    .funcs
                    .get(func)
                    .ok_or_else(|| format!("codegen: call to undefined '{func}'"))?;
                let meta: Vec<_> = argvals.iter().map(|v| (*v).into()).collect();
                let cs = self.builder.build_call(callee, &meta, "call").map_err(le)?;
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
