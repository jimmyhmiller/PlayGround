//! LLVM lowering via inkwell.
//!
//! Conventions (M1/M2):
//! * **`:native`** — emit one LLVM function with a built-in calling convention.
//! * **`:shim`** — a `ccc` `__impl` body + a `naked` trampoline marshalling the
//!   convention's registers ↔ SysV; register-constrained inline-asm call sites.
//!
//! Allocation (M3): values are now `i64` *or* pointers, and a pointer's region
//! picks its storage: `frame` → `alloca`, `static` → a global, `heap` →
//! `malloc`/`free`. The region is checked (see `check.rs`) but at the LLVM level
//! every pointer is just an opaque `ptr`.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum, FunctionType};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue};
use inkwell::{AddressSpace, InlineAsmDialect, IntPredicate};

use crate::ast::*;
use crate::convention::Lowering;

const SYSV_ARG: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

struct ShimInfo<'ctx> {
    impl_fn: FunctionValue<'ctx>,
    param_regs: Vec<String>,
    ret_reg: String,
    clobber: Vec<String>,
}

struct Cg<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    funcs: HashMap<String, FunctionValue<'ctx>>,
    shims: HashMap<String, ShimInfo<'ctx>>,
    globals: Cell<u32>,
}

pub fn compile<'ctx>(ctx: &'ctx Context, program: &Program) -> Result<Module<'ctx>, String> {
    let mut cg = Cg {
        ctx,
        module: ctx.create_module("coil"),
        builder: ctx.create_builder(),
        funcs: HashMap::new(),
        shims: HashMap::new(),
        globals: Cell::new(0),
    };

    // 1a. declare externs (foreign symbols the linker will resolve).
    for e in &program.externs {
        let conv = program
            .conventions
            .get(&e.cc)
            .ok_or_else(|| format!("codegen: unknown convention '{}'", e.cc))?;
        let cc_id = conv
            .native_id()
            .ok_or_else(|| format!("codegen: extern '{}' needs a native convention", e.name))?;
        let fn_ty = cg.fn_type_types(&e.params, &e.ret);
        let fv = cg.module.add_function(&e.name, fn_ty, None);
        fv.set_call_conventions(cc_id);
        cg.funcs.insert(e.name.clone(), fv);
    }

    // 1b. declare all functions (so mutual recursion resolves).
    for f in &program.funcs {
        let conv = program
            .conventions
            .get(&f.cc)
            .ok_or_else(|| format!("codegen: unknown convention '{}'", f.cc))?;
        let fn_ty = cg.fn_type(&f.params, &f.ret);

        match &conv.lowering {
            Lowering::Native(cc) => {
                let fv = cg.module.add_function(&f.name, fn_ty, None);
                fv.set_call_conventions(cc.id());
                cg.funcs.insert(f.name.clone(), fv);
            }
            Lowering::Shim => {
                let ret_reg = conv
                    .ret
                    .clone()
                    .ok_or_else(|| format!("shim convention '{}' needs :ret", conv.name))?;
                let impl_fn = cg.module.add_function(&format!("{}__impl", f.name), fn_ty, None);
                let void_ty = ctx.void_type().fn_type(&[], false);
                let tramp = cg.module.add_function(&f.name, void_ty, None);
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

    cg.module
        .verify()
        .map_err(|e| format!("LLVM module verification failed:\n{}", e.to_string()))?;
    Ok(cg.module)
}

impl<'ctx> Cg<'ctx> {
    fn basic_ty(&self, t: &Type) -> BasicTypeEnum<'ctx> {
        match t {
            Type::I64 => self.ctx.i64_type().into(),
            Type::Ptr(_) => self.ctx.ptr_type(AddressSpace::default()).into(),
        }
    }

    fn fn_type(&self, params: &[Param], ret: &Type) -> FunctionType<'ctx> {
        let types: Vec<Type> = params.iter().map(|p| p.ty.clone()).collect();
        self.fn_type_types(&types, ret)
    }

    fn fn_type_types(&self, params: &[Type], ret: &Type) -> FunctionType<'ctx> {
        let p: Vec<BasicMetadataTypeEnum> = params.iter().map(|t| self.basic_ty(t).into()).collect();
        match ret {
            Type::I64 => self.ctx.i64_type().fn_type(&p, false),
            Type::Ptr(_) => self.ctx.ptr_type(AddressSpace::default()).fn_type(&p, false),
        }
    }

    fn emit_func(&self, f: &Func, function: FunctionValue<'ctx>) -> Result<(), String> {
        let entry = self.ctx.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let mut scope: HashMap<String, BasicValueEnum<'ctx>> = HashMap::new();
        for (i, p) in f.params.iter().enumerate() {
            let v = function.get_nth_param(i as u32).ok_or("codegen: missing param")?;
            v.set_name(&p.name);
            scope.insert(p.name.clone(), v);
        }

        let mut last: BasicValueEnum = self.ctx.i64_type().const_zero().into();
        for e in &f.body {
            last = self.emit_expr(e, &scope)?;
        }
        self.builder.build_return(Some(&last)).map_err(le)?;
        Ok(())
    }

    fn emit_trampoline(
        &self,
        tramp: FunctionValue<'ctx>,
        shim: &ShimInfo<'ctx>,
        n: usize,
    ) -> Result<(), String> {
        let entry = self.ctx.append_basic_block(tramp, "entry");
        self.builder.position_at_end(entry);

        let mut asm = String::new();
        for i in 0..n {
            asm += &format!("movq %{}, %{}\n", shim.param_regs[i], SYSV_ARG[i]);
        }
        asm += "subq $$8, %rsp\n";
        let impl_name = shim.impl_fn.get_name().to_str().map_err(|_| "bad impl name")?;
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
            true,
            false,
            Some(InlineAsmDialect::ATT),
            false,
        );
        self.builder
            .build_indirect_call(void_ty, asm_ptr, &[], "")
            .map_err(le)?;
        self.builder.build_unreachable().map_err(le)?;
        Ok(())
    }

    fn emit_shim_call(
        &self,
        name: &str,
        shim: &ShimInfo<'ctx>,
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let n = args.len();
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

        let arg_types: Vec<BasicMetadataTypeEnum> = args.iter().map(|v| v.get_type().into()).collect();
        let fn_ty = self.ctx.i64_type().fn_type(&arg_types, false);
        let asm_ptr = self.ctx.create_inline_asm(
            fn_ty,
            format!("call {}", name),
            cons,
            true,
            true, // align stack: 16-aligned at the `call`
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
    }

    fn emit_expr(
        &self,
        e: &Expr,
        scope: &HashMap<String, BasicValueEnum<'ctx>>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        match e {
            Expr::Int(n) => Ok(i64t.const_int(*n as u64, true).into()),
            Expr::Var(name) => scope
                .get(name)
                .copied()
                .ok_or_else(|| format!("codegen: unbound '{name}'")),
            Expr::Bin { op, lhs, rhs } => {
                let l = self.emit_expr(lhs, scope)?.into_int_value();
                let r = self.emit_expr(rhs, scope)?.into_int_value();
                let v = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "add"),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "sub"),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "mul"),
                    BinOp::Div => self.builder.build_int_signed_div(l, r, "div"),
                    BinOp::Rem => self.builder.build_int_signed_rem(l, r, "rem"),
                };
                Ok(v.map_err(le)?.into())
            }
            Expr::Cmp { op, lhs, rhs } => {
                let l = self.emit_expr(lhs, scope)?.into_int_value();
                let r = self.emit_expr(rhs, scope)?.into_int_value();
                let pred = match op {
                    CmpOp::Lt => IntPredicate::SLT,
                    CmpOp::Le => IntPredicate::SLE,
                    CmpOp::Gt => IntPredicate::SGT,
                    CmpOp::Ge => IntPredicate::SGE,
                    CmpOp::Eq => IntPredicate::EQ,
                    CmpOp::Ne => IntPredicate::NE,
                };
                let b = self.builder.build_int_compare(pred, l, r, "cmp").map_err(le)?;
                Ok(self.builder.build_int_z_extend(b, i64t, "cmp64").map_err(le)?.into())
            }
            Expr::Do(es) => {
                let mut last: BasicValueEnum = i64t.const_zero().into();
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
                let mut last: BasicValueEnum = i64t.const_zero().into();
                for e in body {
                    last = self.emit_expr(e, &child)?;
                }
                Ok(last)
            }
            Expr::If { cond, then, els } => self.emit_if(cond, then, els, scope),
            Expr::Call { func, args } => {
                let argvals: Vec<BasicValueEnum<'ctx>> = args
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
            }
            Expr::Alloc { region } => self.emit_alloc(*region),
            Expr::Load(p) => {
                let ptr = self.emit_expr(p, scope)?.into_pointer_value();
                Ok(self.builder.build_load(i64t, ptr, "load").map_err(le)?)
            }
            Expr::Store { ptr, val } => {
                let p = self.emit_expr(ptr, scope)?.into_pointer_value();
                let v = self.emit_expr(val, scope)?.into_int_value();
                self.builder.build_store(p, v).map_err(le)?;
                Ok(v.into())
            }
            Expr::Free(p) => {
                let ptr = self.emit_expr(p, scope)?.into_pointer_value();
                self.builder.build_free(ptr).map_err(le)?;
                Ok(i64t.const_zero().into())
            }
        }
    }

    fn emit_alloc(&self, region: Region) -> Result<BasicValueEnum<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        let ptr = match region {
            Region::Frame => self.builder.build_alloca(i64t, "frame.slot").map_err(le)?,
            Region::Heap => self.builder.build_malloc(i64t, "heap.box").map_err(le)?,
            Region::Static => {
                let n = self.globals.get();
                self.globals.set(n + 1);
                let g = self.module.add_global(i64t, None, &format!("g.{n}"));
                g.set_initializer(&i64t.const_zero());
                g.as_pointer_value()
            }
        };
        Ok(ptr.into())
    }

    fn emit_if(
        &self,
        cond: &Expr,
        then: &Expr,
        els: &Expr,
        scope: &HashMap<String, BasicValueEnum<'ctx>>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;

        let c = self.emit_expr(cond, scope)?.into_int_value();
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
        let phi = self.builder.build_phi(tv.get_type(), "ifval").map_err(le)?;
        phi.add_incoming(&[
            (&tv as &dyn BasicValue, then_end),
            (&ev as &dyn BasicValue, else_end),
        ]);
        Ok(phi.as_basic_value())
    }
}

fn le<E: std::fmt::Display>(e: E) -> String {
    format!("llvm: {e}")
}
