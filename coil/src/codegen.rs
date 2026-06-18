//! LLVM lowering via inkwell.
//!
//! Conventions (M1/M2): `:native` emits one function with a built-in LLVM
//! calling convention; `:shim` emits a `ccc` `__impl` + a `naked` trampoline.
//!
//! Allocation/types (M3+): values carry their Coil `Type` through codegen as a
//! `(BasicValueEnum, Type)` pair, so `load`/`store!`/`index` use the right width
//! and pointee, integer widths (i8/i16/i32/i64) and `cast` work, and pointers
//! carry a region + pointee type (e.g. C `char**`).

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType, StructType};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue};
use inkwell::{AddressSpace, InlineAsmDialect, IntPredicate};

use crate::ast::*;
use crate::convention::Lowering;

const SYSV_ARG: [&str; 6] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"];

/// A value plus its Coil type.
type Tv<'ctx> = (BasicValueEnum<'ctx>, Type);

struct ShimInfo<'ctx> {
    impl_fn: FunctionValue<'ctx>,
    param_regs: Vec<String>,
    ret_reg: String,
    clobber: Vec<String>,
}

struct StructInfo<'ctx> {
    fields: Vec<(String, Type)>,
    ty: StructType<'ctx>,
}

/// A sum type's runtime shape: `{ i32 tag, [words x i64] payload }`, plus a
/// per-variant struct type used to read/write the variant's fields out of the
/// payload.
struct SumInfo<'ctx> {
    variants: Vec<(String, Vec<(String, Type)>)>,
    ty: StructType<'ctx>,
    variant_structs: Vec<StructType<'ctx>>,
}

struct Cg<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    funcs: HashMap<String, FunctionValue<'ctx>>,
    shims: HashMap<String, ShimInfo<'ctx>>,
    structs: HashMap<String, StructInfo<'ctx>>,
    sums: HashMap<String, SumInfo<'ctx>>,
    /// Coil return type of every callable (function/extern), for typing calls.
    rets: HashMap<String, Type>,
    /// Full signature of every callable, for `fnptr-of` (cc, params, ret).
    callables: HashMap<String, (String, Vec<Type>, Type)>,
    /// Native LLVM calling-convention id for each convention name.
    conv_ids: HashMap<String, u32>,
    globals: Cell<u32>,
}

pub fn compile<'ctx>(ctx: &'ctx Context, program: &Program) -> Result<Module<'ctx>, String> {
    let mut cg = Cg {
        ctx,
        module: ctx.create_module("coil"),
        builder: ctx.create_builder(),
        funcs: HashMap::new(),
        shims: HashMap::new(),
        structs: HashMap::new(),
        sums: HashMap::new(),
        rets: HashMap::new(),
        callables: HashMap::new(),
        conv_ids: HashMap::new(),
        globals: Cell::new(0),
    };

    for (name, conv) in &program.conventions {
        if let Some(id) = conv.native_id() {
            cg.conv_ids.insert(name.clone(), id);
        }
    }

    // 0. build aggregate types. Two-phase (opaque names first, then bodies) so
    //    definition order doesn't matter (monomorphization emits in any order).
    // 0a. opaque struct names.
    for sd in &program.structs {
        let ty = ctx.opaque_struct_type(&sd.name);
        cg.structs.insert(
            sd.name.clone(),
            StructInfo {
                fields: sd.fields.clone(),
                ty,
            },
        );
    }
    // 0b. sum types `{ i32 tag, [words x i64] payload }` (size from a conservative
    //     layout of the Coil types — no LLVM layout needed, just an upper bound).
    let struct_map: HashMap<&str, &StructDef> =
        program.structs.iter().map(|s| (s.name.as_str(), s)).collect();
    let sum_map: HashMap<&str, &SumDef> =
        program.sums.iter().map(|s| (s.name.as_str(), s)).collect();
    for sd in &program.sums {
        let words = sum_words(sd, &struct_map, &sum_map);
        let payload = ctx.i64_type().array_type(words);
        let ty = ctx.opaque_struct_type(&sd.name);
        ty.set_body(&[ctx.i32_type().into(), payload.into()], false);
        cg.sums.insert(
            sd.name.clone(),
            SumInfo {
                variants: sd.variants.iter().map(|v| (v.name.clone(), v.fields.clone())).collect(),
                ty,
                variant_structs: vec![], // filled in 0d
            },
        );
    }
    // 0c. struct bodies (may reference sums, which are now complete).
    for sd in &program.structs {
        let ty = cg.structs[&sd.name].ty;
        let field_types: Vec<BasicTypeEnum> = sd.fields.iter().map(|(_, t)| cg.basic_ty(t)).collect();
        ty.set_body(&field_types, false);
    }
    // 0d. per-variant field structs (used to read/write payloads).
    for sd in &program.sums {
        let vss: Vec<StructType> = sd
            .variants
            .iter()
            .map(|v| {
                let fs: Vec<BasicTypeEnum> = v.fields.iter().map(|(_, t)| cg.basic_ty(t)).collect();
                ctx.struct_type(&fs, false)
            })
            .collect();
        cg.sums.get_mut(&sd.name).unwrap().variant_structs = vss;
    }

    // 1a. declare externs (foreign symbols resolved at link time).
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
        cg.rets.insert(e.name.clone(), e.ret.clone());
        cg.callables
            .insert(e.name.clone(), (e.cc.clone(), e.params.clone(), e.ret.clone()));
    }

    // 1b. declare all functions (so mutual recursion resolves).
    for f in &program.funcs {
        let conv = program
            .conventions
            .get(&f.cc)
            .ok_or_else(|| format!("codegen: unknown convention '{}'", f.cc))?;
        let fn_ty = cg.fn_type(&f.params, &f.ret);
        cg.rets.insert(f.name.clone(), f.ret.clone());
        cg.callables.insert(
            f.name.clone(),
            (
                f.cc.clone(),
                f.params.iter().map(|p| p.ty.clone()).collect(),
                f.ret.clone(),
            ),
        );

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
            Type::Int(w) => self.ctx.custom_width_int_type(*w).into(),
            Type::Ptr(..) => self.ctx.ptr_type(AddressSpace::default()).into(),
            Type::Struct(name) => self
                .structs
                .get(name)
                .map(|s| s.ty)
                .or_else(|| self.sums.get(name).map(|s| s.ty))
                .unwrap_or_else(|| panic!("unknown nominal type '{name}'"))
                .into(),
            Type::Array(elem, n) => self.basic_ty(elem).array_type(*n).into(),
            Type::Fn(..) => self.ctx.ptr_type(AddressSpace::default()).into(),
            Type::App(..) => unreachable!("generic type survived monomorphization"),
        }
    }

    fn fn_type(&self, params: &[Param], ret: &Type) -> FunctionType<'ctx> {
        let types: Vec<Type> = params.iter().map(|p| p.ty.clone()).collect();
        self.fn_type_types(&types, ret)
    }

    fn fn_type_types(&self, params: &[Type], ret: &Type) -> FunctionType<'ctx> {
        let p: Vec<BasicMetadataTypeEnum> = params.iter().map(|t| self.basic_ty(t).into()).collect();
        self.basic_ty(ret).fn_type(&p, false)
    }

    fn emit_func(&self, f: &Func, function: FunctionValue<'ctx>) -> Result<(), String> {
        let entry = self.ctx.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let mut scope: HashMap<String, Tv<'ctx>> = HashMap::new();
        for (i, p) in f.params.iter().enumerate() {
            let v = function.get_nth_param(i as u32).ok_or("codegen: missing param")?;
            v.set_name(&p.name);
            scope.insert(p.name.clone(), (v, p.ty.clone()));
        }

        let mut last: Tv = (self.ctx.i64_type().const_zero().into(), Type::Int(64));
        for e in &f.body {
            last = self.emit_expr(e, &scope)?;
        }
        self.builder.build_return(Some(&last.0)).map_err(le)?;
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
            true,
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

    fn emit_expr(&self, e: &Expr, scope: &HashMap<String, Tv<'ctx>>) -> Result<Tv<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        match e {
            Expr::Int(n) => Ok((i64t.const_int(*n as u64, true).into(), Type::Int(64))),
            Expr::Var(name) => scope
                .get(name)
                .cloned()
                .ok_or_else(|| format!("codegen: unbound '{name}'")),
            Expr::Bin { op, lhs, rhs } => {
                let (lv, lt) = self.emit_expr(lhs, scope)?;
                let (rv, _) = self.emit_expr(rhs, scope)?;
                let l = lv.into_int_value();
                let r = rv.into_int_value();
                let v = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "add"),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "sub"),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "mul"),
                    BinOp::Div => self.builder.build_int_signed_div(l, r, "div"),
                    BinOp::Rem => self.builder.build_int_signed_rem(l, r, "rem"),
                };
                Ok((v.map_err(le)?.into(), lt))
            }
            Expr::Cmp { op, lhs, rhs } => {
                let (lv, _) = self.emit_expr(lhs, scope)?;
                let (rv, _) = self.emit_expr(rhs, scope)?;
                let pred = match op {
                    CmpOp::Lt => IntPredicate::SLT,
                    CmpOp::Le => IntPredicate::SLE,
                    CmpOp::Gt => IntPredicate::SGT,
                    CmpOp::Ge => IntPredicate::SGE,
                    CmpOp::Eq => IntPredicate::EQ,
                    CmpOp::Ne => IntPredicate::NE,
                };
                let b = self
                    .builder
                    .build_int_compare(pred, lv.into_int_value(), rv.into_int_value(), "cmp")
                    .map_err(le)?;
                Ok((
                    self.builder.build_int_z_extend(b, i64t, "cmp64").map_err(le)?.into(),
                    Type::Int(64),
                ))
            }
            Expr::Do(es) => {
                let mut last: Tv = (i64t.const_zero().into(), Type::Int(64));
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
                let mut last: Tv = (i64t.const_zero().into(), Type::Int(64));
                for e in body {
                    last = self.emit_expr(e, &child)?;
                }
                Ok(last)
            }
            Expr::If { cond, then, els } => self.emit_if(cond, then, els, scope),
            Expr::Call { func, args, .. } => {
                let argtv: Vec<Tv<'ctx>> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope))
                    .collect::<Result<_, _>>()?;
                let ret_ty = self
                    .rets
                    .get(func)
                    .cloned()
                    .ok_or_else(|| format!("codegen: unknown callable '{func}'"))?;
                if let Some(shim) = self.shims.get(func) {
                    let raw: Vec<_> = argtv.iter().map(|(v, _)| *v).collect();
                    let v = self.emit_shim_call(func, shim, &raw)?;
                    return Ok((v, ret_ty));
                }
                let callee = *self
                    .funcs
                    .get(func)
                    .ok_or_else(|| format!("codegen: call to undefined '{func}'"))?;
                let meta: Vec<_> = argtv.iter().map(|(v, _)| (*v).into()).collect();
                let cs = self.builder.build_call(callee, &meta, "call").map_err(le)?;
                cs.set_call_convention(callee.get_call_conventions());
                let v = cs
                    .try_as_basic_value()
                    .left()
                    .ok_or_else(|| "codegen: call returned void".to_string())?;
                Ok((v, ret_ty))
            }
            Expr::Alloc { storage, ty } => self.emit_alloc(*storage, ty),
            Expr::Field { ptr, field } => {
                let (pv, pt) = self.emit_expr(ptr, scope)?;
                let sname = match pt {
                    Type::Ptr(pointee) => match *pointee {
                        Type::Struct(s) => s,
                        other => return Err(format!("codegen: field on (ptr {other:?})")),
                    },
                    other => return Err(format!("codegen: field on non-pointer {other:?}")),
                };
                let info = self
                    .structs
                    .get(&sname)
                    .ok_or_else(|| format!("codegen: unknown struct '{sname}'"))?;
                let idx = info
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or_else(|| format!("codegen: struct '{sname}' has no field '{field}'"))?;
                let fty = info.fields[idx].1.clone();
                let gep = self
                    .builder
                    .build_struct_gep(info.ty, pv.into_pointer_value(), idx as u32, "field")
                    .map_err(le)?;
                Ok((gep.into(), Type::Ptr(Box::new(fty))))
            }
            Expr::Load(p) => {
                let (pv, pt) = self.emit_expr(p, scope)?;
                let pointee = match pt {
                    Type::Ptr(pointee) => *pointee,
                    other => return Err(format!("codegen: load of non-pointer {other:?}")),
                };
                let v = self
                    .builder
                    .build_load(self.basic_ty(&pointee), pv.into_pointer_value(), "load")
                    .map_err(le)?;
                Ok((v, pointee))
            }
            Expr::Store { ptr, val } => {
                let (pv, _) = self.emit_expr(ptr, scope)?;
                let (vv, vt) = self.emit_expr(val, scope)?;
                self.builder
                    .build_store(pv.into_pointer_value(), vv)
                    .map_err(le)?;
                Ok((vv, vt))
            }
            Expr::Index { ptr, idx } => {
                let (pv, pt) = self.emit_expr(ptr, scope)?;
                let (iv, _) = self.emit_expr(idx, scope)?;
                let pointee = match pt {
                    Type::Ptr(pointee) => *pointee,
                    other => return Err(format!("codegen: index of non-pointer {other:?}")),
                };
                let ptr_val = pv.into_pointer_value();
                let i = iv.into_int_value();
                match &pointee {
                    // pointer to an array: GEP [0, i] yields a pointer to elem i.
                    Type::Array(elem, _) => {
                        let zero = self.ctx.i64_type().const_zero();
                        let gep = unsafe {
                            self.builder
                                .build_gep(self.basic_ty(&pointee), ptr_val, &[zero, i], "idx")
                                .map_err(le)?
                        };
                        Ok((gep.into(), Type::Ptr(elem.clone())))
                    }
                    // pointer to a scalar/struct: GEP [i] is pointer arithmetic.
                    _ => {
                        let gep = unsafe {
                            self.builder
                                .build_gep(self.basic_ty(&pointee), ptr_val, &[i], "idx")
                                .map_err(le)?
                        };
                        Ok((gep.into(), Type::Ptr(Box::new(pointee))))
                    }
                }
            }
            Expr::Cast { ty, expr } => {
                let (v, _) = self.emit_expr(expr, scope)?;
                match ty {
                    Type::Int(to) => {
                        let iv = v.into_int_value();
                        let from = iv.get_type().get_bit_width();
                        let target = self.ctx.custom_width_int_type(*to);
                        let out = if *to > from {
                            self.builder.build_int_s_extend(iv, target, "sext").map_err(le)?
                        } else if *to < from {
                            self.builder.build_int_truncate(iv, target, "trunc").map_err(le)?
                        } else {
                            iv
                        };
                        Ok((out.into(), ty.clone()))
                    }
                    // opaque pointers: a reinterpret leaves the value untouched.
                    Type::Ptr(..) => Ok((v, ty.clone())),
                    other => Err(format!("codegen: cannot cast to {other:?}")),
                }
            }
            Expr::SizeOf(ty) => {
                let sz = self
                    .basic_ty(ty)
                    .size_of()
                    .ok_or_else(|| format!("codegen: type {ty:?} has no known size"))?;
                Ok((sz.into(), Type::Int(64)))
            }
            Expr::Free(p) => {
                let (pv, _) = self.emit_expr(p, scope)?;
                self.builder.build_free(pv.into_pointer_value()).map_err(le)?;
                Ok((i64t.const_zero().into(), Type::Int(64)))
            }
            Expr::Construct { sum, variant, args } => self.emit_construct(sum, variant, args, scope),
            Expr::Match { scrut, arms } => self.emit_match(scrut, arms, scope),
            Expr::FnPtrOf(name) => {
                let fv = *self
                    .funcs
                    .get(name)
                    .ok_or_else(|| format!("codegen: fnptr-of unknown '{name}'"))?;
                let (cc, params, ret) = self
                    .callables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("codegen: no signature for '{name}'"))?;
                let ptr = fv.as_global_value().as_pointer_value();
                Ok((ptr.into(), Type::Fn(cc, params, Box::new(ret))))
            }
            Expr::CallPtr { fp, args } => {
                let (fpv, fpt) = self.emit_expr(fp, scope)?;
                let (cc, params, ret) = match fpt {
                    Type::Fn(cc, params, ret) => (cc, params, *ret),
                    other => return Err(format!("codegen: call-ptr on non-fnptr {other:?}")),
                };
                let fn_ty = self.fn_type_types(&params, &ret);
                let argtv: Vec<Tv<'ctx>> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope))
                    .collect::<Result<_, _>>()?;
                let meta: Vec<_> = argtv.iter().map(|(v, _)| (*v).into()).collect();
                let cs = self
                    .builder
                    .build_indirect_call(fn_ty, fpv.into_pointer_value(), &meta, "callptr")
                    .map_err(le)?;
                if let Some(id) = self.conv_ids.get(&cc) {
                    cs.set_call_convention(*id);
                }
                let v = cs
                    .try_as_basic_value()
                    .left()
                    .ok_or_else(|| "codegen: call-ptr returned void".to_string())?;
                Ok((v, ret))
            }
        }
    }

    fn emit_construct(
        &self,
        sum: &str,
        variant: &str,
        args: &[Expr],
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let info = self.sums.get(sum).ok_or_else(|| format!("codegen: unknown sum '{sum}'"))?;
        let vidx = info
            .variants
            .iter()
            .position(|(n, _)| n == variant)
            .ok_or_else(|| format!("codegen: sum '{sum}' has no variant '{variant}'"))?;
        let sum_ty = info.ty;
        let var_struct = info.variant_structs[vidx];

        let tmp = self.builder.build_alloca(sum_ty, "sum.tmp").map_err(le)?;
        let tagptr = self.builder.build_struct_gep(sum_ty, tmp, 0, "tag").map_err(le)?;
        self.builder
            .build_store(tagptr, self.ctx.i32_type().const_int(vidx as u64, false))
            .map_err(le)?;
        let payload = self.builder.build_struct_gep(sum_ty, tmp, 1, "payload").map_err(le)?;
        for (i, a) in args.iter().enumerate() {
            let (v, _) = self.emit_expr(a, scope)?;
            let fptr = self
                .builder
                .build_struct_gep(var_struct, payload, i as u32, "vf")
                .map_err(le)?;
            self.builder.build_store(fptr, v).map_err(le)?;
        }
        let val = self.builder.build_load(sum_ty, tmp, "sum").map_err(le)?;
        Ok((val, Type::Struct(sum.to_string())))
    }

    fn emit_match(
        &self,
        scrut: &Expr,
        arms: &[Arm],
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let (sumval, st) = self.emit_expr(scrut, scope)?;
        let sumname = match st {
            Type::Struct(s) => s,
            other => return Err(format!("codegen: match on non-sum {other:?}")),
        };
        let info = self
            .sums
            .get(&sumname)
            .ok_or_else(|| format!("codegen: match on non-sum '{sumname}'"))?;
        let sum_ty = info.ty;

        // spill the scrutinee so we can GEP into it
        let tmp = self.builder.build_alloca(sum_ty, "match.tmp").map_err(le)?;
        self.builder.build_store(tmp, sumval).map_err(le)?;
        let tagptr = self.builder.build_struct_gep(sum_ty, tmp, 0, "tag").map_err(le)?;
        let tag = self
            .builder
            .build_load(self.ctx.i32_type(), tagptr, "tag")
            .map_err(le)?
            .into_int_value();
        let payload = self.builder.build_struct_gep(sum_ty, tmp, 1, "payload").map_err(le)?;

        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;
        let arm_blocks: Vec<BasicBlock> =
            arms.iter().map(|_| self.ctx.append_basic_block(function, "arm")).collect();
        let default = self.ctx.append_basic_block(function, "match.default");
        let merge = self.ctx.append_basic_block(function, "match.end");

        let cases: Vec<(inkwell::values::IntValue, BasicBlock)> = arms
            .iter()
            .enumerate()
            .map(|(k, arm)| {
                let vidx = info.variants.iter().position(|(n, _)| n == &arm.variant).unwrap();
                (self.ctx.i32_type().const_int(vidx as u64, false), arm_blocks[k])
            })
            .collect();
        self.builder.build_switch(tag, default, &cases).map_err(le)?;
        self.builder.position_at_end(default);
        self.builder.build_unreachable().map_err(le)?;

        let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock)> = Vec::new();
        let mut result_ty = Type::Int(64);
        for (k, arm) in arms.iter().enumerate() {
            self.builder.position_at_end(arm_blocks[k]);
            let vidx = info.variants.iter().position(|(n, _)| n == &arm.variant).unwrap();
            let var_struct = info.variant_structs[vidx];
            let mut child = scope.clone();
            for (i, b) in arm.binds.iter().enumerate() {
                let fptr = self
                    .builder
                    .build_struct_gep(var_struct, payload, i as u32, "vf")
                    .map_err(le)?;
                let fty = info.variants[vidx].1[i].1.clone();
                let fval = self.builder.build_load(self.basic_ty(&fty), fptr, b).map_err(le)?;
                child.insert(b.clone(), (fval, fty));
            }
            let (bval, bty) = self.emit_expr(&arm.body, &child)?;
            let end = self.builder.get_insert_block().unwrap();
            self.builder.build_unconditional_branch(merge).map_err(le)?;
            incoming.push((bval, end));
            result_ty = bty;
        }

        self.builder.position_at_end(merge);
        let phi = self.builder.build_phi(self.basic_ty(&result_ty), "match.val").map_err(le)?;
        let inc: Vec<(&dyn BasicValue, BasicBlock)> =
            incoming.iter().map(|(v, b)| (v as &dyn BasicValue, *b)).collect();
        phi.add_incoming(&inc);
        Ok((phi.as_basic_value(), result_ty))
    }

    fn emit_alloc(&self, storage: Storage, ty: &Type) -> Result<Tv<'ctx>, String> {
        let bt = self.basic_ty(ty);
        let ptr = match storage {
            Storage::Stack => self.builder.build_alloca(bt, "stack.slot").map_err(le)?,
            Storage::Heap => self.builder.build_malloc(bt, "heap.box").map_err(le)?,
            Storage::Static => {
                let n = self.globals.get();
                self.globals.set(n + 1);
                let g = self.module.add_global(bt, None, &format!("g.{n}"));
                g.set_initializer(&bt.const_zero());
                g.as_pointer_value()
            }
        };
        Ok((ptr.into(), Type::Ptr(Box::new(ty.clone()))))
    }

    fn emit_if(
        &self,
        cond: &Expr,
        then: &Expr,
        els: &Expr,
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;

        let (cv, _) = self.emit_expr(cond, scope)?;
        let c = cv.into_int_value();
        let cmp = self
            .builder
            .build_int_compare(IntPredicate::NE, c, c.get_type().const_zero(), "ifc")
            .map_err(le)?;

        let then_bb = self.ctx.append_basic_block(function, "then");
        let else_bb = self.ctx.append_basic_block(function, "else");
        let merge_bb = self.ctx.append_basic_block(function, "ifcont");

        self.builder
            .build_conditional_branch(cmp, then_bb, else_bb)
            .map_err(le)?;

        self.builder.position_at_end(then_bb);
        let (tv, tt) = self.emit_expr(then, scope)?;
        let then_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).map_err(le)?;

        self.builder.position_at_end(else_bb);
        let (ev, _) = self.emit_expr(els, scope)?;
        let else_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).map_err(le)?;

        self.builder.position_at_end(merge_bb);
        let phi = self.builder.build_phi(self.basic_ty(&tt), "ifval").map_err(le)?;
        phi.add_incoming(&[
            (&tv as &dyn BasicValue, then_end),
            (&ev as &dyn BasicValue, else_end),
        ]);
        Ok((phi.as_basic_value(), tt))
    }
}

fn le<E: std::fmt::Display>(e: E) -> String {
    format!("llvm: {e}")
}

fn align8(x: u64) -> u64 {
    x.div_ceil(8) * 8
}

/// Number of i64 words the payload union needs: the largest variant's size
/// (conservative — every field rounded up to 8 bytes; an upper bound is fine
/// since we read/write through the real per-variant struct type).
fn sum_words(sd: &SumDef, structs: &HashMap<&str, &StructDef>, sums: &HashMap<&str, &SumDef>) -> u32 {
    let max_bytes = sd
        .variants
        .iter()
        .map(|v| {
            v.fields
                .iter()
                .map(|(_, t)| align8(type_bytes(t, structs, sums)))
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    (max_bytes / 8) as u32
}

fn type_bytes(t: &Type, structs: &HashMap<&str, &StructDef>, sums: &HashMap<&str, &SumDef>) -> u64 {
    match t {
        Type::Int(w) => (*w as u64) / 8,
        Type::Ptr(_) | Type::Fn(..) => 8,
        Type::Array(e, n) => align8(type_bytes(e, structs, sums)) * (*n as u64),
        Type::Struct(name) => {
            if let Some(s) = structs.get(name.as_str()) {
                s.fields.iter().map(|(_, t)| align8(type_bytes(t, structs, sums))).sum()
            } else if let Some(sm) = sums.get(name.as_str()) {
                8 + (sum_words(sm, structs, sums) as u64) * 8
            } else {
                8
            }
        }
        Type::App(..) => 8,
    }
}
