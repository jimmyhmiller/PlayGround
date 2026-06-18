//! LLVM backend (behind the `llvm` feature), via inkwell + system LLVM 18.
//!
//! Permissions / multiplicities / types are ERASED by the time we get here, so
//! codegen is ordinary lowering: every value is an `i64`, cells are `malloc`'d
//! blocks of i64 slots, fields are slot indices, `free` is libc `free`, and a
//! function is an LLVM function taking/returning `i64`s. We JIT `main` and run.

use crate::ast::{Expr, Func, Program, Stmt};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::types::{IntType, PointerType};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};
use std::collections::HashMap;

/// Toolchain smoke test: build a function returning 42, JIT it, call it.
pub fn smoke() -> i64 {
    let ctx = Context::create();
    let module = ctx.create_module("smoke");
    let builder = ctx.create_builder();
    let i64t = ctx.i64_type();
    let f = module.add_function("answer", i64t.fn_type(&[], false), None);
    let bb = ctx.append_basic_block(f, "entry");
    builder.position_at_end(bb);
    builder.build_return(Some(&i64t.const_int(42, false))).unwrap();
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    unsafe {
        let func: JitFunction<unsafe extern "C" fn() -> i64> = ee.get_function("answer").unwrap();
        func.call()
    }
}

struct Cg<'c> {
    ctx: &'c Context,
    i64t: IntType<'c>,
    ptr: PointerType<'c>,
    malloc: FunctionValue<'c>,
    slot: HashMap<String, u32>,
    fns: HashMap<String, FunctionValue<'c>>,
}

/// Lower a checked program: every `fn` becomes an LLVM function, then JIT-run
/// `main`. Returns whatever `main` evaluates to.
pub fn compile_and_run(prog: &Program) -> i64 {
    let mut slot: HashMap<String, u32> = HashMap::new();
    for f in &prog.funcs {
        for s in &f.body {
            collect_fields_stmt(s, &mut slot);
        }
        if let Some(t) = &f.tail {
            collect_fields_expr(t, &mut slot);
        }
    }

    let ctx = Context::create();
    let module = ctx.create_module("tally");
    let builder = ctx.create_builder();
    let i64t = ctx.i64_type();
    let ptr = ctx.ptr_type(AddressSpace::default());

    let malloc = module.add_function("malloc", ptr.fn_type(&[i64t.into()], false), None);
    let free = module.add_function("free", ctx.void_type().fn_type(&[ptr.into()], false), None);

    // declare all tally functions first (params + return are i64)
    let mut fns = HashMap::new();
    for f in &prog.funcs {
        let params: Vec<_> = f.params.iter().map(|_| i64t.into()).collect();
        let fv = module.add_function(&f.name, i64t.fn_type(&params, false), None);
        fns.insert(f.name.clone(), fv);
    }

    let cg = Cg {
        ctx: &ctx,
        i64t,
        ptr,
        malloc,
        slot,
        fns,
    };

    for f in &prog.funcs {
        cg.lower_fn(&builder, free, f);
    }

    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    unsafe {
        let main: JitFunction<unsafe extern "C" fn() -> i64> = ee.get_function("main").unwrap();
        main.call()
    }
}

impl<'c> Cg<'c> {
    fn lower_fn(&self, builder: &Builder<'c>, free: FunctionValue<'c>, f: &Func) {
        let fv = self.fns[&f.name];
        let entry = self.ctx.append_basic_block(fv, "entry");
        builder.position_at_end(entry);

        let mut env: HashMap<String, BasicValueEnum> = HashMap::new();
        for (i, p) in f.params.iter().enumerate() {
            env.insert(p.name.clone(), fv.get_nth_param(i as u32).unwrap());
        }

        let mut last = self.i64t.const_zero();
        for s in &f.body {
            match s {
                Stmt::Let(name, _ty, rhs) => {
                    let v = self.eval(builder, &env, rhs);
                    env.insert(name.clone(), v.into());
                    last = v;
                }
                Stmt::Write(base, fld, rhs) => {
                    let bv = self.eval(builder, &env, base);
                    let p = builder.build_int_to_ptr(bv, self.ptr, "base").unwrap();
                    let idx = self.slot[fld];
                    let gep = unsafe {
                        builder
                            .build_gep(self.i64t, p, &[self.i64t.const_int(idx as u64, false)], "fld")
                            .unwrap()
                    };
                    let rv = self.eval(builder, &env, rhs);
                    builder.build_store(gep, rv).unwrap();
                }
                Stmt::Free(name) => {
                    let v = env[name].into_int_value();
                    let p = builder.build_int_to_ptr(v, self.ptr, "freep").unwrap();
                    builder.build_call(free, &[p.into()], "").unwrap();
                }
                Stmt::Expr(e) => {
                    last = self.eval(builder, &env, e);
                }
            }
        }
        let ret = match &f.tail {
            Some(e) => self.eval(builder, &env, e),
            None => last,
        };
        builder.build_return(Some(&ret)).unwrap();
    }

    fn eval(&self, builder: &Builder<'c>, env: &HashMap<String, BasicValueEnum<'c>>, e: &Expr) -> IntValue<'c> {
        let nslots = self.slot.len().max(1) as u64;
        match e {
            Expr::Int(n) => self.i64t.const_int(*n as u64, true),
            Expr::Null | Expr::Unit => self.i64t.const_zero(),
            Expr::Var(x) => env[x].into_int_value(),
            Expr::AddrOf(x) => env[x].into_int_value(),
            Expr::Alloc(_, fields) => {
                let sz = self.i64t.const_int(nslots * 8, false);
                let raw = builder
                    .build_call(self.malloc, &[sz.into()], "cell")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_pointer_value();
                for (f, fe) in fields {
                    let idx = self.slot[f];
                    let gep = unsafe {
                        builder
                            .build_gep(self.i64t, raw, &[self.i64t.const_int(idx as u64, false)], "init")
                            .unwrap()
                    };
                    let v = self.eval(builder, env, fe);
                    builder.build_store(gep, v).unwrap();
                }
                builder.build_ptr_to_int(raw, self.i64t, "cellint").unwrap()
            }
            Expr::Field(obj, fld) => {
                let base = self.eval(builder, env, obj);
                let p = builder.build_int_to_ptr(base, self.ptr, "rdbase").unwrap();
                let idx = self.slot[fld];
                let gep = unsafe {
                    builder
                        .build_gep(self.i64t, p, &[self.i64t.const_int(idx as u64, false)], "rdfld")
                        .unwrap()
                };
                builder.build_load(self.i64t, gep, "rd").unwrap().into_int_value()
            }
            Expr::Call(fname, args) => {
                let fv = self.fns[fname];
                let args: Vec<_> = args.iter().map(|a| self.eval(builder, env, a).into()).collect();
                builder
                    .build_call(fv, &args, "call")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_int_value()
            }
        }
    }
}

fn collect_fields_stmt(s: &Stmt, slot: &mut HashMap<String, u32>) {
    match s {
        Stmt::Let(_, _, e) | Stmt::Expr(e) => collect_fields_expr(e, slot),
        Stmt::Write(b, f, r) => {
            add(f, slot);
            collect_fields_expr(b, slot);
            collect_fields_expr(r, slot);
        }
        Stmt::Free(_) => {}
    }
}

fn collect_fields_expr(e: &Expr, slot: &mut HashMap<String, u32>) {
    match e {
        Expr::Alloc(_, fs) => {
            for (f, fe) in fs {
                add(f, slot);
                collect_fields_expr(fe, slot);
            }
        }
        Expr::Field(o, f) => {
            add(f, slot);
            collect_fields_expr(o, slot);
        }
        Expr::Call(_, args) => {
            for a in args {
                collect_fields_expr(a, slot);
            }
        }
        _ => {}
    }
}

fn add(f: &str, slot: &mut HashMap<String, u32>) {
    let n = slot.len() as u32;
    slot.entry(f.to_string()).or_insert(n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn smoke_jit() {
        assert_eq!(smoke(), 42);
    }

    #[test]
    fn alloc_write_read_free_runs() {
        let prog = parse(
            "struct C { val: Int }
             fn main() -> Int {
               let a = alloc C { val: 0 };
               a.val = 42;
               let r = a.val;
               free a;
               r
             }",
        )
        .unwrap();
        assert!(crate::check::check(&prog).is_empty());
        assert_eq!(compile_and_run(&prog), 42);
    }

    #[test]
    fn function_call_runs() {
        // a factory + a consumer, threaded across calls, lowered to native code
        let prog = parse(
            "struct C { val: Int }
             fn make(n: Int) -> Own<C> { alloc C { val: n } }
             fn consume(c: Own<C>) -> Int { let r = c.val; free c; r }
             fn main() -> Int { let a = make(99); consume(a) }",
        )
        .unwrap();
        assert!(crate::check::check(&prog).is_empty(), "{:?}", crate::check::check(&prog));
        assert_eq!(compile_and_run(&prog), 99);
    }
}
