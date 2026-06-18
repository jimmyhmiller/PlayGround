//! LLVM backend (behind the `llvm` feature), via inkwell + system LLVM 18.
//!
//! Because permissions / regions / ghosts are ERASED by the time we get here,
//! codegen is ordinary low-level lowering: cells are malloc'd blocks of i64
//! slots, fields are slot indices, `free` is libc `free`. The safety was all
//! discharged statically in `crate::check`.

use crate::ast::{Expr, Program, Stmt};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::values::{BasicValueEnum, IntValue};
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

/// Lower a checked v0 program to `tally_main() -> i64`, JIT it, and run it.
/// Returns whatever the program's final expression evaluates to as an i64
/// (cells become real malloc'd memory; `free` calls libc free).
pub fn compile_and_run(prog: &Program) -> i64 {
    // assign every field name a stable slot index; cells are uniform i64 arrays
    let mut slot: HashMap<String, u32> = HashMap::new();
    for s in prog {
        collect_fields(s, &mut slot);
    }

    let ctx = Context::create();
    let module = ctx.create_module("tally");
    let builder = ctx.create_builder();
    let i64t = ctx.i64_type();
    let ptr = ctx.ptr_type(AddressSpace::default());

    // extern: i8* malloc(i64), void free(i8*)
    let malloc = module.add_function("malloc", ptr.fn_type(&[i64t.into()], false), None);
    let free = module.add_function("free", ctx.void_type().fn_type(&[ptr.into()], false), None);

    let main = module.add_function("tally_main", i64t.fn_type(&[], false), None);
    let entry = ctx.append_basic_block(main, "entry");
    builder.position_at_end(entry);

    let mut env: HashMap<String, BasicValueEnum> = HashMap::new();
    let mut last = i64t.const_zero();

    for s in prog {
        match s {
            Stmt::Let(name, rhs) => {
                let v = eval_expr(&builder, &ctx, i64t, ptr, malloc, &slot, &env, rhs);
                env.insert(name.clone(), v.into());
                last = v;
            }
            Stmt::Write(base, fld, rhs) => {
                let bv = eval_expr(&builder, &ctx, i64t, ptr, malloc, &slot, &env, base);
                let p = builder.build_int_to_ptr(bv, ptr, "base").unwrap();
                let idx = *slot.get(fld).unwrap();
                let gep = unsafe {
                    builder
                        .build_gep(i64t, p, &[i64t.const_int(idx as u64, false)], "fld")
                        .unwrap()
                };
                let rv = eval_expr(&builder, &ctx, i64t, ptr, malloc, &slot, &env, rhs);
                builder.build_store(gep, rv).unwrap();
            }
            Stmt::Free(name) => {
                let v: IntValue = env.get(name).unwrap().into_int_value();
                let p = builder.build_int_to_ptr(v, ptr, "freep").unwrap();
                builder.build_call(free, &[p.into()], "").unwrap();
            }
            Stmt::Expr(e) => {
                last = eval_expr(&builder, &ctx, i64t, ptr, malloc, &slot, &env, e);
            }
        }
    }
    builder.build_return(Some(&last)).unwrap();

    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    unsafe {
        let f: JitFunction<unsafe extern "C" fn() -> i64> =
            ee.get_function("tally_main").unwrap();
        f.call()
    }
}

fn eval_expr<'c>(
    builder: &inkwell::builder::Builder<'c>,
    ctx: &'c Context,
    i64t: inkwell::types::IntType<'c>,
    ptr: inkwell::types::PointerType<'c>,
    malloc: inkwell::values::FunctionValue<'c>,
    slot: &HashMap<String, u32>,
    env: &HashMap<String, BasicValueEnum<'c>>,
    e: &Expr,
) -> IntValue<'c> {
    let nslots = slot.len().max(1) as u64;
    match e {
        Expr::Int(n) => i64t.const_int(*n as u64, true),
        Expr::Null | Expr::Unit => i64t.const_zero(),
        Expr::Var(x) => env.get(x).unwrap().into_int_value(),
        Expr::AddrOf(x) => env.get(x).unwrap().into_int_value(),
        Expr::Alloc(fields) => {
            let sz = i64t.const_int(nslots * 8, false);
            let raw = builder
                .build_call(malloc, &[sz.into()], "cell")
                .unwrap()
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            for (f, fe) in fields {
                let idx = *slot.get(f).unwrap();
                let gep = unsafe {
                    builder
                        .build_gep(i64t, raw, &[i64t.const_int(idx as u64, false)], "init")
                        .unwrap()
                };
                let v = eval_expr(builder, ctx, i64t, ptr, malloc, slot, env, fe);
                builder.build_store(gep, v).unwrap();
            }
            builder.build_ptr_to_int(raw, i64t, "cellint").unwrap()
        }
        Expr::Field(obj, fld) => {
            let base = eval_expr(builder, ctx, i64t, ptr, malloc, slot, env, obj);
            let p = builder.build_int_to_ptr(base, ptr, "rdbase").unwrap();
            let idx = *slot.get(fld).unwrap();
            let gep = unsafe {
                builder
                    .build_gep(i64t, p, &[i64t.const_int(idx as u64, false)], "rdfld")
                    .unwrap()
            };
            builder
                .build_load(i64t, gep, "rd")
                .unwrap()
                .into_int_value()
        }
    }
}

fn collect_fields(s: &Stmt, slot: &mut HashMap<String, u32>) {
    let add = |f: &str, slot: &mut HashMap<String, u32>| {
        let n = slot.len() as u32;
        slot.entry(f.to_string()).or_insert(n);
    };
    fn fields_in_expr(e: &Expr, add: &mut dyn FnMut(&str)) {
        match e {
            Expr::Alloc(fs) => {
                for (f, fe) in fs {
                    add(f);
                    fields_in_expr(fe, add);
                }
            }
            Expr::Field(o, f) => {
                add(f);
                fields_in_expr(o, add);
            }
            _ => {}
        }
    }
    match s {
        Stmt::Let(_, e) | Stmt::Expr(e) => fields_in_expr(e, &mut |f| add(f, slot)),
        Stmt::Write(b, f, r) => {
            add(f, slot);
            fields_in_expr(b, &mut |f| add(f, slot));
            fields_in_expr(r, &mut |f| add(f, slot));
        }
        Stmt::Free(_) => {}
    }
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
        // allocate a cell, store 42, read it back into the result, free.
        let prog = parse(
            "let a = alloc { val: 0 };
             a.val = 42;
             let r = a.val;
             free a;
             r;",
        )
        .unwrap();
        assert!(crate::check::check(&prog).is_empty());
        assert_eq!(compile_and_run(&prog), 42);
    }
}
