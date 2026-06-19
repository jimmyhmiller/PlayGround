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
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
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
    free: FunctionValue<'c>,
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
    // reserve slots used by the runtime cells of the built-in data structures:
    //   Vec cons cell: $vhd/$vtl ; pair: $fst/$snd ; DLL node: $nnext/$nprev/$nelem
    for f in ["$vhd", "$vtl", "$fst", "$snd", "$nnext", "$nprev", "$nelem"] {
        add(f, &mut slot);
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
        free,
        slot,
        fns,
    };

    for f in &prog.funcs {
        cg.lower_fn(&builder, f);
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
    fn nslots(&self) -> u64 {
        self.slot.len().max(1) as u64
    }
    fn gep(&self, b: &Builder<'c>, p: PointerValue<'c>, idx: u32, name: &str) -> PointerValue<'c> {
        unsafe {
            b.build_gep(self.i64t, p, &[self.i64t.const_int(idx as u64, false)], name)
                .unwrap()
        }
    }
    fn store(&self, b: &Builder<'c>, p: PointerValue<'c>, idx: u32, v: IntValue<'c>) {
        b.build_store(self.gep(b, p, idx, "s"), v).unwrap();
    }
    fn load(&self, b: &Builder<'c>, p: PointerValue<'c>, idx: u32, name: &str) -> IntValue<'c> {
        b.build_load(self.i64t, self.gep(b, p, idx, name), name)
            .unwrap()
            .into_int_value()
    }
    fn malloc_cell(&self, b: &Builder<'c>, name: &str) -> PointerValue<'c> {
        let sz = self.i64t.const_int(self.nslots() * 8, false);
        b.build_call(self.malloc, &[sz.into()], name)
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_pointer_value()
    }
    fn toptr(&self, b: &Builder<'c>, v: IntValue<'c>, name: &str) -> PointerValue<'c> {
        b.build_int_to_ptr(v, self.ptr, name).unwrap()
    }
    fn toint(&self, b: &Builder<'c>, p: PointerValue<'c>, name: &str) -> IntValue<'c> {
        b.build_ptr_to_int(p, self.i64t, name).unwrap()
    }
    fn freep(&self, b: &Builder<'c>, v: IntValue<'c>) {
        let p = self.toptr(b, v, "fp");
        b.build_call(self.free, &[p.into()], "").unwrap();
    }
    /// build a 2-slot pair cell {fst, snd} and return it as an i64
    fn mk_pair(&self, b: &Builder<'c>, fst: IntValue<'c>, snd: IntValue<'c>) -> IntValue<'c> {
        let p = self.malloc_cell(b, "pair");
        self.store(b, p, self.slot["$fst"], fst);
        self.store(b, p, self.slot["$snd"], snd);
        self.toint(b, p, "pairint")
    }

    fn lower_fn(&self, builder: &Builder<'c>, f: &Func) {
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
                    builder.build_call(self.free, &[p.into()], "").unwrap();
                }
                Stmt::LetPair(x, y, rhs) => {
                    let pv = self.eval(builder, &env, rhs);
                    let p = self.toptr(builder, pv, "pp");
                    let xv = self.load(builder, p, self.slot["$fst"], "fst");
                    let yv = self.load(builder, p, self.slot["$snd"], "snd");
                    self.freep(builder, pv);
                    env.insert(x.clone(), xv.into());
                    env.insert(y.clone(), yv.into());
                    last = xv;
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
                // length-indexed Vec built-ins lower to a linked stack of cons
                // cells; the length is erased (never materialised at runtime).
                let hd = self.slot["$vhd"];
                let tl = self.slot["$vtl"];
                let gep = |builder: &Builder<'c>, p, idx: u32, name| unsafe {
                    builder
                        .build_gep(self.i64t, p, &[self.i64t.const_int(idx as u64, false)], name)
                        .unwrap()
                };
                match fname.as_str() {
                    "vnew" => return self.i64t.const_zero(), // empty == null
                    "vpush" => {
                        let v = self.eval(builder, env, &args[0]);
                        let x = self.eval(builder, env, &args[1]);
                        let sz = self.i64t.const_int(nslots * 8, false);
                        let raw = builder
                            .build_call(self.malloc, &[sz.into()], "cons")
                            .unwrap()
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_pointer_value();
                        builder.build_store(gep(builder, raw, hd, "hd"), x).unwrap();
                        builder.build_store(gep(builder, raw, tl, "tl"), v).unwrap();
                        return builder.build_ptr_to_int(raw, self.i64t, "consint").unwrap();
                    }
                    "vhead" => {
                        let v = self.eval(builder, env, &args[0]);
                        let p = builder.build_int_to_ptr(v, self.ptr, "vh").unwrap();
                        return builder
                            .build_load(self.i64t, gep(builder, p, hd, "hd"), "head")
                            .unwrap()
                            .into_int_value();
                    }
                    "vtail" => {
                        let v = self.eval(builder, env, &args[0]);
                        let p = builder.build_int_to_ptr(v, self.ptr, "vt").unwrap();
                        let t = builder
                            .build_load(self.i64t, gep(builder, p, tl, "tl"), "tail")
                            .unwrap()
                            .into_int_value();
                        builder.build_call(self.free, &[p.into()], "").unwrap();
                        return t;
                    }
                    "vfree" => {
                        let _ = self.eval(builder, env, &args[0]); // empty == null, nothing to free
                        return self.i64t.const_zero();
                    }
                    // linear-cursor list = a CIRCULAR doubly-linked list with a
                    // sentinel node (so insert/remove are branch-free). The list
                    // value IS the sentinel; a cursor IS a real node pointer.
                    "lnew" => {
                        let nn = self.slot["$nnext"];
                        let np = self.slot["$nprev"];
                        let s = self.malloc_cell(builder, "sentinel");
                        let si = self.toint(builder, s, "si");
                        self.store(builder, s, nn, si); // s.next = s
                        self.store(builder, s, np, si); // s.prev = s
                        return si;
                    }
                    "linsert" => {
                        let (nn, np, ne) =
                            (self.slot["$nnext"], self.slot["$nprev"], self.slot["$nelem"]);
                        let li = self.eval(builder, env, &args[0]); // sentinel
                        let x = self.eval(builder, env, &args[1]);
                        let s = self.toptr(builder, li, "s");
                        let node = self.malloc_cell(builder, "node");
                        let ni = self.toint(builder, node, "ni");
                        let prev = self.load(builder, s, np, "sp"); // old tail
                        let prevp = self.toptr(builder, prev, "pp");
                        self.store(builder, node, ne, x);
                        self.store(builder, node, np, prev); // node.prev = old tail
                        self.store(builder, node, nn, li); // node.next = sentinel
                        self.store(builder, prevp, nn, ni); // old_tail.next = node
                        self.store(builder, s, np, ni); // sentinel.prev = node
                        return self.mk_pair(builder, ni, li); // (cursor, list)
                    }
                    "lremove" => {
                        let (nn, np, ne) =
                            (self.slot["$nnext"], self.slot["$nprev"], self.slot["$nelem"]);
                        let li = self.eval(builder, env, &args[0]); // sentinel
                        let ci = self.eval(builder, env, &args[1]); // node = cursor
                        let node = self.toptr(builder, ci, "c");
                        let prev = self.load(builder, node, np, "np");
                        let next = self.load(builder, node, nn, "nx");
                        let prevp = self.toptr(builder, prev, "pp");
                        let nextp = self.toptr(builder, next, "xp");
                        self.store(builder, prevp, nn, next); // prev.next = next
                        self.store(builder, nextp, np, prev); // next.prev = prev
                        let elem = self.load(builder, node, ne, "elem");
                        self.freep(builder, ci); // O(1) free of the node
                        return self.mk_pair(builder, elem, li); // (value, list)
                    }
                    "lfree" => {
                        let li = self.eval(builder, env, &args[0]);
                        self.freep(builder, li); // free the sentinel
                        return self.i64t.const_zero();
                    }
                    _ => {}
                }
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
        Stmt::Let(_, _, e) | Stmt::Expr(e) | Stmt::LetPair(_, _, e) => collect_fields_expr(e, slot),
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

    #[test]
    fn dependent_vec_runs() {
        // length-indexed vector lowered to a linked stack; length is erased.
        let prog = parse(
            "fn main() -> Int {
               let v0 = vnew();
               let v1 = vpush(v0, 10);
               let v2 = vpush(v1, 20);
               let top = vhead(v2);
               let v1b = vtail(v2);
               let v0b = vtail(v1b);
               vfree(v0b);
               top
             }",
        )
        .unwrap();
        assert!(crate::check::check(&prog).is_empty(), "{:?}", crate::check::check(&prog));
        assert_eq!(compile_and_run(&prog), 20);
    }

    #[test]
    fn linear_cursor_dll_runs() {
        // a real intrusive doubly-linked list: insert 3, O(1)-remove the MIDDLE
        // by its cursor, then the rest. Lowers to branch-free pointer surgery.
        let prog = parse(
            "fn main() -> Int {
               let l0 = lnew();
               let (c1, l1) = linsert(l0, 10);
               let (c2, l2) = linsert(l1, 20);
               let (c3, l3) = linsert(l2, 30);
               let (x, l4) = lremove(l3, c2);
               let (y, l5) = lremove(l4, c1);
               let (z, l6) = lremove(l5, c3);
               lfree(l6);
               x
             }",
        )
        .unwrap();
        assert!(crate::check::check(&prog).is_empty(), "{:?}", crate::check::check(&prog));
        assert_eq!(compile_and_run(&prog), 20); // removed-middle value
    }
}
