use std::cell::{Cell, RefCell};
use std::process::Command;
use std::rc::Rc;

use dynir::interp::{ExternCallResult, InterpResult, Interpreter};
use dynlower::{JitFunction, call_jit};
use dynvalue::{Decoded, NanBox, TagScheme};

use crate::bytecode::{self, Constant};
use crate::runtime::{self, LuaRuntime, make_nil, is_closure, closure_func_id};
use crate::translate::{self, TranslatedFunction};

// ── JIT extern infrastructure ──────────────────────────────
//
// JIT externs are raw `extern "C"` function pointers — they can't capture
// environment like Rust closures. We use thread-local storage to give them
// access to the LuaRuntime and child JIT code pointers.

struct JitContext {
    rt: *mut LuaRuntime,
    child_jit_ptrs: Vec<*const u8>,
    child_info: Vec<(u8, u8)>,       // (num_params, max_stack) per child
    child_consts: Vec<Vec<String>>,   // constant table per child
}

thread_local! {
    static JIT_CTX: Cell<*mut JitContext> = const { Cell::new(std::ptr::null_mut()) };
}

fn with_jit_ctx<R>(f: impl FnOnce(&JitContext, &mut LuaRuntime) -> R) -> R {
    let ctx_ptr = JIT_CTX.with(|c| c.get());
    let ctx = unsafe { &*ctx_ptr };
    let rt = unsafe { &mut *ctx.rt };
    f(ctx, rt)
}

// ── JIT extern "C" functions ───────────────────────────────
// These must be declared in the SAME ORDER as translate.rs declares externs.

extern "C" fn jit_lua_add(a: u64, b: u64) -> u64 {
    std::panic::catch_unwind(|| with_jit_ctx(|_, rt| rt.lua_add(a, b)))
        .unwrap_or_else(|_| { eprintln!("PANIC jit_lua_add a={:#018x} b={:#018x}", a, b); std::process::exit(99); })
}
extern "C" fn jit_lua_sub(a: u64, b: u64) -> u64 {
    std::panic::catch_unwind(|| with_jit_ctx(|_, rt| rt.lua_sub(a, b)))
        .unwrap_or_else(|_| { eprintln!("PANIC jit_lua_sub a={:#018x} b={:#018x}", a, b); std::process::exit(99); })
}
extern "C" fn jit_lua_mul(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_mul(a, b)) }
extern "C" fn jit_lua_div(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_div(a, b)) }
extern "C" fn jit_lua_mod(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_mod(a, b)) }
extern "C" fn jit_lua_pow(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_pow(a, b)) }
extern "C" fn jit_lua_unm(a: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_unm(a)) }
extern "C" fn jit_lua_not(a: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_not(a)) }
extern "C" fn jit_lua_len(a: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_len(a)) }
extern "C" fn jit_lua_eq(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_eq(a, b)) }
extern "C" fn jit_lua_lt(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_lt(a, b)) }
extern "C" fn jit_lua_le(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_le(a, b)) }
extern "C" fn jit_lua_concat(a: u64, b: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_concat(a, b)) }
extern "C" fn jit_lua_getglobal(name: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_getglobal(name)) }
extern "C" fn jit_lua_setglobal(name: u64, val: u64) -> u64 { with_jit_ctx(|_, rt| { rt.lua_setglobal(name, val); 0 }) }
extern "C" fn jit_lua_newtable() -> u64 { with_jit_ctx(|_, rt| rt.lua_newtable()) }
extern "C" fn jit_lua_gettable(table: u64, key: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_gettable(table, key)) }
extern "C" fn jit_lua_settable(table: u64, key: u64, val: u64) -> u64 { with_jit_ctx(|_, rt| { rt.lua_settable(table, key, val); 0 }) }

extern "C" fn jit_lua_call(func: u64, base: u64, nargs: u64) -> u64 {
    let base = base as usize;
    let nargs = nargs as usize;

    if is_closure(func) {
        let ctx_ptr = JIT_CTX.with(|c| c.get());
        let ctx = unsafe { &*ctx_ptr };
        let rt = unsafe { &mut *ctx.rt };

        let func_id = closure_func_id(func).unwrap();
        let code_ptr = ctx.child_jit_ptrs[func_id];

        // Read args from register_file
        let args: Vec<u64> = (0..nargs).map(|i| rt.register_file[base + i]).collect();

        // Swap constants
        let saved = std::mem::replace(&mut rt.constants, ctx.child_consts[func_id].clone());

        // Build init_regs: [closure, args..., nil_padding...]
        let (_, max_stack) = ctx.child_info[func_id];
        let total = max_stack as usize + 1;
        let mut init_regs = vec![make_nil(); total];
        init_regs[0] = func;
        for (i, &arg) in args.iter().enumerate() {
            if i + 1 < total { init_regs[i + 1] = arg; }
        }

        // Call child JIT code
        let result = unsafe { call_jit(code_ptr, &init_regs) };

        // Restore constants
        let rt = unsafe { &mut *ctx.rt };
        rt.constants = saved;

        result
    } else {
        with_jit_ctx(|_, rt| rt.lua_call(func, base, nargs))
    }
}

extern "C" fn jit_lua_setlist(table: u64, base: u64, offset: u64, count: u64) -> u64 {
    with_jit_ctx(|_, rt| { rt.lua_setlist_from_regfile(table, base as usize, offset as usize, count as usize); 0 })
}
extern "C" fn jit_lua_forprep(init: u64, limit: u64, step: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_forprep(init, limit, step)) }
extern "C" fn jit_lua_forloop(index: u64, limit: u64, step: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_forloop(index, limit, step)) }
extern "C" fn jit_lua_is_nil(v: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_is_nil(v)) }
extern "C" fn jit_lua_self(table: u64, key: u64) -> u64 { with_jit_ctx(|_, rt| rt.lua_gettable(table, key)) }

extern "C" fn jit_lua_store_reg(idx: u64, val: u64) -> u64 {
    with_jit_ctx(|_, rt| {
        let idx = idx as usize;
        if idx >= rt.register_file.len() { rt.register_file.resize(idx + 1, make_nil()); }
        rt.register_file[idx] = val;
        0
    })
}

extern "C" fn jit_lua_make_closure(func_id: u64, _num: u64) -> u64 { runtime::make_closure(func_id as usize, &[]) }
extern "C" fn jit_lua_make_closure_1(func_id: u64, _num: u64, u0: u64) -> u64 { runtime::make_closure(func_id as usize, &[u0]) }
extern "C" fn jit_lua_make_closure_2(func_id: u64, _num: u64, u0: u64, u1: u64) -> u64 { runtime::make_closure(func_id as usize, &[u0, u1]) }
extern "C" fn jit_lua_make_closure_3(func_id: u64, _num: u64, u0: u64, u1: u64, u2: u64) -> u64 { runtime::make_closure(func_id as usize, &[u0, u1, u2]) }
extern "C" fn jit_lua_make_closure_4(func_id: u64, _num: u64, u0: u64, u1: u64, u2: u64, u3: u64) -> u64 { runtime::make_closure(func_id as usize, &[u0, u1, u2, u3]) }

/// Build the extern pointer array matching the order in translate.rs.
fn jit_extern_ptrs() -> Vec<*const u8> {
    vec![
        jit_lua_add as *const u8,
        jit_lua_sub as *const u8,
        jit_lua_mul as *const u8,
        jit_lua_div as *const u8,
        jit_lua_mod as *const u8,
        jit_lua_pow as *const u8,
        jit_lua_unm as *const u8,
        jit_lua_not as *const u8,
        jit_lua_len as *const u8,
        jit_lua_eq as *const u8,
        jit_lua_lt as *const u8,
        jit_lua_le as *const u8,
        jit_lua_concat as *const u8,
        jit_lua_getglobal as *const u8,
        jit_lua_setglobal as *const u8,
        jit_lua_newtable as *const u8,
        jit_lua_gettable as *const u8,
        jit_lua_settable as *const u8,
        jit_lua_call as *const u8,
        jit_lua_setlist as *const u8,
        jit_lua_forprep as *const u8,
        jit_lua_forloop as *const u8,
        jit_lua_is_nil as *const u8,
        jit_lua_self as *const u8,
        jit_lua_store_reg as *const u8,
        jit_lua_make_closure as *const u8,
        jit_lua_make_closure_1 as *const u8,
        jit_lua_make_closure_2 as *const u8,
        jit_lua_make_closure_3 as *const u8,
        jit_lua_make_closure_4 as *const u8,
    ]
}

/// Compile a Lua source string to bytecode using luac 5.1.
fn compile_lua(source: &str) -> Vec<u8> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);

    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("lua2dynir_test_{}.lua", id));
    let out_path = dir.join(format!("lua2dynir_test_{}.luac", id));

    std::fs::write(&src_path, source).unwrap();

    let luac = "/tmp/lua-5.1.5/src/luac";
    let status = Command::new(luac)
        .args(["-o", out_path.to_str().unwrap(), src_path.to_str().unwrap()])
        .status()
        .expect("failed to run luac — build Lua 5.1 at /tmp/lua-5.1.5 first");

    assert!(status.success(), "luac compilation failed");
    let data = std::fs::read(&out_path).unwrap();
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);
    data
}

type CallHandler = Rc<dyn Fn(u64, &[u64]) -> u64>;

/// Bind all extern functions to an interpreter, using the shared runtime and call handler.
fn bind_all_externs(
    interp: &mut Interpreter<NanBox>,
    extern_names: &[String],
    rt: &Rc<RefCell<LuaRuntime>>,
    handler: &CallHandler,
) {
    for name in extern_names {
        match name.as_str() {
            "lua_add" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_add", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_add(args[0], args[1])))
                });
            }
            "lua_sub" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_sub", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_sub(args[0], args[1])))
                });
            }
            "lua_mul" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_mul", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_mul(args[0], args[1])))
                });
            }
            "lua_div" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_div", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_div(args[0], args[1])))
                });
            }
            "lua_mod" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_mod", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_mod(args[0], args[1])))
                });
            }
            "lua_pow" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_pow", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_pow(args[0], args[1])))
                });
            }
            "lua_unm" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_unm", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_unm(args[0])))
                });
            }
            "lua_not" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_not", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_not(args[0])))
                });
            }
            "lua_len" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_len", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_len(args[0])))
                });
            }
            "lua_eq" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_eq", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_eq(args[0], args[1])))
                });
            }
            "lua_lt" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_lt", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_lt(args[0], args[1])))
                });
            }
            "lua_le" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_le", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_le(args[0], args[1])))
                });
            }
            "lua_concat" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_concat", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_concat(args[0], args[1])))
                });
            }
            "lua_getglobal" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_getglobal", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_getglobal(args[0])))
                });
            }
            "lua_setglobal" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_setglobal", move |args| {
                    rt.borrow_mut().lua_setglobal(args[0], args[1]);
                    ExternCallResult::Value(None)
                });
            }
            "lua_newtable" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_newtable", move |_args| {
                    ExternCallResult::Value(Some(rt.borrow_mut().lua_newtable()))
                });
            }
            "lua_gettable" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_gettable", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_gettable(args[0], args[1])))
                });
            }
            "lua_settable" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_settable", move |args| {
                    rt.borrow().lua_settable(args[0], args[1], args[2]);
                    ExternCallResult::Value(None)
                });
            }
            "lua_call" => {
                let rt = rt.clone();
                let handler = handler.clone();
                interp.bind_by_name("lua_call", move |args| {
                    let func = args[0];
                    let base = args[1] as usize;
                    let nargs = args[2] as usize;

                    if is_closure(func) {
                        // Read args from register_file, dispatch to handler
                        let call_args: Vec<u64> = {
                            let r = rt.borrow();
                            (0..nargs).map(|i| r.register_file[base + i]).collect()
                        };
                        let result = handler(func, &call_args);
                        ExternCallResult::Value(Some(result))
                    } else {
                        // Built-in dispatch
                        let result = rt.borrow_mut().lua_call(func, base, nargs);
                        ExternCallResult::Value(Some(result))
                    }
                });
            }
            "lua_setlist" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_setlist", move |args| {
                    let table = args[0];
                    let base = args[1] as usize;
                    let offset = args[2] as usize;
                    let count = args[3] as usize;
                    rt.borrow().lua_setlist_from_regfile(table, base, offset, count);
                    ExternCallResult::Value(None)
                });
            }
            "lua_forprep" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_forprep", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_forprep(args[0], args[1], args[2])))
                });
            }
            "lua_forloop" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_forloop", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_forloop(args[0], args[1], args[2])))
                });
            }
            "lua_is_nil" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_is_nil", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_is_nil(args[0])))
                });
            }
            "lua_self" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_self", move |args| {
                    ExternCallResult::Value(Some(rt.borrow().lua_gettable(args[0], args[1])))
                });
            }
            "lua_store_reg" => {
                let rt = rt.clone();
                interp.bind_by_name("lua_store_reg", move |args| {
                    let idx = args[0] as usize;
                    let val = args[1];
                    let mut r = rt.borrow_mut();
                    if idx >= r.register_file.len() {
                        r.register_file.resize(idx + 1, make_nil());
                    }
                    r.register_file[idx] = val;
                    ExternCallResult::Value(None)
                });
            }
            "lua_make_closure" => {
                interp.bind_by_name("lua_make_closure", move |args| {
                    let func_id = args[0] as usize;
                    let _num_upvals = args[1] as usize;
                    ExternCallResult::Value(Some(runtime::make_closure(func_id, &[])))
                });
            }
            "lua_make_closure_1" => {
                interp.bind_by_name("lua_make_closure_1", move |args| {
                    let func_id = args[0] as usize;
                    let _num_upvals = args[1] as usize;
                    ExternCallResult::Value(Some(runtime::make_closure(func_id, &[args[2]])))
                });
            }
            "lua_make_closure_2" => {
                interp.bind_by_name("lua_make_closure_2", move |args| {
                    let func_id = args[0] as usize;
                    ExternCallResult::Value(Some(runtime::make_closure(func_id, &[args[2], args[3]])))
                });
            }
            "lua_make_closure_3" => {
                interp.bind_by_name("lua_make_closure_3", move |args| {
                    let func_id = args[0] as usize;
                    ExternCallResult::Value(Some(runtime::make_closure(func_id, &[args[2], args[3], args[4]])))
                });
            }
            "lua_make_closure_4" => {
                interp.bind_by_name("lua_make_closure_4", move |args| {
                    let func_id = args[0] as usize;
                    ExternCallResult::Value(Some(runtime::make_closure(func_id, &[args[2], args[3], args[4], args[5]])))
                });
            }
            other => {
                panic!("unknown extern function: {}", other);
            }
        }
    }
}

/// Run a Lua program: compile → parse → translate → interpret via DynIR Interpreter.
/// Returns the result as a NanBox u64 and the LuaRuntime (for checking output).
fn run_lua(source: &str) -> (u64, LuaRuntime) {
    let bytecode_data = compile_lua(source);
    let chunk = bytecode::parse(&bytecode_data).expect("bytecode parse failed");

    // Translate ALL protos up front (main + children)
    let main_tf = translate::translate(&chunk.main);
    let child_tfs: Vec<TranslatedFunction> = chunk.main.protos.iter()
        .map(|p| translate::translate(p))
        .collect();

    // Collect child proto metadata
    let child_info: Vec<(u8, u8)> = chunk.main.protos.iter()
        .map(|p| (p.num_params, p.max_stack_size))
        .collect();

    // Collect per-child constant tables (each proto has its own)
    let child_constants: Vec<Vec<String>> = chunk.main.protos.iter()
        .map(|p| p.constants.iter().map(|c| match c {
            Constant::String(s) => s.clone(),
            _ => String::new(),
        }).collect())
        .collect();

    let rt = Rc::new(RefCell::new(LuaRuntime::new(&chunk.main.constants)));

    // Wrap shared data in Rc for closure capture
    let children = Rc::new(child_tfs);
    let info = Rc::new(child_info);
    let consts = Rc::new(child_constants);

    // Create self-referencing call handler for closure dispatch
    let handler_cell: Rc<RefCell<Option<CallHandler>>> = Rc::new(RefCell::new(None));

    let handler: CallHandler = {
        let rt = rt.clone();
        let children = children.clone();
        let info = info.clone();
        let consts = consts.clone();
        let handler_cell = handler_cell.clone();

        Rc::new(move |closure_val: u64, args: &[u64]| -> u64 {
            let func_id = closure_func_id(closure_val).unwrap();
            let tf = &children[func_id];
            let (_num_params, max_stack) = info[func_id];

            // Swap constants to child's constant table
            let saved_constants = {
                let mut r = rt.borrow_mut();
                std::mem::replace(&mut r.constants, consts[func_id].clone())
            };

            // Create child interpreter
            let mut child_interp = Interpreter::<NanBox>::new(&tf.function);
            let h = handler_cell.borrow().as_ref().unwrap().clone();
            bind_all_externs(&mut child_interp, &tf.extern_names, &rt, &h);

            // Build init regs: [closure, args..., nil_padding...]
            let total = max_stack as usize + 1; // +1 for closure param
            let mut init_regs = vec![make_nil(); total];
            init_regs[0] = closure_val;
            for (i, &arg) in args.iter().enumerate() {
                if i + 1 < total {
                    init_regs[i + 1] = arg;
                }
            }

            let result = match child_interp.run(&init_regs).unwrap() {
                InterpResult::Value(v) => v,
                _ => make_nil(),
            };

            // Restore constants
            {
                let mut r = rt.borrow_mut();
                r.constants = saved_constants;
            }

            result
        })
    };

    *handler_cell.borrow_mut() = Some(handler.clone());

    // Create main interpreter and bind externs
    let result;
    {
        let func = &main_tf.function;
        let mut interp = Interpreter::<NanBox>::new(func);
        bind_all_externs(&mut interp, &main_tf.extern_names, &rt, &handler);

        // Run with [nil_closure, nil_regs...]
        let total = chunk.main.max_stack_size as usize + 1; // +1 for closure param
        let init_regs = vec![make_nil(); total];
        result = match interp.run(&init_regs).unwrap() {
            InterpResult::Value(v) => v,
            _ => make_nil(),
        };
    } // interp dropped here, releasing Rc clones

    // Break reference cycle and unwrap
    *handler_cell.borrow_mut() = None;
    drop(handler);

    // ── JIT path: re-run with JIT-compiled child closures ────────
    // If there are child protos, JIT-compile them and re-run the main chunk
    // with a JIT-based call handler, then compare results.
    if !chunk.main.protos.is_empty() {
        // JIT-compile each child function
        let extern_ptrs = jit_extern_ptrs();
        let child_jits: Vec<JitFunction> = children.iter()
            .map(|tf| JitFunction::compile(&tf.function, &extern_ptrs))
            .collect();
        let child_jit_ptrs: Vec<*const u8> = child_jits.iter()
            .map(|j| j.as_ptr())
            .collect();

        // Create fresh runtime for JIT run
        let mut jit_rt = LuaRuntime::new(&chunk.main.constants);
        let child_consts_owned: Vec<Vec<String>> = consts.to_vec();
        let child_info_owned: Vec<(u8, u8)> = info.to_vec();

        // Set up JIT context in thread-local storage
        let mut jit_ctx = JitContext {
            rt: &mut jit_rt as *mut LuaRuntime,
            child_jit_ptrs,
            child_info: child_info_owned.clone(),
            child_consts: child_consts_owned.clone(),
        };
        JIT_CTX.with(|c| c.set(&mut jit_ctx as *mut JitContext));

        // Create JIT call handler — dispatches closures to JIT code
        let jit_call_handler: CallHandler = {
            let rt_rc = Rc::new(RefCell::new(()));  // dummy for lifetime
            Rc::new(move |closure_val: u64, args: &[u64]| -> u64 {
                let _ = &rt_rc; // prevent drop
                let ctx_ptr = JIT_CTX.with(|c| c.get());
                let ctx = unsafe { &*ctx_ptr };
                let rt = unsafe { &mut *ctx.rt };

                let func_id = closure_func_id(closure_val).unwrap();
                let code_ptr = ctx.child_jit_ptrs[func_id];

                // Swap constants
                let saved = std::mem::replace(&mut rt.constants, ctx.child_consts[func_id].clone());

                // Build init_regs
                let (_, max_stack) = ctx.child_info[func_id];
                let total = max_stack as usize + 1;
                let mut init_regs = vec![make_nil(); total];
                init_regs[0] = closure_val;
                for (i, &arg) in args.iter().enumerate() {
                    if i + 1 < total { init_regs[i + 1] = arg; }
                }

                // Run JIT-compiled child function
                let result = unsafe { call_jit(code_ptr, &init_regs) };

                // Restore constants
                let rt = unsafe { &mut *ctx.rt };
                rt.constants = saved;

                result
            })
        };

        // Re-run main chunk with the JIT call handler for closures
        let jit_result;
        {
            let jit_rt_rc = Rc::new(RefCell::new(()));
            // We need interpreter externs that use the JIT context's runtime
            let func = &main_tf.function;
            let mut interp = Interpreter::<NanBox>::new(func);

            // Bind externs using raw pointer to jit_rt
            let rt_ptr = &mut jit_rt as *mut LuaRuntime;
            bind_jit_main_externs(&mut interp, &main_tf.extern_names, rt_ptr, &jit_call_handler);
            let _ = jit_rt_rc;

            let total = chunk.main.max_stack_size as usize + 1;
            let init_regs = vec![make_nil(); total];
            jit_result = match interp.run(&init_regs).unwrap() {
                InterpResult::Value(v) => v,
                _ => make_nil(),
            };
        }

        // Clear thread-local
        JIT_CTX.with(|c| c.set(std::ptr::null_mut()));

        // Compare results
        assert_eq!(result, jit_result,
            "JIT result ({:#018x}) != interpreter result ({:#018x})",
            jit_result, result);

        // Also keep the child_jits alive until after comparison
        drop(child_jits);
    }

    drop(children);
    drop(info);
    drop(consts);

    let rt_inner = Rc::try_unwrap(rt).ok().unwrap().into_inner();
    (result, rt_inner)
}

/// Bind externs for the main chunk interpreter in JIT mode.
/// Uses a raw pointer to LuaRuntime (from JIT context) instead of Rc<RefCell>.
fn bind_jit_main_externs(
    interp: &mut Interpreter<NanBox>,
    extern_names: &[String],
    rt_ptr: *mut LuaRuntime,
    handler: &CallHandler,
) {
    for name in extern_names {
        match name.as_str() {
            "lua_add" => { let p = rt_ptr; interp.bind_by_name("lua_add", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_add(args[0], args[1]))) }); }
            "lua_sub" => { let p = rt_ptr; interp.bind_by_name("lua_sub", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_sub(args[0], args[1]))) }); }
            "lua_mul" => { let p = rt_ptr; interp.bind_by_name("lua_mul", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_mul(args[0], args[1]))) }); }
            "lua_div" => { let p = rt_ptr; interp.bind_by_name("lua_div", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_div(args[0], args[1]))) }); }
            "lua_mod" => { let p = rt_ptr; interp.bind_by_name("lua_mod", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_mod(args[0], args[1]))) }); }
            "lua_pow" => { let p = rt_ptr; interp.bind_by_name("lua_pow", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_pow(args[0], args[1]))) }); }
            "lua_unm" => { let p = rt_ptr; interp.bind_by_name("lua_unm", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_unm(args[0]))) }); }
            "lua_not" => { let p = rt_ptr; interp.bind_by_name("lua_not", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_not(args[0]))) }); }
            "lua_len" => { let p = rt_ptr; interp.bind_by_name("lua_len", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_len(args[0]))) }); }
            "lua_eq" => { let p = rt_ptr; interp.bind_by_name("lua_eq", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_eq(args[0], args[1]))) }); }
            "lua_lt" => { let p = rt_ptr; interp.bind_by_name("lua_lt", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_lt(args[0], args[1]))) }); }
            "lua_le" => { let p = rt_ptr; interp.bind_by_name("lua_le", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_le(args[0], args[1]))) }); }
            "lua_concat" => { let p = rt_ptr; interp.bind_by_name("lua_concat", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_concat(args[0], args[1]))) }); }
            "lua_getglobal" => { let p = rt_ptr; interp.bind_by_name("lua_getglobal", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_getglobal(args[0]))) }); }
            "lua_setglobal" => { let p = rt_ptr; interp.bind_by_name("lua_setglobal", move |args| { let rt = unsafe { &mut *p }; rt.lua_setglobal(args[0], args[1]); ExternCallResult::Value(None) }); }
            "lua_newtable" => { let p = rt_ptr; interp.bind_by_name("lua_newtable", move |_args| { let rt = unsafe { &mut *p }; ExternCallResult::Value(Some(rt.lua_newtable())) }); }
            "lua_gettable" => { let p = rt_ptr; interp.bind_by_name("lua_gettable", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_gettable(args[0], args[1]))) }); }
            "lua_settable" => { let p = rt_ptr; interp.bind_by_name("lua_settable", move |args| { let rt = unsafe { &*p }; rt.lua_settable(args[0], args[1], args[2]); ExternCallResult::Value(None) }); }
            "lua_call" => {
                let p = rt_ptr;
                let handler = handler.clone();
                interp.bind_by_name("lua_call", move |args| {
                    let func = args[0];
                    let base = args[1] as usize;
                    let nargs = args[2] as usize;
                    if is_closure(func) {
                        let rt = unsafe { &*p };
                        let call_args: Vec<u64> = (0..nargs).map(|i| rt.register_file[base + i]).collect();
                        ExternCallResult::Value(Some(handler(func, &call_args)))
                    } else {
                        let rt = unsafe { &mut *p };
                        ExternCallResult::Value(Some(rt.lua_call(func, base, nargs)))
                    }
                });
            }
            "lua_setlist" => { let p = rt_ptr; interp.bind_by_name("lua_setlist", move |args| { let rt = unsafe { &*p }; rt.lua_setlist_from_regfile(args[0], args[1] as usize, args[2] as usize, args[3] as usize); ExternCallResult::Value(None) }); }
            "lua_forprep" => { let p = rt_ptr; interp.bind_by_name("lua_forprep", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_forprep(args[0], args[1], args[2]))) }); }
            "lua_forloop" => { let p = rt_ptr; interp.bind_by_name("lua_forloop", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_forloop(args[0], args[1], args[2]))) }); }
            "lua_is_nil" => { let p = rt_ptr; interp.bind_by_name("lua_is_nil", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_is_nil(args[0]))) }); }
            "lua_self" => { let p = rt_ptr; interp.bind_by_name("lua_self", move |args| { let rt = unsafe { &*p }; ExternCallResult::Value(Some(rt.lua_gettable(args[0], args[1]))) }); }
            "lua_store_reg" => { let p = rt_ptr; interp.bind_by_name("lua_store_reg", move |args| { let rt = unsafe { &mut *p }; let idx = args[0] as usize; if idx >= rt.register_file.len() { rt.register_file.resize(idx + 1, make_nil()); } rt.register_file[idx] = args[1]; ExternCallResult::Value(None) }); }
            "lua_make_closure" => { interp.bind_by_name("lua_make_closure", move |args| { ExternCallResult::Value(Some(runtime::make_closure(args[0] as usize, &[]))) }); }
            "lua_make_closure_1" => { interp.bind_by_name("lua_make_closure_1", move |args| { ExternCallResult::Value(Some(runtime::make_closure(args[0] as usize, &[args[2]]))) }); }
            "lua_make_closure_2" => { interp.bind_by_name("lua_make_closure_2", move |args| { ExternCallResult::Value(Some(runtime::make_closure(args[0] as usize, &[args[2], args[3]]))) }); }
            "lua_make_closure_3" => { interp.bind_by_name("lua_make_closure_3", move |args| { ExternCallResult::Value(Some(runtime::make_closure(args[0] as usize, &[args[2], args[3], args[4]]))) }); }
            "lua_make_closure_4" => { interp.bind_by_name("lua_make_closure_4", move |args| { ExternCallResult::Value(Some(runtime::make_closure(args[0] as usize, &[args[2], args[3], args[4], args[5]]))) }); }
            other => panic!("unknown extern function: {}", other),
        }
    }
}

/// Extract f64 from a NanBox result.
fn as_number(v: u64) -> f64 {
    match NanBox::decode(v) {
        Decoded::Float(f) => f,
        Decoded::Tagged { tag, payload } => {
            panic!("expected number, got tag={} payload={:#x}", tag, payload);
        }
    }
}

fn is_nil(v: u64) -> bool {
    NanBox::has_tag(v, 1)
}

// ── Phase 1: Bytecode parsing ──────────────────────────────

#[test]
fn test_parse_simple() {
    let data = compile_lua("return 42");
    let chunk = bytecode::parse(&data).unwrap();
    assert_eq!(chunk.main.code.len(), 3);
    assert!(matches!(chunk.main.constants[0], bytecode::Constant::Number(n) if n == 42.0));
}

#[test]
fn test_parse_constants() {
    let data = compile_lua("local a = 'hello'; local b = true; local c = nil; return 3.14");
    let chunk = bytecode::parse(&data).unwrap();
    let consts = &chunk.main.constants;
    assert!(matches!(&consts[0], bytecode::Constant::String(s) if s == "hello"));
    assert!(matches!(&consts[1], bytecode::Constant::Number(n) if *n == 3.14));
}

// ── Phase 1: Simple arithmetic ─────────────────────────────

#[test]
fn test_return_constant() {
    let (result, _) = run_lua("return 42");
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_add() {
    let (result, _) = run_lua("local a = 5; local b = 3; return a + b");
    assert_eq!(as_number(result), 8.0);
}

#[test]
fn test_sub() {
    let (result, _) = run_lua("return 10 - 3");
    assert_eq!(as_number(result), 7.0);
}

#[test]
fn test_mul() {
    let (result, _) = run_lua("return 6 * 7");
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_div() {
    let (result, _) = run_lua("return 10 / 4");
    assert_eq!(as_number(result), 2.5);
}

#[test]
fn test_mod() {
    let (result, _) = run_lua("return 10 % 3");
    assert_eq!(as_number(result), 1.0);
}

#[test]
fn test_pow() {
    let (result, _) = run_lua("return 2 ^ 10");
    assert_eq!(as_number(result), 1024.0);
}

#[test]
fn test_unm() {
    let (result, _) = run_lua("local a = 5; return -a");
    assert_eq!(as_number(result), -5.0);
}

#[test]
fn test_complex_arithmetic() {
    let (result, _) = run_lua("return (2 + 3) * 4 - 1");
    assert_eq!(as_number(result), 19.0);
}

// ── Phase 2: Control flow ──────────────────────────────────

#[test]
fn test_if_true() {
    let (result, _) = run_lua("if 1 < 2 then return 10 else return 20 end");
    assert_eq!(as_number(result), 10.0);
}

#[test]
fn test_if_false() {
    let (result, _) = run_lua("if 2 < 1 then return 10 else return 20 end");
    assert_eq!(as_number(result), 20.0);
}

#[test]
fn test_if_eq() {
    let (result, _) = run_lua("if 5 == 5 then return 1 else return 0 end");
    assert_eq!(as_number(result), 1.0);
}

#[test]
fn test_if_le() {
    let (result, _) = run_lua("if 3 <= 3 then return 1 else return 0 end");
    assert_eq!(as_number(result), 1.0);
}

#[test]
fn test_while_loop() {
    let (result, _) = run_lua(r#"
        local i = 0
        local sum = 0
        while i < 10 do
            i = i + 1
            sum = sum + i
        end
        return sum
    "#);
    assert_eq!(as_number(result), 55.0);
}

#[test]
fn test_numeric_for() {
    let (result, _) = run_lua(r#"
        local sum = 0
        for i = 1, 10 do
            sum = sum + i
        end
        return sum
    "#);
    assert_eq!(as_number(result), 55.0);
}

#[test]
fn test_numeric_for_step() {
    let (result, _) = run_lua(r#"
        local sum = 0
        for i = 0, 10, 2 do
            sum = sum + i
        end
        return sum
    "#);
    assert_eq!(as_number(result), 30.0);
}

#[test]
fn test_numeric_for_negative_step() {
    let (result, _) = run_lua(r#"
        local sum = 0
        for i = 10, 1, -1 do
            sum = sum + i
        end
        return sum
    "#);
    assert_eq!(as_number(result), 55.0);
}

#[test]
fn test_factorial() {
    let (result, _) = run_lua(r#"
        local n = 10
        local f = 1
        for i = 2, n do
            f = f * i
        end
        return f
    "#);
    assert_eq!(as_number(result), 3628800.0);
}

#[test]
fn test_fibonacci_iterative() {
    let (result, _) = run_lua(r#"
        local n = 20
        local a, b = 0, 1
        for i = 1, n do
            a, b = b, a + b
        end
        return a
    "#);
    assert_eq!(as_number(result), 6765.0);
}

// ── Phase 3: Globals + function calls ──────────────────────

#[test]
fn test_global_variable() {
    let (result, _) = run_lua(r#"
        x = 42
        return x
    "#);
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_not_operator() {
    let (result, _) = run_lua("return not false");
    assert!(NanBox::has_tag(result, 2));
    assert_eq!(NanBox::extract_payload(result), 1);
}

#[test]
fn test_nil_return() {
    let (result, _) = run_lua("return nil");
    assert!(is_nil(result));
}

#[test]
fn test_bool_return() {
    let (result, _) = run_lua("return true");
    assert!(NanBox::has_tag(result, 2));
    assert_eq!(NanBox::extract_payload(result), 1);
}

// ── Phase 3: Tables ────────────────────────────────────────

#[test]
fn test_table_basic() {
    let (result, _) = run_lua(r#"
        local t = {}
        t[1] = 42
        return t[1]
    "#);
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_table_multiple_keys() {
    let (result, _) = run_lua(r#"
        local t = {}
        t[1] = 10
        t[2] = 20
        t[3] = 30
        return t[1] + t[2] + t[3]
    "#);
    assert_eq!(as_number(result), 60.0);
}

#[test]
fn test_nested_if_else() {
    let (result, _) = run_lua(r#"
        local x = 15
        if x > 20 then
            return 3
        elseif x > 10 then
            return 2
        else
            return 1
        end
    "#);
    assert_eq!(as_number(result), 2.0);
}

#[test]
fn test_and_or_short_circuit() {
    let (result, _) = run_lua(r#"
        local a = 5
        local b = a > 0 and a or 0
        return b
    "#);
    assert_eq!(as_number(result), 5.0);
}

#[test]
fn test_repeat_until() {
    let (result, _) = run_lua(r#"
        local i = 0
        repeat
            i = i + 1
        until i >= 10
        return i
    "#);
    assert_eq!(as_number(result), 10.0);
}

#[test]
fn test_euclid_gcd() {
    let (result, _) = run_lua(r#"
        local a, b = 48, 18
        while b ~= 0 do
            a, b = b, a % b
        end
        return a
    "#);
    assert_eq!(as_number(result), 6.0);
}

#[test]
fn test_sieve_of_eratosthenes() {
    let (result, _) = run_lua(r#"
        local n = 100
        local is_prime = {}
        for i = 2, n do
            is_prime[i] = true
        end
        for i = 2, n do
            if is_prime[i] then
                local j = i * i
                while j <= n do
                    is_prime[j] = false
                    j = j + i
                end
            end
        end
        local count = 0
        for i = 2, n do
            if is_prime[i] then
                count = count + 1
            end
        end
        return count
    "#);
    assert_eq!(as_number(result), 25.0);
}

// ── Phase 3: Print and function calls ──────────────────────

#[test]
fn test_print_number() {
    let (_, rt) = run_lua(r#"
        print(42)
        return nil
    "#);
    assert_eq!(rt.output.trim(), "42");
}

#[test]
fn test_print_multiple() {
    let (_, rt) = run_lua(r#"
        print(1, 2, 3)
        return nil
    "#);
    assert_eq!(rt.output.trim(), "1\t2\t3");
}

#[test]
fn test_print_expression() {
    let (_, rt) = run_lua(r#"
        print(2 + 3)
        return nil
    "#);
    assert_eq!(rt.output.trim(), "5");
}

#[test]
fn test_print_in_loop() {
    let (_, rt) = run_lua(r#"
        for i = 1, 5 do
            print(i)
        end
        return nil
    "#);
    let lines: Vec<&str> = rt.output.trim().lines().collect();
    assert_eq!(lines, vec!["1", "2", "3", "4", "5"]);
}

#[test]
fn test_global_function_assign() {
    let (result, _) = run_lua(r#"
        x = 10
        y = 20
        return x + y
    "#);
    assert_eq!(as_number(result), 30.0);
}

#[test]
fn test_math_sqrt() {
    let (result, _) = run_lua(r#"
        return math.sqrt(144)
    "#);
    assert_eq!(as_number(result), 12.0);
}

#[test]
fn test_math_abs() {
    let (result, _) = run_lua(r#"
        return math.abs(-42)
    "#);
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_math_floor() {
    let (result, _) = run_lua(r#"
        return math.floor(3.7)
    "#);
    assert_eq!(as_number(result), 3.0);
}

#[test]
fn test_assert_true() {
    let (_, _) = run_lua(r#"
        assert(1 == 1)
        return nil
    "#);
}

#[test]
fn test_nested_table() {
    let (result, _) = run_lua(r#"
        local t = {}
        t[1] = {}
        t[1][1] = 99
        return t[1][1]
    "#);
    assert_eq!(as_number(result), 99.0);
}

#[test]
fn test_table_as_array() {
    let (result, _) = run_lua(r#"
        local t = {}
        for i = 1, 10 do
            t[i] = i * i
        end
        local sum = 0
        for i = 1, 10 do
            sum = sum + t[i]
        end
        return sum
    "#);
    assert_eq!(as_number(result), 385.0);
}

#[test]
fn test_bubble_sort() {
    let (result, _) = run_lua(r#"
        local t = {}
        t[1] = 5; t[2] = 3; t[3] = 8; t[4] = 1; t[5] = 9
        t[6] = 2; t[7] = 7; t[8] = 4; t[9] = 6; t[10] = 10
        local n = 10
        for i = 1, n - 1 do
            for j = 1, n - i do
                if t[j] > t[j + 1] then
                    local tmp = t[j]
                    t[j] = t[j + 1]
                    t[j + 1] = tmp
                end
            end
        end
        return t[1] + t[10]
    "#);
    assert_eq!(as_number(result), 11.0);
}

#[test]
fn test_collatz() {
    let (result, _) = run_lua(r#"
        local n = 27
        local steps = 0
        while n ~= 1 do
            if n % 2 == 0 then
                n = n / 2
            else
                n = 3 * n + 1
            end
            steps = steps + 1
        end
        return steps
    "#);
    assert_eq!(as_number(result), 111.0);
}

#[test]
fn test_matrix_multiply_2x2() {
    let (result, _) = run_lua(r#"
        local a11, a12, a21, a22 = 1, 2, 3, 4
        local b11, b12, b21, b22 = 5, 6, 7, 8
        local c11 = a11*b11 + a12*b21
        local c12 = a11*b12 + a12*b22
        local c21 = a21*b11 + a22*b21
        local c22 = a21*b12 + a22*b22
        return c11 + c12 + c21 + c22
    "#);
    assert_eq!(as_number(result), 134.0);
}

// ── Phase 6: Closures + recursive functions ────────────────

#[test]
fn test_recursive_factorial() {
    let (result, _) = run_lua(r#"
        local function fact(n)
            if n <= 1 then return 1 end
            return n * fact(n - 1)
        end
        return fact(10)
    "#);
    assert_eq!(as_number(result), 3628800.0);
}

#[test]
fn test_recursive_fibonacci() {
    let (result, _) = run_lua(r#"
        local function fib(n)
            if n <= 1 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        return fib(15)
    "#);
    assert_eq!(as_number(result), 610.0);
}

#[test]
fn test_binary_trees() {
    let (result, _) = run_lua(r#"
        local function bottomUpTree(depth)
            if depth > 0 then
                local left = bottomUpTree(depth - 1)
                local right = bottomUpTree(depth - 1)
                local t = {}
                t[1] = left
                t[2] = right
                return t
            else
                local t = {}
                return t
            end
        end

        local function itemCheck(node)
            if node[1] == nil then
                return 1
            end
            return 1 + itemCheck(node[1]) + itemCheck(node[2])
        end

        return itemCheck(bottomUpTree(10))
    "#);
    assert_eq!(as_number(result), 2047.0);
}

#[test]
fn test_binary_trees_benchmark() {
    let (_, rt) = run_lua(r#"
        local function bottomUpTree(depth)
            if depth > 0 then
                local left = bottomUpTree(depth - 1)
                local right = bottomUpTree(depth - 1)
                local t = {}
                t[1] = left
                t[2] = right
                return t
            else
                local t = {}
                return t
            end
        end

        local function itemCheck(node)
            if node[1] == nil then
                return 1
            end
            return 1 + itemCheck(node[1]) + itemCheck(node[2])
        end

        local maxDepth = 10
        if maxDepth < 6 then maxDepth = 6 end
        local stretchDepth = maxDepth + 1
        local check = itemCheck(bottomUpTree(stretchDepth))
        print("stretch tree of depth " .. stretchDepth .. "\t check: " .. check)

        local longLivedTree = bottomUpTree(maxDepth)

        for depth = 4, maxDepth, 2 do
            local iterations = 2 ^ (maxDepth - depth + 4)
            local chk = 0
            for i = 1, iterations do
                chk = chk + itemCheck(bottomUpTree(depth))
            end
            print(iterations .. "\t trees of depth " .. depth .. "\t check: " .. chk)
        end

        print("long lived tree of depth " .. maxDepth .. "\t check: " .. itemCheck(longLivedTree))
        return nil
    "#);
    let output = rt.output.trim();
    let lines: Vec<&str> = output.lines().collect();
    assert!(lines[0].contains("stretch tree of depth 11"));
    assert!(lines[0].contains("check: 4095"));
    assert!(lines.last().unwrap().contains("long lived tree of depth 10"));
    assert!(lines.last().unwrap().contains("check: 2047"));
}

#[test]
fn test_concat() {
    let (_, rt) = run_lua(r#"
        print("hello" .. " " .. "world")
        return nil
    "#);
    assert_eq!(rt.output.trim(), "hello world");
}

#[test]
fn test_number_concat() {
    let (_, rt) = run_lua(r#"
        print("value: " .. 42)
        return nil
    "#);
    assert_eq!(rt.output.trim(), "value: 42");
}
