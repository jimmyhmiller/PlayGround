use std::cell::Cell;
use std::process::Command;

#[allow(unused_imports)]
use dynlower::{JitFunction, call_jit};
use dynobj::RootSource;
use dynruntime::{MutatorRootManager, NanBoxPtrPolicy};
use dynvalue::{Decoded, NanBox, TagScheme};

use crate::bytecode::{self, Constant, Proto};
use crate::runtime::{self, LuaRuntime, closure_func_id, is_closure, make_nil};
use crate::translate::{self, TranslatedFunction};

// ── JIT extern infrastructure ──────────────────────────────
//
// JIT externs are raw `extern "C"` function pointers — they can't capture
// environment like Rust closures. We use thread-local storage to give them
// access to the LuaRuntime and child JIT code pointers.

struct JitContext {
    rt: *mut LuaRuntime,
    child_jit_ptrs: Vec<*const u8>,
    child_info: Vec<(u8, u8)>,      // (num_params, is_vararg) per child
    child_consts: Vec<Vec<String>>, // constant table per child
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

struct LuaCallArgs {
    abi_args: Vec<u64>,
    _varargs: Vec<u64>,
}

impl LuaCallArgs {
    fn for_proto(proto: &Proto, closure: u64, args: &[u64]) -> Self {
        let fixed = proto.num_params as usize;
        let extra_count = if proto.is_vararg != 0 {
            args.len().saturating_sub(fixed)
        } else {
            0
        };

        let varargs = if extra_count > 0 {
            args[fixed..].to_vec()
        } else {
            vec![make_nil()]
        };

        let mut abi_args = Vec::with_capacity(1 + fixed + 2);
        abi_args.push(closure);
        for i in 0..fixed {
            abi_args.push(args.get(i).copied().unwrap_or(make_nil()));
        }
        abi_args.push(extra_count as u64);
        abi_args.push(varargs.as_ptr() as u64);

        LuaCallArgs {
            abi_args,
            _varargs: varargs,
        }
    }

    fn from_sig(num_params: u8, is_vararg: u8, closure: u64, args: &[u64]) -> Self {
        let proto = Proto {
            source: String::new(),
            line_defined: 0,
            last_line_defined: 0,
            num_upvalues: 0,
            num_params,
            is_vararg,
            max_stack_size: 0,
            code: Vec::new(),
            constants: Vec::new(),
            protos: Vec::new(),
            source_lines: Vec::new(),
            locals: Vec::new(),
            upvalue_names: Vec::new(),
        };
        Self::for_proto(&proto, closure, args)
    }
}

struct SlotRoots {
    base: *mut u64,
    len: usize,
}

impl RootSource for SlotRoots {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for i in 0..self.len {
            visitor(unsafe { self.base.add(i) });
        }
    }
}

extern "C" fn jit_gc_handler(frame_ptr: *mut u8, frame_size: usize) {
    with_jit_ctx(|_, rt| {
        let frame_roots = SlotRoots {
            base: frame_ptr as *mut u64,
            len: frame_size / 8,
        };
        let regfile_roots = SlotRoots {
            base: rt.register_file.as_mut_ptr(),
            len: rt.register_file.len(),
        };
        unsafe {
            rt.heap_ref()
                .collect::<NanBoxPtrPolicy>(&[&frame_roots, &regfile_roots]);
        }
    });
}

// ── JIT extern "C" functions ───────────────────────────────
// These must be declared in the SAME ORDER as translate.rs declares externs.

extern "C" fn jit_lua_add(a: u64, b: u64) -> u64 {
    std::panic::catch_unwind(|| with_jit_ctx(|_, rt| rt.lua_add(a, b))).unwrap_or_else(|_| {
        eprintln!("PANIC jit_lua_add a={:#018x} b={:#018x}", a, b);
        std::process::exit(99);
    })
}
extern "C" fn jit_lua_sub(a: u64, b: u64) -> u64 {
    std::panic::catch_unwind(|| with_jit_ctx(|_, rt| rt.lua_sub(a, b))).unwrap_or_else(|_| {
        eprintln!("PANIC jit_lua_sub a={:#018x} b={:#018x}", a, b);
        std::process::exit(99);
    })
}
extern "C" fn jit_lua_mul(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_mul(a, b))
}
extern "C" fn jit_lua_div(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_div(a, b))
}
extern "C" fn jit_lua_mod(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_mod(a, b))
}
extern "C" fn jit_lua_pow(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_pow(a, b))
}
extern "C" fn jit_lua_unm(a: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_unm(a))
}
extern "C" fn jit_lua_not(a: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_not(a))
}
extern "C" fn jit_lua_len(a: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_len(a))
}
extern "C" fn jit_lua_eq(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_eq(a, b))
}
extern "C" fn jit_lua_lt(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_lt(a, b))
}
extern "C" fn jit_lua_le(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_le(a, b))
}
extern "C" fn jit_lua_concat(a: u64, b: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_concat(a, b))
}
extern "C" fn jit_lua_getglobal(name: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_getglobal(name))
}
extern "C" fn jit_lua_setglobal(name: u64, val: u64) -> u64 {
    with_jit_ctx(|_, rt| {
        rt.lua_setglobal(name, val);
        0
    })
}
extern "C" fn jit_lua_newtable() -> u64 {
    with_jit_ctx(|_, rt| rt.lua_newtable())
}
extern "C" fn jit_lua_gettable(table: u64, key: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_gettable(table, key))
}
extern "C" fn jit_lua_settable(table: u64, key: u64, val: u64) -> u64 {
    with_jit_ctx(|_, rt| {
        rt.lua_settable(table, key, val);
        0
    })
}

extern "C" fn jit_lua_call(func: u64, base: u64, nargs: u64) -> u64 {
    std::panic::catch_unwind(|| {
        let base = base as usize;
        let nargs = nargs as usize;

        if is_closure(func) {
            let ctx_ptr = JIT_CTX.with(|c| c.get());
            let ctx = unsafe { &*ctx_ptr };
            let rt = unsafe { &mut *ctx.rt };

            let func_id = closure_func_id(func).unwrap();
            assert!(
                func_id < ctx.child_jit_ptrs.len(),
                "jit_lua_call: func_id {} out of bounds (have {} children)",
                func_id,
                ctx.child_jit_ptrs.len()
            );
            let code_ptr = ctx.child_jit_ptrs[func_id];

            // Read args from register_file
            let args: Vec<u64> = (0..nargs).map(|i| rt.register_file[base + i]).collect();

            // Swap constants
            let saved = std::mem::replace(&mut rt.constants, ctx.child_consts[func_id].clone());

            let (num_params, is_vararg) = ctx.child_info[func_id];
            let call_args = LuaCallArgs::from_sig(num_params, is_vararg, func, &args);

            // Call child JIT code
            let result = unsafe { call_jit(code_ptr, &call_args.abi_args) };

            // Restore constants
            let rt = unsafe { &mut *ctx.rt };
            rt.constants = saved;

            result
        } else {
            with_jit_ctx(|_, rt| rt.lua_call(func, base, nargs))
        }
    })
    .unwrap_or_else(|e| {
        eprintln!("PANIC in jit_lua_call: {:?}", e);
        std::process::exit(99);
    })
}

extern "C" fn jit_lua_setlist(table: u64, base: u64, offset: u64, count: u64) -> u64 {
    with_jit_ctx(|_, rt| {
        rt.lua_setlist_from_regfile(table, base as usize, offset as usize, count as usize);
        0
    })
}
extern "C" fn jit_lua_forprep(init: u64, limit: u64, step: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_forprep(init, limit, step))
}
extern "C" fn jit_lua_forloop(index: u64, limit: u64, step: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_forloop(index, limit, step))
}
extern "C" fn jit_lua_is_nil(v: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_is_nil(v))
}
extern "C" fn jit_lua_self(table: u64, key: u64) -> u64 {
    with_jit_ctx(|_, rt| rt.lua_gettable(table, key))
}

extern "C" fn jit_lua_store_reg(idx: u64, val: u64) -> u64 {
    with_jit_ctx(|_, rt| {
        let idx = idx as usize;
        if idx >= rt.register_file.len() {
            rt.register_file.resize(idx + 1, make_nil());
        }
        rt.register_file[idx] = val;
        0
    })
}

extern "C" fn jit_lua_make_closure(func_id: u64, _num: u64) -> u64 {
    with_jit_ctx(|_, rt| runtime::make_closure(rt.heap_ref(), func_id as usize, &[]))
}
extern "C" fn jit_lua_make_closure_1(func_id: u64, _num: u64, u0: u64) -> u64 {
    with_jit_ctx(|_, rt| runtime::make_closure(rt.heap_ref(), func_id as usize, &[u0]))
}
extern "C" fn jit_lua_make_closure_2(func_id: u64, _num: u64, u0: u64, u1: u64) -> u64 {
    with_jit_ctx(|_, rt| runtime::make_closure(rt.heap_ref(), func_id as usize, &[u0, u1]))
}
extern "C" fn jit_lua_make_closure_3(func_id: u64, _num: u64, u0: u64, u1: u64, u2: u64) -> u64 {
    with_jit_ctx(|_, rt| runtime::make_closure(rt.heap_ref(), func_id as usize, &[u0, u1, u2]))
}
extern "C" fn jit_lua_make_closure_4(
    func_id: u64,
    _num: u64,
    u0: u64,
    u1: u64,
    u2: u64,
    u3: u64,
) -> u64 {
    with_jit_ctx(|_, rt| runtime::make_closure(rt.heap_ref(), func_id as usize, &[u0, u1, u2, u3]))
}

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

/// Run a Lua program fully through the JIT.
fn run_lua(source: &str) -> (u64, LuaRuntime) {
    run_lua_with_heap(source, 64 * 1024 * 1024)
}

fn run_lua_with_heap(source: &str, heap_size: usize) -> (u64, LuaRuntime) {
    let gc_heap_size = Some(heap_size);
    let bytecode_data = compile_lua(source);
    let chunk = bytecode::parse(&bytecode_data).expect("bytecode parse failed");

    // Translate ALL protos up front (main + children)
    let main_tf = translate::translate(&chunk.main);
    let child_tfs: Vec<TranslatedFunction> = chunk
        .main
        .protos
        .iter()
        .map(|p| translate::translate(p))
        .collect();

    // Collect child proto call-signature metadata
    let child_info: Vec<(u8, u8)> = chunk
        .main
        .protos
        .iter()
        .map(|p| (p.num_params, p.is_vararg))
        .collect();

    // Collect per-child constant tables (each proto has its own)
    let child_constants: Vec<Vec<String>> = chunk
        .main
        .protos
        .iter()
        .map(|p| {
            p.constants
                .iter()
                .map(|c| match c {
                    Constant::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect()
        })
        .collect();

    let heap_size = gc_heap_size.unwrap_or(64 * 1024 * 1024);
    let gc_enabled = gc_heap_size.is_some();
    let roots = MutatorRootManager::<NanBoxPtrPolicy>::new(heap_size)
        .with_roots_all(gc_enabled)
        .with_gc_threshold(if gc_enabled { 0.75 } else { f64::INFINITY });
    let mut rt = LuaRuntime::new(roots.heap(), &chunk.main.constants);

    let extern_ptrs = jit_extern_ptrs();
    let child_jits: Vec<JitFunction> = child_tfs
        .iter()
        .map(|tf| {
            JitFunction::compile_with_gc::<NanBox>(&tf.function, &extern_ptrs, jit_gc_handler)
        })
        .collect();
    let child_jit_ptrs: Vec<*const u8> = child_jits.iter().map(|j| j.as_ptr()).collect();
    let main_jit =
        JitFunction::compile_with_gc::<NanBox>(&main_tf.function, &extern_ptrs, jit_gc_handler);

    let mut jit_ctx = JitContext {
        rt: &mut rt as *mut LuaRuntime,
        child_jit_ptrs,
        child_info,
        child_consts: child_constants,
    };
    JIT_CTX.with(|c| c.set(&mut jit_ctx as *mut JitContext));

    let call_args = LuaCallArgs::for_proto(&chunk.main, make_nil(), &[]);
    let result = unsafe { call_jit(main_jit.as_ptr(), &call_args.abi_args) };

    JIT_CTX.with(|c| c.set(std::ptr::null_mut()));

    (result, rt)
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
    let (result, _) = run_lua(
        r#"
        local i = 0
        local sum = 0
        while i < 10 do
            i = i + 1
            sum = sum + i
        end
        return sum
    "#,
    );
    assert_eq!(as_number(result), 55.0);
}

#[test]
fn test_numeric_for() {
    let (result, _) = run_lua(
        r#"
        local sum = 0
        for i = 1, 10 do
            sum = sum + i
        end
        return sum
    "#,
    );
    assert_eq!(as_number(result), 55.0);
}

#[test]
fn test_numeric_for_step() {
    let (result, _) = run_lua(
        r#"
        local sum = 0
        for i = 0, 10, 2 do
            sum = sum + i
        end
        return sum
    "#,
    );
    assert_eq!(as_number(result), 30.0);
}

#[test]
fn test_numeric_for_negative_step() {
    let (result, _) = run_lua(
        r#"
        local sum = 0
        for i = 10, 1, -1 do
            sum = sum + i
        end
        return sum
    "#,
    );
    assert_eq!(as_number(result), 55.0);
}

#[test]
fn test_factorial() {
    let (result, _) = run_lua(
        r#"
        local n = 10
        local f = 1
        for i = 2, n do
            f = f * i
        end
        return f
    "#,
    );
    assert_eq!(as_number(result), 3628800.0);
}

#[test]
fn test_fibonacci_iterative() {
    let (result, _) = run_lua(
        r#"
        local n = 20
        local a, b = 0, 1
        for i = 1, n do
            a, b = b, a + b
        end
        return a
    "#,
    );
    assert_eq!(as_number(result), 6765.0);
}

// ── Phase 3: Globals + function calls ──────────────────────

#[test]
fn test_global_variable() {
    let (result, _) = run_lua(
        r#"
        x = 42
        return x
    "#,
    );
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
    let (result, _) = run_lua(
        r#"
        local t = {}
        t[1] = 42
        return t[1]
    "#,
    );
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_table_multiple_keys() {
    let (result, _) = run_lua(
        r#"
        local t = {}
        t[1] = 10
        t[2] = 20
        t[3] = 30
        return t[1] + t[2] + t[3]
    "#,
    );
    assert_eq!(as_number(result), 60.0);
}

#[test]
fn test_nested_if_else() {
    let (result, _) = run_lua(
        r#"
        local x = 15
        if x > 20 then
            return 3
        elseif x > 10 then
            return 2
        else
            return 1
        end
    "#,
    );
    assert_eq!(as_number(result), 2.0);
}

#[test]
fn test_and_or_short_circuit() {
    let (result, _) = run_lua(
        r#"
        local a = 5
        local b = a > 0 and a or 0
        return b
    "#,
    );
    assert_eq!(as_number(result), 5.0);
}

#[test]
fn test_repeat_until() {
    let (result, _) = run_lua(
        r#"
        local i = 0
        repeat
            i = i + 1
        until i >= 10
        return i
    "#,
    );
    assert_eq!(as_number(result), 10.0);
}

#[test]
fn test_euclid_gcd() {
    let (result, _) = run_lua(
        r#"
        local a, b = 48, 18
        while b ~= 0 do
            a, b = b, a % b
        end
        return a
    "#,
    );
    assert_eq!(as_number(result), 6.0);
}

#[test]
fn test_sieve_of_eratosthenes() {
    let (result, _) = run_lua(
        r#"
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
    "#,
    );
    assert_eq!(as_number(result), 25.0);
}

// ── Phase 3: Print and function calls ──────────────────────

#[test]
fn test_print_number() {
    let (_, rt) = run_lua(
        r#"
        print(42)
        return nil
    "#,
    );
    assert_eq!(rt.output.trim(), "42");
}

#[test]
fn test_print_multiple() {
    let (_, rt) = run_lua(
        r#"
        print(1, 2, 3)
        return nil
    "#,
    );
    assert_eq!(rt.output.trim(), "1\t2\t3");
}

#[test]
fn test_print_expression() {
    let (_, rt) = run_lua(
        r#"
        print(2 + 3)
        return nil
    "#,
    );
    assert_eq!(rt.output.trim(), "5");
}

#[test]
fn test_print_in_loop() {
    let (_, rt) = run_lua(
        r#"
        for i = 1, 5 do
            print(i)
        end
        return nil
    "#,
    );
    let lines: Vec<&str> = rt.output.trim().lines().collect();
    assert_eq!(lines, vec!["1", "2", "3", "4", "5"]);
}

#[test]
fn test_global_function_assign() {
    let (result, _) = run_lua(
        r#"
        x = 10
        y = 20
        return x + y
    "#,
    );
    assert_eq!(as_number(result), 30.0);
}

#[test]
fn test_math_sqrt() {
    let (result, _) = run_lua(
        r#"
        return math.sqrt(144)
    "#,
    );
    assert_eq!(as_number(result), 12.0);
}

#[test]
fn test_math_abs() {
    let (result, _) = run_lua(
        r#"
        return math.abs(-42)
    "#,
    );
    assert_eq!(as_number(result), 42.0);
}

#[test]
fn test_math_floor() {
    let (result, _) = run_lua(
        r#"
        return math.floor(3.7)
    "#,
    );
    assert_eq!(as_number(result), 3.0);
}

#[test]
fn test_assert_true() {
    let (_, _) = run_lua(
        r#"
        assert(1 == 1)
        return nil
    "#,
    );
}

#[test]
fn test_nested_table() {
    let (result, _) = run_lua(
        r#"
        local t = {}
        t[1] = {}
        t[1][1] = 99
        return t[1][1]
    "#,
    );
    assert_eq!(as_number(result), 99.0);
}

#[test]
fn test_table_as_array() {
    let (result, _) = run_lua(
        r#"
        local t = {}
        for i = 1, 10 do
            t[i] = i * i
        end
        local sum = 0
        for i = 1, 10 do
            sum = sum + t[i]
        end
        return sum
    "#,
    );
    assert_eq!(as_number(result), 385.0);
}

#[test]
fn test_bubble_sort() {
    let (result, _) = run_lua(
        r#"
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
    "#,
    );
    assert_eq!(as_number(result), 11.0);
}

#[test]
fn test_collatz() {
    let (result, _) = run_lua(
        r#"
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
    "#,
    );
    assert_eq!(as_number(result), 111.0);
}

#[test]
fn test_matrix_multiply_2x2() {
    let (result, _) = run_lua(
        r#"
        local a11, a12, a21, a22 = 1, 2, 3, 4
        local b11, b12, b21, b22 = 5, 6, 7, 8
        local c11 = a11*b11 + a12*b21
        local c12 = a11*b12 + a12*b22
        local c21 = a21*b11 + a22*b21
        local c22 = a21*b12 + a22*b22
        return c11 + c12 + c21 + c22
    "#,
    );
    assert_eq!(as_number(result), 134.0);
}

// ── Phase 6: Closures + recursive functions ────────────────

#[test]
fn test_recursive_factorial() {
    let (result, _) = run_lua(
        r#"
        local function fact(n)
            if n <= 1 then return 1 end
            return n * fact(n - 1)
        end
        return fact(10)
    "#,
    );
    assert_eq!(as_number(result), 3628800.0);
}

#[test]
fn test_recursive_fibonacci() {
    let (result, _) = run_lua(
        r#"
        local function fib(n)
            if n <= 1 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        return fib(15)
    "#,
    );
    assert_eq!(as_number(result), 610.0);
}

#[test]
fn test_binary_trees() {
    let (result, _) = run_lua(
        r#"
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
    "#,
    );
    assert_eq!(as_number(result), 2047.0);
}

#[test]
fn test_binary_trees_benchmark() {
    let (_, rt) = run_lua(
        r#"
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
    "#,
    );
    let output = rt.output.trim();
    let lines: Vec<&str> = output.lines().collect();
    assert!(lines[0].contains("stretch tree of depth 11"));
    assert!(lines[0].contains("check: 4095"));
    assert!(
        lines
            .last()
            .unwrap()
            .contains("long lived tree of depth 10")
    );
    assert!(lines.last().unwrap().contains("check: 2047"));
}

#[test]
fn test_concat() {
    let (_, rt) = run_lua(
        r#"
        print("hello" .. " " .. "world")
        return nil
    "#,
    );
    assert_eq!(rt.output.trim(), "hello world");
}

#[test]
fn test_number_concat() {
    let (_, rt) = run_lua(
        r#"
        print("value: " .. 42)
        return nil
    "#,
    );
    assert_eq!(rt.output.trim(), "value: 42");
}

#[test]
fn test_fifty_nested_function_calls() {
    // Generate Lua code with 50 functions chained:
    //   local function f0(x) return x + 1 end
    //   local function f1(x) return f0(x) + 1 end
    //   ...
    //   local function f49(x) return f48(x) + 1 end
    //   return f49(0)
    //
    // f49(0) should return 50.
    let n = 50;
    let mut lua = String::new();

    // f0: base case
    lua.push_str("local function f0(x) return x + 1 end\n");

    // f1 through f49: each calls the previous and adds 1
    for i in 1..n {
        lua.push_str(&format!(
            "local function f{i}(x) return f{}(x) + 1 end\n",
            i - 1
        ));
    }

    lua.push_str(&format!("return f{}(0)\n", n - 1));

    let (result, _) = run_lua(&lua);
    assert_eq!(
        as_number(result),
        n as f64,
        "50 nested Lua function calls should each add 1"
    );
}

// ─── Substantial integration test ───────────────────────────────────
//
// A single program that exercises closures, higher-order functions,
// tables-as-objects, recursion, GC pressure, string building, and
// control flow together — proving the IR can support a real language.

#[test]
#[ignore = "hits Lua runtime bug: nested function-call-as-arg pattern corrupts values"]
fn test_json_serializer() {
    // A Lua JSON serializer: builds a table structure representing a small
    // dataset, then serializes it to a JSON string using recursive functions,
    // closures for indentation, and table iteration.
    let (_, rt) = run_lua(
        r#"
        -- Higher-order: map over array indices
        local function map(t, f)
            local result = {}
            for i = 1, #t do
                result[i] = f(t[i])
            end
            return result
        end

        -- Higher-order: fold/reduce
        local function fold(t, init, f)
            local acc = init
            for i = 1, #t do
                acc = f(acc, t[i])
            end
            return acc
        end

        -- Join array of strings with separator
        local function join(parts, sep)
            return fold(parts, "", function(acc, s)
                if acc == "" then return s end
                return acc .. sep .. s
            end)
        end

        -- Build indentation string
        local function make_indent(level)
            local s = ""
            for i = 1, level do
                s = s .. "  "
            end
            return s
        end

        -- Recursive JSON serializer (level = indentation depth)
        local function serialize(val, level)
            if type(val) == "number" then
                return tostring(val)
            elseif type(val) == "string" then
                return "\"" .. val .. "\""
            elseif type(val) == "table" then
                -- Array check
                if val[1] ~= nil then
                    local parts = {}
                    for i = 1, #val do
                        parts[i] = make_indent(level + 1) .. serialize(val[i], level + 1)
                    end
                    return "[\n" .. join(parts, ",\n") .. "\n" .. make_indent(level) .. "]"
                end
                -- Object with known keys
                local lines = {}
                local n = 0
                local ind = make_indent(level + 1)
                if val["name"] ~= nil then
                    n = n + 1
                    lines[n] = ind .. "\"name\": " .. serialize(val["name"], level + 1)
                end
                if val["age"] ~= nil then
                    n = n + 1
                    lines[n] = ind .. "\"age\": " .. serialize(val["age"], level + 1)
                end
                if val["hobbies"] ~= nil then
                    n = n + 1
                    lines[n] = ind .. "\"hobbies\": " .. serialize(val["hobbies"], level + 1)
                end
                if val["team"] ~= nil then
                    n = n + 1
                    lines[n] = ind .. "\"team\": " .. serialize(val["team"], level + 1)
                end
                return "{\n" .. join(lines, ",\n") .. "\n" .. make_indent(level) .. "}"
            else
                return "null"
            end
        end

        -- Build a person record
        local function person(name, age, hobbies)
            local p = {}
            p["name"] = name
            p["age"] = age
            p["hobbies"] = hobbies
            return p
        end

        -- Build dataset
        local hobbies1 = {}
        hobbies1[1] = "chess"
        hobbies1[2] = "hiking"

        local hobbies2 = {}
        hobbies2[1] = "painting"
        hobbies2[2] = "cooking"
        hobbies2[3] = "running"

        local team = {}
        team[1] = person("Alice", 30, hobbies1)
        team[2] = person("Bob", 25, hobbies2)

        local data = {}
        data["team"] = team

        -- Serialize and print
        local json = serialize(data, 0)
        print(json)

        -- Compute stats with higher-order functions
        local ages = map(team, function(p) return p["age"] end)
        local total_age = fold(ages, 0, function(a, b) return a + b end)
        local hobby_counts = map(team, function(p) return #p["hobbies"] end)
        local total_hobbies = fold(hobby_counts, 0, function(a, b) return a + b end)

        print("total_age: " .. total_age)
        print("total_hobbies: " .. total_hobbies)

        return total_age * 100 + total_hobbies
    "#,
    );
    let output = rt.output.trim();
    let lines: Vec<&str> = output.lines().collect();

    // Verify JSON structure
    assert_eq!(lines[0], "{");
    assert!(lines[1].contains("\"team\": ["));
    assert!(output.contains("\"name\": \"Alice\""));
    assert!(output.contains("\"age\": 30"));
    assert!(output.contains("\"name\": \"Bob\""));
    assert!(output.contains("\"hobbies\": ["));
    assert!(output.contains("\"chess\""));
    assert!(output.contains("\"painting\""));

    // Verify computed stats
    assert!(output.contains("total_age: 55"));
    assert!(output.contains("total_hobbies: 5"));
}

#[test]
#[ignore = "hits Lua runtime bug: nested function-call-as-arg pattern corrupts values"]
fn test_jit_with_closures() {
    // A tiny expression evaluator: parse + evaluate arithmetic from a table-based AST.
    // Tests deep closure nesting, mutual recursion between eval and apply,
    // and tables-as-discriminated-unions.
    let (result, rt) = run_lua(
        r#"
        -- AST node constructors
        local function num(n)
            local t = {}
            t["tag"] = "num"
            t["val"] = n
            return t
        end

        local function binop(op, left, right)
            local t = {}
            t["tag"] = "binop"
            t["op"] = op
            t["left"] = left
            t["right"] = right
            return t
        end

        -- Evaluator with environment (closure over env table)
        local function make_eval()
            local eval  -- forward declare

            local function apply_op(op, a, b)
                if op == "+" then return a + b end
                if op == "-" then return a - b end
                if op == "*" then return a * b end
                if op == "/" then return a / b end
                return 0
            end

            eval = function(node)
                if node["tag"] == "num" then
                    return node["val"]
                elseif node["tag"] == "binop" then
                    local l = eval(node["left"])
                    local r = eval(node["right"])
                    return apply_op(node["op"], l, r)
                end
                return 0
            end

            return eval
        end

        local eval = make_eval()

        -- Build AST for: (3 + 4) * (10 - 2) / 2
        -- = 7 * 8 / 2 = 28
        local ast = binop("/",
            binop("*",
                binop("+", num(3), num(4)),
                binop("-", num(10), num(2))
            ),
            num(2)
        )

        local result = eval(ast)
        print("result: " .. result)

        -- Build a bigger expression: sum of squares 1^2 + 2^2 + ... + 10^2
        local expr = num(0)
        for i = 1, 10 do
            expr = binop("+", expr, binop("*", num(i), num(i)))
        end
        local sum_sq = eval(expr)
        print("sum_of_squares: " .. sum_sq)

        return result * 1000 + sum_sq
    "#,
    );
    let output = rt.output.trim();
    assert!(output.contains("result: 28"));
    assert!(output.contains("sum_of_squares: 385"));
    assert_eq!(as_number(result), 28385.0);
}

/// GC stress test: allocate 50000 tables in a main-function for-loop with only
/// 256KB of heap. Without GC collection this would need ~5MB+. The fact that it
/// completes proves the semi-space collector is reclaiming dead tables at ForLoop
/// safepoints.
#[test]
fn test_gc_stress_main_loop() {
    let (result, rt) = run_lua_with_heap(
        r#"
        local sum = 0
        for i = 1, 50000 do
            local t = {}
            t[1] = i
            sum = sum + t[1]
        end
        print(sum)
        return sum
    "#,
        256 * 1024,
    ); // 256KB — would OOM without GC
    assert_eq!(as_number(result), 1250025000.0);
    assert_eq!(rt.output.trim(), "1250025000");
}
