use std::cell::Cell;
use std::collections::HashMap;
use std::process::Command;

#[allow(unused_imports)]
use dynlower::{JitFunction, SafepointHandlerPayloadKind, SafepointRecord, call_jit};
use dynruntime::{
    JitRootTransportRuntime, JitSafepointSession, MutatorRootManager, NanBoxPtrPolicy,
    active_jit_safepoint_handler,
};
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
    /// child_offsets[global_id] = global index where that proto's children start.
    child_offsets: Vec<usize>,
    /// Current function's child offset — `local_bx + current_child_offset = global_id`.
    current_child_offset: usize,
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

/// JIT root transport for Lua: scans the current JIT frame (conservative,
/// all words) plus the Lua register_file which holds values across calls.
struct LuaJitTransport {
    register_file: *const Vec<u64>,
}

// Safety: only used single-threaded; pointer valid during JIT execution.
unsafe impl Send for LuaJitTransport {}
unsafe impl Sync for LuaJitTransport {}

impl JitRootTransportRuntime for LuaJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::FrameSize
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        _safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        // Scan all words in the current JIT frame (same as FrameScanJitTransport)
        let frame_words = payload / 8;
        let base = frame_ptr as *mut u64;
        for i in 0..frame_words {
            visitor(unsafe { base.add(i) });
        }

        // Also scan the Lua register_file (holds values passed to externs)
        let regfile = unsafe { &*self.register_file };
        for i in 0..regfile.len() {
            visitor(unsafe { (regfile.as_ptr() as *mut u64).add(i) });
        }
    }
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
    std::panic::catch_unwind(|| with_jit_ctx(|_, rt| rt.lua_mul(a, b))).unwrap_or_else(|_| {
        eprintln!("PANIC jit_lua_mul a={:#018x} b={:#018x}", a, b);
        std::process::exit(99);
    })
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
            let ctx = unsafe { &mut *ctx_ptr };
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

            // Set child_offset for the callee's scope (no constant swap needed —
            // string constants use global indices).
            let saved_offset = ctx.current_child_offset;
            ctx.current_child_offset = ctx.child_offsets[func_id];

            let (num_params, is_vararg) = ctx.child_info[func_id];
            let call_args = LuaCallArgs::from_sig(num_params, is_vararg, func, &args);

            // Call child JIT code
            let result = unsafe { call_jit(code_ptr, &call_args.abi_args) };

            // Restore
            let ctx = unsafe { &mut *ctx_ptr };
            ctx.current_child_offset = saved_offset;

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

/// Remap a local proto index (bx from CLOSURE instruction) to a global func_id.
fn remap_func_id(ctx: &JitContext, local_bx: usize) -> usize {
    ctx.current_child_offset + local_bx
}

extern "C" fn jit_lua_make_closure(func_id: u64, _num: u64) -> u64 {
    let ctx_ptr = JIT_CTX.with(|c| c.get());
    let ctx = unsafe { &*ctx_ptr };
    let rt = unsafe { &mut *ctx.rt };
    let global_id = remap_func_id(ctx, func_id as usize);
    runtime::make_closure(rt.heap_ref(), global_id, &[])
}
extern "C" fn jit_lua_make_closure_1(func_id: u64, _num: u64, u0: u64) -> u64 {
    let ctx_ptr = JIT_CTX.with(|c| c.get());
    let ctx = unsafe { &*ctx_ptr };
    let rt = unsafe { &mut *ctx.rt };
    let global_id = remap_func_id(ctx, func_id as usize);
    runtime::make_closure(rt.heap_ref(), global_id, &[u0])
}
extern "C" fn jit_lua_make_closure_2(func_id: u64, _num: u64, u0: u64, u1: u64) -> u64 {
    let ctx_ptr = JIT_CTX.with(|c| c.get());
    let ctx = unsafe { &*ctx_ptr };
    let rt = unsafe { &mut *ctx.rt };
    let global_id = remap_func_id(ctx, func_id as usize);
    runtime::make_closure(rt.heap_ref(), global_id, &[u0, u1])
}
extern "C" fn jit_lua_make_closure_3(func_id: u64, _num: u64, u0: u64, u1: u64, u2: u64) -> u64 {
    let ctx_ptr = JIT_CTX.with(|c| c.get());
    let ctx = unsafe { &*ctx_ptr };
    let rt = unsafe { &mut *ctx.rt };
    let global_id = remap_func_id(ctx, func_id as usize);
    runtime::make_closure(rt.heap_ref(), global_id, &[u0, u1, u2])
}
extern "C" fn jit_lua_make_closure_4(
    func_id: u64,
    _num: u64,
    u0: u64,
    u1: u64,
    u2: u64,
    u3: u64,
) -> u64 {
    let ctx_ptr = JIT_CTX.with(|c| c.get());
    let ctx = unsafe { &*ctx_ptr };
    let rt = unsafe { &mut *ctx.rt };
    let global_id = remap_func_id(ctx, func_id as usize);
    runtime::make_closure(rt.heap_ref(), global_id, &[u0, u1, u2, u3])
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

/// Recursively collect all protos depth-first into a flat list.
/// Returns (flat_protos, child_offsets) where child_offsets[global_id] is
/// the global index where that proto's children start.
fn flatten_all_protos(root: &Proto) -> (Vec<Proto>, Vec<usize>) {
    let mut out = Vec::new();
    let mut child_offsets = Vec::new();
    flatten_children(root, &mut out, &mut child_offsets);
    (out, child_offsets)
}

fn flatten_children(parent: &Proto, out: &mut Vec<Proto>, offsets: &mut Vec<usize>) {
    let first_child_global = out.len();

    for child in &parent.protos {
        out.push(child.clone());
        offsets.push(0); // placeholder
    }

    let num_children = parent.protos.len();
    for i in 0..num_children {
        let global_idx = first_child_global + i;
        let grandchild_start = out.len();
        offsets[global_idx] = grandchild_start;
        let child_proto = out[global_idx].clone();
        if !child_proto.protos.is_empty() {
            flatten_children(&child_proto, out, offsets);
        }
    }
}

/// Build a mapping from local constant index → global string index for a proto.
fn build_string_remap(
    proto: &Proto,
    global_strings: &mut Vec<String>,
    string_index_map: &mut HashMap<String, usize>,
) -> Vec<usize> {
    proto.constants.iter().map(|c| {
        match c {
            Constant::String(s) => {
                *string_index_map.entry(s.clone()).or_insert_with(|| {
                    let idx = global_strings.len();
                    global_strings.push(s.clone());
                    idx
                })
            }
            _ => 0,
        }
    }).collect()
}

fn run_lua(source: &str) -> (u64, LuaRuntime) {
    run_lua_with_opts(source, 64 * 1024 * 1024, &dynir::opt::OptConfig::all())
}

fn run_lua_with_heap(source: &str, heap_size: usize) -> (u64, LuaRuntime) {
    run_lua_with_opts(source, heap_size, &dynir::opt::OptConfig::all())
}

fn run_lua_with_opts(source: &str, heap_size: usize, opt: &dynir::opt::OptConfig) -> (u64, LuaRuntime) {
    let gc_heap_size = Some(heap_size);
    let bytecode_data = compile_lua(source);
    let mut chunk = bytecode::parse(&bytecode_data).expect("bytecode parse failed");

    // Flatten ALL protos recursively with global indices.
    let (all_protos, child_offsets) = flatten_all_protos(&chunk.main);

    let mut global_strings: Vec<String> = Vec::new();
    let mut string_index_map: HashMap<String, usize> = HashMap::new();

    let main_string_remap = build_string_remap(
        &chunk.main, &mut global_strings, &mut string_index_map,
    );
    let child_string_remaps: Vec<Vec<usize>> = all_protos
        .iter()
        .map(|p| build_string_remap(p, &mut global_strings, &mut string_index_map))
        .collect();

    let mut main_tf = translate::translate(&chunk.main, Some(&main_string_remap));
    dynir::opt::optimize_with(&mut main_tf.function, opt);
    let mut child_tfs: Vec<TranslatedFunction> = all_protos
        .iter()
        .enumerate()
        .map(|(i, p)| translate::translate(p, Some(&child_string_remaps[i])))
        .collect();
    for tf in &mut child_tfs {
        dynir::opt::optimize_with(&mut tf.function, opt);
    }

    let child_info: Vec<(u8, u8)> = all_protos
        .iter()
        .map(|p| (p.num_params, p.is_vararg))
        .collect();

    let heap_size = gc_heap_size.unwrap_or(64 * 1024 * 1024);
    let gc_enabled = gc_heap_size.is_some();
    let roots = MutatorRootManager::<NanBoxPtrPolicy>::new(heap_size)
        .with_gc_threshold(if gc_enabled { 0.75 } else { f64::INFINITY });
    let mut rt = LuaRuntime::new(roots.heap(), &chunk.main.constants);
    rt.constants = global_strings.clone();

    let extern_ptrs = jit_extern_ptrs();
    let child_jits: Vec<JitFunction> = child_tfs
        .iter()
        .map(|tf| {
            JitFunction::compile_with_gc::<NanBox>(
                &tf.function, &extern_ptrs, active_jit_safepoint_handler,
            )
        })
        .collect();
    let child_jit_ptrs: Vec<*const u8> = child_jits.iter().map(|j| j.as_ptr()).collect();
    let main_jit = JitFunction::compile_with_gc::<NanBox>(
        &main_tf.function, &extern_ptrs, active_jit_safepoint_handler,
    );

    let mut jit_ctx = JitContext {
        rt: &mut rt as *mut LuaRuntime,
        child_jit_ptrs,
        child_info,
        child_offsets: child_offsets.clone(),
        current_child_offset: 0, // main's children start at global index 0
    };
    JIT_CTX.with(|c| c.set(&mut jit_ctx as *mut JitContext));

    let transport = LuaJitTransport {
        register_file: &rt.register_file as *const Vec<u64>,
    };
    let threshold = if gc_enabled { 0.75 } else { f64::INFINITY };
    let session = JitSafepointSession::<NanBoxPtrPolicy, LuaJitTransport>::new(
        roots.heap(), transport, main_jit.safepoints(),
    ).with_gc_threshold(threshold);

    let call_args = LuaCallArgs::for_proto(&chunk.main, make_nil(), &[]);
    let result = session.with_installed(|| {
        unsafe { call_jit(main_jit.as_ptr(), &call_args.abi_args) }
    });

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

// ─── Table-heavy benchmark ─────────────────────────────────
//
// Exercises: integer-keyed tables, string-keyed tables, table creation,
// lookup, iteration, filtering, and GC pressure from many small tables.
// Designed to avoid the nested-call-as-arg bug.

const TABLE_BENCH_SRC: &str = r#"
local N = 50000

local function build_and_sum(n)
    local t = {}
    for i = 1, n do
        t[i] = i * i
    end
    local sum = 0
    for i = 1, n do
        sum = sum + t[i]
    end
    return sum
end

local function filter_odds(n)
    local t = {}
    for i = 1, n do
        t[i] = i * i
    end
    local odds = {}
    local odd_count = 0
    for i = 1, n do
        local v = t[i]
        local half = v / 2
        local floored = half - half % 1
        if floored * 2 ~= v then
            odd_count = odd_count + 1
            odds[odd_count] = v
        end
    end
    local odd_sum = 0
    for i = 1, odd_count do
        odd_sum = odd_sum + odds[i]
    end
    return odd_count + odd_sum
end

local function chained_tables(n)
    local chain_sum = 0
    for i = 1, n / 10 do
        local small = {}
        local base = (i - 1) * 10
        for j = 1, 10 do
            small[j] = base + j
        end
        for j = 1, 10 do
            chain_sum = chain_sum + small[j]
        end
    end
    return chain_sum
end

local sum = build_and_sum(N)
local odd_result = filter_odds(N)
local chain_sum = chained_tables(N)
return sum + odd_result + chain_sum
"#;

#[test]
fn test_table_bench_multi_closure() {
    // Regression: multiple closures in the same chunk used to get corrupted
    // by the GC because the safepoint handler only scanned the innermost
    // JIT frame, not parent frames holding other closure pointers.
    let (result, _) = run_lua(r#"
        local function f1(n)
            local t = {}
            for i = 1, n do t[i] = i end
            local sum = 0
            for i = 1, n do sum = sum + t[i] end
            return sum
        end
        local function f2(n)
            local t = {}
            for i = 1, n do t[i] = i * i end
            local sum = 0
            for i = 1, n do sum = sum + t[i] end
            return sum
        end
        local a = f1(10)
        local b = f2(10)
        return a + b
    "#);
    // 55 + 385 = 440
    assert_eq!(as_number(result), 440.0);
}

/// Proves multi-frame GC root scanning works: f1 allocates 5000 tables in a
/// loop on a 32KB heap. The GC MUST collect many times during f1. Meanwhile,
/// the parent frame holds the closure for f2. If the GC doesn't scan the
/// parent frame (via FP chain walking), f2's closure pointer goes stale
/// after the semi-space copy and the second call crashes or returns garbage.
///
/// We also assert that GC actually collected (>0 collections), proving
/// this isn't just passing because the heap was big enough to avoid GC.
#[test]
fn test_gc_multi_closure_stress() {
    let heap_size = 32 * 1024; // 32KB — each semi-space is 16KB

    let bytecode_data = compile_lua(r#"
        local function f1(n)
            local sum = 0
            for i = 1, n do
                local t = {}
                t[1] = i
                sum = sum + t[1]
            end
            return sum
        end
        local function f2(n)
            local t = {}
            for i = 1, n do t[i] = i * i end
            local sum = 0
            for i = 1, n do sum = sum + t[i] end
            return sum
        end
        local a = f1(5000)
        local b = f2(10)
        return a + b
    "#);
    let chunk = bytecode::parse(&bytecode_data).expect("bytecode parse failed");
    let (all_protos, child_offsets) = flatten_all_protos(&chunk.main);

    let mut global_strings: Vec<String> = Vec::new();
    let mut string_index_map: HashMap<String, usize> = HashMap::new();
    let main_string_remap = build_string_remap(
        &chunk.main, &mut global_strings, &mut string_index_map,
    );
    let child_string_remaps: Vec<Vec<usize>> = all_protos
        .iter()
        .map(|p| build_string_remap(p, &mut global_strings, &mut string_index_map))
        .collect();

    let mut main_tf = translate::translate(&chunk.main, Some(&main_string_remap));
    dynir::opt::optimize(&mut main_tf.function);
    let mut child_tfs: Vec<TranslatedFunction> = all_protos
        .iter()
        .enumerate()
        .map(|(i, p)| translate::translate(p, Some(&child_string_remaps[i])))
        .collect();
    for tf in &mut child_tfs {
        dynir::opt::optimize(&mut tf.function);
    }
    let child_info: Vec<(u8, u8)> = all_protos
        .iter()
        .map(|p| (p.num_params, p.is_vararg))
        .collect();

    let roots = MutatorRootManager::<NanBoxPtrPolicy>::new(heap_size)
        .with_gc_threshold(0.75);
    let mut rt = LuaRuntime::new(roots.heap(), &chunk.main.constants);
    rt.constants = global_strings.clone();

    let extern_ptrs = jit_extern_ptrs();
    let child_jits: Vec<JitFunction> = child_tfs
        .iter()
        .map(|tf| {
            JitFunction::compile_with_gc::<NanBox>(
                &tf.function, &extern_ptrs, active_jit_safepoint_handler,
            )
        })
        .collect();
    let child_jit_ptrs: Vec<*const u8> = child_jits.iter().map(|j| j.as_ptr()).collect();
    let main_jit = JitFunction::compile_with_gc::<NanBox>(
        &main_tf.function, &extern_ptrs, active_jit_safepoint_handler,
    );

    let mut jit_ctx = JitContext {
        rt: &mut rt as *mut LuaRuntime,
        child_jit_ptrs,
        child_info,
        child_offsets: child_offsets.clone(),
        current_child_offset: 0, // main's children start at global index 0
    };
    JIT_CTX.with(|c| c.set(&mut jit_ctx as *mut JitContext));

    let transport = LuaJitTransport {
        register_file: &rt.register_file as *const Vec<u64>,
    };
    let session = JitSafepointSession::<NanBoxPtrPolicy, LuaJitTransport>::new(
        roots.heap(), transport, main_jit.safepoints(),
    ).with_gc_threshold(0.75);

    let call_args = LuaCallArgs::for_proto(&chunk.main, make_nil(), &[]);
    let result = session.with_installed(|| {
        unsafe { call_jit(main_jit.as_ptr(), &call_args.abi_args) }
    });

    JIT_CTX.with(|c| c.set(std::ptr::null_mut()));

    // f1(5000) = sum(1..5000) = 12502500
    // f2(10) = sum(1^2..10^2) = 385
    assert_eq!(as_number(result), 12502885.0);

    // Prove the GC actually ran — 5000 table allocations on a 32KB heap
    // requires many collections.
    let gc_count = roots.collections();
    eprintln!("GC collections: {gc_count}");
    assert!(
        gc_count > 10,
        "expected many GC collections on 32KB heap with 5000 tables, got {gc_count}"
    );
}

#[test]
fn test_nested_call_as_arg() {
    // Simple: g(f(10))
    let (result, _) = run_lua(r#"
        local function f(x) return x end
        local function g(x) return x * 2 end
        local r = g(f(10))
        return r
    "#);
    assert_eq!(as_number(result), 20.0);

    // Two args from nested calls: h(f(3), g(4))
    let (result, _) = run_lua(r#"
        local function f(x) return x + 1 end
        local function g(x) return x * 2 end
        local function h(a, b) return a + b end
        return h(f(3), g(4))
    "#);
    // f(3)=4, g(4)=8, h(4,8)=12
    assert_eq!(as_number(result), 12.0);

    // Three levels deep: f(g(h(5)))
    let (result, _) = run_lua(r#"
        local function f(x) return x + 1 end
        local function g(x) return x * 2 end
        local function h(x) return x * 3 end
        return f(g(h(5)))
    "#);
    // h(5)=15, g(15)=30, f(30)=31
    assert_eq!(as_number(result), 31.0);

    // Nested calls as function args: binop("+", num(3), num(4))
    let (result, _) = run_lua(r#"
        local function num(n) return n end
        local function binop(op, a, b) return a + b end
        return binop("+", num(3), num(4))
    "#);
    assert_eq!(as_number(result), 7.0);

    // Higher-order: fold
    let (result, _) = run_lua(r#"
        local function fold(t, init, f)
            local acc = init
            for i = 1, #t do
                acc = f(acc, t[i])
            end
            return acc
        end
        local t = {}
        t[1] = 10
        t[2] = 20
        t[3] = 30
        return fold(t, 0, function(a, b) return a + b end)
    "#);
    assert_eq!(as_number(result), 60.0);

    // Forward-declared self-referencing closure
    let (result, _) = run_lua(r#"
        local function make_counter()
            local count
            count = function(n)
                if n <= 0 then return 0 end
                return 1 + count(n - 1)
            end
            return count
        end
        local f = make_counter()
        return f(5)
    "#);
    assert_eq!(as_number(result), 5.0);
}

#[test]
fn test_table_bench_correctness() {
    let (result, _rt) = run_lua(TABLE_BENCH_SRC);
    assert_eq!(as_number(result), 62502500050000.0);
}

#[test]
fn bench_table_heavy_jit() {
    // ── Compile once ──────────────────────────────────────────
    let bytecode_data = compile_lua(TABLE_BENCH_SRC);
    let chunk = bytecode::parse(&bytecode_data).expect("bytecode parse failed");
    let (all_protos, child_offsets) = flatten_all_protos(&chunk.main);

    let mut global_strings: Vec<String> = Vec::new();
    let mut string_index_map: HashMap<String, usize> = HashMap::new();
    let main_string_remap = build_string_remap(
        &chunk.main, &mut global_strings, &mut string_index_map,
    );
    let child_string_remaps: Vec<Vec<usize>> = all_protos
        .iter()
        .map(|p| build_string_remap(p, &mut global_strings, &mut string_index_map))
        .collect();

    let mut main_tf = translate::translate(&chunk.main, Some(&main_string_remap));
    dynir::opt::optimize(&mut main_tf.function);
    let mut child_tfs: Vec<TranslatedFunction> = all_protos
        .iter()
        .enumerate()
        .map(|(i, p)| translate::translate(p, Some(&child_string_remaps[i])))
        .collect();
    for tf in &mut child_tfs {
        dynir::opt::optimize(&mut tf.function);
    }
    let child_info: Vec<(u8, u8)> = all_protos
        .iter()
        .map(|p| (p.num_params, p.is_vararg))
        .collect();

    let extern_ptrs = jit_extern_ptrs();
    let child_jits: Vec<JitFunction> = child_tfs
        .iter()
        .map(|tf| {
            JitFunction::compile_with_gc::<NanBox>(
                &tf.function, &extern_ptrs, active_jit_safepoint_handler,
            )
        })
        .collect();
    let child_jit_ptrs: Vec<*const u8> = child_jits.iter().map(|j| j.as_ptr()).collect();
    let main_jit = JitFunction::compile_with_gc::<NanBox>(
        &main_tf.function, &extern_ptrs, active_jit_safepoint_handler,
    );

    // ── Execute many times (execution only) ──────────────────
    let iterations = 10;
    let mut times = Vec::new();
    for _ in 0..iterations {
        // Fresh runtime state each iteration
        let heap_size = 64 * 1024 * 1024;
        let roots = MutatorRootManager::<NanBoxPtrPolicy>::new(heap_size)
            .with_gc_threshold(0.75);
        let mut rt = LuaRuntime::new(roots.heap(), &chunk.main.constants);
        rt.constants = global_strings.clone();

        let mut jit_ctx = JitContext {
            rt: &mut rt as *mut LuaRuntime,
            child_jit_ptrs: child_jit_ptrs.clone(),
            child_info: child_info.clone(),
            child_offsets: child_offsets.clone(),
            current_child_offset: 0,
        };
        JIT_CTX.with(|c| c.set(&mut jit_ctx as *mut JitContext));

        let transport = LuaJitTransport {
            register_file: &rt.register_file as *const Vec<u64>,
        };
        let session = JitSafepointSession::<NanBoxPtrPolicy, LuaJitTransport>::new(
            roots.heap(), transport, main_jit.safepoints(),
        ).with_gc_threshold(0.75);

        let call_args = LuaCallArgs::for_proto(&chunk.main, make_nil(), &[]);

        let start = std::time::Instant::now();
        let result = session.with_installed(|| {
            unsafe { call_jit(main_jit.as_ptr(), &call_args.abi_args) }
        });
        let elapsed = start.elapsed();

        JIT_CTX.with(|c| c.set(std::ptr::null_mut()));
        assert_eq!(as_number(result), 62502500050000.0);
        times.push(elapsed);
    }

    let total: std::time::Duration = times.iter().sum();
    let avg = total / iterations as u32;
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();

    eprintln!("\n── JIT table benchmark ({} runs, exec only) ──", iterations);
    eprintln!("  avg: {:?}", avg);
    eprintln!("  min: {:?}", min);
    eprintln!("  max: {:?}", max);

    // Now run Lua 5.1 interpreter for comparison.
    let lua_src_path = std::env::temp_dir().join("table_bench_cmp.lua");
    let lua_src = TABLE_BENCH_SRC.replace(
        "return sum + odd_result + chain_sum",
        "print(sum + odd_result + chain_sum)",
    );
    std::fs::write(&lua_src_path, &lua_src).unwrap();

    let mut lua_times = Vec::new();
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let status = Command::new("/tmp/lua-5.1.5/src/lua")
            .arg(lua_src_path.to_str().unwrap())
            .stdout(std::process::Stdio::null())
            .status()
            .expect("failed to run lua 5.1");
        let elapsed = start.elapsed();
        assert!(status.success());
        lua_times.push(elapsed);
    }
    let _ = std::fs::remove_file(&lua_src_path);

    let lua_total: std::time::Duration = lua_times.iter().sum();
    let lua_avg = lua_total / iterations as u32;
    let lua_min = lua_times.iter().min().unwrap();
    let lua_max = lua_times.iter().max().unwrap();

    eprintln!("\n── Lua 5.1 interpreter ({} runs, includes parse) ──", iterations);
    eprintln!("  avg: {:?}", lua_avg);
    eprintln!("  min: {:?}", lua_min);
    eprintln!("  max: {:?}", lua_max);

    let ratio = lua_avg.as_secs_f64() / avg.as_secs_f64();
    eprintln!("\n── Ratio: JIT is {:.2}x vs Lua 5.1 ──", ratio);
}

// ─── Optimization pass combination tests ────────────────────────
//
// Run a non-trivial program under every interesting OptConfig
// combination so regressions in individual passes are caught.

/// Source that exercises loops, closures, tables, and arithmetic.
const OPT_TEST_SRC: &str = r#"
    local function build_and_sum(n)
        local t = {}
        for i = 1, n do
            t[i] = i * i
        end
        local sum = 0
        for i = 1, n do
            sum = sum + t[i]
        end
        return sum
    end
    return build_and_sum(200)
"#;

fn opt_test_expected() -> f64 {
    // sum(i^2, i=1..200) = 200*201*401/6 = 2686700
    2686700.0
}

#[test]
fn test_opt_none() {
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &dynir::opt::OptConfig::none());
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_all() {
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &dynir::opt::OptConfig::all());
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_mem2reg_only() {
    let cfg = dynir::opt::OptConfig { mem2reg: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_constfold_only() {
    let cfg = dynir::opt::OptConfig { constant_fold: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_gvn_only() {
    let cfg = dynir::opt::OptConfig { gvn: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_dce_only() {
    let cfg = dynir::opt::OptConfig { dce: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_licm_only() {
    // LICM alone (without mem2reg) should be a no-op but must not break anything.
    let cfg = dynir::opt::OptConfig { licm: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_mem2reg_gvn() {
    let cfg = dynir::opt::OptConfig { mem2reg: true, gvn: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_mem2reg_licm() {
    let cfg = dynir::opt::OptConfig { mem2reg: true, licm: true, ..dynir::opt::OptConfig::none() };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_mem2reg_constfold_gvn_dce() {
    let cfg = dynir::opt::OptConfig {
        mem2reg: true, constant_fold: true, gvn: true, dce: true, licm: false,
        ..dynir::opt::OptConfig::all()
    };
    let (r, _) = run_lua_with_opts(OPT_TEST_SRC, 64 * 1024 * 1024, &cfg);
    assert_eq!(as_number(r), opt_test_expected());
}

#[test]
fn test_opt_all_closures() {
    // Exercises closures under full optimization.
    let (r, _) = run_lua_with_opts(r#"
        local function make_adder(x)
            return function(y) return x + y end
        end
        local add5 = make_adder(5)
        local add10 = make_adder(10)
        return add5(3) + add10(7)
    "#, 64 * 1024 * 1024, &dynir::opt::OptConfig::all());
    assert_eq!(as_number(r), 25.0);
}

#[test]
fn test_opt_all_recursive() {
    // Exercises recursive calls under full optimization.
    let (r, _) = run_lua_with_opts(r#"
        local function fib(n)
            if n < 2 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        return fib(15)
    "#, 64 * 1024 * 1024, &dynir::opt::OptConfig::all());
    assert_eq!(as_number(r), 610.0);
}

#[test]
fn test_opt_none_closures() {
    // Same closure test with NO optimization.
    let (r, _) = run_lua_with_opts(r#"
        local function make_adder(x)
            return function(y) return x + y end
        end
        local add5 = make_adder(5)
        local add10 = make_adder(10)
        return add5(3) + add10(7)
    "#, 64 * 1024 * 1024, &dynir::opt::OptConfig::none());
    assert_eq!(as_number(r), 25.0);
}

#[test]
fn test_opt_none_recursive() {
    // Same recursive test with NO optimization.
    let (r, _) = run_lua_with_opts(r#"
        local function fib(n)
            if n < 2 then return n end
            return fib(n - 1) + fib(n - 2)
        end
        return fib(15)
    "#, 64 * 1024 * 1024, &dynir::opt::OptConfig::none());
    assert_eq!(as_number(r), 610.0);
}
