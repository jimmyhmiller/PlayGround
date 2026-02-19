//! C FFI bindings for embedding rules4 in native programs.
//!
//! All functions take an opaque `*mut Rules4Engine` pointer as the first argument.
//! Create one with `r4_engine_new()` and free it with `r4_engine_free()`.

use crate::engine::Engine;
use crate::term::{TermStore, TermId};
use std::slice;

/// Opaque engine handle for C callers.
pub type Rules4Engine = Engine;

// ── Engine lifecycle ──

#[unsafe(no_mangle)]
pub extern "C" fn r4_engine_new() -> *mut Rules4Engine {
    let store = TermStore::new();
    let engine = Engine::new(store, Vec::new(), Vec::new());
    Box::into_raw(Box::new(engine))
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_engine_free(engine: *mut Rules4Engine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)); }
    }
}

// ── Helpers ──

unsafe fn str_from_c(ptr: *const u8, len: usize) -> &'static str {
    let slice = unsafe { slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(slice).expect("Invalid UTF-8")
}

unsafe fn engine_ref<'a>(engine: *mut Rules4Engine) -> &'a mut Engine {
    unsafe { &mut *engine }
}

// ── Program loading ──

#[unsafe(no_mangle)]
pub extern "C" fn r4_load_program(engine: *mut Rules4Engine, src: *const u8, len: usize) -> u32 {
    let e = unsafe { engine_ref(engine) };
    let s = unsafe { str_from_c(src, len) };
    e.load_program(s).0
}

// ── Eval ──

#[unsafe(no_mangle)]
pub extern "C" fn r4_eval(engine: *mut Rules4Engine, term: u32) -> u32 {
    let e = unsafe { engine_ref(engine) };
    e.invalidate_cache();
    e.reset_eval_counters();
    e.eval(TermId(term)).0
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_eval_step_limit_exceeded(engine: *mut Rules4Engine) -> u32 {
    let e = unsafe { engine_ref(engine) };
    if e.step_limit_exceeded { 1 } else { 0 }
}

// ── Term construction ──

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_num(engine: *mut Rules4Engine, n: i64) -> u32 {
    let e = unsafe { engine_ref(engine) };
    e.make_num(n).0
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_float(engine: *mut Rules4Engine, f: f64) -> u32 {
    let e = unsafe { engine_ref(engine) };
    e.make_float(f).0
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_sym(engine: *mut Rules4Engine, name: *const u8, len: usize) -> u32 {
    let e = unsafe { engine_ref(engine) };
    let s = unsafe { str_from_c(name, len) };
    e.make_sym(s).0
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_call(engine: *mut Rules4Engine, head: u32, args: *const u32, args_len: usize) -> u32 {
    let e = unsafe { engine_ref(engine) };
    let raw = if args_len > 0 && !args.is_null() {
        unsafe { slice::from_raw_parts(args, args_len) }
    } else {
        &[]
    };
    let term_args: Vec<TermId> = raw.iter().map(|&id| TermId(id)).collect();
    e.make_call(TermId(head), &term_args).0
}

// ── Term inspection ──

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_tag(engine: *mut Rules4Engine, id: u32) -> u8 {
    let e = unsafe { engine_ref(engine) };
    e.term_tag(TermId(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_get_num(engine: *mut Rules4Engine, id: u32) -> i64 {
    let e = unsafe { engine_ref(engine) };
    e.term_num(TermId(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_get_float(engine: *mut Rules4Engine, id: u32) -> f64 {
    let e = unsafe { engine_ref(engine) };
    e.term_float(TermId(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_get_sym_name(engine: *mut Rules4Engine, id: u32, out_len: *mut usize) -> *const u8 {
    let e = unsafe { engine_ref(engine) };
    let name = e.term_sym_name(TermId(id));
    if !out_len.is_null() {
        unsafe { *out_len = name.len(); }
    }
    name.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_call_head(engine: *mut Rules4Engine, id: u32) -> u32 {
    let e = unsafe { engine_ref(engine) };
    e.term_call_head(TermId(id)).0
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_call_arity(engine: *mut Rules4Engine, id: u32) -> u32 {
    let e = unsafe { engine_ref(engine) };
    e.term_call_arity(TermId(id)) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_term_call_arg(engine: *mut Rules4Engine, id: u32, idx: u32) -> u32 {
    let e = unsafe { engine_ref(engine) };
    e.term_call_arg(TermId(id), idx as usize).0
}

// ── Display ──

/// Display a term as a string. Returns a pointer to a NUL-terminated string.
/// The returned pointer is valid until the next call to `r4_display_term`.
#[unsafe(no_mangle)]
pub extern "C" fn r4_display_term(engine: *mut Rules4Engine, id: u32, out_len: *mut usize) -> *const u8 {
    thread_local! {
        static BUF: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
    }
    let e = unsafe { engine_ref(engine) };
    let s = e.display(TermId(id));
    BUF.with(|b| {
        let mut buf = b.borrow_mut();
        *buf = s;
        buf.push('\0');
        if !out_len.is_null() {
            unsafe { *out_len = buf.len() - 1; } // exclude NUL
        }
        buf.as_ptr()
    })
}

// ── Generic scope access ──

#[unsafe(no_mangle)]
pub extern "C" fn r4_scope_pending_count(engine: *mut Rules4Engine, scope: *const u8, len: usize) -> u32 {
    let e = unsafe { engine_ref(engine) };
    let name = unsafe { str_from_c(scope, len) };
    e.scope_pending_count(name) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_scope_pending_get(engine: *mut Rules4Engine, scope: *const u8, len: usize, idx: u32) -> u32 {
    let e = unsafe { engine_ref(engine) };
    let name = unsafe { str_from_c(scope, len) };
    e.scope_pending_get(name, idx as usize).0
}

#[unsafe(no_mangle)]
pub extern "C" fn r4_scope_pending_clear(engine: *mut Rules4Engine, scope: *const u8, len: usize) {
    let e = unsafe { engine_ref(engine) };
    let name = unsafe { str_from_c(scope, len) };
    e.scope_pending_clear(name);
}
