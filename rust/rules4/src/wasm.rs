use crate::engine::Engine;
use crate::term::{TermStore, TermId};
use std::cell::RefCell;

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = RefCell::new(None);
}

fn with_engine<T>(f: impl FnOnce(&mut Engine) -> T) -> T {
    ENGINE.with(|e| {
        let mut borrow = e.borrow_mut();
        let engine = borrow.as_mut().expect("Engine not initialized. Call engine_new() first.");
        f(engine)
    })
}

// ── String passing ──

#[unsafe(no_mangle)]
pub extern "C" fn alloc_string(len: u32) -> *mut u8 {
    let mut buf = Vec::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string(ptr: *mut u8, len: u32) {
    unsafe {
        drop(Vec::from_raw_parts(ptr, len as usize, len as usize));
    }
}

// For returning strings to JS — we store the last display result
thread_local! {
    static LAST_STRING: RefCell<String> = RefCell::new(String::new());
}

// ── Engine lifecycle ──

#[unsafe(no_mangle)]
pub extern "C" fn engine_new() {
    ENGINE.with(|e| {
        let store = TermStore::new();
        let engine = Engine::new(store, Vec::new(), Vec::new());
        *e.borrow_mut() = Some(engine);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn engine_reset() {
    engine_new();
}

// ── Program loading ──

#[unsafe(no_mangle)]
pub extern "C" fn load_program(ptr: *const u8, len: u32) -> u32 {
    let src = unsafe {
        let slice = std::slice::from_raw_parts(ptr, len as usize);
        std::str::from_utf8(slice).expect("Invalid UTF-8")
    };
    with_engine(|engine| {
        let term_id = engine.load_program(src);
        term_id.0
    })
}

// ── Eval ──

#[unsafe(no_mangle)]
pub extern "C" fn eval(term_id: u32) -> u32 {
    with_engine(|engine| {
        // Clear cache at each external entry point so side-effectful terms
        // (rule(), retract()) are re-executed even if the term structure is identical.
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let result = engine.eval(TermId(term_id));
        result.0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn eval_step_limit_exceeded() -> u32 {
    with_engine(|engine| if engine.step_limit_exceeded { 1 } else { 0 })
}

// ── Term construction ──

#[unsafe(no_mangle)]
pub extern "C" fn term_num(n: i64) -> u32 {
    with_engine(|engine| engine.make_num(n).0)
}

// For i32 args from JS (JS numbers are f64, but wasm i64 is tricky)
#[unsafe(no_mangle)]
pub extern "C" fn term_num_i32(n: i32) -> u32 {
    with_engine(|engine| engine.make_num(n as i64).0)
}

#[unsafe(no_mangle)]
pub extern "C" fn term_sym(ptr: *const u8, len: u32) -> u32 {
    let name = unsafe {
        let slice = std::slice::from_raw_parts(ptr, len as usize);
        std::str::from_utf8(slice).expect("Invalid UTF-8")
    };
    with_engine(|engine| engine.make_sym(name).0)
}

#[unsafe(no_mangle)]
pub extern "C" fn term_call(head: u32, args_ptr: *const u32, args_len: u32) -> u32 {
    let args_raw = unsafe {
        std::slice::from_raw_parts(args_ptr, args_len as usize)
    };
    let args: Vec<TermId> = args_raw.iter().map(|&id| TermId(id)).collect();
    with_engine(|engine| engine.make_call(TermId(head), &args).0)
}

// ── Float construction ──

#[unsafe(no_mangle)]
pub extern "C" fn term_float(f: f64) -> u32 {
    with_engine(|engine| engine.make_float(f).0)
}

// ── Term inspection ──

#[unsafe(no_mangle)]
pub extern "C" fn term_tag(id: u32) -> u8 {
    with_engine(|engine| engine.term_tag(TermId(id)))
}

#[unsafe(no_mangle)]
pub extern "C" fn term_get_num(id: u32) -> i64 {
    with_engine(|engine| engine.term_num(TermId(id)))
}

#[unsafe(no_mangle)]
pub extern "C" fn term_get_num_i32(id: u32) -> i32 {
    with_engine(|engine| engine.term_num(TermId(id)) as i32)
}

#[unsafe(no_mangle)]
pub extern "C" fn term_get_float(id: u32) -> f64 {
    with_engine(|engine| engine.term_float(TermId(id)))
}

#[unsafe(no_mangle)]
pub extern "C" fn term_get_sym_ptr(id: u32) -> *const u8 {
    with_engine(|engine| {
        let name = engine.term_sym_name(TermId(id));
        name.as_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn term_get_sym_len(id: u32) -> u32 {
    with_engine(|engine| {
        let name = engine.term_sym_name(TermId(id));
        name.len() as u32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn term_call_head(id: u32) -> u32 {
    with_engine(|engine| engine.term_call_head(TermId(id)).0)
}

#[unsafe(no_mangle)]
pub extern "C" fn term_call_arity(id: u32) -> u32 {
    with_engine(|engine| engine.term_call_arity(TermId(id)) as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn term_call_arg(id: u32, idx: u32) -> u32 {
    with_engine(|engine| engine.term_call_arg(TermId(id), idx as usize).0)
}

// ── Dynamic rules ──

#[unsafe(no_mangle)]
pub extern "C" fn assert_rule(lhs: u32, rhs: u32) {
    with_engine(|engine| {
        let rule_sym = engine.make_sym("rule");
        let call = engine.make_call(rule_sym, &[TermId(lhs), TermId(rhs)]);
        engine.eval(call);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn retract_rule(lhs: u32) {
    with_engine(|engine| {
        let retract_sym = engine.make_sym("retract");
        let call = engine.make_call(retract_sym, &[TermId(lhs)]);
        engine.eval(call);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn query_all(tag_id: u32) -> u32 {
    with_engine(|engine| {
        let qa_sym = engine.make_sym("query_all");
        let call = engine.make_call(qa_sym, &[TermId(tag_id)]);
        engine.eval(call).0
    })
}

// ── Display ──

#[unsafe(no_mangle)]
pub extern "C" fn display_term(id: u32) -> *const u8 {
    with_engine(|engine| {
        let s = engine.display(TermId(id));
        LAST_STRING.with(|ls| {
            *ls.borrow_mut() = s;
            ls.borrow().as_ptr()
        })
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn display_term_len() -> u32 {
    LAST_STRING.with(|ls| ls.borrow().len() as u32)
}

// ── Generic scope pending buffer ──

#[unsafe(no_mangle)]
pub extern "C" fn scope_pending_count(name_ptr: *const u8, name_len: u32) -> u32 {
    let name = unsafe {
        let slice = std::slice::from_raw_parts(name_ptr, name_len as usize);
        std::str::from_utf8(slice).expect("Invalid UTF-8")
    };
    with_engine(|engine| engine.scope_pending_count(name) as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn scope_pending_get(name_ptr: *const u8, name_len: u32, idx: u32) -> u32 {
    let name = unsafe {
        let slice = std::slice::from_raw_parts(name_ptr, name_len as usize);
        std::str::from_utf8(slice).expect("Invalid UTF-8")
    };
    with_engine(|engine| engine.scope_pending_get(name, idx as usize).0)
}

#[unsafe(no_mangle)]
pub extern "C" fn scope_pending_clear(name_ptr: *const u8, name_len: u32) {
    let name = unsafe {
        let slice = std::slice::from_raw_parts(name_ptr, name_len as usize);
        std::str::from_utf8(slice).expect("Invalid UTF-8")
    };
    with_engine(|engine| engine.scope_pending_clear(name))
}

// ── Memory access ──

#[unsafe(no_mangle)]
pub extern "C" fn get_memory() -> *const u8 {
    // Helper for JS to get base memory pointer
    std::ptr::null()
}
