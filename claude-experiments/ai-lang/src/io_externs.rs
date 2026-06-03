//! Standard I/O externs registered in the global FFI registry.
//!
//! These are the runtime side of the stdlib's `extern fn` declarations
//! for `print_int`, `print_string`, `println`, `read_line`,
//! `int_to_string`, `string_to_int`. Calling `register_io_externs()`
//! once per process (idempotent — `register_extern` replaces) makes
//! all of them callable from any ai-lang program.
//!
//! ## ABI
//!
//! All externs follow the Layer-1/Layer-2 convention:
//!   `unsafe extern "C" fn(*mut Thread, ...args) -> ret`
//! where each `Int` is `i64` and each `String` is `*const u8`
//! (pointer to an ai-lang heap String — a `Full` header + `count: i64`
//! + raw UTF-8 bytes).
//!
//! Allocations (`int_to_string`, `read_line`) go through `ai_str_new`
//! which routes through the GC. Callers in ai-lang must be ready for
//! a safepoint at every extern call site.

use crate::ast::Type;
use crate::ffi::{heap_str_to_owned, owned_str_to_heap};
use crate::runtime::Thread;
use std::io::{BufRead, Write};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

// =============================================================================
// Process-global tables the runner populates before invoking JIT'd code.
// =============================================================================

/// Command-line args visible to user code via `arg_count()` / `get_arg(i)`.
/// Populated by `set_user_args` before calling `main()`.
static USER_ARGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn user_args_lock() -> &'static Mutex<Vec<String>> {
    USER_ARGS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Set the user-args table that `arg_count()` / `get_arg(i)` read from.
pub fn set_user_args(args: Vec<String>) {
    let m = user_args_lock();
    let mut g = m.lock().unwrap();
    *g = args;
}

/// TCP ports of worker nodes spawned by the runner, exposed to ai-lang
/// code via `node_count()` / `get_node_port(i)`.
static WORKER_NODES: OnceLock<Mutex<Vec<u16>>> = OnceLock::new();

fn worker_nodes_lock() -> &'static Mutex<Vec<u16>> {
    WORKER_NODES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Populate the worker-node table that `node_count()` / `get_node_port(i)` read.
pub fn set_worker_nodes(ports: Vec<u16>) {
    let m = worker_nodes_lock();
    let mut g = m.lock().unwrap();
    *g = ports;
}

fn int_t() -> Type {
    Type::Builtin("Int".to_owned())
}

fn string_t() -> Type {
    Type::Builtin("String".to_owned())
}

// =============================================================================
// Extern implementations
// =============================================================================

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_print_int(_thread: *mut Thread, n: i64) -> i64 {
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let _ = write!(h, "{}", n);
    let _ = h.flush();
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_print_string(_thread: *mut Thread, s: *const u8) -> i64 {
    let owned = unsafe { heap_str_to_owned(s) };
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let _ = h.write_all(owned.as_bytes());
    let _ = h.flush();
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_println(_thread: *mut Thread, s: *const u8) -> i64 {
    let owned = unsafe { heap_str_to_owned(s) };
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let _ = h.write_all(owned.as_bytes());
    let _ = h.write_all(b"\n");
    let _ = h.flush();
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_read_line(thread: *mut Thread) -> *mut u8 {
    let stdin = std::io::stdin();
    let mut line = String::new();
    let n = stdin.lock().read_line(&mut line).unwrap_or(0);
    if n == 0 {
        unsafe { owned_str_to_heap(thread, "") }
    } else {
        // Strip trailing newline if present.
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        unsafe { owned_str_to_heap(thread, &line) }
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_int_to_string(thread: *mut Thread, n: i64) -> *mut u8 {
    let s = n.to_string();
    unsafe { owned_str_to_heap(thread, &s) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_string_to_int(_thread: *mut Thread, s: *const u8) -> i64 {
    let owned = unsafe { heap_str_to_owned(s) };
    owned.trim().parse::<i64>().unwrap_or(0)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_string_is_int(_thread: *mut Thread, s: *const u8) -> i64 {
    let owned = unsafe { heap_str_to_owned(s) };
    if owned.trim().parse::<i64>().is_ok() { 1 } else { 0 }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_sleep_ms(_thread: *mut Thread, ms: i64) -> i64 {
    if ms > 0 {
        std::thread::sleep(Duration::from_millis(ms as u64));
    }
    0
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_arg_count(_thread: *mut Thread) -> i64 {
    user_args_lock().lock().unwrap().len() as i64
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_get_arg(thread: *mut Thread, i: i64) -> *mut u8 {
    let s = user_args_lock()
        .lock()
        .unwrap()
        .get(i as usize)
        .cloned()
        .unwrap_or_default();
    unsafe { owned_str_to_heap(thread, &s) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_node_count(_thread: *mut Thread) -> i64 {
    worker_nodes_lock().lock().unwrap().len() as i64
}

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_io_get_node_port(_thread: *mut Thread, i: i64) -> i64 {
    worker_nodes_lock()
        .lock()
        .unwrap()
        .get(i as usize)
        .copied()
        .map(|p| p as i64)
        .unwrap_or(-1)
}

// =============================================================================
// Registration
// =============================================================================

/// Install the standard I/O externs into the process-global, write-once
/// table. Safe and cheap to call any number of times from any thread:
/// the first call wins and every later call is a no-op (the I/O externs
/// are static, so there is nothing to update or tear down). This is what
/// makes the stdlib's `extern fn print_int(...)` etc. resolve on every
/// build thread without any cross-test races.
pub fn register_io_externs() {
    crate::ffi::install_io_externs(io_extern_entries());
}

fn io_extern_entries() -> Vec<(String, crate::ffi::ExternEntry)> {
    fn e(name: &str, params: Vec<Type>, ret: Type, fn_ptr: usize) -> (String, crate::ffi::ExternEntry) {
        (name.to_owned(), crate::ffi::ExternEntry { params, ret, fn_ptr })
    }
    vec![
        e("print_int", vec![int_t()], int_t(), ai_io_print_int as usize),
        e("print_string", vec![string_t()], int_t(), ai_io_print_string as usize),
        e("println", vec![string_t()], int_t(), ai_io_println as usize),
        e("read_line", vec![], string_t(), ai_io_read_line as usize),
        e("int_to_string", vec![int_t()], string_t(), ai_io_int_to_string as usize),
        e("string_to_int", vec![string_t()], int_t(), ai_io_string_to_int as usize),
        e("string_is_int", vec![string_t()], int_t(), ai_io_string_is_int as usize),
        e("sleep_ms", vec![int_t()], int_t(), ai_io_sleep_ms as usize),
        e("arg_count", vec![], int_t(), ai_io_arg_count as usize),
        e("get_arg", vec![int_t()], string_t(), ai_io_get_arg as usize),
        e("node_count", vec![], int_t(), ai_io_node_count as usize),
        e("get_node_port", vec![int_t()], int_t(), ai_io_get_node_port as usize),
    ]
}
