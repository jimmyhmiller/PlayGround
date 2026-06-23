//! Spike: does a backtrace captured *inside* the global allocator carry the
//! concrete monomorphized type in a demangleable symbol?
//!
//! We install a global allocator that, when "armed" for a single allocation,
//! captures a backtrace and stashes the demangled frame names. A thread-local
//! reentrancy guard prevents the symbolication machinery (which itself
//! allocates) from recursing into our recorder.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};

thread_local! {
    /// True while we are inside our own bookkeeping; alloc calls made here
    /// must bypass recording to avoid infinite recursion.
    static IN_RECORDER: Cell<bool> = const { Cell::new(false) };
    /// When true, the *next* recorded alloc captures a trace.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static CAPTURED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct TracingAlloc;

unsafe impl GlobalAlloc for TracingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        maybe_capture(layout);
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        System.realloc(ptr, layout, new_size)
    }
}

fn maybe_capture(_layout: Layout) {
    let armed = ARMED.with(|a| a.get());
    if !armed {
        return;
    }
    if IN_RECORDER.with(|g| g.get()) {
        return;
    }
    IN_RECORDER.with(|g| g.set(true));
    ARMED.with(|a| a.set(false)); // capture only the first armed alloc

    let mut frames: Vec<String> = Vec::new();
    backtrace::trace(|frame| {
        backtrace::resolve_frame(frame, |sym| {
            if let Some(name) = sym.name() {
                frames.push(format!("{}", name)); // Display = rustc-demangled
            }
        });
        frames.len() < 40
    });
    CAPTURED.with(|c| *c.borrow_mut() = frames);

    IN_RECORDER.with(|g| g.set(false));
}

#[global_allocator]
static A: TracingAlloc = TracingAlloc;

fn probe<F: FnOnce()>(label: &str, needle: &str, f: F) {
    CAPTURED.with(|c| c.borrow_mut().clear());
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));

    println!("\n=== {label} (looking for type containing '{needle}') ===");
    let frames = CAPTURED.with(|c| c.borrow().clone());
    let mut hit = false;
    for (i, fr) in frames.iter().enumerate() {
        let mark = if fr.contains(needle) { hit = true; ">>" } else { "  " };
        if i < 14 {
            println!("{mark} [{i:2}] {fr}");
        }
    }
    println!("    TYPE RECOVERED FROM SYMBOL: {}", if hit { "YES" } else { "NO" });
}

#[derive(Debug)]
#[allow(dead_code)]
struct Widget {
    id: u64,
    name: String,
    coords: [f64; 3],
}

fn main() {
    probe("Box::new::<Widget>", "Widget", || {
        let b = Box::new(Widget { id: 1, name: String::new(), coords: [0.0; 3] });
        std::hint::black_box(&b);
    });

    probe("Vec::<Widget>::push (grow)", "Widget", || {
        let mut v: Vec<Widget> = Vec::new();
        v.push(Widget { id: 2, name: String::new(), coords: [1.0; 3] });
        std::hint::black_box(&v);
    });

    probe("Vec::<u64>::with_capacity", "u64", || {
        let v: Vec<u64> = Vec::with_capacity(128);
        std::hint::black_box(&v);
    });

    probe("String::with_capacity", "alloc", || {
        let s = String::with_capacity(64);
        std::hint::black_box(&s);
    });

    probe("Rc::new::<Widget>", "Widget", || {
        let r = std::rc::Rc::new(Widget { id: 3, name: String::new(), coords: [2.0; 3] });
        std::hint::black_box(&r);
    });
}
