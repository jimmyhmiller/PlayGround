//! Builds a small, interesting heap (a registry of sessions, each owning a
//! boxed user and a vector of message strings, plus a linked list) and writes a
//! JVM HPROF heap dump — open `out.hprof` in Eclipse MAT or VisualVM.
//!
//!   hprofdemo [out.hprof]

use std::collections::HashMap;

use memscope::{MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

// Fields are "read" by the heap walker (via memory + DWARF layout), not by Rust.
#[allow(dead_code)]
struct User {
    _id: u64,
    name: String,
}

#[allow(dead_code)]
struct Session {
    user: Box<User>,
    messages: Vec<String>,
    next: Option<Box<Session>>,
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/hprofdemo.hprof".to_string());
    memscope::set_mode(Mode::Full);

    // A registry of sessions, each owning a boxed user + a vec of messages.
    let mut registry: HashMap<u64, Box<Session>> = HashMap::new();
    for id in 0..200u64 {
        let messages: Vec<String> =
            (0..5).map(|m| format!("session {id} message {m} — some payload text")).collect();
        let session = Box::new(Session {
            user: Box::new(User { _id: id, name: format!("user-{id:04}") }),
            messages,
            next: None,
        });
        registry.insert(id, session);
    }

    // A linked list of sessions so MAT shows a deep dominator chain.
    let mut head: Option<Box<Session>> = None;
    for id in 0..50u64 {
        head = Some(Box::new(Session {
            user: Box::new(User { _id: 1_000_000 + id, name: format!("chain-{id}") }),
            messages: vec![format!("chain node {id}")],
            next: head.take(),
        }));
    }

    std::hint::black_box(&registry);
    std::hint::black_box(&head);

    // Dump the heap. registry + head are live, so MAT will show them retained.
    match memscope::heap_dump(&out) {
        Ok(s) => eprintln!(
            "[hprofdemo] {} objects across {} classes -> {out}\n  open it in Eclipse MAT / VisualVM",
            s.objects, s.classes
        ),
        Err(e) => {
            eprintln!("[hprofdemo] heap dump failed: {e}");
            std::process::exit(1);
        }
    }

    std::hint::black_box(&registry);
    std::hint::black_box(&head);
}
