//! A completely ordinary Rust program — NO memscope, NO custom allocator, no
//! instrumentation of any kind. Used to prove `memscope-preload` can dump its
//! heap purely by injection (DYLD_INSERT_LIBRARIES / LD_PRELOAD).
//!
//! It builds a registry of accounts (each owning a boxed profile + a vec of
//! tag strings), holds it live, prints its pid, and waits — so you can
//! `kill -USR1 <pid>` to trigger a heap dump.

use std::collections::HashMap;

#[allow(dead_code)]
struct Profile {
    id: u64,
    display_name: String,
}

#[allow(dead_code)]
struct Account {
    profile: Box<Profile>,
    tags: Vec<String>,
}

fn main() {
    let mut registry: HashMap<u64, Box<Account>> = HashMap::new();
    for id in 0..500u64 {
        let tags: Vec<String> =
            (0..4).map(|t| format!("account-{id}-tag-{t}-payload")).collect();
        registry.insert(
            id,
            Box::new(Account {
                profile: Box::new(Profile { id, display_name: format!("account-{id:05}") }),
                tags,
            }),
        );
    }
    std::hint::black_box(&registry);

    println!("plainheap pid={} — kill -USR1 it for a heap dump", std::process::id());
    // Stay alive so the dump can be triggered; exit after a while.
    for _ in 0..600 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        std::hint::black_box(&registry);
    }
}
