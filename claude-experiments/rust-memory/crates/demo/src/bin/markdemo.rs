//! Exercises `memscope::mark` checkpoints + a recording, so `memscope marks`
//! and `memscope diff` can be run against the output.
//!
//!   markdemo <out.mscope>
//!
//! Phases: warm up a steady-state buffer, mark "after_warmup", then simulate a
//! per-request leak (entries pushed into a cache that's never drained) plus a
//! lot of transient churn that frees, then mark "end". A diff of the two marks
//! should show the leaking `String`/`Vec` growing with freed_in_window == 0,
//! while the churn nets to ~zero.

use memscope::{MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

struct Session {
    _id: u64,
    _token: String,
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "/tmp/markdemo.mscope".to_string());

    memscope::set_mode(Mode::Full);
    memscope::record_to_file(&out).expect("failed to start recording");

    // --- warmup: a steady-state working set we hold for the whole run ---
    let mut working: Vec<Vec<u8>> = Vec::new();
    for i in 0..200 {
        working.push(vec![0u8; 256 + (i & 63)]);
    }
    std::hint::black_box(&working);

    memscope::mark("after_warmup");

    // --- the leak: a cache that grows per "request" and is never evicted ---
    let mut cache: Vec<Session> = Vec::new();
    for req in 0..5_000u64 {
        cache.push(Session {
            _id: req,
            _token: format!("token-{req:08x}-session-data-padding"),
        });

        // --- transient churn: allocate + free within the loop (nets to zero) ---
        for k in 0..8 {
            let tmp: Vec<u8> = Vec::with_capacity(128 + k * 16);
            std::hint::black_box(&tmp);
            // tmp dropped here
        }
    }
    std::hint::black_box(&cache);
    std::hint::black_box(&working);

    memscope::mark("end");

    // Let the recorder pump drain the last batch before we exit.
    std::thread::sleep(std::time::Duration::from_millis(200));
    eprintln!(
        "[markdemo] wrote {out}: held {} working buffers + {} leaked sessions",
        working.len(),
        cache.len()
    );
}
