//! End-to-end test of the `meta!` path through the *real* tracking allocator and
//! the async reconstruction pump — not the isolated read-side logic, but the
//! whole write path: `meta!` -> ring -> pump -> sink, then correlation.
//!
//! A capturing [`EventSink`] records every drained `RawEvent`; the workload tags
//! a couple of distinctly-sized allocations with metadata; we flush the pump and
//! assert each allocation carries the metadata that was live when it happened.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use memscope::{EventKind, EventSink, MemScope, Mode, RawEvent};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

/// Sink that appends every drained event to a shared buffer.
struct Capture(Arc<Mutex<Vec<RawEvent>>>);
impl EventSink for Capture {
    fn consume(&mut self, events: &[RawEvent]) {
        // Runs on the pump thread, which is excluded from tracking, so this
        // push doesn't recurse into the recorder.
        self.0.lock().unwrap().extend_from_slice(events);
    }
}

/// Re-derive each allocation's merged metadata by replaying the captured stream
/// per thread (the same idea the CLI's `correlate_meta` uses), resolving context
/// ids through the live recorder's tables.
fn tagged_allocs(events: &[RawEvent]) -> Vec<(u64, BTreeMap<String, String>)> {
    let mut stacks: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut out = Vec::new();
    for e in events {
        match e.kind {
            EventKind::MetaEnter => stacks.entry(e.thread).or_default().push(e.site.0),
            EventKind::MetaExit => {
                if let Some(s) = stacks.get_mut(&e.thread) {
                    if let Some(p) = s.iter().rposition(|&m| m == e.site.0) {
                        s.remove(p);
                    } else {
                        s.pop();
                    }
                }
            }
            EventKind::Alloc | EventKind::ReallocGrow => {
                let mut m = BTreeMap::new();
                if let Some(s) = stacks.get(&e.thread) {
                    for mid in s {
                        if let Some(kvs) = memscope::meta_context(*mid) {
                            for (kid, val) in kvs {
                                if let Some(name) = memscope::key_name(kid) {
                                    m.insert(name, val.to_display());
                                }
                            }
                        }
                    }
                }
                out.push((e.size, m));
            }
            EventKind::Dealloc | EventKind::Mark => {}
        }
    }
    out
}

#[test]
fn metadata_attaches_to_real_allocations() {
    memscope::set_mode(Mode::Full);
    let buf = Arc::new(Mutex::new(Vec::new()));
    memscope::spawn_consumer(Box::new(Capture(buf.clone())), Duration::from_millis(1));

    // A distinctly-sized allocation inside each scope so we can find it by size.
    const PHYS_LEN: usize = 4096; // Vec<u64> -> 32768 bytes
    const IO_LEN: usize = 9000; // Vec<u8>  -> 9000 bytes

    {
        let _m = memscope::meta!(subsystem = "physics", shard = 3u32);
        let v: Vec<u64> = (0..PHYS_LEN as u64).collect();
        std::hint::black_box(&v);
    }
    {
        let _m = memscope::meta!(subsystem = "io");
        let v: Vec<u8> = vec![7u8; IO_LEN];
        std::hint::black_box(&v);
    }
    // An allocation outside every scope must carry no metadata.
    let untagged: Vec<u64> = (0..2048).collect();
    std::hint::black_box(&untagged);

    // Flush the async pump so every event reaches the sink, then read it.
    for _ in 0..5 {
        let _ = memscope::stats();
        std::thread::sleep(Duration::from_millis(20));
    }
    let events = buf.lock().unwrap().clone();
    assert!(!events.is_empty(), "pump delivered no events");

    let tagged = tagged_allocs(&events);

    // The physics Vec<u64> (>= 32768 bytes) must be tagged subsystem=physics, shard=3.
    let phys = tagged
        .iter()
        .find(|(sz, m)| *sz >= (PHYS_LEN * 8) as u64 && m.get("subsystem").map(String::as_str) == Some("physics"));
    let (_, pm) = phys.expect("no physics-tagged allocation of the expected size found");
    assert_eq!(pm.get("shard").map(String::as_str), Some("3"));

    // The io Vec<u8> must be tagged subsystem=io and must NOT inherit `shard`
    // (that scope already closed).
    let io = tagged
        .iter()
        .find(|(sz, m)| *sz == IO_LEN as u64 && m.get("subsystem").map(String::as_str) == Some("io"));
    let (_, im) = io.expect("no io-tagged allocation found");
    assert!(im.get("shard").is_none(), "io allocation wrongly inherited shard");

    // At least one allocation carried no metadata at all (the untagged Vec).
    assert!(
        tagged.iter().any(|(_, m)| m.is_empty()),
        "expected some untagged allocations"
    );
}
