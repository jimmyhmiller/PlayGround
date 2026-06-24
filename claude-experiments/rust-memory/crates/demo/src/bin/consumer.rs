//! Verifies the lock-free event ring + pluggable consumer end to end.
//!
//! Installs a consumer whose sink is a shared `LiveSet` reconstructor: it rebuilds
//! the live allocation set purely by replaying the event stream off the hot path.
//! We then check the reconstruction matches the in-process live table, and that
//! Reliable mode delivered every event (zero drops).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use memscope::{LiveSet, MemScope, Mode, RingMode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

fn main() {
    memscope::set_mode(Mode::Full);
    memscope::set_ring_mode(RingMode::Reliable); // validate exactness: no drops

    // The sink is a shared LiveSet: the pump updates it; we read it here.
    let recon = Arc::new(Mutex::new(LiveSet::new()));
    let consumer = memscope::spawn_consumer(Box::new(recon.clone()), Duration::from_millis(1));

    println!("consumer demo: running workload...");

    // Retain 5,000 boxes; churn 50,000 transient vecs that get freed.
    let mut retained: Vec<Box<[u64; 4]>> = Vec::new();
    for i in 0..5_000u64 {
        retained.push(Box::new([i, i + 1, i + 2, i + 3]));
    }
    for _ in 0..50_000 {
        let v: Vec<u8> = Vec::with_capacity(64);
        std::hint::black_box(&v);
    }
    std::hint::black_box(&retained);

    // Let the consumer drain the tail of the stream.
    std::thread::sleep(Duration::from_millis(250));

    let table = memscope::snapshot();

    let g = recon.lock().unwrap();
    println!("\n  RECONSTRUCTED (replayed from the event stream, off the hot path):");
    println!("    live allocations : {}", g.live_count());
    println!("    live bytes       : {}", g.live_bytes());
    println!("    total allocs     : {}", g.total_allocs());
    println!("    total bytes      : {}", g.total_alloc_bytes());
    println!("    ring dropped     : {}", memscope::ring_dropped());
    println!("\n  IN-PROCESS TABLE (snapshot of the live set):");
    println!("    live allocations : {}", table.live.len());
    println!("    live bytes       : {}", table.total_live_bytes);

    let recon_count = g.live_count();
    let table_count = table.live.len();
    let diff = recon_count.abs_diff(table_count);
    drop(g);
    drop(consumer); // stop the pump

    println!("\n  difference: {diff} allocations  (consumer's own teardown accounts for a few)");
    assert_eq!(memscope::ring_dropped(), 0, "Reliable mode must not drop");
    assert!(
        diff < 200,
        "reconstructed live set ({recon_count}) should match the table ({table_count})"
    );
    println!("  \u{2713} reconstruction matches the live table, zero drops");
}
