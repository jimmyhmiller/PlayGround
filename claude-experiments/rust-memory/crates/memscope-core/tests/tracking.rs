//! Integration test: install the tracking allocator and verify the live table
//! reflects real allocations and frees, in both Full and Sampled modes.

use std::sync::Mutex;

use memscope_core::{self as mem, MemScope, Mode};

#[global_allocator]
static GLOBAL: MemScope = MemScope::system();

/// The recording mode is process-global, so these tests must not run
/// concurrently. Serialize them.
static SERIAL: Mutex<()> = Mutex::new(());

#[test]
fn full_mode_tracks_live_and_free() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    mem::set_mode(Mode::Full);

    // A precisely-sized allocation we can find by address + size.
    let v: Vec<u64> = Vec::with_capacity(1000); // 8000 bytes
    let addr = v.as_ptr() as u64;

    let snap = mem::snapshot();
    let found = snap
        .live
        .iter()
        .find(|l| l.addr == addr)
        .expect("allocation should be live in the snapshot");
    assert_eq!(found.size, 8000, "tracked size must match the layout");
    assert!(found.site.is_some(), "a site should have been captured");

    drop(v);

    let snap2 = mem::snapshot();
    assert!(
        !snap2.live.iter().any(|l| l.addr == addr),
        "freed allocation must leave the live set"
    );
}

#[test]
fn live_bytes_returns_to_baseline_after_freeing() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    mem::set_mode(Mode::Full);
    let base = mem::stats().live_bytes;

    let mut held: Vec<Vec<u8>> = Vec::new();
    for _ in 0..200 {
        held.push(vec![0u8; 1024]);
    }
    let peak = mem::stats().live_bytes;
    assert!(
        peak >= base + 200 * 1024,
        "live bytes should grow by at least the bytes we held"
    );

    drop(held);
    let after = mem::stats().live_bytes;
    // Allow slack for unrelated test-harness allocations, but the 200 KiB we
    // freed must be gone.
    assert!(
        after < peak - 150 * 1024,
        "freeing should drop live bytes well below the peak (peak={peak}, after={after})"
    );
}

#[test]
fn sampled_mode_records_a_fraction() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    mem::set_mode(Mode::Sampled);
    mem::set_sample_rate(50);
    let before = mem::stats().total_allocs;

    let mut keep: Vec<Vec<u8>> = Vec::new();
    for _ in 0..5000 {
        keep.push(vec![0u8; 8]);
    }
    std::hint::black_box(&keep);

    let recorded = mem::stats().total_allocs - before;
    // ~5000/50 = 100 expected; assert it's clearly a sampled fraction, not all.
    assert!(
        recorded > 10 && recorded < 1000,
        "sampled recording should be a fraction of 5000 allocations, got {recorded}"
    );

    mem::set_mode(Mode::Off);
}
