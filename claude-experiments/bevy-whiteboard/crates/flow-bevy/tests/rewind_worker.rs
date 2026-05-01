//! Worker-mode rewind tests. Unlike `rewind_e2e` and `rewind_ui`
//! which use `SimDriver::direct` (synchronous, single-threaded),
//! these tests use `SimDriver::worker` so we exercise the real
//! channel + background-thread path:
//!
//!   * `driver.rewind(target)` sends `WorkerMsg::Rewind` and
//!      returns immediately.
//!   * The worker drains queued messages, coalesces multiple
//!      `Rewind`s, then runs `do_rewind` on the latest target.
//!   * Snapshot publication is async; we poll the published
//!      snapshot to observe the rewind landing.
//!
//! This is the path that produces the user-reported drag-lockup
//! behavior — a flood of `Rewind` messages must NOT serially run
//! the strategy for every stale target.
//!
//! Tests use real `std::thread::sleep` to give the worker time to
//! process. Tight loops with `try_recv` would race against the
//! worker's own scheduling.

use std::thread::sleep;
use std::time::{Duration, Instant};

use flow::Sim;
use flow_bevy::sim_driver::SimDriver;

/// Build a fresh worker driver with a dense Client/Worker
/// topology in `sim`. Returns the driver alone; tests don't need
/// the event channel for these scenarios.
fn worker_driver(num_pairs: usize) -> SimDriver {
    let mut sim = Sim::new(1);
    flow_bevy::gadgets::install_default_params(&mut sim);
    for i in 0..num_pairs {
        let client = flow_bevy::gadgets::spawn(
            &mut sim,
            flow_bevy::gadgets::Kind::Client,
            &format!("Cli_{}", i),
            i % 3,
        );
        let worker = flow_bevy::gadgets::spawn(
            &mut sim,
            flow_bevy::gadgets::Kind::Worker,
            &format!("Wkr_{}", i),
            i % 3,
        );
        if let Some(n) = sim.nodes.get_mut(&client) {
            n.slots.insert("period_ns".into(), flow::Value::Int(50_000_000));
        }
        if let Some(n) = sim.nodes.get_mut(&worker) {
            n.slots.insert("service_ns".into(), flow::Value::Int(10_000_000));
        }
        sim.add_edge(client, worker, flow::Expr::int(1_000_000));
    }
    let (driver, _events_rx) = SimDriver::worker(sim, 1.0);
    driver
}

/// Wait up to `budget` for the worker to publish a snapshot whose
/// `now_ns` matches `predicate`. Returns the elapsed time on
/// success, or `None` if the budget expired.
fn wait_for<F: Fn(u64) -> bool>(
    driver: &SimDriver,
    predicate: F,
    budget: Duration,
) -> Option<Duration> {
    let start = Instant::now();
    while start.elapsed() < budget {
        if predicate(driver.snapshot().now_ns) {
            return Some(start.elapsed());
        }
        sleep(Duration::from_millis(2));
    }
    None
}

// ─────────────────────────────────────────────────────────────────
// `<<` (rewind to t=0) in worker mode
// ─────────────────────────────────────────────────────────────────

#[test]
fn worker_rewind_to_zero_lands_at_zero() {
    let mut driver = worker_driver(4);

    // Let the worker advance for a couple seconds.
    driver.control().set_paused(false);
    let _ = wait_for(&driver, |t| t > 1_000_000_000, Duration::from_secs(3))
        .expect("worker didn't advance past 1s within 3s");
    driver.control().set_paused(true);
    sleep(Duration::from_millis(50));

    // Now rewind to 0.
    driver.rewind(0);
    let elapsed = wait_for(&driver, |t| t == 0, Duration::from_secs(2))
        .expect("worker rewind to 0 didn't land within 2s");

    assert!(
        elapsed.as_millis() < 500,
        "worker rewind to 0 took {}ms — too slow",
        elapsed.as_millis(),
    );
}

// ─────────────────────────────────────────────────────────────────
// Drag-scrubbing: rewind coalescing pays off here
// ─────────────────────────────────────────────────────────────────

/// The actual user-reported lockup scenario: a fast drag posts
/// many `Rewind` messages in rapid succession. With coalescing,
/// the worker drops stale ones and only runs `do_rewind` for the
/// latest target. Without coalescing, each rewind runs serially
/// and the worker spends multiple seconds catching up.
#[test]
fn worker_drag_scrub_completes_quickly() {
    let mut driver = worker_driver(8);

    // Run forward for a while.
    driver.control().set_paused(false);
    let _ = wait_for(&driver, |t| t > 5_000_000_000, Duration::from_secs(8))
        .expect("worker didn't reach 5s within 8s");
    driver.control().set_paused(true);
    sleep(Duration::from_millis(50));

    let now = driver.snapshot().now_ns;
    assert!(now >= 5_000_000_000);

    // Simulate a drag: 50 rewind targets in quick succession,
    // sweeping from current time back toward 0. With coalescing,
    // the worker should only execute the *last* one.
    let drag_start = Instant::now();
    for i in 0..50 {
        let frac = 1.0 - (i as f64 / 50.0);
        let target = ((now as f64) * frac) as u64;
        driver.rewind(target);
    }
    let dispatch_elapsed = drag_start.elapsed();

    // Final target is at i=49, frac = 0.02 → ~2% of `now`.
    let final_target = ((now as f64) * 0.02) as u64;

    // Wait for the worker to settle on the final target. With
    // coalescing this is one rewind's worth of work; without it,
    // it could be 50× that.
    let landed_at = wait_for(
        &driver,
        |t| t.abs_diff(final_target) < 100_000_000,
        Duration::from_secs(5),
    );
    let total = drag_start.elapsed();

    assert!(
        landed_at.is_some(),
        "worker didn't catch up to final drag target ({}ns) within 5s — \
         current sim_now={}ns, dispatched 50 rewinds in {}ms",
        final_target,
        driver.snapshot().now_ns,
        dispatch_elapsed.as_millis(),
    );
    assert!(
        total.as_millis() < 1_500,
        "drag-scrub took {}ms total — coalescing likely broken",
        total.as_millis(),
    );
}

/// Same idea but harder: long session, all targets outside the
/// snapshot ring's coverage. Each non-coalesced rewind would
/// require running sim forward from t=0, multiplying the cost.
#[test]
fn worker_long_distance_drag_doesnt_lockup() {
    let mut driver = worker_driver(8);

    // Run the worker for ~10 sim-seconds — far enough that the
    // ring (16s coverage) still has the recent past but most
    // sweep positions land outside it.
    driver.control().set_paused(false);
    driver.control().set_multiplier(20.0); // 20× speedup so we get there in real time
    let _ = wait_for(&driver, |t| t > 10_000_000_000, Duration::from_secs(8))
        .expect("worker didn't reach 10s within 8s wall");
    driver.control().set_paused(true);
    driver.control().set_multiplier(1.0);
    sleep(Duration::from_millis(50));

    let now = driver.snapshot().now_ns;
    assert!(now >= 10_000_000_000);

    // 100 sweep stops back to ~0.
    let drag_start = Instant::now();
    for i in 0..100 {
        let frac = 1.0 - (i as f64 / 100.0);
        let target = ((now as f64) * frac) as u64;
        driver.rewind(target);
    }
    let final_target = ((now as f64) * 0.0) as u64; // i=99 → 1% off, but the LAST iteration uses i=99 which gives frac = 0.01

    let landed_at = wait_for(
        &driver,
        |t| t < 500_000_000, // landed somewhere near 0
        Duration::from_secs(5),
    );
    let total = drag_start.elapsed();

    assert!(
        landed_at.is_some(),
        "worker didn't land near 0 within 5s of dragging — sim_now={}ns",
        driver.snapshot().now_ns,
    );
    assert!(
        total.as_millis() < 2_500,
        "long-distance drag took {}ms — UI would be locked up; final_target={}ns",
        total.as_millis(), final_target,
    );
}

// ─────────────────────────────────────────────────────────────────
// Coalescing assertion: 50 rewinds should NOT cost 50× one rewind
// ─────────────────────────────────────────────────────────────────

/// Direct evidence the coalescing fix works. Each `do_rewind`
/// bumps `rewind_epoch` by one; with coalescing the worker bumps
/// it once for the whole batch, regardless of how many `Rewind`
/// messages were queued. Without coalescing, the epoch climbs by
/// (≈ batch size).
///
/// We use the snapshot's published `rewind_epoch` rather than wall
/// time so the assertion isn't sensitive to per-rewind cost in the
/// test environment.
#[test]
fn worker_coalesces_queued_rewinds() {
    let mut driver = worker_driver(20);

    driver.control().set_paused(false);
    driver.control().set_multiplier(10.0);
    let _ = wait_for(&driver, |t| t > 5_000_000_000, Duration::from_secs(8))
        .expect("worker didn't reach 5s within 8s wall");
    driver.control().set_paused(true);
    driver.control().set_multiplier(1.0);
    sleep(Duration::from_millis(100));

    let now = driver.snapshot().now_ns;
    assert!(now >= 5_000_000_000);
    let epoch_start = driver.snapshot().rewind_epoch;

    // Queue 30 rewinds in tight succession. Targets sweep down
    // toward 0 so the worker can't dedupe via "same target".
    let mut final_target = 0u64;
    for i in 0..30 {
        let frac = 0.9 - (i as f64 * 0.85 / 29.0); // 0.9 → 0.05
        let target = (now as f64 * frac) as u64;
        driver.rewind(target);
        final_target = target;
    }

    // Wait for the worker to settle on the final dispatched target.
    let landed = wait_for(
        &driver,
        |t| t.abs_diff(final_target) < 200_000_000,
        Duration::from_secs(10),
    );
    assert!(
        landed.is_some(),
        "worker never settled on final target after 30-rewind drag",
    );
    sleep(Duration::from_millis(50));

    let epoch_end = driver.snapshot().rewind_epoch;
    let epochs_advanced = epoch_end - epoch_start;
    eprintln!(
        "coalescing: 30 rewinds dispatched, epoch advanced by {}",
        epochs_advanced,
    );
    // With coalescing the worker drains the queue → one
    // `do_rewind` per drain cycle. The dispatch loop on the test
    // thread is faster than the worker's drain so all 30 should
    // land in 1-2 batches; we allow up to 4 to absorb the timing
    // slack of the test runner. Without coalescing, this would
    // be 30.
    assert!(
        epochs_advanced <= 4,
        "30 queued rewinds bumped rewind_epoch {} times — coalescing is broken (expected ≤ 4)",
        epochs_advanced,
    );
}
