//! Regression: pausing the sim, waiting in real time, then resuming
//! must NOT jump `sim.now_ns` forward by the pause duration.
//!
//! Before the fix, `worker_loop` left `last_tick` untouched while
//! `control.paused()` was true. On resume, `elapsed = now - last_tick`
//! spanned the entire pause window and the worker advanced the sim
//! by that whole real-time span on the next iteration.

use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;

use flow::Sim;
use flow_bevy::sim_driver::SimDriver;

fn now_ns(driver: &SimDriver) -> u64 {
    driver.snapshot().now_ns
}

#[test]
fn pause_then_wait_then_resume_does_not_jump_sim_clock() {
    // Empty sim is enough: `run_until` advances `now_ns` to the
    // deadline unconditionally when there are no scheduled events.
    let sim = Sim::new(0);
    let (driver, _events_rx): (SimDriver, mpsc::Receiver<flow::Event>) =
        SimDriver::worker(sim, 1.0);

    // Run forward briefly to establish a non-trivial sim time.
    sleep(Duration::from_millis(150));
    let before_pause = now_ns(&driver);
    assert!(
        before_pause > 50_000_000,
        "sim should be advancing while running; got {}ns after 150ms wall",
        before_pause,
    );

    // Pause and wait in real time. With the bug, the worker keeps
    // accumulating wall-clock elapsed against `last_tick`.
    driver.control().set_paused(true);
    // Give the worker a tick to observe `paused=true` before we start
    // counting the pause window.
    sleep(Duration::from_millis(20));
    let pause_started_at = now_ns(&driver);

    sleep(Duration::from_millis(500));

    // While paused, the sim clock must not advance.
    let pause_ended_at = now_ns(&driver);
    assert!(
        pause_ended_at <= pause_started_at + 5_000_000,
        "sim clock advanced during pause: {} -> {} (delta {}ns)",
        pause_started_at,
        pause_ended_at,
        pause_ended_at - pause_started_at,
    );

    // Resume. Immediately measure now_ns. Without the fix, the
    // worker's next tick would jump forward by ~500ms (the pause
    // duration). With the fix, it should advance by at most a few ms
    // (one worker iteration's worth).
    driver.control().set_paused(false);
    // One worker iteration is at most ~1ms (the recv_timeout window),
    // so 30ms of real time is plenty for the worker to run a couple
    // of iterations and surface the bug if present.
    sleep(Duration::from_millis(30));
    let after_resume = now_ns(&driver);

    let post_resume_advance = after_resume - pause_ended_at;
    assert!(
        post_resume_advance < 100_000_000,
        "post-resume jump too large: advanced {}ns in 30ms wall (pause was 500ms; \
         a jump near 500ms means `last_tick` carried the pause window across resume)",
        post_resume_advance,
    );
}
