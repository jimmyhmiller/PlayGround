//! Pause must freeze visuals, not just the sim. Before the fix:
//! visual packets used `time.elapsed_secs_f64()` (Bevy wall clock),
//! which keeps ticking through pause. Dots kept flying along edges
//! while the sim sat still. Now every visual system reads
//! `SimClock.visual_now`, which only advances when `paused == false`.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::SimClock;

/// Drive a few real frames so Bevy's wall clock moves. We can't
/// literally sleep in a test, but `app.update()` ticks the Time
/// resource (on headless `test_app_headless` with manual updates,
/// you have to insert delta yourself — but Bevy's `Time` advances
/// by its internal counter regardless). We just need enough frames
/// to prove `visual_now` accumulates vs. stays still.
fn tick_frames(app: &mut App, n: usize) {
    for _ in 0..n {
        app.update();
    }
}

#[test]
fn visual_clock_advances_while_unpaused() {
    let mut app = make_app();
    assert!(!app.world().resource::<SimClock>().paused);
    let t0 = app.world().resource::<SimClock>().visual_now;
    tick_frames(&mut app, 5);
    let t1 = app.world().resource::<SimClock>().visual_now;
    // Should have advanced by some positive amount.
    assert!(t1 > t0, "visual_now should grow while unpaused: {} -> {}", t0, t1);
}

#[test]
fn visual_clock_frozen_while_paused() {
    let mut app = make_app();
    app.world_mut().resource_mut::<SimClock>().paused = true;
    let t0 = app.world().resource::<SimClock>().visual_now;
    tick_frames(&mut app, 20);
    let t1 = app.world().resource::<SimClock>().visual_now;
    assert_eq!(t0, t1, "visual_now must not advance while paused");
}

#[test]
fn visual_clock_resumes_from_freeze_point() {
    let mut app = make_app();
    tick_frames(&mut app, 3);
    let pre_pause = app.world().resource::<SimClock>().visual_now;

    // Pause for a bunch of frames.
    app.world_mut().resource_mut::<SimClock>().paused = true;
    tick_frames(&mut app, 20);
    let during_pause = app.world().resource::<SimClock>().visual_now;
    assert_eq!(pre_pause, during_pause, "no drift during pause");

    // Unpause; visual_now should resume from where it was, not jump
    // forward to catch up with wall clock.
    app.world_mut().resource_mut::<SimClock>().paused = false;
    tick_frames(&mut app, 1);
    let post_resume = app.world().resource::<SimClock>().visual_now;
    // A single frame should add roughly one frame's delta, not 20.
    // Upper bound: assert < (pre_pause + some huge gap) would fail
    // if we jumped. Lower bound: must have moved at all.
    assert!(post_resume > pre_pause, "should resume advancing");
    // Allow up to ~0.1s for a single tick; anything more suggests
    // we leaked wall-clock time accumulated during pause.
    let jump = post_resume - pre_pause;
    assert!(jump < 0.5, "single post-pause tick jumped by {} s (should be tiny)", jump);
}
