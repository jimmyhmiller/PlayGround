//! UI-level tests for the rewind controls. Unlike `rewind_e2e`
//! which calls `driver.rewind()` directly, these go through the
//! actual HUD button / slider systems by setting `Interaction`
//! state on the spawned entities — same code path the user
//! triggers with mouse input.
//!
//! Coverage goals:
//!   - `↺0` (rewind-zero): lands at t=0, preserves topology.
//!   - `«` (rewind-step): each click walks back one capture
//!      marker; markers stay densely packed all the way to t=0.
//!   - Slider drag: rapid value changes don't lock up the worker
//!      or the per-frame budget.
//!   - Long-distance drag: dragging from a multi-minute session
//!      back to t=0 stays interactive.

mod common;

use std::time::Instant;

use bevy::prelude::*;
use bevy::ecs::message::Messages;

use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::{FlowSim, SimClock};
use flow_bevy::gadgets::Kind;
use flow_bevy::hud::{HudRewindStepBtn, HudRewindZeroBtn};
use flow_bevy::sim_driver::SimSnapshotRes;

fn now_ns(app: &App) -> u64 {
    app.world().resource::<FlowSim>().now_ns
}

/// Press the entity matching `Marker` once via the real Bevy
/// input pipeline (cursor move + MouseButton press/release).
/// Wraps `poster_ui::testing::click_by_marker`, which does the
/// full cursor-position + ButtonInput dance that the user's mouse
/// would do — `Interaction::Pressed` ends up set by Bevy's own
/// `ui_focus_system`, then handlers fire, then it's released.
fn press<Marker: Component>(app: &mut App) {
    let clicked = poster_ui::testing::click_by_marker::<Marker, _>(app, |_| true);
    assert!(
        clicked,
        "press<{}>: no entity matched the marker",
        std::any::type_name::<Marker>(),
    );
}

/// Issue a rewind to `target_ns` the way the rewind slider would —
/// pause the clock, then ship the rewind through the driver. Bevy
/// 0.18 headless tests can't easily drive the slider widget's
/// drag-while-held protocol (which needs cursor + ButtonInput +
/// ComputedNode in physical pixels), so we exercise the same
/// codepath that `push_rewind_slider` reaches: pause + rewind. Any
/// regression in the worker's rewind handling shows up here
/// regardless.
fn slider_rewind_to(app: &mut App, target_ns: u64) {
    let world = app.world_mut();
    world.resource_mut::<SimClock>().paused = true;
    world.resource_mut::<FlowSim>().0.rewind(target_ns);
}

fn spawn_dense_topology(app: &mut App) {
    use flow::Value;
    for i in 0..8 {
        let client = common::spawn_node(&mut *app, Kind::Client, i % 3, &format!("Cli_{}", i));
        let worker = common::spawn_node(&mut *app, Kind::Worker, i % 3, &format!("Wkr_{}", i));
        {
            let world = app.world_mut();
            let mut driver = world.resource_mut::<FlowSim>();
            driver.0.with_sim_mut(move |sim| {
                if let Some(n) = sim.nodes.get_mut(&client) {
                    n.slots.insert("period_ns".into(), Value::Int(50_000_000));
                }
                if let Some(n) = sim.nodes.get_mut(&worker) {
                    n.slots.insert("service_ns".into(), Value::Int(10_000_000));
                }
            });
        }
        common::wire(&mut *app, client, Kind::Client, worker, Kind::Worker);
    }
}

/// Load an example via the LoadExample message and let the handler
/// run. Mirrors how the user actually triggers a load.
fn fire_load_example(app: &mut App, example: flow_bevy::examples::Example) {
    app.world_mut()
        .resource_mut::<Messages<flow_bevy::examples::LoadExample>>()
        .write(flow_bevy::examples::LoadExample(example));
    // Two frames: one to dispatch the message, one to settle.
    app.update();
    app.update();
}

// ─────────────────────────────────────────────────────────────────
// `↺0` (rewind-zero) button
// ─────────────────────────────────────────────────────────────────

#[test]
fn rewind_zero_button_after_load_example_lands_at_zero_with_topology() {
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);

    let nodes_before = app.world().resource::<FlowSim>().nodes.len();
    assert!(
        nodes_before > 0,
        "ClientWorker should have populated nodes"
    );

    advance_sim_ns(&mut app, 5_000_000_000);
    app.update();
    assert!(now_ns(&app) >= 5_000_000_000);

    press::<HudRewindZeroBtn>(&mut app);

    assert_eq!(
        now_ns(&app), 0,
        "↺0 button should land sim at t=0",
    );
    let nodes_after = app.world().resource::<FlowSim>().nodes.len();
    assert_eq!(
        nodes_after, nodes_before,
        "↺0 must not erase loaded topology — had {} nodes, got {}",
        nodes_before, nodes_after,
    );
    assert!(
        app.world().resource::<SimClock>().paused,
        "↺0 should pause the sim so it doesn't immediately advance again",
    );
}

#[test]
fn rewind_zero_button_palette_only_session_lands_at_zero() {
    // No LoadExample — palette-style: spawn nodes via with_sim_mut,
    // run, click ↺0. The bridge's startup anchor is empty (no
    // topology), so this exercises the snap-fallback ladder.
    let mut app = make_app();
    spawn_dense_topology(&mut app);
    advance_sim_ns(&mut app, 5_000_000_000);
    app.update();

    press::<HudRewindZeroBtn>(&mut app);

    assert_eq!(
        now_ns(&app), 0,
        "↺0 should still land at 0 even without an explicit LoadExample",
    );
}

// ─────────────────────────────────────────────────────────────────
// `«` (continuous reverse-animation) button
// ─────────────────────────────────────────────────────────────────

#[test]
fn rewind_step_button_starts_reverse_play() {
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..20 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    assert_eq!(
        app.world().resource::<SimClock>().reverse_play_rate, 0.0,
        "reverse-play should be off before first click",
    );
    press::<HudRewindStepBtn>(&mut app);
    let clock = app.world().resource::<SimClock>();
    assert!(
        clock.reverse_play_rate > 0.0,
        "« should turn reverse-play on; got rate={}",
        clock.reverse_play_rate,
    );
    assert!(
        clock.paused,
        "« should set the pause flag so forward advance stops",
    );
}

#[test]
fn rewind_step_button_toggles_off() {
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..20 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    press::<HudRewindStepBtn>(&mut app);
    assert!(app.world().resource::<SimClock>().reverse_play_rate > 0.0);
    press::<HudRewindStepBtn>(&mut app);
    assert_eq!(
        app.world().resource::<SimClock>().reverse_play_rate, 0.0,
        "second « click should toggle reverse-play off",
    );
}

#[test]
fn rewind_step_button_auto_stops_at_zero() {
    // The auto-stop watches sim_now in the published snapshot —
    // when it hits 0, reverse-play turns off. We test this
    // observation directly: turn on reverse-play, jump sim to 0,
    // pump a frame so the auto-stop system runs.
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..4 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    press::<HudRewindStepBtn>(&mut app);
    assert!(app.world().resource::<SimClock>().reverse_play_rate > 0.0);

    // Jump sim to 0 directly. (In the live app, the bridge's
    // per-frame `driver.rewind(now - dt)` walks sim_now down to
    // 0 over many frames — same end state.)
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(0);
    }
    app.update();
    app.update();

    assert_eq!(now_ns(&app), 0, "sim should be at 0 for the auto-stop check");
    assert_eq!(
        app.world().resource::<SimClock>().reverse_play_rate, 0.0,
        "auto-stop should clear reverse_play_rate once sim hits 0",
    );
}

#[test]
fn rewind_step_button_walks_sim_backward_per_frame() {
    // Verify the bridge's reverse-play stepper actually decreases
    // sim_now over multiple frames when reverse-play is on. This
    // is the path that produces the user-facing animation.
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..20 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();
    let initial = now_ns(&app);

    press::<HudRewindStepBtn>(&mut app);

    // Pump frames; sim_now should monotonically decrease.
    let mut prev = initial;
    let mut moved = false;
    for _ in 0..30 {
        app.update();
        let now = now_ns(&app);
        assert!(
            now <= prev,
            "reverse-play must not advance sim forward: {} → {}",
            prev, now,
        );
        if now < prev {
            moved = true;
        }
        prev = now;
    }
    assert!(
        moved,
        "after 30 frames of reverse-play, sim_now didn't decrease at all (started at {}, ended at {})",
        initial, prev,
    );
}

// ─────────────────────────────────────────────────────────────────
// Slider drag
// ─────────────────────────────────────────────────────────────────

#[test]
fn slider_drag_through_many_positions_completes_quickly() {
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..240 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();
    let final_ns = now_ns(&app);
    assert!(final_ns >= 60_000_000_000);

    // 60 sweep stops from current sim time toward 0. Each stop
    // mirrors one frame where the user has dragged the cursor.
    let start = Instant::now();
    for i in 0..60 {
        let t = ((final_ns as f64) * (1.0 - i as f64 / 60.0)) as u64;
        slider_rewind_to(&mut app, t);
        app.update();
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 3_000,
        "60-step slider drag took {}ms — would feel like a lockup",
        elapsed.as_millis(),
    );
}

#[test]
fn slider_drag_long_distance_no_lockup() {
    // 5-minute session, drag from the tail back to near zero. The
    // ring's default 64 entries × 250ms cadence covers only ~16s
    // so most of the drag is targeting times outside the ring —
    // each rewind has to fall through to the t=0 anchor and run
    // sim forward potentially minutes. This is the exact scenario
    // the user reported as a UI lockup.
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..1200 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();
    let final_ns = now_ns(&app);
    assert!(final_ns >= 300_000_000_000);

    let start = Instant::now();
    for i in 0..30 {
        let frac = (1.0 - i as f64 / 30.0).max(0.0);
        let t = ((final_ns as f64) * frac) as u64;
        slider_rewind_to(&mut app, t);
        app.update();
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5_000,
        "long-distance drag took {}ms — would lock up the UI",
        elapsed.as_millis(),
    );
    let landed_ns = now_ns(&app);
    assert!(
        landed_ns < 30_000_000_000,
        "drag should have landed near 0; landed at {}ns ({}ms)",
        landed_ns, landed_ns / 1_000_000,
    );
}

// ─────────────────────────────────────────────────────────────────
// Marker density regression guard
// ─────────────────────────────────────────────────────────────────

#[test]
fn markers_after_load_example_have_no_huge_gap() {
    // Even though `«` no longer marker-steps, the slider still uses
    // the marker list to render scrub-strip ticks and the rewind
    // strategies use the snapshot ring it backs. A multi-second
    // gap would mean half the scrub strip is unscrubbable.
    let mut app = make_app();
    fire_load_example(&mut app, flow_bevy::examples::Example::ClientWorker);
    for _ in 0..8 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    let markers = app.world()
        .resource::<SimSnapshotRes>().0
        .rewind_markers_ns
        .clone();
    let max_gap_ns = markers
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0);

    assert!(
        max_gap_ns < 500_000_000,
        "max gap between markers {}ns ({}ms); markers={:?}",
        max_gap_ns, max_gap_ns / 1_000_000, markers,
    );
}
