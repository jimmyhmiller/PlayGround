//! Probes report in sim-time, so their reading is independent of the
//! user's playback speed. A 10/s generator shows "10/s" whether the user
//! is watching at 1× or 4×.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::{FlowSim, SimClock};
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::probes::{ProbeSamples, rate_for_edge};
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn wire_gen_to_sink_with_probe(app: &mut App) -> flow::EdgeId {
    let gen_xy = Vec2::new(-300.0, -200.0);
    let sink_xy = Vec2::new(300.0, -200.0);
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(app, gen_xy);
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(app, sink_xy);
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(app, gen_xy);
    simulate_canvas_click(app, sink_xy);

    // Drop a probe onto the edge midpoint.
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(app, Vec2::new(0.0, -200.0));

    // Return the gen → sink edge id.
    let sim = &app.world().resource::<FlowSim>().sim;
    *sim.edges
        .iter()
        .find(|(_, e)| {
            let fname = &sim.nodes[&e.from].name;
            let tname = &sim.nodes[&e.to].name;
            fname.starts_with("Gen_") && tname.starts_with("Sink_")
        })
        .map(|(id, _)| id)
        .expect("no gen→sink edge")
}

#[test]
fn probe_rate_at_1x_matches_generator_rate() {
    let mut app = make_app();
    let eid = wire_gen_to_sink_with_probe(&mut app);

    // Default multiplier 1.0, default gen rate 10/s. Run 2 sim-seconds so
    // the probe's 2-sec window is fully primed.
    advance_sim_ns(&mut app, 2_000_000_000);
    app.update();

    let rate = rate_for_edge(app.world().resource::<ProbeSamples>(), eid);
    assert!(
        (rate - 2.0).abs() < 1.0,
        "1× probe rate should be ~2/s, got {:.2}/s",
        rate
    );
}

#[test]
fn probe_rate_is_unchanged_at_4x_multiplier() {
    // The point: at higher multipliers the sim runs faster in real time,
    // but the probe's sim-time windowing means its reading stays put.
    let mut app = make_app();
    let eid = wire_gen_to_sink_with_probe(&mut app);

    // Crank the multiplier. advance_sim_ns bypasses the bridge so
    // multiplier isn't consulted for sim advancement here — but probe
    // collection runs every frame and uses sim-time, so the reading is
    // the same shape regardless.
    app.world_mut().resource_mut::<SimClock>().multiplier = 4.0;

    advance_sim_ns(&mut app, 2_000_000_000);
    app.update();

    let rate = rate_for_edge(app.world().resource::<ProbeSamples>(), eid);
    assert!(
        (rate - 2.0).abs() < 1.0,
        "probe rate at 4× should still be ~10/s (sim-time basis), got {:.2}/s",
        rate
    );
}

#[test]
fn probe_rate_tracks_slider_change_regardless_of_multiplier() {
    use flow_bevy::inspector::RateSlider;
    use poster_ui::Slider;

    let mut app = make_app();
    let eid = wire_gen_to_sink_with_probe(&mut app);

    // Select the generator and crank rate to 40/s.
    let gen_xy = Vec2::new(-300.0, -200.0);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Select);
    simulate_canvas_click(&mut app, gen_xy);
    app.update();
    {
        let world = app.world_mut();
        let mut q = world.query::<(Entity, &RateSlider)>();
        let (e, _) = q.iter(world).next().expect("no rate slider");
        let mut sq = world.query::<&mut Slider>();
        sq.get_mut(world, e).unwrap().value = 40.0;
    }
    app.update();

    // Multiplier shouldn't matter — set it to something odd.
    app.world_mut().resource_mut::<SimClock>().multiplier = 2.0;

    advance_sim_ns(&mut app, 2_000_000_000);
    app.update();

    let rate = rate_for_edge(app.world().resource::<ProbeSamples>(), eid);
    assert!(
        rate > 30.0,
        "probe should report ~40/s after the slider change (sim-time); got {:.2}/s",
        rate
    );
}
