//! End-to-end check: a slider-driven rate change must actually alter the
//! live emission rate, not just the `period_ns` slot. Counts real
//! `PacketEmitted` events in a sim window before and after a slider move.
//!
//! The earlier `rate_control::faster_rate_produces_more_emissions` test
//! compared two *different* generators at two different rates. This test
//! compares the *same* generator before and after a slider drag, which is
//! what a user experiences in the app.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Event;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::inspector::RateSlider;
use flow_bevy::palette::ToolBtn;
use flow_bevy::tool::Tool;
use poster_ui::Slider;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn latest_of_kind(app: &App, kind: Kind) -> flow::NodeId {
    let sim = &app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .filter_map(|(id, n)| {
            n.name
                .strip_prefix(&prefix)
                .and_then(|s| s.parse::<u32>().ok())
                .map(|num| (*id, num))
        })
        .max_by_key(|(_, num)| *num)
        .map(|(id, _)| id)
        .expect("no node of that kind")
}

fn set_rate_via_slider(app: &mut App, nid: flow::NodeId, value: f32) {
    let world = app.world_mut();
    let target = {
        let mut q = world.query::<(Entity, &RateSlider)>();
        q.iter(world)
            .find(|(_, rs)| rs.node == nid)
            .map(|(e, _)| e)
            .expect("no RateSlider — did you select the node?")
    };
    let mut q = world.query::<&mut Slider>();
    q.get_mut(world, target).expect("slider missing").value = value;
    app.update();
}

/// Count `PacketEmitted` events where the emitter is `nid` and the target
/// isn't itself (so self-loop ticks don't inflate the count). Reads the
/// sim's event log in place.
fn count_emissions_from(app: &App, nid: flow::NodeId) -> usize {
    let sim = &app.world().resource::<FlowSim>();
    sim.log
        .events
        .iter()
        .filter(|ev| {
            if let Event::PacketEmitted { from, to, .. } = ev {
                *from == nid && *from != *to
            } else {
                false
            }
        })
        .count()
}

#[test]
fn slider_change_speeds_up_live_emissions() {
    let mut app = make_app();

    // Drop a generator wired to a sink so PacketEmitted events actually
    // land somewhere (self-loop ticks don't show up in our count).
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let gen_nid = latest_of_kind(&app, Kind::Generator);
    let gen_xy = Vec2::new(-300.0, -200.0);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(300.0, -200.0));
    let sink_xy = Vec2::new(300.0, -200.0);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(&mut app, gen_xy);
    simulate_canvas_click(&mut app, sink_xy);

    // Run 3 sim-seconds at the default 2/s rate. Expect ~6 emissions.
    advance_sim_ns(&mut app, 3_000_000_000);
    let baseline_window = count_emissions_from(&app, gen_nid);
    assert!(
        (3..=9).contains(&baseline_window),
        "baseline 3s window should be ~6 emissions, got {}",
        baseline_window
    );

    // Back to Select tool so the next canvas click picks the node for
    // the inspector rather than restarting a Connect gesture.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Select);
    simulate_canvas_click(&mut app, gen_xy);
    app.update();
    set_rate_via_slider(&mut app, gen_nid, 40.0);

    // Snapshot count right after the slider change so we can measure the
    // fresh window independently.
    let cutoff = count_emissions_from(&app, gen_nid);

    // Run another 1 sim-second. At 40/s the window should have ~40 emissions.
    advance_sim_ns(&mut app, 1_000_000_000);
    let after_window = count_emissions_from(&app, gen_nid) - cutoff;

    assert!(
        after_window >= baseline_window * 2,
        "slider-driven rate bump should double (or more) the emission rate. \
         baseline 1s={}, post-slider 1s={}",
        baseline_window, after_window
    );
    // Loose expectation: the first cycle after the slider change is
    // still being delivered on the old period, so the effective rate
    // over the 1-second window mixes old+new. 15 is well above the
    // baseline's ~6 and accounts for that.
    assert!(
        after_window >= 15,
        "post-slider window should be noticeably higher (~40/s steady-state), got {}",
        after_window
    );
}

#[test]
#[ignore = "Composite migration: pattern-matches monolithic node shape (event-log from/to or direct slot access on the shim). Re-enable after rewriting to use Sim::compound_outermost / read_slot_resolved."]
fn slider_change_slows_down_live_emissions() {
    let mut app = make_app();

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let gen_nid = latest_of_kind(&app, Kind::Generator);
    let gen_xy = Vec2::new(-300.0, -200.0);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(300.0, -200.0));
    let sink_xy = Vec2::new(300.0, -200.0);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(&mut app, gen_xy);
    simulate_canvas_click(&mut app, sink_xy);

    // Baseline at a higher rate first: bump gen to 10/s so there's
    // meaningful headroom for "slower" to show up. Without this, default
    // 2/s barely produces any packets in a 1s window.
    {
        let world = app.world_mut();
        let mut flow = world.resource_mut::<FlowSim>();
        flow.nodes.get_mut(&gen_nid).unwrap()
            .slots.insert("period_ns".into(), flow::Value::Int(100_000_000));
    }
    // Warm-up: let the previously-scheduled 500ms tick land, then a
    // second window to sample the bumped rate.
    advance_sim_ns(&mut app, 500_000_000);
    let warm = count_emissions_from(&app, gen_nid);
    advance_sim_ns(&mut app, 1_000_000_000);
    let baseline_window = count_emissions_from(&app, gen_nid) - warm;
    assert!(
        baseline_window >= 5,
        "baseline should accumulate at least 5 emissions in a 1s sample window at 10/s, got {}",
        baseline_window
    );

    // Back to Select so the canvas click targets the generator for the
    // inspector. Then drop the rate via the slider to 2/s.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Select);
    simulate_canvas_click(&mut app, gen_xy);
    app.update();
    set_rate_via_slider(&mut app, gen_nid, 2.0);

    let cutoff = count_emissions_from(&app, gen_nid);

    // Run 1s at the slower rate → expect ~2 emissions.
    advance_sim_ns(&mut app, 1_000_000_000);
    let after_window = count_emissions_from(&app, gen_nid) - cutoff;

    assert!(
        after_window <= baseline_window / 2,
        "slower slider rate should produce fewer emissions. \
         baseline 1s={}, post-slider 1s={}",
        baseline_window, after_window
    );
}
