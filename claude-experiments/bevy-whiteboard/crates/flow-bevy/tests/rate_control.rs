//! Per-node rate control: dropping a Generator gives it a `period_ns`
//! slot; selecting the node spawns a `RateSlider` + underlying
//! `poster_ui::Slider`; writing to the slider's value pushes a new
//! `period_ns` into the sim.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::{Event, Value};
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::inspector::{RateSlider, RateSliderKind};
use flow_bevy::nodes::Selection;
use flow_bevy::palette::ToolBtn;
use flow_bevy::tool::Tool;
use poster_ui::Slider;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn latest_generator(app: &App) -> flow::NodeId {
    let sim = &app.world().resource::<FlowSim>();
    sim.nodes
        .iter()
        .filter_map(|(id, n)| {
            n.name
                .strip_prefix("Gen_")
                .and_then(|s| s.parse::<u32>().ok())
                .map(|num| (*id, num))
        })
        .max_by_key(|(_, num)| *num)
        .map(|(id, _)| id)
        .expect("no generator dropped")
}

fn period_ns(app: &App, nid: flow::NodeId) -> i64 {
    match app.world().resource::<FlowSim>().nodes[&nid].slots["period_ns"] {
        Value::Int(i) => i,
        _ => panic!("period_ns isn't Int"),
    }
}

/// Find the Slider entity belonging to the given node and set its value.
/// Mirrors what a drag would do — the `Changed<Slider>` handler in
/// inspector pushes the new rate into the sim.
fn set_rate_slider(app: &mut App, nid: flow::NodeId, new_value: f32) {
    let world = app.world_mut();
    let target = {
        let mut q = world.query::<(Entity, &RateSlider)>();
        q.iter(world)
            .find(|(_, rs)| rs.node == nid)
            .map(|(e, _)| e)
            .expect("no RateSlider for that node — is the node selected?")
    };
    let mut q = world.query::<&mut Slider>();
    let mut slider = q
        .get_mut(world, target)
        .expect("slider entity missing");
    slider.value = new_value;
    // Give the `push_rate_slider_to_sim` system a frame to run.
    app.update();
}

/// Select a dropped node by clicking it on the canvas (Select is the
/// resting tool after any drop).
fn select_node_on_canvas(app: &mut App, world_xy: Vec2) {
    simulate_canvas_click(app, world_xy);
}

#[test]
fn generator_has_default_period_on_drop() {
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    let nid = latest_generator(&app);
    // Default 500ms in `gen_generator` (2/s).
    assert_eq!(period_ns(&app, nid), 500_000_000);
}

#[test]
fn selecting_generator_spawns_rate_slider() {
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_generator(&app);

    select_node_on_canvas(&mut app, Vec2::new(-300.0, -200.0));
    assert!(app.world().resource::<Selection>().entity.is_some());

    // After one frame for the inspector rebuild to run.
    app.update();

    let world = app.world_mut();
    let mut q = world.query::<&RateSlider>();
    let found = q.iter(world).find(|rs| rs.node == nid);
    let rs = found.expect("no RateSlider spawned for the selected generator");
    assert_eq!(rs.kind, RateSliderKind::EmitPerSecond);
}

#[test]
fn setting_slider_value_updates_period_ns() {
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_generator(&app);
    select_node_on_canvas(&mut app, Vec2::new(-300.0, -200.0));
    app.update();

    // Default: 10/s → 100ms. Slide to 4/s → 250ms.
    set_rate_slider(&mut app, nid, 4.0);
    let got = period_ns(&app, nid);
    // Allow a 1% margin — slider uses f32, sim uses i64 ns.
    let want = 250_000_000i64;
    assert!(
        (got - want).abs() < want / 100,
        "expected ~{}ns, got {}ns",
        want, got
    );

    // Slide to 20/s → 50ms.
    set_rate_slider(&mut app, nid, 20.0);
    let got = period_ns(&app, nid);
    assert!(
        (got - 50_000_000).abs() < 500_000,
        "expected ~50_000_000 ns, got {}",
        got
    );
}

#[test]
fn faster_rate_produces_more_emissions() {
    let mut app = make_app();

    // Baseline generator at default 10/s.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let baseline = latest_generator(&app);

    // Fast generator at 20/s via slider.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -350.0));
    let fast = latest_generator(&app);
    select_node_on_canvas(&mut app, Vec2::new(-300.0, -350.0));
    app.update();
    set_rate_slider(&mut app, fast, 20.0);

    assert!(period_ns(&app, fast) < period_ns(&app, baseline));

    advance_sim_ns(&mut app, 2_000_000_000);

    let sim = &app.world().resource::<FlowSim>();
    let mut base_count = 0;
    let mut fast_count = 0;
    for ev in sim.log.events.iter() {
        if let Event::PacketEmitted { from, to, .. } = ev {
            if from == to { continue; }
            if *from == baseline { base_count += 1; }
            if *from == fast { fast_count += 1; }
        }
    }
    if base_count == 0 && fast_count == 0 {
        // No downstream edges wired — fall back to the per-node
        // `emitted` counter, which ticks on every self-fired rule.
        let base_slot = match &sim.nodes[&baseline].slots["emitted"] {
            Value::Int(i) => *i,
            _ => 0,
        };
        let fast_slot = match &sim.nodes[&fast].slots["emitted"] {
            Value::Int(i) => *i,
            _ => 0,
        };
        assert!(
            fast_slot > base_slot,
            "fast gen (slot value {}) should emit more than baseline ({})",
            fast_slot, base_slot
        );
    } else {
        assert!(
            fast_count > base_count,
            "fast gen should emit more PacketEmitted events: baseline={}, fast={}",
            base_count, fast_count
        );
    }
}
