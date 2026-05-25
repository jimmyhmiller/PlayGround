//! End-to-end colour tests.
//!
//! Unit tests in `edges.rs::color_tests` cover the pure `packet_color`
//! resolver. These tests layer on top: they build a real Bevy graph,
//! tick the sim, and assert every `PacketEmitted` the engine recorded
//! would render in the expected theme colour.
//!
//! Each test builds its own topology — nothing depends on the opening
//! demo. Change the demo shape without breaking these.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, build_pull_chain, make_app};
use flow::Event;
use flow_bevy::bridge::FlowSim;
use flow_bevy::edges::{packet_color, payload_slot};
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::{ColorSwatch, ToolBtn};
use flow_bevy::tool::{NodeColors, Tool};
use poster_ui::Theme;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

/// Resolve the colour that the render pipeline would assign to a given
/// `PacketEmitted` event in the supplied app.
fn resolved_color_of(app: &App, from: flow::NodeId, payload: &flow::Value) -> Color {
    let node_colors = app.world().resource::<NodeColors>();
    let theme = app.world().resource::<Theme>();
    packet_color(
        payload,
        node_colors.0.get(&from).copied(),
        &theme.data,
        theme.accent,
    )
}

fn theme_slot(app: &App, slot: usize) -> Color {
    app.world().resource::<Theme>().data[slot]
}

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

fn drop_at(app: &mut App, slot: usize, kind: Kind, pos: Vec2) -> (flow::NodeId, Vec2) {
    click_by_marker::<ColorSwatch, _>(app, |s| s.0 == slot);
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(kind));
    simulate_canvas_click(app, pos);
    (latest_of_kind(app, kind), pos)
}

fn drop_router_at(app: &mut App, pos: Vec2) -> (flow::NodeId, Vec2) {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(Kind::Router));
    simulate_canvas_click(app, pos);
    (latest_of_kind(app, Kind::Router), pos)
}

fn connect(app: &mut App, from_xy: Vec2, to_xy: Vec2) {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(app, from_xy);
    simulate_canvas_click(app, to_xy);
}

// -----------------------------------------------------------------------------
// Single-stream chain: every data packet takes the colour of its slot.
// -----------------------------------------------------------------------------

#[test]
fn single_stream_data_packets_resolve_to_stream_color() {
    // Slot 0 (red) chain: Gen → Queue ← Worker → Sink. Every
    // PacketEmitted carrying `packet(Int(0))` or `req(Int(0))` must
    // resolve to theme.data[0].
    let mut app = make_app();
    let _ = build_pull_chain(&mut app);
    advance_sim_ns(&mut app, 2_000_000_000);

    let red = theme_slot(&app, 0);
    let mut data_events = 0usize;
    let events = app.world().resource::<FlowSim>().log.events.clone();
    for ev in events.iter() {
        let Event::PacketEmitted { from, payload, .. } = ev else { continue; };
        if payload_slot(payload).is_none() { continue; }
        data_events += 1;
        let got = resolved_color_of(&app, *from, payload);
        assert_eq!(
            got, red,
            "data packet from {:?} ({:?}) resolved to {:?}, want red",
            from, payload, got,
        );
    }
    assert!(data_events > 0, "expected at least one data packet in 2s run");
}

#[test]
fn single_stream_control_packets_fall_back_to_emitter_color() {
    // `pull(self)` / `wake(nil)` / `tick(nil)` have no slot tag. Their
    // render colour comes from NodeColors of the emitter. In a slot-0
    // chain every emitter is red, so every control packet is red too.
    let mut app = make_app();
    let _ = build_pull_chain(&mut app);
    advance_sim_ns(&mut app, 1_000_000_000);

    let red = theme_slot(&app, 0);
    let mut control_events = 0usize;
    let events = app.world().resource::<FlowSim>().log.events.clone();
    for ev in events.iter() {
        let Event::PacketEmitted { from, payload, .. } = ev else { continue; };
        if payload_slot(payload).is_some() { continue; }
        control_events += 1;
        let got = resolved_color_of(&app, *from, payload);
        assert_eq!(got, red, "control payload {:?} from {:?} resolved to {:?}, want red", payload, from, got);
    }
    assert!(control_events > 0, "expected control payloads in chain");
}

// -----------------------------------------------------------------------------
// Router: two colour streams, each retains its own colour end-to-end.
// -----------------------------------------------------------------------------

#[test]
#[ignore = "Composite migration: pattern-matches monolithic node shape (event-log from/to or direct slot access on the shim). Re-enable after rewriting to use Sim::compound_outermost / read_slot_resolved."]
fn router_preserves_per_stream_colors_end_to_end() {
    let mut app = make_app();

    let (gen_red,    gen_red_xy)    = drop_at(&mut app, 0, Kind::Generator, Vec2::new(-500.0, -100.0));
    let (gen_yel,    gen_yel_xy)    = drop_at(&mut app, 1, Kind::Generator, Vec2::new(-500.0, -300.0));
    let (_router,    router_xy)     = drop_router_at(&mut app, Vec2::new(-100.0, -200.0));
    let (queue_red,  queue_red_xy)  = drop_at(&mut app, 0, Kind::Queue, Vec2::new(300.0, -100.0));
    let (queue_yel,  queue_yel_xy)  = drop_at(&mut app, 1, Kind::Queue, Vec2::new(300.0, -300.0));

    connect(&mut app, gen_red_xy,   router_xy);
    connect(&mut app, gen_yel_xy,   router_xy);
    connect(&mut app, router_xy,    queue_red_xy);
    connect(&mut app, router_xy,    queue_yel_xy);

    advance_sim_ns(&mut app, 3_000_000_000);

    let red    = theme_slot(&app, 0);
    let yellow = theme_slot(&app, 1);

    let mut red_seen = 0;
    let mut yel_seen = 0;
    let events = app.world().resource::<FlowSim>().log.events.clone();
    for ev in events.iter() {
        let Event::PacketEmitted { from, to, payload, .. } = ev else { continue; };
        let Some(slot) = payload_slot(payload) else { continue; };

        let resolved = resolved_color_of(&app, *from, payload);
        let expected = theme_slot(&app, slot);
        assert_eq!(
            resolved, expected,
            "payload slot {} (from={:?}, to={:?}) resolved to {:?}, want {:?}",
            slot, from, to, resolved, expected,
        );

        if *to == queue_red {
            red_seen += 1;
            assert_eq!(resolved, red, "packet to queue_red resolved to {:?}", resolved);
        }
        if *to == queue_yel {
            yel_seen += 1;
            assert_eq!(resolved, yellow, "packet to queue_yel resolved to {:?}", resolved);
        }

        if *from == gen_red { assert_eq!(resolved, red); }
        if *from == gen_yel { assert_eq!(resolved, yellow); }
    }
    assert!(red_seen > 0, "no packets reached queue_red");
    assert!(yel_seen > 0, "no packets reached queue_yel");
}

// -----------------------------------------------------------------------------
// Indicator-dot consistency: NodeColors for a node spawned at slot N
// always equals theme.data[N]. This is what keeps the on-node indicator
// dot in sync with the renderer's emitter fallback.
// -----------------------------------------------------------------------------

#[test]
fn node_colors_match_slot_index_across_palette() {
    let mut app = make_app();
    // Drop one of every non-router kind at every palette slot.
    let kinds = [Kind::Generator, Kind::Client, Kind::Queue, Kind::Worker, Kind::Sink];
    let palette_size = app.world().resource::<Theme>().data.len();

    let mut expected: Vec<(flow::NodeId, Color)> = Vec::new();
    let mut y = -300.0;
    for slot in 0..palette_size {
        let expected_color = theme_slot(&app, slot);
        for &kind in &kinds {
            let (id, _) = drop_at(&mut app, slot, kind, Vec2::new(-500.0 + 60.0 * slot as f32, y));
            expected.push((id, expected_color));
            y += 60.0;
            if y > 300.0 { y = -300.0; }
        }
    }

    let node_colors = app.world().resource::<NodeColors>();
    for (id, want) in expected {
        let got = node_colors.0.get(&id).copied()
            .unwrap_or_else(|| panic!("NodeColors missing entry for {:?}", id));
        assert_eq!(got, want, "node {:?}: NodeColors {:?} != theme.data slot color {:?}", id, got, want);
    }
}
