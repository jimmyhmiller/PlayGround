//! After a drill-in / drill-out roundtrip, the sim+canvas must still
//! behave correctly:
//!   - every sink keeps absorbing its own color (not just other lanes)
//!   - packets continue to flow for a meaningful amount of sim time
//!   - the visual `NodeColors` resource still maps every gadget shim
//!     that participates in palette tagging
//!
//! These are the downstream symptoms of the drill-in/out bug: when the
//! compound shim was re-spawned as a generic `LabeledBox` it lost its
//! Kind/NodeKind, and any per-entity bookkeeping that hung off the
//! original spawn (like `NodeColors`) silently drifted.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::compound::CurrentScope;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;
use flow_bevy::tool::NodeColors;

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

fn first_shim_of(app: &App, kind: Kind) -> flow::NodeId {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .find(|(_, n)| n.is_compound() && n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no compound shim of kind {:?}", kind))
}

fn sinks(app: &App) -> Vec<flow::NodeId> {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", Kind::Sink.label());
    let mut out: Vec<flow::NodeId> = sim
        .nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .collect();
    out.sort_by_key(|nid| shim_filter_match(app, *nid));
    out
}

fn shim_filter_match(app: &App, nid: flow::NodeId) -> i64 {
    let sim = app.world().resource::<FlowSim>();
    let Some(n) = sim.nodes.get(&nid) else { return -1 };
    let prefix = format!("{}::", n.name);
    for inner in sim.nodes.values() {
        if inner.name.starts_with(&prefix) {
            if let Some(Value::Int(i)) = inner.slots.get("match") {
                return *i;
            }
        }
    }
    -1
}

fn sink_count(app: &App, nid: flow::NodeId) -> i64 {
    match app.world().resource::<FlowSim>().read_slot_resolved(nid, "count") {
        Some(Value::Int(i)) => *i,
        _ => -1,
    }
}

fn drill_in_out(app: &mut App, target: flow::NodeId) {
    app.world_mut().resource_mut::<CurrentScope>().0 = Some(target);
    for _ in 0..6 {
        app.update();
    }
    app.world_mut().resource_mut::<CurrentScope>().0 = None;
    for _ in 0..6 {
        app.update();
    }
}

/// After drilling in and out of the Generator, every sink should still
/// keep absorbing packets of its own color. The user reported red (0)
/// in particular stopped being absorbed; this test fails if ANY lane
/// stops counting after the roundtrip.
#[test]
fn after_drill_roundtrip_every_sink_still_absorbs() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    // Warm up a bit so the queue/worker pipeline is steady-state.
    advance_sim_ns(&mut app, 1_000_000_000);

    let gen_id = first_shim_of(&app, Kind::Generator);
    drill_in_out(&mut app, gen_id);

    let sink_ids = sinks(&app);
    let before: Vec<i64> = sink_ids.iter().map(|n| sink_count(&app, *n)).collect();
    advance_sim_ns(&mut app, 5_000_000_000);
    let after: Vec<i64> = sink_ids.iter().map(|n| sink_count(&app, *n)).collect();

    eprintln!("post-drill sink counts: before={:?}, after={:?}", before, after);
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            a > b,
            "lane {} (color {}) stopped absorbing after drill roundtrip: {} -> {}",
            i, shim_filter_match(&app, sink_ids[i]), b, a,
        );
    }
}

/// After drilling in/out of EVERY shim in the example, the NodeColors
/// resource must still have entries for every color-tagged gadget. A
/// missing entry would make the packet renderer fall back to the
/// accent color for everything that gadget emits — visually a global
/// palette drift, which matches the "everything breaks" report.
#[test]
fn drill_every_kind_preserves_node_colors() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    let sim = app.world().resource::<FlowSim>();
    let tagged: Vec<flow::NodeId> = sim
        .nodes
        .iter()
        .filter(|(id, n)| {
            n.is_compound()
                && !n.name.contains("::")
                && flow_bevy::gadgets::has_color_slot(sim, **id)
        })
        .map(|(id, _)| *id)
        .collect();

    let before_colors = app.world().resource::<NodeColors>().0.len();
    assert!(before_colors > 0, "no color-tagged gadgets at start");

    for nid in &tagged {
        drill_in_out(&mut app, *nid);
    }

    let after = app.world().resource::<NodeColors>();
    for nid in &tagged {
        assert!(
            after.0.contains_key(nid),
            "NodeColors entry for {:?} dropped after drill roundtrip",
            nid,
        );
    }
}

/// After a generator drill-roundtrip, the visible-edge count must not
/// drop. Edges anchored on the re-spawned generator would silently
/// disappear from the canvas otherwise — the user's "everything
/// breaks" hand-wave.
#[test]
fn drill_roundtrip_does_not_lose_edges() {
    use flow_bevy::bridge::EntityMaps;
    use flow_bevy::edges::HiddenEdges;
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    let visible = |app: &App| -> usize {
        let sim = app.world().resource::<FlowSim>();
        let hidden = app.world().resource::<HiddenEdges>();
        let maps = app.world().resource::<EntityMaps>();
        sim.edges
            .iter()
            .filter(|(eid, e)| {
                !hidden.set.contains(eid)
                    && maps.node_to_entity.contains_key(&e.from)
                    && maps.node_to_entity.contains_key(&e.to)
                    && e.from != e.to
            })
            .count()
    };

    let before = visible(&app);
    let gen_id = first_shim_of(&app, Kind::Generator);
    drill_in_out(&mut app, gen_id);
    let after = visible(&app);
    assert_eq!(
        before, after,
        "edges visible at top scope changed after drill roundtrip: {} -> {}",
        before, after,
    );
}
