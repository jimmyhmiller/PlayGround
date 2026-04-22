//! Pull is one-directional.
//!
//!  * Drawing **Worker → Queue** (the pull-arrow direction) wires up
//!    pull semantics: worker.upstream := Queue, plus a hidden reverse
//!    data edge so the queue can emit back to the worker.
//!  * Drawing **Queue → Worker** creates only an ordinary edge — no
//!    upstream slot wiring, nothing flows.
//!
//! Both tests build their graph from scratch through palette clicks so
//! they exercise the real drop + connect UI path.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow::Value;
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::edges::HiddenEdges;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn latest_of_kind(app: &App, kind: Kind) -> flow::NodeId {
    let sim = &app.world().resource::<FlowSim>().sim;
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

fn drop_at(app: &mut App, kind: Kind, pos: Vec2) -> (flow::NodeId, Vec2) {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(kind));
    simulate_canvas_click(app, pos);
    (latest_of_kind(app, kind), pos)
}

fn connect(app: &mut App, from_xy: Vec2, to_xy: Vec2) {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(app, from_xy);
    simulate_canvas_click(app, to_xy);
}

fn world_pos_of(app: &App, nid: flow::NodeId) -> Vec2 {
    let maps = app.world().resource::<EntityMaps>();
    let ent = *maps.node_to_entity.get(&nid).unwrap();
    app.world().get::<Transform>(ent).unwrap().translation.truncate()
}

#[test]
fn drawing_worker_to_queue_creates_pull() {
    let mut app = make_app();
    let (queue, queue_xy)   = drop_at(&mut app, Kind::Queue,  Vec2::new(-200.0, 0.0));
    let (worker, worker_xy) = drop_at(&mut app, Kind::Worker, Vec2::new( 200.0, 0.0));

    // Pull direction: Worker → Queue.
    connect(&mut app, worker_xy, queue_xy);
    let _ = world_pos_of(&app, queue); // keep helper referenced

    let sim = &app.world().resource::<FlowSim>().sim;
    assert_eq!(
        sim.nodes[&worker].slots["upstream"],
        Value::NodeRef(queue),
        "worker.upstream should be set to the queue after a pull-draw",
    );

    // Exactly one VISIBLE edge between the pair (the one the user drew).
    // A second reverse-direction edge exists for data delivery, but it's
    // marked hidden so it doesn't visually duplicate the arrow.
    let hidden = &app.world().resource::<HiddenEdges>().set;
    let visible = sim
        .edges
        .iter()
        .filter(|(eid, e)| {
            !hidden.contains(eid)
                && ((e.from == queue && e.to == worker) || (e.from == worker && e.to == queue))
        })
        .count();
    assert_eq!(visible, 1, "exactly one visible edge between the pair");

    // And that visible edge runs in the pull direction (Worker → Queue).
    let visible_dir = sim
        .edges
        .iter()
        .filter(|(eid, e)| {
            !hidden.contains(eid)
                && ((e.from == worker && e.to == queue) || (e.from == queue && e.to == worker))
        })
        .next()
        .map(|(_, e)| if e.from == worker && e.to == queue { "pull" } else { "data" });
    assert_eq!(visible_dir, Some("pull"));
}

#[test]
fn drawing_queue_to_worker_does_not_establish_pull() {
    let mut app = make_app();
    let (_queue, queue_xy)   = drop_at(&mut app, Kind::Queue,  Vec2::new(-200.0, 0.0));
    let (worker, worker_xy)  = drop_at(&mut app, Kind::Worker, Vec2::new( 200.0, 0.0));

    // Wrong direction: Queue → Worker. Shouldn't establish pull.
    connect(&mut app, queue_xy, worker_xy);

    let sim = &app.world().resource::<FlowSim>().sim;
    assert_eq!(
        sim.nodes[&worker].slots["upstream"],
        Value::Nil,
        "worker.upstream should remain nil when the user drew the non-pull direction",
    );
}
