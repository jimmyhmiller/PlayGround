//! Connect every kind pairing the whiteboard cares about: Gen→Worker,
//! Gen→Queue, Queue→Worker, Router→Sink, Client→Worker. Each row drops
//! fresh source + target offset from the seeded chain, connects them,
//! and verifies an edge exists in the sim from source to target.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

/// Drop a node of `kind` at `pos` and return the newly-created `NodeId`.
/// Relies on the fact that gadget naming uses a monotonic counter, so the
/// node whose name has the *highest* counter suffix is the one we just
/// placed.
fn drop_and_latest(app: &mut App, kind: Kind, pos: Vec2) -> flow::NodeId {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(kind));
    simulate_canvas_click(app, pos);

    let sim = &app.world().resource::<FlowSim>().sim;
    // Match against the kind's `label()` which is the prefix gadget-spawning
    // uses. Highest numeric suffix = most recent drop.
    let prefix = format!("{}_", kind.label());
    let mut winner: Option<(flow::NodeId, u32)> = None;
    for (id, node) in sim.nodes.iter() {
        if let Some(num) = node
            .name
            .strip_prefix(&prefix)
            .and_then(|s| s.parse::<u32>().ok())
        {
            if winner.map_or(true, |(_, best)| num > best) {
                winner = Some((*id, num));
            }
        }
    }
    winner.expect("couldn't find newly-dropped node").0
}

fn connect_in_ui(app: &mut App, source_pos: Vec2, target_pos: Vec2) {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(app, source_pos);
    simulate_canvas_click(app, target_pos);
}

fn assert_edge(app: &App, from: flow::NodeId, to: flow::NodeId) {
    let sim = &app.world().resource::<FlowSim>().sim;
    let found = sim.edges.values().any(|e| e.from == from && e.to == to);
    assert!(
        found,
        "no sim edge from {:?} to {:?} — edges present: {:?}",
        from,
        to,
        sim.edges
            .values()
            .map(|e| (e.from, e.to))
            .collect::<Vec<_>>()
    );
}

fn pair_case(src_kind: Kind, dst_kind: Kind, src_xy: Vec2, dst_xy: Vec2) {
    let mut app = make_app();
    let src = drop_and_latest(&mut app, src_kind, src_xy);
    let dst = drop_and_latest(&mut app, dst_kind, dst_xy);
    connect_in_ui(&mut app, src_xy, dst_xy);
    assert_edge(&app, src, dst);
}

// Offsets above (y = +250) and below (y = -250) the seeded demo chain so
// hit-testing isn't confused by the seeded nodes sitting near y=0.

#[test]
fn gen_to_worker() {
    pair_case(Kind::Generator, Kind::Worker, Vec2::new(-300.0, 250.0), Vec2::new(300.0, 250.0));
}

#[test]
fn gen_to_queue() {
    pair_case(Kind::Generator, Kind::Queue, Vec2::new(-300.0, 250.0), Vec2::new(300.0, 250.0));
}

#[test]
fn queue_to_worker() {
    pair_case(Kind::Queue, Kind::Worker, Vec2::new(-300.0, -250.0), Vec2::new(300.0, -250.0));
}

#[test]
fn router_to_sink() {
    pair_case(Kind::Router, Kind::Sink, Vec2::new(-300.0, -250.0), Vec2::new(300.0, -250.0));
}

#[test]
fn client_to_worker() {
    pair_case(Kind::Client, Kind::Worker, Vec2::new(-300.0, 250.0), Vec2::new(300.0, 250.0));
}

#[test]
fn worker_to_router() {
    pair_case(Kind::Worker, Kind::Router, Vec2::new(-300.0, -250.0), Vec2::new(300.0, -250.0));
}
