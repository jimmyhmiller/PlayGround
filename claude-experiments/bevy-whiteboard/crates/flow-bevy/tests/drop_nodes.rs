//! Exercise the palette → canvas drop pathway for each node kind.
//!
//! Each test: click the palette button for a kind (sets `ActiveTool`), then
//! simulate a canvas click at world (0, 0) (dispatches `handle_drop`), then
//! assert the sim gained exactly one node of that kind.

mod common;

use bevy::prelude::*;
use common::{make_app, node_count};
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

/// Count *outer* sim nodes whose name matches the `{kind.label()}_N`
/// pattern. The seed_demo_graph already created one Gen_, one Queue_,
/// one Worker_, one Sink_ — so delta against the baseline is what we
/// check. Composite inner nodes (e.g. `Worker_2::L`, `Worker_2::F`)
/// also start with the kind prefix and would inflate the count — we
/// filter them out with the `::` discriminator.
fn count_by_label_prefix(app: &App, prefix: &str) -> usize {
    app.world()
        .resource::<FlowSim>()
        .nodes
        .values()
        .filter(|n| n.name.starts_with(prefix) && !n.name.contains("::"))
        .count()
}

fn drop_and_assert(kind: Kind, prefix: &str) {
    let mut app = make_app();
    let before_total = node_count(&app);
    let before_kind = count_by_label_prefix(&app, prefix);

    let clicked = click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(kind));
    assert!(clicked, "no palette button found for {:?}", kind);

    // Canvas click at world origin — squarely inside the viewport, nowhere
    // near the right-side palette. `pointer_over_ui` sees all Interactions
    // at None, so `handle_drop` fires.
    simulate_canvas_click(&mut app, Vec2::new(0.0, 0.0));

    let after_total = node_count(&app);
    let after_kind = count_by_label_prefix(&app, prefix);
    // Composites spawn N inner nodes plus the port shim, so the total
    // delta is >1. The kind-prefix delta is still exactly 1 — only the
    // shim has the user-facing `<Kind.label()>_N` name.
    assert!(
        after_total > before_total,
        "expected at least one new node after dropping {:?}, got delta {}",
        kind,
        after_total as i64 - before_total as i64
    );
    assert_eq!(
        after_kind,
        before_kind + 1,
        "new node isn't labelled with {:?} prefix",
        prefix
    );
}

#[test]
fn drop_generator() { drop_and_assert(Kind::Generator, "Gen_"); }

#[test]
fn drop_client() { drop_and_assert(Kind::Client, "Client_"); }

#[test]
fn drop_worker() { drop_and_assert(Kind::Worker, "Worker_"); }

#[test]
fn drop_router() { drop_and_assert(Kind::Router, "Router_"); }

#[test]
fn drop_queue() { drop_and_assert(Kind::Queue, "Queue_"); }

#[test]
fn drop_sink() { drop_and_assert(Kind::Sink, "Sink_"); }

#[test]
fn drop_then_active_resets_to_select() {
    // `handle_drop` flips back to Select after a single placement so the
    // user can't accidentally spam nodes on every click.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Worker));
    simulate_canvas_click(&mut app, Vec2::new(0.0, 0.0));
    assert_eq!(
        app.world().resource::<flow_bevy::tool::ActiveTool>().0,
        Tool::Select,
        "ActiveTool should reset to Select after a drop"
    );
}
