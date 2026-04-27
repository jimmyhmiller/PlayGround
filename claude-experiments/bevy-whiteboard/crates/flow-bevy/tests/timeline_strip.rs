//! Bottom-of-canvas timeline strip — display-only, no authoring.
//! Renders `sim.timeline` events as markers and tracks the now-cursor
//! against `sim.now_ns`. The strip hides itself when the timeline is
//! empty so plain whiteboards (no scenario) don't get unwanted chrome.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::timeline::TimelineStripRoot;

fn strip_visibility(app: &mut App) -> Visibility {
    let world = app.world_mut();
    let mut q = world.query_filtered::<&Visibility, With<TimelineStripRoot>>();
    let v = q.iter(world).next().expect("timeline strip root entity should exist");
    *v
}

#[test]
fn strip_hidden_when_timeline_is_empty() {
    let mut app = make_app();
    app.update();
    app.update();
    let v = strip_visibility(&mut app);
    assert_eq!(v, Visibility::Hidden);
}

#[test]
fn strip_visible_when_timeline_has_events() {
    let mut app = make_app();
    {
        let mut flow = app.world_mut().resource_mut::<FlowSim>();
        let nid = flow.add_node(
            "test",
            std::collections::BTreeMap::from([("x".to_string(), Value::Int(0))]),
            Vec::new(),
        );
        flow.timeline.schedule(1_000_000_000, nid, "x".into(), Value::Int(1));
    }
    app.update();
    app.update();
    app.update();

    let v = strip_visibility(&mut app);
    assert_ne!(v, Visibility::Hidden, "strip should reveal once events exist");
}
