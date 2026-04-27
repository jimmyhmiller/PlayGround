//! Data-palette interaction tests. Covers:
//!
//! * Clicking each of the six swatches updates the `ActiveSlot` resource.
//! * The `ActiveSlot` default (slot 0) is what seeded nodes get in `NodeColors`.
//! * Dropping a node after selecting slot N tags it with `theme.data[N]`.
//! * Theme swap preserves the selected slot index — the colour under the
//!   swatch updates, but selection survives.
//! * Two drops under different slots end up with different colours.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::{ActionBtn, ColorSwatch, ToolBtn};
use flow_bevy::tool::{ActiveSlot, NodeColors, Tool};
use poster_ui::{DATA_SLOT_COUNT, Theme};
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn latest_of_kind(app: &App, kind: Kind) -> Option<flow::NodeId> {
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
}

#[test]
fn default_slot_is_zero() {
    let app = make_app();
    assert_eq!(
        app.world().resource::<ActiveSlot>().0,
        0,
        "ActiveSlot should start at 0 (the dominant accent)"
    );
}

#[test]
fn click_each_swatch_updates_active_slot() {
    let mut app = make_app();
    for i in 0..DATA_SLOT_COUNT {
        let clicked = click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == i);
        assert!(clicked, "no ColorSwatch for slot {}", i);
        assert_eq!(
            app.world().resource::<ActiveSlot>().0,
            i,
            "clicking swatch {} should set ActiveSlot to {}",
            i,
            i
        );
    }
}

#[test]
fn drop_after_swatch_records_node_color() {
    let mut app = make_app();

    // Swatch 2 is the moss/olive slot in iso50. Pick it, drop a worker at
    // (0, 300) — well away from the seeded chain.
    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 2);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Worker));
    simulate_canvas_click(&mut app, Vec2::new(0.0, 300.0));

    let nid = latest_of_kind(&app, Kind::Worker).expect("worker wasn't dropped");
    let expected = {
        let theme = app.world().resource::<Theme>();
        theme.data[2]
    };
    let recorded = *app
        .world()
        .resource::<NodeColors>()
        .0
        .get(&nid)
        .expect("NodeColors missing entry for dropped worker");
    assert_eq!(
        recorded, expected,
        "dropped worker tagged with wrong colour: got {:?}, want {:?} (slot 2)",
        recorded, expected
    );
}

#[test]
fn two_drops_under_different_slots_get_different_colors() {
    let mut app = make_app();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 1);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-200.0, 300.0));
    let a = latest_of_kind(&app, Kind::Generator).unwrap();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 4);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(200.0, 300.0));
    let b = latest_of_kind(&app, Kind::Generator).unwrap();

    let colors = &app.world().resource::<NodeColors>().0;
    let ca = colors.get(&a).copied().expect("no color for first gen");
    let cb = colors.get(&b).copied().expect("no color for second gen");
    assert_ne!(
        ca, cb,
        "two generators dropped under different slots should have different colours"
    );
}

#[test]
fn theme_swap_preserves_slot_index() {
    let mut app = make_app();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 3);
    let before_theme = app.world().resource::<Theme>().name;

    click_by_marker::<ActionBtn, _>(&mut app, |a| matches!(a, ActionBtn::NextTheme));

    assert_ne!(
        app.world().resource::<Theme>().name,
        before_theme,
        "theme should have changed"
    );
    assert_eq!(
        app.world().resource::<ActiveSlot>().0,
        3,
        "swatch selection survives theme swap (slot index, not colour value)"
    );
}

#[test]
fn snapshotted_color_does_not_follow_theme_swap() {
    // Drop a node under iso50's slot 2, then swap theme. The node's
    // recorded colour stays the iso50 colour — intentional: data tags
    // are a user-made typology choice, not a theme decoration.
    let mut app = make_app();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 2);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(0.0, -300.0));
    let nid = latest_of_kind(&app, Kind::Sink).unwrap();

    let pre_color = *app.world().resource::<NodeColors>().0.get(&nid).unwrap();

    click_by_marker::<ActionBtn, _>(&mut app, |a| matches!(a, ActionBtn::NextTheme));

    let post_color = *app.world().resource::<NodeColors>().0.get(&nid).unwrap();
    assert_eq!(
        pre_color, post_color,
        "node's snapshotted colour should survive theme swap unchanged"
    );
}
