//! End-to-end check that loading the DSL Life canvas tags inner
//! cells with [`Inside`] markers and hides them at the top level,
//! while the compound's body itself stays visible. Drilling into
//! the compound (toggling [`CurrentScope`] manually — the
//! double-click input path needs window state we don't have in
//! headless tests) flips the visibility correctly.

use std::path::PathBuf;

use bevy::prelude::*;
use flow_bevy::compound::{
    CompoundBodyMarker, CompoundMembership, CompoundParamRegistry, CurrentScope, EdgeInside,
    GridCellPaintRef, Inside,
};
use flow_bevy::sim_driver::{SimDriver, SimDriverRes, SimEventRx};
use flow_bevy::{CanvasSeedPlugin, FlowBevyPlugins};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Build a headless app with [`CanvasSeedPlugin`] installed *before*
/// any Startup pump, so `seed_from_path` runs against the requested
/// `.whiteboard` directory on the very first frame. Replaces the
/// worker-mode driver the bridge installs with a Direct one for
/// determinism. Mirrors the relevant parts of `common::make_app` but
/// skips the pre-update so the canvas seed plugin's Startup system
/// gets a chance to run.
fn boot_with_life_dsl() -> App {
    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(FlowBevyPlugins);
    app.world_mut().resource_mut::<flow_bevy::bridge::SimClock>().multiplier = 1.0;
    {
        let mut tl = app.world_mut().resource_mut::<flow_bevy::edges::VisualTimelineRes>();
        tl.0.k = 1.0;
    }
    {
        let world = app.world_mut();
        let mut sim = flow::Sim::new(1);
        flow_bevy::gadgets::install_default_params(&mut sim);
        let (driver, events_rx) = SimDriver::direct(sim, 1.0);
        world.insert_resource(SimDriverRes(driver));
        world.insert_resource(SimEventRx(std::sync::Mutex::new(events_rx)));
    }
    let path = project_root().join("examples/life_5x5_dsl.whiteboard");
    app.add_plugins(CanvasSeedPlugin(path));
    // First update runs Startup (the canvas seed). Second lets the
    // deferred Commands::insert_resource / spawn_entity / .insert
    // calls flush AND lets the visibility system run.
    app.update();
    app.update();
    app
}

#[test]
fn life_compound_loads_with_inner_cells_hidden() {
    let mut app = boot_with_life_dsl();
    let world = app.world_mut();

    // 25 cells should have `Inside` markers — they all belong to
    // `Life`. The compound body itself does not (it has no parent).
    let mut inside_count = 0;
    let mut q = world.query::<&Inside>();
    for _ in q.iter(world) { inside_count += 1; }
    assert_eq!(
        inside_count, 25,
        "expected 25 cells tagged Inside(Life), got {}", inside_count
    );

    // Each `Inside`-tagged entity should be Hidden at the top level.
    let mut q = world.query::<(&Inside, &Visibility)>();
    for (_, vis) in q.iter(world) {
        assert!(matches!(vis, Visibility::Hidden),
            "inner cell should be hidden at top level, got {:?}", vis);
    }

    // The compound body should be visible at top level.
    let mut q = world.query::<(&CompoundBodyMarker, &Visibility)>();
    let bodies: Vec<_> = q.iter(world).collect();
    assert_eq!(bodies.len(), 1, "expected exactly one compound body, got {}", bodies.len());
    assert!(matches!(bodies[0].1, Visibility::Visible),
        "compound body should be Visible at top level, got {:?}", bodies[0].1);

    // At least the 200 internal edges (8 per cell × 25) should be
    // tagged EdgeInside and hidden.
    let mut q = world.query::<(&EdgeInside, &Visibility)>();
    let edge_pairs: Vec<_> = q.iter(world).collect();
    assert!(edge_pairs.len() >= 200,
        "expected ≥ 200 internal edges, got {}", edge_pairs.len());
    for (_, vis) in &edge_pairs {
        assert!(matches!(vis, Visibility::Hidden),
            "internal edge should be hidden at top level, got {:?}", vis);
    }
}

#[test]
fn drilling_into_life_reveals_cells_and_hides_compound_body() {
    let mut app = boot_with_life_dsl();

    // Find the compound's NodeId via the membership map (any cell's
    // parent is Life).
    let life_id = {
        let world = app.world();
        let membership = world.resource::<CompoundMembership>();
        let any_parent = membership
            .parent
            .values()
            .next()
            .copied()
            .expect("compound membership populated");
        any_parent
    };

    // Set scope and tick so the visibility system runs.
    app.world_mut().resource_mut::<CurrentScope>().0 = Some(life_id);
    app.update();

    let world = app.world_mut();

    // Cells now Visible.
    let mut q = world.query::<(&Inside, &Visibility)>();
    for (inside, vis) in q.iter(world) {
        assert_eq!(inside.0, life_id);
        assert!(matches!(vis, Visibility::Visible),
            "cell should be visible when drilled into Life, got {:?}", vis);
    }

    // Compound body now Hidden.
    let mut q = world.query::<(&CompoundBodyMarker, &Visibility)>();
    for (_, vis) in q.iter(world) {
        assert!(matches!(vis, Visibility::Hidden),
            "compound body should hide when we're inside it, got {:?}", vis);
    }

    // Internal edges now Visible.
    let mut q = world.query::<(&EdgeInside, &Visibility)>();
    for (_, vis) in q.iter(world) {
        assert!(matches!(vis, Visibility::Visible),
            "internal edge should be visible when drilled in, got {:?}", vis);
    }

    // Pop scope back to top-level — everything reverts.
    app.world_mut().resource_mut::<CurrentScope>().0 = None;
    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(&Inside, &Visibility)>();
    for (_, vis) in q.iter(world) {
        assert!(matches!(vis, Visibility::Hidden),
            "cell should re-hide at top level, got {:?}", vis);
    }
}

#[test]
fn compound_param_registry_carries_authoring_metadata() {
    let mut app = boot_with_life_dsl();
    let world = app.world_mut();
    let registry = world.resource::<CompoundParamRegistry>();
    let life = registry
        .by_name
        .get("Life")
        .expect("Life compound should be in CompoundParamRegistry");

    // Param order matches DSL declaration: width, height, period_ns,
    // cell_latency_ns. We only assert the names + that defaults
    // round-tripped — the values are checked numerically below.
    let names: Vec<&str> = life.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["width", "height", "period_ns", "cell_latency_ns"]);

    // width default should be Int(5).
    use flow::dsl::expand::CtValue;
    let width = life.iter().find(|p| p.name == "width").unwrap();
    assert!(matches!(width.default, Some(CtValue::Int(5))));
    let height = life.iter().find(|p| p.name == "height").unwrap();
    assert!(matches!(height.default, Some(CtValue::Int(5))));
    let period = life.iter().find(|p| p.name == "period_ns").unwrap();
    assert!(matches!(period.default, Some(CtValue::Int(200_000_000))));
}

#[test]
fn life_compound_renders_grid_cells_from_visual_json() {
    let mut app = boot_with_life_dsl();
    let world = app.world_mut();

    // Visual.json declares a 5×5 grid; spawn should produce 25
    // GridCellPaintRef-tagged entities.
    let mut q = world.query::<&GridCellPaintRef>();
    let cells: Vec<_> = q.iter(world).collect();
    assert_eq!(
        cells.len(), 25,
        "expected 25 grid-cell paint refs, got {}", cells.len()
    );

    // Every cell's source should resolve to a real NodeId (the
    // member_pattern matched all 25 names) and the slot should be
    // "alive" as declared in visual.json.
    for paint in &cells {
        assert!(paint.source.is_some(), "grid cell has unresolved source");
        assert_eq!(paint.slot, "alive");
    }

    // The blinker pattern starts with three live cells at row 2,
    // columns 1..=3. Their source NodeIds should resolve to nodes
    // whose `alive` slot is currently 1 in the snapshot.
    let live_count = {
        let snapshot = world.resource::<flow_bevy::sim_driver::SimSnapshotRes>();
        cells.iter().filter(|paint| {
            paint.source
                .and_then(|nid| snapshot.0.nodes.get(&nid))
                .and_then(|n| n.slots.get("alive"))
                .map(|v| matches!(v, flow::Value::Int(1)))
                .unwrap_or(false)
        }).count()
    };
    assert_eq!(live_count, 3, "expected 3 initially-live cells (blinker), got {}", live_count);
}
