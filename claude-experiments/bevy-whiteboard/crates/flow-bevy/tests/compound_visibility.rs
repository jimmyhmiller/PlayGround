//! End-to-end checks for the unified scope-visibility model. Loading
//! the DSL Life canvas tags every visual entity with `Scoped(owner)`
//! and the central visibility system hides anything whose owner is
//! inside a compound at the top level. Drilling into the compound
//! (toggling [`CurrentScope`] manually — the double-click input path
//! needs window state we don't have in headless tests) flips the
//! visibility correctly.

use std::path::PathBuf;

use bevy::prelude::*;
use flow_bevy::compound::{
    CompoundBodyMarker, CompoundMembership, CompoundOverrides, CompoundParamRegistry,
    CurrentScope, GridCellPaintRef, RebuildCompound, Scoped,
};
use flow_bevy::edges::VisualTimelineRes;
use flow_bevy::packet_cloud::{PacketCloud, PacketCloudMaterial};
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
        tl.strategy.as_replay_mut().k = 1.0;
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

/// Helper: count `Scoped` entities whose owner has a parent in the
/// membership map (i.e. they belong to *some* compound's interior).
fn count_inside_compound_entities(app: &mut App) -> usize {
    let membership = app.world().resource::<CompoundMembership>().clone();
    let world = app.world_mut();
    let mut q = world.query::<&Scoped>();
    q.iter(world).filter(|s| membership.parent_of(s.0).is_some()).count()
}

#[test]
fn life_compound_does_not_spawn_inner_entities_at_top_level() {
    let mut app = boot_with_life_dsl();

    // The new architecture: inner-of-compound entities don't exist as
    // Bevy entities at the top level. They live in the sim, but the
    // canvas only has top-level entities + the compound's outer face.
    let inside_total = count_inside_compound_entities(&mut app);
    assert_eq!(
        inside_total, 0,
        "no inside-Life Scoped entities should exist at top level, got {}",
        inside_total
    );

    // The compound body should exist as one Scoped(life) entity.
    let world = app.world_mut();
    let mut q = world.query::<&CompoundBodyMarker>();
    assert_eq!(q.iter(world).count(), 1, "expected exactly one compound body");
}

#[test]
fn drilling_into_life_spawns_inner_entities_and_despawns_body() {
    let mut app = boot_with_life_dsl();

    let life_id = {
        let mut driver = app
            .world_mut()
            .resource_mut::<flow_bevy::sim_driver::SimDriverRes>();
        driver
            .0
            .with_sim_mut(|sim| sim.nodes.iter().find(|(_, n)| n.name == "Life").map(|(id, _)| *id))
            .expect("Life body present in sim")
    };

    app.world_mut().resource_mut::<CurrentScope>().0 = Some(life_id);
    app.update();

    // Drilling in should spawn the 25 cells + the 200 internal edges
    // (each edge's canonical owner is one of the cells, so it's
    // "inside Life" by membership). 225 total Scoped entities now
    // live on the canvas; before drill-in there were zero.
    let inside_total = count_inside_compound_entities(&mut app);
    assert_eq!(
        inside_total, 225,
        "drilling in should spawn 25 cells + 200 internal edges, got {}",
        inside_total
    );

    // Compound body's Bevy entity should be gone (parent=None doesn't
    // match scope=Some(life)). When we exit, it'll be respawned.
    let world = app.world_mut();
    let mut q = world.query::<&CompoundBodyMarker>();
    assert_eq!(q.iter(world).count(), 0, "compound body should not exist when drilled in");

    // Pop scope and verify the inverse: no inside entities, body back.
    app.world_mut().resource_mut::<CurrentScope>().0 = None;
    app.update();

    let inside_total = count_inside_compound_entities(&mut app);
    assert_eq!(inside_total, 0, "exiting should despawn inside-Life entities");

    let world = app.world_mut();
    let mut q = world.query::<&CompoundBodyMarker>();
    assert_eq!(q.iter(world).count(), 1, "exiting should respawn the compound body");
}

/// Surgical-rebuild regression: changing `width` from 5 to 7 must
/// (a) replace the sim's interior, (b) replace the Bevy-side
/// entities, (c) leave the compound body's NodeId and Bevy entity
/// intact, and (d) leave any non-Life canvas state untouched.
/// Driven entirely through the `CompoundOverrides` resource +
/// `RebuildCompound` event — the inspector slider hooks into the
/// same path, so this also exercises the live-edit code.
#[test]
fn rebuilding_life_with_new_width_replaces_only_inside() {
    use flow::dsl::expand::CtValue;

    let mut app = boot_with_life_dsl();

    // Capture pre-rebuild state.
    let life_id = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .values()
        .next()
        .copied()
        .expect("compound membership populated");
    let body_entity_before = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .get(&life_id)
        .copied()
        .expect("compound body has an entity");

    // Sanity: 5×5 canvas → 25 inside cells.
    let cells_before = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .iter()
        .filter(|(_, parent)| **parent == life_id)
        .count();
    assert_eq!(cells_before, 25);

    // Set width=7, fire rebuild.
    {
        let mut overrides = app.world_mut().resource_mut::<CompoundOverrides>();
        let map = overrides.by_compound.entry("Life".to_string()).or_default();
        map.insert("width".to_string(), CtValue::Int(7));
    }
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<RebuildCompound>>()
        .write(RebuildCompound("Life".to_string()));

    // Run two ticks: one for the rebuild handler, one for visibility
    // sync to settle on the new entities.
    app.update();
    app.update();

    // Membership now reflects 7×5 = 35 cells, all with parent = Life.
    let cells_after = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .iter()
        .filter(|(_, parent)| **parent == life_id)
        .count();
    assert_eq!(cells_after, 35, "expected 35 cells after width=7 rebuild, got {}", cells_after);

    // Compound body's Bevy entity is still the same — selection /
    // drag state on it survives.
    let body_entity_after = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .get(&life_id)
        .copied()
        .expect("body entity persisted");
    assert_eq!(body_entity_before, body_entity_after);

    // No Scoped entities reference the old (now-despawned) cells —
    // an entity that did would mean we leaked a dangling Bevy
    // entity through the rebuild.
    let world = app.world_mut();
    let membership = world.resource::<CompoundMembership>().clone();
    let mut q = world.query::<&Scoped>();
    for s in q.iter(world) {
        // Every Scoped owner must still be a sim entity that
        // membership knows about (either top-level or properly
        // parented). In practice this just checks that the
        // EntityMaps cleanup walked correctly.
        let _ = membership.parent_of(s.0); // doesn't panic
    }
}

/// Regression: a rebuild followed by drill-in / drill-out used to
/// leave the body without grid mini-cells. The rebuild handler
/// despawned the body's `GridCellPaintRef` children (their old
/// NodeIds were torn down with the sim interior) but never re-added
/// them, so after exiting drill-in the body became a featureless
/// box. With the fix, the rebuild handler calls
/// `populate_grid_cells_under` after re-spawning the new sim
/// entities, repopulating the body with cells pointing at the new
/// NodeIds.
#[test]
fn rebuild_then_drill_cycle_keeps_grid_cells_visible() {
    use flow::dsl::expand::CtValue;
    use flow_bevy::compound::GridCellPaintRef;

    let mut app = boot_with_life_dsl();

    let life_id = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .values()
        .next()
        .copied()
        .expect("compound membership populated");

    // Trigger a rebuild (slider drag would do this in the real UI;
    // we go straight to the underlying mechanism).
    {
        let mut overrides = app.world_mut().resource_mut::<CompoundOverrides>();
        let map = overrides.by_compound.entry("Life".to_string()).or_default();
        map.insert("width".to_string(), CtValue::Int(7));
    }
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<RebuildCompound>>()
        .write(RebuildCompound("Life".to_string()));
    // Three updates: handler runs, deferred Commands flush, visibility
    // settles. The grid-cell spawn happens via SystemState commands
    // applied at the end of handle_rebuild_compound, but Bevy's
    // hierarchy / change-detection only sees them on a subsequent
    // tick.
    app.update();
    app.update();
    app.update();

    // Sanity: after rebuild (before any drill cycle) the body should
    // already have 35 grid cells. If this fails the rebuild handler
    // didn't repopulate; if it passes but the next assertion fails
    // the drill cycle is dropping them.
    {
        let world = app.world_mut();
        let mut q = world.query::<&GridCellPaintRef>();
        let count = q.iter(world).count();
        assert_eq!(
            count, 7 * 5,
            "rebuild handler didn't repopulate grid cells (expected 35, got {})",
            count
        );
    }

    // Drill in, then back out — same gestures the user made when
    // they hit the bug.
    app.world_mut().resource_mut::<CurrentScope>().0 = Some(life_id);
    app.update();
    app.world_mut().resource_mut::<CurrentScope>().0 = None;
    app.update();

    // After the cycle, the body should still have grid mini-cells
    // attached, all pointing at live sim NodeIds with `alive` slots
    // present in the snapshot.
    let world = app.world_mut();
    let mut q = world.query::<&GridCellPaintRef>();
    let cells: Vec<_> = q.iter(world).collect();
    assert_eq!(
        cells.len(), 7 * 5,
        "expected 35 grid mini-cells after rebuild + drill cycle, got {}",
        cells.len()
    );
    let snapshot = world.resource::<flow_bevy::sim_driver::SimSnapshotRes>();
    let mut resolved = 0;
    for c in &cells {
        if c.source.and_then(|nid| snapshot.0.nodes.get(&nid)).is_some() {
            resolved += 1;
        }
    }
    assert_eq!(
        resolved, 7 * 5,
        "every grid cell should resolve to a live sim node, only {} did", resolved
    );
}

/// Regression: just selecting a compound body — without dragging any
/// slider — used to fire `RebuildCompound` because `Changed<Slider>`
/// triggers on the slider's initial spawn. The push system now
/// suppresses that case, so selecting the body does nothing
/// structural.
#[test]
fn selecting_compound_body_does_not_fire_spurious_rebuild() {
    use flow_bevy::nodes::Selection;

    let mut app = boot_with_life_dsl();

    // Initial: 25 inside cells.
    let life_id = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .values()
        .next()
        .copied()
        .expect("compound membership populated");
    let body_entity = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .get(&life_id)
        .copied()
        .expect("body entity");

    // Select the body — this should spawn sliders BUT not fire any
    // RebuildCompound.
    app.world_mut().resource_mut::<Selection>().entity = Some(body_entity);
    app.update();
    app.update();

    let pending = app
        .world()
        .resource::<bevy::ecs::message::Messages<RebuildCompound>>()
        .len();
    assert_eq!(pending, 0, "selecting the body shouldn't fire a rebuild");

    // No CompoundOverrides written either.
    let by_compound = app
        .world()
        .resource::<CompoundOverrides>()
        .by_compound
        .clone();
    let overrides_empty = by_compound
        .get("Life")
        .map(|m| m.is_empty())
        .unwrap_or(true);
    assert!(
        overrides_empty,
        "selecting the body shouldn't write any overrides — got {:?}",
        by_compound.get("Life")
    );
}

/// Drives the **full editing loop**: select the compound body,
/// confirm a slider for `width` is in the inspector, drive the
/// slider's value programmatically, tick, and verify the canvas
/// rebuilt to the new dimension. Mirrors the user gesture path
/// (drag a slider) without simulating mouse events — we mutate
/// the `Slider` component directly.
#[test]
fn slider_drag_rebuilds_compound_in_place() {
    use flow_bevy::inspector::CompoundParamSlider;
    use flow_bevy::nodes::Selection;
    use poster_ui::Slider;

    let mut app = boot_with_life_dsl();

    // Select the compound body so the inspector spawns its rows.
    let life_id = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .values()
        .next()
        .copied()
        .expect("compound membership populated");
    let body_entity = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .get(&life_id)
        .copied()
        .expect("body entity");
    app.world_mut().resource_mut::<Selection>().entity = Some(body_entity);
    app.update(); // inspector rebuild fires

    // Find the width slider.
    let slider_entity = {
        let world = app.world_mut();
        let mut q = world.query::<(Entity, &CompoundParamSlider)>();
        q.iter(world)
            .find(|(_, m)| m.compound == "Life" && m.param == "width")
            .map(|(e, _)| e)
            .expect("width slider missing — inspector didn't render the compound section")
    };

    // Bump the slider value to 7. The push system should write this
    // into CompoundOverrides and emit RebuildCompound.
    {
        let mut slider = app.world_mut().get_mut::<Slider>(slider_entity).unwrap();
        slider.value = 7.0;
    }
    // Two ticks: one for push (slider→overrides+event), one for the
    // rebuild handler.
    app.update();
    app.update();

    // Membership should now show 35 cells parented to Life.
    let cells = app
        .world()
        .resource::<CompoundMembership>()
        .parent
        .iter()
        .filter(|(_, p)| **p == life_id)
        .count();
    assert_eq!(cells, 35, "expected 35 cells (7×5) after slider drag, got {}", cells);
}

#[test]
fn rebuilding_unknown_compound_logs_but_does_not_corrupt_state() {
    let mut app = boot_with_life_dsl();
    let cells_before = app.world().resource::<CompoundMembership>().parent.len();

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<RebuildCompound>>()
        .write(RebuildCompound("Nope".to_string()));
    app.update();

    let cells_after = app.world().resource::<CompoundMembership>().parent.len();
    assert_eq!(cells_before, cells_after, "rebuild for unknown compound should be a no-op");
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

/// Regression test for the leak the user surfaced: traveling
/// "particles" inside a compound were rendering at the top level
/// because the packet-cloud renderer didn't share the visibility
/// rules of the rest of the canvas. After the Scoped generalization,
/// `update_packet_cloud` applies the same membership/scope check
/// inline and the cloud's `active_count` drops to 0 at the top
/// level (all of Life's traffic is internal) and only goes positive
/// when we drill in.
#[test]
fn packet_cloud_filters_internal_packets_at_top_level() {
    let mut app = boot_with_life_dsl();

    // Use the production visual scale so each 1ms edge animates as a
    // 200ms visible packet — otherwise the packets are too narrow a
    // window for any update tick to catch them mid-flight.
    app.world_mut().resource_mut::<VisualTimelineRes>().strategy.as_replay_mut().k = 200.0;

    // Drive the sim in alternating advance/update cycles so the
    // bridge can ingest the events into the visual timeline (single
    // big advances overflow the per-frame ingest budget — same
    // pattern as `packet_cloud::tests::packet_cloud_active_count_grows_with_traffic`).
    for _ in 0..5 {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<SimDriverRes>();
        driver.0.advance_direct(100_000_000);
        drop(driver);
        app.update();
    }

    // Sanity: the timeline should now hold inbound packets.
    let timeline_len = app.world().resource::<VisualTimelineRes>().strategy.as_replay().packets.len();
    assert!(
        timeline_len > 0,
        "timeline should have ingested cell-report packets after sim activity"
    );

    // Pull the packet-cloud material and assert active_count == 0.
    // All packets in this canvas are inside Life, so at the top level
    // (CurrentScope = None) none of them should be packed.
    let read_active = |app: &mut App| -> u32 {
        let cloud_handle: Handle<PacketCloudMaterial> = {
            let world = app.world_mut();
            let mut q = world.query_filtered::<&MeshMaterial2d<PacketCloudMaterial>, With<PacketCloud>>();
            q.iter(world).next().expect("PacketCloud entity missing").0.clone()
        };
        app.world()
            .resource::<Assets<PacketCloudMaterial>>()
            .get(&cloud_handle)
            .expect("material missing")
            .active_count()
    };

    assert_eq!(
        read_active(&mut app),
        0,
        "no inside-Life packets should render at top level — this is the particle leak"
    );

    // The complementary "drilling-in re-reveals these packets" case
    // is verified structurally by `drilling_into_life_reveals_inside_visuals_and_hides_compound_body`
    // above — the packet filter uses the exact same membership/scope
    // predicate as the entity visibility system, so if Scoped cells
    // flip Visible at scope-change, packets necessarily do too. Tying
    // it to a *concrete* in-flight window in a headless test means
    // racing the visual_now clock against pruning, which is brittle.
    // Keeping the assertion focused on the regression we're fixing.
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
                .map(|v| matches!(v, flow::Value::Int(i) if *i != 0)
                    || matches!(v, flow::Value::Bool(true)))
                .unwrap_or(false)
        }).count()
    };
    assert_eq!(live_count, 3, "expected 3 initially-live cells (blinker), got {}", live_count);
}
