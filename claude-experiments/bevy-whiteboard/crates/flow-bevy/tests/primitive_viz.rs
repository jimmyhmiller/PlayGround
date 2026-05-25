//! Headless verification that high-fidelity primitive visuals attach
//! and animate. Loads the life_cell_pure_primitives whiteboard
//! (a single Life cell decomposed into 10 primitive instances spanning
//! 8 distinct classes — Tick, Switch, ConstantPacket, FanOut,
//! Aggregator, Filter, ConstantSignal) and asserts:
//!
//! - Every primitive instance carries [`PrimitiveKind`] + [`PrimitivePulse`]
//! - Child entities (Pendulum, Blade, Spokes, Beads, etc.) get spawned
//! - Sim-driven slot changes propagate to material/transform updates
//!   (Aggregator bead count tracks the `seen` slot; Switch blade angle
//!   follows the `passing` slot)

use std::path::PathBuf;

use bevy::prelude::*;
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::primitive_viz::{
    AggregatorBead, FanoutSpoke, PrimitiveBuilt, PrimitiveKind, PrimitivePulse, SwitchBlade,
    TickPendulum,
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

fn boot(canvas: &str) -> App {
    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(FlowBevyPlugins);
    app.world_mut()
        .resource_mut::<flow_bevy::bridge::SimClock>()
        .multiplier = 1.0;
    {
        let mut tl = app
            .world_mut()
            .resource_mut::<flow_bevy::edges::VisualTimelineRes>();
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
    let path = project_root().join("examples").join(canvas);
    app.add_plugins(CanvasSeedPlugin(path));
    // Startup, then a few updates to let Commands flush and the
    // construct_primitive_visuals system run.
    app.update();
    app.update();
    app.update();
    app
}

#[test]
fn every_primitive_in_pure_life_cell_gets_a_kind_marker() {
    let mut app = boot("life_cell_pure_primitives.whiteboard");

    let world = app.world_mut();
    let mut q = world.query::<&PrimitiveKind>();
    let kinds: Vec<PrimitiveKind> = q.iter(world).copied().collect();
    assert!(!kinds.is_empty(), "expected primitive viz markers");

    // The canvas has these primitives (per main.flow):
    //   T (Tick), SW (Switch), C1/C0 (Constant x2 packet-flavored),
    //   B (FanOut), A (Aggregator), F3/F2 (Filter x2),
    //   ToAlv/ToDed (Constant x2 signal-flavored)
    let count = |k: PrimitiveKind| kinds.iter().filter(|x| **x == k).count();
    assert_eq!(count(PrimitiveKind::Tick), 1, "kinds={:?}", kinds);
    assert_eq!(count(PrimitiveKind::Switch), 1, "kinds={:?}", kinds);
    // Unified Constant — covers both the value-source (C1/C0) and the
    // signal-source (ToAlv/ToDed) roles, distinguished only by the
    // `out_kind` slot. Four total in this scene.
    assert_eq!(count(PrimitiveKind::Constant), 4, "kinds={:?}", kinds);
    assert_eq!(count(PrimitiveKind::FanOut), 1, "kinds={:?}", kinds);
    assert_eq!(count(PrimitiveKind::Egress), 1, "kinds={:?}", kinds);
    assert_eq!(count(PrimitiveKind::Aggregator), 1, "kinds={:?}", kinds);
    assert_eq!(count(PrimitiveKind::Filter), 2, "kinds={:?}", kinds);
}

#[test]
fn child_entities_are_built_for_each_primitive() {
    let mut app = boot("life_cell_pure_primitives.whiteboard");
    let world = app.world_mut();

    // At least one pendulum (Tick), one blade (Switch), one fanout
    // spoke, and 8 aggregator beads should exist after construction.
    let mut pend = world.query::<&TickPendulum>();
    let mut blade = world.query::<&SwitchBlade>();
    let mut spoke = world.query::<&FanoutSpoke>();
    let mut bead = world.query::<&AggregatorBead>();
    assert_eq!(pend.iter(world).count(), 1, "Tick pendulum missing");
    assert_eq!(blade.iter(world).count(), 1, "Switch blade missing");
    assert_eq!(
        spoke.iter(world).count(),
        8,
        "FanOut should have 8 radial spokes"
    );
    assert_eq!(
        bead.iter(world).count(),
        8,
        "Aggregator should lay out 8 beads (3x3 minus center)"
    );

    // Every primitive should be marked Built.
    let mut built = world.query::<(&PrimitiveKind, &PrimitiveBuilt)>();
    let mut all = world.query::<&PrimitiveKind>();
    assert_eq!(
        built.iter(world).count(),
        all.iter(world).count(),
        "every PrimitiveKind entity should have PrimitiveBuilt"
    );
}

#[test]
fn pulse_advances_on_packet_delivery() {
    let mut app = boot("life_cell_pure_primitives.whiteboard");

    // Run sim ~1 second — Tick fires 5 packets (period=200ms), each
    // ripples through SW → C1/C0 → B and into A's neighbour port.
    // Any primitive on the active path should accumulate non-default
    // last_arrival or last_emit timestamps.
    {
        let world = app.world_mut();
        world.resource_mut::<FlowSim>().0.advance_direct(1_000_000_000);
    }
    app.update();
    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(&PrimitiveKind, &PrimitivePulse)>();
    let mut tick_saw_emit = false;
    for (kind, pulse) in q.iter(world) {
        if *kind == PrimitiveKind::Tick {
            // Tick emits every 200ms — in 1s of sim we expect ≥1 emit.
            if pulse.last_emit > 0.0 {
                tick_saw_emit = true;
            }
        }
    }
    assert!(
        tick_saw_emit,
        "Tick primitive should have a recorded emit pulse after 1s of sim time"
    );
}

#[test]
fn switch_blade_tracks_passing_slot() {
    let mut app = boot("life_cell_pure_primitives.whiteboard");

    // Capture the Switch's NodeId, then advance the sim and toggle
    // its `passing` slot. The blade transform should drift toward the
    // divert angle (-0.52 rad) when passing=0.
    let switch_id = {
        let world = app.world_mut();
        let sim = world.resource::<FlowSim>();
        sim.nodes
            .iter()
            .find(|(_, n)| n.name == "SW")
            .map(|(id, _)| *id)
            .expect("SW node exists")
    };

    // Flip passing to 0 via a sim command.
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<SimDriverRes>();
        driver.0.send_command(flow_bevy::sim_driver::SimCommand::new(move |sim| {
            if let Some(n) = sim.nodes.get_mut(&switch_id) {
                n.slots.insert("passing".into(), flow::Value::Int(0));
            }
        }));
        driver.0.advance_direct(10_000_000);
    }

    // Run several frames + sim advance so the tween (driven by
    // visual_now) can converge.
    for _ in 0..20 {
        {
            let world = app.world_mut();
            world.resource_mut::<FlowSim>().0.advance_direct(50_000_000);
        }
        app.update();
    }

    let world = app.world_mut();
    let maps = world.resource::<EntityMaps>().clone_handles();
    let switch_entity = maps
        .get(&switch_id)
        .copied()
        .expect("SW has a Bevy entity");
    let children: Vec<Entity> = world
        .entity(switch_entity)
        .get::<Children>()
        .expect("SW entity has children")
        .iter()
        .collect();
    let mut blade_q = world.query::<(Entity, &SwitchBlade)>();
    let blade_angle = blade_q
        .iter(world)
        .find_map(|(e, b)| {
            if children.contains(&e) {
                Some(b.current_angle)
            } else {
                None
            }
        })
        .expect("SW has a blade child");
    // Target is -0.52; after ~1.5s of tween at 8/s it should be very
    // close. Sanity-check it's at least past halfway.
    assert!(
        blade_angle < -0.20,
        "expected blade to swing toward divert (-0.52), got {}",
        blade_angle
    );
}

#[test]
fn aggregator_beads_light_up_with_seen_slot() {
    let mut app = boot("life_cell_pure_primitives.whiteboard");

    let agg_id = {
        let world = app.world_mut();
        let sim = world.resource::<FlowSim>();
        sim.nodes
            .iter()
            .find(|(_, n)| n.name == "A")
            .map(|(id, _)| *id)
            .expect("A node exists")
    };

    // Force seen=5 and check that exactly 5 beads (slots 0..5) are lit.
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<SimDriverRes>();
        driver.0.send_command(flow_bevy::sim_driver::SimCommand::new(move |sim| {
            if let Some(n) = sim.nodes.get_mut(&agg_id) {
                n.slots.insert("seen".into(), flow::Value::Int(5));
            }
        }));
        driver.0.advance_direct(1_000_000);
    }
    app.update();
    app.update();

    let world = app.world_mut();
    let agg_entity = world
        .resource::<EntityMaps>()
        .node_to_entity
        .get(&agg_id)
        .copied()
        .expect("agg has entity");
    let bead_children: Vec<Entity> = world
        .entity(agg_entity)
        .get::<Children>()
        .expect("agg has children")
        .iter()
        .collect();
    let mut bead_q = world.query::<(Entity, &AggregatorBead, &MeshMaterial2d<ColorMaterial>)>();
    let mats = world.resource::<Assets<ColorMaterial>>();
    let mut lit_count = 0;
    for (e, bead, mh) in bead_q.iter(world) {
        if !bead_children.contains(&e) {
            continue;
        }
        let c = mats.get(&mh.0).expect("bead material exists").color;
        let s = c.to_srgba();
        // Lit color is warm-yellow (R≈0.95). Dim is dark purple-grey.
        if s.red > 0.6 {
            lit_count += 1;
        }
        // Sanity: slot index 0..5 should be lit; 5..8 should be dim.
        if bead.slot < 5 {
            assert!(
                s.red > 0.6,
                "bead {} should be lit, color={:?}",
                bead.slot,
                s
            );
        } else {
            assert!(
                s.red < 0.6,
                "bead {} should be dim, color={:?}",
                bead.slot,
                s
            );
        }
    }
    assert_eq!(lit_count, 5, "expected exactly 5 lit beads for seen=5");
}

/// Small helper trait so we don't need to copy EntityMaps wholesale.
trait EntityMapsExt {
    fn clone_handles(&self) -> std::collections::HashMap<flow::NodeId, Entity>;
}
impl EntityMapsExt for EntityMaps {
    fn clone_handles(&self) -> std::collections::HashMap<flow::NodeId, Entity> {
        self.node_to_entity.clone()
    }
}
