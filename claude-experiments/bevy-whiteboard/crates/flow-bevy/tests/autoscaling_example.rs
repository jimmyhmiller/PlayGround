//! Bevy-level test for the AutoScaling example: the fleet grows in the
//! sim AND each spawned worker gets a top-level Bevy entity (no drill-in)
//! — exercising the topology reconciler + the fact that spawned workers
//! live at the top level.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;
use flow_bevy::nodes::NodeKind;

fn project_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The whiteboard's upstream node must render as a real **Generator**
/// (with the generator shape + rate slider), not a generic Worker box.
/// Regression test for whiteboard-loaded composite gadgets coming
/// through untyped and defaulting to `Kind::Worker`.
#[test]
fn whiteboard_generator_is_typed_as_a_generator() {
    let path = project_root().join("examples/autoscaling.whiteboard");
    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(flow_bevy::FlowBevyPlugins);
    app.insert_resource(flow_bevy::PendingCanvas(Some(path)))
        .add_systems(Startup, flow_bevy::canvas::seed_from_path);
    app.update();
    app.update();

    let kind_of = |app: &App, name: &str| -> Option<Kind> {
        let snap = app.world().resource::<flow_bevy::sim_driver::SimSnapshotRes>();
        let nid = snap.0.nodes.iter().find(|(_, v)| v.name == name).map(|(id, _)| *id)?;
        let ent = *app.world().resource::<EntityMaps>().node_to_entity.get(&nid)?;
        app.world().entity(ent).get::<NodeKind>().map(|k| k.0)
    };

    assert_eq!(
        kind_of(&app, "Generator"),
        Some(Kind::Generator),
        "the whiteboard's Generator must be typed as a Generator (so it \
         renders correctly and shows the rate slider)"
    );
    assert_eq!(kind_of(&app, "ASG"), Some(Kind::AutoScalingGroup));
    assert_eq!(kind_of(&app, "Sink"), Some(Kind::Sink));

    // Selecting the Generator must produce a rate slider bound to it —
    // i.e. the rate control actually shows, even though it's a compound.
    let gen_nid = {
        let snap = app.world().resource::<flow_bevy::sim_driver::SimSnapshotRes>();
        snap.0.nodes.iter().find(|(_, v)| v.name == "Generator").map(|(id, _)| *id).unwrap()
    };
    let gen_ent = *app.world().resource::<EntityMaps>().node_to_entity.get(&gen_nid).unwrap();
    app.world_mut().resource_mut::<flow_bevy::nodes::Selection>().entity = Some(gen_ent);
    app.update();
    app.update();

    let mut found_rate_slider = false;
    let mut q = app.world_mut().query::<&flow_bevy::inspector::RateSlider>();
    for rs in q.iter(app.world()) {
        if rs.node == gen_nid {
            found_rate_slider = true;
        }
    }
    assert!(
        found_rate_slider,
        "selecting the Generator should show a rate slider bound to it"
    );
}

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

/// Top-level worker shims spawned by the ASG (`WorkerComposite_<n>`).
fn worker_ids(app: &App) -> Vec<flow::NodeId> {
    let sim = &app.world().resource::<FlowSim>();
    sim.nodes
        .iter()
        .filter(|(_, n)| {
            n.name.starts_with("WorkerComposite_") && !n.name.contains("::")
        })
        .map(|(id, _)| *id)
        .collect()
}

#[test]
fn autoscaling_example_fans_out_workers_at_top_level() {
    let mut app = make_app();
    load(&mut app, Example::AutoScaling);

    for _ in 0..40 {
        advance_sim_ns(&mut app, 150_000_000); // ~6 s
        app.update();
    }

    let workers = worker_ids(&app);
    assert!(
        workers.len() > 1,
        "ASG should have scaled the fleet up under load, got {} workers",
        workers.len()
    );

    // Every live worker has a TOP-LEVEL canvas entity (no drill-in).
    let maps = app.world().resource::<EntityMaps>();
    let rendered = workers
        .iter()
        .filter(|w| maps.node_to_entity.contains_key(w))
        .count();
    assert_eq!(
        rendered,
        workers.len(),
        "all {} workers should be rendered at the top level, only {} were",
        workers.len(),
        rendered
    );
}
