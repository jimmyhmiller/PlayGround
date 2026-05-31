//! Quick smoke test: verify visible edges exist after loading examples.

use bevy::prelude::*;
use flow::Sim;
use flow_bevy::bridge::{FlowSim, EntityMaps};
use flow_bevy::edges::HiddenEdges;
use flow_bevy::examples::{Example, LoadExample};

mod common;

fn count_visible_edges(app: &App) -> usize {
    let sim: &Sim = app.world().resource::<FlowSim>();
    let hidden: &HiddenEdges = app.world().resource();
    let maps: &EntityMaps = app.world().resource();
    sim.edges.iter()
        .filter(|(eid, e)| {
            !hidden.set.contains(eid)
                && maps.node_to_entity.contains_key(&e.from)
                && maps.node_to_entity.contains_key(&e.to)
                && e.from != e.to
        })
        .count()
}

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

/// `handle_load_example` queues `commands.insert_resource(membership)`
/// — if `sync_canvas_population` runs before that command flushes, it
/// sees empty membership and spawns Bevy entities for every inner
/// node. The fix is to order `handle_load_example` before
/// `sync_canvas_population` so commands flush at the boundary. Catch
/// regressions by asserting no inner-entity spawns survive a single
/// extra update — if ordering broke and we double-spawned, the next
/// frame's despawn would race against the failed first-frame spawn.
#[test]
fn load_example_runs_before_population_syncer() {
    use flow_bevy::compound::CompoundMembership;
    let mut app = common::make_app();
    load(&mut app, Example::ThreeLaneFanout);
    // After exactly two updates, the membership and entity maps must
    // be in sync. No inner entities, all shims present.
    let sim: &Sim = app.world().resource::<FlowSim>();
    let maps: &EntityMaps = app.world().resource();
    let membership: &CompoundMembership = app.world().resource();
    // Membership must have at least one inner→compound mapping —
    // otherwise the recompute didn't happen.
    let any_mapped = sim.nodes.iter()
        .any(|(nid, _)| membership.parent_of(*nid).is_some());
    assert!(any_mapped, "CompoundMembership wasn't recomputed after example load");
    // No inner entities at top scope.
    let leaked: Vec<String> = sim.nodes.values()
        .filter(|n| n.name.contains("::") && maps.node_to_entity.contains_key(&n.id))
        .map(|n| n.name.clone())
        .collect();
    assert!(leaked.is_empty(),
        "inner-of-compound entities leaked into top scope: {:?}", leaked);
}

/// Every compound shim spawned by an example must carry the
/// `CompoundBodyMarker` component — without it the
/// `drill_in_on_double_click` handler can't find the entity and
/// double-clicking a composite does nothing.
#[test]
fn compound_shims_carry_drill_in_marker() {
    use flow_bevy::compound::CompoundBodyMarker;
    let mut app = common::make_app();
    for ex in [
        Example::ThreeLaneFanout,
        Example::ClientWorker,
        Example::ClientRouterWorker,
        Example::TwoClientsOneWorker,
        Example::ClientQueueWorker,
        Example::BackoffHerd,
    ] {
        load(&mut app, ex);
        app.update();
        app.update();
        let sim: &Sim = app.world().resource::<FlowSim>();
        let maps: &EntityMaps = app.world().resource();
        let mut missing: Vec<String> = Vec::new();
        for (nid, node) in sim.nodes.iter() {
            if !node.is_compound() { continue; }
            // Only check top-level compounds — nested compounds (if
            // any) sit inside a parent compound and don't have an
            // entity at top scope by design.
            if node.name.contains("::") { continue; }
            let Some(entity) = maps.node_to_entity.get(nid).copied() else {
                missing.push(format!("{}: no entity", node.name));
                continue;
            };
            if app.world().entity(entity).get::<CompoundBodyMarker>().is_none() {
                missing.push(format!("{}: no CompoundBodyMarker", node.name));
            }
        }
        assert!(missing.is_empty(),
            "{:?}: compound shims missing drill-in marker: {:?}", ex, missing);
    }
}

/// Inner-of-compound nodes should NOT have Bevy entities at top
/// scope — the user sees a single composite shim, not a soup of
/// primitives. Failed when `handle_load_example` forgot to recompute
/// `CompoundMembership` after building the example.
#[test]
fn no_inner_entities_at_top_scope() {
    let mut app = common::make_app();
    for ex in [
        Example::ThreeLaneFanout,
        Example::ClientWorker,
        Example::ClientRouterWorker,
        Example::TwoClientsOneWorker,
        Example::ClientQueueWorker,
        Example::BackoffHerd,
    ] {
        load(&mut app, ex);
        // sync_canvas_population reacts to the membership-changed flag
        // a frame later; give it two more updates to settle.
        app.update();
        app.update();
        let sim: &Sim = app.world().resource::<FlowSim>();
        let maps: &EntityMaps = app.world().resource();
        let inner_with_entities: Vec<String> = sim.nodes.values()
            .filter(|n| n.name.contains("::"))
            .filter(|n| maps.node_to_entity.contains_key(&n.id))
            .map(|n| n.name.clone())
            .collect();
        assert!(
            inner_with_entities.is_empty(),
            "{:?}: inner-of-compound nodes have Bevy entities at top scope: {:?}",
            ex, inner_with_entities,
        );
    }
}

#[test]
fn each_example_has_visible_edges() {
    let mut app = common::make_app();
    for ex in [
        Example::ThreeLaneFanout,
        Example::ClientWorker,
        Example::ClientRouterWorker,
        Example::TwoClientsOneWorker,
        Example::ClientQueueWorker,
        Example::BackoffHerd,
    ] {
        load(&mut app, ex);
        let count = count_visible_edges(&app);
        assert!(count > 0, "{:?}: no visible edges", ex);
        eprintln!("{:?}: visible_edges = {}", ex, count);
    }
}
