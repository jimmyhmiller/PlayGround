//! Diagnostic for the "saga chain with two Worker backends" rendering bug.
//! Loads `examples/saga_chain_repro.whiteboard`, runs the sim a couple of
//! seconds, and prints what's actually happening: packet event counts,
//! per-tag breakdown, slot snapshots, error_counts, and visual timeline
//! ingestion. Lets us see whether the simulator is fine and the bug is
//! purely visual, or whether the sim is itself stuck.

use std::path::PathBuf;

use flow::{Event, Sim, Value};
use flow_bevy::canvas::load_canvas;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

fn slot_i(sim: &Sim, name: &str, slot: &str) -> Option<i64> {
    let n = sim.nodes.values().find(|n| n.name == name)?;
    match n.slots.get(slot) {
        Some(Value::Int(i)) => Some(*i),
        _ => None,
    }
}

#[test]
fn saga_chain_repro_diagnostics() {
    let path = project_root().join("examples/saga_chain_repro.whiteboard");
    let mut canvas = load_canvas(&path, 7).expect("load saga_chain_repro");

    eprintln!("---- nodes ({}) ----", canvas.sim.nodes.len());
    let mut names: Vec<&String> = canvas.sim.nodes.values().map(|n| &n.name).collect();
    names.sort();
    for n in &names { eprintln!("  {}", n); }

    eprintln!("---- edges ({}) ----", canvas.sim.edges.len());
    let by_id: std::collections::HashMap<flow::NodeId, String> =
        canvas.sim.nodes.iter().map(|(id, n)| (*id, n.name.clone())).collect();
    let mut elines: Vec<String> = canvas.sim.edges.iter()
        .map(|(_, e)| format!("  {} -> {}",
            by_id.get(&e.from).cloned().unwrap_or("?".into()),
            by_id.get(&e.to).cloned().unwrap_or("?".into())))
        .collect();
    elines.sort();
    for l in &elines { eprintln!("{}", l); }

    canvas.sim.run_until(canvas.sim.now_ns + 2_500_000_000);

    eprintln!("---- error_counts ----");
    if canvas.sim.error_counts.is_empty() {
        eprintln!("  (none)");
    } else {
        for (k, v) in &canvas.sim.error_counts {
            eprintln!("  {:>30}: {}", k, v);
        }
    }

    eprintln!("---- key slot values ----");
    for (name, slots) in [
        ("WriteClient",  &["completed", "in_flight"][..]),
        ("ReserveInv",   &["done", "compensated", "phase"]),
        ("ChargePay",    &["done", "compensated", "phase"]),
        ("InvShard0",    &["served"]),
        ("PaymentWorker",&["served"]),
    ] {
        eprintln!("  {}:", name);
        for s in slots {
            let v = canvas.sim.nodes.values()
                .find(|n| &n.name == name)
                .and_then(|n| n.slots.get(*s).cloned());
            eprintln!("    {} = {:?}", s, v);
        }
    }

    let mut pkt_count = 0usize;
    let mut by_tag = std::collections::BTreeMap::<String, usize>::new();
    let mut visible_count = 0usize;
    let mut by_edge = std::collections::BTreeMap::<(String, String), usize>::new();
    for ev in &canvas.sim.log.events {
        if let Event::PacketEmitted { from, to, payload, at_ns, arrives_at_ns, .. } = ev {
            pkt_count += 1;
            let tag = if let Value::Variant { tag, .. } = payload {
                tag.clone()
            } else {
                "<non-variant>".into()
            };
            *by_tag.entry(tag.clone()).or_default() += 1;
            if from != to && arrives_at_ns > at_ns && tag != "pull" && tag != "wake" {
                visible_count += 1;
                let k = (
                    by_id.get(from).cloned().unwrap_or("?".into()),
                    by_id.get(to).cloned().unwrap_or("?".into()),
                );
                *by_edge.entry(k).or_default() += 1;
            }
        }
    }
    eprintln!("---- packet events ({} total, {} 'visible') ----", pkt_count, visible_count);
    eprintln!("by tag:");
    for (t, c) in &by_tag { eprintln!("  {:>16}: {}", t, c); }
    eprintln!("visible packets by edge:");
    for ((f, t), c) in &by_edge { eprintln!("  {:>16} -> {:<16}: {}", f, t, c); }

    // Now run the same packets through VisualTimeline and see what it picks up.
    use flow_bevy::visual::VisualTimeline;
    let mut tl = VisualTimeline::new(1.0);
    let mut real_now = 0.0;
    let mut ingested = 0usize;
    for ev in &canvas.sim.log.events {
        if let Event::PacketEmitted { at_ns, .. } = ev {
            real_now = (*at_ns as f64) * 1e-9;
        }
        if tl.ingest(ev, real_now).is_some() {
            ingested += 1;
        }
    }
    eprintln!("---- visual timeline ----");
    eprintln!("ingested = {}, packets.len() = {}", ingested, tl.packets.len());
    for (i, p) in tl.packets.iter().enumerate().take(20) {
        eprintln!("  pkt[{}] from={:?} to={:?} emit={:.3} arrive={:.3}",
            i,
            by_id.get(&p.from).cloned().unwrap_or("?".into()),
            by_id.get(&p.to).cloned().unwrap_or("?".into()),
            p.emit_real, p.arrive_real);
    }
}

/// Baseline: no pre-canvas, just LoadExample. How many packets in 600ms?
#[test]
fn baseline_load_example_only_worker_mode() {
    use std::thread::sleep;
    use std::time::Duration;

    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(flow_bevy::FlowBevyPlugins);
    app.world_mut().resource_mut::<flow_bevy::bridge::SimClock>().multiplier = 1.0;

    let pump = |app: &mut bevy::prelude::App, secs: f64| {
        let frames = (secs * 30.0) as usize;
        for _ in 0..frames {
            app.update();
            sleep(Duration::from_millis(33));
        }
    };
    pump(&mut app, 0.2); // boot

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<flow_bevy::examples::LoadExample>>()
        .write(flow_bevy::examples::LoadExample(flow_bevy::examples::Example::ClientWorker));
    app.update();
    app.update();

    pump(&mut app, 6.0);

    let n = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().0.as_replay().packets.len();
    let visual_now = app.world().resource::<flow_bevy::bridge::SimClock>().visual_now;
    eprintln!("baseline: timeline.packets={}, visual_now={:.3}", n, visual_now);
    let timeline = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().0.as_replay();
    for (i, p) in timeline.packets.iter().enumerate().take(20) {
        eprintln!("  pkt[{}]: emit={:.3} arrive={:.3} from={:?} to={:?}",
            i, p.emit_real, p.arrive_real, p.from, p.to);
    }
}

/// Cross-canvas test: load saga_chain_repro, then fire LoadExample for
/// ClientWorker. After load, the example should produce visible packets
/// in the timeline. Mirrors `region_then_example_worker_mode_packets_visible`
/// but for our minimal repro.
#[test]
fn repro_then_example_worker_mode_packets_visible() {
    use std::thread::sleep;
    use std::time::Duration;

    let path = project_root().join("examples/saga_chain_repro.whiteboard");
    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(flow_bevy::FlowBevyPlugins);
    app.insert_resource(flow_bevy::PendingCanvas(Some(path.clone())))
        .add_systems(bevy::prelude::Startup, flow_bevy::canvas::seed_from_path);
    app.world_mut().resource_mut::<flow_bevy::bridge::SimClock>().multiplier = 1.0;

    let pump = |app: &mut bevy::prelude::App, secs: f64| {
        let frames = (secs * 30.0) as usize;
        for _ in 0..frames {
            app.update();
            sleep(Duration::from_millis(33));
        }
    };
    pump(&mut app, 3.0);

    let repro_packets = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().0.as_replay().packets.len();
    eprintln!("repro phase: timeline.packets.len() = {}", repro_packets);

    let repro_node_ids: std::collections::HashSet<flow::NodeId> = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .keys().copied().collect();
    eprintln!("repro NodeIds: {:?}", repro_node_ids);

    // Click an example. ThreeLaneFanout matches the user's repro screenshot.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<flow_bevy::examples::LoadExample>>()
        .write(flow_bevy::examples::LoadExample(flow_bevy::examples::Example::ThreeLaneFanout));
    app.update();
    app.update();

    pump(&mut app, 0.6);

    // Inspect the sim AFTER LoadExample is processed: does it have the
    // ClientWorker nodes/edges? If sim is empty here, the build didn't
    // run (or didn't reach the worker).
    {
        let driver = &mut app.world_mut().resource_mut::<flow_bevy::sim_driver::SimDriverRes>().0;
        let (nc, ec, names) = driver.with_sim_mut(|sim| {
            let names: Vec<String> = sim.nodes.values().map(|n| n.name.clone()).collect();
            (sim.nodes.len(), sim.edges.len(), names)
        });
        eprintln!("[after-load] sim nodes={} edges={} names={:?}", nc, ec, names);
    }

    pump(&mut app, 2.4);

    {
        let driver = &mut app.world_mut().resource_mut::<flow_bevy::sim_driver::SimDriverRes>().0;
        let (nc, ec, names, errors) = driver.with_sim_mut(|sim| {
            let names: Vec<String> = sim.nodes.values().map(|n| n.name.clone()).collect();
            let errors: Vec<(String, u64)> = sim.error_counts.iter()
                .map(|(k, v)| (k.clone(), *v)).collect();
            (sim.nodes.len(), sim.edges.len(), names, errors)
        });
        eprintln!("[steady] sim nodes={} edges={} names={:?} errors={:?}", nc, ec, names, errors);
    }

    let timeline = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().0.as_replay();
    let visual_now = app.world().resource::<flow_bevy::bridge::SimClock>().visual_now;
    let known: std::collections::HashSet<flow::NodeId> = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .keys().copied().collect();
    let total_packets = timeline.packets.len();
    let in_flight_now = timeline.packets.iter()
        .filter(|p| p.emit_real <= visual_now && visual_now < p.arrive_real)
        .count();
    let stale = timeline.packets.iter()
        .filter(|p| !known.contains(&p.from) || !known.contains(&p.to))
        .count();
    eprintln!(
        "after example: timeline.packets={}, in_flight_now={}, visual_now={:.3}, stale={}",
        total_packets, in_flight_now, visual_now, stale,
    );
    eprintln!("current NodeIds: {:?}", known);
    for (i, p) in timeline.packets.iter().enumerate().take(20) {
        eprintln!(
            "  pkt[{}]: emit={:.3} arrive={:.3} from={:?} to={:?} visible_now={}",
            i, p.emit_real, p.arrive_real, p.from, p.to,
            p.emit_real <= visual_now && visual_now < p.arrive_real,
        );
    }

    assert!(total_packets > 5,
        "ClientWorker should produce visible packets after LoadExample on top of saga_chain_repro; got {}",
        total_packets);
}
