//! Diagnostic: what payloads does the Router actually emit, and what
//! visual color would each one resolve to?
//!
//! User report: "Router does not work. It is spitting out green in all
//! directions" — green is data slot 2 (moss/olive). We dump every
//! PacketEmitted event whose `from` is inside the router compound and
//! check the payload slot.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

fn router_inner_ids(app: &App) -> Vec<flow::NodeId> {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", Kind::Router.label());
    // Find the router shim name, then collect all inner nodes.
    let shim_name = sim
        .nodes
        .iter()
        .find(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(_, n)| n.name.clone())
        .expect("no router");
    let inner_prefix = format!("{}::", shim_name);
    sim.nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&inner_prefix))
        .map(|(id, _)| *id)
        .collect()
}

fn packet_slot(v: &Value) -> Option<i64> {
    let (tag, inner) = v.as_variant()?;
    if tag != "packet" && tag != "req" {
        return None;
    }
    match inner {
        Value::Int(i) => Some(*i),
        _ => None,
    }
}

#[test]
fn router_emits_all_three_colors_not_just_one() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    advance_sim_ns(&mut app, 3_000_000_000);

    let inner = router_inner_ids(&app);
    let sim = app.world().resource::<FlowSim>();

    // Name lookup for nice diagnostics.
    let name_of = |id: flow::NodeId| sim.nodes.get(&id).map(|n| n.name.clone()).unwrap_or_default();

    let mut slot_counts: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    let mut by_from: std::collections::BTreeMap<String, std::collections::BTreeMap<i64, usize>> =
        std::collections::BTreeMap::new();
    for ev in sim.log.iter() {
        let flow::Event::PacketEmitted { from, payload, .. } = ev else { continue };
        if !inner.contains(from) {
            continue;
        }
        let Some(slot) = packet_slot(payload) else { continue };
        *slot_counts.entry(slot).or_default() += 1;
        *by_from.entry(name_of(*from)).or_default().entry(slot).or_default() += 1;
    }

    eprintln!("router inner emits by slot: {:?}", slot_counts);
    for (from, slots) in &by_from {
        eprintln!("  {} -> {:?}", from, slots);
    }

    // The router broadcasts all three generator colors. We must see
    // packets of slot 0, 1, AND 2 leave the router — not just one
    // color. If only slot 2 (green) shows up, the payload is being
    // rewritten somewhere inside the router.
    assert!(slot_counts.contains_key(&0), "router never emitted slot 0 (red): {:?}", slot_counts);
    assert!(slot_counts.contains_key(&1), "router never emitted slot 1 (amber): {:?}", slot_counts);
    assert!(slot_counts.contains_key(&2), "router never emitted slot 2 (green): {:?}", slot_counts);
}

/// The router routes by colour: every packet leaving the Router shim
/// must land on the downstream queue whose colour matches the packet.
/// No packet should ever travel toward a wrong-colour lane (the old
/// broadcast behaviour, which the user saw as "green in all directions").
#[test]
fn router_never_sends_packet_to_wrong_color_lane() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    advance_sim_ns(&mut app, 3_000_000_000);

    let sim = app.world().resource::<FlowSim>();

    // Router shim + its inner node ids. The colour-matched boundary
    // fan-out (`to out output matching`) records `from` = the router
    // SHIM (like ToOutPort), but accept inner ids too for robustness.
    let prefix = format!("{}_", Kind::Router.label());
    let (router_shim, router_name) = sim
        .nodes
        .iter()
        .find(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::") && n.is_compound())
        .map(|(id, n)| (*id, n.name.clone()))
        .expect("no router shim");
    let router_inner_prefix = format!("{}::", router_name);
    let mut router_sources: Vec<flow::NodeId> = sim
        .nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&router_inner_prefix))
        .map(|(id, _)| *id)
        .collect();
    router_sources.push(router_shim);
    let router_sources = router_sources;
    // The queue shims are the router's external downstream peers.
    let queue_prefix = format!("{}_", Kind::Queue.label());
    let queue_shims: std::collections::HashSet<flow::NodeId> = sim
        .nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&queue_prefix) && !n.name.contains("::") && n.is_compound())
        .map(|(id, _)| *id)
        .collect();

    // For each downstream queue shim, its colour = inner Filter's `match`.
    let lane_color = |nid: flow::NodeId| -> Option<i64> {
        let n = sim.nodes.get(&nid)?;
        let inner_prefix = format!("{}::", n.name);
        sim.nodes
            .values()
            .filter(|m| m.name.starts_with(&inner_prefix))
            .find_map(|m| match m.slots.get("match") {
                Some(Value::Int(i)) => Some(*i),
                _ => None,
            })
    };

    let mut crossings: Vec<(i64, i64)> = Vec::new();
    let mut total = 0usize;
    for ev in sim.log.iter() {
        let flow::Event::PacketEmitted { from, to, payload, .. } = ev else { continue };
        // Forward emit out of the router into one of its queues.
        if !router_sources.contains(from) || !queue_shims.contains(to) {
            continue;
        }
        let Some(pkt) = packet_slot(payload) else { continue };
        total += 1;
        if let Some(lane) = lane_color(*to) {
            if lane != pkt {
                crossings.push((pkt, lane));
            }
        }
    }
    eprintln!("router→queue emits: {} total, {} crossings", total, crossings.len());
    assert!(total > 0, "router emitted nothing to its queues");
    assert!(
        crossings.is_empty(),
        "router sent packets to wrong-colour lanes (packet_color, lane_color): {:?}",
        crossings.iter().take(10).collect::<Vec<_>>(),
    );
}
