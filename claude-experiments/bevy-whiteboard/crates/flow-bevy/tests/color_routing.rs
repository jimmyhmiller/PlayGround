//! UI test for the color-routing bug.
//!
//! Scenario built entirely through simulated UI clicks:
//!   Gen_red (slot 0) ─┐
//!                     ├─► Router ─┬─► Queue_red    (slot 0)
//!   Gen_yellow (slot 1) ┘         └─► Queue_yellow (slot 1)
//!
//! Expected behaviour: each queue only receives packets that originated at
//! its matching-colour generator. I.e. Queue_red's inbox should be all red,
//! Queue_yellow's all yellow.
//!
//! Observed behaviour (as of writing): the flow router gadget emits every
//! input to `EmitTo::DefaultOut` — the first outbound edge — so all packets
//! funnel into whichever queue happened to be connected first. The other
//! queue receives nothing.
//!
//! Two layers would need to change to make color routing actually work:
//!
//!  1. **Packets must carry a colour tag in their sim payload.** Today a
//!     Generator emits `packet(nil)`; it would need to emit
//!     `packet(slot_idx)` where `slot_idx` is the generator's data-palette
//!     slot. That requires `ActiveSlot` to reach `gen_generator()` at
//!     spawn time, and the node to persist the slot in a sim slot.
//!  2. **The router must match on payload colour when forwarding.**
//!     Either with per-colour rules keyed on pattern, or by querying
//!     outbound-neighbour colour slots — either way, a new `EmitTo`
//!     variant beyond `DefaultOut` is needed.
//!
//! This test captures the expectation so the gap is visible. It will fail
//! until the two changes above land.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Event;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::{ColorSwatch, ToolBtn};
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn latest_of_kind(app: &App, kind: Kind) -> flow::NodeId {
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
        .expect("no node of that kind in sim")
}

fn drop_at(app: &mut App, slot: usize, kind: Kind, pos: Vec2) -> (flow::NodeId, Vec2) {
    click_by_marker::<ColorSwatch, _>(app, |s| s.0 == slot);
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(kind));
    simulate_canvas_click(app, pos);
    (latest_of_kind(app, kind), pos)
}

fn drop_router_at(app: &mut App, pos: Vec2) -> (flow::NodeId, Vec2) {
    // Routers are neutral — swatch doesn't matter, but we still click one
    // for the interaction to look realistic.
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Drop(Kind::Router));
    simulate_canvas_click(app, pos);
    (latest_of_kind(app, Kind::Router), pos)
}

fn connect(app: &mut App, from_xy: Vec2, to_xy: Vec2) {
    click_by_marker::<ToolBtn, _>(app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(app, from_xy);
    simulate_canvas_click(app, to_xy);
}

#[test]
#[ignore = "Pattern-matches PacketEmitted from==router directly; with composites the emit comes from `router::F` (inner Filter). Also reads `queue.slots[len]` directly on the shim — len lives on the inner Buffer. Re-enable after rewriting to use `sim.compound_outermost` for event matching and `read_slot_resolved` for slot access."]
fn yellow_packets_route_to_yellow_queue() {
    let mut app = make_app();

    // Build the diamond offset well below the seeded demo chain so hit-tests
    // pick our fresh nodes.
    // `simulate_canvas_click` now panics loudly if a target projects off-
    // viewport or lands under a UI element, so tests fail at the offending
    // call site instead of silently dropping the click. Positions below
    // are in the safe band (left-of-palette, inside a 1400×900 window).
    let (_gen_red, gen_red_xy) = drop_at(&mut app, 0, Kind::Generator, Vec2::new(-500.0, -100.0));
    let (_gen_yel, gen_yel_xy) = drop_at(&mut app, 1, Kind::Generator, Vec2::new(-500.0, -300.0));
    let (router, router_xy) = drop_router_at(&mut app, Vec2::new(-100.0, -200.0));
    let (queue_red, queue_red_xy) = drop_at(&mut app, 0, Kind::Queue, Vec2::new(300.0, -100.0));
    let (queue_yel, queue_yel_xy) = drop_at(&mut app, 1, Kind::Queue, Vec2::new(300.0, -300.0));

    connect(&mut app, gen_red_xy, router_xy);
    connect(&mut app, gen_yel_xy, router_xy);
    connect(&mut app, router_xy, queue_red_xy);
    connect(&mut app, router_xy, queue_yel_xy);

    // Let the sim run long enough for a few dozen emissions from each gen.
    // Default gen period is 100ms, so 3s ≈ 30 emissions per generator.
    advance_sim_ns(&mut app, 3_000_000_000);

    // Count packets the router forwarded to each downstream queue.
    let sim = &app.world().resource::<FlowSim>();
    let mut to_red = 0usize;
    let mut to_yel = 0usize;
    for ev in sim.log.events.iter() {
        if let Event::PacketEmitted { from, to, .. } = ev {
            if *from == router {
                if *to == queue_red {
                    to_red += 1;
                } else if *to == queue_yel {
                    to_yel += 1;
                }
            }
        }
    }

    // Diagnostic: full edge list around router + queues.
    {
        let sim = &app.world().resource::<FlowSim>();
        eprintln!("router id: {:?}, queue_red: {:?}, queue_yel: {:?}",
            router, queue_red, queue_yel);
        eprintln!("router name: {:?}", sim.nodes.get(&router).map(|n| &n.name));
        eprintln!("queue_red name: {:?}", sim.nodes.get(&queue_red).map(|n| &n.name));
        eprintln!("queue_yel name: {:?}", sim.nodes.get(&queue_yel).map(|n| &n.name));
        for (eid, edge) in sim.edges.iter() {
            if edge.from == router || edge.to == router {
                eprintln!("  edge {:?} {} -> {}", eid,
                    sim.nodes.get(&edge.from).map(|n| n.name.as_str()).unwrap_or("?"),
                    sim.nodes.get(&edge.to).map(|n| n.name.as_str()).unwrap_or("?"));
            }
        }
    }
    // Coarse check: both queues should receive packets — with two upstream
    // generators and two downstream queues, no reasonable routing strategy
    // leaves one queue empty. (DefaultOut fails this; round-robin and
    // colour-matching both pass it.)
    assert!(to_red > 0, "Queue_red received 0 packets from the router");
    assert!(to_yel > 0, "Queue_yellow received 0 packets from the router");

    // Fine check: with colour-matched routing, each queue should receive
    // roughly as many packets as its own-colour generator produced (both
    // gens run at the same rate, so ~50/50 ± jitter). If one queue holds
    // almost everything, routing is blind.
    let total = to_red + to_yel;
    let ratio_red = to_red as f32 / total as f32;
    assert!(
        (0.3..=0.7).contains(&ratio_red),
        "Router split is lopsided — {} to Queue_red vs {} to Queue_yellow \
         (ratio {:.2}). Two equal-rate generators should produce a ~50/50 \
         split under colour-matched routing.",
        to_red, to_yel, ratio_red
    );

    // Bonus: queues should end up holding roughly equal totals (since both
    // generators emit at the same rate). This would catch the case where
    // the sim silently merges both colours into one queue.
    let queue_red_fill = sim
        .nodes
        .get(&queue_red)
        .and_then(|n| n.slots.get("len"))
        .and_then(|v| match v { flow::Value::Int(i) => Some(*i as usize), _ => None })
        .unwrap_or(0);
    let queue_yel_fill = sim
        .nodes
        .get(&queue_yel)
        .and_then(|n| n.slots.get("len"))
        .and_then(|v| match v { flow::Value::Int(i) => Some(*i as usize), _ => None })
        .unwrap_or(0);
    assert!(
        queue_red_fill > 0 && queue_yel_fill > 0,
        "Both queue buffers should hold at least one packet — \
         red={} yellow={}",
        queue_red_fill, queue_yel_fill
    );
}
