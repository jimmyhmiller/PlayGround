//! When a user-scheduled timeline event fires, the visual layer
//! drops every pending in-flight packet so the canvas reflects the
//! new sim state instead of replaying pre-change traffic.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow::Value;
use flow_bevy::bridge::{FlowSim, SimClock};
use flow_bevy::edges::VisualTimelineRes;

/// Number of packets currently in the visual timeline. Used to be a
/// `Query<&TravelingPacket>::iter().count()` over per-packet entities;
/// the entities are gone now, so we count records on the timeline
/// resource directly. Same data, no entity churn.
fn count_packets(app: &mut App) -> usize {
    app.world().resource::<VisualTimelineRes>().0.packets.len()
}

/// Pin visual_now to sim_now after each chunk (in headless tests
/// Bevy's Time delta is near-zero, so advance_visual_clock alone
/// won't keep up).
fn step_locked(app: &mut App, total_ns: u64) {
    let chunk_ns: u64 = 50_000_000;
    let mut remaining = total_ns;
    while remaining > 0 {
        let step = remaining.min(chunk_ns);
        {
            let mut flow = app.world_mut().resource_mut::<FlowSim>();
            let target = flow.now_ns + step;
            flow.run_until(target);
        }
        let now_ns = app.world().resource::<FlowSim>().now_ns;
        app.world_mut().resource_mut::<SimClock>().visual_now = now_ns as f64 / 1e9;
        app.update();
        remaining -= step;
    }
}

#[test]
fn timeline_event_clears_pending_visuals() {
    let mut app = make_app();
    // Use the live default k so the visual queue actually backs up
    // — at k=1 there's nothing meaningful to drop.
    {
        let mut tl = app.world_mut().resource_mut::<VisualTimelineRes>();
        tl.0.set_k(410.0);
    }

    // Build a tiny chain that emits packets continuously: a
    // self-ticking node that sends to a sink.
    let (source, sink) = {
        use flow::rule::{Effect, EmitTo, Rule, When};
        use flow::value::Pattern;
        use flow::expr::Expr;
        use std::collections::BTreeMap;

        let mut sim = app.world_mut().resource_mut::<FlowSim>();

        // Sink: just a node with a flag we can flip via timeline.
        let sink = sim.add_node(
            "sink",
            BTreeMap::from([("flag".to_string(), Value::Int(0))]),
            Vec::new(),
        );
        // Source: emits a packet to sink every 50ms.
        let source = sim.add_node(
            "source",
            BTreeMap::new(),
            vec![Rule::new("tick")
                .when(When::input(Pattern::variant("tick", Pattern::wild())))
                .do_(Effect::emit(
                    Expr::variant("packet", Expr::lit(Value::Int(0))),
                    EmitTo::ToTarget("sink".into()),
                ))
                .do_(Effect::emit(
                    Expr::variant("tick", Expr::lit(Value::Nil)),
                    EmitTo::ToTargetExpr(Expr::self_ref()),
                ))],
        );
        sim.add_edge(source, sink, Expr::int(1_000_000)); // 1ms
        sim.add_edge(source, source, Expr::int(50_000_000)); // 50ms self-tick
        sim.inject(source, Value::variant("tick", Value::Nil));

        // Schedule a timeline event 1.5s out.
        sim.timeline.schedule(1_500_000_000, sink, "flag".into(), Value::Int(1));
        (source, sink)
    };
    let _ = (source, sink);

    // Drive to ~1.0s — well past the visual queue's steady-state
    // fill window (~410 ms) but well before the event at 1.5s.
    step_locked(&mut app, 1_000_000_000);
    let pre_event = count_packets(&mut app);
    assert!(pre_event >= 4, "expected backlog before event, got {}", pre_event);

    // Cross the firing instant in a tight chunk so we sample
    // immediately after the event fires.
    {
        let mut flow = app.world_mut().resource_mut::<FlowSim>();
        let target = 1_510_000_000u64; // 10ms past the event at 1.5s
        flow.run_until(target);
    }
    let now_ns = app.world().resource::<FlowSim>().now_ns;
    app.world_mut().resource_mut::<SimClock>().visual_now = now_ns as f64 / 1e9;
    app.update();

    // The drop only nukes the FUTURE-QUEUED backlog. Packets
    // currently animating (`emit_real <= visual_now`) and
    // already-arrived ones are kept — they're real recent past, the
    // user is watching them mid-flight.
    let visual_now = app.world().resource::<SimClock>().visual_now;
    let tl = &app.world().resource::<VisualTimelineRes>().0;
    let future_queued = tl.packets.iter()
        .filter(|p| p.emit_real > visual_now + 1e-6)
        .count();
    assert_eq!(
        future_queued, 0,
        "future-queued packets (emit_real > visual_now) should be dropped \
         after TimelineEventFired; visual_now={}, packets={:?}",
        visual_now,
        tl.packets.iter().map(|p| (p.emit_real, p.arrive_real)).collect::<Vec<_>>()
    );

    // Sanity: the in-flight ones (emit_real <= visual_now < arrive_real)
    // must have survived. With pre_event >= 4 and a 10ms post-event
    // sim window, a chunk of those should still be visible.
    let in_flight = tl.packets.iter()
        .filter(|p| p.emit_real <= visual_now && p.arrive_real > visual_now)
        .count();
    assert!(
        in_flight > 0,
        "drop should NOT have killed the currently-animating packets; \
         pre_event was {}, in_flight after is {}", pre_event, in_flight
    );
}

/// Manual edits via `Sim::user_edit_slot` (the path the inspector
/// takes for toggle clicks and slider drags) emit the same kind of
/// boundary as a scheduled timeline event firing — so the visual
/// queue's future-clamped backlog gets dropped.
#[test]
fn manual_user_edit_also_drops_pending_visuals() {
    let mut app = make_app();
    {
        let mut tl = app.world_mut().resource_mut::<VisualTimelineRes>();
        tl.0.set_k(410.0);
    }

    let (source, sink) = {
        use flow::rule::{Effect, EmitTo, Rule, When};
        use flow::value::Pattern;
        use flow::expr::Expr;
        use std::collections::BTreeMap;

        let mut sim = app.world_mut().resource_mut::<FlowSim>();
        let sink = sim.add_node(
            "sink",
            BTreeMap::from([("flag".to_string(), Value::Bool(false))]),
            Vec::new(),
        );
        let source = sim.add_node(
            "source",
            BTreeMap::new(),
            vec![Rule::new("tick")
                .when(When::input(Pattern::variant("tick", Pattern::wild())))
                .do_(Effect::emit(
                    Expr::variant("packet", Expr::lit(Value::Int(0))),
                    EmitTo::ToTarget("sink".into()),
                ))
                .do_(Effect::emit(
                    Expr::variant("tick", Expr::lit(Value::Nil)),
                    EmitTo::ToTargetExpr(Expr::self_ref()),
                ))],
        );
        sim.add_edge(source, sink, Expr::int(1_000_000));
        sim.add_edge(source, source, Expr::int(50_000_000));
        sim.inject(source, Value::variant("tick", Value::Nil));
        (source, sink)
    };
    let _ = source;

    // Build the visual backlog up.
    step_locked(&mut app, 1_000_000_000);
    let pre_edit = count_packets(&mut app);
    assert!(pre_edit >= 4, "expected backlog before edit, got {}", pre_edit);

    // Manual user edit — same path the inspector takes for toggles
    // and slider drags. Should emit a UserSlotEdit boundary that the
    // visual layer treats identically to TimelineEventFired.
    {
        let mut sim = app.world_mut().resource_mut::<FlowSim>();
        sim.user_edit_slot(sink, "flag", Value::Bool(true));
    }
    let now_ns = app.world().resource::<FlowSim>().now_ns;
    app.world_mut().resource_mut::<SimClock>().visual_now = now_ns as f64 / 1e9;
    app.update();

    let visual_now = app.world().resource::<SimClock>().visual_now;
    let tl = &app.world().resource::<VisualTimelineRes>().0;
    let future_queued = tl.packets.iter()
        .filter(|p| p.emit_real > visual_now + 1e-6)
        .count();
    assert_eq!(future_queued, 0, "manual edit should drop future-queued backlog");

    let in_flight = tl.packets.iter()
        .filter(|p| p.emit_real <= visual_now && p.arrive_real > visual_now)
        .count();
    assert!(in_flight > 0, "in-flight packets should be kept on manual edit");
}
