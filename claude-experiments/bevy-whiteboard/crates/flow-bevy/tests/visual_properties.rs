//! Pure property tests for the visual timeline.
//!
//! These run against `VisualTimeline` directly — no Bevy world, no
//! example scenarios, no Time/Transform queries. They generate random
//! event streams and assert the F1 invariants hold for every input.
//!
//! Matching property numbering in `src/visual.rs`:
//!
//!   V1 — `arrive_real > emit_real` for every ingested packet.
//!   V2 — every ingested packet corresponds to exactly one
//!        visible `PacketEmitted`.
//!   V3 — `visible_at(t)` only returns packets with
//!        `emit_real <= t < arrive_real`.
//!   V4 — progress is monotonic in `t`: at `emit_real` it's 0.0,
//!        and it grows with `t` up to 1.0 at `arrive_real`.
//!   V5 — no spontaneous creation:
//!        `|visible_at(t)| <= |{p : emit_real(p) <= t}|`.
//!   V6 — ingesting the same event stream twice produces the same
//!        packets vector (determinism).
//!   V7 — control-plane variants (`pull`, `wake`) and self-loops
//!        are filtered out.

use flow::{Event, NodeId, PacketId, Value};
use flow_bevy::visual::{VisualPacket, VisualTimeline};
use proptest::prelude::*;

// ────────────────────────────────────────────────────────────
// Generators
// ────────────────────────────────────────────────────────────

/// Arbitrary payload that the visual layer might receive. Mix of
/// data packets (visible) and control plane (filtered).
fn arb_payload() -> impl Strategy<Value = Value> {
    prop_oneof![
        // data packet with a slot 0..3
        (0i64..4).prop_map(|slot| Value::variant("packet", Value::Int(slot))),
        (0i64..4).prop_map(|slot| Value::variant("req",    Value::Int(slot))),
        (0i64..4).prop_map(|slot| Value::variant("resp",   Value::Int(slot))),
        // control-plane — should be filtered by V7
        Just(Value::variant("pull", Value::Nil)),
        Just(Value::variant("wake", Value::Nil)),
    ]
}

/// Arbitrary `PacketEmitted`. Node ids drawn from a small space so
/// self-loops and matching pairs both occur. Latency is always > 0.
fn arb_emit() -> impl Strategy<Value = Event> {
    (
        0u64..1_000,              // packet id
        0u64..6,                  // from
        0u64..6,                  // to
        0u64..10_000_000,         // at_ns, up to 10 ms
        1u64..2_000_000,          // latency, up to 2 ms
        arb_payload(),
    ).prop_map(|(pid, from, to, at, latency, payload)| {
        Event::PacketEmitted {
            packet: PacketId(pid),
            from: NodeId(from),
            to: NodeId(to),
            at_ns: at,
            arrives_at_ns: at + latency,
            payload,
        }
    })
}

/// A stream of `N` emit events. Not sorted — the timeline shouldn't
/// require chronological order for correctness of the invariants.
fn arb_event_stream(max: usize) -> impl Strategy<Value = Vec<Event>> {
    prop::collection::vec(arb_emit(), 0..=max)
}

// ────────────────────────────────────────────────────────────
// V1 — every packet has positive duration
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v1_positive_duration(events in arb_event_stream(50), k in 0.5f64..2000.0) {
        let mut tl = VisualTimeline::new(k);
        for ev in &events { tl.ingest(ev, 0.0); }
        for p in &tl.packets {
            prop_assert!(p.arrive_real > p.emit_real,
                "V1 violated: emit={} arrive={}", p.emit_real, p.arrive_real);
        }
    }
}

// ────────────────────────────────────────────────────────────
// V2 — each ingested packet corresponds to one visible emit
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v2_each_packet_has_one_matching_emit(events in arb_event_stream(50), k in 0.5f64..2000.0) {
        let mut tl = VisualTimeline::new(k);
        for ev in &events { tl.ingest(ev, 0.0); }

        // Every VisualPacket must be traceable to an event in the
        // input stream by (packet_id, from, to, payload).
        for vp in &tl.packets {
            let matched = events.iter().filter(|ev| match ev {
                Event::PacketEmitted { packet, from, to, payload, .. } => {
                    *packet == vp.packet_id
                        && *from == vp.from
                        && *to == vp.to
                        && *payload == vp.payload
                }
                _ => false,
            }).count();
            prop_assert!(matched >= 1,
                "V2 violated: no matching emit for packet_id {:?}", vp.packet_id);
        }

        // Conversely, the COUNT of visible-eligible events in the
        // stream must equal the packets vector length. We filter
        // events by the same rule as `is_visible_event`.
        let expected: usize = events.iter().filter(|ev| VisualTimeline::is_visible_event(ev)
            && match ev {
                Event::PacketEmitted { at_ns, arrives_at_ns, .. } => arrives_at_ns > at_ns,
                _ => false,
            }
        ).count();
        prop_assert_eq!(tl.packets.len(), expected,
            "V2 violated: packet count {} != expected visible-eligible emits {}",
            tl.packets.len(), expected);
    }
}

// ────────────────────────────────────────────────────────────
// V3 — visible_at returns packets strictly inside the window
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v3_visible_at_respects_window(
        events in arb_event_stream(30),
        k in 0.5f64..2000.0,
        probes in prop::collection::vec(0f64..10.0, 1..10),
    ) {
        let mut tl = VisualTimeline::new(k);
        for ev in &events { tl.ingest(ev, 0.0); }
        for t in probes {
            for (p, prog) in tl.visible_at(t) {
                prop_assert!(t >= p.emit_real,
                    "V3 violated: t={} < emit_real={}", t, p.emit_real);
                prop_assert!(t < p.arrive_real,
                    "V3 violated: t={} >= arrive_real={}", t, p.arrive_real);
                prop_assert!(prog >= 0.0 && prog <= 1.0,
                    "V3 violated: progress {} out of [0,1]", prog);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// V4 — progress monotonicity + endpoints
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v4_progress_starts_at_zero_grows_to_one(
        events in arb_event_stream(20),
        k in 0.5f64..2000.0,
    ) {
        let mut tl = VisualTimeline::new(k);
        for ev in &events { tl.ingest(ev, 0.0); }

        for p in &tl.packets {
            // Identify the outer packet by pointer equality. Fuzzed
            // event streams can emit multiple VisualPackets that share
            // `(packet_id, from, to)` with different durations — a
            // key-based filter would alias them and sample progress
            // from the wrong packet's window.
            let same = |vp: &VisualPacket| std::ptr::eq(vp, p);

            // At emit_real, progress = 0.
            let visible_at_emit: Vec<_> = tl.visible_at(p.emit_real)
                .filter(|(vp, _)| same(vp))
                .collect();
            if let Some((_, prog0)) = visible_at_emit.first() {
                prop_assert!(*prog0 < 1e-3,
                    "V4: progress at emit_real should be ~0, got {}", prog0);
            }
            // Just before arrive_real, progress approaches 1.
            let denom = p.arrive_real - p.emit_real;
            let almost = p.arrive_real - denom * 0.001;
            let visible_at_end: Vec<_> = tl.visible_at(almost)
                .filter(|(vp, _)| same(vp))
                .collect();
            if let Some((_, prog_end)) = visible_at_end.first() {
                prop_assert!(*prog_end > 0.99,
                    "V4: progress near arrive_real should be ~1, got {}", prog_end);
            }

            // Monotonicity: for two times inside the window,
            // later t => higher progress.
            let t_a = p.emit_real + denom * 0.25;
            let t_b = p.emit_real + denom * 0.75;
            let prog_a = tl.visible_at(t_a)
                .filter(|(vp, _)| same(vp))
                .map(|(_, pr)| pr).next();
            let prog_b = tl.visible_at(t_b)
                .filter(|(vp, _)| same(vp))
                .map(|(_, pr)| pr).next();
            if let (Some(a), Some(b)) = (prog_a, prog_b) {
                prop_assert!(b > a,
                    "V4 monotonicity: progress(t_b) {} should exceed progress(t_a) {}",
                    b, a);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// V5 — no spontaneous creation: visible count bounded by emits
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v5_no_spontaneous_visible(
        events in arb_event_stream(40),
        k in 0.5f64..2000.0,
        probes in prop::collection::vec(0f64..10.0, 1..10),
    ) {
        let mut tl = VisualTimeline::new(k);
        for ev in &events { tl.ingest(ev, 0.0); }
        for t in probes {
            let visible = tl.visible_at(t).count();
            let already_emitted = tl.packets.iter()
                .filter(|p| p.emit_real <= t)
                .count();
            prop_assert!(visible <= already_emitted,
                "V5: {} visible but only {} emitted by t={}",
                visible, already_emitted, t);
        }
    }
}

// ────────────────────────────────────────────────────────────
// V6 — determinism: same events twice = same packets
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v6_ingestion_is_deterministic(
        events in arb_event_stream(30),
        k in 0.5f64..2000.0,
    ) {
        let mut a = VisualTimeline::new(k);
        let mut b = VisualTimeline::new(k);
        for ev in &events { a.ingest(ev, 0.0); }
        for ev in &events { b.ingest(ev, 0.0); }
        prop_assert_eq!(&a.packets, &b.packets);
    }
}

// ────────────────────────────────────────────────────────────
// V7 — control-plane & self-loops filtered
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn v7_filter_removes_control_and_self_loops(
        events in arb_event_stream(50),
        k in 0.5f64..2000.0,
    ) {
        let mut tl = VisualTimeline::new(k);
        for ev in &events { tl.ingest(ev, 0.0); }
        for vp in &tl.packets {
            // No self-loops.
            prop_assert!(vp.from != vp.to);
            // No control-plane variants.
            if let Value::Variant { tag, .. } = &vp.payload {
                prop_assert!(tag != "pull" && tag != "wake",
                    "V7: {} payload leaked into timeline", tag);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// Causality invariant C — if events are causally ordered in
// sim time, their visual windows are causally ordered in real
// time too. This encodes the user's "1 in → 1 out" physical
// constraint: if sim emits B only after A arrives, then B's
// visual emit_real is >= A's visual arrive_real.
// ────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn causality_preserved_under_scaling(
        // Two events where B's sim at_ns >= A's arrives_at_ns.
        a_at in 0u64..1_000_000,
        a_lat in 1u64..1_000_000,
        b_gap in 0u64..500_000, // sim gap between A's arrival and B's emit
        b_lat in 1u64..1_000_000,
        k in 0.5f64..2000.0,
    ) {
        let a_arr = a_at + a_lat;
        let b_at = a_arr + b_gap;
        let a = Event::PacketEmitted {
            packet: PacketId(1), from: NodeId(1), to: NodeId(2),
            at_ns: a_at, arrives_at_ns: a_arr,
            payload: Value::variant("packet", Value::Int(0)),
        };
        let b = Event::PacketEmitted {
            packet: PacketId(2), from: NodeId(2), to: NodeId(3),
            at_ns: b_at, arrives_at_ns: b_at + b_lat,
            payload: Value::variant("packet", Value::Int(0)),
        };
        let mut tl = VisualTimeline::new(k);
        tl.ingest(&a, 0.0).unwrap();
        tl.ingest(&b, 0.0).unwrap();
        let a_vp = &tl.packets[0];
        let b_vp = &tl.packets[1];
        prop_assert!(b_vp.emit_real >= a_vp.arrive_real - 1e-9,
            "causality: B.emit_real {} < A.arrive_real {}", b_vp.emit_real, a_vp.arrive_real);
    }
}
