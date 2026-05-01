//! Rewind: restore the sim to a past moment so the visual layer can
//! re-render that frame.
//!
//! There is one path. Pick the deepest captured snapshot at or before
//! `target_ns - lookback_ns` (with a fallback chain if none qualifies),
//! restore, run forward to `target_ns`, bump `rewind_epoch`. The
//! visual layer reacts to the epoch bump by calling
//! `VisualStrategy::invalidate` and consuming the events that the
//! sim emitted while running forward — same forward-play machinery as
//! the live frame loop.
//!
//! `lookback_ns` is sized by the active visual strategy (`SimMirror`
//! returns 0; event-history strategies return enough to cover their
//! visibility window plus chain reach). The host pushes the value to
//! `SimControl.rewind_lookback_ns_bits` each frame; the worker reads
//! it on rewind.

use std::sync::Arc;

use flow::{Edge, Event, Sim, Snapshot, SnapshotRing, Time};

/// Compute `max(latency_ns)` across non-self-loop edges. The visual
/// strategy combines this with `k` to size its lookback window.
/// Self-loops are skipped — their latencies represent service times
/// and dwells that the visual filter drops (`from == to`).
pub fn max_edge_latency_ns(edges: impl IntoIterator<Item = (impl std::borrow::Borrow<Edge>,)>) -> u64 {
    let mut max_lat: u64 = 0;
    for (edge,) in edges {
        let edge = edge.borrow();
        if edge.from == edge.to {
            continue;
        }
        if let flow::Expr::Lit(flow::Value::Int(n)) = &edge.latency_ns {
            if *n > 0 && (*n as u64) > max_lat {
                max_lat = *n as u64;
            }
        }
    }
    max_lat
}

/// Convenience for reading `max_edge_latency_ns` straight off a
/// `Sim`. Filters self-loops; ignores edges with non-literal latency
/// exprs (caller falls back to a constant ceiling in those cases).
pub fn sim_max_edge_latency_ns(sim: &Sim) -> u64 {
    let mut max_lat: u64 = 0;
    for edge in sim.edges.values() {
        if edge.from == edge.to {
            continue;
        }
        if let flow::Expr::Lit(flow::Value::Int(n)) = &edge.latency_ns {
            if *n > 0 && (*n as u64) > max_lat {
                max_lat = *n as u64;
            }
        }
    }
    max_lat
}

/// Pick the latest entry whose `sim_now_ns ≤ ceiling`. Falls back to
/// the sticky anchor if no entry qualifies — but only if the anchor
/// itself is `≤ ceiling`.
fn snap_at_or_before(ring: &SnapshotRing, ceiling: Time) -> Option<Snapshot> {
    ring.entries
        .iter()
        .rev()
        .find(|s| s.sim_now_ns <= ceiling)
        .cloned()
        .or_else(|| {
            ring.anchor
                .as_ref()
                .filter(|a| a.sim_now_ns <= ceiling)
                .cloned()
        })
}

/// Pick a snapshot that covers the strategy's lookback window. Prefer
/// one ≤ `target_ns − lookback_ns` so the run-forward emits the full
/// window of events the visual strategy needs; fall back to the
/// latest snapshot ≤ target_ns when no deep-enough capture exists.
fn pick_snapshot(ring: &SnapshotRing, target_ns: Time, lookback_ns: u64) -> Option<Snapshot> {
    let anchor_ns = target_ns.saturating_sub(lookback_ns);
    let has_topology = |s: &Snapshot| !s.sim.nodes.is_empty();
    ring.entries
        .iter()
        .rev()
        .find(|s| s.sim_now_ns <= anchor_ns && has_topology(s))
        .or_else(|| {
            ring.entries
                .iter()
                .rev()
                .find(|s| s.sim_now_ns <= target_ns && has_topology(s))
        })
        .cloned()
        .or_else(|| snap_at_or_before(ring, target_ns))
}

/// Do the rewind. Restores from a snapshot that covers the lookback
/// window, runs forward to `target_ns`, returns the events the sim
/// emitted during run-forward (including synthesized in-flight events
/// for packets already mid-flight at the snapshot moment — without
/// these, the visual strategy can't "see" the chain heads that
/// preceded the snapshot).
///
/// Returns `None` if no snapshot covers `target_ns` — caller leaves
/// the sim untouched.
pub fn do_rewind(
    sim: &mut Sim,
    ring: &SnapshotRing,
    prev_log_index: &mut u64,
    rewind_epoch: &mut u64,
    target_ns: Time,
    lookback_ns: u64,
) -> Option<Arc<Vec<Event>>> {
    let snap = pick_snapshot(ring, target_ns, lookback_ns)?;
    let debug = std::env::var("FLOW_BEVY_REWIND_DEBUG").is_ok();
    if debug {
        eprintln!(
            "[rewind] target_ns={} lookback_ns={} snap_ns={}",
            target_ns, lookback_ns, snap.sim_now_ns,
        );
    }
    sim.restore_from(snap.sim);

    let in_flight_at_snap: Vec<flow::sim::Scheduled> =
        sim.in_flight.iter().map(|r| r.0.clone()).collect();
    let edges_at_snap = sim.edges.clone();

    let start_idx = sim.log.total_recorded;
    if sim.now_ns < target_ns {
        sim.run_until(target_ns);
    }
    let end_idx = sim.log.total_recorded;

    let mut replay: Vec<Event> = Vec::with_capacity(in_flight_at_snap.len() + 64);

    // Synthesize PacketEmitted events for packets that were already
    // mid-flight at the snapshot — those won't be re-emitted by
    // run_until (they were emitted before the snapshot was taken).
    // Without these, the visual strategy misses the "chain head"
    // visuals that started earlier than the snapshot moment.
    for s in &in_flight_at_snap {
        let edge = match edges_at_snap.get(&s.edge) {
            Some(e) => e,
            None => continue,
        };
        let from = if s.deliver_to == edge.to {
            edge.from
        } else {
            edge.to
        };
        replay.push(Event::PacketEmitted {
            packet: s.packet.id,
            from,
            to: s.deliver_to,
            at_ns: s.packet.emitted_at_ns,
            arrives_at_ns: s.arrives_at_ns,
            payload: s.packet.payload.clone(),
        });
    }

    // Append the post-snap delta from the log. `start_idx`/`end_idx`
    // bracket the events `run_until` just emitted.
    let total = sim.log.total_recorded;
    let resident = sim.log.events.len() as u64;
    let first_resident = total.saturating_sub(resident);
    let from = start_idx.max(first_resident);
    let to = end_idx.min(total);
    if from < to {
        let skip = (from - first_resident) as usize;
        let take = (to - from) as usize;
        for ev in sim.log.events.iter().skip(skip).take(take) {
            if matches!(ev, Event::PacketEmitted { .. }) {
                replay.push(ev.clone());
            }
        }
    }
    replay.sort_by_key(|e| match e {
        Event::PacketEmitted { at_ns, .. } => *at_ns,
        _ => 0,
    });

    if debug {
        eprintln!(
            "[rewind] in_flight_at_snap={} log_range=[{},{}) total replay events={}",
            in_flight_at_snap.len(),
            start_idx,
            end_idx,
            replay.len(),
        );
    }
    *prev_log_index = sim.log.total_recorded;
    *rewind_epoch += 1;
    Some(Arc::new(replay))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn sim_with_one_edge(latency_ns: i64) -> Sim {
        let mut sim = Sim::new(1);
        let n0 = sim.add_node("n0", BTreeMap::new(), Vec::new());
        let n1 = sim.add_node("n1", BTreeMap::new(), Vec::new());
        sim.add_edge(n0, n1, flow::Expr::int(latency_ns));
        sim
    }

    #[test]
    fn max_edge_latency_skips_self_loops() {
        let mut sim = Sim::new(1);
        let n0 = sim.add_node("n0", BTreeMap::new(), Vec::new());
        let n1 = sim.add_node("n1", BTreeMap::new(), Vec::new());
        sim.add_edge(n0, n1, flow::Expr::int(2_000_000));
        sim.add_edge(n0, n0, flow::Expr::int(50_000_000)); // self-loop
        assert_eq!(sim_max_edge_latency_ns(&sim), 2_000_000);
    }

    #[test]
    fn max_edge_latency_returns_zero_when_only_self_loops() {
        let mut sim = Sim::new(1);
        let n0 = sim.add_node("n0", BTreeMap::new(), Vec::new());
        sim.add_edge(n0, n0, flow::Expr::int(50_000_000));
        assert_eq!(sim_max_edge_latency_ns(&sim), 0);
    }

    #[test]
    fn max_edge_latency_picks_the_max() {
        let mut sim = Sim::new(1);
        let n0 = sim.add_node("n0", BTreeMap::new(), Vec::new());
        let n1 = sim.add_node("n1", BTreeMap::new(), Vec::new());
        let n2 = sim.add_node("n2", BTreeMap::new(), Vec::new());
        sim.add_edge(n0, n1, flow::Expr::int(1_000_000));
        sim.add_edge(n1, n2, flow::Expr::int(5_000_000));
        sim.add_edge(n0, n2, flow::Expr::int(3_000_000));
        assert_eq!(sim_max_edge_latency_ns(&sim), 5_000_000);
    }

    #[test]
    fn one_edge_present_maxes_correctly() {
        let sim = sim_with_one_edge(1_500_000);
        assert_eq!(sim_max_edge_latency_ns(&sim), 1_500_000);
    }
}
