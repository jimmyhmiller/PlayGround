//! Rewind strategies — pluggable policies for reconstructing the
//! visual layer's state at a past sim time.
//!
//! Every variant ultimately yields a `Vec<Event>` of `PacketEmitted`
//! that the visual layer ingests with a synthetic time mapping
//! (`crate::edges::apply_rewind_reset`). Strategies differ in
//!
//!   * **which snapshot they restore from** (close to target vs.
//!     deep enough to cover the chain-reach window), and
//!   * **which events they package** (full log, post-snap delta +
//!     in-flight, in-flight only, run-forward trajectory + in-flight
//!     at anchor).
//!
//! Each strategy owns its full rewind operation — pick snapshot,
//! restore, run sim forward, collect events, advance the bookkeeping
//! cursors. This is a change from the previous shape (trait returned
//! events; shell did snapshot+restore+run); it was forced by
//! `AnchorReplay` needing to anchor on a different snapshot than the
//! at-or-before-target one the others use.

use std::collections::BTreeMap;
use std::sync::Arc;

use flow::{Edge, EdgeId, Event, Sim, Snapshot, SnapshotRing, Time, Value};
use flow::sim::Scheduled;

// ─────────────────────────────────────────────────────────────────
// Lookback derivation (used by AnchorReplay)
// ─────────────────────────────────────────────────────────────────

/// Conservative ceiling on the depth of a causal-clamp chain that
/// can still affect currently-visible packets. A chain of 8 hops at
/// the slowest edge in the sim is the lookback window AnchorReplay
/// restores from.
pub const MAX_CHAIN_DEPTH: u32 = 8;

/// Fallback ceiling per non-trivially-typed edge `latency_ns` Expr.
/// Most edges in this codebase use `Expr::int(N)` literals which we
/// extract directly; dynamic Exprs (random distributions, slot
/// reads) can't be bounded statically, so we cap them at 100ms.
const FALLBACK_MAX_EDGE_LAT_NS: u64 = 100_000_000;

fn edge_lat_ceiling_ns(edge: &Edge) -> u64 {
    match &edge.latency_ns {
        flow::Expr::Lit(Value::Int(n)) if *n > 0 => *n as u64,
        _ => FALLBACK_MAX_EDGE_LAT_NS,
    }
}

/// Sim-time lookback window for `AnchorReplay`: how far before
/// `target_ns` we must restore from for visuals to reconstruct
/// correctly.
///
/// Two terms:
///
/// 1. **Visibility window**: a packet emitted at `at_ns` is on
///    screen iff `visual_now < arrive_real`, which under the
///    lockstep mapping `arrive_real = synth_emit_real + sim_lat × k`
///    means the oldest visible packet has
///    `at_ns ≥ sim_now_ns − sim_lat_ns × k`. So we must replay
///    events back to `max_edge_lat × k` in sim ns. Without this
///    term, the gc trail goes missing on rewind: visuals already
///    travelling out of workers/sinks aren't replayed because
///    their emit events were ignored.
/// 2. **Chain reach**: `max_edge_lat × MAX_CHAIN_DEPTH` covers
///    causal-clamp predecessors.
///
/// We bound by **edge latencies only** — not the `in_flight` heap.
/// `in_flight` includes internal wake/tick events whose
/// `arrives_at_ns − emitted_at_ns` reflects scheduling intervals
/// (e.g. Generator periods), not visual edge crossings, and they
/// inflate the lookback by orders of magnitude in busy sims.
/// `k` is the live visual-scale value pulled from `SimControl`.
pub fn rewind_lookback_ns(sim: &Sim, k: f64) -> Time {
    let mut max_lat: u64 = 0;
    for edge in sim.edges.values() {
        let lat = edge_lat_ceiling_ns(edge);
        if lat > max_lat { max_lat = lat; }
    }
    if max_lat == 0 {
        max_lat = FALLBACK_MAX_EDGE_LAT_NS;
    }
    let k_eff = k.max(1.0);
    let visibility = (max_lat as f64 * k_eff) as u64;
    let chain = max_lat.saturating_mul(MAX_CHAIN_DEPTH as u64);
    visibility.saturating_add(chain)
}

// ─────────────────────────────────────────────────────────────────
// Trait + dispatch
// ─────────────────────────────────────────────────────────────────

/// Contract: do the entire rewind. Restore from some snapshot, run
/// sim forward to `target_ns`, advance `prev_log_index` past the
/// re-emitted events (so subsequent forward play doesn't double-feed
/// them), bump `rewind_epoch`, and return the events the visual
/// layer should ingest. Return `None` only when no usable snapshot
/// exists for `target_ns` — caller leaves sim untouched.
pub trait RewindStrategy: Send + Sync + 'static {
    fn do_rewind(
        &self,
        sim: &mut Sim,
        ring: &SnapshotRing,
        prev_log_index: &mut u64,
        rewind_epoch: &mut u64,
        target_ns: Time,
        visual_k: f64,
    ) -> Option<Arc<Vec<Event>>>;
}

/// Selector for picking a strategy variant by name (env var, HUD
/// cycle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewindStrategyKind {
    /// Re-execute sim from a snapshot deep enough to cover the
    /// chain-reach window, plus synthesize in-flight emits at that
    /// anchor. Default — bounded cost, accurate visuals (including
    /// the gc trail) without keeping unbounded event log history.
    AnchorReplay,
    /// Replay every `PacketEmitted` in the bounded sim log. O(log
    /// size). Accurate including all gc-window history, but cost
    /// grows with session length until the log evicts.
    FullLog,
    /// Snapshot at-or-before target, plus synthesized
    /// `PacketEmitted` for in-flight at the snapshot moment, plus
    /// the post-snap delta. Bounded cost; misses chain predecessors
    /// older than the snapshot.
    PostSnap,
    /// Only synthesized emits for the rewound sim's current
    /// in-flight heap. Smallest output; no gc trail.
    InFlightOnly,
}

impl RewindStrategyKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::AnchorReplay => "anchor",
            Self::FullLog => "full-log",
            Self::PostSnap => "post-snap",
            Self::InFlightOnly => "in-flight-only",
        }
    }

    pub fn build(self) -> RewindStrategyDispatch {
        match self {
            Self::AnchorReplay => RewindStrategyDispatch::AnchorReplay(AnchorReplay),
            Self::FullLog => RewindStrategyDispatch::FullLog(FullLogStrategy),
            Self::PostSnap => RewindStrategyDispatch::PostSnap(PostSnapStrategy),
            Self::InFlightOnly => RewindStrategyDispatch::InFlightOnly(InFlightOnlyStrategy),
        }
    }
}

#[derive(Clone)]
pub enum RewindStrategyDispatch {
    AnchorReplay(AnchorReplay),
    FullLog(FullLogStrategy),
    PostSnap(PostSnapStrategy),
    InFlightOnly(InFlightOnlyStrategy),
}

impl RewindStrategy for RewindStrategyDispatch {
    fn do_rewind(
        &self,
        sim: &mut Sim,
        ring: &SnapshotRing,
        prev_log_index: &mut u64,
        rewind_epoch: &mut u64,
        target_ns: Time,
        visual_k: f64,
    ) -> Option<Arc<Vec<Event>>> {
        match self {
            Self::AnchorReplay(s) => s.do_rewind(sim, ring, prev_log_index, rewind_epoch, target_ns, visual_k),
            Self::FullLog(s) => s.do_rewind(sim, ring, prev_log_index, rewind_epoch, target_ns, visual_k),
            Self::PostSnap(s) => s.do_rewind(sim, ring, prev_log_index, rewind_epoch, target_ns, visual_k),
            Self::InFlightOnly(s) => s.do_rewind(sim, ring, prev_log_index, rewind_epoch, target_ns, visual_k),
        }
    }
}

impl RewindStrategyDispatch {
    pub fn kind(&self) -> RewindStrategyKind {
        match self {
            Self::AnchorReplay(_) => RewindStrategyKind::AnchorReplay,
            Self::FullLog(_) => RewindStrategyKind::FullLog,
            Self::PostSnap(_) => RewindStrategyKind::PostSnap,
            Self::InFlightOnly(_) => RewindStrategyKind::InFlightOnly,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnchorReplay;
#[derive(Clone, Copy, Debug)]
pub struct FullLogStrategy;
#[derive(Clone, Copy, Debug)]
pub struct PostSnapStrategy;
#[derive(Clone, Copy, Debug)]
pub struct InFlightOnlyStrategy;

// ─────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────

/// Pick the latest entry whose `sim_now_ns ≤ target_ns`. Falls back
/// to the sticky anchor if no entry qualifies — but only if the
/// anchor itself is `≤ target_ns`, otherwise None (rewind to a time
/// earlier than every available capture cleanly fails).
fn snap_at_or_before(ring: &SnapshotRing, target_ns: Time) -> Option<Snapshot> {
    ring.entries.iter()
        .rev()
        .find(|s| s.sim_now_ns <= target_ns)
        .cloned()
        .or_else(|| {
            ring.anchor.as_ref()
                .filter(|a| a.sim_now_ns <= target_ns)
                .cloned()
        })
}

/// Pick a snapshot for `AnchorReplay`: prefer one deep enough to
/// cover the chain-reach window, fall back to the latest
/// topology-bearing entry ≤ target_ns, then to the sticky anchor
/// (filtered ≤ target_ns).
fn snap_anchor_replay(ring: &SnapshotRing, target_ns: Time, anchor_ns: Time) -> Option<Snapshot> {
    let has_topology = |s: &Snapshot| !s.sim.nodes.is_empty();
    ring.entries.iter()
        .rev()
        .find(|s| s.sim_now_ns <= anchor_ns && has_topology(s))
        .or_else(|| {
            ring.entries.iter()
                .rev()
                .find(|s| s.sim_now_ns <= target_ns && has_topology(s))
        })
        .cloned()
        .or_else(|| {
            ring.anchor.as_ref()
                .filter(|a| a.sim_now_ns <= target_ns)
                .cloned()
        })
}

/// Synthesize a `PacketEmitted` event for each `Scheduled` entry
/// using the provided edge map for `from`/`to` resolution. Used by
/// strategies that include in-flight packets emitted before the
/// snapshot's now_ns (those won't be re-emitted by run_until).
fn synthesize_in_flight(
    out: &mut Vec<Event>,
    in_flight: &[Scheduled],
    edges: &BTreeMap<EdgeId, Edge>,
) {
    for s in in_flight {
        let edge = match edges.get(&s.edge) {
            Some(e) => e,
            None => continue,
        };
        let from = if s.deliver_to == edge.to { edge.from } else { edge.to };
        out.push(Event::PacketEmitted {
            packet: s.packet.id,
            from,
            to: s.deliver_to,
            at_ns: s.packet.emitted_at_ns,
            arrives_at_ns: s.arrives_at_ns,
            payload: s.packet.payload.clone(),
        });
    }
}

/// Append `PacketEmitted` events from `sim.log` whose absolute log
/// index is in `[start_idx, end_idx)`. Strategies use this to pick
/// up the post-snap delta after `run_until`.
fn append_log_range(
    out: &mut Vec<Event>,
    sim: &Sim,
    start_idx: u64,
    end_idx: u64,
) {
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
                out.push(ev.clone());
            }
        }
    }
}

fn sort_by_at_ns(events: &mut [Event]) {
    events.sort_by_key(|e| match e {
        Event::PacketEmitted { at_ns, .. } => *at_ns,
        _ => 0,
    });
}

// ─────────────────────────────────────────────────────────────────
// Strategy implementations
// ─────────────────────────────────────────────────────────────────

impl RewindStrategy for AnchorReplay {
    fn do_rewind(
        &self,
        sim: &mut Sim,
        ring: &SnapshotRing,
        prev_log_index: &mut u64,
        rewind_epoch: &mut u64,
        target_ns: Time,
        visual_k: f64,
    ) -> Option<Arc<Vec<Event>>> {
        let lookback = rewind_lookback_ns(sim, visual_k);
        let anchor_ns = target_ns.saturating_sub(lookback);
        let snap = snap_anchor_replay(ring, target_ns, anchor_ns)?;
        sim.restore_from(snap.sim);

        let in_flight_at_anchor: Vec<Scheduled> =
            sim.in_flight.iter().map(|r| r.0.clone()).collect();
        let edges_at_anchor = sim.edges.clone();

        let start_idx = sim.log.total_recorded;
        if sim.now_ns < target_ns {
            sim.run_until(target_ns);
        }
        let end_idx = sim.log.total_recorded;

        let mut replay: Vec<Event> = Vec::with_capacity(in_flight_at_anchor.len() + 64);
        synthesize_in_flight(&mut replay, &in_flight_at_anchor, &edges_at_anchor);
        append_log_range(&mut replay, sim, start_idx, end_idx);
        sort_by_at_ns(&mut replay);

        *prev_log_index = sim.log.total_recorded;
        *rewind_epoch += 1;
        Some(Arc::new(replay))
    }
}

impl RewindStrategy for FullLogStrategy {
    fn do_rewind(
        &self,
        sim: &mut Sim,
        ring: &SnapshotRing,
        prev_log_index: &mut u64,
        rewind_epoch: &mut u64,
        target_ns: Time,
        _visual_k: f64,
    ) -> Option<Arc<Vec<Event>>> {
        let snap = snap_at_or_before(ring, target_ns)?;
        sim.restore_from(snap.sim);

        if sim.now_ns < target_ns {
            sim.run_until(target_ns);
        }

        let mut replay: Vec<Event> = sim.log.events.iter()
            .filter(|e| matches!(e, Event::PacketEmitted { .. }))
            .cloned()
            .collect();
        sort_by_at_ns(&mut replay);

        *prev_log_index = sim.log.total_recorded;
        *rewind_epoch += 1;
        Some(Arc::new(replay))
    }
}

impl RewindStrategy for PostSnapStrategy {
    fn do_rewind(
        &self,
        sim: &mut Sim,
        ring: &SnapshotRing,
        prev_log_index: &mut u64,
        rewind_epoch: &mut u64,
        target_ns: Time,
        _visual_k: f64,
    ) -> Option<Arc<Vec<Event>>> {
        let snap = snap_at_or_before(ring, target_ns)?;
        sim.restore_from(snap.sim);

        let in_flight_at_snap: Vec<Scheduled> =
            sim.in_flight.iter().map(|r| r.0.clone()).collect();
        let edges_at_snap = sim.edges.clone();

        let start_idx = sim.log.total_recorded;
        if sim.now_ns < target_ns {
            sim.run_until(target_ns);
        }
        let end_idx = sim.log.total_recorded;

        let mut replay: Vec<Event> = Vec::with_capacity(in_flight_at_snap.len() + 64);
        synthesize_in_flight(&mut replay, &in_flight_at_snap, &edges_at_snap);
        append_log_range(&mut replay, sim, start_idx, end_idx);
        sort_by_at_ns(&mut replay);

        *prev_log_index = sim.log.total_recorded;
        *rewind_epoch += 1;
        Some(Arc::new(replay))
    }
}

impl RewindStrategy for InFlightOnlyStrategy {
    fn do_rewind(
        &self,
        sim: &mut Sim,
        ring: &SnapshotRing,
        prev_log_index: &mut u64,
        rewind_epoch: &mut u64,
        target_ns: Time,
        _visual_k: f64,
    ) -> Option<Arc<Vec<Event>>> {
        let snap = snap_at_or_before(ring, target_ns)?;
        sim.restore_from(snap.sim);

        if sim.now_ns < target_ns {
            sim.run_until(target_ns);
        }

        let in_flight_now: Vec<Scheduled> =
            sim.in_flight.iter().map(|r| r.0.clone()).collect();

        let mut replay: Vec<Event> = Vec::with_capacity(in_flight_now.len());
        synthesize_in_flight(&mut replay, &in_flight_now, &sim.edges);
        sort_by_at_ns(&mut replay);

        *prev_log_index = sim.log.total_recorded;
        *rewind_epoch += 1;
        Some(Arc::new(replay))
    }
}

// ─────────────────────────────────────────────────────────────────
// Selection helpers
// ─────────────────────────────────────────────────────────────────

/// Pick a strategy *kind* from `FLOW_BEVY_REWIND_STRATEGY`. Default:
/// `AnchorReplay` — bounded cost with correct chain reconstruction.
pub fn initial_strategy_kind() -> RewindStrategyKind {
    std::env::var("FLOW_BEVY_REWIND_STRATEGY")
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "anchor" | "anchor-replay" | "anchor_replay" => Some(RewindStrategyKind::AnchorReplay),
            "full" | "full-log" | "full_log" | "fulllog" => Some(RewindStrategyKind::FullLog),
            "post-snap" | "post_snap" | "postsnap" | "snap" => Some(RewindStrategyKind::PostSnap),
            "in-flight" | "in_flight" | "inflight" | "in-flight-only" | "inflightonly" => {
                Some(RewindStrategyKind::InFlightOnly)
            }
            _ => None,
        })
        .unwrap_or(RewindStrategyKind::AnchorReplay)
}

pub fn initial_strategy() -> RewindStrategyDispatch {
    initial_strategy_kind().build()
}

/// Strategy variants in cycle order — used by the HUD's rotate
/// button. AnchorReplay first (the default), then the legacy three
/// in their original order.
pub const STRATEGY_CYCLE: &[RewindStrategyKind] = &[
    RewindStrategyKind::AnchorReplay,
    RewindStrategyKind::FullLog,
    RewindStrategyKind::PostSnap,
    RewindStrategyKind::InFlightOnly,
];

impl RewindStrategyKind {
    pub fn next(self) -> Self {
        let i = STRATEGY_CYCLE.iter().position(|k| *k == self).unwrap_or(0);
        STRATEGY_CYCLE[(i + 1) % STRATEGY_CYCLE.len()]
    }
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

    /// Regression: lookback must scale with `k` so high-`k`
    /// sessions don't drop the gc trail on rewind. Reported
    /// symptom: ThreeLaneFanout at default k=200 had no visual
    /// packets reverse out of sinks/workers; lowering k masked
    /// the bug.
    #[test]
    fn lookback_scales_with_visual_k() {
        let sim = sim_with_one_edge(1_000_000); // 1 ms edge

        let lb_k1 = rewind_lookback_ns(&sim, 1.0);
        let lb_k200 = rewind_lookback_ns(&sim, 200.0);
        let lb_k400 = rewind_lookback_ns(&sim, 400.0);

        assert!(
            lb_k200 >= 200_000_000,
            "k=200 lookback {}ns < expected 200ms — gc trail will go missing",
            lb_k200,
        );
        assert!(
            lb_k400 > lb_k200,
            "lookback should grow with k: k=200 → {}, k=400 → {}",
            lb_k200, lb_k400,
        );
        assert!(
            lb_k1 < lb_k200,
            "k=1 lookback {} should be smaller than k=200 lookback {}",
            lb_k1, lb_k200,
        );
    }

    #[test]
    fn lookback_handles_zero_or_subunit_k() {
        let sim = sim_with_one_edge(1_000_000);

        let lb_k0 = rewind_lookback_ns(&sim, 0.0);
        let lb_kneg = rewind_lookback_ns(&sim, -5.0);
        let lb_kfrac = rewind_lookback_ns(&sim, 0.5);

        assert!(lb_k0 > 0, "k=0 produced zero lookback — would skip every chain predecessor");
        assert!(lb_kneg > 0, "k=-5 produced zero lookback");
        assert!(lb_kfrac > 0, "k=0.5 produced zero lookback");
    }
}
