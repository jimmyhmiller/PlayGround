//! Rewind strategies — pluggable policies for reconstructing the
//! visual layer's state after the sim is restored to a past time.
//!
//! Mirrors the [`crate::visual`] split: a small trait, a dispatch
//! enum, and a [`RewindStrategyKind`] selector with an env-var
//! initializer for A/B testing.
//!
//! # The contract
//!
//! Every rewind goes through the same shell on the driver side:
//!
//!   1. Pick the latest snapshot at-or-before `target_ns`.
//!   2. `sim.restore_from(snap.sim)` — sim now matches the
//!      snapshot's state.
//!   3. Capture `in_flight_at_snap` + `edges_at_snap` + the log's
//!      `total_recorded` (call this `start_idx`) before any
//!      forward run.
//!   4. `sim.run_until(target_ns)` — sim re-runs forward from the
//!      snapshot point to the rewind target. New events accumulate
//!      in `sim.log` at indices `[start_idx, end_idx)`.
//!   5. Strategy decides which events the visual layer should
//!      replay, returning them in chronological order.
//!
//! Strategies differ only in *which subset* of the available events
//! they emit. Smaller subsets mean less work per rewind (no growth
//! with session length); larger subsets give richer past-history
//! reconstruction.
//!
//! # Time mapping (strategies don't have to think about this)
//!
//! Whatever events a strategy returns get ingested by
//! [`crate::edges::apply_rewind_reset`] with
//! `synth_real_now = visual_now + (at_ns - sim_now_ns) * 1e-9`.
//! Lockstep formula — no `k` factor, see that function for why.

use std::collections::BTreeMap;

use flow::sim::Scheduled;
use flow::{EdgeId, Edge, Event, Sim, Time};

/// Inputs available to a strategy when it decides what to replay.
/// Bundled into a struct so future strategies can take what they
/// need without us thrashing the trait signature.
pub struct RewindContext<'a> {
    /// The sim *after* `restore_from(snap)` and `run_until(target)`.
    /// `sim.now_ns == target_ns`. `sim.log` includes every event
    /// the snapshot already had plus the freshly-replayed events
    /// from the run-forward.
    pub sim: &'a Sim,
    /// Snapshot of `sim.in_flight` taken right after restore, *before*
    /// run_until. Holds the packets that crossed the snapshot boundary
    /// (emitted before `snap_t`, not yet arrived). The post-run sim
    /// no longer has these as a distinct set — `run_until` may have
    /// delivered some — so we capture them up front.
    pub in_flight_at_snap: &'a [Scheduled],
    /// Edge map snapshot at `snap_t`. Used to recover the `from`
    /// node for each scheduled packet (forward traversal vs.
    /// reverse-direction reply).
    pub edges_at_snap: &'a BTreeMap<EdgeId, Edge>,
    /// Absolute log index at the moment the snapshot was taken
    /// (`snap.sim.log.total_recorded`). The events in
    /// `sim.log.events` whose absolute index is `>= start_idx` are
    /// the ones produced by `run_until` — the post-snap delta.
    pub start_idx: u64,
    /// Absolute log index after `run_until`. Equals
    /// `sim.log.total_recorded`. Events in `[start_idx, end_idx)`
    /// are the post-snap delta.
    pub end_idx: u64,
    /// The sim time the rewind landed at (== `sim.now_ns`).
    pub target_ns: Time,
}

/// A rewind strategy. Implementations are stateless data-flow
/// policies: read the context, return a chronological event list.
pub trait RewindStrategy: Send + Sync + 'static {
    fn compute_replay(&self, ctx: &RewindContext<'_>) -> Vec<Event>;
}

/// Selector for picking a strategy variant by name (env var, future
/// runtime swap, etc.). Keeping this separate from the dispatch
/// enum lets us thread a cheap `Copy` value through code paths that
/// don't want to own the strategy itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewindStrategyKind {
    /// Replay every `PacketEmitted` in the bounded sim log. O(log
    /// size). The original behaviour — accurate reconstruction
    /// including the `gc`-window of recently-arrived past visuals,
    /// but cost grows with session length until the log evicts.
    FullLog,
    /// Replay only the post-snap delta plus a synthetic
    /// `PacketEmitted` for each entry in `in_flight_at_snap`. O(snapshot
    /// cadence × event density + in-flight size). Bounded regardless
    /// of session length. Misses packets that were emitted *and*
    /// arrived before `snap_t` (those don't contribute to the
    /// in-flight set at `target` anyway).
    PostSnap,
    /// Replay only synthesised emit events for the rewound sim's
    /// current `in_flight` heap. Smallest output — only currently
    /// in-flight packets — and so doesn't reconstruct the
    /// recently-arrived gc trail. Useful for measuring what's
    /// strictly necessary.
    InFlightOnly,
}

impl RewindStrategyKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::FullLog => "full-log",
            Self::PostSnap => "post-snap",
            Self::InFlightOnly => "in-flight-only",
        }
    }

    /// Construct the dispatch enum for this kind.
    pub fn build(self) -> RewindStrategyDispatch {
        match self {
            Self::FullLog => RewindStrategyDispatch::FullLog(FullLogStrategy),
            Self::PostSnap => RewindStrategyDispatch::PostSnap(PostSnapStrategy),
            Self::InFlightOnly => RewindStrategyDispatch::InFlightOnly(InFlightOnlyStrategy),
        }
    }
}

/// Dispatch enum holding one concrete strategy. Owned by the sim
/// driver so worker-mode rewinds (which run on a background thread)
/// can carry it without boxing.
#[derive(Clone)]
pub enum RewindStrategyDispatch {
    FullLog(FullLogStrategy),
    PostSnap(PostSnapStrategy),
    InFlightOnly(InFlightOnlyStrategy),
}

impl RewindStrategy for RewindStrategyDispatch {
    fn compute_replay(&self, ctx: &RewindContext<'_>) -> Vec<Event> {
        match self {
            Self::FullLog(s) => s.compute_replay(ctx),
            Self::PostSnap(s) => s.compute_replay(ctx),
            Self::InFlightOnly(s) => s.compute_replay(ctx),
        }
    }
}

impl RewindStrategyDispatch {
    pub fn kind(&self) -> RewindStrategyKind {
        match self {
            Self::FullLog(_) => RewindStrategyKind::FullLog,
            Self::PostSnap(_) => RewindStrategyKind::PostSnap,
            Self::InFlightOnly(_) => RewindStrategyKind::InFlightOnly,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FullLogStrategy;
#[derive(Clone, Copy, Debug)]
pub struct PostSnapStrategy;
#[derive(Clone, Copy, Debug)]
pub struct InFlightOnlyStrategy;

impl RewindStrategy for FullLogStrategy {
    fn compute_replay(&self, ctx: &RewindContext<'_>) -> Vec<Event> {
        ctx.sim
            .log
            .events
            .iter()
            .filter(|e| matches!(e, Event::PacketEmitted { .. }))
            .cloned()
            .collect()
    }
}

impl RewindStrategy for PostSnapStrategy {
    fn compute_replay(&self, ctx: &RewindContext<'_>) -> Vec<Event> {
        let mut out: Vec<Event> = Vec::with_capacity(ctx.in_flight_at_snap.len() + 64);

        // Synthesize PacketEmitted for each entry in the snapshot's
        // in-flight set. These are emissions that happened before
        // `snap_t` and so won't appear in the post-snap log delta.
        for s in ctx.in_flight_at_snap {
            let edge = match ctx.edges_at_snap.get(&s.edge) {
                Some(e) => e,
                None => continue, // edge dropped between snap and now — skip
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

        // Append the post-snap delta — events whose absolute log
        // index is in [start_idx, end_idx). The log is a deque
        // bounded at 100K; events outside the resident window are
        // already gone (and predate the snapshot anyway).
        let total = ctx.sim.log.total_recorded;
        let resident = ctx.sim.log.events.len() as u64;
        let first_resident = total.saturating_sub(resident);
        let from = ctx.start_idx.max(first_resident);
        let to = ctx.end_idx.min(total);
        if from < to {
            let skip = (from - first_resident) as usize;
            let take = (to - from) as usize;
            for ev in ctx.sim.log.events.iter().skip(skip).take(take) {
                if matches!(ev, Event::PacketEmitted { .. }) {
                    out.push(ev.clone());
                }
            }
        }

        // Synthesised in-flight events have `at_ns` from before
        // `snap_t`; delta events are after. Sort by emit time so
        // the visual ingest sees a chronological stream and
        // strategies' causal clamps build up in the right order.
        out.sort_by_key(|e| match e {
            Event::PacketEmitted { at_ns, .. } => *at_ns,
            _ => 0,
        });
        out
    }
}

impl RewindStrategy for InFlightOnlyStrategy {
    fn compute_replay(&self, ctx: &RewindContext<'_>) -> Vec<Event> {
        let mut out = Vec::with_capacity(ctx.sim.in_flight.len());
        for r in &ctx.sim.in_flight {
            let s = &r.0;
            let edge = match ctx.sim.edges.get(&s.edge) {
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
        out.sort_by_key(|e| match e {
            Event::PacketEmitted { at_ns, .. } => *at_ns,
            _ => 0,
        });
        out
    }
}

/// Pick a strategy *kind* from the `FLOW_BEVY_REWIND_STRATEGY` env
/// var. Default: `PostSnap` — the bounded-cost approach.
pub fn initial_strategy_kind() -> RewindStrategyKind {
    std::env::var("FLOW_BEVY_REWIND_STRATEGY")
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "full" | "full-log" | "full_log" | "fulllog" => Some(RewindStrategyKind::FullLog),
            "post-snap" | "post_snap" | "postsnap" | "snap" => Some(RewindStrategyKind::PostSnap),
            "in-flight" | "in_flight" | "inflight" | "in-flight-only" | "inflightonly" => {
                Some(RewindStrategyKind::InFlightOnly)
            }
            _ => None,
        })
        .unwrap_or(RewindStrategyKind::PostSnap)
}

/// Pick the initial strategy dispatch from the env var (or default).
pub fn initial_strategy() -> RewindStrategyDispatch {
    initial_strategy_kind().build()
}

/// Strategy variants in cycle order — used by the HUD's rotate
/// button. Listed coarsest-to-finest so cycling visibly walks
/// through "more reconstruction" → "less".
pub const STRATEGY_CYCLE: &[RewindStrategyKind] = &[
    RewindStrategyKind::FullLog,
    RewindStrategyKind::PostSnap,
    RewindStrategyKind::InFlightOnly,
];

impl RewindStrategyKind {
    /// Next entry in [`STRATEGY_CYCLE`], wrapping back to the start
    /// at the end. Used by the HUD click-to-cycle handler.
    pub fn next(self) -> Self {
        let i = STRATEGY_CYCLE.iter().position(|k| *k == self).unwrap_or(0);
        STRATEGY_CYCLE[(i + 1) % STRATEGY_CYCLE.len()]
    }
}
