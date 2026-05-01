//! Visual layer: maps sim state + (optional) event stream into the
//! set of `VisualPacket`s the renderer should draw this frame.
//!
//! # The trait
//!
//! [`VisualStrategy::frame_into`] is the entire production API. Each
//! frame, the host hands a strategy the current `TimeCursor`, a view
//! over the live sim's edges + in-flight heap, and the events the sim
//! emitted since the last frame. The strategy writes its visible
//! packet set into `out`. That's it — no per-event ingest, no
//! `visible_at(t)`, no GC scheduler.
//!
//! Strategies that maintain derived state (Replay's causal-clamp log,
//! the rate-sampled edge windows) consume `ctx.new_events`
//! incrementally and call their internal helpers. Strategies that
//! project current sim state (the limiting case [`SimMirror`]) read
//! `ctx.sim.in_flight` and ignore events entirely.
//!
//! # Replay-friendliness
//!
//! On rewind the host calls [`VisualStrategy::invalidate`] to drop
//! derived state, then resumes the normal frame loop. The next
//! `frame_into` call sees a fresh `new_events` slice (the events the
//! sim emitted while running forward from the rewind anchor to the
//! target). Strategies declare how far back the rewind needs to
//! anchor via [`VisualStrategy::rewind_lookback_ns`] — `SimMirror`
//! returns `0` (no history needed); event-history strategies return
//! enough sim-ns to cover the visibility window plus chain reach.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use flow::sim::Scheduled;
use flow::{Edge, EdgeId, Event, NodeId, PacketId, Time, Value};

/// How far past `visual_now` strategies retain past-arrived packet
/// records. Matches the snapshot ring's scrub-back horizon so a
/// rewind to anywhere in that window can reconstruct what was on
/// screen — strategies that GC tighter than this would silently
/// blank out the past on rewind.
pub const VISUAL_GC_KEEP_PAST_S: f64 = 30.0;

/// The current moment, in both axes the layer cares about.
///
/// Under the host's lockstep mapping `visual_now ≈ sim_now × 1e-9`
/// (modulo a stable offset across topology resets). Strategies that
/// need a wall-clock anchor read `visual_now`; strategies that key
/// off sim-time read `sim_now_ns`.
#[derive(Clone, Copy, Debug)]
pub struct TimeCursor {
    pub sim_now_ns: Time,
    pub visual_now: f64,
    /// Per-strategy visual-stretch knob. Event-history strategies use
    /// this to scale `sim_latency × k` into a visible window;
    /// [`SimMirror`] ignores it.
    pub k: f64,
}

/// Borrow into the sim's currently-needed render data. Cheap — both
/// fields are references the host pulls from snapshot Arcs each
/// frame, no per-strategy clones.
pub struct SimView<'a> {
    pub edges: &'a BTreeMap<EdgeId, Edge>,
    pub in_flight: &'a [Scheduled],
}

/// Everything a strategy needs to compute a frame.
pub struct VisualFrameCtx<'a> {
    pub time: TimeCursor,
    pub sim: SimView<'a>,
    /// Events emitted by the sim since the previous `frame_into`
    /// call. Includes the run-forward-from-rewind events when a
    /// rewind just landed (the host calls `invalidate` first, then
    /// passes those events through here).
    pub new_events: &'a [Event],
}

/// The single visual abstraction. Implementations decide what they
/// look at — sim state, event stream, or both — to produce the
/// per-frame visible set.
pub trait VisualStrategy: Send + Sync + 'static {
    /// Compute the frame's visible packet set. Strategies that
    /// accumulate state should run their per-event ingest over
    /// `ctx.new_events` first, then write the visible-packet view
    /// into `out`. Pure projections (SimMirror) ignore events and
    /// derive directly from `ctx.sim`.
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>);

    /// Drop derived state. The host calls this on rewind / topology
    /// reset before passing the post-rewind event slice through
    /// `frame_into`.
    fn invalidate(&mut self);

    /// How far before `target_ns` the host's rewind machinery must
    /// re-run the sim for this strategy to render correctly. `0`
    /// means "don't bother, current sim state is enough."
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64;

    fn set_k(&mut self, new_k: f64);
    fn k(&self) -> f64;
}

/// Concrete strategy enum the host stores as a Bevy resource.
#[derive(Clone, Debug)]
pub enum Strategy {
    /// Limiting case: no `k`, no causal clamp, no event history.
    /// Renders `sim.in_flight` directly with `progress = sim-time
    /// ratio`. Replay-friendly by construction — rewind to any
    /// snapshot ≤ target and the projection is right.
    SimMirror(SimMirror),
    /// Faithful one-visual-per-event replay with causal clamping
    /// and a `k`-stretched visible window. Needs a lookback covering
    /// the visibility window + chain reach.
    Replay(VisualTimeline),
    /// Per-edge rolling-window flow-rate sampling. Throttles bursty
    /// sims so the eye can track each packet, at the cost of
    /// dropping individual events.
    RateSampled(RateSampled),
    /// RateSampled-style throttle that also drops responses whose
    /// triggering request was throttled — every visible interaction
    /// is a complete chain.
    DropOrphans(DropOrphans),
    /// RateSampled-style throttle that retroactively promotes a
    /// throttled cause when a downstream event needs it.
    CausalRateSampled(CausalRateSampled),
    /// Coalesces bursts on the same edge within a sim-time window
    /// into one visual.
    BundleSummarized(BundleSummarized),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyKind {
    SimMirror,
    Replay,
    RateSampled,
    DropOrphans,
    CausalRateSampled,
    BundleSummarized,
}

impl StrategyKind {
    pub const ALL: &'static [StrategyKind] = &[
        StrategyKind::SimMirror,
        StrategyKind::Replay,
        StrategyKind::RateSampled,
        StrategyKind::DropOrphans,
        StrategyKind::CausalRateSampled,
        StrategyKind::BundleSummarized,
    ];

    pub fn label(self) -> &'static str {
        match self {
            StrategyKind::SimMirror => "sim-mirror",
            StrategyKind::Replay => "replay",
            StrategyKind::RateSampled => "rate-sampled",
            StrategyKind::DropOrphans => "drop-orphans",
            StrategyKind::CausalRateSampled => "causal-rate",
            StrategyKind::BundleSummarized => "bundle",
        }
    }

    fn ordinal(self) -> usize {
        Self::ALL.iter().position(|k| *k == self).unwrap_or(0)
    }

    pub fn next(self) -> StrategyKind {
        Self::ALL[(self.ordinal() + 1) % Self::ALL.len()]
    }
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::Replay(VisualTimeline::default())
    }
}

impl Strategy {
    pub fn new_of_kind(kind: StrategyKind, k: f64) -> Self {
        match kind {
            StrategyKind::SimMirror => Strategy::SimMirror(SimMirror::new()),
            StrategyKind::Replay => Strategy::Replay(VisualTimeline::new(k)),
            StrategyKind::RateSampled => Strategy::RateSampled(RateSampled::new(k)),
            StrategyKind::DropOrphans => Strategy::DropOrphans(DropOrphans::new(k)),
            StrategyKind::CausalRateSampled => {
                Strategy::CausalRateSampled(CausalRateSampled::new(k))
            }
            StrategyKind::BundleSummarized => {
                Strategy::BundleSummarized(BundleSummarized::new(k))
            }
        }
    }

    pub fn kind(&self) -> StrategyKind {
        match self {
            Strategy::SimMirror(_) => StrategyKind::SimMirror,
            Strategy::Replay(_) => StrategyKind::Replay,
            Strategy::RateSampled(_) => StrategyKind::RateSampled,
            Strategy::DropOrphans(_) => StrategyKind::DropOrphans,
            Strategy::CausalRateSampled(_) => StrategyKind::CausalRateSampled,
            Strategy::BundleSummarized(_) => StrategyKind::BundleSummarized,
        }
    }

    pub fn switch_to(&mut self, kind: StrategyKind) {
        if self.kind() == kind {
            return;
        }
        let k = self.k();
        *self = Strategy::new_of_kind(kind, k);
    }

    pub fn cycle(&mut self) {
        self.switch_to(self.kind().next());
    }

    /// Borrow the inner [`VisualTimeline`] when the active strategy
    /// is `Replay`. Panics otherwise — used by tests that depend on
    /// Replay-specific internal layout (`packets`, causal-arrival
    /// records).
    pub fn as_replay(&self) -> &VisualTimeline {
        match self {
            Strategy::Replay(t) => t,
            other => panic!(
                "as_replay() called on non-Replay strategy ({:?})",
                other.kind()
            ),
        }
    }

    pub fn as_replay_mut(&mut self) -> &mut VisualTimeline {
        match self {
            Strategy::Replay(t) => t,
            other => panic!(
                "as_replay_mut() called on non-Replay strategy ({:?})",
                other.kind()
            ),
        }
    }
}

impl VisualStrategy for Strategy {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        match self {
            Strategy::SimMirror(s) => s.frame_into(ctx, out),
            Strategy::Replay(s) => s.frame_into(ctx, out),
            Strategy::RateSampled(s) => s.frame_into(ctx, out),
            Strategy::DropOrphans(s) => s.frame_into(ctx, out),
            Strategy::CausalRateSampled(s) => s.frame_into(ctx, out),
            Strategy::BundleSummarized(s) => s.frame_into(ctx, out),
        }
    }
    fn invalidate(&mut self) {
        match self {
            Strategy::SimMirror(s) => s.invalidate(),
            Strategy::Replay(s) => s.invalidate(),
            Strategy::RateSampled(s) => s.invalidate(),
            Strategy::DropOrphans(s) => s.invalidate(),
            Strategy::CausalRateSampled(s) => s.invalidate(),
            Strategy::BundleSummarized(s) => s.invalidate(),
        }
    }
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64 {
        match self {
            Strategy::SimMirror(s) => s.rewind_lookback_ns(k, max_edge_lat_ns),
            Strategy::Replay(s) => s.rewind_lookback_ns(k, max_edge_lat_ns),
            Strategy::RateSampled(s) => s.rewind_lookback_ns(k, max_edge_lat_ns),
            Strategy::DropOrphans(s) => s.rewind_lookback_ns(k, max_edge_lat_ns),
            Strategy::CausalRateSampled(s) => s.rewind_lookback_ns(k, max_edge_lat_ns),
            Strategy::BundleSummarized(s) => s.rewind_lookback_ns(k, max_edge_lat_ns),
        }
    }
    fn set_k(&mut self, new_k: f64) {
        match self {
            Strategy::SimMirror(s) => s.set_k(new_k),
            Strategy::Replay(s) => s.set_k(new_k),
            Strategy::RateSampled(s) => s.set_k(new_k),
            Strategy::DropOrphans(s) => s.set_k(new_k),
            Strategy::CausalRateSampled(s) => s.set_k(new_k),
            Strategy::BundleSummarized(s) => s.set_k(new_k),
        }
    }
    fn k(&self) -> f64 {
        match self {
            Strategy::SimMirror(s) => s.k(),
            Strategy::Replay(s) => s.k(),
            Strategy::RateSampled(s) => s.k(),
            Strategy::DropOrphans(s) => s.k(),
            Strategy::CausalRateSampled(s) => s.k(),
            Strategy::BundleSummarized(s) => s.k(),
        }
    }
}

/// Filter for events that can become visible packets. Drops
/// control-plane variants (`pull`, `wake`), self-loops, and
/// zero-duration hops. Used by all event-driven strategies.
pub fn is_visible_event(ev: &Event) -> bool {
    let Event::PacketEmitted {
        from,
        to,
        payload,
        at_ns,
        arrives_at_ns,
        ..
    } = ev
    else {
        return false;
    };
    if from == to {
        return false;
    }
    if arrives_at_ns <= at_ns {
        return false;
    }
    if let Value::Variant { tag, .. } = payload {
        if tag == "pull" || tag == "wake" {
            return false;
        }
    }
    true
}

/// Per-event lockstep mapping from sim time to a synthetic real_now,
/// matching the host's wall-clock ↔ sim-clock coupling. Strategies
/// that need a `real_now` per ingest derive it via this helper so
/// forward play and post-rewind frames produce identical visuals
/// for identical event sequences.
fn real_now_for_event(ev: &Event, time: &TimeCursor) -> f64 {
    let sim_now_s = time.sim_now_ns as f64 * 1e-9;
    match ev {
        Event::PacketEmitted { at_ns, .. } => time.visual_now + (*at_ns as f64 * 1e-9 - sim_now_s),
        _ => time.visual_now,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisualPacket {
    pub packet_id: PacketId,
    pub from: NodeId,
    pub to: NodeId,
    pub payload: Value,
    pub emit_real: f64,
    pub arrive_real: f64,
}

impl VisualPacket {
    pub fn is_visible_at(&self, now: f64) -> bool {
        self.emit_real <= now && now < self.arrive_real
    }
    pub fn progress_at(&self, now: f64) -> f32 {
        let denom = (self.arrive_real - self.emit_real).max(1e-9);
        ((now - self.emit_real) / denom).clamp(0.0, 1.0) as f32
    }
}

// ────────────────────────────────────────────────────────────
// SimMirror — the limiting case
// ────────────────────────────────────────────────────────────

/// Project `sim.in_flight` straight to the screen with `progress =
/// (sim_now − emitted_at) / sim_latency`. No `k`, no causal clamp,
/// no internal state. The visual is a 1:1 mirror of what the sim is
/// doing right now.
///
/// Trade-offs vs the event-history strategies:
///
/// - **No visual stretch.** A 1ms edge at multiplier=1 is on screen
///   for 1ms wall — to make individual packets readable, slow
///   `multiplier` rather than stretching with `k`.
/// - **No trail past sim arrival.** Once `arrives_at_ns ≤ sim_now`
///   the packet leaves `in_flight` and disappears.
/// - **Replay-friendly by construction.** Rewind = restore + run
///   forward + frame; no derived state to reconstruct.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimMirror;

impl SimMirror {
    pub const fn new() -> Self {
        Self
    }
}

impl VisualStrategy for SimMirror {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        let TimeCursor {
            sim_now_ns,
            visual_now,
            ..
        } = ctx.time;
        for s in ctx.sim.in_flight {
            // Skip control-plane and zero-duration entries — they
            // contribute no visible motion. Self-loops are filtered
            // via the (from == deliver_to) check below.
            if s.arrives_at_ns <= s.packet.emitted_at_ns {
                continue;
            }
            if let Value::Variant { tag, .. } = &s.packet.payload {
                if tag == "pull" || tag == "wake" {
                    continue;
                }
            }
            let edge = match ctx.sim.edges.get(&s.edge) {
                Some(e) => e,
                None => continue,
            };
            // Direction: forward when delivering to edge.to, reverse
            // when delivering to edge.from (req/resp on the same edge).
            let from = if s.deliver_to == edge.to {
                edge.from
            } else {
                edge.to
            };
            if from == s.deliver_to {
                continue;
            }

            let sim_lat_ns = s.arrives_at_ns - s.packet.emitted_at_ns;
            let elapsed_ns = sim_now_ns.saturating_sub(s.packet.emitted_at_ns);
            // Don't render packets that haven't started yet (in_flight
            // is sorted by arrival, but emit can be > sim_now in a
            // freshly-restored snapshot before run_until catches up).
            if elapsed_ns == 0 {
                // emitted exactly now: progress 0 is fine, render.
            }
            let prog = (elapsed_ns as f64 / sim_lat_ns as f64).clamp(0.0, 1.0) as f32;

            let elapsed_s = elapsed_ns as f64 * 1e-9;
            let remaining_s = (s.arrives_at_ns.saturating_sub(sim_now_ns)) as f64 * 1e-9;
            let emit_real = visual_now - elapsed_s;
            let arrive_real = visual_now + remaining_s;

            out.push((
                VisualPacket {
                    packet_id: s.packet.id,
                    from,
                    to: s.deliver_to,
                    payload: s.packet.payload.clone(),
                    emit_real,
                    arrive_real,
                },
                prog,
            ));
        }
    }

    fn invalidate(&mut self) {}

    fn rewind_lookback_ns(&self, _k: f64, _max_edge_lat_ns: u64) -> u64 {
        0
    }

    fn set_k(&mut self, _new_k: f64) {}

    fn k(&self) -> f64 {
        1.0
    }
}

// ────────────────────────────────────────────────────────────
// Causal-clamp arrival index (shared by Replay/Bundle/Causal)
// ────────────────────────────────────────────────────────────

/// Sorted-by-sim-time record of visual arrivals at some node. Used
/// to find a causal trigger when a later packet emits from that
/// node. Parallel `Vec`s so we can binary-search the `ns` side.
#[derive(Clone, Debug, Default)]
struct NodeArrivals {
    arrives_ns: Vec<u64>,
    arrives_real: Vec<f64>,
}

impl NodeArrivals {
    fn push(&mut self, ns: u64, real: f64) {
        self.arrives_ns.push(ns);
        self.arrives_real.push(real);
    }
    fn trigger_for(&self, at_ns: u64) -> Option<f64> {
        self.trigger_for_full(at_ns).map(|(_, real)| real)
    }
    fn trigger_for_full(&self, at_ns: u64) -> Option<(u64, f64)> {
        let idx = self.arrives_ns.partition_point(|ns| *ns <= at_ns);
        if idx == 0 {
            None
        } else {
            Some((self.arrives_ns[idx - 1], self.arrives_real[idx - 1]))
        }
    }
    fn len(&self) -> usize {
        self.arrives_ns.len()
    }
    fn gc(&mut self, cutoff: f64) {
        let keep_from = self
            .arrives_real
            .iter()
            .position(|r| *r >= cutoff)
            .unwrap_or(self.arrives_real.len());
        if keep_from > 0 {
            self.arrives_ns.drain(..keep_from);
            self.arrives_real.drain(..keep_from);
        }
    }
}

// ────────────────────────────────────────────────────────────
// Replay (VisualTimeline)
// ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct VisualTimeline {
    pub packets: Vec<VisualPacket>,
    pub k: f64,
    node_arrivals: HashMap<NodeId, NodeArrivals>,
}

impl Default for VisualTimeline {
    fn default() -> Self {
        Self {
            packets: Vec::new(),
            k: Self::K_DEFAULT,
            node_arrivals: HashMap::new(),
        }
    }
}

impl VisualTimeline {
    pub const K_MIN: f64 = 0.1;
    pub const K_MAX: f64 = 100_000.0;
    pub const K_DEFAULT: f64 = 410.0;

    pub fn new(k: f64) -> Self {
        Self {
            k: k.clamp(Self::K_MIN, Self::K_MAX),
            ..Self::default()
        }
    }

    /// Per-event ingest. Public so tests can drive it directly with a
    /// chosen `real_now`; production calls go through `frame_into`.
    pub fn ingest(&mut self, ev: &Event, real_now: f64) -> Option<usize> {
        let Event::PacketEmitted {
            packet,
            from,
            to,
            at_ns,
            arrives_at_ns,
            payload,
        } = ev
        else {
            return None;
        };
        if arrives_at_ns <= at_ns {
            return None;
        }
        if let Value::Variant { tag, .. } = payload {
            if tag == "pull" || tag == "wake" {
                return None;
            }
        }

        let k = self.k.clamp(Self::K_MIN, Self::K_MAX);
        let clamp = self
            .node_arrivals
            .get(from)
            .and_then(|n| n.trigger_for(*at_ns))
            .unwrap_or(f64::NEG_INFINITY);
        let emit_real = real_now.max(clamp);
        let sim_latency_s = (arrives_at_ns - at_ns) as f64 * 1e-9;

        if from == to {
            // Self-loop: no spatial travel, but the dwell contributes
            // a causal arrival at the node so subsequent emits clamp
            // correctly. Unscaled (no `k`) so a 200ms service time
            // produces a 200ms causal step, not `200ms × k`.
            let arrive_real = emit_real + sim_latency_s;
            self.node_arrivals
                .entry(*to)
                .or_default()
                .push(*arrives_at_ns, arrive_real);
            return None;
        }

        let arrive_real = emit_real + sim_latency_s * k;
        self.node_arrivals
            .entry(*to)
            .or_default()
            .push(*arrives_at_ns, arrive_real);

        let vp = VisualPacket {
            packet_id: *packet,
            from: *from,
            to: *to,
            payload: payload.clone(),
            emit_real,
            arrive_real,
        };
        self.packets.push(vp);
        Some(self.packets.len() - 1)
    }

    /// Iterator over packets visible at `t`, with their progress.
    /// Public for tests; the trait method writes into a buffer
    /// instead.
    pub fn visible_at<'a>(&'a self, t: f64) -> impl Iterator<Item = (&'a VisualPacket, f32)> + 'a {
        self.packets.iter().filter_map(move |p| {
            if t < p.emit_real || t >= p.arrive_real {
                return None;
            }
            let denom = p.arrive_real - p.emit_real;
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            Some((p, prog))
        })
    }

    fn gc_internal(&mut self, cutoff: f64) {
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.node_arrivals.values_mut() {
            n.gc(cutoff);
        }
        self.node_arrivals.retain(|_, n| n.len() > 0);
    }

    pub fn reset(&mut self) {
        self.packets.clear();
        self.node_arrivals.clear();
    }
}

impl VisualStrategy for VisualTimeline {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        for ev in ctx.new_events {
            let real_now = real_now_for_event(ev, &ctx.time);
            self.ingest(ev, real_now);
        }
        self.gc_internal(ctx.time.visual_now - VISUAL_GC_KEEP_PAST_S);
        let t = ctx.time.visual_now;
        for p in &self.packets {
            if t < p.emit_real || t >= p.arrive_real {
                continue;
            }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            out.push((p.clone(), prog));
        }
    }
    fn invalidate(&mut self) {
        self.reset();
    }
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64 {
        // Two terms: visibility (sim_lat × k in real time =
        // sim_lat × k in sim ns under lockstep) + chain reach
        // (max_edge_lat × MAX_CHAIN_DEPTH).
        let max_lat = if max_edge_lat_ns == 0 {
            FALLBACK_MAX_EDGE_LAT_NS
        } else {
            max_edge_lat_ns
        };
        let k_eff = k.max(1.0);
        let visibility = (max_lat as f64 * k_eff) as u64;
        let chain = max_lat.saturating_mul(MAX_CHAIN_DEPTH as u64);
        visibility.saturating_add(chain)
    }
    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(Self::K_MIN, Self::K_MAX);
    }
    fn k(&self) -> f64 {
        self.k
    }
}

/// Conservative chain depth used by event-history strategies' rewind
/// lookback. Deep enough to cover request → router → worker →
/// response chains plus a safety margin.
pub const MAX_CHAIN_DEPTH: u32 = 8;
/// Fallback edge latency ceiling when the host can't statically
/// bound it (random-distribution latencies, slot-driven exprs).
pub const FALLBACK_MAX_EDGE_LAT_NS: u64 = 100_000_000;

// ────────────────────────────────────────────────────────────
// Common per-event parsing for event-history strategies
// ────────────────────────────────────────────────────────────

struct ParsedEmit<'a> {
    packet_id: PacketId,
    from: NodeId,
    to: NodeId,
    payload: &'a Value,
    at_ns: u64,
    arrives_at_ns: u64,
    sim_latency_s: f64,
    is_self_loop: bool,
}

fn parse_emit(ev: &Event) -> Option<ParsedEmit<'_>> {
    let Event::PacketEmitted {
        packet,
        from,
        to,
        at_ns,
        arrives_at_ns,
        payload,
    } = ev
    else {
        return None;
    };
    if arrives_at_ns <= at_ns {
        return None;
    }
    if let Value::Variant { tag, .. } = payload {
        if tag == "pull" || tag == "wake" {
            return None;
        }
    }
    Some(ParsedEmit {
        packet_id: *packet,
        from: *from,
        to: *to,
        payload,
        at_ns: *at_ns,
        arrives_at_ns: *arrives_at_ns,
        sim_latency_s: (arrives_at_ns - at_ns) as f64 * 1e-9,
        is_self_loop: from == to,
    })
}

#[derive(Clone, Debug, Default)]
struct EdgeStream {
    recent: VecDeque<f64>,
    last_visual_emit_real: f64,
}

// ────────────────────────────────────────────────────────────
// RateSampled
// ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RateSampled {
    edges: HashMap<(NodeId, NodeId), EdgeStream>,
    packets: Vec<VisualPacket>,
    k: f64,
}

impl Default for RateSampled {
    fn default() -> Self {
        Self {
            edges: HashMap::new(),
            packets: Vec::new(),
            k: VisualTimeline::K_DEFAULT,
        }
    }
}

impl RateSampled {
    pub const RATE_WINDOW_SEC: f64 = 1.0;
    pub const MAX_VISUAL_RATE_PER_EDGE: f64 = 10.0;

    pub fn new(k: f64) -> Self {
        Self {
            k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX),
            ..Self::default()
        }
    }

    pub fn ingest(&mut self, ev: &Event, real_now: f64) {
        if !is_visible_event(ev) {
            return;
        }
        let Event::PacketEmitted {
            packet,
            from,
            to,
            at_ns,
            arrives_at_ns,
            payload,
        } = ev
        else {
            return;
        };
        let stream = self.edges.entry((*from, *to)).or_insert(EdgeStream {
            recent: VecDeque::new(),
            last_visual_emit_real: f64::NEG_INFINITY,
        });
        stream.recent.push_back(real_now);
        let cutoff = real_now - Self::RATE_WINDOW_SEC;
        while stream.recent.front().map_or(false, |&t| t < cutoff) {
            stream.recent.pop_front();
        }
        let rate = stream.recent.len() as f64 / Self::RATE_WINDOW_SEC;
        let visual_period = if rate <= Self::MAX_VISUAL_RATE_PER_EDGE {
            0.0
        } else {
            1.0 / Self::MAX_VISUAL_RATE_PER_EDGE
        };
        if real_now < stream.last_visual_emit_real + visual_period {
            return;
        }
        stream.last_visual_emit_real = real_now;

        let k = self.k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
        let sim_latency_s = (arrives_at_ns - at_ns) as f64 * 1e-9;
        let arrive_real = real_now + sim_latency_s * k;

        self.packets.push(VisualPacket {
            packet_id: *packet,
            from: *from,
            to: *to,
            payload: payload.clone(),
            emit_real: real_now,
            arrive_real,
        });
    }

    fn gc_internal(&mut self, cutoff: f64) {
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for stream in self.edges.values_mut() {
            while stream.recent.front().map_or(false, |&ts| ts < cutoff) {
                stream.recent.pop_front();
            }
        }
        self.edges.retain(|_, s| !s.recent.is_empty());
    }
}

impl VisualStrategy for RateSampled {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        for ev in ctx.new_events {
            let real_now = real_now_for_event(ev, &ctx.time);
            self.ingest(ev, real_now);
        }
        self.gc_internal(ctx.time.visual_now - VISUAL_GC_KEEP_PAST_S);
        let t = ctx.time.visual_now;
        for p in &self.packets {
            if t < p.emit_real || t >= p.arrive_real {
                continue;
            }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            out.push((p.clone(), prog));
        }
    }
    fn invalidate(&mut self) {
        self.edges.clear();
        self.packets.clear();
    }
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64 {
        let max_lat = if max_edge_lat_ns == 0 {
            FALLBACK_MAX_EDGE_LAT_NS
        } else {
            max_edge_lat_ns
        };
        let k_eff = k.max(1.0);
        (max_lat as f64 * k_eff) as u64
    }
    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }
    fn k(&self) -> f64 {
        self.k
    }
}

// ────────────────────────────────────────────────────────────
// DropOrphans
// ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DropOrphans {
    edges: HashMap<(NodeId, NodeId), EdgeStream>,
    packets: Vec<VisualPacket>,
    visible_arrivals: HashMap<NodeId, NodeArrivals>,
    any_arrivals: HashMap<NodeId, NodeArrivals>,
    k: f64,
}

impl Default for DropOrphans {
    fn default() -> Self {
        Self {
            edges: HashMap::new(),
            packets: Vec::new(),
            visible_arrivals: HashMap::new(),
            any_arrivals: HashMap::new(),
            k: VisualTimeline::K_DEFAULT,
        }
    }
}

impl DropOrphans {
    pub fn new(k: f64) -> Self {
        Self {
            k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX),
            ..Self::default()
        }
    }

    pub fn ingest(&mut self, ev: &Event, real_now: f64) {
        let Some(p) = parse_emit(ev) else {
            return;
        };
        if p.is_self_loop {
            self.any_arrivals
                .entry(p.to)
                .or_default()
                .push(p.arrives_at_ns, real_now + p.sim_latency_s);
            return;
        }
        let trigger_visible = self
            .visible_arrivals
            .get(&p.from)
            .and_then(|n| n.trigger_for(p.at_ns));
        let trigger_any = self
            .any_arrivals
            .get(&p.from)
            .and_then(|n| n.trigger_for(p.at_ns));
        let is_orphan = trigger_any.is_some() && trigger_visible.is_none();
        if is_orphan {
            self.any_arrivals
                .entry(p.to)
                .or_default()
                .push(p.arrives_at_ns, real_now);
            return;
        }

        let stream = self.edges.entry((p.from, p.to)).or_insert(EdgeStream {
            recent: VecDeque::new(),
            last_visual_emit_real: f64::NEG_INFINITY,
        });
        stream.recent.push_back(real_now);
        let cutoff = real_now - RateSampled::RATE_WINDOW_SEC;
        while stream.recent.front().map_or(false, |&t| t < cutoff) {
            stream.recent.pop_front();
        }
        let rate = stream.recent.len() as f64 / RateSampled::RATE_WINDOW_SEC;
        let visual_period = if rate <= RateSampled::MAX_VISUAL_RATE_PER_EDGE {
            0.0
        } else {
            1.0 / RateSampled::MAX_VISUAL_RATE_PER_EDGE
        };
        if real_now < stream.last_visual_emit_real + visual_period {
            self.any_arrivals
                .entry(p.to)
                .or_default()
                .push(p.arrives_at_ns, real_now);
            return;
        }
        stream.last_visual_emit_real = real_now;

        let k = self.k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
        let clamp = trigger_visible.unwrap_or(f64::NEG_INFINITY);
        let emit_real = real_now.max(clamp);
        let arrive_real = emit_real + p.sim_latency_s * k;

        self.packets.push(VisualPacket {
            packet_id: p.packet_id,
            from: p.from,
            to: p.to,
            payload: p.payload.clone(),
            emit_real,
            arrive_real,
        });
        self.visible_arrivals
            .entry(p.to)
            .or_default()
            .push(p.arrives_at_ns, arrive_real);
        self.any_arrivals
            .entry(p.to)
            .or_default()
            .push(p.arrives_at_ns, arrive_real);
    }

    fn gc_internal(&mut self, cutoff: f64) {
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.visible_arrivals.values_mut() {
            n.gc(cutoff);
        }
        for n in self.any_arrivals.values_mut() {
            n.gc(cutoff);
        }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
        self.any_arrivals.retain(|_, n| n.len() > 0);
        for stream in self.edges.values_mut() {
            while stream.recent.front().map_or(false, |&ts| ts < cutoff) {
                stream.recent.pop_front();
            }
        }
        self.edges.retain(|_, s| !s.recent.is_empty());
    }
}

impl VisualStrategy for DropOrphans {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        for ev in ctx.new_events {
            let real_now = real_now_for_event(ev, &ctx.time);
            self.ingest(ev, real_now);
        }
        self.gc_internal(ctx.time.visual_now - VISUAL_GC_KEEP_PAST_S);
        let t = ctx.time.visual_now;
        for p in &self.packets {
            if t < p.emit_real || t >= p.arrive_real {
                continue;
            }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            out.push((p.clone(), prog));
        }
    }
    fn invalidate(&mut self) {
        self.edges.clear();
        self.packets.clear();
        self.visible_arrivals.clear();
        self.any_arrivals.clear();
    }
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64 {
        let max_lat = if max_edge_lat_ns == 0 {
            FALLBACK_MAX_EDGE_LAT_NS
        } else {
            max_edge_lat_ns
        };
        let k_eff = k.max(1.0);
        let visibility = (max_lat as f64 * k_eff) as u64;
        let chain = max_lat.saturating_mul(MAX_CHAIN_DEPTH as u64);
        visibility.saturating_add(chain)
    }
    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }
    fn k(&self) -> f64 {
        self.k
    }
}

// ────────────────────────────────────────────────────────────
// CausalRateSampled
// ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CausalRateSampled {
    edges: HashMap<(NodeId, NodeId), EdgeStream>,
    packets: Vec<VisualPacket>,
    visible_arrivals: HashMap<NodeId, NodeArrivals>,
    throttled: HashMap<NodeId, VecDeque<ThrottledEmit>>,
    k: f64,
}

#[derive(Clone, Debug)]
struct ThrottledEmit {
    packet_id: PacketId,
    from: NodeId,
    to: NodeId,
    payload: Value,
    arrives_at_ns: u64,
    sim_latency_s: f64,
    ingest_real: f64,
}

const CAUSAL_THROTTLED_BUFFER_LEN: usize = 32;

impl Default for CausalRateSampled {
    fn default() -> Self {
        Self {
            edges: HashMap::new(),
            packets: Vec::new(),
            visible_arrivals: HashMap::new(),
            throttled: HashMap::new(),
            k: VisualTimeline::K_DEFAULT,
        }
    }
}

impl CausalRateSampled {
    pub fn new(k: f64) -> Self {
        Self {
            k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX),
            ..Self::default()
        }
    }

    fn pop_throttled_trigger(&mut self, node: NodeId, at_ns_limit: u64) -> Option<ThrottledEmit> {
        let buf = self.throttled.get_mut(&node)?;
        let pos = buf.iter().rposition(|t| t.arrives_at_ns <= at_ns_limit)?;
        buf.remove(pos)
    }

    fn promote(&mut self, ev: ThrottledEmit, real_now: f64) -> f64 {
        let k = self.k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
        let promo_duration = (ev.sim_latency_s * k).min(0.15);
        let emit_real = real_now;
        let arrive_real = emit_real + promo_duration;
        self.packets.push(VisualPacket {
            packet_id: ev.packet_id,
            from: ev.from,
            to: ev.to,
            payload: ev.payload,
            emit_real,
            arrive_real,
        });
        self.visible_arrivals
            .entry(ev.to)
            .or_default()
            .push(ev.arrives_at_ns, arrive_real);
        arrive_real
    }

    pub fn ingest(&mut self, ev: &Event, real_now: f64) {
        let Some(p) = parse_emit(ev) else {
            return;
        };
        if p.is_self_loop {
            self.visible_arrivals
                .entry(p.to)
                .or_default()
                .push(p.arrives_at_ns, real_now + p.sim_latency_s);
            return;
        }

        let stream = self.edges.entry((p.from, p.to)).or_insert(EdgeStream {
            recent: VecDeque::new(),
            last_visual_emit_real: f64::NEG_INFINITY,
        });
        stream.recent.push_back(real_now);
        let cutoff = real_now - RateSampled::RATE_WINDOW_SEC;
        while stream.recent.front().map_or(false, |&t| t < cutoff) {
            stream.recent.pop_front();
        }
        let rate = stream.recent.len() as f64 / RateSampled::RATE_WINDOW_SEC;
        let visual_period = if rate <= RateSampled::MAX_VISUAL_RATE_PER_EDGE {
            0.0
        } else {
            1.0 / RateSampled::MAX_VISUAL_RATE_PER_EDGE
        };
        let is_throttled = real_now < stream.last_visual_emit_real + visual_period;

        if is_throttled {
            let buf = self.throttled.entry(p.to).or_default();
            if buf.len() >= CAUSAL_THROTTLED_BUFFER_LEN {
                buf.pop_front();
            }
            buf.push_back(ThrottledEmit {
                packet_id: p.packet_id,
                from: p.from,
                to: p.to,
                payload: p.payload.clone(),
                arrives_at_ns: p.arrives_at_ns,
                sim_latency_s: p.sim_latency_s,
                ingest_real: real_now,
            });
            return;
        }
        stream.last_visual_emit_real = real_now;

        let visible_trigger = self
            .visible_arrivals
            .get(&p.from)
            .and_then(|n| n.trigger_for_full(p.at_ns));
        let buffered_trigger_at_ns = self
            .throttled
            .get(&p.from)
            .and_then(|buf| buf.iter().rev().find(|t| t.arrives_at_ns <= p.at_ns))
            .map(|t| t.arrives_at_ns);
        let prefer_buffered = match (buffered_trigger_at_ns, visible_trigger) {
            (Some(b), Some((v, _))) => b > v,
            (Some(_), None) => true,
            _ => false,
        };
        let clamp = if prefer_buffered {
            let throttled_cause = self
                .pop_throttled_trigger(p.from, p.at_ns)
                .expect("buffered trigger must still be present after lookup");
            Some(self.promote(throttled_cause, real_now))
        } else {
            visible_trigger.map(|(_, real)| real)
        };

        let k = self.k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
        let clamp = clamp.unwrap_or(f64::NEG_INFINITY);
        let emit_real = real_now.max(clamp);
        let arrive_real = emit_real + p.sim_latency_s * k;

        self.packets.push(VisualPacket {
            packet_id: p.packet_id,
            from: p.from,
            to: p.to,
            payload: p.payload.clone(),
            emit_real,
            arrive_real,
        });
        self.visible_arrivals
            .entry(p.to)
            .or_default()
            .push(p.arrives_at_ns, arrive_real);
    }

    fn gc_internal(&mut self, cutoff: f64) {
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.visible_arrivals.values_mut() {
            n.gc(cutoff);
        }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
        for buf in self.throttled.values_mut() {
            while buf.front().map_or(false, |t| t.ingest_real < cutoff) {
                buf.pop_front();
            }
        }
        self.throttled.retain(|_, b| !b.is_empty());
        for stream in self.edges.values_mut() {
            while stream.recent.front().map_or(false, |&ts| ts < cutoff) {
                stream.recent.pop_front();
            }
        }
        self.edges.retain(|_, s| !s.recent.is_empty());
    }
}

impl VisualStrategy for CausalRateSampled {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        for ev in ctx.new_events {
            let real_now = real_now_for_event(ev, &ctx.time);
            self.ingest(ev, real_now);
        }
        self.gc_internal(ctx.time.visual_now - VISUAL_GC_KEEP_PAST_S);
        let t = ctx.time.visual_now;
        for p in &self.packets {
            if t < p.emit_real || t >= p.arrive_real {
                continue;
            }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            out.push((p.clone(), prog));
        }
    }
    fn invalidate(&mut self) {
        self.edges.clear();
        self.packets.clear();
        self.visible_arrivals.clear();
        self.throttled.clear();
    }
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64 {
        let max_lat = if max_edge_lat_ns == 0 {
            FALLBACK_MAX_EDGE_LAT_NS
        } else {
            max_edge_lat_ns
        };
        let k_eff = k.max(1.0);
        let visibility = (max_lat as f64 * k_eff) as u64;
        let chain = max_lat.saturating_mul(MAX_CHAIN_DEPTH as u64);
        visibility.saturating_add(chain)
    }
    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }
    fn k(&self) -> f64 {
        self.k
    }
}

// ────────────────────────────────────────────────────────────
// BundleSummarized
// ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BundleSummarized {
    last_bundle_at_ns: HashMap<(NodeId, NodeId), u64>,
    packets: Vec<VisualPacket>,
    visible_arrivals: HashMap<NodeId, NodeArrivals>,
    k: f64,
}

impl Default for BundleSummarized {
    fn default() -> Self {
        Self {
            last_bundle_at_ns: HashMap::new(),
            packets: Vec::new(),
            visible_arrivals: HashMap::new(),
            k: VisualTimeline::K_DEFAULT,
        }
    }
}

impl BundleSummarized {
    pub const BUNDLE_WINDOW_NS: u64 = 100_000_000;

    pub fn new(k: f64) -> Self {
        Self {
            k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX),
            ..Self::default()
        }
    }

    pub fn ingest(&mut self, ev: &Event, real_now: f64) {
        let Some(p) = parse_emit(ev) else {
            return;
        };
        if p.is_self_loop {
            self.visible_arrivals
                .entry(p.to)
                .or_default()
                .push(p.arrives_at_ns, real_now + p.sim_latency_s);
            return;
        }
        let key = (p.from, p.to);
        let new_bundle = match self.last_bundle_at_ns.get(&key) {
            Some(&last_at) => p.at_ns.saturating_sub(last_at) >= Self::BUNDLE_WINDOW_NS,
            None => true,
        };
        if !new_bundle {
            return;
        }
        self.last_bundle_at_ns.insert(key, p.at_ns);

        let trigger = self
            .visible_arrivals
            .get(&p.from)
            .and_then(|n| n.trigger_for(p.at_ns));
        let k = self.k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
        let clamp = trigger.unwrap_or(f64::NEG_INFINITY);
        let emit_real = real_now.max(clamp);
        let arrive_real = emit_real + p.sim_latency_s * k;

        self.packets.push(VisualPacket {
            packet_id: p.packet_id,
            from: p.from,
            to: p.to,
            payload: p.payload.clone(),
            emit_real,
            arrive_real,
        });
        self.visible_arrivals
            .entry(p.to)
            .or_default()
            .push(p.arrives_at_ns, arrive_real);
    }

    fn gc_internal(&mut self, cutoff: f64) {
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.visible_arrivals.values_mut() {
            n.gc(cutoff);
        }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
    }
}

impl VisualStrategy for BundleSummarized {
    fn frame_into(&mut self, ctx: &VisualFrameCtx<'_>, out: &mut Vec<(VisualPacket, f32)>) {
        for ev in ctx.new_events {
            let real_now = real_now_for_event(ev, &ctx.time);
            self.ingest(ev, real_now);
        }
        self.gc_internal(ctx.time.visual_now - VISUAL_GC_KEEP_PAST_S);
        let t = ctx.time.visual_now;
        for p in &self.packets {
            if t < p.emit_real || t >= p.arrive_real {
                continue;
            }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            out.push((p.clone(), prog));
        }
    }
    fn invalidate(&mut self) {
        self.last_bundle_at_ns.clear();
        self.packets.clear();
        self.visible_arrivals.clear();
    }
    fn rewind_lookback_ns(&self, k: f64, max_edge_lat_ns: u64) -> u64 {
        let max_lat = if max_edge_lat_ns == 0 {
            FALLBACK_MAX_EDGE_LAT_NS
        } else {
            max_edge_lat_ns
        };
        let k_eff = k.max(1.0);
        let visibility = (max_lat as f64 * k_eff) as u64;
        let chain = max_lat.saturating_mul(MAX_CHAIN_DEPTH as u64);
        visibility.saturating_add(chain)
    }
    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }
    fn k(&self) -> f64 {
        self.k
    }
}

// Suppress unused-warnings for the `HashSet` import — kept available
// for downstream test helpers that historically built sets of
// `PacketId`. Remove if no consumer materialises.
#[allow(dead_code)]
fn _hashset_keep_alive() -> HashSet<PacketId> {
    HashSet::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flow::{NodeId, PacketId};

    fn emit(id: u64, from: u64, to: u64, at: u64, arr: u64, payload: Value) -> Event {
        Event::PacketEmitted {
            packet: PacketId(id),
            from: NodeId(from),
            to: NodeId(to),
            at_ns: at,
            arrives_at_ns: arr,
            payload,
        }
    }
    fn data_pkt(slot: i64) -> Value {
        Value::variant("packet", Value::Int(slot))
    }

    #[test]
    fn gen_emit_uses_real_now_when_no_trigger() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 5.0);
        let p = &tl.packets[0];
        assert!((p.emit_real - 5.0).abs() < 1e-12);
        assert!((p.arrive_real - 5.1).abs() < 1e-9);
    }

    #[test]
    fn response_waits_for_specific_triggering_arrival() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        tl.ingest(&emit(2, 20, 30, 1_000_000, 2_000_000, data_pkt(0)), 0.001);
        let gen_pkt = &tl.packets[0];
        let resp = &tl.packets[1];
        assert!((resp.emit_real - gen_pkt.arrive_real).abs() < 1e-9);
    }

    #[test]
    fn router_does_not_wait_for_unrelated_later_arrival() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        tl.ingest(&emit(2, 20, 30, 1_000_000, 2_000_000, data_pkt(0)), 0.001);
        tl.ingest(&emit(3, 11, 20, 9_000_000, 10_000_000, data_pkt(0)), 0.009);
        let resp = &tl.packets[1];
        let gen1 = &tl.packets[0];
        assert!((resp.emit_real - gen1.arrive_real).abs() < 1e-9);
    }

    #[test]
    fn ingest_skips_pull_wake_selfloop_zero_duration() {
        let mut tl = VisualTimeline::new(100.0);
        assert!(tl
            .ingest(
                &emit(1, 10, 20, 0, 1_000_000, Value::variant("pull", Value::Nil)),
                0.0
            )
            .is_none());
        assert!(tl
            .ingest(
                &emit(2, 10, 20, 0, 1_000_000, Value::variant("wake", Value::Nil)),
                0.0
            )
            .is_none());
        assert!(tl
            .ingest(&emit(3, 10, 10, 0, 1_000_000, data_pkt(0)), 0.0)
            .is_none());
        assert!(tl
            .ingest(&emit(4, 10, 20, 100, 100, data_pkt(0)), 0.0)
            .is_none());
        assert!(tl.packets.is_empty());
    }

    #[test]
    fn visible_at_window() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        assert_eq!(tl.visible_at(0.0).count(), 1);
        assert_eq!(tl.visible_at(0.05).count(), 1);
        assert_eq!(tl.visible_at(0.1).count(), 0);
        assert_eq!(tl.visible_at(-0.001).count(), 0);
    }

    #[test]
    fn parallel_emits_overlap_on_same_edge() {
        let mut tl = VisualTimeline::new(100.0);
        for i in 0..30 {
            let at = (i * 33_333_333) as u64;
            let real = (i as f64) * (1.0 / 30.0);
            tl.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
        }
        let vis = tl.visible_at(0.5).count();
        assert!(vis >= 3, "expected concurrent visuals, got {}", vis);
    }

    #[test]
    fn set_k_doesnt_teleport_existing_packets() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        let before = tl.packets[0].clone();
        VisualStrategy::set_k(&mut tl, 10.0);
        assert_eq!(tl.packets[0], before);
        tl.ingest(&emit(2, 11, 20, 0, 1_000_000, data_pkt(0)), 1.0);
        let p2 = &tl.packets[1];
        assert!((p2.arrive_real - p2.emit_real - 0.01).abs() < 1e-9);
    }

    // ── RateSampled ──

    #[test]
    fn rate_sampled_below_cap_emits_1_to_1() {
        let mut rs = RateSampled::new(100.0);
        for i in 0..5 {
            let real = i as f64 * 0.2;
            let at = (i * 200_000_000) as u64;
            rs.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
        }
        assert_eq!(rs.packets.len(), 5);
    }

    #[test]
    fn rate_sampled_above_cap_throttles() {
        let mut rs = RateSampled::new(100.0);
        for i in 0..100 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            rs.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
        }
        assert!(rs.packets.len() < 30 && rs.packets.len() > 5);
    }

    #[test]
    fn rate_sampled_per_edge_independent() {
        let mut rs = RateSampled::new(100.0);
        for i in 0..50 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            rs.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
            rs.ingest(
                &emit(1000 + i as u64, 30, 40, at, at + 1_000_000, data_pkt(0)),
                real,
            );
        }
        let edge_a = rs.packets.iter().filter(|p| p.from == NodeId(10)).count();
        let edge_b = rs.packets.iter().filter(|p| p.from == NodeId(30)).count();
        assert!(edge_a > 0 && edge_b > 0);
    }

    #[test]
    fn rate_sampled_skips_pull_wake_selfloop() {
        let mut rs = RateSampled::new(100.0);
        rs.ingest(
            &emit(1, 10, 20, 0, 1_000_000, Value::variant("pull", Value::Nil)),
            0.0,
        );
        rs.ingest(
            &emit(2, 10, 20, 0, 1_000_000, Value::variant("wake", Value::Nil)),
            0.0,
        );
        rs.ingest(&emit(3, 10, 10, 0, 1_000_000, data_pkt(0)), 0.0);
        assert!(rs.packets.is_empty());
    }

    #[test]
    fn strategy_cycle_round_trips() {
        let mut s = Strategy::new_of_kind(StrategyKind::ALL[0], 50.0);
        for expected in StrategyKind::ALL
            .iter()
            .skip(1)
            .chain(std::iter::once(&StrategyKind::ALL[0]))
        {
            s.cycle();
            assert_eq!(&s.kind(), expected);
        }
    }

    // ── DropOrphans ──

    #[test]
    fn drop_orphans_root_emit_visible() {
        let mut s = DropOrphans::new(100.0);
        s.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        assert_eq!(s.packets.len(), 1);
    }

    #[test]
    fn drop_orphans_response_dropped_when_request_throttled() {
        let mut s = DropOrphans::new(100.0);
        for i in 0..50 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            s.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
            s.ingest(
                &emit(
                    1000 + i as u64,
                    20,
                    10,
                    at + 1_500_000,
                    at + 2_500_000,
                    data_pkt(0),
                ),
                real + 0.0015,
            );
        }
        let req_count = s.packets.iter().filter(|p| p.from == NodeId(10)).count();
        let resp_count = s.packets.iter().filter(|p| p.from == NodeId(20)).count();
        assert!(resp_count <= req_count);
    }

    // ── CausalRateSampled ──

    #[test]
    fn causal_rate_sampled_promotes_throttled_cause() {
        let mut s = CausalRateSampled::new(100.0);
        for i in 0..11 {
            let real = i as f64 * 0.001;
            let at = (i * 1_000_000) as u64;
            s.ingest(
                &emit(i as u64, 10, 20, at, at + 100_000, data_pkt(0)),
                real,
            );
        }
        let req_count_before_resp = s.packets.iter().filter(|p| p.from == NodeId(10)).count();
        assert_eq!(req_count_before_resp, 10);
        s.ingest(
            &emit(9999, 20, 10, 10_500_000, 10_600_000, data_pkt(0)),
            0.012,
        );
        let req_count_after = s.packets.iter().filter(|p| p.from == NodeId(10)).count();
        let resp_count = s.packets.iter().filter(|p| p.from == NodeId(20)).count();
        assert_eq!(resp_count, 1);
        assert_eq!(req_count_after, req_count_before_resp + 1);
    }

    #[test]
    fn causal_rate_sampled_preserves_low_rate_chains() {
        let mut s = CausalRateSampled::new(100.0);
        for i in 0..3 {
            let real = i as f64 * 0.2;
            let at = (i * 200_000_000) as u64;
            s.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
            s.ingest(
                &emit(
                    1000 + i as u64,
                    20,
                    10,
                    at + 2_000_000,
                    at + 3_000_000,
                    data_pkt(0),
                ),
                real + 0.002,
            );
        }
        assert_eq!(s.packets.len(), 6);
    }

    // ── BundleSummarized ──

    #[test]
    fn bundle_summarized_collapses_close_emits() {
        let mut s = BundleSummarized::new(100.0);
        for i in 0..10 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            s.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
        }
        assert_eq!(s.packets.len(), 1);
    }

    #[test]
    fn bundle_summarized_separates_distant_emits() {
        let mut s = BundleSummarized::new(100.0);
        for i in 0..3 {
            let real = i as f64 * 0.2;
            let at = (i * 200_000_000) as u64;
            s.ingest(
                &emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)),
                real,
            );
        }
        assert_eq!(s.packets.len(), 3);
    }

    // ── SimMirror ──

    #[test]
    fn sim_mirror_renders_in_flight_packets() {
        use flow::sim::Scheduled;
        use flow::Packet;
        use std::collections::BTreeMap;

        let mut edges: BTreeMap<EdgeId, Edge> = BTreeMap::new();
        edges.insert(
            EdgeId(1),
            Edge {
                id: EdgeId(1),
                from: NodeId(10),
                from_port: None,
                to: NodeId(20),
                to_port: None,
                latency_ns: flow::Expr::int(1_000_000),
                last_sent_seq: None,
            },
        );
        let in_flight = vec![Scheduled {
            arrives_at_ns: 1_000_000,
            packet: Packet {
                id: PacketId(7),
                payload: data_pkt(0),
                from_edge: None,
                metadata: BTreeMap::new(),
                return_path: Vec::new(),
                emitted_at_ns: 0,
            },
            edge: EdgeId(1),
            deliver_to: NodeId(20),
        }];
        let mut sm = SimMirror::new();
        let ctx = VisualFrameCtx {
            time: TimeCursor {
                sim_now_ns: 500_000,
                visual_now: 0.5,
                k: 1.0,
            },
            sim: SimView {
                edges: &edges,
                in_flight: &in_flight,
            },
            new_events: &[],
        };
        let mut out = Vec::new();
        sm.frame_into(&ctx, &mut out);
        assert_eq!(out.len(), 1);
        let (p, prog) = &out[0];
        assert_eq!(p.from, NodeId(10));
        assert_eq!(p.to, NodeId(20));
        assert!((prog - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sim_mirror_skips_self_loop_and_control() {
        use flow::sim::Scheduled;
        use flow::Packet;
        use std::collections::BTreeMap;

        let mut edges: BTreeMap<EdgeId, Edge> = BTreeMap::new();
        edges.insert(
            EdgeId(1),
            Edge {
                id: EdgeId(1),
                from: NodeId(10),
                from_port: None,
                to: NodeId(10),
                to_port: None,
                latency_ns: flow::Expr::int(1_000_000),
                last_sent_seq: None,
            },
        );
        let in_flight = vec![Scheduled {
            arrives_at_ns: 1_000_000,
            packet: Packet {
                id: PacketId(7),
                payload: Value::variant("pull", Value::Nil),
                from_edge: None,
                metadata: BTreeMap::new(),
                return_path: Vec::new(),
                emitted_at_ns: 0,
            },
            edge: EdgeId(1),
            deliver_to: NodeId(10),
        }];
        let mut sm = SimMirror::new();
        let ctx = VisualFrameCtx {
            time: TimeCursor {
                sim_now_ns: 500_000,
                visual_now: 0.5,
                k: 1.0,
            },
            sim: SimView {
                edges: &edges,
                in_flight: &in_flight,
            },
            new_events: &[],
        };
        let mut out = Vec::new();
        sm.frame_into(&ctx, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn sim_mirror_lookback_is_zero() {
        let s = SimMirror::new();
        assert_eq!(s.rewind_lookback_ns(400.0, 1_000_000), 0);
    }
}
