//! Pure-data visual timeline — the formal mapping from sim events to
//! on-screen packets. NO Bevy types in here.
//!
//! # The F12 formalism — real-time ingest + per-node causal clamp
//!
//! Two knobs, independent:
//!
//!   1. `SimClock.multiplier` — how fast sim advances per wall-second.
//!   2. `VisualTimeline.k` — real-seconds of animation per sim-second
//!      of packet latency.
//!
//! Ingestion (given `real_now` = wall-clock at ingest):
//!
//!   trigger       = the specific prior arrival at `from` whose
//!                   `arrives_at_ns <= this.at_ns`, picked by an
//!                   upper-bound search over recorded arrivals.
//!   clamp         = trigger.arrive_real  (or -∞ if no trigger)
//!   emit_real     = max(real_now, clamp)
//!   arrive_real   = emit_real + (arrives_at_ns - at_ns) * 1e-9 * k
//!
//! Then record this packet's `(arrives_at_ns, arrive_real)` under
//! node `to`, so future emits leaving `to` can find it as a trigger.
//!
//! ## What this buys
//!
//! - Packets appear on screen as the sim emits them (`emit_real` is
//!   close to `real_now`, except when causality pushes it forward).
//! - Each packet is visible for `latency * k` real seconds — user
//!   controls that with `-` / `=`.
//! - Multiple packets on the same edge can overlap — a busy sim
//!   reads as a busy canvas (the user-asked "constant stream").
//! - Causal pairs are preserved: a router's response to request R
//!   starts only after R visually arrived. The clamp is tied to R
//!   specifically (via `at_ns <= emit.at_ns` upper-bound search),
//!   NOT to the max of all arrivals — so unrelated traffic at the
//!   node can't stall the response.
//!
//! ## What it gives up vs a stretched-sim replay
//!
//! Absolute timing between unrelated packets isn't preserved.
//! "Gen 1 emitted 10 ms before Gen 2" may collapse to "same frame"
//! if both ingest at `real_now`. That's the trade: we prioritized
//! "show it now" over "render sim time faithfully." Causal
//! relationships still hold — which is what the user flagged as
//! non-negotiable.
//!
//! # Invariants (property-tested)
//!
//! - V1: `arrive_real > emit_real` for every packet.
//! - V2: 1:1 with visible-eligible `PacketEmitted` events.
//! - V3: `visible_at(t)` returns only packets with
//!       `emit_real <= t < arrive_real`.
//! - V4: progress monotonic 0 → 1 across the window.
//! - V5: `|visible_at(t)| ≤ |{p : emit_real ≤ t}|`.
//! - V6: determinism (same events + same `real_now` sequence ⇒
//!       same packets).
//! - V7: filters `pull`, `wake`, self-loops.
//! - C (causality): if sim event Q causally depends on P
//!   (`Q.at_ns ≥ P.arrives_at_ns` and `Q.from = P.to`), then
//!   `Q.emit_real ≥ P.arrive_real`.

use flow::{Event, NodeId, PacketId, Value};
use std::collections::{HashMap, HashSet, VecDeque};

/// Pluggable visual strategy. Implementations decide how sim events
/// map to on-screen packets — replay one-to-one, aggregate by flow
/// rate, etc. The host (edges.rs, packet_cloud.rs) talks to whatever
/// strategy is wired in via this trait.
///
/// `Strategy` (below) is the dispatch enum the Bevy resource holds.
pub trait VisualStrategy: Send + Sync + 'static {
    /// Ingest one sim event at wall-clock `real_now`. Strategies may
    /// emit zero, one, or many internal visual records — return value
    /// is intentionally not exposed (tests that need the index use
    /// the Replay-specific accessor).
    fn ingest(&mut self, ev: &Event, real_now: f64);

    /// Iterate visual packets currently on-screen at time `t`, paired
    /// with their normalised progress in [0, 1]. The packet-cloud
    /// renderer consumes this directly each frame.
    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a>;

    /// Drop the future-queued backlog at wall-clock `t`. Returns the
    /// packet ids that were removed so callers can despawn matching
    /// Bevy entities.
    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId>;

    /// Trim history older than `t - keep_past_s`.
    fn gc_before(&mut self, t: f64, keep_past_s: f64);

    /// Wipe all internal state.
    fn reset(&mut self);

    fn set_k(&mut self, new_k: f64);
    fn k(&self) -> f64;
}

/// Dispatch enum held by `VisualTimelineRes`. One variant per concrete
/// strategy; `VisualStrategy` impl below match-dispatches to the
/// inner type.
#[derive(Clone, Debug)]
pub enum Strategy {
    /// Faithful one-visual-per-sim-event replay with causal clamping.
    /// The historical default — preserves V3 (no reply visible before
    /// its req) and is what the property tests assume.
    Replay(VisualTimeline),
    /// Per-edge rolling-window flow-rate sampling. Emits visuals 1:1
    /// when an edge is below `MAX_VISUAL_RATE_PER_EDGE`, throttles
    /// once the sim outpaces the visual budget. Designed for
    /// readability under bursty/fast sims, at the cost of dropping
    /// individual events. Causal pairing across hops is NOT preserved
    /// (responses can animate without their cause).
    RateSampled(RateSampled),
    /// RateSampled-style throttle, plus drops downstream events whose
    /// triggering cause was throttled — every visible interaction is
    /// a complete chain. Sparser than RateSampled under bursts, but
    /// no orphaned responses.
    DropOrphans(DropOrphans),
    /// RateSampled-style throttle, but retroactively promotes a
    /// throttled cause when a downstream event needs it — chains stay
    /// intact without stacking unrelated traffic. The "best of both"
    /// path; uses Replay's causal clamping on the promoted leg.
    CausalRateSampled(CausalRateSampled),
    /// Coalesces bursts into one visual per causal bundle (events
    /// within a sim-time window on the same edge collapse into the
    /// first). Preserves chains, hides multiplicity. Future work:
    /// thicken the packet to convey bundle size.
    BundleSummarized(BundleSummarized),
}

/// Stable identifier for a strategy variant. Useful for HUD readouts,
/// the cycle hotkey, and persisting the user's choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyKind {
    Replay,
    RateSampled,
    DropOrphans,
    CausalRateSampled,
    BundleSummarized,
}

impl StrategyKind {
    pub const ALL: &'static [StrategyKind] = &[
        StrategyKind::Replay,
        StrategyKind::RateSampled,
        StrategyKind::DropOrphans,
        StrategyKind::CausalRateSampled,
        StrategyKind::BundleSummarized,
    ];

    pub fn label(self) -> &'static str {
        match self {
            StrategyKind::Replay => "replay",
            StrategyKind::RateSampled => "rate-sampled",
            StrategyKind::DropOrphans => "drop-orphans",
            StrategyKind::CausalRateSampled => "causal-rate",
            StrategyKind::BundleSummarized => "bundle",
        }
    }

    /// Index into `ALL`. Used to drive cycle-to-next without
    /// hand-rolling the table.
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
    /// Construct a fresh strategy of the requested kind, carrying
    /// over the visual scale `k`. Internal per-strategy state
    /// (causal records, rate windows, scheduled packets) is **not**
    /// migrated — the next batch of ingested events bootstraps the
    /// new strategy from empty. Cheap and predictable: the trade-off
    /// is a brief visual gap on switch, which is honest.
    pub fn new_of_kind(kind: StrategyKind, k: f64) -> Self {
        match kind {
            StrategyKind::Replay => Strategy::Replay(VisualTimeline::new(k)),
            StrategyKind::RateSampled => Strategy::RateSampled(RateSampled::new(k)),
            StrategyKind::DropOrphans => Strategy::DropOrphans(DropOrphans::new(k)),
            StrategyKind::CausalRateSampled => Strategy::CausalRateSampled(CausalRateSampled::new(k)),
            StrategyKind::BundleSummarized => Strategy::BundleSummarized(BundleSummarized::new(k)),
        }
    }

    pub fn kind(&self) -> StrategyKind {
        match self {
            Strategy::Replay(_) => StrategyKind::Replay,
            Strategy::RateSampled(_) => StrategyKind::RateSampled,
            Strategy::DropOrphans(_) => StrategyKind::DropOrphans,
            Strategy::CausalRateSampled(_) => StrategyKind::CausalRateSampled,
            Strategy::BundleSummarized(_) => StrategyKind::BundleSummarized,
        }
    }

    /// Replace `self` with a fresh strategy of `kind`, preserving
    /// `k`. Used by the HUD/hotkey toggle for runtime A/B.
    pub fn switch_to(&mut self, kind: StrategyKind) {
        if self.kind() == kind { return; }
        let k = self.k();
        *self = Strategy::new_of_kind(kind, k);
    }

    /// Cycle to the next strategy in `StrategyKind::ALL`. Bound to
    /// the `V` key in the palette/hotkey layer.
    pub fn cycle(&mut self) {
        self.switch_to(self.kind().next());
    }

    /// Borrow the inner `VisualTimeline` if this is the Replay
    /// strategy. Panics with a clear message otherwise — used by
    /// tests and diagnostics that depend on Replay's internal layout
    /// (the `packets` Vec, causal-arrival records, etc.) which other
    /// strategies don't have.
    pub fn as_replay(&self) -> &VisualTimeline {
        match self {
            Strategy::Replay(t) => t,
            other => panic!(
                "as_replay() called on non-Replay strategy ({:?}); \
                 Replay-specific fields aren't available here",
                other.kind(),
            ),
        }
    }

    pub fn as_replay_mut(&mut self) -> &mut VisualTimeline {
        match self {
            Strategy::Replay(t) => t,
            other => panic!(
                "as_replay_mut() called on non-Replay strategy ({:?}); \
                 Replay-specific fields aren't available here",
                other.kind(),
            ),
        }
    }
}

impl VisualStrategy for Strategy {
    fn ingest(&mut self, ev: &Event, real_now: f64) {
        match self {
            Strategy::Replay(t) => <VisualTimeline as VisualStrategy>::ingest(t, ev, real_now),
            Strategy::RateSampled(t) => t.ingest(ev, real_now),
            Strategy::DropOrphans(t) => t.ingest(ev, real_now),
            Strategy::CausalRateSampled(t) => t.ingest(ev, real_now),
            Strategy::BundleSummarized(t) => t.ingest(ev, real_now),
        }
    }
    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a> {
        match self {
            Strategy::Replay(s) => <VisualTimeline as VisualStrategy>::visible_at(s, t),
            Strategy::RateSampled(s) => s.visible_at(t),
            Strategy::DropOrphans(s) => s.visible_at(t),
            Strategy::CausalRateSampled(s) => s.visible_at(t),
            Strategy::BundleSummarized(s) => s.visible_at(t),
        }
    }
    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId> {
        match self {
            Strategy::Replay(s) => s.drop_pending_after(t),
            Strategy::RateSampled(s) => s.drop_pending_after(t),
            Strategy::DropOrphans(s) => s.drop_pending_after(t),
            Strategy::CausalRateSampled(s) => s.drop_pending_after(t),
            Strategy::BundleSummarized(s) => s.drop_pending_after(t),
        }
    }
    fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        match self {
            Strategy::Replay(s) => s.gc_before(t, keep_past_s),
            Strategy::RateSampled(s) => s.gc_before(t, keep_past_s),
            Strategy::DropOrphans(s) => s.gc_before(t, keep_past_s),
            Strategy::CausalRateSampled(s) => s.gc_before(t, keep_past_s),
            Strategy::BundleSummarized(s) => s.gc_before(t, keep_past_s),
        }
    }
    fn reset(&mut self) {
        match self {
            Strategy::Replay(s) => s.reset(),
            Strategy::RateSampled(s) => s.reset(),
            Strategy::DropOrphans(s) => s.reset(),
            Strategy::CausalRateSampled(s) => s.reset(),
            Strategy::BundleSummarized(s) => s.reset(),
        }
    }
    fn set_k(&mut self, new_k: f64) {
        match self {
            Strategy::Replay(s) => s.set_k(new_k),
            Strategy::RateSampled(s) => s.set_k(new_k),
            Strategy::DropOrphans(s) => s.set_k(new_k),
            Strategy::CausalRateSampled(s) => s.set_k(new_k),
            Strategy::BundleSummarized(s) => s.set_k(new_k),
        }
    }
    fn k(&self) -> f64 {
        match self {
            Strategy::Replay(s) => s.k,
            Strategy::RateSampled(s) => s.k(),
            Strategy::DropOrphans(s) => s.k(),
            Strategy::CausalRateSampled(s) => s.k(),
            Strategy::BundleSummarized(s) => s.k(),
        }
    }
}

/// Whether an event is eligible to become a visible packet under any
/// strategy. Filters control-plane variants (`pull`, `wake`),
/// self-loops, and zero-duration hops. Strategy-agnostic — moved out
/// of `VisualTimeline` so other strategies can reuse it.
pub fn is_visible_event(ev: &Event) -> bool {
    let Event::PacketEmitted { from, to, payload, at_ns, arrives_at_ns, .. } = ev else { return false; };
    if from == to { return false; }
    if arrives_at_ns <= at_ns { return false; }
    if let Value::Variant { tag, .. } = payload {
        if tag == "pull" || tag == "wake" { return false; }
    }
    true
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
    /// True when `now` falls inside this packet's animation window
    /// (`emit_real <= now < arrive_real`). Used by tests and the
    /// packet-cloud renderer to decide what's currently on-screen.
    pub fn is_visible_at(&self, now: f64) -> bool {
        self.emit_real <= now && now < self.arrive_real
    }

    /// Progress along the packet's edge in `[0, 1]` at `now`. Outside
    /// the animation window the value is clamped (0 if not yet emitted,
    /// 1 if already arrived). Same formula the GPU vertex shader uses.
    pub fn progress_at(&self, now: f64) -> f32 {
        let denom = (self.arrive_real - self.emit_real).max(1e-9);
        ((now - self.emit_real) / denom).clamp(0.0, 1.0) as f32
    }
}

/// Sorted-by-sim-time record of a visual arrival at some node.
/// Used to find a causal trigger when later packets emit from that
/// same node. Kept as parallel `Vec`s rather than `Vec<(u64,f64)>`
/// so we can binary-search the `ns` side directly.
#[derive(Clone, Debug, Default)]
struct NodeArrivals {
    /// Sim `arrives_at_ns` values, monotonically non-decreasing
    /// under sim-order ingestion.
    arrives_ns: Vec<u64>,
    /// Visual `arrive_real` for the same index.
    arrives_real: Vec<f64>,
}

impl NodeArrivals {
    fn push(&mut self, ns: u64, real: f64) {
        self.arrives_ns.push(ns);
        self.arrives_real.push(real);
    }
    /// Latest arrival with `arrives_at_ns ≤ at_ns`. Returns its
    /// `arrive_real`, or `None` if the node has no prior qualifying
    /// arrival.
    fn trigger_for(&self, at_ns: u64) -> Option<f64> {
        self.trigger_for_full(at_ns).map(|(_, real)| real)
    }
    /// Same as `trigger_for` but returns the matched
    /// `(arrives_at_ns, arrive_real)` pair. Strategies that want to
    /// compare visible vs. buffered trigger recency need both halves.
    fn trigger_for_full(&self, at_ns: u64) -> Option<(u64, f64)> {
        let idx = self.arrives_ns.partition_point(|ns| *ns <= at_ns);
        if idx == 0 { None } else {
            Some((self.arrives_ns[idx - 1], self.arrives_real[idx - 1]))
        }
    }
    fn len(&self) -> usize { self.arrives_ns.len() }
    /// Drop entries with `arrive_real < cutoff`. Relies on
    /// `arrives_real` NOT being strictly sorted (it's usually close
    /// to sorted but causal clamps can reorder), so we do a linear
    /// scan to find the split and retain from there. Cheap because
    /// we call GC rarely.
    fn gc(&mut self, cutoff: f64) {
        // Partition so earlier entries (< cutoff) are dropped.
        let keep_from = self.arrives_real.iter().position(|r| *r >= cutoff).unwrap_or(self.arrives_real.len());
        if keep_from > 0 {
            self.arrives_ns.drain(..keep_from);
            self.arrives_real.drain(..keep_from);
        }
    }

    /// Drop entries with `arrive_real > t`. Counterpart to `gc` —
    /// used when canceling future-clamped arrivals after a state
    /// change. We pair-iterate both vecs since arrival_real isn't
    /// strictly sorted.
    fn gc_after(&mut self, t: f64) {
        let mut i = 0;
        while i < self.arrives_real.len() {
            if self.arrives_real[i] > t {
                self.arrives_ns.swap_remove(i);
                self.arrives_real.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct VisualTimeline {
    pub packets: Vec<VisualPacket>,
    /// Visual scale: real-seconds per sim-second of packet latency.
    /// Controls the length of an on-screen packet's travel animation,
    /// independently of `SimClock.multiplier`.
    pub k: f64,
    /// Per-node record of visual arrivals, for causal trigger lookup.
    node_arrivals: HashMap<NodeId, NodeArrivals>,
}

impl Default for VisualTimeline {
    fn default() -> Self {
        // Manual impl — `#[derive(Default)]` would set `k = 0.0`,
        // which the ingest-time clamp then pins at `K_MIN = 0.1`,
        // producing nearly-invisible packets on app startup. We
        // want the user-chosen default here.
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
    /// Default so a 1 ms edge animates for 410 ms real — the value
    /// the user settled on after experimenting with `-` / `=`.
    pub const K_DEFAULT: f64 = 410.0;

    pub fn new(k: f64) -> Self {
        Self { k: k.clamp(Self::K_MIN, Self::K_MAX), ..Self::default() }
    }

    pub fn reset(&mut self) {
        self.packets.clear();
        self.node_arrivals.clear();
    }

    /// Set the visual scale. Already-baked packets are untouched —
    /// they finish their current windows at the old rate. New
    /// ingestions use `new_k`.
    pub fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(Self::K_MIN, Self::K_MAX);
    }

    /// Strategy-agnostic visibility filter. Delegates to the free
    /// `is_visible_event` so callers that have a `VisualTimeline`
    /// handy (or that pre-date the trait) keep compiling.
    pub fn is_visible_event(ev: &Event) -> bool {
        is_visible_event(ev)
    }

    /// Ingest a sim event at wall-clock time `real_now`. Returns the
    /// index of the new packet if one was created.
    ///
    /// Must be called in sim-time order for causal clamping to use
    /// fresh-enough data. The engine's event log is already ordered
    /// by `at_ns`, so the natural ingestion loop preserves this.
    pub fn ingest(&mut self, ev: &Event, real_now: f64) -> Option<usize> {
        let Event::PacketEmitted { packet, from, to, at_ns, arrives_at_ns, payload } = ev
        else { return None; };
        // Control-plane / zero-duration events contribute nothing — not
        // a visible packet, not a causal clamp.
        if arrives_at_ns <= at_ns { return None; }
        if let Value::Variant { tag, .. } = payload {
            if tag == "pull" || tag == "wake" { return None; }
        }

        let k = self.k.clamp(Self::K_MIN, Self::K_MAX);
        let clamp = self.node_arrivals.get(from)
            .and_then(|n| n.trigger_for(*at_ns))
            .unwrap_or(f64::NEG_INFINITY);
        let emit_real = real_now.max(clamp);
        let sim_latency_s = (arrives_at_ns - at_ns) as f64 * 1e-9;

        // Self-loops (service-time dwells, tick periods, wake pulses)
        // have no spatial representation — no edge to fly along — so
        // they don't become visible packets. They DO need to contribute
        // to `node_arrivals` though: a Worker's `done_req` self-hop is
        // the only place the service delay lives, and without it the
        // subsequent `resp` emit would clamp back to the req's arrival
        // time and appear instant. Unscaled (no `k`) so a 200ms
        // service produces a 200ms visible dwell instead of `200ms*k`
        // which would stretch the simulation into minutes.
        if from == to {
            let arrive_real = emit_real + sim_latency_s;
            self.node_arrivals.entry(*to).or_default().push(*arrives_at_ns, arrive_real);
            return None;
        }

        let arrive_real = emit_real + sim_latency_s * k;

        self.node_arrivals.entry(*to).or_default().push(*arrives_at_ns, arrive_real);

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

    pub fn visible_at<'a>(&'a self, t: f64) -> impl Iterator<Item = (&'a VisualPacket, f32)> + 'a {
        self.packets.iter().filter_map(move |p| {
            if t < p.emit_real || t >= p.arrive_real { return None; }
            let denom = p.arrive_real - p.emit_real;
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            Some((p, prog))
        })
    }

    /// Drop packets + arrival records older than `keep_past_s` past
    /// `t`. Trims both the visible-packets log and the per-node
    /// causal-trigger records; long sessions would otherwise grow
    /// `node_arrivals` unboundedly.
    pub fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        let cutoff = t - keep_past_s;
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.node_arrivals.values_mut() {
            n.gc(cutoff);
        }
        self.node_arrivals.retain(|_, n| n.len() > 0);
    }

    /// Drop packets that haven't started animating yet at wall-clock
    /// time `t` — the future-queued backlog. Currently-animating
    /// packets (`emit_real <= t < arrive_real`) and already-arrived
    /// packets (`arrive_real <= t`) are kept untouched. Returns the
    /// set of packet ids that were dropped, so callers can despawn
    /// the matching Bevy entities.
    ///
    /// Used by the host UI's "drop the queued backlog when a user
    /// state change fires" pass. We don't want to murder packets
    /// the user is currently watching — those are recent past, not
    /// stale future.
    pub fn drop_pending_after(&mut self, t: f64) -> HashSet<flow::PacketId> {
        let mut dropped: HashSet<flow::PacketId> = HashSet::new();
        self.packets.retain(|p| {
            if p.emit_real > t {
                dropped.insert(p.packet_id);
                false
            } else {
                true
            }
        });
        // Also trim future-arrival entries from node_arrivals so the
        // causal clamp doesn't keep waiting for them.
        for n in self.node_arrivals.values_mut() {
            n.gc_after(t);
        }
        self.node_arrivals.retain(|_, n| n.len() > 0);
        dropped
    }
}

// ────────────────────────────────────────────────────────────
// Rate-sampled strategy
// ────────────────────────────────────────────────────────────

/// Visual rate sampler. Each edge tracks a rolling window of recent
/// ingest timestamps; if the inferred rate is below the per-edge
/// budget, every event becomes a visual packet (Replay-equivalent).
/// Once the rate exceeds the budget, ingests are throttled to
/// `1 / MAX_VISUAL_RATE_PER_EDGE` second between visuals — the burst
/// is still legible to the eye but doesn't stack into a wall of
/// simultaneous starts at `real_now`.
///
/// Causal pairing across hops is **not** preserved: a request packet
/// that gets dropped under throttle leaves no trace, so the response
/// it triggered may animate without a visible cause. This is the
/// trade for legibility — Replay is the strategy to use when
/// "show every event" is non-negotiable.
#[derive(Clone, Debug)]
pub struct RateSampled {
    edges: HashMap<(NodeId, NodeId), EdgeStream>,
    packets: Vec<VisualPacket>,
    k: f64,
}

#[derive(Clone, Debug, Default)]
struct EdgeStream {
    /// Real-clock timestamps of recent ingests (within
    /// `RATE_WINDOW_SEC` of `real_now`). Length / window = rate.
    recent: VecDeque<f64>,
    /// Last `real_now` at which we emitted a visual on this edge.
    /// `f64::NEG_INFINITY` means "no prior emit, allow immediately".
    last_visual_emit_real: f64,
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
    /// Window over which incoming events are counted for rate
    /// estimation. One second is a comfortable trade-off — long
    /// enough to smooth jitter, short enough to react when a sim
    /// burst starts or ends.
    pub const RATE_WINDOW_SEC: f64 = 1.0;
    /// Per-edge cap on visuals per real second. Above this rate,
    /// ingests are dropped (still counted toward rate). Tuned so a
    /// human can visually track each packet — beyond ~10/s the eye
    /// can't separate them anyway, so the cap costs nothing.
    pub const MAX_VISUAL_RATE_PER_EDGE: f64 = 10.0;

    pub fn new(k: f64) -> Self {
        Self {
            k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX),
            ..Self::default()
        }
    }
}

impl VisualStrategy for RateSampled {
    fn ingest(&mut self, ev: &Event, real_now: f64) {
        if !is_visible_event(ev) { return; }
        let Event::PacketEmitted { packet, from, to, at_ns, arrives_at_ns, payload } = ev
        else { return; };

        let stream = self.edges.entry((*from, *to)).or_insert(EdgeStream {
            recent: VecDeque::new(),
            last_visual_emit_real: f64::NEG_INFINITY,
        });

        // Update rolling window.
        stream.recent.push_back(real_now);
        let cutoff = real_now - Self::RATE_WINDOW_SEC;
        while stream.recent.front().map_or(false, |&t| t < cutoff) {
            stream.recent.pop_front();
        }

        let rate = stream.recent.len() as f64 / Self::RATE_WINDOW_SEC;
        // Below the cap: emit 1:1 (visual_period = 0). Above:
        // throttle to `1 / cap` seconds between visuals.
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

    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a> {
        Box::new(self.packets.iter().filter_map(move |p| {
            if t < p.emit_real || t >= p.arrive_real { return None; }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            Some((p, prog))
        }))
    }

    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId> {
        let mut dropped = HashSet::new();
        self.packets.retain(|p| {
            if p.emit_real > t {
                dropped.insert(p.packet_id);
                false
            } else {
                true
            }
        });
        dropped
    }

    fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        let cutoff = t - keep_past_s;
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for stream in self.edges.values_mut() {
            while stream.recent.front().map_or(false, |&ts| ts < cutoff) {
                stream.recent.pop_front();
            }
        }
        self.edges.retain(|_, s| !s.recent.is_empty());
    }

    fn reset(&mut self) {
        self.edges.clear();
        self.packets.clear();
    }

    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }

    fn k(&self) -> f64 { self.k }
}

// ────────────────────────────────────────────────────────────
// Common helpers shared by causality-aware strategies
// ────────────────────────────────────────────────────────────

/// Carve a single ingested `PacketEmitted` into the fields the
/// strategies all need. Returns `None` for control-plane / zero-
/// duration events; returns `Some` with `is_self_loop=true` for
/// self-loop events (which don't render but DO carry causal info).
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
    let Event::PacketEmitted { packet, from, to, at_ns, arrives_at_ns, payload } = ev
    else { return None; };
    if arrives_at_ns <= at_ns { return None; }
    if let Value::Variant { tag, .. } = payload {
        if tag == "pull" || tag == "wake" { return None; }
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

// ────────────────────────────────────────────────────────────
// DropOrphans strategy (Option A)
// ────────────────────────────────────────────────────────────

/// Throttle like RateSampled, but **also drop responses whose
/// triggering request was throttled out** — the result is a sparser
/// canvas under bursts where every visible interaction is a complete
/// request/response chain. No "uncaused" events.
///
/// Mechanism: track two arrival logs per node: `visible` (arrivals
/// we actually rendered) and `any` (every sim arrival, including
/// dropped/self-loop/etc). On each ingest, look up the trigger in
/// both. If `any` has a trigger but `visible` doesn't, the chain is
/// broken — drop. If neither has one, this is a root emit (Client,
/// Generator) and is allowed.
#[derive(Clone, Debug)]
pub struct DropOrphans {
    edges: HashMap<(NodeId, NodeId), EdgeStream>,
    packets: Vec<VisualPacket>,
    /// Arrivals corresponding to packets we actually rendered.
    /// Same shape as Replay's `node_arrivals`.
    visible_arrivals: HashMap<NodeId, NodeArrivals>,
    /// Arrivals for *every* sim event we observed (including
    /// throttled, self-loop, dropped). Used to distinguish a root
    /// emit ("from has never received anything") from a broken
    /// chain ("from received something we didn't show").
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
        Self { k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX), ..Self::default() }
    }
}

impl VisualStrategy for DropOrphans {
    fn ingest(&mut self, ev: &Event, real_now: f64) {
        let Some(p) = parse_emit(ev) else { return };

        // Self-loop: not visible but contributes to `any_arrivals`
        // so downstream events from this node aren't classified as
        // orphans. (Mirrors Replay's self-loop handling.)
        if p.is_self_loop {
            self.any_arrivals.entry(p.to).or_default()
                .push(p.arrives_at_ns, real_now + p.sim_latency_s);
            return;
        }

        // Orphan check: any prior arrival at `from` but no visible
        // one means the trigger was throttled out. Drop.
        let trigger_visible = self.visible_arrivals.get(&p.from)
            .and_then(|n| n.trigger_for(p.at_ns));
        let trigger_any = self.any_arrivals.get(&p.from)
            .and_then(|n| n.trigger_for(p.at_ns));
        let is_orphan = trigger_any.is_some() && trigger_visible.is_none();
        if is_orphan {
            // Still record the arrival so downstream consequences
            // also get classified as orphans.
            self.any_arrivals.entry(p.to).or_default()
                .push(p.arrives_at_ns, real_now);
            return;
        }

        // Throttle (per-edge, same as RateSampled).
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
            // Throttled. Record any-arrival at `to` so consequences
            // know this hop happened sim-side.
            self.any_arrivals.entry(p.to).or_default()
                .push(p.arrives_at_ns, real_now);
            return;
        }
        stream.last_visual_emit_real = real_now;

        // Emit, with Replay's causal clamp.
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
        self.visible_arrivals.entry(p.to).or_default()
            .push(p.arrives_at_ns, arrive_real);
        self.any_arrivals.entry(p.to).or_default()
            .push(p.arrives_at_ns, arrive_real);
    }

    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a> {
        Box::new(self.packets.iter().filter_map(move |p| {
            if t < p.emit_real || t >= p.arrive_real { return None; }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            Some((p, prog))
        }))
    }

    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId> {
        let mut dropped = HashSet::new();
        self.packets.retain(|p| {
            if p.emit_real > t { dropped.insert(p.packet_id); false } else { true }
        });
        for n in self.visible_arrivals.values_mut() { n.gc_after(t); }
        for n in self.any_arrivals.values_mut() { n.gc_after(t); }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
        self.any_arrivals.retain(|_, n| n.len() > 0);
        dropped
    }

    fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        let cutoff = t - keep_past_s;
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.visible_arrivals.values_mut() { n.gc(cutoff); }
        for n in self.any_arrivals.values_mut() { n.gc(cutoff); }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
        self.any_arrivals.retain(|_, n| n.len() > 0);
        for stream in self.edges.values_mut() {
            while stream.recent.front().map_or(false, |&ts| ts < cutoff) {
                stream.recent.pop_front();
            }
        }
        self.edges.retain(|_, s| !s.recent.is_empty());
    }

    fn reset(&mut self) {
        self.edges.clear();
        self.packets.clear();
        self.visible_arrivals.clear();
        self.any_arrivals.clear();
    }

    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }

    fn k(&self) -> f64 { self.k }
}

// ────────────────────────────────────────────────────────────
// CausalRateSampled strategy (Option B — best of both)
// ────────────────────────────────────────────────────────────

/// Throttle like RateSampled, but when a downstream event arrives
/// whose triggering cause was throttled out, **promote the cause to
/// a visual just-in-time** so the chain stays intact. Visually, you
/// get short bursts where a request flies in and is immediately
/// chased by the response, instead of either stacking (Replay) or
/// causeless responses (RateSampled).
///
/// Mechanism: maintain Replay-style `node_arrivals` (visible only),
/// plus a small per-edge ring buffer of recent throttled events
/// retaining enough info to retroactively schedule them. On each
/// ingest, if the trigger lookup misses but a buffered candidate
/// exists, promote it before processing the consequence.
#[derive(Clone, Debug)]
pub struct CausalRateSampled {
    edges: HashMap<(NodeId, NodeId), EdgeStream>,
    packets: Vec<VisualPacket>,
    visible_arrivals: HashMap<NodeId, NodeArrivals>,
    /// Per-`to`-node ring of recently-throttled events that could
    /// later be promoted to satisfy a causal lookup.
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
    /// Real-clock ingest time. Used as the visual `emit_real` if
    /// this event gets promoted (we stretch it slightly so progress
    /// reads as "fly through quickly" rather than instant teleport).
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
        Self { k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX), ..Self::default() }
    }

    /// Find the most recent throttled event ending at `node` whose
    /// `arrives_at_ns ≤ at_ns_limit`. Returns the popped record on hit.
    fn pop_throttled_trigger(&mut self, node: NodeId, at_ns_limit: u64) -> Option<ThrottledEmit> {
        let buf = self.throttled.get_mut(&node)?;
        // Linear scan from newest to oldest — buffer is tiny.
        let pos = buf.iter().rposition(|t| t.arrives_at_ns <= at_ns_limit)?;
        buf.remove(pos)
    }

    /// Schedule a previously-throttled event as a visual now.
    /// The promoted packet animates over the same `sim_latency_s * k`
    /// window starting at the moment of promotion — so it reads as
    /// "the cause that we couldn't show at the time, shown now in
    /// quick succession with its consequence."
    fn promote(&mut self, ev: ThrottledEmit, real_now: f64) -> f64 {
        let k = self.k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
        // Tighten the promoted animation to a small fraction of the
        // normal duration so it reads as a "catch-up" rather than a
        // full-speed packet. Empirically nicer than just `* k`.
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
        self.visible_arrivals.entry(ev.to).or_default()
            .push(ev.arrives_at_ns, arrive_real);
        arrive_real
    }
}

impl VisualStrategy for CausalRateSampled {
    fn ingest(&mut self, ev: &Event, real_now: f64) {
        let Some(p) = parse_emit(ev) else { return };

        if p.is_self_loop {
            self.visible_arrivals.entry(p.to).or_default()
                .push(p.arrives_at_ns, real_now + p.sim_latency_s);
            return;
        }

        // Throttle decision first.
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
            // Stash for possible later promotion.
            let buf = self.throttled.entry(p.to).or_default();
            if buf.len() >= CAUSAL_THROTTLED_BUFFER_LEN { buf.pop_front(); }
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

        // Trigger lookup. We want the *most recent* cause to be the
        // visual trigger, regardless of whether it was visible or
        // throttled. If a buffered (throttled) candidate is fresher
        // than the most recent visible arrival, promote it — that's
        // the visual we wish had been there. Otherwise use the
        // visible trigger (or none, for root emits).
        let visible_trigger = self.visible_arrivals.get(&p.from)
            .and_then(|n| n.trigger_for_full(p.at_ns));
        let buffered_trigger_at_ns = self.throttled.get(&p.from)
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
        self.visible_arrivals.entry(p.to).or_default()
            .push(p.arrives_at_ns, arrive_real);
    }

    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a> {
        Box::new(self.packets.iter().filter_map(move |p| {
            if t < p.emit_real || t >= p.arrive_real { return None; }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            Some((p, prog))
        }))
    }

    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId> {
        let mut dropped = HashSet::new();
        self.packets.retain(|p| {
            if p.emit_real > t { dropped.insert(p.packet_id); false } else { true }
        });
        for n in self.visible_arrivals.values_mut() { n.gc_after(t); }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
        self.throttled.clear();
        dropped
    }

    fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        let cutoff = t - keep_past_s;
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.visible_arrivals.values_mut() { n.gc(cutoff); }
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

    fn reset(&mut self) {
        self.edges.clear();
        self.packets.clear();
        self.visible_arrivals.clear();
        self.throttled.clear();
    }

    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }

    fn k(&self) -> f64 { self.k }
}

// ────────────────────────────────────────────────────────────
// BundleSummarized strategy (Option C — coalesce bursts)
// ────────────────────────────────────────────────────────────

/// Coalesce bursts into single visuals: events on the same edge
/// arriving within `BUNDLE_WINDOW_SEC` (sim time) of the previous
/// emit fold into the current bundle and don't emit a new visual.
/// Causal pairing across hops is preserved via Replay-style
/// `node_arrivals` (recorded only for the bundle representatives).
///
/// This is what "summarization" would look like if we collapsed N
/// events into 1 visual without conveying N. A future iteration
/// could thicken the packet or add a count badge to communicate
/// bundle size — left as data-model work.
#[derive(Clone, Debug)]
pub struct BundleSummarized {
    /// Per-edge: sim-time of the most recent emitted bundle leader.
    /// New events within `BUNDLE_WINDOW_NS` of that fold into the
    /// existing bundle; outside the window starts a new bundle.
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
    /// Sim-time window over which contiguous events on the same
    /// edge are treated as a single bundle. 100 ms is a comfortable
    /// default — short enough that genuinely separate request bursts
    /// stay separate, long enough to coalesce a tight retry storm.
    pub const BUNDLE_WINDOW_NS: u64 = 100_000_000;

    pub fn new(k: f64) -> Self {
        Self { k: k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX), ..Self::default() }
    }
}

impl VisualStrategy for BundleSummarized {
    fn ingest(&mut self, ev: &Event, real_now: f64) {
        let Some(p) = parse_emit(ev) else { return };

        if p.is_self_loop {
            self.visible_arrivals.entry(p.to).or_default()
                .push(p.arrives_at_ns, real_now + p.sim_latency_s);
            return;
        }

        // Bundle check: is this within the recent burst on this edge?
        let key = (p.from, p.to);
        let new_bundle = match self.last_bundle_at_ns.get(&key) {
            Some(&last_at) => p.at_ns.saturating_sub(last_at) >= Self::BUNDLE_WINDOW_NS,
            None => true,
        };
        if !new_bundle { return; }
        self.last_bundle_at_ns.insert(key, p.at_ns);

        // Bundle leader → render. Replay-style causal clamp on the
        // visualized chain.
        let trigger = self.visible_arrivals.get(&p.from)
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
        self.visible_arrivals.entry(p.to).or_default()
            .push(p.arrives_at_ns, arrive_real);
    }

    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a> {
        Box::new(self.packets.iter().filter_map(move |p| {
            if t < p.emit_real || t >= p.arrive_real { return None; }
            let denom = (p.arrive_real - p.emit_real).max(1e-9);
            let prog = ((t - p.emit_real) / denom).clamp(0.0, 1.0) as f32;
            Some((p, prog))
        }))
    }

    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId> {
        let mut dropped = HashSet::new();
        self.packets.retain(|p| {
            if p.emit_real > t { dropped.insert(p.packet_id); false } else { true }
        });
        for n in self.visible_arrivals.values_mut() { n.gc_after(t); }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
        // Bundle ledger: clearing entirely is safest — any sim-time
        // cursor we kept is post-edit-stale anyway.
        self.last_bundle_at_ns.clear();
        dropped
    }

    fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        let cutoff = t - keep_past_s;
        self.packets.retain(|p| p.arrive_real >= cutoff);
        for n in self.visible_arrivals.values_mut() { n.gc(cutoff); }
        self.visible_arrivals.retain(|_, n| n.len() > 0);
    }

    fn reset(&mut self) {
        self.last_bundle_at_ns.clear();
        self.packets.clear();
        self.visible_arrivals.clear();
    }

    fn set_k(&mut self, new_k: f64) {
        self.k = new_k.clamp(VisualTimeline::K_MIN, VisualTimeline::K_MAX);
    }

    fn k(&self) -> f64 { self.k }
}

/// `VisualTimeline` is itself the Replay strategy.
impl VisualStrategy for VisualTimeline {
    fn ingest(&mut self, ev: &Event, real_now: f64) {
        let _ = VisualTimeline::ingest(self, ev, real_now);
    }
    fn visible_at<'a>(
        &'a self,
        t: f64,
    ) -> Box<dyn Iterator<Item = (&'a VisualPacket, f32)> + 'a> {
        Box::new(VisualTimeline::visible_at(self, t))
    }
    fn drop_pending_after(&mut self, t: f64) -> HashSet<PacketId> {
        VisualTimeline::drop_pending_after(self, t)
    }
    fn gc_before(&mut self, t: f64, keep_past_s: f64) {
        VisualTimeline::gc_before(self, t, keep_past_s)
    }
    fn reset(&mut self) {
        VisualTimeline::reset(self)
    }
    fn set_k(&mut self, new_k: f64) {
        VisualTimeline::set_k(self, new_k)
    }
    fn k(&self) -> f64 {
        self.k
    }
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
        // Gen has no prior arrivals; its outgoing packets emit at
        // exactly `real_now` (no causal clamp).
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 5.0);
        let p = &tl.packets[0];
        assert!((p.emit_real - 5.0).abs() < 1e-12);
        // duration = 1ms * 100 = 100ms
        assert!((p.arrive_real - 5.1).abs() < 1e-9);
    }

    #[test]
    fn response_waits_for_specific_triggering_arrival() {
        // Gen → Router → Sink. Router's response should emit at the
        // real time Router received Gen's packet, NOT at its own
        // ingestion real_now.
        let mut tl = VisualTimeline::new(100.0);
        // Gen packet ingested at real=0; gen→router, sim=0 → 1ms.
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        // Router emits response at sim=1ms (right after gen packet
        // arrives); ingested at real=0.001 (fast sim).
        tl.ingest(&emit(2, 20, 30, 1_000_000, 2_000_000, data_pkt(0)), 0.001);
        let gen_pkt = &tl.packets[0];
        let resp = &tl.packets[1];
        // Response's emit = gen's arrive, not 0.001.
        assert!((resp.emit_real - gen_pkt.arrive_real).abs() < 1e-9,
            "expected response emit {} = gen arrive {}, got {}",
            gen_pkt.arrive_real, gen_pkt.arrive_real, resp.emit_real);
    }

    #[test]
    fn router_does_not_wait_for_unrelated_later_arrival() {
        // If a second packet arrives at Router LATER than Router's
        // outgoing emit, the router's response shouldn't wait for
        // the unrelated arrival. Only arrivals with arrives_at_ns
        // ≤ emit.at_ns count.
        let mut tl = VisualTimeline::new(100.0);
        // Gen1 → Router, sim=0 → 1ms.
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        // Router emits response at sim=1ms (reacts to Gen1), well
        // BEFORE Gen2's packet arrives (sim=10ms).
        tl.ingest(&emit(2, 20, 30, 1_000_000, 2_000_000, data_pkt(0)), 0.001);
        // Gen2 → Router, sim=9ms → 10ms. Arrives LATER than the
        // response was emitted.
        tl.ingest(&emit(3, 11, 20, 9_000_000, 10_000_000, data_pkt(0)), 0.009);
        let resp = &tl.packets[1];
        let gen1 = &tl.packets[0];
        // Response emit should still be clamped only by Gen1's
        // arrive, not Gen2's (whose arrives_at_ns > resp.at_ns).
        assert!((resp.emit_real - gen1.arrive_real).abs() < 1e-9);
    }

    #[test]
    fn ingest_skips_pull_wake_selfloop_zero_duration() {
        let mut tl = VisualTimeline::new(100.0);
        assert!(tl.ingest(&emit(1, 10, 20, 0, 1_000_000, Value::variant("pull", Value::Nil)), 0.0).is_none());
        assert!(tl.ingest(&emit(2, 10, 20, 0, 1_000_000, Value::variant("wake", Value::Nil)), 0.0).is_none());
        assert!(tl.ingest(&emit(3, 10, 10, 0, 1_000_000, data_pkt(0)), 0.0).is_none());
        assert!(tl.ingest(&emit(4, 10, 20, 100, 100, data_pkt(0)), 0.0).is_none());
        assert!(tl.packets.is_empty());
    }

    #[test]
    fn visible_at_window() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        // emit=0, duration=0.1, so window [0, 0.1).
        assert_eq!(tl.visible_at(0.0).count(), 1);
        assert_eq!(tl.visible_at(0.05).count(), 1);
        assert_eq!(tl.visible_at(0.1).count(), 0);
        assert_eq!(tl.visible_at(-0.001).count(), 0);
    }

    #[test]
    fn parallel_emits_overlap_on_same_edge() {
        // 30 emits per real second into the same edge, fast sim.
        // Each packet lasts 100ms real. Expect heavy overlap.
        let mut tl = VisualTimeline::new(100.0);
        for i in 0..30 {
            let at = (i * 33_333_333) as u64; // 30/sec in ns
            let real = (i as f64) * (1.0 / 30.0);
            tl.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
        }
        // At real=0.5 (halfway through the second), many dots visible.
        let vis = tl.visible_at(0.5).count();
        assert!(vis >= 3,
            "expected multiple concurrent visuals on a 30/s stream at k=100, got {}", vis);
    }

    #[test]
    fn set_k_doesnt_teleport_existing_packets() {
        let mut tl = VisualTimeline::new(100.0);
        tl.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        let before = tl.packets[0].clone();
        tl.set_k(10.0);
        assert_eq!(tl.packets[0], before);
        tl.ingest(&emit(2, 11, 20, 0, 1_000_000, data_pkt(0)), 1.0);
        let p2 = &tl.packets[1];
        // New k=10 → 1ms * 10 = 10ms duration.
        assert!((p2.arrive_real - p2.emit_real - 0.01).abs() < 1e-9);
    }

    // ────────────────────────────────────────────────────────────
    // RateSampled
    // ────────────────────────────────────────────────────────────

    #[test]
    fn rate_sampled_below_cap_emits_1_to_1() {
        // 5 events/sec on one edge — well under MAX_VISUAL_RATE_PER_EDGE=10.
        // Every event should produce a visual.
        let mut rs = RateSampled::new(100.0);
        for i in 0..5 {
            let real = i as f64 * 0.2;
            let at = (i * 200_000_000) as u64;
            rs.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
        }
        assert_eq!(rs.packets.len(), 5,
            "low-rate edge should emit one visual per event");
    }

    #[test]
    fn rate_sampled_above_cap_throttles() {
        // 100 events/sec on one edge — way above the 10/s cap.
        // Window-based throttle should land us near the cap, NOT 100.
        let mut rs = RateSampled::new(100.0);
        for i in 0..100 {
            let real = i as f64 * 0.01; // 100/sec
            let at = (i * 10_000_000) as u64;
            rs.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
        }
        // The first ~window-worth emit freely (rate hasn't been
        // observed yet); after that, throttle kicks in at ~10/s.
        // Over 1 sec of input we expect well under 100 visuals — bound
        // it loosely to absorb the warm-up burst while still proving
        // throttling happens.
        assert!(rs.packets.len() < 30,
            "100/s sim should be throttled below 30 visuals/s, got {}",
            rs.packets.len());
        assert!(rs.packets.len() > 5,
            "throttled stream should still produce SOME visuals, got {}",
            rs.packets.len());
    }

    #[test]
    fn rate_sampled_per_edge_independent() {
        // Two edges flowing 100/s each. Each should be throttled
        // independently — total visuals ≤ 2 * cap, not capped to 10.
        let mut rs = RateSampled::new(100.0);
        for i in 0..50 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            rs.ingest(&emit(i as u64,        10, 20, at, at + 1_000_000, data_pkt(0)), real);
            rs.ingest(&emit(1000 + i as u64, 30, 40, at, at + 1_000_000, data_pkt(0)), real);
        }
        // Per-edge counts should each be > 0 and bounded by their own
        // throttle, not by a shared budget.
        let edge_a = rs.packets.iter().filter(|p| p.from == NodeId(10)).count();
        let edge_b = rs.packets.iter().filter(|p| p.from == NodeId(30)).count();
        assert!(edge_a > 0 && edge_b > 0,
            "both edges should produce visuals, got A={} B={}", edge_a, edge_b);
    }

    #[test]
    fn rate_sampled_skips_pull_wake_selfloop() {
        let mut rs = RateSampled::new(100.0);
        rs.ingest(&emit(1, 10, 20, 0, 1_000_000, Value::variant("pull", Value::Nil)), 0.0);
        rs.ingest(&emit(2, 10, 20, 0, 1_000_000, Value::variant("wake", Value::Nil)), 0.0);
        rs.ingest(&emit(3, 10, 10, 0, 1_000_000, data_pkt(0)), 0.0);
        assert!(rs.packets.is_empty());
    }

    #[test]
    fn strategy_cycle_round_trips() {
        // Cycle through every variant in `StrategyKind::ALL` and
        // back to the start. `k` is preserved across each step.
        let mut s = Strategy::new_of_kind(StrategyKind::ALL[0], 50.0);
        for expected in StrategyKind::ALL.iter().skip(1).chain(std::iter::once(&StrategyKind::ALL[0])) {
            s.cycle();
            assert_eq!(&s.kind(), expected,
                "cycle should advance to {:?}", expected);
            assert!((s.k() - 50.0).abs() < 1e-9, "k must carry over");
        }
    }

    // ────────────────────────────────────────────────────────────
    // DropOrphans
    // ────────────────────────────────────────────────────────────

    #[test]
    fn drop_orphans_root_emit_visible() {
        // Client (NodeId 10) has never received anything → its emit
        // is a root, must be visible. Mirrors `gen_emit_uses_real_now`.
        let mut s = DropOrphans::new(100.0);
        s.ingest(&emit(1, 10, 20, 0, 1_000_000, data_pkt(0)), 0.0);
        assert_eq!(s.packets.len(), 1);
    }

    #[test]
    fn drop_orphans_response_dropped_when_request_throttled() {
        // 100/s requests on edge Client→Worker → throttle drops most.
        // The response edges Worker→Client should also drop because
        // their causes (Worker received nothing visible) were thrown
        // out by throttle.
        let mut s = DropOrphans::new(100.0);
        for i in 0..50 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            // Client → Worker
            s.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
            // Worker → Client (response)
            s.ingest(&emit(1000 + i as u64, 20, 10,
                at + 1_500_000, at + 2_500_000, data_pkt(0)), real + 0.0015);
        }
        // Many requests get visualized (within the per-edge cap),
        // but every visible response should pair with a visible
        // request that arrived before it.
        let req_count = s.packets.iter().filter(|p| p.from == NodeId(10)).count();
        let resp_count = s.packets.iter().filter(|p| p.from == NodeId(20)).count();
        // Concrete property: responses ≤ requests (no orphans).
        assert!(resp_count <= req_count,
            "DropOrphans must not produce more responses than requests; \
             got reqs={} resps={}", req_count, resp_count);
    }

    // ────────────────────────────────────────────────────────────
    // CausalRateSampled
    // ────────────────────────────────────────────────────────────

    #[test]
    fn causal_rate_sampled_promotes_throttled_cause() {
        // Send 11 requests in a tight 11ms real-time burst. The first
        // 10 hit the per-edge cap (10/s) under the 1s rate window;
        // the 11th trips the throttle and goes into the buffer. Then
        // fire a response whose at_ns falls between the 10th request's
        // arrival and the 11th's — the most recent cause is the
        // throttled 11th, so the strategy should promote it.
        let mut s = CausalRateSampled::new(100.0);
        for i in 0..11 {
            let real = i as f64 * 0.001;
            let at = (i * 1_000_000) as u64;
            s.ingest(&emit(i as u64, 10, 20, at, at + 100_000, data_pkt(0)), real);
        }
        let req_count_before_resp = s.packets.iter().filter(|p| p.from == NodeId(10)).count();
        assert_eq!(req_count_before_resp, 10, "warm-up should emit 10 requests");
        // Response at sim 10.5ms — newer than i=10's arrives_at_ns
        // (10.1ms) but older than any later request would be.
        s.ingest(&emit(9999, 20, 10, 10_500_000, 10_600_000, data_pkt(0)), 0.012);
        let req_count_after = s.packets.iter().filter(|p| p.from == NodeId(10)).count();
        let resp_count = s.packets.iter().filter(|p| p.from == NodeId(20)).count();
        assert_eq!(resp_count, 1, "response should emit");
        assert_eq!(req_count_after, req_count_before_resp + 1,
            "promotion should add exactly one request visual (was {}, became {})",
            req_count_before_resp, req_count_after);
    }

    #[test]
    fn causal_rate_sampled_preserves_low_rate_chains() {
        // Below the throttle cap, behavior should match Replay's
        // request/response pairing.
        let mut s = CausalRateSampled::new(100.0);
        for i in 0..3 {
            let real = i as f64 * 0.2;
            let at = (i * 200_000_000) as u64;
            s.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
            s.ingest(&emit(1000 + i as u64, 20, 10,
                at + 2_000_000, at + 3_000_000, data_pkt(0)), real + 0.002);
        }
        assert_eq!(s.packets.len(), 6,
            "low-rate chain should produce 3 reqs + 3 resps under CausalRateSampled");
    }

    // ────────────────────────────────────────────────────────────
    // BundleSummarized
    // ────────────────────────────────────────────────────────────

    #[test]
    fn bundle_summarized_collapses_close_emits() {
        // 10 events on the same edge, 10ms apart sim-time → all
        // within one BUNDLE_WINDOW_NS (100ms). Should produce a
        // single visual.
        let mut s = BundleSummarized::new(100.0);
        for i in 0..10 {
            let real = i as f64 * 0.01;
            let at = (i * 10_000_000) as u64;
            s.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
        }
        assert_eq!(s.packets.len(), 1,
            "10 emits within one bundle window should collapse to 1 visual");
    }

    #[test]
    fn bundle_summarized_separates_distant_emits() {
        // 3 events 200ms apart sim-time → each starts a new bundle.
        let mut s = BundleSummarized::new(100.0);
        for i in 0..3 {
            let real = i as f64 * 0.2;
            let at = (i * 200_000_000) as u64;
            s.ingest(&emit(i as u64, 10, 20, at, at + 1_000_000, data_pkt(0)), real);
        }
        assert_eq!(s.packets.len(), 3,
            "emits separated by > BUNDLE_WINDOW_NS should each emit a visual");
    }

    #[test]
    fn strategy_dispatches_through_trait() {
        // Same event sequence into both variants; both should produce
        // visible packets at low rates. Replay's `as_replay()` and
        // RateSampled's match arm should each light up.
        let mut s_replay = Strategy::new_of_kind(StrategyKind::Replay, 100.0);
        let mut s_rate = Strategy::new_of_kind(StrategyKind::RateSampled, 100.0);
        for i in 0..3 {
            let ev = emit(i as u64, 10, 20, (i * 200_000_000) as u64,
                          (i * 200_000_000 + 1_000_000) as u64, data_pkt(0));
            s_replay.ingest(&ev, i as f64 * 0.2);
            s_rate.ingest(&ev, i as f64 * 0.2);
        }
        assert_eq!(s_replay.visible_at(0.05).count() + s_replay.visible_at(0.25).count() +
                   s_replay.visible_at(0.45).count(), 3);
        assert_eq!(s_rate.visible_at(0.05).count() + s_rate.visible_at(0.25).count() +
                   s_rate.visible_at(0.45).count(), 3);
    }
}
