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
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct VisualPacket {
    pub packet_id: PacketId,
    pub from: NodeId,
    pub to: NodeId,
    pub payload: Value,
    pub emit_real: f64,
    pub arrive_real: f64,
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
        // `partition_point` gives us the first index where predicate
        // is false; everything before is ≤ at_ns.
        let idx = self.arrives_ns.partition_point(|ns| *ns <= at_ns);
        if idx == 0 { None } else { Some(self.arrives_real[idx - 1]) }
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

    /// Should this event become a visible packet? Skips control-plane
    /// variants (`pull`, `wake`), self-loops, and zero-duration hops.
    pub fn is_visible_event(ev: &Event) -> bool {
        let Event::PacketEmitted { from, to, payload, at_ns, arrives_at_ns, .. } = ev else { return false; };
        if from == to { return false; }
        if arrives_at_ns <= at_ns { return false; }
        if let Value::Variant { tag, .. } = payload {
            if tag == "pull" || tag == "wake" { return false; }
        }
        true
    }

    /// Ingest a sim event at wall-clock time `real_now`. Returns the
    /// index of the new packet if one was created.
    ///
    /// Must be called in sim-time order for causal clamping to use
    /// fresh-enough data. The engine's event log is already ordered
    /// by `at_ns`, so the natural ingestion loop preserves this.
    pub fn ingest(&mut self, ev: &Event, real_now: f64) -> Option<usize> {
        if !Self::is_visible_event(ev) { return None; }
        let Event::PacketEmitted { packet, from, to, at_ns, arrives_at_ns, payload } = ev
        else { return None; };

        let k = self.k.clamp(Self::K_MIN, Self::K_MAX);
        let clamp = self.node_arrivals.get(from)
            .and_then(|n| n.trigger_for(*at_ns))
            .unwrap_or(f64::NEG_INFINITY);
        let emit_real = real_now.max(clamp);
        let sim_latency_s = (arrives_at_ns - at_ns) as f64 * 1e-9;
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
}
