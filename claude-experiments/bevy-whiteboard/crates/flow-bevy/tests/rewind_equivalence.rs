//! Cross-strategy equivalence: FullLog vs AnchorReplay should
//! produce visually-identical rewinds. AnchorReplay is supposed to
//! be a bounded-cost approximation of FullLog — same on-screen
//! state, just cheaper to compute. Any divergence is a bug.
//!
//! These tests build the same sim+state twice, rewind once with
//! each strategy, then compare the resulting `VisualTimeline`'s
//! packet records. Mismatched emit/arrive times or missing entries
//! mean the lookback window or the apply-rewind-reset filter is
//! out of sync.

mod common;

use bevy::prelude::*;
use bevy::ecs::message::Messages;

use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::FlowSim;
use flow_bevy::edges::VisualTimelineRes;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::rewind::RewindStrategyKind;
use flow_bevy::visual::{VisualPacket, VisualStrategy};
use flow::PacketId;

fn fire_load_example(app: &mut App, example: Example) {
    app.world_mut()
        .resource_mut::<Messages<LoadExample>>()
        .write(LoadExample(example));
    app.update();
    app.update();
}

/// What we capture after a rewind so we can compare strategies
/// fairly: the visual packet records *and* the `visual_now` the
/// app actually shows at that moment.
#[derive(Debug)]
struct RewoundState {
    packets: Vec<VisualPacket>,
    visual_now: f64,
}

/// Run the same sim setup, advance, then rewind to `target_ns`
/// using `kind`.
fn rewind_and_collect(
    example: Example,
    advance_chunks_ns: &[u64],
    target_ns: u64,
    kind: RewindStrategyKind,
    k: f64,
) -> RewoundState {
    let mut app = make_app();
    {
        let world = app.world_mut();
        world.resource_mut::<VisualTimelineRes>().0.set_k(k);
        world.resource_mut::<FlowSim>().0.set_rewind_strategy(kind);
    }
    fire_load_example(&mut app, example);
    for &chunk in advance_chunks_ns {
        advance_sim_ns(&mut app, chunk);
    }
    app.update();

    {
        let world = app.world_mut();
        world.resource_mut::<flow_bevy::bridge::SimClock>().paused = true;
        world.resource_mut::<FlowSim>().0.rewind(target_ns);
    }
    app.update();

    let timeline = app.world().resource::<VisualTimelineRes>();
    let visual_now = app.world().resource::<flow_bevy::bridge::SimClock>().visual_now;
    RewoundState {
        packets: timeline.0.as_replay().packets.clone(),
        visual_now,
    }
}

/// Collapse a packet list to a comparable shape: the visible-now
/// subset (those whose emit_real ≤ visual_now < arrive_real)
/// keyed by `packet_id`. Strategies are allowed to disagree about
/// far-past gc-trail records — what the user sees on screen is
/// the visible set.
fn visible_keys(
    packets: &[VisualPacket],
    visual_now: f64,
) -> std::collections::BTreeSet<PacketId> {
    packets
        .iter()
        .filter(|p| p.emit_real <= visual_now && visual_now < p.arrive_real)
        .map(|p| p.packet_id)
        .collect()
}

/// Run a 3-second ThreeLaneFanout and rewind both ways. Visible
/// packet sets must match. Any difference means the strategies
/// disagree on what's on screen for the same sim moment.
fn assert_strategies_agree(
    chunks: &[u64],
    target: u64,
    k: f64,
) {
    let full = rewind_and_collect(
        Example::ThreeLaneFanout, chunks, target,
        RewindStrategyKind::FullLog, k,
    );
    let anchor = rewind_and_collect(
        Example::ThreeLaneFanout, chunks, target,
        RewindStrategyKind::AnchorReplay, k,
    );

    eprintln!(
        "k={} target={}ms: full-log packets={} visual_now={:.3} | anchor packets={} visual_now={:.3}",
        k, target / 1_000_000,
        full.packets.len(), full.visual_now,
        anchor.packets.len(), anchor.visual_now,
    );

    // Sample at visual_now values across the post-rewind window.
    // The user's actual visual_now after a rewind is the rebased
    // wall clock, which depends on how long they ran the sim
    // before clicking. Test must pass at *every* sample so the
    // strategies agree regardless of what visual_now happens to
    // be when the user looks.
    let probe_visual_nows: Vec<f64> = (0..20)
        .map(|i| anchor.visual_now - 0.1 + (i as f64 * 0.01))
        .collect();
    for &probe in &probe_visual_nows {
        let visible_full = visible_keys(&full.packets, probe);
        let visible_anchor = visible_keys(&anchor.packets, probe);
        let only_full: Vec<_> = visible_full.difference(&visible_anchor).copied().collect();
        let only_anchor: Vec<_> = visible_anchor.difference(&visible_full).copied().collect();

        if !only_full.is_empty() || !only_anchor.is_empty() {
            panic!(
                "visible-packet sets differ at visual_now={:.3} (k={}, target={}ms)\n\
                 missing from anchor (full-log has them):  {} packets\n\
                 only in anchor (full-log doesn't have):   {} packets\n\
                 visible@probe: full-log={} anchor={}\n",
                probe, k, target / 1_000_000,
                only_full.len(), only_anchor.len(),
                visible_full.len(), visible_anchor.len(),
            );
        }
    }
}

#[test]
fn anchor_matches_full_log_three_lane_k_200() {
    let chunks: Vec<u64> = std::iter::repeat(250_000_000_u64).take(12).collect();
    assert_strategies_agree(&chunks, 1_500_000_000, 200.0);
}

#[test]
fn anchor_matches_full_log_three_lane_k_400() {
    let chunks: Vec<u64> = std::iter::repeat(250_000_000_u64).take(12).collect();
    assert_strategies_agree(&chunks, 2_000_000_000, 400.0);
}

#[test]
fn anchor_matches_full_log_three_lane_k_800() {
    // High `k` stresses the visibility term in the lookback —
    // packets stay on screen for much longer real time, so the
    // window of relevant past events grows.
    let chunks: Vec<u64> = std::iter::repeat(250_000_000_u64).take(12).collect();
    assert_strategies_agree(&chunks, 1_500_000_000, 800.0);
}
