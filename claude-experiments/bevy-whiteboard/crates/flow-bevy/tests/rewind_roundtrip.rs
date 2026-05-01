//! Round-trip equivalence: forward-play to t=x, capture the
//! "screenshot" (visible packets + their on-edge progress),
//! continue forward, then rewind to t=x and capture again. The
//! two snapshots must be identical.
//!
//! This is the strongest contract we can express for a rewind
//! strategy: "rewinding to a sim moment must reproduce exactly
//! what the user saw at that sim moment." Anything weaker and
//! AnchorReplay's drift slips through.
//!
//! "Identical" is checked by comparing the (packet_id, from, to,
//! progress) tuple of every visible packet. Progress is a float
//! in [0, 1] indicating where along its edge the packet is
//! rendered — the visible position of every dot in the screenshot
//! is determined by exactly this set, with the from/to positions
//! coming from the static node layout.

mod common;

use bevy::prelude::*;
use bevy::ecs::message::Messages;

use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::{FlowSim, SimClock};
use flow_bevy::edges::VisualTimelineRes;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::rewind::RewindStrategyKind;
use flow_bevy::visual::VisualStrategy;
use flow::{NodeId, PacketId};

/// One visible packet in a "screenshot" — everything that
/// determines a rendered position on screen, modulo the static
/// node layout.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ScreenshotPacket {
    packet_id: PacketId,
    from: NodeId,
    to: NodeId,
    /// Quantised progress in [0, 1] — quantised so floating-point
    /// noise in the synth-time math doesn't flag false negatives.
    /// 0.001 ≈ 1ms of motion at typical packet speed; well below
    /// the user's perceptual threshold.
    progress_q: u32,
}

fn quantise(progress: f32) -> u32 {
    (progress.clamp(0.0, 1.0) * 1000.0).round() as u32
}

/// Capture the on-screen state from the visual layer.
fn screenshot(app: &App) -> Vec<ScreenshotPacket> {
    let visual_now = app.world().resource::<SimClock>().visual_now;
    let timeline = app.world().resource::<VisualTimelineRes>();
    let mut shot: Vec<ScreenshotPacket> = timeline
        .visible_at(visual_now)
        .map(|(p, prog)| ScreenshotPacket {
            packet_id: p.packet_id,
            from: p.from,
            to: p.to,
            progress_q: quantise(prog),
        })
        .collect();
    // Stable ordering for diffing.
    shot.sort_by_key(|p| (p.from.0, p.to.0, p.packet_id.0));
    shot
}

fn fire_load_example(app: &mut App, example: Example) {
    app.world_mut()
        .resource_mut::<Messages<LoadExample>>()
        .write(LoadExample(example));
    app.update();
    app.update();
}

/// Set the strategy + visual scale on a fresh app.
fn configure(app: &mut App, kind: RewindStrategyKind, k: f64) {
    let world = app.world_mut();
    world.resource_mut::<VisualTimelineRes>().0.set_k(k);
    world.resource_mut::<FlowSim>().0.set_rewind_strategy(kind);
}

/// Diff two screenshots into a human-readable failure message.
fn diff(forward: &[ScreenshotPacket], rewound: &[ScreenshotPacket]) -> Option<String> {
    if forward == rewound {
        return None;
    }
    use std::collections::BTreeSet;
    let f: BTreeSet<_> = forward.iter().collect();
    let r: BTreeSet<_> = rewound.iter().collect();
    let only_forward: Vec<_> = f.difference(&r).copied().cloned().collect();
    let only_rewound: Vec<_> = r.difference(&f).copied().cloned().collect();
    Some(format!(
        "round-trip mismatch:\n  \
         forward count: {}\n  \
         rewound count: {}\n  \
         only in forward (lost on rewind): {} entries\n    {:#?}\n  \
         only in rewound (drift introduced): {} entries\n    {:#?}",
        forward.len(), rewound.len(),
        only_forward.len(), &only_forward.iter().take(8).collect::<Vec<_>>(),
        only_rewound.len(), &only_rewound.iter().take(8).collect::<Vec<_>>(),
    ))
}

/// Drive forward play frame-by-frame: advance sim a small chunk
/// then run an app.update so events ingest into the visual layer
/// at the same real_now they were emitted at, mirroring live-app
/// behavior (rather than batching everything into one update,
/// which collapses every event's `emit_real` and makes visuals
/// unrepresentative of the live render).
fn forward_play(app: &mut App, total_ns: u64) {
    let frame_ns = 16_000_000_u64; // ~60fps
    let frames = (total_ns / frame_ns).max(1) as usize;
    for _ in 0..frames {
        advance_sim_ns(app, frame_ns);
        app.update();
    }
}

/// Core scenario.
fn run_roundtrip(
    example: Example,
    forward_to_ns: u64,
    extra_advance_ns: u64,
    kind: RewindStrategyKind,
    k: f64,
) -> (Vec<ScreenshotPacket>, Vec<ScreenshotPacket>) {
    let mut app = make_app();
    configure(&mut app, kind, k);
    fire_load_example(&mut app, example);

    // Forward play in small frame-sized chunks until sim_now is
    // at-or-just-past `forward_to_ns`. The real `sim_now` at
    // capture time is what we'll rewind back to (Bevy's per-frame
    // advance can overshoot the requested deadline by one tick).
    while app.world().resource::<FlowSim>().now_ns < forward_to_ns {
        forward_play(&mut app, 16_000_000);
    }
    let forward_shot = screenshot(&app);
    let forward_visual_now = app.world().resource::<SimClock>().visual_now;
    let forward_sim_now = app.world().resource::<FlowSim>().now_ns;
    let total_packets = app.world().resource::<VisualTimelineRes>()
        .0.as_replay().packets.len();
    eprintln!(
        "forward: sim_now={}ms visual_now={:.3} packets_visible={} packets_in_timeline={}",
        forward_sim_now / 1_000_000,
        forward_visual_now,
        forward_shot.len(),
        total_packets,
    );

    // Continue advancing past x.
    forward_play(&mut app, extra_advance_ns);

    // Rewind to the exact sim_now we captured the forward shot at.
    {
        let world = app.world_mut();
        world.resource_mut::<SimClock>().paused = true;
        world.resource_mut::<FlowSim>().0.rewind(forward_sim_now);
    }
    app.update();
    let rewound_shot = screenshot(&app);
    let rewound_visual_now = app.world().resource::<SimClock>().visual_now;
    let rewound_sim_now = app.world().resource::<FlowSim>().now_ns;
    eprintln!(
        "rewound: sim_now={}ms visual_now={:.3} packets_visible={}",
        rewound_sim_now / 1_000_000,
        rewound_visual_now,
        rewound_shot.len(),
    );

    (forward_shot, rewound_shot)
}

// ─────────────────────────────────────────────────────────────────
// FullLog: should round-trip cleanly. If THIS one fails, the
// problem is in the visual layer's reset+ingest path, not
// strategy-specific.
// ─────────────────────────────────────────────────────────────────

#[test]
fn fulllog_three_lane_roundtrip_at_1500ms() {
    let (forward, rewound) = run_roundtrip(
        Example::ThreeLaneFanout,
        1_500_000_000,
        2_000_000_000,
        RewindStrategyKind::FullLog,
        200.0,
    );
    if let Some(msg) = diff(&forward, &rewound) {
        panic!("[FullLog] {}", msg);
    }
}

// ─────────────────────────────────────────────────────────────────
// AnchorReplay: the test the user actually wants. If this passes,
// AnchorReplay reproduces the exact on-screen state at the rewind
// target.
// ─────────────────────────────────────────────────────────────────

#[test]
fn anchor_three_lane_roundtrip_at_1500ms_k200() {
    let (forward, rewound) = run_roundtrip(
        Example::ThreeLaneFanout,
        1_500_000_000,
        2_000_000_000,
        RewindStrategyKind::AnchorReplay,
        200.0,
    );
    if let Some(msg) = diff(&forward, &rewound) {
        panic!("[AnchorReplay] {}", msg);
    }
}

#[test]
fn anchor_three_lane_roundtrip_at_3s_k200() {
    let (forward, rewound) = run_roundtrip(
        Example::ThreeLaneFanout,
        3_000_000_000,
        2_000_000_000,
        RewindStrategyKind::AnchorReplay,
        200.0,
    );
    if let Some(msg) = diff(&forward, &rewound) {
        panic!("[AnchorReplay 3s] {}", msg);
    }
}

#[test]
fn anchor_three_lane_roundtrip_at_3s_k400() {
    let (forward, rewound) = run_roundtrip(
        Example::ThreeLaneFanout,
        3_000_000_000,
        2_000_000_000,
        RewindStrategyKind::AnchorReplay,
        400.0,
    );
    if let Some(msg) = diff(&forward, &rewound) {
        panic!("[AnchorReplay 3s k=400] {}", msg);
    }
}
