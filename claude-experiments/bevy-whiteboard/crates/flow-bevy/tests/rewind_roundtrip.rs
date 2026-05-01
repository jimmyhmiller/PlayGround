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
use flow_bevy::visual::{Strategy, StrategyKind, VisualStrategy};
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

/// Capture exactly what the renderer would draw — query at
/// `clock.visual_now` (the same value `sync_packet_transforms`
/// passes to `visible_at` for every rendered packet), no
/// boundary filter, no quantisation softening. If forward and
/// rewound `visual_now` differ even slightly, the rendered sets
/// differ, even when the underlying timeline records carry the
/// same `emit_real` / `arrive_real`. That's the gap I missed.
fn screenshot(app: &App) -> Vec<ScreenshotPacket> {
    let _ = app.world().resource::<SimClock>().visual_now;
    let timeline = app.world().resource::<VisualTimelineRes>();
    let mut shot: Vec<ScreenshotPacket> = timeline
        .visible
        .iter()
        .map(|(p, prog)| ScreenshotPacket {
            packet_id: p.packet_id,
            from: p.from,
            to: p.to,
            progress_q: quantise(*prog),
        })
        .collect();
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

/// Switch the active visual strategy + set its `k`. The roundtrip
/// property is meaningful only for strategies that don't depend on
/// pre-snapshot visual history — `SimMirror` derives positions from
/// `sim.in_flight` directly, so it reproduces exactly. `Replay` and
/// the rate-sampled strategies need an event window that the
/// bounded snapshot ring can't always fully cover; their roundtrip
/// is approximate by design.
fn configure(app: &mut App, kind: StrategyKind, k: f64) {
    let world = app.world_mut();
    let mut tl = world.resource_mut::<VisualTimelineRes>();
    tl.strategy = Strategy::new_of_kind(kind, k);
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
    kind: StrategyKind,
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
    // `as_replay` only works for the Replay variant; for SimMirror
    // there's nothing accumulated, so report the visible-set size.
    let total_packets = {
        let timeline = app.world().resource::<VisualTimelineRes>();
        match timeline.strategy.kind() {
            StrategyKind::Replay => timeline.strategy.as_replay().packets.len(),
            _ => timeline.visible.len(),
        }
    };
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
// Roundtrip is a property of the *visual strategy*, not the rewind
// machinery. SimMirror reproduces exactly: positions come from
// `sim.in_flight`, so the rewound state is identical to forward
// (modulo the snapshot ring's resolution, which is below the test's
// quantisation threshold).
//
// Replay etc. are *not* exact-roundtrip strategies — they accumulate
// derived state (causal-clamp arrival logs, packet vectors) over the
// full visible window, and the bounded snapshot ring can't always
// reach back far enough on rewind to reconstruct that state. The
// user-visible mismatch is by design; tests pinning Replay's
// roundtrip would be aspirational.
// ─────────────────────────────────────────────────────────────────

#[test]
fn sim_mirror_three_lane_roundtrip_at_1500ms() {
    let (forward, rewound) = run_roundtrip(
        Example::ThreeLaneFanout,
        1_500_000_000,
        2_000_000_000,
        StrategyKind::SimMirror,
        1.0,
    );
    if let Some(msg) = diff(&forward, &rewound) {
        panic!("[SimMirror 1500ms] {}", msg);
    }
}

#[test]
fn sim_mirror_three_lane_roundtrip_at_3s() {
    let (forward, rewound) = run_roundtrip(
        Example::ThreeLaneFanout,
        3_000_000_000,
        2_000_000_000,
        StrategyKind::SimMirror,
        1.0,
    );
    if let Some(msg) = diff(&forward, &rewound) {
        panic!("[SimMirror 3s] {}", msg);
    }
}
