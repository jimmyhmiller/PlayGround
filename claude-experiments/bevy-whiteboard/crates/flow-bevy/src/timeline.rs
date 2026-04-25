//! Display-only timeline strip across the bottom of the canvas.
//!
//! Renders `sim.timeline` (the global, sim-level scenario timeline)
//! as a horizontal bar with:
//!   - integer-second tick marks across the visible range
//!   - a `now` cursor advancing left-to-right with sim time, paused
//!     when `SimClock.paused`
//!   - one marker per scheduled event, positioned at its `at_ns`,
//!     labeled with the affected `node.slot := value`
//!
//! Time relationship to the canvas: at the global clock level
//! `sim_now / 1e9 == visual_now` (when `multiplier == 1`), so the
//! strip's cursor is exactly aligned with what slot probes show.
//! Per-packet animation duration is a separate factor (`k`); a packet
//! on a 1 ms edge takes `k * 1ms` of wall time to cross. So in-flight
//! packets on the canvas show traffic from up to `k * edge_latency`
//! ago in sim time. Slot states and timeline events are point-in-
//! time correct; packet animations are bounded-history.

use bevy::prelude::*;
use flow::{TimelineEvent, Value};
use poster_ui::{Bold, Mono, Theme};

use crate::bridge::FlowSim;

pub struct TimelinePlugin;
impl Plugin for TimelinePlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Startup, spawn_strip)
            .add_systems(Update, (
                rebuild_strip_contents,
                sync_now_cursor,
            ));
    }
}

/// Marker on the strip's outer container. Public for tests.
#[derive(Component)]
pub struct TimelineStripRoot;

/// Marker on the inner positioning track that holds tick marks,
/// markers and the now-cursor as absolutely-positioned children.
#[derive(Component)]
struct TimelineTrack;

/// Marker on the now-cursor (a vertical accent line).
#[derive(Component)]
struct TimelineNowCursor;

/// Marker on each event marker entity. Carries the event id so
/// rebuilds can despawn the matching set.
#[derive(Component, Debug, Clone)]
struct TimelineEventMarker {
    event_id: u64,
}

const STRIP_HEIGHT: f32 = 64.0;
const TRACK_PAD_X: f32 = 16.0;
const MARKER_SIZE: f32 = 12.0;
const NOW_WIDTH: f32 = 2.0;
const RANGE_BUFFER_NS: u64 = 5_000_000_000;
const MIN_RANGE_NS: u64 = 5_000_000_000;
/// Pixel inset between the track's right edge and the rightmost
/// usable percent for marker placement, so a marker at the END of
/// the visible range doesn't overflow off-screen.
const _MARKER_RIGHT_GUARD_PCT: f32 = 0.5;

fn spawn_strip(mut commands: Commands, theme: Res<Theme>) {
    let root = commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            right: Val::Px(284.0),    // clear the right palette
            bottom: Val::Px(72.0),    // above the bottom HUD
            height: Val::Px(STRIP_HEIGHT),
            padding: UiRect::all(Val::Px(8.0)),
            flex_direction: FlexDirection::Column,
            ..default()
        },
        BackgroundColor(theme.paper_alt),
        BorderColor::all(theme.rule),
        Visibility::Hidden,
        TimelineStripRoot,
    )).id();

    commands.entity(root).with_children(|outer| {
        // Header — minimal for v1. The rest of the strip carries the
        // information.
        outer.spawn((
            Text::new("Timeline"),
            TextFont { font_size: 9.0, ..default() },
            TextColor(theme.ink_soft),
            Bold,
        ));

        // Track: positioning context for ticks, markers, cursor.
        // All children use `position_type: Absolute` so absolute
        // pixel/percent left placement works.
        outer
            .spawn((
                Node {
                    position_type: PositionType::Relative,
                    width: Val::Percent(100.0),
                    height: Val::Px(36.0),
                    margin: UiRect::top(Val::Px(4.0)),
                    padding: UiRect::horizontal(Val::Px(TRACK_PAD_X)),
                    border: UiRect::top(Val::Px(1.0)),
                    ..default()
                },
                BorderColor::all(theme.rule),
                TimelineTrack,
            ))
            .with_children(|track| {
                // Spawn the now-cursor INSIDE the track from the
                // start so its `Val::Percent` left position is
                // relative to the track's content box. Re-parenting
                // afterwards via `ChildOf` was the bug that broke
                // cursor rendering.
                track.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        top: Val::Px(0.0),
                        bottom: Val::Px(0.0),
                        width: Val::Px(NOW_WIDTH),
                        left: Val::Px(0.0),
                        ..default()
                    },
                    BackgroundColor(theme.accent),
                    Visibility::Hidden,
                    TimelineNowCursor,
                ));
            });
    });
}

/// Compute the strip's visible time range:
///   start = 0
///   end   = max(sim.now_ns, last_event_at_ns, MIN_RANGE_NS) + buffer
fn visible_range(sim: &FlowSim) -> (u64, u64) {
    let last = sim.sim.timeline.last_at_ns().unwrap_or(0);
    let high_water = sim.sim.now_ns.max(last).max(MIN_RANGE_NS);
    (0, high_water + RANGE_BUFFER_NS)
}

fn time_to_fraction(at_ns: u64, start: u64, end: u64) -> f32 {
    if end <= start { return 0.0; }
    let dt = at_ns.saturating_sub(start) as f64;
    let span = (end - start) as f64;
    (dt / span).clamp(0.0, 1.0) as f32
}

/// Refresh ticks and markers on every frame the timeline contents
/// or visible range change. Cheaper alternative to rebuilding every
/// frame: hash a signature.
#[derive(Default)]
struct StripSignature {
    sig: (Vec<(u64, (u64, usize, bool))>, (u64, u64)),
}

fn rebuild_strip_contents(
    sim: Res<FlowSim>,
    theme: Res<Theme>,
    track_q: Query<Entity, With<TimelineTrack>>,
    root_q: Query<Entity, With<TimelineStripRoot>>,
    existing_markers: Query<(Entity, &TimelineEventMarker)>,
    existing_ticks: Query<Entity, With<TickLabel>>,
    mut commands: Commands,
    mut last_sig: Local<StripSignature>,
) {
    // Toggle whole-strip visibility based on whether anything's
    // scheduled. Plain whiteboards stay clean.
    let want_visible = !sim.sim.timeline.events.is_empty();
    if let Ok(root) = root_q.single() {
        commands.entity(root).insert(if want_visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        });
    }
    if !want_visible { return; }

    let range = visible_range(&sim);
    let cur_event_sig: Vec<(u64, (u64, usize, bool))> = sim.sim.timeline.events
        .iter()
        .map(|e| (e.id, (e.at_ns, e.actions.len(), e.fired)))
        .collect();
    let cur_sig = (cur_event_sig.clone(), range);
    if cur_sig == last_sig.sig && !theme.is_changed() {
        return;
    }
    last_sig.sig = cur_sig;

    // Despawn old markers + ticks.
    for (e, _) in existing_markers.iter() {
        commands.entity(e).despawn();
    }
    for e in existing_ticks.iter() {
        commands.entity(e).despawn();
    }

    let Ok(track) = track_q.single() else { return };
    let (start, end) = range;

    commands.entity(track).with_children(|t| {
        spawn_tick_marks(t, &theme, start, end);
        for ev in &sim.sim.timeline.events {
            spawn_event_marker(t, &theme, &sim, ev, start, end);
        }
    });
}

/// Marker on tick-mark entities so we can despawn them on rebuild.
#[derive(Component)]
struct TickLabel;

fn spawn_tick_marks(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    start: u64,
    end: u64,
) {
    // Pick a tick interval: 1s if range ≤ 15s, 2s if ≤ 30s, 5s otherwise.
    let span_s = ((end - start) as f64 / 1e9).ceil() as u64;
    let step_s: u64 = if span_s <= 15 { 1 }
                       else if span_s <= 30 { 2 }
                       else if span_s <= 90 { 5 }
                       else { 10 };

    let start_s = start / 1_000_000_000;
    let end_s = end / 1_000_000_000;
    let mut s = start_s;
    if s == 0 && step_s > 0 { s = step_s; }  // skip the 0 tick (label clutter)
    while s <= end_s {
        let at_ns = s * 1_000_000_000;
        let frac = time_to_fraction(at_ns, start, end);
        parent.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(frac * 100.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Px(1.0),
                ..default()
            },
            BackgroundColor(theme.rule),
            TickLabel,
        ));
        parent.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(frac * 100.0),
                bottom: Val::Px(0.0),
                margin: UiRect::left(Val::Px(2.0)),
                ..default()
            },
            TickLabel,
            children![(
                Text::new(format!("{}s", s)),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                Mono,
            )],
        ));
        s += step_s;
    }
}

fn spawn_event_marker(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    sim: &FlowSim,
    ev: &TimelineEvent,
    range_start: u64,
    range_end: u64,
) {
    let frac = time_to_fraction(ev.at_ns, range_start, range_end);
    let dot_color = if ev.fired { theme.muted } else { theme.accent };
    parent.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(frac * 100.0),
            top: Val::Px(10.0),
            width: Val::Px(MARKER_SIZE),
            height: Val::Px(MARKER_SIZE),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border: UiRect::all(Val::Px(1.5)),
            border_radius: BorderRadius::all(Val::Px(MARKER_SIZE / 2.0)),
            ..default()
        },
        BackgroundColor(dot_color),
        BorderColor::all(theme.ink),
        TimelineEventMarker { event_id: ev.id },
        children![(
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(-12.0),
                left: Val::Px(MARKER_SIZE + 4.0),
                ..default()
            },
            children![(
                Text::new(format_event_label(sim, ev)),
                TextFont { font_size: 10.0, ..default() },
                TextColor(theme.ink),
                Mono,
            )],
        )],
    ));
}

/// "2.0s Worker.up := 0" for single-action events; for compound
/// events, summarize as "2.0s × 3" (count) and let a future tooltip
/// expand them.
fn format_event_label(sim: &FlowSim, ev: &TimelineEvent) -> String {
    let sec = ev.at_ns as f64 / 1e9;
    if ev.actions.len() == 1 {
        let a = &ev.actions[0];
        let node_name = sim.sim.nodes.get(&a.node)
            .map(|n| n.name.as_str())
            .unwrap_or("?");
        format!("{:.1}s  {}.{} := {}", sec, node_name, a.slot, format_value(&a.value))
    } else {
        format!("{:.1}s  ×{}", sec, ev.actions.len())
    }
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Int(i) => format!("{}", i),
        Value::Float(f) => format!("{:.3}", f),
        Value::Bool(b) => format!("{}", b),
        Value::Str(s) => s.clone(),
        Value::Nil => "nil".into(),
        _ => "?".into(),
    }
}

/// Slide the now-cursor to its current sim-time position. Runs every
/// frame so the cursor advances smoothly.
fn sync_now_cursor(
    sim: Res<FlowSim>,
    mut q: Query<(&mut Node, &mut Visibility), With<TimelineNowCursor>>,
) {
    let Ok((mut node, mut vis)) = q.single_mut() else { return };
    if sim.sim.timeline.events.is_empty() {
        *vis = Visibility::Hidden;
        return;
    }
    let (start, end) = visible_range(&sim);
    let frac = time_to_fraction(sim.sim.now_ns, start, end);
    node.left = Val::Percent(frac * 100.0);
    *vis = Visibility::Inherited;
}
