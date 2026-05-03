//! Bottom-left HUD: play/pause, step, sim-time readout, speed chips, counters.
//! Built from poster-ui cell primitives; this module owns the domain wiring
//! (which flow-bevy state each cell reads / writes).

use bevy::prelude::*;
use poster_ui::{
    HudButtonFill, HudButtonStyle, Mono, Slider, Theme,
    hud_bottom_left, hud_button_cell, hud_chip_strip, hud_counter, hud_counter_strip,
    hud_speed_chip, hud_step_cell, spawn_hud_bar, spawn_slider_with_step,
};

use crate::bridge::SimClock;
use crate::edges::VisualTimelineRes;
use crate::sim_driver::{SimCommand, SimDriverRes, SimSnapshotRes};

pub struct HudPlugin;
impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingRewindTarget>()
            .add_systems(Startup, (spawn_hud, spawn_vis_k_popup, spawn_rewind_popup))
            .add_systems(
                Update,
                (
                    update_time_readout,
                    update_vis_readout,
                    update_counts,
                    handle_play_button,
                    handle_step_button,
                    handle_rewind_zero_button,
                    handle_rewind_step_button,
                    handle_reverse_play_button,
                    handle_visual_strategy_click,
                    update_visual_strategy_readout,
                    clear_pending_rewind_when_caught_up,
                    auto_stop_reverse_play_at_zero,
                ),
            )
            .add_systems(
                Update,
                (
                    handle_rewind_time_click,
                    push_rewind_slider,
                    sync_rewind_slider_from_snapshot,
                    sync_rewind_slider_max,
                    handle_speed_chips,
                    handle_edge_scale_chips,
                    handle_vis_k_click,
                    push_vis_k_slider,
                    sync_vis_k_slider_from_resource,
                    sync_play_button_visual,
                    sync_speed_chip_visuals,
                    sync_edge_scale_chip_visuals,
                ),
            );
    }
}

// ────────────────────────────────────────────────────────────
// Domain markers
// ────────────────────────────────────────────────────────────

#[derive(Component)]
pub struct HudPlayBtn;

#[derive(Component)]
pub struct HudStepBtn;

/// Rewind to t=0 — uses the sticky anchor in the snapshot ring.
#[derive(Component)]
pub struct HudRewindZeroBtn;

/// Rewind one capture marker back from current sim time.
#[derive(Component)]
pub struct HudRewindStepBtn;

/// Marker on the cell wrapping the visual-strategy text. Clicking
/// the cell rotates through `StrategyKind::ALL`.
#[derive(Component)]
struct HudVisualStrategyCell;

/// Marker on the inner `Text` of the visual-strategy cell so the
/// readout system can find and update it.
#[derive(Component)]
struct HudVisualStrategyText;

/// Most recent rewind target the HUD asked for. Worker-mode rewinds
/// are async — the snapshot's `now_ns` lags. When the user clicks «
/// rapidly, every click would otherwise read the same stale `now_ns`
/// and compute the same target. This resource lets each click base
/// its "previous marker" lookup on the *previously requested* target,
/// so successive clicks step further back. Cleared once the published
/// snapshot's `now_ns` reaches the target.
#[derive(Resource, Default)]
pub struct PendingRewindTarget(pub Option<u64>);

#[derive(Component)]
struct HudTimeText;

/// Visual scale readout — shows `VisualTimeline.k` so the user
/// knows what `-` / `=` are set to. Displayed as "vis k=NNN" so
/// the units are clear (real-ms per sim-ms of packet latency =
/// scale factor applied to packet animation duration).
#[derive(Component)]
struct HudVisScale;

#[derive(Component)]
struct HudNodeCount;

#[derive(Component)]
struct HudEdgeCount;

#[derive(Component)]
struct HudErrorCount;

#[derive(Component, Clone, Copy)]
struct HudSpeedChip(f64);

const SPEED_OPTIONS: [(f64, &str); 5] = [
    (0.25, "¼x"),
    (0.5,  "½x"),
    (1.0,  "1x"),
    (2.0,  "2x"),
    (4.0,  "4x"),
];

#[derive(Component, Clone, Copy)]
struct HudEdgeScaleChip(f64);

/// Edge-latency scale chips. `1×` is neutral; `2× / 4× / 10×` make
/// every edge latency that much longer (and rescale in-flight packets
/// so the change feels uniform). Independent of the sim-speed chips:
/// you can run the sim at `4×` speed but with edge transit `4×`
/// slower to see "fast logic, slow lines".
const EDGE_SCALE_OPTIONS: [(f64, &str); 4] = [
    (1.0,  "e1x"),
    (2.0,  "e2x"),
    (4.0,  "e4x"),
    (10.0, "e10x"),
];

// ────────────────────────────────────────────────────────────
// Spawn
// ────────────────────────────────────────────────────────────

fn spawn_hud(mut commands: Commands, theme: Res<Theme>, clock: Res<SimClock>) {
    let hud = spawn_hud_bar(&mut commands, &theme, hud_bottom_left());
    let play_style = if clock.paused { HudButtonStyle::Ink } else { HudButtonStyle::Accent };
    let play_glyph = if clock.paused { "▶" } else { "❚❚" };

    commands.entity(hud).with_children(|bar| {
        hud_button_cell(bar, &theme, play_style, play_glyph, HudPlayBtn);
        hud_step_cell(bar, &theme, "›|", HudStepBtn);
        // Rewind controls. `«` rolls back one snapshot marker;
        // `↺0` jumps to the sticky anchor at t=0; `◀◀` toggles
        // continuous reverse playback at real-time speed.
        hud_step_cell(bar, &theme, "«", HudRewindStepBtn);
        hud_step_cell(bar, &theme, "↺0", HudRewindZeroBtn);
        hud_step_cell(bar, &theme, "◀◀", HudReversePlayBtn);
        // Time readout: clickable cell that toggles the rewind
        // scrubber popup. Same shape as the vis-k cell — a plain
        // `Button` so `Interaction` tracking just works.
        bar.spawn((
            Button,
            Node {
                min_width: Val::Px(96.0),
                padding: UiRect::horizontal(Val::Px(12.0)),
                border: UiRect::right(Val::Px(1.5)),
                align_items: AlignItems::Center,
                ..default()
            },
            BorderColor::all(theme.ink),
        ))
        .with_children(|cell| {
            cell.spawn((
                Text::new("0.000 ms"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(theme.ink),
                Mono,
                HudTimeText,
            ));
        });
        // The vis-k cell is clickable: tapping it toggles a small
        // popup slider so the user can dial `k` directly without
        // memorising the `-` / `=` keybindings. Bare `Button` makes
        // it Interaction-tracked the same way the speed chips are.
        bar.spawn((
            Button,
            Node {
                min_width: Val::Px(78.0),
                padding: UiRect::horizontal(Val::Px(12.0)),
                border: UiRect::right(Val::Px(1.5)),
                align_items: AlignItems::Center,
                ..default()
            },
            BorderColor::all(theme.ink),
        ))
        .with_children(|cell| {
            cell.spawn((
                Text::new("vis k=…"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(theme.ink),
                Mono,
                HudVisScale,
            ));
        });

        // Rewind-strategy cycler: click rotates through the
        // available strategies (anchor-replay default, then the
        // legacy three for A/B testing).
        bar.spawn((
            Button,
            Node {
                min_width: Val::Px(120.0),
                padding: UiRect::horizontal(Val::Px(12.0)),
                border: UiRect::right(Val::Px(1.5)),
                align_items: AlignItems::Center,
                ..default()
            },
            BorderColor::all(theme.ink),
            HudVisualStrategyCell,
        ))
        .with_children(|cell| {
            cell.spawn((
                Text::new("vis: …"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(theme.ink),
                Mono,
                HudVisualStrategyText,
            ));
        });

        hud_chip_strip(bar, &theme, |chips| {
            for (i, (val, label)) in SPEED_OPTIONS.iter().enumerate() {
                let active = !clock.paused && (clock.multiplier - val).abs() < 1e-3;
                let is_last = i + 1 == SPEED_OPTIONS.len();
                hud_speed_chip(chips, &theme, label, active, is_last, HudSpeedChip(*val));
            }
        });

        hud_chip_strip(bar, &theme, |chips| {
            for (i, (val, label)) in EDGE_SCALE_OPTIONS.iter().enumerate() {
                let active = (clock.edge_latency_scale - val).abs() < 1e-3;
                let is_last = i + 1 == EDGE_SCALE_OPTIONS.len();
                hud_speed_chip(chips, &theme, label, active, is_last, HudEdgeScaleChip(*val));
            }
        });

        hud_counter_strip(bar, |cs| {
            hud_counter(cs, &theme, "0 nodes", HudNodeCount);
            hud_counter(cs, &theme, "0 edges", HudEdgeCount);
            hud_counter(cs, &theme, "0 errors", HudErrorCount);
        });
    });
}

// ────────────────────────────────────────────────────────────
// Per-frame updates
// ────────────────────────────────────────────────────────────

fn update_time_readout(
    snapshot: Res<SimSnapshotRes>,
    mut q: Query<&mut Text, With<HudTimeText>>,
) {
    let Ok(mut text) = q.single_mut() else { return };
    let t_ms = snapshot.0.now_ns as f64 / 1_000_000.0;
    let new = format!("{:.3} ms", t_ms);
    if text.0 != new { text.0 = new; }
}

/// Show the current visual scale so `-` / `=` adjustments are
/// legible. Rounded to an integer — `k` always gets halved/doubled
/// from the 100 default so fractions only show up after many presses.
fn update_vis_readout(
    timeline: Res<VisualTimelineRes>,
    mut q: Query<&mut Text, With<HudVisScale>>,
) {
    let Ok(mut text) = q.single_mut() else { return };
    let k = timeline.k();
    let strategy = timeline.strategy.kind().label();
    let new = if k >= 10.0 {
        format!("vis k={} [{}]", k.round() as i64, strategy)
    } else {
        format!("vis k={:.2} [{}]", k, strategy)
    };
    if text.0 != new { text.0 = new; }
}

fn update_counts(
    snapshot: Res<SimSnapshotRes>,
    mut nodes: Query<&mut Text, (With<HudNodeCount>, Without<HudEdgeCount>, Without<HudErrorCount>)>,
    mut edges: Query<&mut Text, (With<HudEdgeCount>, Without<HudNodeCount>, Without<HudErrorCount>)>,
    mut errors: Query<&mut Text, (With<HudErrorCount>, Without<HudNodeCount>, Without<HudEdgeCount>)>,
) {
    let n = format!("{} nodes", snapshot.0.nodes.len());
    let e = format!("{} edges", snapshot.0.edges.len());
    // Total error count — sum across all kinds. Lets the user notice
    // a new error happened at a glance; clicking through to the
    // event log (future) would show which kinds and where.
    let err_total: u64 = snapshot.0.error_counts.values().sum();
    let err = format!("{} errors", err_total);
    for mut t in nodes.iter_mut() { if t.0 != n { t.0 = n.clone(); } }
    for mut t in edges.iter_mut() { if t.0 != e { t.0 = e.clone(); } }
    for mut t in errors.iter_mut() { if t.0 != err { t.0 = err.clone(); } }
}

fn handle_play_button(
    q: Query<&Interaction, (Changed<Interaction>, With<HudPlayBtn>)>,
    mut clock: ResMut<SimClock>,
) {
    for i in q.iter() {
        if *i == Interaction::Pressed {
            // Toggling play implicitly cancels reverse playback.
            // Without this the user would have to first toggle
            // reverse off, then unpause, to get forward play back.
            clock.reverse_play_rate = 0.0;
            clock.paused = !clock.paused;
        }
    }
}

fn handle_step_button(
    q: Query<&Interaction, (Changed<Interaction>, With<HudStepBtn>)>,
    mut driver: ResMut<SimDriverRes>,
) {
    for i in q.iter() {
        if *i == Interaction::Pressed {
            driver.0.send_command(SimCommand::new(|sim| crate::palette::step_to_visible(sim)));
        }
    }
}

/// Drive a rewind to `target_ns`. The visual layer recomputes its
/// on-screen state from the sim's event log on the next frame —
/// see `edges::apply_rewind_reset` — so this is just a one-shot
/// driver call.
fn apply_rewind(
    target_ns: u64,
    driver: &mut SimDriverRes,
) {
    driver.0.rewind(target_ns);
}

fn handle_rewind_zero_button(
    q: Query<&Interaction, (Changed<Interaction>, With<HudRewindZeroBtn>)>,
    mut driver: ResMut<SimDriverRes>,
    mut clock: ResMut<SimClock>,
    mut pending: ResMut<PendingRewindTarget>,
) {
    for i in q.iter() {
        if *i == Interaction::Pressed {
            apply_rewind(0, &mut driver);
            clock.paused = true;
            pending.0 = Some(0);
        }
    }
}

/// Debug-mode reverse play: continuous rewind at real-time speed.
/// Toggle on, packets visually flow backward along edges; toggle
/// off to resume normal forward play. Sim is paused while reverse
/// is active (forward-advance is suppressed inside
/// `push_clock_to_driver`), so the only thing changing is sim time
/// shifting backward via per-frame `driver.rewind(...)` calls.
#[derive(Component)]
struct HudReversePlayBtn;

fn handle_reverse_play_button(
    q: Query<&Interaction, (Changed<Interaction>, With<HudReversePlayBtn>)>,
    mut clock: ResMut<SimClock>,
) {
    for i in q.iter() {
        if *i == Interaction::Pressed {
            if clock.reverse_play_rate > 0.0 {
                clock.reverse_play_rate = 0.0;
            } else {
                clock.reverse_play_rate = 1.0;
                // Reverse-play implies the user is no longer
                // forward-playing; surface the pause flag so the
                // play button visual reflects "not advancing
                // forward" until they toggle reverse off.
                clock.paused = true;
            }
        }
    }
}

/// Toggle continuous animated reverse-playback. First click flips
/// reverse-play on (sim time walks backward at real-time speed via
/// `bridge::push_clock_to_driver`). Second click stops it. Auto-stops
/// when sim_now reaches 0 — see `auto_stop_reverse_play_at_zero`.
fn handle_rewind_step_button(
    q: Query<&Interaction, (Changed<Interaction>, With<HudRewindStepBtn>)>,
    mut clock: ResMut<SimClock>,
) {
    for i in q.iter() {
        if *i == Interaction::Pressed {
            if clock.reverse_play_rate > 0.0 {
                clock.reverse_play_rate = 0.0;
            } else {
                clock.reverse_play_rate = 1.0;
                // Reverse-play paths in `bridge` only fire while
                // forward advance is suppressed. Setting paused
                // here matches `handle_reverse_play_button`.
                clock.paused = true;
            }
        }
    }
}

/// While reverse-playback is active, stop it as soon as sim time
/// hits 0. Without this, `bridge::push_clock_to_driver` keeps
/// dispatching `driver.rewind(0)` every frame — the worker
/// processes them but burns cycles for no visible effect.
fn auto_stop_reverse_play_at_zero(
    snapshot: Res<SimSnapshotRes>,
    mut clock: ResMut<SimClock>,
) {
    if clock.reverse_play_rate > 0.0 && snapshot.0.now_ns == 0 {
        clock.reverse_play_rate = 0.0;
    }
}

/// Clear `PendingRewindTarget` once the worker has caught up — the
/// next click should anchor off the live snapshot, not the stale
/// pending value. Also clear if the user unpaused (they're moving
/// forward; the pending rewind chain is no longer relevant).
fn clear_pending_rewind_when_caught_up(
    snapshot: Res<SimSnapshotRes>,
    clock: Res<SimClock>,
    mut pending: ResMut<PendingRewindTarget>,
) {
    let Some(t) = pending.0 else { return };
    if !clock.paused || snapshot.0.now_ns <= t {
        pending.0 = None;
    }
}

fn handle_speed_chips(
    q: Query<(&Interaction, &HudSpeedChip), Changed<Interaction>>,
    mut clock: ResMut<SimClock>,
) {
    for (interaction, chip) in q.iter() {
        if *interaction == Interaction::Pressed {
            clock.multiplier = chip.0;
            clock.paused = false;
        }
    }
}

fn handle_edge_scale_chips(
    q: Query<(&Interaction, &HudEdgeScaleChip), Changed<Interaction>>,
    mut clock: ResMut<SimClock>,
) {
    for (interaction, chip) in q.iter() {
        if *interaction == Interaction::Pressed {
            clock.edge_latency_scale = chip.0;
        }
    }
}

fn sync_edge_scale_chip_visuals(
    theme: Res<Theme>,
    clock: Res<SimClock>,
    mut chips: Query<(&HudEdgeScaleChip, &Children, &mut BackgroundColor)>,
    mut text_q: Query<&mut TextColor, Without<Mono>>,
) {
    for (chip, children, mut bg) in chips.iter_mut() {
        let active = (clock.edge_latency_scale - chip.0).abs() < 1e-3;
        bg.0 = if active { theme.ink } else { Color::NONE };
        for c in children.iter() {
            if let Ok(mut tc) = text_q.get_mut(c) {
                tc.0 = if active { theme.paper } else { theme.ink_soft };
            }
        }
    }
}

fn sync_play_button_visual(
    theme: Res<Theme>,
    clock: Res<SimClock>,
    mut btn_q: Query<
        (&mut BackgroundColor, &mut HudButtonFill, &Children),
        With<HudPlayBtn>,
    >,
    mut text_q: Query<&mut Text>,
) {
    if !clock.is_changed() && !theme.is_changed() { return; }
    for (mut bg, mut fill, children) in btn_q.iter_mut() {
        let (style, glyph) = if clock.paused {
            (HudButtonStyle::Ink, "▶")
        } else {
            (HudButtonStyle::Accent, "❚❚")
        };
        fill.0 = style;
        bg.0 = match style {
            HudButtonStyle::Accent => theme.accent,
            HudButtonStyle::Ink => theme.ink,
            HudButtonStyle::Muted => theme.paper_alt,
        };
        for c in children.iter() {
            if let Ok(mut text) = text_q.get_mut(c) {
                if text.0 != glyph { text.0 = glyph.into(); }
            }
        }
    }
}

fn sync_speed_chip_visuals(
    theme: Res<Theme>,
    clock: Res<SimClock>,
    mut chips: Query<(&HudSpeedChip, &Children, &mut BackgroundColor)>,
    mut text_q: Query<&mut TextColor, Without<Mono>>,
) {
    for (chip, children, mut bg) in chips.iter_mut() {
        let active = !clock.paused && (clock.multiplier - chip.0).abs() < 1e-3;
        bg.0 = if active { theme.ink } else { Color::NONE };
        for c in children.iter() {
            if let Ok(mut tc) = text_q.get_mut(c) {
                tc.0 = if active { theme.paper } else { theme.ink_soft };
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// Vis-k popup slider
// ────────────────────────────────────────────────────────────

/// Marker on the popup container so we can toggle its visibility.
#[derive(Component)]
struct VisKPopup;

/// Marker on the slider entity inside the popup. The push system
/// converts the slider's value back into `VisualTimelineRes.set_k`;
/// the sync system writes the resource's value back when something
/// else (the `-` / `=` keys, a snapshot load) changes `k`.
#[derive(Component)]
struct VisKSlider;

const VIS_K_MIN: f32 = 1.0;
const VIS_K_MAX: f32 = 2000.0;

/// Spawn the (initially hidden) popup just above where the bottom-left
/// HUD sits. Positioning is absolute relative to the screen so we
/// don't need to thread a layout-anchor through the HUD bar; eyeball
/// values that line up with `hud_bottom_left()` are good enough today
/// (and easy to revisit when the HUD shape changes).
fn spawn_vis_k_popup(mut commands: Commands, theme: Res<Theme>, timeline: Res<VisualTimelineRes>) {
    let initial = (timeline.k() as f32).clamp(VIS_K_MIN, VIS_K_MAX);
    commands
        .spawn((
            VisKPopup,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(258.0),
                bottom: Val::Px(56.0),
                width: Val::Px(220.0),
                padding: UiRect::all(Val::Px(10.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.ink),
            Visibility::Hidden,
        ))
        .with_children(|panel| {
            spawn_slider_with_step(
                panel,
                &theme,
                "vis k",
                VIS_K_MIN,
                VIS_K_MAX,
                /*step=*/ 1.0,
                initial,
                "",
                VisKSlider,
            );
        });
}

/// Toggle popup visibility on click of the vis-k cell. Pinned to the
/// `Pressed` transition so a held click doesn't keep flipping.
fn handle_vis_k_click(
    cells: Query<&Interaction, (Changed<Interaction>, With<Button>)>,
    cell_text_q: Query<&ChildOf, With<HudVisScale>>,
    mut popup: Query<&mut Visibility, With<VisKPopup>>,
) {
    // Find the cell entity that owns the HudVisScale text. (We could
    // attach the marker to the cell itself, but doing it through the
    // text child keeps the spawn shape symmetric with the other HUD
    // text cells.)
    let Some(cell_entity) = cell_text_q.iter().next().map(|p| p.parent()) else { return };
    let Ok(interaction) = cells.get(cell_entity) else { return };
    if *interaction != Interaction::Pressed { return; }
    let Ok(mut vis) = popup.single_mut() else { return };
    *vis = match *vis {
        Visibility::Hidden => Visibility::Visible,
        _ => Visibility::Hidden,
    };
}

/// Slider drag → write into `VisualTimelineRes.k`. Uses `set_k` so
/// the clamp to `[K_MIN, K_MAX]` happens once and stays consistent
/// with the keybind path.
fn push_vis_k_slider(
    sliders: Query<&Slider, (Changed<Slider>, With<VisKSlider>)>,
    mut timeline: ResMut<VisualTimelineRes>,
) {
    for slider in sliders.iter() {
        let new_k = slider.value as f64;
        if (timeline.k() - new_k).abs() < 1e-3 { continue; }
        timeline.set_k(new_k);
    }
}

// ────────────────────────────────────────────────────────────
// Rewind scrubber popup
// ────────────────────────────────────────────────────────────

#[derive(Component)]
struct RewindPopup;

#[derive(Component)]
pub struct RewindSlider;

/// Bottom-aligned popup with a horizontal slider whose range is
/// `[0, max(observed_now_ns, last_marker_ns)]`. The slider value is
/// in nanoseconds — direct `rewind(target_ns)` mapping. Hidden until
/// the time-readout cell is clicked.
fn spawn_rewind_popup(mut commands: Commands, theme: Res<Theme>) {
    commands
        .spawn((
            RewindPopup,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(180.0),
                bottom: Val::Px(56.0),
                width: Val::Px(360.0),
                padding: UiRect::all(Val::Px(10.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.ink),
            Visibility::Hidden,
        ))
        .with_children(|panel| {
            // Range is in *milliseconds* on the slider widget so
            // values fit comfortably in `f32`. The push system
            // converts back to nanoseconds (×1e6). Without this scale
            // the slider's `f32` would lose precision past ~16ms of
            // sim time.
            spawn_slider_with_step(
                panel,
                &theme,
                "rewind",
                /*min=*/ 0.0,
                /*max=*/ 1.0,
                /*step=*/ 0.0,
                /*initial=*/ 0.0,
                "ms",
                RewindSlider,
            );
        });
}

/// Toggle the rewind popup when the time-readout cell is clicked.
/// Same idiom as `handle_vis_k_click` — locate the parent button via
/// the `HudTimeText` child marker.
fn handle_rewind_time_click(
    cells: Query<&Interaction, (Changed<Interaction>, With<Button>)>,
    cell_text_q: Query<&ChildOf, With<HudTimeText>>,
    mut popup: Query<&mut Visibility, With<RewindPopup>>,
) {
    let Some(cell_entity) = cell_text_q.iter().next().map(|p| p.parent()) else { return };
    let Ok(interaction) = cells.get(cell_entity) else { return };
    if *interaction != Interaction::Pressed { return; }
    let Ok(mut vis) = popup.single_mut() else { return };
    *vis = match *vis {
        Visibility::Hidden => Visibility::Visible,
        _ => Visibility::Hidden,
    };
}

/// Slider drag → rewind to the chosen sim time. Value is in ms;
/// converted to ns for the driver call. Gated on
/// `Interaction::Pressed` so the system only fires while the user is
/// actually touching the slider — otherwise the cadence-driven
/// `sync_rewind_slider_max` writes would trigger `Changed<Slider>`
/// every frame, and we'd interpret each frame's stale value as a
/// fresh rewind request.
fn push_rewind_slider(
    sliders: Query<(&Slider, &Interaction), (Changed<Slider>, With<RewindSlider>)>,
    snapshot: Res<SimSnapshotRes>,
    mut driver: ResMut<SimDriverRes>,
    mut clock: ResMut<SimClock>,
    mut pending: ResMut<PendingRewindTarget>,
) {
    for (slider, interaction) in sliders.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let target_ns = (slider.value as f64 * 1_000_000.0) as u64;
        if target_ns.abs_diff(snapshot.0.now_ns) < 1_000_000 { continue; }
        apply_rewind(target_ns, &mut driver);
        clock.paused = true;
        pending.0 = Some(target_ns);
    }
}

/// When the slider isn't being dragged, mirror the current sim time
/// onto its value. Reading `Interaction` lets us distinguish "user
/// is touching the slider" (don't fight them) from "sim is running
/// forward" (slider tracks live time).
fn sync_rewind_slider_from_snapshot(
    snapshot: Res<SimSnapshotRes>,
    mut sliders: Query<(&mut Slider, &Interaction), With<RewindSlider>>,
) {
    let now_ms = snapshot.0.now_ns as f32 / 1_000_000.0;
    for (mut slider, interaction) in sliders.iter_mut() {
        if *interaction == Interaction::Pressed { continue; }
        if (slider.value - now_ms).abs() > 0.5 {
            slider.value = now_ms.clamp(slider.min, slider.max);
        }
    }
}

/// Update the slider's `max` to `max(latest_marker, current_now)` so
/// it grows as the sim advances. Slider values past the new max are
/// clamped by the widget itself on next interaction.
fn sync_rewind_slider_max(
    snapshot: Res<SimSnapshotRes>,
    mut sliders: Query<&mut Slider, With<RewindSlider>>,
) {
    let max_ns = snapshot
        .0
        .rewind_markers_ns
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(snapshot.0.now_ns);
    // Always at least 1ms wide so the slider is interactable even
    // before any sim time has elapsed.
    let max_ms = ((max_ns as f32) / 1_000_000.0).max(1.0);
    for mut slider in sliders.iter_mut() {
        if (slider.max - max_ms).abs() > 0.5 {
            slider.max = max_ms;
        }
    }
}

// ────────────────────────────────────────────────────────────
// Rewind-strategy cell
// ────────────────────────────────────────────────────────────

/// Cycle the active visual strategy on click. Goes straight at the
/// `VisualTimelineRes` — there's no separate setting resource because
/// the timeline already owns the truth.
fn handle_visual_strategy_click(
    cells: Query<&Interaction, (Changed<Interaction>, With<HudVisualStrategyCell>)>,
    mut timeline: ResMut<VisualTimelineRes>,
) {
    for interaction in cells.iter() {
        if *interaction == Interaction::Pressed {
            timeline.strategy.cycle();
        }
    }
}

/// Keep the readout text in sync with the live strategy.
fn update_visual_strategy_readout(
    timeline: Res<VisualTimelineRes>,
    mut q: Query<&mut Text, With<HudVisualStrategyText>>,
) {
    let Ok(mut text) = q.single_mut() else { return; };
    let label = format!("vis: {}", timeline.strategy.kind().label());
    if text.0 != label { text.0 = label; }
}

/// Push `k` resource changes (e.g. `-` / `=` keybind) back to the
/// slider so the popup matches reality whenever it's opened. Only
/// writes when the values disagree — avoids fighting an in-flight
/// drag.
fn sync_vis_k_slider_from_resource(
    timeline: Res<VisualTimelineRes>,
    mut sliders: Query<&mut Slider, With<VisKSlider>>,
) {
    if !timeline.is_changed() { return; }
    let k = (timeline.k() as f32).clamp(VIS_K_MIN, VIS_K_MAX);
    for mut slider in sliders.iter_mut() {
        if (slider.value - k).abs() > 1e-3 {
            slider.value = k;
        }
    }
}
