//! Bottom-left HUD: play/pause, step, sim-time readout, speed chips, counters.
//! Built from poster-ui cell primitives; this module owns the domain wiring
//! (which flow-bevy state each cell reads / writes).

use bevy::prelude::*;
use poster_ui::{
    HudButtonFill, HudButtonStyle, Mono, Slider, Theme,
    hud_bottom_left, hud_button_cell, hud_chip_strip, hud_counter, hud_counter_strip,
    hud_speed_chip, hud_step_cell, hud_text_cell, spawn_hud_bar, spawn_slider_with_step,
};

use crate::bridge::SimClock;
use crate::edges::VisualTimelineRes;
use crate::sim_driver::{SimCommand, SimDriverRes, SimSnapshotRes};

pub struct HudPlugin;
impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_hud, spawn_vis_k_popup))
            .add_systems(
                Update,
                (
                    update_time_readout,
                    update_vis_readout,
                    update_counts,
                    handle_play_button,
                    handle_step_button,
                    handle_speed_chips,
                    handle_vis_k_click,
                    push_vis_k_slider,
                    sync_vis_k_slider_from_resource,
                    sync_play_button_visual,
                    sync_speed_chip_visuals,
                ),
            );
    }
}

// ────────────────────────────────────────────────────────────
// Domain markers
// ────────────────────────────────────────────────────────────

#[derive(Component)]
struct HudPlayBtn;

#[derive(Component)]
struct HudStepBtn;

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
        hud_text_cell(bar, &theme, "0.000 ms", HudTimeText);
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

        hud_chip_strip(bar, &theme, |chips| {
            for (i, (val, label)) in SPEED_OPTIONS.iter().enumerate() {
                let active = !clock.paused && (clock.multiplier - val).abs() < 1e-3;
                let is_last = i + 1 == SPEED_OPTIONS.len();
                hud_speed_chip(chips, &theme, label, active, is_last, HudSpeedChip(*val));
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
    let k = timeline.0.k;
    let new = if k >= 10.0 {
        format!("vis k={}", k.round() as i64)
    } else {
        format!("vis k={:.2}", k)
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
    let initial = (timeline.0.k as f32).clamp(VIS_K_MIN, VIS_K_MAX);
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
        if (timeline.0.k - new_k).abs() < 1e-3 { continue; }
        timeline.0.set_k(new_k);
    }
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
    let k = (timeline.0.k as f32).clamp(VIS_K_MIN, VIS_K_MAX);
    for mut slider in sliders.iter_mut() {
        if (slider.value - k).abs() > 1e-3 {
            slider.value = k;
        }
    }
}
