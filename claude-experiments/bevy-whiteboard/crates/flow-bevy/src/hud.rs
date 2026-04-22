//! Bottom-left HUD: play/pause, step, sim-time readout, speed chips, counters.
//! Built from poster-ui cell primitives; this module owns the domain wiring
//! (which flow-bevy state each cell reads / writes).

use bevy::prelude::*;
use poster_ui::{
    HudButtonFill, HudButtonStyle, Mono, Theme,
    hud_bottom_left, hud_button_cell, hud_chip_strip, hud_counter, hud_counter_strip,
    hud_speed_chip, hud_step_cell, hud_text_cell, spawn_hud_bar,
};

use crate::bridge::{FlowSim, SimClock};

pub struct HudPlugin;
impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_hud)
            .add_systems(
                Update,
                (
                    update_time_readout,
                    update_counts,
                    handle_play_button,
                    handle_step_button,
                    handle_speed_chips,
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
struct HudPlayIcon;

#[derive(Component)]
struct HudStepBtn;

#[derive(Component)]
struct HudTimeText;

#[derive(Component)]
struct HudNodeCount;

#[derive(Component)]
struct HudEdgeCount;

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
        hud_button_cell(bar, &theme, play_style, play_glyph, (HudPlayBtn, HudPlayIcon));
        hud_step_cell(bar, &theme, "›|", HudStepBtn);
        hud_text_cell(bar, &theme, "0.000 ms", HudTimeText);

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
        });
    });
}

// ────────────────────────────────────────────────────────────
// Per-frame updates
// ────────────────────────────────────────────────────────────

fn update_time_readout(
    flow: Res<FlowSim>,
    mut q: Query<&mut Text, With<HudTimeText>>,
) {
    let Ok(mut text) = q.single_mut() else { return };
    let t_ms = flow.sim.now_ns as f64 / 1_000_000.0;
    let new = format!("{:.3} ms", t_ms);
    if text.0 != new { text.0 = new; }
}

fn update_counts(
    flow: Res<FlowSim>,
    mut nodes: Query<&mut Text, (With<HudNodeCount>, Without<HudEdgeCount>)>,
    mut edges: Query<&mut Text, (With<HudEdgeCount>, Without<HudNodeCount>)>,
) {
    let n = format!("{} nodes", flow.sim.nodes.len());
    let e = format!("{} edges", flow.sim.edges.len());
    for mut t in nodes.iter_mut() { if t.0 != n { t.0 = n.clone(); } }
    for mut t in edges.iter_mut() { if t.0 != e { t.0 = e.clone(); } }
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
    mut flow: ResMut<FlowSim>,
) {
    for i in q.iter() {
        if *i == Interaction::Pressed {
            crate::palette::step_to_visible(&mut flow.sim);
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
    mut text_q: Query<&mut Text, With<HudPlayIcon>>,
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
