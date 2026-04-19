//! Floating UI overlays that frame the canvas: top-left Studio stamp,
//! top-center hint line, bottom-left HUD (play/pause/step/time/speed/counts).
//! Each overlay reads the live [`Theme`] and re-skins in place when the
//! theme changes — same pattern as `palette.rs`.

use crate::bridge::{SimResource, SimSpeed};
use crate::sim::NS_PER_S;
use crate::theme::Theme;
use bevy::prelude::*;

pub struct ChromePlugin;

impl Plugin for ChromePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_bottom_hud).add_systems(
            Update,
            (
                re_skin_chrome,
                update_hud_time,
                update_hud_counts,
                handle_hud_buttons,
                sync_hud_visuals,
            ),
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// Bottom-left HUD
// ──────────────────────────────────────────────────────────────────

#[derive(Component)]
struct HudBg;
#[derive(Component)]
struct HudPlayBtn;
#[derive(Component)]
struct HudStepBtn;
#[derive(Component)]
struct HudPlayBtnIcon; // Text2d/Text inside the play button — ▶ or ❚❚
#[derive(Component, Clone, Copy)]
struct HudSpeedBtn(f64);
#[derive(Component)]
struct HudTimeText;
#[derive(Component)]
struct HudPktCountText;
#[derive(Component)]
struct HudDoneCountText;
#[derive(Component)]
struct HudDivider;
#[derive(Component)]
struct HudInkText;

const HUD_HEIGHT: f32 = 44.0;
const SPEED_OPTIONS: [(f64, &str); 5] = [
    (0.25, "1/4x"),
    (0.5, "1/2x"),
    (1.0, "1x"),
    (2.0, "2x"),
    (4.0, "4x"),
];

fn spawn_bottom_hud(mut commands: Commands, theme: Res<Theme>, speed: Res<SimSpeed>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0),
                bottom: Val::Px(20.0),
                height: Val::Px(HUD_HEIGHT),
                border: UiRect::all(Val::Px(1.5)),
                border_radius: BorderRadius::all(Val::Px(8.0)),
                align_items: AlignItems::Stretch,
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.ink),
            HudBg,
        ))
        .with_children(|hud| {
            // Play/pause cell
            hud.spawn((
                Button,
                Node {
                    width: Val::Px(44.0),
                    border: UiRect::right(Val::Px(1.5)),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                BackgroundColor(if speed.paused { theme.ink } else { theme.accent }),
                BorderColor::all(theme.ink),
                HudPlayBtn,
            ))
            .with_children(|p| {
                p.spawn((
                    Text::new(if speed.paused { "▶" } else { "❚❚" }),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(theme.paper),
                    HudPlayBtnIcon,
                ));
            });

            // Step cell
            hud.spawn((
                Button,
                Node {
                    width: Val::Px(36.0),
                    border: UiRect::right(Val::Px(1.5)),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                BackgroundColor(theme.paper_alt),
                BorderColor::all(theme.ink),
                HudStepBtn,
                HudDivider,
            ))
            .with_children(|p| {
                p.spawn((
                    Text::new("›|"),
                    TextFont { font_size: 14.0, ..default() },
                    TextColor(theme.ink),
                    HudInkText,
                ));
            });

            // Time readout
            hud.spawn((
                Node {
                    min_width: Val::Px(78.0),
                    padding: UiRect::horizontal(Val::Px(12.0)),
                    border: UiRect::right(Val::Px(1.5)),
                    align_items: AlignItems::Center,
                    ..default()
                },
                BorderColor::all(theme.ink),
                HudDivider,
            ))
            .with_children(|p| {
                p.spawn((
                    Text::new("0.00s"),
                    TextFont { font_size: 13.0, ..default() },
                    TextColor(theme.ink),
                    HudTimeText,
                ));
            });

            // Speed chips
            hud.spawn((
                Node {
                    border: UiRect::right(Val::Px(1.5)),
                    align_items: AlignItems::Stretch,
                    ..default()
                },
                BorderColor::all(theme.ink),
                HudDivider,
            ))
            .with_children(|chips| {
                for (i, (val, label)) in SPEED_OPTIONS.iter().enumerate() {
                    let active = !speed.paused && (speed.multiplier - val).abs() < 1e-3;
                    chips.spawn((
                        Button,
                        Node {
                            min_width: Val::Px(34.0),
                            padding: UiRect::horizontal(Val::Px(8.0)),
                            border: if i + 1 < SPEED_OPTIONS.len() {
                                UiRect::right(Val::Px(1.0))
                            } else {
                                UiRect::default()
                            },
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BackgroundColor(if active { theme.ink } else { Color::NONE }),
                        BorderColor::all(theme.rule),
                        HudSpeedBtn(*val),
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new(*label),
                            TextFont { font_size: 11.0, ..default() },
                            TextColor(if active { theme.paper } else { theme.ink_soft }),
                        ));
                    });
                }
            });

            // Counters
            hud.spawn((
                Node {
                    padding: UiRect::horizontal(Val::Px(12.0)),
                    column_gap: Val::Px(10.0),
                    align_items: AlignItems::Center,
                    ..default()
                },
            ))
            .with_children(|p| {
                p.spawn((
                    Text::new("0 pkts"),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(theme.ink_soft),
                    HudPktCountText,
                ));
                p.spawn((
                    Text::new("0 done"),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(theme.ink_soft),
                    HudDoneCountText,
                ));
            });
        });
}

fn update_hud_time(
    sim: Res<SimResource>,
    mut q: Query<&mut Text, With<HudTimeText>>,
) {
    let secs = sim.0.now_ns as f64 / NS_PER_S as f64;
    let new = format!("{:.2}s", secs);
    for mut text in q.iter_mut() {
        if text.0 != new {
            text.0 = new.clone();
        }
    }
}

fn update_hud_counts(
    sim: Res<SimResource>,
    mut pkt_q: Query<&mut Text, (With<HudPktCountText>, Without<HudDoneCountText>)>,
    mut done_q: Query<&mut Text, (With<HudDoneCountText>, Without<HudPktCountText>)>,
) {
    use crate::sim::NodeKind;
    let mut workers_busy = 0u32;
    let mut queued = 0u32;
    let mut sunk = 0u32;
    for n in sim.0.nodes.values() {
        match n.kind {
            NodeKind::Worker => {
                if n.holding.is_some() {
                    workers_busy += 1;
                }
            }
            NodeKind::Queue => queued += n.buffer.len() as u32,
            NodeKind::Sink => sunk += n.sink_total,
            _ => {}
        }
    }
    let in_flight = workers_busy + queued;
    let new_pkt = format!("{} pkts", in_flight);
    let new_done = format!("{} done", sunk);
    for mut t in pkt_q.iter_mut() {
        if t.0 != new_pkt {
            t.0 = new_pkt.clone();
        }
    }
    for mut t in done_q.iter_mut() {
        if t.0 != new_done {
            t.0 = new_done.clone();
        }
    }
}

fn handle_hud_buttons(
    mut speed: ResMut<SimSpeed>,
    play: Query<&Interaction, (Changed<Interaction>, With<HudPlayBtn>)>,
    step: Query<&Interaction, (Changed<Interaction>, With<HudStepBtn>)>,
    chips: Query<(&Interaction, &HudSpeedBtn), Changed<Interaction>>,
) {
    for i in play.iter() {
        if *i == Interaction::Pressed {
            speed.paused = !speed.paused;
        }
    }
    for i in step.iter() {
        if *i == Interaction::Pressed {
            // Step a single 100ms tick at the current multiplier — same
            // budget the keyboard `.` shortcut uses.
            let ns = (0.1f64 * NS_PER_S as f64) as u64;
            speed.step_once_ns = Some(ns);
        }
    }
    for (i, btn) in chips.iter() {
        if *i == Interaction::Pressed {
            speed.multiplier = btn.0;
            speed.paused = false;
        }
    }
}

fn sync_hud_visuals(
    theme: Res<Theme>,
    speed: Res<SimSpeed>,
    mut play_q: Query<
        (&mut BackgroundColor, &Children),
        (With<HudPlayBtn>, Without<HudSpeedBtn>),
    >,
    mut play_icon_q: Query<(&mut Text, &mut TextColor), With<HudPlayBtnIcon>>,
    mut chip_q: Query<
        (&HudSpeedBtn, &Children, &mut BackgroundColor),
        (With<Button>, Without<HudPlayBtn>),
    >,
    mut chip_text_q: Query<&mut TextColor, (Without<HudPlayBtnIcon>, Without<HudPlayBtn>)>,
) {
    for (mut bg, children) in play_q.iter_mut() {
        bg.0 = if speed.paused { theme.ink } else { theme.accent };
        for child in children.iter() {
            if let Ok((mut text, mut color)) = play_icon_q.get_mut(child) {
                text.0 = if speed.paused { "▶" } else { "❚❚" }.to_string();
                color.0 = theme.paper;
            }
        }
    }
    for (chip, children, mut bg) in chip_q.iter_mut() {
        let active = !speed.paused && (speed.multiplier - chip.0).abs() < 1e-3;
        bg.0 = if active { theme.ink } else { Color::NONE };
        for child in children.iter() {
            if let Ok(mut tc) = chip_text_q.get_mut(child) {
                tc.0 = if active { theme.paper } else { theme.ink_soft };
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Theme re-skin
// ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn re_skin_chrome(
    theme: Res<Theme>,
    mut hud_bg: Query<&mut BackgroundColor, With<HudBg>>,
    mut hud_border: Query<&mut BorderColor, With<HudBg>>,
    mut divider_borders: Query<&mut BorderColor, (With<HudDivider>, Without<HudBg>)>,
    mut hud_time: Query<&mut TextColor, (With<HudTimeText>, Without<HudInkText>, Without<HudPktCountText>, Without<HudDoneCountText>)>,
    mut hud_ink: Query<&mut TextColor, (With<HudInkText>, Without<HudTimeText>, Without<HudPktCountText>, Without<HudDoneCountText>)>,
    mut hud_count_pkt: Query<&mut TextColor, (With<HudPktCountText>, Without<HudTimeText>, Without<HudInkText>, Without<HudDoneCountText>)>,
    mut hud_count_done: Query<&mut TextColor, (With<HudDoneCountText>, Without<HudTimeText>, Without<HudInkText>, Without<HudPktCountText>)>,
) {
    if !theme.is_changed() {
        return;
    }
    for mut bg in hud_bg.iter_mut() { bg.0 = theme.paper_alt; }
    for mut b in hud_border.iter_mut() { *b = BorderColor::all(theme.ink); }
    for mut b in divider_borders.iter_mut() { *b = BorderColor::all(theme.ink); }
    for mut t in hud_time.iter_mut() { t.0 = theme.ink; }
    for mut t in hud_ink.iter_mut() { t.0 = theme.ink; }
    for mut t in hud_count_pkt.iter_mut() { t.0 = theme.ink_soft; }
    for mut t in hud_count_done.iter_mut() { t.0 = theme.ink_soft; }
}
