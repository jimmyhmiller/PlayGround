//! Bottom / top HUD chrome — a thin bordered pill of stretch-to-fit cells.
//! Each cell is either a button (play / step / speed chip) or a readout
//! (time, counter). Consumers compose the HUD by spawning a bar root and
//! stuffing it with cells:
//!
//! ```ignore
//! let hud = spawn_hud_bar(&mut commands, &theme, hud_bottom_left());
//! commands.entity(hud).with_children(|bar| {
//!     hud_button_cell(bar, &theme, HudButtonStyle::Accent, "❚❚", PlayBtn);
//!     hud_button_cell(bar, &theme, HudButtonStyle::Muted, "›|", StepBtn);
//!     hud_text_cell(bar, &theme, "0.00s", (HudTimeText,));
//!     hud_speed_chips(bar, &theme, &[(0.25, "1/4x"), (1.0, "1x"), ...], 1.0,
//!         |multiplier| SpeedChip(multiplier));
//!     hud_counter_cell(bar, &theme, "0 pkts", (HudPktCount,));
//! });
//! ```
//!
//! Consumer-defined markers (`PlayBtn`, `SpeedChip(_)`, `HudTimeText`, etc.)
//! carry the click / update wiring. The library only owns the look.

use crate::theme::Theme;
use crate::typography::{Bold, Mono};
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;

pub struct HudReskinPlugin;

impl Plugin for HudReskinPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, re_skin_hud);
    }
}

pub const HUD_HEIGHT: f32 = 44.0;

// ────────────────────────────────────────────────────────────
// Re-skin marker components
// ────────────────────────────────────────────────────────────

/// HUD bar root background.
#[derive(Component)]
pub struct HudBg;

/// Cell that should track `theme.ink` borders & ink colour.
#[derive(Component)]
pub struct HudDivider;

/// Text that should track `theme.ink`.
#[derive(Component)]
pub struct HudInkText;

/// Text that should track `theme.ink_soft`.
#[derive(Component)]
pub struct HudMutedText;

/// Paper-coloured text (lives on accent/ink fills).
#[derive(Component)]
pub struct HudPaperText;

// ────────────────────────────────────────────────────────────
// Button visual styles
// ────────────────────────────────────────────────────────────

/// Which background fill a HUD button uses. Affects the re-skin system and
/// the inner text colour at spawn.
#[derive(Clone, Copy)]
pub enum HudButtonStyle {
    /// Accent fill (active CTA — e.g. play when running).
    Accent,
    /// Ink fill (the inverted "depressed" look — e.g. play when paused).
    Ink,
    /// Transparent on paper — the resting look.
    Muted,
}

/// Marker that records which [`HudButtonStyle`] a button should re-skin back
/// to when the theme swaps. Consumers can swap the style by mutating this
/// component and changing the `BackgroundColor` themselves; the re-skin
/// system follows this marker on theme change.
#[derive(Component, Clone, Copy)]
pub struct HudButtonFill(pub HudButtonStyle);

// ────────────────────────────────────────────────────────────
// Layout presets
// ────────────────────────────────────────────────────────────

/// Node-layout preset: bottom-left corner at 20px inset.
pub fn hud_bottom_left() -> Node {
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
    }
}

/// Top-left corner at 20px inset.
pub fn hud_top_left() -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: Val::Px(20.0),
        top: Val::Px(20.0),
        height: Val::Px(HUD_HEIGHT),
        border: UiRect::all(Val::Px(1.5)),
        border_radius: BorderRadius::all(Val::Px(8.0)),
        align_items: AlignItems::Stretch,
        overflow: Overflow::clip(),
        ..default()
    }
}

/// Spawn the HUD bar root entity. Fill it with cells via `.with_children`.
pub fn spawn_hud_bar(commands: &mut Commands, theme: &Theme, layout: Node) -> Entity {
    commands
        .spawn((
            layout,
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.ink),
            HudBg,
        ))
        .id()
}

// ────────────────────────────────────────────────────────────
// Cells
// ────────────────────────────────────────────────────────────

/// A button cell — fixed-width, centered glyph, right-border divider.
/// `extra` is the consumer-defined marker for click routing.
pub fn hud_button_cell(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    style: HudButtonStyle,
    glyph: &str,
    extra: impl Bundle,
) {
    let (bg, text_c) = hud_fill_colors(theme, style);
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(44.0),
                border: UiRect::right(Val::Px(1.5)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(bg),
            BorderColor::all(theme.ink),
            HudDivider,
            HudButtonFill(style),
            extra,
        ))
        .with_children(|p| {
            let mut text_entity = p.spawn((
                Text::new(glyph),
                TextFont { font_size: 14.0, ..default() },
                TextColor(text_c),
            ));
            match style {
                HudButtonStyle::Accent | HudButtonStyle::Ink => {
                    text_entity.insert(HudPaperText);
                }
                HudButtonStyle::Muted => {
                    text_entity.insert(HudInkText);
                }
            }
        });
}

/// A narrower button cell (36px) — for secondary actions like step-forward.
pub fn hud_step_cell(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    glyph: &str,
    extra: impl Bundle,
) {
    parent
        .spawn((
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
            HudDivider,
            HudButtonFill(HudButtonStyle::Muted),
            extra,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(glyph),
                TextFont { font_size: 14.0, ..default() },
                TextColor(theme.ink),
                HudInkText,
            ));
        });
}

/// A text readout cell — monospaced, with horizontal padding and a
/// right-border divider. `extra` typically carries a marker like
/// `HudTimeText` so the consumer's per-frame update system can find it.
pub fn hud_text_cell(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    text: &str,
    extra: impl Bundle,
) {
    parent
        .spawn((
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
                Text::new(text.to_string()),
                TextFont { font_size: 13.0, ..default() },
                TextColor(theme.ink),
                HudInkText,
                Mono,
                extra,
            ));
        });
}

/// One speed chip. Active chips paint with `theme.ink` fill + paper text;
/// inactive with a rule-coloured border and ink-soft text. `is_last` drops
/// the trailing inner divider.
pub fn hud_speed_chip(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    active: bool,
    is_last: bool,
    extra: impl Bundle,
) {
    parent
        .spawn((
            Button,
            Node {
                min_width: Val::Px(34.0),
                padding: UiRect::horizontal(Val::Px(8.0)),
                border: if is_last {
                    UiRect::default()
                } else {
                    UiRect::right(Val::Px(1.0))
                },
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(if active { theme.ink } else { Color::NONE }),
            BorderColor::all(theme.rule),
            extra,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(label),
                TextFont { font_size: 11.0, ..default() },
                TextColor(if active { theme.paper } else { theme.ink_soft }),
            ));
        });
}

/// Horizontal chip strip wrapper. Use as the parent for [`hud_speed_chip`]
/// calls so they're laid out in a row with a trailing divider.
pub fn hud_chip_strip(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    body_fn: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            Node {
                border: UiRect::right(Val::Px(1.5)),
                align_items: AlignItems::Stretch,
                ..default()
            },
            BorderColor::all(theme.ink),
            HudDivider,
        ))
        .with_children(body_fn);
}

/// A horizontal run of small counter labels. Put [`hud_counter`] calls
/// inside.
pub fn hud_counter_strip(
    parent: &mut ChildSpawnerCommands,
    body_fn: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn(Node {
            padding: UiRect::horizontal(Val::Px(12.0)),
            column_gap: Val::Px(10.0),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(body_fn);
}

/// A small muted-text counter label ("4 pkts", "12 done").
pub fn hud_counter(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    text: &str,
    extra: impl Bundle,
) {
    parent.spawn((
        Text::new(text.to_string()),
        TextFont { font_size: 11.0, ..default() },
        TextColor(theme.ink_soft),
        HudMutedText,
        Mono,
        Bold,
        extra,
    ));
}

// ────────────────────────────────────────────────────────────
// Color resolution
// ────────────────────────────────────────────────────────────

fn hud_fill_colors(theme: &Theme, style: HudButtonStyle) -> (Color, Color) {
    match style {
        HudButtonStyle::Accent => (theme.accent, theme.paper),
        HudButtonStyle::Ink => (theme.ink, theme.paper),
        HudButtonStyle::Muted => (theme.paper_alt, theme.ink),
    }
}

// ────────────────────────────────────────────────────────────
// Re-skin system
// ────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn re_skin_hud(
    theme: Res<Theme>,
    mut hud_bg: Query<&mut BackgroundColor, (With<HudBg>, Without<HudButtonFill>)>,
    mut hud_border: Query<&mut BorderColor, With<HudBg>>,
    mut divider_borders: Query<&mut BorderColor, (With<HudDivider>, Without<HudBg>)>,
    mut button_fills: Query<(&HudButtonFill, &mut BackgroundColor), Without<HudBg>>,
    mut ink_text: Query<&mut TextColor, (With<HudInkText>, Without<HudMutedText>, Without<HudPaperText>)>,
    mut muted_text: Query<&mut TextColor, (With<HudMutedText>, Without<HudInkText>, Without<HudPaperText>)>,
    mut paper_text: Query<&mut TextColor, (With<HudPaperText>, Without<HudInkText>, Without<HudMutedText>)>,
) {
    if !theme.is_changed() {
        return;
    }
    for mut bg in hud_bg.iter_mut() { bg.0 = theme.paper_alt; }
    for mut b in hud_border.iter_mut() { *b = BorderColor::all(theme.ink); }
    for mut b in divider_borders.iter_mut() { *b = BorderColor::all(theme.ink); }
    for (fill, mut bg) in button_fills.iter_mut() {
        bg.0 = hud_fill_colors(&theme, fill.0).0;
    }
    for mut t in ink_text.iter_mut() { t.0 = theme.ink; }
    for mut t in muted_text.iter_mut() { t.0 = theme.ink_soft; }
    for mut t in paper_text.iter_mut() { t.0 = theme.paper; }
}
