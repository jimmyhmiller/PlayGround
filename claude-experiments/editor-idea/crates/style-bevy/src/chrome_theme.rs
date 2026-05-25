//! Bridge: theme tokens → pane chrome look.
//!
//! `pane-bevy` owns `ChromeStyle` and the SDF material that reads it.
//! It can't depend on `style-bevy` (that's the wrong direction —
//! style-bevy depends on pane-bevy), so the theme→chrome wiring lives
//! here.
//!
//! Whenever the active theme changes, we copy the `PANE_*` tokens
//! into `ChromeStyle`. `pane-bevy::sync_chrome_uniforms` already
//! watches the resource via `Res::is_changed` and re-pushes uniforms
//! for every pane in the same frame.

use bevy::prelude::*;
use pane_bevy::ChromeStyle;

use crate::theme::{tokens, Theme, ThemeChanged};

pub struct ChromeThemePlugin;

impl Plugin for ChromeThemePlugin {
    fn build(&self, app: &mut App) {
        // Run once at startup so panes spawned in the first frame see
        // theme-driven look, not the bare ChromeStyle::default().
        app.add_systems(Startup, apply_theme_to_chrome)
            .add_systems(Update, apply_on_theme_changed);
    }
}

fn apply_on_theme_changed(
    mut events: MessageReader<ThemeChanged>,
    theme: Res<Theme>,
    style: ResMut<ChromeStyle>,
) {
    if events.read().last().is_none() {
        return;
    }
    write_style(&theme, style);
}

fn apply_theme_to_chrome(theme: Res<Theme>, style: ResMut<ChromeStyle>) {
    write_style(&theme, style);
}

fn write_style(theme: &Theme, mut style: ResMut<ChromeStyle>) {
    let bg = theme.color(tokens::PANE_BG);
    let border = theme.color(tokens::PANE_BORDER);
    let border_focused = theme.color(tokens::PANE_BORDER_FOCUSED);
    let focus_glow = theme.color(tokens::PANE_FOCUS_GLOW);

    style.bg = Vec4::new(bg.red, bg.green, bg.blue, bg.alpha);
    style.border = Vec4::new(border.red, border.green, border.blue, border.alpha);
    style.border_focused = Vec4::new(
        border_focused.red,
        border_focused.green,
        border_focused.blue,
        border_focused.alpha,
    );
    style.focus_glow = Vec4::new(focus_glow.red, focus_glow.green, focus_glow.blue, 1.0);
    style.corner_radius = theme.f32(tokens::PANE_CORNER_RADIUS);
    style.border_width = theme.f32(tokens::PANE_BORDER_WIDTH);
    style.border_width_focused = theme.f32(tokens::PANE_BORDER_WIDTH_FOCUSED);
    style.focus_width = theme.f32(tokens::PANE_FOCUS_WIDTH);
    style.focus_strength = theme.f32(tokens::PANE_FOCUS_STRENGTH);

    let shadow = theme.color(tokens::PANE_SHADOW_COLOR);
    style.shadow_color = Vec4::new(shadow.red, shadow.green, shadow.blue, shadow.alpha);
    style.shadow_blur = theme.f32(tokens::PANE_SHADOW_BLUR);
    style.shadow_offset_y = theme.f32(tokens::PANE_SHADOW_OFFSET_Y);
}
