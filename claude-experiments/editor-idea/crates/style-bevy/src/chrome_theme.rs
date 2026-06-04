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
use pane_bevy::{
    ChromeStyle, ChromeTextStyle, PaneChromeShader, PaneChromeStyle, PaneProject, PaneTag,
};

use crate::presets::StylePresetRegistry;
use crate::state::ProjectStyleState;
use crate::theme::{tokens, ProjectThemes, Theme, ThemeChanged};

/// Embedded default chrome fragment shader (the rounded-rect SDF). Used
/// for projects whose preset doesn't ship a custom `chrome.wgsl`.
const DEFAULT_CHROME_SHADER: &str = "embedded://pane_bevy/chrome_material.wgsl";

pub struct ChromeThemePlugin;

impl Plugin for ChromeThemePlugin {
    fn build(&self, app: &mut App) {
        // Run once at startup so panes spawned in the first frame see
        // theme-driven look, not the bare ChromeStyle::default().
        app.add_systems(Startup, apply_theme_to_chrome)
            .add_systems(Update, (apply_on_theme_changed, apply_per_project_chrome));
    }
}

/// Stamp each pane with its OWN project's chrome look (`PaneChromeStyle`)
/// from the per-project theme cache, so a pane's card/border/title match
/// its project — in the cube overview and in flat view. Panes whose
/// project has no cached theme fall back to the global `ChromeStyle`
/// (the override is removed). Recomputes when the cache changes or a pane
/// is new / changes project.
#[allow(clippy::type_complexity)]
fn apply_per_project_chrome(
    mut commands: Commands,
    themes: Res<ProjectThemes>,
    style_state: Res<ProjectStyleState>,
    registry: Res<StylePresetRegistry>,
    asset_server: Res<AssetServer>,
    panes: Query<
        (Entity, &PaneProject, Option<&PaneChromeStyle>),
        (With<PaneTag>, Or<(Changed<PaneProject>, Added<PaneProject>)>),
    >,
    all_panes: Query<(Entity, &PaneProject, Option<&PaneChromeStyle>), With<PaneTag>>,
) {
    // Refresh everything when the cache or preset registry moves (preset
    // pick / theme edit / project switch / preset discovery); otherwise
    // only newly-spawned / reparented panes need stamping.
    let full = themes.is_changed() || registry.is_changed();
    let ctx = ChromeCtx {
        themes: &themes,
        style_state: &style_state,
        registry: &registry,
        asset_server: &asset_server,
    };
    if full {
        for (e, proj, existing) in &all_panes {
            stamp(&mut commands, e, proj.0, existing.is_some(), &ctx);
        }
    } else {
        for (e, proj, existing) in &panes {
            stamp(&mut commands, e, proj.0, existing.is_some(), &ctx);
        }
    }
}

struct ChromeCtx<'a> {
    themes: &'a ProjectThemes,
    style_state: &'a ProjectStyleState,
    registry: &'a StylePresetRegistry,
    asset_server: &'a AssetServer,
}

fn stamp(
    commands: &mut Commands,
    entity: Entity,
    project_id: u64,
    has_override: bool,
    ctx: &ChromeCtx,
) {
    match ctx.themes.get(project_id) {
        Some(theme) => {
            // The project's preset chrome shader (or the embedded default).
            let url = ctx
                .style_state
                .preset_of(project_id)
                .and_then(|name| {
                    ctx.registry
                        .presets
                        .iter()
                        .find(|p| p.name == name)
                        .and_then(|p| p.chrome_shader.clone())
                })
                .unwrap_or_else(|| DEFAULT_CHROME_SHADER.to_string());
            commands.entity(entity).insert((
                PaneChromeStyle(build_chrome_style(theme)),
                PaneChromeShader(ctx.asset_server.load::<bevy::shader::Shader>(url)),
            ));
        }
        None if has_override => {
            commands
                .entity(entity)
                .remove::<PaneChromeStyle>()
                .remove::<PaneChromeShader>();
        }
        None => {}
    }
}

fn apply_on_theme_changed(
    mut events: MessageReader<ThemeChanged>,
    theme: Res<Theme>,
    style: ResMut<ChromeStyle>,
    text_style: ResMut<ChromeTextStyle>,
) {
    if events.read().last().is_none() {
        return;
    }
    write_style(&theme, style);
    write_text_style(&theme, text_style);
}

fn apply_theme_to_chrome(
    theme: Res<Theme>,
    style: ResMut<ChromeStyle>,
    text_style: ResMut<ChromeTextStyle>,
) {
    write_style(&theme, style);
    write_text_style(&theme, text_style);
}

fn write_text_style(theme: &Theme, mut text_style: ResMut<ChromeTextStyle>) {
    let title = theme.color(tokens::CHROME_TITLE);
    let title_focused = theme.color(tokens::CHROME_TITLE_FOCUSED);
    let divider = theme.color(tokens::CHROME_DIVIDER);
    let close = theme.color(tokens::CHROME_CLOSE);
    let handle = theme.color(tokens::CHROME_HANDLE);
    text_style.title = Color::LinearRgba(title);
    text_style.title_focused = Color::LinearRgba(title_focused);
    text_style.divider = Color::LinearRgba(divider);
    text_style.close = Color::LinearRgba(close);
    text_style.handle = Color::LinearRgba(handle);
}

fn write_style(theme: &Theme, mut style: ResMut<ChromeStyle>) {
    *style = build_chrome_style(theme);
}

/// Build a complete [`ChromeStyle`] from a theme's `PANE_*`/`CHROME_*`
/// tokens. Used both for the global style and for each pane's
/// per-project override.
pub fn build_chrome_style(theme: &Theme) -> ChromeStyle {
    let v4 = |c: bevy::color::LinearRgba| Vec4::new(c.red, c.green, c.blue, c.alpha);
    let focus_glow = theme.color(tokens::PANE_FOCUS_GLOW);
    let mut style = ChromeStyle::default();
    style.bg = v4(theme.color(tokens::PANE_BG));
    style.border = v4(theme.color(tokens::PANE_BORDER));
    style.border_focused = v4(theme.color(tokens::PANE_BORDER_FOCUSED));
    style.focus_glow = Vec4::new(focus_glow.red, focus_glow.green, focus_glow.blue, 1.0);
    style.corner_radius = theme.f32(tokens::PANE_CORNER_RADIUS);
    style.border_width = theme.f32(tokens::PANE_BORDER_WIDTH);
    style.border_width_focused = theme.f32(tokens::PANE_BORDER_WIDTH_FOCUSED);
    style.focus_width = theme.f32(tokens::PANE_FOCUS_WIDTH);
    style.focus_strength = theme.f32(tokens::PANE_FOCUS_STRENGTH);
    style.title_bg = v4(theme.color(tokens::CHROME_TITLE_BG));
    style.title_bg_focused = v4(theme.color(tokens::CHROME_TITLE_BG_FOCUSED));
    style.shadow_color = v4(theme.color(tokens::PANE_SHADOW_COLOR));
    style.shadow_blur = theme.f32(tokens::PANE_SHADOW_BLUR);
    style.shadow_offset_y = theme.f32(tokens::PANE_SHADOW_OFFSET_Y);
    style
}
