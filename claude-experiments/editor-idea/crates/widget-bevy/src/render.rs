//! Element tree → Bevy entities + layout.
//!
//! Single-pass top-down walk. Each container is told its origin and max
//! width and returns the size it actually consumed. Children measure
//! intrinsically (no shaping — text width is approximated as
//! `chars * font_size * 0.55`); single-line, no wrapping. The visible
//! frame is rebuilt from scratch every time the widget emits one — the
//! tree size is small enough (tens of nodes) that diffing isn't worth
//! the complexity yet.

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextBounds};
use pane_bevy::PaneFontMetrics;

use crate::protocol::{parse_hex_color, Align, ButtonKind, Element, Style, TabItem, Weight};
use crate::{ClickKind, ClickTarget, LinkTarget, WidgetTargets};

const DEFAULT_FONT_SIZE: f32 = 13.0;
const LINE_HEIGHT_MUL: f32 = 1.4;

// Colors all flow through `LayoutCtx::palette` (theme-driven). See
// `WidgetPalette::from_theme` below.

const BUTTON_PAD_X: f32 = 8.0;
const BUTTON_PAD_Y: f32 = 4.0;
const BADGE_PAD_X: f32 = 6.0;
const BADGE_PAD_Y: f32 = 2.0;
const BADGE_FONT_SIZE: f32 = 11.0;

// Toggle pill dimensions. Track height = knob diameter; the knob slides
// across the inner length.
pub const TOGGLE_TRACK_W: f32 = 34.0;
pub const TOGGLE_TRACK_H: f32 = 18.0;
pub const TOGGLE_KNOB_PAD: f32 = 2.0;

// Input height = body text + symmetric padding. Width comes from the
// element itself (defaults to 160 px).
pub const INPUT_HEIGHT: f32 = 26.0;
pub const INPUT_PAD_X: f32 = 8.0;

// Tabs row.
pub const TAB_PAD_X: f32 = 10.0;
pub const TAB_PAD_Y: f32 = 6.0;
pub const TAB_GAP: f32 = 4.0;
pub const TAB_INDICATOR_H: f32 = 2.0;

pub struct LayoutCtx {
    /// Fallback font (mono). Per-element `family` overrides resolve
    /// through [`Self::font_for`].
    pub font: Handle<Font>,
    pub metrics: PaneFontMetrics,
    pub content_root: Entity,
    pub content_size: Vec2,
    /// Theme-derived colors for every primitive element. Carried in
    /// the ctx so render fns don't need to take `Res<Theme>` as a
    /// separate arg.
    pub palette: WidgetPalette,
    /// Snapshot of the active theme — render fns use this to resolve
    /// token-named colors / numbers in `Style` overrides.
    pub theme: style_bevy::Theme,
    /// Font registry; per-element `family` lookups go through here.
    pub fonts: style_bevy::FontRegistry,
    /// While focused on this pane: id + buffered value + caret pos for
    /// an Element::Input. None when no input has focus.
    pub focused_input: Option<crate::WidgetInputFocus>,
    /// Caret blink state — true when the bar is visible this frame.
    pub caret_visible: bool,
}

impl LayoutCtx {
    /// Resolve a color string. Accepts either a theme-token name
    /// (`"surface_2"`, `"accent_500"`) or a literal color form
    /// (`"#1a1d24"`, `"oklch(0.6, 0.04, 250)"`).
    pub fn resolve_color(&self, s: &str) -> Option<Color> {
        if let Some(style_bevy::TokenValue::Color(c)) = self.theme.get_by_name(s) {
            return Some(Color::LinearRgba(c));
        }
        style_bevy::theme::parse_color_string(s)
            .ok()
            .map(Color::LinearRgba)
    }

    /// Resolve an `f32` string. Accepts a theme-token name
    /// (`"radius_md"`, `"space_lg"`) or a numeric literal.
    pub fn resolve_f32(&self, s: &str) -> Option<f32> {
        if let Some(style_bevy::TokenValue::F32(v)) = self.theme.get_by_name(s) {
            return Some(v);
        }
        s.parse::<f32>().ok()
    }

    /// Resolve a shadow triple. `token` (if set) names a shadow_* group;
    /// otherwise fall back to explicit color/blur/offset_y on the
    /// `Shadow` struct, then to `shadow_md_*` theme defaults.
    pub fn resolve_shadow(&self, s: &crate::protocol::Shadow) -> (Color, f32, f32) {
        // Pick the token group: caller-supplied or "shadow_md".
        let group = s.token.as_deref().unwrap_or("shadow_md");
        let color_key = format!("{}_color", group);
        let blur_key = format!("{}_blur", group);
        let offset_key = format!("{}_offset_y", group);
        let mut color = self
            .resolve_color(&color_key)
            .unwrap_or(Color::srgba(0.0, 0.0, 0.0, 0.32));
        let mut blur = self.resolve_f32(&blur_key).unwrap_or(12.0);
        let mut offset_y = self.resolve_f32(&offset_key).unwrap_or(4.0);
        // Per-field overrides.
        if let Some(c) = s.color.as_deref().and_then(|c| self.resolve_color(c)) {
            color = c;
        }
        if let Some(b) = s.blur {
            blur = b;
        }
        if let Some(o) = s.offset_y {
            offset_y = o;
        }
        (color, blur, offset_y)
    }

    /// Resolve a font family. Accepts a registry name (`"serif"`,
    /// `"sans"`, `"mono"`) or a theme token whose value is the
    /// family name (`"font_family_heading"`).
    pub fn font_for(&self, family: &str) -> Option<Handle<Font>> {
        if let Some(style_bevy::TokenValue::Str(name)) = self.theme.get_by_name(family) {
            return Some(self.fonts.resolve(&name));
        }
        Some(self.fonts.resolve(family))
    }
}

#[derive(Clone, Debug)]
pub struct WidgetPalette {
    pub text: Color,
    pub text_muted: Color,
    pub link: Color,
    pub divider: Color,
    pub button_bg: Color,
    pub button_label: Color,
    pub badge_bg: Color,
    pub badge_label: Color,
    pub bar_track: Color,
    pub bar_fill: Color,
    // --- Button SDF style ---
    pub button_corner_radius: f32,
    pub button_border: Color,
    pub button_border_width: f32,
    pub button_shadow_color: Color,
    pub button_shadow_blur: f32,
    pub button_shadow_offset_y: f32,
}

impl WidgetPalette {
    pub fn from_theme(theme: &style_bevy::Theme) -> Self {
        use style_bevy::tokens as t;
        let c = |id| Color::LinearRgba(theme.color(id));
        let f = |id| theme.f32(id);
        Self {
            text: c(t::FG),
            text_muted: c(t::FG_MUTED),
            link: c(t::WIDGET_LINK),
            divider: c(t::CHROME_DIVIDER),
            button_bg: c(t::BUTTON_BG),
            button_label: c(t::BUTTON_LABEL),
            badge_bg: c(t::WIDGET_BADGE_BG),
            badge_label: c(t::WIDGET_BADGE_LABEL),
            bar_track: c(t::WIDGET_BAR_TRACK),
            bar_fill: c(t::WIDGET_BAR_FILL),
            button_corner_radius: f(t::WIDGET_BUTTON_CORNER_RADIUS),
            button_border: c(t::WIDGET_BUTTON_BORDER),
            button_border_width: f(t::WIDGET_BUTTON_BORDER_WIDTH),
            button_shadow_color: c(t::WIDGET_BUTTON_SHADOW_COLOR),
            button_shadow_blur: f(t::WIDGET_BUTTON_SHADOW_BLUR),
            button_shadow_offset_y: f(t::WIDGET_BUTTON_SHADOW_OFFSET_Y),
        }
    }
}

/// Measure an element's intrinsic size without spawning entities. Used
/// by stack layout (hstack pre-measure for alignment) to decide x/y of
/// next sibling.
pub fn measure(el: &Element, metrics: &PaneFontMetrics) -> Vec2 {
    match el {
        Element::Text { value, size, .. } => {
            let s = size.unwrap_or(DEFAULT_FONT_SIZE);
            Vec2::new(metrics.measure(value, s), s * LINE_HEIGHT_MUL)
        }
        Element::Vstack {
            gap,
            pad,
            children,
            ..
        } => measure_stack(children, *gap, *pad, true, metrics),
        Element::Hstack {
            gap,
            pad,
            children,
            ..
        } => measure_stack(children, *gap, *pad, false, metrics),
        Element::Frame {
            gap,
            pad,
            children,
            ..
        } => measure_stack(children, *gap, *pad, true, metrics),
        Element::Scroll {
            gap,
            pad,
            children,
        } => measure_stack(children, *gap, *pad, true, metrics),
        Element::Divider => Vec2::new(0.0, 1.0),
        Element::Spacer { size } => Vec2::new(*size, *size),
        Element::Badge { value, .. } => Vec2::new(
            metrics.measure(value, BADGE_FONT_SIZE) + BADGE_PAD_X * 2.0,
            BADGE_FONT_SIZE * LINE_HEIGHT_MUL + BADGE_PAD_Y * 2.0,
        ),
        Element::Button { label, .. } => Vec2::new(
            metrics.measure(label, DEFAULT_FONT_SIZE) + BUTTON_PAD_X * 2.0,
            DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL + BUTTON_PAD_Y * 2.0,
        ),
        Element::Link { label, .. } => Vec2::new(
            metrics.measure(label, DEFAULT_FONT_SIZE),
            DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL,
        ),
        Element::Bar { width, height, .. } => Vec2::new(*width, *height),
        Element::Swatch { size, .. } => Vec2::new(*size, *size),
        Element::SwatchButton { size, .. } => Vec2::new(*size, *size),
        Element::Tabs { items, .. } => {
            // Pre-measure each tab as bold-default text + tab padding.
            let h = DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL + 12.0;
            let w: f32 = items
                .iter()
                .map(|t| metrics.measure(&t.label, DEFAULT_FONT_SIZE) + 20.0)
                .sum::<f32>()
                + ((items.len().saturating_sub(1)) as f32) * 4.0;
            Vec2::new(w, h)
        }
        Element::Toggle { label, .. } => {
            let label_w = if label.is_empty() {
                0.0
            } else {
                metrics.measure(label, DEFAULT_FONT_SIZE) + 8.0
            };
            Vec2::new(label_w + TOGGLE_TRACK_W, TOGGLE_TRACK_H.max(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL))
        }
        Element::ListItem {
            gap, pad, children, ..
        } => measure_stack(children, *gap, *pad, true, metrics),
        Element::Input { width, .. } => Vec2::new(*width, INPUT_HEIGHT),
        // Canvas is handled by a separate render path in
        // `widget_bevy::render_canvas_items` and never reaches the
        // flow-layout `render`. Return zero so any accidental nesting
        // doesn't contribute to its parent's measured size.
        Element::Canvas { .. } => Vec2::ZERO,
    }
}

fn measure_stack(
    children: &[Element],
    gap: f32,
    pad: f32,
    vertical: bool,
    metrics: &PaneFontMetrics,
) -> Vec2 {
    let mut main: f32 = 0.0;
    let mut cross: f32 = 0.0;
    for (i, c) in children.iter().enumerate() {
        let cs = measure(c, metrics);
        if vertical {
            cross = cross.max(cs.x);
            main += cs.y;
        } else {
            cross = cross.max(cs.y);
            main += cs.x;
        }
        if i + 1 < children.len() {
            main += gap;
        }
    }
    if vertical {
        Vec2::new(cross + pad * 2.0, main + pad * 2.0)
    } else {
        Vec2::new(main + pad * 2.0, cross + pad * 2.0)
    }
}

/// Render `el` at `origin` (pixels-from-content-top-left, y-down).
/// Returns the consumed size.
pub fn render(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    el: &Element,
    origin: Vec2,
    max_w: f32,
    z: f32,
) -> Vec2 {
    // Measure container size first so background/border/shadow have
    // dimensions to paint. For non-style containers this is the same
    // as before; for elements with `style` we use the size to render
    // a Style background under the children.
    let measured = match el {
        Element::Vstack { style, .. }
        | Element::Hstack { style, .. }
        | Element::Frame { style, .. }
        | Element::ListItem { style, .. } => {
            let outer = measure(el, &ctx.metrics);
            paint_style_background(commands, ctx, style.as_ref(), origin, outer, z);
            outer
        }
        _ => Vec2::ZERO,
    };
    let _ = measured;

    match el {
        Element::Vstack {
            gap,
            pad,
            children,
            ..
        } => render_stack(commands, ctx, targets, children, origin, *gap, *pad, true, max_w, z),
        Element::Frame {
            gap,
            pad,
            children,
            ..
        } => render_stack(commands, ctx, targets, children, origin, *gap, *pad, true, max_w, z),
        Element::Scroll {
            gap,
            pad,
            children,
        } => render_stack(commands, ctx, targets, children, origin, *gap, *pad, true, max_w, z),
        Element::Hstack {
            gap,
            pad,
            children,
            align,
            ..
        } => render_hstack(
            commands, ctx, targets, children, origin, *gap, *pad, *align, max_w, z,
        ),
        Element::Text {
            value,
            color,
            size,
            weight,
            family,
        } => render_text(
            commands,
            ctx,
            value,
            color.as_deref(),
            *size,
            *weight,
            family.as_deref(),
            origin,
            max_w,
            z,
        ),
        Element::Divider => render_divider(commands, ctx, origin, max_w, z),
        Element::Spacer { size } => Vec2::new(*size, *size),
        Element::Badge { value, color, .. } => {
            render_badge(commands, ctx, value, color.as_deref(), origin, z)
        }
        Element::Button { id, label, kind, style } => render_button(
            commands, ctx, targets, id, label, *kind, style.as_ref(), origin, z,
        ),
        Element::Link { url, label } => render_link(commands, ctx, targets, url, label, origin, z),
        Element::Bar {
            value,
            max,
            color,
            track,
            width,
            height,
        } => render_bar(
            commands,
            ctx,
            *value,
            *max,
            color.as_deref(),
            track.as_deref(),
            *width,
            *height,
            origin,
            z,
        ),
        Element::Swatch { color, size, id } => render_swatch(
            commands, ctx, targets, color, *size, id.as_deref(), origin, z,
        ),
        Element::SwatchButton { id, color, size } => render_swatch(
            commands, ctx, targets, color, *size, Some(id), origin, z,
        ),
        Element::Tabs {
            id,
            items,
            selected,
            ..
        } => render_tabs(commands, ctx, targets, id, items, selected, origin, z),
        Element::Toggle {
            id,
            label,
            checked,
            ..
        } => render_toggle(commands, ctx, targets, id, label, *checked, origin, z),
        Element::ListItem {
            id,
            children,
            gap,
            pad,
            selected,
            ..
        } => render_list_item(
            commands, ctx, targets, id, children, *gap, *pad, *selected, origin, max_w, z,
        ),
        Element::Input {
            id,
            value,
            placeholder,
            focused,
            width,
            ..
        } => render_input(
            commands, ctx, targets, id, value, placeholder, *focused, *width, origin, z,
        ),
        // Canvas only renders at the top level — see
        // `widget_bevy::render_canvas_items`. If we hit it nested
        // inside a stack, silently drop it (zero size, no entities)
        // rather than panic.
        Element::Canvas { .. } => Vec2::ZERO,
    }
}

#[allow(clippy::too_many_arguments)]
fn render_swatch(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    color_str: &str,
    size: f32,
    id: Option<&str>,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let color = parse_color_or_default(color_str, ctx.palette.button_bg);
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color,
            custom_size: Some(Vec2::new(size, size)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    if let Some(id) = id {
        targets.clicks.push(ClickTarget {
            id: id.to_string(),
            kind: ClickKind::Button,
            rect: Rect::new(origin.x, origin.y, origin.x + size, origin.y + size),
        });
    }
    Vec2::new(size, size)
}

fn parse_color_or_default(s: &str, fallback: Color) -> Color {
    // Accept the same forms as the theme parser: hex / oklch / oklab / rgb.
    style_bevy::theme::parse_color_string(s)
        .map(Color::LinearRgba)
        .unwrap_or(fallback)
}

#[allow(clippy::too_many_arguments)]
fn render_stack(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    children: &[Element],
    origin: Vec2,
    gap: f32,
    pad: f32,
    vertical: bool,
    max_w: f32,
    z: f32,
) -> Vec2 {
    let inner_x = origin.x + pad;
    let inner_y = origin.y + pad;
    let inner_max = (max_w - pad * 2.0).max(0.0);

    let mut cursor = if vertical { inner_y } else { inner_x };
    let mut cross: f32 = 0.0;
    for (i, c) in children.iter().enumerate() {
        let cs = if vertical {
            render(
                commands,
                ctx,
                targets,
                c,
                Vec2::new(inner_x, cursor),
                inner_max,
                z + 0.01,
            )
        } else {
            render(
                commands,
                ctx,
                targets,
                c,
                Vec2::new(cursor, inner_y),
                inner_max,
                z + 0.01,
            )
        };
        if vertical {
            cursor += cs.y;
            cross = cross.max(cs.x);
        } else {
            cursor += cs.x;
            cross = cross.max(cs.y);
        }
        if i + 1 < children.len() {
            cursor += gap;
        }
    }
    let main = cursor - if vertical { inner_y } else { inner_x };
    if vertical {
        Vec2::new(cross + pad * 2.0, main + pad * 2.0)
    } else {
        Vec2::new(main + pad * 2.0, cross + pad * 2.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn render_hstack(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    children: &[Element],
    origin: Vec2,
    gap: f32,
    pad: f32,
    align: Align,
    max_w: f32,
    z: f32,
) -> Vec2 {
    let inner_x = origin.x + pad;
    let inner_y = origin.y + pad;
    let inner_max = (max_w - pad * 2.0).max(0.0);

    // Pre-measure children for cross-axis alignment.
    let sizes: Vec<Vec2> = children.iter().map(|c| measure(c, &ctx.metrics)).collect();
    let row_h = sizes.iter().map(|s| s.y).fold(0.0_f32, f32::max);

    let mut cursor = inner_x;
    for (i, (c, cs)) in children.iter().zip(sizes.iter()).enumerate() {
        let dy = match align {
            Align::Start => 0.0,
            Align::Center => (row_h - cs.y) * 0.5,
            Align::End => row_h - cs.y,
        };
        let _ = render(
            commands,
            ctx,
            targets,
            c,
            Vec2::new(cursor, inner_y + dy),
            (inner_max - (cursor - inner_x)).max(0.0),
            z + 0.01,
        );
        cursor += cs.x;
        if i + 1 < children.len() {
            cursor += gap;
        }
    }
    Vec2::new(cursor - inner_x + pad * 2.0, row_h + pad * 2.0)
}

#[allow(clippy::too_many_arguments)]
fn render_text(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    value: &str,
    color: Option<&str>,
    size: Option<f32>,
    weight: Option<Weight>,
    family: Option<&str>,
    origin: Vec2,
    max_w: f32,
    z: f32,
) -> Vec2 {
    let s = size.unwrap_or(DEFAULT_FONT_SIZE);
    let col = color
        .and_then(|c| ctx.resolve_color(c))
        .unwrap_or(ctx.palette.text);
    let col = match weight {
        Some(Weight::Bold) => brighten(col, 0.08),
        _ => col,
    };
    let font = family
        .and_then(|f| ctx.font_for(f))
        .unwrap_or_else(|| ctx.font.clone());
    let (display, w, h) = fit_single_line(value, s, max_w, &ctx.metrics);
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(display),
        TextFont {
            font,
            font_size: s,
            ..default()
        },
        LineHeight::Px(h),
        TextColor(col),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        TextBounds {
            width: Some(max_w.max(0.0)),
            height: Some(h),
        },
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    Vec2::new(w, h)
}

/// Truncate `value` to fit in `max_w` pixels using the host-supplied
/// `PaneFontMetrics`. Returns the display string (with `…` appended if
/// truncated), the actual width consumed, and the single line height.
///
/// This is the single source of truth for "how wide is this text?" —
/// callers that need both the rendered string and its size should go
/// through here so the renderer's intrinsic measure matches what we
/// actually draw on screen. Uses real per-character advance from the
/// host font instead of approximating, so the layout stays correct as
/// the pane resizes.
pub fn fit_single_line(
    value: &str,
    font_size: f32,
    max_w: f32,
    metrics: &PaneFontMetrics,
) -> (String, f32, f32) {
    let h = font_size * LINE_HEIGHT_MUL;
    let intrinsic_w = metrics.measure(value, font_size);
    if intrinsic_w <= max_w || max_w <= 0.0 {
        return (value.to_string(), intrinsic_w.min(max_w.max(0.0)), h);
    }
    let char_w = metrics.char_width(font_size);
    if char_w <= 0.0 {
        return (String::new(), 0.0, h);
    }
    let ellipsis_w = char_w; // '…' is one char wide on monospace
    let avail = (max_w - ellipsis_w).max(0.0);
    let max_chars = (avail / char_w).floor() as usize;
    let mut s: String = value.chars().take(max_chars).collect();
    s.push('…');
    let w = (max_chars as f32 + 1.0) * char_w;
    (s, w.min(max_w), h)
}

fn render_divider(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    origin: Vec2,
    max_w: f32,
    z: f32,
) -> Vec2 {
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: ctx.palette.divider,
            custom_size: Some(Vec2::new(max_w.max(0.0), 1.0)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    Vec2::new(max_w.max(0.0), 1.0)
}

fn render_badge(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    value: &str,
    color: Option<&str>,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let bg = color
        .and_then(parse_hex_color)
        .map(|[r, g, b]| Color::srgb(r, g, b))
        .unwrap_or(ctx.palette.badge_bg);
    let text_w = ctx.metrics.measure(value, BADGE_FONT_SIZE);
    let w = text_w + BADGE_PAD_X * 2.0;
    let h = BADGE_FONT_SIZE * LINE_HEIGHT_MUL + BADGE_PAD_Y * 2.0;
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: bg,
            custom_size: Some(Vec2::new(w, h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(value.to_string()),
        TextFont {
            font: ctx.font.clone(),
            font_size: BADGE_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(BADGE_FONT_SIZE * LINE_HEIGHT_MUL),
        TextColor(ctx.palette.badge_label),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + BADGE_PAD_X, -(origin.y + BADGE_PAD_Y), z + 0.01),
    ));
    Vec2::new(w, h)
}

#[allow(clippy::too_many_arguments)]
fn render_button(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    label: &str,
    kind: ButtonKind,
    style: Option<&Style>,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let text_w = ctx.metrics.measure(label, DEFAULT_FONT_SIZE);
    let w = text_w + BUTTON_PAD_X * 2.0;
    let h = DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL + BUTTON_PAD_Y * 2.0;

    // Resolve visual style. Filled = palette button_bg + label. Outline
    // = transparent bg + accent border + accent label. Ghost = fully
    // transparent + accent label (no shadow either). Style overrides
    // win over the kind defaults.
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let (default_bg, default_border, default_border_w, default_label, draw_shadow) = match kind {
        ButtonKind::Filled => (
            ctx.palette.button_bg,
            ctx.palette.button_border,
            ctx.palette.button_border_width,
            ctx.palette.button_label,
            true,
        ),
        ButtonKind::Outline => (
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            accent,
            1.5,
            accent,
            false,
        ),
        ButtonKind::Ghost => (
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            0.0,
            accent,
            false,
        ),
    };

    let bg_color = style
        .and_then(|s| s.background.as_deref())
        .and_then(|c| ctx.resolve_color(c))
        .unwrap_or(default_bg);
    let border_color = style
        .and_then(|s| s.border.as_ref())
        .and_then(|b| ctx.resolve_color(&b.color))
        .unwrap_or(default_border);
    let border_w = style
        .and_then(|s| s.border.as_ref().map(|b| b.width))
        .unwrap_or(default_border_w);
    let radius = style
        .and_then(|s| s.radius.as_deref())
        .and_then(|r| ctx.resolve_f32(r))
        .unwrap_or(ctx.palette.button_corner_radius);
    let (shadow_color, shadow_blur, shadow_offset_y) = if let Some(sh) = style.and_then(|s| s.shadow.as_ref()) {
        ctx.resolve_shadow(sh)
    } else if draw_shadow {
        (
            ctx.palette.button_shadow_color,
            ctx.palette.button_shadow_blur,
            ctx.palette.button_shadow_offset_y,
        )
    } else {
        (Color::srgba(0.0, 0.0, 0.0, 0.0), 0.0, 0.0)
    };

    paint_rounded_panel(
        commands,
        ctx,
        origin,
        Vec2::new(w, h),
        radius,
        bg_color,
        border_color,
        border_w,
        shadow_color,
        shadow_blur,
        shadow_offset_y,
        z,
    );

    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(label.to_string()),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL),
        TextColor(default_label),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + BUTTON_PAD_X, -(origin.y + BUTTON_PAD_Y), z + 0.01),
    ));
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::Button,
        rect: Rect::new(origin.x, origin.y, origin.x + w, origin.y + h),
    });
    Vec2::new(w, h)
}

/// Spawn a rounded-rect SDF panel with optional border + drop shadow.
/// Shared between Button, Frame, ListItem, Toggle, Input, Tabs.
#[allow(clippy::too_many_arguments)]
fn paint_rounded_panel(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    origin: Vec2,
    size: Vec2,
    corner_radius: f32,
    bg: Color,
    border_color: Color,
    border_width: f32,
    shadow_color: Color,
    shadow_blur: f32,
    shadow_offset_y: f32,
    z: f32,
) {
    use crate::button_material::{ButtonParams, WidgetButtonMaterial, WidgetButtonMesh};

    if size.x <= 0.0 || size.y <= 0.0 {
        return;
    }
    // Clamp pill radius to half the shorter side.
    let radius = corner_radius.min(size.x * 0.5).min(size.y * 0.5).max(0.0);
    let mesh_w = size.x + 2.0 * shadow_blur;
    let mesh_h = size.y + 2.0 * shadow_blur;
    let params = ButtonParams {
        mesh_size: Vec2::new(mesh_w, mesh_h),
        button_size: size,
        corner_radius: radius,
        border_width,
        bg: lin_vec4(bg),
        border: lin_vec4(border_color),
        shadow_color: lin_vec4(shadow_color),
        shadow_blur,
        shadow_offset_y,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let entity = commands
        .spawn((
            ChildOf(ctx.content_root),
            Transform::from_xyz(
                origin.x + size.x * 0.5,
                -(origin.y + size.y * 0.5),
                z,
            )
            .with_scale(Vec3::new(mesh_w, mesh_h, 1.0)),
            Visibility::Inherited,
        ))
        .id();
    commands.queue(move |world: &mut World| {
        let mesh = match world.get_resource::<WidgetButtonMesh>() {
            Some(m) => m.0.clone(),
            None => return,
        };
        let mat = world
            .resource_mut::<Assets<WidgetButtonMaterial>>()
            .add(WidgetButtonMaterial { params });
        if let Ok(mut ec) = world.get_entity_mut(entity) {
            ec.insert((
                bevy::mesh::Mesh2d(mesh),
                bevy::sprite_render::MeshMaterial2d(mat),
            ));
        }
    });
}

/// Paint the `style.background` (color + image) and border/shadow for
/// a Vstack/Hstack/Frame/ListItem behind its children. Has no effect
/// when `style` is None.
pub fn paint_style_background(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    style: Option<&Style>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let Some(style) = style else { return };
    if size.x <= 0.0 || size.y <= 0.0 {
        return;
    }
    let bg = style
        .background
        .as_deref()
        .and_then(|c| ctx.resolve_color(c));
    let radius = style
        .radius
        .as_deref()
        .and_then(|r| ctx.resolve_f32(r))
        .unwrap_or(0.0);
    let (border_color, border_w) = match style.border.as_ref() {
        Some(b) => (
            ctx.resolve_color(&b.color)
                .unwrap_or(Color::srgba(0.0, 0.0, 0.0, 0.0)),
            b.width,
        ),
        None => (Color::srgba(0.0, 0.0, 0.0, 0.0), 0.0),
    };
    let (shadow_color, shadow_blur, shadow_offset_y) = match style.shadow.as_ref() {
        Some(sh) => ctx.resolve_shadow(sh),
        None => (Color::srgba(0.0, 0.0, 0.0, 0.0), 0.0, 0.0),
    };
    // Nothing to paint if everything is transparent and there's no image.
    let has_panel = bg.is_some()
        || border_w > 0.0
        || shadow_color.to_srgba().alpha > 0.0;
    if has_panel {
        paint_rounded_panel(
            commands,
            ctx,
            origin,
            size,
            radius,
            bg.unwrap_or(Color::srgba(0.0, 0.0, 0.0, 0.0)),
            border_color,
            border_w,
            shadow_color,
            shadow_blur,
            shadow_offset_y,
            z - 0.005,
        );
    }
    // Background image — Bevy image asset by path, stretched to fit.
    // Note: the actual image load is deferred to the rerender system
    // via WidgetImageCache. Here we record a sprite placeholder; the
    // image will swap in when loaded. The host code already plumbs
    // image loading via `widget_bevy::render_canvas_items`, so we
    // queue a similar load.
    if let Some(path) = style.background_image.as_deref() {
        let path = path.to_string();
        let entity = commands
            .spawn((
                ChildOf(ctx.content_root),
                Sprite {
                    custom_size: Some(size),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(origin.x, -origin.y, z - 0.003),
            ))
            .id();
        commands.queue(move |world: &mut World| {
            let server = world.resource::<AssetServer>().clone();
            let handle = server.load::<Image>(path);
            if let Ok(mut ec) = world.get_entity_mut(entity) {
                if let Some(mut sp) = ec.get_mut::<Sprite>() {
                    sp.image = handle;
                }
            }
        });
    }
}

/// Convert a `Color` to its linear-RGBA `Vec4` for shader uniforms.
fn lin_vec4(c: Color) -> Vec4 {
    let lin = c.to_linear();
    Vec4::new(lin.red, lin.green, lin.blue, lin.alpha)
}

fn render_link(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    url: &str,
    label: &str,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let w = ctx.metrics.measure(label, DEFAULT_FONT_SIZE);
    let h = DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL;
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(label.to_string()),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(h),
        TextColor(ctx.palette.link),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    targets.links.push(LinkTarget {
        url: url.to_string(),
        rect: Rect::new(origin.x, origin.y, origin.x + w, origin.y + h),
    });
    Vec2::new(w, h)
}

#[allow(clippy::too_many_arguments)]
fn render_bar(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    value: f32,
    max: f32,
    color: Option<&str>,
    track: Option<&str>,
    width: f32,
    height: f32,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let fill = color
        .and_then(parse_hex_color)
        .map(|[r, g, b]| Color::srgb(r, g, b))
        .unwrap_or(ctx.palette.bar_fill);
    let bg = track
        .and_then(parse_hex_color)
        .map(|[r, g, b]| Color::srgb(r, g, b))
        .unwrap_or(ctx.palette.bar_track);
    let ratio = if max <= 0.0 {
        0.0
    } else {
        (value / max).clamp(0.0, 1.0)
    };
    let fill_w = (width * ratio).max(0.0);

    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: bg,
            custom_size: Some(Vec2::new(width, height)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    if fill_w > 0.0 {
        commands.spawn((
            ChildOf(ctx.content_root),
            Sprite {
                color: fill,
                custom_size: Some(Vec2::new(fill_w, height)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(origin.x, -origin.y, z + 0.01),
        ));
    }
    Vec2::new(width, height)
}

fn brighten(c: Color, delta: f32) -> Color {
    let s = c.to_srgba();
    Color::srgb(
        (s.red + delta).clamp(0.0, 1.0),
        (s.green + delta).clamp(0.0, 1.0),
        (s.blue + delta).clamp(0.0, 1.0),
    )
}

fn render_tabs(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    items: &[TabItem],
    selected: &str,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let row_h = DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL + TAB_PAD_Y * 2.0;
    let mut cursor = origin.x;
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    for (i, tab) in items.iter().enumerate() {
        let label_w = ctx.metrics.measure(&tab.label, DEFAULT_FONT_SIZE);
        let tab_w = label_w + TAB_PAD_X * 2.0;
        let is_selected = tab.id == selected;
        let label_color = if is_selected {
            accent
        } else {
            ctx.palette.text_muted
        };
        commands.spawn((
            ChildOf(ctx.content_root),
            Text2d::new(tab.label.clone()),
            TextFont {
                font: ctx.font.clone(),
                font_size: DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL),
            TextColor(label_color),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(cursor + TAB_PAD_X, -(origin.y + TAB_PAD_Y), z + 0.01),
        ));
        if is_selected {
            commands.spawn((
                ChildOf(ctx.content_root),
                Sprite {
                    color: accent,
                    custom_size: Some(Vec2::new(tab_w, TAB_INDICATOR_H)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(
                    cursor,
                    -(origin.y + row_h - TAB_INDICATOR_H),
                    z + 0.02,
                ),
            ));
        }
        targets.clicks.push(ClickTarget {
            id: id.to_string(),
            kind: ClickKind::TabSelect { tab: tab.id.clone() },
            rect: Rect::new(cursor, origin.y, cursor + tab_w, origin.y + row_h),
        });
        cursor += tab_w;
        if i + 1 < items.len() {
            cursor += TAB_GAP;
        }
    }
    Vec2::new(cursor - origin.x, row_h)
}

fn render_toggle(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    label: &str,
    checked: bool,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    // Optional label sits to the left of the track.
    let mut cursor_x = origin.x;
    let row_h = TOGGLE_TRACK_H.max(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL);
    let label_offset_y = (row_h - DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL) * 0.5;
    if !label.is_empty() {
        let lw = ctx.metrics.measure(label, DEFAULT_FONT_SIZE);
        commands.spawn((
            ChildOf(ctx.content_root),
            Text2d::new(label.to_string()),
            TextFont {
                font: ctx.font.clone(),
                font_size: DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL),
            TextColor(ctx.palette.text),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(cursor_x, -(origin.y + label_offset_y), z + 0.01),
        ));
        cursor_x += lw + 8.0;
    }

    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let track_origin = Vec2::new(cursor_x, origin.y + (row_h - TOGGLE_TRACK_H) * 0.5);
    let track_color = if checked {
        accent
    } else {
        ctx.palette.bar_track
    };
    paint_rounded_panel(
        commands,
        ctx,
        track_origin,
        Vec2::new(TOGGLE_TRACK_W, TOGGLE_TRACK_H),
        TOGGLE_TRACK_H * 0.5,
        track_color,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        0.0,
        z,
    );
    // Knob.
    let knob_d = TOGGLE_TRACK_H - TOGGLE_KNOB_PAD * 2.0;
    let knob_x = if checked {
        track_origin.x + TOGGLE_TRACK_W - knob_d - TOGGLE_KNOB_PAD
    } else {
        track_origin.x + TOGGLE_KNOB_PAD
    };
    paint_rounded_panel(
        commands,
        ctx,
        Vec2::new(knob_x, track_origin.y + TOGGLE_KNOB_PAD),
        Vec2::new(knob_d, knob_d),
        knob_d * 0.5,
        Color::WHITE,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        Color::srgba(0.0, 0.0, 0.0, 0.2),
        2.0,
        1.0,
        z + 0.01,
    );

    let total_w = cursor_x + TOGGLE_TRACK_W - origin.x;
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::Toggle { new_checked: !checked },
        rect: Rect::new(origin.x, origin.y, origin.x + total_w, origin.y + row_h),
    });
    Vec2::new(total_w, row_h)
}

#[allow(clippy::too_many_arguments)]
fn render_list_item(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    children: &[Element],
    gap: f32,
    pad: f32,
    selected: bool,
    origin: Vec2,
    max_w: f32,
    z: f32,
) -> Vec2 {
    // Selected items get a subtle accent fill behind the children.
    // Style overrides applied in the outer `render` block already drew
    // the user-supplied background; we layer the selected highlight
    // *above* that so the indicator wins.
    let inner = measure_stack(children, gap, pad, true, &ctx.metrics);
    let panel_size = Vec2::new(inner.x.max(max_w * 0.95), inner.y);
    if selected {
        let accent = ctx
            .resolve_color("accent")
            .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
        let sel_bg = accent.with_alpha(0.18);
        paint_rounded_panel(
            commands,
            ctx,
            origin,
            panel_size,
            ctx
                .resolve_f32("radius_sm")
                .unwrap_or(4.0),
            sel_bg,
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            0.0,
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            0.0,
            0.0,
            z + 0.001,
        );
    }
    let consumed = render_stack(
        commands,
        ctx,
        targets,
        children,
        origin,
        gap,
        pad,
        true,
        max_w,
        z + 0.002,
    );
    let hit_w = consumed.x.max(panel_size.x);
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::Button,
        rect: Rect::new(origin.x, origin.y, origin.x + hit_w, origin.y + consumed.y),
    });
    consumed
}

#[allow(clippy::too_many_arguments)]
fn render_input(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    value: &str,
    placeholder: &str,
    focused_attr: bool,
    width: f32,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    // Host owns the editing state while an input is focused — so when
    // ctx.focused_input matches our id we render from that buffer
    // instead of the widget's stale value.
    let focus_match = ctx
        .focused_input
        .as_ref()
        .filter(|f| f.id == id);
    let is_focused = focus_match.is_some() || focused_attr;
    let display_value = match focus_match {
        Some(f) => f.value.clone(),
        None => value.to_string(),
    };
    let caret_pos = focus_match.map(|f| f.caret).unwrap_or(0);
    let show_caret = focus_match.is_some() && ctx.caret_visible;

    let radius = ctx.resolve_f32("radius_sm").unwrap_or(4.0);
    let bg = ctx
        .resolve_color("input_bg")
        .unwrap_or(Color::srgb(0.10, 0.11, 0.13));
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let border_color = if is_focused {
        accent
    } else {
        ctx.palette.divider
    };
    paint_rounded_panel(
        commands,
        ctx,
        origin,
        Vec2::new(width, INPUT_HEIGHT),
        radius,
        bg,
        border_color,
        1.0,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        0.0,
        z,
    );

    // Text or placeholder.
    let (txt, txt_color) = if display_value.is_empty() && !is_focused {
        (placeholder.to_string(), ctx.palette.text_muted)
    } else if display_value.is_empty() {
        (String::new(), ctx.palette.text)
    } else {
        (display_value.clone(), ctx.palette.text)
    };
    let text_y = (INPUT_HEIGHT - DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL) * 0.5;
    if !txt.is_empty() {
        commands.spawn((
            ChildOf(ctx.content_root),
            Text2d::new(txt),
            TextFont {
                font: ctx.font.clone(),
                font_size: DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL),
            TextColor(txt_color),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(origin.x + INPUT_PAD_X, -(origin.y + text_y), z + 0.01),
        ));
    }
    // Caret.
    if show_caret {
        // Width of value up to caret position, in chars.
        let prefix: String = display_value.chars().take(caret_pos).collect();
        let caret_x = origin.x + INPUT_PAD_X + ctx.metrics.measure(&prefix, DEFAULT_FONT_SIZE);
        let caret_h = DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL;
        commands.spawn((
            ChildOf(ctx.content_root),
            Sprite {
                color: accent,
                custom_size: Some(Vec2::new(1.5, caret_h)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(caret_x, -(origin.y + text_y), z + 0.02),
        ));
    }

    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::InputFocus,
        rect: Rect::new(origin.x, origin.y, origin.x + width, origin.y + INPUT_HEIGHT),
    });
    Vec2::new(width, INPUT_HEIGHT)
}
