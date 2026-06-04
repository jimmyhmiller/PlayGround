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

use crate::protocol::{parse_hex_color, Align, ButtonKind, Edges, Element, Style, TabItem, Weight};
use crate::{ClickKind, ClickTarget, LinkTarget, TextSpan, WidgetTargets};

pub const DEFAULT_FONT_SIZE: f32 = 13.0;
pub const LINE_HEIGHT_MUL: f32 = 1.4;
/// Threshold above which text is considered a display heading. Display
/// type uses a tighter line height because the natural 1.4-multiplier
/// "phone-book" line spacing leaves an obvious empty band above the
/// glyph at 36px+ sizes.
const DISPLAY_FONT_THRESHOLD: f32 = 28.0;
const DISPLAY_LINE_HEIGHT_MUL: f32 = 1.05;

// Colors all flow through `LayoutCtx::palette` (theme-driven). See
// `WidgetPalette::from_theme` below.

pub const BUTTON_PAD_X: f32 = 8.0;
pub const BUTTON_PAD_Y: f32 = 4.0;
pub const BADGE_PAD_X: f32 = 6.0;
pub const BADGE_PAD_Y: f32 = 2.0;
pub const BADGE_FONT_SIZE: f32 = 11.0;

// Toggle pill dimensions. Track height = knob diameter; the knob slides
// across the inner length.
pub const TOGGLE_TRACK_W: f32 = 34.0;
pub const TOGGLE_TRACK_H: f32 = 18.0;
pub const TOGGLE_KNOB_PAD: f32 = 2.0;

// Input height = body text + symmetric padding. Width comes from the
// element itself (defaults to 160 px).
pub const INPUT_HEIGHT: f32 = 26.0;
pub const INPUT_PAD_X: f32 = 8.0;

// TextArea: symmetric vertical padding around the text block. Total box
// height = rows * line_height + 2 * TEXTAREA_PAD_Y (see layout.rs).
pub const TEXTAREA_PAD_Y: f32 = 6.0;

// Table cell padding (inside each grid cell).
pub const TABLE_CELL_PAD_X: f32 = 8.0;
pub const TABLE_CELL_PAD_Y: f32 = 5.0;
// Gap between table columns.
pub const TABLE_COL_GAP: f32 = 12.0;

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
    /// ID of the clickable currently under the mouse pointer in this
    /// pane (`None` if no hover, or the hover is over a non-clickable
    /// region). Render fns compare against their element id to pick
    /// hover-state colors.
    pub hovered_click_id: Option<String>,
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
    pub button_bg_hover: Color,
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
            button_bg_hover: c(t::BUTTON_BG_HOVER),
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

/// Resolve a flow container's effective padding. If the Style carries
/// an explicit `padding` (asymmetric), it wins; otherwise the
/// element's symmetric `pad: f32` becomes Edges::all(pad).
pub(crate) fn effective_padding(style: Option<&Style>, pad: f32) -> Edges {
    style
        .and_then(|s| s.padding.as_ref())
        .copied()
        .unwrap_or_else(|| Edges::all(pad))
}

/// Effective per-line height for `font_size`. Display headings use a
/// tighter ratio so big titles don't carry an empty band above the
/// glyph that reads as accidental top margin.
pub fn line_height(font_size: f32) -> f32 {
    let mul = if font_size >= DISPLAY_FONT_THRESHOLD {
        DISPLAY_LINE_HEIGHT_MUL
    } else {
        LINE_HEIGHT_MUL
    };
    font_size * mul
}

/// Render `el` at `origin` (pixels-from-content-top-left, y-down).
/// Returns the consumed size.
///
/// Layout is computed via [`crate::layout`] (Taffy) before any
/// entities are spawned. The render walk then reads each node's
/// computed `(x, y, width, height)` from Taffy rather than recomputing
/// stack positions by hand.
pub fn render(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    el: &Element,
    origin: Vec2,
    max_w: f32,
    z: f32,
) -> Vec2 {
    let mut laid = crate::layout::build_tree(el, &ctx.metrics);
    // Available height for layout. The pane content height is already
    // known (ctx.content_size.y); passing it as the definite available
    // height lets a root opt into filling the pane vertically — `height:
    // "100%"` resolves against it and `flex_grow` distributes real
    // vertical space, symmetric with how `max_w` makes width work.
    // A root with no explicit/flex height still sizes to its content
    // (Taffy uses available space only for percentage/flex resolution),
    // so taller-than-pane content keeps growing and scrolls as before.
    // Fall back to INFINITY when the pane height isn't known yet.
    let avail_h = if ctx.content_size.y.is_finite() && ctx.content_size.y > 0.0 {
        ctx.content_size.y
    } else {
        f32::INFINITY
    };
    crate::layout::compute(&mut laid, max_w, avail_h, &ctx.metrics);
    let root_layout = laid.layout(laid.root);
    let root_origin = origin
        + Vec2::new(root_layout.location.x, root_layout.location.y);
    render_node(
        commands,
        ctx,
        targets,
        &laid,
        laid.root,
        el,
        root_origin,
        z,
    );
    Vec2::new(root_layout.size.width, root_layout.size.height)
}

/// Walk the Taffy tree in lockstep with the Element tree, spawning the
/// appropriate primitives at each node's computed position + size.
#[allow(clippy::too_many_arguments)]
fn render_node(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    laid: &crate::layout::LaidOut,
    node_id: taffy::NodeId,
    el: &Element,
    origin: Vec2,
    z: f32,
) {
    let layout = laid.layout(node_id);
    let size = Vec2::new(layout.size.width, layout.size.height);

    // Container helper: paint the Style background under the children
    // (if any) and recurse, using Taffy's child-node positions for
    // each child rather than computing them ourselves.
    let recurse_children = |commands: &mut Commands,
                            targets: &mut WidgetTargets,
                            children: &[Element],
                            style: Option<&Style>| {
        paint_style_background(commands, ctx, style, origin, size, z);
        let child_ids = laid.taffy.children(node_id).unwrap_or_default();
        for (cid, child) in child_ids.iter().zip(children.iter()) {
            let cl = laid.layout(*cid);
            let cpos = origin + Vec2::new(cl.location.x, cl.location.y);
            render_node(commands, ctx, targets, laid, *cid, child, cpos, z + 0.01);
        }
    };

    match el {
        Element::Vstack { children, style, .. }
        | Element::Hstack { children, style, .. }
        | Element::Frame { children, style, .. } => {
            recurse_children(commands, targets, children, style.as_ref());
        }
        Element::Scroll { children, .. } => {
            recurse_children(commands, targets, children, None);
        }
        Element::ListItem {
            id,
            children,
            selected,
            style,
            ..
        } => {
            if *selected {
                let sel_bg = ctx
                    .resolve_color("surface_3")
                    .unwrap_or(Color::srgb(0.13, 0.14, 0.17));
                paint_rounded_panel(
                    commands,
                    ctx,
                    origin,
                    size,
                    ctx.resolve_f32("radius_sm").unwrap_or(4.0),
                    sel_bg,
                    Color::srgba(0.0, 0.0, 0.0, 0.0),
                    0.0,
                    Color::srgba(0.0, 0.0, 0.0, 0.0),
                    0.0,
                    0.0,
                    z + 0.001,
                );
                let accent = ctx
                    .resolve_color("accent")
                    .unwrap_or(Color::srgb(0.79, 0.66, 0.42));
                commands.spawn((
                    ChildOf(ctx.content_root),
                    Sprite {
                        color: accent,
                        custom_size: Some(Vec2::new(2.0, size.y)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(origin.x, -origin.y, z + 0.003),
                ));
            }
            recurse_children(commands, targets, children, style.as_ref());
            targets.clicks.push(ClickTarget {
                id: id.clone(),
                kind: ClickKind::Button,
                rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
            });
        }
        Element::Text {
            value,
            color,
            size: font_size,
            weight,
            family,
            selectable,
        } => render_text_at(
            commands,
            ctx,
            targets,
            value,
            color.as_deref(),
            font_size.unwrap_or(DEFAULT_FONT_SIZE),
            *weight,
            family.as_deref(),
            *selectable,
            origin,
            size,
            z,
        ),
        Element::Divider => render_divider_at(commands, ctx, origin, size, z),
        Element::Spacer { .. } => {}
        Element::Badge {
            value,
            color,
            selectable,
            ..
        } => {
            render_badge_at(
                commands,
                ctx,
                targets,
                value,
                color.as_deref(),
                *selectable,
                origin,
                size,
                z,
            );
        }
        Element::Button {
            id,
            label,
            kind,
            style,
        } => render_button_at(
            commands,
            ctx,
            targets,
            id,
            label,
            *kind,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Link { url, label } => {
            render_link_at(commands, ctx, targets, url, label, origin, size, z);
        }
        Element::Bar {
            value,
            max,
            color,
            track,
            ..
        } => render_bar_at(
            commands,
            ctx,
            *value,
            *max,
            color.as_deref(),
            track.as_deref(),
            origin,
            size,
            z,
        ),
        Element::Swatch { color, id, .. } => render_swatch_at(
            commands,
            ctx,
            targets,
            color,
            id.as_deref(),
            origin,
            size,
            z,
        ),
        Element::SwatchButton { id, color, .. } => render_swatch_at(
            commands,
            ctx,
            targets,
            color,
            Some(id),
            origin,
            size,
            z,
        ),
        Element::Tabs {
            id,
            items,
            selected,
            ..
        } => render_tabs_at(commands, ctx, targets, laid, node_id, id, items, selected, origin, z),
        Element::Toggle {
            id,
            label,
            checked,
            ..
        } => render_toggle_at(commands, ctx, targets, laid, node_id, id, label, *checked, origin, size, z),
        Element::Input {
            id,
            value,
            placeholder,
            focused,
            ..
        } => render_input_at(
            commands, ctx, targets, id, value, placeholder, *focused, origin, size, z,
        ),
        Element::TextArea {
            id,
            value,
            placeholder,
            focused,
            ..
        } => render_textarea_at(
            commands, ctx, targets, id, value, placeholder, *focused, origin, size, z,
        ),
        Element::Table {
            columns,
            rows,
            zebra,
            selectable,
            ..
        } => render_table_at(
            commands, ctx, targets, laid, node_id, columns, rows, *zebra, *selectable, origin,
            size, z,
        ),
        // Canvas renders only at the top level via render_canvas_items.
        // Nested Canvas inside flow layout becomes a 0-size leaf.
        Element::Canvas { .. } => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn render_swatch_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    color_str: &str,
    id: Option<&str>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let color = parse_color_or_default(color_str, ctx.palette.button_bg);
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color,
            custom_size: Some(size),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    if let Some(id) = id {
        targets.clicks.push(ClickTarget {
            id: id.to_string(),
            kind: ClickKind::Button,
            rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
        });
    }
}

fn parse_color_or_default(s: &str, fallback: Color) -> Color {
    // Accept the same forms as the theme parser: hex / oklch / oklab / rgb.
    style_bevy::theme::parse_color_string(s)
        .map(Color::LinearRgba)
        .unwrap_or(fallback)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn render_text_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    value: &str,
    color: Option<&str>,
    font_size: f32,
    weight: Option<Weight>,
    family: Option<&str>,
    selectable: bool,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
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
    // Multi-line via Bevy's text layout: let it wrap inside the
    // TextBounds box. font_size is passed in (not inferred from
    // size.y — that would treat multi-line height as line-height
    // and over-scale the glyphs).
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(value.to_string()),
        TextFont {
            font,
            font_size,
            ..default()
        },
        LineHeight::Px(line_height(font_size)),
        TextColor(col),
        Anchor::TOP_LEFT,
        TextBounds {
            width: Some(size.x.max(0.0)),
            height: Some(size.y.max(0.0)),
        },
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    // Drag-select: register this run as a selectable text span. Char
    // offsets are measured left-to-right from `origin`, so use the
    // glyph-origin rect (one line tall for the common single-line value).
    if selectable && !value.is_empty() {
        let line_h = line_height(font_size);
        targets.spans.push(TextSpan {
            text: value.to_string(),
            rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + line_h),
            font_size,
        });
    }
}

fn render_divider_at(commands: &mut Commands, ctx: &LayoutCtx, origin: Vec2, size: Vec2, z: f32) {
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: ctx.palette.divider,
            custom_size: Some(Vec2::new(size.x.max(0.0), size.y.max(1.0))),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn render_badge_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    value: &str,
    color: Option<&str>,
    selectable: bool,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let bg = color
        .and_then(parse_hex_color)
        .map(|[r, g, b]| Color::srgb(r, g, b))
        .unwrap_or(ctx.palette.badge_bg);
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: bg,
            custom_size: Some(size),
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
        LineHeight::Px(line_height(BADGE_FONT_SIZE)),
        TextColor(ctx.palette.badge_label),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + BADGE_PAD_X, -(origin.y + BADGE_PAD_Y), z + 0.01),
    ));
    if selectable && !value.is_empty() {
        let tx = origin.x + BADGE_PAD_X;
        let ty = origin.y + BADGE_PAD_Y;
        let line_h = line_height(BADGE_FONT_SIZE);
        targets.spans.push(TextSpan {
            text: value.to_string(),
            rect: Rect::new(tx, ty, tx + size.x, ty + line_h),
            font_size: BADGE_FONT_SIZE,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn render_button_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    label: &str,
    kind: ButtonKind,
    style: Option<&Style>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let w = size.x;
    let h = size.y;

    // Resolve visual style. Filled = palette button_bg + label. Outline
    // = transparent bg + accent border + accent label. Ghost = fully
    // transparent + accent label (no shadow either). Style overrides
    // win over the kind defaults.
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let fg = ctx
        .resolve_color("fg")
        .unwrap_or(Color::srgb(0.91, 0.90, 0.86));
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
            fg,
            false,
        ),
        ButtonKind::Ghost => (
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            0.0,
            fg,
            false,
        ),
    };

    let mut bg_color = style
        .and_then(|s| s.background.as_deref())
        .and_then(|c| ctx.resolve_color(c))
        .unwrap_or(default_bg);
    let is_hovered = ctx.hovered_click_id.as_deref() == Some(id);
    if is_hovered {
        // Hover state: substitute the theme's hover bg for the kind's
        // default opaque bg, or lift a transparent bg (Outline / Ghost)
        // toward the hover color at a low alpha so the affordance is
        // visible without filling in the entire button.
        bg_color = match kind {
            ButtonKind::Filled => ctx.palette.button_bg_hover,
            ButtonKind::Outline | ButtonKind::Ghost => {
                let hover = ctx.palette.button_bg_hover.to_linear();
                Color::LinearRgba(LinearRgba {
                    red: hover.red,
                    green: hover.green,
                    blue: hover.blue,
                    alpha: 0.18,
                })
            }
        };
    }
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
    // Shift the mesh up by shadow_offset_y so the rendered button rect
    // lands at the LOGICAL origin (the shader places the SDF rect at
    // p_button = p_mesh - vec2(0, offset_y), which moves the rect down
    // by offset_y within the mesh). Without this compensation, tiles
    // with bigger shadows visually slide downward in their row even
    // though layout puts them at the same y.
    let entity = commands
        .spawn((
            ChildOf(ctx.content_root),
            Transform::from_xyz(
                origin.x + size.x * 0.5,
                -(origin.y + size.y * 0.5) + shadow_offset_y,
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
    // Background image — load from disk through the shared
    // WidgetImageCache so repeated references decode once. AssetServer
    // doesn't handle absolute paths reliably (widgets often resolve
    // textures to absolute paths because they're a separate process
    // and can't assume any cwd), so we do the std::fs read ourselves
    // via Commands::queue.
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
            // Cache check — borrow + lookup, drop before mutating
            // Assets<Image>.
            let path_buf = std::path::PathBuf::from(&path);
            let cached = world
                .get_resource::<crate::WidgetImageCache>()
                .and_then(|c| c.by_path.get(&path_buf).cloned());
            let handle = if let Some(h) = cached {
                h
            } else {
                let Ok(bytes) = std::fs::read(&path_buf) else {
                    return;
                };
                let Ok(decoded) = image::load_from_memory(&bytes) else {
                    return;
                };
                let rgba = decoded.to_rgba8();
                let (w, h) = (rgba.width(), rgba.height());
                let img = crate::make_nearest_image(rgba.into_raw(), w, h);
                let handle = world.resource_mut::<Assets<Image>>().add(img);
                if let Some(mut cache) = world.get_resource_mut::<crate::WidgetImageCache>()
                {
                    cache.by_path.insert(path_buf, handle.clone());
                }
                handle
            };
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

#[allow(clippy::too_many_arguments)]
fn render_link_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    url: &str,
    label: &str,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(label.to_string()),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
        TextColor(ctx.palette.link),
        Anchor::TOP_LEFT,
        TextBounds {
            width: Some(size.x.max(0.0)),
            height: Some(size.y.max(0.0)),
        },
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    targets.links.push(LinkTarget {
        url: url.to_string(),
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
    });
}

#[allow(clippy::too_many_arguments)]
fn render_bar_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    value: f32,
    max: f32,
    color: Option<&str>,
    track: Option<&str>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
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
    let fill_w = (size.x * ratio).max(0.0);

    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: bg,
            custom_size: Some(size),
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
                custom_size: Some(Vec2::new(fill_w, size.y)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(origin.x, -origin.y, z + 0.01),
        ));
    }
}

fn brighten(c: Color, delta: f32) -> Color {
    let s = c.to_srgba();
    Color::srgb(
        (s.red + delta).clamp(0.0, 1.0),
        (s.green + delta).clamp(0.0, 1.0),
        (s.blue + delta).clamp(0.0, 1.0),
    )
}

#[allow(clippy::too_many_arguments)]
fn render_tabs_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    laid: &crate::layout::LaidOut,
    node_id: taffy::NodeId,
    id: &str,
    items: &[TabItem],
    selected: &str,
    origin: Vec2,
    z: f32,
) {
    // Walk Taffy's per-tab children — each cell got its own NodeId
    // when the tree was built, so we can place tabs exactly where the
    // layout solver put them rather than re-measuring labels here.
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let child_ids = laid.taffy.children(node_id).unwrap_or_default();
    for (cid, tab) in child_ids.iter().zip(items.iter()) {
        let cl = laid.layout(*cid);
        let cell_pos = origin + Vec2::new(cl.location.x, cl.location.y);
        let cell_size = Vec2::new(cl.size.width, cl.size.height);
        let is_selected = tab.id == selected;
        let label_color = if is_selected { accent } else { ctx.palette.text_muted };
        commands.spawn((
            ChildOf(ctx.content_root),
            Text2d::new(tab.label.clone()),
            TextFont {
                font: ctx.font.clone(),
                font_size: DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
            TextColor(label_color),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(cell_pos.x + TAB_PAD_X, -(cell_pos.y + TAB_PAD_Y), z + 0.01),
        ));
        if is_selected {
            commands.spawn((
                ChildOf(ctx.content_root),
                Sprite {
                    color: accent,
                    custom_size: Some(Vec2::new(cell_size.x, TAB_INDICATOR_H)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(
                    cell_pos.x,
                    -(cell_pos.y + cell_size.y - TAB_INDICATOR_H),
                    z + 0.02,
                ),
            ));
        }
        targets.clicks.push(ClickTarget {
            id: id.to_string(),
            kind: ClickKind::TabSelect { tab: tab.id.clone() },
            rect: Rect::new(
                cell_pos.x,
                cell_pos.y,
                cell_pos.x + cell_size.x,
                cell_pos.y + cell_size.y,
            ),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn render_toggle_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    laid: &crate::layout::LaidOut,
    node_id: taffy::NodeId,
    id: &str,
    label: &str,
    checked: bool,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    // Taffy already laid out (label, track) inside the toggle row
    // when present; iterate the child positions.
    let (track_pos, track_size) = if label.is_empty() {
        (origin, size)
    } else {
        let kids = laid.taffy.children(node_id).unwrap_or_default();
        // [0] = label, [1] = track. Render label first.
        if let Some(label_id) = kids.first() {
            let ll = laid.layout(*label_id);
            commands.spawn((
                ChildOf(ctx.content_root),
                Text2d::new(label.to_string()),
                TextFont {
                    font: ctx.font.clone(),
                    font_size: DEFAULT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
                TextColor(ctx.palette.text),
                Anchor::TOP_LEFT,
                bevy::text::TextLayout::new_with_no_wrap(),
                Transform::from_xyz(
                    origin.x + ll.location.x,
                    -(origin.y + ll.location.y),
                    z + 0.01,
                ),
            ));
        }
        if let Some(track_id) = kids.get(1) {
            let tl = laid.layout(*track_id);
            (
                origin + Vec2::new(tl.location.x, tl.location.y),
                Vec2::new(tl.size.width, tl.size.height),
            )
        } else {
            (origin, Vec2::new(TOGGLE_TRACK_W, TOGGLE_TRACK_H))
        }
    };

    let track_color = if checked { accent } else { ctx.palette.bar_track };
    paint_rounded_panel(
        commands,
        ctx,
        track_pos,
        track_size,
        track_size.y * 0.5,
        track_color,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        0.0,
        z,
    );
    let knob_d = track_size.y - TOGGLE_KNOB_PAD * 2.0;
    let knob_x = if checked {
        track_pos.x + track_size.x - knob_d - TOGGLE_KNOB_PAD
    } else {
        track_pos.x + TOGGLE_KNOB_PAD
    };
    paint_rounded_panel(
        commands,
        ctx,
        Vec2::new(knob_x, track_pos.y + TOGGLE_KNOB_PAD),
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

    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::Toggle { new_checked: !checked },
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
    });
}

#[allow(clippy::too_many_arguments)]
fn render_input_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    value: &str,
    placeholder: &str,
    focused_attr: bool,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
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
        size,
        radius,
        bg,
        border_color,
        1.0,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        0.0,
        z,
    );

    let line_h = line_height(DEFAULT_FONT_SIZE);
    let text_y = (size.y - line_h) * 0.5;
    let content_w = (size.x - 2.0 * INPUT_PAD_X).max(0.0);

    if display_value.is_empty() && !is_focused {
        // Placeholder (muted), shown only when empty and unfocused.
        if !placeholder.is_empty() {
            commands.spawn((
                ChildOf(ctx.content_root),
                Text2d::new(placeholder.to_string()),
                TextFont {
                    font: ctx.font.clone(),
                    font_size: DEFAULT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(line_h),
                TextColor(ctx.palette.text_muted),
                Anchor::TOP_LEFT,
                bevy::text::TextLayout::new_with_no_wrap(),
                bevy::text::TextBounds { width: Some(content_w), height: None },
                pane_bevy::PaneContentNoClip,
                Transform::from_xyz(origin.x + INPUT_PAD_X, -(origin.y + text_y), z + 0.01),
            ));
        }
    } else {
        // Single line: render only the character window that fits the box
        // (scrolled to keep the caret visible) so a long value never
        // overflows into sibling elements. Monospace widget font, so a
        // fixed char width maps chars↔pixels exactly.
        let char_w = ctx.metrics.char_width(DEFAULT_FONT_SIZE).max(1.0);
        let max_chars = (content_w / char_w).floor().max(1.0) as usize;
        let chars: Vec<char> = display_value.chars().collect();
        let caret_pos = caret_pos.min(chars.len());
        let start = caret_pos.saturating_sub(max_chars);
        let end = (start + max_chars).min(chars.len());
        let visible: String = chars[start..end].iter().collect();
        if !visible.is_empty() {
            commands.spawn((
                ChildOf(ctx.content_root),
                Text2d::new(visible),
                TextFont {
                    font: ctx.font.clone(),
                    font_size: DEFAULT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(line_h),
                TextColor(ctx.palette.text),
                Anchor::TOP_LEFT,
                bevy::text::TextLayout::new_with_no_wrap(),
                bevy::text::TextBounds { width: Some(content_w), height: None },
                pane_bevy::PaneContentNoClip,
                Transform::from_xyz(origin.x + INPUT_PAD_X, -(origin.y + text_y), z + 0.01),
            ));
        }
        if show_caret {
            let caret_x = origin.x + INPUT_PAD_X + (caret_pos - start) as f32 * char_w;
            commands.spawn((
                ChildOf(ctx.content_root),
                Sprite {
                    color: accent,
                    custom_size: Some(Vec2::new(1.5, line_h)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(caret_x, -(origin.y + text_y), z + 0.02),
            ));
        }
    }

    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::InputFocus,
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
    });
}

/// Multi-line variant of [`render_input_at`]. Renders the value with
/// hard newlines (no soft wrap), top-aligned, with a line-aware caret.
/// Like `Input`, while the host owns focus (`ctx.focused_input`) the
/// displayed text and caret come from that live buffer.
#[allow(clippy::too_many_arguments)]
fn render_textarea_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    value: &str,
    placeholder: &str,
    focused_attr: bool,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let focus_match = ctx.focused_input.as_ref().filter(|f| f.id == id);
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
        size,
        radius,
        bg,
        border_color,
        1.0,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        0.0,
        z,
    );

    let line_h = line_height(DEFAULT_FONT_SIZE);
    let avail = (size.x - 2.0 * INPUT_PAD_X).max(1.0);

    if display_value.is_empty() && !is_focused {
        // Placeholder: short, render as-is on the first line.
        if !placeholder.is_empty() {
            commands.spawn((
                ChildOf(ctx.content_root),
                Text2d::new(placeholder.to_string()),
                TextFont {
                    font: ctx.font.clone(),
                    font_size: DEFAULT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(line_h),
                TextColor(ctx.palette.text_muted),
                Anchor::TOP_LEFT,
                bevy::text::TextLayout::new_with_no_wrap(),
                Transform::from_xyz(
                    origin.x + INPUT_PAD_X,
                    -(origin.y + TEXTAREA_PAD_Y),
                    z + 0.01,
                ),
            ));
        }
    } else {
        // Word-wrap to the box width (honoring hard newlines), then we
        // can render the wrapped text and place the caret against the
        // same visual layout. We pre-wrap ourselves (rather than letting
        // the text engine wrap) so the caret math matches exactly.
        let chars: Vec<char> = display_value.chars().collect();
        let visual = wrap_visual_lines(&chars, &ctx.metrics, avail);
        let display: String = visual
            .iter()
            .map(|&(s, e)| chars[s..e].iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        if !display.is_empty() {
            commands.spawn((
                ChildOf(ctx.content_root),
                Text2d::new(display),
                TextFont {
                    font: ctx.font.clone(),
                    font_size: DEFAULT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(line_h),
                TextColor(ctx.palette.text),
                Anchor::TOP_LEFT,
                bevy::text::TextLayout::new_with_no_wrap(),
                Transform::from_xyz(
                    origin.x + INPUT_PAD_X,
                    -(origin.y + TEXTAREA_PAD_Y),
                    z + 0.01,
                ),
            ));
        }

        if show_caret {
            let caret_pos = caret_pos.min(chars.len());
            // Locate the visual line holding the caret. When the caret
            // sits exactly at a soft-wrap boundary, it belongs at the
            // start of the next visual line.
            let mut cl = 0usize;
            let mut col_start = 0usize;
            for (li, &(s, e)) in visual.iter().enumerate() {
                cl = li;
                col_start = s;
                let is_last = li + 1 == visual.len();
                let next_soft = visual.get(li + 1).map(|&(ns, _)| ns == e).unwrap_or(false);
                if caret_pos < e || (caret_pos == e && (is_last || !next_soft)) {
                    break;
                }
            }
            let prefix: String = chars[col_start..caret_pos].iter().collect();
            let caret_x = origin.x + INPUT_PAD_X + ctx.metrics.measure(&prefix, DEFAULT_FONT_SIZE);
            let caret_y = origin.y + TEXTAREA_PAD_Y + cl as f32 * line_h;
            commands.spawn((
                ChildOf(ctx.content_root),
                Sprite {
                    color: accent,
                    custom_size: Some(Vec2::new(1.5, line_h)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(caret_x, -caret_y, z + 0.02),
            ));
        }
    }

    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::InputFocus,
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
    });
}

/// Word-wrap `chars` into visual lines that each fit within `avail`
/// pixels, honoring hard newlines. Greedy: breaks at the last space that
/// fits; for a single word longer than the line, breaks mid-word.
/// Returns each visual line's char range `[start, end)`. Shared by the
/// textarea renderer and the layout pass so the laid-out box height
/// matches the rendered wrap exactly.
pub(crate) fn wrap_visual_lines(
    chars: &[char],
    metrics: &PaneFontMetrics,
    avail: f32,
) -> Vec<(usize, usize)> {
    let n = chars.len();
    let mut lines: Vec<(usize, usize)> = Vec::new();
    let mut para_start = 0usize;
    loop {
        // End of the current hard-newline paragraph.
        let mut para_end = para_start;
        while para_end < n && chars[para_end] != '\n' {
            para_end += 1;
        }
        // Greedy word-wrap within [para_start, para_end).
        let mut line_start = para_start;
        let mut last_space: Option<usize> = None;
        let mut k = para_start;
        while k < para_end {
            if chars[k] == ' ' {
                last_space = Some(k + 1);
            }
            let seg: String = chars[line_start..=k].iter().collect();
            if k > line_start && metrics.measure(&seg, DEFAULT_FONT_SIZE) > avail {
                let brk = match last_space {
                    Some(b) if b > line_start && b <= k => b,
                    _ => k,
                };
                lines.push((line_start, brk));
                line_start = brk;
                last_space = None;
                k = brk;
                continue;
            }
            k += 1;
        }
        lines.push((line_start, para_end));

        if para_end < n {
            para_start = para_end + 1;
            // Trailing newline → one empty final line.
            if para_start == n {
                lines.push((n, n));
                break;
            }
        } else {
            break;
        }
    }
    if lines.is_empty() {
        lines.push((0, 0));
    }
    lines
}

/// Render an `Element::Table`. The grid cells were laid out as children
/// of `node_id` in row-major order (header row, then data rows); we read
/// their computed positions and draw each cell's text, plus a header
/// tint + divider and optional zebra striping. Cell text wraps within
/// its column and aligns per the column's `align`.
#[allow(clippy::too_many_arguments)]
fn render_table_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    laid: &crate::layout::LaidOut,
    node_id: taffy::NodeId,
    columns: &[crate::protocol::TableColumn],
    rows: &[Vec<String>],
    zebra: bool,
    selectable: bool,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    if columns.is_empty() {
        return;
    }
    let ncols = columns.len();
    let cells = laid.taffy.children(node_id).unwrap_or_default();
    if cells.is_empty() {
        return;
    }

    // Paint the panel + row backgrounds from the grid's CONTENT BOX
    // (the union of the laid-out cell extents), not the node `size`.
    // The node can be stretched wider than its tracks (flex parent with
    // align-items: stretch → dead space, panel wider than cells) or, in
    // the content-sized root, the tracks can exceed the clamped node
    // width (last column painting past the panel edge). Measuring the
    // cells directly makes the panel and zebra/header fills line up
    // with the cells exactly in every case.
    let mut content = Vec2::ZERO;
    for &cell in &cells {
        let cl = laid.layout(cell);
        content.x = content.x.max(cl.location.x + cl.size.width);
        content.y = content.y.max(cl.location.y + cl.size.height);
    }
    // Paint exactly the cells' extent. Falls back to the node size if a
    // degenerate (zero-area) content box is measured.
    let size = Vec2::new(
        if content.x > 0.0 { content.x } else { size.x },
        if content.y > 0.0 { content.y } else { size.y },
    );

    // Outer panel: subtle surface + border.
    let radius = ctx.resolve_f32("radius_sm").unwrap_or(4.0);
    let panel_bg = ctx
        .resolve_color("surface_2")
        .unwrap_or(Color::srgb(0.10, 0.11, 0.13));
    paint_rounded_panel(
        commands,
        ctx,
        origin,
        size,
        radius,
        panel_bg,
        ctx.palette.divider,
        1.0,
        Color::srgba(0.0, 0.0, 0.0, 0.0),
        0.0,
        0.0,
        z,
    );

    let line_h = line_height(DEFAULT_FONT_SIZE);
    let nrows_total = cells.len() / ncols; // header + data rows
    let header_bg = ctx
        .resolve_color("surface_3")
        .unwrap_or(Color::srgb(0.13, 0.14, 0.17));
    let zebra_bg = {
        let l = header_bg.to_linear();
        Color::LinearRgba(LinearRgba { alpha: 0.4, ..l })
    };

    // Row backgrounds: header tint + optional zebra on alternate data
    // rows. Row geometry comes from the first cell in each row (grid
    // tracks give every cell in a row the same top + height).
    for r in 0..nrows_total {
        let Some(&first) = cells.get(r * ncols) else {
            continue;
        };
        let cl = laid.layout(first);
        let row_top = origin.y + cl.location.y;
        let row_h = cl.size.height;
        let fill = if r == 0 {
            Some(header_bg)
        } else if zebra && r % 2 == 0 {
            Some(zebra_bg)
        } else {
            None
        };
        if let Some(color) = fill {
            commands.spawn((
                ChildOf(ctx.content_root),
                Sprite {
                    color,
                    custom_size: Some(Vec2::new(size.x, row_h)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(origin.x, -row_top, z + 0.001),
            ));
        }
    }

    // Divider under the header row.
    {
        let cl = laid.layout(cells[0]);
        let y = origin.y + cl.location.y + cl.size.height;
        commands.spawn((
            ChildOf(ctx.content_root),
            Sprite {
                color: ctx.palette.divider,
                custom_size: Some(Vec2::new(size.x, 1.0)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(origin.x, -y, z + 0.002),
        ));
    }

    // Cell text.
    for (k, &cell) in cells.iter().enumerate() {
        let r = k / ncols;
        let ci = k % ncols;
        let text = if r == 0 {
            columns[ci].header.clone()
        } else {
            rows.get(r - 1)
                .and_then(|row| row.get(ci))
                .cloned()
                .unwrap_or_default()
        };
        if text.is_empty() {
            continue;
        }
        let cl = laid.layout(cell);
        let cell_origin = origin + Vec2::new(cl.location.x, cl.location.y);
        let content_w = (cl.size.width - 2.0 * TABLE_CELL_PAD_X).max(0.0);
        let left = cell_origin.x + TABLE_CELL_PAD_X;
        let top = cell_origin.y + TABLE_CELL_PAD_Y;
        // Alignment by explicit positioning. Left-aligned cells wrap
        // within the column (TextBounds width); End/Center are measured
        // single-line and offset — this avoids a Justify+TextBounds
        // interaction that mis-places the text.
        let (text_x, bounds_w) = match columns[ci].align {
            Align::End => {
                let tw = ctx.metrics.measure(&text, DEFAULT_FONT_SIZE);
                (left + (content_w - tw).max(0.0), None)
            }
            Align::Center => {
                let tw = ctx.metrics.measure(&text, DEFAULT_FONT_SIZE);
                (left + ((content_w - tw) * 0.5).max(0.0), None)
            }
            _ => (left, Some(content_w)),
        };
        // Drag-select: register the cell as a selectable span. `text_x` is
        // the glyph origin (already alignment-shifted), and the run reaches
        // the cell's right content edge, so char offsets measured from
        // `text_x` land correctly for left/center/right alignment alike.
        if selectable {
            let line_h = line_height(DEFAULT_FONT_SIZE);
            targets.spans.push(TextSpan {
                text: text.clone(),
                rect: Rect::new(text_x, top, left + content_w, top + line_h),
                font_size: DEFAULT_FONT_SIZE,
            });
        }
        commands.spawn((
            ChildOf(ctx.content_root),
            Text2d::new(text),
            TextFont {
                font: ctx.font.clone(),
                font_size: DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(line_h),
            TextColor(ctx.palette.text),
            Anchor::TOP_LEFT,
            TextBounds {
                width: bounds_w,
                height: None,
            },
            // Keep our per-column wrap width: opt out of the pane-wide
            // TextBounds enforcement that would otherwise rewrap cells to
            // the full pane width.
            pane_bevy::PaneContentNoClip,
            Transform::from_xyz(text_x, -top, z + 0.01),
        ));
    }
}
