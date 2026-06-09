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

use crate::protocol::{
    Align, ButtonKind, Edges, Element, GlazeLayer, GradientStop, Sides, Style, TabItem, Weight,
    parse_hex_color,
};
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
pub const CHECKBOX_SIZE: f32 = 18.0;
// Stepper: [− button][field][+ button].
pub const STEPPER_BTN: f32 = 26.0;
pub const STEPPER_FIELD_W: f32 = 56.0;
pub const STEPPER_GAP: f32 = 2.0;
pub const STEPPER_H: f32 = 26.0;
pub const STEPPER_W: f32 = STEPPER_BTN * 2.0 + STEPPER_FIELD_W + STEPPER_GAP * 2.0;
// Select trigger + dropdown menu.
pub const SELECT_H: f32 = 28.0;
pub const SELECT_PAD_X: f32 = 10.0;
pub const SELECT_ITEM_H: f32 = 26.0;
pub const SELECT_MENU_PAD: f32 = 4.0;

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
// Radio group.
pub const RADIO_RING: f32 = 16.0;
pub const RADIO_GAP: f32 = 8.0;
pub const RADIO_PAD_Y: f32 = 5.0;
pub const RADIO_GROUP_GAP: f32 = 6.0;

pub struct LayoutCtx {
    /// Fallback font (mono). Per-element `family` overrides resolve
    /// through [`Self::font_for`].
    pub font: Handle<Font>,
    pub metrics: PaneFontMetrics,
    /// Pane that owns this render tree. Glaze material entities retain this
    /// identity so their interaction uniforms can update without rebuilding.
    pub owner_pane: Entity,
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
    let root_origin = origin + Vec2::new(root_layout.location.x, root_layout.location.y);
    render_node(commands, ctx, targets, &laid, laid.root, el, root_origin, z);
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
        Element::Vstack {
            children, style, ..
        }
        | Element::Hstack {
            children, style, ..
        }
        | Element::Frame {
            children, style, ..
        } => {
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
            style,
            ..
        } => render_bar_at(
            commands,
            ctx,
            *value,
            *max,
            color.as_deref(),
            track.as_deref(),
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Slider {
            id,
            value,
            min,
            max,
            step,
            style,
            ..
        } => render_slider_at(
            commands,
            ctx,
            targets,
            id,
            *value,
            *min,
            *max,
            *step,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Stepper {
            id,
            value,
            min,
            max,
            step,
            style,
        } => render_stepper_at(
            commands,
            ctx,
            targets,
            id,
            *value,
            *min,
            *max,
            *step,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Tooltip { label, text, style } => render_tooltip_at(
            commands,
            ctx,
            targets,
            label,
            text,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Select {
            id,
            options,
            value,
            placeholder,
            style,
            ..
        } => render_select_trigger_at(
            commands,
            ctx,
            targets,
            id,
            options,
            value,
            placeholder,
            style.as_ref(),
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
        Element::SwatchButton { id, color, .. } => {
            render_swatch_at(commands, ctx, targets, color, Some(id), origin, size, z)
        }
        Element::Tabs {
            id,
            items,
            selected,
            style,
        } => render_tabs_at(
            commands,
            ctx,
            targets,
            laid,
            node_id,
            id,
            items,
            selected,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Toggle {
            id,
            label,
            checked,
            style,
        } => render_toggle_at(
            commands,
            ctx,
            targets,
            laid,
            node_id,
            id,
            label,
            *checked,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Checkbox {
            id,
            label,
            checked,
            style,
        } => render_checkbox_at(
            commands,
            ctx,
            targets,
            laid,
            node_id,
            id,
            label,
            *checked,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::RadioGroup {
            id,
            options,
            selected,
            style,
        } => render_radio_at(
            commands,
            ctx,
            targets,
            laid,
            node_id,
            id,
            options,
            selected,
            style.as_ref(),
            origin,
            z,
        ),
        Element::Input {
            id,
            value,
            placeholder,
            focused,
            style,
            ..
        } => render_input_at(
            commands,
            ctx,
            targets,
            id,
            value,
            placeholder,
            *focused,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::TextArea {
            id,
            value,
            placeholder,
            focused,
            style,
            ..
        } => render_textarea_at(
            commands,
            ctx,
            targets,
            id,
            value,
            placeholder,
            *focused,
            style.as_ref(),
            origin,
            size,
            z,
        ),
        Element::Table {
            columns,
            rows,
            zebra,
            selectable,
            style,
        } => render_table_at(
            commands,
            ctx,
            targets,
            laid,
            node_id,
            columns,
            rows,
            *zebra,
            *selectable,
            style.as_ref(),
            origin,
            size,
            z,
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
        ButtonKind::Outline => (Color::srgba(0.0, 0.0, 0.0, 0.0), accent, 1.5, fg, false),
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
    let (shadow_color, shadow_blur, shadow_offset_y) =
        if let Some(sh) = style.and_then(|s| s.shadow.as_ref()) {
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

    let has_glaze_layers = style.is_some_and(|s| !s.glaze_layers.is_empty());
    if has_glaze_layers {
        paint_glaze_layers(
            commands,
            ctx,
            style.expect("checked above"),
            origin,
            Vec2::new(w, h),
            z,
            Some(id),
        );
    } else {
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
    }

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
        Transform::from_xyz(
            origin.x + BUTTON_PAD_X,
            -(origin.y + BUTTON_PAD_Y),
            z + 0.01,
        ),
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
pub(crate) fn paint_rounded_panel(
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
    if !style.glaze_layers.is_empty() {
        paint_glaze_layers(commands, ctx, style, origin, size, z, None);
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
    let has_panel = bg.is_some() || border_w > 0.0 || shadow_color.to_srgba().alpha > 0.0;
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
                if let Some(mut cache) = world.get_resource_mut::<crate::WidgetImageCache>() {
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
    // Glaze shader layer — painted over the background, under the children.
    // Pass the element's corner radius so the shader masks to the rounded rect.
    if let Some(spec) = style.shader.as_ref() {
        paint_shader_layer(
            commands,
            ctx,
            &spec.body,
            origin,
            size,
            radius,
            z - 0.004,
            None,
        );
    }
}

fn paint_glaze_layers(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    style: &Style,
    origin: Vec2,
    size: Vec2,
    z: f32,
    element_id: Option<&str>,
) {
    let radius = style
        .radius
        .as_deref()
        .and_then(|r| ctx.resolve_f32(r))
        .unwrap_or(0.0);
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);
    let mut layer_index = 0usize;

    for layer in &style.glaze_layers {
        let layer_z = z - 0.005 + layer_index as f32 * 0.0005;
        layer_index += 1;
        match layer {
            GlazeLayer::Fill { color } => {
                let fill = ctx.resolve_color(color).unwrap_or(transparent);
                paint_rounded_panel(
                    commands,
                    ctx,
                    origin,
                    size,
                    radius,
                    fill,
                    transparent,
                    0.0,
                    transparent,
                    0.0,
                    0.0,
                    layer_z,
                );
            }
            GlazeLayer::LinearGradient { angle, stops } => {
                // A gradient lowers to a tiny generated WGSL body and rides the
                // existing shader path (which masks to the rounded rect for us).
                let body = gradient_wgsl(ctx, *angle, stops);
                paint_shader_layer(
                    commands, ctx, &body, origin, size, radius, layer_z, element_id,
                );
            }
            GlazeLayer::Border {
                color,
                width,
                sides,
            } => {
                let border = ctx.resolve_color(color).unwrap_or(transparent);
                if sides.is_all() {
                    paint_rounded_panel(
                        commands, ctx, origin, size, radius, transparent, border, *width,
                        transparent, 0.0, 0.0, layer_z,
                    );
                } else {
                    // Partial borders are sharp edge rects (a rounded corner can't
                    // belong to a single side).
                    paint_border_edges(commands, ctx, origin, size, border, *width, sides, layer_z);
                }
            }
            GlazeLayer::Shadow {
                color,
                blur,
                offset_x,
                offset_y,
                spread,
                inset,
            } => {
                let shadow = ctx.resolve_color(color).unwrap_or(transparent);
                if *inset {
                    // Inner shadow → generated SDF body on the shader path.
                    let body = inset_shadow_wgsl(shadow, *blur, *offset_x, *offset_y, *spread);
                    paint_shader_layer(
                        commands, ctx, &body, origin, size, radius, layer_z, element_id,
                    );
                } else {
                    // Outset drop shadow: grow the box by `spread`, fold offset_x
                    // into the origin, let the panel SDF apply blur + offset_y.
                    let so = Vec2::new(origin.x - spread + offset_x, origin.y - spread);
                    let ss = Vec2::new(size.x + 2.0 * spread, size.y + 2.0 * spread);
                    paint_rounded_panel(
                        commands,
                        ctx,
                        so,
                        ss,
                        radius + spread,
                        transparent,
                        transparent,
                        0.0,
                        shadow,
                        *blur,
                        *offset_y,
                        layer_z,
                    );
                }
            }
            GlazeLayer::Shader { body, .. } => {
                paint_shader_layer(
                    commands, ctx, body, origin, size, radius, layer_z, element_id,
                );
            }
        }
    }
}

/// Paint a compiled Glaze shader layer on a quad at the element's rect. The
/// fragment body comes from the `glaze` compiler; the material's per-instance
/// shader handle is cached by body hash and pinned in `specialize`.
fn paint_shader_layer(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    body: &str,
    origin: Vec2,
    size: Vec2,
    radius: f32,
    z: f32,
    element_id: Option<&str>,
) {
    use crate::button_material::WidgetButtonMesh;
    use crate::glaze_material::{
        GlazeInteractionTarget, GlazeMaterial, GlazeShaderCache, GlazeUniforms,
    };

    if size.x <= 0.0 || size.y <= 0.0 {
        return;
    }
    let body = body.to_string();
    let interaction = GlazeInteractionTarget {
        pane: ctx.owner_pane,
        element_id: element_id.map(str::to_string),
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
    };
    let entity = commands
        .spawn((
            ChildOf(ctx.content_root),
            interaction,
            // Unit quad centered on the element box (Y flipped: content uses a
            // top-left origin, Bevy 2D is Y-up). uv runs 0..1 across the quad.
            Transform::from_xyz(origin.x + size.x * 0.5, -(origin.y + size.y * 0.5), z)
                .with_scale(Vec3::new(size.x, size.y, 1.0)),
            Visibility::Inherited,
        ))
        .id();
    commands.queue(move |world: &mut World| {
        let Some(mesh) = world
            .get_resource::<WidgetButtonMesh>()
            .map(|m| m.0.clone())
        else {
            return;
        };
        // Get-or-create the shader handle (needs Assets<Shader> + the cache).
        let handle = world.resource_scope(|world, mut cache: Mut<GlazeShaderCache>| {
            let mut shaders = world.resource_mut::<Assets<Shader>>();
            cache.handle_for(&body, &mut shaders)
        });
        let mat = world
            .resource_mut::<Assets<GlazeMaterial>>()
            .add(GlazeMaterial {
                u: GlazeUniforms {
                    size,
                    resolution: size,
                    radius,
                    ..Default::default()
                },
                fragment: handle,
            });
        if let Ok(mut ec) = world.get_entity_mut(entity) {
            ec.insert((
                bevy::mesh::Mesh2d(mesh),
                bevy::sprite_render::MeshMaterial2d(mat),
            ));
        }
    });
}

/// Paint a per-side border as sharp filled edge rectangles. Used when a Glaze
/// `border_top`/`border_left`/… targets a subset of edges (a uniform border
/// keeps the rounded-rect SDF path instead).
#[allow(clippy::too_many_arguments)]
fn paint_border_edges(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    origin: Vec2,
    size: Vec2,
    color: Color,
    width: f32,
    sides: &Sides,
    z: f32,
) {
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);
    let mut edge = |o: Vec2, s: Vec2| {
        paint_rounded_panel(
            commands, ctx, o, s, 0.0, color, transparent, 0.0, transparent, 0.0, 0.0, z,
        );
    };
    if sides.top {
        edge(origin, Vec2::new(size.x, width));
    }
    if sides.bottom {
        edge(
            Vec2::new(origin.x, origin.y + size.y - width),
            Vec2::new(size.x, width),
        );
    }
    if sides.left {
        edge(origin, Vec2::new(width, size.y));
    }
    if sides.right {
        edge(
            Vec2::new(origin.x + size.x - width, origin.y),
            Vec2::new(width, size.y),
        );
    }
}

/// Format a float for WGSL source (always a decimal point so it parses as `f32`).
fn wgsl_f(x: f32) -> String {
    format!("{x:.6}")
}

/// Generate the `glaze_body` fragment for a linear gradient. The stop colors are
/// resolved to linear RGBA and baked in as constants; `t` is the position along
/// the gradient axis derived from `angle` (0° = left→right, 90° = bottom→top).
fn gradient_wgsl(ctx: &LayoutCtx, angle_deg: f32, stops: &[GradientStop]) -> String {
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);
    let vec4_of = |c: Color| {
        let v = lin_vec4(c);
        format!(
            "vec4<f32>({}, {}, {}, {})",
            wgsl_f(v.x),
            wgsl_f(v.y),
            wgsl_f(v.z),
            wgsl_f(v.w)
        )
    };
    let cols: Vec<(f32, String)> = stops
        .iter()
        .map(|s| {
            let c = ctx.resolve_color(&s.color).unwrap_or(transparent);
            (s.offset, vec4_of(c))
        })
        .collect();

    let mut b = String::new();
    b.push_str(&format!("    let a = radians({});\n", wgsl_f(angle_deg)));
    b.push_str("    let dir = vec2<f32>(cos(a), -sin(a));\n");
    b.push_str(
        "    let t = clamp(dot(in.uv - vec2<f32>(0.5, 0.5), dir) + 0.5, 0.0, 1.0);\n",
    );
    b.push_str(&format!("    var col = {};\n", cols[0].1));
    for pair in cols.windows(2) {
        let (o0, _) = &pair[0];
        let (o1, c1) = &pair[1];
        let denom = (o1 - o0).max(1e-5);
        b.push_str(&format!(
            "    col = mix(col, {}, clamp((t - {}) / {}, 0.0, 1.0));\n",
            c1,
            wgsl_f(*o0),
            wgsl_f(denom)
        ));
    }
    b.push_str("    return col;\n");
    b
}

/// Generate the `glaze_body` fragment for an inner (inset) shadow: a rounded-rect
/// SDF that darkens inward from the edges, offset by `(offx, offy)` and pulled in
/// by `spread`. The outer rounded clip is still applied by `assemble_wgsl`.
fn inset_shadow_wgsl(color: Color, blur: f32, offx: f32, offy: f32, spread: f32) -> String {
    let c = lin_vec4(color);
    format!(
        "    let p = (in.uv - vec2<f32>(0.5, 0.5)) * u.size - vec2<f32>({offx}, {offy});\n\
         \x20   let hh = u.size * 0.5;\n\
         \x20   let rr = min(u.radius, min(hh.x, hh.y));\n\
         \x20   let qq = abs(p) - hh + vec2<f32>(rr, rr);\n\
         \x20   let dd = length(max(qq, vec2<f32>(0.0, 0.0))) + min(max(qq.x, qq.y), 0.0) - rr;\n\
         \x20   let inner = 1.0 - smoothstep(0.0, max({blur}, 1.0), -dd - {spread});\n\
         \x20   return vec4<f32>({r}, {g}, {b}, {a} * clamp(inner, 0.0, 1.0));\n",
        offx = wgsl_f(offx),
        offy = wgsl_f(offy),
        blur = wgsl_f(blur),
        spread = wgsl_f(spread),
        r = wgsl_f(c.x),
        g = wgsl_f(c.y),
        b = wgsl_f(c.z),
        a = wgsl_f(c.w),
    )
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
/// Render an `Element::Slider`: a full-width `track`, a value-driven `range`
/// fill, and a `thumb` handle centred on the value. Each slot paints from its
/// Glaze plan when present, else a sensible default. Also registers the drag
/// hit-region so the host can map cursor x → value.
#[allow(clippy::too_many_arguments)]
fn render_slider_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
    style: Option<&crate::protocol::SliderStyle>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let span = max - min;
    let ratio = if span.abs() < 1e-6 {
        0.0
    } else {
        ((value - min) / span).clamp(0.0, 1.0)
    };
    let thumb_d = size.y;
    let thumb_x = origin.x + ratio * (size.x - thumb_d);
    let thumb_center_x = thumb_x + thumb_d * 0.5;

    let track_plan = style.and_then(|s| s.track.as_ref());
    let range_plan = style.and_then(|s| s.range.as_ref());
    let thumb_plan = style.and_then(|s| s.thumb.as_ref());

    // Track height: from the track plan's `height` if given, else a thin groove.
    let track_h = track_plan
        .and_then(|p| p.height.as_deref())
        .and_then(|h| ctx.resolve_f32(h))
        .unwrap_or((size.y * 0.3).max(4.0));
    let track_y = origin.y + (size.y - track_h) * 0.5;
    let track_origin = Vec2::new(origin.x, track_y);
    let track_size = Vec2::new(size.x, track_h);

    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);

    // --- track ---
    if let Some(plan) = track_plan {
        paint_style_background(commands, ctx, Some(plan), track_origin, track_size, z);
    } else {
        paint_rounded_panel(
            commands,
            ctx,
            track_origin,
            track_size,
            track_h * 0.5,
            ctx.palette.bar_track,
            transparent,
            0.0,
            transparent,
            0.0,
            0.0,
            z,
        );
    }

    // --- range (leading fill up to the thumb centre) ---
    let range_w = (thumb_center_x - origin.x).max(0.0);
    if range_w > 0.0 {
        let range_size = Vec2::new(range_w, track_h);
        if let Some(plan) = range_plan {
            paint_style_background(commands, ctx, Some(plan), track_origin, range_size, z + 0.01);
        } else {
            let accent = ctx
                .resolve_color("accent")
                .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
            paint_rounded_panel(
                commands,
                ctx,
                track_origin,
                range_size,
                track_h * 0.5,
                accent,
                transparent,
                0.0,
                transparent,
                0.0,
                0.0,
                z + 0.01,
            );
        }
    }

    // --- thumb (handle) ---
    let thumb_origin = Vec2::new(thumb_x, origin.y);
    let thumb_size = Vec2::splat(thumb_d);
    if let Some(plan) = thumb_plan {
        paint_style_background(commands, ctx, Some(plan), thumb_origin, thumb_size, z + 0.02);
    } else {
        paint_rounded_panel(
            commands,
            ctx,
            thumb_origin,
            thumb_size,
            thumb_d * 0.5,
            Color::WHITE,
            transparent,
            0.0,
            Color::srgba(0.0, 0.0, 0.0, 0.25),
            2.0,
            1.0,
            z + 0.02,
        );
    }

    // --- drag hit-region ---
    targets.sliders.push(crate::SliderTarget {
        id: id.to_string(),
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
        value_x0: origin.x + thumb_d * 0.5,
        value_span: (size.x - thumb_d).max(1.0),
        min,
        max,
        step,
    });
}

/// Render an `Element::Tooltip`'s in-pane label (an underlined hint anchor) and
/// record its hover-region; the floating hint is drawn by the overlay system.
#[allow(clippy::too_many_arguments)]
fn render_tooltip_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    label: &str,
    text: &str,
    style: Option<&crate::protocol::TooltipStyle>,
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
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x, -origin.y, z + 0.01),
    ));
    targets.tooltips.push(crate::TooltipTarget {
        anchor: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
        text: text.to_string(),
        style: style.cloned(),
    });
}

/// Render the closed, in-pane `trigger` of an `Element::Select`: a box showing
/// the selected option's label (or the placeholder) and a chevron. Records the
/// trigger as a `SelectTarget` (anchor + data) and a `SelectTrigger` click
/// target; the floating menu is drawn separately when the host marks it open.
#[allow(clippy::too_many_arguments)]
fn render_select_trigger_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    options: &[TabItem],
    value: &str,
    placeholder: &str,
    style: Option<&crate::protocol::SelectStyle>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);
    if let Some(plan) = style.and_then(|s| s.trigger.as_ref()) {
        paint_style_background(commands, ctx, Some(plan), origin, size, z);
    } else {
        paint_rounded_panel(
            commands,
            ctx,
            origin,
            size,
            6.0,
            ctx.palette.bar_track,
            ctx.palette.divider,
            1.0,
            transparent,
            0.0,
            0.0,
            z,
        );
    }

    let selected = options.iter().find(|o| o.id == value);
    let (label, color) = match selected {
        Some(o) => (o.label.clone(), ctx.palette.text),
        None => (placeholder.to_string(), ctx.palette.text_muted),
    };
    let cy = origin.y + size.y * 0.5;
    // label (left, vertically centred)
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(label),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
        TextColor(color),
        Anchor::CENTER_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + SELECT_PAD_X, -cy, z + 0.01),
    ));
    // chevron (right)
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new("\u{25be}"),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
        TextColor(ctx.palette.text_muted),
        Anchor::CENTER_RIGHT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + size.x - SELECT_PAD_X, -cy, z + 0.01),
    ));

    let rect = Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y);
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::SelectTrigger,
        rect,
    });
    targets.selects.push(crate::SelectTarget {
        id: id.to_string(),
        anchor: rect,
        options: options.to_vec(),
        value: value.to_string(),
        width: size.x,
        style: style.cloned(),
    });
}

/// Spawn a single line of text centred within `(origin, size)`.
fn spawn_centered_text(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    text: &str,
    origin: Vec2,
    size: Vec2,
    color: Color,
    z: f32,
) {
    let center = origin + size * 0.5;
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(text.to_string()),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
        TextColor(color),
        Anchor::CENTER,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(center.x, -center.y, z),
    ));
}

/// Render an `Element::Stepper`: `[− button][value field][+ button]`. The
/// buttons carry the precomputed (clamped) target value, so a click is a plain
/// `NumberChange` — the renderer owns the arithmetic.
#[allow(clippy::too_many_arguments)]
fn render_stepper_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
    style: Option<&crate::protocol::StepperStyle>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);
    let field_plan = style.and_then(|s| s.field.as_ref());
    let button_plan = style.and_then(|s| s.button.as_ref());

    let minus_origin = origin;
    let btn_size = Vec2::new(STEPPER_BTN, size.y);
    let field_origin = Vec2::new(origin.x + STEPPER_BTN + STEPPER_GAP, origin.y);
    let field_size = Vec2::new(STEPPER_FIELD_W, size.y);
    let plus_origin = Vec2::new(field_origin.x + STEPPER_FIELD_W + STEPPER_GAP, origin.y);

    let paint_button = |commands: &mut Commands, o: Vec2| {
        if let Some(plan) = button_plan {
            paint_style_background(commands, ctx, Some(plan), o, btn_size, z);
        } else {
            paint_rounded_panel(
                commands,
                ctx,
                o,
                btn_size,
                4.0,
                ctx.palette.bar_track,
                ctx.palette.divider,
                1.0,
                transparent,
                0.0,
                0.0,
                z,
            );
        }
    };
    paint_button(commands, minus_origin);
    paint_button(commands, plus_origin);

    if let Some(plan) = field_plan {
        paint_style_background(commands, ctx, Some(plan), field_origin, field_size, z);
    } else {
        paint_rounded_panel(
            commands,
            ctx,
            field_origin,
            field_size,
            4.0,
            ctx.palette.bar_track,
            ctx.palette.divider,
            1.0,
            transparent,
            0.0,
            0.0,
            z,
        );
    }

    // glyphs + value (whole numbers print without a decimal)
    let value_str = if (value - value.round()).abs() < 1e-4 {
        format!("{}", value.round() as i64)
    } else {
        format!("{value:.1}")
    };
    spawn_centered_text(commands, ctx, "−", minus_origin, btn_size, ctx.palette.text, z + 0.01);
    spawn_centered_text(commands, ctx, &value_str, field_origin, field_size, ctx.palette.text, z + 0.01);
    spawn_centered_text(commands, ctx, "+", plus_origin, btn_size, ctx.palette.text, z + 0.01);

    // click targets carry the clamped target value
    let dec = (value - step).clamp(min.min(max), min.max(max));
    let inc = (value + step).clamp(min.min(max), min.max(max));
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::NumberChange { value: dec },
        rect: Rect::new(
            minus_origin.x,
            minus_origin.y,
            minus_origin.x + btn_size.x,
            minus_origin.y + btn_size.y,
        ),
    });
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::NumberChange { value: inc },
        rect: Rect::new(
            plus_origin.x,
            plus_origin.y,
            plus_origin.x + btn_size.x,
            plus_origin.y + btn_size.y,
        ),
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
    style: Option<&crate::protocol::BarStyle>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let ratio = if max <= 0.0 {
        0.0
    } else {
        (value / max).clamp(0.0, 1.0)
    };
    let fill_w = (size.x * ratio).max(0.0);
    let fill_size = Vec2::new(fill_w, size.y);

    // Value-driven sub-layout: the `track` slot spans the full rect, the `fill`
    // slot is the leading `ratio` of it. Each slot paints from its Glaze plan
    // when present, else from the flat `track`/`color` colors (legacy path).
    let track_plan = style.and_then(|s| s.track.as_ref());
    let fill_plan = style.and_then(|s| s.fill.as_ref());

    if let Some(plan) = track_plan {
        paint_style_background(commands, ctx, Some(plan), origin, size, z);
    } else {
        let bg = track
            .and_then(parse_hex_color)
            .map(|[r, g, b]| Color::srgb(r, g, b))
            .unwrap_or(ctx.palette.bar_track);
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
    }

    if fill_w <= 0.0 {
        return;
    }
    if let Some(plan) = fill_plan {
        paint_style_background(commands, ctx, Some(plan), origin, fill_size, z + 0.01);
    } else {
        let fill = color
            .and_then(parse_hex_color)
            .map(|[r, g, b]| Color::srgb(r, g, b))
            .unwrap_or(ctx.palette.bar_fill);
        commands.spawn((
            ChildOf(ctx.content_root),
            Sprite {
                color: fill,
                custom_size: Some(fill_size),
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
    style: Option<&crate::protocol::TabsStyle>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    // Walk Taffy's per-tab children — each cell got its own NodeId
    // when the tree was built, so we can place tabs exactly where the
    // layout solver put them rather than re-measuring labels here.
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    // `strip` slot: a background behind the whole tab bar.
    if let Some(plan) = style.and_then(|s| s.strip.as_ref()) {
        paint_style_background(commands, ctx, Some(plan), origin, size, z - 0.01);
    }
    let child_ids = laid.taffy.children(node_id).unwrap_or_default();
    for (cid, tab) in child_ids.iter().zip(items.iter()) {
        let cl = laid.layout(*cid);
        let cell_pos = origin + Vec2::new(cl.location.x, cl.location.y);
        let cell_size = Vec2::new(cl.size.width, cl.size.height);
        let is_selected = tab.id == selected;
        // `tab` slot: swap the precomputed `:selected` plan in for the active tab.
        let tab_plan = style.and_then(|s| {
            if is_selected {
                s.tab_selected.as_ref().or(s.tab.as_ref())
            } else {
                s.tab.as_ref()
            }
        });
        if let Some(plan) = tab_plan {
            paint_style_background(commands, ctx, Some(plan), cell_pos, cell_size, z);
        }
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
            LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
            TextColor(label_color),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(cell_pos.x + TAB_PAD_X, -(cell_pos.y + TAB_PAD_Y), z + 0.01),
        ));
        if is_selected {
            // `indicator` slot: the underline under the active tab (value-driven
            // position = the selected cell). A styled plan can give it a height,
            // gradient, glow, etc.; otherwise the flat accent rule.
            let indicator_plan = style.and_then(|s| s.indicator.as_ref());
            if let Some(plan) = indicator_plan {
                let ind_h = plan
                    .height
                    .as_deref()
                    .and_then(|h| ctx.resolve_f32(h))
                    .unwrap_or(TAB_INDICATOR_H);
                let ind_origin = Vec2::new(cell_pos.x, cell_pos.y + cell_size.y - ind_h);
                paint_style_background(
                    commands,
                    ctx,
                    Some(plan),
                    ind_origin,
                    Vec2::new(cell_size.x, ind_h),
                    z + 0.02,
                );
            } else {
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
        }
        targets.clicks.push(ClickTarget {
            id: id.to_string(),
            kind: ClickKind::TabSelect {
                tab: tab.id.clone(),
            },
            rect: Rect::new(
                cell_pos.x,
                cell_pos.y,
                cell_pos.x + cell_size.x,
                cell_pos.y + cell_size.y,
            ),
        });
    }
}

/// Render an `Element::RadioGroup`: a column of option rows, each a `ring` plus
/// (when selected) a `dot`, then the label. Emits `RadioSelect` on click.
#[allow(clippy::too_many_arguments)]
fn render_radio_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    laid: &crate::layout::LaidOut,
    node_id: taffy::NodeId,
    id: &str,
    options: &[TabItem],
    selected: &str,
    style: Option<&crate::protocol::RadioGroupStyle>,
    origin: Vec2,
    z: f32,
) {
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);
    let cells = laid.taffy.children(node_id).unwrap_or_default();
    for (cid, opt) in cells.iter().zip(options.iter()) {
        let cl = laid.layout(*cid);
        let cell_pos = origin + Vec2::new(cl.location.x, cl.location.y);
        let cell_size = Vec2::new(cl.size.width, cl.size.height);
        let is_selected = opt.id == selected;

        // ring (left, vertically centred within the row)
        let ring_pos = Vec2::new(
            cell_pos.x,
            cell_pos.y + (cell_size.y - RADIO_RING) * 0.5,
        );
        let ring_size = Vec2::splat(RADIO_RING);
        if let Some(plan) = style.and_then(|s| s.ring.as_ref()) {
            paint_style_background(commands, ctx, Some(plan), ring_pos, ring_size, z);
        } else {
            let border = if is_selected { accent } else { ctx.palette.text_muted };
            paint_rounded_panel(
                commands,
                ctx,
                ring_pos,
                ring_size,
                RADIO_RING * 0.5,
                ctx.palette.bar_track,
                border,
                1.5,
                transparent,
                0.0,
                0.0,
                z,
            );
        }

        // dot (only when selected)
        if is_selected {
            let inset = RADIO_RING * 0.3;
            let dot_pos = ring_pos + Vec2::splat(inset);
            let dot_size = Vec2::splat(RADIO_RING - inset * 2.0);
            if let Some(plan) = style.and_then(|s| s.dot.as_ref()) {
                paint_style_background(commands, ctx, Some(plan), dot_pos, dot_size, z + 0.01);
            } else {
                paint_rounded_panel(
                    commands,
                    ctx,
                    dot_pos,
                    dot_size,
                    dot_size.x * 0.5,
                    accent,
                    transparent,
                    0.0,
                    transparent,
                    0.0,
                    0.0,
                    z + 0.01,
                );
            }
        }

        // label, after the ring's reserved padding
        commands.spawn((
            ChildOf(ctx.content_root),
            Text2d::new(opt.label.clone()),
            TextFont {
                font: ctx.font.clone(),
                font_size: DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(line_height(DEFAULT_FONT_SIZE)),
            TextColor(if is_selected { ctx.palette.text } else { ctx.palette.text_muted }),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(
                cell_pos.x + RADIO_RING + RADIO_GAP,
                -(cell_pos.y + RADIO_PAD_Y),
                z + 0.01,
            ),
        ));

        targets.clicks.push(ClickTarget {
            id: id.to_string(),
            kind: ClickKind::RadioSelect {
                option: opt.id.clone(),
            },
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
    style: Option<&crate::protocol::ToggleStyle>,
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

    // Track slot: from its Glaze plan (resolved with the `:checked` state) when
    // present, else the hardcoded accent/muted pill.
    let track_plan = style.and_then(|s| s.track.as_ref());
    if let Some(plan) = track_plan {
        paint_style_background(commands, ctx, Some(plan), track_pos, track_size, z);
    } else {
        let track_color = if checked {
            accent
        } else {
            ctx.palette.bar_track
        };
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
    }
    // Knob slot: value-driven x-position (left when off, right when on); the
    // slot plan styles the dot itself.
    let knob_d = track_size.y - TOGGLE_KNOB_PAD * 2.0;
    let knob_x = if checked {
        track_pos.x + track_size.x - knob_d - TOGGLE_KNOB_PAD
    } else {
        track_pos.x + TOGGLE_KNOB_PAD
    };
    let knob_pos = Vec2::new(knob_x, track_pos.y + TOGGLE_KNOB_PAD);
    let knob_size = Vec2::new(knob_d, knob_d);
    let knob_plan = style.and_then(|s| s.knob.as_ref());
    if let Some(plan) = knob_plan {
        paint_style_background(commands, ctx, Some(plan), knob_pos, knob_size, z + 0.01);
    } else {
        paint_rounded_panel(
            commands,
            ctx,
            knob_pos,
            knob_size,
            knob_d * 0.5,
            Color::WHITE,
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            0.0,
            Color::srgba(0.0, 0.0, 0.0, 0.2),
            2.0,
            1.0,
            z + 0.01,
        );
    }

    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::Toggle {
            new_checked: !checked,
        },
        rect: Rect::new(origin.x, origin.y, origin.x + size.x, origin.y + size.y),
    });
}

/// Render an `Element::Checkbox`: a `box` square plus, when `checked`, an inner
/// `check` mark. Both slot-styled; emits the same `Toggle` click event so the
/// host/script reuse `on_toggle`.
#[allow(clippy::too_many_arguments)]
fn render_checkbox_at(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    laid: &crate::layout::LaidOut,
    node_id: taffy::NodeId,
    id: &str,
    label: &str,
    checked: bool,
    style: Option<&crate::protocol::CheckboxStyle>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    let accent = ctx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);

    // The box is the whole element when unlabeled, else the first Taffy child.
    let (box_pos, box_size) = if label.is_empty() {
        (origin, size)
    } else {
        let kids = laid.taffy.children(node_id).unwrap_or_default();
        if let Some(label_id) = kids.get(1) {
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
        match kids.first() {
            Some(box_id) => {
                let bl = laid.layout(*box_id);
                (
                    origin + Vec2::new(bl.location.x, bl.location.y),
                    Vec2::new(bl.size.width, bl.size.height),
                )
            }
            None => (origin, Vec2::splat(CHECKBOX_SIZE)),
        }
    };

    // --- box slot ---
    let box_plan = style.and_then(|s| s.square.as_ref());
    if let Some(plan) = box_plan {
        paint_style_background(commands, ctx, Some(plan), box_pos, box_size, z);
    } else {
        let border = if checked { accent } else { ctx.palette.text_muted };
        paint_rounded_panel(
            commands,
            ctx,
            box_pos,
            box_size,
            4.0,
            ctx.palette.bar_track,
            border,
            1.5,
            transparent,
            0.0,
            0.0,
            z,
        );
    }

    // --- check mark (only when checked) ---
    if checked {
        let inset = box_size * 0.28;
        let check_pos = box_pos + inset;
        let check_size = (box_size - inset * 2.0).max(Vec2::ZERO);
        let check_plan = style.and_then(|s| s.check.as_ref());
        if let Some(plan) = check_plan {
            paint_style_background(commands, ctx, Some(plan), check_pos, check_size, z + 0.01);
        } else {
            paint_rounded_panel(
                commands,
                ctx,
                check_pos,
                check_size,
                2.0,
                accent,
                transparent,
                0.0,
                transparent,
                0.0,
                0.0,
                z + 0.01,
            );
        }
    }

    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        kind: ClickKind::Toggle {
            new_checked: !checked,
        },
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
    style: Option<&Style>,
    origin: Vec2,
    size: Vec2,
    z: f32,
) {
    // Host owns the editing state while an input is focused — so when
    // ctx.focused_input matches our id we render from that buffer
    // instead of the widget's stale value.
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
    if let Some(style) = style.filter(|s| !s.glaze_layers.is_empty()) {
        paint_glaze_layers(commands, ctx, style, origin, size, z, Some(id));
    } else {
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
    }

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
                bevy::text::TextBounds {
                    width: Some(content_w),
                    height: None,
                },
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
                bevy::text::TextBounds {
                    width: Some(content_w),
                    height: None,
                },
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
    style: Option<&Style>,
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
    if let Some(style) = style.filter(|s| !s.glaze_layers.is_empty()) {
        paint_glaze_layers(commands, ctx, style, origin, size, z, Some(id));
    } else {
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
    }

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
    style: Option<&crate::protocol::TableStyle>,
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

    // Outer `panel` slot: from its Glaze plan when present, else the hardcoded
    // surface + border.
    if let Some(plan) = style.and_then(|s| s.panel.as_ref()) {
        paint_style_background(commands, ctx, Some(plan), origin, size, z);
    } else {
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
    }

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
    let header_plan = style.and_then(|s| s.header.as_ref());
    let zebra_plan = style.and_then(|s| s.zebra.as_ref());
    for r in 0..nrows_total {
        let Some(&first) = cells.get(r * ncols) else {
            continue;
        };
        let cl = laid.layout(first);
        let row_top = origin.y + cl.location.y;
        let row_h = cl.size.height;
        let row_origin = Vec2::new(origin.x, row_top);
        let row_size = Vec2::new(size.x, row_h);
        // header / zebra `slot` plans win over the hardcoded fills.
        let is_header = r == 0;
        let is_zebra = zebra && r % 2 == 0;
        if is_header && header_plan.is_some() {
            paint_style_background(commands, ctx, header_plan, row_origin, row_size, z + 0.001);
        } else if is_zebra && zebra_plan.is_some() {
            paint_style_background(commands, ctx, zebra_plan, row_origin, row_size, z + 0.001);
        } else {
            let fill = if is_header {
                Some(header_bg)
            } else if is_zebra {
                Some(zebra_bg)
            } else {
                None
            };
            if let Some(color) = fill {
                commands.spawn((
                    ChildOf(ctx.content_root),
                    Sprite {
                        color,
                        custom_size: Some(row_size),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(origin.x, -row_top, z + 0.001),
                ));
            }
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
