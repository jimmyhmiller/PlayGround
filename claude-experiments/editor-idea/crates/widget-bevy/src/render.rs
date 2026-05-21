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

use crate::protocol::{parse_hex_color, Align, Element, Weight};
use crate::{ClickTarget, LinkTarget, WidgetTargets};

const DEFAULT_FONT_SIZE: f32 = 13.0;
const LINE_HEIGHT_MUL: f32 = 1.4;

const COLOR_TEXT: Color = Color::srgb(0.85, 0.86, 0.90);
const COLOR_LINK: Color = Color::srgb(0.55, 0.75, 1.00);
const COLOR_DIVIDER: Color = Color::srgb(0.165, 0.170, 0.188);
const COLOR_BUTTON_BG: Color = Color::srgb(0.18, 0.20, 0.24);
const COLOR_BUTTON_TEXT: Color = Color::srgb(0.92, 0.94, 0.98);
const COLOR_BADGE_DEFAULT: Color = Color::srgb(0.20, 0.42, 0.28);
const COLOR_BADGE_TEXT: Color = Color::srgb(0.94, 0.98, 0.95);
const COLOR_BAR_FILL: Color = Color::srgb(0.42, 0.62, 0.92);
const COLOR_BAR_TRACK: Color = Color::srgb(0.13, 0.15, 0.19);

const BUTTON_PAD_X: f32 = 8.0;
const BUTTON_PAD_Y: f32 = 4.0;
const BADGE_PAD_X: f32 = 6.0;
const BADGE_PAD_Y: f32 = 2.0;
const BADGE_FONT_SIZE: f32 = 11.0;

pub struct LayoutCtx {
    pub font: Handle<Font>,
    pub metrics: PaneFontMetrics,
    pub content_root: Entity,
    pub content_size: Vec2,
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
        } => measure_stack(children, *gap, *pad, true, metrics),
        Element::Hstack {
            gap,
            pad,
            children,
            ..
        } => measure_stack(children, *gap, *pad, false, metrics),
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
    match el {
        Element::Vstack {
            gap,
            pad,
            children,
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
        } => render_hstack(
            commands, ctx, targets, children, origin, *gap, *pad, *align, max_w, z,
        ),
        Element::Text {
            value,
            color,
            size,
            weight,
        } => render_text(
            commands,
            ctx,
            value,
            color.as_deref(),
            *size,
            *weight,
            origin,
            max_w,
            z,
        ),
        Element::Divider => render_divider(commands, ctx, origin, max_w, z),
        Element::Spacer { size } => Vec2::new(*size, *size),
        Element::Badge { value, color } => {
            render_badge(commands, ctx, value, color.as_deref(), origin, z)
        }
        Element::Button { id, label } => {
            render_button(commands, ctx, targets, id, label, origin, z)
        }
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
    }
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
    origin: Vec2,
    max_w: f32,
    z: f32,
) -> Vec2 {
    let s = size.unwrap_or(DEFAULT_FONT_SIZE);
    let col = color
        .and_then(parse_hex_color)
        .map(|[r, g, b]| Color::srgb(r, g, b))
        .unwrap_or(COLOR_TEXT);
    let col = match weight {
        Some(Weight::Bold) => brighten(col, 0.08),
        _ => col,
    };
    let (display, w, h) = fit_single_line(value, s, max_w, &ctx.metrics);
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(display),
        TextFont {
            font: ctx.font.clone(),
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
            color: COLOR_DIVIDER,
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
        .unwrap_or(COLOR_BADGE_DEFAULT);
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
        TextColor(COLOR_BADGE_TEXT),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + BADGE_PAD_X, -(origin.y + BADGE_PAD_Y), z + 0.01),
    ));
    Vec2::new(w, h)
}

fn render_button(
    commands: &mut Commands,
    ctx: &LayoutCtx,
    targets: &mut WidgetTargets,
    id: &str,
    label: &str,
    origin: Vec2,
    z: f32,
) -> Vec2 {
    let text_w = ctx.metrics.measure(label, DEFAULT_FONT_SIZE);
    let w = text_w + BUTTON_PAD_X * 2.0;
    let h = DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL + BUTTON_PAD_Y * 2.0;
    commands.spawn((
        ChildOf(ctx.content_root),
        Sprite {
            color: COLOR_BUTTON_BG,
            custom_size: Some(Vec2::new(w, h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, z),
    ));
    commands.spawn((
        ChildOf(ctx.content_root),
        Text2d::new(label.to_string()),
        TextFont {
            font: ctx.font.clone(),
            font_size: DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(DEFAULT_FONT_SIZE * LINE_HEIGHT_MUL),
        TextColor(COLOR_BUTTON_TEXT),
        Anchor::TOP_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(origin.x + BUTTON_PAD_X, -(origin.y + BUTTON_PAD_Y), z + 0.01),
    ));
    targets.clicks.push(ClickTarget {
        id: id.to_string(),
        rect: Rect::new(origin.x, origin.y, origin.x + w, origin.y + h),
    });
    Vec2::new(w, h)
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
        TextColor(COLOR_LINK),
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
        .unwrap_or(COLOR_BAR_FILL);
    let bg = track
        .and_then(parse_hex_color)
        .map(|[r, g, b]| Color::srgb(r, g, b))
        .unwrap_or(COLOR_BAR_TRACK);
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
