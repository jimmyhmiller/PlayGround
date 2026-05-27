//! Layout pass — turns an Element tree into computed (x, y, width,
//! height) per node via the [taffy](https://docs.rs/taffy) flex/grid
//! engine.
//!
//! Pipeline:
//!   1. [`build_tree`] walks the Element tree and constructs a parallel
//!      Taffy tree, mirroring the parent/child relationship and
//!      translating each Element's style into a Taffy [`taffy::Style`].
//!      Text leaves carry a [`MeasureCtx`] so the measure callback can
//!      size them later.
//!   2. [`compute`] calls Taffy's layout solver, providing a measure
//!      function that handles text wrapping for leaves.
//!   3. The renderer walks the Element tree alongside the Taffy tree
//!      and reads each node's computed `Layout` to spawn entities.
//!
//! ## Why Taffy
//!
//! Previously, layout was hand-rolled in `render.rs::measure` +
//! `render_stack`. That worked for trivial trees but couldn't:
//!   - distribute extra width among siblings (`flex-grow`)
//!   - equalize sibling heights (`align-items: stretch`)
//!   - constrain with min/max widths
//!   - wrap text at the available width
//!
//! Taffy gives us all of those declaratively. It's the same engine
//! `bevy_ui` uses, so the dependency is already in the workspace.

use bevy::math::Vec2;
use pane_bevy::PaneFontMetrics;
use taffy::prelude::*;

use crate::protocol::{Align, Border, ButtonKind, Edges, Element, Shadow, Style as PStyle, Weight};

/// Per-text-leaf context passed to Taffy's measure callback.
#[derive(Clone, Debug)]
pub struct MeasureCtx {
    pub value: String,
    pub font_size: f32,
}

/// Built layout: the Taffy tree + the root node + a parallel vector of
/// (Element-tree-position → NodeId) entries.
///
/// `nodes` is in pre-order so a renderer that walks the Element tree
/// in the same pre-order can pair each Element with its layout by
/// incrementing a counter.
pub struct LaidOut {
    pub taffy: TaffyTree<MeasureCtx>,
    pub root: NodeId,
}

impl LaidOut {
    pub fn layout(&self, id: NodeId) -> Layout {
        *self.taffy.layout(id).expect("missing layout for node")
    }
}

/// Build the Taffy tree mirroring `el`. Returns the root NodeId.
pub fn build_tree(el: &Element) -> LaidOut {
    let mut taffy = TaffyTree::new();
    let root = build_node(&mut taffy, el);
    LaidOut { taffy, root }
}

fn build_node(taffy: &mut TaffyTree<MeasureCtx>, el: &Element) -> NodeId {
    match el {
        Element::Vstack {
            gap,
            pad,
            children,
            style,
        } => {
            let st = stack_style(*gap, *pad, style.as_ref(), FlexDirection::Column);
            let kids: Vec<NodeId> = children.iter().map(|c| build_node(taffy, c)).collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::Hstack {
            gap,
            pad,
            align,
            children,
            style,
        } => {
            let mut st = stack_style(*gap, *pad, style.as_ref(), FlexDirection::Row);
            st.align_items = Some(align_to_taffy(*align));
            let kids: Vec<NodeId> = children.iter().map(|c| build_node(taffy, c)).collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::Frame {
            gap,
            pad,
            children,
            style,
        } => {
            let st = stack_style(*gap, *pad, style.as_ref(), FlexDirection::Column);
            let kids: Vec<NodeId> = children.iter().map(|c| build_node(taffy, c)).collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::Scroll {
            gap,
            pad,
            children,
        } => {
            let st = stack_style(*gap, *pad, None, FlexDirection::Column);
            let kids: Vec<NodeId> = children.iter().map(|c| build_node(taffy, c)).collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::ListItem {
            gap,
            pad,
            children,
            style,
            ..
        } => {
            let st = stack_style(*gap, *pad, style.as_ref(), FlexDirection::Column);
            let kids: Vec<NodeId> = children.iter().map(|c| build_node(taffy, c)).collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::Text { value, size, .. } => {
            let font_size = size.unwrap_or(crate::render::DEFAULT_FONT_SIZE);
            let st = taffy::Style {
                ..taffy::Style::DEFAULT
            };
            taffy
                .new_leaf_with_context(
                    st,
                    MeasureCtx {
                        value: value.clone(),
                        font_size,
                    },
                )
                .unwrap()
        }
        Element::Divider => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::auto(),
                    height: Dimension::length(1.0),
                },
                min_size: Size {
                    width: Dimension::length(20.0),
                    height: Dimension::length(1.0),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Spacer { size } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*size),
                    height: Dimension::length(*size),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Badge { value, .. } => taffy
            .new_leaf_with_context(
                taffy::Style {
                    padding: Rect {
                        left: LengthPercentage::length(crate::render::BADGE_PAD_X),
                        right: LengthPercentage::length(crate::render::BADGE_PAD_X),
                        top: LengthPercentage::length(crate::render::BADGE_PAD_Y),
                        bottom: LengthPercentage::length(crate::render::BADGE_PAD_Y),
                    },
                    ..taffy::Style::DEFAULT
                },
                MeasureCtx {
                    value: value.clone(),
                    font_size: crate::render::BADGE_FONT_SIZE,
                },
            )
            .unwrap(),
        Element::Button { label, .. } => taffy
            .new_leaf_with_context(
                taffy::Style {
                    padding: Rect {
                        left: LengthPercentage::length(crate::render::BUTTON_PAD_X),
                        right: LengthPercentage::length(crate::render::BUTTON_PAD_X),
                        top: LengthPercentage::length(crate::render::BUTTON_PAD_Y),
                        bottom: LengthPercentage::length(crate::render::BUTTON_PAD_Y),
                    },
                    ..taffy::Style::DEFAULT
                },
                MeasureCtx {
                    value: label.clone(),
                    font_size: crate::render::DEFAULT_FONT_SIZE,
                },
            )
            .unwrap(),
        Element::Link { label, .. } => taffy
            .new_leaf_with_context(
                taffy::Style::DEFAULT,
                MeasureCtx {
                    value: label.clone(),
                    font_size: crate::render::DEFAULT_FONT_SIZE,
                },
            )
            .unwrap(),
        Element::Bar { width, height, .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(*height),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Swatch { size, .. } | Element::SwatchButton { size, .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*size),
                    height: Dimension::length(*size),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Tabs { items, .. } => {
            // Tabs render as an hstack of tab cells; build a row of
            // text-shaped leaves so Taffy gives each one its own slot.
            let cell_kids: Vec<NodeId> = items
                .iter()
                .map(|t| {
                    taffy
                        .new_leaf_with_context(
                            taffy::Style {
                                padding: Rect {
                                    left: LengthPercentage::length(crate::render::TAB_PAD_X),
                                    right: LengthPercentage::length(crate::render::TAB_PAD_X),
                                    top: LengthPercentage::length(crate::render::TAB_PAD_Y),
                                    bottom: LengthPercentage::length(
                                        crate::render::TAB_PAD_Y + crate::render::TAB_INDICATOR_H,
                                    ),
                                },
                                ..taffy::Style::DEFAULT
                            },
                            MeasureCtx {
                                value: t.label.clone(),
                                font_size: crate::render::DEFAULT_FONT_SIZE,
                            },
                        )
                        .unwrap()
                })
                .collect();
            taffy
                .new_with_children(
                    taffy::Style {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Row,
                        gap: Size {
                            width: LengthPercentage::length(crate::render::TAB_GAP),
                            height: LengthPercentage::length(0.0),
                        },
                        ..taffy::Style::DEFAULT
                    },
                    &cell_kids,
                )
                .unwrap()
        }
        Element::Toggle { label, .. } => {
            // The toggle is its own bounding box: track + optional label.
            let track_w = crate::render::TOGGLE_TRACK_W;
            let track_h = crate::render::TOGGLE_TRACK_H;
            if label.is_empty() {
                taffy
                    .new_leaf(taffy::Style {
                        size: Size {
                            width: Dimension::length(track_w),
                            height: Dimension::length(track_h),
                        },
                        ..taffy::Style::DEFAULT
                    })
                    .unwrap()
            } else {
                let label_node = taffy
                    .new_leaf_with_context(
                        taffy::Style::DEFAULT,
                        MeasureCtx {
                            value: label.clone(),
                            font_size: crate::render::DEFAULT_FONT_SIZE,
                        },
                    )
                    .unwrap();
                let track_node = taffy
                    .new_leaf(taffy::Style {
                        size: Size {
                            width: Dimension::length(track_w),
                            height: Dimension::length(track_h),
                        },
                        flex_shrink: 0.0,
                        ..taffy::Style::DEFAULT
                    })
                    .unwrap();
                taffy
                    .new_with_children(
                        taffy::Style {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: Some(AlignItems::Center),
                            gap: Size {
                                width: LengthPercentage::length(8.0),
                                height: LengthPercentage::length(0.0),
                            },
                            ..taffy::Style::DEFAULT
                        },
                        &[label_node, track_node],
                    )
                    .unwrap()
            }
        }
        Element::Input { width, .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(crate::render::INPUT_HEIGHT),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Canvas { .. } => taffy.new_leaf(taffy::Style::DEFAULT).unwrap(),
    }
}

/// Build a Taffy style for a flex stack (vstack / hstack / frame /
/// scroll / list-item).
fn stack_style(gap: f32, pad: f32, style: Option<&PStyle>, dir: FlexDirection) -> taffy::Style {
    let padding = effective_padding(style, pad);
    let margin = effective_margin(style);
    let mut s = taffy::Style {
        display: Display::Flex,
        flex_direction: dir,
        gap: Size {
            width: LengthPercentage::length(gap),
            height: LengthPercentage::length(gap),
        },
        padding: rect_from(padding),
        margin: rect_from_signed(margin),
        ..taffy::Style::DEFAULT
    };
    apply_style_overrides(&mut s, style);
    s
}

fn effective_padding(style: Option<&PStyle>, pad: f32) -> Edges {
    style
        .and_then(|s| s.padding.as_ref())
        .copied()
        .unwrap_or_else(|| Edges::all(pad))
}

fn effective_margin(style: Option<&PStyle>) -> Edges {
    style
        .and_then(|s| s.margin.as_ref())
        .copied()
        .unwrap_or_default()
}

fn rect_from(e: Edges) -> Rect<LengthPercentage> {
    Rect {
        left: LengthPercentage::length(e.left),
        right: LengthPercentage::length(e.right),
        top: LengthPercentage::length(e.top),
        bottom: LengthPercentage::length(e.bottom),
    }
}

fn rect_from_signed(e: Edges) -> Rect<LengthPercentageAuto> {
    Rect {
        left: LengthPercentageAuto::length(e.left),
        right: LengthPercentageAuto::length(e.right),
        top: LengthPercentageAuto::length(e.top),
        bottom: LengthPercentageAuto::length(e.bottom),
    }
}

/// Pull min/max/explicit size from the Style overrides.
fn apply_style_overrides(s: &mut taffy::Style, style: Option<&PStyle>) {
    let Some(style) = style else { return };
    if let Some(g) = style.flex_grow {
        s.flex_grow = g;
    }
    if let Some(sh) = style.flex_shrink {
        s.flex_shrink = sh;
    }
    if let Some(w) = style.width.as_deref().and_then(parse_dimension) {
        s.size.width = w;
    }
    if let Some(h) = style.height.as_deref().and_then(parse_dimension) {
        s.size.height = h;
    }
    if let Some(mw) = style.min_width.as_deref().and_then(parse_dimension) {
        s.min_size.width = mw;
    }
    if let Some(mh) = style.min_height.as_deref().and_then(parse_dimension) {
        s.min_size.height = mh;
    }
    if let Some(mw) = style.max_width.as_deref().and_then(parse_dimension) {
        s.max_size.width = mw;
    }
    if let Some(mh) = style.max_height.as_deref().and_then(parse_dimension) {
        s.max_size.height = mh;
    }
    if let Some(a) = style.align_self {
        s.align_self = Some(align_to_taffy(a));
    }
}

/// Parse a Style width/height/min/max string.
/// - `"123"` or `"123.5"` → pixels
/// - `"50%"` → percent of parent
/// - `"auto"` → intrinsic
/// Returns `None` for unparseable values; the caller leaves the default.
fn parse_dimension(s: &str) -> Option<Dimension> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("auto") {
        return Some(Dimension::auto());
    }
    if let Some(rest) = t.strip_suffix('%') {
        return rest.trim().parse::<f32>().ok().map(|n| Dimension::percent(n / 100.0));
    }
    t.parse::<f32>().ok().map(Dimension::length)
}

fn align_to_taffy(a: Align) -> AlignItems {
    match a {
        Align::Start => AlignItems::FlexStart,
        Align::Center => AlignItems::Center,
        Align::End => AlignItems::FlexEnd,
        Align::Stretch => AlignItems::Stretch,
    }
}

/// Compute layout for the tree rooted at `root` within the given
/// `(max_w, max_h)` viewport. `metrics` is used to size text leaves.
pub fn compute(laid: &mut LaidOut, max_w: f32, max_h: f32, metrics: &PaneFontMetrics) {
    let m = *metrics;
    laid.taffy
        .compute_layout_with_measure(
            laid.root,
            Size {
                width: AvailableSpace::Definite(max_w),
                height: AvailableSpace::Definite(max_h),
            },
            move |known, available, _node, context, _style| {
                let Some(ctx) = context else {
                    return Size::ZERO;
                };
                measure_text(ctx, known, available, &m)
            },
        )
        .expect("taffy compute_layout");
}

/// Measure callback for text + button + badge leaves. Wraps at the
/// available width when the text overflows; returns (width, height).
fn measure_text(
    ctx: &MeasureCtx,
    known: Size<Option<f32>>,
    available: Size<AvailableSpace>,
    metrics: &PaneFontMetrics,
) -> Size<f32> {
    let line_h = crate::render::line_height(ctx.font_size);
    if let (Some(w), Some(h)) = (known.width, known.height) {
        return Size { width: w, height: h };
    }
    // Available width caps the wrap. When unbounded (MIN/MAX_CONTENT),
    // treat as infinite for word-wrap purposes — the intrinsic single-
    // line width is returned and Taffy picks the bounding.
    let max_w = match available.width {
        AvailableSpace::Definite(w) => w,
        AvailableSpace::MinContent => 0.0,
        AvailableSpace::MaxContent => f32::INFINITY,
    };
    let intrinsic_w = metrics.measure(&ctx.value, ctx.font_size);
    if intrinsic_w <= max_w {
        return Size {
            width: intrinsic_w,
            height: line_h,
        };
    }
    // Word-wrap: break on whitespace, accumulate words per line until
    // the next word would overflow, then start a new line.
    let char_w = metrics.char_width(ctx.font_size);
    if char_w <= 0.0 || max_w <= 0.0 {
        return Size {
            width: intrinsic_w,
            height: line_h,
        };
    }
    let mut lines: u32 = 1;
    let mut max_line_w: f32 = 0.0;
    let mut line_w: f32 = 0.0;
    let mut first_word = true;
    for word in ctx.value.split_whitespace() {
        let w = word.chars().count() as f32 * char_w;
        let added = if first_word { w } else { char_w + w };
        if !first_word && line_w + added > max_w {
            max_line_w = max_line_w.max(line_w);
            lines += 1;
            line_w = w;
        } else {
            line_w += added;
            first_word = false;
        }
    }
    max_line_w = max_line_w.max(line_w);
    Size {
        width: max_line_w.min(max_w),
        height: line_h * lines as f32,
    }
}

/// Convenience: absolute position of a node within the root's coord
/// frame. Taffy stores per-node `location` relative to the node's
/// parent; absolute position requires summing across the ancestor
/// chain. The renderer walks the tree top-down anyway, so it carries
/// the running origin itself — this helper exists for ad-hoc
/// queries (e.g. snapshot tools).
pub fn absolute_position(laid: &LaidOut, mut node: NodeId, root: NodeId) -> Vec2 {
    let mut total = Vec2::ZERO;
    while node != root {
        let layout = laid.layout(node);
        total.x += layout.location.x;
        total.y += layout.location.y;
        let Some(parent) = laid.taffy.parent(node) else {
            break;
        };
        node = parent;
    }
    let root_layout = laid.layout(root);
    total.x += root_layout.location.x;
    total.y += root_layout.location.y;
    total
}

// Compile-time checks that the Style overrides import compiles
// against the protocol types we expect.
#[allow(dead_code)]
fn _check_style_imports(_b: Border, _sh: Shadow, _w: Weight, _k: ButtonKind) {}
