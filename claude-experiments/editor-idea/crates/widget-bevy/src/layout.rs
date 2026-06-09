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
/// `metrics` is used to pre-size leaves whose height depends on wrapped
/// text (currently `TextArea`).
pub fn build_tree(el: &Element, metrics: &PaneFontMetrics) -> LaidOut {
    let mut taffy = TaffyTree::new();
    let root = build_node(&mut taffy, el, metrics);
    LaidOut { taffy, root }
}

fn build_node(
    taffy: &mut TaffyTree<MeasureCtx>,
    el: &Element,
    metrics: &PaneFontMetrics,
) -> NodeId {
    match el {
        Element::Vstack {
            gap,
            pad,
            children,
            style,
        } => {
            let st = stack_style(*gap, *pad, style.as_ref(), FlexDirection::Column);
            let kids: Vec<NodeId> = children
                .iter()
                .map(|c| build_node(taffy, c, metrics))
                .collect();
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
            let kids: Vec<NodeId> = children
                .iter()
                .map(|c| build_node(taffy, c, metrics))
                .collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::Frame {
            gap,
            pad,
            children,
            style,
        } => {
            let st = stack_style(*gap, *pad, style.as_ref(), FlexDirection::Column);
            let kids: Vec<NodeId> = children
                .iter()
                .map(|c| build_node(taffy, c, metrics))
                .collect();
            taffy.new_with_children(st, &kids).unwrap()
        }
        Element::Scroll { gap, pad, children } => {
            let st = stack_style(*gap, *pad, None, FlexDirection::Column);
            let kids: Vec<NodeId> = children
                .iter()
                .map(|c| build_node(taffy, c, metrics))
                .collect();
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
            let kids: Vec<NodeId> = children
                .iter()
                .map(|c| build_node(taffy, c, metrics))
                .collect();
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
        Element::Tooltip { label, .. } => taffy
            .new_leaf_with_context(
                taffy::Style::DEFAULT,
                MeasureCtx {
                    value: label.clone(),
                    font_size: crate::render::DEFAULT_FONT_SIZE,
                },
            )
            .unwrap(),
        Element::Bar { width, height, .. } | Element::Slider { width, height, .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(*height),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Stepper { .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(crate::render::STEPPER_W),
                    height: Dimension::length(crate::render::STEPPER_H),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Select { width, .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(crate::render::SELECT_H),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        Element::Popover { width, .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(crate::render::SELECT_H),
                },
                ..taffy::Style::DEFAULT
            })
            .unwrap(),
        // Dialog + Toast have no in-pane footprint — they render on the overlay.
        Element::Dialog { .. } | Element::Toast { .. } => taffy
            .new_leaf(taffy::Style {
                size: Size {
                    width: Dimension::length(0.0),
                    height: Dimension::length(0.0),
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
        Element::RadioGroup { options, .. } => {
            // A column of option rows. Each cell measures its label with a left
            // padding reserving room for the ring (drawn into that padding).
            let cells: Vec<NodeId> = options
                .iter()
                .map(|o| {
                    taffy
                        .new_leaf_with_context(
                            taffy::Style {
                                padding: Rect {
                                    left: LengthPercentage::length(
                                        crate::render::RADIO_RING + crate::render::RADIO_GAP,
                                    ),
                                    right: LengthPercentage::length(0.0),
                                    top: LengthPercentage::length(crate::render::RADIO_PAD_Y),
                                    bottom: LengthPercentage::length(crate::render::RADIO_PAD_Y),
                                },
                                ..taffy::Style::DEFAULT
                            },
                            MeasureCtx {
                                value: o.label.clone(),
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
                        flex_direction: FlexDirection::Column,
                        gap: Size {
                            width: LengthPercentage::length(0.0),
                            height: LengthPercentage::length(crate::render::RADIO_GROUP_GAP),
                        },
                        ..taffy::Style::DEFAULT
                    },
                    &cells,
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
        Element::Checkbox { label, .. } => {
            // box + optional label, laid out as a centered row (like Toggle).
            let box_d = crate::render::CHECKBOX_SIZE;
            if label.is_empty() {
                taffy
                    .new_leaf(taffy::Style {
                        size: Size {
                            width: Dimension::length(box_d),
                            height: Dimension::length(box_d),
                        },
                        ..taffy::Style::DEFAULT
                    })
                    .unwrap()
            } else {
                let box_node = taffy
                    .new_leaf(taffy::Style {
                        size: Size {
                            width: Dimension::length(box_d),
                            height: Dimension::length(box_d),
                        },
                        flex_shrink: 0.0,
                        ..taffy::Style::DEFAULT
                    })
                    .unwrap();
                let label_node = taffy
                    .new_leaf_with_context(
                        taffy::Style::DEFAULT,
                        MeasureCtx {
                            value: label.clone(),
                            font_size: crate::render::DEFAULT_FONT_SIZE,
                        },
                    )
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
                        &[box_node, label_node],
                    )
                    .unwrap()
            }
        }
        Element::Input { width, style, .. } => {
            // The `width` field is the DEFAULT; `style` (flex_grow,
            // width/height incl. `"100%"`, min/max) overrides it so an
            // input can fill / grow within its pane like any stack does.
            let mut s = taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(crate::render::INPUT_HEIGHT),
                },
                ..taffy::Style::DEFAULT
            };
            apply_style_overrides(&mut s, style.as_ref());
            taffy.new_leaf(s).unwrap()
        }
        Element::TextArea {
            width,
            rows,
            value,
            style,
            ..
        } => {
            // Auto-grow: height fits the wrapped content, with `rows` as
            // the minimum. Wrapping here uses the same routine + width as
            // the renderer, so the box height matches the drawn text.
            // (While typing, this tracks the element's `value`, which a
            // widget refreshes from `on_input_change`.)
            let line_h = crate::render::line_height(crate::render::DEFAULT_FONT_SIZE);
            let avail = (*width - 2.0 * crate::render::INPUT_PAD_X).max(1.0);
            let chars: Vec<char> = value.chars().collect();
            let wrapped = crate::render::wrap_visual_lines(&chars, metrics, avail).len() as u32;
            let lines = wrapped.max((*rows).max(1));
            let height = lines as f32 * line_h + 2.0 * crate::render::TEXTAREA_PAD_Y;
            // `width`/`rows` are the DEFAULTS; `style` overrides them so a
            // query editor can `flex_grow` / `height: "100%"` to fill a
            // docked editor pane instead of staying a fixed rows-tall box.
            let mut s = taffy::Style {
                size: Size {
                    width: Dimension::length(*width),
                    height: Dimension::length(height),
                },
                ..taffy::Style::DEFAULT
            };
            apply_style_overrides(&mut s, style.as_ref());
            taffy.new_leaf(s).unwrap()
        }
        Element::Table { columns, rows, .. } => {
            use taffy::style::{
                GridTemplateComponent, MaxTrackSizingFunction, MinTrackSizingFunction,
            };
            // Max width an auto (unsized) column grows to before its text
            // wraps. Keeps a long cell from ballooning the whole table.
            const COL_CAP: f32 = 260.0;
            let ncols = columns.len().max(1);
            // One grid track per column. A fixed `width` becomes a rigid
            // track; otherwise the column fits its content, capped at
            // COL_CAP (longer text wraps). We avoid `fr` here because the
            // widget layout root is content-sized, so `fr` has no
            // definite width to distribute and one column would balloon.
            let mut tracks: Vec<GridTemplateComponent<_>> = columns
                .iter()
                .map(|c| {
                    let track = match c.width {
                        Some(w) => minmax(
                            MinTrackSizingFunction::length(w),
                            MaxTrackSizingFunction::length(w),
                        ),
                        // Grow to content, but cap at COL_CAP so a long
                        // cell wraps instead of stretching the table. A
                        // fixed-length max caps even when the grid has no
                        // definite outer width (unlike `fit-content`).
                        None => minmax(
                            MinTrackSizingFunction::auto(),
                            MaxTrackSizingFunction::length(COL_CAP),
                        ),
                    };
                    GridTemplateComponent::Single(track)
                })
                .collect();
            if tracks.is_empty() {
                tracks.push(GridTemplateComponent::Single(minmax(
                    MinTrackSizingFunction::auto(),
                    MaxTrackSizingFunction::length(COL_CAP),
                )));
            }
            let st = taffy::Style {
                display: Display::Grid,
                grid_template_columns: tracks,
                // Column gap so adjacent cells (esp. a wrapped cell next
                // to a right-aligned one) don't visually touch.
                gap: Size {
                    width: LengthPercentage::length(crate::render::TABLE_COL_GAP),
                    height: LengthPercentage::length(0.0),
                },
                ..taffy::Style::DEFAULT
            };
            // Cells in row-major order: header row first, then data rows.
            // Each is a measured text leaf so it wraps within its column
            // and the row track grows to the tallest cell.
            let mut cells: Vec<NodeId> = Vec::with_capacity(ncols * (rows.len() + 1));
            for c in columns {
                cells.push(table_cell_leaf(taffy, &c.header));
            }
            for row in rows {
                for ci in 0..ncols {
                    let txt = row.get(ci).map(|s| s.as_str()).unwrap_or("");
                    cells.push(table_cell_leaf(taffy, txt));
                }
            }
            taffy.new_with_children(st, &cells).unwrap()
        }
        Element::Canvas { .. } => taffy.new_leaf(taffy::Style::DEFAULT).unwrap(),
    }
}

/// One table cell: a measured text leaf with cell padding, so it wraps
/// to its column width and contributes its height to the row track.
fn table_cell_leaf(taffy: &mut TaffyTree<MeasureCtx>, text: &str) -> NodeId {
    taffy
        .new_leaf_with_context(
            taffy::Style {
                padding: Rect {
                    left: LengthPercentage::length(crate::render::TABLE_CELL_PAD_X),
                    right: LengthPercentage::length(crate::render::TABLE_CELL_PAD_X),
                    top: LengthPercentage::length(crate::render::TABLE_CELL_PAD_Y),
                    bottom: LengthPercentage::length(crate::render::TABLE_CELL_PAD_Y),
                },
                ..taffy::Style::DEFAULT
            },
            MeasureCtx {
                value: text.to_string(),
                font_size: crate::render::DEFAULT_FONT_SIZE,
            },
        )
        .unwrap()
}

/// Build a Taffy style for a flex stack (vstack / hstack / frame /
/// scroll / list-item).
fn stack_style(gap: f32, pad: f32, style: Option<&PStyle>, dir: FlexDirection) -> taffy::Style {
    let padding = effective_padding(style, pad);
    let margin = effective_margin(style);
    // A Glaze `direction` override (e.g. from a `when` breakpoint) can flip the
    // container between row and column independent of the Element variant.
    let dir = match style.and_then(|s| s.flex_direction.as_deref()) {
        Some("row") => FlexDirection::Row,
        Some("column") | Some("col") => FlexDirection::Column,
        _ => dir,
    };
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
        return rest
            .trim()
            .parse::<f32>()
            .ok()
            .map(|n| Dimension::percent(n / 100.0));
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
    let _prof = pane_bevy::prof::sys_span_nested("taffy_layout");
    // Force the root to fill the available content width. Without this the root
    // (auto width) shrinks to its content, so `grow`/stretch children have no
    // free space to distribute and text leaves measure/wrap at a collapsed
    // width. Pinning the root to the content width makes flex layouts reflow as
    // the pane resizes, and text wrap at the true available width.
    if let Ok(style) = laid.taffy.style(laid.root) {
        let mut s = style.clone();
        s.size.width = Dimension::length(max_w);
        let _ = laid.taffy.set_style(laid.root, s);
    }
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
        return Size {
            width: w,
            height: h,
        };
    }
    let intrinsic_w = metrics.measure(&ctx.value, ctx.font_size);
    let char_w = metrics.char_width(ctx.font_size);
    // Min-content width is the widest *unbreakable word*, NOT the whole line.
    // If a text leaf reported its full single-line width as its minimum, flex
    // containers could never shrink it — so content would always overflow the
    // pane and the layout wouldn't reflow on resize. With the longest word as
    // the floor, text shrinks and wraps, and containers can adapt to width.
    let longest_word = ctx
        .value
        .split_whitespace()
        .map(|w| w.chars().count() as f32 * char_w)
        .fold(0.0_f32, f32::max);
    let max_w = match available.width {
        AvailableSpace::Definite(w) => w,
        AvailableSpace::MinContent => longest_word.max(char_w),
        AvailableSpace::MaxContent => f32::INFINITY,
    };
    if intrinsic_w <= max_w {
        return Size {
            width: intrinsic_w,
            height: line_h,
        };
    }
    // Word-wrap: break on whitespace, accumulate words per line until
    // the next word would overflow, then start a new line.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Style;

    fn metrics() -> PaneFontMetrics {
        PaneFontMetrics {
            cell_width: 8.4,
            font_size: 14.0,
        }
    }

    fn pct_height(p: &str) -> Style {
        Style {
            height: Some(p.into()),
            ..Default::default()
        }
    }

    /// A root vstack with `height:"100%"` and a `flex_grow:1` child fills
    /// the available pane height — the whole point of passing the known
    /// pane height (content_size.y) instead of INFINITY. With INFINITY,
    /// the percentage and flex have nothing to distribute and the child
    /// stays content-tall.
    #[test]
    fn root_fills_pane_height_when_height_100pct() {
        let m = metrics();
        let pane_h = 600.0_f32;
        let el = Element::Vstack {
            gap: 0.0,
            pad: 0.0,
            style: Some(pct_height("100%")),
            children: vec![Element::Frame {
                gap: 0.0,
                pad: 0.0,
                style: Some(Style {
                    flex_grow: Some(1.0),
                    ..Default::default()
                }),
                children: vec![],
            }],
        };
        let mut laid = build_tree(&el, &m);
        compute(&mut laid, 400.0, pane_h, &m);

        let root = laid.layout(laid.root);
        assert!(
            (root.size.height - pane_h).abs() < 0.5,
            "root should fill pane height {pane_h}, got {}",
            root.size.height
        );
        let child = laid.taffy.children(laid.root).unwrap()[0];
        let cl = laid.layout(child);
        assert!(
            (cl.size.height - pane_h).abs() < 0.5,
            "flex_grow child should fill {pane_h}, got {}",
            cl.size.height
        );
    }

    /// The table render fix paints the panel/rows from the cells' CONTENT
    /// BOX (union of laid-out cell extents) rather than the node `size`.
    /// This guards the two properties that make that correct in every
    /// case: (1) the content box bounds every cell (so no column can ever
    /// paint outside the panel — the reported bug), and (2) when the grid
    /// tracks already fill the node, the content box equals the node, so
    /// the fix never shrinks a correctly-filled table.
    #[test]
    fn table_content_box_bounds_cells_and_matches_filled_node() {
        use crate::protocol::TableColumn;
        let m = metrics();
        let col = |h: &str, a| TableColumn {
            header: h.into(),
            width: None,
            align: a,
        };
        let table = Element::Table {
            columns: vec![col("name", Align::Start), col("age", Align::End)],
            rows: vec![
                vec!["Widget".into(), "30".into()],
                vec!["Gadget".into(), "25".into()],
            ],
            zebra: true,
            selectable: false,
            style: None,
        };
        let el = Element::Vstack {
            gap: 0.0,
            pad: 0.0,
            style: Some(Style {
                width: Some("400".into()),
                ..Default::default()
            }),
            children: vec![table],
        };
        let mut laid = build_tree(&el, &m);
        compute(&mut laid, 400.0, 600.0, &m);

        let table_node = laid.taffy.children(laid.root).unwrap()[0];
        let node_w = laid.layout(table_node).size.width;

        // Cells' content box, computed exactly like render_table_at.
        let cells = laid.taffy.children(table_node).unwrap();
        let mut content_x = 0.0_f32;
        for &c in &cells {
            let cl = laid.layout(c);
            let right = cl.location.x + cl.size.width;
            // (1) every cell is within the content box by construction.
            assert!(
                right <= content_x.max(right) + 0.01,
                "cell right edge {right} must be within content box"
            );
            content_x = content_x.max(right);
        }
        assert!(content_x > 0.0, "content box should be non-degenerate");
        // (2) non-destructive: when tracks fill the node, panel == cells.
        assert!(
            (content_x - node_w).abs() < 1.0,
            "content box ({content_x}) should match the filled node ({node_w}) \
             so the fix doesn't shrink a correct table"
        );
    }

    /// Default (no height set) still content-sizes, so taller-than-pane
    /// content grows past the pane and scrolls — passing the pane height
    /// as the available space must NOT clamp an auto-height root.
    #[test]
    fn root_without_height_still_content_sizes() {
        let m = metrics();
        // Three stacked fixed-height frames = 300px of content, in a
        // 100px pane. An auto-height root must report ~300, not 100.
        let frame = |h: f32| Element::Frame {
            gap: 0.0,
            pad: 0.0,
            style: Some(Style {
                height: Some(format!("{h}")),
                ..Default::default()
            }),
            children: vec![],
        };
        let el = Element::Vstack {
            gap: 0.0,
            pad: 0.0,
            style: None,
            children: vec![frame(100.0), frame(100.0), frame(100.0)],
        };
        let mut laid = build_tree(&el, &m);
        compute(&mut laid, 400.0, 100.0, &m);
        let root = laid.layout(laid.root);
        assert!(
            (root.size.height - 300.0).abs() < 0.5,
            "auto-height root should grow to content 300, got {}",
            root.size.height
        );
    }
}
