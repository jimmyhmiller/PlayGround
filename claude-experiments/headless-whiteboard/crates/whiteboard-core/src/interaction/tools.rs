//! Per-tool element creation for the drag-to-create gesture.
//!
//! Reimplemented from Excalidraw's new-element logic
//! (`packages/excalidraw/element/newElement.ts` and the pointer-down branch of
//! `App.tsx`'s `handleCanvasPointerDown`). A pointer-down with a creation tool
//! spawns an element seeded from the start point; pointer-move reshapes it; and
//! pointer-up commits it. Generic shapes (rectangle/ellipse/diamond) grow a box;
//! linear tools (line/arrow) track a two-point path; freedraw accumulates points.
//!
//! Element ids and RNG seeds are supplied by the caller (the interaction state's
//! deterministic allocator) so creation stays free of hidden global state — the
//! same requirement `Element::new` documents.

use crate::element::{Element, ElementId, ElementKind, FreedrawData, LinearData};
use crate::geometry::Point;
use crate::interaction::Tool;

/// Whether a tool creates an element by dragging.
pub fn is_creation_tool(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::Rectangle | Tool::Ellipse | Tool::Diamond | Tool::Line | Tool::Arrow | Tool::Freedraw
    )
}

/// The geometric family a creation tool produces, which decides how
/// pointer-move updates the in-progress element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateKind {
    /// Grows an axis-aligned box from the start corner (rect/ellipse/diamond).
    Box,
    /// A two-point line/arrow from the start anchor to the cursor.
    Linear,
    /// A freehand stroke accumulating every move point.
    Freedraw,
}

/// Classify a creation tool. Returns `None` for non-creation tools.
pub fn create_kind(tool: Tool) -> Option<CreateKind> {
    match tool {
        Tool::Rectangle | Tool::Ellipse | Tool::Diamond => Some(CreateKind::Box),
        Tool::Line | Tool::Arrow => Some(CreateKind::Linear),
        Tool::Freedraw => Some(CreateKind::Freedraw),
        _ => None,
    }
}

/// Build the initial element for `tool` at `start` (scene coords).
///
/// The element starts degenerate (zero size, or a single-point path); the
/// drag-update functions reshape it as the pointer moves. Returns `None` if
/// `tool` is not a creation tool.
pub fn begin_element(tool: Tool, id: ElementId, seed: u32, start: Point) -> Option<Element> {
    let kind = match tool {
        Tool::Rectangle => ElementKind::Rectangle,
        Tool::Ellipse => ElementKind::Ellipse,
        Tool::Diamond => ElementKind::Diamond,
        Tool::Line => ElementKind::Line(LinearData::line(vec![Point::ORIGIN, Point::ORIGIN])),
        Tool::Arrow => ElementKind::Arrow(LinearData::arrow(vec![Point::ORIGIN, Point::ORIGIN])),
        Tool::Freedraw => ElementKind::Freedraw(FreedrawData::new(vec![Point::ORIGIN])),
        _ => return None,
    };
    Some(Element::new(id, seed, start.x, start.y, 0.0, 0.0, kind))
}

/// Update a box-family element so its opposite corner is `cursor`.
///
/// `start` is the anchor corner the drag began at. With `keep_square` the box is
/// constrained to a square sized by the larger axis delta (Excalidraw's
/// shift-to-constrain behavior).
pub fn update_box(el: &mut Element, start: Point, cursor: Point, keep_square: bool) {
    let mut dx = cursor.x - start.x;
    let mut dy = cursor.y - start.y;
    if keep_square {
        let size = dx.abs().max(dy.abs());
        dx = size * if dx < 0.0 { -1.0 } else { 1.0 };
        dy = size * if dy < 0.0 { -1.0 } else { 1.0 };
    }
    // Normalize so x/y stay the min corner even when dragging up/left.
    let (x, w) = if dx < 0.0 {
        (start.x + dx, -dx)
    } else {
        (start.x, dx)
    };
    let (y, h) = if dy < 0.0 {
        (start.y + dy, -dy)
    } else {
        (start.y, dy)
    };
    el.x = x;
    el.y = y;
    el.width = w;
    el.height = h;
}

/// Update a linear (line/arrow) element's endpoint to `cursor`.
///
/// The element origin stays at `start`; the path is `[(0,0), cursor-start]`.
/// With `keep_axis` the segment snaps to the nearest of horizontal / vertical /
/// 45-degree diagonal (Excalidraw's shift-constrain for linear drags).
pub fn update_linear(el: &mut Element, start: Point, cursor: Point, keep_axis: bool) {
    let mut ex = cursor.x - start.x;
    let mut ey = cursor.y - start.y;
    if keep_axis {
        (ex, ey) = snap_axis(ex, ey);
    }
    let pts = linear_points_mut(el);
    pts.clear();
    pts.push(Point::ORIGIN);
    pts.push(Point::new(ex, ey));
    sync_linear_box(el, start);
}

/// Append a point to a freedraw element (scene coords).
pub fn push_freedraw(el: &mut Element, start: Point, cursor: Point) {
    let rel = Point::new(cursor.x - start.x, cursor.y - start.y);
    if let ElementKind::Freedraw(fd) = &mut el.kind {
        // Skip a duplicate of the last point to avoid zero-length segments.
        if fd.points.last().map(|p| *p != rel).unwrap_or(true) {
            fd.points.push(rel);
        }
    }
    sync_linear_box(el, start);
}

/// Snap a delta to horizontal, vertical, or 45-degree diagonal — whichever the
/// vector is closest to.
fn snap_axis(dx: f64, dy: f64) -> (f64, f64) {
    let adx = dx.abs();
    let ady = dy.abs();
    // Closer to an axis than to the diagonal?
    if adx > ady * 2.0 {
        (dx, 0.0) // horizontal
    } else if ady > adx * 2.0 {
        (0.0, dy) // vertical
    } else {
        let m = adx.max(ady);
        (m * dx.signum(), m * dy.signum()) // diagonal
    }
}

/// Mutable access to a linear element's point list.
fn linear_points_mut(el: &mut Element) -> &mut Vec<Point> {
    match &mut el.kind {
        ElementKind::Line(l) | ElementKind::Arrow(l) => &mut l.points,
        _ => panic!("update_linear called on a non-linear element"),
    }
}

/// Recompute `width`/`height` (and origin) of a point-based element from its
/// points, keeping the element origin at `origin`.
///
/// Excalidraw stores point-based elements with the origin at the box top-left and
/// points possibly extending into negative coordinates; it then re-bases points
/// so the smallest point is `(0,0)`. We do the same so `raw_box()` is correct.
fn sync_linear_box(el: &mut Element, origin: Point) {
    let pts: Vec<Point> = match &el.kind {
        ElementKind::Line(l) | ElementKind::Arrow(l) => l.points.clone(),
        ElementKind::Freedraw(f) => f.points.clone(),
        _ => return,
    };
    if pts.is_empty() {
        return;
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in &pts {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    // Re-base points so the minimum is at (0,0), and shift the origin to match.
    if min_x != 0.0 || min_y != 0.0 {
        let shift = Point::new(min_x, min_y);
        match &mut el.kind {
            ElementKind::Line(l) | ElementKind::Arrow(l) => {
                for p in &mut l.points {
                    p.x -= shift.x;
                    p.y -= shift.y;
                }
            }
            ElementKind::Freedraw(f) => {
                for p in &mut f.points {
                    p.x -= shift.x;
                    p.y -= shift.y;
                }
            }
            _ => {}
        }
        el.x = origin.x + min_x;
        el.y = origin.y + min_y;
    } else {
        el.x = origin.x;
        el.y = origin.y;
    }
    el.width = max_x - min_x;
    el.height = max_y - min_y;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_creation_tools() {
        assert!(is_creation_tool(Tool::Rectangle));
        assert!(is_creation_tool(Tool::Arrow));
        assert!(is_creation_tool(Tool::Freedraw));
        assert!(!is_creation_tool(Tool::Select));
        assert!(!is_creation_tool(Tool::Pan));
        assert_eq!(create_kind(Tool::Ellipse), Some(CreateKind::Box));
        assert_eq!(create_kind(Tool::Line), Some(CreateKind::Linear));
        assert_eq!(create_kind(Tool::Freedraw), Some(CreateKind::Freedraw));
        assert_eq!(create_kind(Tool::Text), None);
    }

    #[test]
    fn begin_rectangle_is_degenerate() {
        let el = begin_element(
            Tool::Rectangle,
            ElementId::from("a"),
            1,
            Point::new(5.0, 6.0),
        )
        .unwrap();
        assert_eq!(el.x, 5.0);
        assert_eq!(el.y, 6.0);
        assert_eq!(el.width, 0.0);
        assert_eq!(el.height, 0.0);
        assert_eq!(el.type_name(), "rectangle");
    }

    #[test]
    fn box_update_grows_and_normalizes() {
        let start = Point::new(10.0, 10.0);
        let mut el = begin_element(Tool::Rectangle, ElementId::from("a"), 1, start).unwrap();
        // drag down-right
        update_box(&mut el, start, Point::new(40.0, 30.0), false);
        assert_eq!(
            el.raw_box(),
            crate::geometry::Rect::new(10.0, 10.0, 30.0, 20.0)
        );
        // drag up-left past the origin: box normalizes to min corner
        update_box(&mut el, start, Point::new(0.0, 5.0), false);
        assert_eq!(el.x, 0.0);
        assert_eq!(el.y, 5.0);
        assert_eq!(el.width, 10.0);
        assert_eq!(el.height, 5.0);
    }

    #[test]
    fn box_square_constraint() {
        let start = Point::new(0.0, 0.0);
        let mut el = begin_element(Tool::Rectangle, ElementId::from("a"), 1, start).unwrap();
        update_box(&mut el, start, Point::new(50.0, 20.0), true);
        assert_eq!(el.width, 50.0);
        assert_eq!(el.height, 50.0);
    }

    #[test]
    fn linear_tracks_endpoint() {
        let start = Point::new(100.0, 100.0);
        let mut el = begin_element(Tool::Arrow, ElementId::from("a"), 1, start).unwrap();
        update_linear(&mut el, start, Point::new(160.0, 140.0), false);
        // origin stays at start; box spans the delta
        assert_eq!(el.x, 100.0);
        assert_eq!(el.y, 100.0);
        assert_eq!(el.width, 60.0);
        assert_eq!(el.height, 40.0);
        if let ElementKind::Arrow(l) = &el.kind {
            assert_eq!(l.points, vec![Point::ORIGIN, Point::new(60.0, 40.0)]);
        } else {
            panic!("not an arrow");
        }
    }

    #[test]
    fn linear_drag_up_left_rebases_origin() {
        let start = Point::new(100.0, 100.0);
        let mut el = begin_element(Tool::Line, ElementId::from("a"), 1, start).unwrap();
        update_linear(&mut el, start, Point::new(60.0, 70.0), false);
        // endpoint is up-left of start, so the box origin moves to the endpoint
        assert_eq!(el.x, 60.0);
        assert_eq!(el.y, 70.0);
        assert_eq!(el.width, 40.0);
        assert_eq!(el.height, 30.0);
        if let ElementKind::Line(l) = &el.kind {
            // start point is now at (40,30), endpoint at (0,0)
            assert_eq!(l.points, vec![Point::new(40.0, 30.0), Point::ORIGIN]);
        } else {
            panic!();
        }
    }

    #[test]
    fn linear_axis_snap_horizontal() {
        let start = Point::ORIGIN;
        let mut el = begin_element(Tool::Line, ElementId::from("a"), 1, start).unwrap();
        update_linear(&mut el, start, Point::new(100.0, 5.0), true);
        if let ElementKind::Line(l) = &el.kind {
            assert_eq!(l.points[1], Point::new(100.0, 0.0));
        } else {
            panic!();
        }
    }

    #[test]
    fn freedraw_accumulates_points() {
        let start = Point::new(0.0, 0.0);
        let mut el = begin_element(Tool::Freedraw, ElementId::from("a"), 1, start).unwrap();
        push_freedraw(&mut el, start, Point::new(10.0, 0.0));
        push_freedraw(&mut el, start, Point::new(10.0, 10.0));
        push_freedraw(&mut el, start, Point::new(10.0, 10.0)); // duplicate ignored
        if let ElementKind::Freedraw(f) = &el.kind {
            assert_eq!(
                f.points,
                vec![Point::ORIGIN, Point::new(10.0, 0.0), Point::new(10.0, 10.0)]
            );
        } else {
            panic!();
        }
        assert_eq!(el.width, 10.0);
        assert_eq!(el.height, 10.0);
    }
}
