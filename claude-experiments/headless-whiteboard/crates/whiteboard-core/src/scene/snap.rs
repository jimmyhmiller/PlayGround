//! Object snapping with visual alignment guides (Excalidraw's pink guides).
//!
//! Reimplemented in Rust from Excalidraw's `packages/element/src/snapping.ts`
//! (the `getElementsCorners` / `getReferenceSnapPoints` / `snapDraggedElements`
//! machinery), simplified to the bounding-box case. We do not vendor any
//! JavaScript; the algorithm below is a from-scratch reimplementation.
//!
//! The idea: when a selection is being moved (or created), compare the three
//! candidate offsets on each axis — its left/center/right edges (X) and
//! top/center/bottom edges (Y) — against the same three reference lines of every
//! *non-moving* element. The closest pair within `threshold` wins per axis,
//! producing a tiny correction `offset` that lands the selection exactly on the
//! reference line, plus guide line segments spanning the combined extent of the
//! moving box and the element it snapped to.
//!
//! All coordinates are **scene** units. The functions are pure over `&Scene`.

use crate::element::ElementId;
use crate::geometry::{element_bounds, Point, Rect, Vec2};
use crate::scene::Scene;

/// Which axis a snap acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapAxis {
    /// Horizontal correction (a vertical guide line).
    X,
    /// Vertical correction (a horizontal guide line).
    Y,
}

/// A single alignment guide line segment in **scene** coordinates. The editor
/// maps these to screen space and draws them as thin guide-colored strokes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapGuide {
    pub a: Point,
    pub b: Point,
}

/// The result of snapping a moving/creating selection.
///
/// `offset` is the small correction to add to the moving elements' positions so
/// they land exactly on the nearest reference line(s); it is [`Vec2::ZERO`] when
/// nothing was within `threshold`. `guides` holds the guide line segments to draw
/// (empty when nothing snapped).
#[derive(Debug, Clone, PartialEq)]
pub struct SnapResult {
    pub offset: Vec2,
    pub guides: Vec<SnapGuide>,
}

impl SnapResult {
    /// No snap occurred.
    fn none() -> Self {
        SnapResult {
            offset: Vec2::ZERO,
            guides: Vec::new(),
        }
    }
}

/// The three reference coordinates of a rect on one axis: min edge, center,
/// max edge.
fn axis_lines(min: f64, max: f64) -> [f64; 3] {
    [min, (min + max) / 2.0, max]
}

/// The best snap found on a single axis: the signed correction to apply and the
/// reference coordinate it lands on.
#[derive(Clone, Copy)]
struct AxisSnap {
    /// Correction to add to the moving box on this axis (reference - moving).
    delta: f64,
    /// Absolute distance |delta|, for picking the nearest.
    dist: f64,
    /// The reference line coordinate the moving box lands on.
    line: f64,
    /// The non-moving element this snap aligned to (for guide extent).
    other: Rect,
}

/// Snap the moving selection (given its would-be `proposed_bounds`) to the
/// nearest non-moving element's edges/centers within `threshold`.
///
/// Returns the small `offset` that lands it on the snap plus guide line segments
/// spanning the aligned extent. Returns a zero offset and empty guides when
/// nothing is within `threshold`.
pub fn snap_moving(
    scene: &Scene,
    moving_ids: &[ElementId],
    proposed_bounds: Rect,
    threshold: f64,
) -> SnapResult {
    if proposed_bounds.is_empty() {
        return SnapResult::none();
    }

    // Candidate reference rects: every live element NOT in the moving set.
    let candidates: Vec<Rect> = scene
        .iter_live()
        .filter(|e| !moving_ids.contains(&e.id))
        .map(element_bounds)
        .collect();
    if candidates.is_empty() {
        return SnapResult::none();
    }

    let moving_x = axis_lines(proposed_bounds.min_x(), proposed_bounds.max_x());
    let moving_y = axis_lines(proposed_bounds.min_y(), proposed_bounds.max_y());

    let mut best_x: Option<AxisSnap> = None;
    let mut best_y: Option<AxisSnap> = None;

    for cand in &candidates {
        let ref_x = axis_lines(cand.min_x(), cand.max_x());
        let ref_y = axis_lines(cand.min_y(), cand.max_y());

        for &m in &moving_x {
            for &r in &ref_x {
                consider(&mut best_x, m, r, *cand, threshold);
            }
        }
        for &m in &moving_y {
            for &r in &ref_y {
                consider(&mut best_y, m, r, *cand, threshold);
            }
        }
    }

    let mut offset = Vec2::ZERO;
    let mut guides = Vec::new();

    // The bounds after applying the chosen per-axis corrections — used so the
    // guide spans the *snapped* extent, not the pre-snap one.
    let snapped = Rect::from_min_max(
        proposed_bounds.min_x() + best_x.map(|s| s.delta).unwrap_or(0.0),
        proposed_bounds.min_y() + best_y.map(|s| s.delta).unwrap_or(0.0),
        proposed_bounds.max_x() + best_x.map(|s| s.delta).unwrap_or(0.0),
        proposed_bounds.max_y() + best_y.map(|s| s.delta).unwrap_or(0.0),
    );

    if let Some(s) = best_x {
        offset.x = s.delta;
        // Vertical guide at x = s.line, spanning the combined y-extent of the
        // snapped moving box and the element it aligned to.
        let y0 = snapped.min_y().min(s.other.min_y());
        let y1 = snapped.max_y().max(s.other.max_y());
        guides.push(SnapGuide {
            a: Point::new(s.line, y0),
            b: Point::new(s.line, y1),
        });
    }
    if let Some(s) = best_y {
        offset.y = s.delta;
        // Horizontal guide at y = s.line, spanning the combined x-extent.
        let x0 = snapped.min_x().min(s.other.min_x());
        let x1 = snapped.max_x().max(s.other.max_x());
        guides.push(SnapGuide {
            a: Point::new(x0, s.line),
            b: Point::new(x1, s.line),
        });
    }

    SnapResult { offset, guides }
}

/// Consider a candidate snap (moving line `m` -> reference line `r`) for the
/// running best on one axis. Keeps the nearest within `threshold`.
fn consider(best: &mut Option<AxisSnap>, m: f64, r: f64, other: Rect, threshold: f64) {
    let delta = r - m;
    let dist = delta.abs();
    if dist > threshold {
        return;
    }
    let better = match best {
        None => true,
        Some(b) => dist < b.dist,
    };
    if better {
        *best = Some(AxisSnap {
            delta,
            dist,
            line: r,
            other,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementKind};

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    fn scene_with(els: impl IntoIterator<Item = Element>) -> Scene {
        let mut s = Scene::new();
        for e in els {
            s.insert(e);
        }
        s
    }

    #[test]
    fn snaps_left_edge_when_within_threshold() {
        // Stationary rect at x=100. Moving rect proposed at x=103 (3px off).
        let scene = scene_with([rect("a", 100.0, 0.0, 50.0, 50.0)]);
        let moving = Rect::new(103.0, 200.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        // Correction of -3 lands the left edge exactly on x=100.
        assert!(
            (res.offset.x + 3.0).abs() < 1e-9,
            "offset.x={}",
            res.offset.x
        );
        assert_eq!(res.offset.y, 0.0);
        // One vertical guide at x=100.
        assert_eq!(res.guides.len(), 1);
        let g = res.guides[0];
        assert!((g.a.x - 100.0).abs() < 1e-9);
        assert!((g.b.x - 100.0).abs() < 1e-9);
        // The guide is vertical.
        assert!((g.a.x - g.b.x).abs() < 1e-9);
        assert!((g.a.y - g.b.y).abs() > 1.0);
    }

    #[test]
    fn no_snap_when_far_apart() {
        let scene = scene_with([rect("a", 100.0, 0.0, 50.0, 50.0)]);
        let moving = Rect::new(400.0, 400.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        assert_eq!(res.offset, Vec2::ZERO);
        assert!(res.guides.is_empty());
    }

    #[test]
    fn snaps_center_alignment() {
        // Stationary rect center x = 125 (100..150). Moving width 50, proposed
        // so its center is at 127 (left at 102) -> snaps center to 125.
        let scene = scene_with([rect("a", 100.0, 0.0, 50.0, 50.0)]);
        let moving = Rect::new(102.0, 200.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        // Closest reference is center 125 to moving center 127 -> delta -2 (vs
        // left edge 100 to 102 -> -2 as well; tie, both land guide). Either way
        // the magnitude is 2.
        assert!(
            (res.offset.x.abs() - 2.0).abs() < 1e-9,
            "offset.x={}",
            res.offset.x
        );
    }

    #[test]
    fn snaps_both_axes_independently() {
        // Stationary rect at (100,100) 50x50. Moving proposed at (103, 96) so
        // left edge 3px right of 100 and top edge 4px below 100.
        let scene = scene_with([rect("a", 100.0, 100.0, 50.0, 50.0)]);
        let moving = Rect::new(103.0, 96.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        assert!((res.offset.x + 3.0).abs() < 1e-9);
        assert!((res.offset.y - 4.0).abs() < 1e-9);
        // One vertical + one horizontal guide.
        assert_eq!(res.guides.len(), 2);
    }

    #[test]
    fn moving_element_is_excluded_from_candidates() {
        // The only element is the moving one; nothing to snap to.
        let scene = scene_with([rect("m", 100.0, 0.0, 50.0, 50.0)]);
        let moving = Rect::new(101.0, 0.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        assert_eq!(res.offset, Vec2::ZERO);
        assert!(res.guides.is_empty());
    }

    #[test]
    fn guide_spans_combined_extent() {
        // Stationary rect a at y 0..50; moving snapped box at y 200..250. The
        // vertical guide should span from min(0,200)=0 to max(50,250)=250.
        let scene = scene_with([rect("a", 100.0, 0.0, 50.0, 50.0)]);
        let moving = Rect::new(102.0, 200.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        let g = res.guides[0];
        let ymin = g.a.y.min(g.b.y);
        let ymax = g.a.y.max(g.b.y);
        assert!((ymin - 0.0).abs() < 1e-9, "ymin={ymin}");
        assert!((ymax - 250.0).abs() < 1e-9, "ymax={ymax}");
    }

    #[test]
    fn nearest_of_two_candidates_wins() {
        // Two stationary rects: left edges at 100 and at 108. Moving left edge at
        // 105 -> nearer to 108 (3) than 100 (5). Snaps to 108.
        let scene = scene_with([
            rect("a", 100.0, 0.0, 50.0, 50.0),
            rect("b", 108.0, 300.0, 50.0, 50.0),
        ]);
        let moving = Rect::new(105.0, 600.0, 50.0, 50.0);
        let res = snap_moving(&scene, &[ElementId::from("m")], moving, 6.0);
        assert!(
            (res.offset.x - 3.0).abs() < 1e-9,
            "offset.x={}",
            res.offset.x
        );
        assert!((res.guides[0].a.x - 108.0).abs() < 1e-9);
    }

    #[test]
    fn empty_proposed_bounds_does_not_snap() {
        let scene = scene_with([rect("a", 100.0, 0.0, 50.0, 50.0)]);
        let res = snap_moving(&scene, &[ElementId::from("m")], Rect::EMPTY, 6.0);
        assert_eq!(res.offset, Vec2::ZERO);
        assert!(res.guides.is_empty());
    }
}
