//! Alignment and distribution as pure scene operations.
//!
//! Reimplemented from Excalidraw's `packages/element/src/align.ts`
//! (`alignElements` / `distributeElements`). No JavaScript is vendored — the
//! algorithms below are reimplementations in Rust.
//!
//! Both operations work in scene space using tight element bounds
//! ([`crate::geometry::element_bounds`], which accounts for rotation and curve
//! extents) and move elements by adjusting their `x`/`y`. Because linear and
//! freedraw points are element-relative, translating `x`/`y` moves the whole
//! element rigidly — so a single `(dx, dy)` per element is sufficient for every
//! element kind.
//!
//! Like the rest of this lane these are pure scene ops: they do not bump element
//! versions or record undo. The editor layer owns snapshotting.

use crate::element::ElementId;
use crate::geometry::{element_bounds, Rect};

use super::Scene;

/// One of the six alignment targets, matching Excalidraw's align actions.
///
/// `Left`/`CenterX`/`Right` operate on the horizontal axis; `Top`/`CenterY`/
/// `Bottom` on the vertical axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    CenterX,
    Right,
    Top,
    CenterY,
    Bottom,
}

/// Distribution axis, matching Excalidraw's distribute actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribute {
    Horizontal,
    Vertical,
}

/// Collect `(id, bounds)` for every requested id that exists and is live, in the
/// order requested. Skips missing/deleted ids.
fn live_bounds(scene: &Scene, ids: &[ElementId]) -> Vec<(ElementId, Rect)> {
    ids.iter()
        .filter_map(|id| {
            let el = scene.get(id)?;
            if el.is_deleted {
                return None;
            }
            Some((id.clone(), element_bounds(el)))
        })
        .collect()
}

/// Translate the element `id` by `(dx, dy)` in scene space.
///
/// Returns whether the element moved (a non-zero delta on a present, live
/// element).
fn translate(scene: &mut Scene, id: &ElementId, dx: f64, dy: f64) -> bool {
    if dx == 0.0 && dy == 0.0 {
        return false;
    }
    if let Some(el) = scene.get_mut(id) {
        el.x += dx;
        el.y += dy;
        true
    } else {
        false
    }
}

/// Align the selected elements (>= 2) to the selection's bounding box.
///
/// Each element's matching bounds edge (or center) is moved to coincide with the
/// corresponding edge/center of the union of all selected elements' bounds.
/// Missing/deleted ids are skipped; fewer than two live elements is a no-op.
/// Returns whether anything moved.
pub fn align(scene: &mut Scene, ids: &[ElementId], how: Align) -> bool {
    let items = live_bounds(scene, ids);
    if items.len() < 2 {
        return false;
    }

    // Selection bounding box (union of tight element bounds).
    let selection = items.iter().fold(Rect::EMPTY, |acc, (_, b)| acc.union(b));

    let mut changed = false;
    for (id, b) in &items {
        let (dx, dy) = match how {
            Align::Left => (selection.min_x() - b.min_x(), 0.0),
            Align::Right => (selection.max_x() - b.max_x(), 0.0),
            Align::CenterX => (selection.center().x - b.center().x, 0.0),
            Align::Top => (0.0, selection.min_y() - b.min_y()),
            Align::Bottom => (0.0, selection.max_y() - b.max_y()),
            Align::CenterY => (0.0, selection.center().y - b.center().y),
        };
        changed |= translate(scene, id, dx, dy);
    }
    changed
}

/// Distribute the selected elements (>= 3) so the gaps between adjacent elements
/// along `how`'s axis are equal.
///
/// Follows Excalidraw's `distributeElements`: the extreme elements (min-start and
/// max-end along the axis) stay put, defining the total span; the interior space
/// not occupied by the elements is divided evenly into `n - 1` equal gaps, and
/// each element is repositioned in start order so consecutive elements are
/// separated by that gap.
///
/// Missing/deleted ids are skipped; fewer than three live elements is a no-op.
/// Returns whether anything moved.
pub fn distribute(scene: &mut Scene, ids: &[ElementId], how: Distribute) -> bool {
    let items = live_bounds(scene, ids);
    if items.len() < 3 {
        return false;
    }

    // Project each element onto the axis: (start, size).
    let axis = |b: &Rect| match how {
        Distribute::Horizontal => (b.min_x(), b.width),
        Distribute::Vertical => (b.min_y(), b.height),
    };

    // Sort indices by start (then by end) so the extremes anchor the span.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&i, &j| {
        let (si, szi) = axis(&items[i].1);
        let (sj, szj) = axis(&items[j].1);
        si.partial_cmp(&sj)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                (si + szi)
                    .partial_cmp(&(sj + szj))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Span from the first element's start to the last element's end.
    let first = &items[order[0]].1;
    let last = &items[*order.last().unwrap()].1;
    let span_start = axis(first).0;
    let (last_start, last_size) = axis(last);
    let span_end = last_start + last_size;
    let span = span_end - span_start;

    // Total size occupied by the elements; the rest is gap space.
    let total_size: f64 = order.iter().map(|&i| axis(&items[i].1).1).sum();
    let gap = (span - total_size) / (order.len() as f64 - 1.0);

    // Walk in start order, placing each element after the previous with `gap`.
    let mut cursor = span_start;
    let mut changed = false;
    for &i in &order {
        let (start, size) = axis(&items[i].1);
        let target = cursor;
        let delta = target - start;
        let (dx, dy) = match how {
            Distribute::Horizontal => (delta, 0.0),
            Distribute::Vertical => (0.0, delta),
        };
        changed |= translate(scene, &items[i].0, dx, dy);
        cursor += size + gap;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementKind};

    /// A rectangle whose bounds equal its raw box (angle 0), so the test
    /// coordinates are exact and easy to reason about.
    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    fn id(s: &str) -> ElementId {
        ElementId::from(s)
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn align_left_moves_all_to_min_x() {
        let mut s = Scene::new();
        s.insert(rect("a", 10.0, 0.0, 20.0, 20.0));
        s.insert(rect("b", 50.0, 0.0, 30.0, 20.0));
        s.insert(rect("c", 100.0, 0.0, 10.0, 20.0));
        let changed = align(&mut s, &[id("a"), id("b"), id("c")], Align::Left);
        assert!(changed);
        // Min-x across the selection is 10.0; all left edges move there.
        approx(s.get(&id("a")).unwrap().x, 10.0);
        approx(s.get(&id("b")).unwrap().x, 10.0);
        approx(s.get(&id("c")).unwrap().x, 10.0);
        // y is untouched.
        approx(s.get(&id("b")).unwrap().y, 0.0);
    }

    #[test]
    fn align_right_moves_all_to_max_x() {
        let mut s = Scene::new();
        s.insert(rect("a", 10.0, 0.0, 20.0, 20.0)); // right edge 30
        s.insert(rect("b", 50.0, 0.0, 30.0, 20.0)); // right edge 80
        s.insert(rect("c", 100.0, 0.0, 10.0, 20.0)); // right edge 110 (max)
        align(&mut s, &[id("a"), id("b"), id("c")], Align::Right);
        // Selection max-x is 110; each element's right edge -> 110.
        approx(s.get(&id("a")).unwrap().x, 110.0 - 20.0);
        approx(s.get(&id("b")).unwrap().x, 110.0 - 30.0);
        approx(s.get(&id("c")).unwrap().x, 110.0 - 10.0);
    }

    #[test]
    fn align_center_x_centers_all() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0)); // min-x 0
        s.insert(rect("b", 80.0, 0.0, 20.0, 20.0)); // max-x 100
                                                    // Selection box is [0, 100], center x = 50.
        align(&mut s, &[id("a"), id("b")], Align::CenterX);
        // Each 20-wide element centered on 50 -> x = 40.
        approx(s.get(&id("a")).unwrap().x, 40.0);
        approx(s.get(&id("b")).unwrap().x, 40.0);
    }

    #[test]
    fn align_top_moves_all_to_min_y() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 30.0, 20.0, 20.0));
        s.insert(rect("b", 40.0, 10.0, 20.0, 20.0)); // min-y 10
        s.insert(rect("c", 80.0, 60.0, 20.0, 20.0));
        align(&mut s, &[id("a"), id("b"), id("c")], Align::Top);
        approx(s.get(&id("a")).unwrap().y, 10.0);
        approx(s.get(&id("b")).unwrap().y, 10.0);
        approx(s.get(&id("c")).unwrap().y, 10.0);
    }

    #[test]
    fn align_bottom_moves_all_to_max_y() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0)); // bottom 20
        s.insert(rect("b", 40.0, 0.0, 20.0, 50.0)); // bottom 50 (max)
        align(&mut s, &[id("a"), id("b")], Align::Bottom);
        approx(s.get(&id("a")).unwrap().y, 50.0 - 20.0);
        approx(s.get(&id("b")).unwrap().y, 50.0 - 50.0);
    }

    #[test]
    fn align_center_y_centers_all() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0)); // min-y 0
        s.insert(rect("b", 0.0, 80.0, 20.0, 20.0)); // max-y 100
                                                    // Selection box [0,100], center y = 50.
        align(&mut s, &[id("a"), id("b")], Align::CenterY);
        approx(s.get(&id("a")).unwrap().y, 40.0);
        approx(s.get(&id("b")).unwrap().y, 40.0);
    }

    #[test]
    fn align_requires_two_elements() {
        let mut s = Scene::new();
        s.insert(rect("a", 10.0, 0.0, 20.0, 20.0));
        assert!(!align(&mut s, &[id("a")], Align::Left));
        // Still at original position.
        approx(s.get(&id("a")).unwrap().x, 10.0);
    }

    #[test]
    fn align_skips_missing_and_deleted() {
        let mut s = Scene::new();
        s.insert(rect("a", 10.0, 0.0, 20.0, 20.0));
        s.insert(rect("b", 50.0, 0.0, 20.0, 20.0));
        let mut d = rect("c", 200.0, 0.0, 20.0, 20.0);
        d.is_deleted = true;
        s.insert(d);
        // Deleted "c" and missing "z" do not affect the selection box.
        align(&mut s, &[id("a"), id("b"), id("c"), id("z")], Align::Left);
        approx(s.get(&id("a")).unwrap().x, 10.0);
        approx(s.get(&id("b")).unwrap().x, 10.0);
        // Deleted element untouched.
        approx(s.get(&id("c")).unwrap().x, 200.0);
    }

    #[test]
    fn distribute_horizontal_equalizes_gaps() {
        let mut s = Scene::new();
        // Three 20-wide rects. Extremes at x=0 (end 20) and x=200 (end 220).
        // Middle one starts off-center at 50.
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0));
        s.insert(rect("b", 50.0, 0.0, 20.0, 20.0));
        s.insert(rect("c", 200.0, 0.0, 20.0, 20.0));
        let changed = distribute(&mut s, &[id("a"), id("b"), id("c")], Distribute::Horizontal);
        assert!(changed);
        // Span = 220 - 0 = 220; total size = 60; gap = (220-60)/2 = 80.
        // a stays at 0; b at 0+20+80 = 100; c at 100+20+80 = 200.
        approx(s.get(&id("a")).unwrap().x, 0.0);
        approx(s.get(&id("b")).unwrap().x, 100.0);
        approx(s.get(&id("c")).unwrap().x, 200.0);
        // Gaps are now equal: 100-20 = 80 and 200-120 = 80.
        let ax = s.get(&id("a")).unwrap();
        let bx = s.get(&id("b")).unwrap();
        let cx = s.get(&id("c")).unwrap();
        approx((bx.x) - (ax.x + ax.width), (cx.x) - (bx.x + bx.width));
    }

    #[test]
    fn distribute_vertical_equalizes_gaps() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0)); // y end 20
        s.insert(rect("b", 0.0, 30.0, 20.0, 40.0)); // size 40
        s.insert(rect("c", 0.0, 200.0, 20.0, 20.0)); // y end 220
        distribute(&mut s, &[id("a"), id("b"), id("c")], Distribute::Vertical);
        // Span = 220; total size = 20+40+20 = 80; gap = (220-80)/2 = 70.
        // a at 0; b at 0+20+70 = 90; c at 90+40+70 = 200.
        approx(s.get(&id("a")).unwrap().y, 0.0);
        approx(s.get(&id("b")).unwrap().y, 90.0);
        approx(s.get(&id("c")).unwrap().y, 200.0);
    }

    #[test]
    fn distribute_is_order_independent_of_input() {
        // Same three elements supplied out of spatial order; result must be the
        // same as the sorted-input case.
        let mut s = Scene::new();
        s.insert(rect("c", 200.0, 0.0, 20.0, 20.0));
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0));
        s.insert(rect("b", 50.0, 0.0, 20.0, 20.0));
        distribute(&mut s, &[id("c"), id("a"), id("b")], Distribute::Horizontal);
        approx(s.get(&id("a")).unwrap().x, 0.0);
        approx(s.get(&id("b")).unwrap().x, 100.0);
        approx(s.get(&id("c")).unwrap().x, 200.0);
    }

    #[test]
    fn distribute_requires_three_elements() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0));
        s.insert(rect("b", 100.0, 0.0, 20.0, 20.0));
        assert!(!distribute(
            &mut s,
            &[id("a"), id("b")],
            Distribute::Horizontal
        ));
        approx(s.get(&id("b")).unwrap().x, 100.0);
    }

    #[test]
    fn distribute_skips_missing_and_deleted() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0, 20.0, 20.0));
        s.insert(rect("b", 50.0, 0.0, 20.0, 20.0));
        s.insert(rect("c", 200.0, 0.0, 20.0, 20.0));
        let mut d = rect("dead", 500.0, 0.0, 20.0, 20.0);
        d.is_deleted = true;
        s.insert(d);
        // "dead" is deleted and "z" missing; only a,b,c distribute.
        distribute(
            &mut s,
            &[id("a"), id("b"), id("c"), id("dead"), id("z")],
            Distribute::Horizontal,
        );
        approx(s.get(&id("a")).unwrap().x, 0.0);
        approx(s.get(&id("b")).unwrap().x, 100.0);
        approx(s.get(&id("c")).unwrap().x, 200.0);
        approx(s.get(&id("dead")).unwrap().x, 500.0);
    }
}
