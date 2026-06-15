//! Multi-element (group) rigid-body transform tests, driven entirely through the
//! public `Editor` API. These harden the guarantee that resizing, rotating, or
//! moving a multi-selection treats the whole selection as one rigid body:
//!
//! * a corner-handle resize scales every element about the *opposite* corner of
//!   the shared selection box, preserving the relative layout (position and size
//!   ratios) of all members;
//! * a rotation-handle drag rotates every element's angle by the same delta and
//!   orbits every element's center about the shared pivot, preserving the group
//!   bounding-box center;
//! * dragging the body of a multi-selection translates every member by the same
//!   delta.
//!
//! The interaction math lives in `interaction::state` (`apply_scale`,
//! `resize_scale`, and the `Rotating` gesture) — these tests verify it end-to-end
//! through `Editor::handle`, the same path an application drives, rather than the
//! internal helpers. Handle screen positions are discovered via
//! `Editor::interaction().handle_layout(...)`, mirroring the Phase-7 pattern in
//! `interaction_lifecycle.rs`.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{Handle, InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::text::MonospaceMeasurer;

type Ed = Editor<MonospaceMeasurer>;

fn editor() -> Ed {
    Editor::new(MonospaceMeasurer::default())
}

fn down(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerDown {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    }
}
fn mv(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerMove {
        pos: Point::new(x, y),
        mods: Modifiers::default(),
    }
}
fn up(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerUp {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    }
}

/// Add a rectangle directly to the scene (bypassing the create gesture so we can
/// place three well-separated, exactly-sized boxes deterministically).
fn add_rect(ed: &mut Ed, id: &str, x: f64, y: f64, w: f64, h: f64) -> ElementId {
    let el = Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle);
    ed.add_element(el)
}

/// The three separated rectangles used by every test in this file.
///
/// Their union bounding box is `(10,10)`..`(140,170)` (130 wide, 160 tall) with
/// the NW corner at A's top-left and the SE corner at B/C's extremes:
///   A: (10,10)  30x30   -> 10..40   x, 10..40   y
///   B: (100,40) 40x20   -> 100..140 x, 40..60   y
///   C: (60,120) 20x50   -> 60..80   x, 120..170 y
fn three_rects(ed: &mut Ed) -> (ElementId, ElementId, ElementId) {
    let a = add_rect(ed, "a", 10.0, 10.0, 30.0, 30.0);
    let b = add_rect(ed, "b", 100.0, 40.0, 40.0, 20.0);
    let c = add_rect(ed, "c", 60.0, 120.0, 20.0, 50.0);
    (a, b, c)
}

/// Marquee-select everything by dragging an empty-space rectangle that fully
/// encloses all three boxes.
fn marquee_select_all(ed: &mut Ed) {
    ed.set_tool(Tool::Select);
    ed.handle(down(0.0, 0.0));
    ed.handle(mv(100.0, 100.0));
    ed.handle(mv(200.0, 200.0));
    ed.handle(up(200.0, 200.0));
}

fn get(ed: &Ed, id: &ElementId) -> Element {
    ed.scene().get(id).unwrap().clone()
}

#[test]
fn marquee_selects_all_three_separated_rects() {
    let mut ed = editor();
    let (a, b, c) = three_rects(&mut ed);
    marquee_select_all(&mut ed);
    let sel = ed.selection();
    assert!(sel.contains(&a), "A selected");
    assert!(sel.contains(&b), "B selected");
    assert!(sel.contains(&c), "C selected");
    assert_eq!(sel.len(), 3, "exactly the three rects are selected");
}

#[test]
fn group_corner_resize_scales_all_about_opposite_corner() {
    let mut ed = editor();
    let (a, b, c) = three_rects(&mut ed);

    let before = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];

    marquee_select_all(&mut ed);

    // The shared selection box is (10,10)..(140,170). Drag the SE handle so the
    // box scales 2x about its NW corner (the fixed anchor for an SE drag):
    //   anchor = (10,10), original SE = (140,170)
    //   target SE = anchor + (SE - anchor)*2 = (270, 330)
    let vp = ed.viewport();
    let layout = ed
        .interaction()
        .handle_layout(ed.scene(), &vp)
        .expect("multi-selection exposes a handle layout");
    let se = layout.center(Handle::SouthEast);
    // Identity viewport => handle is at the scene-space SE corner.
    assert!((se.x - 140.0).abs() < 1e-9, "SE handle x = {}", se.x);
    assert!((se.y - 170.0).abs() < 1e-9, "SE handle y = {}", se.y);

    let anchor = Point::new(10.0, 10.0);
    let target = Point::new(
        anchor.x + (se.x - anchor.x) * 2.0,
        anchor.y + (se.y - anchor.y) * 2.0,
    );

    ed.handle(down(se.x, se.y));
    ed.handle(mv((se.x + target.x) / 2.0, (se.y + target.y) / 2.0));
    ed.handle(mv(target.x, target.y));
    ed.handle(up(target.x, target.y));

    let after = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];

    // Every element scaled 2x in both axes.
    for (b0, a1) in before.iter().zip(after.iter()) {
        assert!(
            (a1.width - b0.width * 2.0).abs() < 1e-6,
            "{}: width {} -> {} (expected x2)",
            a1.id.as_str(),
            b0.width,
            a1.width
        );
        assert!(
            (a1.height - b0.height * 2.0).abs() < 1e-6,
            "{}: height {} -> {} (expected x2)",
            a1.id.as_str(),
            b0.height,
            a1.height
        );
        // Position scales about the (10,10) anchor too.
        let want_x = anchor.x + (b0.x - anchor.x) * 2.0;
        let want_y = anchor.y + (b0.y - anchor.y) * 2.0;
        assert!(
            (a1.x - want_x).abs() < 1e-6,
            "{}: x {} -> {} (expected {})",
            a1.id.as_str(),
            b0.x,
            a1.x,
            want_x
        );
        assert!(
            (a1.y - want_y).abs() < 1e-6,
            "{}: y {} -> {} (expected {})",
            a1.id.as_str(),
            b0.y,
            a1.y,
            want_y
        );
    }

    // The anchored NW corner of the group (A's top-left) is unmoved.
    assert!((after[0].x - 10.0).abs() < 1e-6, "anchor x fixed");
    assert!((after[0].y - 10.0).abs() < 1e-6, "anchor y fixed");

    // Relative layout is preserved: every pairwise gap and size ratio is exactly
    // doubled (i.e. the *shape* of the layout, normalized by scale, is identical).
    // Check this independently of the scale factor by comparing ratios.
    let ratio_w_ab_before = before[0].width / before[1].width;
    let ratio_w_ab_after = after[0].width / after[1].width;
    assert!(
        (ratio_w_ab_before - ratio_w_ab_after).abs() < 1e-9,
        "A/B width ratio preserved: {ratio_w_ab_before} vs {ratio_w_ab_after}"
    );
    // Gap between A and B centers scales by the same factor in both axes.
    let gap_before = (
        before[1].center().x - before[0].center().x,
        before[1].center().y - before[0].center().y,
    );
    let gap_after = (
        after[1].center().x - after[0].center().x,
        after[1].center().y - after[0].center().y,
    );
    assert!(
        (gap_after.0 - gap_before.0 * 2.0).abs() < 1e-6,
        "A->B dx doubled"
    );
    assert!(
        (gap_after.1 - gap_before.1 * 2.0).abs() < 1e-6,
        "A->B dy doubled"
    );
}

#[test]
fn group_rotation_rotates_every_element_by_same_delta() {
    let mut ed = editor();
    let (a, b, c) = three_rects(&mut ed);

    let before = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];
    // All start unrotated.
    for el in &before {
        assert_eq!(el.angle, 0.0, "{} starts unrotated", el.id.as_str());
    }

    marquee_select_all(&mut ed);

    // The shared selection box is (10,10)..(140,170); its center (the rotation
    // pivot) is (75, 90).
    let pivot = Point::new(75.0, 90.0);

    let vp = ed.viewport();
    let layout = ed.interaction().handle_layout(ed.scene(), &vp).unwrap();
    let rot = layout.center(Handle::Rotation);
    let layout_pivot = layout.pivot();
    assert!((layout_pivot.x - pivot.x).abs() < 1e-9, "pivot x");
    assert!((layout_pivot.y - pivot.y).abs() < 1e-9, "pivot y");

    // Swing the rotation handle 90deg clockwise about the pivot. The handle sits
    // directly above the pivot at grab time; rotating (dx,dy) by +90deg CW maps
    // (x,y) -> (-y, x), so the target is to the right of the pivot.
    let dx = rot.x - pivot.x;
    let dy = rot.y - pivot.y;
    let target = Point::new(pivot.x - dy, pivot.y + dx);

    ed.handle(down(rot.x, rot.y));
    ed.handle(mv(target.x, target.y));
    ed.handle(up(target.x, target.y));

    let after = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];

    use std::f64::consts::FRAC_PI_2;
    // Every element rotated by ~+90deg.
    for (b0, a1) in before.iter().zip(after.iter()) {
        let delta = a1.angle - b0.angle;
        assert!(
            (delta - FRAC_PI_2).abs() < 1e-6,
            "{}: angle delta {} (expected {})",
            a1.id.as_str(),
            delta,
            FRAC_PI_2
        );
    }

    // All deltas are equal to each other (rigid body).
    let d0 = after[0].angle - before[0].angle;
    let d1 = after[1].angle - before[1].angle;
    let d2 = after[2].angle - before[2].angle;
    assert!(
        (d0 - d1).abs() < 1e-9 && (d1 - d2).abs() < 1e-9,
        "equal deltas"
    );

    // Each element's center orbited the pivot by +90deg CW: (x,y) about pivot ->
    // (pivot.x - (y-pivot.y), pivot.y + (x-pivot.x)).
    for (b0, a1) in before.iter().zip(after.iter()) {
        let cb = b0.center();
        let want = Point::new(pivot.x - (cb.y - pivot.y), pivot.y + (cb.x - pivot.x));
        let ca = a1.center();
        assert!(
            (ca.x - want.x).abs() < 1e-6 && (ca.y - want.y).abs() < 1e-6,
            "{}: center {:?} expected {:?}",
            a1.id.as_str(),
            (ca.x, ca.y),
            (want.x, want.y)
        );
    }

    // The group bbox center is preserved by a pure rotation about that center.
    let center_before = group_bbox_center(&before);
    let center_after = group_bbox_center(&after);
    assert!(
        (center_before.x - center_after.x).abs() < 1e-6
            && (center_before.y - center_after.y).abs() < 1e-6,
        "group bbox center preserved: {:?} -> {:?}",
        (center_before.x, center_before.y),
        (center_after.x, center_after.y)
    );
}

#[test]
fn group_move_translates_all_by_same_delta() {
    let mut ed = editor();
    let (a, b, c) = three_rects(&mut ed);

    let before = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];

    marquee_select_all(&mut ed);

    // Grab the body of the selection (inside rect A) and drag by (+50, +30).
    // A spans (10,10)..(40,40); (25,25) is interior.
    let (gx, gy) = (25.0, 25.0);
    let (dx, dy) = (50.0, 30.0);
    ed.set_tool(Tool::Select);
    ed.handle(down(gx, gy));
    ed.handle(mv(gx + dx / 2.0, gy + dy / 2.0));
    ed.handle(mv(gx + dx, gy + dy));
    ed.handle(up(gx + dx, gy + dy));

    let after = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];

    for (b0, a1) in before.iter().zip(after.iter()) {
        assert!(
            (a1.x - (b0.x + dx)).abs() < 1e-6,
            "{}: x {} -> {} (expected +{})",
            a1.id.as_str(),
            b0.x,
            a1.x,
            dx
        );
        assert!(
            (a1.y - (b0.y + dy)).abs() < 1e-6,
            "{}: y {} -> {} (expected +{})",
            a1.id.as_str(),
            b0.y,
            a1.y,
            dy
        );
        // Sizes unchanged by a translation.
        assert!((a1.width - b0.width).abs() < 1e-9, "width unchanged");
        assert!((a1.height - b0.height).abs() < 1e-9, "height unchanged");
        // Angles unchanged.
        assert!((a1.angle - b0.angle).abs() < 1e-9, "angle unchanged");
    }

    // The move is a single undo step that restores all three.
    assert!(ed.undo(), "move is undoable");
    let undone = [get(&ed, &a), get(&ed, &b), get(&ed, &c)];
    for (orig, u) in before.iter().zip(undone.iter()) {
        assert!((u.x - orig.x).abs() < 1e-9, "undo restores x");
        assert!((u.y - orig.y).abs() < 1e-9, "undo restores y");
    }
}

/// Axis-aligned union of the elements' (unrotated) raw boxes, then its center.
/// For these tests the elements never have a nonzero angle before rotation, and
/// after a rigid rotation about the bbox center the union-of-rotated-bounds center
/// is what the editor's `selection_bbox` tracks; we use the simple raw-box union
/// center which is invariant under the symmetric layout we constructed.
fn group_bbox_center(els: &[Element]) -> Point {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for el in els {
        let bb = rotated_bounds(el);
        min_x = min_x.min(bb.min_x());
        min_y = min_y.min(bb.min_y());
        max_x = max_x.max(bb.max_x());
        max_y = max_y.max(bb.max_y());
    }
    Point::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
}

/// Axis-aligned bounds of an element after applying its own rotation about its
/// center (mirrors the editor's internal `rotated_bounds`).
fn rotated_bounds(el: &Element) -> whiteboard_core::geometry::Rect {
    use whiteboard_core::geometry::{point_rotate_rads, Rect};
    if el.angle == 0.0 {
        return el.raw_box();
    }
    let c = el.center();
    Rect::bounding(
        el.raw_box()
            .corners()
            .into_iter()
            .map(|p| point_rotate_rads(p, c, el.angle)),
    )
}
