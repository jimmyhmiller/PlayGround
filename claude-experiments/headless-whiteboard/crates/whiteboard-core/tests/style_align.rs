//! Editor-level integration for Phase-10 property mutation, alignment, and
//! distribution, plus elbow-arrow generation.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, LinearData};
use whiteboard_core::geometry::{PathSegment, Point};
use whiteboard_core::render::{Color, DrawCommand, FillStyle};
use whiteboard_core::scene::{Align, Distribute, StyleChange};
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}

fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
    Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
}

#[test]
fn set_style_changes_selection_and_undoes() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 0.0, 0.0, 10.0, 10.0));
    let b = ed.add_element(rect("b", 20.0, 0.0, 10.0, 10.0));
    ed.select([a.clone(), b.clone()]);

    assert!(ed.set_style(&StyleChange::StrokeColor(Color::rgb(255, 0, 0))));
    assert_eq!(
        ed.scene().get(&a).unwrap().stroke_color,
        Color::rgb(255, 0, 0)
    );
    assert_eq!(
        ed.scene().get(&b).unwrap().stroke_color,
        Color::rgb(255, 0, 0)
    );

    assert!(ed.set_style(&StyleChange::FillStyle(FillStyle::CrossHatch)));
    assert_eq!(
        ed.scene().get(&a).unwrap().fill_style,
        FillStyle::CrossHatch
    );

    // Each style change is its own undo step.
    assert!(ed.undo());
    assert_eq!(ed.scene().get(&a).unwrap().fill_style, FillStyle::Hachure);
    assert!(ed.undo());
    assert_eq!(
        ed.scene().get(&a).unwrap().stroke_color,
        Color::rgb(30, 30, 30)
    );
}

#[test]
fn opacity_is_clamped() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 0.0, 0.0, 10.0, 10.0));
    ed.select([a.clone()]);
    ed.set_style(&StyleChange::Opacity(250.0));
    assert!(ed.scene().get(&a).unwrap().opacity <= 100.0);
}

#[test]
fn align_left_moves_all_to_min_x() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 100.0, 0.0, 20.0, 20.0));
    let b = ed.add_element(rect("b", 50.0, 40.0, 20.0, 20.0));
    let c = ed.add_element(rect("c", 200.0, 80.0, 20.0, 20.0));
    ed.select([a.clone(), b.clone(), c.clone()]);

    assert!(ed.align(Align::Left));
    // The minimum x among the three was 50 (b); all left edges now == 50.
    assert!((ed.scene().get(&a).unwrap().x - 50.0).abs() < 1e-6);
    assert!((ed.scene().get(&b).unwrap().x - 50.0).abs() < 1e-6);
    assert!((ed.scene().get(&c).unwrap().x - 50.0).abs() < 1e-6);

    assert!(ed.undo());
    assert!((ed.scene().get(&a).unwrap().x - 100.0).abs() < 1e-6);
}

#[test]
fn distribute_horizontal_equalizes_gaps() {
    let mut ed = editor();
    // Three same-size rects with uneven horizontal spacing.
    let a = ed.add_element(rect("a", 0.0, 0.0, 20.0, 20.0));
    let b = ed.add_element(rect("b", 30.0, 0.0, 20.0, 20.0));
    let c = ed.add_element(rect("c", 200.0, 0.0, 20.0, 20.0));
    ed.select([a.clone(), b.clone(), c.clone()]);

    assert!(ed.distribute(Distribute::Horizontal));
    // Extremes (a at 0, c at 200) are anchored; b sits so gaps are equal.
    let xa = ed.scene().get(&a).unwrap().x;
    let xb = ed.scene().get(&b).unwrap().x;
    let xc = ed.scene().get(&c).unwrap().x;
    assert!(
        (xa - 0.0).abs() < 1e-6 && (xc - 200.0).abs() < 1e-6,
        "extremes anchored"
    );
    let gap1 = xb - (xa + 20.0);
    let gap2 = xc - (xb + 20.0);
    assert!((gap1 - gap2).abs() < 1e-6, "gaps equal: {gap1} vs {gap2}");
}

#[test]
fn elbowed_arrow_is_axis_aligned() {
    let mut ed = editor();
    // A diagonal arrow, but elbowed ⇒ orthogonal routing.
    let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(120.0, 80.0)]);
    data.elbowed = true;
    ed.add_element(Element::new(
        ElementId::from("arr"),
        1,
        10.0,
        10.0,
        120.0,
        80.0,
        ElementKind::Arrow(data),
    ));

    // The first stroked path is the elbow body (arrowheads are later paths).
    // Every body segment must be axis-aligned.
    let scene = ed.render();
    let body = scene
        .commands
        .iter()
        .find_map(|cmd| match cmd {
            DrawCommand::StrokePath { path, .. } => Some(path),
            _ => None,
        })
        .expect("an elbow body polyline was rendered");

    let pts: Vec<Point> = body
        .segments
        .iter()
        .filter_map(|s| match s {
            PathSegment::MoveTo(p) | PathSegment::LineTo(p) => Some(*p),
            _ => None,
        })
        .collect();
    assert!(pts.len() >= 2, "body polyline has points");
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let horizontal = (a.y - b.y).abs() < 1e-6;
        let vertical = (a.x - b.x).abs() < 1e-6;
        assert!(
            horizontal || vertical,
            "elbow body segment must be axis-aligned: {a:?}->{b:?}"
        );
    }
    // A diagonal start→end forced through an elbow must bend (>2 points).
    assert!(pts.len() >= 3, "diagonal elbow introduces a corner");
}
