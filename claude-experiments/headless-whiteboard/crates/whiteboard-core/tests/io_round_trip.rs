//! Round-trip integrity for the internal `.excalidraw` save/load path across a
//! rich scene: every element type, groups, custom styling, opacity, rotation,
//! and soft-deleted elements. Saving then loading must reproduce the scene.

use whiteboard_core::element::{
    BoundElement, BoundElementKind, Element, ElementId, ElementKind, FreedrawData, GroupId,
    ImageData, LinearData, TextData,
};
use whiteboard_core::geometry::Point;
use whiteboard_core::io::{load_from_str, save_to_string};
use whiteboard_core::render::{Color, FillStyle, StrokeStyle};
use whiteboard_core::scene::Scene;

fn rich_scene() -> Scene {
    let mut scene = Scene::new();

    // A styled, rotated, grouped rectangle.
    let mut rect = Element::new(
        ElementId::from("rect"),
        11,
        10.0,
        20.0,
        100.0,
        50.0,
        ElementKind::Rectangle,
    );
    rect.angle = 0.5;
    rect.opacity = 60.0;
    rect.stroke_color = Color::rgb(200, 50, 50);
    rect.background_color = Color::rgb(255, 224, 178);
    rect.fill_style = FillStyle::CrossHatch;
    rect.stroke_style = StrokeStyle::Dashed;
    rect.stroke_width = 2.5;
    rect.group_ids = vec![GroupId::from("g1")];
    rect.bound_elements = vec![BoundElement {
        id: ElementId::from("label"),
        kind: BoundElementKind::Text,
    }];
    scene.insert(rect);

    // Its bound text label.
    let mut text = Element::new(
        ElementId::from("label"),
        12,
        15.0,
        30.0,
        80.0,
        25.0,
        ElementKind::Text({
            let mut t = TextData::new("hello");
            t.container_id = Some(ElementId::from("rect"));
            t
        }),
    );
    text.group_ids = vec![GroupId::from("g1")];
    scene.insert(text);

    // An ellipse and a diamond.
    scene.insert(Element::new(
        ElementId::from("ell"),
        13,
        140.0,
        20.0,
        80.0,
        60.0,
        ElementKind::Ellipse,
    ));
    scene.insert(Element::new(
        ElementId::from("dia"),
        14,
        140.0,
        100.0,
        60.0,
        60.0,
        ElementKind::Diamond,
    ));

    // An arrow with arrowheads + bindings.
    let arrow = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(90.0, 30.0)]);
    scene.insert(Element::new(
        ElementId::from("arr"),
        15,
        10.0,
        120.0,
        90.0,
        30.0,
        ElementKind::Arrow(arrow),
    ));

    // A freedraw with pressures.
    let mut free = FreedrawData::new(vec![
        Point::new(0.0, 0.0),
        Point::new(5.0, 8.0),
        Point::new(12.0, 4.0),
    ]);
    free.pressures = vec![0.2, 0.6, 0.9];
    scene.insert(Element::new(
        ElementId::from("free"),
        16,
        10.0,
        200.0,
        12.0,
        8.0,
        ElementKind::Freedraw(free),
    ));

    // An image and a frame.
    scene.insert(Element::new(
        ElementId::from("img"),
        17,
        220.0,
        20.0,
        64.0,
        64.0,
        ElementKind::Image(ImageData::new("file-xyz")),
    ));

    // A soft-deleted element (must survive the round-trip as deleted).
    let mut gone = Element::new(
        ElementId::from("gone"),
        18,
        0.0,
        0.0,
        10.0,
        10.0,
        ElementKind::Rectangle,
    );
    gone.is_deleted = true;
    scene.insert(gone);

    scene
}

#[test]
fn rich_scene_round_trips_exactly() {
    let original = rich_scene();
    let json = save_to_string(&original).expect("save");
    let loaded = load_from_str(&json).expect("load");

    assert_eq!(loaded.len(), original.len(), "element count preserved");

    for id in original.order() {
        let a = original.get(id).unwrap();
        let b = loaded
            .get(id)
            .unwrap_or_else(|| panic!("element {id} missing after round-trip"));
        // The whole Element derives PartialEq, so this checks every field
        // including kind-specific payloads, styling, groups, and bindings.
        assert_eq!(a, b, "element {id} changed across round-trip");
    }
}

#[test]
fn paint_order_is_preserved() {
    let original = rich_scene();
    let json = save_to_string(&original).unwrap();
    let loaded = load_from_str(&json).unwrap();
    assert_eq!(
        original.order(),
        loaded.order(),
        "z-order must survive save/load"
    );
}

#[test]
fn soft_deleted_survives_as_deleted() {
    let loaded = load_from_str(&save_to_string(&rich_scene()).unwrap()).unwrap();
    assert!(
        loaded.get(&ElementId::from("gone")).unwrap().is_deleted,
        "soft-deleted flag preserved"
    );
    assert!(
        !loaded.iter_live().any(|e| e.id.as_str() == "gone"),
        "deleted element excluded from live iteration"
    );
}

#[test]
fn group_membership_preserved() {
    let loaded = load_from_str(&save_to_string(&rich_scene()).unwrap()).unwrap();
    let rect = loaded.get(&ElementId::from("rect")).unwrap();
    assert_eq!(rect.group_ids, vec![GroupId::from("g1")]);
    let label = loaded.get(&ElementId::from("label")).unwrap();
    assert_eq!(label.group_ids, vec![GroupId::from("g1")]);
}
