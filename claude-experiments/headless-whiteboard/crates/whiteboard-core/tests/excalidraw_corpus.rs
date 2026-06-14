//! Real-format `.excalidraw` compatibility test.
//!
//! Parses a document in Excalidraw's actual on-disk JSON shape (flat camelCase
//! keys, `type` discriminator, integer `fontFamily` codes) and asserts it maps
//! onto our `Element` model with the right values, then that bounds/hit-test and
//! rendering work on the loaded scene. This is the integration check that the
//! file format the rest of the world produces actually loads.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::ElementKind;
use whiteboard_core::geometry::{element_bounds, hit_test, Point};
use whiteboard_core::io::load_excalidraw_str;
use whiteboard_core::scene::Scene;
use whiteboard_core::text::MonospaceMeasurer;

/// A minimal but real-shaped Excalidraw export: a rectangle, an arrow, and a
/// text element. Keys and value conventions match what the Excalidraw app emits.
const SAMPLE: &str = r##"{
  "type": "excalidraw",
  "version": 2,
  "source": "https://excalidraw.com",
  "elements": [
    {
      "id": "rect-1",
      "type": "rectangle",
      "x": 100.0,
      "y": 120.0,
      "width": 200.0,
      "height": 90.0,
      "angle": 0.0,
      "strokeColor": "#1e1e1e",
      "backgroundColor": "#ffc9c9",
      "fillStyle": "hachure",
      "strokeWidth": 2,
      "strokeStyle": "solid",
      "roughness": 1,
      "opacity": 100,
      "seed": 12345,
      "version": 5,
      "versionNonce": 99887766,
      "groupIds": [],
      "boundElements": [],
      "frameId": null,
      "link": null,
      "locked": false,
      "isDeleted": false
    },
    {
      "id": "arrow-1",
      "type": "arrow",
      "x": 320.0,
      "y": 140.0,
      "width": 160.0,
      "height": 0.0,
      "angle": 0.0,
      "strokeColor": "#1971c2",
      "backgroundColor": "transparent",
      "fillStyle": "solid",
      "strokeWidth": 2,
      "strokeStyle": "solid",
      "roughness": 1,
      "opacity": 100,
      "seed": 222,
      "version": 3,
      "versionNonce": 1,
      "groupIds": [],
      "boundElements": [],
      "frameId": null,
      "link": null,
      "locked": false,
      "isDeleted": false,
      "points": [[0.0, 0.0], [160.0, 0.0]],
      "startBinding": null,
      "endBinding": null,
      "startArrowhead": null,
      "endArrowhead": "arrow"
    },
    {
      "id": "text-1",
      "type": "text",
      "x": 110.0,
      "y": 230.0,
      "width": 120.0,
      "height": 25.0,
      "angle": 0.0,
      "strokeColor": "#1e1e1e",
      "backgroundColor": "transparent",
      "fillStyle": "solid",
      "strokeWidth": 1,
      "strokeStyle": "solid",
      "roughness": 1,
      "opacity": 100,
      "seed": 333,
      "version": 2,
      "versionNonce": 2,
      "groupIds": [],
      "boundElements": [],
      "frameId": null,
      "link": null,
      "locked": false,
      "isDeleted": false,
      "text": "hello",
      "fontSize": 20,
      "fontFamily": 1,
      "textAlign": "left",
      "verticalAlign": "top",
      "containerId": null,
      "lineHeight": 1.25
    }
  ]
}"##;

#[test]
fn loads_real_excalidraw_format() {
    let elements = load_excalidraw_str(SAMPLE).expect("parse real excalidraw json");
    assert_eq!(elements.len(), 3);

    let rect = &elements[0];
    assert_eq!(rect.id.as_str(), "rect-1");
    assert!(matches!(rect.kind, ElementKind::Rectangle));
    assert_eq!(rect.width, 200.0);
    assert_eq!(rect.seed, 12345);
    // backgroundColor "#ffc9c9" parsed to RGB.
    assert_eq!(
        (
            rect.background_color.r,
            rect.background_color.g,
            rect.background_color.b
        ),
        (255, 201, 201)
    );

    let arrow = &elements[1];
    match &arrow.kind {
        ElementKind::Arrow(data) => {
            assert_eq!(data.points.len(), 2);
            assert_eq!(data.points[1], Point::new(160.0, 0.0));
            assert!(data.end_arrowhead.is_some(), "arrow has an end arrowhead");
        }
        other => panic!("expected arrow, got {other:?}"),
    }

    let text = &elements[2];
    match &text.kind {
        ElementKind::Text(t) => {
            assert_eq!(t.text, "hello");
            assert_eq!(t.font_size, 20.0);
            // fontFamily code 1 => HandDrawn.
            assert_eq!(t.font_family, whiteboard_core::text::FontFamily::HandDrawn);
        }
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn loaded_scene_has_working_bounds_and_hit_test() {
    let elements = load_excalidraw_str(SAMPLE).unwrap();
    let rect = &elements[0];

    // Tight bounds of the unrotated rectangle == its raw box.
    let b = element_bounds(rect);
    assert!((b.min_x() - 100.0).abs() < 1e-6);
    assert!((b.max_x() - 300.0).abs() < 1e-6);

    // A point inside the rectangle hits it; a far point does not.
    assert!(hit_test(rect, Point::new(150.0, 150.0), 2.0));
    assert!(!hit_test(rect, Point::new(5.0, 5.0), 2.0));
}

#[test]
fn loaded_scene_renders() {
    let elements = load_excalidraw_str(SAMPLE).unwrap();
    let mut scene = Scene::new();
    for el in elements {
        scene.insert(el);
    }

    // Drive the loaded scene through the editor's render pass.
    let mut editor = Editor::new_rough(MonospaceMeasurer::default());
    for id in scene.order().to_vec() {
        editor.add_element(scene.get(&id).unwrap().clone());
    }
    assert!(!editor.render().is_empty(), "loaded scene must render");
}
