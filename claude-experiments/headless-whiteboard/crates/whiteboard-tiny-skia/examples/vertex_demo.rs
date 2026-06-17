use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, LinearData};
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::geometry::Point;
use whiteboard_core::render::{Backend, Color};
use whiteboard_tiny_skia::{FontMeasurer, TinySkiaBackend};

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "vertex.png".into());
    let mut ed = Editor::new(FontMeasurer::new());
    // A 4-point arrow.
    let data = LinearData::arrow(vec![
        Point::new(0.0,0.0), Point::new(80.0,0.0), Point::new(80.0,80.0), Point::new(180.0,80.0),
    ]);
    let id = ed.add_element(Element::new(ElementId::from("a"),1, 40.0,40.0, 180.0,80.0, ElementKind::Arrow(data)));
    ed.scene().get(&id); // noop
    ed.select([id.clone()]);
    ed.set_tool(Tool::Select);
    // Drag the 2nd vertex (at scene 120,40) up-left to reshape.
    let d=|x,y|InputEvent::PointerDown{pos:Point::new(x,y),button:PointerButton::Primary,mods:Modifiers::default()};
    let m=|x,y|InputEvent::PointerMove{pos:Point::new(x,y),mods:Modifiers::default()};
    let u=|x,y|InputEvent::PointerUp{pos:Point::new(x,y),button:PointerButton::Primary,mods:Modifiers::default()};
    ed.handle(d(120.0,40.0)); ed.handle(m(150.0,20.0)); ed.handle(u(160.0,15.0));

    let scene = ed.render_with_overlay();
    let mut backend = TinySkiaBackend::new(300, 200).with_background(Color::WHITE);
    backend.render(&scene);
    backend.save_png(&out).expect("png");
    println!("rendered vertex demo ({} commands)", scene.commands.len());
}
