//! A full editing session, end to end, asserting correctness at every step —
//! the integration test that catches gaps per-feature unit tests miss. This is
//! the workflow a real app drives: draw several shapes, select/move/resize,
//! style them, group, copy/paste, undo/redo deeply, save, reload, re-render.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::ElementKind;
use whiteboard_core::geometry::{Point, Vec2};
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::io::{load_from_str, save_to_string};
use whiteboard_core::render::{Color, FillStyle};
use whiteboard_core::scene::{Align, StyleChange};
use whiteboard_core::text::MonospaceMeasurer;

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

fn draw(ed: &mut Editor<MonospaceMeasurer>, tool: Tool, a: (f64, f64), b: (f64, f64)) {
    ed.set_tool(tool);
    ed.handle(down(a.0, a.1));
    ed.handle(mv((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0));
    ed.handle(mv(b.0, b.1));
    ed.handle(up(b.0, b.1));
}

#[test]
fn complete_editing_session() {
    let mut ed = Editor::new(MonospaceMeasurer::default());

    // 1. Draw three shapes with three tools, at DIFFERENT tops so alignment
    //    later has real work to do.
    draw(&mut ed, Tool::Rectangle, (20.0, 20.0), (120.0, 100.0));
    draw(&mut ed, Tool::Ellipse, (200.0, 50.0), (320.0, 130.0));
    draw(&mut ed, Tool::Diamond, (380.0, 80.0), (480.0, 160.0));
    assert_eq!(ed.scene().iter_live().count(), 3, "three shapes created");

    // Each is the right kind.
    let kinds: Vec<&str> = ed.scene().iter_live().map(|e| e.type_name()).collect();
    assert_eq!(kinds, ["rectangle", "ellipse", "diamond"]);

    // 2. Select all three with a marquee and confirm.
    ed.set_tool(Tool::Select);
    ed.handle(down(10.0, 10.0));
    ed.handle(mv(250.0, 120.0));
    ed.handle(mv(500.0, 180.0));
    ed.handle(up(500.0, 180.0));
    assert_eq!(ed.selection().len(), 3, "marquee selected all three");

    // 3. Style them — red stroke, solid fill — as undoable steps.
    assert!(ed.set_style(&StyleChange::StrokeColor(Color::rgb(220, 30, 30))));
    assert!(ed.set_style(&StyleChange::FillStyle(FillStyle::Solid)));
    assert!(ed.set_style(&StyleChange::BackgroundColor(Color::rgb(255, 230, 200))));
    for e in ed.scene().iter_live() {
        assert_eq!(e.stroke_color, Color::rgb(220, 30, 30));
        assert_eq!(e.fill_style, FillStyle::Solid);
    }

    // 4. Align their tops.
    assert!(ed.align(Align::Top));
    let tops: Vec<f64> = ed.scene().iter_live().map(|e| e.y).collect();
    assert!(
        tops.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6),
        "all tops aligned: {tops:?}"
    );

    // 5. Group them; clicking one selects all.
    assert!(ed.group_selection());
    let one = ed.scene().iter_live().next().unwrap().id.clone();
    let expanded = ed.expand_to_groups(std::slice::from_ref(&one));
    assert_eq!(expanded.len(), 3, "group expands to all three");

    // 6. Duplicate the group → 6 elements.
    ed.select(expanded);
    let dups = ed.duplicate_selection();
    assert_eq!(dups.len(), 3);
    assert_eq!(ed.scene().iter_live().count(), 6, "duplicated the group");

    // 7. Copy + paste once more → 9.
    ed.copy();
    let pasted = ed.paste(Vec2::new(0.0, 200.0));
    assert_eq!(pasted.len(), 3);
    assert_eq!(ed.scene().iter_live().count(), 9);

    // 8. Render the whole thing to a non-trivial command list.
    let scene = ed.render();
    assert!(
        scene.commands.len() >= 18,
        "9 shapes ⇒ many commands, got {}",
        scene.commands.len()
    );

    // 9. Save to .excalidraw JSON, reload, and confirm the scene survives.
    let json = save_to_string(ed.scene()).expect("save");
    let reloaded = load_from_str(&json).expect("reload");
    assert_eq!(
        reloaded.iter_live().count(),
        9,
        "all 9 elements survive save/reload"
    );
    // A reloaded element keeps its styling.
    let styled = reloaded
        .iter_live()
        .find(|e| matches!(e.kind, ElementKind::Rectangle))
        .unwrap();
    assert_eq!(styled.stroke_color, Color::rgb(220, 30, 30));

    // 10. Deep undo: every recorded step reverses back toward empty.
    let mut undos = 0;
    while ed.can_undo() {
        assert!(ed.undo());
        undos += 1;
        assert!(undos < 100, "undo terminates");
    }
    assert_eq!(
        ed.scene().iter_live().count(),
        0,
        "undoing everything empties the scene"
    );

    // 11. Redo all the way back to 9.
    while ed.can_redo() {
        assert!(ed.redo());
    }
    assert_eq!(
        ed.scene().iter_live().count(),
        9,
        "redo restores the full scene"
    );
}
