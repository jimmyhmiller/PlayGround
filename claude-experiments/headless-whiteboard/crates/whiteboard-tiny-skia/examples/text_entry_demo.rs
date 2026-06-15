use whiteboard_core::editor::Editor;
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Key, Modifiers, PointerButton, Tool};
use whiteboard_core::render::{Backend, Color};
use whiteboard_tiny_skia::{FontMeasurer, TinySkiaBackend};

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "text_entry.png".into());
    let mut ed = Editor::new(FontMeasurer::new());

    // Place + type a heading using the real text-entry path.
    ed.set_tool(Tool::Text);
    let click = |x, y| {
        [
            InputEvent::PointerDown {
                pos: Point::new(x, y),
                button: PointerButton::Primary,
                mods: Modifiers::default(),
            },
            InputEvent::PointerUp {
                pos: Point::new(x, y),
                button: PointerButton::Primary,
                mods: Modifiers::default(),
            },
        ]
    };
    for ev in click(40.0, 40.0) {
        ed.handle(ev);
    }
    for c in "Typed via text entry!".chars() {
        ed.handle(InputEvent::KeyDown {
            key: Key::Char(c),
            mods: Modifiers::default(),
        });
    }
    ed.handle(InputEvent::KeyDown {
        key: Key::Enter,
        mods: Modifiers::default(),
    });
    for c in "second line".chars() {
        ed.handle(InputEvent::KeyDown {
            key: Key::Char(c),
            mods: Modifiers::default(),
        });
    }
    ed.handle(InputEvent::KeyDown {
        key: Key::Escape,
        mods: Modifiers::default(),
    });

    let scene = ed.render();
    let mut backend = TinySkiaBackend::new(360, 160).with_background(Color::WHITE);
    backend.render(&scene);
    backend.save_png(&out).expect("png");
    println!(
        "text element box: {:?}",
        ed.scene().iter_live().next().map(|e| (e.width, e.height))
    );
}
