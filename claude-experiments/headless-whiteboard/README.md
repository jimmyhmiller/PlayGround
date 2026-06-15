# headless-whiteboard

A **headless, Excalidraw-parity whiteboard engine** written in Rust. It owns the
entire scene and interaction lifecycle — elements, tools, selection, hit-testing,
geometry, undo/redo, input handling — and emits a backend-agnostic **draw-command
list** each frame. It never rasterizes anything itself, so any renderer plugs in:
tiny-skia, Vello, wgpu, a web canvas, SVG, even a TUI.

This is an independent reimplementation, not a fork. The element model, file
format, geometry/hit-test math, and binding behavior are reimplemented from
[Excalidraw](https://github.com/excalidraw/excalidraw) (MIT); the hand-drawn
sketch generation is reimplemented from [Rough.js](https://github.com/rough-stuff/rough)
(MIT). See [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Why headless

The library is the *model + controller*; you bring the *view*. An app holds an
`Editor`, feeds it raw input events, and asks it to render:

```rust
use whiteboard_core::editor::Editor;
use whiteboard_core::text::MonospaceMeasurer;

// Inject a text measurer (only a real backend has fonts); pick clean or rough.
let mut editor = Editor::new_rough(MonospaceMeasurer::default());

editor.set_tool(Tool::Rectangle);
editor.handle(InputEvent::PointerDown { pos, button, mods }); // drag to draw…
editor.handle(InputEvent::PointerMove { pos, mods });
editor.handle(InputEvent::PointerUp   { pos, button, mods });

let scene = editor.render();   // -> RenderScene { commands: Vec<DrawCommand>, .. }
backend.render(&scene);        // any backend consumes the command list
```

The only things that flow *into* the library are capabilities a renderer owns:
**text measurement** (`TextMeasurer`) and image dimensions. Everything else —
including all interaction — lives in the headless core.

## Workspace

| Crate | Role |
|-------|------|
| `whiteboard-core` | **The library.** Headless, zero GPU/windowing deps. |
| `whiteboard-tiny-skia` | Reference CPU backend; rasterizes the command list to pixels/PNG. Powers snapshot tests. |
| `whiteboard-vello` | Vello GPU backend: builds a `vello::Scene` from the command list. |
| `whiteboard-svg` | SVG export backend: the command list → a standalone SVG document. |
| `examples/winit-draw` | Runnable window: draw with the mouse. |

`whiteboard-core` modules: `element` (model), `geometry` (primitives + bounds +
hit-test), `scene` (store, z-order, groups, frames), `rough` (rough.js port),
`shape` (element → geometry), `render` (draw commands + tessellator),
`interaction` (tool state machine), `history` (undo/redo), `io` (`.excalidraw`),
`text` (measurement + layout).

## Features

- **All Excalidraw element types**: rectangle, ellipse, diamond, line, arrow
  (bindings + arrowheads), freedraw, text, image, frame.
- **Sketchy look** by default — seeded, deterministic rough.js-style rendering
  with hachure / cross-hatch / zigzag / dots fills. A clean generator is also
  available.
- **Full interaction lifecycle**: drag-to-create, click/shift-click/marquee
  select, move, 8 resize handles, rotation, pan, wheel zoom, keyboard.
- **Undo/redo**, `.excalidraw` JSON load/save (real-format + internal),
  tight rotated bounds and precise per-shape hit-testing.
- **Real text rendering**: the core lays text out (word-wrap, alignment,
  container fitting) against an injected `TextMeasurer` and emits `DrawText`
  commands; the tiny-skia backend rasterizes glyphs with bundled DejaVu fonts
  (via `fontdue`). Use `whiteboard_tiny_skia::FontMeasurer` for font-accurate
  layout.
- **Frame clipping**: frames clip their children in the tessellator.
- **Selection overlay**: `Editor::render_with_overlay()` appends the selection
  bounding box, 8 resize handles, rotation handle, and the active marquee as
  draw commands a backend draws over the scene.
- **Arrow binding**: arrows bound to shapes follow them on move/resize — wired
  into the editor's event loop.
- **Elbow arrows, align/distribute, style edits**: orthogonal arrow routing
  (`LinearData.elbowed`); `Editor::align`/`distribute`/`set_style` over the
  selection (undoable); bound-text container auto-grow.
- **Eraser + laser tools**: eraser-drag soft-deletes (one undo per stroke);
  laser draws a transient pointer trail that never touches the scene.
- **Web backend**: `whiteboard-web` lowers the command list to HTML5 Canvas 2D
  ops (wasm-bindgen/web-sys), proving the browser target.
- **Load real `.excalidraw` files**: `render-file <in.excalidraw> <out.png|svg>`.
- **Clipboard + grouping**: copy/cut/paste/duplicate, group/ungroup, and
  frame membership reassignment on drag — all wired into the `Editor`.
- **Multiple backends**: a CPU rasterizer (tiny-skia), a GPU scene builder
  (Vello), and a vector exporter (SVG) all consume the same command list.

## Try it

```sh
# Headless render to a PNG (no window):
cargo run -p whiteboard-tiny-skia --example render_png -- out.png

# Interactive: draw with the mouse
cargo run -p winit-draw
#   r/o/d rect/ellipse/diamond · l/a line/arrow · f freedraw · v select
#   u undo · shift+u redo · Delete · wheel scroll · ctrl+wheel zoom · Esc quit
```

## Develop

```sh
cargo test --workspace          # 521 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

## License

MIT. See [`LICENSE`](LICENSE) and [`ATTRIBUTION.md`](ATTRIBUTION.md).
