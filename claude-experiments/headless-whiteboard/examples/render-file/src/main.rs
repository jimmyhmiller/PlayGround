//! `render-file` — load a real `.excalidraw` file and render it to an image.
//!
//! Usage:
//!
//! ```text
//! render-file <input.excalidraw> <output.png|output.svg>
//! ```
//!
//! The output format is chosen from the output file extension:
//!
//! * `.png` — rasterized with the `whiteboard-tiny-skia` backend.
//! * `.svg` — exported as a vector document with the `whiteboard-svg` backend.
//!
//! The canvas size is auto-fit to the scene's bounding box (plus a small
//! margin), and the viewport is scrolled so the whole scene is visible.
//!
//! This binary is glue around the library: it never re-implements parsing or
//! rendering — it loads via [`whiteboard_core::io::excalidraw::load_excalidraw_str`],
//! builds an [`Editor`], and renders with [`Editor::render`].

use std::path::Path;
use std::process::ExitCode;

use whiteboard_core::editor::Editor;
use whiteboard_core::element::Element;
use whiteboard_core::interaction::Viewport;
use whiteboard_core::io::excalidraw::load_excalidraw_str;
use whiteboard_core::render::RenderScene;
use whiteboard_core::Vec2;

use whiteboard_core::render::Backend;
use whiteboard_svg::to_svg;
use whiteboard_tiny_skia::{FontMeasurer, TinySkiaBackend};

/// Margin (scene units) added around the scene bounds when sizing the canvas.
const MARGIN: f64 = 24.0;
/// Fallback canvas size when the scene is empty or degenerate.
const FALLBACK_SIZE: u32 = 64;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: render-file <input.excalidraw> <output.png|output.svg>");
        return ExitCode::FAILURE;
    }
    let input = &args[1];
    let output = &args[2];

    match run(input, output) {
        Ok(()) => {
            eprintln!("wrote {output}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Output formats this CLI can write, selected by extension.
enum Format {
    Png,
    Svg,
}

fn detect_format(output: &str) -> Result<Format, String> {
    match Path::new(output)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Ok(Format::Png),
        Some("svg") => Ok(Format::Svg),
        other => Err(format!(
            "unsupported output extension {other:?}; use .png or .svg"
        )),
    }
}

fn run(input: &str, output: &str) -> Result<(), String> {
    let format = detect_format(output)?;

    let text = std::fs::read_to_string(input).map_err(|e| format!("reading {input}: {e}"))?;
    let elements = load_excalidraw_str(&text).map_err(|e| format!("parsing {input}: {e}"))?;

    // Auto-fit: compute scene bounds, derive canvas size + scroll.
    let (width, height, viewport) = fit(&elements);

    let mut editor = Editor::new_rough(FontMeasurer::new());
    editor.set_viewport(viewport);
    for el in elements {
        editor.add_element(el);
    }

    let scene = editor.render();

    match format {
        Format::Png => write_png(&scene, width, height, output),
        Format::Svg => write_svg(&scene, width, height, output),
    }
}

/// Compute the canvas dimensions and viewport that fit `elements` (plus a
/// margin) into the visible area. Returns `(width, height, viewport)`.
fn fit(elements: &[Element]) -> (u32, u32, Viewport) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for el in elements {
        if el.is_deleted {
            continue;
        }
        min_x = min_x.min(el.x);
        min_y = min_y.min(el.y);
        max_x = max_x.max(el.x + el.width);
        max_y = max_y.max(el.y + el.height);
    }

    if !min_x.is_finite() || !min_y.is_finite() || max_x < min_x || max_y < min_y {
        // No live elements: emit a small placeholder canvas at the origin.
        return (FALLBACK_SIZE, FALLBACK_SIZE, Viewport::default());
    }

    let origin_x = min_x - MARGIN;
    let origin_y = min_y - MARGIN;
    let width = (max_x - min_x + 2.0 * MARGIN).ceil().max(1.0) as u32;
    let height = (max_y - min_y + 2.0 * MARGIN).ceil().max(1.0) as u32;

    // Scroll so the (min - margin) corner lands at the screen origin; zoom 1.0.
    let viewport = Viewport {
        scroll: Vec2::new(origin_x, origin_y),
        zoom: 1.0,
    };
    (width, height, viewport)
}

fn write_png(scene: &RenderScene, width: u32, height: u32, output: &str) -> Result<(), String> {
    let mut backend = TinySkiaBackend::new(width, height);
    backend.render(scene);
    backend
        .save_png(output)
        .map_err(|e| format!("writing PNG {output}: {e}"))
}

fn write_svg(scene: &RenderScene, width: u32, height: u32, output: &str) -> Result<(), String> {
    let doc = to_svg(scene, width, height);
    std::fs::write(output, doc).map_err(|e| format!("writing SVG {output}: {e}"))
}
