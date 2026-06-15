//! The backend contract: a flat, immutable list of draw commands.
//!
//! This is the heart of the "headless" design. The library never touches a GPU,
//! a canvas, or a font. Instead, [`crate::editor::Editor::render`] produces a
//! [`RenderScene`] — an ordered `Vec<DrawCommand>` — and *any* backend consumes
//! it: tiny-skia, Vello, wgpu, a web canvas, an SVG writer, even a TUI.
//!
//! Commands are emitted in paint order (first = bottom). Transform and clip
//! commands form balanced push/pop pairs that backends apply as a stack.
//!
//! The [`Tessellator`] that turns a `Scene` into commands lands in Phase 1; this
//! module defines the vocabulary every backend and that tessellator share.

mod clip;
mod overlay;
mod paint;
mod tessellate;

pub use overlay::{selection_overlay, OverlayStyle};
pub use paint::{Color, FillStyle, LineCap, LineJoin, Paint, Stroke, StrokeStyle};
pub use tessellate::{tessellate, RenderOptions};

use crate::geometry::{Path, Rect, Transform};
use crate::text::TextRun;
use serde::{Deserialize, Serialize};

/// Opaque handle identifying image bytes the backend has been given out-of-band
/// (the core never holds pixel data). The backend resolves it to a texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageId(pub String);

/// A single drawing instruction. Backends translate these into their native API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DrawCommand {
    /// Intersect the clip region with `Rect` until the matching `PopClip`.
    PushClip(Rect),
    PopClip,
    /// Push a transform onto the stack until the matching `PopTransform`.
    /// Subsequent geometry is in the pre-transform coordinate space.
    PushTransform(Transform),
    PopTransform,
    /// Fill a path's interior (non-zero winding).
    FillPath {
        path: Path,
        paint: Paint,
    },
    /// Stroke a path's outline.
    StrokePath {
        path: Path,
        stroke: Stroke,
        paint: Paint,
    },
    /// Draw a laid-out line of text.
    DrawText {
        run: TextRun,
        paint: Paint,
    },
    /// Draw an image into `dst`, scaled to fit, at the given opacity.
    DrawImage {
        id: ImageId,
        dst: Rect,
        opacity: f32,
    },
}

/// The full output of a render pass: ordered commands plus the scene bounds they
/// cover (useful for sizing a surface or computing dirty regions).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderScene {
    pub commands: Vec<DrawCommand>,
    pub bounds: Rect,
}

impl RenderScene {
    pub fn new() -> Self {
        RenderScene {
            commands: Vec::new(),
            bounds: Rect::EMPTY,
        }
    }

    pub fn push(&mut self, cmd: DrawCommand) {
        self.commands.push(cmd);
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }
}

/// Convenience trait a backend may implement. Implementing it is optional — a
/// backend can just match on `&[DrawCommand]` directly — but it documents the
/// expected entry point and lets generic code accept "any backend".
pub trait Backend {
    fn render(&mut self, scene: &RenderScene);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Path, Point};

    #[test]
    fn render_scene_collects_commands() {
        let mut scene = RenderScene::new();
        assert!(scene.is_empty());
        scene.push(DrawCommand::FillPath {
            path: Path::polygon(&[
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 1.0),
            ]),
            paint: Paint::solid(Color::BLACK),
        });
        assert_eq!(scene.len(), 1);
    }

    #[test]
    fn commands_are_serializable() {
        // The whole command vocabulary must round-trip through serde so that
        // snapshot tests and tooling can inspect frames as data.
        let cmd = DrawCommand::PushClip(Rect::new(0.0, 0.0, 10.0, 10.0));
        let json = serde_json::to_string(&cmd).unwrap();
        let back: DrawCommand = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, back);
    }
}
