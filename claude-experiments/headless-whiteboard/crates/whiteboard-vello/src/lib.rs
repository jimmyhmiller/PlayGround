//! GPU backend scaffold for `whiteboard-core` targeting [Vello].
//!
//! The full GPU path (pulling in `vello`/`wgpu`) lands in a later phase. To keep
//! the workspace buildable and dependency-light until then, this crate currently
//! defines the *conversion* from our backend-neutral [`DrawCommand`] vocabulary
//! into an intermediate, renderer-agnostic representation. When the Vello
//! dependency is added, the wgpu surface wiring slots in around this conversion
//! without changing the command contract.
//!
//! [Vello]: https://github.com/linebender/vello

use whiteboard_core::render::{DrawCommand, RenderScene};

/// A flattened, GPU-friendly view of one draw command. This is the seam a Vello
/// (or any retained-mode GPU) backend builds its scene from. It deliberately
/// carries the same information as [`DrawCommand`]; the value is in proving the
/// command list is renderer-agnostic and giving the GPU backend a typed target.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuOp<'a> {
    PushLayer,
    PopLayer,
    Fill(&'a DrawCommand),
    Stroke(&'a DrawCommand),
    Text(&'a DrawCommand),
    Image(&'a DrawCommand),
}

/// Walk a render scene and yield GPU ops in paint order. Pure and allocation-
/// free over the borrowed scene, so it is trivially testable without a GPU.
pub fn to_gpu_ops(scene: &RenderScene) -> Vec<GpuOp<'_>> {
    scene
        .commands
        .iter()
        .map(|cmd| match cmd {
            DrawCommand::PushClip(_) | DrawCommand::PushTransform(_) => GpuOp::PushLayer,
            DrawCommand::PopClip | DrawCommand::PopTransform => GpuOp::PopLayer,
            DrawCommand::FillPath { .. } => GpuOp::Fill(cmd),
            DrawCommand::StrokePath { .. } => GpuOp::Stroke(cmd),
            DrawCommand::DrawText { .. } => GpuOp::Text(cmd),
            DrawCommand::DrawImage { .. } => GpuOp::Image(cmd),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::geometry::{Path, Point};
    use whiteboard_core::render::{Color, Paint};

    #[test]
    fn maps_fill_command() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::FillPath {
            path: Path::polygon(&[
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 1.0),
            ]),
            paint: Paint::solid(Color::BLACK),
        });
        let ops = to_gpu_ops(&scene);
        assert!(matches!(ops.as_slice(), [GpuOp::Fill(_)]));
    }
}
