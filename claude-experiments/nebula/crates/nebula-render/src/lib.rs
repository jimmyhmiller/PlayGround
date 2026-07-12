//! `nebula-render` — GPU rendering and GPU compute layout (wgpu).
//!
//! The public surface is small: build a [`Graph`](nebula_core::Graph), seed
//! positions with a [`Layout`](nebula_layout::Layout), then hand both to
//! [`App::run`]. Positions and velocities then live entirely on the GPU: the
//! compute layout writes them and the renderer reads them with no CPU round-trip.

pub mod app;
pub mod camera;
pub mod coloring;
pub mod gpu;
pub mod layout_gpu;
pub mod readback;
pub mod render;
pub mod scene;

pub use app::{App, ColorMode, RunOptions};
pub use layout_gpu::LayoutSettings;
