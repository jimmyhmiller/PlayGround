//! GPU rendering for the flame graph viewer. Holds wgpu state, instance buffer,
//! glyphon text, and viewport pan/zoom math.
//!
//! The renderer does **not** own a window or a swapchain. The caller supplies a
//! `wgpu::Device`, `wgpu::Queue`, and target `wgpu::TextureFormat`, and on each
//! frame passes a `wgpu::TextureView` to render into. This makes the same code
//! usable from a standalone winit app (`flame-viewer`) or embedded inside a
//! Bevy app (`flame-bevy`) where Bevy owns the device, queue, and final
//! presentation.
//!
//! See `crates/flame-bevy` for the Bevy integration and the workspace README
//! for embedding instructions.

pub mod instance;
pub mod palette;
pub mod renderer;
pub mod viewport;

pub use renderer::{
    Direction, MainTab, MergeMode, Renderer, SidebarTab, TrackLayout, ROW_HEIGHT_PX,
};
pub use viewport::Viewport;
