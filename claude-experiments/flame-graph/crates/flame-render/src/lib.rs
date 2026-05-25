//! GPU rendering for the flame graph viewer. Holds wgpu state, instance buffer,
//! glyphon text, and viewport pan/zoom math. The renderer is decoupled from the
//! windowing layer (winit) — `flame-viewer` wires the two together.

pub mod instance;
pub mod palette;
pub mod renderer;
pub mod viewport;

pub use renderer::{
    Direction, MainTab, MergeMode, Renderer, SidebarTab, TrackLayout, ROW_HEIGHT_PX,
};
pub use viewport::Viewport;
