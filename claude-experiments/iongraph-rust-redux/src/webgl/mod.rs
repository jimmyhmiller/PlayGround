//! WebGL 2 renderer for IonGraph
//!
//! This module provides a WebGL-based rendering backend that implements
//! the `LayoutProvider` trait, enabling hardware-accelerated graph visualization
//! with full interactivity (pan, zoom, selection, hover).

pub mod element;
pub mod scene;
pub mod state;
pub mod shaders;
pub mod text;
pub mod tessellation;
pub mod renderer;
pub mod interaction;
pub mod provider;

// Re-export main types
pub use element::{WebGLElement, ElementId, Rect};
pub use scene::{Scene, DrawCall, DrawCallOwner, Primitive};
pub use state::{Viewport, InteractionState};
pub use renderer::WebGLRenderer;
pub use provider::WebGLLayoutProvider;
