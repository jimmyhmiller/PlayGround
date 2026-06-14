//! # whiteboard-core
//!
//! A **headless** Excalidraw-parity whiteboard engine. It owns the entire scene
//! and interaction lifecycle — elements, tools, selection, hit-testing,
//! geometry, undo/redo, input handling — and emits a backend-agnostic
//! [`render::RenderScene`] (a flat list of [`render::DrawCommand`]s) each frame.
//!
//! The library never rasterizes anything. Plug in any renderer (tiny-skia,
//! Vello, wgpu, a web canvas, SVG, a TUI) by consuming the draw-command list,
//! and inject text measurement via [`text::TextMeasurer`]. See [`editor::Editor`]
//! for the top-level facade an application holds.
//!
//! ## Layout
//! - [`geometry`] — points, vectors, rects, transforms, paths, bounds, hit-test
//! - [`element`] — the element model (the shared foundation type)
//! - [`scene`] — the element store, z-order, groups, frames
//! - [`rough`] — hand-drawn ("sketchy") shape generation (rough.js port)
//! - [`shape`] — element → geometry path generation
//! - [`render`] — the draw-command vocabulary + tessellator
//! - [`interaction`] — tools and the pointer/keyboard state machine
//! - [`history`] — undo / redo
//! - [`io`] — `.excalidraw` load / save
//! - [`text`] — text measurement trait + layout
//!
//! Attribution for the Excalidraw and Rough.js algorithms reimplemented here is
//! in `ATTRIBUTION.md` at the repository root.

pub mod element;
pub mod geometry;
pub mod history;
pub mod interaction;
pub mod io;
pub mod render;
pub mod rough;
pub mod scene;
pub mod shape;
pub mod text;

pub mod editor;

// Re-export the most commonly used types at the crate root for ergonomics.
pub use element::{Element, ElementId, ElementKind};
pub use geometry::{Point, Rect, Transform, Vec2};
pub use render::{Color, DrawCommand, RenderScene};
pub use text::{TextMeasurer, TextMetrics};
