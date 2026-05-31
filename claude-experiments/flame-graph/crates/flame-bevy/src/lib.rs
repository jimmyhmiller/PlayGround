//! Embed the flame-graph renderer inside a Bevy app.
//!
//! # Quick start
//!
//! ```no_run
//! use bevy::prelude::*;
//! use flame_bevy::{FlameGraph, FlameGraphInput, FlameGraphPlugin};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(FlameGraphPlugin)
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
//!     let image = images.add(FlameGraph::blank_image(800, 600));
//!     commands.spawn((
//!         FlameGraph::new(image.clone(), (800, 600)),
//!         FlameGraphInput::default(),
//!         Sprite::from_image(image),
//!     ));
//! }
//! ```
//!
//! # Architecture
//!
//! `flame-render` pins glyphon 0.11, which forces wgpu 29. Bevy 0.18 pins
//! wgpu 27. Their `wgpu::Device` types are distinct Rust types from
//! different crate versions, so they cannot share GPU resources directly.
//!
//! This plugin works around the mismatch by giving the flame-graph its own
//! isolated wgpu 29 stack (see `gpu::FlameGpu`). Each frame, the renderer
//! draws into an offscreen texture, the pixels are read back to CPU, and
//! the bytes are copied into a `bevy::Image` asset. Bevy then composites
//! that image however the host app wants — as a `Sprite`, a `UiImage`, a
//! material on a 3D mesh, anything that takes an `Image` handle.
//!
//! The readback path is the price we pay for version isolation. Cost is one
//! tightly-packed RGBA8 copy per redraw: ~8 MB at 1080p, sub-millisecond
//! upload, dominated by the synchronous `map_async` poll (a few ms). The
//! system only redraws when state actually changed, so a static profile
//! sitting on screen costs ~zero.
//!
//! # Public surface
//!
//! - [`FlameGraphPlugin`]: add to your Bevy `App`.
//! - [`FlameGraph`]: per-entity component holding the renderer state and the
//!   target `Handle<Image>`. Mutate via [`FlameGraph::set_profile`],
//!   [`FlameGraph::renderer_mut`], etc.
//! - [`FlameGraphInput`]: per-entity component. Add it on the same entity to
//!   get Bevy input forwarded to the renderer; omit it if you want to drive
//!   the renderer manually.
//!
//! # Embed modes
//!
//! - **Embedded panel (RTT, recommended):** spawn one entity with a
//!   `FlameGraph` and place its `Handle<Image>` into a `Sprite` or `UiImage`.
//!   Set `FlameGraphInput::panel_origin` to where the panel renders in the
//!   window so cursor input maps correctly.
//! - **Full-window:** spawn the same entity but size it to the window and
//!   leave `panel_origin` at `Vec2::ZERO`. Drop the resize-image system from
//!   the schedule if you want a fixed canvas; otherwise see
//!   `examples/embed_fullscreen.rs` for a window-tracking setup.
//!
//! See `examples/embed_panel.rs` and `examples/embed_fullscreen.rs`.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use flame_core::Profile;
use flame_render::Renderer;

mod gpu;
mod input;

use gpu::FlameGpu;

pub use flame_core;
pub use flame_render;
pub use input::{forward_input, FlameGraphInput};

/// Add to your Bevy `App` to enable embedded flame-graph rendering.
///
/// Registers the [`FlameGraph`] component, the input-forwarding system, and
/// the render-and-upload system that copies pixels into your `Image` assets.
pub struct FlameGraphPlugin;

impl Plugin for FlameGraphPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                sync_input_panel_size,
                input::forward_input,
                resize_image_to_panel,
                render_and_upload,
            )
                .chain(),
        );
    }
}

/// Per-entity flame-graph state.
///
/// Owns a private wgpu device and a `flame-render::Renderer`. The component
/// is `!Send + !Sync` (wgpu types are not portable across threads on all
/// platforms); Bevy will keep it on the main thread automatically.
///
/// Construct via [`FlameGraph::new`] passing the `Image` handle you want the
/// flame graph painted into and the panel's pixel size.
#[derive(Component)]
pub struct FlameGraph {
    gpu: FlameGpu,
    image: Handle<Image>,
    /// Set to `true` when something the renderer paints has changed, cleared
    /// when `render_and_upload` re-paints. `FlameGraphInput` sets this when
    /// it mutates renderer state; the host should also call
    /// `mark_dirty()` after mutating directly via `renderer_mut()`.
    dirty: bool,
}

impl FlameGraph {
    /// Build a fresh flame-graph attached to `image` at `size` pixels.
    ///
    /// `image` is the asset handle that will receive the rendered pixels each
    /// frame. The image's existing data is overwritten on the first redraw;
    /// callers typically create it via [`FlameGraph::blank_image`] and put it
    /// straight into `Assets<Image>`.
    pub fn new(image: Handle<Image>, size: (u32, u32)) -> Self {
        let mut gpu = FlameGpu::new(size);
        gpu.renderer.rebuild_instances();
        Self {
            gpu,
            image,
            dirty: true,
        }
    }

    /// Convenience: build a transparent `bevy::Image` of `(w, h)` pixels,
    /// suitable as a starting point for `FlameGraph::new`. Uses the same
    /// `Rgba8UnormSrgb` format that the readback produces.
    pub fn blank_image(w: u32, h: u32) -> Image {
        let mut img = Image::new_fill(
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            bevy::asset::RenderAssetUsages::default(),
        );
        // We blit raw pixels into `data` from CPU, no GPU upload involved on
        // the read side; allow main-world access.
        img.asset_usage = bevy::asset::RenderAssetUsages::default();
        img
    }

    /// Asset handle that the flame graph paints into. Hand this to a
    /// `Sprite`, `UiImage`, or any other consumer of `Handle<Image>`.
    pub fn image(&self) -> Handle<Image> {
        self.image.clone()
    }

    /// Current panel size in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.gpu.size
    }

    /// Resize the rendered canvas. Triggers a rebuild on the next frame.
    /// Cheap when the size hasn't actually changed.
    pub fn set_size(&mut self, w: u32, h: u32) {
        if (w, h) == self.gpu.size {
            return;
        }
        self.gpu.resize(w, h);
        self.dirty = true;
    }

    /// Replace the profile being displayed. Fits the viewport.
    pub fn set_profile(&mut self, profile: Arc<Profile>) {
        self.gpu.renderer.set_profile(profile);
        self.gpu.renderer.rebuild_instances();
        self.dirty = true;
    }

    /// Direct access to the underlying renderer for advanced use (sandwich
    /// view, group key, sequence-lifeline key, etc.). Always call
    /// [`Self::mark_dirty`] after mutating, or no redraw will be scheduled.
    pub fn renderer_mut(&mut self) -> &mut Renderer {
        &mut self.gpu.renderer
    }

    /// Read-only access to the renderer (selection, layouts, viewport).
    pub fn renderer(&self) -> &Renderer {
        &self.gpu.renderer
    }

    /// Schedule a redraw on the next frame. Call after mutating the renderer
    /// via [`Self::renderer_mut`] from outside the plugin.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}

/// System: when an entity has a `FlameGraphInput`, keep its `panel_size`
/// matched to the `FlameGraph`'s current canvas size. Host code only needs
/// to set `panel_origin`.
fn sync_input_panel_size(mut q: Query<(&FlameGraph, &mut FlameGraphInput)>) {
    for (flame, mut input) in &mut q {
        let (w, h) = flame.gpu.size;
        let want = Vec2::new(w as f32, h as f32);
        if input.panel_size != want {
            input.panel_size = want;
        }
    }
}

/// System: if the target `Image` asset's size disagrees with the panel size
/// (e.g. user code resized it externally), update the renderer to match. We
/// treat the `Image` extent as the source of truth so host UI layout drives
/// canvas size.
fn resize_image_to_panel(mut q: Query<&mut FlameGraph>, images: Res<Assets<Image>>) {
    for mut flame in &mut q {
        let Some(img) = images.get(&flame.image) else { continue };
        let want = (img.texture_descriptor.size.width, img.texture_descriptor.size.height);
        if want != flame.gpu.size && want.0 > 0 && want.1 > 0 {
            flame.gpu.resize(want.0, want.1);
            flame.dirty = true;
        }
    }
}

/// System: redraw any dirty `FlameGraph`s and blit the pixels into the
/// linked `Image` asset.
fn render_and_upload(mut q: Query<&mut FlameGraph>, mut images: ResMut<Assets<Image>>) {
    for mut flame in &mut q {
        if !flame.dirty {
            continue;
        }
        // Split borrow: we need &mut Image and &mut FlameGpu separately.
        let (w, h) = flame.gpu.render_and_readback();
        let Some(img) = images.get_mut(&flame.image) else {
            log::warn!(
                "flame-bevy: FlameGraph's image handle is not in Assets<Image>; \
                 dropping this frame"
            );
            flame.dirty = false;
            continue;
        };
        let expected_len = (w * h * 4) as usize;
        if img
            .data
            .as_ref()
            .map(|d| d.len())
            .unwrap_or(0)
            != expected_len
        {
            // Either the image was freshly created (data: None) or our size
            // changed underneath the asset. Reallocate.
            img.texture_descriptor.size = Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            };
            img.data = Some(vec![0; expected_len]);
        }
        if let Some(dst) = img.data.as_mut() {
            dst.copy_from_slice(&flame.gpu.pixels);
        }
        flame.dirty = false;
    }
}
