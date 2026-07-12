//! A 2D pan/zoom camera. World space is graph coordinates; the camera maps a
//! world point to normalized device coordinates (NDC) via a center + per-axis
//! scale, keeping the transform trivial (no full matrix needed) and aspect-correct.

use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy)]
pub struct Camera2D {
    /// World coordinate currently at the center of the screen.
    pub center: glam::Vec2,
    /// Zoom: world-units-per-... higher = more zoomed in. Specifically the
    /// number of pixels one world unit occupies.
    pub zoom: f32,
    pub viewport: glam::Vec2,
}

impl Camera2D {
    pub fn new(viewport: glam::Vec2) -> Self {
        Camera2D {
            center: glam::Vec2::ZERO,
            zoom: 1.0,
            viewport,
        }
    }

    /// Fit the camera to a world-space bounding box with a margin.
    pub fn fit_bounds(&mut self, min: glam::Vec2, max: glam::Vec2) {
        let size = (max - min).max(glam::Vec2::splat(1.0));
        self.center = (min + max) * 0.5;
        let zx = self.viewport.x / size.x;
        let zy = self.viewport.y / size.y;
        self.zoom = zx.min(zy) * 0.9; // 10% margin
        if !self.zoom.is_finite() || self.zoom <= 0.0 {
            self.zoom = 1.0;
        }
    }

    /// Zoom by a factor about a screen-space anchor (in pixels, origin top-left),
    /// keeping the world point under the cursor fixed.
    pub fn zoom_about(&mut self, factor: f32, screen: glam::Vec2) {
        let before = self.screen_to_world(screen);
        self.zoom = (self.zoom * factor).clamp(1e-7, 1e7);
        let after = self.screen_to_world(screen);
        self.center += before - after;
    }

    /// Pan by a pixel delta.
    pub fn pan_pixels(&mut self, delta: glam::Vec2) {
        self.center -= glam::vec2(delta.x, -delta.y) / self.zoom;
    }

    pub fn screen_to_world(&self, screen: glam::Vec2) -> glam::Vec2 {
        // screen: pixels, origin top-left, y down.
        let centered = screen - self.viewport * 0.5;
        let world_offset = glam::vec2(centered.x, -centered.y) / self.zoom;
        self.center + world_offset
    }

    pub fn uniform(&self) -> CameraUniform {
        // world -> NDC:  ndc = (world - center) * zoom / (viewport/2)
        let scale = glam::vec2(
            2.0 * self.zoom / self.viewport.x,
            2.0 * self.zoom / self.viewport.y,
        );
        CameraUniform {
            center: self.center.into(),
            scale: scale.into(),
            viewport: self.viewport.into(),
            zoom: self.zoom,
            _pad: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub center: [f32; 2],
    pub scale: [f32; 2],
    pub viewport: [f32; 2],
    pub zoom: f32,
    pub _pad: f32,
}
