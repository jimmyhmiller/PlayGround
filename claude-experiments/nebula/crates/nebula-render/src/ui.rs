//! egui-based control panel. Holds the egui plumbing (context, winit state,
//! wgpu renderer) and the persistent widget state for the panel. The actual
//! panel contents are built in `app.rs` (`App::build_ui`) so they can read and
//! mutate the full application state; this module owns only the integration.

use std::sync::Arc;
use winit::window::Window;

/// egui integration state, created once the window/GPU exist.
pub struct Ui {
    pub ctx: egui::Context,
    pub state: egui_winit::State,
    pub renderer: egui_wgpu::Renderer,
}

impl Ui {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &Arc<Window>,
    ) -> Self {
        let ctx = egui::Context::default();
        // Slightly larger default text for a native-app feel.
        ctx.style_mut(|s| {
            for (_, f) in s.text_styles.iter_mut() {
                f.size *= 1.05;
            }
        });
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );
        Ui { ctx, state, renderer }
    }

    /// Record an egui draw into `enc`, compositing over `view` (loads, does not
    /// clear). Call `free_textures` after the encoder is submitted.
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &UiFrame,
    ) {
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.0, size.1],
            pixels_per_point: frame.pixels_per_point,
        };
        for (id, delta) in &frame.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        self.renderer.update_buffers(device, queue, enc, &frame.jobs, &sd);
        let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.renderer.render(&mut pass.forget_lifetime(), &frame.jobs, &sd);
    }

    /// Free egui textures released during `frame`. Call after submitting.
    pub fn free_textures(&mut self, frame: &UiFrame) {
        for id in &frame.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

/// Tessellated egui output captured during the UI pass, consumed during render.
pub struct UiFrame {
    pub jobs: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}
