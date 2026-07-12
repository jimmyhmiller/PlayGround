//! The windowed application: winit event loop, camera input, per-frame GPU
//! layout stepping, rendering, and wiring graph algorithms to node colors.

use std::sync::Arc;
use std::time::Instant;

use nebula_core::{algorithms, Graph, Pos};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::Camera2D;
use crate::coloring;
use crate::gpu::Gpu;
use crate::layout_gpu::{LayoutGpu, LayoutSettings};
use crate::render::{RenderParams, Renderer};
use crate::scene::{pack_rgba, GpuGraph};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    Uniform,
    Components,
    Degree,
    PageRank,
    Coloring,
    Communities,
}

impl ColorMode {
    fn label(&self) -> &'static str {
        match self {
            ColorMode::Uniform => "uniform",
            ColorMode::Components => "components",
            ColorMode::Degree => "degree",
            ColorMode::PageRank => "pagerank",
            ColorMode::Coloring => "greedy-coloring",
            ColorMode::Communities => "communities",
        }
    }
}

/// Options passed in from the CLI.
pub struct RunOptions {
    pub title: String,
    pub k: f32,
    pub settings: LayoutSettings,
    /// If set, exit after this many rendered frames (for headless capture/bench).
    pub max_frames: Option<u64>,
    /// If set, save a PNG of the final frame before exiting.
    pub screenshot: Option<String>,
    /// Color mode to apply on startup.
    pub color_mode: ColorMode,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            title: "nebula".into(),
            k: 30.0,
            settings: LayoutSettings::default(),
            max_frames: None,
            screenshot: None,
            color_mode: ColorMode::Uniform,
        }
    }
}

/// GPU + scene state, alive only after the window exists.
struct Live {
    window: Arc<Window>,
    gpu: Gpu,
    graph_gpu: GpuGraph,
    renderer: Renderer,
    layout: LayoutGpu,
}

pub struct App {
    // Immutable inputs.
    graph: Graph,
    seed_positions: Vec<Pos>,
    opts: RunOptions,

    // Runtime.
    live: Option<Live>,
    camera: Camera2D,
    settings: LayoutSettings,
    render_params: RenderParams,
    color_mode: ColorMode,

    // Input state.
    cursor: glam::Vec2,
    dragging: bool,
    last_cursor: glam::Vec2,

    // Timing / stats.
    last_frame: Instant,
    frame_count: u64,
    fps_timer: Instant,
    fps: f32,
    total_steps: u64,
    rendered_frames: u64,
    should_exit: bool,
}

impl App {
    pub fn new(mut graph: Graph, seed_positions: Vec<Pos>, opts: RunOptions) -> Self {
        graph.ensure_csr();
        let settings = opts.settings;
        let color_mode = opts.color_mode;
        App {
            graph,
            seed_positions,
            opts,
            live: None,
            camera: Camera2D::new(glam::vec2(1280.0, 800.0)),
            settings,
            render_params: RenderParams::default(),
            color_mode,
            cursor: glam::Vec2::ZERO,
            dragging: false,
            last_cursor: glam::Vec2::ZERO,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            fps: 0.0,
            total_steps: 0,
            rendered_frames: 0,
            should_exit: false,
        }
    }

    pub fn run(self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = self;
        event_loop.run_app(&mut app)?;
        Ok(())
    }

    fn init_live(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title(&self.opts.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let gpu = pollster::block_on(Gpu::new(window.clone())).expect("gpu init");
        let graph_gpu =
            GpuGraph::upload(&gpu.device, &gpu.queue, &self.graph, &self.seed_positions, self.opts.k);
        let renderer = Renderer::new(&gpu.device, gpu.config.format, &graph_gpu);
        let layout = LayoutGpu::new(&gpu.device, &graph_gpu, &self.settings);

        // Fit camera to the seeded layout.
        self.camera.viewport = glam::vec2(gpu.size.width as f32, gpu.size.height as f32);
        let (min, max) = bounds(&self.seed_positions);
        self.camera.fit_bounds(min, max);

        self.live = Some(Live { window, gpu, graph_gpu, renderer, layout });
        self.apply_color_mode();
        self.update_title();
    }

    fn apply_color_mode(&mut self) {
        let Some(live) = self.live.as_mut() else { return };
        let n = self.graph.num_nodes() as usize;
        let (colors, sizes): (Vec<u32>, Option<Vec<f32>>) = match self.color_mode {
            ColorMode::Uniform => (vec![pack_rgba(120, 170, 255, 255); n], None),
            ColorMode::Components => {
                let labels = algorithms::connected_components(&self.graph);
                (coloring::categorical_colors(&labels), None)
            }
            ColorMode::Degree => {
                let deg = self.graph.degrees();
                let degf: Vec<f32> = deg.iter().map(|&d| d as f32).collect();
                (
                    coloring::sequential_colors_u32(&deg, true),
                    Some(coloring::sizes_from_scalar(&degf, 6.0)),
                )
            }
            ColorMode::PageRank => {
                let pr = algorithms::pagerank(&mut self.graph, 40, 0.85);
                (
                    coloring::sequential_colors_f32(&pr, true),
                    Some(coloring::sizes_from_scalar(&pr, 8.0)),
                )
            }
            ColorMode::Coloring => {
                let c = algorithms::greedy_coloring(&mut self.graph);
                (coloring::categorical_colors(&c), None)
            }
            ColorMode::Communities => {
                let labels = algorithms::label_propagation(&mut self.graph, 20);
                (coloring::categorical_colors(&labels), None)
            }
        };
        live.graph_gpu.set_colors(&live.gpu.queue, &colors);
        match sizes {
            Some(s) => live.graph_gpu.set_sizes(&live.gpu.queue, &s),
            None => live
                .graph_gpu
                .set_sizes(&live.gpu.queue, &vec![1.0f32; n]),
        }
        log::info!("color mode -> {}", self.color_mode.label());
    }

    fn update_title(&self) {
        let Some(live) = self.live.as_ref() else { return };
        let title = format!(
            "{} — {} nodes · {} edges · {} · {} · {:.0} fps · steps {}{}",
            self.opts.title,
            self.graph.num_nodes(),
            self.graph.num_edges(),
            self.color_mode.label(),
            if self.settings.running { "sim" } else { "paused" },
            self.fps,
            self.total_steps,
            if self.render_params.base_radius_px > 0.0 { "" } else { "" },
        );
        live.window.set_title(&title);
    }

    fn render(&mut self) {
        let Some(live) = self.live.as_mut() else { return };

        // Advance layout.
        if self.settings.running {
            let mut enc = live
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim") });
            for _ in 0..self.settings.substeps {
                live.layout.step(&mut enc);
                self.total_steps += 1;
            }
            live.gpu.queue.submit(Some(enc.finish()));
        }

        // Camera uniform.
        live.renderer.update_camera(&live.gpu.queue, &self.camera.uniform());
        live.renderer.update_params(&live.gpu.queue, &self.render_params);

        let frame = match live.gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                live.gpu.surface.configure(&live.gpu.device, &live.gpu.config);
                return;
            }
            Err(e) => {
                log::warn!("surface error: {e:?}");
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = live
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("draw") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.015,
                            g: 0.015,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            live.renderer.draw(&mut pass);
        }
        live.gpu.queue.submit(Some(enc.finish()));
        frame.present();

        // Headless frame limit / screenshot.
        self.rendered_frames += 1;
        if let Some(maxf) = self.opts.max_frames {
            if self.rendered_frames >= maxf {
                // Fit to the settled layout so the capture frames the real result.
                self.fit_view();
                self.capture_if_requested();
                self.should_exit = true;
            }
        }

        // FPS.
        self.frame_count += 1;
        let now = Instant::now();
        if now.duration_since(self.fps_timer).as_secs_f32() >= 0.5 {
            self.fps = self.frame_count as f32 / now.duration_since(self.fps_timer).as_secs_f32();
            self.frame_count = 0;
            self.fps_timer = now;
            self.update_title();
        }
        self.last_frame = now;
    }

    fn capture_if_requested(&self) {
        let (Some(path), Some(live)) = (self.opts.screenshot.as_ref(), self.live.as_ref()) else {
            return;
        };
        let (w, h) = (live.gpu.size.width, live.gpu.size.height);
        if let Err(e) = crate::screenshot::capture(
            &live.gpu,
            &live.graph_gpu,
            w,
            h,
            &self.camera.uniform(),
            &self.render_params,
            live.renderer.draw_edges,
            live.renderer.draw_nodes,
            path,
        ) {
            log::error!("screenshot failed: {e}");
        }
    }

    fn set_color_mode(&mut self, mode: ColorMode) {
        if self.color_mode != mode {
            self.color_mode = mode;
            self.apply_color_mode();
            self.update_title();
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => {
                self.settings.running = !self.settings.running;
                self.update_title();
            }
            KeyCode::Digit1 => self.set_color_mode(ColorMode::Uniform),
            KeyCode::Digit2 => self.set_color_mode(ColorMode::Components),
            KeyCode::Digit3 => self.set_color_mode(ColorMode::Degree),
            KeyCode::Digit4 => self.set_color_mode(ColorMode::PageRank),
            KeyCode::Digit5 => self.set_color_mode(ColorMode::Coloring),
            KeyCode::Digit6 => self.set_color_mode(ColorMode::Communities),
            KeyCode::KeyE => {
                if let Some(live) = self.live.as_mut() {
                    live.renderer.draw_edges = !live.renderer.draw_edges;
                }
            }
            KeyCode::KeyN => {
                if let Some(live) = self.live.as_mut() {
                    live.renderer.draw_nodes = !live.renderer.draw_nodes;
                }
            }
            KeyCode::KeyF => self.fit_view(),
            KeyCode::Equal => {
                self.render_params.base_radius_px = (self.render_params.base_radius_px * 1.3).min(64.0);
            }
            KeyCode::Minus => {
                self.render_params.base_radius_px = (self.render_params.base_radius_px / 1.3).max(0.5);
            }
            KeyCode::BracketLeft => {
                self.render_params.edge_alpha = (self.render_params.edge_alpha / 1.4).max(0.01);
                self.push_params();
            }
            KeyCode::BracketRight => {
                self.render_params.edge_alpha = (self.render_params.edge_alpha * 1.4).min(1.0);
                self.push_params();
            }
            _ => {}
        }
    }

    fn push_params(&mut self) {
        if let Some(live) = self.live.as_ref() {
            live.renderer.update_params(&live.gpu.queue, &self.render_params);
        }
    }

    /// Re-fit the camera. For manageable sizes we read positions back from the
    /// GPU for an exact fit; for very large graphs we fit to the world bounds.
    fn fit_view(&mut self) {
        let Some(live) = self.live.as_ref() else { return };
        if self.graph.num_nodes() <= 5_000_000 {
            if let Some(pos) = crate::readback::read_positions(&live.gpu, &live.graph_gpu) {
                let (min, max) = bounds(&pos);
                self.camera.fit_bounds(min, max);
                return;
            }
        }
        let h = live.graph_gpu.world_size * 0.35;
        self.camera.fit_bounds(glam::vec2(-h, -h), glam::vec2(h, h));
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.live.is_none() {
            self.init_live(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(live) = self.live.as_mut() {
                    live.gpu.resize(size);
                    self.camera.viewport = glam::vec2(size.width as f32, size.height as f32);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        self.handle_key(code, event_loop);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                    self.last_cursor = self.cursor;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new = glam::vec2(position.x as f32, position.y as f32);
                if self.dragging {
                    let delta = new - self.last_cursor;
                    self.camera.pan_pixels(delta);
                }
                self.last_cursor = new;
                self.cursor = new;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                let factor = (1.0 + scroll * 0.12).clamp(0.2, 5.0);
                self.camera.zoom_about(factor, self.cursor);
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if self.should_exit {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.should_exit {
            event_loop.exit();
            return;
        }
        if let Some(live) = self.live.as_ref() {
            live.window.request_redraw();
        }
    }
}

fn bounds(pos: &[Pos]) -> (glam::Vec2, glam::Vec2) {
    let mut min = glam::vec2(f32::MAX, f32::MAX);
    let mut max = glam::vec2(f32::MIN, f32::MIN);
    for p in pos {
        min = min.min(glam::vec2(p[0], p[1]));
        max = max.max(glam::vec2(p[0], p[1]));
    }
    if !min.x.is_finite() {
        min = glam::vec2(-100.0, -100.0);
        max = glam::vec2(100.0, 100.0);
    }
    (min, max)
}
