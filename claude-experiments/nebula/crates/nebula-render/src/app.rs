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
use crate::overlay::Overlay;
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
    pub draw_edges: bool,
    pub draw_nodes: bool,
    /// Preselect a node (mainly for scripted/headless captures).
    pub select: Option<u32>,
    /// Start with the help overlay visible.
    pub show_help: bool,
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
            draw_edges: true,
            draw_nodes: true,
            select: None,
            show_help: false,
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
    overlay: Overlay,
}

pub struct App {
    // Immutable inputs.
    graph: Graph,
    seed_positions: Vec<Pos>,
    opts: RunOptions,
    labels: Option<Vec<String>>,

    // Runtime.
    live: Option<Live>,
    camera: Camera2D,
    settings: LayoutSettings,
    render_params: RenderParams,
    color_mode: ColorMode,

    // Selection / overlay.
    selected: Option<u32>,
    selected_pos: Option<glam::Vec2>,
    last_values: Option<Vec<f32>>,
    value_name: &'static str,
    show_help: bool,
    show_hud: bool,

    // Input state.
    cursor: glam::Vec2,
    dragging: bool,
    press_pos: glam::Vec2,
    moved_since_press: bool,
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
    pub fn new(graph: Graph, seed_positions: Vec<Pos>, opts: RunOptions) -> Self {
        Self::with_labels(graph, seed_positions, opts, None)
    }

    pub fn with_labels(
        mut graph: Graph,
        seed_positions: Vec<Pos>,
        opts: RunOptions,
        labels: Option<Vec<String>>,
    ) -> Self {
        graph.ensure_csr();
        let settings = opts.settings;
        let color_mode = opts.color_mode;
        let selected = opts.select;
        let show_help = opts.show_help;
        App {
            graph,
            seed_positions,
            opts,
            labels,
            live: None,
            camera: Camera2D::new(glam::vec2(1280.0, 800.0)),
            settings,
            render_params: RenderParams::default(),
            color_mode,
            selected,
            selected_pos: None,
            last_values: None,
            value_name: "",
            show_help,
            show_hud: true,
            cursor: glam::Vec2::ZERO,
            dragging: false,
            press_pos: glam::Vec2::ZERO,
            moved_since_press: false,
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
        let overlay = Overlay::new(&gpu.device, gpu.config.format);

        // Fit camera to the seeded layout.
        self.camera.viewport = glam::vec2(gpu.size.width as f32, gpu.size.height as f32);
        let (min, max) = bounds(&self.seed_positions);
        self.camera.fit_bounds(min, max);

        let mut renderer = renderer;
        renderer.draw_edges = self.opts.draw_edges;
        renderer.draw_nodes = self.opts.draw_nodes;
        self.live = Some(Live { window, gpu, graph_gpu, renderer, layout, overlay });
        self.apply_color_mode();
        self.update_title();
    }

    fn apply_color_mode(&mut self) {
        if self.live.is_none() {
            return;
        }
        let n = self.graph.num_nodes() as usize;
        self.value_name = "";
        self.last_values = None;
        let (colors, sizes): (Vec<u32>, Option<Vec<f32>>) = match self.color_mode {
            ColorMode::Uniform => (vec![pack_rgba(120, 170, 255, 255); n], None),
            ColorMode::Components => {
                let labels = algorithms::connected_components(&self.graph);
                self.value_name = "component";
                self.last_values = Some(labels.iter().map(|&l| l as f32).collect());
                (coloring::categorical_colors(&labels), None)
            }
            ColorMode::Degree => {
                let deg = self.graph.degrees();
                let degf: Vec<f32> = deg.iter().map(|&d| d as f32).collect();
                self.value_name = "degree";
                self.last_values = Some(degf.clone());
                (
                    coloring::sequential_colors_u32(&deg, true),
                    Some(coloring::sizes_from_scalar(&degf, 6.0)),
                )
            }
            ColorMode::PageRank => {
                let pr = algorithms::pagerank(&mut self.graph, 40, 0.85);
                self.value_name = "pagerank";
                self.last_values = Some(pr.clone());
                (
                    coloring::sequential_colors_f32(&pr, true),
                    Some(coloring::sizes_from_scalar(&pr, 8.0)),
                )
            }
            ColorMode::Coloring => {
                let c = algorithms::greedy_coloring(&mut self.graph);
                self.value_name = "color";
                self.last_values = Some(c.iter().map(|&x| x as f32).collect());
                (coloring::categorical_colors(&c), None)
            }
            ColorMode::Communities => {
                let labels = algorithms::label_propagation(&mut self.graph, 20);
                self.value_name = "community";
                self.last_values = Some(labels.iter().map(|&l| l as f32).collect());
                (coloring::categorical_colors(&labels), None)
            }
        };
        let Some(live) = self.live.as_mut() else { return };
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
        if self.live.is_none() {
            return;
        }

        // Refresh the selected node's position (cheap 8-byte readback) so the
        // marker + info panel track it as the simulation moves it.
        if let Some(sel) = self.selected {
            let sp = self
                .live
                .as_ref()
                .and_then(|live| crate::readback::read_one_position(&live.gpu, &live.graph_gpu, sel))
                .map(|p| glam::vec2(p[0], p[1]));
            if sp.is_none() {
                log::warn!("selected node {sel}: position readback returned None");
            }
            self.selected_pos = sp;
        }
        // Build overlay commands before borrowing live mutably.
        let overlay_cmds = self.build_overlay();

        let live = self.live.as_mut().unwrap();

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

            // Overlay (HUD, info panel, selection marker) on top.
            live.overlay.begin();
            for (x, y, w, h, col) in &overlay_cmds.rects {
                live.overlay.rect(*x, *y, *w, *h, *col);
            }
            for (x, y, s, col, txt) in &overlay_cmds.texts {
                live.overlay.text(*x, *y, *s, *col, txt);
            }
            let vp = (live.gpu.size.width as f32, live.gpu.size.height as f32);
            live.overlay
                .draw(&live.gpu.device, &live.gpu.queue, vp, &mut pass);
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
            log::info!(
                "{:.1} fps · {} sim-steps/frame · {} nodes · {} edges",
                self.fps,
                self.settings.substeps,
                self.graph.num_nodes(),
                self.graph.num_edges()
            );
        }
        self.last_frame = now;
    }

    /// Build owned overlay draw commands (panels + text). Reads only `&self` so
    /// it composes without fighting the borrow checker against the live GPU state.
    fn build_overlay(&self) -> OverlayCmds {
        let mut c = OverlayCmds::default();
        let vp = self.camera.viewport;
        let scale = (vp.y / 800.0).clamp(1.5, 3.0); // hidpi-aware text size
        let line = crate::overlay::GLYPH_H * scale + 4.0;
        let white = pack_rgba(235, 238, 245, 255);
        let dim = pack_rgba(150, 158, 175, 255);
        let panel_bg = pack_rgba(12, 14, 26, 220);
        let accent = pack_rgba(120, 200, 255, 255);

        // --- Top-left stats HUD ---
        if self.show_hud {
            let pad = 10.0;
            let lines = vec![
                (accent, "nebula".to_string()),
                (
                    white,
                    format!("{} nodes  {} edges", commafy(self.graph.num_nodes()), commafy(self.graph.num_edges())),
                ),
                (
                    dim,
                    format!(
                        "{:.0} fps   {}   {}",
                        self.fps,
                        self.color_mode.label(),
                        if self.settings.running { "simulating" } else { "paused" }
                    ),
                ),
            ];
            self.panel(&mut c, pad, pad, scale, line, panel_bg, &lines);
        }

        // --- Bottom-left controls ---
        if self.show_help {
            let help = [
                "drag pan / scroll zoom / F fit",
                "click a node to inspect it",
                "1 uniform  2 components  3 degree",
                "4 pagerank  5 coloring  6 communities",
                "space pause / E edges / N nodes",
                "+/- node size / [ ] edge brightness",
                "H help / Tab hud / C clear / Esc quit",
            ];
            let lines: Vec<(u32, String)> = help.iter().map(|s| (dim, s.to_string())).collect();
            let h = help.len() as f32 * line + 12.0;
            self.panel(&mut c, 10.0, vp.y - h - 10.0, scale, line, panel_bg, &lines);
        } else {
            c.texts.push((10.0, vp.y - line - 6.0, scale, dim, "H  help".to_string()));
        }

        // --- Selected node info panel + marker ---
        if let (Some(id), Some(wp)) = (self.selected, self.selected_pos) {
            let mut lines: Vec<(u32, String)> = Vec::new();
            let label = self
                .labels
                .as_ref()
                .and_then(|l| l.get(id as usize))
                .cloned()
                .unwrap_or_else(|| id.to_string());
            lines.push((accent, format!("node {label}")));
            lines.push((dim, format!("index {id}")));
            if let Some(csr) = self.graph.csr_ref() {
                let deg = csr.degree(id);
                lines.push((white, format!("degree {deg}")));
                // First few neighbors.
                let nbrs = csr.neighbors(id);
                let show: Vec<String> = nbrs
                    .iter()
                    .take(8)
                    .map(|&nb| {
                        self.labels
                            .as_ref()
                            .and_then(|l| l.get(nb as usize))
                            .cloned()
                            .unwrap_or_else(|| nb.to_string())
                    })
                    .collect();
                let mut s = show.join(" ");
                if nbrs.len() > 8 {
                    s.push_str(" …");
                }
                lines.push((dim, format!("adj: {s}")));
            }
            if let Some(vals) = self.last_values.as_ref() {
                if let Some(v) = vals.get(id as usize) {
                    lines.push((white, format!("{} {:.4}", self.value_name, v)));
                }
            }
            lines.push((dim, format!("pos {:.0}, {:.0}", wp.x, wp.y)));

            // Panel top-right.
            let maxw = lines
                .iter()
                .map(|(_, s)| crate::overlay::Overlay::text_width(s, scale))
                .fold(0.0f32, f32::max);
            let pw = maxw + 20.0;
            let px = vp.x - pw - 10.0;
            self.panel(&mut c, px, 10.0, scale, line, panel_bg, &lines);

            // Marker: a square outline around the node's screen position.
            let sp = self.camera.world_to_screen(wp);
            let r = 12.0;
            let t = 2.0;
            let ring = accent;
            c.rects.push((sp.x - r, sp.y - r, 2.0 * r, t, ring)); // top
            c.rects.push((sp.x - r, sp.y + r - t, 2.0 * r, t, ring)); // bottom
            c.rects.push((sp.x - r, sp.y - r, t, 2.0 * r, ring)); // left
            c.rects.push((sp.x + r - t, sp.y - r, t, 2.0 * r, ring)); // right
        }

        c
    }

    /// Draw a panel: a background rect sized to the text plus the lines.
    fn panel(
        &self,
        c: &mut OverlayCmds,
        x: f32,
        y: f32,
        scale: f32,
        line: f32,
        bg: u32,
        lines: &[(u32, String)],
    ) {
        let maxw = lines
            .iter()
            .map(|(_, s)| crate::overlay::Overlay::text_width(s, scale))
            .fold(0.0f32, f32::max);
        let w = maxw + 20.0;
        let h = lines.len() as f32 * line + 12.0;
        c.rects.push((x, y, w, h, bg));
        let mut ty = y + 8.0;
        for (col, s) in lines {
            c.texts.push((x + 10.0, ty, scale, *col, s.clone()));
            ty += line;
        }
    }

    fn capture_if_requested(&mut self) {
        if self.opts.screenshot.is_none() {
            return;
        }
        // Render the real scene (graph + overlay) to an offscreen texture matching
        // the swapchain format, then save it — so the capture includes the HUD.
        let overlay_cmds = self.build_overlay();
        let path = self.opts.screenshot.clone().unwrap();
        let Some(live) = self.live.as_mut() else { return };
        let (w, h) = (live.gpu.size.width, live.gpu.size.height);
        let format = live.gpu.config.format;

        live.renderer.update_camera(&live.gpu.queue, &self.camera.uniform());
        live.renderer.update_params(&live.gpu.queue, &self.render_params);

        let tex = live.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture_tex"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = live
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("capture_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.015, g: 0.015, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            live.renderer.draw(&mut pass);
            live.overlay.begin();
            for (x, y, rw, rh, col) in &overlay_cmds.rects {
                live.overlay.rect(*x, *y, *rw, *rh, *col);
            }
            for (x, y, s, col, txt) in &overlay_cmds.texts {
                live.overlay.text(*x, *y, *s, *col, txt);
            }
            live.overlay
                .draw(&live.gpu.device, &live.gpu.queue, (w as f32, h as f32), &mut pass);
        }
        live.gpu.queue.submit(Some(enc.finish()));

        if let Err(e) = crate::screenshot::save_texture(&live.gpu, &tex, w, h, format, &path) {
            log::error!("screenshot failed: {e}");
        }
    }

    /// Pick a node under a screen pixel and select it (updating the info panel).
    fn pick_at(&mut self, screen: glam::Vec2) {
        // Ensure the pick pass uses the current camera.
        let picked = {
            let Some(live) = self.live.as_ref() else { return };
            live.renderer.update_camera(&live.gpu.queue, &self.camera.uniform());
            let (w, h) = (live.gpu.size.width, live.gpu.size.height);
            live.renderer.pick(
                &live.gpu.device,
                &live.gpu.queue,
                w,
                h,
                screen.x.max(0.0) as u32,
                screen.y.max(0.0) as u32,
            )
        };
        self.selected = picked;
        if let Some(id) = picked {
            let deg = self.graph.csr_ref().map(|c| c.degree(id)).unwrap_or(0);
            let label = self
                .labels
                .as_ref()
                .and_then(|l| l.get(id as usize))
                .cloned()
                .unwrap_or_else(|| id.to_string());
            log::info!("selected node {label} (index {id}, degree {deg})");
        } else {
            self.selected_pos = None;
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
            KeyCode::KeyH => self.show_help = !self.show_help,
            KeyCode::Tab => self.show_hud = !self.show_hud,
            KeyCode::KeyC => {
                self.selected = None;
                self.selected_pos = None;
            }
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
                    match state {
                        ElementState::Pressed => {
                            self.dragging = true;
                            self.press_pos = self.cursor;
                            self.moved_since_press = false;
                            self.last_cursor = self.cursor;
                        }
                        ElementState::Released => {
                            self.dragging = false;
                            // A click (negligible movement) selects a node.
                            if !self.moved_since_press {
                                self.pick_at(self.cursor);
                            }
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new = glam::vec2(position.x as f32, position.y as f32);
                if self.dragging {
                    let delta = new - self.last_cursor;
                    self.camera.pan_pixels(delta);
                    if (new - self.press_pos).length() > 4.0 {
                        self.moved_since_press = true;
                    }
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

/// Owned overlay draw list: rects `(x,y,w,h,color)` and text `(x,y,scale,color,str)`.
#[derive(Default)]
struct OverlayCmds {
    rects: Vec<(f32, f32, f32, f32, u32)>,
    texts: Vec<(f32, f32, f32, u32, String)>,
}

/// Group digits with thousands separators: 1234567 -> "1,234,567".
fn commafy(n: u64) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in s.chars().enumerate() {
        if i != 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
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
