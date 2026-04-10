use bytemuck::{Pod, Zeroable};
use font8x8::UnicodeFonts;
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use crate::AppState;
use crate::viz;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OverlayVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LineVertex { position: [f32; 3] }

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
    _pad: f32,
}

const SIDEBAR_WIDTH_FRAC: f32 = 0.20;
const SIDEBAR_MARGIN: f32 = 10.0;
const SIDEBAR_GLYPH_SCALE: f32 = 2.0;
const SIDEBAR_LINE_HEIGHT: f32 = 20.0;

/// Wall camera: looks at the XY wall from +Z. Pan with drag, zoom with scroll.
/// Slight tilt available via right-drag for seeing 3D attention structures.
struct WallCamera {
    /// Center of view in world XY
    center_x: f32,
    center_y: f32,
    /// Distance from wall (zoom level)
    distance: f32,
    /// Slight tilt angles for peeking at 3D structure
    tilt_x: f32,
    tilt_y: f32,
}

impl WallCamera {
    fn new() -> Self {
        Self { center_x: 0.0, center_y: -15.0, distance: 40.0, tilt_x: 0.0, tilt_y: 0.0 }
    }

    fn view_matrix(&self) -> Mat4 {
        let eye = Vec3::new(
            self.center_x + self.tilt_x * self.distance * 0.3,
            self.center_y + self.tilt_y * self.distance * 0.3,
            self.distance,
        );
        let target = Vec3::new(self.center_x, self.center_y, 0.0);
        Mat4::look_at_rh(eye, target, Vec3::Y)
    }

    fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 500.0)
    }

    /// Convert a pixel drag to world-space pan delta.
    fn pan_scale(&self) -> f32 {
        self.distance * 0.003
    }
}

fn cube_wireframe_vertices(min: Vec3, max: Vec3) -> Vec<LineVertex> {
    let c = [
        Vec3::new(min.x, min.y, min.z), Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z), Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z), Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z), Vec3::new(min.x, max.y, max.z),
    ];
    [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        .iter().flat_map(|&(a,b)| [LineVertex { position: c[a].into() }, LineVertex { position: c[b].into() }])
        .collect()
}

// --- Sidebar ---
fn pixel_to_ndc(x: f32, y: f32, w: f32, h: f32) -> [f32; 2] {
    [(x / w.max(1.0)) * 2.0 - 1.0, 1.0 - (y / h.max(1.0)) * 2.0]
}
fn push_quad(v: &mut Vec<OverlayVertex>, w: f32, h: f32, x0: f32, y0: f32, x1: f32, y1: f32, c: [f32; 4]) {
    let p00 = pixel_to_ndc(x0,y0,w,h); let p10 = pixel_to_ndc(x1,y0,w,h);
    let p11 = pixel_to_ndc(x1,y1,w,h); let p01 = pixel_to_ndc(x0,y1,w,h);
    v.extend_from_slice(&[
        OverlayVertex{position:p00,color:c}, OverlayVertex{position:p10,color:c},
        OverlayVertex{position:p11,color:c}, OverlayVertex{position:p00,color:c},
        OverlayVertex{position:p11,color:c}, OverlayVertex{position:p01,color:c},
    ]);
}

fn build_sidebar(text: &str, width: f32, height: f32) -> Vec<OverlayVertex> {
    let sw = width * SIDEBAR_WIDTH_FRAC;
    let mut verts = Vec::new();
    push_quad(&mut verts, width, height, 0.0, 0.0, sw, height, [0.05, 0.06, 0.08, 0.96]);
    push_quad(&mut verts, width, height, sw - 2.0, 0.0, sw, height, [0.16, 0.18, 0.22, 1.0]);

    let glyph_px = 8.0 * SIDEBAR_GLYPH_SCALE;
    let max_chars = ((sw - SIDEBAR_MARGIN * 2.0).max(glyph_px) / glyph_px).floor().max(1.0) as usize;
    let max_lines = ((height - SIDEBAR_MARGIN * 2.0) / SIDEBAR_LINE_HEIGHT).floor().max(1.0) as usize;

    let lines: Vec<&str> = text.lines().take(max_lines).collect();
    let normal = [0.9, 0.92, 0.96, 1.0];
    let dim = [0.5, 0.55, 0.6, 1.0];
    let highlight = [0.3, 0.9, 0.5, 1.0];

    for (li, line) in lines.iter().enumerate() {
        let y = SIDEBAR_MARGIN + li as f32 * SIDEBAR_LINE_HEIGHT;
        let color = if line.starts_with(">>>") { highlight }
            else if line.starts_with("---") || line.starts_with("===") { dim }
            else { normal };
        for (ci, ch) in line.chars().take(max_chars).enumerate() {
            let Some(glyph) = font8x8::BASIC_FONTS.get(ch) else { continue };
            let x = SIDEBAR_MARGIN + ci as f32 * glyph_px;
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits >> col) & 1 == 0 { continue; }
                    let px = x + col as f32 * SIDEBAR_GLYPH_SCALE;
                    let py = y + row as f32 * SIDEBAR_GLYPH_SCALE;
                    push_quad(&mut verts, width, height, px, py,
                        px + SIDEBAR_GLYPH_SCALE, py + SIDEBAR_GLYPH_SCALE, color);
                }
            }
        }
    }
    verts
}

// --- GPU state ---
struct GpuState {
    surface: wgpu::Surface<'static>, device: wgpu::Device, queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    point_pipeline: wgpu::RenderPipeline, line_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    point_buffer: wgpu::Buffer, point_count: u32,
    line_buffer: wgpu::Buffer, line_vertex_count: u32,
    sidebar_buffer: wgpu::Buffer, sidebar_vertex_count: u32,
    uniform_buffer: wgpu::Buffer, bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
}

pub struct App {
    window: Option<Arc<Window>>, gpu: Option<GpuState>,
    camera: WallCamera,
    left_pressed: bool,
    right_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    point_size: f32,
    state: AppState,
    inspect_text: Option<String>,
    current_labels: Vec<String>,
    dirty: bool,
}

impl App {
    pub fn new(state: AppState) -> Self {
        Self {
            window: None, gpu: None,
            camera: WallCamera::new(),
            left_pressed: false, right_pressed: false, last_mouse: None,
            point_size: 3.0,
            state, inspect_text: None, current_labels: vec![], dirty: true,
        }
    }

    pub fn run(self) {
        let event_loop = EventLoop::new().unwrap();
        let mut app = self;
        event_loop.run_app(&mut app).unwrap();
    }

    fn sidebar_text(&self) -> String {
        let s = &self.state;
        let n_total = s.n_nodes();
        let n_done = s.n_computed();
        let n_tok = s.n_tokens();

        // Show tokens with focus highlighted
        let tok_parts: Vec<String> = s.all_token_strings.iter().take(n_tok).enumerate()
            .map(|(i, t)| {
                let clean = t.replace("Ġ", " ");
                if i == s.focus_token { format!("[{}]", clean.trim()) } else { clean }
            }).collect();
        let tok_display = tok_parts.join("");

        let mut lines = vec![
            "GPT-2 Wall".into(),
            "===========================".into(),
            format!("FOCUS  {}/{n_tok}: {:?}", s.focus_token + 1, s.focus_token_str()),
            format!(">>> {tok_display}"),
            format!("NODES  {n_done}/{n_total}"),
            if s.computing { "COMPUTING...".into() } else { String::new() },
        ];

        if let Some(logits) = &s.logits {
            let preds = logits.top_k_predictions(&s.model.config, 5);
            lines.push("NEXT TOKEN".into());
            for (i, &(idx, logit)) in preds.iter().enumerate() {
                let tok = s.vocab_strings.get(idx).map(|t| t.as_str()).unwrap_or("?");
                lines.push(format!("  {}: {:?} ({:.1})", i + 1, tok, logit));
            }
        } else {
            lines.push("COMPUTING...".into());
        }

        if let Some(sel) = &self.inspect_text {
            lines.push(String::new());
            lines.push("SELECTED".into());
            lines.extend(sel.lines().map(str::to_owned));
        }

        lines.push(String::new());
        lines.push("---".into());
        lines.push("L-drag   pan".into());
        lines.push("R-drag   tilt (3D)".into());
        lines.push("wheel    zoom".into());
        lines.push("left/right step window".into());
        lines.push("up/down    pan".into());
        lines.push("space      generate".into());
        lines.push("+/-        point size".into());
        lines.join("\n")
    }

    fn rebuild_viz(&mut self) {
        if !self.dirty { return; }
        self.dirty = false;

        let tiles = viz::build_wall(
            &self.state.tile_values,
            &self.state.model.node_infos,
            &self.state.model.config,
            self.state.n_tokens(),
            self.state.focus_token,
            self.state.computing,
        );

        let mut all_verts = Vec::new();
        let mut all_labels = Vec::new();
        for (tile, _, _) in &tiles {
            let base = all_verts.len();
            all_verts.extend_from_slice(&tile.vertices);
            all_labels.extend(tile.labels.iter().cloned());
        }
        self.current_labels = all_labels;

        if let Some(gpu) = self.gpu.as_mut() {
            if !all_verts.is_empty() {
                gpu.point_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("points"), contents: bytemuck::cast_slice(&all_verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                gpu.point_count = all_verts.len() as u32;
            } else {
                gpu.point_count = 0;
            }
        }
        self.update_sidebar();
        self.update_title();
    }

    fn update_sidebar(&mut self) {
        let text = self.sidebar_text();
        let Some(gpu) = self.gpu.as_mut() else { return };
        let verts = build_sidebar(&text, gpu.config.width as f32, gpu.config.height as f32);
        gpu.sidebar_vertex_count = verts.len() as u32;
        gpu.sidebar_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sidebar"), contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
    }

    fn update_title(&self) {
        if let Some(w) = &self.window {
            w.set_title(&format!("GPT-2 Wall — {} tokens, {}/{} nodes",
                self.state.all_token_strings.len(), self.state.n_computed(), self.state.n_nodes()));
        }
    }

    fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..Default::default() });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { compatible_surface: Some(&surface), ..Default::default() }).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits { max_buffer_size: 1024 * 1024 * 1024, ..wgpu::Limits::default() }, ..Default::default()
        }).await.unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format,
            width: size.width.max(1), height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync, alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"), source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms { view_proj: Mat4::IDENTITY.to_cols_array_2d(), point_size: self.point_size, screen_width: 1280.0, screen_height: 960.0, _pad: 0.0 }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"), entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"), layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"), bind_group_layouts: &[&bgl], immediate_size: 0,
        });

        let ds = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less, stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default(),
        });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"), layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<PointVertex>() as u64, step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x4 }],
                }], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent::OVER }), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: ds.clone(), multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"), layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_line"),
                buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<LineVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 }],
                }], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: ds.clone(), multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"), layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_overlay"),
                buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<OverlayVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                        wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 }],
                }], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_overlay"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent::OVER }), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"), contents: &[0u8; std::mem::size_of::<PointVertex>()], usage: wgpu::BufferUsages::VERTEX });
        let line_verts = cube_wireframe_vertices(Vec3::splat(-1.0), Vec3::splat(1.0));
        let line_vertex_count = line_verts.len() as u32;
        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lines"), contents: bytemuck::cast_slice(&line_verts), usage: wgpu::BufferUsages::VERTEX });
        let sidebar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sidebar"), contents: &[0u8; std::mem::size_of::<OverlayVertex>()], usage: wgpu::BufferUsages::VERTEX });
        let depth_texture_view = Self::create_depth_texture(&device, config.width, config.height);

        self.gpu = Some(GpuState {
            surface, device, queue, config, point_pipeline, line_pipeline, overlay_pipeline,
            point_buffer, point_count: 0, line_buffer, line_vertex_count,
            sidebar_buffer, sidebar_vertex_count: 0, uniform_buffer, bind_group, depth_texture_view,
        });
        self.rebuild_viz();
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let gpu = self.gpu.as_mut().unwrap();
        if new_size.width > 0 && new_size.height > 0 {
            gpu.config.width = new_size.width; gpu.config.height = new_size.height;
            gpu.surface.configure(&gpu.device, &gpu.config);
            gpu.depth_texture_view = Self::create_depth_texture(&gpu.device, new_size.width, new_size.height);
        }
        self.update_sidebar();
    }

    fn render(&mut self) {
        // Poll for background forward result
        if self.state.poll() {
            self.dirty = true;
        }
        self.rebuild_viz();
        let gpu = self.gpu.as_ref().unwrap();
        let w = gpu.config.width as f32; let h = gpu.config.height as f32;
        let vp_x = w * SIDEBAR_WIDTH_FRAC; let vp_w = w - vp_x;

        let vp = self.camera.projection_matrix(vp_w / h) * self.camera.view_matrix();
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&Uniforms {
            view_proj: vp.to_cols_array_2d(), point_size: self.point_size, screen_width: vp_w, screen_height: h, _pad: 0.0,
        }));

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => { self.resize(PhysicalSize::new(gpu.config.width, gpu.config.height)); return; }
            Err(e) => { eprintln!("Surface: {e}"); return; }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let gpu = self.gpu.as_ref().unwrap();
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, depth_slice: None, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }), store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_texture_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None,
                }), ..Default::default()
            });
            pass.set_viewport(vp_x, 0.0, vp_w, h, 0.0, 1.0);
            if gpu.point_count > 0 {
                pass.set_pipeline(&gpu.point_pipeline); pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.point_buffer.slice(..));
                pass.draw(0..6, 0..gpu.point_count);
            }
        }

        if gpu.sidebar_vertex_count > 0 {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sidebar"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, depth_slice: None, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, ..Default::default()
            });
            pass.set_pipeline(&gpu.overlay_pipeline); pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.sidebar_buffer.slice(..));
            pass.draw(0..gpu.sidebar_vertex_count, 0..1);
        }

        gpu.queue.submit(std::iter::once(enc.finish()));
        frame.present();
    }

    fn inspect_at_cursor(&mut self, cx: f64, cy: f64) {
        if self.current_labels.is_empty() { return; }
        let Some(gpu) = self.gpu.as_ref() else { return };
        let w = gpu.config.width as f32; let h = gpu.config.height as f32;
        let vp_x = w * SIDEBAR_WIDTH_FRAC; let vp_w = w - vp_x;
        if cx < vp_x as f64 { return; }

        // Rebuild points to get positions (TODO: cache)
        let tiles = viz::build_wall(&self.state.tile_values, &self.state.model.node_infos,
            &self.state.model.config, self.state.n_tokens(), self.state.focus_token, false);
        let mut all_verts = Vec::new();
        let mut all_labels = Vec::new();
        for (tile, _, _) in &tiles {
            all_verts.extend_from_slice(&tile.vertices);
            all_labels.extend(tile.labels.iter().cloned());
        }

        let vp = self.camera.projection_matrix(vp_w / h) * self.camera.view_matrix();
        let threshold_sq = (self.point_size * 4.0).max(8.0).powi(2);
        let mut best: Option<(usize, f32, f32)> = None;
        for (i, v) in all_verts.iter().enumerate() {
            let clip = vp * Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
            if clip.w <= 0.0 { continue; }
            let ndc = clip.truncate() / clip.w;
            if ndc.z < -1.0 || ndc.z > 1.0 { continue; }
            let sx = vp_x + (ndc.x * 0.5 + 0.5) * vp_w;
            let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * h;
            let d2 = (sx - cx as f32).powi(2) + (sy - cy as f32).powi(2);
            if d2 > threshold_sq { continue; }
            match best { Some((_, bd, bz)) if d2 > bd && ndc.z >= bz => {} _ => best = Some((i, d2, ndc.z)) }
        }
        if let Some((idx, _, _)) = best {
            if let Some(label) = all_labels.get(idx) {
                eprintln!("Inspect: {}", label.replace('\n', " | "));
                self.inspect_text = Some(label.clone());
                self.update_sidebar();
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes().with_title("GPT-2 Wall").with_inner_size(PhysicalSize::new(1400u32, 960));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());
            pollster::block_on(self.init_gpu(window));
        }
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => { self.resize(size); self.window.as_ref().unwrap().request_redraw(); }
            WindowEvent::RedrawRequested => { self.render(); self.window.as_ref().unwrap().request_redraw(); }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.left_pressed = state == ElementState::Pressed;
                if !self.left_pressed { self.last_mouse = None; }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                self.right_pressed = state == ElementState::Pressed;
                if !self.right_pressed { self.last_mouse = None; }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = self.last_mouse {
                    let dx = (position.x - lx) as f32;
                    let dy = (position.y - ly) as f32;
                    if self.left_pressed {
                        // Pan across the wall
                        let scale = self.camera.pan_scale();
                        self.camera.center_x -= dx * scale;
                        self.camera.center_y += dy * scale;
                    }
                    if self.right_pressed {
                        // Tilt to see 3D structure
                        self.camera.tilt_x = (self.camera.tilt_x + dx * 0.003).clamp(-0.8, 0.8);
                        self.camera.tilt_y = (self.camera.tilt_y - dy * 0.003).clamp(-0.8, 0.8);
                    }
                }
                self.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta { MouseScrollDelta::LineDelta(_, y) => y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01 };
                self.camera.distance = (self.camera.distance * (1.0 - scroll * 0.08)).clamp(1.0, 100.0);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                use winit::keyboard::{Key, NamedKey};
                match event.logical_key {
                    Key::Named(NamedKey::ArrowDown) => {
                        self.camera.center_y -= 2.0;
                        self.dirty = true;
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        self.camera.center_y += 2.0;
                        self.dirty = true;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.state.focus_prev();
                        self.dirty = true;
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        self.state.focus_next();
                        self.dirty = true;
                    }
                    Key::Named(NamedKey::Space) => {
                        self.state.generate_one();
                        self.dirty = true;
                    }
                    Key::Character(ref c) if c.as_str() == "=" || c.as_str() == "+" => { self.point_size = (self.point_size + 0.5).min(20.0); }
                    Key::Character(ref c) if c.as_str() == "-" => { self.point_size = (self.point_size - 0.5).max(1.0); }
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}
