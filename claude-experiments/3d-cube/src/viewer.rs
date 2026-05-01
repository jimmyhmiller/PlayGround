use crate::ViewMode;
use crate::source::{
    InspectPoint, LoadResult, MinimapRow, MinimapVertex, PointCloudSource, PointVertex,
};
use bytemuck::{Pod, Zeroable};
use font8x8::UnicodeFonts;
use glam::{Mat4, Vec3, Vec4};
use memmap2::Mmap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

const SIDEBAR_WIDTH_FRAC: f32 = 0.24;
const MINIMAP_HEIGHT_FRAC: f32 = 0.26;
const SIDEBAR_MARGIN: f32 = 10.0;
const SIDEBAR_GLYPH_SCALE: f32 = 2.5;
const SIDEBAR_LINE_HEIGHT: f32 = 24.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
    _pad: f32,
}

struct OrbitalCamera {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitalCamera {
    fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.0,
            yaw: 0.8,
            pitch: 0.6,
        }
    }

    fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.sin();
        self.target + Vec3::new(x, y, z)
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 100.0)
    }
}

fn cube_wireframe_vertices(min: Vec3, max: Vec3) -> Vec<LineVertex> {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    edges
        .iter()
        .flat_map(|&(a, b)| {
            [
                LineVertex {
                    position: corners[a].into(),
                },
                LineVertex {
                    position: corners[b].into(),
                },
            ]
        })
        .collect()
}

fn build_minimap_vertices(rows: &[MinimapRow], sel_start: f32, sel_end: f32) -> Vec<MinimapVertex> {
    if rows.is_empty() {
        return vec![];
    }

    let num_rows = rows.len();
    let x_left = -1.0_f32;
    let x_right = -1.0 + 2.0 * SIDEBAR_WIDTH_FRAC;
    let row_height = 2.0 / num_rows as f32;

    let mut verts = Vec::with_capacity(num_rows * 6 + 24);

    for (row, info) in rows.iter().enumerate() {
        let t = info.avg_byte / 255.0;
        let r = (t * 2.0).clamp(0.0, 1.0);
        let g = (1.0 - (t - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let b = (1.0 - t).clamp(0.0, 1.0) * 0.8;
        let brightness = 0.3 + info.entropy * 0.7;

        let y_top = 1.0 - row as f32 * row_height;
        let y_bot = y_top - row_height;
        let color = [r * brightness, g * brightness, b * brightness, 1.0];

        verts.push(MinimapVertex {
            position: [x_left, y_top],
            color,
        });
        verts.push(MinimapVertex {
            position: [x_right, y_top],
            color,
        });
        verts.push(MinimapVertex {
            position: [x_right, y_bot],
            color,
        });
        verts.push(MinimapVertex {
            position: [x_left, y_top],
            color,
        });
        verts.push(MinimapVertex {
            position: [x_right, y_bot],
            color,
        });
        verts.push(MinimapVertex {
            position: [x_left, y_bot],
            color,
        });
    }

    // Dim outside selection
    let sel_y_top = 1.0 - sel_start * 2.0;
    let sel_y_bot = 1.0 - sel_end * 2.0;
    let dim_color = [0.0_f32, 0.0, 0.0, 0.7];

    if sel_start > 0.001 {
        verts.push(MinimapVertex {
            position: [x_left, 1.0],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, 1.0],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, sel_y_top],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_left, 1.0],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, sel_y_top],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_left, sel_y_top],
            color: dim_color,
        });
    }

    if sel_end < 0.999 {
        verts.push(MinimapVertex {
            position: [x_left, sel_y_bot],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, sel_y_bot],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, -1.0],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_left, sel_y_bot],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, -1.0],
            color: dim_color,
        });
        verts.push(MinimapVertex {
            position: [x_left, -1.0],
            color: dim_color,
        });
    }

    let border_h = 0.003;
    let border_color = [1.0_f32, 1.0, 1.0, 0.9];
    for &y in &[sel_y_top, sel_y_bot] {
        verts.push(MinimapVertex {
            position: [x_left, y + border_h],
            color: border_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, y + border_h],
            color: border_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, y - border_h],
            color: border_color,
        });
        verts.push(MinimapVertex {
            position: [x_left, y + border_h],
            color: border_color,
        });
        verts.push(MinimapVertex {
            position: [x_right, y - border_h],
            color: border_color,
        });
        verts.push(MinimapVertex {
            position: [x_left, y - border_h],
            color: border_color,
        });
    }

    verts
}

fn pixel_to_ndc(x: f32, y: f32, width: f32, height: f32) -> [f32; 2] {
    [
        (x / width.max(1.0)) * 2.0 - 1.0,
        1.0 - (y / height.max(1.0)) * 2.0,
    ]
}

fn push_quad(
    verts: &mut Vec<MinimapVertex>,
    width: f32,
    height: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    let p00 = pixel_to_ndc(x0, y0, width, height);
    let p10 = pixel_to_ndc(x1, y0, width, height);
    let p11 = pixel_to_ndc(x1, y1, width, height);
    let p01 = pixel_to_ndc(x0, y1, width, height);
    verts.push(MinimapVertex {
        position: p00,
        color,
    });
    verts.push(MinimapVertex {
        position: p10,
        color,
    });
    verts.push(MinimapVertex {
        position: p11,
        color,
    });
    verts.push(MinimapVertex {
        position: p00,
        color,
    });
    verts.push(MinimapVertex {
        position: p11,
        color,
    });
    verts.push(MinimapVertex {
        position: p01,
        color,
    });
}

fn wrap_text_line(line: &str, max_chars: usize, out: &mut Vec<String>) {
    if max_chars == 0 {
        return;
    }
    if line.is_empty() {
        out.push(String::new());
        return;
    }

    let mut current = String::new();
    for word in line.split_whitespace() {
        let word_len = word.chars().count();
        let current_len = current.chars().count();
        let sep = if current.is_empty() { 0 } else { 1 };

        if current_len + sep + word_len <= max_chars {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
            continue;
        }

        if !current.is_empty() {
            out.push(current);
            current = String::new();
        }

        if word_len <= max_chars {
            current.push_str(word);
            continue;
        }

        let chars: Vec<char> = word.chars().collect();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + max_chars).min(chars.len());
            out.push(chars[start..end].iter().collect());
            start = end;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }
}

fn build_sidebar_vertices(text: &str, width: f32, height: f32) -> Vec<MinimapVertex> {
    let sidebar_width = width * SIDEBAR_WIDTH_FRAC;
    let text_bottom = height * (1.0 - MINIMAP_HEIGHT_FRAC);
    let panel_right = (sidebar_width - 1.0).max(0.0);
    let panel_bottom = (text_bottom - 1.0).max(0.0);
    let mut verts = Vec::new();

    push_quad(
        &mut verts,
        width,
        height,
        0.0,
        0.0,
        panel_right,
        panel_bottom,
        [0.05, 0.06, 0.08, 0.96],
    );
    push_quad(
        &mut verts,
        width,
        height,
        panel_right,
        0.0,
        sidebar_width,
        panel_bottom,
        [0.16, 0.18, 0.22, 1.0],
    );

    let glyph_px = 8.0 * SIDEBAR_GLYPH_SCALE;
    let usable_width = (sidebar_width - SIDEBAR_MARGIN * 2.0).max(glyph_px);
    let max_chars = (usable_width / glyph_px).floor().max(1.0) as usize;
    let max_lines = ((text_bottom - SIDEBAR_MARGIN * 2.0) / SIDEBAR_LINE_HEIGHT)
        .floor()
        .max(1.0) as usize;

    let mut lines = Vec::new();
    for line in text.lines() {
        wrap_text_line(line, max_chars, &mut lines);
    }
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        if let Some(last) = lines.last_mut() {
            let keep = last.chars().count().saturating_sub(3);
            *last = if keep == 0 {
                String::from("...")
            } else {
                format!("{}...", last.chars().take(keep).collect::<String>())
            };
        }
    }

    let text_color = [0.9, 0.92, 0.96, 1.0];
    for (line_index, line) in lines.iter().enumerate() {
        let y = SIDEBAR_MARGIN + line_index as f32 * SIDEBAR_LINE_HEIGHT;
        for (char_index, ch) in line.chars().enumerate() {
            let Some(glyph) = font8x8::BASIC_FONTS.get(ch) else {
                continue;
            };
            let x = SIDEBAR_MARGIN + char_index as f32 * glyph_px;
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits >> col) & 1 == 0 {
                        continue;
                    }
                    let px0 = x + col as f32 * SIDEBAR_GLYPH_SCALE;
                    let py0 = y + row as f32 * SIDEBAR_GLYPH_SCALE;
                    push_quad(
                        &mut verts,
                        width,
                        height,
                        px0,
                        py0,
                        px0 + SIDEBAR_GLYPH_SCALE,
                        py0 + SIDEBAR_GLYPH_SCALE,
                        text_color,
                    );
                }
            }
        }
    }

    verts
}

// --- GPU state ---

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    point_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    minimap_pipeline: wgpu::RenderPipeline,
    point_buffer: wgpu::Buffer,
    line_buffer: wgpu::Buffer,
    line_vertex_count: u32,
    sidebar_buffer: wgpu::Buffer,
    sidebar_vertex_count: u32,
    minimap_buffer: wgpu::Buffer,
    minimap_vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
}

/// Loaded data ready for rendering.
struct LoadedData {
    vertices: Vec<PointVertex>,
    inspect_points: Vec<InspectPoint>,
    info_lines: Vec<String>,
    block_ranges: Vec<(u32, u32)>,
    minimap_rows: Vec<MinimapRow>,
    num_blocks: usize,
}

pub struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    camera: OrbitalCamera,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    point_size: f32,
    file_path: Option<PathBuf>,
    source: Arc<dyn PointCloudSource>,
    source_factory:
        Arc<dyn Fn(&std::path::Path, ViewMode) -> Arc<dyn PointCloudSource> + Send + Sync>,
    pending_load: Option<mpsc::Receiver<(String, LoadResult)>>,
    loaded: Option<LoadedData>,
    loading: bool,
    file_data: Option<Arc<Mmap>>,
    file_name: String,
    sel_start: f32,
    sel_end: f32,
    minimap_dragging: bool,
    minimap_resizing: bool,
    drag_offset: f32,
    resize_anchor: f32,
    cmd_held: bool,
    view_mode: ViewMode,
    inspect_text: Option<String>,
}

impl App {
    pub fn new(
        source: Arc<dyn PointCloudSource>,
        source_factory: Arc<
            dyn Fn(&std::path::Path, ViewMode) -> Arc<dyn PointCloudSource> + Send + Sync,
        >,
        view_mode: ViewMode,
        file_path: Option<PathBuf>,
    ) -> Self {
        Self {
            window: None,
            gpu: None,
            camera: OrbitalCamera::new(),
            mouse_pressed: false,
            last_mouse: None,
            point_size: 3.0,
            file_path,
            source,
            source_factory,
            pending_load: None,
            loaded: None,
            loading: false,
            file_data: None,
            file_name: String::new(),
            sel_start: 0.0,
            sel_end: 1.0,
            minimap_dragging: false,
            minimap_resizing: false,
            drag_offset: 0.0,
            resize_anchor: 0.0,
            cmd_held: false,
            view_mode,
            inspect_text: None,
        }
    }

    pub fn run(self) {
        let event_loop = EventLoop::new().unwrap();
        let mut app = self;
        event_loop.run_app(&mut app).unwrap();
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn is_in_minimap(&self, x: f64) -> bool {
        if let Some(gpu) = &self.gpu {
            x < gpu.config.width as f64 * SIDEBAR_WIDTH_FRAC as f64
        } else {
            false
        }
    }

    fn pixel_y_to_file_pos(&self, y: f64) -> f32 {
        if let Some(gpu) = &self.gpu {
            let minimap_top = gpu.config.height as f32 * (1.0 - MINIMAP_HEIGHT_FRAC);
            ((y as f32 - minimap_top) / (gpu.config.height as f32 * MINIMAP_HEIGHT_FRAC))
                .clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn load_file(&mut self, path: &std::path::Path) {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open {}: {e}", path.display());
                return;
            }
        };
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to mmap {}: {e}", path.display());
                return;
            }
        };

        self.file_path = Some(path.to_owned());
        self.file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let mmap = Arc::new(mmap);
        self.file_data = Some(mmap.clone());
        self.sel_start = 0.0;
        self.sel_end = 1.0;
        self.loaded = None;
        self.inspect_text = None;
        self.update_sidebar();

        self.view_mode = ViewMode::Binary;
        self.source = (self.source_factory)(path, self.view_mode);

        let name = self.file_name.clone();
        let data = mmap;
        let (tx, rx) = mpsc::channel();
        self.pending_load = Some(rx);
        self.loading = true;

        if let Some(window) = &self.window {
            window.set_title(&format!("Point Cloud Viewer - loading {name}..."));
        }

        let source = self.source.clone();
        let file_path_clone = path.to_owned();
        std::thread::spawn(move || {
            let start_time = std::time::Instant::now();
            let result = source.load(&file_path_clone, &data);
            let elapsed = start_time.elapsed();
            let size_mb = data.len() as f64 / (1024.0 * 1024.0);
            eprintln!(
                "Loaded {name} ({size_mb:.1} MB) in {:.2}s -> {} vertices",
                elapsed.as_secs_f64(),
                result.vertices.len(),
            );
            let _ = tx.send((name, result));
        });
    }

    fn update_minimap(&mut self) {
        if let (Some(loaded), Some(gpu)) = (&self.loaded, &mut self.gpu) {
            let verts = build_minimap_vertices(&loaded.minimap_rows, self.sel_start, self.sel_end);
            gpu.minimap_vertex_count = verts.len() as u32;
            gpu.minimap_buffer = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("minimap"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        }
    }

    fn sidebar_text(&self) -> String {
        let mut lines = vec![
            format!("VIEW  {}", self.view_mode.label()),
            format!("FILE  {}", self.file_name),
        ];

        if self.sel_start > 0.0 || self.sel_end < 0.999 {
            let pct_start = (self.sel_start * 100.0) as u32;
            let pct_end = (self.sel_end * 100.0) as u32;
            lines.push(format!("SLICE {}%-{}%", pct_start, pct_end));
        }

        lines.push(String::new());

        if let Some(selected) = &self.inspect_text {
            lines.push(String::from("SELECTED"));
            lines.extend(selected.lines().map(str::to_owned));
            lines.push(String::new());
        }

        if let Some(loaded) = &self.loaded {
            lines.push(String::from("SUMMARY"));
            lines.extend(loaded.info_lines.iter().cloned());
        } else if self.loading {
            lines.push(String::from("LOADING"));
            lines.push(String::from("Building point cloud..."));
        } else {
            lines.push(String::from("DROP A FILE"));
            lines.push(String::from("Open a binary or .hprof file."));
        }

        lines.push(String::new());
        lines.push(String::from("CONTROLS"));
        lines.push(String::from("drag orbit"));
        lines.push(String::from("wheel zoom"));
        lines.push(String::from("right-click inspect"));
        lines.push(String::from("v cycle views"));
        lines.push(String::from("r reset slice"));
        lines.join("\n")
    }

    fn update_sidebar(&mut self) {
        let text = self.sidebar_text();
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let verts =
            build_sidebar_vertices(&text, gpu.config.width as f32, gpu.config.height as f32);
        gpu.sidebar_vertex_count = verts.len() as u32;
        gpu.sidebar_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sidebar"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
    }

    fn selection_instance_range(&self) -> (u32, u32) {
        if let Some(loaded) = &self.loaded {
            let num_blocks = loaded.num_blocks;
            let start_block = (self.sel_start * num_blocks as f32) as usize;
            let end_block = ((self.sel_end * num_blocks as f32).ceil() as usize)
                .min(num_blocks)
                .min(loaded.block_ranges.len());

            if start_block >= end_block {
                return (0, 0);
            }

            let first_instance = loaded.block_ranges[start_block].0;
            let last = &loaded.block_ranges[end_block - 1];
            let end_instance = last.0 + last.1;
            (first_instance, end_instance - first_instance)
        } else {
            (0, 0)
        }
    }

    fn update_title(&self) {
        if let Some(window) = &self.window {
            if self.sel_start == 0.0 && self.sel_end >= 0.999 {
                window.set_title(&format!(
                    "Point Cloud Viewer - {} [{}]",
                    self.file_name,
                    self.view_mode.label(),
                ));
            } else {
                let pct_start = (self.sel_start * 100.0) as u32;
                let pct_end = (self.sel_end * 100.0) as u32;
                window.set_title(&format!(
                    "Point Cloud Viewer - {} [{}] [{pct_start}%-{pct_end}%]",
                    self.file_name,
                    self.view_mode.label(),
                ));
            }
        }
    }

    fn print_loaded_summary(&self) {
        let Some(loaded) = &self.loaded else {
            return;
        };
        eprintln!();
        eprintln!("=== {} [{}] ===", self.file_name, self.view_mode.label());
        for line in &loaded.info_lines {
            eprintln!("{line}");
        }
        eprintln!();
        eprintln!(
            "Controls: drag=orbit  wheel=zoom  right-click=inspect  v=cycle views  r=reset slice"
        );
        eprintln!("Sidebar: drag to scrub visible blocks, cmd+drag to resize slice");
        eprintln!();
    }

    fn print_selection(&self, text: &str) {
        let pct_start = (self.sel_start * 100.0) as u32;
        let pct_end = (self.sel_end * 100.0) as u32;
        eprintln!();
        eprintln!(
            "=== Selected [{}] [{}%-{}%] ===",
            self.view_mode.label(),
            pct_start,
            pct_end
        );
        eprintln!("{text}");
        eprintln!();
    }

    fn check_pending_load(&mut self) {
        if let Some(rx) = &self.pending_load {
            if let Ok((name, result)) = rx.try_recv() {
                let num_blocks = self.source.num_blocks();
                if let Some(gpu) = &mut self.gpu {
                    if !result.vertices.is_empty() {
                        gpu.point_buffer =
                            gpu.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("points"),
                                    contents: bytemuck::cast_slice(&result.vertices),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });
                    }
                }
                self.loaded = Some(LoadedData {
                    vertices: result.vertices,
                    inspect_points: result.inspect_points,
                    info_lines: result.info_lines,
                    block_ranges: result.block_ranges,
                    minimap_rows: result.minimap_rows,
                    num_blocks,
                });
                self.pending_load = None;
                self.loading = false;
                self.file_name = name;
                self.update_minimap();
                self.update_title();
                self.update_sidebar();
                self.print_loaded_summary();
            }
        }
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_buffer_size: 512 * 1024 * 1024,
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                point_size: 3.0,
                screen_width: 1280.0,
                screen_height: 960.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniforms_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let minimap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("minimap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_minimap"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MinimapVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_minimap"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"),
            contents: &[0u8; std::mem::size_of::<PointVertex>()],
            usage: wgpu::BufferUsages::VERTEX,
        });
        let line_verts = cube_wireframe_vertices(Vec3::splat(-1.0), Vec3::splat(1.0));
        let line_vertex_count = line_verts.len() as u32;
        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lines"),
            contents: bytemuck::cast_slice(&line_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let minimap_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("minimap"),
            contents: &[0u8; std::mem::size_of::<MinimapVertex>()],
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sidebar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sidebar"),
            contents: &[0u8; std::mem::size_of::<MinimapVertex>()],
            usage: wgpu::BufferUsages::VERTEX,
        });

        let depth_texture_view = Self::create_depth_texture(&device, config.width, config.height);

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            point_pipeline,
            line_pipeline,
            minimap_pipeline,
            point_buffer,
            line_buffer,
            line_vertex_count,
            sidebar_buffer,
            sidebar_vertex_count: 0,
            minimap_buffer,
            minimap_vertex_count: 0,
            uniform_buffer,
            bind_group,
            depth_texture_view,
        });

        if let Some(path) = self.file_path.clone() {
            self.load_file(&path);
        }
        self.update_sidebar();
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let gpu = self.gpu.as_mut().unwrap();
        if new_size.width > 0 && new_size.height > 0 {
            gpu.config.width = new_size.width;
            gpu.config.height = new_size.height;
            gpu.surface.configure(&gpu.device, &gpu.config);
            gpu.depth_texture_view =
                Self::create_depth_texture(&gpu.device, new_size.width, new_size.height);
        }
        self.update_sidebar();
    }

    fn render(&mut self) {
        let (inst_start, inst_count) = self.selection_instance_range();
        let gpu = self.gpu.as_ref().unwrap();
        let width = gpu.config.width as f32;
        let height = gpu.config.height as f32;

        let viewport_x = width * SIDEBAR_WIDTH_FRAC;
        let viewport_w = width - viewport_x;
        let aspect = viewport_w / height;

        let view_proj = self.camera.projection_matrix(aspect) * self.camera.view_matrix();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            point_size: self.point_size,
            screen_width: viewport_w,
            screen_height: height,
            _pad: 0.0,
        };
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = PhysicalSize::new(gpu.config.width, gpu.config.height);
                self.resize(size);
                return;
            }
            Err(e) => {
                eprintln!("Surface error: {e}");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let gpu = self.gpu.as_ref().unwrap();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_viewport(viewport_x, 0.0, viewport_w, height, 0.0, 1.0);

            if inst_count > 0 {
                pass.set_pipeline(&gpu.point_pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.point_buffer.slice(..));
                pass.draw(0..6, inst_start..inst_start + inst_count);
            }

            pass.set_pipeline(&gpu.line_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.line_buffer.slice(..));
            pass.draw(0..gpu.line_vertex_count, 0..1);
        }

        if gpu.minimap_vertex_count > 0 {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("minimap_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            let minimap_top = height * (1.0 - MINIMAP_HEIGHT_FRAC);
            pass.set_viewport(0.0, minimap_top, viewport_x, height - minimap_top, 0.0, 1.0);
            pass.set_pipeline(&gpu.minimap_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.minimap_buffer.slice(..));
            pass.draw(0..gpu.minimap_vertex_count, 0..1);
        }

        if gpu.sidebar_vertex_count > 0 {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sidebar_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&gpu.minimap_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.sidebar_buffer.slice(..));
            pass.draw(0..gpu.sidebar_vertex_count, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    fn current_view_proj(&self) -> Option<(Mat4, f32, f32, f32)> {
        let gpu = self.gpu.as_ref()?;
        let width = gpu.config.width as f32;
        let height = gpu.config.height as f32;
        let viewport_x = width * SIDEBAR_WIDTH_FRAC;
        let viewport_w = width - viewport_x;
        let aspect = viewport_w / height.max(1.0);
        let view_proj = self.camera.projection_matrix(aspect) * self.camera.view_matrix();
        Some((view_proj, viewport_x, viewport_w, height))
    }

    fn inspect_at_cursor(&mut self, cursor_x: f64, cursor_y: f64) {
        let Some(loaded) = &self.loaded else {
            return;
        };
        if loaded.inspect_points.len() != loaded.vertices.len() || loaded.inspect_points.is_empty()
        {
            return;
        }
        let Some((view_proj, viewport_x, viewport_w, height)) = self.current_view_proj() else {
            return;
        };
        if cursor_x < viewport_x as f64 {
            return;
        }

        let (inst_start, inst_count) = self.selection_instance_range();
        let start = inst_start as usize;
        let end = (inst_start + inst_count) as usize;
        if start >= end || end > loaded.vertices.len() {
            return;
        }

        let threshold_sq = (self.point_size * 4.0).max(10.0).powi(2);
        let mut best: Option<(usize, f32, f32)> = None;

        for index in start..end {
            let pos = loaded.vertices[index].position;
            let clip = view_proj * Vec4::new(pos[0], pos[1], pos[2], 1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            if ndc.z < -1.0 || ndc.z > 1.0 {
                continue;
            }
            let screen_x = viewport_x + (ndc.x * 0.5 + 0.5) * viewport_w;
            let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * height;
            let dx = screen_x - cursor_x as f32;
            let dy = screen_y - cursor_y as f32;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > threshold_sq {
                continue;
            }
            match best {
                Some((_, best_dist, best_depth)) if dist_sq > best_dist && ndc.z >= best_depth => {}
                _ => best = Some((index, dist_sq, ndc.z)),
            }
        }

        if let Some((index, _, _)) = best {
            let text = loaded.inspect_points[index].label.to_string();
            self.inspect_text = Some(text.clone());
            self.update_title();
            self.update_sidebar();
            self.print_selection(&text);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Point Cloud Viewer — drop a file to visualize")
                .with_inner_size(PhysicalSize::new(1280u32, 960));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());
            pollster::block_on(self.init_gpu(window));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.resize(size);
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.check_pending_load();
                self.render();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                self.load_file(&path);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                if pressed {
                    if let Some((mx, my)) = self.last_mouse {
                        if self.is_in_minimap(mx) && self.loaded.is_some() {
                            let pos = self.pixel_y_to_file_pos(my);
                            if self.cmd_held {
                                self.minimap_resizing = true;
                                let dist_to_start = (pos - self.sel_start).abs();
                                let dist_to_end = (pos - self.sel_end).abs();
                                self.resize_anchor = if dist_to_start > dist_to_end {
                                    self.sel_start
                                } else {
                                    self.sel_end
                                };
                            } else {
                                let sel_center = (self.sel_start + self.sel_end) / 2.0;
                                self.drag_offset = sel_center - pos;
                                self.minimap_dragging = true;
                            }
                        } else {
                            self.mouse_pressed = true;
                        }
                    } else {
                        self.mouse_pressed = true;
                    }
                } else {
                    self.minimap_dragging = false;
                    self.minimap_resizing = false;
                    self.mouse_pressed = false;
                    self.last_mouse = None;
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                if let Some((mx, my)) = self.last_mouse {
                    self.inspect_at_cursor(mx, my);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.cmd_held = modifiers.state().super_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = self.pixel_y_to_file_pos(position.y);

                if self.minimap_dragging {
                    let range = self.sel_end - self.sel_start;
                    let center = pos + self.drag_offset;
                    let new_start = (center - range / 2.0).clamp(0.0, 1.0 - range);
                    self.sel_start = new_start;
                    self.sel_end = (new_start + range).min(1.0);
                    self.update_minimap();
                    self.update_title();
                    self.update_sidebar();
                } else if self.minimap_resizing {
                    self.sel_start = self.resize_anchor.min(pos);
                    self.sel_end = self.resize_anchor.max(pos).max(self.sel_start + 0.005);
                    self.update_minimap();
                    self.update_title();
                    self.update_sidebar();
                } else if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.camera.yaw += dx * 0.005;
                        self.camera.pitch += dy * 0.005;
                        self.camera.pitch = self.camera.pitch.clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.01,
                            std::f32::consts::FRAC_PI_2 - 0.01,
                        );
                    }
                }
                self.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let in_minimap = self
                    .last_mouse
                    .map(|(x, _)| self.is_in_minimap(x))
                    .unwrap_or(false);

                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };

                if in_minimap && self.loaded.is_some() {
                    let shift = -scroll * 0.02;
                    let range = self.sel_end - self.sel_start;
                    let new_start = (self.sel_start + shift).clamp(0.0, 1.0 - range);
                    self.sel_start = new_start;
                    self.sel_end = new_start + range;
                    self.update_minimap();
                    self.update_title();
                    self.update_sidebar();
                } else {
                    self.camera.distance = (self.camera.distance - scroll * 0.3).clamp(0.5, 20.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    use winit::keyboard::{Key, NamedKey};
                    match event.logical_key {
                        Key::Character(ref c) if c.as_str() == "=" || c.as_str() == "+" => {
                            self.point_size = (self.point_size + 0.5).min(20.0);
                        }
                        Key::Character(ref c) if c.as_str() == "-" => {
                            self.point_size = (self.point_size - 0.5).max(1.0);
                        }
                        Key::Character(ref c) if c.as_str() == "r" => {
                            self.sel_start = 0.0;
                            self.sel_end = 1.0;
                            self.update_minimap();
                            self.update_title();
                            self.update_sidebar();
                        }
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}
