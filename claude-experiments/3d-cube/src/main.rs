use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use memmap2::Mmap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

// --- Constants ---

const MINIMAP_WIDTH_FRAC: f32 = 0.08;
const MINIMAP_ROWS: usize = 512;
const NUM_BLOCKS: usize = 256;

// --- Data types ---

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PointVertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MinimapVertex {
    position: [f32; 2],
    color: [f32; 4],
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

// --- Camera ---

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

// --- Cube wireframe ---

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
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    edges
        .iter()
        .flat_map(|&(a, b)| {
            [
                LineVertex { position: corners[a].into() },
                LineVertex { position: corners[b].into() },
            ]
        })
        .collect()
}

// --- Binary visualization (trigram method) ---

const TRIGRAM_COUNT: usize = 256 * 256 * 256;

#[inline(always)]
fn trigram_index(b0: u8, b1: u8, b2: u8) -> usize {
    (b0 as usize) << 16 | (b1 as usize) << 8 | (b2 as usize)
}

/// Sparse trigram data for one block: Vec of (trigram_index, count, position_sum)
struct BlockTrigrams {
    entries: Vec<(u32, u32, f32)>, // (idx, count, pos_sum)
}

/// Precomputed block data for instant scrubbing.
/// Uses incremental add/subtract so moving selection by 1 block
/// only touches ~500K entries instead of ~50M.
struct PrecomputedBlocks {
    blocks: Vec<BlockTrigrams>,
    count: Vec<u32>,
    pos_sum: Vec<f32>,
    dirty: Vec<u32>,
    points: Vec<PointVertex>,
    positions: Vec<[f32; 3]>,
    // Track current block range for incremental updates
    cur_start: usize,
    cur_end: usize,
}

impl PrecomputedBlocks {
    fn new(blocks: Vec<BlockTrigrams>) -> Self {
        let mut positions = Vec::with_capacity(TRIGRAM_COUNT);
        for idx in 0..TRIGRAM_COUNT {
            let b0 = (idx >> 16) as u16;
            let b1 = ((idx >> 8) & 0xFF) as u16;
            let b2 = (idx & 0xFF) as u16;
            positions.push([
                (b0 as f32 / 127.5) - 1.0,
                (b1 as f32 / 127.5) - 1.0,
                (b2 as f32 / 127.5) - 1.0,
            ]);
        }
        Self {
            blocks,
            count: vec![0u32; TRIGRAM_COUNT],
            pos_sum: vec![0f32; TRIGRAM_COUNT],
            dirty: Vec::with_capacity(TRIGRAM_COUNT / 2),
            points: Vec::with_capacity(TRIGRAM_COUNT / 2),
            positions,
            cur_start: 0,
            cur_end: 0,
        }
    }

    /// Add a block's entries to the accumulator.
    fn add_block(&mut self, block_idx: usize) {
        for &(idx, c, ps) in &self.blocks[block_idx].entries {
            let i = idx as usize;
            if self.count[i] == 0 {
                self.dirty.push(idx);
            }
            self.count[i] += c;
            self.pos_sum[i] += ps;
        }
    }

    /// Subtract a block's entries from the accumulator.
    fn sub_block(&mut self, block_idx: usize) {
        for &(idx, c, ps) in &self.blocks[block_idx].entries {
            let i = idx as usize;
            self.count[i] = self.count[i].saturating_sub(c);
            self.pos_sum[i] -= ps;
        }
    }

    /// Update to a new block range, using incremental add/subtract when possible.
    fn update_range(&mut self, new_start: usize, new_end: usize) -> usize {
        let new_end = new_end.min(self.blocks.len());
        let old_start = self.cur_start;
        let old_end = self.cur_end;

        if new_start == old_start && new_end == old_end {
            return self.points.len();
        }

        // Check if there's overlap we can exploit
        let has_overlap = old_end > 0 && new_start < old_end && new_end > old_start;

        if has_overlap {
            // Subtract blocks that left the window
            for b in old_start..new_start.min(old_end) {
                self.sub_block(b);
            }
            for b in new_end.max(old_start)..old_end {
                self.sub_block(b);
            }
            // Add blocks that entered the window
            for b in new_start..old_start.min(new_end) {
                self.add_block(b);
            }
            for b in old_end.max(new_start)..new_end {
                self.add_block(b);
            }
        } else {
            // No overlap: full clear and recompute
            for &idx in &self.dirty {
                self.count[idx as usize] = 0;
                self.pos_sum[idx as usize] = 0.0;
            }
            self.dirty.clear();
            for b in new_start..new_end {
                self.add_block(b);
            }
        }

        self.cur_start = new_start;
        self.cur_end = new_end;

        self.rebuild_vertices()
    }

    /// Rebuild vertex buffer from dirty list.
    /// Skips rare trigrams to keep point count manageable for GPU.
    fn rebuild_vertices(&mut self) -> usize {
        let mut max_freq: u32 = 0;
        let count = &self.count;
        self.dirty.retain(|&idx| {
            let c = count[idx as usize];
            if c > max_freq {
                max_freq = c;
            }
            c > 0
        });

        if max_freq == 0 {
            self.points.clear();
            return 0;
        }

        // Find a count threshold that keeps us under MAX_POINTS.
        // Binary search for the right cutoff.
        const MAX_POINTS: usize = 500_000;
        let min_count = if self.dirty.len() > MAX_POINTS {
            // Build a histogram of log2(count) buckets to find cutoff fast
            let mut above = self.dirty.len();
            let mut threshold = 1u32;
            // Double threshold until we're under budget
            while above > MAX_POINTS && threshold < max_freq {
                threshold *= 2;
                above = 0;
                for &idx in &self.dirty {
                    if self.count[idx as usize] >= threshold {
                        above += 1;
                    }
                }
            }
            // Back off one step and refine
            let lo = threshold / 2;
            let mut best = lo;
            // Binary search between lo and threshold
            let (mut a, mut b) = (lo, threshold);
            while a < b {
                let mid = a + (b - a) / 2;
                let cnt = self.dirty.iter().filter(|&&idx| self.count[idx as usize] >= mid).count();
                if cnt > MAX_POINTS {
                    a = mid + 1;
                    best = a;
                } else {
                    b = mid;
                    best = mid;
                }
            }
            best
        } else {
            1
        };

        let ln_max = (max_freq as f32).ln();

        self.points.clear();
        for &idx in &self.dirty {
            let i = idx as usize;
            let c = self.count[i];
            if c < min_count {
                continue;
            }
            let avg_pos = self.pos_sum[i] / c as f32;
            let brightness = ((c as f32).ln() / ln_max).clamp(0.05, 1.0);
            let (r, g, b_color) = file_position_color(avg_pos);

            self.points.push(PointVertex {
                position: self.positions[i],
                color: [r * brightness, g * brightness, b_color * brightness, 1.0],
            });
        }
        self.points.len()
    }
}

/// Precompute trigram data for NUM_BLOCKS blocks, in parallel.
/// Returns sparse per-block data suitable for fast range queries.
fn precompute_blocks(data: &[u8]) -> Vec<BlockTrigrams> {
    if data.len() < 3 {
        return vec![];
    }

    let block_size = data.len() / NUM_BLOCKS;
    if block_size < 3 {
        return vec![];
    }

    let inv_file_len = 1.0 / data.len() as f32;

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(NUM_BLOCKS);
        for b in 0..NUM_BLOCKS {
            let start = b * block_size;
            let end = if b == NUM_BLOCKS - 1 {
                data.len()
            } else {
                // Overlap by 2 to capture trigrams at boundaries
                ((b + 1) * block_size + 2).min(data.len())
            };
            let chunk = &data[start..end];
            handles.push(s.spawn(move || {
                let mut count = vec![0u32; TRIGRAM_COUNT];
                let mut pos_sum = vec![0f32; TRIGRAM_COUNT];

                for i in 0..chunk.len() - 2 {
                    let idx = trigram_index(chunk[i], chunk[i + 1], chunk[i + 2]);
                    unsafe {
                        *count.get_unchecked_mut(idx) += 1;
                        *pos_sum.get_unchecked_mut(idx) += (start + i) as f32 * inv_file_len;
                    }
                }

                // Convert to sparse
                let mut entries = Vec::new();
                for idx in 0..TRIGRAM_COUNT {
                    let c = count[idx];
                    if c > 0 {
                        entries.push((idx as u32, c, pos_sum[idx]));
                    }
                }

                BlockTrigrams { entries }
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn file_position_color(pos: f32) -> (f32, f32, f32) {
    if pos < 0.5 {
        let t = pos * 2.0;
        (1.0, 1.0, t)
    } else {
        let t = (pos - 0.5) * 2.0;
        (1.0 - t, 1.0 - t, 1.0)
    }
}

// --- Minimap ---

fn build_minimap_vertices(data: &[u8], sel_start: f32, sel_end: f32) -> Vec<MinimapVertex> {
    if data.is_empty() {
        return vec![];
    }

    let rows = MINIMAP_ROWS;
    let chunk_size = data.len() / rows;
    if chunk_size == 0 {
        return vec![];
    }

    let x_left = -1.0_f32;
    let x_right = -1.0 + 2.0 * MINIMAP_WIDTH_FRAC;
    let row_height = 2.0 / rows as f32;

    let mut verts = Vec::with_capacity(rows * 6 + 24);

    for row in 0..rows {
        let start = row * chunk_size;
        let end = if row == rows - 1 { data.len() } else { start + chunk_size };
        let slice = &data[start..end];

        let mut sum: u64 = 0;
        let mut entropy_counts = [0u32; 256];
        for &b in slice {
            sum += b as u64;
            entropy_counts[b as usize] += 1;
        }
        let avg = sum as f32 / slice.len() as f32;

        let len_f = slice.len() as f32;
        let mut entropy: f32 = 0.0;
        for &c in &entropy_counts {
            if c > 0 {
                let p = c as f32 / len_f;
                entropy -= p * p.log2();
            }
        }
        let norm_entropy = (entropy / 8.0).clamp(0.0, 1.0);

        let t = avg / 255.0;
        let r = (t * 2.0).clamp(0.0, 1.0);
        let g = (1.0 - (t - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let b = (1.0 - t).clamp(0.0, 1.0) * 0.8;
        let brightness = 0.3 + norm_entropy * 0.7;

        let y_top = 1.0 - row as f32 * row_height;
        let y_bot = y_top - row_height;
        let color = [r * brightness, g * brightness, b * brightness, 1.0];

        verts.push(MinimapVertex { position: [x_left, y_top], color });
        verts.push(MinimapVertex { position: [x_right, y_top], color });
        verts.push(MinimapVertex { position: [x_right, y_bot], color });
        verts.push(MinimapVertex { position: [x_left, y_top], color });
        verts.push(MinimapVertex { position: [x_right, y_bot], color });
        verts.push(MinimapVertex { position: [x_left, y_bot], color });
    }

    // Dim overlay OUTSIDE the selection (dark semi-transparent quads above and below)
    let sel_y_top = 1.0 - sel_start * 2.0;
    let sel_y_bot = 1.0 - sel_end * 2.0;
    let dim_color = [0.0_f32, 0.0, 0.0, 0.7];

    // Dim region above selection (top of minimap to selection start)
    if sel_start > 0.001 {
        verts.push(MinimapVertex { position: [x_left, 1.0], color: dim_color });
        verts.push(MinimapVertex { position: [x_right, 1.0], color: dim_color });
        verts.push(MinimapVertex { position: [x_right, sel_y_top], color: dim_color });
        verts.push(MinimapVertex { position: [x_left, 1.0], color: dim_color });
        verts.push(MinimapVertex { position: [x_right, sel_y_top], color: dim_color });
        verts.push(MinimapVertex { position: [x_left, sel_y_top], color: dim_color });
    }

    // Dim region below selection (selection end to bottom of minimap)
    if sel_end < 0.999 {
        verts.push(MinimapVertex { position: [x_left, sel_y_bot], color: dim_color });
        verts.push(MinimapVertex { position: [x_right, sel_y_bot], color: dim_color });
        verts.push(MinimapVertex { position: [x_right, -1.0], color: dim_color });
        verts.push(MinimapVertex { position: [x_left, sel_y_bot], color: dim_color });
        verts.push(MinimapVertex { position: [x_right, -1.0], color: dim_color });
        verts.push(MinimapVertex { position: [x_left, -1.0], color: dim_color });
    }

    // Selection border lines (bright edges at top and bottom of selection)
    let border_h = 0.003;
    let border_color = [1.0_f32, 1.0, 1.0, 0.9];
    for &y in &[sel_y_top, sel_y_bot] {
        verts.push(MinimapVertex { position: [x_left, y + border_h], color: border_color });
        verts.push(MinimapVertex { position: [x_right, y + border_h], color: border_color });
        verts.push(MinimapVertex { position: [x_right, y - border_h], color: border_color });
        verts.push(MinimapVertex { position: [x_left, y + border_h], color: border_color });
        verts.push(MinimapVertex { position: [x_right, y - border_h], color: border_color });
        verts.push(MinimapVertex { position: [x_left, y - border_h], color: border_color });
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
    point_count: u32,
    line_buffer: wgpu::Buffer,
    line_vertex_count: u32,
    minimap_buffer: wgpu::Buffer,
    minimap_vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
}

// --- App ---

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    camera: OrbitalCamera,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    point_size: f32,
    file_path: Option<PathBuf>,
    // Block precomputation (background)
    pending_blocks: Option<mpsc::Receiver<(String, PrecomputedBlocks)>>,
    blocks: Option<PrecomputedBlocks>,
    loading: bool,
    // File data
    file_data: Option<Arc<Mmap>>,
    file_name: String,
    // Selection [0, 1]
    sel_start: f32,
    sel_end: f32,
    // Minimap interaction
    minimap_dragging: bool,       // regular drag = move
    minimap_resizing: bool,       // cmd+drag = resize
    drag_offset: f32,             // offset from mouse to sel center when moving
    resize_anchor: f32,           // the fixed edge during resize
    cmd_held: bool,               // track Cmd key state
}

impl App {
    fn new(file_path: Option<PathBuf>) -> Self {
        Self {
            window: None,
            gpu: None,
            camera: OrbitalCamera::new(),
            mouse_pressed: false,
            last_mouse: None,
            point_size: 3.0,
            file_path,
            pending_blocks: None,
            blocks: None,
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
        }
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
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
            x < gpu.config.width as f64 * MINIMAP_WIDTH_FRAC as f64
        } else {
            false
        }
    }

    fn pixel_y_to_file_pos(&self, y: f64) -> f32 {
        if let Some(gpu) = &self.gpu {
            (y as f32 / gpu.config.height as f32).clamp(0.0, 1.0)
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

        self.file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let mmap = Arc::new(mmap);
        self.file_data = Some(mmap.clone());
        self.sel_start = 0.0;
        self.sel_end = 1.0;
        self.blocks = None;

        self.update_minimap();

        let name = self.file_name.clone();
        let data = mmap;
        let (tx, rx) = mpsc::channel();
        self.pending_blocks = Some(rx);
        self.loading = true;

        if let Some(window) = &self.window {
            window.set_title(&format!("Point Cloud Viewer - loading {name}..."));
        }

        std::thread::spawn(move || {
            let start_time = std::time::Instant::now();
            let block_data = precompute_blocks(&data);
            let elapsed = start_time.elapsed();
            let size_mb = data.len() as f64 / (1024.0 * 1024.0);
            eprintln!(
                "Precomputed {NUM_BLOCKS} blocks for {name} ({size_mb:.1} MB) in {:.2}s",
                elapsed.as_secs_f64(),
            );

            let precomputed = PrecomputedBlocks::new(block_data);
            let _ = tx.send((name, precomputed));
        });
    }

    fn update_minimap(&mut self) {
        if let (Some(data), Some(gpu)) = (&self.file_data, &mut self.gpu) {
            let verts = build_minimap_vertices(data, self.sel_start, self.sel_end);
            gpu.minimap_vertex_count = verts.len() as u32;
            gpu.minimap_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("minimap"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        }
    }

    /// Instantly recompute trigrams from precomputed blocks for the current selection.
    fn recompute_from_blocks(&mut self) {
        if let Some(blocks) = &mut self.blocks {
            let start_block = (self.sel_start * NUM_BLOCKS as f32) as usize;
            let end_block = ((self.sel_end * NUM_BLOCKS as f32).ceil() as usize).min(NUM_BLOCKS);

            let point_count = blocks.update_range(start_block, end_block);

            if let Some(gpu) = &mut self.gpu {
                gpu.point_count = point_count as u32;
                if point_count > 0 {
                    let byte_size = point_count * std::mem::size_of::<PointVertex>();
                    // Recreate buffer only if current one is too small
                    if gpu.point_buffer.size() < byte_size as u64 {
                        gpu.point_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("points"),
                            // Allocate 2x to reduce future reallocations
                            size: (byte_size * 2) as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                    }
                    gpu.queue.write_buffer(
                        &gpu.point_buffer,
                        0,
                        bytemuck::cast_slice(&blocks.points),
                    );
                }
            }
        }

        self.update_title();
    }

    fn update_title(&self) {
        if let Some(window) = &self.window {
            if self.sel_start == 0.0 && self.sel_end >= 0.999 {
                window.set_title(&format!("Point Cloud Viewer - {}", self.file_name));
            } else {
                let pct_start = (self.sel_start * 100.0) as u32;
                let pct_end = (self.sel_end * 100.0) as u32;
                window.set_title(&format!(
                    "Point Cloud Viewer - {} [{pct_start}%-{pct_end}%]",
                    self.file_name
                ));
            }
        }
    }

    fn check_pending_blocks(&mut self) {
        if let Some(rx) = &self.pending_blocks {
            if let Ok((name, blocks)) = rx.try_recv() {
                self.blocks = Some(blocks);
                self.pending_blocks = None;
                self.loading = false;
                self.file_name = name;
                // Initial full-file combine
                self.recompute_from_blocks();
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
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
            point_count: 0,
            line_buffer,
            line_vertex_count,
            minimap_buffer,
            minimap_vertex_count: 0,
            uniform_buffer,
            bind_group,
            depth_texture_view,
        });

        if let Some(path) = self.file_path.take() {
            self.load_file(&path);
        }
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
    }

    fn render(&mut self) {
        let gpu = self.gpu.as_ref().unwrap();
        let width = gpu.config.width as f32;
        let height = gpu.config.height as f32;

        let viewport_x = width * MINIMAP_WIDTH_FRAC;
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
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let gpu = self.gpu.as_ref().unwrap();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02, g: 0.02, b: 0.03, a: 1.0,
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

            if gpu.point_count > 0 {
                pass.set_pipeline(&gpu.point_pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.point_buffer.slice(..));
                pass.draw(0..6, 0..gpu.point_count);
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

            pass.set_pipeline(&gpu.minimap_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.minimap_buffer.slice(..));
            pass.draw(0..gpu.minimap_vertex_count, 0..1);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
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
                self.check_pending_blocks();
                self.render();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                self.load_file(&path);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                let pressed = state == ElementState::Pressed;
                if pressed {
                    if let Some((mx, my)) = self.last_mouse {
                        if self.is_in_minimap(mx) && self.blocks.is_some() {
                            let pos = self.pixel_y_to_file_pos(my);

                            if self.cmd_held {
                                // Cmd+click = resize: anchor the far edge
                                self.minimap_resizing = true;
                                let dist_to_start = (pos - self.sel_start).abs();
                                let dist_to_end = (pos - self.sel_end).abs();
                                self.resize_anchor = if dist_to_start > dist_to_end {
                                    self.sel_start
                                } else {
                                    self.sel_end
                                };
                            } else {
                                // Regular click = move selection
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
            WindowEvent::ModifiersChanged(modifiers) => {
                self.cmd_held = modifiers.state().super_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = self.pixel_y_to_file_pos(position.y);

                if self.minimap_dragging {
                    // Move selection, keeping size constant
                    let range = self.sel_end - self.sel_start;
                    let center = pos + self.drag_offset;
                    let new_start = (center - range / 2.0).clamp(0.0, 1.0 - range);
                    self.sel_start = new_start;
                    self.sel_end = (new_start + range).min(1.0);
                    self.update_minimap();
                    self.recompute_from_blocks();
                } else if self.minimap_resizing {
                    // Resize: anchor stays, other edge follows mouse
                    self.sel_start = self.resize_anchor.min(pos);
                    self.sel_end = self.resize_anchor.max(pos).max(self.sel_start + 0.005);
                    self.update_minimap();
                    self.recompute_from_blocks();
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

                if in_minimap && self.blocks.is_some() {
                    let shift = -scroll * 0.02;
                    let range = self.sel_end - self.sel_start;
                    let new_start = (self.sel_start + shift).clamp(0.0, 1.0 - range);
                    self.sel_start = new_start;
                    self.sel_end = new_start + range;
                    self.update_minimap();
                    self.recompute_from_blocks();
                } else {
                    self.camera.distance =
                        (self.camera.distance - scroll * 0.3).clamp(0.5, 20.0);
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
                            self.recompute_from_blocks();
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

fn main() {
    let file_path = std::env::args().nth(1).map(PathBuf::from);
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(file_path);
    event_loop.run_app(&mut app).unwrap();
}
