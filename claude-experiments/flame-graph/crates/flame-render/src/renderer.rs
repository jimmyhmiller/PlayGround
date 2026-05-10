use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use flame_core::{Profile, StringId, TrackId};
use glyphon::{
    Attrs, Buffer, Cache as GlyphonCache, Color as GlyphonColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport as GlyphonViewport, Wrap,
};
use wgpu::util::DeviceExt;

use crate::instance::{SliceInstance, Uniforms};
use crate::palette;
use crate::viewport::Viewport;

const SLICE_LABEL_MIN_PX: f32 = 76.0;
const STATUS_BAR_HEIGHT_PX: f32 = 60.0;
pub const ROW_HEIGHT_PX: f32 = 52.0;
const LABEL_FONT_SIZE: f32 = 30.0;
const LABEL_LINE_HEIGHT: f32 = 38.0;
const TRACK_HEADER_HEIGHT_PX: f32 = 36.0;
const TRACK_GAP_PX: f32 = 8.0;
const HEADER_FONT_SIZE: f32 = 22.0;
const HEADER_LINE_HEIGHT: f32 = 28.0;
/// Sentinel stored in `slice_indices` for instances that aren't slices (track
/// headers). Hit-testing skips them.
const NON_SLICE_SENTINEL: u32 = u32::MAX;
// ────────── Theme ──────────
// Palette inspired by VS Code Dark+ and Firefox profiler dark; warm orange
// accent that picks up the flame-graph hue family.

/// Track-header band color.
const HEADER_COLOR: [f32; 4] = [0.16, 0.18, 0.22, 1.0];
/// Inspector & full-tab content row stripe (even).
const ROW_COLOR_EVEN: [f32; 4] = [0.13, 0.14, 0.17, 1.0];
/// Inspector & full-tab content row stripe (odd).
const ROW_COLOR_ODD: [f32; 4] = [0.16, 0.17, 0.20, 1.0];
/// Top tab bar background — slightly lower than the main canvas so the bar
/// reads as a separate strip.
const TOP_TAB_INACTIVE: [f32; 4] = [0.15, 0.16, 0.20, 1.0];
const TOP_TAB_ACTIVE: [f32; 4] = [0.27, 0.30, 0.36, 1.0];
const TOP_TAB_ACCENT: [f32; 4] = [0.96, 0.66, 0.28, 1.0]; // active tab underline
const SANDWICH_DIVIDER: [f32; 4] = [0.10, 0.11, 0.13, 1.0];

/// Vertical layout direction.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Root at top, depth grows downward (Speedscope/Perfetto/Chrome dev tools).
    Icicle,
    /// Root at bottom, depth grows upward (classic Brendan Gregg flame graph).
    Flame,
}

impl Direction {
    pub fn flipped(self) -> Self {
        match self {
            Direction::Icicle => Direction::Flame,
            Direction::Flame => Direction::Icicle,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Direction::Icicle => "icicle",
            Direction::Flame => "flame",
        }
    }
}

/// Per-track vertical layout. Computed each `rebuild_instances` from the
/// loaded profile. Coordinates are canvas-space (before scroll).
#[derive(Clone, Debug)]
pub struct TrackLayout {
    pub track: TrackId,
    pub y_top: f32,
    pub header_h: f32,
    pub rows: u16,
    pub content_h: f32,
}

/// Aggregated stats per function name across the entire profile.
#[derive(Clone, Debug)]
pub struct HotspotEntry {
    pub name: StringId,
    pub total_dur_ns: u64,
    pub self_dur_ns: u64,
    pub count: u32,
    /// Slice index of the largest single instance — used for jump-to-instance.
    pub exemplar_slice_idx: u32,
}

/// Aggregated callers/callees of one function (used in the Sandwich view).
#[derive(Clone, Debug)]
pub struct CallerCalleeAggregate {
    /// (function_name, total_time_along_this_edge, count_of_edges, exemplar_slice_idx)
    pub callers: Vec<(StringId, u64, u32, u32)>,
    pub callees: Vec<(StringId, u64, u32, u32)>,
}

/// One node in the top-down call tree. Same-named sibling slices are merged at
/// every level, across all tracks, so a single node represents every place a
/// function was called from a given parent.
#[derive(Clone, Debug)]
pub struct CallTreeNode {
    pub name: StringId,
    /// Sum of `dur_ns` across all slices that contributed to this node.
    pub total_ns: u64,
    /// Sum of self-time (`dur - children_dur`) across contributing slices.
    pub self_ns: u64,
    /// Number of slices merged into this node.
    pub count: u32,
    /// Tree depth (root = 0).
    pub depth: u16,
    /// Indices into `CallTree::nodes`.
    pub children: Vec<u32>,
    /// Index into `CallTree::nodes`, or `u32::MAX` for roots.
    pub parent: u32,
    /// Largest contributing slice index — used for "jump to flame" actions.
    pub exemplar_slice_idx: u32,
}

#[derive(Default, Clone, Debug)]
pub struct CallTree {
    pub nodes: Vec<CallTreeNode>,
    pub roots: Vec<u32>,
    /// Sum of root totals — denominator for percentages.
    pub total_ns: u64,
}

/// Which top-level view is currently active.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MainTab {
    /// The timeline / flame graph view (default). Sandwich/callers/callees of the
    /// selected slice live in this tab's right-side inspector panel.
    Flame,
    /// Top-down call tree, all calls aggregated by name at each level. Click
    /// a row to expand/collapse. Firefox-profiler "Call Tree" equivalent.
    CallTree,
    /// Top functions by **total** duration.
    Hotspots,
    /// Top functions by **self** duration (Firefox bottom-up call tree at depth 1).
    BottomUp,
    /// Full sortable function table.
    Table,
}

impl MainTab {
    pub const ALL: &'static [MainTab] = &[
        MainTab::Flame,
        MainTab::CallTree,
        MainTab::Hotspots,
        MainTab::BottomUp,
        MainTab::Table,
    ];
    pub fn label(self) -> &'static str {
        match self {
            MainTab::Flame => "FLAME",
            MainTab::CallTree => "CALL TREE",
            MainTab::Hotspots => "HOTSPOTS",
            MainTab::BottomUp => "BOTTOM-UP",
            MainTab::Table => "ALL FUNCTIONS",
        }
    }
}

/// How the FLAME view lays out slices on the time axis.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LayoutMode {
    /// x-axis is wall time. Each slice at its real `start_ns`. Default.
    TimeOrdered,
    /// x-axis is total time spent. Identical stacks collapsed into one wide bar,
    /// siblings sorted by duration desc. Brendan Gregg / Speedscope "Left Heavy".
    LeftHeavy,
}

impl LayoutMode {
    pub const ALL: &'static [LayoutMode] = &[LayoutMode::TimeOrdered, LayoutMode::LeftHeavy];
    pub fn label(self) -> &'static str {
        match self {
            LayoutMode::TimeOrdered => "TIME",
            LayoutMode::LeftHeavy => "AGGREGATED",
        }
    }
}

/// One inspector-side clickable hotspot row's pixel rect + the slice it
/// jumps to. Recomputed each `rebuild_instances`.
#[derive(Clone, Debug)]
pub struct InspectorHotspotRect {
    pub rect: [f32; 4], // x, y, w, h
    pub slice_idx: u32,
}

pub const INSPECTOR_WIDTH_PX: f32 = 640.0;
pub const MAX_HOTSPOTS: usize = 18;
const INSPECTOR_BG_COLOR: [f32; 4] = [0.115, 0.125, 0.150, 1.0];
const INSPECTOR_PADDING_PX: f32 = 24.0;
const INSPECTOR_LINE_HEIGHT_PX: f32 = 38.0;
const INSPECTOR_HEADING_FONT: f32 = 22.0;
const INSPECTOR_BODY_FONT: f32 = 26.0;
const INSPECTOR_TITLE_FONT: f32 = 38.0;
const INSPECTOR_TITLE_LINE_HEIGHT: f32 = 48.0;
const SECTION_GAP_PX: f32 = 24.0;
#[allow(dead_code)]
const TAB_HEIGHT_PX: f32 = 56.0;
/// Height of the global tab bar at the top of the window.
pub const TAB_BAR_HEIGHT_PX: f32 = 70.0;
/// One row in the CALL TREE view.
const CALL_TREE_ROW_H: f32 = 36.0;
/// Column-header band above the CALL TREE rows.
const CALL_TREE_HEADER_H: f32 = 44.0;
/// Pixels of indent per call-tree depth level.
const CALL_TREE_INDENT_PX: f32 = 18.0;
/// Color of the tab bar background.
const TAB_BAR_BG_COLOR: [f32; 4] = [0.095, 0.103, 0.122, 1.0];

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    instance_count: u32,
    /// Mirror of the GPU instance buffer; used for hover hit-testing.
    pub instances: Vec<SliceInstance>,
    /// `slice_indices[i]` is the SoA index of `instances[i]`. Lets hover map to slice.
    pub slice_indices: Vec<u32>,

    pub viewport: Viewport,
    pub profile: Option<Arc<Profile>>,
    pub hovered: Option<u32>,
    pub status_text: String,
    pub direction: Direction,
    pub track_layouts: Vec<TrackLayout>,
    pub canvas_height_px: f32,

    /// Currently selected slice (SoA index, persistent across rebuilds).
    pub selected_slice: Option<u32>,
    /// Per-slice self time = dur - sum(direct children dur). Computed once on
    /// profile load. Same length as `profile.slices`.
    pub self_dur_ns: Vec<u64>,
    /// All functions aggregated, sorted desc by total_dur_ns.
    pub functions_by_total: Vec<HotspotEntry>,
    /// All functions aggregated, sorted desc by self_dur_ns.
    pub functions_by_self: Vec<HotspotEntry>,
    /// Window width in px. The timeline occupies `[0, window_w - INSPECTOR_WIDTH_PX]`.
    pub window_w_px: f32,
    /// Inspector clickable rows (any tab) — populated each rebuild for click hit-test.
    pub inspector_hotspot_rects: Vec<InspectorHotspotRect>,
    /// Inspector tab buttons — populated each rebuild for click hit-test.
    pub inspector_tab_rects: Vec<(MainTab, [f32; 4])>,
    /// Currently active inspector tab.
    pub active_tab: MainTab,
    /// Track header rectangles for click-to-collapse hit-test (FLAME tab only).
    pub track_header_rects: Vec<(TrackId, [f32; 4])>,
    /// Tracks the user has collapsed via header click.
    pub collapsed_tracks: AHashSet<TrackId>,
    /// Layout mode for the FLAME view: time-ordered or left-heavy aggregated.
    pub layout_mode: LayoutMode,
    /// Floating layout-mode toggle button rects (bottom-left of timeline).
    pub layout_button_rects: Vec<(LayoutMode, [f32; 4])>,
    /// Pre-computed left-heavy slice table (one per profile load). Wrapped in
    /// `Arc` so the renderer can take a cheap reference into the GPU rebuild
    /// loop alongside `&mut self` for `self.instances.push(...)`.
    pub aggregated_slices: Option<Arc<flame_core::SliceTable>>,
    /// Time range of the aggregated layout (different scale than wall time).
    pub aggregated_time_range: (u64, u64),
    /// Per-track row counts for the aggregated layout.
    pub aggregated_track_rows: Vec<u16>,
    /// Cached sandwich aggregate for the currently selected slice's function name.
    pub sandwich_cache: Option<(StringId, CallerCalleeAggregate)>,
    /// Top-down call tree, built once on profile load. None until a profile loads.
    pub call_tree: Option<CallTree>,
    /// Set of expanded node indices in `call_tree.nodes`. Roots are always
    /// considered visible regardless of this set; this gates whether each
    /// node's `children` are walked.
    pub expanded_tree_nodes: AHashSet<u32>,
    /// Per-frame: clickable rect + node idx for each visible tree row.
    pub call_tree_row_rects: Vec<(u32, [f32; 4])>,
    /// CallTree-tab-specific vertical scroll offset (in px). Decoupled from the
    /// timeline's `viewport.scroll_y_px` so switching tabs doesn't lose either.
    pub call_tree_scroll_y_px: f32,

    // glyphon text state
    font_system: FontSystem,
    swash_cache: SwashCache,
    #[allow(dead_code)]
    glyphon_cache: GlyphonCache,
    glyphon_viewport: GlyphonViewport,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    /// Persistent cosmic-text buffers, one per text area we want to render. Reused
    /// across frames; surplus buffers are kept around.
    text_buffers: Vec<Buffer>,
}

impl Renderer {
    pub async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no compatible adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("flame-render device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        // ---- Slice pipeline ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slice.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/slice.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniforms bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                viewport_size_px: [size.width as f32, size.height as f32],
                hovered: u32::MAX,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slice pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("slice pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[SliceInstance::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let initial_capacity: u64 = 4096;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: initial_capacity * std::mem::size_of::<SliceInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- glyphon text ----
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyphon_cache = GlyphonCache::new(&device);
        let mut glyphon_viewport = GlyphonViewport::new(&device, &glyphon_cache);
        glyphon_viewport.update(
            &queue,
            Resolution {
                width: size.width,
                height: size.height,
            },
        );
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyphon_cache, format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        let timeline_w = (size.width as f32 - INSPECTOR_WIDTH_PX).max(1.0);
        let mut viewport = Viewport::new((timeline_w, size.height as f32));
        viewport.row_height_px = ROW_HEIGHT_PX;

        Self {
            device,
            queue,
            surface,
            surface_config,
            pipeline,
            uniform_buffer,
            bind_group,
            instance_buffer,
            instance_capacity: initial_capacity,
            instance_count: 0,
            instances: Vec::new(),
            slice_indices: Vec::new(),
            viewport,
            profile: None,
            hovered: None,
            status_text: String::from("Drop a trace file or pass one on the CLI"),
            direction: Direction::Icicle,
            track_layouts: Vec::new(),
            canvas_height_px: 0.0,
            selected_slice: None,
            self_dur_ns: Vec::new(),
            functions_by_total: Vec::new(),
            functions_by_self: Vec::new(),
            window_w_px: size.width as f32,
            inspector_hotspot_rects: Vec::new(),
            inspector_tab_rects: Vec::new(),
            active_tab: MainTab::Flame,
            sandwich_cache: None,
            call_tree: None,
            expanded_tree_nodes: AHashSet::new(),
            call_tree_row_rects: Vec::new(),
            call_tree_scroll_y_px: 0.0,
            track_header_rects: Vec::new(),
            collapsed_tracks: AHashSet::new(),
            layout_mode: LayoutMode::TimeOrdered,
            layout_button_rects: Vec::new(),
            aggregated_slices: None,
            aggregated_time_range: (0, 0),
            aggregated_track_rows: Vec::new(),
            font_system,
            swash_cache,
            glyphon_cache,
            glyphon_viewport,
            text_atlas,
            text_renderer,
            text_buffers: Vec::new(),
        }
    }

    pub fn set_profile(&mut self, profile: Arc<Profile>) {
        let (start_ns, end_ns) = profile.time_range;
        let started = std::time::Instant::now();
        self.self_dur_ns = compute_self_dur(&profile);
        let by_name = aggregate_functions(&profile, &self.self_dur_ns);
        self.functions_by_total = sort_functions_desc_by(&by_name, false);
        self.functions_by_self = sort_functions_desc_by(&by_name, true);
        let (agg_slices, agg_range, agg_rows) = build_left_heavy_layout(&profile);
        self.aggregated_slices = Some(Arc::new(agg_slices));
        self.aggregated_time_range = agg_range;
        self.aggregated_track_rows = agg_rows;

        let tree = build_call_tree(&profile, &self.self_dur_ns);
        // Default-expand all roots so the user sees something useful immediately.
        self.expanded_tree_nodes.clear();
        for &r in &tree.roots {
            self.expanded_tree_nodes.insert(r);
        }
        self.call_tree = Some(tree);
        self.call_tree_scroll_y_px = 0.0;
        log::debug!(
            "computed aggregates for {} slices in {:?}",
            profile.slices.len(),
            started.elapsed()
        );
        self.profile = Some(profile);
        self.hovered = None;
        self.selected_slice = None;
        self.sandwich_cache = None;
        self.collapsed_tracks.clear();
        self.layout_mode = LayoutMode::TimeOrdered;
        self.viewport.fit_time(start_ns, end_ns);
        self.viewport.row_height_px = ROW_HEIGHT_PX;
        self.viewport.scroll_y_px = 0.0;
        self.clamp_viewport();
        self.refresh_status_for_idle();
    }

    /// Active SliceTable based on `layout_mode`. Used by all rendering /
    /// hit-testing / stats paths.
    pub fn current_slices<'a>(&'a self, profile: &'a Profile) -> &'a flame_core::SliceTable {
        match (self.layout_mode, &self.aggregated_slices) {
            (LayoutMode::LeftHeavy, Some(s)) => s,
            _ => &profile.slices,
        }
    }

    /// Active time range for the current layout.
    pub fn current_time_range(&self) -> (u64, u64) {
        match self.layout_mode {
            LayoutMode::LeftHeavy => self.aggregated_time_range,
            LayoutMode::TimeOrdered => self.profile.as_ref().map(|p| p.time_range).unwrap_or((0, 0)),
        }
    }

    /// Per-track row count for the current layout.
    pub fn current_track_row_count(&self, t_idx: usize) -> u16 {
        match self.layout_mode {
            LayoutMode::LeftHeavy => self.aggregated_track_rows.get(t_idx).copied().unwrap_or(0),
            LayoutMode::TimeOrdered => self
                .profile
                .as_ref()
                .and_then(|p| p.tracks.get(t_idx))
                .map(|t| t.row_count)
                .unwrap_or(0),
        }
    }

    /// Switch layout mode. Refits the viewport so the user lands on the new
    /// layout's time range.
    pub fn set_layout_mode(&mut self, mode: LayoutMode) {
        if self.layout_mode == mode {
            return;
        }
        self.layout_mode = mode;
        let (s, e) = self.current_time_range();
        self.viewport.fit_time(s, e);
        self.viewport.row_height_px = ROW_HEIGHT_PX;
        self.viewport.scroll_y_px = 0.0;
        self.clamp_viewport();
        self.selected_slice = None;
    }

    /// Toggle collapsed state of a track.
    pub fn toggle_track_collapsed(&mut self, t: TrackId) {
        if !self.collapsed_tracks.remove(&t) {
            self.collapsed_tracks.insert(t);
        }
    }

    pub fn is_track_collapsed(&self, t: TrackId) -> bool {
        self.collapsed_tracks.contains(&t)
    }

    /// Hit-test the floating layout-mode toggle buttons in the bottom-left.
    pub fn hit_test_layout_button(&self, x: f32, y: f32) -> Option<LayoutMode> {
        for (mode, rect) in &self.layout_button_rects {
            if x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3] {
                return Some(*mode);
            }
        }
        None
    }

    /// Hit-test track headers (FLAME tab only). Returns the track ID to toggle.
    pub fn hit_test_track_header(&self, x: f32, y: f32) -> Option<TrackId> {
        for (t, rect) in &self.track_header_rects {
            if x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3] {
                return Some(*t);
            }
        }
        None
    }

    /// Switch the active inspector tab. Caller must follow with rebuild_instances + redraw.
    pub fn set_tab(&mut self, tab: MainTab) {
        self.active_tab = tab;
    }

    /// Per-tab list of (slice_idx, label) rows to display. Each row's slice_idx
    /// is the exemplar slice the row links to (click → select that slice).
    pub fn tab_rows(&mut self) -> Vec<(u32, String)> {
        let n_max = MAX_HOTSPOTS * if self.active_tab == MainTab::Flame { 1 } else { 4 };
        let Some(profile) = &self.profile else { return Vec::new() };
        let total_dur_ns = profile.duration_ns().max(1) as f64;
        let strings = &profile.strings;
        match self.active_tab {
            MainTab::Flame => Vec::new(),
            // CallTree has its own custom rendering path; tab_rows is unused.
            MainTab::CallTree => Vec::new(),
            MainTab::Hotspots => self
                .functions_by_total
                .iter()
                .take(n_max)
                .map(|h| {
                    let pct = (h.total_dur_ns as f64 / total_dur_ns) * 100.0;
                    let label = format!(
                        "{:<24.24}  total {:>10}  ×{}\n{:.1}%  ·  self {}",
                        strings.get(h.name),
                        format_duration(h.total_dur_ns),
                        h.count,
                        pct,
                        format_duration(h.self_dur_ns),
                    );
                    (h.exemplar_slice_idx, label)
                })
                .collect(),
            MainTab::BottomUp => self
                .functions_by_self
                .iter()
                .take(n_max)
                .map(|h| {
                    let pct = (h.self_dur_ns as f64 / total_dur_ns) * 100.0;
                    let label = format!(
                        "{:<24.24}  self  {:>10}  ×{}\n{:.1}% self  ·  total {}",
                        strings.get(h.name),
                        format_duration(h.self_dur_ns),
                        h.count,
                        pct,
                        format_duration(h.total_dur_ns),
                    );
                    (h.exemplar_slice_idx, label)
                })
                .collect(),
            MainTab::Table => self
                .functions_by_total
                .iter()
                .take(n_max)
                .map(|h| {
                    let label = format!(
                        "{:<26.26}  {:>10}  {:>10}  ×{}",
                        strings.get(h.name),
                        format_duration(h.total_dur_ns),
                        format_duration(h.self_dur_ns),
                        h.count,
                    );
                    (h.exemplar_slice_idx, label)
                })
                .collect(),
        }
    }

    /// Y-coordinate where the SANDWICH heading begins in the FLAME-tab inspector.
    /// Used by both `rebuild_instances` and `prepare_text` so chrome and text agree.
    pub fn sandwich_section_y(&self) -> Option<f32> {
        let sel = self.selected_slice?;
        let pad = INSPECTOR_PADDING_PX;
        let line = INSPECTOR_LINE_HEIGHT_PX;
        let mut y = TAB_BAR_HEIGHT_PX + pad;
        y += line; // SELECTED heading
        y += INSPECTOR_TITLE_LINE_HEIGHT; // name
        y += line; // meta
        y += SECTION_GAP_PX;
        y += line; // STACK heading
        let stack_len = self.reconstruct_stack(sel).len();
        let lines_shown = stack_len.min(8);
        y += line * lines_shown as f32;
        if stack_len > 8 {
            y += line; // "…" overflow line
        }
        y += SECTION_GAP_PX;
        Some(y)
    }

    /// Sandwich rows for the inspector sidebar — top-N callers + top-N callees
    /// of the currently-selected slice's function. Returns `None` when there's
    /// no selection.
    pub fn sandwich_rows(&mut self, max_each: usize) -> Option<Vec<(u32, String)>> {
        self.selected_slice?;
        let agg = self.ensure_sandwich()?.clone();
        let profile = self.profile.as_ref()?;
        let strings = &profile.strings;
        let mut rows = Vec::new();
        rows.push((u32::MAX, "── CALLERS ──".into()));
        if agg.callers.is_empty() {
            rows.push((u32::MAX, "  (root function)".into()));
        }
        for (name, total, count, ex) in agg.callers.iter().take(max_each) {
            rows.push((
                *ex,
                format!(
                    "↑ {:<20.20}  {:>10}  ×{}",
                    strings.get(*name),
                    format_duration(*total),
                    count
                ),
            ));
        }
        rows.push((u32::MAX, "── CALLEES ──".into()));
        if agg.callees.is_empty() {
            rows.push((u32::MAX, "  (leaf function)".into()));
        }
        for (name, total, count, ex) in agg.callees.iter().take(max_each) {
            rows.push((
                *ex,
                format!(
                    "↓ {:<20.20}  {:>10}  ×{}",
                    strings.get(*name),
                    format_duration(*total),
                    count
                ),
            ));
        }
        Some(rows)
    }

    /// Get-or-compute the sandwich aggregate for the selected slice's function.
    /// Cached on `self.sandwich_cache` keyed by the function name.
    pub fn ensure_sandwich(&mut self) -> Option<&CallerCalleeAggregate> {
        let sel = self.selected_slice?;
        let profile = self.profile.clone()?;
        let i = sel as usize;
        if i >= profile.slices.len() {
            return None;
        }
        let target = profile.slices.name[i];
        if !matches!(&self.sandwich_cache, Some((n, _)) if *n == target) {
            self.sandwich_cache = Some((target, compute_sandwich(&profile, target)));
        }
        self.sandwich_cache.as_ref().map(|(_, a)| a)
    }

    /// Time-range bounds (start, end) of the loaded profile, or `(0, 0)` if none.
    pub fn time_range(&self) -> (u64, u64) {
        self.profile.as_ref().map(|p| p.time_range).unwrap_or((0, 0))
    }

    /// Apply the viewport's pan/zoom bounds against the loaded profile.
    pub fn clamp_viewport(&mut self) {
        let (start, end) = self.time_range();
        self.viewport.clamp(start, end);
        // Vertical scroll bounds: stop at top of first track, allow scrolling so
        // that the last row stays visible above the status bar.
        let usable_h = (self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX).max(0.0);
        let max_scroll = (self.canvas_height_px - usable_h * 0.5).max(0.0);
        if self.viewport.scroll_y_px > max_scroll {
            self.viewport.scroll_y_px = max_scroll;
        }
        if self.viewport.scroll_y_px < 0.0 {
            self.viewport.scroll_y_px = 0.0;
        }
    }

    /// Toggle or set the icicle/flame direction. Caller must follow with
    /// `rebuild_instances` and a redraw.
    pub fn set_direction(&mut self, dir: Direction) {
        self.direction = dir;
    }
    pub fn flip_direction(&mut self) {
        self.direction = self.direction.flipped();
    }

    /// Reset both axes so the entire trace fits in the viewport: time fills
    /// width, and row height shrinks (down to 0.5px) so all tracks' rows fit
    /// vertically. Useful as a "show me the whole thing" reset.
    pub fn fit_all(&mut self) {
        let Some(p) = &self.profile else { return };
        // Horizontal: full timeline width.
        let (start_ns, end_ns) = p.time_range;
        self.viewport.fit_time(start_ns, end_ns);

        // Vertical: row height that makes total content == usable height.
        let total_rows: u32 = p.tracks.iter().map(|t| t.row_count as u32).sum();
        let usable_h = (self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX).max(1.0);
        let n_tracks = p.tracks.len() as f32;
        let header_total = TRACK_HEADER_HEIGHT_PX * n_tracks;
        let gap_total = TRACK_GAP_PX * (n_tracks - 1.0).max(0.0);
        let row_h = if total_rows == 0 {
            ROW_HEIGHT_PX
        } else {
            ((usable_h - header_total - gap_total) / total_rows as f32)
                .clamp(0.5, ROW_HEIGHT_PX)
        };
        self.viewport.row_height_px = row_h;
        self.viewport.scroll_y_px = 0.0;
        self.clamp_viewport();
    }

    fn refresh_status_for_idle(&mut self) {
        let Some(p) = &self.profile else { return };
        let n = p.slices.len();
        let dur = format_duration(p.duration_ns());
        self.status_text = format!("{} slices · duration {}", n, dur);
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.surface_config.width = w.max(1);
        self.surface_config.height = h.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        self.window_w_px = w as f32;
        let timeline_w = (w as f32 - INSPECTOR_WIDTH_PX).max(1.0);
        self.viewport.resize(timeline_w, h as f32);
        self.clamp_viewport();
        self.glyphon_viewport.update(
            &self.queue,
            Resolution { width: w, height: h },
        );
    }

    /// X-coordinate where the inspector panel begins.
    pub fn inspector_x(&self) -> f32 {
        self.window_w_px - INSPECTOR_WIDTH_PX
    }

    /// True if `x` falls inside the inspector area (right side of the window).
    pub fn cursor_in_inspector(&self, x: f32) -> bool {
        x >= self.inspector_x()
    }

    /// If the cursor is over a hotspot row in the inspector, returns the
    /// slice index to jump to. Skips section-heading rows whose slice_idx is
    /// `u32::MAX`.
    pub fn hit_test_inspector(&self, cursor_x: f32, cursor_y: f32) -> Option<u32> {
        let raw = hit_test_hotspot_rect(&self.inspector_hotspot_rects, cursor_x, cursor_y)?;
        if raw == u32::MAX { None } else { Some(raw) }
    }

    /// If the cursor is over a tab button, returns the tab to switch to.
    pub fn hit_test_inspector_tab(&self, cursor_x: f32, cursor_y: f32) -> Option<MainTab> {
        for (tab, rect) in &self.inspector_tab_rects {
            let [x, y, w, h] = *rect;
            if cursor_x >= x && cursor_x < x + w && cursor_y >= y && cursor_y < y + h {
                return Some(*tab);
            }
        }
        None
    }

    /// Persistent selection. Stores the SoA slice index so it survives
    /// `rebuild_instances` (which renumbers `instance_id`s).
    pub fn select_slice(&mut self, slice_idx: Option<u32>) {
        self.selected_slice = slice_idx;
        if let Some(_) = slice_idx {
            // Refresh status to reflect selection if nothing is hovered.
            if self.hovered.is_none() {
                self.refresh_status_for_idle();
            }
        }
    }

    /// Convert a hovered/clicked instance_id into the underlying SoA slice index.
    pub fn instance_to_slice(&self, instance_id: u32) -> Option<u32> {
        let raw = *self.slice_indices.get(instance_id as usize)?;
        if raw == NON_SLICE_SENTINEL { None } else { Some(raw) }
    }

    /// Walk the spatial parent chain of a slice on the same track. For each
    /// shallower depth, find the slice whose interval contains this slice's
    /// start. Returns root-first.
    pub fn reconstruct_stack(&self, slice_idx: u32) -> Vec<u32> {
        let Some(profile) = &self.profile else { return Vec::new() };
        let i = slice_idx as usize;
        if i >= profile.slices.len() {
            return Vec::new();
        }
        let track = profile.slices.track[i];
        let depth = profile.slices.depth[i];
        let start = profile.slices.start_ns[i];
        let mut chain = Vec::with_capacity(depth as usize + 1);
        for d in 0..depth {
            // visible_in_row treats `[lo, hi)` so use a 1-ns window at `start`.
            let row = profile.slices.visible_in_row(track, d, start, start + 1);
            // Find the slice in `row` whose interval covers `start`.
            for r in row.start..row.end {
                let s = profile.slices.start_ns[r as usize];
                let e = s + profile.slices.dur_ns[r as usize];
                if s <= start && start < e {
                    chain.push(r);
                    break;
                }
            }
        }
        chain.push(slice_idx);
        chain
    }

    /// Re-cull and re-pack the instance buffer based on the current viewport.
    pub fn rebuild_instances(&mut self) {
        self.instances.clear();
        self.slice_indices.clear();
        self.track_layouts.clear();
        self.inspector_hotspot_rects.clear();
        self.inspector_tab_rects.clear();
        self.track_header_rects.clear();
        self.layout_button_rects.clear();

        let Some(profile) = self.profile.clone() else {
            // Empty state — still draw the tab bar.
            self.emit_top_tab_bar();
            self.instance_count = 0;
            return;
        };

        // Non-flame tabs render a full-window content list below the tab bar.
        if self.active_tab == MainTab::CallTree {
            self.emit_call_tree_content();
            self.emit_top_tab_bar();
            self.finalize_instance_buffer();
            return;
        }
        if self.active_tab != MainTab::Flame {
            self.emit_full_tab_content();
            // Top tab bar last so it overdraws any spillover from rows.
            self.emit_top_tab_bar();
            self.finalize_instance_buffer();
            return;
        }

        let lo_ns = self.viewport.start_ns.max(0.0) as u64;
        let hi_ns = self.viewport.end_ns().max(0.0) as u64;
        let view_w_px = self.viewport.size_px.0;
        let row_h = self.viewport.row_height_px;
        // Usable timeline area lives below the tab bar and above the status bar.
        let top_offset = TAB_BAR_HEIGHT_PX;
        let view_h_px = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX - top_offset;
        let scroll_y = self.viewport.scroll_y_px;
        let direction = self.direction;
        let layout_mode = self.layout_mode;
        // Hold the active SliceTable behind an Arc so the rest of this method
        // can read it freely while still mutating `self` for instance pushes.
        let slices_arc: Arc<flame_core::SliceTable> = match (
            layout_mode,
            self.aggregated_slices.clone(),
        ) {
            (LayoutMode::LeftHeavy, Some(s)) => s,
            _ => Arc::new(flame_core::SliceTable::default()), // placeholder; real path uses profile
        };
        // For TimeOrdered we use profile.slices directly (no Arc).
        let use_aggregated = layout_mode == LayoutMode::LeftHeavy && self.aggregated_slices.is_some();
        macro_rules! slices {
            () => {
                if use_aggregated { &*slices_arc } else { &profile.slices }
            };
        }

        // Compute per-track canvas-space layout. Collapsed tracks render only
        // their header (content_h = 0). Empty tracks still list their header
        // so the user knows they exist.
        let mut y = 0.0_f32;
        for (i, _track) in profile.tracks.iter().enumerate() {
            let track_id = TrackId(i as u32);
            let rows = if use_aggregated {
                self.aggregated_track_rows.get(i).copied().unwrap_or(0)
            } else {
                profile.tracks[i].row_count
            };
            let collapsed = self.collapsed_tracks.contains(&track_id);
            let content_h = if collapsed { 0.0 } else { rows as f32 * row_h };
            let layout = TrackLayout {
                track: track_id,
                y_top: y,
                header_h: TRACK_HEADER_HEIGHT_PX,
                rows: if collapsed { 0 } else { rows },
                content_h,
            };
            y += TRACK_HEADER_HEIGHT_PX + content_h + TRACK_GAP_PX;
            self.track_layouts.push(layout);
        }
        self.canvas_height_px = y;

        for layout in &self.track_layouts {
            let track_y = layout.y_top - scroll_y;
            let total_h = layout.header_h + layout.content_h;

            // Whole track outside the visible band? skip.
            if track_y + total_h <= 0.0 || track_y >= view_h_px {
                continue;
            }

            // Header bar instance: full-width at the top of the track. Click
            // the header to toggle collapse.
            let header_y = track_y;
            if header_y + layout.header_h > 0.0 && header_y < view_h_px {
                let inst_id = self.instances.len() as u32;
                let rect = [0.0, header_y + top_offset, view_w_px, layout.header_h - 2.0];
                self.instances.push(SliceInstance {
                    rect_px: rect,
                    color: HEADER_COLOR,
                    instance_id: inst_id,
                    flags: 1, // bit 0 = header
                    _pad: [0; 2],
                });
                self.slice_indices.push(NON_SLICE_SENTINEL);
                // Restrict click hit-test to timeline width (don't trigger when
                // the user clicks the inspector area at the same y).
                let header_w = view_w_px;
                self.track_header_rects.push((
                    layout.track,
                    [0.0, header_y + top_offset, header_w, layout.header_h - 2.0],
                ));
            }

            // Collapsed → no slice content for this track.
            if self.collapsed_tracks.contains(&layout.track) {
                continue;
            }

            // Compute the slice of `depth` values that actually fall in the
            // viewport. For 1000-deep tracks this is the difference between
            // walking 1000 rows and walking ~14.
            let track_content_top = track_y + layout.header_h;
            let lo_y = -row_h;
            let hi_y = view_h_px;
            let (depth_start, depth_end) = visible_depth_range(
                direction,
                layout.rows,
                track_content_top,
                row_h,
                lo_y,
                hi_y,
            );

            for depth in depth_start..depth_end {
                let y_in_track = layout.header_h
                    + match direction {
                        Direction::Icicle => depth as f32 * row_h,
                        Direction::Flame => (layout.rows - 1 - depth) as f32 * row_h,
                    };
                let y_top = track_y + y_in_track;
                if y_top + row_h <= 0.0 || y_top >= view_h_px {
                    continue;
                }

                let active_slices: &flame_core::SliceTable = slices!();
                let row = active_slices.visible_in_row(layout.track, depth, lo_ns, hi_ns);
                let max_x = view_w_px;

                // Sub-pixel slices get *bucketed* by pixel column instead of
                // culled. For every pixel column with sub-pixel slices we emit
                // one 1px-wide instance representing the longest-duration
                // slice in that column — that way deep rows still show content
                // at zoom-out, but we don't push millions of overlapping
                // instances. Wide slices (>= 1px) emit normally.
                let mut pending_col: Option<(u32, i32, u64, [f32; 4])> = None;

                for i in row.start..row.end {
                    let idx = i as usize;
                    let s = active_slices.start_ns[idx];
                    let d = active_slices.dur_ns[idx];
                    let x = self.viewport.ns_to_x(s);
                    let w_px = (d as f64 / self.viewport.ns_per_pixel) as f32;
                    let name_id = active_slices.name[idx];
                    let name = profile.strings.get(name_id);
                    let color = palette::color_for(name);

                    if w_px >= 1.0 {
                        // Flush pending sub-pixel bucket first.
                        if let Some((bidx, col, _, bcolor)) = pending_col.take() {
                            push_subpx_instance(
                                &mut self.instances,
                                &mut self.slice_indices,
                                bidx,
                                col as f32,
                                y_top + top_offset,
                                row_h,
                                bcolor,
                                self.selected_slice == Some(bidx),
                            );
                        }
                        let x_clamped = x.max(-2.0);
                        let w_clamped = (w_px - (x_clamped - x)).min(max_x - x_clamped);
                        if w_clamped <= 0.0 {
                            continue;
                        }
                        let inst_id = self.instances.len() as u32;
                        let mut flags: u32 = 0;
                        if Some(i) == self.selected_slice {
                            flags |= 2;
                        }
                        self.instances.push(SliceInstance {
                            rect_px: [x_clamped, y_top + top_offset, w_clamped, row_h - 1.0],
                            color,
                            instance_id: inst_id,
                            flags,
                            _pad: [0; 2],
                        });
                        self.slice_indices.push(i);
                    } else {
                        let col = x.floor().max(0.0).min(max_x - 1.0) as i32;
                        match pending_col {
                            Some((bidx, bcol, bdur, bcolor)) if bcol == col => {
                                if d > bdur {
                                    pending_col = Some((i, col, d, color));
                                } else {
                                    pending_col = Some((bidx, bcol, bdur, bcolor));
                                }
                            }
                            Some((bidx, bcol, _bdur, bcolor)) => {
                                push_subpx_instance(
                                    &mut self.instances,
                                    &mut self.slice_indices,
                                    bidx,
                                    bcol as f32,
                                    y_top + top_offset,
                                    row_h,
                                    bcolor,
                                    self.selected_slice == Some(bidx),
                                );
                                pending_col = Some((i, col, d, color));
                            }
                            None => {
                                pending_col = Some((i, col, d, color));
                            }
                        }
                    }
                }
                if let Some((bidx, col, _, bcolor)) = pending_col.take() {
                    push_subpx_instance(
                        &mut self.instances,
                        &mut self.slice_indices,
                        bidx,
                        col as f32,
                        y_top + top_offset,
                        row_h,
                        bcolor,
                        self.selected_slice == Some(bidx),
                    );
                }
            }
        }

        // Right-side inspector (only on FLAME tab). Sits below the tab bar.
        let inspector_x = self.window_w_px - INSPECTOR_WIDTH_PX;
        let inspector_top = TAB_BAR_HEIGHT_PX;
        let inspector_h = self.viewport.size_px.1 - inspector_top;
        if INSPECTOR_WIDTH_PX > 0.0 {
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [inspector_x, inspector_top, INSPECTOR_WIDTH_PX, inspector_h],
                color: INSPECTOR_BG_COLOR,
                instance_id: inst_id,
                flags: 1, // header bit so hit_test ignores it
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);
        }

        // Floating layout-mode toggle buttons in the bottom-left of the timeline.
        {
            let btn_h = 44.0_f32;
            let btn_w = 156.0_f32;
            let btn_gap = 4.0_f32;
            let pad = 16.0_f32;
            let total_w = btn_w * LayoutMode::ALL.len() as f32
                + btn_gap * (LayoutMode::ALL.len() as f32 - 1.0);
            let bar_w = total_w + pad * 2.0;
            let bar_h = btn_h + pad;
            let bar_x = 12.0_f32;
            let bar_y = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX - bar_h - 12.0;

            // Backing pill so the buttons read against the timeline.
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [bar_x, bar_y, bar_w, bar_h],
                color: [0.10, 0.11, 0.14, 0.92],
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);

            for (i, &mode) in LayoutMode::ALL.iter().enumerate() {
                let x = bar_x + pad + (btn_w + btn_gap) * i as f32;
                let y = bar_y + pad * 0.5;
                let active = mode == self.layout_mode;
                let bg = if active { TOP_TAB_ACTIVE } else { TOP_TAB_INACTIVE };
                let inst_id = self.instances.len() as u32;
                self.instances.push(SliceInstance {
                    rect_px: [x, y, btn_w, btn_h],
                    color: bg,
                    instance_id: inst_id,
                    flags: 1,
                    _pad: [0; 2],
                });
                self.slice_indices.push(NON_SLICE_SENTINEL);
                self.layout_button_rects.push((mode, [x, y, btn_w, btn_h]));
                if active {
                    let inst_id = self.instances.len() as u32;
                    self.instances.push(SliceInstance {
                        rect_px: [x, y + btn_h - 3.0, btn_w, 3.0],
                        color: TOP_TAB_ACCENT,
                        instance_id: inst_id,
                        flags: 1,
                        _pad: [0; 2],
                    });
                    self.slice_indices.push(NON_SLICE_SENTINEL);
                }
            }
        }

        // SANDWICH section — only when a slice is selected. Renders below STACK.
        if let Some(section_y) = self.sandwich_section_y() {
            let pad = INSPECTOR_PADDING_PX;
            let line = INSPECTOR_LINE_HEIGHT_PX;
            // Skip the SANDWICH heading line; chrome rows start below it.
            let mut y = section_y + line + 4.0;
            let row_h = line * 1.4;
            let band_x = inspector_x + pad;
            let band_w = INSPECTOR_WIDTH_PX - pad * 2.0;
            let max_y = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX;

            if let Some(rows) = self.sandwich_rows(5) {
                for (i, (slice_idx, _)) in rows.iter().enumerate() {
                    if y + row_h > max_y {
                        break;
                    }
                    // Section markers ("── CALLERS ──") get a faint divider band
                    // instead of a normal alternating row.
                    let is_marker = *slice_idx == u32::MAX;
                    let bg = if is_marker {
                        SANDWICH_DIVIDER
                    } else if i % 2 == 0 {
                        ROW_COLOR_EVEN
                    } else {
                        ROW_COLOR_ODD
                    };
                    let inst_id = self.instances.len() as u32;
                    self.instances.push(SliceInstance {
                        rect_px: [band_x, y, band_w, row_h - 2.0],
                        color: bg,
                        instance_id: inst_id,
                        flags: 1,
                        _pad: [0; 2],
                    });
                    self.slice_indices.push(NON_SLICE_SENTINEL);
                    self.inspector_hotspot_rects.push(InspectorHotspotRect {
                        rect: [band_x, y, band_w, row_h - 2.0],
                        slice_idx: *slice_idx,
                    });
                    y += row_h;
                }
            }
        }

        // Top tab bar last so it always overdraws any slice that scrolled up
        // into its y-range. (Same applies to the tab content path; the no-tab
        // empty state path emits its tab bar earlier.)
        self.emit_top_tab_bar();

        self.finalize_instance_buffer();
    }

    /// Top-of-window global tab bar: dark background + button per tab.
    fn emit_top_tab_bar(&mut self) {
        // Background.
        let bg_id = self.instances.len() as u32;
        self.instances.push(SliceInstance {
            rect_px: [0.0, 0.0, self.window_w_px, TAB_BAR_HEIGHT_PX],
            color: TAB_BAR_BG_COLOR,
            instance_id: bg_id,
            flags: 1,
            _pad: [0; 2],
        });
        self.slice_indices.push(NON_SLICE_SENTINEL);

        // Buttons evenly spaced across the full window.
        let n = MainTab::ALL.len();
        let outer_pad = 16.0_f32;
        let gap = 8.0_f32;
        let usable = self.window_w_px - outer_pad * 2.0 - gap * (n as f32 - 1.0);
        let btn_w = usable / n as f32;
        let btn_h = TAB_BAR_HEIGHT_PX - 16.0;
        let btn_y = 8.0_f32;
        for (i, &tab) in MainTab::ALL.iter().enumerate() {
            let x = outer_pad + (btn_w + gap) * i as f32;
            let bg = if tab == self.active_tab { TOP_TAB_ACTIVE } else { TOP_TAB_INACTIVE };
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [x, btn_y, btn_w, btn_h],
                color: bg,
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);
            self.inspector_tab_rects.push((tab, [x, btn_y, btn_w, btn_h]));

            // Active tab gets a warm accent strip along its bottom edge.
            if tab == self.active_tab {
                let strip_h = 4.0;
                let inst_id = self.instances.len() as u32;
                self.instances.push(SliceInstance {
                    rect_px: [x, btn_y + btn_h - strip_h, btn_w, strip_h],
                    color: TOP_TAB_ACCENT,
                    instance_id: inst_id,
                    flags: 1,
                    _pad: [0; 2],
                });
                self.slice_indices.push(NON_SLICE_SENTINEL);
            }
        }
    }

    /// Walk the call tree in display order. Yields `(node_idx, has_children)`
    /// for every node whose ancestors are all expanded. Roots are always yielded.
    pub fn visible_tree_nodes(&self) -> Vec<(u32, bool)> {
        let Some(tree) = &self.call_tree else { return Vec::new() };
        let mut out: Vec<(u32, bool)> = Vec::new();
        let mut stack: Vec<u32> = tree.roots.iter().rev().copied().collect();
        while let Some(node_idx) = stack.pop() {
            let node = &tree.nodes[node_idx as usize];
            let has_children = !node.children.is_empty();
            out.push((node_idx, has_children));
            if has_children && self.expanded_tree_nodes.contains(&node_idx) {
                for &c in node.children.iter().rev() {
                    stack.push(c);
                }
            }
        }
        out
    }

    /// Maximum scroll offset so the last row stays at least partially in view.
    fn call_tree_max_scroll(&self) -> f32 {
        let visible = self.visible_tree_nodes().len() as f32;
        let row_h = CALL_TREE_ROW_H;
        let view_h = (self.viewport.size_px.1
            - STATUS_BAR_HEIGHT_PX
            - TAB_BAR_HEIGHT_PX
            - CALL_TREE_HEADER_H)
            .max(0.0);
        let total = visible * row_h;
        (total - view_h * 0.5).max(0.0)
    }

    /// Pan the call-tree view by `dy_px` (positive = scroll down).
    pub fn pan_call_tree(&mut self, dy_px: f32) {
        self.call_tree_scroll_y_px = (self.call_tree_scroll_y_px - dy_px).max(0.0);
        let max = self.call_tree_max_scroll();
        if self.call_tree_scroll_y_px > max {
            self.call_tree_scroll_y_px = max;
        }
    }

    /// Toggle expansion of one tree node.
    pub fn toggle_tree_node(&mut self, node_idx: u32) {
        if !self.expanded_tree_nodes.remove(&node_idx) {
            self.expanded_tree_nodes.insert(node_idx);
        }
    }

    /// Hit-test the rendered call tree rows. Returns the node index whose
    /// rect contains the cursor, or `None`.
    pub fn hit_test_call_tree(&self, x: f32, y: f32) -> Option<u32> {
        for (node_idx, rect) in &self.call_tree_row_rects {
            let [rx, ry, rw, rh] = *rect;
            if x >= rx && x < rx + rw && y >= ry && y < ry + rh {
                return Some(*node_idx);
            }
        }
        None
    }

    /// Emit row backgrounds + click rects for the CallTree tab. Text labels are
    /// produced in `prepare_text` against the same `call_tree_row_rects`.
    fn emit_call_tree_content(&mut self) {
        self.call_tree_row_rects.clear();
        let top = TAB_BAR_HEIGHT_PX + CALL_TREE_HEADER_H;
        let view_bottom = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX;
        let view_h = view_bottom - top;
        if view_h <= 0.0 {
            return;
        }
        let pad = 16.0_f32;
        let band_x = pad;
        let band_w = (self.window_w_px - pad * 2.0).max(0.0);
        let row_h = CALL_TREE_ROW_H;

        let visible = self.visible_tree_nodes();
        let scroll = self.call_tree_scroll_y_px;
        for (i, (node_idx, _has_children)) in visible.iter().enumerate() {
            let y = top + i as f32 * row_h - scroll;
            // Skip rows entirely above or below the visible band.
            if y + row_h <= top || y >= view_bottom {
                continue;
            }
            // Even/odd alternating row colors keep tree depth readable.
            let bg = if i % 2 == 0 { ROW_COLOR_EVEN } else { ROW_COLOR_ODD };
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [band_x, y, band_w, row_h - 1.0],
                color: bg,
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);
            // Clip click rect to the visible content band so clicks on rows
            // partially scrolled under the heading don't trigger.
            let click_top = y.max(top);
            let click_bottom = (y + row_h - 1.0).min(view_bottom);
            if click_bottom > click_top {
                self.call_tree_row_rects.push((
                    *node_idx,
                    [band_x, click_top, band_w, click_bottom - click_top],
                ));
            }
        }
    }

    /// Full-window content view used by HOT / SELF / SAND / ALL tabs.
    fn emit_full_tab_content(&mut self) {
        // Header band underneath the tab bar — empty so the tab content has space.
        let top = TAB_BAR_HEIGHT_PX;
        let view_h = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX - top;
        if view_h <= 0.0 {
            return;
        }
        let pad = INSPECTOR_PADDING_PX;
        let row_h = INSPECTOR_LINE_HEIGHT_PX * 2.4;
        let band_x = pad;
        let band_w = self.window_w_px - pad * 2.0;
        // Reserve room at the top for a section heading line.
        let mut y = top + pad + INSPECTOR_LINE_HEIGHT_PX + 8.0;

        let row_entries = self.tab_rows();
        for (i, (slice_idx, _)) in row_entries.iter().enumerate() {
            if y + row_h > top + view_h {
                break;
            }
            let bg_color = if i % 2 == 0 { ROW_COLOR_EVEN } else { ROW_COLOR_ODD };
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [band_x, y, band_w, row_h - 2.0],
                color: bg_color,
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);
            self.inspector_hotspot_rects.push(InspectorHotspotRect {
                rect: [band_x, y, band_w, row_h - 2.0],
                slice_idx: *slice_idx,
            });
            y += row_h;
        }
    }

    fn finalize_instance_buffer(&mut self) {
        self.instance_count = self.instances.len() as u32;

        // Grow GPU buffer if needed.
        let needed_capacity = self.instances.len() as u64;
        if needed_capacity > self.instance_capacity {
            self.instance_capacity = (needed_capacity * 2).max(self.instance_capacity * 2);
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: self.instance_capacity
                    * std::mem::size_of::<SliceInstance>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.instances.is_empty() {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
    }

    /// Resolve which slice instance the cursor is over (if any). Header
    /// instances (flags bit 0) are skipped.
    pub fn hit_test(&self, cursor_x: f32, cursor_y: f32) -> Option<u32> {
        for inst in self.instances.iter().rev() {
            if inst.flags & 1 != 0 {
                continue;
            }
            let [x, y, w, h] = inst.rect_px;
            if cursor_x >= x && cursor_x < x + w && cursor_y >= y && cursor_y < y + h {
                return Some(inst.instance_id);
            }
        }
        None
    }

    pub fn set_hover(&mut self, instance: Option<u32>) {
        if self.hovered == instance {
            return;
        }
        self.hovered = instance;
        if let (Some(id), Some(profile)) = (instance, &self.profile) {
            let slot = id as usize;
            let slice_idx_raw = self.slice_indices.get(slot).copied().unwrap_or(NON_SLICE_SENTINEL);
            if slice_idx_raw != NON_SLICE_SENTINEL {
                let slice_idx = slice_idx_raw as usize;
                let name = profile.strings.get(profile.slices.name[slice_idx]);
                let dur = profile.slices.dur_ns[slice_idx];
                let depth = profile.slices.depth[slice_idx];
                let track_idx = profile.slices.track[slice_idx].0 as usize;
                let track_name = profile
                    .tracks
                    .get(track_idx)
                    .map(|t| profile.strings.get(t.name))
                    .unwrap_or("?");
                self.status_text = format!(
                    "{}    duration {}    depth {}    track {}",
                    name,
                    format_duration(dur),
                    depth,
                    track_name,
                );
            } else {
                self.refresh_status_for_idle();
            }
        } else {
            self.refresh_status_for_idle();
        }
    }

    fn prepare_text(&mut self) {
        let Some(profile) = self.profile.clone() else {
            self.text_buffers.clear();
            return;
        };

        // Each label entry: (text, x, y, max_w, font_metric, color, zone).
        // `zone` tells us where the label lives — it determines glyphon's
        // clip bounds so timeline labels don't bleed into the tab bar (and
        // tab bar labels don't bleed below it).
        #[derive(Copy, Clone)]
        enum Metric { Slice, Header, InspectorHeading, InspectorTitle, InspectorBody }
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum Zone {
            /// Top tab bar (y in [0, TAB_BAR_HEIGHT_PX]). Anything below clipped.
            TabBar,
            /// Timeline / inspector content (clip top at TAB_BAR_HEIGHT_PX,
            /// bottom at status-bar top).
            Below,
            /// Status bar (no extra clipping).
            Status,
        }
        let row_h = self.viewport.row_height_px;
        let mut labels: Vec<(String, f32, f32, f32, Metric, GlyphonColor, Zone)> = Vec::new();
        let inner_pad = 6.0_f32;
        let scroll_y = self.viewport.scroll_y_px;
        let top_offset = TAB_BAR_HEIGHT_PX;
        let view_h_px = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX - top_offset;

        // Top tab bar labels (always present).
        let label_color = GlyphonColor::rgb(220, 220, 224);
        let dim_color = GlyphonColor::rgb(150, 150, 158);
        let title_color = GlyphonColor::rgb(255, 255, 255);
        for (tab, rect) in self.inspector_tab_rects.clone() {
            let color = if tab == self.active_tab { title_color } else { dim_color };
            let tx = rect[0];
            let ty = rect[1] + (rect[3] - INSPECTOR_LINE_HEIGHT_PX) * 0.5;
            labels.push((
                format!("  {}", tab.label()),
                tx,
                ty,
                rect[2],
                Metric::InspectorBody,
                color,
                Zone::TabBar,
            ));
        }

        // Floating layout-mode button labels.
        for (mode, rect) in self.layout_button_rects.clone() {
            let active = mode == self.layout_mode;
            let color = if active { title_color } else { dim_color };
            let lh = INSPECTOR_LINE_HEIGHT_PX;
            let tx = rect[0];
            let ty = rect[1] + (rect[3] - lh) * 0.5;
            labels.push((
                format!("  {}", mode.label()),
                tx,
                ty,
                rect[2],
                Metric::InspectorHeading,
                color,
                Zone::Below,
            ));
        }

        // Timeline-only content (FLAME tab only).
        if self.active_tab == MainTab::Flame {
            // Track header labels.
            for layout in &self.track_layouts {
                let track_y = layout.y_top - scroll_y;
                if track_y + layout.header_h <= 0.0 || track_y >= view_h_px {
                    continue;
                }
                let track_data = &profile.tracks[layout.track.0 as usize];
                let track_name = profile.strings.get(track_data.name).to_owned();
                let dir_tag = match self.direction {
                    Direction::Icicle => " (icicle)",
                    Direction::Flame => " (flame)",
                };
                let label = if layout.rows == 0 {
                    format!("{}  ·  empty", track_name)
                } else {
                    format!("{}  ·  {} rows{}", track_name, layout.rows, dir_tag)
                };
                labels.push((
                    label,
                    12.0,
                    track_y + top_offset + (layout.header_h - HEADER_FONT_SIZE) * 0.5,
                    self.viewport.size_px.0 - 24.0,
                    Metric::Header,
                    label_color,
                    Zone::Below,
                ));
            }

            // Slice labels.
            let slice_label_min_h = LABEL_FONT_SIZE + 4.0;
            let render_slice_labels = row_h >= slice_label_min_h;
            for (slot, inst) in self.instances.iter().enumerate() {
                if !render_slice_labels { break; }
                if inst.flags & 1 != 0 { continue; }
                if inst.rect_px[2] < SLICE_LABEL_MIN_PX { continue; }
                let slice_idx_raw = self.slice_indices[slot];
                if slice_idx_raw == NON_SLICE_SENTINEL { continue; }
                let name = profile.strings.get(profile.slices.name[slice_idx_raw as usize]).to_owned();
                labels.push((
                    name,
                    inst.rect_px[0] + inner_pad,
                    inst.rect_px[1] + (row_h - LABEL_FONT_SIZE) * 0.5,
                    (inst.rect_px[2] - inner_pad * 2.0).max(0.0),
                    Metric::Slice,
                    GlyphonColor::rgb(20, 20, 22),
                    Zone::Below,
                ));
            }
        }

        // Inspector content (selected slice, stack, hotspots).
        let inspector_x = self.window_w_px - INSPECTOR_WIDTH_PX;
        let pad = INSPECTOR_PADDING_PX;
        let line = INSPECTOR_LINE_HEIGHT_PX;

        let mut iy = top_offset + pad;
        let inspector_text_x = inspector_x + pad;
        let inspector_text_w = INSPECTOR_WIDTH_PX - pad * 2.0;

        // FLAME tab inspector content (SELECTED + STACK only). Other tabs show
        // their own full-window content view below the tab bar.
        if self.active_tab == MainTab::Flame {
        if let Some(sel_idx) = self.selected_slice {
            let i = sel_idx as usize;
            if i < profile.slices.len() {
                let name = profile.strings.get(profile.slices.name[i]).to_owned();
                let dur = profile.slices.dur_ns[i];
                let depth = profile.slices.depth[i];
                let track_idx = profile.slices.track[i].0 as usize;
                let track_name = profile.tracks.get(track_idx).map(|t| profile.strings.get(t.name)).unwrap_or("?");

                labels.push(("SELECTED".into(), inspector_text_x, iy, inspector_text_w, Metric::InspectorHeading, dim_color, Zone::Below));
                iy += line;
                labels.push((name, inspector_text_x, iy, inspector_text_w, Metric::InspectorTitle, title_color, Zone::Below));
                iy += INSPECTOR_TITLE_LINE_HEIGHT;
                let meta = format!("{}    depth {}    {}", format_duration(dur), depth, track_name);
                labels.push((meta, inspector_text_x, iy, inspector_text_w, Metric::InspectorBody, dim_color, Zone::Below));
                iy += line + SECTION_GAP_PX;

                // Reconstruct call stack.
                let stack = self.reconstruct_stack(sel_idx);
                labels.push(("STACK".into(), inspector_text_x, iy, inspector_text_w, Metric::InspectorHeading, dim_color, Zone::Below));
                iy += line;
                // Show up to last 8 entries; deeper traces get truncated with an ellipsis.
                let max_lines = 8;
                let show: Vec<&u32> = if stack.len() > max_lines {
                    stack.iter().rev().take(max_lines).collect::<Vec<_>>().into_iter().rev().collect()
                } else {
                    stack.iter().collect()
                };
                if stack.len() > max_lines {
                    labels.push(("…".into(), inspector_text_x + 16.0, iy, inspector_text_w - 16.0, Metric::InspectorBody, dim_color, Zone::Below));
                    iy += line;
                }
                for sidx_ref in show {
                    let sidx = *sidx_ref as usize;
                    let sname = profile.strings.get(profile.slices.name[sidx]);
                    let sdur = profile.slices.dur_ns[sidx];
                    let sdep = profile.slices.depth[sidx];
                    let dur_str = format_duration(sdur);
                    let line_text = format!("{:2} {:<26.26}{:>10}", sdep, sname, dur_str);
                    labels.push((
                        line_text,
                        inspector_text_x,
                        iy,
                        inspector_text_w,
                        Metric::InspectorBody,
                        label_color,
                        Zone::Below,
                    ));
                    iy += line;
                }
                iy += SECTION_GAP_PX;

                // SANDWICH section heading + per-row labels (chrome was emitted
                // in rebuild_instances; rects are in `inspector_hotspot_rects`).
                if !self.inspector_hotspot_rects.is_empty() {
                    labels.push((
                        "SANDWICH".into(),
                        inspector_text_x,
                        iy,
                        inspector_text_w,
                        Metric::InspectorHeading,
                        dim_color,
                        Zone::Below,
                    ));
                    let row_data = self.sandwich_rows(5).unwrap_or_default();
                    for (hi, rect) in self.inspector_hotspot_rects.clone().iter().enumerate() {
                        let Some((_, txt)) = row_data.get(hi) else { break };
                        let row_y = rect.rect[1] + (rect.rect[3] - line) * 0.5;
                        let color = if txt.starts_with("──") || txt.starts_with("  (")
                        {
                            dim_color
                        } else {
                            label_color
                        };
                        labels.push((
                            txt.clone(),
                            rect.rect[0] + 12.0,
                            row_y,
                            rect.rect[2] - 24.0,
                            Metric::InspectorBody,
                            color,
                            Zone::Below,
                        ));
                    }
                }
            }
        }
        // Close FLAME-only branch.
        } else if self.active_tab == MainTab::CallTree {
            // Column-header band: FUNCTION ......... TOTAL %TOTAL SELF COUNT.
            let header_pad_x = 16.0_f32;
            let header_y = top_offset + (CALL_TREE_HEADER_H - LABEL_LINE_HEIGHT) * 0.5;
            let band_x = header_pad_x;
            let band_w = (self.window_w_px - header_pad_x * 2.0).max(0.0);
            let metrics_w = 460.0_f32.min(band_w * 0.55);
            let name_col_w = (band_w - metrics_w).max(0.0);
            let metrics_x = band_x + name_col_w;

            labels.push((
                "FUNCTION".into(),
                band_x + 12.0,
                header_y,
                name_col_w - 12.0,
                Metric::InspectorHeading,
                dim_color,
                Zone::Below,
            ));
            labels.push((
                "  TOTAL    %TOTAL      SELF      ×".into(),
                metrics_x,
                header_y,
                metrics_w,
                Metric::InspectorHeading,
                dim_color,
                Zone::Below,
            ));

            // Per-row labels: chevron + indented name on the left, fixed
            // metrics column on the right.
            let total_ns = self
                .call_tree
                .as_ref()
                .map(|t| t.total_ns)
                .unwrap_or(0)
                .max(1) as f64;
            for (node_idx, rect) in self.call_tree_row_rects.clone() {
                let Some(tree) = &self.call_tree else { break };
                let node = &tree.nodes[node_idx as usize];
                let chev = if node.children.is_empty() {
                    "  "
                } else if self.expanded_tree_nodes.contains(&node_idx) {
                    "▾ "
                } else {
                    "▸ "
                };
                let indent_px = node.depth as f32 * CALL_TREE_INDENT_PX;
                let name = profile.strings.get(node.name);
                let left_text = format!("{chev}{name}");
                let row_y = rect[1] + (rect[3] - LABEL_LINE_HEIGHT) * 0.5;
                let left_x = rect[0] + 12.0 + indent_px;
                let left_w = (name_col_w - 12.0 - indent_px).max(0.0);
                labels.push((
                    left_text,
                    left_x,
                    row_y,
                    left_w,
                    Metric::InspectorBody,
                    label_color,
                    Zone::Below,
                ));
                let pct = (node.total_ns as f64 / total_ns) * 100.0;
                let metrics_text = format!(
                    "{:>9}  {:>6.2}%  {:>9}  ×{}",
                    format_duration(node.total_ns),
                    pct,
                    format_duration(node.self_ns),
                    node.count,
                );
                labels.push((
                    metrics_text,
                    metrics_x,
                    row_y,
                    metrics_w,
                    Metric::InspectorBody,
                    dim_color,
                    Zone::Below,
                ));
            }
        } else {
            // Full-window content for HOT / SELF / SAND / ALL tabs.
            // Section heading at the top.
            let heading_y = top_offset + pad;
            labels.push((
                self.active_tab.label().into(),
                pad,
                heading_y,
                self.window_w_px - pad * 2.0,
                Metric::InspectorHeading,
                dim_color,
                Zone::Below,
            ));

            let rows_data: Vec<(u32, String)> = self.tab_rows();
            for (hi, rect) in self.inspector_hotspot_rects.iter().enumerate() {
                let Some((_slice_idx, label_text)) = rows_data.get(hi) else { break };
                if label_text.starts_with("──") {
                    labels.push((
                        label_text.clone(),
                        rect.rect[0] + 12.0,
                        rect.rect[1] + (rect.rect[3] - line) * 0.5,
                        rect.rect[2] - 24.0,
                        Metric::InspectorHeading,
                        dim_color,
                        Zone::Below,
                    ));
                    continue;
                }
                let mut iter = label_text.splitn(2, '\n');
                let main = iter.next().unwrap_or("");
                let sub = iter.next().unwrap_or("");
                labels.push((
                    main.to_string(),
                    rect.rect[0] + 12.0,
                    rect.rect[1] + 6.0,
                    rect.rect[2] - 24.0,
                    Metric::InspectorBody,
                    label_color,
                    Zone::Below,
                ));
                if !sub.is_empty() {
                    labels.push((
                        sub.to_string(),
                        rect.rect[0] + 12.0,
                        rect.rect[1] + 6.0 + line * 0.95,
                        rect.rect[2] - 24.0,
                        Metric::InspectorHeading,
                        dim_color,
                        Zone::Below,
                    ));
                }
            }
        }

        // Status bar.
        let status_y = self.viewport.size_px.1
            - STATUS_BAR_HEIGHT_PX
            + (STATUS_BAR_HEIGHT_PX - LABEL_LINE_HEIGHT) * 0.5;
        labels.push((
            self.status_text.clone(),
            12.0,
            status_y,
            self.viewport.size_px.0 - 24.0,
            Metric::Slice,
            GlyphonColor::rgb(220, 220, 224),
            Zone::Status,
        ));

        // Resize the buffer pool to match.
        while self.text_buffers.len() < labels.len() {
            let mut buf = Buffer::new(
                &mut self.font_system,
                Metrics::new(LABEL_FONT_SIZE, LABEL_LINE_HEIGHT),
            );
            buf.set_wrap(&mut self.font_system, Wrap::None);
            self.text_buffers.push(buf);
        }
        if self.text_buffers.len() > labels.len() + 256 {
            self.text_buffers.truncate(labels.len() + 256);
        }

        for (i, (text, _x, _y, max_w, metric, _color, _zone)) in labels.iter().enumerate() {
            let buf = &mut self.text_buffers[i];
            let (fs, lh) = match metric {
                Metric::Slice => (LABEL_FONT_SIZE, LABEL_LINE_HEIGHT),
                Metric::Header => (HEADER_FONT_SIZE, HEADER_LINE_HEIGHT),
                Metric::InspectorHeading => {
                    (INSPECTOR_HEADING_FONT, INSPECTOR_LINE_HEIGHT_PX)
                }
                Metric::InspectorTitle => (INSPECTOR_TITLE_FONT, INSPECTOR_TITLE_LINE_HEIGHT),
                Metric::InspectorBody => (INSPECTOR_BODY_FONT, INSPECTOR_LINE_HEIGHT_PX),
            };
            buf.set_metrics(&mut self.font_system, Metrics::new(fs, lh));
            buf.set_size(&mut self.font_system, Some(*max_w), Some(lh));
            buf.set_text(
                &mut self.font_system,
                text,
                &Attrs::new().family(Family::SansSerif),
                Shaping::Advanced,
                None,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
        }

        // Glyphon bounds use the full window — inspector labels live to the right
        // of the timeline.
        let view_w = self.window_w_px as i32;
        let view_h = self.viewport.size_px.1 as i32;

        let text_areas: Vec<TextArea> = labels
            .iter()
            .enumerate()
            .map(|(i, (_, x, y, max_w, metric, color, zone))| {
                let lh = match metric {
                    Metric::Slice => LABEL_LINE_HEIGHT,
                    Metric::Header => HEADER_LINE_HEIGHT,
                    Metric::InspectorHeading | Metric::InspectorBody => INSPECTOR_LINE_HEIGHT_PX,
                    Metric::InspectorTitle => INSPECTOR_TITLE_LINE_HEIGHT,
                };
                let bounds_left = x.floor() as i32;
                let bounds_top = y.floor() as i32;
                let bounds_right = (x + max_w).ceil() as i32;
                let bounds_bottom = (y + lh).ceil() as i32;
                let tab_bar_h = TAB_BAR_HEIGHT_PX as i32;
                let status_top = (self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX) as i32;
                let (clip_top, clip_bot) = match zone {
                    Zone::TabBar => (0, tab_bar_h),
                    Zone::Below => (tab_bar_h, status_top),
                    Zone::Status => (status_top, view_h),
                };
                TextArea {
                    buffer: &self.text_buffers[i],
                    left: *x,
                    top: *y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: bounds_left.max(0),
                        top: bounds_top.max(clip_top),
                        right: bounds_right.min(view_w),
                        bottom: bounds_bottom.min(clip_bot),
                    },
                    default_color: *color,
                    custom_glyphs: &[],
                }
            })
            .collect();

        if let Err(err) = self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.glyphon_viewport,
            text_areas,
            &mut self.swash_cache,
        ) {
            log::warn!("glyphon prepare failed: {err:?}");
        }
    }

    pub fn render(&mut self) {
        // Update uniforms.
        let uniforms = Uniforms {
            // Pixel-to-NDC mapping covers the *whole window*, not just the timeline.
            // Inspector instances live in [timeline_w, window_w] and would otherwise
            // map to NDC.x > 1 and be clipped.
            viewport_size_px: [self.window_w_px, self.viewport.size_px.1],
            hovered: self.hovered.unwrap_or(u32::MAX),
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.prepare_text();

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.078,
                            g: 0.086,
                            b: 0.105,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if self.instance_count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
                pass.draw(0..6, 0..self.instance_count);
            }

            if let Err(e) = self.text_renderer.render(
                &self.text_atlas,
                &self.glyphon_viewport,
                &mut pass,
            ) {
                log::warn!("glyphon render failed: {e:?}");
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        self.text_atlas.trim();
    }
}

/// Y-coordinate where the tab content rows begin. Must agree with `prepare_text`.
#[allow(dead_code)]
fn compute_hotspot_section_y(has_selection: bool, _tab: MainTab) -> f32 {
    let pad = INSPECTOR_PADDING_PX;
    let line = INSPECTOR_LINE_HEIGHT_PX;
    let mut y = pad;
    y += TAB_HEIGHT_PX + SECTION_GAP_PX;
    if has_selection {
        y += line; // SELECTED heading
        y += INSPECTOR_TITLE_LINE_HEIGHT; // name
        y += line; // meta
        y += SECTION_GAP_PX;
        y += line; // STACK heading
        y += line * 8.0; // up to 8 stack lines
        y += SECTION_GAP_PX;
    }
    y += line; // tab heading
    y += 6.0;
    y
}

/// Hit-test the inspector's hotspot rows. Returns the slice index to jump to.
fn hit_test_hotspot_rect(
    rects: &[InspectorHotspotRect],
    cursor_x: f32,
    cursor_y: f32,
) -> Option<u32> {
    for r in rects {
        let [x, y, w, h] = r.rect;
        if cursor_x >= x && cursor_x < x + w && cursor_y >= y && cursor_y < y + h {
            return Some(r.slice_idx);
        }
    }
    None
}

/// For each slice whose function name matches `target`, find its parent on the
/// same track at depth-1 (caller) and its direct children at depth+1
/// (callees). Aggregate by name; returns lists sorted desc by total time.
fn compute_sandwich(profile: &Profile, target: StringId) -> CallerCalleeAggregate {
    // (total, count, exemplar_idx, exemplar_dur)
    let mut callers: AHashMap<StringId, (u64, u32, u32, u64)> = AHashMap::new();
    let mut callees: AHashMap<StringId, (u64, u32, u32, u64)> = AHashMap::new();
    for i in 0..profile.slices.len() {
        if profile.slices.name[i] != target {
            continue;
        }
        let track = profile.slices.track[i];
        let depth = profile.slices.depth[i];
        let s = profile.slices.start_ns[i];
        let dur = profile.slices.dur_ns[i];
        let e = s + dur;

        // Parent: shallower slice on same track containing `s`.
        if depth > 0 {
            let row = profile.slices.visible_in_row(track, depth - 1, s, s + 1);
            for r in row.start..row.end {
                let ps = profile.slices.start_ns[r as usize];
                let pe = ps + profile.slices.dur_ns[r as usize];
                if ps <= s && s < pe {
                    let pname = profile.slices.name[r as usize];
                    let entry = callers.entry(pname).or_insert((0, 0, r, 0));
                    entry.0 += dur;
                    entry.1 += 1;
                    if dur > entry.3 {
                        entry.2 = r;
                        entry.3 = dur;
                    }
                    break;
                }
            }
        }

        // Callees: direct children of this slice.
        let row = profile.slices.visible_in_row(track, depth + 1, s, e);
        for j in row.start..row.end {
            let cs = profile.slices.start_ns[j as usize];
            let cd = profile.slices.dur_ns[j as usize];
            if cs >= s && cs + cd <= e {
                let cname = profile.slices.name[j as usize];
                let entry = callees.entry(cname).or_insert((0, 0, j, 0));
                entry.0 += cd;
                entry.1 += 1;
                if cd > entry.3 {
                    entry.2 = j;
                    entry.3 = cd;
                }
            }
        }
    }
    let to_list = |m: AHashMap<StringId, (u64, u32, u32, u64)>| {
        let mut v: Vec<(StringId, u64, u32, u32)> = m
            .into_iter()
            .map(|(n, (total, count, ex, _))| (n, total, count, ex))
            .collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        v
    };
    CallerCalleeAggregate { callers: to_list(callers), callees: to_list(callees) }
}

/// Build a top-down call tree across all tracks. Same-named siblings under the
/// same parent merge into one node. Root level merges depth-0 slices from
/// every track that share a name (so a thread named `start_thread` and a
/// process root named `start_thread` show up as one node).
fn build_call_tree(profile: &Profile, self_dur: &[u64]) -> CallTree {
    use std::collections::hash_map::Entry;

    let mut nodes: Vec<CallTreeNode> = Vec::new();

    // Each frame on the recursion stack is the work-list for one (parent, depth)
    // pair: (parent_node_idx, depth, slice_indices_to_group).
    fn build_level(
        profile: &Profile,
        self_dur: &[u64],
        parent: u32,
        depth: u16,
        slice_indices: &[u32],
        nodes: &mut Vec<CallTreeNode>,
    ) -> Vec<u32> {
        let mut by_name: AHashMap<StringId, u32> = AHashMap::new();
        let mut order: Vec<u32> = Vec::new();
        // Stash this level's children-to-recurse-on grouped by which node they belong to.
        let mut kids_by_node: AHashMap<u32, Vec<u32>> = AHashMap::new();

        for &sidx in slice_indices {
            let i = sidx as usize;
            let name = profile.slices.name[i];
            let total = profile.slices.dur_ns[i];
            let self_t = self_dur.get(i).copied().unwrap_or(0);

            let node_idx = match by_name.entry(name) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let new_idx = nodes.len() as u32;
                    nodes.push(CallTreeNode {
                        name,
                        total_ns: 0,
                        self_ns: 0,
                        count: 0,
                        depth,
                        children: Vec::new(),
                        parent,
                        exemplar_slice_idx: sidx,
                    });
                    e.insert(new_idx);
                    order.push(new_idx);
                    new_idx
                }
            };
            {
                let n = &mut nodes[node_idx as usize];
                n.total_ns += total;
                n.self_ns += self_t;
                n.count += 1;
                let cur_ex_dur = profile.slices.dur_ns[n.exemplar_slice_idx as usize];
                if total > cur_ex_dur {
                    n.exemplar_slice_idx = sidx;
                }
            }

            // Direct children of this slice on the same track.
            let track = profile.slices.track[i];
            let s = profile.slices.start_ns[i];
            let e = s + total;
            let slice_depth = profile.slices.depth[i];
            let row = profile.slices.visible_in_row(track, slice_depth + 1, s, e);
            let pool = kids_by_node.entry(node_idx).or_default();
            for c in row.start..row.end {
                let cs = profile.slices.start_ns[c as usize];
                let cd = profile.slices.dur_ns[c as usize];
                if cs >= s && cs + cd <= e {
                    pool.push(c);
                }
            }
        }

        for &node_idx in &order {
            let kids = kids_by_node.remove(&node_idx).unwrap_or_default();
            if !kids.is_empty() {
                let cn = build_level(profile, self_dur, node_idx, depth + 1, &kids, nodes);
                nodes[node_idx as usize].children = cn;
            }
        }

        order
    }

    // Collect all depth-0 slices across every track.
    let mut roots_input: Vec<u32> = Vec::new();
    for t in 0..profile.tracks.len() {
        let track = TrackId(t as u32);
        let row_0 = profile
            .slices
            .rows
            .get(&(track, 0))
            .cloned()
            .unwrap_or(0..0);
        for r in row_0.start..row_0.end {
            roots_input.push(r);
        }
    }

    let roots = build_level(profile, self_dur, u32::MAX, 0, &roots_input, &mut nodes);
    // Sort roots by total desc so the "hot" path is at the top.
    let mut roots = roots;
    roots.sort_by_key(|&i| std::cmp::Reverse(nodes[i as usize].total_ns));
    // Same for every node's children — recursive sort is cheap (one pass).
    for n in 0..nodes.len() {
        let mut kids = std::mem::take(&mut nodes[n].children);
        kids.sort_by_key(|&i| std::cmp::Reverse(nodes[i as usize].total_ns));
        nodes[n].children = kids;
    }
    let total_ns: u64 = roots.iter().map(|&i| nodes[i as usize].total_ns).sum();
    CallTree { nodes, roots, total_ns }
}

/// Push a 1-pixel-wide bucket instance representing one or more sub-pixel
/// slices that all share a pixel column. Indexed back to a real SoA slice so
/// hover/click still work (the largest-duration slice in the bucket wins).
fn push_subpx_instance(
    instances: &mut Vec<SliceInstance>,
    slice_indices: &mut Vec<u32>,
    slice_idx: u32,
    x_px: f32,
    y_px: f32,
    row_h: f32,
    color: [f32; 4],
    selected: bool,
) {
    let inst_id = instances.len() as u32;
    let flags: u32 = if selected { 2 } else { 0 };
    instances.push(SliceInstance {
        rect_px: [x_px, y_px, 1.0, row_h - 1.0],
        color,
        instance_id: inst_id,
        flags,
        _pad: [0; 2],
    });
    slice_indices.push(slice_idx);
}

struct AggNode {
    frame: StringId,
    category: flame_core::CategoryId,
    total: u64,
    children: Vec<usize>,
}

/// Build a left-heavy / aggregated SliceTable from the time-ordered profile.
/// For each track, identical (parent_chain → frame) paths collapse into one
/// wide bar; siblings sort by total duration desc. The new x-axis represents
/// total time spent (0..track_total), not wall time.
fn build_left_heavy_layout(
    profile: &Profile,
) -> (flame_core::SliceTable, (u64, u64), Vec<u16>) {
    let n_tracks = profile.tracks.len();
    let mut row_counts = vec![0u16; n_tracks];
    let mut max_total: u64 = 0;

    let mut out_track: Vec<TrackId> = Vec::new();
    let mut out_depth: Vec<u16> = Vec::new();
    let mut out_start: Vec<u64> = Vec::new();
    let mut out_dur: Vec<u64> = Vec::new();
    let mut out_name: Vec<StringId> = Vec::new();
    let mut out_cat: Vec<flame_core::CategoryId> = Vec::new();

    for t_idx in 0..n_tracks {
        let track_id = TrackId(t_idx as u32);
        let mut nodes: Vec<AggNode> = Vec::new();
        let row_0 = profile
            .slices
            .rows
            .get(&(track_id, 0))
            .cloned()
            .unwrap_or(0..0);
        let roots: Vec<u32> = (row_0.start..row_0.end).collect();
        let root_nodes = aggregate_children_lh(profile, track_id, 0, &roots, &mut nodes);

        let mut max_depth = 0u16;
        let total = lay_out_lh(
            &nodes,
            &root_nodes,
            track_id,
            0,
            0,
            &mut out_track,
            &mut out_depth,
            &mut out_start,
            &mut out_dur,
            &mut out_name,
            &mut out_cat,
            &mut max_depth,
        );
        row_counts[t_idx] = if root_nodes.is_empty() { 0 } else { max_depth + 1 };
        max_total = max_total.max(total);
    }

    // Sort all collected slices into the canonical (track, depth, start) order
    // and build the SliceTable + rows index.
    let n = out_start.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by_key(|&i| (out_track[i].0, out_depth[i], out_start[i]));
    let track = idx.iter().map(|&i| out_track[i]).collect::<Vec<_>>();
    let depth = idx.iter().map(|&i| out_depth[i]).collect::<Vec<_>>();
    let start_ns = idx.iter().map(|&i| out_start[i]).collect::<Vec<_>>();
    let dur_ns = idx.iter().map(|&i| out_dur[i]).collect::<Vec<_>>();
    let name = idx.iter().map(|&i| out_name[i]).collect::<Vec<_>>();
    let category = idx.iter().map(|&i| out_cat[i]).collect::<Vec<_>>();
    let stack = vec![None; n];

    let mut rows: AHashMap<(TrackId, u16), std::ops::Range<u32>> = AHashMap::new();
    let mut row_start: u32 = 0;
    let mut cur_key: Option<(TrackId, u16)> = None;
    for i in 0..n {
        let key = (track[i], depth[i]);
        if Some(key) != cur_key {
            if let Some(prev) = cur_key {
                rows.insert(prev, row_start..i as u32);
            }
            row_start = i as u32;
            cur_key = Some(key);
        }
    }
    if let Some(prev) = cur_key {
        rows.insert(prev, row_start..n as u32);
    }

    let table = flame_core::SliceTable {
        track,
        depth,
        start_ns,
        dur_ns,
        name,
        category,
        stack,
        rows,
    };
    (table, (0, max_total), row_counts)
}

fn aggregate_children_lh(
    profile: &Profile,
    track_id: TrackId,
    depth: u16,
    slice_indices: &[u32],
    nodes: &mut Vec<AggNode>,
) -> Vec<usize> {
    let mut by_frame: AHashMap<StringId, usize> = AHashMap::new();
    let mut order: Vec<usize> = Vec::new();
    let mut children_per_frame: AHashMap<StringId, Vec<u32>> = AHashMap::new();

    for &idx in slice_indices {
        let i = idx as usize;
        let frame = profile.slices.name[i];
        let dur = profile.slices.dur_ns[i];
        let cat = profile.slices.category[i];

        let node_idx = match by_frame.entry(frame) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let new_idx = nodes.len();
                nodes.push(AggNode {
                    frame,
                    category: cat,
                    total: 0,
                    children: Vec::new(),
                });
                e.insert(new_idx);
                order.push(new_idx);
                new_idx
            }
        };
        nodes[node_idx].total += dur;

        // Gather direct children of this slice into the merged children pool.
        let s = profile.slices.start_ns[i];
        let end = s + dur;
        let row = profile.slices.visible_in_row(track_id, depth + 1, s, end);
        let cv = children_per_frame.entry(frame).or_default();
        for c in row.start..row.end {
            let cs = profile.slices.start_ns[c as usize];
            let cd = profile.slices.dur_ns[c as usize];
            if cs >= s && cs + cd <= end {
                cv.push(c);
            }
        }
    }

    // Recurse on each unique frame's merged children.
    for &node_idx in &order {
        let frame = nodes[node_idx].frame;
        let kids = children_per_frame.remove(&frame).unwrap_or_default();
        if !kids.is_empty() {
            let cn = aggregate_children_lh(profile, track_id, depth + 1, &kids, nodes);
            nodes[node_idx].children = cn;
        }
    }
    order
}

fn lay_out_lh(
    nodes: &[AggNode],
    roots: &[usize],
    track: TrackId,
    x_offset: u64,
    depth: u16,
    out_track: &mut Vec<TrackId>,
    out_depth: &mut Vec<u16>,
    out_start: &mut Vec<u64>,
    out_dur: &mut Vec<u64>,
    out_name: &mut Vec<StringId>,
    out_cat: &mut Vec<flame_core::CategoryId>,
    max_depth: &mut u16,
) -> u64 {
    let mut sorted = roots.to_vec();
    sorted.sort_by_key(|&i| std::cmp::Reverse(nodes[i].total));
    let mut x = x_offset;
    for &i in &sorted {
        let n = &nodes[i];
        out_track.push(track);
        out_depth.push(depth);
        out_start.push(x);
        out_dur.push(n.total);
        out_name.push(n.frame);
        out_cat.push(n.category);
        *max_depth = (*max_depth).max(depth);
        lay_out_lh(
            nodes,
            &n.children,
            track,
            x,
            depth + 1,
            out_track,
            out_depth,
            out_start,
            out_dur,
            out_name,
            out_cat,
            max_depth,
        );
        x += n.total;
    }
    x - x_offset
}

/// Compute per-slice self-time: `dur - sum(direct-children dur)`. Children are
/// slices on the same track at depth+1 whose interval lies inside this slice's
/// interval. O(N log N) over the SoA, runs once on profile load.
fn compute_self_dur(profile: &Profile) -> Vec<u64> {
    let n = profile.slices.len();
    let mut self_dur = vec![0u64; n];
    for i in 0..n {
        let track = profile.slices.track[i];
        let depth = profile.slices.depth[i];
        let s = profile.slices.start_ns[i];
        let dur = profile.slices.dur_ns[i];
        let e = s + dur;
        let row = profile.slices.visible_in_row(track, depth + 1, s, e);
        let mut children_total: u64 = 0;
        for j in row.start..row.end {
            let cs = profile.slices.start_ns[j as usize];
            let cd = profile.slices.dur_ns[j as usize];
            // Restrict to children fully inside [s, e). visible_in_row may
            // include a slice whose interval just-barely-touches.
            if cs >= s && cs + cd <= e {
                children_total += cd;
            }
        }
        self_dur[i] = dur.saturating_sub(children_total);
    }
    self_dur
}

/// Aggregate slices by function name, optionally sorting by self or total.
/// All variants share the same per-slice scan; sorting is what differs.
fn aggregate_functions(
    profile: &Profile,
    self_dur: &[u64],
) -> AHashMap<StringId, (u64, u64, u32, u32, u64)> {
    // (total_dur, self_dur, count, exemplar_idx, exemplar_dur)
    let mut by_name: AHashMap<StringId, (u64, u64, u32, u32, u64)> = AHashMap::new();
    for i in 0..profile.slices.len() {
        let name = profile.slices.name[i];
        let dur = profile.slices.dur_ns[i];
        let s_dur = self_dur.get(i).copied().unwrap_or(0);
        let entry = by_name.entry(name).or_insert((0, 0, 0, i as u32, 0));
        entry.0 += dur;
        entry.1 += s_dur;
        entry.2 += 1;
        if dur > entry.4 {
            entry.3 = i as u32;
            entry.4 = dur;
        }
    }
    by_name
}

fn sort_functions_desc_by(
    by_name: &AHashMap<StringId, (u64, u64, u32, u32, u64)>,
    by_self: bool,
) -> Vec<HotspotEntry> {
    let mut v: Vec<HotspotEntry> = by_name
        .iter()
        .map(|(&name, &(total, self_d, count, ex_idx, _))| HotspotEntry {
            name,
            total_dur_ns: total,
            self_dur_ns: self_d,
            count,
            exemplar_slice_idx: ex_idx,
        })
        .collect();
    if by_self {
        v.sort_by_key(|h| std::cmp::Reverse(h.self_dur_ns));
    } else {
        v.sort_by_key(|h| std::cmp::Reverse(h.total_dur_ns));
    }
    v
}

/// Compute the half-open `[start, end)` range of slice depths whose row
/// rectangles intersect the screen-y band `[lo_y, hi_y)`. `track_content_top`
/// is the y of the first row in canvas-after-scroll space.
fn visible_depth_range(
    direction: Direction,
    rows: u16,
    track_content_top: f32,
    row_h: f32,
    lo_y: f32,
    hi_y: f32,
) -> (u16, u16) {
    if rows == 0 || row_h <= 0.0 {
        return (0, 0);
    }
    // For a given depth d, the row's screen y is:
    //   Icicle:  track_content_top + d * row_h
    //   Flame:   track_content_top + (rows - 1 - d) * row_h
    // Solve for d such that y < hi_y && y + row_h > lo_y.
    let rows_i = rows as i32;
    match direction {
        Direction::Icicle => {
            let lo_d = ((lo_y - track_content_top) / row_h).floor() as i32;
            let hi_d = ((hi_y - track_content_top) / row_h).ceil() as i32 + 1;
            let start = lo_d.clamp(0, rows_i) as u16;
            let end = hi_d.clamp(0, rows_i) as u16;
            (start, end)
        }
        Direction::Flame => {
            // Substitute d' = rows - 1 - d into the icicle solution.
            let lo_dp = ((lo_y - track_content_top) / row_h).floor() as i32;
            let hi_dp = ((hi_y - track_content_top) / row_h).ceil() as i32 + 1;
            let dp_start = lo_dp.clamp(0, rows_i);
            let dp_end = hi_dp.clamp(0, rows_i);
            // d range is reversed: [rows - dp_end, rows - dp_start)
            let start = (rows_i - dp_end).max(0) as u16;
            let end = (rows_i - dp_start).max(0) as u16;
            (start, end)
        }
    }
}

fn format_duration(ns: u64) -> String {
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.2} µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.3} s", ns as f64 / 1_000_000_000.0)
    }
}
