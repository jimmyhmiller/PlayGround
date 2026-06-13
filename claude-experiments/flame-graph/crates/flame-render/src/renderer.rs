use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use flame_core::{Profile, SliceTable, StringId, Track, TrackId, TrackKind};
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
    /// Distributed-tracing sequence diagram view: per-trace, lifelines per
    /// participant (e.g. service), activation boxes per span, arrows for
    /// cross-participant hops.
    Sequence,
}

impl MainTab {
    pub const ALL: &'static [MainTab] = &[
        MainTab::Flame,
        MainTab::CallTree,
        MainTab::Hotspots,
        MainTab::BottomUp,
        MainTab::Table,
        MainTab::Sequence,
    ];
    pub fn label(self) -> &'static str {
        match self {
            MainTab::Flame => "FLAME",
            MainTab::CallTree => "CALL TREE",
            MainTab::Hotspots => "HOTSPOTS",
            MainTab::BottomUp => "BOTTOM-UP",
            MainTab::Table => "ALL FUNCTIONS",
            MainTab::Sequence => "SEQUENCE",
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

/// Whether to display each source track separately or merge them all onto a
/// single synthetic track (greedy row-pack so colors-by-category become bands).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MergeMode {
    /// One row strip per source track (default).
    Multi,
    /// All slices on a single synthetic track; depth assigned by greedy
    /// first-fit so no two slices overlap in a row.
    Single,
}

impl MergeMode {
    pub fn label(self) -> &'static str {
        match self {
            MergeMode::Multi => "multi",
            MergeMode::Single => "single",
        }
    }
}

/// Tabs inside the persistent right-side panel (visible on the FLAME view).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SidebarTab {
    /// Selected-slice details + stack + sandwich callers/callees. Default.
    Inspect,
    /// Picker for the timeline grouping attribute key. Lists `(none)` plus
    /// every key the loaded profile carries.
    Group,
}

impl SidebarTab {
    pub const ALL: &'static [SidebarTab] = &[SidebarTab::Inspect, SidebarTab::Group];
    pub fn label(self) -> &'static str {
        match self {
            SidebarTab::Inspect => "INSPECT",
            SidebarTab::Group => "GROUP",
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
/// Height of the per-sidebar tab strip (GROUP / INSPECT). Sits below the
/// global tab bar at the top of the right inspector area.
pub const SIDEBAR_TAB_BAR_H: f32 = 44.0;

/// SEQUENCE tab: width of the left-side trace picker.
pub const SEQUENCE_PICKER_WIDTH_PX: f32 = 360.0;
/// SEQUENCE tab: height of the participant lifeline header band.
pub const SEQUENCE_HEADER_H: f32 = 56.0;
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
    /// Color format of the texture views we'll render into. Used to recreate
    /// pipelines (and would be needed if we ever rebuilt the glyphon atlas).
    pub target_format: wgpu::TextureFormat,
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
    /// When `Some`, overrides `status_text` in the status bar. Used by live
    /// mode to surface samples/rate/elapsed without competing with the
    /// hover / selection messages that `refresh_status_for_*` writes.
    pub live_status: Option<String>,
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
    /// Track-merge mode: render each source track separately or merge into one.
    pub merge_mode: MergeMode,
    /// The profile as loaded — preserved so we can swap back from merged.
    pub original_profile: Option<Arc<Profile>>,
    /// Single-track version of the original. Built lazily on first toggle.
    pub merged_profile: Option<Arc<Profile>>,
    /// Current attribute key the timeline is grouped by, or None for no grouping.
    /// Mutually exclusive with `merge_mode == Single`.
    pub group_key: Option<StringId>,
    /// Cache of grouped profiles by key. Built lazily on first selection of
    /// each key so repeated toggling is cheap.
    pub grouped_profiles: AHashMap<StringId, Arc<Profile>>,
    /// Rows of the GROUP sidebar tab's key list. (None, _) is the "(none)" reset row.
    pub group_picker_row_rects: Vec<(Option<StringId>, [f32; 4])>,
    /// Active sidebar tab (GROUP / INSPECT). Only matters when MainTab::Flame.
    pub sidebar_tab: SidebarTab,
    /// Sidebar tab strip button rects for click hit-test.
    pub sidebar_tab_rects: Vec<(SidebarTab, [f32; 4])>,
    /// Vertical scroll offset (in px) for the sidebar body content. The tab
    /// strip itself stays fixed; only what's below the strip moves.
    pub sidebar_scroll_y_px: f32,
    /// Total height of the sidebar body content emitted last frame. Used to
    /// clamp scroll so the user can't pan past the bottom.
    pub sidebar_content_h_px: f32,

    // ───── SEQUENCE tab state ─────
    /// Attribute key whose distinct values define the participant lifelines.
    /// Defaults to "service" if present, else first available key.
    pub sequence_lifeline_key: Option<StringId>,
    /// Vertical scroll offset (px) for the sequence diagram body.
    pub sequence_scroll_y_px: f32,
    /// Time-axis zoom for the sequence diagram. 1.0 fits the whole capture
    /// into the body height; larger expands the time axis so dense bursts of
    /// activity spread out and become legible. Anchored on zoom via
    /// [`Renderer::zoom_sequence`].
    pub sequence_time_zoom: f32,
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
    /// Build a renderer that draws into externally-supplied wgpu resources.
    ///
    /// The caller owns the swapchain (or render target). On each frame the
    /// caller must invoke [`Renderer::render`] with a `TextureView` whose
    /// underlying texture format matches `target_format`.
    ///
    /// `size_px` is the initial pixel size of the canvas the renderer will
    /// draw into. Call [`Renderer::resize`] whenever that changes.
    ///
    /// `device` and `queue` are cloned (both are cheap `Arc`-backed handles in
    /// wgpu), so the caller can keep using them too.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        size_px: (u32, u32),
    ) -> Self {
        let device = device.clone();
        let queue = queue.clone();
        let format = target_format;
        struct Size { width: u32, height: u32 }
        let size = Size { width: size_px.0.max(1), height: size_px.1.max(1) };

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
            target_format: format,
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
            live_status: None,
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
            merge_mode: MergeMode::Multi,
            original_profile: None,
            merged_profile: None,
            group_key: None,
            grouped_profiles: AHashMap::new(),
            group_picker_row_rects: Vec::new(),
            sidebar_tab: SidebarTab::Inspect,
            sidebar_tab_rects: Vec::new(),
            sidebar_scroll_y_px: 0.0,
            sidebar_content_h_px: 0.0,
            sequence_lifeline_key: None,
            sequence_scroll_y_px: 0.0,
            sequence_time_zoom: 1.0,
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

    /// Replace the current profile while preserving every bit of user
    /// interaction state we know how to re-resolve in the new one. Use this
    /// for live-streaming snapshot refreshes; use [`Renderer::set_profile`]
    /// for fresh trace loads where you want the viewport/selection reset.
    ///
    /// Identifies things by *name* rather than index so the swap survives the
    /// fact that snapshot indices are rebuilt from scratch each tick:
    /// - collapsed tracks → by track name
    /// - selected slice → by (track name + root-to-leaf frame name path)
    /// - group key / sequence lifeline key → by string, re-interned
    /// - merge_mode and layout_mode are kept verbatim; the merged profile
    ///   (if active) is dropped so the next access re-derives it.
    /// - hovered is cleared; the next mouse move re-hit-tests it for free.
    pub fn set_profile_live(&mut self, profile: Arc<Profile>) {
        let saved = self.capture_live_state();
        self.set_profile(profile);
        self.restore_live_state(saved);
    }

    fn capture_live_state(&self) -> LiveState {
        let strings_lookup = |id: StringId| -> String {
            self.profile
                .as_ref()
                .map(|p| p.strings.get(id).to_string())
                .unwrap_or_default()
        };
        let collapsed_track_names: Vec<String> = self
            .profile
            .as_ref()
            .map(|p| {
                self.collapsed_tracks
                    .iter()
                    .filter_map(|t| {
                        p.tracks
                            .get(t.0 as usize)
                            .map(|tr| p.strings.get(tr.name).to_string())
                    })
                    .collect()
            })
            .unwrap_or_default();

        let selected = self
            .selected_slice
            .and_then(|idx| self.profile.as_ref().map(|p| (p, idx)))
            .and_then(|(p, idx)| slice_identity(p, idx));

        LiveState {
            viewport: self.viewport.clone(),
            active_tab: self.active_tab,
            sidebar_tab: self.sidebar_tab,
            layout_mode: self.layout_mode,
            merge_mode: self.merge_mode,
            direction: self.direction,
            group_key_name: self.group_key.map(strings_lookup),
            sequence_lifeline_key_name: self.sequence_lifeline_key.map(strings_lookup),
            collapsed_track_names,
            selected,
            sidebar_scroll_y_px: self.sidebar_scroll_y_px,
            sequence_scroll_y_px: self.sequence_scroll_y_px,
            sequence_time_zoom: self.sequence_time_zoom,
            call_tree_scroll_y_px: self.call_tree_scroll_y_px,
        }
    }

    fn restore_live_state(&mut self, s: LiveState) {
        self.set_tab(s.active_tab);
        self.sidebar_tab = s.sidebar_tab;
        self.layout_mode = s.layout_mode;
        self.direction = s.direction;
        self.sidebar_scroll_y_px = s.sidebar_scroll_y_px;
        self.sequence_scroll_y_px = s.sequence_scroll_y_px;
        self.sequence_time_zoom = if s.sequence_time_zoom > 0.0 {
            s.sequence_time_zoom
        } else {
            1.0
        };
        self.call_tree_scroll_y_px = s.call_tree_scroll_y_px;
        // Preserve vertical scroll and row height. Do NOT restore the
        // horizontal viewport (start_ns / ns_per_pixel) — in live mode the
        // trace's time range grows constantly, and pinning to the old
        // coordinates strands the user looking at a stale window while new
        // data goes off-screen to the right. Always re-fit horizontally to
        // the active layout's full range.
        self.viewport.scroll_y_px = s.viewport.scroll_y_px;
        self.viewport.row_height_px = s.viewport.row_height_px;
        let (rstart, rend) = self.current_time_range();
        if rend > rstart {
            self.viewport.fit_time(rstart, rend);
        }

        // merge_mode: setting it via set_merge_mode would rebuild merged
        // profile here; we let it rebuild lazily on the next access.
        self.merge_mode = s.merge_mode;
        self.merged_profile = None;
        self.grouped_profiles.clear();

        if let Some(p) = self.profile.clone() {
            if let Some(name) = &s.group_key_name {
                self.group_key = p.strings.lookup(name);
            }
            if let Some(name) = &s.sequence_lifeline_key_name {
                self.sequence_lifeline_key = p.strings.lookup(name);
            }
            // Re-resolve collapsed track names → new TrackIds.
            let mut wanted: AHashSet<TrackId> = AHashSet::new();
            for (i, t) in p.tracks.iter().enumerate() {
                let name = p.strings.get(t.name);
                if s.collapsed_track_names.iter().any(|n| n == name) {
                    wanted.insert(TrackId(i as u32));
                }
            }
            self.collapsed_tracks = wanted;
            // Re-resolve selected slice by name path.
            if let Some(identity) = &s.selected {
                self.selected_slice = find_slice_by_identity(&p, identity);
            }
        }

        self.clamp_viewport();
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
        self.original_profile = Some(profile.clone());
        // Drop any stale merged / grouped profiles from a previous load.
        self.merged_profile = None;
        self.grouped_profiles.clear();
        self.merge_mode = MergeMode::Multi;
        self.group_key = None;
        self.sidebar_scroll_y_px = 0.0;
        self.sidebar_content_h_px = 0.0;
        self.sequence_scroll_y_px = 0.0;
        self.sequence_time_zoom = 1.0;
        // Default the SEQUENCE lifeline key to "service" if present, else
        // first available attr key alphabetically.
        self.sequence_lifeline_key = profile
            .strings
            .lookup("service")
            .filter(|k| profile.attrs.key_lookup.contains_key(k))
            .or_else(|| {
                let mut keys: Vec<(StringId, String)> = profile
                    .attrs
                    .keys
                    .iter()
                    .map(|&sid| (sid, profile.strings.get(sid).to_string()))
                    .collect();
                keys.sort_by(|a, b| a.1.cmp(&b.1));
                keys.first().map(|(sid, _)| *sid)
            });
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

    /// Switch between multi-track and single-track display. The merged profile
    /// is built lazily and cached. Caller must follow with `rebuild_instances`
    /// + redraw.
    pub fn set_merge_mode(&mut self, mode: MergeMode) {
        if self.merge_mode == mode {
            return;
        }
        self.merge_mode = mode;
        let target = match mode {
            MergeMode::Multi => self.original_profile.clone(),
            MergeMode::Single => {
                if self.merged_profile.is_none() {
                    if let Some(orig) = &self.original_profile {
                        let built = build_single_track_profile(orig);
                        self.merged_profile = Some(Arc::new(built));
                    }
                }
                self.merged_profile.clone()
            }
        };
        if let Some(p) = target {
            // Track IDs change between modes — drop selection / hover / collapse
            // state to avoid dangling references.
            self.selected_slice = None;
            self.hovered = None;
            self.collapsed_tracks.clear();
            // Aggregated (left-heavy) layout is built per-track; rebuild it for
            // the new track shape so the AGGREGATED toggle keeps working.
            let (agg_slices, agg_range, agg_rows) = build_left_heavy_layout(&p);
            self.aggregated_slices = Some(Arc::new(agg_slices));
            self.aggregated_time_range = agg_range;
            self.aggregated_track_rows = agg_rows;
            let (s, e) = p.time_range;
            self.profile = Some(p);
            self.viewport.fit_time(s, e);
            self.viewport.row_height_px = ROW_HEIGHT_PX;
            self.viewport.scroll_y_px = 0.0;
            self.clamp_viewport();
        }
    }

    pub fn toggle_merge_mode(&mut self) {
        let next = match self.merge_mode {
            MergeMode::Multi => MergeMode::Single,
            MergeMode::Single => MergeMode::Multi,
        };
        self.set_merge_mode(next);
    }

    /// Group the timeline by an attribute key (one track per distinct value),
    /// or `None` to clear grouping and restore the original per-trace layout.
    /// The key is a `StringId` interned in the *original* profile's string
    /// table (look it up via `original_profile.strings`).
    ///
    /// Grouping and single-track merge are mutually exclusive — selecting a
    /// group key flips merge mode back to Multi.
    pub fn set_group_key(&mut self, key: Option<StringId>) {
        if self.group_key == key && self.merge_mode == MergeMode::Multi {
            return;
        }
        self.group_key = key;
        self.merge_mode = MergeMode::Multi;
        let target = match key {
            None => self.original_profile.clone(),
            Some(k) => {
                if !self.grouped_profiles.contains_key(&k) {
                    if let Some(orig) = &self.original_profile {
                        let built = build_grouped_profile(orig, k);
                        self.grouped_profiles.insert(k, Arc::new(built));
                    }
                }
                self.grouped_profiles.get(&k).cloned()
            }
        };
        if let Some(p) = target {
            // Track IDs change between groupings — drop selection / hover /
            // collapse state so they don't dangle.
            self.selected_slice = None;
            self.hovered = None;
            self.collapsed_tracks.clear();
            let (agg_slices, agg_range, agg_rows) = build_left_heavy_layout(&p);
            self.aggregated_slices = Some(Arc::new(agg_slices));
            self.aggregated_time_range = agg_range;
            self.aggregated_track_rows = agg_rows;
            let (s, e) = p.time_range;
            self.profile = Some(p);
            self.viewport.fit_time(s, e);
            self.viewport.row_height_px = ROW_HEIGHT_PX;
            self.viewport.scroll_y_px = 0.0;
            self.clamp_viewport();
        }
    }

    /// Sorted (by string) list of attribute keys available for grouping in the
    /// *original* profile, with the key's StringId and human-readable name.
    pub fn available_group_keys(&self) -> Vec<(StringId, String)> {
        let Some(orig) = &self.original_profile else { return Vec::new() };
        let mut keys: Vec<(StringId, String)> = orig
            .attrs
            .keys
            .iter()
            .map(|&sid| (sid, orig.strings.get(sid).to_string()))
            .collect();
        keys.sort_by(|a, b| a.1.cmp(&b.1));
        keys
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

    /// Hit-test the sidebar tab strip (GROUP / INSPECT) at the top of the
    /// right inspector panel.
    pub fn hit_test_sidebar_tab(&self, x: f32, y: f32) -> Option<SidebarTab> {
        for (tab, rect) in &self.sidebar_tab_rects {
            if x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3] {
                return Some(*tab);
            }
        }
        None
    }

    /// Hit-test a row in the GROUP sidebar tab's key list. Returns the picked
    /// value: `Some(None)` = "(none)" row, `Some(Some(sid))` = a specific key.
    pub fn hit_test_group_row(&self, x: f32, y: f32) -> Option<Option<StringId>> {
        if self.sidebar_tab != SidebarTab::Group {
            return None;
        }
        for (val, rect) in &self.group_picker_row_rects {
            if x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3] {
                return Some(*val);
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
            // SEQUENCE also has its own custom rendering path.
            MainTab::Sequence => Vec::new(),
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
        let mut y = self.inspector_content_top() + pad;
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
        // ATTRIBUTES heading + per-attr lines (each attribute's value may wrap
        // across several lines; `attrs_section_lines` returns the total).
        let attr_lines = self.attrs_section_lines(sel);
        y += line * attr_lines as f32;
        y += SECTION_GAP_PX;
        Some(y)
    }

    /// Approximate number of wrapped lines a single attribute display takes
    /// in the inspector body font. Caps at 8 lines (rest gets clipped).
    pub fn attr_wrap_lines(&self, key: &str, value: &str) -> u16 {
        // INSPECTOR_BODY_FONT ≈ 26px; inspector content width ≈ 552px after
        // padding. Average glyph width in our sans-serif at that size is ~13px,
        // so ~42 chars per line. The leading "key: " counts toward line 1.
        const CHARS_PER_LINE: usize = 42;
        let total = key.len() + 2 + value.len(); // "key: value"
        let raw = ((total + CHARS_PER_LINE - 1) / CHARS_PER_LINE).max(1);
        raw.min(8) as u16
    }

    /// Total inspector lines the ATTRIBUTES section occupies for a slice
    /// (heading + one entry per attr × its wrap lines, or one "(none)" line).
    pub fn attrs_section_lines(&self, slice_idx: u32) -> u16 {
        let attrs = self.attrs_for_slice(slice_idx);
        let mut total: u16 = 1; // heading
        if attrs.is_empty() {
            total += 1; // "(none)"
        } else {
            for (k, v) in &attrs {
                total = total.saturating_add(self.attr_wrap_lines(k, v));
            }
        }
        total
    }

    /// Return the attribute (key, value) pairs for a slice, sorted by key.
    /// Empty vec if the slice has no attrs or the profile carries no attrs.
    pub fn attrs_for_slice(&self, slice_idx: u32) -> Vec<(String, String)> {
        let Some(profile) = &self.profile else { return Vec::new() };
        let Some(row) = profile.attrs.per_slice.get(slice_idx as usize) else {
            return Vec::new();
        };
        let mut out: Vec<(String, String)> = row
            .iter()
            .filter_map(|(k_idx, v_sid)| {
                let key_sid = profile.attrs.keys.get(*k_idx as usize).copied()?;
                Some((
                    profile.strings.get(key_sid).to_string(),
                    profile.strings.get(*v_sid).to_string(),
                ))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
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

        // Also reset the SEQUENCE time axis so 'a' / Home / Esc give a clean
        // fit on that tab too.
        self.sequence_scroll_y_px = 0.0;
        self.sequence_time_zoom = 1.0;
    }

    fn refresh_status_for_idle(&mut self) {
        let Some(p) = &self.profile else { return };
        let n = p.slices.len();
        let dur = format_duration(p.duration_ns());
        self.status_text = format!("{} slices · duration {}", n, dur);
    }

    /// Notify the renderer that the texture it draws into has changed size.
    /// The caller (which owns the swapchain / RTT texture) is responsible for
    /// reconfiguring that surface separately — the renderer just adjusts its
    /// internal layout math and the glyphon viewport.
    pub fn resize(&mut self, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
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

    /// Y where the right-sidebar *content* (below the per-sidebar tab strip)
    /// begins. Use this anywhere the inspector lays out body content.
    pub fn inspector_content_top(&self) -> f32 {
        TAB_BAR_HEIGHT_PX + SIDEBAR_TAB_BAR_H
    }

    /// Body geometry (top, height) of the SEQUENCE diagram for the current
    /// viewport. Shared by the emit pass and the pan/zoom handlers so scroll
    /// clamping and focal-point math stay consistent.
    fn sequence_body_geometry(&self) -> (f32, f32) {
        let top = TAB_BAR_HEIGHT_PX;
        let bot = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX;
        let body_top = top + SEQUENCE_HEADER_H;
        let body_bot = bot - 16.0;
        (body_top, (body_bot - body_top).max(1.0))
    }

    /// Max scroll for the SEQUENCE body at the current zoom.
    fn sequence_max_scroll(&self) -> f32 {
        let (_, body_h) = self.sequence_body_geometry();
        (body_h * self.sequence_time_zoom - body_h).max(0.0)
    }

    /// Vertical scroll on the SEQUENCE diagram. `dy > 0` moves content up.
    /// Clamped to the zoomed content height so you can't scroll into the void.
    pub fn pan_sequence(&mut self, dy: f32) {
        let max_scroll = self.sequence_max_scroll();
        self.sequence_scroll_y_px = (self.sequence_scroll_y_px + dy).clamp(0.0, max_scroll);
    }

    /// Zoom the SEQUENCE time axis around `focus_y` (screen px). `factor > 1`
    /// zooms in (spreads the time axis out), `< 1` zooms out. The time under
    /// `focus_y` is kept stationary so wheel / keyboard zoom feels anchored.
    pub fn zoom_sequence(&mut self, focus_y: f32, factor: f32) {
        let (body_top, body_h) = self.sequence_body_geometry();
        let old_zoom = self.sequence_time_zoom;
        let new_zoom = (old_zoom * factor).clamp(1.0, 5000.0);
        if (new_zoom - old_zoom).abs() < f32::EPSILON {
            return;
        }
        // Fraction of the whole capture currently sitting under the focus line.
        let old_content_h = body_h * old_zoom;
        let focus = (focus_y - body_top).max(0.0);
        let frac = ((focus + self.sequence_scroll_y_px) / old_content_h).clamp(0.0, 1.0);

        self.sequence_time_zoom = new_zoom;
        let new_content_h = body_h * new_zoom;
        let max_scroll = (new_content_h - body_h).max(0.0);
        self.sequence_scroll_y_px = (frac * new_content_h - focus).clamp(0.0, max_scroll);
    }

    pub fn set_sidebar_tab(&mut self, tab: SidebarTab) {
        if self.sidebar_tab != tab {
            // Reset scroll when switching tabs so the new tab starts at top.
            self.sidebar_scroll_y_px = 0.0;
        }
        self.sidebar_tab = tab;
    }

    /// Add `dy` pixels to the sidebar scroll offset, clamped to the content
    /// height. `dy > 0` scrolls toward the bottom (content moves up).
    pub fn pan_sidebar(&mut self, dy: f32) {
        let visible = (self.viewport.size_px.1 - self.inspector_content_top() - STATUS_BAR_HEIGHT_PX)
            .max(0.0);
        let max_scroll = (self.sidebar_content_h_px - visible).max(0.0);
        self.sidebar_scroll_y_px = (self.sidebar_scroll_y_px + dy).clamp(0.0, max_scroll);
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
        if self.active_tab == MainTab::Sequence {
            self.emit_sequence_content();
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
                    let cat_id = active_slices.category[idx];
                    let cat_name_id = profile.categories[cat_id.0 as usize].name;
                    let cat_name = profile.strings.get(cat_name_id);
                    // Formats that don't carry meaningful categories fall back
                    // to the bland built-in "default" — for those we use the
                    // classic name-based warm palette so each function still
                    // varies visually.
                    let color = if cat_name == "default" {
                        palette::color_for(name)
                    } else {
                        palette::color_for_blend(cat_name, name)
                    };

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

        // Right-sidebar body content (GROUP key list when GROUP tab is
        // active). INSPECT body content (SANDWICH row chrome) is emitted in
        // its own block further down. Both are scroll-shifted by
        // `sidebar_scroll_y_px` and clipped to the inspector body area; the
        // tab strip itself is emitted at the very end so it always masks
        // anything that scrolled into its y range.
        self.group_picker_row_rects.clear();
        self.sidebar_tab_rects.clear();
        let sidebar_scroll = self.sidebar_scroll_y_px;
        let body_top = self.inspector_content_top();
        let body_bot = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX;
        if self.active_tab == MainTab::Flame {
            // GROUP-tab content: list of (none) + every attribute key.
            if self.sidebar_tab == SidebarTab::Group {
                let pad = INSPECTOR_PADDING_PX;
                let row_h = 40.0_f32;
                let content_x = inspector_x + pad;
                let content_w = INSPECTOR_WIDTH_PX - pad * 2.0;
                // Natural (unscrolled) start; scroll applied per-row below so
                // we know which rows fall outside the visible band.
                let base_y = body_top + pad;
                let mut natural_y = base_y;

                let emit_row = |rect: [f32; 4],
                                selected: bool,
                                even: bool,
                                instances: &mut Vec<SliceInstance>,
                                indices: &mut Vec<u32>| {
                    let bg = if selected {
                        TOP_TAB_ACTIVE
                    } else if even {
                        ROW_COLOR_EVEN
                    } else {
                        ROW_COLOR_ODD
                    };
                    let inst_id = instances.len() as u32;
                    instances.push(SliceInstance {
                        rect_px: rect,
                        color: bg,
                        instance_id: inst_id,
                        flags: 1,
                        _pad: [0; 2],
                    });
                    indices.push(NON_SLICE_SENTINEL);
                };

                let keys = self.available_group_keys();
                let emit = |val: Option<StringId>,
                                row_idx: usize,
                                natural_y: f32,
                                instances: &mut Vec<SliceInstance>,
                                indices: &mut Vec<u32>,
                                rects: &mut Vec<(Option<StringId>, [f32; 4])>,
                                selected: bool| {
                    let y = natural_y - sidebar_scroll;
                    // Skip rows that are entirely above or below the visible
                    // body band — saves draw work and keeps rects from being
                    // hit-tested where they shouldn't be.
                    if y + row_h <= body_top || y >= body_bot {
                        return;
                    }
                    let rect = [content_x, y, content_w, row_h - 2.0];
                    emit_row(rect, selected, row_idx % 2 == 0, instances, indices);
                    rects.push((val, rect));
                };

                // "(none)" row.
                emit(
                    None,
                    0,
                    natural_y,
                    &mut self.instances,
                    &mut self.slice_indices,
                    &mut self.group_picker_row_rects,
                    self.group_key.is_none(),
                );
                natural_y += row_h;
                for (i, (sid, _name)) in keys.iter().enumerate() {
                    emit(
                        Some(*sid),
                        i + 1,
                        natural_y,
                        &mut self.instances,
                        &mut self.slice_indices,
                        &mut self.group_picker_row_rects,
                        self.group_key == Some(*sid),
                    );
                    natural_y += row_h;
                }
                self.sidebar_content_h_px = (natural_y - base_y).max(0.0);
            }
        }

        // SANDWICH section — only when a slice is selected. Renders below STACK.
        // Skipped entirely when the sidebar is on the GROUP tab so the key
        // list owns the inspector area.
        if self.active_tab == MainTab::Flame
            && self.sidebar_tab == SidebarTab::Inspect
        {
        if let Some(section_y) = self.sandwich_section_y() {
            let pad = INSPECTOR_PADDING_PX;
            let line = INSPECTOR_LINE_HEIGHT_PX;
            // Skip the SANDWICH heading line; chrome rows start below it.
            // `section_y` is in unscrolled (natural) coordinates.
            let mut natural_y = section_y + line + 4.0;
            let row_h = line * 1.4;
            let band_x = inspector_x + pad;
            let band_w = INSPECTOR_WIDTH_PX - pad * 2.0;

            if let Some(rows) = self.sandwich_rows(5) {
                for (i, (slice_idx, _)) in rows.iter().enumerate() {
                    let y = natural_y - sidebar_scroll;
                    natural_y += row_h;
                    // Skip rows entirely outside the visible sidebar body band.
                    if y + row_h <= body_top || y >= body_bot {
                        continue;
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
                }
            }
            // Total INSPECT body content height = how far the natural y went
            // past the body top.
            self.sidebar_content_h_px = (natural_y - body_top).max(0.0);
        } else {
            self.sidebar_content_h_px = 0.0;
        }
        }

        // Sidebar tab strip — emitted AFTER any body content so it always
        // overdraws what may have scrolled up into its y range.
        if self.active_tab == MainTab::Flame {
            let strip_x = inspector_x;
            let strip_y = TAB_BAR_HEIGHT_PX;
            let strip_w = INSPECTOR_WIDTH_PX;
            let strip_h = SIDEBAR_TAB_BAR_H;

            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [strip_x, strip_y, strip_w, strip_h],
                color: TAB_BAR_BG_COLOR,
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);

            let n = SidebarTab::ALL.len();
            let outer = 12.0_f32;
            let gap = 6.0_f32;
            let btn_w = (strip_w - outer * 2.0 - gap * (n as f32 - 1.0)) / n as f32;
            let btn_h = strip_h - 8.0;
            let btn_y = strip_y + 4.0;
            for (i, &tab) in SidebarTab::ALL.iter().enumerate() {
                let x = strip_x + outer + (btn_w + gap) * i as f32;
                let active = tab == self.sidebar_tab;
                let bg = if active { TOP_TAB_ACTIVE } else { TOP_TAB_INACTIVE };
                let inst_id = self.instances.len() as u32;
                self.instances.push(SliceInstance {
                    rect_px: [x, btn_y, btn_w, btn_h],
                    color: bg,
                    instance_id: inst_id,
                    flags: 1,
                    _pad: [0; 2],
                });
                self.slice_indices.push(NON_SLICE_SENTINEL);
                self.sidebar_tab_rects.push((tab, [x, btn_y, btn_w, btn_h]));
                if active {
                    let inst_id = self.instances.len() as u32;
                    self.instances.push(SliceInstance {
                        rect_px: [x, btn_y + btn_h - 3.0, btn_w, 3.0],
                        color: TOP_TAB_ACCENT,
                        instance_id: inst_id,
                        flags: 1,
                        _pad: [0; 2],
                    });
                    self.slice_indices.push(NON_SLICE_SENTINEL);
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

    /// Distinct values of `key` across the whole original profile, in stable
    /// alphabetical order. `None` slot reserved at the end for spans missing
    /// the key. Used for SEQUENCE column ordering.
    pub fn sequence_global_lifelines(&self, key: StringId) -> Vec<Option<StringId>> {
        let Some(orig) = &self.original_profile else { return Vec::new() };
        let mut set: std::collections::HashSet<Option<StringId>> = std::collections::HashSet::new();
        for i in 0..orig.slices.len() as u32 {
            set.insert(orig.attrs.get(i, key));
        }
        let mut out: Vec<Option<StringId>> = set.into_iter().collect();
        out.sort_by(|a, b| match (a, b) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, _) => std::cmp::Ordering::Greater,
            (_, None) => std::cmp::Ordering::Less,
            (Some(av), Some(bv)) => orig.strings.get(*av).cmp(orig.strings.get(*bv)),
        });
        out
    }

    /// Emit the SEQUENCE tab: full-window swimlane. Vertical columns = the
    /// distinct values of the configured lifeline key (default `service`).
    /// Time runs top→bottom over the entire capture. Every span becomes an
    /// activation box on its column. Cross-host parent→child relationships
    /// get a solid request line at the child's start and a dashed return at
    /// its end.
    fn emit_sequence_content(&mut self) {
        let top = TAB_BAR_HEIGHT_PX;
        let bot = self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX;
        let w = self.window_w_px;

        // Full-window dark background.
        let inst_id = self.instances.len() as u32;
        self.instances.push(SliceInstance {
            rect_px: [0.0, top, w, bot - top],
            color: [0.085, 0.092, 0.110, 1.0],
            instance_id: inst_id,
            flags: 1,
            _pad: [0; 2],
        });
        self.slice_indices.push(NON_SLICE_SENTINEL);

        let (Some(key), Some(orig)) = (self.sequence_lifeline_key, self.original_profile.clone())
        else {
            return;
        };
        let lifelines = self.sequence_global_lifelines(key);
        if lifelines.is_empty() || orig.slices.len() == 0 {
            return;
        }

        let (cap_start, cap_end) = orig.time_range;
        let duration = (cap_end.saturating_sub(cap_start)).max(1);

        let header_h = SEQUENCE_HEADER_H;
        let body_top = top + header_h;
        let body_bot = bot - 16.0;
        let body_h = (body_bot - body_top).max(1.0);

        // Time scale: the whole capture spans `content_h = body_h * zoom`. At
        // zoom 1 it fits the body exactly; zooming in spreads dense bursts of
        // activity apart. `scroll` shifts the window down the (taller) content,
        // re-clamped here in case the viewport was resized since the last pan.
        let content_h = body_h * self.sequence_time_zoom;
        let max_scroll = (content_h - body_h).max(0.0);
        let scroll = self.sequence_scroll_y_px.clamp(0.0, max_scroll);
        self.sequence_scroll_y_px = scroll;
        let ns_to_y = |ns: u64| -> f32 {
            let dt = ns.saturating_sub(cap_start) as f64;
            body_top + (dt / duration as f64) as f32 * content_h - scroll
        };

        let n_lanes = lifelines.len() as f32;
        let lane_w = (w / n_lanes).max(60.0);
        let lifeline_x = |lane_idx: usize| -> f32 { lane_w * (lane_idx as f32 + 0.5) };

        // Vertical lifeline rules.
        for li in 0..lifelines.len() {
            let x = lifeline_x(li);
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [x - 1.0, body_top, 2.0, body_h],
                color: [0.30, 0.33, 0.40, 1.0],
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);
        }

        // Header band.
        let inst_id = self.instances.len() as u32;
        self.instances.push(SliceInstance {
            rect_px: [0.0, top, w, header_h],
            color: HEADER_COLOR,
            instance_id: inst_id,
            flags: 1,
            _pad: [0; 2],
        });
        self.slice_indices.push(NON_SLICE_SENTINEL);

        // Build span_id → slice_idx map for parent lookups.
        let span_id_key = orig.strings.lookup("span_id");
        let parent_key = orig.strings.lookup("parent_span_id");
        let span_by_id: AHashMap<StringId, u32> = if let Some(sk) = span_id_key {
            (0..orig.slices.len() as u32)
                .filter_map(|i| orig.attrs.get(i, sk).map(|sid| (sid, i)))
                .collect()
        } else {
            AHashMap::new()
        };

        // Per-span: (lane_idx, y0, y1). Precompute so the arrow pass can
        // reuse the same x/y math without re-doing the lookup.
        let n = orig.slices.len();
        let mut lane_of: Vec<i32> = vec![-1; n];
        for i in 0..n as u32 {
            if let Some(val) = orig.attrs.get(i, key) {
                if let Some(li) = lifelines.iter().position(|l| *l == Some(val)) {
                    lane_of[i as usize] = li as i32;
                }
            } else if let Some(li) = lifelines.iter().position(|l| l.is_none()) {
                lane_of[i as usize] = li as i32;
            }
        }

        // Activation boxes — one per span.
        let box_w = (lane_w * 0.18).clamp(10.0, 32.0);
        for i in 0..n {
            let li = lane_of[i];
            if li < 0 {
                continue;
            }
            let y0 = ns_to_y(orig.slices.start_ns[i]);
            let y1 = ns_to_y(orig.slices.start_ns[i] + orig.slices.dur_ns[i]);
            if y1 < body_top - 4.0 || y0 > body_bot + 4.0 {
                continue;
            }
            let cat_id = orig.slices.category[i];
            let cat_name = orig.strings.get(orig.categories[cat_id.0 as usize].name);
            let name = orig.strings.get(orig.slices.name[i]);
            let color = if cat_name == "default" {
                palette::color_for(name)
            } else {
                palette::color_for_blend(cat_name, name)
            };
            let x = lifeline_x(li as usize) - box_w * 0.5;
            let h = (y1 - y0).max(3.0);
            let inst_id = self.instances.len() as u32;
            self.instances.push(SliceInstance {
                rect_px: [x, y0, box_w, h],
                color,
                instance_id: inst_id,
                flags: 1,
                _pad: [0; 2],
            });
            self.slice_indices.push(NON_SLICE_SENTINEL);
        }

        // Cross-lane arrows. For each span with a parent that resolves to a
        // different lane: solid line at child.start from parent→child,
        // dashed line at child.end from child→parent.
        if let (Some(pk), true) = (parent_key, !span_by_id.is_empty()) {
            let request_color = [0.92, 0.62, 0.30, 0.85];
            let response_color = [0.55, 0.62, 0.78, 0.70];
            for i in 0..n as u32 {
                let child_li = lane_of[i as usize];
                if child_li < 0 {
                    continue;
                }
                let Some(pid) = orig.attrs.get(i, pk) else { continue };
                let Some(&parent_i) = span_by_id.get(&pid) else { continue };
                let parent_li = lane_of[parent_i as usize];
                if parent_li < 0 || parent_li == child_li {
                    continue;
                }
                let start = orig.slices.start_ns[i as usize];
                let end = start + orig.slices.dur_ns[i as usize];
                let y_req = ns_to_y(start);
                let y_resp = ns_to_y(end);
                let px = lifeline_x(parent_li as usize);
                let cx = lifeline_x(child_li as usize);
                let lo = px.min(cx) + box_w * 0.5;
                let hi = px.max(cx) - box_w * 0.5;
                if hi - lo < 2.0 {
                    continue;
                }
                // Request: solid horizontal line at child start.
                if y_req >= body_top - 2.0 && y_req <= body_bot + 2.0 {
                    let inst_id = self.instances.len() as u32;
                    self.instances.push(SliceInstance {
                        rect_px: [lo, y_req - 1.0, hi - lo, 2.0],
                        color: request_color,
                        instance_id: inst_id,
                        flags: 1,
                        _pad: [0; 2],
                    });
                    self.slice_indices.push(NON_SLICE_SENTINEL);
                }
                // Response: dashed horizontal line at child end (segments).
                if y_resp >= body_top - 2.0 && y_resp <= body_bot + 2.0 && y_resp != y_req {
                    let seg = 6.0_f32;
                    let gap = 4.0_f32;
                    let mut x = lo;
                    while x < hi {
                        let w = seg.min(hi - x);
                        let inst_id = self.instances.len() as u32;
                        self.instances.push(SliceInstance {
                            rect_px: [x, y_resp - 1.0, w, 2.0],
                            color: response_color,
                            instance_id: inst_id,
                            flags: 1,
                            _pad: [0; 2],
                        });
                        self.slice_indices.push(NON_SLICE_SENTINEL);
                        x += seg + gap;
                    }
                }
            }
        }
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
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum Metric { Slice, Header, InspectorHeading, InspectorTitle, InspectorBody, MonoBody }
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum Zone {
            /// Top tab bar (y in [0, TAB_BAR_HEIGHT_PX]). Anything below clipped.
            TabBar,
            /// Timeline / inspector content (clip top at TAB_BAR_HEIGHT_PX,
            /// bottom at status-bar top).
            Below,
            /// Sidebar body — like `Below` but clip top is `inspector_content_top()`
            /// so scrolled body labels never bleed up into the sidebar tab strip.
            SidebarBody,
            /// Status bar (no extra clipping).
            Status,
        }
        let row_h = self.viewport.row_height_px;
        let mut labels: Vec<(String, f32, f32, f32, Metric, GlyphonColor, Zone)> = Vec::new();
        // Per-label wrap line count. 0/1 = single line (no wrap). N > 1 enables
        // word wrap into up to N lines; the y-bounds and clip rect grow with it.
        // Parallel to `labels`; only entries we explicitly override get N > 1.
        let mut wrap_lines_overrides: Vec<(usize, u16)> = Vec::new();
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

        // Sidebar tab-strip labels (GROUP / INSPECT).
        if self.active_tab == MainTab::Flame {
            let rects = self.sidebar_tab_rects.clone();
            for (tab, rect) in rects {
                let active = tab == self.sidebar_tab;
                let color = if active { title_color } else { dim_color };
                let lh = INSPECTOR_LINE_HEIGHT_PX;
                // Center text in the button.
                let approx_glyph_w = 11.0_f32;
                let text_w = tab.label().len() as f32 * approx_glyph_w;
                let tx = rect[0] + ((rect[2] - text_w) * 0.5).max(0.0);
                labels.push((
                    tab.label().to_string(),
                    tx,
                    rect[1] + (rect[3] - lh) * 0.5,
                    rect[2],
                    Metric::InspectorHeading,
                    color,
                    Zone::Below,
                ));
            }

            // GROUP-tab row labels. Rects already carry the scroll offset
            // (applied at chrome-emit time), and SidebarBody clips so labels
            // scrolled into the tab-strip area don't bleed through.
            if self.sidebar_tab == SidebarTab::Group {
                let rows = self.group_picker_row_rects.clone();
                let lh = INSPECTOR_LINE_HEIGHT_PX;
                for (val, rect) in rows {
                    let text = match val {
                        None => "(none)".to_string(),
                        Some(sid) => self
                            .original_profile
                            .as_ref()
                            .map(|p| p.strings.get(sid).to_string())
                            .unwrap_or_else(|| "?".into()),
                    };
                    let selected = self.group_key == val;
                    let color = if selected { title_color } else { dim_color };
                    labels.push((
                        format!("  {text}"),
                        rect[0] + 12.0,
                        rect[1] + (rect[3] - lh) * 0.5,
                        rect[2] - 12.0,
                        Metric::InspectorBody,
                        color,
                        Zone::SidebarBody,
                    ));
                }
            }
        }

        // In aggregated mode `slice_indices` and `selected_slice` index into
        // `aggregated_slices`, not `profile.slices` — labels and the inspector
        // both need to read from the same table the bars were drawn from.
        let label_slices: &flame_core::SliceTable = match (
            self.layout_mode,
            self.aggregated_slices.as_ref(),
        ) {
            (LayoutMode::LeftHeavy, Some(s)) => s,
            _ => &profile.slices,
        };

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
                let name = profile.strings.get(label_slices.name[slice_idx_raw as usize]).to_owned();
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

        // Account for the sidebar tab strip that sits between the global tab
        // bar and the inspector body. Body labels then apply the sidebar
        // scroll offset and use Zone::SidebarBody so they're clipped to the
        // body area (never bleed up into the tab strip).
        let mut iy = self.inspector_content_top() + pad - self.sidebar_scroll_y_px;
        let _ = top_offset; // shadowed for inspector y; keep var for other consumers
        let inspector_text_x = inspector_x + pad;
        let inspector_text_w = INSPECTOR_WIDTH_PX - pad * 2.0;

        // FLAME tab inspector content (SELECTED + STACK only). Skipped when
        // the sidebar's GROUP tab owns the inspector area. Other MainTabs
        // render their own full-window content view below the tab bar.
        if self.active_tab == MainTab::Flame && self.sidebar_tab == SidebarTab::Inspect {
        if let Some(sel_idx) = self.selected_slice {
            let i = sel_idx as usize;
            if i < label_slices.len() {
                let name = profile.strings.get(label_slices.name[i]).to_owned();
                let dur = label_slices.dur_ns[i];
                let depth = label_slices.depth[i];
                let track_idx = label_slices.track[i].0 as usize;
                let track_name = profile.tracks.get(track_idx).map(|t| profile.strings.get(t.name)).unwrap_or("?");

                labels.push(("SELECTED".into(), inspector_text_x, iy, inspector_text_w, Metric::InspectorHeading, dim_color, Zone::SidebarBody));
                iy += line;
                labels.push((name, inspector_text_x, iy, inspector_text_w, Metric::InspectorTitle, title_color, Zone::SidebarBody));
                iy += INSPECTOR_TITLE_LINE_HEIGHT;
                let meta = format!("{}    depth {}    {}", format_duration(dur), depth, track_name);
                labels.push((meta, inspector_text_x, iy, inspector_text_w, Metric::InspectorBody, dim_color, Zone::SidebarBody));
                iy += line + SECTION_GAP_PX;

                // Reconstruct call stack.
                let stack = self.reconstruct_stack(sel_idx);
                labels.push(("STACK".into(), inspector_text_x, iy, inspector_text_w, Metric::InspectorHeading, dim_color, Zone::SidebarBody));
                iy += line;
                // Show up to last 8 entries; deeper traces get truncated with an ellipsis.
                let max_lines = 8;
                let show: Vec<&u32> = if stack.len() > max_lines {
                    stack.iter().rev().take(max_lines).collect::<Vec<_>>().into_iter().rev().collect()
                } else {
                    stack.iter().collect()
                };
                if stack.len() > max_lines {
                    labels.push(("…".into(), inspector_text_x + 16.0, iy, inspector_text_w - 16.0, Metric::InspectorBody, dim_color, Zone::SidebarBody));
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
                        Zone::SidebarBody,
                    ));
                    iy += line;
                }
                iy += SECTION_GAP_PX;

                // ATTRIBUTES section: every (key, value) on the selected
                // span, sorted by key. Includes inherited attributes for OTel
                // (the loader resolves the parent-chain at load time). Long
                // values (SQL queries etc.) word-wrap across multiple lines
                // via `attr_wrap_lines`.
                labels.push((
                    "ATTRIBUTES".into(),
                    inspector_text_x,
                    iy,
                    inspector_text_w,
                    Metric::InspectorHeading,
                    dim_color,
                    Zone::SidebarBody,
                ));
                iy += line;
                let attrs = self.attrs_for_slice(sel_idx);
                if attrs.is_empty() {
                    labels.push((
                        "(none)".into(),
                        inspector_text_x + 16.0,
                        iy,
                        inspector_text_w - 16.0,
                        Metric::InspectorBody,
                        dim_color,
                        Zone::SidebarBody,
                    ));
                    iy += line;
                } else {
                    for (k, v) in &attrs {
                        let display_v = v.replace('\n', " ");
                        let line_text = format!("{k}:  {display_v}");
                        let wraps = self.attr_wrap_lines(k, &display_v);
                        let idx = labels.len();
                        labels.push((
                            line_text,
                            inspector_text_x,
                            iy,
                            inspector_text_w,
                            Metric::InspectorBody,
                            label_color,
                            Zone::SidebarBody,
                        ));
                        if wraps > 1 {
                            wrap_lines_overrides.push((idx, wraps));
                        }
                        iy += line * wraps as f32;
                    }
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
                        Zone::SidebarBody,
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
                            Zone::SidebarBody,
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
                title_color,
                Zone::Below,
            ));
            // Column widths must match the row format below: {:>9}  {:>6}%  {:>9}  {:>5}
            let header_text = format!(
                "{:>9}  {:>6}{}  {:>9}  {:>5}",
                "TOTAL", "%TOTAL", " ", "SELF", "×",
            );
            labels.push((
                header_text,
                metrics_x,
                header_y,
                metrics_w,
                Metric::MonoBody,
                title_color,
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
                    title_color,
                    Zone::Below,
                ));
                let pct = (node.total_ns as f64 / total_ns) * 100.0;
                let metrics_text = format!(
                    "{:>9}  {:>6.2}%  {:>9}  {:>5}",
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
                    Metric::MonoBody,
                    label_color,
                    Zone::Below,
                ));
            }
        } else if self.active_tab == MainTab::Sequence {
            // SEQUENCE tab: column-header labels for each lifeline (no trace
            // picker — the diagram covers the entire capture).
            let lh = INSPECTOR_LINE_HEIGHT_PX;
            if let (Some(key), Some(orig)) =
                (self.sequence_lifeline_key, self.original_profile.clone())
            {
                let lifelines = self.sequence_global_lifelines(key);
                if !lifelines.is_empty() {
                    let n_lanes = lifelines.len() as f32;
                    let lane_w = (self.window_w_px / n_lanes).max(60.0);
                    let key_label = orig.strings.get(key).to_string();
                    labels.push((
                        key_label.to_uppercase(),
                        12.0,
                        top_offset + 4.0,
                        self.window_w_px - 24.0,
                        Metric::InspectorBody,
                        dim_color,
                        Zone::Below,
                    ));
                    // Right-aligned controls hint + current time-axis zoom so
                    // the zoom/pan affordance is discoverable.
                    let hint = format!(
                        "scroll/pinch = zoom · drag = pan · {:.1}×",
                        self.sequence_time_zoom
                    );
                    let hint_w = (hint.len() as f32 * 7.0).min(self.window_w_px - 24.0);
                    labels.push((
                        hint,
                        self.window_w_px - 12.0 - hint_w,
                        top_offset + 4.0,
                        hint_w,
                        Metric::InspectorBody,
                        dim_color,
                        Zone::Below,
                    ));
                    for (li, val) in lifelines.iter().enumerate() {
                        let txt = match val {
                            None => format!("(no {})", key_label),
                            Some(sid) => orig.strings.get(*sid).to_string(),
                        };
                        let cx = lane_w * (li as f32 + 0.5);
                        let approx_w = (txt.len() as f32 * 11.0).min(lane_w - 8.0);
                        labels.push((
                            txt,
                            cx - approx_w * 0.5,
                            top_offset + SEQUENCE_HEADER_H - lh - 4.0,
                            approx_w,
                            Metric::InspectorHeading,
                            title_color,
                            Zone::Below,
                        ));
                    }
                }
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
        let status_for_bar = self
            .live_status
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.status_text.clone());
        labels.push((
            status_for_bar,
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

        // Materialize per-label wrap-line counts: default 1, overridden where
        // the label requested word-wrap (currently only ATTRIBUTES values).
        let mut wrap_lines: Vec<u16> = vec![1; labels.len()];
        for &(idx, n) in &wrap_lines_overrides {
            if idx < wrap_lines.len() {
                wrap_lines[idx] = n;
            }
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
                Metric::MonoBody => (INSPECTOR_BODY_FONT, INSPECTOR_LINE_HEIGHT_PX),
            };
            buf.set_metrics(&mut self.font_system, Metrics::new(fs, lh));
            let n_lines = wrap_lines[i].max(1) as f32;
            let wrap_mode = if n_lines > 1.0 { Wrap::Word } else { Wrap::None };
            buf.set_wrap(&mut self.font_system, wrap_mode);
            buf.set_size(&mut self.font_system, Some(*max_w), Some(lh * n_lines));
            let family = if *metric == Metric::MonoBody {
                Family::Monospace
            } else {
                Family::SansSerif
            };
            buf.set_text(
                &mut self.font_system,
                text,
                &Attrs::new().family(family),
                Shaping::Advanced,
                None,
            );
            buf.shape_until_scroll(&mut self.font_system, false);
        }

        // Glyphon bounds use the full window — inspector labels live to the right
        // of the timeline.
        let view_w = self.window_w_px as i32;
        let view_h = self.viewport.size_px.1 as i32;
        // Precompute so the closure below doesn't reborrow &self.
        let inspector_body_top = self.inspector_content_top() as i32;

        let text_areas: Vec<TextArea> = labels
            .iter()
            .enumerate()
            .map(|(i, (_, x, y, max_w, metric, color, zone))| {
                let lh = match metric {
                    Metric::Slice => LABEL_LINE_HEIGHT,
                    Metric::Header => HEADER_LINE_HEIGHT,
                    Metric::InspectorHeading
                    | Metric::InspectorBody
                    | Metric::MonoBody => INSPECTOR_LINE_HEIGHT_PX,
                    Metric::InspectorTitle => INSPECTOR_TITLE_LINE_HEIGHT,
                };
                let n_lines = wrap_lines[i].max(1) as f32;
                let bounds_left = x.floor() as i32;
                let bounds_top = y.floor() as i32;
                let bounds_right = (x + max_w).ceil() as i32;
                let bounds_bottom = (y + lh * n_lines).ceil() as i32;
                let tab_bar_h = TAB_BAR_HEIGHT_PX as i32;
                let status_top = (self.viewport.size_px.1 - STATUS_BAR_HEIGHT_PX) as i32;
                let (clip_top, clip_bot) = match zone {
                    Zone::TabBar => (0, tab_bar_h),
                    Zone::Below => (tab_bar_h, status_top),
                    Zone::SidebarBody => (inspector_body_top, status_top),
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

    /// Render one frame into `view`. The caller is responsible for acquiring
    /// the view (e.g. from a swapchain surface texture or a render-to-texture
    /// `wgpu::Texture`) and for presenting it afterwards if applicable.
    ///
    /// The texture format of `view` must match the `target_format` passed to
    /// [`Renderer::new`].
    pub fn render(&mut self, view: &wgpu::TextureView) {
        // Update uniforms.
        let uniforms = Uniforms {
            // Pixel-to-NDC mapping covers the *whole canvas*, not just the
            // timeline. Inspector instances live in [timeline_w, window_w] and
            // would otherwise map to NDC.x > 1 and be clipped.
            viewport_size_px: [self.window_w_px, self.viewport.size_px.1],
            hovered: self.hovered.unwrap_or(u32::MAX),
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.prepare_text();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flame-render frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("flame-render main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
#[doc(hidden)]
pub fn build_left_heavy_layout(
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

/// Build a single-track view of a profile. Strings, categories, processes,
/// threads, stacks, and samples are cloned verbatim. The track list is
/// replaced with one synthetic "all spans" track; every slice is repacked
/// onto it with greedy first-fit row assignment so no two slices overlap
/// in the same row.
fn build_single_track_profile(orig: &Profile) -> Profile {
    let n = orig.slices.len();

    // Greedy first-fit row pack. `row_end[r]` is the last end_ns assigned to
    // row r. Walk spans in start-order; pick the smallest-indexed row whose
    // last-end is <= this start, else open a new row.
    let mut visit: Vec<u32> = (0..n as u32).collect();
    visit.sort_by_key(|&i| orig.slices.start_ns[i as usize]);

    let mut row_end: Vec<u64> = Vec::new();
    let mut new_depth: Vec<u16> = vec![0; n];
    for &i in &visit {
        let i = i as usize;
        let start = orig.slices.start_ns[i];
        let end = start.saturating_add(orig.slices.dur_ns[i]);
        let mut chose: Option<usize> = None;
        for (r, &e) in row_end.iter().enumerate() {
            if e <= start {
                chose = Some(r);
                break;
            }
        }
        let d = match chose {
            Some(r) => {
                row_end[r] = end;
                r
            }
            None => {
                row_end.push(end);
                row_end.len() - 1
            }
        };
        new_depth[i] = d as u16;
    }
    let row_count = row_end.len() as u16;

    // Sort entries into (depth, start_ns) order so the SoA respects the
    // flame-graph invariant (sorted by track, depth, start_ns) and the
    // per-row range index is contiguous.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| {
        let i = i as usize;
        (new_depth[i], orig.slices.start_ns[i])
    });

    let mut strings = orig.strings.clone();
    let track_name = strings.intern("all spans (merged)");

    let synth_track = Track {
        kind: TrackKind::Global,
        name: track_name,
        parent: None,
        row_count,
    };
    let track_id = TrackId(0);

    let mut slices = SliceTable {
        track: Vec::with_capacity(n),
        depth: Vec::with_capacity(n),
        start_ns: Vec::with_capacity(n),
        dur_ns: Vec::with_capacity(n),
        name: Vec::with_capacity(n),
        category: Vec::with_capacity(n),
        stack: Vec::with_capacity(n),
        rows: ahash::AHashMap::new(),
    };
    let mut row_start: u32 = 0;
    let mut cur_depth: Option<u16> = None;
    let mut per_slice_attrs: Vec<Vec<(u16, StringId)>> = Vec::with_capacity(n);
    for (pos, &src) in order.iter().enumerate() {
        let src = src as usize;
        let d = new_depth[src];
        if Some(d) != cur_depth {
            if let Some(prev) = cur_depth {
                slices.rows.insert((track_id, prev), row_start..pos as u32);
            }
            row_start = pos as u32;
            cur_depth = Some(d);
        }
        slices.track.push(track_id);
        slices.depth.push(d);
        slices.start_ns.push(orig.slices.start_ns[src]);
        slices.dur_ns.push(orig.slices.dur_ns[src]);
        slices.name.push(orig.slices.name[src]);
        slices.category.push(orig.slices.category[src]);
        slices.stack.push(orig.slices.stack[src]);
        per_slice_attrs.push(
            orig.attrs
                .per_slice
                .get(src)
                .cloned()
                .unwrap_or_default(),
        );
    }
    if let Some(prev) = cur_depth {
        slices.rows.insert((track_id, prev), row_start..n as u32);
    }

    let attrs = flame_core::AttrTable {
        keys: orig.attrs.keys.clone(),
        key_lookup: orig.attrs.key_lookup.clone(),
        per_slice: per_slice_attrs,
    };

    Profile {
        strings,
        categories: orig.categories.clone(),
        processes: orig.processes.clone(),
        threads: orig.threads.clone(),
        tracks: vec![synth_track],
        stacks: orig.stacks.clone(),
        slices,
        samples: orig.samples.clone(),
        attrs,
        time_range: orig.time_range,
    }
}

/// Bucket slices by their value of attribute `key`, then build one track per
/// bucket (alphabetically by value), greedy-row-packed within each. Spans
/// missing the key land in an "(none)" bucket.
fn build_grouped_profile(orig: &Profile, key: StringId) -> Profile {
    let n = orig.slices.len();

    // Step 1: bucket assignment per slice. The bucket key is the value
    // StringId (or None for absent).
    let mut bucket_for: Vec<Option<StringId>> = Vec::with_capacity(n);
    for i in 0..n {
        bucket_for.push(orig.attrs.get(i as u32, key));
    }

    // Step 2: ordered bucket list (stable: sort by value string).
    let mut distinct_values: Vec<Option<StringId>> = bucket_for
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    distinct_values.sort_by(|a, b| match (a, b) {
        (None, None) => std::cmp::Ordering::Equal,
        // Put "(none)" track at the bottom of the list.
        (None, _) => std::cmp::Ordering::Greater,
        (_, None) => std::cmp::Ordering::Less,
        (Some(av), Some(bv)) => orig.strings.get(*av).cmp(orig.strings.get(*bv)),
    });
    let bucket_idx: AHashMap<Option<StringId>, u32> = distinct_values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    // Step 3: per-bucket greedy first-fit row packing. Track row_end[bucket] =
    // Vec<u64> (last end_ns per row). Walk slices in start order.
    let mut visit: Vec<u32> = (0..n as u32).collect();
    visit.sort_by_key(|&i| orig.slices.start_ns[i as usize]);

    let mut row_end_per_bucket: Vec<Vec<u64>> = vec![Vec::new(); distinct_values.len()];
    let mut new_track: Vec<u32> = vec![0; n];
    let mut new_depth: Vec<u16> = vec![0; n];
    for &i in &visit {
        let i = i as usize;
        let b = bucket_idx[&bucket_for[i]];
        let start = orig.slices.start_ns[i];
        let end = start.saturating_add(orig.slices.dur_ns[i]);
        let rows = &mut row_end_per_bucket[b as usize];
        let mut chose: Option<usize> = None;
        for (r, &e) in rows.iter().enumerate() {
            if e <= start {
                chose = Some(r);
                break;
            }
        }
        let d = match chose {
            Some(r) => {
                rows[r] = end;
                r
            }
            None => {
                rows.push(end);
                rows.len() - 1
            }
        };
        new_track[i] = b;
        new_depth[i] = d as u16;
    }

    // Step 4: re-intern strings (so original + this profile can coexist in
    // Arcs without aliasing surprises) and build per-bucket Track entries.
    let mut strings = orig.strings.clone();
    let mut tracks: Vec<Track> = Vec::with_capacity(distinct_values.len());
    for (b, val) in distinct_values.iter().enumerate() {
        let label = match val {
            Some(sid) => format!("{} = {}", strings.get(key), strings.get(*sid)),
            None => format!("(no {})", strings.get(key)),
        };
        let name_id = strings.intern(&label);
        tracks.push(Track {
            kind: TrackKind::Global,
            name: name_id,
            parent: None,
            row_count: row_end_per_bucket[b].len() as u16,
        });
    }

    // Step 5: sort all slices into (track, depth, start_ns) order for the SoA
    // invariant, build SliceTable + parallel attrs.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| {
        let i = i as usize;
        (new_track[i], new_depth[i], orig.slices.start_ns[i])
    });

    let mut slices = SliceTable {
        track: Vec::with_capacity(n),
        depth: Vec::with_capacity(n),
        start_ns: Vec::with_capacity(n),
        dur_ns: Vec::with_capacity(n),
        name: Vec::with_capacity(n),
        category: Vec::with_capacity(n),
        stack: Vec::with_capacity(n),
        rows: ahash::AHashMap::new(),
    };
    let mut per_slice_attrs: Vec<Vec<(u16, StringId)>> = Vec::with_capacity(n);
    let mut row_start: u32 = 0;
    let mut cur_key: Option<(TrackId, u16)> = None;
    for (pos, &src) in order.iter().enumerate() {
        let src = src as usize;
        let t = TrackId(new_track[src]);
        let d = new_depth[src];
        let key2 = (t, d);
        if Some(key2) != cur_key {
            if let Some(prev) = cur_key {
                slices.rows.insert(prev, row_start..pos as u32);
            }
            row_start = pos as u32;
            cur_key = Some(key2);
        }
        slices.track.push(t);
        slices.depth.push(d);
        slices.start_ns.push(orig.slices.start_ns[src]);
        slices.dur_ns.push(orig.slices.dur_ns[src]);
        slices.name.push(orig.slices.name[src]);
        slices.category.push(orig.slices.category[src]);
        slices.stack.push(orig.slices.stack[src]);
        per_slice_attrs.push(
            orig.attrs
                .per_slice
                .get(src)
                .cloned()
                .unwrap_or_default(),
        );
    }
    if let Some(prev) = cur_key {
        slices.rows.insert(prev, row_start..n as u32);
    }

    let attrs = flame_core::AttrTable {
        keys: orig.attrs.keys.clone(),
        key_lookup: orig.attrs.key_lookup.clone(),
        per_slice: per_slice_attrs,
    };

    Profile {
        strings,
        categories: orig.categories.clone(),
        processes: orig.processes.clone(),
        threads: orig.threads.clone(),
        tracks,
        stacks: orig.stacks.clone(),
        slices,
        samples: orig.samples.clone(),
        attrs,
        time_range: orig.time_range,
    }
}

/// Snapshot of user state that needs to survive a live profile swap.
struct LiveState {
    viewport: Viewport,
    active_tab: MainTab,
    sidebar_tab: SidebarTab,
    layout_mode: LayoutMode,
    merge_mode: MergeMode,
    direction: Direction,
    /// Name of the attribute key the timeline is grouped by, or None.
    group_key_name: Option<String>,
    /// Name of the SEQUENCE lifeline key, or None.
    sequence_lifeline_key_name: Option<String>,
    /// Names of tracks the user has collapsed.
    collapsed_track_names: Vec<String>,
    /// Identity of the currently selected slice, content-keyed so we can
    /// re-resolve it to the corresponding slice in the new profile.
    selected: Option<SliceIdentity>,
    sidebar_scroll_y_px: f32,
    sequence_scroll_y_px: f32,
    sequence_time_zoom: f32,
    call_tree_scroll_y_px: f32,
}

/// Content-keyed identity of a slice: which track it lives on (by name) and
/// the sequence of frame names from root to the slice itself. For the
/// synthesized flame-graph layout this uniquely identifies a slice across
/// snapshots because slices are nested perfectly within their parents.
struct SliceIdentity {
    track_name: String,
    /// Frame names from outermost (root) to innermost (the slice itself).
    name_path: Vec<String>,
}

/// Walk a slice up to its root, returning (track_name, name_path). Returns
/// None if the slice index is out of bounds or the track is missing.
fn slice_identity(p: &Profile, slice_idx: u32) -> Option<SliceIdentity> {
    let i = slice_idx as usize;
    if i >= p.slices.len() {
        return None;
    }
    let track = p.slices.track[i];
    let track_name = p.strings.get(p.tracks.get(track.0 as usize)?.name).to_string();
    let depth = p.slices.depth[i];

    let mut chain = Vec::with_capacity(depth as usize + 1);
    chain.push(i as u32);
    let mut cur_depth = depth;
    let mut cur_start = p.slices.start_ns[i];
    let mut cur_end = cur_start + p.slices.dur_ns[i];

    while cur_depth > 0 {
        let parent_depth = cur_depth - 1;
        let row = p.slices.rows.get(&(track, parent_depth))?;
        // Synthesized flame-graph layout: parents contain children exactly,
        // so the parent's [start, end) is a superset of the child's.
        let mut parent_idx: Option<u32> = None;
        for j in row.clone() {
            let ps = p.slices.start_ns[j as usize];
            let pe = ps + p.slices.dur_ns[j as usize];
            if ps <= cur_start && pe >= cur_end {
                parent_idx = Some(j);
                break;
            }
        }
        let pi = parent_idx?;
        chain.push(pi);
        cur_depth = parent_depth;
        cur_start = p.slices.start_ns[pi as usize];
        cur_end = cur_start + p.slices.dur_ns[pi as usize];
    }

    chain.reverse();
    let name_path: Vec<String> = chain
        .iter()
        .map(|&idx| p.strings.get(p.slices.name[idx as usize]).to_string())
        .collect();
    Some(SliceIdentity {
        track_name,
        name_path,
    })
}

/// Inverse of `slice_identity`: in `p`, find the slice with the given track
/// name and root-to-leaf name path. None if anything along the path is
/// missing.
fn find_slice_by_identity(p: &Profile, id: &SliceIdentity) -> Option<u32> {
    let track_id = p.tracks.iter().enumerate().find_map(|(i, t)| {
        (p.strings.get(t.name) == id.track_name).then_some(TrackId(i as u32))
    })?;
    if id.name_path.is_empty() {
        return None;
    }
    // Walk down each depth, picking the first slice that matches the name
    // and (for depth > 0) is contained in the current parent's range.
    let mut parent_range: Option<(u64, u64)> = None;
    let mut found: Option<u32> = None;
    for (depth_idx, frame_name) in id.name_path.iter().enumerate() {
        let depth = depth_idx as u16;
        let row = p.slices.rows.get(&(track_id, depth))?;
        let mut next: Option<u32> = None;
        for j in row.clone() {
            let n = p.strings.get(p.slices.name[j as usize]);
            if n != frame_name {
                continue;
            }
            let s = p.slices.start_ns[j as usize];
            let e = s + p.slices.dur_ns[j as usize];
            if let Some((ps, pe)) = parent_range {
                if s < ps || e > pe {
                    continue;
                }
            }
            next = Some(j);
            break;
        }
        let n = next?;
        parent_range = Some((
            p.slices.start_ns[n as usize],
            p.slices.start_ns[n as usize] + p.slices.dur_ns[n as usize],
        ));
        found = Some(n);
    }
    found
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
