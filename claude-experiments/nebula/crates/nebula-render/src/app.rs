//! The windowed application: winit event loop, camera input, per-frame GPU
//! layout stepping, rendering, and wiring graph algorithms to node colors.

use std::sync::Arc;
use std::time::Instant;

use nebula_core::{algorithms, Graph, Pos};
use nebula_layout::{CircleLayout, GridLayout, Layout, LayeredLayout, RadialLayout, RandomLayout};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::Camera2D;
use crate::coloring;
use crate::density::Density;
use crate::gpu::Gpu;
use crate::layout_gpu::{LayoutGpu, LayoutSettings};
use crate::overlay::Overlay;
use crate::render::{RenderParams, Renderer};
use crate::scene::{pack_rgba, GpuGraph};
use crate::ui::{Ui, UiFrame};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    Uniform,
    Components,
    Degree,
    PageRank,
    Coloring,
    Communities,
    /// Color by a loaded node attribute, indexed into `App::attr_keys`.
    Attribute(usize),
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
            ColorMode::Attribute(_) => "attribute",
        }
    }
}

/// Color modes shown in the UI panel, paired with human labels.
const COLOR_MODES: [(ColorMode, &str); 6] = [
    (ColorMode::Uniform, "Uniform"),
    (ColorMode::Components, "Connected components"),
    (ColorMode::Degree, "Degree"),
    (ColorMode::PageRank, "PageRank"),
    (ColorMode::Coloring, "Greedy coloring"),
    (ColorMode::Communities, "Communities"),
];

/// A named set of edges (relationship type) shown over the shared node set.
struct EdgeType {
    name: String,
    /// Packed RGBA8 tint, or None to color edges by endpoint node color.
    color: Option<u32>,
    visible: bool,
    edges: Vec<[u32; 2]>,
}

/// Which layout to (re)apply when the user clicks a layout button.
#[derive(Clone, Copy)]
enum Seed {
    Random,
    Grid,
    Circle,
    /// Fixed hierarchical (layered DAG) layout; pauses the simulation.
    Hierarchical,
    /// Fixed radial (concentric DAG) layout; pauses the simulation.
    Radial,
}

/// Comparison used by the attribute filter.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Contains,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl FilterOp {
    /// Parse an operator name (for the CLI): contains/eq/ne/gt/ge/lt/le.
    pub fn from_name(s: &str) -> Option<FilterOp> {
        Some(match s {
            "contains" => FilterOp::Contains,
            "eq" => FilterOp::Eq,
            "ne" => FilterOp::Ne,
            "gt" => FilterOp::Gt,
            "ge" => FilterOp::Ge,
            "lt" => FilterOp::Lt,
            "le" => FilterOp::Le,
            _ => return None,
        })
    }

    const ALL: [FilterOp; 7] = [
        FilterOp::Contains,
        FilterOp::Eq,
        FilterOp::Ne,
        FilterOp::Gt,
        FilterOp::Ge,
        FilterOp::Lt,
        FilterOp::Le,
    ];
    fn label(self) -> &'static str {
        match self {
            FilterOp::Contains => "contains",
            FilterOp::Eq => "=",
            FilterOp::Ne => "≠",
            FilterOp::Gt => ">",
            FilterOp::Ge => "≥",
            FilterOp::Lt => "<",
            FilterOp::Le => "≤",
        }
    }
    /// Evaluate `actual <op> target`. Numeric ops parse both sides as numbers
    /// and fail closed (no match) when either side isn't numeric.
    fn eval(self, actual: &str, target: &str) -> bool {
        match self {
            FilterOp::Contains => actual.contains(target),
            FilterOp::Eq => actual == target,
            FilterOp::Ne => actual != target,
            _ => {
                let (Ok(a), Ok(b)) = (actual.trim().parse::<f64>(), target.trim().parse::<f64>())
                else {
                    return false;
                };
                match self {
                    FilterOp::Gt => a > b,
                    FilterOp::Ge => a >= b,
                    FilterOp::Lt => a < b,
                    FilterOp::Le => a <= b,
                    _ => false,
                }
            }
        }
    }
}

/// "Show only" filter on a node attribute. When enabled, non-matching nodes are
/// hidden (rgb + size zeroed, which also drops their additively-blended edges).
struct NodeFilter {
    enabled: bool,
    key: usize,
    op: FilterOp,
    value: String,
}

impl Default for NodeFilter {
    fn default() -> Self {
        NodeFilter {
            enabled: false,
            key: 0,
            op: FilterOp::Contains,
            value: String::new(),
        }
    }
}

/// A legend highlight predicate over per-node `last_values`.
#[derive(Clone, Copy, PartialEq)]
enum Highlight {
    /// Keep nodes whose value equals this category label.
    Category(f32),
    /// Keep nodes whose value falls in `[lo, hi]` (a slice of the scalar ramp).
    ValueBand(f32, f32),
}

/// What the pointer did to the legend this frame.
#[derive(Default)]
struct LegendInteraction {
    /// Entry under the pointer (temporary preview highlight).
    hovered: Option<Highlight>,
    /// Entry clicked this frame (toggles the pinned highlight).
    clicked: Option<Highlight>,
}

/// What the color legend should show for the active color mode.
#[derive(Clone)]
enum Legend {
    None,
    /// Single uniform color (no legend needed).
    Uniform,
    /// A continuous scalar with a turbo gradient between `lo` and `hi`.
    Scalar { lo: f32, hi: f32, log: bool, name: String },
    /// A categorical field: the most frequent classes as `(label, color, freq)`
    /// plus the total distinct count.
    Categorical { count: usize, name: String, top: Vec<(f32, u32, usize)> },
}

impl Legend {
    /// Build a scalar legend from the current per-node values.
    fn scalar(values: &Option<Vec<f32>>, name: &str, log: bool) -> Legend {
        match values {
            Some(v) if !v.is_empty() => {
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for &x in v {
                    if x.is_finite() {
                        lo = lo.min(x);
                        hi = hi.max(x);
                    }
                }
                if lo > hi {
                    Legend::None
                } else {
                    Legend::Scalar { lo, hi, log, name: name.to_string() }
                }
            }
            _ => Legend::None,
        }
    }

    /// Build a categorical legend from label values: the 12 most frequent
    /// classes (with their real colors) plus the distinct count.
    fn categorical(values: &Option<Vec<f32>>, name: &str) -> Legend {
        match values {
            Some(v) => {
                let mut freq: std::collections::HashMap<i64, usize> =
                    std::collections::HashMap::new();
                for &x in v {
                    *freq.entry(x as i64).or_insert(0) += 1;
                }
                let count = freq.len();
                let mut items: Vec<(i64, usize)> = freq.into_iter().collect();
                items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let top = items
                    .iter()
                    .take(12)
                    .map(|&(label, f)| {
                        (label as f32, crate::coloring::categorical(label as u32), f)
                    })
                    .collect();
                Legend::Categorical { count, name: name.to_string(), top }
            }
            None => Legend::None,
        }
    }
}

/// Greyscale a packed RGBA8 color for the "fade": desaturate to luminance (so
/// non-matching nodes read as muted context) and lower the alpha a little.
fn grey_color(c: u32) -> u32 {
    let r = (c & 0xff) as f32;
    let g = ((c >> 8) & 0xff) as f32;
    let b = ((c >> 16) & 0xff) as f32;
    let lum = (0.299 * r + 0.587 * g + 0.114 * b) * 0.4;
    let v = lum.clamp(0.0, 255.0) as u8;
    pack_rgba(v, v, v, 95)
}

/// Unpack a packed RGBA8 color into an egui color (drops alpha).
fn egui_rgb(c: u32) -> egui::Color32 {
    egui::Color32::from_rgb((c & 0xff) as u8, ((c >> 8) & 0xff) as u8, ((c >> 16) & 0xff) as u8)
}

/// Format a legend endpoint: integers without decimals, else a few places.
fn fmt_legend_num(v: f32) -> String {
    if (v - v.round()).abs() < 1e-4 && v.abs() < 1e7 {
        format!("{}", v.round() as i64)
    } else if v.abs() < 1e-2 || v.abs() >= 1e5 {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

/// Forward map a value to a `[0,1]` position on the (optionally log) gradient.
fn scalar_t(v: f32, lo: f32, hi: f32, log: bool) -> f32 {
    if log {
        let a = (lo.max(0.0) + 1.0).ln();
        let b = (hi.max(0.0) + 1.0).ln();
        if b <= a {
            0.0
        } else {
            (((v.max(0.0) + 1.0).ln()) - a) / (b - a)
        }
    } else if hi <= lo {
        0.0
    } else {
        (v - lo) / (hi - lo)
    }
    .clamp(0.0, 1.0)
}

/// Draw the color legend for the active mode. Entries are hoverable (temporary
/// preview) and clickable (toggles a persistent pin). `pinned` is the currently
/// pinned highlight, drawn with a persistent marker.
fn draw_legend(ui: &mut egui::Ui, legend: &Legend, pinned: Option<Highlight>) -> LegendInteraction {
    let mut out = LegendInteraction::default();
    let accent = egui::Color32::from_rgb(120, 200, 255);
    match legend {
        Legend::None | Legend::Uniform => {}
        Legend::Scalar { lo, hi, log, name } => {
            ui.add_space(2.0);
            let w = ui.available_width().min(220.0);
            let (rect, resp) =
                ui.allocate_exact_size(egui::vec2(w, 14.0), egui::Sense::click());
            let painter = ui.painter();
            let steps = 64;
            for i in 0..steps {
                let t = i as f32 / (steps - 1) as f32;
                let col = egui_rgb(crate::coloring::turbo_rgba(t));
                let x0 = rect.left() + rect.width() * i as f32 / steps as f32;
                let x1 = rect.left() + rect.width() * (i + 1) as f32 / steps as f32;
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x0, rect.top()),
                        egui::pos2(x1, rect.bottom()),
                    ),
                    0.0,
                    col,
                );
            }
            let band = 0.06;
            let inv = |tt: f32| -> f32 {
                let tt = tt.clamp(0.0, 1.0);
                if *log {
                    let a = (lo.max(0.0) + 1.0).ln();
                    let b = (hi.max(0.0) + 1.0).ln();
                    (a + tt * (b - a)).exp() - 1.0
                } else {
                    lo + tt * (hi - lo)
                }
            };
            let marker = |painter: &egui::Painter, t: f32, color: egui::Color32| {
                let mx = rect.left() + t.clamp(0.0, 1.0) * rect.width();
                painter.line_segment(
                    [egui::pos2(mx, rect.top() - 2.0), egui::pos2(mx, rect.bottom() + 2.0)],
                    egui::Stroke::new(2.0, color),
                );
            };
            // Persistent marker for a pinned band.
            if let Some(Highlight::ValueBand(plo, phi)) = pinned {
                marker(painter, scalar_t((plo + phi) * 0.5, *lo, *hi, *log), accent);
            }
            // Hover preview + click to pin.
            if let Some(p) = resp.hover_pos() {
                let t = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                out.hovered = Some(Highlight::ValueBand(inv(t - band), inv(t + band)));
                marker(painter, t, egui::Color32::WHITE);
            }
            if resp.clicked() {
                if let Some(p) = resp.interact_pointer_pos() {
                    let t = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                    out.clicked = Some(Highlight::ValueBand(inv(t - band), inv(t + band)));
                }
            }
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(fmt_legend_num(*lo)).weak().small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(fmt_legend_num(*hi)).weak().small());
                });
            });
            let suffix = if *log { "  (log scale)" } else { "" };
            ui.label(egui::RichText::new(format!("{name}{suffix}")).weak().small());
        }
        Legend::Categorical { count, name, top } => {
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                for &(label, color, freq) in top {
                    let (rect, resp) =
                        ui.allocate_exact_size(egui::vec2(15.0, 15.0), egui::Sense::click());
                    ui.painter().rect_filled(rect, 2.0, egui_rgb(color));
                    let is_pinned = pinned == Some(Highlight::Category(label));
                    let hovered = resp.hovered();
                    if hovered || is_pinned {
                        let stroke_col = if hovered { egui::Color32::WHITE } else { accent };
                        ui.painter().rect_stroke(
                            rect,
                            2.0,
                            egui::Stroke::new(2.0, stroke_col),
                            egui::StrokeKind::Outside,
                        );
                    }
                    if hovered {
                        out.hovered = Some(Highlight::Category(label));
                        resp.clone().on_hover_text(format!("{freq} nodes — click to pin"));
                    }
                    if resp.clicked() {
                        out.clicked = Some(Highlight::Category(label));
                    }
                }
                if *count > top.len() {
                    ui.label(egui::RichText::new("…").weak());
                }
            });
            ui.label(
                egui::RichText::new(format!("{name}: {count} categories")).weak().small(),
            );
        }
    }
    out
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
    /// Start with node labels visible.
    pub show_labels: bool,
    /// Force the density-aggregation LOD on (otherwise auto for >2M nodes).
    pub aggregate: bool,
    /// Initial node radius in pixels (runtime-adjustable with +/-).
    pub node_size: f32,
    /// Max edges drawn per frame (sampled with alpha compensation beyond this).
    pub edge_budget: u32,
    /// Optional startup "show only" filter: (attribute key index, op, value).
    pub filter: Option<(usize, FilterOp, String)>,
    /// Start in the hierarchical (layered DAG) layout instead of force-directed.
    pub start_hierarchical: bool,
    /// Start in the radial (concentric DAG) layout instead of force-directed.
    pub start_radial: bool,
    /// Headless preview of the legend-hover highlight (the most frequent class).
    pub preview_highlight: bool,
    /// Edge set for the algorithms: "all" (union) or an edge type name.
    pub compute_over: Option<String>,
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
            show_labels: false,
            aggregate: false,
            node_size: 3.0,
            edge_budget: 1_000_000,
            filter: None,
            start_hierarchical: false,
            start_radial: false,
            preview_highlight: false,
            compute_over: None,
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
    density: Density,
}

pub struct App {
    // Immutable inputs.
    graph: Graph,
    seed_positions: Vec<Pos>,
    opts: RunOptions,
    labels: Option<Vec<String>>,
    /// Optional per-node attributes (key/value), indexed by node id. Shown in
    /// the inspection panel when a node is selected.
    node_attrs: Option<Vec<Vec<(String, String)>>>,
    /// Distinct attribute keys (union over all nodes), in first-seen order.
    /// Drives the "color by attribute" and filter UI.
    attr_keys: Vec<String>,
    /// Named edge sets shown over the graph (e.g. children, deps).
    edge_types: Vec<EdgeType>,
    /// Which edge set the edge-aware algorithms (coloring, layouts) compute over.
    /// None = union of all types; Some(i) = edge type `i`. Applied on the next
    /// manual recolor / relayout, not automatically.
    algo_edge_set: Option<usize>,
    /// Which edge set the GPU force springs currently reflect (rebuilt on reseed
    /// when it differs from `algo_edge_set`).
    spring_edge_set: Option<usize>,
    /// Attribute "show only" filter; non-matching nodes are hidden.
    filter: NodeFilter,
    /// Node count matching the current filter (for the UI), or None if disabled.
    filter_match_count: Option<usize>,
    /// Legend descriptor for the active color mode.
    legend: Legend,
    /// Unfiltered colors/sizes from the active color mode; the filter masks
    /// these into the GPU buffers so toggling the filter doesn't recompute.
    base_colors: Vec<u32>,
    base_sizes: Vec<f32>,

    // Runtime.
    live: Option<Live>,
    /// egui control panel plumbing (created with the window).
    ui: Option<Ui>,
    /// Tessellated egui output for the current frame, drawn during `render`.
    ui_frame: Option<UiFrame>,
    /// Whether the control panel is shown.
    show_panel: bool,
    camera: Camera2D,
    settings: LayoutSettings,
    render_params: RenderParams,
    color_mode: ColorMode,

    // Selection / overlay.
    selected: Option<u32>,
    /// Effective highlight in effect this frame (hover preview, else pinned).
    highlight: Option<Highlight>,
    /// Highlight pinned by clicking a legend entry; persists without hovering.
    pinned_highlight: Option<Highlight>,
    selected_pos: Option<glam::Vec2>,
    neighbor_positions: Vec<glam::Vec2>,
    last_values: Option<Vec<f32>>,
    value_name: String,
    show_help: bool,
    show_hud: bool,
    show_labels: bool,
    cached_positions: Option<Vec<Pos>>,
    show_density: bool,

    // Input state.
    cursor: glam::Vec2,
    dragging: bool,
    press_pos: glam::Vec2,
    moved_since_press: bool,
    last_cursor: glam::Vec2,
    last_click: Instant,
    last_click_pos: glam::Vec2,

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
        Self::with_labels(graph, seed_positions, opts, None, None, Vec::new())
    }

    pub fn with_labels(
        mut graph: Graph,
        seed_positions: Vec<Pos>,
        opts: RunOptions,
        labels: Option<Vec<String>>,
        node_attrs: Option<Vec<Vec<(String, String)>>>,
        loaded_edge_types: Vec<nebula_core::formats::LoadedEdgeType>,
    ) -> Self {
        graph.ensure_csr();
        // Build edge types from the load; synthesize a single node-colored set
        // for formats/generators that don't carry types.
        let edge_types: Vec<EdgeType> = if loaded_edge_types.is_empty() {
            vec![EdgeType {
                name: "edges".to_string(),
                color: None,
                visible: true,
                edges: graph.edges().to_vec(),
            }]
        } else {
            loaded_edge_types
                .into_iter()
                .map(|t| EdgeType { name: t.name, color: t.color, visible: true, edges: t.edges })
                .collect()
        };
        let init_algo_edge_set: Option<usize> = match opts.compute_over.as_deref() {
            None | Some("all") | Some("union") => None,
            Some(name) => edge_types.iter().position(|t| t.name == name),
        };
        let settings = opts.settings;
        let color_mode = opts.color_mode;
        let selected = opts.select;
        let show_help = opts.show_help;
        let show_labels = opts.show_labels;
        let node_count = graph.num_nodes();
        let aggregate = opts.aggregate;
        let init_filter = opts
            .filter
            .clone()
            .map(|(key, op, value)| NodeFilter { enabled: true, key, op, value })
            .unwrap_or_default();
        let render_params = RenderParams {
            base_radius_px: opts.node_size.clamp(0.5, 64.0),
            ..RenderParams::default()
        };
        App {
            graph,
            seed_positions,
            opts,
            attr_keys: attribute_keys(node_attrs.as_deref()),
            algo_edge_set: init_algo_edge_set,
            edge_types,
            spring_edge_set: None,
            filter: init_filter,
            filter_match_count: None,
            legend: Legend::None,
            base_colors: Vec::new(),
            base_sizes: Vec::new(),
            labels,
            node_attrs,
            live: None,
            ui: None,
            ui_frame: None,
            show_panel: true,
            camera: Camera2D::new(glam::vec2(1280.0, 800.0)),
            settings,
            render_params,
            color_mode,
            selected,
            highlight: None,
            pinned_highlight: None,
            selected_pos: None,
            neighbor_positions: Vec::new(),
            last_values: None,
            value_name: String::new(),
            show_help,
            show_hud: true,
            show_labels,
            cached_positions: None,
            // Auto-enable aggregation for graphs too large to click through.
            show_density: aggregate || node_count > 2_000_000,
            cursor: glam::Vec2::ZERO,
            dragging: false,
            press_pos: glam::Vec2::ZERO,
            moved_since_press: false,
            last_cursor: glam::Vec2::ZERO,
            last_click: Instant::now(),
            last_click_pos: glam::Vec2::splat(-1e6),
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
        let density = Density::new(&gpu.device, gpu.config.format, &graph_gpu);
        self.ui = Some(Ui::new(&gpu.device, gpu.config.format, &window));

        // Fit camera to the seeded layout.
        self.camera.viewport = glam::vec2(gpu.size.width as f32, gpu.size.height as f32);
        let (min, max) = bounds(&self.seed_positions);
        self.camera.fit_bounds(min, max);

        let mut renderer = renderer;
        renderer.draw_edges = self.opts.draw_edges;
        renderer.draw_nodes = self.opts.draw_nodes;
        renderer.edge_budget = self.opts.edge_budget.max(1);
        // Upload the edge types to the renderer.
        let specs: Vec<crate::render::EdgeTypeInput> = self
            .edge_types
            .iter()
            .map(|t| crate::render::EdgeTypeInput {
                name: &t.name,
                color: t.color,
                visible: t.visible,
                edges: &t.edges,
            })
            .collect();
        renderer.set_edge_types(&gpu.device, &graph_gpu, &specs);
        drop(specs);
        self.live = Some(Live { window, gpu, graph_gpu, renderer, layout, overlay, density });
        self.apply_color_mode();
        if self.opts.start_radial {
            self.apply_radial();
        } else if self.opts.start_hierarchical {
            self.apply_hierarchical();
        }
        // Headless: preview the legend hover highlight (most frequent class).
        if self.opts.max_frames.is_some() && self.opts.preview_highlight {
            let h = match &self.legend {
                Legend::Categorical { top, .. } => {
                    top.first().map(|&(label, _, _)| Highlight::Category(label))
                }
                Legend::Scalar { lo, hi, log, .. } => {
                    // Mimic hovering ~55% along the (log) bar, ±8%.
                    let inv = |tt: f32| {
                        if *log {
                            let a = (lo.max(0.0) + 1.0).ln();
                            let b = (hi.max(0.0) + 1.0).ln();
                            (a + tt * (b - a)).exp() - 1.0
                        } else {
                            lo + tt * (hi - lo)
                        }
                    };
                    Some(Highlight::ValueBand(inv(0.47), inv(0.63)))
                }
                _ => None,
            };
            self.set_highlight(h);
        }
        self.update_title();
    }

    fn apply_color_mode(&mut self) {
        if self.live.is_none() {
            return;
        }
        let n = self.graph.num_nodes() as usize;
        self.value_name = String::new();
        self.last_values = None;
        // Edge-aware coloring computes over the chosen algorithm edge set.
        let mut ag = self.algo_graph();
        let (colors, sizes): (Vec<u32>, Option<Vec<f32>>) = match self.color_mode {
            ColorMode::Uniform => (vec![pack_rgba(120, 170, 255, 255); n], None),
            ColorMode::Components => {
                let labels = algorithms::connected_components(&ag);
                self.value_name = "component".to_string();
                self.last_values = Some(labels.iter().map(|&l| l as f32).collect());
                (coloring::categorical_colors(&labels), None)
            }
            ColorMode::Degree => {
                let deg = ag.degrees();
                let degf: Vec<f32> = deg.iter().map(|&d| d as f32).collect();
                self.value_name = "degree".to_string();
                self.last_values = Some(degf.clone());
                (
                    coloring::sequential_colors_u32(&deg, true),
                    Some(coloring::sizes_from_scalar(&degf, 6.0)),
                )
            }
            ColorMode::PageRank => {
                let pr = algorithms::pagerank(&mut ag, 40, 0.85);
                self.value_name = "pagerank".to_string();
                self.last_values = Some(pr.clone());
                (
                    coloring::sequential_colors_f32(&pr, true),
                    Some(coloring::sizes_from_scalar(&pr, 8.0)),
                )
            }
            ColorMode::Coloring => {
                let c = algorithms::greedy_coloring(&mut ag);
                self.value_name = "color".to_string();
                self.last_values = Some(c.iter().map(|&x| x as f32).collect());
                (coloring::categorical_colors(&c), None)
            }
            ColorMode::Communities => {
                let labels = algorithms::label_propagation(&mut ag, 20);
                self.value_name = "community".to_string();
                self.last_values = Some(labels.iter().map(|&l| l as f32).collect());
                (coloring::categorical_colors(&labels), None)
            }
            ColorMode::Attribute(i) => self.colors_for_attribute(i, n),
        };
        self.base_colors = colors;
        self.base_sizes = sizes.unwrap_or_else(|| vec![1.0f32; n]);
        // Legend for the active mode (the attribute case is set in
        // `colors_for_attribute`, which knows whether it went scalar).
        match self.color_mode {
            ColorMode::Uniform => self.legend = Legend::Uniform,
            ColorMode::Degree => self.legend = Legend::scalar(&self.last_values, "degree", true),
            ColorMode::PageRank => self.legend = Legend::scalar(&self.last_values, "pagerank", true),
            ColorMode::Components => {
                self.legend = Legend::categorical(&self.last_values, "component")
            }
            ColorMode::Coloring => self.legend = Legend::categorical(&self.last_values, "color"),
            ColorMode::Communities => {
                self.legend = Legend::categorical(&self.last_values, "community")
            }
            ColorMode::Attribute(_) => {}
        }
        self.push_filtered();
        log::info!("color mode -> {}", self.color_mode.label());
    }

    /// Compute the filter mask (`true` = visible), or `None` when disabled.
    fn filter_mask(&self, n: usize) -> Option<Vec<bool>> {
        if !self.filter.enabled {
            return None;
        }
        let key = self.attr_keys.get(self.filter.key)?.clone();
        let attrs = self.node_attrs.as_ref()?;
        let f = &self.filter;
        Some(
            (0..n)
                .map(|id| {
                    let val = attrs
                        .get(id)
                        .and_then(|kv| kv.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str()))
                        .unwrap_or("");
                    f.op.eval(val, &f.value)
                })
                .collect(),
        )
    }

    /// Push `base_colors`/`base_sizes` to the GPU, applying the attribute filter
    /// (hides non-matching) and the legend-hover highlight (fades nodes that
    /// don't match the hovered legend entry — keeps only "that color" bright).
    fn push_filtered(&mut self) {
        let n = self.graph.num_nodes() as usize;
        let mask = self.filter_mask(n);
        self.filter_match_count = mask.as_ref().map(|m| m.iter().filter(|&&v| v).count());

        // Per-node highlight match against the hovered legend entry.
        let matched: Option<Vec<bool>> = self.highlight.map(|hl| {
            let vals = self.last_values.as_ref();
            (0..n)
                .map(|i| {
                    let v = vals.and_then(|v| v.get(i)).copied();
                    match (hl, v) {
                        (Highlight::Category(c), Some(x)) => x == c,
                        (Highlight::ValueBand(lo, hi), Some(x)) => x >= lo && x <= hi,
                        _ => false,
                    }
                })
                .collect()
        });

        let Some(live) = self.live.as_ref() else { return };

        if mask.is_none() && matched.is_none() {
            // Fast path: nothing to modify.
            live.graph_gpu.set_colors(&live.gpu.queue, &self.base_colors);
            live.graph_gpu.set_sizes(&live.gpu.queue, &self.base_sizes);
            return;
        }

        let mut colors = vec![0u32; n];
        let mut sizes = vec![0f32; n];
        for i in 0..n {
            if !mask.as_ref().map_or(true, |m| m[i]) {
                continue; // hidden by filter (color 0, size 0)
            }
            let base = self.base_colors[i];
            match &matched {
                Some(m) if !m[i] => {
                    colors[i] = grey_color(base);
                    sizes[i] = self.base_sizes[i] * 0.7;
                }
                _ => {
                    colors[i] = base;
                    sizes[i] = self.base_sizes[i];
                }
            }
        }
        live.graph_gpu.set_colors(&live.gpu.queue, &colors);
        live.graph_gpu.set_sizes(&live.gpu.queue, &sizes);
    }

    /// Update the legend-hover highlight and re-push colors if it changed.
    fn set_highlight(&mut self, h: Option<Highlight>) {
        if h != self.highlight {
            self.highlight = h;
            self.push_filtered();
        }
    }

    /// Build `(colors, sizes)` for coloring by attribute key index `i`. Numeric
    /// attributes get a turbo scalar ramp; everything else (booleans, strings)
    /// is colored categorically.
    fn colors_for_attribute(&mut self, i: usize, n: usize) -> (Vec<u32>, Option<Vec<f32>>) {
        let Some(key) = self.attr_keys.get(i).cloned() else {
            return (vec![pack_rgba(120, 170, 255, 255); n], None);
        };
        self.value_name = key.clone();

        // Snapshot the attribute's value for each node (owned), so the borrow of
        // `self.node_attrs` is released before we mutate `self` below.
        let vals: Vec<Option<String>> = (0..n)
            .map(|id| {
                self.node_attrs
                    .as_ref()
                    .and_then(|a| a.get(id))
                    .and_then(|kv| kv.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()))
            })
            .collect();

        // Numeric if every present value parses as a number (and at least one is).
        let nums: Vec<Option<f32>> = vals
            .iter()
            .map(|o| o.as_ref().and_then(|s| s.trim().parse::<f32>().ok()))
            .collect();
        let any_present = vals.iter().any(|o| o.is_some());
        let all_numeric = vals.iter().zip(&nums).all(|(v, n)| v.is_none() || n.is_some());

        if any_present && all_numeric {
            let scalar: Vec<f32> = nums.iter().map(|o| o.unwrap_or(0.0)).collect();
            self.last_values = Some(scalar.clone());
            self.legend = Legend::scalar(&self.last_values, &key, true);
            // Log scale: counts (read_count, parent_count, …) are heavily skewed.
            return (coloring::sequential_colors_f32(&scalar, true), None);
        }

        // Categorical: intern distinct string values into dense labels.
        let mut map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        let mut labels = vec![0u32; n];
        for (id, v) in vals.into_iter().enumerate() {
            let key = v.unwrap_or_default();
            let next = map.len() as u32;
            labels[id] = *map.entry(key).or_insert(next);
        }
        self.last_values = Some(labels.iter().map(|&l| l as f32).collect());
        self.legend = Legend::categorical(&self.last_values, &key);
        (coloring::categorical_colors(&labels), None)
    }

    fn update_title(&self) {
        let Some(live) = self.live.as_ref() else { return };
        let title = format!(
            "{} — {} nodes · {} edges · {} · {} · {:.0} fps · steps {}{}",
            self.opts.title,
            self.graph.num_nodes(),
            self.graph.num_edges(),
            self.color_mode.label(),
            self.sim_status(),
            self.fps,
            self.total_steps,
            if self.render_params.base_radius_px > 0.0 { "" } else { "" },
        );
        live.window.set_title(&title);
    }

    /// Run the egui control panel for this frame: feed input, build the panel,
    /// tessellate. Stores the paint jobs in `self.ui_frame` for `render`.
    fn run_ui(&mut self) {
        let Some(mut ui) = self.ui.take() else { return };
        let Some(window) = self.live.as_ref().map(|l| l.window.clone()) else {
            self.ui = Some(ui);
            return;
        };
        let raw = ui.state.take_egui_input(window.as_ref());
        let ctx = ui.ctx.clone();
        let full = ctx.run(raw, |ctx| self.build_ui(ctx));
        ui.state.handle_platform_output(window.as_ref(), full.platform_output);
        let jobs = ctx.tessellate(full.shapes, full.pixels_per_point);
        self.ui_frame = Some(UiFrame {
            jobs,
            textures_delta: full.textures_delta,
            pixels_per_point: full.pixels_per_point,
        });
        self.ui = Some(ui);
    }

    /// Build the control panel. Widgets edit local snapshots (never `self`
    /// directly, to avoid borrow conflicts inside egui's nested closures); the
    /// diffs are applied to `self` after the panel closes.
    fn build_ui(&mut self, ctx: &egui::Context) {
        if !self.show_panel {
            // Panel hidden: no hover preview, but keep any pinned highlight.
            self.set_highlight(self.pinned_highlight);
            return;
        }

        let mut running = self.settings.running;
        let mut substeps = self.settings.substeps;
        let mut node_size = self.render_params.base_radius_px;
        let mut edge_alpha = self.render_params.edge_alpha;
        let mut draw_edges = self.live.as_ref().map(|l| l.renderer.draw_edges).unwrap_or(true);
        let mut draw_nodes = self.live.as_ref().map(|l| l.renderer.draw_nodes).unwrap_or(true);
        let mut edge_budget =
            self.live.as_ref().map(|l| l.renderer.edge_budget).unwrap_or(1_000_000);
        let visible_edges =
            self.live.as_ref().map(|l| l.renderer.visible_edge_count()).unwrap_or(0);
        let mut show_density = self.show_density;
        let mut show_labels = self.show_labels;
        let cur_color = self.color_mode;
        let n = self.graph.num_nodes();
        let e = self.graph.num_edges();
        let fps = self.fps;
        let status = self.sim_status();
        let mut filter_enabled = self.filter.enabled;
        let mut filter_key = self.filter.key;
        let mut filter_op = self.filter.op;
        let mut filter_value = self.filter.value.clone();
        let filter_count = self.filter_match_count;
        let legend = self.legend.clone();
        let edge_info: Vec<(String, Option<u32>, usize)> = self
            .edge_types
            .iter()
            .map(|t| (t.name.clone(), t.color, t.edges.len()))
            .collect();
        let mut edge_visible: Vec<bool> = self.edge_types.iter().map(|t| t.visible).collect();
        let mut algo_sel = self.algo_edge_set;
        let spring_set = self.spring_edge_set;

        // Actions requiring heavier work (GPU recompute) are recorded and run
        // after the panel closure, when `self` is freely borrowable again.
        let mut act_fit = false;
        let mut act_color: Option<ColorMode> = None;
        let mut act_reseed: Option<Seed> = None;
        let pinned = self.pinned_highlight;
        let mut legend_iact = LegendInteraction::default();

        egui::SidePanel::left("nebula_controls")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("nebula");
                ui.label(format!("{n} nodes · {e} edges"));
                ui.label(format!("{fps:.0} fps · {status}"));
                ui.separator();

                ui.strong("Simulation");
                ui.horizontal(|ui| {
                    if ui.button(if running { "⏸ Pause" } else { "▶ Resume" }).clicked() {
                        running = !running;
                    }
                    if ui.button("Fit view").clicked() {
                        act_fit = true;
                    }
                });
                ui.add(egui::Slider::new(&mut substeps, 1..=16).text("steps / frame"));
                ui.separator();

                ui.strong("Layout");
                ui.horizontal(|ui| {
                    if ui
                        .button("⬇ Hierarchical")
                        .on_hover_text("Layered top-down DAG layout by dependency depth; pauses physics")
                        .clicked()
                    {
                        act_reseed = Some(Seed::Hierarchical);
                    }
                    if ui
                        .button("◎ Radial")
                        .on_hover_text("Concentric DAG layout: root at center, layers as rings; pauses physics")
                        .clicked()
                    {
                        act_reseed = Some(Seed::Radial);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Force reseed:");
                    if ui.button("Random").clicked() {
                        act_reseed = Some(Seed::Random);
                    }
                    if ui.button("Grid").clicked() {
                        act_reseed = Some(Seed::Grid);
                    }
                    if ui.button("Circle").clicked() {
                        act_reseed = Some(Seed::Circle);
                    }
                });
                ui.separator();

                ui.strong("Color by");
                for (mode, label) in COLOR_MODES {
                    if ui.selectable_label(cur_color == mode, label).clicked() {
                        act_color = Some(mode);
                    }
                }
                if !self.attr_keys.is_empty() {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("attributes").weak());
                    for (i, k) in self.attr_keys.iter().enumerate() {
                        let mode = ColorMode::Attribute(i);
                        if ui.selectable_label(cur_color == mode, k).clicked() {
                            act_color = Some(mode);
                        }
                    }
                }
                legend_iact = draw_legend(ui, &legend, pinned);
                ui.separator();

                ui.strong("Display");
                ui.checkbox(&mut draw_nodes, "Nodes");
                ui.checkbox(&mut draw_edges, "Edges");
                ui.checkbox(&mut show_density, "Aggregate (density LOD)");
                ui.checkbox(&mut show_labels, "Labels");
                ui.add(
                    egui::Slider::new(&mut node_size, 0.5..=64.0)
                        .logarithmic(true)
                        .text("node size"),
                );
                ui.add(
                    egui::Slider::new(&mut edge_alpha, 0.01..=1.0)
                        .logarithmic(true)
                        .text("edge glow"),
                );
                if draw_edges && visible_edges > 100_000 {
                    ui.add(
                        egui::Slider::new(&mut edge_budget, 50_000..=16_000_000)
                            .logarithmic(true)
                            .text("edge budget"),
                    );
                    if (edge_budget as u64) < visible_edges {
                        ui.label(
                            egui::RichText::new(format!(
                                "sampling {} of {} edges",
                                commafy(edge_budget as u64),
                                commafy(visible_edges)
                            ))
                            .weak()
                            .small(),
                        );
                    }
                }

                // Edge types: per-type render visibility + color, and which edge
                // set the edge-aware algorithms compute over.
                if edge_info.len() > 1 {
                    ui.separator();
                    ui.strong("Edge types");
                    ui.label(egui::RichText::new("show").weak().small());
                    for (i, (name, color, count)) in edge_info.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut edge_visible[i], "");
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(13.0, 13.0), egui::Sense::hover());
                            let col = match color {
                                Some(c) => egui_rgb(*c),
                                None => egui::Color32::from_gray(150),
                            };
                            ui.painter().rect_filled(rect, 2.0, col);
                            ui.label(format!("{name}  ({count})"));
                        });
                    }
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new("compute over").weak().small());
                    ui.horizontal_wrapped(|ui| {
                        if ui.selectable_label(algo_sel.is_none(), "all").clicked() {
                            algo_sel = None;
                        }
                        for (i, (name, _, _)) in edge_info.iter().enumerate() {
                            if ui.selectable_label(algo_sel == Some(i), name).clicked() {
                                algo_sel = Some(i);
                            }
                        }
                    });
                    // Nudge when the pending selection hasn't been applied yet.
                    let dag_stale = algo_sel != self.algo_edge_set;
                    let force_stale = algo_sel != spring_set;
                    if dag_stale || force_stale {
                        ui.label(
                            egui::RichText::new("recolor or re-layout to apply")
                                .weak()
                                .small()
                                .italics(),
                        );
                    }
                }

                if !self.attr_keys.is_empty() {
                    ui.separator();
                    ui.strong("Filter (show only)");
                    ui.checkbox(&mut filter_enabled, "Enable filter");
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("filter_key")
                            .selected_text(
                                self.attr_keys
                                    .get(filter_key)
                                    .map(|s| s.as_str())
                                    .unwrap_or("—"),
                            )
                            .show_ui(ui, |ui| {
                                for (i, k) in self.attr_keys.iter().enumerate() {
                                    ui.selectable_value(&mut filter_key, i, k);
                                }
                            });
                        egui::ComboBox::from_id_salt("filter_op")
                            .width(72.0)
                            .selected_text(filter_op.label())
                            .show_ui(ui, |ui| {
                                for op in FilterOp::ALL {
                                    ui.selectable_value(&mut filter_op, op, op.label());
                                }
                            });
                    });
                    ui.add(
                        egui::TextEdit::singleline(&mut filter_value)
                            .hint_text("value (e.g. true, 5, resolve)"),
                    );
                    if filter_enabled {
                        match filter_count {
                            Some(c) => {
                                ui.label(egui::RichText::new(format!("{c} of {n} shown")).weak());
                            }
                            None => {
                                ui.label(egui::RichText::new("…").weak());
                            }
                        }
                    }
                }
            });

        // --- apply diffs ---
        if running != self.settings.running {
            self.settings.running = running;
            if running {
                self.settings.alpha = self.settings.alpha.max(self.settings.alpha_reheat);
            }
        }
        self.settings.substeps = substeps;
        if node_size != self.render_params.base_radius_px {
            self.render_params.base_radius_px = node_size;
            self.push_params();
        }
        if edge_alpha != self.render_params.edge_alpha {
            self.render_params.edge_alpha = edge_alpha;
            self.push_params();
        }
        self.show_density = show_density;
        self.show_labels = show_labels;
        self.algo_edge_set = algo_sel.filter(|&i| i < self.edge_types.len());
        for (i, &v) in edge_visible.iter().enumerate() {
            if self.edge_types.get(i).map(|t| t.visible) != Some(v) {
                if let Some(t) = self.edge_types.get_mut(i) {
                    t.visible = v;
                }
                if let Some(live) = self.live.as_mut() {
                    live.renderer.set_edge_visible(i, v);
                }
            }
        }
        if let Some(live) = self.live.as_mut() {
            live.renderer.draw_edges = draw_edges;
            live.renderer.draw_nodes = draw_nodes;
            live.renderer.edge_budget = edge_budget.max(1);
        }
        let filter_changed = filter_enabled != self.filter.enabled
            || filter_key != self.filter.key
            || filter_op != self.filter.op
            || filter_value != self.filter.value;
        if filter_changed {
            self.filter.enabled = filter_enabled;
            self.filter.key = filter_key;
            self.filter.op = filter_op;
            self.filter.value = filter_value;
            self.push_filtered();
        }
        if act_fit {
            self.fit_view();
        }
        if let Some(m) = act_color {
            self.set_color_mode(m);
        }
        // Clicking a legend entry toggles the persistent pin.
        if let Some(clicked) = legend_iact.clicked {
            self.pinned_highlight =
                if self.pinned_highlight == Some(clicked) { None } else { Some(clicked) };
        }
        // In headless capture there's no live pointer; keep any preset preview.
        // Otherwise the effective highlight is the hover preview, else the pin.
        if self.opts.max_frames.is_none() {
            let effective = legend_iact.hovered.or(self.pinned_highlight);
            self.set_highlight(effective);
        }
        if let Some(seed) = act_reseed {
            match seed {
                Seed::Random => {
                    let ext = self.opts.k * (n.max(1) as f32).sqrt();
                    self.reseed(&RandomLayout { extent: ext });
                }
                Seed::Grid => self.reseed(&GridLayout { spacing: self.opts.k }),
                Seed::Circle => {
                    let r = self.opts.k * (n.max(1) as f32).sqrt() * 0.5;
                    self.reseed(&CircleLayout { radius: r });
                }
                Seed::Hierarchical => self.apply_hierarchical(),
                Seed::Radial => self.apply_radial(),
            }
        }
    }

    fn render(&mut self) {
        if self.live.is_none() {
            return;
        }
        self.run_ui();

        // Refresh selection geometry (node + neighbor positions) so the marker,
        // info panel, and connection lines track the live simulation.
        if let Some(sel) = self.selected {
            self.refresh_selection(sel);
        }
        // For label rendering on modest graphs, keep a CPU copy of positions.
        if self.show_labels && self.graph.num_nodes() <= 50_000 {
            let pos = self
                .live
                .as_ref()
                .and_then(|live| crate::readback::read_positions(&live.gpu, &live.graph_gpu));
            if pos.is_some() {
                self.cached_positions = pos;
            }
        } else {
            self.cached_positions = None;
        }
        // Build overlay commands before borrowing live mutably.
        let overlay_cmds = self.build_overlay();

        // Take the egui state out of `self` so it can be used alongside the
        // `&mut self.live` borrow below (disjoint owned locals, no conflict).
        let mut ui_taken = self.ui.take();
        let ui_frame = self.ui_frame.take();

        let live = self.live.as_mut().unwrap();

        // Advance layout, cooling alpha so the simulation converges and stops.
        if self.settings.running {
            let decay = (1.0 - self.settings.alpha_decay).powi(self.settings.substeps as i32);
            self.settings.alpha *= decay;
            live.layout.update_settings(&live.gpu.queue, &self.settings);

            let mut enc = live
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim") });
            for _ in 0..self.settings.substeps {
                live.layout.step(&mut enc);
                self.total_steps += 1;
            }
            live.gpu.queue.submit(Some(enc.finish()));

            if self.settings.alpha <= self.settings.alpha_min {
                self.settings.running = false;
                log::info!("layout settled after {} steps", self.total_steps);
            }
        }

        // Camera uniform.
        live.renderer.update_camera(&live.gpu.queue, &self.camera.uniform());
        let params = effective_params(&self.render_params, &live.renderer);
        live.renderer.update_params(&live.gpu.queue, &params);
        if self.show_density {
            let (vw, vh) = (live.gpu.size.width as f32, live.gpu.size.height as f32);
            live.density.update(&live.gpu.queue, &self.camera.uniform(), vw, vh);
        }

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
        // Bin nodes into screen tiles before the render pass (same encoder).
        if self.show_density {
            live.density.record_compute(&mut enc);
        }
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
            if self.show_density {
                live.density.draw(&mut pass);
            } else {
                live.renderer.draw(&mut pass);
            }

            // Overlay (HUD, info panel, selection marker) on top.
            live.overlay.begin();
            for (x, y, w, h, col) in &overlay_cmds.rects {
                live.overlay.rect(*x, *y, *w, *h, *col);
            }
            for (a, b, th, col) in &overlay_cmds.lines {
                live.overlay.line(*a, *b, *th, *col);
            }
            for (x, y, s, col, txt) in &overlay_cmds.texts {
                live.overlay.text(*x, *y, *s, *col, txt);
            }
            let vp = (live.gpu.size.width as f32, live.gpu.size.height as f32);
            live.overlay
                .draw(&live.gpu.device, &live.gpu.queue, vp, &mut pass);
        }

        // egui control panel, composited over the scene + overlay.
        let size = (live.gpu.size.width, live.gpu.size.height);
        if let (Some(ui), Some(uf)) = (ui_taken.as_mut(), ui_frame.as_ref()) {
            ui.record(&live.gpu.device, &live.gpu.queue, size, &mut enc, &view, uf);
        }

        live.gpu.queue.submit(Some(enc.finish()));
        frame.present();

        // Free egui textures released this frame, then restore the UI state.
        if let (Some(ui), Some(uf)) = (ui_taken.as_mut(), ui_frame.as_ref()) {
            ui.free_textures(uf);
        }
        self.ui = ui_taken;

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

        // --- Top-left stats HUD (hidden when the egui panel shows the same) ---
        if self.show_hud && !self.show_panel {
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
                        "{:.0} fps   {}   {}{}",
                        self.fps,
                        self.color_mode.label(),
                        self.sim_status(),
                        if self.show_density { "   [aggregated]" } else { "" }
                    ),
                ),
            ];
            self.panel(&mut c, pad, pad, scale, line, panel_bg, &lines);

            // Gradient legend for scalar color modes.
            if matches!(self.color_mode, ColorMode::Degree | ColorMode::PageRank) {
                if let Some((lo, hi)) = self.value_range() {
                    let lx = pad;
                    // Below the stats panel (3 lines + padding) with room for a label.
                    let ly = pad + 4.0 * line + 30.0;
                    let bar_w = 180.0 * (scale / 2.0).max(0.7);
                    let bar_h = 10.0 * scale.max(1.0);
                    let steps = 48;
                    for i in 0..steps {
                        let t = i as f32 / (steps - 1) as f32;
                        let col = coloring::turbo_rgba(t);
                        c.rects.push((lx + t * bar_w, ly, bar_w / steps as f32 + 1.0, bar_h, col));
                    }
                    let ts = scale * 0.85;
                    c.texts.push((lx, ly + bar_h + 3.0, ts, dim, format!("{lo:.3}")));
                    let hs = format!("{hi:.3}");
                    c.texts.push((
                        lx + bar_w - crate::overlay::Overlay::text_width(&hs, ts),
                        ly + bar_h + 3.0,
                        ts,
                        dim,
                        hs,
                    ));
                    c.texts.push((lx, ly - line, scale * 0.9, dim, self.value_name.to_string()));
                }
            }
        }

        // --- Bottom-left controls (hidden when the egui panel is up) ---
        if self.show_help && !self.show_panel {
            let help = [
                "drag pan / scroll zoom / F fit",
                "click a node to inspect it",
                "1 uniform  2 components  3 degree",
                "4 pagerank  5 coloring  6 communities",
                "space pause / E edges / N nodes",
                "Y hierarchical / U radial / RGO seed",
                "A aggregate (density) / L labels",
                "+/- node size / [ ] edge brightness",
                "P panel / H help / Tab hud / Esc quit",
            ];
            let lines: Vec<(u32, String)> = help.iter().map(|s| (dim, s.to_string())).collect();
            let h = help.len() as f32 * line + 12.0;
            self.panel(&mut c, 10.0, vp.y - h - 10.0, scale, line, panel_bg, &lines);
        } else if !self.show_panel {
            c.texts.push((10.0, vp.y - line - 6.0, scale, dim, "H  help".to_string()));
        }

        // --- Node labels (only when few are on screen, to avoid clutter) ---
        if self.show_labels {
            if let Some(pos) = self.cached_positions.as_ref() {
                let lscale = scale * 0.75;
                let mut visible: Vec<(glam::Vec2, u32)> = Vec::new();
                for (i, p) in pos.iter().enumerate() {
                    let sp = self.camera.world_to_screen(glam::vec2(p[0], p[1]));
                    if sp.x >= 0.0 && sp.x <= vp.x && sp.y >= 0.0 && sp.y <= vp.y {
                        visible.push((sp, i as u32));
                        if visible.len() > 400 {
                            break; // too many on screen -> skip labels this frame
                        }
                    }
                }
                if visible.len() <= 400 {
                    for (sp, i) in visible {
                        let label = self
                            .labels
                            .as_ref()
                            .and_then(|l| l.get(i as usize))
                            .cloned()
                            .unwrap_or_else(|| i.to_string());
                        c.texts.push((sp.x + 6.0, sp.y - 4.0, lscale, white, label));
                    }
                }
            }
        }

        // --- Selected node info panel + marker ---
        if let (Some(id), Some(wp)) = (self.selected, self.selected_pos) {
            let mut lines: Vec<(u32, String)> = Vec::new();
            // Per-node attributes carried from the source file (e.g. JSON).
            let attrs = self.node_attrs.as_ref().and_then(|a| a.get(id as usize));
            // Prefer a meaningful attribute (`ty`/`type`/`name`/`label`) as the
            // panel title; fall back to the interned label or the numeric id.
            let title = attrs
                .and_then(|a| {
                    a.iter()
                        .find(|(k, _)| matches!(k.as_str(), "ty" | "type" | "name" | "label"))
                        .map(|(_, v)| v.clone())
                })
                .or_else(|| {
                    self.labels
                        .as_ref()
                        .and_then(|l| l.get(id as usize))
                        .cloned()
                })
                .unwrap_or_else(|| id.to_string());
            lines.push((accent, title));
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
                    s.push_str(&format!(" +{} more", nbrs.len() - 8));
                }
                lines.push((dim, format!("adj: {s}")));
            }
            if self.value_name != "degree"
                && !self.value_name.is_empty()
                && !matches!(self.color_mode, ColorMode::Attribute(_))
            {
                if let Some(v) = self.last_values.as_ref().and_then(|vals| vals.get(id as usize)) {
                    // Integer-valued modes read cleaner without decimals.
                    if matches!(self.color_mode, ColorMode::Components | ColorMode::Coloring | ColorMode::Communities) {
                        lines.push((white, format!("{} {}", self.value_name, *v as i64)));
                    } else {
                        lines.push((white, format!("{} {:.5}", self.value_name, v)));
                    }
                }
            }
            // Source attributes: one line per key/value (title field already
            // shown above, so skip it here to avoid duplication).
            if let Some(attrs) = attrs {
                for (k, v) in attrs {
                    if matches!(k.as_str(), "ty" | "type" | "name" | "label") {
                        continue;
                    }
                    lines.push((white, format!("{k}: {v}")));
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

            // Connection lines to neighbors + a ring on each neighbor.
            let sp = self.camera.world_to_screen(wp);
            let link = pack_rgba(255, 220, 120, 90);
            let nbr_ring = pack_rgba(255, 210, 90, 230);
            for np in &self.neighbor_positions {
                let ns = self.camera.world_to_screen(*np);
                c.lines.push((sp, ns, 1.5, link));
                let rr = 5.0;
                c.rects.push((ns.x - rr, ns.y - rr, 2.0 * rr, 1.5, nbr_ring));
                c.rects.push((ns.x - rr, ns.y + rr - 1.5, 2.0 * rr, 1.5, nbr_ring));
                c.rects.push((ns.x - rr, ns.y - rr, 1.5, 2.0 * rr, nbr_ring));
                c.rects.push((ns.x + rr - 1.5, ns.y - rr, 1.5, 2.0 * rr, nbr_ring));
            }

            // Marker: a bright square outline around the selected node.
            let r = 13.0;
            let t = 2.5;
            let ring = accent;
            c.rects.push((sp.x - r, sp.y - r, 2.0 * r, t, ring)); // top
            c.rects.push((sp.x - r, sp.y + r - t, 2.0 * r, t, ring)); // bottom
            c.rects.push((sp.x - r, sp.y - r, t, 2.0 * r, ring)); // left
            c.rects.push((sp.x + r - t, sp.y - r, t, 2.0 * r, ring)); // right
        }

        c
    }

    /// Human-readable simulation status for the HUD.
    fn sim_status(&self) -> String {
        if self.settings.running {
            format!("cooling a={:.3}", self.settings.alpha)
        } else if self.settings.alpha <= self.settings.alpha_min {
            "settled".to_string()
        } else {
            "paused".to_string()
        }
    }

    /// Min/max of the current scalar values (for the legend).
    fn value_range(&self) -> Option<(f32, f32)> {
        let vals = self.last_values.as_ref()?;
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &v in vals {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if lo <= hi {
            Some((lo, hi))
        } else {
            None
        }
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
        if let Some(path) = self.opts.screenshot.clone() {
            self.save_frame(&path);
        }
    }

    /// Render the real scene (graph + overlay) to an offscreen texture matching
    /// the swapchain format and save it as a PNG — so the capture includes the HUD.
    fn save_frame(&mut self, path: &str) {
        // Refresh the egui panel so captured screenshots include the UI.
        self.run_ui();
        let overlay_cmds = self.build_overlay();
        let mut ui_taken = self.ui.take();
        let ui_frame = self.ui_frame.take();
        let Some(live) = self.live.as_mut() else {
            self.ui = ui_taken;
            return;
        };
        let (w, h) = (live.gpu.size.width, live.gpu.size.height);
        let format = live.gpu.config.format;

        live.renderer.update_camera(&live.gpu.queue, &self.camera.uniform());
        let eff_params = effective_params(&self.render_params, &live.renderer);
        live.renderer.update_params(&live.gpu.queue, &eff_params);
        if self.show_density {
            live.density.update(&live.gpu.queue, &self.camera.uniform(), w as f32, h as f32);
        }

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
        if self.show_density {
            live.density.record_compute(&mut enc);
        }
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
            if self.show_density {
                live.density.draw(&mut pass);
            } else {
                live.renderer.draw(&mut pass);
            }
            live.overlay.begin();
            for (x, y, rw, rh, col) in &overlay_cmds.rects {
                live.overlay.rect(*x, *y, *rw, *rh, *col);
            }
            for (a, b, th, col) in &overlay_cmds.lines {
                live.overlay.line(*a, *b, *th, *col);
            }
            for (x, y, s, col, txt) in &overlay_cmds.texts {
                live.overlay.text(*x, *y, *s, *col, txt);
            }
            live.overlay
                .draw(&live.gpu.device, &live.gpu.queue, (w as f32, h as f32), &mut pass);
        }

        // Composite the egui panel into the capture too.
        if let (Some(ui), Some(uf)) = (ui_taken.as_mut(), ui_frame.as_ref()) {
            ui.record(&live.gpu.device, &live.gpu.queue, (w, h), &mut enc, &view, uf);
        }

        live.gpu.queue.submit(Some(enc.finish()));
        if let (Some(ui), Some(uf)) = (ui_taken.as_mut(), ui_frame.as_ref()) {
            ui.free_textures(uf);
        }
        self.ui = ui_taken;

        if let Err(e) = crate::screenshot::save_texture(&live.gpu, &tex, w, h, format, &path) {
            log::error!("screenshot failed: {e}");
        }
    }

    /// Re-seed node positions from a one-shot CPU layout and restart the sim.
    /// Demonstrates the pluggable `Layout` trait at runtime.
    /// Rebuild the GPU force springs to reflect the current algorithm edge set,
    /// if it differs from what the springs currently use. Called on force reseed.
    fn rebuild_spring_edges(&mut self) {
        if self.spring_edge_set == self.algo_edge_set {
            return;
        }
        let g = self.algo_graph();
        if let (Some(live), Some(csr)) = (self.live.as_mut(), g.csr_ref()) {
            live.graph_gpu.set_csr(&live.gpu.device, csr);
            live.layout.rebind(&live.gpu.device, &live.graph_gpu);
        }
        self.spring_edge_set = self.algo_edge_set;
        log::info!("force springs now use '{}' edges", self.algo_edge_set_name());
    }

    fn reseed(&mut self, layout: &dyn Layout) {
        self.rebuild_spring_edges();
        let n = self.graph.num_nodes() as usize;
        let mut pos = vec![[0.0f32; 2]; n];
        let seed = 0x51ED ^ self.total_steps;
        layout.place(&self.graph, &mut pos, seed);
        if let Some(live) = self.live.as_ref() {
            live.graph_gpu.set_positions(&live.gpu.queue, &pos);
        }
        let (min, max) = bounds(&pos);
        self.camera.fit_bounds(min, max);
        self.settings.running = true;
        self.settings.alpha = 1.0; // full reheat: lay out from scratch
        if let Some(live) = self.live.as_ref() {
            live.layout.update_settings(&live.gpu.queue, &self.settings);
        }
        log::info!("re-seeded with {} layout", layout.name());
    }

    /// A `Graph` over the edge set the algorithms should compute on: the union
    /// (`algo_edge_set == None`) or a single edge type. CSR is built.
    fn algo_graph(&self) -> Graph {
        let n = self.graph.num_nodes();
        let mut g = match self.algo_edge_set.and_then(|i| self.edge_types.get(i)) {
            Some(t) => Graph::new(n, t.edges.clone()),
            None => Graph::new(n, self.graph.edges().to_vec()),
        };
        g.ensure_csr();
        g
    }

    /// Human name of the current algorithm edge set.
    fn algo_edge_set_name(&self) -> &str {
        match self.algo_edge_set.and_then(|i| self.edge_types.get(i)) {
            Some(t) => &t.name,
            None => "all",
        }
    }

    /// Apply a fixed (non-physics) layout over `graph`: place nodes, upload, fit
    /// the view, and stop the simulation so the arrangement stays put.
    fn apply_fixed_layout(&mut self, layout: &dyn Layout, graph: &Graph) {
        let n = self.graph.num_nodes() as usize;
        let mut pos = vec![[0.0f32; 2]; n];
        layout.place(graph, &mut pos, 0);
        if let Some(live) = self.live.as_ref() {
            live.graph_gpu.set_positions(&live.gpu.queue, &pos);
        }
        let (min, max) = bounds(&pos);
        self.camera.fit_bounds(min, max);
        self.settings.running = false;
        if let Some(live) = self.live.as_ref() {
            live.layout.update_settings(&live.gpu.queue, &self.settings);
        }
        log::info!(
            "applied {} layout over '{}' edges (simulation paused)",
            layout.name(),
            self.algo_edge_set_name()
        );
    }

    /// Hierarchical (layered DAG) layout over the algorithm edge set.
    fn apply_hierarchical(&mut self) {
        let layout = LayeredLayout {
            target_height: self.opts.k * (self.graph.num_nodes().max(1) as f32).sqrt(),
            aspect: 1.8,
            sweeps: 4,
        };
        let g = self.algo_graph();
        self.apply_fixed_layout(&layout, &g);
    }

    /// Radial (concentric) DAG layout over the algorithm edge set.
    fn apply_radial(&mut self) {
        let layout = RadialLayout {
            radius: self.opts.k * (self.graph.num_nodes().max(1) as f32).sqrt() * 0.6,
            sweeps: 4,
        };
        let g = self.algo_graph();
        self.apply_fixed_layout(&layout, &g);
    }

    /// Read back positions needed to draw the selection: the node itself and its
    /// neighbors, so we can draw connection lines. Uses a gather readback whose
    /// cost scales with the neighbor cap, not the graph, so any graph size gets
    /// live neighbor lines.
    fn refresh_selection(&mut self, sel: u32) {
        const MAX_LINES: usize = 512;
        let mut indices: Vec<u32> = vec![sel];
        if let Some(csr) = self.graph.csr_ref() {
            indices.extend(csr.neighbors(sel).iter().copied().take(MAX_LINES));
        }
        let pos = self
            .live
            .as_ref()
            .and_then(|live| crate::readback::read_positions_at(&live.gpu, &live.graph_gpu, &indices));
        if let Some(pos) = pos {
            self.selected_pos = pos.first().map(|p| glam::vec2(p[0], p[1]));
            self.neighbor_positions =
                pos.iter().skip(1).map(|p| glam::vec2(p[0], p[1])).collect();
        }
    }

    /// Zoom/center the camera on the selected node and its neighborhood.
    fn focus_selected(&mut self) {
        let Some(anchor) = self.selected_pos else { return };
        let mut min = anchor;
        let mut max = anchor;
        for np in &self.neighbor_positions {
            min = min.min(*np);
            max = max.max(*np);
        }
        // Pad so the node isn't flush against the edge; ensure a minimum span so
        // an isolated node doesn't zoom to infinity.
        let span = (max - min).max(glam::Vec2::splat(self.opts.k * 6.0));
        let c = (min + max) * 0.5;
        self.camera.fit_bounds(c - span * 0.6, c + span * 0.6);
    }

    /// Pick a node under a screen pixel and select it (updating the info panel).
    /// GPU id-pick the node under a screen pixel (no state mutation).
    fn pick_id(&self, screen: glam::Vec2) -> Option<u32> {
        let live = self.live.as_ref()?;
        // Ensure the pick pass uses the current camera.
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
    }

    fn pick_at(&mut self, screen: glam::Vec2) {
        let picked = self.pick_id(screen);
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
            // Categories/ranges differ across modes; drop any highlight + pin.
            self.highlight = None;
            self.pinned_highlight = None;
            self.apply_color_mode();
            self.update_title();
        }
    }

    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => {
                self.settings.running = !self.settings.running;
                if self.settings.running {
                    // Reheat a settled layout so resuming actually moves it.
                    self.settings.alpha = self.settings.alpha.max(self.settings.alpha_reheat);
                }
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
            KeyCode::KeyR => {
                let ext = self.opts.k * (self.graph.num_nodes().max(1) as f32).sqrt();
                self.reseed(&RandomLayout { extent: ext });
            }
            KeyCode::KeyG => {
                self.reseed(&GridLayout { spacing: self.opts.k });
            }
            KeyCode::KeyO => {
                let r = self.opts.k * (self.graph.num_nodes().max(1) as f32).sqrt() * 0.5;
                self.reseed(&CircleLayout { radius: r });
            }
            KeyCode::KeyY => self.apply_hierarchical(),
            KeyCode::KeyU => self.apply_radial(),
            KeyCode::KeyA => {
                self.show_density = !self.show_density;
                log::info!("aggregation (density heatmap) {}", if self.show_density { "on" } else { "off" });
            }
            KeyCode::KeyH => self.show_help = !self.show_help,
            KeyCode::KeyL => {
                self.show_labels = !self.show_labels;
                if self.show_labels && self.graph.num_nodes() > 50_000 {
                    log::info!("labels only render for graphs up to 50k nodes");
                }
            }
            KeyCode::KeyS => {
                let path = format!("nebula-{}.png", self.total_steps);
                self.save_frame(&path);
            }
            KeyCode::Tab => self.show_hud = !self.show_hud,
            KeyCode::KeyP => self.show_panel = !self.show_panel,
            KeyCode::KeyC => {
                self.selected = None;
                self.selected_pos = None;
            }
            KeyCode::Equal | KeyCode::NumpadAdd => {
                self.render_params.base_radius_px = (self.render_params.base_radius_px * 1.3).min(64.0);
                self.push_params();
                log::info!("node size: {:.1} px", self.render_params.base_radius_px);
            }
            KeyCode::Minus | KeyCode::NumpadSubtract => {
                self.render_params.base_radius_px = (self.render_params.base_radius_px / 1.3).max(0.5);
                self.push_params();
                log::info!("node size: {:.1} px", self.render_params.base_radius_px);
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
            let params = effective_params(&self.render_params, &live.renderer);
            live.renderer.update_params(&live.gpu.queue, &params);
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
        // Let egui see the event first. If the pointer/keyboard is over a panel
        // and egui wants the event, don't also drive the camera/selection.
        let egui_consumed = if let (Some(ui), Some(live)) = (self.ui.as_mut(), self.live.as_ref()) {
            ui.state.on_window_event(live.window.as_ref(), &event).consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(live) = self.live.as_mut() {
                    live.gpu.resize(size);
                    self.camera.viewport = glam::vec2(size.width as f32, size.height as f32);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !egui_consumed && event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        self.handle_key(code, event_loop);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if egui_consumed {
                    // Cancel any in-progress drag so releasing over a panel is clean.
                    self.dragging = false;
                } else if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            self.dragging = true;
                            self.press_pos = self.cursor;
                            self.moved_since_press = false;
                            self.last_cursor = self.cursor;
                        }
                        ElementState::Released => {
                            self.dragging = false;
                            // A click (negligible movement) selects a node; a
                            // double-click also focuses the camera on it.
                            if !self.moved_since_press {
                                self.pick_at(self.cursor);
                                let now = Instant::now();
                                let dbl = now.duration_since(self.last_click).as_millis() < 350
                                    && (self.cursor - self.last_click_pos).length() < 6.0;
                                if dbl {
                                    self.focus_selected();
                                }
                                self.last_click = now;
                                self.last_click_pos = self.cursor;
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
                if !egui_consumed {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                    };
                    let factor = (1.0 + scroll * 0.12).clamp(0.2, 5.0);
                    self.camera.zoom_about(factor, self.cursor);
                }
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
    lines: Vec<(glam::Vec2, glam::Vec2, f32, u32)>,
}

/// Render params as uploaded to the GPU: when the edge budget samples only a
/// fraction of the edges, boost `edge_alpha` by 1/fraction (capped at 1.0) so
/// the additively-accumulated brightness matches drawing everything.
fn effective_params(params: &RenderParams, renderer: &Renderer) -> RenderParams {
    let frac = renderer.edge_sample_frac();
    let mut p = *params;
    if frac < 1.0 {
        p.edge_alpha = (p.edge_alpha / frac).min(1.0);
    }
    p
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

/// Distinct attribute keys across all nodes, in first-seen order.
fn attribute_keys(attrs: Option<&[Vec<(String, String)>]>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut keys = Vec::new();
    if let Some(attrs) = attrs {
        for node in attrs {
            for (k, _) in node {
                if seen.insert(k.clone()) {
                    keys.push(k.clone());
                }
            }
        }
    }
    keys
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
