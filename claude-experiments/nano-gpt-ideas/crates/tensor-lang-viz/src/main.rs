mod model;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glyphon::{
    Attrs, Buffer as GlyphonBuffer, Cache, Color as GlyphonColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use model::{Adam, ModelRunner, TOKENS, VOCAB_SIZE};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// ─── Vertex type ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ColorVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl ColorVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ColorVertex>() as u64,
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
        }
    }
}

fn viridis(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let r = (0.267 + t * (0.993 - 0.267 + t * (-1.02 + t * 0.76))).clamp(0.0, 1.0);
    let g = (0.004 + t * (1.53 + t * (-1.74 + t * 0.76))).clamp(0.0, 1.0);
    let b = (0.329 + t * (1.16 + t * (-2.24 + t * 1.15))).clamp(0.0, 1.0);
    [r, g, b, 1.0]
}

fn px_to_ndc(px_x: f32, px_y: f32, win_w: f32, win_h: f32) -> (f32, f32) {
    (px_x / win_w * 2.0 - 1.0, 1.0 - px_y / win_h * 2.0)
}

fn quad_px(
    vertices: &mut Vec<ColorVertex>,
    px_x: f32, px_y: f32, px_w: f32, px_h: f32,
    win_w: f32, win_h: f32, color: [f32; 4],
) {
    let (x0, y0) = px_to_ndc(px_x, px_y, win_w, win_h);
    let (x1, y1) = px_to_ndc(px_x + px_w, px_y + px_h, win_w, win_h);
    vertices.extend_from_slice(&[
        ColorVertex { position: [x0, y0], color },
        ColorVertex { position: [x1, y0], color },
        ColorVertex { position: [x1, y1], color },
        ColorVertex { position: [x0, y0], color },
        ColorVertex { position: [x1, y1], color },
        ColorVertex { position: [x0, y1], color },
    ]);
}

// ─── Text helper ──────────────────────────────────────────────────────────

struct TextEntry {
    text: String,
    x: f32,
    y: f32,
    size: f32,
    color: GlyphonColor,
}

/// Collapse an ndarray to 2D for heatmap visualization.
/// Takes first element of leading dims, last two dims become rows/cols.
fn collapse_to_2d(arr: &ndarray::ArrayD<f32>) -> (usize, usize, Vec<f32>) {
    let shape = arr.shape();
    if shape.is_empty() {
        return (1, 1, arr.iter().copied().collect());
    }
    if shape.len() == 1 {
        return (1, shape[0], arr.iter().copied().collect());
    }
    // Take last two dims
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    // Index into first element of all leading dims
    let stride: usize = rows * cols;
    let data: Vec<f32> = arr.iter().take(stride).copied().collect();
    (rows, cols, data)
}

// ─── Mode ─────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
enum Mode {
    Paused,
    Training,
}

#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    Sequence,
    Graph,
    DAG,
}

// ─── Node Graph Editor ────────────────────────────────────────────────────

struct DagNode {
    name: &'static str,
    op_symbol: &'static str,
    output_node: usize,
    input_nodes: Vec<usize>,       // graph node indices for input tensor previews
    input_labels: Vec<&'static str>,
    output_label: &'static str,
    base_x: f32,                   // position before layout adjustments
    y: f32,
    header_color: [f32; 4],
    node_range: (usize, usize),    // graph node index range (inclusive)
    expand_t: f32,
    expanded: bool,
}

struct Wire {
    src_node: usize,
    dst_node: usize,
    dst_port: usize,
}

struct DagGraph {
    nodes: Vec<DagNode>,
    wires: Vec<Wire>,
}

const NODE_W: f32 = 340.0;
const NODE_H: f32 = 180.0;
const SUB_NODE_W: f32 = 160.0;
const SUB_NODE_H: f32 = 160.0;

fn build_dag_graph() -> DagGraph {
    let ci = [0.35, 0.55, 0.75, 1.0];
    let ce = [0.25, 0.50, 0.80, 1.0];
    let cn = [0.30, 0.65, 0.60, 1.0];
    let ca = [0.55, 0.35, 0.75, 1.0];
    let cm = [0.80, 0.55, 0.25, 1.0];
    let co = [0.75, 0.30, 0.30, 1.0];

    let rm = 250.0; let rt = 30.0; let rb = 470.0; let sp = 400.0;

    macro_rules! nd {
        ($name:expr, $op:expr, $out:expr, $ins:expr, $il:expr, $ol:expr,
         $x:expr, $y:expr, $c:expr, $r0:expr, $r1:expr) => {
            DagNode {
                name: $name, op_symbol: $op, output_node: $out,
                input_nodes: $ins, input_labels: $il, output_label: $ol,
                base_x: $x, y: $y, header_color: $c,
                node_range: ($r0, $r1), expand_t: 0.0, expanded: false,
            }
        }
    }

    let nodes = vec![
        nd!("Tokens",      "",           0,   vec![],        vec![],              "[1,T]",      0.0,     rt, ci, 0, 0),
        nd!("WTE Weights",  "",          1,   vec![],        vec![],              "[V,D]",      0.0,     rb, ci, 1, 1),
        nd!("Embed",        "one_hot×",  21,  vec![0, 1],    vec!["tok","wte"],   "[1,T,D]",    sp,      rm, ce, 2, 21),
        nd!("ALiBi+Causal", "mask",      45,  vec![],        vec![],              "[1,H,T,T]",  sp*4.0,  rb, ca, 22, 45),
        nd!("LayerNorm 1",  "norm",      73,  vec![21],      vec!["x"],           "[1,T,D]",    sp*2.0,  rm, cn, 58, 73),
        nd!("QKV Linear",   "Wx+b",      81,  vec![73, 48],  vec!["x","W"],       "[1,T,3D]",   sp*3.0,  rm, ca, 74, 81),
        nd!("Q K V Split",  "split",     89,  vec![81],      vec!["qkv"],         "Q [H,T,S]",  sp*4.0,  rm, ca, 82, 92),
        nd!("Attn Scores",  "Q×K^T",     102, vec![89,92,45],vec!["Q","K^T","b"], "[H,T,T]",    sp*5.0,  rm, ca, 93, 102),
        nd!("Softmax",      "σ",         111, vec![102],     vec!["scores"],      "[H,T,T]",    sp*6.0,  rm, ca, 103, 111),
        nd!("Attn×V+Proj",  "×W+res",    129, vec![111,91],  vec!["attn","V"],    "[1,T,D]",    sp*7.0,  rm, ca, 112, 129),
        nd!("LayerNorm 2",  "norm",      145, vec![129],     vec!["x"],           "[1,T,D]",    sp*8.0,  rm, cn, 130, 145),
        nd!("MLP",          "FC→GELU→FC",194, vec![145],     vec!["x"],           "[1,T,D]",    sp*9.0,  rm, cm, 146, 194),
        nd!("Final LN",     "norm",      212, vec![194],     vec!["x"],           "[1,T,D]",    sp*10.0, rm, cn, 197, 212),
        nd!("Logits",       "× wte^T",   222, vec![212, 1],  vec!["x","wte^T"],   "[1,T,V]",    sp*11.0, rm, co, 213, 222),
    ];

    let wires = vec![
        Wire { src_node: 0, dst_node: 2, dst_port: 0 },
        Wire { src_node: 1, dst_node: 2, dst_port: 1 },
        Wire { src_node: 2, dst_node: 4, dst_port: 0 },
        Wire { src_node: 4, dst_node: 5, dst_port: 0 },
        Wire { src_node: 5, dst_node: 6, dst_port: 0 },
        Wire { src_node: 6, dst_node: 7, dst_port: 0 },
        Wire { src_node: 6, dst_node: 7, dst_port: 1 },
        Wire { src_node: 3, dst_node: 7, dst_port: 2 },
        Wire { src_node: 7, dst_node: 8, dst_port: 0 },
        Wire { src_node: 8, dst_node: 9, dst_port: 0 },
        Wire { src_node: 6, dst_node: 9, dst_port: 1 },
        Wire { src_node: 9, dst_node: 10, dst_port: 0 },
        Wire { src_node: 10, dst_node: 11, dst_port: 0 },
        Wire { src_node: 11, dst_node: 12, dst_port: 0 },
        Wire { src_node: 12, dst_node: 13, dst_port: 0 },
        Wire { src_node: 1, dst_node: 13, dst_port: 1 },
    ];

    DagGraph { nodes, wires }
}

const SUB_COL_GAP: f32 = 40.0;
const SUB_ROW_GAP: f32 = 24.0;

fn node_effective_w(node: &DagNode, z: f32, graph: &tensor_lang_graph::Graph) -> f32 {
    if node.node_range.1 <= node.node_range.0 { return NODE_W * z; }
    let layout = compute_sub_layout(graph, node.node_range.0, node.node_range.1);
    let expanded_w = (layout.n_cols as f32 * (SUB_NODE_W + SUB_COL_GAP) + 30.0).max(NODE_W);
    lerp(NODE_W, expanded_w, node.expand_t) * z
}

fn node_effective_h(node: &DagNode, z: f32, graph: &tensor_lang_graph::Graph) -> f32 {
    if node.node_range.1 <= node.node_range.0 { return NODE_H * z; }
    let layout = compute_sub_layout(graph, node.node_range.0, node.node_range.1);
    let expanded_h = (layout.n_rows as f32 * (SUB_NODE_H + SUB_ROW_GAP) + 60.0).max(NODE_H);
    lerp(NODE_H, expanded_h, node.expand_t) * z
}

fn compute_layout_x(nodes: &[DagNode], z: f32, graph: &tensor_lang_graph::Graph) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..nodes.len()).collect();
    indices.sort_by(|&a, &b| nodes[a].base_x.partial_cmp(&nodes[b].base_x).unwrap());

    let mut x_positions = vec![0.0f32; nodes.len()];
    let mut right_edge = f32::MIN;

    for &idx in &indices {
        let node = &nodes[idx];
        let desired_x = node.base_x * z;
        let actual_x = desired_x.max(right_edge + 20.0 * z);
        x_positions[idx] = actual_x;

        let ew = node_effective_w(node, z, graph);
        if (node.y - 250.0).abs() < 100.0 {
            right_edge = right_edge.max(actual_x + ew);
        }
    }

    x_positions
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

/// Layered layout for sub-graph. Returns (col, row) for each node in range,
/// plus lists of external input indices and their positions.
struct SubLayout {
    positions: Vec<(usize, usize)>,   // (col, row) for each node in range
    n_cols: usize,
    n_rows: usize,
    ext_inputs: Vec<usize>,           // graph node indices of external inputs
    ext_positions: Vec<(usize, usize)>, // (col=0, row) for each external input
}

fn compute_sub_layout(
    graph: &tensor_lang_graph::Graph,
    range_start: usize, range_end: usize,
) -> SubLayout {
    let n = range_end - range_start + 1;
    if n == 0 {
        return SubLayout { positions: vec![], n_cols: 0, n_rows: 0,
            ext_inputs: vec![], ext_positions: vec![] };
    }

    // Find external inputs (nodes outside range that feed into this sub-graph)
    let mut ext_set = std::collections::BTreeSet::new();
    for idx in range_start..=range_end {
        for inp in &graph.nodes[idx].inputs {
            if inp.0 < range_start || inp.0 > range_end {
                ext_set.insert(inp.0);
            }
        }
    }
    let ext_inputs: Vec<usize> = ext_set.into_iter().collect();

    // Compute topological depth. External inputs are at depth 0,
    // so internal nodes that only depend on externals start at depth 1.
    let mut depth = vec![0usize; n];
    for idx in range_start..=range_end {
        let i = idx - range_start;
        let node = &graph.nodes[idx];
        let has_internal_input = node.inputs.iter()
            .any(|inp| inp.0 >= range_start && inp.0 <= range_end);
        let has_external_input = node.inputs.iter()
            .any(|inp| inp.0 < range_start || inp.0 > range_end);

        if !has_internal_input && !has_external_input {
            // Pure leaf (constants, arange) — depth 0
            depth[i] = 0;
        } else if !has_internal_input && has_external_input {
            // Only external inputs — depth 1 (after ext inputs column)
            depth[i] = 1;
        } else {
            // Has internal inputs — max of internal sources + 1
            for inp in &node.inputs {
                if inp.0 >= range_start && inp.0 <= range_end {
                    let src_i = inp.0 - range_start;
                    depth[i] = depth[i].max(depth[src_i] + 1);
                }
            }
            // Also ensure we're after any external input
            if has_external_input {
                depth[i] = depth[i].max(1);
            }
        }
    }

    // Group constants together: put them in the same column as the first node that uses them
    // This prevents constants from piling up at column 0
    for idx in range_start..=range_end {
        let i = idx - range_start;
        let node = &graph.nodes[idx];
        if matches!(node.op, tensor_lang_graph::Op::Constant(_)) && node.inputs.is_empty() {
            // Find the minimum depth of any consumer of this constant
            let mut min_consumer_depth = depth[i];
            for cidx in range_start..=range_end {
                let ci = cidx - range_start;
                if graph.nodes[cidx].inputs.iter().any(|inp| inp.0 == idx) {
                    min_consumer_depth = min_consumer_depth.max(depth[ci].saturating_sub(1));
                }
            }
            depth[i] = min_consumer_depth;
        }
    }

    // Assign rows within columns — try to position nodes near their consumers
    let max_col = depth.iter().copied().max().unwrap_or(0);
    let mut col_members: Vec<Vec<usize>> = vec![vec![]; max_col + 1];
    for i in 0..n {
        col_members[depth[i]].push(i);
    }

    // Sort members within each column by average consumer row (barycenter heuristic)
    // First pass: just use index order
    let mut positions = vec![(0usize, 0usize); n];
    for col in 0..=max_col {
        for (row, &i) in col_members[col].iter().enumerate() {
            positions[i] = (col, row);
        }
    }

    // External input positions: column 0, stacked
    let ext_positions: Vec<(usize, usize)> = ext_inputs.iter().enumerate()
        .map(|(row, _)| (0, row))
        .collect();

    // Shift internal columns right by 1 if we have external inputs
    let col_offset = if !ext_inputs.is_empty() { 1 } else { 0 };
    for p in &mut positions {
        p.0 += col_offset;
    }

    let n_cols = max_col + 1 + col_offset;
    let max_internal_rows = col_members.iter().map(|c| c.len()).max().unwrap_or(1);
    let n_rows = max_internal_rows.max(ext_inputs.len()).max(1);

    SubLayout { positions, n_cols, n_rows, ext_inputs, ext_positions }
}

fn bezier(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32), t: f32) -> (f32, f32) {
    let u = 1.0 - t;
    (u*u*u*p0.0 + 3.0*u*u*t*p1.0 + 3.0*u*t*t*p2.0 + t*t*t*p3.0,
     u*u*u*p0.1 + 3.0*u*u*t*p1.1 + 3.0*u*t*t*p2.1 + t*t*t*p3.1)
}

fn render_mini_heatmap(
    vertices: &mut Vec<ColorVertex>,
    val: &ndarray::ArrayD<f32>,
    px: f32, py: f32, pw: f32, ph: f32,
    win_w: f32, win_h: f32,
) {
    let flat: Vec<f32> = val.iter().copied().collect();
    if flat.is_empty() || pw < 2.0 || ph < 2.0 { return; }
    let fmin = flat.iter().copied().fold(f32::MAX, f32::min);
    let fmax = flat.iter().copied().fold(f32::MIN, f32::max);
    let frange = (fmax - fmin).max(0.001);
    let shape = val.shape();

    if shape.len() >= 2 {
        let (rows, cols, data) = collapse_to_2d(val);
        let max_r = rows.min(32);
        let max_c = cols.min(64);
        let cell_w = pw / max_c as f32;
        let cell_h = ph / max_r as f32;
        for r in 0..max_r {
            for c in 0..max_c {
                let v = data[r * cols + c];
                let t = (v - fmin) / frange;
                quad_px(vertices, px + c as f32 * cell_w, py + r as f32 * cell_h,
                    cell_w, cell_h, win_w, win_h, viridis(t));
            }
        }
    } else {
        let n = flat.len().min(64);
        let bar_w = pw / n as f32;
        for i in 0..n {
            let t = (flat[i] - fmin) / frange;
            let h = t * ph;
            quad_px(vertices, px + i as f32 * bar_w, py + ph - h,
                bar_w.max(1.0), h, win_w, win_h, viridis(t));
        }
    }
}

// ─── GPU state ────────────────────────────────────────────────────────────

struct GpuState {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    max_vertices: usize,
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
}

// ─── Application ──────────────────────────────────────────────────────────

struct App {
    gpu: Option<GpuState>,

    // Model (owned by UI thread — no background thread)
    runner: ModelRunner,
    adam: Adam,

    // Current example being inspected
    current_example: Vec<usize>,
    cursor_pos: usize,

    // Cached inference results for current example
    attn_weights: Vec<f32>,
    logits: Vec<f32>,
    predicted: Vec<usize>,

    // Training stats
    loss_history: Vec<f32>,
    step: usize,

    mode: Mode,
    view_mode: ViewMode,

    // Graph explorer state
    all_node_values: Vec<ndarray::ArrayD<f32>>,
    graph_cursor: usize,
    graph_scroll: usize,   // first visible row in node list
    mouse_pos: (f32, f32),
    // DAG view state
    dag: DagGraph,
    dag_offset: (f32, f32),
    dag_zoom: f32,
    dag_dragging: bool,
    dag_selected: Option<usize>,
    start_time: std::time::Instant,
}

impl App {
    fn new(runner: ModelRunner) -> Self {
        let sizes: Vec<usize> = runner.weights.iter().map(|w| w.len()).collect();
        let adam = Adam::new(&sizes, 0.001);
        let mut app = App {
            gpu: None,
            runner,
            adam,
            current_example: vec![],
            cursor_pos: 0,
            attn_weights: vec![],
            logits: vec![],
            predicted: vec![],
            loss_history: vec![],
            step: 0,
            mode: Mode::Paused,
            view_mode: ViewMode::Sequence,
            all_node_values: vec![],
            graph_cursor: 0,
            graph_scroll: 0,
            mouse_pos: (0.0, 0.0),
            dag: build_dag_graph(),
            dag_offset: (50.0, 50.0),
            dag_zoom: 0.7,
            dag_dragging: false,
            dag_selected: None,
            start_time: std::time::Instant::now(),
        };
        // Generate initial example
        app.new_example();
        app
    }

    /// Generate a new random example and run inference on it.
    fn new_example(&mut self) {
        let seq = model::generate_sequence(&mut rand::thread_rng());
        self.set_example(seq);
    }

    /// Set a specific example and run inference.
    fn set_example(&mut self, seq: Vec<usize>) {
        self.current_example = seq;
        self.cursor_pos = 0;
        self.run_inference();
    }

    /// Run inference on current_example with current weights.
    fn run_inference(&mut self) {
        let tokens = ModelRunner::pad_sequence(&self.current_example, self.runner.cfg.seq_len);
        let all = self.runner.forward(&tokens);
        self.attn_weights = all[self.runner.inf_map.attn_weights].iter().copied().collect();
        self.logits = all[self.runner.inf_map.logits].iter().copied().collect();
        self.all_node_values = all;

        let v = self.runner.cfg.vocab_size;
        let t = self.runner.cfg.seq_len;
        self.predicted.clear();
        for pos in 0..t {
            let start = pos * v;
            let end = start + v;
            if end <= self.logits.len() {
                let slice = &self.logits[start..end];
                let max_idx = slice.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                self.predicted.push(max_idx);
            }
        }
    }

    /// Train on one new random example, then show it.
    fn train_one_step(&mut self) {
        let seq = model::generate_sequence(&mut rand::thread_rng());
        let tokens = ModelRunner::pad_sequence(&seq, self.runner.cfg.seq_len);
        let mut target_seq = seq[1..].to_vec();
        target_seq.push(0);
        let targets = ModelRunner::pad_sequence(&target_seq, self.runner.cfg.seq_len);

        let (loss, grads) = self.runner.train_step(&tokens, &targets);
        self.adam.step(&mut self.runner.weights, &grads);
        if loss.is_finite() {
            self.loss_history.push(loss);
        }
        self.step += 1;

        // Show the example we just trained on
        self.set_example(seq);
    }

    fn render_dag(
        &self,
        vertices: &mut Vec<ColorVertex>,
        texts: &mut Vec<TextEntry>,
        win_w: f32, win_h: f32,
    ) {
        let tw = GlyphonColor::rgb(230, 230, 230);
        let td = GlyphonColor::rgb(150, 150, 165);
        let to = GlyphonColor::rgb(200, 180, 130);
        let z = self.dag_zoom;
        let (offx, offy) = self.dag_offset;
        let time = self.start_time.elapsed().as_secs_f32();
        let dag = &self.dag;

        // Compute dynamic layout (nodes push right when neighbors expand)
        let inf_graph = &self.runner.inf_graph;
        let layout_x = compute_layout_x(&dag.nodes, z, inf_graph);

        // ── Background with grid ──
        quad_px(vertices, 0.0, 0.0, win_w, win_h, win_w, win_h, [0.08, 0.08, 0.11, 1.0]);
        let grid = 60.0 * z;
        if grid > 8.0 {
            let gc = [0.10, 0.10, 0.14, 1.0];
            let mut gx = offx % grid;
            while gx < win_w { quad_px(vertices, gx, 0.0, 1.0, win_h, win_w, win_h, gc); gx += grid; }
            let mut gy = offy % grid;
            while gy < win_h { quad_px(vertices, 0.0, gy, win_w, 1.0, win_w, win_h, gc); gy += grid; }
        }

        // Helper: draw a bezier wire
        let draw_wire = |vertices: &mut Vec<ColorVertex>, sx: f32, sy: f32, dx: f32, dy: f32, color: [f32; 4]| {
            let ctrl = ((sx - dx).abs() * 0.4).max(40.0 * z);
            let p0 = (sx, sy); let p1 = (sx + ctrl, sy);
            let p2 = (dx - ctrl, dy); let p3 = (dx, dy);
            let segs = 20;
            let thick = 2.0 * z;
            for s in 0..segs {
                let t0 = s as f32 / segs as f32;
                let t1 = (s+1) as f32 / segs as f32;
                let (x0,y0) = bezier(p0,p1,p2,p3,t0);
                let (x1,y1) = bezier(p0,p1,p2,p3,t1);
                let ddx=x1-x0; let ddy=y1-y0;
                let l=(ddx*ddx+ddy*ddy).sqrt().max(0.1);
                let nx=-ddy/l*thick; let ny=ddx/l*thick;
                let (a,b)=(px_to_ndc(x0+nx,y0+ny,win_w,win_h),px_to_ndc(x0-nx,y0-ny,win_w,win_h));
                let (c,d)=(px_to_ndc(x1+nx,y1+ny,win_w,win_h),px_to_ndc(x1-nx,y1-ny,win_w,win_h));
                vertices.extend_from_slice(&[
                    ColorVertex{position:[a.0,a.1],color},ColorVertex{position:[b.0,b.1],color},
                    ColorVertex{position:[c.0,c.1],color},ColorVertex{position:[b.0,b.1],color},
                    ColorVertex{position:[c.0,c.1],color},ColorVertex{position:[d.0,d.1],color},
                ]);
            }
        };

        // ── Draw inter-node wires ──
        for wire in &dag.wires {
            let src = &dag.nodes[wire.src_node];
            let dst = &dag.nodes[wire.dst_node];
            let src_ew = node_effective_w(src, z, inf_graph);
            let src_eh = node_effective_h(src, z, inf_graph);
            let dst_eh = node_effective_h(dst, z, inf_graph);

            let sx = layout_x[wire.src_node] + offx + src_ew;
            let sy = src.y * z + offy + src_eh * 0.5;
            let port_y = (32.0 + wire.dst_port as f32 * 28.0 + 14.0) * z;
            let dx = layout_x[wire.dst_node] + offx;
            let dy = dst.y * z + offy + port_y.min(dst_eh * 0.8);

            let wc = { let c = src.header_color; [c[0]*0.5, c[1]*0.5, c[2]*0.5, 0.45] };
            draw_wire(vertices, sx, sy, dx, dy, wc);

            let pd = 5.0 * z;
            quad_px(vertices, sx-pd, sy-pd, pd*2.0, pd*2.0, win_w, win_h, src.header_color);
            quad_px(vertices, dx-pd, dy-pd, pd*2.0, pd*2.0, win_w, win_h, dst.header_color);
        }

        // ── Draw nodes ──
        for (ni, node) in dag.nodes.iter().enumerate() {
            let et = node.expand_t;
            let cur_w = node_effective_w(node, z, inf_graph);
            let cur_h = node_effective_h(node, z, inf_graph);
            let bx = layout_x[ni] + offx;
            let by = node.y * z + offy - (cur_h - NODE_H * z) * 0.5;
            if bx + cur_w < 0.0 || bx > win_w || by + cur_h < 0.0 || by > win_h { continue; }

            let is_sel = self.dag_selected == Some(ni);
            let header_h = 32.0 * z;
            let expandable = node.node_range.1 > node.node_range.0;

            // Glow
            if is_sel {
                let glow = (time * 2.0).sin() * 0.08 + 0.12;
                let gc = [node.header_color[0]*0.5, node.header_color[1]*0.5, node.header_color[2]*0.5, glow];
                quad_px(vertices, bx-6.0, by-6.0, cur_w+12.0, cur_h+12.0, win_w, win_h, gc);
            }

            // Body + header + border
            let bg = if is_sel { [0.18, 0.18, 0.24, 1.0] } else { [0.13, 0.13, 0.17, 1.0] };
            quad_px(vertices, bx, by, cur_w, cur_h, win_w, win_h, bg);
            quad_px(vertices, bx, by, cur_w, header_h, win_w, win_h, node.header_color);
            let bc = if is_sel { [0.6, 0.7, 1.0, 0.9] } else { [0.25, 0.25, 0.35, 0.5] };
            for &(rx,ry,rw,rh) in &[(bx,by,cur_w,1.5),(bx,by+cur_h-1.5,cur_w,1.5),(bx,by,1.5,cur_h),(bx+cur_w-1.5,by,1.5,cur_h)] {
                quad_px(vertices, rx, ry, rw, rh, win_w, win_h, bc);
            }

            // Title
            let ts = (16.0 * z).max(8.0);
            let icon = if expandable { if et > 0.5 { "▼ " } else { "▶ " } } else { "" };
            texts.push(TextEntry {
                text: format!("{}{}", icon, node.name), x: bx + 8.0*z, y: by + 6.0*z, size: ts, color: tw,
            });
            if !node.op_symbol.is_empty() {
                texts.push(TextEntry {
                    text: node.op_symbol.to_string(),
                    x: bx + cur_w - (node.op_symbol.len() as f32 + 1.0) * ts * 0.6,
                    y: by + 6.0*z, size: ts, color: to,
                });
            }

            if et < 0.1 {
                // ── Collapsed view: input heatmaps → output heatmap ──
                let py0 = by + header_h + 4.0*z;
                let ph = cur_h - header_h - 8.0*z;
                let n_in = node.input_nodes.len();
                let total = n_in + 1;
                let gap = 6.0 * z;
                let sw = (cur_w - gap * (total as f32 + 1.0)) / total as f32;
                let psz = sw.min(ph - 16.0*z).max(4.0);

                for (ii, &inp) in node.input_nodes.iter().enumerate() {
                    let px = bx + gap + ii as f32 * (sw + gap);
                    let py = py0 + 14.0*z;
                    let label = if ii < node.input_labels.len() { node.input_labels[ii] } else { "?" };
                    texts.push(TextEntry { text: label.to_string(), x: px, y: py0, size: (11.0*z).max(6.0), color: td });
                    if inp < self.all_node_values.len() {
                        render_mini_heatmap(vertices, &self.all_node_values[inp], px, py, psz, psz, win_w, win_h);
                    }
                }
                if n_in > 0 {
                    let ax = bx + gap + n_in as f32 * (sw + gap) - gap*0.5;
                    texts.push(TextEntry { text: "→".into(), x: ax, y: py0 + ph*0.4, size: (14.0*z).max(7.0), color: td });
                }
                let px = bx + gap + n_in as f32 * (sw + gap);
                texts.push(TextEntry { text: node.output_label.to_string(), x: px, y: py0, size: (11.0*z).max(6.0), color: td });
                if node.output_node < self.all_node_values.len() {
                    render_mini_heatmap(vertices, &self.all_node_values[node.output_node], px, py0+14.0*z, psz, psz, win_w, win_h);
                }
            } else {
                // ── Expanded: dataflow layout of every primitive op ──
                let inner_x = bx + 10.0*z;
                let inner_y = by + header_h + 8.0*z;
                let sub_w = SUB_NODE_W * z;
                let sub_h = SUB_NODE_H * z;
                let alpha = et.min(1.0);
                let col_gap = SUB_COL_GAP * z;
                let row_gap = SUB_ROW_GAP * z;

                let layout = compute_sub_layout(
                    inf_graph, node.node_range.0, node.node_range.1);

                let sub_pos = |i: usize| -> (f32, f32) {
                    let (col, row) = layout.positions[i];
                    (inner_x + col as f32 * (sub_w + col_gap),
                     inner_y + row as f32 * (sub_h + row_gap))
                };

                let ext_pos = |ei: usize| -> (f32, f32) {
                    let (col, row) = layout.ext_positions[ei];
                    (inner_x + col as f32 * (sub_w + col_gap),
                     inner_y + row as f32 * (sub_h + row_gap))
                };

                let wire_color = [
                    node.header_color[0] * 0.7, node.header_color[1] * 0.7,
                    node.header_color[2] * 0.7, 0.7 * alpha,
                ];
                let port_color = [
                    node.header_color[0] * 0.9, node.header_color[1] * 0.9,
                    node.header_color[2] * 0.9, 0.9 * alpha,
                ];
                let ext_wire_color = [0.4, 0.4, 0.5, 0.5 * alpha];
                let sh = 20.0 * z;

                // Draw external input nodes (small labeled ports)
                for (ei, &ext_idx) in layout.ext_inputs.iter().enumerate() {
                    let (ex, ey) = ext_pos(ei);
                    let ew = sub_w * 0.7;
                    let eh = 28.0 * z;
                    // Small box
                    quad_px(vertices, ex, ey, ew, eh, win_w, win_h, [0.15, 0.15, 0.22, alpha]);
                    quad_px(vertices, ex, ey, ew, 1.0, win_w, win_h, [0.3, 0.3, 0.5, 0.4*alpha]);
                    quad_px(vertices, ex, ey+eh-1.0, ew, 1.0, win_w, win_h, [0.3, 0.3, 0.5, 0.4*alpha]);
                    quad_px(vertices, ex, ey, 1.0, eh, win_w, win_h, [0.3, 0.3, 0.5, 0.4*alpha]);
                    quad_px(vertices, ex+ew-1.0, ey, 1.0, eh, win_w, win_h, [0.3, 0.3, 0.5, 0.4*alpha]);
                    // Label
                    let ext_name = if let tensor_lang_graph::Op::Input { name } = &inf_graph.nodes[ext_idx].op {
                        name.clone()
                    } else {
                        format!("#{}", ext_idx)
                    };
                    let es = (9.0*z).max(5.0);
                    let ec = GlyphonColor::rgba(180, 180, 200, (alpha * 255.0) as u8);
                    texts.push(TextEntry { text: ext_name, x: ex + 3.0*z, y: ey + 4.0*z, size: es, color: ec });
                    // Output port dot
                    let pd = 4.0 * z;
                    quad_px(vertices, ex + ew - pd, ey + eh*0.5 - pd, pd*2.0, pd*2.0,
                        win_w, win_h, port_color);
                }

                // Draw wires: internal→internal and external→internal
                for idx in node.node_range.0..=node.node_range.1 {
                    let gn = &inf_graph.nodes[idx];
                    let dst_i = idx - node.node_range.0;
                    let (dx_s, dy_s) = sub_pos(dst_i);
                    let n_inputs = gn.inputs.len().max(1) as f32;

                    for (port, inp_id) in gn.inputs.iter().enumerate() {
                        let src_idx = inp_id.0;
                        let port_spacing = (sub_h - sh) / (n_inputs + 1.0);
                        let wire_dy = dy_s + sh + port_spacing * (port as f32 + 1.0);
                        let wire_dx = dx_s;
                        let pd = 4.0 * z;

                        if src_idx >= node.node_range.0 && src_idx <= node.node_range.1 {
                            // Internal wire
                            let src_i = src_idx - node.node_range.0;
                            let (sx_s, sy_s) = sub_pos(src_i);
                            let wire_sx = sx_s + sub_w;
                            let wire_sy = sy_s + sub_h * 0.5;

                            draw_wire(vertices, wire_sx, wire_sy, wire_dx, wire_dy, wire_color);
                            quad_px(vertices, wire_sx-pd, wire_sy-pd, pd*2.0, pd*2.0, win_w, win_h, port_color);
                            quad_px(vertices, wire_dx-pd, wire_dy-pd, pd*2.0, pd*2.0, win_w, win_h, port_color);
                        } else if let Some(ei) = layout.ext_inputs.iter().position(|&e| e == src_idx) {
                            // External wire
                            let (ex, ey) = ext_pos(ei);
                            let ew = sub_w * 0.7;
                            let wire_sx = ex + ew;
                            let wire_sy = ey + 14.0 * z;

                            draw_wire(vertices, wire_sx, wire_sy, wire_dx, wire_dy, ext_wire_color);
                            quad_px(vertices, wire_dx-pd, wire_dy-pd, pd*2.0, pd*2.0, win_w, win_h, port_color);
                        }
                    }
                }

                // Draw sub-nodes
                for idx in node.node_range.0..=node.node_range.1 {
                    let ci = idx - node.node_range.0;
                    let (cx, cy) = sub_pos(ci);
                    let gn = &inf_graph.nodes[idx];

                    // Background
                    quad_px(vertices, cx, cy, sub_w, sub_h, win_w, win_h, [0.10, 0.10, 0.14, alpha]);
                    // Header
                    let shc = [node.header_color[0]*0.6, node.header_color[1]*0.6, node.header_color[2]*0.6, alpha];
                    quad_px(vertices, cx, cy, sub_w, sh, win_w, win_h, shc);
                    // Border
                    let sbc = [0.3, 0.3, 0.4, 0.5 * alpha];
                    quad_px(vertices, cx, cy, sub_w, 1.0, win_w, win_h, sbc);
                    quad_px(vertices, cx, cy+sub_h-1.0, sub_w, 1.0, win_w, win_h, sbc);
                    quad_px(vertices, cx, cy, 1.0, sub_h, win_w, win_h, sbc);
                    quad_px(vertices, cx+sub_w-1.0, cy, 1.0, sub_h, win_w, win_h, sbc);

                    // Op name in header
                    let op_name = match &gn.op {
                        tensor_lang_graph::Op::Input { name } => format!("In({})", name),
                        tensor_lang_graph::Op::Constant(v) => format!("{:.3}", v),
                        tensor_lang_graph::Op::Arange { .. } => "Arange".into(),
                        tensor_lang_graph::Op::Neg => "Neg".into(),
                        tensor_lang_graph::Op::Recip => "1/x".into(),
                        tensor_lang_graph::Op::Exp2 => "Exp2".into(),
                        tensor_lang_graph::Op::Log2 => "Log2".into(),
                        tensor_lang_graph::Op::Sqrt => "√".into(),
                        tensor_lang_graph::Op::Add => "Add".into(),
                        tensor_lang_graph::Op::Mul => "Mul".into(),
                        tensor_lang_graph::Op::Max => "Max".into(),
                        tensor_lang_graph::Op::CmpLt => "CmpLt".into(),
                        tensor_lang_graph::Op::ReduceSum { axis } => format!("Σ ax={}", axis),
                        tensor_lang_graph::Op::ReduceMax { axis } => format!("Max ax={}", axis),
                        tensor_lang_graph::Op::Reshape { .. } => "Reshape".into(),
                        tensor_lang_graph::Op::Permute { .. } => "Permute".into(),
                        tensor_lang_graph::Op::Expand { .. } => "Expand".into(),
                        tensor_lang_graph::Op::Pad { .. } => "Pad".into(),
                        tensor_lang_graph::Op::Shrink { .. } => "Shrink".into(),
                    };
                    let ss = (11.0*z).max(6.0);
                    let ac = GlyphonColor::rgba(230, 230, 230, (alpha * 255.0) as u8);
                    texts.push(TextEntry {
                        text: op_name, x: cx + 4.0*z, y: cy + 3.0*z, size: ss, color: ac,
                    });

                    // Shape label below header
                    let shape_str: String = gn.shape.iter()
                        .map(|d| d.as_usize().map(|v| v.to_string()).unwrap_or("?".into()))
                        .collect::<Vec<_>>().join("×");
                    let ds = (9.0*z).max(5.0);
                    let dc = GlyphonColor::rgba(150, 150, 170, (alpha * 255.0) as u8);
                    texts.push(TextEntry {
                        text: format!("[{}]", shape_str),
                        x: cx + 4.0*z, y: cy + sh + 2.0*z, size: ds, color: dc,
                    });

                    // Output port dot on right edge
                    let out_y = cy + sub_h * 0.5;
                    let pd = 4.0 * z;
                    quad_px(vertices, cx + sub_w - pd, out_y - pd, pd*2.0, pd*2.0,
                        win_w, win_h, port_color);

                    // Mini heatmap
                    if idx < self.all_node_values.len() && alpha > 0.3 {
                        let hx = cx + 3.0*z;
                        let hy = cy + sh + 14.0*z;
                        let hw = sub_w - 6.0*z;
                        let hh = (sub_h - sh - 18.0*z).max(2.0);
                        render_mini_heatmap(vertices, &self.all_node_values[idx], hx, hy, hw, hh, win_w, win_h);
                    }
                }
            }
        }

        // Title
        texts.push(TextEntry {
            text: "Transformer Dataflow".into(), x: 12.0, y: 8.0, size: 28.0, color: tw,
        });

        // Input sequence overlay (top-right corner)
        {
            let seq_str: String = self.current_example.iter()
                .enumerate()
                .map(|(i, &tok)| {
                    let name = model::token_name(tok);
                    if i == self.cursor_pos { format!("[{}]", name) } else { name.to_string() }
                })
                .collect::<Vec<_>>().join(" ");
            let fs = 16.0;
            let text_w = seq_str.len() as f32 * fs * 0.6 + 16.0;
            let ox = (win_w - text_w - 10.0).max(10.0);
            quad_px(vertices, ox - 6.0, 2.0, text_w + 12.0, 28.0,
                win_w, win_h, [0.06, 0.06, 0.09, 0.9]);
            texts.push(TextEntry {
                text: seq_str, x: ox, y: 6.0, size: fs, color: td,
            });
        }
    }

    fn render_graph_panels(
        runner: &ModelRunner,
        all_node_values: &[ndarray::ArrayD<f32>],
        graph_cursor: usize,
        graph_scroll: usize,
        vertices: &mut Vec<ColorVertex>,
        texts: &mut Vec<TextEntry>,
        m: f32, left_w: f32, right_x: f32, right_w: f32,
        bot_y: f32, bot_h: f32, win_w: f32, win_h: f32,
    ) {
        let tw = GlyphonColor::rgb(220, 220, 220);
        let tg = GlyphonColor::rgb(100, 220, 120);
        let ty = GlyphonColor::rgb(220, 200, 80);
        let td = GlyphonColor::rgb(140, 140, 140);
        let tr = GlyphonColor::rgb(230, 100, 80);

        let graph = &runner.inf_graph;
        let logits_idx = runner.inf_logits_idx;
        let n_nodes = graph.nodes.len();

        // ── Left panel: Node list ──
        let list_x = m + 10.0;
        let list_y = bot_y + 10.0;
        let row_h = 22.0;
        let visible_rows = ((bot_h - 20.0) / row_h) as usize;

        texts.push(TextEntry {
            text: format!("Graph: {} nodes (click to inspect)", n_nodes),
            x: list_x, y: bot_y + 2.0, size: 18.0, color: ty,
        });

        let list_top = list_y + 22.0;
        for i in 0..visible_rows {
            let node_idx = graph_scroll + i;
            if node_idx >= n_nodes { break; }

            let y = list_top + i as f32 * row_h;
            let is_selected = node_idx == graph_cursor;

            // Color by section
            let section_color = if node_idx <= logits_idx {
                GlyphonColor::rgb(100, 160, 255)  // blue = forward
            } else {
                GlyphonColor::rgb(255, 120, 80)   // red = loss+backward
            };

            // Highlight selected row
            if is_selected {
                quad_px(vertices, list_x - 4.0, y - 2.0,
                    left_w - 2.0 * m - 12.0, row_h,
                    win_w, win_h, [0.2, 0.2, 0.3, 1.0]);
            }

            let desc = model::describe_node(graph, node_idx);
            // Truncate to fit
            let max_chars = ((left_w - 2.0 * m - 30.0) / 10.0) as usize;
            let display = if desc.len() > max_chars { &desc[..max_chars] } else { &desc };
            texts.push(TextEntry {
                text: display.to_string(),
                x: list_x, y,
                size: 18.0,
                color: if is_selected { tw } else { section_color },
            });
        }

        // Scrollbar indicator
        if n_nodes > visible_rows {
            let sb_x = m + left_w - 2.0 * m - 8.0;
            let sb_h = bot_h - 30.0;
            let thumb_h = (visible_rows as f32 / n_nodes as f32 * sb_h).max(10.0);
            let thumb_y = list_top + (graph_scroll as f32 / n_nodes as f32) * sb_h;
            quad_px(vertices, sb_x, list_top, 4.0, sb_h, win_w, win_h, [0.2, 0.2, 0.2, 1.0]);
            quad_px(vertices, sb_x, thumb_y, 4.0, thumb_h, win_w, win_h, [0.5, 0.5, 0.5, 1.0]);
        }

        // ── Right panel: Value inspector ──
        let insp_x = right_x + 10.0;
        let insp_y = bot_y + 10.0;

        if graph_cursor < all_node_values.len() {
            let node = &graph.nodes[graph_cursor];
            let val = &all_node_values[graph_cursor];
            let shape = val.shape();

            // Header
            let desc = model::describe_node(graph, graph_cursor);
            texts.push(TextEntry {
                text: desc, x: insp_x, y: insp_y, size: 18.0, color: ty,
            });

            // Stats
            let flat: Vec<f32> = val.iter().copied().collect();
            if !flat.is_empty() {
                let min = flat.iter().copied().fold(f32::MAX, f32::min);
                let max = flat.iter().copied().fold(f32::MIN, f32::max);
                let mean = flat.iter().sum::<f32>() / flat.len() as f32;
                texts.push(TextEntry {
                    text: format!("min={:.4} max={:.4} mean={:.4} n={}", min, max, mean, flat.len()),
                    x: insp_x, y: insp_y + 22.0, size: 16.0, color: td,
                });
            }

            // Visualize value
            let viz_y = insp_y + 50.0;
            let viz_w = right_w - 30.0;
            let viz_h = bot_h - 80.0;

            if shape.is_empty() || shape.iter().all(|&d| d == 1) {
                // Scalar — show big number
                let v = flat.first().copied().unwrap_or(0.0);
                texts.push(TextEntry {
                    text: format!("{:.6}", v),
                    x: insp_x + 20.0, y: viz_y + viz_h * 0.3, size: 48.0, color: tw,
                });
            } else if shape.len() == 1 || (shape.len() >= 2 && shape.iter().filter(|&&d| d > 1).count() == 1) {
                // 1D — bar chart
                let flat_abs_max = flat.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(0.001);
                let n = flat.len().min(128); // cap at 128 bars
                let bar_w = viz_w / n as f32;
                let mid_y = viz_y + viz_h * 0.5;

                for i in 0..n {
                    let v = flat[i];
                    let h = (v / flat_abs_max) * (viz_h * 0.45);
                    let color = if v >= 0.0 { [0.3, 0.7, 0.9, 1.0] } else { [0.9, 0.4, 0.3, 1.0] };
                    if h >= 0.0 {
                        quad_px(vertices, insp_x + i as f32 * bar_w + 1.0, mid_y - h,
                            bar_w - 2.0, h, win_w, win_h, color);
                    } else {
                        quad_px(vertices, insp_x + i as f32 * bar_w + 1.0, mid_y,
                            bar_w - 2.0, -h, win_w, win_h, color);
                    }
                }
                // Zero line
                quad_px(vertices, insp_x, mid_y, viz_w, 1.0, win_w, win_h, [0.5, 0.5, 0.5, 0.5]);
            } else {
                // 2D+ — heatmap (use first 2D slice)
                // Collapse to 2D: take first element of all leading dims
                let (rows, cols, data) = collapse_to_2d(val);
                if rows > 0 && cols > 0 {
                    let cell_w = (viz_w / cols as f32).min(viz_h / rows as f32);
                    let cell_h = cell_w;

                    // Find range for colormap
                    let dmin = data.iter().copied().fold(f32::MAX, f32::min);
                    let dmax = data.iter().copied().fold(f32::MIN, f32::max);
                    let drange = (dmax - dmin).max(0.001);

                    let max_r = ((viz_h / cell_h) as usize).min(rows);
                    let max_c = ((viz_w / cell_w) as usize).min(cols);

                    for r in 0..max_r {
                        for c in 0..max_c {
                            let v = data[r * cols + c];
                            let t = (v - dmin) / drange;
                            quad_px(vertices,
                                insp_x + c as f32 * cell_w,
                                viz_y + r as f32 * cell_h,
                                cell_w - 1.0, cell_h - 1.0,
                                win_w, win_h, viridis(t));
                        }
                    }

                    texts.push(TextEntry {
                        text: format!("{}x{}", rows, cols),
                        x: insp_x, y: viz_y + max_r as f32 * cell_h + 4.0,
                        size: 16.0, color: td,
                    });
                }
            }
        } else {
            texts.push(TextEntry {
                text: "No data — run inference first".into(),
                x: insp_x, y: insp_y, size: 20.0, color: td,
            });
        }
    }

    fn render(&mut self) {
        let (win_w, win_h) = {
            let gpu = self.gpu.as_ref().unwrap();
            let size = gpu.window.inner_size();
            (size.width as f32, size.height as f32)
        };
        if win_w < 1.0 || win_h < 1.0 { return; }

        let mut vertices: Vec<ColorVertex> = Vec::with_capacity(50_000);
        let mut texts: Vec<TextEntry> = Vec::new();

        // DAG mode: full-screen graph, skip all panels
        if self.view_mode == ViewMode::DAG {
            self.render_dag(&mut vertices, &mut texts, win_w, win_h);
            let td = GlyphonColor::rgb(140, 140, 140);
            texts.push(TextEntry {
                text: "[Tab]=Seq  [Drag]=Pan  [Pinch]=Zoom  [Click]=Expand  [S]=Train  [N]=New  [Left/Right]=Pos  [Q]=Quit".into(),
                x: 5.0, y: win_h - 30.0, size: 20.0, color: td,
            });

            // Upload + draw (jump to end of render)
            let gpu = self.gpu.as_mut().unwrap();
            let vertex_count = vertices.len();
            if vertex_count == 0 { return; }
            if vertex_count > gpu.max_vertices {
                gpu.vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("vertex_buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
                gpu.max_vertices = vertex_count;
            } else {
                gpu.queue.write_buffer(&gpu.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            }
            let mut text_buffers: Vec<GlyphonBuffer> = Vec::new();
            for entry in &texts {
                let mut buffer = GlyphonBuffer::new(&mut gpu.font_system, Metrics::new(entry.size, entry.size * 1.3));
                buffer.set_size(&mut gpu.font_system, Some(win_w), Some(win_h));
                buffer.set_text(&mut gpu.font_system, &entry.text,
                    &Attrs::new().family(Family::Monospace).color(entry.color), Shaping::Basic, None);
                buffer.shape_until_scroll(&mut gpu.font_system, false);
                text_buffers.push(buffer);
            }
            let text_areas: Vec<TextArea> = text_buffers.iter().zip(texts.iter()).map(|(buf, entry)| {
                TextArea { buffer: buf, left: entry.x, top: entry.y, scale: 1.0,
                    bounds: TextBounds { left: 0, top: 0, right: win_w as i32, bottom: win_h as i32 },
                    default_color: entry.color, custom_glyphs: &[] }
            }).collect();
            gpu.viewport.update(&gpu.queue, Resolution { width: win_w as u32, height: win_h as u32 });
            gpu.text_renderer.prepare(&gpu.device, &gpu.queue, &mut gpu.font_system,
                &mut gpu.text_atlas, &gpu.viewport, text_areas, &mut gpu.swash_cache).unwrap();
            let frame = match gpu.surface.get_current_texture() { Ok(f) => f, Err(_) => return };
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("dag_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view, depth_slice: None, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.08, b: 0.1, a: 1.0 }),
                            store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None, timestamp_writes: None,
                    occlusion_query_set: None, multiview_mask: None,
                });
                pass.set_pipeline(&gpu.pipeline);
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                pass.draw(0..vertex_count as u32, 0..1);
                gpu.text_renderer.render(&gpu.text_atlas, &gpu.viewport, &mut pass).unwrap();
            }
            gpu.queue.submit(std::iter::once(encoder.finish()));
            frame.present();
            gpu.text_atlas.trim();
            return;
        }

        let green = [0.2, 0.8, 0.4, 1.0];
        let red_highlight = [0.9, 0.3, 0.1, 1.0];
        let blue_bar = [0.3, 0.5, 0.8, 1.0];

        let tw = GlyphonColor::rgb(220, 220, 220);
        let tg = GlyphonColor::rgb(100, 220, 120);
        let ty = GlyphonColor::rgb(220, 200, 80);
        let td = GlyphonColor::rgb(140, 140, 140);

        let n_head = self.runner.cfg.n_head;
        let seq_len = self.runner.cfg.seq_len;
        let vocab_size = self.runner.cfg.vocab_size;
        let n_embd = self.runner.cfg.n_embd;

        // ── Panel layout ──
        let m = 10.0;
        let left_w = win_w * 0.6;
        let right_x = left_w + m;
        let right_w = win_w - right_x - m;
        let top_h = win_h * 0.55;
        let bot_y = top_h + m;
        let bot_h = win_h - bot_y - m;

        let bg = [0.12, 0.12, 0.15, 1.0];
        quad_px(&mut vertices, m, m, left_w - 2.0 * m, top_h - m, win_w, win_h, bg);
        quad_px(&mut vertices, right_x, m, right_w - m, top_h - m, win_w, win_h, bg);
        quad_px(&mut vertices, m, bot_y, left_w - 2.0 * m, bot_h, win_w, win_h, bg);
        quad_px(&mut vertices, right_x, bot_y, right_w - m, bot_h, win_w, win_h, bg);

        // ── Title ──
        let mode_str = match self.mode {
            Mode::Paused => " [PAUSED]",
            Mode::Training => " [TRAINING]",
        };
        let loss_val = self.loss_history.last().copied().unwrap_or(0.0);
        texts.push(TextEntry {
            text: format!("Step {} | Loss: {:.4}{}", self.step, loss_val, mode_str),
            x: m + 5.0, y: 2.0, size: 36.0, color: tw,
        });

        // ── 1. Attention heatmaps (top-left) ──
        {
            let t = seq_len;
            let hm_size = (top_h - 110.0).min((left_w - 2.0 * m - 40.0) / n_head as f32 - 15.0);
            let cell = hm_size / t as f32;

            for head in 0..n_head {
                let ox = m + 20.0 + head as f32 * (hm_size + 30.0);
                let oy = 80.0;

                texts.push(TextEntry {
                    text: format!("Head {}", head),
                    x: ox, y: oy - 34.0, size: 28.0, color: td,
                });

                for row in 0..t {
                    let is_cursor_row = row == self.cursor_pos;
                    for col in 0..t {
                        let idx = head * t * t + row * t + col;
                        let val = self.attn_weights.get(idx).copied().unwrap_or(0.0);
                        let mut color = viridis(val);
                        if !is_cursor_row {
                            color[3] = 0.4;
                        }
                        quad_px(
                            &mut vertices,
                            ox + col as f32 * cell, oy + row as f32 * cell,
                            cell - 1.0, cell - 1.0,
                            win_w, win_h, color,
                        );
                    }
                    if is_cursor_row {
                        let border = [1.0, 1.0, 1.0, 0.6];
                        quad_px(&mut vertices, ox, oy + row as f32 * cell - 1.0,
                            t as f32 * cell, 1.0, win_w, win_h, border);
                        quad_px(&mut vertices, ox, oy + (row + 1) as f32 * cell,
                            t as f32 * cell, 1.0, win_w, win_h, border);
                    }
                }
            }
        }

        // ── 2. Loss curve + sequence display (top-right) ──
        {
            texts.push(TextEntry {
                text: "Loss".into(), x: right_x + 5.0, y: 42.0, size: 28.0, color: tg,
            });

            let hist = &self.loss_history;
            if hist.len() > 1 {
                let ox = right_x + 20.0;
                let oy = 76.0;
                let chart_w = right_w - 100.0;
                let chart_h = top_h * 0.35;

                let min_loss = hist.iter().copied().fold(f32::MAX, f32::min).max(0.0);
                let max_loss = hist.iter().copied().fold(f32::MIN, f32::max);
                let range = (max_loss - min_loss).max(0.1);

                let start = hist.len().saturating_sub(2000);
                let visible = &hist[start..];

                for i in 0..visible.len().saturating_sub(1) {
                    let x0 = ox + (i as f32 / visible.len() as f32) * chart_w;
                    let x1 = ox + ((i + 1) as f32 / visible.len() as f32) * chart_w;
                    let y0 = oy + chart_h - ((visible[i] - min_loss) / range) * chart_h;
                    let y1 = oy + chart_h - ((visible[i + 1] - min_loss) / range) * chart_h;

                    let dx = x1 - x0;
                    let dy = y1 - y0;
                    let len = (dx * dx + dy * dy).sqrt().max(0.001);
                    let nx = -dy / len * 1.0;
                    let ny = dx / len * 1.0;

                    let (ax, ay) = px_to_ndc(x0 + nx, y0 + ny, win_w, win_h);
                    let (bx, by) = px_to_ndc(x0 - nx, y0 - ny, win_w, win_h);
                    let (cx, cy) = px_to_ndc(x1 + nx, y1 + ny, win_w, win_h);
                    let (dx2, dy2) = px_to_ndc(x1 - nx, y1 - ny, win_w, win_h);

                    vertices.extend_from_slice(&[
                        ColorVertex { position: [ax, ay], color: green },
                        ColorVertex { position: [bx, by], color: green },
                        ColorVertex { position: [cx, cy], color: green },
                        ColorVertex { position: [bx, by], color: green },
                        ColorVertex { position: [cx, cy], color: green },
                        ColorVertex { position: [dx2, dy2], color: green },
                    ]);
                }

                texts.push(TextEntry {
                    text: format!("{:.2}", max_loss),
                    x: ox + chart_w + 5.0, y: oy, size: 28.0, color: td,
                });
                texts.push(TextEntry {
                    text: format!("{:.2}", min_loss),
                    x: ox + chart_w + 5.0, y: oy + chart_h - 28.0, size: 28.0, color: td,
                });
            }

            // Sequence display with cursor
            let iy = top_h * 0.5 + 70.0;
            let pos = self.cursor_pos;
            let ex_len = self.current_example.len();
            let actual_next = if pos + 1 < ex_len {
                Some(self.current_example[pos + 1])
            } else {
                None
            };
            let predicted_next = self.predicted.get(pos).copied();

            texts.push(TextEntry {
                text: format!("Input sequence (pos {}/{})", pos, ex_len.saturating_sub(1)),
                x: right_x + 10.0, y: iy, size: 28.0, color: ty,
            });

            let mut sx = right_x + 10.0;
            let tok_y = iy + 36.0;
            for (i, &tok) in self.current_example.iter().enumerate() {
                let name = model::token_name(tok);
                let is_cursor = i == pos;
                let is_next = i == pos + 1;
                let color = if is_cursor { tw } else if is_next { tg } else { td };
                if is_cursor {
                    let label_w = name.len() as f32 * 20.0 + 8.0;
                    quad_px(&mut vertices, sx - 4.0, tok_y - 4.0, label_w, 38.0,
                        win_w, win_h, [0.25, 0.25, 0.35, 1.0]);
                }
                texts.push(TextEntry {
                    text: name.to_string(),
                    x: sx, y: tok_y, size: 32.0, color,
                });
                sx += name.len() as f32 * 20.0 + 14.0;
                if sx > win_w - 20.0 { break; }
            }

            let info_y = tok_y + 42.0;
            if let Some(pred) = predicted_next {
                let pred_name = model::token_name(pred);
                let is_correct = actual_next == Some(pred);
                let pred_color = if is_correct { tg } else { GlyphonColor::rgb(230, 80, 60) };
                texts.push(TextEntry {
                    text: format!("Predicted: {}", pred_name),
                    x: right_x + 10.0, y: info_y, size: 32.0, color: pred_color,
                });
            }
            if let Some(actual) = actual_next {
                texts.push(TextEntry {
                    text: format!("Actual:    {}", model::token_name(actual)),
                    x: right_x + 10.0, y: info_y + 38.0, size: 32.0, color: tg,
                });
            }
        }

        // ── Bottom panels: depends on view mode ──
      match self.view_mode {
        ViewMode::Sequence => {

        // ── 3. Logit bar chart (bottom-left) ──
        {
            let pos = self.cursor_pos;
            let actual_next = if pos + 1 < self.current_example.len() {
                Some(self.current_example[pos + 1])
            } else {
                None
            };

            texts.push(TextEntry {
                text: format!("P(next | pos {})", pos),
                x: m + 5.0, y: bot_y + 5.0, size: 28.0, color: ty,
            });

            let v = vocab_size;
            let ox = m + 20.0;
            let oy = bot_y + 40.0;
            let chart_w = left_w - 2.0 * m - 40.0;
            let chart_h = bot_h - 100.0;
            let bar_w = chart_w / v as f32;

            let start = pos * v;
            let end = start + v;
            if end <= self.logits.len() {
                let logits_slice = &self.logits[start..end];
                let max_logit = logits_slice.iter().copied().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits_slice.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

                let predicted = self.predicted.get(pos).copied().unwrap_or(0);

                for (i, &p) in probs.iter().enumerate() {
                    let h = p * chart_h;
                    let color = if Some(i) == actual_next {
                        [0.2, 0.85, 0.4, 1.0]
                    } else if i == predicted {
                        red_highlight
                    } else {
                        blue_bar
                    };
                    quad_px(
                        &mut vertices,
                        ox + i as f32 * bar_w + 2.0, oy + chart_h - h,
                        bar_w - 4.0, h,
                        win_w, win_h, color,
                    );

                    if p > 0.05 {
                        texts.push(TextEntry {
                            text: format!("{:.0}%", p * 100.0),
                            x: ox + i as f32 * bar_w + 2.0,
                            y: oy + chart_h - h - 24.0,
                            size: 20.0, color: tw,
                        });
                    }
                }

                for (i, &name) in TOKENS.iter().enumerate() {
                    if i >= v { break; }
                    let label = if name.len() > 6 { &name[..6] } else { name };
                    texts.push(TextEntry {
                        text: label.to_string(),
                        x: ox + i as f32 * bar_w + 2.0,
                        y: oy + chart_h + 6.0,
                        size: 20.0, color: td,
                    });
                }
            }
        }

        // ── 4. Embedding scatter (bottom-right) ──
        {
            texts.push(TextEntry {
                text: "Embeddings (dim 0 vs 1)".into(),
                x: right_x + 5.0, y: bot_y + 5.0, size: 28.0, color: ty,
            });

            let v = vocab_size;
            let d = n_embd;
            let emb = &self.runner.weights[0]; // wte
            if emb.len() == v * d {
                let ox = right_x + 20.0;
                let oy = bot_y + 42.0;
                let plot_w = right_w - 60.0;
                let plot_h = bot_h - 60.0;

                let mut points = Vec::new();
                for i in 0..v {
                    points.push((emb[i * d], emb[i * d + 1]));
                }
                let min_x = points.iter().map(|p| p.0).fold(f32::MAX, f32::min);
                let max_x = points.iter().map(|p| p.0).fold(f32::MIN, f32::max);
                let min_y = points.iter().map(|p| p.1).fold(f32::MAX, f32::min);
                let max_y = points.iter().map(|p| p.1).fold(f32::MIN, f32::max);
                let rx = (max_x - min_x).max(0.01);
                let ry = (max_y - min_y).max(0.01);

                for (i, &(px, py)) in points.iter().enumerate() {
                    let sx = ox + ((px - min_x) / rx) * plot_w;
                    let sy = oy + ((py - min_y) / ry) * plot_h;

                    let color = match i {
                        0..=2 => [0.5, 0.5, 0.5, 1.0],
                        3 => [1.0, 0.8, 0.2, 1.0],
                        4..=5 | 8 => [0.3, 0.7, 1.0, 1.0],
                        6..=7 => [0.2, 0.9, 0.4, 1.0],
                        9..=11 => [0.9, 0.4, 0.7, 1.0],
                        12..=13 => [1.0, 0.5, 0.2, 1.0],
                        _ => [0.7, 0.3, 0.9, 1.0],
                    };

                    quad_px(&mut vertices, sx - 10.0, sy - 10.0, 20.0, 20.0, win_w, win_h, color);

                    texts.push(TextEntry {
                        text: TOKENS[i].to_string(),
                        x: sx + 10.0, y: sy - 10.0, size: 28.0, color: td,
                    });
                }
            }
        }

        } // end ViewMode::Sequence

        ViewMode::Graph => {
            App::render_graph_panels(
                &self.runner, &self.all_node_values, self.graph_cursor, self.graph_scroll,
                &mut vertices, &mut texts, m, left_w, right_x, right_w, bot_y, bot_h, win_w, win_h,
            );
        }
        ViewMode::DAG => {
            // DAG uses full screen — panels above are still drawn but DAG overlays the bottom
            self.render_dag(&mut vertices, &mut texts, win_w, win_h);
        }
      } // end match view_mode

        // Controls
        let controls = match self.view_mode {
            ViewMode::Sequence =>
                "[Tab]=Graph  [Left/Right]=Step  [S]=Train  [N]=New  [Space]=Run/Pause  [R]=Reset  [Q]=Quit",
            ViewMode::Graph =>
                "[Tab]=DAG  [Click/Up/Down]=Node  [Scroll]=Browse  [S]=Train  [N]=New  [Space]=Run/Pause  [Q]=Quit",
            ViewMode::DAG =>
                "[Tab]=Seq  [Drag]=Pan  [Scroll]=Zoom  [Click]=Select  [S]=Train  [N]=New  [Q]=Quit",
        };
        texts.push(TextEntry {
            text: controls.into(),
            x: 5.0, y: win_h - 30.0, size: 24.0, color: td,
        });

        // ── Upload geometry and render ──
        let gpu = self.gpu.as_mut().unwrap();
        let vertex_count = vertices.len();
        if vertex_count == 0 { return; }

        if vertex_count > gpu.max_vertices {
            gpu.vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            gpu.max_vertices = vertex_count;
        } else {
            gpu.queue.write_buffer(&gpu.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        // ── Prepare text ──
        let mut text_buffers: Vec<GlyphonBuffer> = Vec::new();
        for entry in &texts {
            let mut buffer = GlyphonBuffer::new(&mut gpu.font_system, Metrics::new(entry.size, entry.size * 1.3));
            buffer.set_size(&mut gpu.font_system, Some(win_w), Some(win_h));
            buffer.set_text(
                &mut gpu.font_system, &entry.text,
                &Attrs::new().family(Family::Monospace).color(entry.color),
                Shaping::Basic, None,
            );
            buffer.shape_until_scroll(&mut gpu.font_system, false);
            text_buffers.push(buffer);
        }

        let text_areas: Vec<TextArea> = text_buffers.iter().zip(texts.iter()).map(|(buf, entry)| {
            TextArea {
                buffer: buf,
                left: entry.x,
                top: entry.y,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0, top: 0,
                    right: win_w as i32,
                    bottom: win_h as i32,
                },
                default_color: entry.color,
                custom_glyphs: &[],
            }
        }).collect();

        gpu.viewport.update(&gpu.queue, Resolution {
            width: win_w as u32, height: win_h as u32,
        });

        gpu.text_renderer.prepare(
            &gpu.device, &gpu.queue, &mut gpu.font_system,
            &mut gpu.text_atlas, &gpu.viewport, text_areas,
            &mut gpu.swash_cache,
        ).unwrap();

        // ── Render pass ──
        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.08, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&gpu.pipeline);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            pass.draw(0..vertex_count as u32, 0..1);

            gpu.text_renderer.render(&gpu.text_atlas, &gpu.viewport, &mut pass).unwrap();
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        gpu.text_atlas.trim();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("Transformer Visualizer")
                    .with_inner_size(winit::dpi::LogicalSize::new(1400, 850)),
            ).unwrap()
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device").into(),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        )).unwrap();

        let size = window.inner_size();
        let config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).unwrap();
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ColorVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            multiview_mask: None,
            cache: None,
        });

        let max_vertices = 200_000;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_buffer"),
            size: (max_vertices * std::mem::size_of::<ColorVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, config.format);
        let text_renderer = TextRenderer::new(&mut text_atlas, &device, wgpu::MultisampleState::default(), None);
        let viewport = Viewport::new(&device, &cache);

        self.gpu = Some(GpuState {
            window, device, queue, surface, config, pipeline,
            vertex_buffer, max_vertices,
            font_system, swash_cache, text_atlas, text_renderer, viewport,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x as f32, position.y as f32);
                if self.dag_dragging {
                    self.dag_offset.0 += new_pos.0 - self.mouse_pos.0;
                    self.dag_offset.1 += new_pos.1 - self.mouse_pos.1;
                }
                self.mouse_pos = new_pos;
            }
            WindowEvent::PinchGesture { delta, .. } => {
                if self.view_mode == ViewMode::DAG {
                    let old_zoom = self.dag_zoom;
                    self.dag_zoom = (self.dag_zoom * (1.0 + delta as f32)).clamp(0.15, 3.0);
                    // Zoom toward mouse position
                    let (mx, my) = self.mouse_pos;
                    let scale = self.dag_zoom / old_zoom;
                    self.dag_offset.0 = mx - (mx - self.dag_offset.0) * scale;
                    self.dag_offset.1 = my - (my - self.dag_offset.1) * scale;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.view_mode == ViewMode::Graph {
                    let scroll_amount = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => (y * -3.0) as isize,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y * -0.1) as isize,
                    };
                    let n_nodes = self.runner.inf_graph.nodes.len();
                    let new_scroll = (self.graph_scroll as isize + scroll_amount).max(0) as usize;
                    self.graph_scroll = new_scroll.min(n_nodes.saturating_sub(1));
                } else if self.view_mode == ViewMode::DAG {
                    // Scroll wheel pans in DAG mode
                    let (pan_dx, pan_dy) = match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => (x * 30.0, y * 30.0),
                        winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                    };
                    self.dag_offset.0 += pan_dx;
                    self.dag_offset.1 += pan_dy;
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: winit::event::MouseButton::Left, .. } => {
                if self.view_mode == ViewMode::DAG {
                    let (mx, my) = self.mouse_pos;
                    let mut hit = false;
                    let lx = compute_layout_x(&self.dag.nodes, self.dag_zoom, &self.runner.inf_graph);
                    for (i, node) in self.dag.nodes.iter_mut().enumerate() {
                        let cur_w = node_effective_w(node, self.dag_zoom, &self.runner.inf_graph);
                        let cur_h = node_effective_h(node, self.dag_zoom, &self.runner.inf_graph);
                        let nx = lx[i] + self.dag_offset.0;
                        let ny = node.y * self.dag_zoom + self.dag_offset.1 - (cur_h - NODE_H * self.dag_zoom) * 0.5;
                        if mx >= nx && mx <= nx + cur_w && my >= ny && my <= ny + cur_h {
                            if node.node_range.1 > node.node_range.0 {
                                node.expanded = !node.expanded;
                            }
                            self.dag_selected = Some(i);
                            hit = true;
                            break;
                        }
                    }
                    if !hit {
                        self.dag_dragging = true;
                        self.dag_selected = None;
                    }
                } else if self.view_mode == ViewMode::Graph {
                    // Hit test against node list
                    let (mx, my) = self.mouse_pos;
                    let m = 10.0;
                    let bot_y = if let Some(gpu) = &self.gpu {
                        gpu.window.inner_size().height as f32 * 0.55 + m
                    } else { 0.0 };
                    let list_top = bot_y + 10.0 + 22.0;
                    let row_h = 22.0;
                    if mx > m && mx < self.gpu.as_ref().map(|g| g.window.inner_size().width as f32 * 0.6).unwrap_or(0.0)
                        && my > list_top
                    {
                        let row = ((my - list_top) / row_h) as usize;
                        let node_idx = self.graph_scroll + row;
                        if node_idx < self.runner.inf_graph.nodes.len() {
                            self.graph_cursor = node_idx;
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Released, button: winit::event::MouseButton::Left, .. } => {
                self.dag_dragging = false;
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, ..
            } => {
                let max_pos = self.current_example.len().saturating_sub(1);
                let n_nodes = self.runner.inf_graph.nodes.len();
                match logical_key.as_ref() {
                    Key::Named(NamedKey::Tab) => {
                        self.view_mode = match self.view_mode {
                            ViewMode::Sequence => ViewMode::Graph,
                            ViewMode::Graph => ViewMode::DAG,
                            ViewMode::DAG => ViewMode::Sequence,
                        };
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        self.cursor_pos = (self.cursor_pos + 1).min(max_pos);
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.cursor_pos = self.cursor_pos.saturating_sub(1);
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if self.view_mode == ViewMode::Graph {
                            self.graph_cursor = (self.graph_cursor + 1).min(n_nodes.saturating_sub(1));
                            // Auto-scroll to keep cursor visible
                            let win_h = self.gpu.as_ref().map(|g| g.window.inner_size().height as f32).unwrap_or(850.0);
                            let bot_h = win_h - (win_h * 0.55 + 10.0) - 10.0;
                            let visible_rows = ((bot_h - 20.0) / 22.0) as usize;
                            if self.graph_cursor >= self.graph_scroll + visible_rows {
                                self.graph_scroll = self.graph_cursor - visible_rows + 1;
                            }
                        }
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        if self.view_mode == ViewMode::Graph {
                            self.graph_cursor = self.graph_cursor.saturating_sub(1);
                            if self.graph_cursor < self.graph_scroll {
                                self.graph_scroll = self.graph_cursor;
                            }
                        }
                    }
                    Key::Character("s") => {
                        self.train_one_step();
                        self.mode = Mode::Paused;
                    }
                    Key::Character("n") => {
                        self.new_example();
                    }
                    Key::Named(NamedKey::Space) => {
                        self.mode = match self.mode {
                            Mode::Paused => Mode::Training,
                            Mode::Training => Mode::Paused,
                        };
                    }
                    Key::Character("r") => {
                        self.runner.weights = model::init_weights_pub(&self.runner.cfg);
                        let sizes: Vec<usize> = self.runner.weights.iter().map(|w| w.len()).collect();
                        self.adam = Adam::new(&sizes, 0.001);
                        self.loss_history.clear();
                        self.step = 0;
                        self.new_example();
                    }
                    Key::Character("q") => {
                        event_loop.exit();
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                if self.mode == Mode::Training {
                    self.train_one_step();
                }
                // Animate DAG node expansion
                let anim_speed = 0.08;
                for node in &mut self.dag.nodes {
                    let target = if node.expanded { 1.0 } else { 0.0 };
                    node.expand_t += (target - node.expand_t) * anim_speed;
                    if (node.expand_t - target).abs() < 0.005 {
                        node.expand_t = target;
                    }
                }
                if self.gpu.is_some() {
                    self.render();
                }
            }
            _ => {}
        }

        if let Some(gpu) = &self.gpu {
            gpu.window.request_redraw();
        }
    }
}

const SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let runner = if args.iter().any(|a| a == "--pretrained") {
        let idx = args.iter().position(|a| a == "--pretrained").unwrap();
        let default = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .join("chord_grammar_model.bin");
        let path = if idx + 1 < args.len() && !args[idx + 1].starts_with('-') {
            std::path::PathBuf::from(&args[idx + 1])
        } else {
            default
        };
        ModelRunner::from_pretrained(&path)
    } else {
        ModelRunner::new()
    };

    if args.iter().any(|a| a == "--dump-graph") {
        for i in 0..runner.inf_graph.nodes.len() {
            eprintln!("{}", model::describe_node(&runner.inf_graph, i));
        }
        eprintln!("--- {} total nodes, logits_idx={} ---", runner.inf_graph.nodes.len(), runner.inf_logits_idx);
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new(runner);
    event_loop.run_app(&mut app).unwrap();
}
