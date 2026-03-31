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
    // Which graph to show: inference or training
    show_train_graph: bool,
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
            show_train_graph: false,
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
        let gpu = self.gpu.as_mut().unwrap();
        let (win_w, win_h) = {
            let size = gpu.window.inner_size();
            (size.width as f32, size.height as f32)
        };
        if win_w < 1.0 || win_h < 1.0 { return; }

        let mut vertices: Vec<ColorVertex> = Vec::with_capacity(50_000);
        let mut texts: Vec<TextEntry> = Vec::new();

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
      } // end match view_mode

        // Controls
        let controls = match self.view_mode {
            ViewMode::Sequence =>
                "[Tab]=Graph view  [Left/Right]=Step pos  [S]=Train step  [N]=New example  [Space]=Run/Pause  [R]=Reset  [Q]=Quit",
            ViewMode::Graph =>
                "[Tab]=Seq view  [Click/Up/Down]=Select node  [Scroll]=Browse  [S]=Train step  [N]=New example  [Space]=Run/Pause  [Q]=Quit",
        };
        texts.push(TextEntry {
            text: controls.into(),
            x: 5.0, y: win_h - 30.0, size: 24.0, color: td,
        });

        // ── Upload geometry ──
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
                self.mouse_pos = (position.x as f32, position.y as f32);
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
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: winit::event::MouseButton::Left, .. } => {
                if self.view_mode == ViewMode::Graph {
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
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. }, ..
            } => {
                let max_pos = self.current_example.len().saturating_sub(1);
                let n_nodes = self.runner.inf_graph.nodes.len();
                match logical_key.as_ref() {
                    Key::Named(NamedKey::Tab) => {
                        self.view_mode = match self.view_mode {
                            ViewMode::Sequence => ViewMode::Graph,
                            ViewMode::Graph => ViewMode::Sequence,
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

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new(runner);
    event_loop.run_app(&mut app).unwrap();
}
