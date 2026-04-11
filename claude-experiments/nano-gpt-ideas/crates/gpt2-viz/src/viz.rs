use crate::gpt2::{Gpt2Config, NodeInfo, SemanticLayout, MAX_SEQ_LEN};
use crate::viewer::{LineVertex, PointVertex};

fn viridis(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let r = (0.267 + t * (0.993 - 0.267 + t * (-1.02 + t * 0.76))).clamp(0.0, 1.0);
    let g = (0.004 + t * (1.53 + t * (-1.74 + t * 0.76))).clamp(0.0, 1.0);
    let b = (0.329 + t * (1.16 + t * (-2.24 + t * 1.15))).clamp(0.0, 1.0);
    [r, g, b, 1.0]
}

fn head_color(idx: usize, count: usize) -> [f32; 4] {
    let hue = idx as f32 / count.max(1) as f32;
    let h = hue * 6.0;
    let c = 0.95 * 0.85;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = 0.95 - c;
    let (r, g, b) = match h as usize {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m, 1.0]
}

pub struct MachineGeometry {
    pub points: Vec<PointVertex>,
    pub lines: Vec<LineVertex>,
    pub labels: Vec<String>,
}

// Layout constants
const NODE_X_SPACING: f32 = 1.5;         // spacing between nodes within a layer (X)
const LAYER_Y_SPACING: f32 = 5.0;        // spacing between layers (Y)
const TOKEN_Z_SPACING: f32 = 3.5;        // spacing between tokens (Z)
const RAIL_FOOTPRINT: f32 = 1.0;         // width/depth of one rail's heightfield
const HEIGHT_SCALE: f32 = 1.2;           // Y range for values
const DIM_SUBSAMPLE: usize = 4;          // render every Nth dim

/// Assign each node to a layer based on graph_idx position relative to attention nodes.
fn compute_node_layers(
    node_infos: &[NodeInfo],
    layout: &SemanticLayout,
    n_layer: usize,
) -> Vec<Option<usize>> {
    // Get the graph_idx of each layer's attention node (mid-layer marker)
    let mut layer_anchors: Vec<usize> = Vec::with_capacity(n_layer);
    for l in &layout.layers {
        if let Some(ni) = l.attention {
            layer_anchors.push(node_infos[ni].graph_idx);
        } else {
            // Fallback: use residual_attn
            if let Some(ni) = l.residual_attn {
                layer_anchors.push(node_infos[ni].graph_idx);
            } else {
                layer_anchors.push(usize::MAX);
            }
        }
    }

    // Compute boundaries: node belongs to layer i if graph_idx is between
    // (layer_anchors[i-1] + 1) and (layer_anchors[i] + stuff from next layer's start)
    // Simpler: find the midpoint between consecutive layer anchors
    let mut boundaries = Vec::with_capacity(n_layer + 1);
    boundaries.push(0);
    for i in 0..n_layer.saturating_sub(1) {
        let mid = (layer_anchors[i] + layer_anchors[i + 1]) / 2;
        boundaries.push(mid);
    }
    // Last boundary: include everything after the last layer's anchor
    if let Some(&last) = layer_anchors.last() {
        boundaries.push(last.saturating_add(1000));
    } else {
        boundaries.push(usize::MAX);
    }

    // Embedding boundary: before layer 0's anchor's earliest predecessor
    // We'll treat everything before the first layer's anchor minus some buffer as embedding
    let first_layer_start = if n_layer > 0 {
        // Find the earliest node classified as belonging to layer 0
        // Simple heuristic: everything before the midpoint to layer 1
        if layer_anchors.len() >= 2 {
            (layer_anchors[0] + layer_anchors[1]) / 2
        } else {
            layer_anchors[0]
        }
    } else {
        usize::MAX
    };
    let _ = first_layer_start;

    node_infos.iter().map(|info| {
        let gid = info.graph_idx;
        // Find which layer's range contains this gid
        for l in 0..n_layer {
            let lo = if l == 0 { 0 } else { boundaries[l] };
            let hi = boundaries[l + 1];
            if gid >= lo && gid < hi {
                return Some(l);
            }
        }
        None
    }).collect()
}

/// Build the whole machine: every materialized node as a heightfield.
pub fn build_machine(
    tile_values: &[Option<Vec<f32>>],
    node_infos: &[NodeInfo],
    layout: &SemanticLayout,
    config: &Gpt2Config,
    n_tokens: usize,
    focus: usize,
    stale: bool,
) -> MachineGeometry {
    let mut points = Vec::new();
    let mut lines = Vec::new();
    let mut labels = Vec::new();

    // Assign each node to a layer
    let node_layer = compute_node_layers(node_infos, layout, config.n_layer);

    // Group nodes by layer, preserving graph order
    let mut layer_nodes: Vec<Vec<usize>> = vec![Vec::new(); config.n_layer];
    let mut pre_layer: Vec<usize> = Vec::new(); // embedding and pre-layer stuff
    let mut post_layer: Vec<usize> = Vec::new(); // final LN, logits
    for (ni, layer_opt) in node_layer.iter().enumerate() {
        match layer_opt {
            Some(l) => layer_nodes[*l].push(ni),
            None => {
                // Before any layer or after all layers
                // Use graph_idx to decide: before first layer anchor = pre, after last = post
                if !layer_nodes.iter().any(|v| !v.is_empty()) {
                    pre_layer.push(ni);
                } else {
                    // Check if it's before everything
                    let first_assigned_gid = node_infos[
                        *layer_nodes.iter().flatten().next().unwrap_or(&0)
                    ].graph_idx;
                    if node_infos[ni].graph_idx < first_assigned_gid {
                        pre_layer.push(ni);
                    } else {
                        post_layer.push(ni);
                    }
                }
            }
        }
    }

    // Render pre-layer nodes (embedding) at the top
    let mut current_x = 0.0f32;
    let pre_y = 2.0;
    for &ni in &pre_layer {
        if let Some(data) = &tile_values[ni] {
            emit_node_rails(&mut points, &mut labels, data, &node_infos[ni],
                config, n_tokens, focus, current_x, pre_y, stale);
        }
        current_x += NODE_X_SPACING;
    }

    // Render each layer as a row
    for (layer_idx, nodes) in layer_nodes.iter().enumerate() {
        let layer_y = -(layer_idx as f32 + 1.0) * LAYER_Y_SPACING;
        let mut x = 0.0f32;

        // Track the X position of this layer's attention node for arc anchoring
        let mut attention_x: Option<f32> = None;
        let mut attention_ni: Option<usize> = None;

        for &ni in nodes {
            if let Some(data) = &tile_values[ni] {
                emit_node_rails(&mut points, &mut labels, data, &node_infos[ni],
                    config, n_tokens, focus, x, layer_y, stale);

                // Is this the attention softmax node?
                if let crate::gpt2::NodeRole::Attention(_) = &node_infos[ni].role {
                    attention_x = Some(x);
                    attention_ni = Some(ni);
                }
            }
            x += NODE_X_SPACING;
        }

        // Emit attention arcs at the attention node's position
        if let (Some(ax), Some(ani)) = (attention_x, attention_ni) {
            if let Some(data) = &tile_values[ani] {
                emit_attention_arcs(&mut lines, data, config, n_tokens, focus, ax, layer_y, stale);
            }
        }
    }

    // Render post-layer (final LN, logits) below the last layer
    let post_y = -((config.n_layer as f32 + 1.0) * LAYER_Y_SPACING);
    let mut x = 0.0f32;
    for &ni in &post_layer {
        if let Some(data) = &tile_values[ni] {
            emit_node_rails(&mut points, &mut labels, data, &node_infos[ni],
                config, n_tokens, focus, x, post_y, stale);
        }
        x += NODE_X_SPACING;
    }

    MachineGeometry { points, lines, labels }
}

/// Emit heightfield rails for one node across all tokens.
fn emit_node_rails(
    points: &mut Vec<PointVertex>,
    labels: &mut Vec<String>,
    data: &[f32],
    info: &NodeInfo,
    config: &Gpt2Config,
    n_tokens: usize,
    focus: usize,
    base_x: f32,
    base_y: f32,
    stale: bool,
) {
    let shape = &info.shape;

    // Find per-token stride from shape
    // Common shapes: [1, T, D], [1, H, T, T], [1, T, 1], etc.
    let (per_token_size, is_attention) = classify_shape(shape, config);

    if per_token_size == 0 { return; }

    // Find global value range
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for &v in data.iter() { if v.is_finite() { vmin = vmin.min(v); vmax = vmax.max(v); } }
    let vrange = (vmax - vmin).max(1e-7);

    for token in 0..n_tokens {
        let z_center = token as f32 * TOKEN_Z_SPACING;
        let focused = token == focus;

        // Extract this token's slice of values
        let values = extract_token_slice(data, shape, token, is_attention, config);
        if values.is_empty() { continue; }

        // Lay out values in a grid, Y = value
        let n = values.len();
        let n_render = (n + DIM_SUBSAMPLE - 1) / DIM_SUBSAMPLE;
        if n_render == 0 { continue; }
        let grid_w = ((n_render as f32).sqrt().ceil() as usize).max(1);
        let grid_h = (n_render + grid_w - 1) / grid_w;

        let mut render_idx = 0;
        for di in (0..n).step_by(DIM_SUBSAMPLE) {
            let v = values[di];
            let norm = ((v - vmin) / vrange).clamp(0.0, 1.0);
            let gx = render_idx % grid_w;
            let gy = render_idx / grid_w;
            render_idx += 1;

            let x = base_x + (gx as f32 / grid_w as f32 - 0.5) * RAIL_FOOTPRINT;
            let z = z_center + (gy as f32 / grid_h.max(1) as f32 - 0.5) * RAIL_FOOTPRINT;
            let y = base_y + (norm - 0.5) * HEIGHT_SCALE;

            let mut color = viridis(norm);
            if !focused {
                for c in color.iter_mut().take(3) { *c *= 0.25; }
                color[3] = 0.3;
            }
            if stale { color[3] *= 0.4; }

            points.push(PointVertex { position: [x, y, z], color });
            labels.push(format!("[{}] {} tok={token} dim={di} val={v:.4}", info.graph_idx, info.op_name));
        }
    }
}

/// Return (elements_per_token, is_attention_pattern) for various shapes.
fn classify_shape(shape: &[usize], config: &Gpt2Config) -> (usize, bool) {
    match shape.len() {
        // [T] or [D] — 1D
        1 => (shape[0], false),
        // [B, T] or [D1, D2]
        2 => {
            if shape[0] == MAX_SEQ_LEN { (shape[1], false) }
            else { (shape[1], false) }
        }
        // [1, T, D]
        3 if shape[1] == MAX_SEQ_LEN => (shape[2], false),
        // [1, H, T, T] attention
        4 if shape[1] == config.n_head && shape[2] == MAX_SEQ_LEN && shape[3] == MAX_SEQ_LEN => {
            (config.n_head * MAX_SEQ_LEN, true)
        }
        // [1, H, T, S] (Q/K/V per head)
        4 if shape[1] <= 16 && shape[2] == MAX_SEQ_LEN => {
            (shape[1] * shape[3], false)
        }
        _ => {
            // Last dim as "per token" fallback
            (*shape.last().unwrap_or(&0), false)
        }
    }
}

/// Extract a token's slice of the tensor.
fn extract_token_slice<'a>(
    data: &'a [f32],
    shape: &[usize],
    token: usize,
    is_attention: bool,
    config: &Gpt2Config,
) -> Vec<f32> {
    match shape.len() {
        1 => {
            // Shared across tokens
            data.to_vec()
        }
        2 if shape[0] == MAX_SEQ_LEN => {
            let d = shape[1];
            let start = token * d;
            data.get(start..start+d).map(|s| s.to_vec()).unwrap_or_default()
        }
        2 => data.to_vec(),
        3 if shape[1] == MAX_SEQ_LEN => {
            let d = shape[2];
            let start = token * d;
            data.get(start..start+d).map(|s| s.to_vec()).unwrap_or_default()
        }
        4 if is_attention => {
            // [1, H, T, T] — extract row for focused token as query across all heads
            let h = config.n_head;
            let full_t = MAX_SEQ_LEN;
            let mut out = Vec::with_capacity(h * full_t);
            for head in 0..h {
                for k in 0..full_t {
                    let idx = head * full_t * full_t + token * full_t + k;
                    out.push(data.get(idx).copied().unwrap_or(0.0));
                }
            }
            out
        }
        4 if shape[1] <= 16 && shape[2] == MAX_SEQ_LEN => {
            // [1, H, T, S]
            let h = shape[1];
            let s = shape[3];
            let mut out = Vec::with_capacity(h * s);
            for head in 0..h {
                for si in 0..s {
                    let idx = head * MAX_SEQ_LEN * s + token * s + si;
                    out.push(data.get(idx).copied().unwrap_or(0.0));
                }
            }
            out
        }
        _ => data.to_vec(),
    }
}

/// Emit attention arcs connecting tokens at the attention node's position.
fn emit_attention_arcs(
    lines: &mut Vec<LineVertex>,
    data: &[f32],
    config: &Gpt2Config,
    n_tokens: usize,
    focus: usize,
    base_x: f32,
    base_y: f32,
    stale: bool,
) {
    let h = config.n_head;
    let full_t = MAX_SEQ_LEN;
    let head_x_spread = 0.6;

    for head in 0..h {
        let hx = base_x + (head as f32 / (h - 1).max(1) as f32 - 0.5) * head_x_spread;
        let base_c = head_color(head, h);

        for q in 0..n_tokens {
            for k in 0..=q {
                let idx = head * full_t * full_t + q * full_t + k;
                let w = data.get(idx).copied().unwrap_or(0.0);
                if w < 0.02 { continue; }

                let focused = q == focus || k == focus;

                let z_q = q as f32 * TOKEN_Z_SPACING;
                let z_k = k as f32 * TOKEN_Z_SPACING;

                let arc_height = ((q - k) as f32 * 0.5).max(0.1) + 0.5;
                let n_segs = 6;
                let mut prev: Option<[f32; 3]> = None;

                let brightness = w.sqrt();
                let alpha_base = if focused { 1.0 } else { 0.2 };
                let color = [
                    base_c[0] * (0.3 + 0.7 * brightness),
                    base_c[1] * (0.3 + 0.7 * brightness),
                    base_c[2] * (0.3 + 0.7 * brightness),
                    (brightness * alpha_base).clamp(0.05, 1.0) * if stale { 0.4 } else { 1.0 },
                ];

                for seg in 0..=n_segs {
                    let t = seg as f32 / n_segs as f32;
                    let u = 1.0 - t;
                    let mid_z = (z_q + z_k) * 0.5;
                    let mid_y = base_y + arc_height;
                    let px = hx;
                    let py = u * u * base_y + 2.0 * u * t * mid_y + t * t * base_y;
                    let pz = u * u * z_k + 2.0 * u * t * mid_z + t * t * z_q;
                    let cur = [px, py, pz];
                    if let Some(prev) = prev {
                        lines.push(LineVertex { position: prev, color });
                        lines.push(LineVertex { position: cur, color });
                    }
                    prev = Some(cur);
                }
            }
        }
    }
}
