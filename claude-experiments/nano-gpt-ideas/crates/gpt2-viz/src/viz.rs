use crate::gpt2::{Gpt2Config, NodeInfo, MAX_SEQ_LEN};
use crate::viewer::PointVertex;

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

pub struct WallTile {
    pub vertices: Vec<PointVertex>,
    pub labels: Vec<String>,
}

/// Build the wall of all materialized nodes.
/// `tile_values`: per-node data, Some = computed, None = pending.
/// `view_bounds`: optional (min_x, max_x, min_y, max_y) for frustum culling.
pub fn build_wall(
    tile_values: &[Option<Vec<f32>>],
    node_infos: &[NodeInfo],
    config: &Gpt2Config,
    n_tokens: usize,
    focus_token: usize,
    stale: bool,
) -> Vec<(WallTile, f32, f32)> {
    let cols = 30;
    let tile_spacing_x = 2.2;
    let tile_spacing_y = 2.2;
    let tile_w = 2.0;
    let tile_h = 2.0;
    let total = node_infos.len();

    let mut result = Vec::new();

    for node_idx in 0..total {
        let info = &node_infos[node_idx];

        let col = node_idx % cols;
        let row = node_idx / cols;
        let base_x = col as f32 * tile_spacing_x - (cols as f32 * tile_spacing_x) / 2.0;
        let base_y = -(row as f32 * tile_spacing_y) + 4.0;

        let mut tile = match &tile_values[node_idx] {
            Some(data) => build_tile(data, info, config, n_tokens, focus_token, base_x, base_y, tile_w, tile_h),
            None => build_placeholder(info, base_x, base_y, tile_w, tile_h),
        };
        // Dim stale tiles (old data while recomputing)
        if stale && tile_values[node_idx].is_some() {
            for v in &mut tile.vertices {
                v.color[3] *= 0.3;
            }
        }
        result.push((tile, base_x, base_y));
    }

    result
}

/// Dim outline placeholder for tiles not yet computed.
fn build_placeholder(
    _info: &NodeInfo, base_x: f32, base_y: f32, tile_w: f32, tile_h: f32,
) -> WallTile {
    let color = [0.15, 0.15, 0.2, 0.3];
    // Just 4 corner points as a dim outline
    let mut vertices = Vec::new();
    let mut labels = Vec::new();
    for &(dx, dy) in &[(0.0, 0.0), (tile_w, 0.0), (0.0, -tile_h), (tile_w, -tile_h)] {
        vertices.push(PointVertex { position: [base_x + dx, base_y + dy, 0.0], color });
        labels.push("loading...".into());
    }
    // Fill in the outline edges with a few more points
    let n = 8;
    for i in 0..n {
        let t = i as f32 / n as f32;
        vertices.push(PointVertex { position: [base_x + t * tile_w, base_y, 0.0], color });
        labels.push("loading...".into());
        vertices.push(PointVertex { position: [base_x + t * tile_w, base_y - tile_h, 0.0], color });
        labels.push("loading...".into());
        vertices.push(PointVertex { position: [base_x, base_y - t * tile_h, 0.0], color });
        labels.push("loading...".into());
        vertices.push(PointVertex { position: [base_x + tile_w, base_y - t * tile_h, 0.0], color });
        labels.push("loading...".into());
    }
    WallTile { vertices, labels }
}

pub fn count_interesting_nodes(node_infos: &[NodeInfo]) -> usize {
    node_infos.len()
}

/// Choose the right 3D representation based on tensor shape.
fn build_tile(
    data: &[f32],
    info: &NodeInfo,
    config: &Gpt2Config,
    n_tokens: usize,
    focus: usize,
    base_x: f32, base_y: f32,
    tile_w: f32, tile_h: f32,
) -> WallTile {
    let shape = &info.shape;

    if shape.len() == 4 && shape[1] == config.n_head
        && shape[2] == MAX_SEQ_LEN && shape[3] == MAX_SEQ_LEN
    {
        return build_attention_3d(data, info, config, n_tokens, focus, base_x, base_y, tile_w, tile_h);
    }

    if shape.len() == 3 && shape[1] == MAX_SEQ_LEN && shape[2] >= 16 {
        return build_hidden_3d(data, info, config, n_tokens, focus, base_x, base_y, tile_w, tile_h);
    }

    if shape.len() == 4 && shape[1] <= 16 && shape[2] == MAX_SEQ_LEN {
        return build_4d_3d(data, info, n_tokens, focus, base_x, base_y, tile_w, tile_h);
    }

    build_flat_heatmap(data, info, n_tokens, base_x, base_y, tile_w, tile_h)
}

// ─── [1, H, T, T] attention patterns — heads as Z depth ────────────────

fn build_attention_3d(
    data: &[f32], info: &NodeInfo, config: &Gpt2Config, n_tokens: usize, focus: usize,
    base_x: f32, base_y: f32, tile_w: f32, tile_h: f32,
) -> WallTile {
    let n_head = config.n_head;
    let t = n_tokens.min(MAX_SEQ_LEN);
    let full_t = MAX_SEQ_LEN;
    let mut vertices = Vec::new();
    let mut labels = Vec::new();

    for head in 0..n_head {
        for q in 0..t {
            for k in 0..=q {
                let idx = head * full_t * full_t + q * full_t + k;
                let weight = data.get(idx).copied().unwrap_or(0.0);

                let x = base_x + (k as f32 / t.max(1) as f32) * tile_w;
                let y = base_y - (q as f32 / t.max(1) as f32) * tile_h;
                let z = (head as f32 / (n_head - 1).max(1) as f32) * 2.0 - 1.0;

                let base_c = head_color(head, n_head);
                let w = weight.sqrt();
                let focused = q == focus;
                let dim = if focused { 1.0 } else { 0.15 };
                let color = [
                    base_c[0] * (0.3 + 0.7 * w) * dim,
                    base_c[1] * (0.3 + 0.7 * w) * dim,
                    base_c[2] * (0.3 + 0.7 * w) * dim,
                    (w * 1.5).clamp(0.05, 1.0) * dim,
                ];
                vertices.push(PointVertex { position: [x, y, z], color });
                labels.push(format!("[{}] {} H{head} q={q} k={k} w={weight:.4}", info.graph_idx, info.op_name));
            }
        }
    }

    WallTile { vertices, labels }
}

// ─── [1, T, D] hidden states — X=dim, Y=value (height), Z=token ────────

fn build_hidden_3d(
    data: &[f32], info: &NodeInfo, _config: &Gpt2Config, n_tokens: usize, focus: usize,
    base_x: f32, base_y: f32, tile_w: f32, tile_h: f32,
) -> WallTile {
    let t = n_tokens.min(MAX_SEQ_LEN);
    let d = info.shape[2];
    let full_t = MAX_SEQ_LEN;

    // Subsample dimensions
    let max_d = 64;
    let d_step = (d / max_d).max(1);
    let n_d = (d / d_step).min(max_d);

    // Find value range
    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for pos in 0..t {
        let row_start = pos * d;
        for di in (0..d).step_by(d_step) {
            let v = data.get(row_start + di).copied().unwrap_or(0.0);
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    let vrange = (vmax - vmin).max(1e-7);

    let mut vertices = Vec::new();
    let mut labels = Vec::new();

    for pos in 0..t {
        let row_start = pos * d;
        for (di_idx, di) in (0..d).step_by(d_step).enumerate().take(n_d) {
            let v = data.get(row_start + di).copied().unwrap_or(0.0);
            let norm = (v - vmin) / vrange; // 0..1

            // X = dimension index spread across tile width
            let x = base_x + (di_idx as f32 / n_d.max(1) as f32) * tile_w;
            // Y = base + height proportional to value (rises off the wall)
            let y_base = base_y - (tile_h * 0.5);
            let y = y_base + (norm - 0.5) * tile_h;
            // Z = token position (depth into screen)
            let z = (pos as f32 / t.max(1) as f32) * 2.0 - 1.0;

            let base_c = head_color(pos % 12, 12);
            let intensity = (norm * 2.0 - 1.0).abs();
            let focused = pos == focus;
            let dim = if focused { 1.0 } else { 0.15 };
            let color = [
                base_c[0] * (0.3 + 0.7 * intensity) * dim,
                base_c[1] * (0.3 + 0.7 * intensity) * dim,
                base_c[2] * (0.3 + 0.7 * intensity) * dim,
                (0.2 + intensity * 0.8).min(1.0) * dim,
            ];
            vertices.push(PointVertex { position: [x, y, z], color });
            labels.push(format!("[{}] {} pos={pos} dim={di} val={v:.4}", info.graph_idx, info.op_name));
        }
    }

    WallTile { vertices, labels }
}

// ─── [1, H, T, S] or similar 4D — heads in Z, T×S as XY ───────────────

fn build_4d_3d(
    data: &[f32], info: &NodeInfo, n_tokens: usize, focus: usize,
    base_x: f32, base_y: f32, tile_w: f32, tile_h: f32,
) -> WallTile {
    let shape = &info.shape;
    let n_head = shape[1];
    let t = n_tokens.min(shape[2]);
    let s = shape[3];
    let full_t = shape[2];

    let max_s = s.min(32);
    let s_step = (s / max_s).max(1);

    let mut vmin = f32::MAX;
    let mut vmax = f32::MIN;
    for &v in data.iter() { vmin = vmin.min(v); vmax = vmax.max(v); }
    let vrange = (vmax - vmin).max(1e-7);

    let mut vertices = Vec::new();
    let mut labels = Vec::new();

    for head in 0..n_head {
        for pos in 0..t {
            for si in (0..s).step_by(s_step).take(max_s) {
                let idx = head * full_t * s + pos * s + si;
                let v = data.get(idx).copied().unwrap_or(0.0);
                let norm = (v - vmin) / vrange;

                let x = base_x + (si as f32 / s.max(1) as f32) * tile_w;
                let y = base_y - (pos as f32 / t.max(1) as f32) * tile_h;
                let z = (head as f32 / n_head.max(1) as f32) * 2.0 - 1.0;

                let mut color = viridis(norm);
                if pos != focus { for c in &mut color[..4] { *c *= 0.15; } }
                vertices.push(PointVertex { position: [x, y, z], color });
                labels.push(format!("[{}] {} H{head} pos={pos} s={si} val={v:.4}", info.graph_idx, info.op_name));
            }
        }
    }

    WallTile { vertices, labels }
}

// ─── Flat heatmap fallback for small/1D/2D tensors ──────────────────────

fn build_flat_heatmap(
    data: &[f32], info: &NodeInfo, n_tokens: usize,
    base_x: f32, base_y: f32, tile_w: f32, tile_h: f32,
) -> WallTile {
    let shape = &info.shape;
    let mut vertices = Vec::new();
    let mut labels = Vec::new();

    let (rows, cols, flat) = collapse_to_2d(data, shape, n_tokens);
    if rows == 0 || cols == 0 {
        return WallTile { vertices, labels };
    }

    let max_r = rows.min(48);
    let max_c = cols.min(96);
    let row_step = (rows / max_r).max(1);
    let col_step = (cols / max_c).max(1);

    let fmin = flat.iter().copied().fold(f32::MAX, f32::min);
    let fmax = flat.iter().copied().fold(f32::MIN, f32::max);
    let frange = (fmax - fmin).max(1e-7);

    for ri in 0..max_r {
        let r = ri * row_step;
        for ci in 0..max_c {
            let c = ci * col_step;
            let v = flat[r * cols + c];
            let t = (v - fmin) / frange;
            let x = base_x + (ci as f32 / max_c as f32) * tile_w;
            let y = base_y - (ri as f32 / max_r as f32) * tile_h;
            let color = viridis(t);
            vertices.push(PointVertex { position: [x, y, 0.0], color });
            labels.push(format!("[{}] {} r={r} c={c} val={v:.4}", info.graph_idx, info.op_name));
        }
    }

    WallTile { vertices, labels }
}

fn collapse_to_2d<'a>(data: &'a [f32], shape: &[usize], n_tokens: usize) -> (usize, usize, &'a [f32]) {
    if shape.is_empty() || data.is_empty() { return (0, 0, &[]); }
    match shape.len() {
        1 => (1, shape[0].min(n_tokens * 4), data),
        2 => (shape[0].min(n_tokens), shape[1], data),
        3 => (shape[1].min(n_tokens), shape[2], data),
        _ => {
            let r = shape[shape.len() - 2];
            let c = shape[shape.len() - 1];
            (r, c, data)
        }
    }
}
