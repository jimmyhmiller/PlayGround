use crate::source::{InspectPoint, LoadResult, MinimapRow, PointCloudSource, PointVertex};
use heapster::hprof::parser::HprofFile;
use heapster::hprof::types::Id;
use std::collections::HashMap;
use std::sync::Arc;

pub struct HeapLandscapeSource {
    pub num_blocks: usize,
    pub max_objects: usize,
}

struct LandscapeVertex {
    vertex: PointVertex,
    depth: u32,
    retained: u64,
    inspect: InspectPoint,
}

impl Default for HeapLandscapeSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_objects: 450_000,
        }
    }
}

impl PointCloudSource for HeapLandscapeSource {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn load(&self, path: &std::path::Path, _data: &[u8]) -> LoadResult {
        eprintln!("[heap-landscape] Loading: {}", path.display());

        let hprof = match HprofFile::open(path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Failed to parse heap dump: {e}");
                return empty_result(self.num_blocks);
            }
        };

        let dominator = match hprof.dominator_tree() {
            Ok(tree) => tree,
            Err(e) => {
                eprintln!("Failed to build dominator tree: {e}");
                return empty_result(self.num_blocks);
            }
        };

        let mut object_class: HashMap<Id, Id> = HashMap::new();
        if let Err(e) = hprof.visit_metadata_objects(|obj| {
            object_class.insert(obj.object_id, obj.class_id);
            Ok(())
        }) {
            eprintln!("Failed to scan objects: {e}");
            return empty_result(self.num_blocks);
        }

        let node_count = dominator.node_count();
        if node_count <= 1 {
            return empty_result(self.num_blocks);
        }

        let mut children: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        for node_idx in 1..node_count {
            if let Some(parent_idx) = dominator.immediate_dominator_index(node_idx) {
                if parent_idx < node_count && parent_idx != node_idx {
                    children[parent_idx].push(node_idx);
                }
            }
        }

        let mut depth = vec![0u32; node_count];
        let mut root_branch = vec![0usize; node_count];
        let mut stack = vec![0usize];
        while let Some(node_idx) = stack.pop() {
            let branch = root_branch[node_idx];
            let next_depth = depth[node_idx] + 1;
            for &child in &children[node_idx] {
                depth[child] = next_depth;
                root_branch[child] = if node_idx == 0 { child } else { branch };
                stack.push(child);
            }
        }

        let max_depth = depth.iter().copied().max().unwrap_or(1).max(1);
        let root_retained = dominator.retained_size_at(0).unwrap_or(1).max(1);
        let root_children = &children[0];
        if root_children.is_empty() {
            return empty_result(self.num_blocks);
        }
        let mut branch_slot: HashMap<usize, usize> = HashMap::new();
        for (slot, &child) in root_children.iter().enumerate() {
            branch_slot.insert(child, slot);
        }

        let mut intervals = vec![(0.0f32, 1.0f32); node_count];
        let mut x_cursor = 0.0f32;
        for &child in root_children {
            let retained = dominator.retained_size_at(child).unwrap_or(1).max(1);
            let width = retained as f32 / root_retained as f32;
            let end = (x_cursor + width).min(1.0);
            intervals[child] = (x_cursor, end);
            assign_subtree_intervals(child, &children, &dominator, &mut intervals);
            x_cursor = end;
        }
        if let Some(&last) = root_children.last() {
            intervals[last].1 = 1.0;
            assign_subtree_intervals(last, &children, &dominator, &mut intervals);
        }

        let mut nodes: Vec<NodeVisual> = Vec::with_capacity(node_count - 1);
        for node_idx in 1..node_count {
            let object_id = match dominator.object_id_at(node_idx) {
                Some(id) if id != 0 => id,
                _ => continue,
            };
            let retained = dominator.retained_size_at(node_idx).unwrap_or(0);
            let class_id = object_class.get(&object_id).copied().unwrap_or(0);
            let class_name = hprof.class_name(class_id);
            let package = package_name(&class_name);
            let package_color = package_color(package);
            let branch_hue = hue_from_id(root_branch[node_idx] as u64);
            let branch_slot = branch_slot
                .get(&root_branch[node_idx])
                .copied()
                .unwrap_or(0);
            let retained_t = (retained as f32 / root_retained as f32).clamp(0.0, 1.0);
            let label: Arc<str> = format!(
                "{}  object=0x{:x}  retained={}  depth={}  owner={}",
                class_name, object_id, retained, depth[node_idx], package
            )
            .into();
            nodes.push(NodeVisual {
                object_id,
                retained,
                label,
                depth: depth[node_idx],
                interval: intervals[node_idx],
                branch_hue,
                branch_slot,
                package_color,
                retained_t,
            });
        }

        nodes.sort_unstable_by(|a, b| b.retained.cmp(&a.retained));
        if nodes.len() > self.max_objects {
            nodes.truncate(self.max_objects);
        }

        let max_retained = nodes
            .iter()
            .map(|node| node.retained)
            .max()
            .unwrap_or(1)
            .max(1);
        let log_max_retained = (max_retained as f32 + 1.0).ln().max(1.0);

        let mut verts: Vec<LandscapeVertex> = Vec::with_capacity(nodes.len());
        for node in &nodes {
            let (x0, x1) = node.interval;
            let interval_width = (x1 - x0).max(0.0005);
            let center_x = (x0 + x1) * 0.5;
            let depth_t = node.depth as f32 / max_depth as f32;

            let branch_count = root_children.len().max(1) as f32;
            let branch_center = if branch_count <= 1.0 {
                0.0
            } else {
                -0.85 + (node.branch_slot as f32 / (branch_count - 1.0)) * 1.7
            };
            let subtree_bias = ((center_x * 2.0) - 1.0) * 0.22;
            let pile_center_x = ((center_x * 2.0) - 1.0) * 0.9;
            let pile_center_z = (branch_center + subtree_bias).clamp(-0.9, 0.9);

            let base_radius_x = (interval_width * 1.2).clamp(0.015, 0.22);
            let base_radius_z = (0.06 + interval_width * 0.45).clamp(0.03, 0.18);
            let angle = hash01(node.object_id, 0) * std::f32::consts::TAU;
            let radius = hash01(node.object_id, 1).sqrt();
            let local_x = angle.cos() * radius;
            let local_z = angle.sin() * radius;
            let x = (pile_center_x + local_x * base_radius_x).clamp(-1.0, 1.0);
            let z = (pile_center_z + local_z * base_radius_z).clamp(-1.0, 1.0);

            let peak_height =
                ((node.retained as f32 + 1.0).ln() / log_max_retained).clamp(0.0, 1.0);
            let radial = radius.clamp(0.0, 1.0);
            let surface = -1.0 + peak_height * 1.9 * (1.0 - radial.powf(1.15));
            let fill = hash01(node.object_id, 2).powf(0.28 + depth_t * 0.35);
            let y = lerp(-1.0, surface.max(-0.98), fill).clamp(-1.0, 1.0);

            let brightness = 0.34 + 0.66 * peak_height;
            let [pr, pg, pb] = node.package_color;
            let [rr, rg, rb] = hsv_to_rgb(node.branch_hue * 360.0, 0.45, 0.95);
            let color = [
                (pr * 0.7 + rr * 0.3) * brightness,
                (pg * 0.7 + rg * 0.3) * brightness,
                (pb * 0.7 + rb * 0.3) * brightness,
                (0.16 + node.retained_t * 0.42).clamp(0.16, 0.62),
            ];

            verts.push(LandscapeVertex {
                vertex: PointVertex {
                    position: [x, y, z],
                    color,
                },
                depth: node.depth,
                retained: node.retained,
                inspect: InspectPoint {
                    label: node.label.clone(),
                },
            });
        }

        verts.sort_unstable_by(|a, b| a.depth.cmp(&b.depth).then(b.retained.cmp(&a.retained)));

        let vertices: Vec<PointVertex> = verts.iter().map(|v| v.vertex).collect();
        let inspect_points: Vec<InspectPoint> = verts.iter().map(|v| v.inspect.clone()).collect();
        let block_ranges = contiguous_block_ranges(vertices.len(), self.num_blocks);
        let minimap_rows = build_minimap_rows(&verts, &block_ranges, max_depth, max_retained);

        eprintln!(
            "[heap-landscape] Done: {} vertices, max_depth={}, max_retained={}",
            vertices.len(),
            max_depth,
            max_retained
        );

        LoadResult {
            vertices,
            inspect_points,
            info_lines: vec![],
            block_ranges,
            minimap_rows,
        }
    }
}

struct NodeVisual {
    object_id: Id,
    retained: u64,
    label: Arc<str>,
    depth: u32,
    interval: (f32, f32),
    branch_hue: f32,
    branch_slot: usize,
    package_color: [f32; 3],
    retained_t: f32,
}

fn assign_subtree_intervals(
    root_idx: usize,
    children: &[Vec<usize>],
    dominator: &heapster::hprof::dominator::DominatorTree,
    intervals: &mut [(f32, f32)],
) {
    let (x0, x1) = intervals[root_idx];
    let total = dominator.retained_size_at(root_idx).unwrap_or(1).max(1) as f32;
    let mut cursor = x0;
    let child_indices = &children[root_idx];
    for &child_idx in child_indices {
        let weight = dominator.retained_size_at(child_idx).unwrap_or(1).max(1) as f32;
        let width = (x1 - x0) * (weight / total);
        let end = if child_idx == *child_indices.last().unwrap() {
            x1
        } else {
            (cursor + width).min(x1)
        };
        intervals[child_idx] = (cursor, end);
        assign_subtree_intervals(child_idx, children, dominator, intervals);
        cursor = end;
    }
}

fn build_minimap_rows(
    verts: &[impl VertexDepthRetained],
    block_ranges: &[(u32, u32)],
    max_depth: u32,
    max_retained: u64,
) -> Vec<MinimapRow> {
    let max_retained_f = max_retained as f32;
    (0..512)
        .map(|row| {
            let block = row * block_ranges.len().max(1) / 512;
            let (start, count) = block_ranges.get(block).copied().unwrap_or((0, 0));
            let end = (start + count) as usize;
            let start = start as usize;
            if start >= end || end > verts.len() {
                return MinimapRow {
                    avg_byte: 0.0,
                    entropy: 0.0,
                };
            }
            let slice = &verts[start..end];
            let avg_depth =
                slice.iter().map(|v| v.depth()).sum::<u32>() as f32 / slice.len() as f32;
            let avg_retained =
                slice.iter().map(|v| v.retained()).sum::<u64>() as f32 / slice.len() as f32;
            MinimapRow {
                avg_byte: ((avg_depth / max_depth.max(1) as f32) * 160.0
                    + (avg_retained / max_retained_f.max(1.0)) * 95.0)
                    .clamp(0.0, 255.0),
                entropy: (count as f32 / verts.len().max(1) as f32 * 28.0).clamp(0.0, 1.0),
            }
        })
        .collect()
}

trait VertexDepthRetained {
    fn depth(&self) -> u32;
    fn retained(&self) -> u64;
}

impl VertexDepthRetained for LandscapeVertex {
    fn depth(&self) -> u32 {
        self.depth
    }

    fn retained(&self) -> u64 {
        self.retained
    }
}

impl VertexDepthRetained for (PointVertex, u32, u64) {
    fn depth(&self) -> u32 {
        self.1
    }

    fn retained(&self) -> u64 {
        self.2
    }
}

fn contiguous_block_ranges(total_vertices: usize, num_blocks: usize) -> Vec<(u32, u32)> {
    let block_size = (total_vertices + num_blocks.saturating_sub(1)) / num_blocks.max(1);
    let mut block_ranges = Vec::with_capacity(num_blocks);
    for block in 0..num_blocks {
        let start = (block * block_size).min(total_vertices) as u32;
        let end = ((block + 1) * block_size).min(total_vertices) as u32;
        block_ranges.push((start, end - start));
    }
    block_ranges
}

fn package_name(class_name: &str) -> &str {
    if let Some((package, _)) = class_name.rsplit_once('.') {
        package
    } else {
        "<root>"
    }
}

fn package_color(package: &str) -> [f32; 3] {
    let mut hash = 0u64;
    for b in package.as_bytes() {
        hash = hash.wrapping_mul(131).wrapping_add(*b as u64);
    }
    hsv_to_rgb(hue_from_id(hash) * 360.0, 0.72, 0.92)
}

fn hue_from_id(id: u64) -> f32 {
    ((id.wrapping_mul(2654435761)) & 0xFFFF_FFFF) as f32 / u32::MAX as f32
}

fn hash01(id: Id, seed: u64) -> f32 {
    let mixed = id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((mixed >> 11) & 0xFFFF) as f32 / 65535.0
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0..60 => (c, x, 0.0),
        60..120 => (x, c, 0.0),
        120..180 => (0.0, c, x),
        180..240 => (0.0, x, c),
        240..300 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

fn empty_result(num_blocks: usize) -> LoadResult {
    LoadResult {
        vertices: vec![],
        inspect_points: vec![],
        info_lines: vec![],
        block_ranges: vec![(0, 0); num_blocks],
        minimap_rows: vec![],
    }
}
