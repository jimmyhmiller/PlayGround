use crate::source::{LoadResult, MinimapRow, PointCloudSource, PointVertex};
use heapster::hprof::parser::HprofFile;
use heapster::hprof::refs::RefGraph;
use heapster::hprof::segment::ObjectMeta;
use heapster::hprof::types::Id;
use rand::Rng;
use std::collections::HashMap;

pub struct HeapDumpSource {
    pub num_blocks: usize,
    pub max_objects: usize,
    pub layout_iterations: usize,
    pub max_points_per_block: usize,
}

impl Default for HeapDumpSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_objects: 200_000,
            layout_iterations: 50,
            max_points_per_block: 4_000,
        }
    }
}

/// Hash a class ID to an RGB color.
fn class_color(class_id: Id) -> [f32; 3] {
    // Use a hash to generate a hue, then HSV -> RGB with high saturation
    let h = ((class_id.wrapping_mul(2654435761)) & 0xFFFFFFFF) as f32 / u32::MAX as f32;
    let s = 0.7;
    let v = 0.9;
    hsv_to_rgb(h * 360.0, s, v)
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

impl PointCloudSource for HeapDumpSource {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn load(&self, path: &std::path::Path, _data: &[u8]) -> LoadResult {
        self.load_from_path(&path.to_string_lossy())
    }
}

impl HeapDumpSource {
    fn load_from_path(&self, path: &str) -> LoadResult {
        eprintln!("Loading heap dump: {path}");

        let hprof = match HprofFile::open(std::path::Path::new(path)) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Failed to parse heap dump: {e}");
                return empty_result(self.num_blocks);
            }
        };

        eprintln!("Collecting objects...");
        let mut objects: Vec<ObjectMeta> = Vec::new();
        if let Err(e) = hprof.visit_metadata_objects(|obj| {
            objects.push(obj);
            Ok(())
        }) {
            eprintln!("Failed to scan objects: {e}");
            return empty_result(self.num_blocks);
        }
        eprintln!("Found {} objects", objects.len());

        eprintln!("Building reference graph...");
        let ref_graph = match hprof.ref_graph() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to build ref graph: {e}");
                return empty_result(self.num_blocks);
            }
        };
        eprintln!("Ref graph: {} edges", ref_graph.edge_count());

        // Sample if too many objects
        let sampled = if objects.len() > self.max_objects {
            eprintln!("Sampling {} objects from {}", self.max_objects, objects.len());
            sample_objects(&objects, self.max_objects, &ref_graph)
        } else {
            objects
        };

        // Build ID -> index map
        let mut id_to_idx: HashMap<Id, usize> = HashMap::with_capacity(sampled.len());
        for (i, obj) in sampled.iter().enumerate() {
            id_to_idx.insert(obj.object_id, i);
        }

        // Laplacian relaxation layout
        eprintln!("Running layout ({} iterations on {} objects)...", self.layout_iterations, sampled.len());
        let positions = self.layout(&sampled, &ref_graph, &id_to_idx);

        // Assign objects to blocks by BFS depth from high-degree nodes
        let depths = compute_depths(&sampled, &ref_graph, &id_to_idx);
        let max_depth = depths.iter().copied().max().unwrap_or(1).max(1);

        // Build per-block vertices
        let num_blocks = self.num_blocks;
        let mut block_verts: Vec<Vec<PointVertex>> = vec![vec![]; num_blocks];

        for (i, obj) in sampled.iter().enumerate() {
            let block = ((depths[i] as f64 / max_depth as f64) * (num_blocks - 1) as f64) as usize;
            let block = block.min(num_blocks - 1);
            let [r, g, b] = class_color(obj.class_id);
            block_verts[block].push(PointVertex {
                position: positions[i],
                color: [r, g, b, 0.3],
            });
        }

        // Cap per block
        for verts in &mut block_verts {
            if verts.len() > self.max_points_per_block {
                verts.truncate(self.max_points_per_block);
            }
        }

        // Concatenate
        let total: usize = block_verts.iter().map(|v| v.len()).sum();
        let mut all_vertices = Vec::with_capacity(total);
        let mut block_ranges = Vec::with_capacity(num_blocks);
        for verts in block_verts {
            let start = all_vertices.len() as u32;
            let count = verts.len() as u32;
            all_vertices.extend_from_slice(&verts);
            block_ranges.push((start, count));
        }

        // Build minimap rows from depth distribution
        let minimap_rows = (0..512)
            .map(|row| {
                let frac = row as f32 / 512.0;
                let depth = (frac * max_depth as f32) as u32;
                let count = depths.iter().filter(|&&d| d == depth).count();
                MinimapRow {
                    avg_byte: frac * 255.0,
                    entropy: (count as f32 / sampled.len() as f32 * 50.0).clamp(0.0, 1.0),
                }
            })
            .collect();

        eprintln!("Layout complete: {} vertices in {} blocks", all_vertices.len(), num_blocks);

        LoadResult {
            vertices: all_vertices,
            block_ranges,
            minimap_rows,
        }
    }

    /// Laplacian relaxation: iteratively move each node toward the average
    /// position of its neighbors.
    fn layout(
        &self,
        objects: &[ObjectMeta],
        ref_graph: &RefGraph,
        id_to_idx: &HashMap<Id, usize>,
    ) -> Vec<[f32; 3]> {
        let n = objects.len();
        let mut rng = rand::thread_rng();

        // Initialize with random positions in [-1, 1]^3
        let mut pos: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                [
                    rng.r#gen::<f32>() * 2.0 - 1.0,
                    rng.r#gen::<f32>() * 2.0 - 1.0,
                    rng.r#gen::<f32>() * 2.0 - 1.0,
                ]
            })
            .collect();

        // Pre-build adjacency in terms of indices (not IDs) for speed
        let mut neighbors: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, obj) in objects.iter().enumerate() {
            for &target_id in ref_graph.outgoing(obj.object_id) {
                if let Some(&j) = id_to_idx.get(&target_id) {
                    neighbors[i].push(j);
                    neighbors[j].push(i); // bidirectional
                }
            }
        }

        // Iterative relaxation
        let mut new_pos = vec![[0.0f32; 3]; n];
        for iter in 0..self.layout_iterations {
            let alpha = 0.5 * (1.0 - iter as f32 / self.layout_iterations as f32); // decay

            for i in 0..n {
                if neighbors[i].is_empty() {
                    new_pos[i] = pos[i];
                    continue;
                }

                // Average of neighbors
                let mut avg = [0.0f32; 3];
                for &j in &neighbors[i] {
                    avg[0] += pos[j][0];
                    avg[1] += pos[j][1];
                    avg[2] += pos[j][2];
                }
                let k = neighbors[i].len() as f32;
                avg[0] /= k;
                avg[1] /= k;
                avg[2] /= k;

                // Move toward average
                new_pos[i] = [
                    pos[i][0] + alpha * (avg[0] - pos[i][0]),
                    pos[i][1] + alpha * (avg[1] - pos[i][1]),
                    pos[i][2] + alpha * (avg[2] - pos[i][2]),
                ];
            }

            std::mem::swap(&mut pos, &mut new_pos);

            // Rescale to [-1, 1] to prevent collapse
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for p in &pos {
                for d in 0..3 {
                    min[d] = min[d].min(p[d]);
                    max[d] = max[d].max(p[d]);
                }
            }
            for p in &mut pos {
                for d in 0..3 {
                    let range = max[d] - min[d];
                    if range > 0.0 {
                        p[d] = (p[d] - min[d]) / range * 2.0 - 1.0;
                    }
                }
            }
        }

        pos
    }
}

/// Sample objects, preferring those with more references (more "interesting").
fn sample_objects(objects: &[ObjectMeta], max: usize, ref_graph: &RefGraph) -> Vec<ObjectMeta> {
    // Score each object by degree (in + out references)
    let mut scored: Vec<(usize, u32)> = objects
        .iter()
        .enumerate()
        .map(|(i, obj)| {
            let degree = ref_graph.outgoing(obj.object_id).len() as u32;
            (i, degree)
        })
        .collect();

    // Sort by degree descending, take top max
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored.truncate(max);

    scored.iter().map(|&(i, _)| objects[i]).collect()
}

/// Compute BFS depth from high-degree "root" nodes.
fn compute_depths(
    objects: &[ObjectMeta],
    ref_graph: &RefGraph,
    id_to_idx: &HashMap<Id, usize>,
) -> Vec<u32> {
    let n = objects.len();
    let mut depth = vec![u32::MAX; n];
    let mut queue = std::collections::VecDeque::new();

    // Seed from objects with high out-degree (likely roots/containers)
    let mut degrees: Vec<(usize, usize)> = objects
        .iter()
        .enumerate()
        .map(|(i, obj)| (i, ref_graph.outgoing(obj.object_id).len()))
        .collect();
    degrees.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    for &(i, _) in degrees.iter().take(100) {
        depth[i] = 0;
        queue.push_back(i);
    }

    // BFS
    while let Some(i) = queue.pop_front() {
        let d = depth[i];
        for &target_id in ref_graph.outgoing(objects[i].object_id) {
            if let Some(&j) = id_to_idx.get(&target_id) {
                if depth[j] == u32::MAX {
                    depth[j] = d + 1;
                    queue.push_back(j);
                }
            }
        }
    }

    // Assign remaining (disconnected) objects depth 0
    for d in &mut depth {
        if *d == u32::MAX {
            *d = 0;
        }
    }

    depth
}

fn empty_result(num_blocks: usize) -> LoadResult {
    LoadResult {
        vertices: vec![],
        block_ranges: vec![(0, 0); num_blocks],
        minimap_rows: vec![],
    }
}
