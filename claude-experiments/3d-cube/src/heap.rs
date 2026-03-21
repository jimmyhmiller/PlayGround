use crate::source::{LoadResult, MinimapRow, PointCloudSource, PointVertex};
use heapster::hprof::parser::HprofFile;
use heapster::hprof::segment::ObjectMeta;
use heapster::hprof::types::Id;
use rand::Rng;
use std::collections::HashMap;

pub struct HeapDumpSource {
    pub num_blocks: usize,
    pub max_objects: usize,
    pub class_layout_iterations: usize,
}

impl Default for HeapDumpSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_objects: 500_000,
            class_layout_iterations: 100,
        }
    }
}

fn class_color(class_id: Id) -> [f32; 3] {
    let h = ((class_id.wrapping_mul(2654435761)) & 0xFFFFFFFF) as f32 / u32::MAX as f32;
    hsv_to_rgb(h * 360.0, 0.7, 0.9)
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

#[derive(Clone)]
struct ClassEdge {
    target_class: usize,
    weight: u32,
}

impl PointCloudSource for HeapDumpSource {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn load(&self, path: &std::path::Path, _data: &[u8]) -> LoadResult {
        self.load_from_path(path)
    }
}

impl HeapDumpSource {
    fn load_from_path(&self, path: &std::path::Path) -> LoadResult {
        eprintln!("Loading heap dump: {}", path.display());

        let hprof = match HprofFile::open(path) {
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

        // object_id -> class_id lookup
        let mut obj_class: HashMap<Id, Id> = HashMap::with_capacity(objects.len());
        for obj in &objects {
            obj_class.insert(obj.object_id, obj.class_id);
        }

        // --- Step 1: Build class-level graph ---
        eprintln!("Building class-level graph...");

        let mut class_to_idx: HashMap<Id, usize> = HashMap::new();
        let mut class_ids: Vec<Id> = Vec::new();
        let mut class_instance_count: Vec<u32> = Vec::new();
        for obj in &objects {
            let idx = class_to_idx.len();
            let entry = class_to_idx.entry(obj.class_id).or_insert(idx);
            if *entry == idx {
                class_ids.push(obj.class_id);
                class_instance_count.push(0);
            }
            class_instance_count[*entry] += 1;
        }
        let num_classes = class_ids.len();
        eprintln!("{num_classes} unique classes");

        // Count inter-class references
        let mut class_edges: HashMap<(usize, usize), u32> = HashMap::new();
        for obj in &objects {
            let src_class = class_to_idx[&obj.class_id];
            for &target_id in ref_graph.outgoing(obj.object_id) {
                if let Some(&target_class_id) = obj_class.get(&target_id) {
                    let tgt_class = class_to_idx[&target_class_id];
                    if src_class != tgt_class {
                        *class_edges.entry((src_class, tgt_class)).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut class_adj: Vec<Vec<ClassEdge>> = vec![vec![]; num_classes];
        for (&(src, tgt), &weight) in &class_edges {
            class_adj[src].push(ClassEdge {
                target_class: tgt,
                weight,
            });
            class_adj[tgt].push(ClassEdge {
                target_class: src,
                weight,
            });
        }

        // --- Step 2: Force-directed layout on class graph ---
        eprintln!(
            "Laying out {} classes ({} inter-class edges)...",
            num_classes,
            class_edges.len()
        );
        let class_positions = self.layout_classes(num_classes, &class_adj, &class_instance_count);

        // --- Step 3: Place objects near their class center ---
        // Sample if needed
        let sampled = if objects.len() > self.max_objects {
            eprintln!(
                "Sampling {} from {} objects",
                self.max_objects,
                objects.len()
            );
            sample_objects(&objects, self.max_objects)
        } else {
            objects
        };
        eprintln!("Placing {} objects...", sampled.len());

        let mut rng = rand::thread_rng();

        // Class radius proportional to sqrt(instance_count) relative to max
        let max_instances = class_instance_count.iter().copied().max().unwrap_or(1) as f32;
        let class_radii: Vec<f32> = class_instance_count
            .iter()
            .map(|&c| (c as f32 / max_instances).sqrt() * 0.12 + 0.005)
            .collect();

        // --- Step 3b: BFS from GC roots to compute depth per object ---
        eprintln!("Computing BFS depths from GC roots...");
        let gc_roots = hprof.gc_roots().unwrap_or_default();
        let mut obj_depth: HashMap<Id, u32> = HashMap::with_capacity(sampled.len());
        {
            let sampled_set: std::collections::HashSet<Id> =
                sampled.iter().map(|o| o.object_id).collect();
            let mut queue = std::collections::VecDeque::new();

            // Seed BFS from GC roots that are in our sampled set
            for &root_id in &gc_roots {
                if sampled_set.contains(&root_id) && !obj_depth.contains_key(&root_id) {
                    obj_depth.insert(root_id, 0);
                    queue.push_back(root_id);
                }
            }

            eprintln!("BFS seeded with {} GC roots", obj_depth.len());

            // BFS traversal
            while let Some(id) = queue.pop_front() {
                let d = obj_depth[&id];
                for &target_id in ref_graph.outgoing(id) {
                    if sampled_set.contains(&target_id) && !obj_depth.contains_key(&target_id) {
                        obj_depth.insert(target_id, d + 1);
                        queue.push_back(target_id);
                    }
                }
            }
        }

        let max_depth = obj_depth.values().copied().max().unwrap_or(1).max(1);
        let unreachable_count = sampled.len() - obj_depth.len();
        eprintln!(
            "BFS complete: max depth {max_depth}, {} reachable, {} unreachable",
            obj_depth.len(),
            unreachable_count
        );

        // Build vertices with depth info, then sort by depth
        struct VertexWithDepth {
            vertex: PointVertex,
            depth: u32,
        }

        let mut verts_with_depth: Vec<VertexWithDepth> = Vec::with_capacity(sampled.len());

        for obj in &sampled {
            let ci = class_to_idx[&obj.class_id];
            let center = class_positions[ci];
            let radius = class_radii[ci];

            // Uniform random point inside a sphere
            let (ox, oy, oz) = loop {
                let x = rng.r#gen::<f32>() * 2.0 - 1.0;
                let y = rng.r#gen::<f32>() * 2.0 - 1.0;
                let z = rng.r#gen::<f32>() * 2.0 - 1.0;
                if x * x + y * y + z * z <= 1.0 {
                    break (x * radius, y * radius, z * radius);
                }
            };

            let pos = [center[0] + ox, center[1] + oy, center[2] + oz];
            let [cr, cg, cb] = class_color(obj.class_id);
            let depth = obj_depth.get(&obj.object_id).copied().unwrap_or(max_depth);

            verts_with_depth.push(VertexWithDepth {
                vertex: PointVertex {
                    position: pos,
                    color: [cr, cg, cb, 0.4],
                },
                depth,
            });
        }

        // Sort by depth so blocks correspond to reference depth layers
        verts_with_depth.sort_unstable_by_key(|v| v.depth);

        let all_vertices: Vec<PointVertex> = verts_with_depth.iter().map(|v| v.vertex).collect();

        // --- Step 4: Assign blocks by depth ---
        // Each block covers a range of depths. Objects at depth 0 (GC roots) are in
        // the first blocks, deepest objects in the last blocks.
        let num_blocks = self.num_blocks;
        let block_size = (all_vertices.len() + num_blocks - 1) / num_blocks.max(1);
        let mut block_ranges = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let start = (b * block_size).min(all_vertices.len()) as u32;
            let end = ((b + 1) * block_size).min(all_vertices.len()) as u32;
            block_ranges.push((start, end - start));
        }

        // Minimap: show depth distribution — color by depth, brightness by density
        let minimap_rows: Vec<MinimapRow> = (0..512)
            .map(|row| {
                let block = row * num_blocks / 512;
                let count = block_ranges.get(block).map(|r| r.1).unwrap_or(0);

                // Figure out the average depth for this block's vertices
                let start = block_ranges.get(block).map(|r| r.0 as usize).unwrap_or(0);
                let end = block_ranges
                    .get(block)
                    .map(|r| (r.0 + r.1) as usize)
                    .unwrap_or(0);
                let avg_depth = if end > start {
                    let sum: u32 = verts_with_depth[start..end].iter().map(|v| v.depth).sum();
                    sum as f32 / (end - start) as f32
                } else {
                    0.0
                };

                MinimapRow {
                    // Map depth to color (low depth = warm/yellow, high depth = cool/blue)
                    avg_byte: (avg_depth / max_depth as f32 * 255.0).clamp(0.0, 255.0),
                    entropy: (count as f32 / block_size.max(1) as f32).clamp(0.0, 1.0),
                }
            })
            .collect();

        eprintln!(
            "Done: {} vertices in {} blocks",
            all_vertices.len(),
            num_blocks
        );

        LoadResult {
            vertices: all_vertices,
            inspect_points: vec![],
            info_lines: vec![],
            block_ranges,
            minimap_rows,
        }
    }

    /// Layout classes by starting evenly spread on a sphere,
    /// then letting connected classes attract each other.
    /// No repulsion needed since initial positions are well-separated.
    fn layout_classes(
        &self,
        num_classes: usize,
        adj: &[Vec<ClassEdge>],
        _instance_counts: &[u32],
    ) -> Vec<[f32; 3]> {
        if num_classes <= 1 {
            return vec![[0.0; 3]; num_classes];
        }

        // Initialize: evenly distributed on a sphere using Fibonacci sphere
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut pos: Vec<[f32; 3]> = (0..num_classes)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / golden;
                let phi = (1.0 - 2.0 * (i as f64 + 0.5) / num_classes as f64).acos();
                let r = 0.85;
                [
                    (r * phi.sin() * theta.cos()) as f32,
                    (r * phi.sin() * theta.sin()) as f32,
                    (r * phi.cos()) as f32,
                ]
            })
            .collect();

        // Attraction-only: connected classes pull toward each other.
        // Use small alpha so they don't all collapse — just nudge neighbors closer.
        for _iter in 0..self.class_layout_iterations {
            let mut disp = vec![[0.0f32; 3]; num_classes];

            for i in 0..num_classes {
                // Count total edge weight for normalization
                let total_weight: f32 =
                    adj[i].iter().map(|e| (e.weight as f32).ln().max(1.0)).sum();
                if total_weight == 0.0 {
                    continue;
                }

                for edge in &adj[i] {
                    let j = edge.target_class;
                    let dx = pos[j][0] - pos[i][0];
                    let dy = pos[j][1] - pos[i][1];
                    let dz = pos[j][2] - pos[i][2];

                    // Normalized weight: fraction of this class's total connectivity
                    let w = (edge.weight as f32).ln().max(1.0) / total_weight;

                    disp[i][0] += dx * w;
                    disp[i][1] += dy * w;
                    disp[i][2] += dz * w;
                }
            }

            // Move each node 5% toward its weighted neighbor centroid
            let alpha = 0.05;
            for i in 0..num_classes {
                pos[i][0] += disp[i][0] * alpha;
                pos[i][1] += disp[i][1] * alpha;
                pos[i][2] += disp[i][2] * alpha;
            }
        }

        // Final normalize to [-0.85, 0.85]
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
                let range = (max[d] - min[d]).max(1e-6);
                p[d] = (p[d] - min[d]) / range * 1.7 - 0.85;
            }
        }

        pos
    }
}

fn sample_objects(objects: &[ObjectMeta], max: usize) -> Vec<ObjectMeta> {
    let step = objects.len() / max;
    objects
        .iter()
        .step_by(step.max(1))
        .copied()
        .take(max)
        .collect()
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
