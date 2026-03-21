use crate::source::{LoadResult, MinimapRow, PointCloudSource, PointVertex};
use heapster::hprof::parser::HprofFile;
use heapster::hprof::segment::ObjectMeta;
use heapster::hprof::types::Id;
use std::collections::HashMap;

/// Heap dump visualization mapped into the cube like trigrams:
///   X = class identity (deterministic hash)
///   Y = reference depth from GC roots
///   Z = outgoing reference count
/// Color = class. Blocks = depth layers for scrubbing.
pub struct HeapCubeSource {
    pub num_blocks: usize,
    pub max_objects: usize,
}

impl Default for HeapCubeSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_objects: 500_000,
        }
    }
}

impl PointCloudSource for HeapCubeSource {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn load(&self, path: &std::path::Path, _data: &[u8]) -> LoadResult {
        eprintln!("[heap-cube] Loading: {}", path.display());

        let hprof = match HprofFile::open(path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Failed to parse heap dump: {e}");
                return empty_result(self.num_blocks);
            }
        };

        // Collect objects
        let mut objects: Vec<ObjectMeta> = Vec::new();
        if let Err(e) = hprof.visit_metadata_objects(|obj| {
            objects.push(obj);
            Ok(())
        }) {
            eprintln!("Failed to scan objects: {e}");
            return empty_result(self.num_blocks);
        }
        eprintln!("[heap-cube] {} objects", objects.len());

        // Reference graph
        let ref_graph = match hprof.ref_graph() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to build ref graph: {e}");
                return empty_result(self.num_blocks);
            }
        };

        // Sample if needed
        let sampled = if objects.len() > self.max_objects {
            eprintln!("[heap-cube] Sampling {} objects", self.max_objects);
            let step = objects.len() / self.max_objects;
            objects
                .iter()
                .step_by(step.max(1))
                .copied()
                .take(self.max_objects)
                .collect::<Vec<_>>()
        } else {
            objects
        };

        // BFS depth from GC roots (over ALL objects, not just sampled)
        eprintln!("[heap-cube] BFS from GC roots...");
        let gc_roots = hprof.gc_roots().unwrap_or_default();
        let sampled_set: std::collections::HashSet<Id> =
            sampled.iter().map(|o| o.object_id).collect();
        let mut obj_depth: HashMap<Id, u32> = HashMap::with_capacity(sampled.len());
        {
            let mut queue = std::collections::VecDeque::new();
            for &root_id in &gc_roots {
                if sampled_set.contains(&root_id) && !obj_depth.contains_key(&root_id) {
                    obj_depth.insert(root_id, 0);
                    queue.push_back(root_id);
                }
            }
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
        eprintln!(
            "[heap-cube] max depth={max_depth}, reachable={}, unreachable={}",
            obj_depth.len(),
            sampled.len() - obj_depth.len()
        );

        // Incoming ref counts (how "important" — how many things point to this object)
        let in_counts: HashMap<Id, u32> = sampled
            .iter()
            .map(|obj| {
                let c = ref_graph.referrers(obj.object_id).len() as u32;
                (obj.object_id, c)
            })
            .collect();
        let max_in = in_counts.values().copied().max().unwrap_or(1).max(1);
        let log_max_in = (max_in as f32 + 1.0).ln();

        // Map each class to a deterministic X position using golden-ratio hash
        // This spreads classes evenly across [-1, 1]
        let mut class_x: HashMap<Id, f32> = HashMap::new();
        {
            let mut class_counter = 0u32;
            for obj in &sampled {
                class_x.entry(obj.class_id).or_insert_with(|| {
                    class_counter += 1;
                    // Golden ratio spacing mod 1 → even distribution
                    let frac = (class_counter as f64 * 0.6180339887498949) % 1.0;
                    frac as f32 * 2.0 - 1.0
                });
            }
        }

        // Build vertices: x=class, y=depth(rank), z=shallow_size(rank)
        // Brightness = incoming ref count (more referenced = brighter)

        // Find max shallow size for log scaling
        let max_size = sampled
            .iter()
            .map(|o| o.shallow_size)
            .max()
            .unwrap_or(1)
            .max(1);
        let log_max_size = (max_size as f32 + 1.0).ln();

        struct VertexWithDepth {
            vertex: PointVertex,
            depth: u32,
        }

        // Sort by depth, then use rank for Y
        struct ObjSorted {
            obj: ObjectMeta,
            depth: u32,
            in_count: u32,
        }
        let mut sorted: Vec<ObjSorted> = sampled
            .iter()
            .map(|obj| {
                let depth = obj_depth.get(&obj.object_id).copied().unwrap_or(max_depth);
                let in_count = in_counts.get(&obj.object_id).copied().unwrap_or(0);
                ObjSorted {
                    obj: *obj,
                    depth,
                    in_count,
                }
            })
            .collect();
        sorted.sort_unstable_by_key(|o| o.depth);

        let n = sorted.len().max(1) as f32;

        let mut verts: Vec<VertexWithDepth> = Vec::with_capacity(sorted.len());
        for (rank, entry) in sorted.iter().enumerate() {
            // X: class identity with jitter
            let cx = class_x[&entry.obj.class_id];
            let jx = micro_hash(entry.obj.object_id, 0) * 0.02;

            // Y: rank in depth-sorted list (top = shallowest, bottom = deepest)
            let y_base = 1.0 - 2.0 * (rank as f32 / n);
            let jy = micro_hash(entry.obj.object_id, 1) * 0.01;

            // Z: shallow size (log scale)
            let z_base = (entry.obj.shallow_size as f32 + 1.0).ln() / log_max_size * 2.0 - 1.0;
            let jz = micro_hash(entry.obj.object_id, 2) * 0.02;

            let [r, g, b] = class_color(entry.obj.class_id);

            // Brightness from incoming ref count (more referenced = brighter/more opaque)
            let brightness = if entry.in_count == 0 {
                0.3
            } else {
                0.3 + 0.7 * ((entry.in_count as f32 + 1.0).ln() / log_max_in).clamp(0.0, 1.0)
            };

            verts.push(VertexWithDepth {
                vertex: PointVertex {
                    position: [
                        (cx + jx).clamp(-1.0, 1.0),
                        (y_base + jy).clamp(-1.0, 1.0),
                        (z_base + jz).clamp(-1.0, 1.0),
                    ],
                    color: [r * brightness, g * brightness, b * brightness, 0.5],
                },
                depth: entry.depth,
            });
        }

        let all_vertices: Vec<PointVertex> = verts.iter().map(|v| v.vertex).collect();

        // Blocks by depth
        let num_blocks = self.num_blocks;
        let block_size = (all_vertices.len() + num_blocks - 1) / num_blocks.max(1);
        let mut block_ranges = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let start = (b * block_size).min(all_vertices.len()) as u32;
            let end = ((b + 1) * block_size).min(all_vertices.len()) as u32;
            block_ranges.push((start, end - start));
        }

        // Minimap: depth distribution
        let minimap_rows: Vec<MinimapRow> = (0..512)
            .map(|row| {
                let block = row * num_blocks / 512;
                let count = block_ranges.get(block).map(|r| r.1).unwrap_or(0);
                let start = block_ranges.get(block).map(|r| r.0 as usize).unwrap_or(0);
                let end = block_ranges
                    .get(block)
                    .map(|r| (r.0 + r.1) as usize)
                    .unwrap_or(0);
                let avg_depth = if end > start {
                    let sum: u32 = verts[start..end].iter().map(|v| v.depth).sum();
                    sum as f32 / (end - start) as f32
                } else {
                    0.0
                };
                MinimapRow {
                    avg_byte: (avg_depth / max_depth as f32 * 255.0).clamp(0.0, 255.0),
                    entropy: (count as f32 / block_size.max(1) as f32).clamp(0.0, 1.0),
                }
            })
            .collect();

        eprintln!(
            "[heap-cube] Done: {} vertices, max_depth={max_depth}, max_shallow_size={max_size}, max_in_refs={max_in}",
            all_vertices.len()
        );

        LoadResult {
            vertices: all_vertices,
            inspect_points: vec![],
            block_ranges,
            minimap_rows,
        }
    }
}

/// Deterministic small jitter from object ID. Returns value in [-1, 1].
fn micro_hash(id: Id, seed: u64) -> f32 {
    let h = id
        .wrapping_mul(2654435761)
        .wrapping_add(seed.wrapping_mul(1442695040888963407));
    (h & 0xFFFF) as f32 / 32768.0 - 1.0
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

fn empty_result(num_blocks: usize) -> LoadResult {
    LoadResult {
        vertices: vec![],
        inspect_points: vec![],
        block_ranges: vec![(0, 0); num_blocks],
        minimap_rows: vec![],
    }
}
