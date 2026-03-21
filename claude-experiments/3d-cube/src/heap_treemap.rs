use crate::source::{LoadResult, MinimapRow, PointCloudSource, PointVertex};
use heapster::hprof::parser::HprofFile;
use heapster::hprof::types::Id;
use std::collections::{HashMap, HashSet, VecDeque};

struct TreemapVertexRow {
    vertex: PointVertex,
    package_rank: usize,
    class_rank: usize,
    depth: u32,
    size: u32,
}

pub struct HeapTreemapSource {
    pub num_blocks: usize,
    pub max_points: usize,
}

impl Default for HeapTreemapSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_points: 450_000,
        }
    }
}

impl PointCloudSource for HeapTreemapSource {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn load(&self, path: &std::path::Path, _data: &[u8]) -> LoadResult {
        eprintln!("[heap-treemap] Loading: {}", path.display());

        let hprof = match HprofFile::open(path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Failed to parse heap dump: {e}");
                return empty_result(self.num_blocks);
            }
        };

        let mut objects = Vec::new();
        if let Err(e) = hprof.visit_metadata_objects(|obj| {
            objects.push(obj);
            Ok(())
        }) {
            eprintln!("Failed to scan objects: {e}");
            return empty_result(self.num_blocks);
        }

        if objects.is_empty() {
            return empty_result(self.num_blocks);
        }

        let ref_graph = match hprof.ref_graph() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to build ref graph: {e}");
                return empty_result(self.num_blocks);
            }
        };

        let sampled = if objects.len() > self.max_points {
            let step = (objects.len() / self.max_points).max(1);
            objects
                .iter()
                .step_by(step)
                .copied()
                .take(self.max_points)
                .collect::<Vec<_>>()
        } else {
            objects
        };
        eprintln!("[heap-treemap] Using {} objects", sampled.len());

        let depth_by_id = compute_gc_depths(&hprof, &ref_graph, &sampled);
        let max_depth = depth_by_id.values().copied().max().unwrap_or(1).max(1);

        let mut package_map: HashMap<String, PackageGroup> = HashMap::new();
        for obj in &sampled {
            let class_name = hprof.class_name(obj.class_id);
            let package_name = package_name(&class_name).to_string();
            let package = package_map.entry(package_name).or_default();
            package.total_size += obj.shallow_size as u64;
            package.count += 1;

            let class = package.classes.entry(obj.class_id).or_default();
            class.total_size += obj.shallow_size as u64;
            class.count += 1;
            class.depth_sum += depth_by_id
                .get(&obj.object_id)
                .copied()
                .unwrap_or(max_depth) as u64;
        }

        let mut packages: Vec<PackageLayout> = package_map
            .into_iter()
            .map(|(name, group)| PackageLayout::from_group(name, group))
            .collect();
        packages.sort_unstable_by(|a, b| b.total_size.cmp(&a.total_size));
        assign_package_boxes(&mut packages);

        let max_size = sampled
            .iter()
            .map(|obj| obj.shallow_size)
            .max()
            .unwrap_or(1)
            .max(1);
        let total_size: u64 = packages.iter().map(|package| package.total_size).sum();
        let max_class_size = packages
            .iter()
            .flat_map(|package| package.classes.iter())
            .map(|class| class.total_size)
            .max()
            .unwrap_or(1)
            .max(1) as f32;

        let mut rows = Vec::new();
        for (package_rank, package) in packages.iter().enumerate() {
            for (class_rank, class) in package.classes.iter().enumerate() {
                let class_box = class.bounds.map_into(package.bounds).inset(0.006);
                let target_points = if total_size == 0 {
                    16
                } else {
                    ((class.total_size as f64 / total_size as f64) * self.max_points as f64).round()
                        as usize
                }
                .clamp(12, 20_000);
                let avg_depth = if class.count == 0 {
                    max_depth
                } else {
                    (class.depth_sum / class.count as u64) as u32
                };
                let depth_t = avg_depth as f32 / max_depth as f32;
                let size_t = (class.total_size as f32 / max_class_size)
                    .sqrt()
                    .clamp(0.0, 1.0);
                let [r, g, b] = class_color(class.class_id);
                let brightness = 0.52 + 0.28 * size_t + 0.16 * (1.0 - depth_t);

                for point_index in 0..target_points {
                    let [x, y, z] = stratified_point_in_box(
                        class_box,
                        target_points,
                        point_index,
                        class.class_id,
                    );
                    rows.push(TreemapVertexRow {
                        vertex: PointVertex {
                            position: [x, y, z],
                            color: [r * brightness, g * brightness, b * brightness, 0.32],
                        },
                        package_rank,
                        class_rank,
                        depth: avg_depth,
                        size: class.total_size.min(u32::MAX as u64) as u32,
                    });
                }
            }
        }

        rows.sort_unstable_by(|a, b| {
            a.package_rank
                .cmp(&b.package_rank)
                .then(a.class_rank.cmp(&b.class_rank))
                .then(b.size.cmp(&a.size))
        });

        let vertices: Vec<PointVertex> = rows.iter().map(|row| row.vertex).collect();
        let block_ranges = contiguous_block_ranges(vertices.len(), self.num_blocks);
        let minimap_rows = build_minimap_rows(&rows, &block_ranges, max_depth, max_size);

        eprintln!(
            "[heap-treemap] Done: {} vertices across {} packages",
            vertices.len(),
            packages.len()
        );

        LoadResult {
            vertices,
            inspect_points: vec![],
            block_ranges,
            minimap_rows,
        }
    }
}

#[derive(Default)]
struct PackageGroup {
    total_size: u64,
    count: u32,
    classes: HashMap<Id, ClassGroup>,
}

#[derive(Default)]
struct ClassGroup {
    total_size: u64,
    count: u32,
    depth_sum: u64,
}

struct PackageLayout {
    total_size: u64,
    bounds: Box3,
    classes: Vec<ClassLayout>,
}

impl PackageLayout {
    fn from_group(_name: String, group: PackageGroup) -> Self {
        let mut classes: Vec<ClassLayout> = group
            .classes
            .into_iter()
            .map(|(class_id, class)| ClassLayout {
                class_id,
                total_size: class.total_size,
                count: class.count,
                depth_sum: class.depth_sum,
                bounds: Box3::FULL,
            })
            .collect();
        classes.sort_unstable_by(|a, b| b.total_size.cmp(&a.total_size));
        assign_class_boxes(&mut classes);

        Self {
            total_size: group.total_size,
            bounds: Box3::FULL,
            classes,
        }
    }
}

struct ClassLayout {
    class_id: Id,
    total_size: u64,
    count: u32,
    depth_sum: u64,
    bounds: Box3,
}

fn compute_gc_depths(
    hprof: &HprofFile,
    ref_graph: &heapster::hprof::refs::RefGraph,
    sampled: &[heapster::hprof::segment::ObjectMeta],
) -> HashMap<Id, u32> {
    let gc_roots = hprof.gc_roots().unwrap_or_default();
    let sampled_ids: HashSet<Id> = sampled.iter().map(|obj| obj.object_id).collect();
    let mut depth_by_id = HashMap::with_capacity(sampled.len());
    let mut queue = VecDeque::new();

    for &root_id in &gc_roots {
        if sampled_ids.contains(&root_id) && !depth_by_id.contains_key(&root_id) {
            depth_by_id.insert(root_id, 0);
            queue.push_back(root_id);
        }
    }

    while let Some(id) = queue.pop_front() {
        let depth = depth_by_id[&id];
        for &target_id in ref_graph.outgoing(id) {
            if sampled_ids.contains(&target_id) && !depth_by_id.contains_key(&target_id) {
                depth_by_id.insert(target_id, depth + 1);
                queue.push_back(target_id);
            }
        }
    }

    depth_by_id
}

fn assign_package_boxes(packages: &mut [PackageLayout]) {
    let weights: Vec<f32> = packages
        .iter()
        .map(|package| package.total_size.max(1) as f32)
        .collect();
    let bounds = partition_3d(&weights, Box3::FULL);
    for (package, bound) in packages.iter_mut().zip(bounds) {
        package.bounds = bound;
    }
}

fn assign_class_boxes(classes: &mut [ClassLayout]) {
    let weights: Vec<f32> = classes
        .iter()
        .map(|class| class.total_size.max(1) as f32)
        .collect();
    let bounds = partition_3d(&weights, Box3::FULL);
    for (class, bound) in classes.iter_mut().zip(bounds) {
        class.bounds = bound;
    }
}

fn build_minimap_rows(
    rows: &[TreemapVertexRow],
    block_ranges: &[(u32, u32)],
    max_depth: u32,
    max_size: u32,
) -> Vec<MinimapRow> {
    let max_size_f = max_size as f32;
    (0..512)
        .map(|row| {
            let block = row * block_ranges.len().max(1) / 512;
            let (start, count) = block_ranges.get(block).copied().unwrap_or((0, 0));
            let end = (start + count) as usize;
            let start = start as usize;
            if start >= end || end > rows.len() {
                return MinimapRow {
                    avg_byte: 0.0,
                    entropy: 0.0,
                };
            }

            let slice = &rows[start..end];
            let avg_depth =
                slice.iter().map(|row| row.depth).sum::<u32>() as f32 / slice.len() as f32;
            let avg_size =
                slice.iter().map(|row| row.size).sum::<u32>() as f32 / slice.len() as f32;
            MinimapRow {
                avg_byte: ((avg_depth / max_depth.max(1) as f32) * 180.0
                    + (avg_size / max_size_f.max(1.0)) * 75.0)
                    .clamp(0.0, 255.0),
                entropy: (count as f32 / rows.len().max(1) as f32 * 24.0).clamp(0.0, 1.0),
            }
        })
        .collect()
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

fn partition_3d(weights: &[f32], bounds: Box3) -> Vec<Box3> {
    let mut assignments = vec![Box3::FULL; weights.len()];
    let indices: Vec<usize> = (0..weights.len()).collect();
    assign_partition_recursive(weights, &indices, bounds, 0, &mut assignments);
    assignments
}

fn assign_partition_recursive(
    weights: &[f32],
    indices: &[usize],
    bounds: Box3,
    axis_hint: usize,
    assignments: &mut [Box3],
) {
    if indices.is_empty() {
        return;
    }
    if indices.len() == 1 {
        assignments[indices[0]] = bounds;
        return;
    }

    let axis = if bounds.axis_len(axis_hint % 3) >= bounds.longest_axis_len() * 0.7 {
        axis_hint % 3
    } else {
        bounds.longest_axis()
    };
    let total: f32 = indices.iter().map(|&index| weights[index].max(1e-6)).sum();
    let half = total * 0.5;
    let mut left_total = 0.0;
    let mut split = 0usize;

    for (i, &index) in indices.iter().enumerate() {
        left_total += weights[index].max(1e-6);
        split = i + 1;
        if left_total >= half {
            break;
        }
    }

    if split == 0 || split >= indices.len() {
        split = indices.len() / 2;
    }

    let left_indices = &indices[..split];
    let right_indices = &indices[split..];
    let left_weight: f32 = left_indices
        .iter()
        .map(|&index| weights[index].max(1e-6))
        .sum();
    let ratio = (left_weight / total).clamp(0.05, 0.95);
    let (left_box, right_box) = bounds.split(axis, ratio);

    assign_partition_recursive(weights, left_indices, left_box, (axis + 1) % 3, assignments);
    assign_partition_recursive(
        weights,
        right_indices,
        right_box,
        (axis + 1) % 3,
        assignments,
    );
}

fn hash01(id: Id, seed: u64) -> f32 {
    let mixed = id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((mixed >> 11) & 0xFFFF) as f32 / 65535.0
}

fn stratified_point_in_box(
    bounds: Box3,
    total_points: usize,
    point_index: usize,
    seed: Id,
) -> [f32; 3] {
    let total_points = total_points.max(1);
    let dims = bounds.dimensions();
    let volume = (dims[0] * dims[1] * dims[2]).max(1e-6);
    let density = total_points as f32 / volume;

    let mut nx = (dims[0] * density.cbrt()).round().max(1.0) as usize;
    let mut ny = (dims[1] * density.cbrt()).round().max(1.0) as usize;
    let mut nz = (dims[2] * density.cbrt()).round().max(1.0) as usize;

    while nx * ny * nz < total_points {
        let sx = dims[0] / nx as f32;
        let sy = dims[1] / ny as f32;
        let sz = dims[2] / nz as f32;
        if sx >= sy && sx >= sz {
            nx += 1;
        } else if sy >= sz {
            ny += 1;
        } else {
            nz += 1;
        }
    }

    let ix = point_index % nx;
    let iy = (point_index / nx) % ny;
    let iz = (point_index / (nx * ny)) % nz;
    let point_id = seed ^ point_index as u64;

    let tx = (ix as f32 + hash01(point_id, 0)) / nx as f32;
    let ty = (iy as f32 + hash01(point_id, 1)) / ny as f32;
    let tz = (iz as f32 + hash01(point_id, 2)) / nz as f32;

    [
        lerp(bounds.x_min, bounds.x_max, tx.clamp(0.0, 1.0)),
        lerp(bounds.y_min, bounds.y_max, ty.clamp(0.0, 1.0)),
        lerp(bounds.z_min, bounds.z_max, tz.clamp(0.0, 1.0)),
    ]
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t
}

#[derive(Clone, Copy)]
struct Box3 {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    z_min: f32,
    z_max: f32,
}

impl Box3 {
    const FULL: Self = Self {
        x_min: -1.0,
        x_max: 1.0,
        y_min: -1.0,
        y_max: 1.0,
        z_min: -1.0,
        z_max: 1.0,
    };

    fn width(self) -> f32 {
        self.x_max - self.x_min
    }

    fn height(self) -> f32 {
        self.y_max - self.y_min
    }

    fn depth(self) -> f32 {
        self.z_max - self.z_min
    }

    fn longest_axis(self) -> usize {
        let dims = [self.width(), self.height(), self.depth()];
        let mut axis = 0usize;
        for i in 1..3 {
            if dims[i] > dims[axis] {
                axis = i;
            }
        }
        axis
    }

    fn longest_axis_len(self) -> f32 {
        self.axis_len(self.longest_axis())
    }

    fn axis_len(self, axis: usize) -> f32 {
        match axis {
            0 => self.width(),
            1 => self.height(),
            _ => self.depth(),
        }
    }

    fn dimensions(self) -> [f32; 3] {
        [self.width(), self.height(), self.depth()]
    }

    fn split(self, axis: usize, ratio: f32) -> (Self, Self) {
        match axis {
            0 => {
                let mid = lerp(self.x_min, self.x_max, ratio);
                (Self { x_max: mid, ..self }, Self { x_min: mid, ..self })
            }
            1 => {
                let mid = lerp(self.y_min, self.y_max, ratio);
                (Self { y_max: mid, ..self }, Self { y_min: mid, ..self })
            }
            _ => {
                let mid = lerp(self.z_min, self.z_max, ratio);
                (Self { z_max: mid, ..self }, Self { z_min: mid, ..self })
            }
        }
    }

    fn inset(self, amount: f32) -> Self {
        let dx = amount.min(self.width() * 0.35);
        let dy = amount.min(self.height() * 0.35);
        let dz = amount.min(self.depth() * 0.35);
        Self {
            x_min: self.x_min + dx,
            x_max: self.x_max - dx,
            y_min: self.y_min + dy,
            y_max: self.y_max - dy,
            z_min: self.z_min + dz,
            z_max: self.z_max - dz,
        }
    }

    fn map_into(self, outer: Self) -> Self {
        let nx0 = (self.x_min + 1.0) * 0.5;
        let nx1 = (self.x_max + 1.0) * 0.5;
        let ny0 = (self.y_min + 1.0) * 0.5;
        let ny1 = (self.y_max + 1.0) * 0.5;
        let nz0 = (self.z_min + 1.0) * 0.5;
        let nz1 = (self.z_max + 1.0) * 0.5;
        Self {
            x_min: lerp(outer.x_min, outer.x_max, nx0),
            x_max: lerp(outer.x_min, outer.x_max, nx1),
            y_min: lerp(outer.y_min, outer.y_max, ny0),
            y_max: lerp(outer.y_min, outer.y_max, ny1),
            z_min: lerp(outer.z_min, outer.z_max, nz0),
            z_max: lerp(outer.z_min, outer.z_max, nz1),
        }
    }
}

fn class_color(class_id: Id) -> [f32; 3] {
    let h = ((class_id.wrapping_mul(2654435761)) & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
    hsv_to_rgb(h * 360.0, 0.65, 0.95)
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
