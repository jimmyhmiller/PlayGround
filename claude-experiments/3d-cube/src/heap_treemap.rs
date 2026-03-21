use crate::source::{InspectPoint, LoadResult, MinimapRow, PointCloudSource, PointVertex};
use heapster::hprof::parser::HprofFile;
use heapster::hprof::types::Id;
use std::collections::HashMap;
use std::sync::Arc;

pub struct HeapTreemapSource {
    pub num_blocks: usize,
    pub max_items: usize,
    pub max_points: usize,
    pub max_depth: usize,
    pub min_fraction: f32,
}

impl Default for HeapTreemapSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_items: 2_500,
            max_points: 320_000,
            max_depth: 5,
            min_fraction: 0.002,
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

        let root_retained = dominator.retained_size_at(0).unwrap_or(1).max(1);
        let mut items = Vec::new();
        let mut next_slot = 0usize;
        collect_items(
            &dominator,
            &children,
            &object_class,
            &hprof,
            0,
            root_retained,
            0,
            self,
            &mut next_slot,
            &mut items,
        );

        if items.is_empty() {
            return empty_result(self.num_blocks);
        }

        let total_size: u64 = items.iter().map(|item| item.retained).sum();
        if total_size == 0 {
            return empty_result(self.num_blocks);
        }

        let max_retained = items
            .iter()
            .map(|item| item.retained)
            .max()
            .unwrap_or(1)
            .max(1);
        let max_depth_seen = items
            .iter()
            .map(|item| item.depth)
            .max()
            .unwrap_or(1)
            .max(1);
        assign_item_rects(&mut items, Rect::FULL);

        let mut rows = Vec::new();
        for item in &items {
            let point_count = ((item.retained as f64 / total_size as f64) * self.max_points as f64)
                .round() as usize;
            let point_count = point_count.clamp(12, 24_000);

            let rect = item.rect.inset(0.006);
            let retained_t = (item.retained as f32 / max_retained as f32)
                .sqrt()
                .clamp(0.0, 1.0);
            let depth_t = item.depth as f32 / max_depth_seen as f32;
            let [r, g, b] = package_color(&item.package);
            let alpha = (0.18 + retained_t * 0.32).clamp(0.18, 0.55);
            let brightness = 0.48 + retained_t * 0.4 - depth_t * 0.12;

            for point_index in 0..point_count {
                let [x, z] =
                    stratified_point_in_rect(rect, point_count, point_index, item.object_id);
                let y = lerp(-0.12, 0.12, hash01(item.object_id ^ point_index as u64, 2));
                rows.push(TreemapRow {
                    vertex: PointVertex {
                        position: [x, y, z],
                        color: [r * brightness, g * brightness, b * brightness, alpha],
                    },
                    depth: item.depth as u32,
                    retained: item.retained.min(u32::MAX as u64) as u32,
                    inspect: InspectPoint {
                        label: item.label.clone(),
                    },
                });
            }
        }

        rows.sort_unstable_by(|a, b| a.depth.cmp(&b.depth).then(b.retained.cmp(&a.retained)));

        let vertices: Vec<PointVertex> = rows.iter().map(|row| row.vertex).collect();
        let inspect_points: Vec<InspectPoint> =
            rows.iter().map(|row| row.inspect.clone()).collect();
        let block_ranges = contiguous_block_ranges(vertices.len(), self.num_blocks);
        let minimap_rows = build_minimap_rows(
            &rows,
            &block_ranges,
            max_depth_seen as u32,
            max_retained as u32,
        );
        let mut info_lines = vec![
            format!("Top retained subtrees: {}", items.len()),
            format!("Total retained shown: {}", format_bytes(total_size)),
            String::from("Right-click a region to inspect it."),
            String::from("Press v to cycle views."),
            String::new(),
        ];
        for (index, item) in items.iter().take(12).enumerate() {
            info_lines.push(format!(
                "{:>2}. {}  {}",
                index + 1,
                truncate_label(&item.class_name, 42),
                format_bytes(item.retained)
            ));
        }

        eprintln!(
            "[heap-treemap] Done: {} leaves, {} vertices, max_retained={}",
            items.len(),
            vertices.len(),
            max_retained
        );

        LoadResult {
            vertices,
            inspect_points,
            info_lines,
            block_ranges,
            minimap_rows,
        }
    }
}

struct TreemapItem {
    object_id: Id,
    retained: u64,
    depth: usize,
    class_name: Arc<str>,
    package: String,
    label: Arc<str>,
    rect: Rect,
    children: Vec<TreemapItem>,
}

struct TreemapRow {
    vertex: PointVertex,
    depth: u32,
    retained: u32,
    inspect: InspectPoint,
}

fn collect_items(
    dominator: &heapster::hprof::dominator::DominatorTree,
    children: &[Vec<usize>],
    object_class: &HashMap<Id, Id>,
    hprof: &HprofFile,
    node_idx: usize,
    root_retained: u64,
    depth: usize,
    config: &HeapTreemapSource,
    next_slot: &mut usize,
    out: &mut Vec<TreemapItem>,
) {
    let mut child_indices = children[node_idx].clone();
    child_indices.sort_unstable_by_key(|&child| {
        std::cmp::Reverse(dominator.retained_size_at(child).unwrap_or(0))
    });

    for child_idx in child_indices {
        if *next_slot >= config.max_items {
            return;
        }
        let retained = dominator.retained_size_at(child_idx).unwrap_or(0);
        if retained == 0 {
            continue;
        }
        let fraction = retained as f32 / root_retained as f32;
        if depth > 0 && fraction < config.min_fraction {
            continue;
        }
        let object_id = match dominator.object_id_at(child_idx) {
            Some(id) if id != 0 => id,
            _ => continue,
        };
        let class_id = object_class.get(&object_id).copied().unwrap_or(0);
        let class_name = hprof.class_name(class_id);
        let package = package_name(&class_name).to_string();
        let heap_share = (retained as f64 / root_retained.max(1) as f64) * 100.0;
        let child_count = children[child_idx].len();
        let label: Arc<str> = format!(
            "Class: {class_name}\nPackage: {package}\nObject: 0x{object_id:x}\nRetained: {}\nHeap share: {heap_share:.2}%\nDominator depth: {}\nOwned children: {child_count}",
            format_bytes(retained),
            depth + 1,
        )
        .into();

        *next_slot += 1;
        let mut item = TreemapItem {
            object_id,
            retained,
            depth: depth + 1,
            class_name: class_name.into(),
            package,
            label,
            rect: Rect::FULL,
            children: Vec::new(),
        };

        if depth + 1 < config.max_depth && !children[child_idx].is_empty() {
            collect_items(
                dominator,
                children,
                object_class,
                hprof,
                child_idx,
                root_retained,
                depth + 1,
                config,
                next_slot,
                &mut item.children,
            );
        }

        out.push(item);
    }
}

fn assign_item_rects(items: &mut [TreemapItem], bounds: Rect) {
    if items.is_empty() {
        return;
    }

    let total: u64 = items.iter().map(|item| item.retained.max(1)).sum();
    let horizontal = bounds.width() >= bounds.height();
    let mut cursor = if horizontal {
        bounds.x_min
    } else {
        bounds.z_min
    };

    for index in 0..items.len() {
        let weight = items[index].retained.max(1) as f32 / total as f32;
        let is_last = index == items.len() - 1;
        let rect = if horizontal {
            let end = if is_last {
                bounds.x_max
            } else {
                cursor + bounds.width() * weight
            };
            let rect = Rect {
                x_min: cursor,
                x_max: end.min(bounds.x_max),
                z_min: bounds.z_min,
                z_max: bounds.z_max,
            };
            cursor = rect.x_max;
            rect
        } else {
            let end = if is_last {
                bounds.z_max
            } else {
                cursor + bounds.height() * weight
            };
            let rect = Rect {
                x_min: bounds.x_min,
                x_max: bounds.x_max,
                z_min: cursor,
                z_max: end.min(bounds.z_max),
            };
            cursor = rect.z_max;
            rect
        };

        items[index].rect = rect;
        if !items[index].children.is_empty() {
            assign_item_rects(&mut items[index].children, rect.inset(0.004));
        }
    }
}

fn build_minimap_rows(
    rows: &[TreemapRow],
    block_ranges: &[(u32, u32)],
    max_depth: u32,
    max_retained: u32,
) -> Vec<MinimapRow> {
    let max_retained_f = max_retained.max(1) as f32;
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
            let avg_retained =
                slice.iter().map(|row| row.retained).sum::<u32>() as f32 / slice.len() as f32;
            MinimapRow {
                avg_byte: ((avg_depth / max_depth.max(1) as f32) * 170.0
                    + (avg_retained / max_retained_f) * 85.0)
                    .clamp(0.0, 255.0),
                entropy: (count as f32 / rows.len().max(1) as f32 * 30.0).clamp(0.0, 1.0),
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

fn stratified_point_in_rect(
    rect: Rect,
    total_points: usize,
    point_index: usize,
    seed: Id,
) -> [f32; 2] {
    let total_points = total_points.max(1);
    let aspect = (rect.width() / rect.height().max(1e-6)).max(0.2);
    let nx = ((total_points as f32 * aspect).sqrt().round() as usize).max(1);
    let ny = total_points.div_ceil(nx).max(1);
    let ix = point_index % nx;
    let iy = (point_index / nx) % ny;
    let point_id = seed ^ point_index as u64;
    let tx = (ix as f32 + hash01(point_id, 0)) / nx as f32;
    let tz = (iy as f32 + hash01(point_id, 1)) / ny as f32;
    [
        lerp(rect.x_min, rect.x_max, tx.clamp(0.0, 1.0)),
        lerp(rect.z_min, rect.z_max, tz.clamp(0.0, 1.0)),
    ]
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
    hsv_to_rgb(hue_from_id(hash) * 360.0, 0.68, 0.9)
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

#[derive(Clone, Copy)]
struct Rect {
    x_min: f32,
    x_max: f32,
    z_min: f32,
    z_max: f32,
}

impl Rect {
    const FULL: Self = Self {
        x_min: -1.0,
        x_max: 1.0,
        z_min: -1.0,
        z_max: 1.0,
    };

    fn width(self) -> f32 {
        (self.x_max - self.x_min).max(1e-6)
    }

    fn height(self) -> f32 {
        (self.z_max - self.z_min).max(1e-6)
    }

    fn inset(self, amount: f32) -> Self {
        let dx = amount.min(self.width() * 0.45);
        let dz = amount.min(self.height() * 0.45);
        Self {
            x_min: self.x_min + dx,
            x_max: self.x_max - dx,
            z_min: self.z_min + dz,
            z_max: self.z_max - dz,
        }
    }
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

fn truncate_label(label: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, ch) in label.chars().enumerate() {
        if i >= max_chars {
            out.push_str("...");
            break;
        }
        out.push(ch);
    }
    out
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
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
