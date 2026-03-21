use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// A point in the cloud: position in [-1,1]^3 + RGBA color.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Minimap vertex for the 2D sidebar.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MinimapVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// Result of loading a file: pre-built block vertices for instant scrubbing,
/// plus minimap data for the sidebar.
pub struct LoadResult {
    /// All point vertices for all blocks, concatenated.
    pub vertices: Vec<PointVertex>,
    /// Optional inspection metadata aligned with `vertices`.
    pub inspect_points: Vec<InspectPoint>,
    /// (start_instance, instance_count) per block.
    pub block_ranges: Vec<(u32, u32)>,
    /// Minimap row data: one entry per row with (avg_byte, entropy).
    pub minimap_rows: Vec<MinimapRow>,
}

#[derive(Clone)]
pub struct InspectPoint {
    pub label: Arc<str>,
}

pub struct MinimapRow {
    pub avg_byte: f32,
    pub entropy: f32,
}

/// Trait for anything that can produce a point cloud from a file.
/// Implementations do the heavy lifting of turning raw bytes into vertices.
pub trait PointCloudSource: Send + Sync + 'static {
    /// Process a file into block-based vertex data.
    /// Receives both the file path and the mmap'd bytes.
    /// Called in a background thread.
    fn load(&self, path: &std::path::Path, data: &[u8]) -> LoadResult;

    /// Number of blocks the file is divided into.
    fn num_blocks(&self) -> usize;
}

// --- Trigram implementation ---

const TRIGRAM_COUNT: usize = 256 * 256 * 256;

#[inline(always)]
fn trigram_index(b0: u8, b1: u8, b2: u8) -> usize {
    (b0 as usize) << 16 | (b1 as usize) << 8 | (b2 as usize)
}

pub struct TrigramSource {
    pub num_blocks: usize,
    pub max_points_per_block: usize,
    pub minimap_rows: usize,
}

impl Default for TrigramSource {
    fn default() -> Self {
        Self {
            num_blocks: 256,
            max_points_per_block: 4_000,
            minimap_rows: 512,
        }
    }
}

impl PointCloudSource for TrigramSource {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn load(&self, _path: &std::path::Path, data: &[u8]) -> LoadResult {
        let minimap_rows = self.build_minimap_rows(data);
        let (vertices, block_ranges) = self.build_block_vertices(data);
        LoadResult {
            vertices,
            inspect_points: vec![],
            block_ranges,
            minimap_rows,
        }
    }
}

impl TrigramSource {
    fn build_minimap_rows(&self, data: &[u8]) -> Vec<MinimapRow> {
        if data.is_empty() {
            return vec![];
        }
        let rows = self.minimap_rows;
        let chunk_size = data.len() / rows;
        if chunk_size == 0 {
            return vec![];
        }

        (0..rows)
            .map(|row| {
                let start = row * chunk_size;
                let end = if row == rows - 1 {
                    data.len()
                } else {
                    start + chunk_size
                };
                let slice = &data[start..end];

                let mut sum: u64 = 0;
                let mut counts = [0u32; 256];
                for &b in slice {
                    sum += b as u64;
                    counts[b as usize] += 1;
                }
                let avg_byte = sum as f32 / slice.len() as f32;

                let len_f = slice.len() as f32;
                let mut entropy: f32 = 0.0;
                for &c in &counts {
                    if c > 0 {
                        let p = c as f32 / len_f;
                        entropy -= p * p.log2();
                    }
                }

                MinimapRow {
                    avg_byte,
                    entropy: (entropy / 8.0).clamp(0.0, 1.0),
                }
            })
            .collect()
    }

    fn build_block_vertices(&self, data: &[u8]) -> (Vec<PointVertex>, Vec<(u32, u32)>) {
        let num_blocks = self.num_blocks;
        let max_per_block = self.max_points_per_block;

        if data.len() < 3 {
            return (vec![], vec![(0, 0); num_blocks]);
        }

        let block_size = data.len() / num_blocks;
        if block_size < 3 {
            return (vec![], vec![(0, 0); num_blocks]);
        }

        let block_verts: Vec<Vec<PointVertex>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_blocks);
            for b in 0..num_blocks {
                let start = b * block_size;
                let end = if b == num_blocks - 1 {
                    data.len()
                } else {
                    ((b + 1) * block_size + 2).min(data.len())
                };
                let chunk = &data[start..end];
                let block_pos = (b as f32 + 0.5) / num_blocks as f32;

                handles.push(s.spawn(move || {
                    let mut count = vec![0u32; TRIGRAM_COUNT];
                    let mut max_freq: u32 = 0;

                    for i in 0..chunk.len() - 2 {
                        let idx = trigram_index(chunk[i], chunk[i + 1], chunk[i + 2]);
                        unsafe {
                            let c = count.get_unchecked_mut(idx);
                            *c += 1;
                            if *c > max_freq {
                                max_freq = *c;
                            }
                        }
                    }

                    if max_freq == 0 {
                        return vec![];
                    }

                    let mut entries: Vec<(usize, u32)> = Vec::new();
                    for idx in 0..TRIGRAM_COUNT {
                        let c = count[idx];
                        if c > 0 {
                            entries.push((idx, c));
                        }
                    }

                    if entries.len() > max_per_block {
                        entries.select_nth_unstable_by(max_per_block, |a, b| b.1.cmp(&a.1));
                        entries.truncate(max_per_block);
                        max_freq = entries.iter().map(|e| e.1).max().unwrap_or(1);
                    }

                    let ln_max = (max_freq as f32).ln().max(1.0);
                    let (cr, cg, cb) = file_position_color(block_pos);

                    entries
                        .iter()
                        .map(|&(idx, c)| {
                            let b0 = (idx >> 16) as u16;
                            let b1 = ((idx >> 8) & 0xFF) as u16;
                            let b2 = (idx & 0xFF) as u16;
                            let brightness = ((c as f32).ln() / ln_max).clamp(0.1, 1.0);
                            PointVertex {
                                position: [
                                    (b0 as f32 / 127.5) - 1.0,
                                    (b1 as f32 / 127.5) - 1.0,
                                    (b2 as f32 / 127.5) - 1.0,
                                ],
                                color: [cr * brightness, cg * brightness, cb * brightness, 0.15],
                            }
                        })
                        .collect()
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let total: usize = block_verts.iter().map(|v| v.len()).sum();
        let mut all_vertices = Vec::with_capacity(total);
        let mut block_ranges = Vec::with_capacity(num_blocks);

        for verts in block_verts {
            let start = all_vertices.len() as u32;
            let count = verts.len() as u32;
            all_vertices.extend_from_slice(&verts);
            block_ranges.push((start, count));
        }

        (all_vertices, block_ranges)
    }
}

fn file_position_color(pos: f32) -> (f32, f32, f32) {
    if pos < 0.5 {
        let t = pos * 2.0;
        (1.0, 1.0, t)
    } else {
        let t = (pos - 0.5) * 2.0;
        (1.0 - t, 1.0 - t, 1.0)
    }
}
