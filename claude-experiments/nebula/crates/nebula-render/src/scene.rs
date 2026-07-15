//! GPU-resident graph state. This is where the "never round-trip to CPU" promise
//! is kept: positions and velocities live here and are shared, by binding, between
//! the compute (layout) passes and the render passes.

use nebula_core::{Graph, Pos};
use wgpu::util::DeviceExt;

/// All GPU buffers describing the graph and its live layout.
pub struct GpuGraph {
    pub num_nodes: u64,
    pub num_edges: u64,

    /// `array<vec2<f32>>` — node positions (world space). Shared compute+render.
    pub positions: wgpu::Buffer,
    /// `array<vec2<f32>>` — node velocities. Compute only.
    pub velocities: wgpu::Buffer,
    /// `array<u32>` — flat edge endpoints `[s0,d0,s1,d1,...]`. len = 2*E.
    pub edges: wgpu::Buffer,
    /// `array<u32>` — packed RGBA8 color per node.
    pub colors: wgpu::Buffer,
    /// `array<f32>` — per-node size multiplier.
    pub sizes: wgpu::Buffer,

    /// CSR offsets (`array<u32>`, len N+1) and targets (`array<u32>`, len 2E) for
    /// GPU spring attraction. Offsets are u32; for >4B directed entries this would
    /// need u64, which is out of scope until we go out-of-core.
    pub csr_offsets: wgpu::Buffer,
    pub csr_targets: wgpu::Buffer,

    /// Uniform spatial grid for O(N) repulsion: a per-cell atomic counter, plus
    /// node ids counting-sorted by cell. Cell `c` owns the run
    /// `node_order[cell_starts[c] .. cell_starts[c] + grid_counts[c]]`.
    ///
    /// The sort is what makes `forces` cache-friendly (it walks nodes in cell
    /// order), and it costs N entries instead of the `dim^2 * cap` a
    /// fixed-capacity per-cell list would need.
    pub grid_counts: wgpu::Buffer,
    pub node_order: wgpu::Buffer,
    pub cell_starts: wgpu::Buffer,
    /// Scratch: per-cell insert cursor for the scatter.
    pub cell_cursor: wgpu::Buffer,
    /// Scratch: per-block totals for the two-level prefix-sum scan.
    pub scan_sums: wgpu::Buffer,
    pub grid_dim: u32,
    /// Cap on how many nodes of a cell's run the near field sums. Bounds the
    /// worst case; does not affect storage.
    pub grid_cap: u32,
    pub world_size: f32,

    /// Center-of-mass pyramid for far-field repulsion: level 0 mirrors the fine
    /// grid, each level above halves the dimension (power-of-two aligned), down
    /// to 4x4. Levels are packed contiguously; `pyr_levels` holds (offset, dim).
    ///
    /// `array<vec2<u32>>`, one 8-byte entry per cell: x = center of mass in
    /// 16-bit-per-axis cell-relative fixed point, y = mass as an exact node
    /// count. See `layout.wgsl` for why it is packed this way.
    pub pyr: wgpu::Buffer,
    pub pyr_levels: Vec<(u32, u32)>,
}

impl GpuGraph {
    /// Upload a graph plus its seeded positions. `graph` must already have CSR
    /// built (call `graph.ensure_csr()` first) — we borrow it here.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        graph: &Graph,
        positions: &[Pos],
        k: f32,
    ) -> Self {
        Self::upload_with_grid_dim(device, queue, graph, positions, k, None)
    }

    /// As [`upload`], but forces the spatial grid's dimension instead of deriving
    /// it. Past ~2M nodes the derived dimension hits its 2048 clamp, so cells grow
    /// wider than `k` and near-field work per node climbs; this override exists to
    /// measure that effect. Raising it is close to a wash — finer cells cut
    /// near-field work but add a pyramid level and quadruple the pyramid — so
    /// this is a measurement knob, not a tuning one.
    pub fn upload_with_grid_dim(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        graph: &Graph,
        positions: &[Pos],
        k: f32,
        grid_dim_override: Option<u32>,
    ) -> Self {
        let num_nodes = graph.num_nodes();
        let num_edges = graph.num_edges();
        let n = num_nodes as usize;

        let positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("positions"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let velocities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("velocities"),
            size: (n * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero velocities.
        queue.write_buffer(&velocities, 0, &vec![0u8; (n * 8).max(4)]);

        let edges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edges"),
            contents: if num_edges > 0 {
                bytemuck::cast_slice(graph.edges_flat())
            } else {
                bytemuck::cast_slice(&[0u32, 0u32]) // avoid zero-sized buffer
            },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Default colors (steel blue) and unit sizes.
        let default_color = pack_rgba(120, 170, 255, 255);
        let colors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&vec![default_color; n.max(1)]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let sizes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sizes"),
            contents: bytemuck::cast_slice(&vec![1.0f32; n.max(1)]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // CSR (build if not present is caller's job; here we require it).
        let csr = graph
            .csr_ref()
            .expect("GpuGraph::upload requires graph.ensure_csr() first");
        let offsets_u32: Vec<u32> = csr.offsets.iter().map(|&o| o as u32).collect();
        let csr_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_offsets"),
            contents: bytemuck::cast_slice(&offsets_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let csr_targets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_targets"),
            contents: if csr.targets.is_empty() {
                bytemuck::cast_slice(&[0u32])
            } else {
                bytemuck::cast_slice(&csr.targets)
            },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // --- Spatial grid sizing ---------------------------------------------
        // Aim for a cell roughly the size of the optimal edge length k, with the
        // world sized to hold the natural spread (~k*sqrt(n)). Cap the dimension
        // so grid memory stays bounded for huge graphs (accuracy degrades softly
        // via the per-cell capacity clamp). Rounded to a power of two so the
        // far-field COM pyramid's parent/child cells align exactly (2:1).
        let world_size = (k * (num_nodes.max(1) as f32).sqrt() * 1.6).max(k * 4.0);
        let ideal_dim = (world_size / k).ceil() as u32;
        let grid_dim = match grid_dim_override {
            // 4096 is the ceiling the two-level scan supports (see the assert below).
            Some(d) => d.next_power_of_two().clamp(4, 4096),
            None => ideal_dim.next_power_of_two().clamp(4, 2048),
        };
        let grid_cap = 32u32;
        let num_cells = (grid_dim as u64) * (grid_dim as u64);

        let grid_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_counts"),
            size: num_cells * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let storage_buf = |label: &'static str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                // max(1) — wgpu rejects zero-sized buffers, and an empty graph
                // still has to bind something.
                size: size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let node_order = storage_buf("node_order", num_nodes * 4);
        let cell_starts = storage_buf("cell_starts", num_cells * 4);
        let cell_cursor = storage_buf("cell_cursor", num_cells * 4);
        // Two levels of block totals: one per 256 cells, then one per 256 of those.
        let nblocks0 = num_cells.div_ceil(256);
        let nblocks1 = nblocks0.div_ceil(256);
        // `scan_sums1` finishes the scan in a single 256-thread workgroup, which
        // only works while the top level fits in one. Two levels cover 256^3 =
        // 16.7M cells (grid_dim 4096); beyond that the scan needs a third level,
        // and getting it wrong would silently corrupt every cell offset rather
        // than fail — so refuse loudly instead.
        assert!(
            nblocks1 <= 256,
            "grid_dim {grid_dim} needs {nblocks1} top-level scan blocks, but the \
             two-level scan tops out at 256 (grid_dim 4096). Add a third scan level."
        );
        let scan_sums = storage_buf("scan_sums", (nblocks0 + nblocks1) * 4);

        // Far-field COM pyramid: level 0 mirrors the fine grid, then halve down
        // to 4x4. The forces pass walks ~27 cells per level (fast-multipole
        // interaction lists) instead of scanning one big coarse grid.
        let mut pyr_levels: Vec<(u32, u32)> = Vec::new();
        let mut off = 0u32;
        let mut d = grid_dim;
        loop {
            pyr_levels.push((off, d));
            off += d * d;
            if d <= 4 {
                break;
            }
            d /= 2;
        }
        let pyr_cells = off as u64;
        let pyr = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pyr"),
            size: pyr_cells * 8, // vec2<u32>: packed COM + mass
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::info!(
            "GpuGraph: {} nodes, {} edges, grid {}x{} cap {}, pyramid {} levels / {} cells (world {:.0})",
            num_nodes, num_edges, grid_dim, grid_dim, grid_cap, pyr_levels.len(), pyr_cells, world_size
        );

        GpuGraph {
            num_nodes,
            num_edges,
            positions: positions_buf,
            velocities,
            edges,
            colors,
            sizes,
            csr_offsets,
            csr_targets,
            grid_counts,
            node_order,
            cell_starts,
            cell_cursor,
            scan_sums,
            grid_dim,
            grid_cap,
            world_size,
            pyr,
            pyr_levels,
        }
    }

    /// Overwrite node colors (packed RGBA8) from an algorithm result.
    pub fn set_colors(&self, queue: &wgpu::Queue, colors: &[u32]) {
        queue.write_buffer(&self.colors, 0, bytemuck::cast_slice(colors));
    }

    /// Overwrite per-node size multipliers.
    pub fn set_sizes(&self, queue: &wgpu::Queue, sizes: &[f32]) {
        queue.write_buffer(&self.sizes, 0, bytemuck::cast_slice(sizes));
    }

    /// Re-seed positions (e.g. when switching layouts) and zero velocities.
    pub fn set_positions(&self, queue: &wgpu::Queue, positions: &[Pos]) {
        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(positions));
        queue.write_buffer(
            &self.velocities,
            0,
            &vec![0u8; (self.num_nodes as usize * 8).max(4)],
        );
    }

    /// Replace the CSR buffers (spring attraction edges) from a new CSR. The
    /// buffers are recreated (sizes change with the edge set); callers must
    /// rebind anything referencing them (e.g. `LayoutGpu::rebind`).
    pub fn set_csr(&mut self, device: &wgpu::Device, csr: &nebula_core::Csr) {
        let offsets_u32: Vec<u32> = csr.offsets.iter().map(|&o| o as u32).collect();
        self.csr_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_offsets"),
            contents: bytemuck::cast_slice(&offsets_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.csr_targets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csr_targets"),
            contents: if csr.targets.is_empty() {
                bytemuck::cast_slice(&[0u32])
            } else {
                bytemuck::cast_slice(&csr.targets)
            },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.num_edges = (csr.targets.len() / 2) as u64;
    }
}

#[inline]
pub fn pack_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}
