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

    /// Uniform spatial grid for O(N) repulsion: per-cell atomic counter + a
    /// fixed-capacity item list. Sized from node count at build time.
    pub grid_counts: wgpu::Buffer,
    pub grid_items: wgpu::Buffer,
    pub grid_dim: u32,
    pub grid_cap: u32,
    pub world_size: f32,

    /// Coarse center-of-mass grid for far-field repulsion.
    pub coarse_com: wgpu::Buffer,
    pub coarse_mass: wgpu::Buffer,
    pub coarse_dim: u32,
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
        // via the per-cell capacity clamp).
        let world_size = (k * (num_nodes.max(1) as f32).sqrt() * 1.6).max(k * 4.0);
        let ideal_dim = (world_size / k).ceil() as u32;
        let grid_dim = ideal_dim.clamp(4, 2048);
        let grid_cap = 32u32;
        let num_cells = (grid_dim as u64) * (grid_dim as u64);

        let grid_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_counts"),
            size: num_cells * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid_items = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_items"),
            size: num_cells * grid_cap as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Coarse grid for far-field: kept small so the per-node far-field loop
        // (coarse_dim^2 iterations) stays cheap, but large enough that a coarse
        // cell is only a few fine cells wide.
        let coarse_dim = grid_dim.min(64).max(1);
        let ncoarse = (coarse_dim as u64) * (coarse_dim as u64);
        let coarse_com = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coarse_com"),
            size: ncoarse * 8, // vec2<f32>
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let coarse_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coarse_mass"),
            size: ncoarse * 4, // f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::info!(
            "GpuGraph: {} nodes, {} edges, grid {}x{} cap {}, coarse {}x{} (world {:.0})",
            num_nodes, num_edges, grid_dim, grid_dim, grid_cap, coarse_dim, coarse_dim, world_size
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
            grid_items,
            grid_dim,
            grid_cap,
            world_size,
            coarse_com,
            coarse_mass,
            coarse_dim,
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
}

#[inline]
pub fn pack_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}
