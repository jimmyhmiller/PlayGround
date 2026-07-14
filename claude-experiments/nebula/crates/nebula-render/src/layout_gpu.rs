//! GPU force-directed layout driver: owns the compute pipelines and the params
//! uniform, and dispatches the four passes that make up one simulation step.
//!
//! Dispatch tiling: the workgroup count for a billion nodes is ~4M, far past the
//! 65535 per-dimension limit, so we spread it across a 2D grid and the shader
//! reconstructs a linear index from `workgroup_id` + `num_workgroups`.

use crate::scene::GpuGraph;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Matches `Params` in `layout.wgsl` (64 bytes + the 12-entry level table).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LayoutParams {
    pub num_nodes: u32,
    pub grid_dim: u32,
    pub grid_cap: u32,
    pub num_levels: u32,
    pub world_size: f32,
    pub k: f32,
    pub repulsion: f32,
    pub attraction: f32,
    pub gravity: f32,
    pub damping: f32,
    pub dt: f32,
    pub max_speed: f32,
    pub alpha: f32,
    pub _p1: f32,
    pub _p2: f32,
    pub _p3: f32,
    /// COM pyramid level table: x = cell offset, y = dim (vec4 for std140).
    pub levels: [[u32; 4]; 12],
}

/// Tunable physics knobs surfaced to the UI.
#[derive(Clone, Copy)]
pub struct LayoutSettings {
    pub k: f32,
    pub repulsion: f32,
    pub attraction: f32,
    pub gravity: f32,
    pub damping: f32,
    pub dt: f32,
    pub max_speed: f32,
    /// Simulation substeps per rendered frame.
    pub substeps: u32,
    pub running: bool,
    /// Global cooling. `alpha` scales all forces; it decays by `alpha_decay` per
    /// simulation step and the sim auto-pauses once it reaches `alpha_min`.
    pub alpha: f32,
    pub alpha_decay: f32,
    pub alpha_min: f32,
    /// Alpha to "reheat" to when the user resumes a settled layout.
    pub alpha_reheat: f32,
}

impl Default for LayoutSettings {
    fn default() -> Self {
        LayoutSettings {
            k: 30.0,
            repulsion: 1.0,
            attraction: 1.0,
            gravity: 0.005,
            damping: 0.9,
            dt: 0.3,
            max_speed: 50.0,
            substeps: 1,
            running: true,
            alpha: 1.0,
            alpha_decay: 0.015,
            alpha_min: 0.004,
            alpha_reheat: 0.4,
        }
    }
}

pub struct LayoutGpu {
    params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
    data_bg: wgpu::BindGroup,
    data_bgl: wgpu::BindGroupLayout,
    clear_pipeline: wgpu::ComputePipeline,
    build_pipeline: wgpu::ComputePipeline,
    pyr_l0_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    /// One dynamic-offset slot per reduce pass, telling it which level to write.
    reduce_bg: wgpu::BindGroup,
    forces_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    num_nodes: u32,
    grid_dim: u32,
    grid_cap: u32,
    levels: Vec<(u32, u32)>,
    world_size: f32,
}

/// Pack the (offset, dim) level list into the fixed uniform table.
fn level_table(levels: &[(u32, u32)]) -> [[u32; 4]; 12] {
    let mut t = [[0u32; 4]; 12];
    for (i, &(off, dim)) in levels.iter().take(12).enumerate() {
        t[i] = [off, dim, 0, 0];
    }
    t
}

impl LayoutGpu {
    pub fn new(device: &wgpu::Device, graph: &GpuGraph, settings: &LayoutSettings) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layout.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layout.wgsl").into()),
        });

        let params = LayoutParams {
            num_nodes: graph.num_nodes as u32,
            grid_dim: graph.grid_dim,
            grid_cap: graph.grid_cap,
            num_levels: graph.pyr_levels.len() as u32,
            world_size: graph.world_size,
            k: settings.k,
            repulsion: settings.repulsion,
            attraction: settings.attraction,
            gravity: settings.gravity,
            damping: settings.damping,
            dt: settings.dt,
            max_speed: settings.max_speed,
            alpha: settings.alpha,
            _p1: 0.0,
            _p2: 0.0,
            _p3: 0.0,
            levels: level_table(&graph.pyr_levels),
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layout_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // group(0): params uniform
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_params_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // group(1): storage buffers (positions, velocities, csr_offsets,
        // csr_targets, grid_counts, grid_items).
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_data_bgl"),
            entries: &[
                storage(0, false), // positions (rw)
                storage(1, false), // velocities (rw)
                storage(2, true),  // csr_offsets (ro)
                storage(3, true),  // csr_targets (ro)
                storage(4, false), // grid_counts (rw, atomic)
                storage(5, false), // grid_items (rw)
                storage(6, false), // pyr_com (rw)
                storage(7, false), // pyr_mass (rw)
            ],
        });

        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layout_params_bg"),
            layout: &params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        let data_bg = make_data_bg(device, &data_bgl, graph);

        // group(2) for reduce passes: which pyramid level to write, selected per
        // pass with a dynamic offset into one small uniform buffer.
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_reduce_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            }],
        });
        const REDUCE_STRIDE: u64 = 256; // min uniform buffer offset alignment
        let mut reduce_contents = vec![0u8; (REDUCE_STRIDE as usize) * 12];
        for lvl in 1..12u32 {
            let at = lvl as usize * REDUCE_STRIDE as usize;
            reduce_contents[at..at + 4].copy_from_slice(&lvl.to_le_bytes());
        }
        let reduce_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layout_reduce_levels"),
            contents: &reduce_contents,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layout_reduce_bg"),
            layout: &reduce_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &reduce_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(4),
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout_pl"),
            bind_group_layouts: &[&params_bgl, &data_bgl],
            push_constant_ranges: &[],
        });
        let reduce_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("layout_reduce_pl"),
                bind_group_layouts: &[&params_bgl, &data_bgl, &reduce_bgl],
                push_constant_ranges: &[],
            });

        let make = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        LayoutGpu {
            params_buf,
            params_bg,
            data_bg,
            data_bgl,
            clear_pipeline: make("clear_grid", &pipeline_layout),
            build_pipeline: make("build_grid", &pipeline_layout),
            pyr_l0_pipeline: make("build_pyr_l0", &pipeline_layout),
            reduce_pipeline: make("reduce_pyr", &reduce_pipeline_layout),
            reduce_bg,
            forces_pipeline: make("forces", &pipeline_layout),
            integrate_pipeline: make("integrate", &pipeline_layout),
            num_nodes: graph.num_nodes as u32,
            grid_dim: graph.grid_dim,
            grid_cap: graph.grid_cap,
            levels: graph.pyr_levels.clone(),
            world_size: graph.world_size,
        }
    }

    /// Rebind the data bind group to the graph's current buffers. Call after
    /// `GpuGraph::set_csr` swaps the spring edges (a different edge set).
    pub fn rebind(&mut self, device: &wgpu::Device, graph: &GpuGraph) {
        self.data_bg = make_data_bg(device, &self.data_bgl, graph);
    }

    /// Push updated physics settings to the GPU (call when the UI changes them).
    pub fn update_settings(&self, queue: &wgpu::Queue, settings: &LayoutSettings) {
        let params = LayoutParams {
            num_nodes: self.num_nodes,
            grid_dim: self.grid_dim,
            grid_cap: self.grid_cap,
            num_levels: self.levels.len() as u32,
            world_size: self.world_size,
            k: settings.k,
            repulsion: settings.repulsion,
            attraction: settings.attraction,
            gravity: settings.gravity,
            damping: settings.damping,
            dt: settings.dt,
            max_speed: settings.max_speed,
            alpha: settings.alpha,
            _p1: 0.0,
            _p2: 0.0,
            _p3: 0.0,
            levels: level_table(&self.levels),
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Encode one simulation step.
    pub fn step(&self, encoder: &mut wgpu::CommandEncoder) {
        let cells = self.grid_dim as u64 * self.grid_dim as u64;
        let cell_groups = dispatch_dims(cells.div_ceil(256));
        let node_groups = dispatch_dims((self.num_nodes as u64).div_ceil(256));

        // Each stage in its own pass so wgpu inserts the needed memory barriers.
        self.pass(encoder, &self.clear_pipeline, cell_groups, "clear_grid");
        self.pass(encoder, &self.build_pipeline, node_groups, "build_grid");
        self.pass(encoder, &self.pyr_l0_pipeline, cell_groups, "build_pyr_l0");
        // Reduce the pyramid one level at a time (each pass reads the previous).
        for lvl in 1..self.levels.len() {
            let dim = self.levels[lvl].1 as u64;
            let groups = dispatch_dims((dim * dim).div_ceil(256));
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce_pyr"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.reduce_pipeline);
            cpass.set_bind_group(0, &self.params_bg, &[]);
            cpass.set_bind_group(1, &self.data_bg, &[]);
            cpass.set_bind_group(2, &self.reduce_bg, &[lvl as u32 * 256]);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.pass(encoder, &self.forces_pipeline, node_groups, "forces");
        self.pass(encoder, &self.integrate_pipeline, node_groups, "integrate");
    }

    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: (u32, u32, u32),
        label: &str,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.params_bg, &[]);
        cpass.set_bind_group(1, &self.data_bg, &[]);
        cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }
}

/// Spread `total` workgroups across up to 2 dimensions to respect the 65535
/// per-dimension limit. The shader linearizes via workgroup_id + num_workgroups.
pub fn dispatch_dims(total: u64) -> (u32, u32, u32) {
    const MAX: u64 = 65535;
    if total == 0 {
        return (1, 1, 1);
    }
    if total <= MAX {
        return (total as u32, 1, 1);
    }
    let x = MAX;
    let y = total.div_ceil(x);
    // y could in theory exceed MAX only past ~4.29e9 workgroups (~1.1e12 nodes),
    // which is beyond any addressable buffer, so 2D suffices here.
    (x as u32, y as u32, 1)
}

/// Build the layout compute data bind group from a graph's buffers.
fn make_data_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    graph: &GpuGraph,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("layout_data_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: graph.positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: graph.velocities.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: graph.csr_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: graph.csr_targets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: graph.grid_counts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: graph.grid_items.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: graph.pyr_com.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: graph.pyr_mass.as_entire_binding() },
        ],
    })
}
