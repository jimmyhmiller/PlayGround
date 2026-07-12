//! Screen-space density aggregation (level-of-detail). One O(N) compute pass bins
//! every node into a screen tile; the render pass draws one quad per tile colored
//! by count. Rendering is O(tiles), independent of N — this is how you *view* a
//! graph far larger than you could ever rasterize node-by-node.

use crate::camera::CameraUniform;
use crate::layout_gpu::dispatch_dims;
use crate::scene::GpuGraph;
use bytemuck::{Pod, Zeroable};

const MAX_TILES: u64 = 1 << 21; // ~2.1M tiles — covers 5K displays at 3px tiles

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DParams {
    tiles_x: u32,
    tiles_y: u32,
    tile_px: f32,
    num_nodes: u32,
    vw: f32,
    vh: f32,
    gamma: f32,
    _p: f32,
}

pub struct Density {
    camera_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    compute_view_bg: wgpu::BindGroup,
    compute_data_bg: wgpu::BindGroup,
    render_view_bg: wgpu::BindGroup,
    render_data_bg: wgpu::BindGroup,
    clear_pipeline: wgpu::ComputePipeline,
    accum_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    num_nodes: u32,
    tiles_x: u32,
    tiles_y: u32,
    pub tile_px: f32,
    pub gamma: f32,
}

impl Density {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, graph: &GpuGraph) -> Self {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density.wgsl").into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density_render.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/density_render.wgsl").into()),
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density_camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density_params"),
            size: std::mem::size_of::<DParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density_counts"),
            size: MAX_TILES * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let max_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density_max"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniform = |vis| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |binding: u32, vis: wgpu::ShaderStages, read_only: bool| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };

        // Compute layouts.
        let cv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_cv_bgl"),
            entries: &[
                uniform(wgpu::ShaderStages::COMPUTE),
                wgpu::BindGroupLayoutEntry { binding: 1, ..uniform(wgpu::ShaderStages::COMPUTE) },
            ],
        });
        let cd_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_cd_bgl"),
            entries: &[
                storage(0, wgpu::ShaderStages::COMPUTE, true),  // positions
                storage(1, wgpu::ShaderStages::COMPUTE, false), // counts (atomic)
                storage(2, wgpu::ShaderStages::COMPUTE, false), // max (atomic)
            ],
        });
        // Render layouts.
        let rv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_rv_bgl"),
            entries: &[uniform(wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let rd_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_rd_bgl"),
            entries: &[
                storage(0, wgpu::ShaderStages::VERTEX, true), // counts (read-only)
                storage(1, wgpu::ShaderStages::VERTEX, true), // max (read-only)
            ],
        });

        let compute_view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_cv_bg"),
            layout: &cv_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });
        let compute_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_cd_bg"),
            layout: &cd_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: graph.positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: max_buf.as_entire_binding() },
            ],
        });
        let render_view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_rv_bg"),
            layout: &rv_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() }],
        });
        let render_data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_rd_bg"),
            layout: &rd_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: max_buf.as_entire_binding() },
            ],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density_compute_pl"),
            bind_group_layouts: &[&cv_bgl, &cd_bgl],
            push_constant_ranges: &[],
        });
        let make_compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&compute_layout),
                module: &compute_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let clear_pipeline = make_compute("clear_density");
        let accum_pipeline = make_compute("accumulate_density");

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density_render_pl"),
            bind_group_layouts: &[&rv_bgl, &rd_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("density_render"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_density"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_density"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Density {
            camera_buf,
            params_buf,
            compute_view_bg,
            compute_data_bg,
            render_view_bg,
            render_data_bg,
            clear_pipeline,
            accum_pipeline,
            render_pipeline,
            num_nodes: graph.num_nodes as u32,
            tiles_x: 1,
            tiles_y: 1,
            tile_px: 4.0,
            gamma: 0.6,
        }
    }

    /// Update camera + tile grid for the current viewport. Call each frame.
    pub fn update(&mut self, queue: &wgpu::Queue, cam: &CameraUniform, vw: f32, vh: f32) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
        let tp = self.tile_px.max(1.0);
        let mut tx = (vw / tp).ceil() as u32;
        let mut ty = (vh / tp).ceil() as u32;
        // Respect the tile budget.
        while (tx as u64) * (ty as u64) > MAX_TILES {
            tx = (tx + 1) / 2;
            ty = (ty + 1) / 2;
        }
        self.tiles_x = tx.max(1);
        self.tiles_y = ty.max(1);
        let params = DParams {
            tiles_x: self.tiles_x,
            tiles_y: self.tiles_y,
            tile_px: tp,
            num_nodes: self.num_nodes,
            vw,
            vh,
            gamma: self.gamma,
            _p: 0.0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Record the clear + accumulate compute passes (binning nodes into tiles).
    pub fn record_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let tiles = self.tiles_x as u64 * self.tiles_y as u64;
        let tile_groups = dispatch_dims(tiles.div_ceil(256));
        let node_groups = dispatch_dims((self.num_nodes as u64).div_ceil(256));

        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density_clear"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.clear_pipeline);
            cp.set_bind_group(0, &self.compute_view_bg, &[]);
            cp.set_bind_group(1, &self.compute_data_bg, &[]);
            cp.dispatch_workgroups(tile_groups.0, tile_groups.1, tile_groups.2);
        }
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density_accumulate"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.accum_pipeline);
            cp.set_bind_group(0, &self.compute_view_bg, &[]);
            cp.set_bind_group(1, &self.compute_data_bg, &[]);
            cp.dispatch_workgroups(node_groups.0, node_groups.1, node_groups.2);
        }
    }

    /// Draw the heatmap into an active render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_view_bg, &[]);
        pass.set_bind_group(1, &self.render_data_bg, &[]);
        let tiles = self.tiles_x * self.tiles_y;
        pass.draw(0..6, 0..tiles);
    }
}
