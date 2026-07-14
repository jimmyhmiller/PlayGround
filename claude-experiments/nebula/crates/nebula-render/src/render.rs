//! Node and edge render pipelines. Both read the same GPU position buffer the
//! compute layout writes, so there is zero copy between simulating and drawing.
//!
//! Edges use additive blending so overlapping lines accumulate into bright
//! bundles — the natural way to read structure in a million-edge hairball.
//! Nodes use standard alpha-over with an SDF-antialiased circle.

use crate::camera::CameraUniform;
use crate::scene::GpuGraph;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RenderParams {
    pub base_radius_px: f32,
    pub min_radius_px: f32,
    pub max_radius_px: f32,
    pub size_gamma: f32,
    pub edge_alpha: f32,
    pub node_alpha: f32,
    pub _p0: f32,
    pub _p1: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        RenderParams {
            base_radius_px: 3.0,
            min_radius_px: 1.0,
            max_radius_px: 64.0,
            size_gamma: 1.0,
            edge_alpha: 0.12,
            node_alpha: 1.0,
            _p0: 0.0,
            _p1: 0.0,
        }
    }
}

/// Per-edge-type style uniform. `mode` 0 = color edges by node color, 1 = tint.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EdgeStyle {
    color: [f32; 4],
    mode: u32,
    _pad: [u32; 3],
}

/// Input describing one edge type to upload (borrowed from the app).
pub struct EdgeTypeInput<'a> {
    pub name: &'a str,
    /// Packed RGBA8 tint, or None to color edges by endpoint node color.
    pub color: Option<u32>,
    pub visible: bool,
    pub edges: &'a [[u32; 2]],
}

/// GPU state for one edge type: its own edge buffer + bind group, plus the
/// compacted visible-edge index buffer and indirect draw args the cull
/// compute pass maintains.
struct EdgeTypeGpu {
    bind_group: wgpu::BindGroup,
    /// Bind group for the edge_cull compute passes.
    cull_bg: wgpu::BindGroup,
    /// Group-0 bind group for the edge_raster compute pass.
    raster_bg: wgpu::BindGroup,
    /// Visible edges as (a, b) node-id pairs in original edge order — used as
    /// the index buffer for the edge draw.
    compact: wgpu::Buffer,
    /// DrawIndexedIndirect args, written entirely on the GPU by cull_scan.
    indirect: wgpu::Buffer,
    num_edges: u32,
    visible: bool,
}

fn unpack_norm(c: u32) -> [f32; 4] {
    [
        (c & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 24) & 0xff) as f32 / 255.0,
    ]
}

pub struct Renderer {
    camera_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    graph_bg: wgpu::BindGroup,
    graph_bgl: wgpu::BindGroupLayout,
    /// Per-edge-type GPU state (buffers + bind groups). Set via `set_edge_types`.
    edge_types: Vec<EdgeTypeGpu>,
    node_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    pick_pipeline: wgpu::RenderPipeline,
    cull_bgl: wgpu::BindGroupLayout,
    cull_count_pipeline: wgpu::ComputePipeline,
    cull_scan_pipeline: wgpu::ComputePipeline,
    cull_emit_pipeline: wgpu::ComputePipeline,
    raster_bgl: wgpu::BindGroupLayout,
    raster_accum_bgl: wgpu::BindGroupLayout,
    tile_count_pipeline: wgpu::ComputePipeline,
    tile_scan_pipeline: wgpu::ComputePipeline,
    tile_emit_pipeline: wgpu::ComputePipeline,
    tile_raster_pipeline: wgpu::ComputePipeline,
    resolve_bgl: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    /// Accumulation target for the compute raster path (recreated on resize).
    accum: Option<AccumGpu>,
    /// True when the prepared edge data (compacted index sets on the hardware
    /// path, the accumulation buffer on the compute path) is stale.
    cull_dirty: bool,
    last_cam: Option<CameraUniform>,
    /// edge_alpha is baked into the compute-raster accumulation, so a change
    /// must re-raster (the hardware path reads it live and doesn't care).
    last_edge_alpha: Option<f32>,
    num_nodes: u32,
    pub draw_edges: bool,
    pub draw_nodes: bool,
    /// Runtime toggle: rasterize edges in a compute shader instead of the
    /// hardware LineList pipeline. Switch via `set_edge_raster`.
    edge_raster: bool,
}

/// Screen-tile edge length in pixels (must match TILE in edge_raster.wgsl).
const TILE: u32 = 32;
/// Capacity of the (tile, edge) pair scratch, in pairs. Edges are processed
/// in batches sized so a batch's worst-case pair count fits; batching cannot
/// change the image because integer accumulation commutes.
const PAIRS_CAP: u64 = 64 << 20;
/// Max batches per frame (batch params buffer stride slots).
const MAX_BATCHES: usize = 1024;
/// Dynamic-offset stride for the per-batch uniform.
const BATCH_STRIDE: u64 = 256;

/// GPU state for the tiled compute raster path (recreated on resize or when
/// the edge sets change).
struct AccumGpu {
    buf: wgpu::Buffer,
    batch_buf: wgpu::Buffer,
    w: u32,
    h: u32,
    tiles_x: u32,
    tiles_y: u32,
    /// Pair scratch capacity actually allocated (≤ PAIRS_CAP).
    pairs_cap: u64,
    /// Group-1 bind group for the tile passes (dims, accum, tile scratch,
    /// pairs, batch uniform with dynamic offset).
    raster_bg: wgpu::BindGroup,
    /// Group-0 bind group for the fullscreen resolve (dims + read-only accum).
    resolve_bg: wgpu::BindGroup,
}

/// Edges per cull_count/cull_emit workgroup (must match CHUNK in edge_cull.wgsl).
const CULL_CHUNK: u32 = 4096;

impl Renderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, graph: &GpuGraph) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_params"),
            contents: bytemuck::bytes_of(&RenderParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view_bgl"),
            entries: &[uniform_entry(0), uniform_entry(1)],
        });

        let ro_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let edge_style_entry = wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let graph_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph_bgl"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2), ro_storage(3), edge_style_entry],
        });
        // Default edge style (node-colored) for the node/base bind group.
        let default_edge_style = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("default_edge_style"),
            contents: bytemuck::bytes_of(&EdgeStyle { color: [0.0; 4], mode: 0, _pad: [0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view_bg"),
            layout: &view_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });
        let graph_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("graph_bg"),
            layout: &graph_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: graph.positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: graph.colors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: graph.sizes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: graph.edges.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: default_edge_style.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pl"),
            bind_group_layouts: &[&view_bgl, &graph_bgl],
            push_constant_ranges: &[],
        });

        let alpha_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        // Additive: overlapping edges brighten. Premultiplied by src alpha.
        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("node_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_node"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_node"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(alpha_blend),
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

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_edge"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_edge"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pick_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_pick"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_pick"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
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

        // Edge cull/compaction compute pipelines (see edge_cull.wgsl).
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/edge_cull.wgsl").into()),
        });
        let rw_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edge_cull_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                cull_ro(1),
                cull_ro(2),
                rw_storage(3),
                rw_storage(4),
                rw_storage(5),
            ],
        });
        let cull_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edge_cull_pl"),
            bind_group_layouts: &[&cull_bgl],
            push_constant_ranges: &[],
        });
        let cull_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&cull_pl),
                module: &cull_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let cull_count_pipeline = cull_pipe("cull_count");
        let cull_scan_pipeline = cull_pipe("cull_scan");
        let cull_emit_pipeline = cull_pipe("cull_emit");

        // Compute edge rasterizer (see edge_raster.wgsl / edge_resolve.wgsl).
        let raster_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge_raster.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/edge_raster.wgsl").into()),
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge_resolve.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/edge_resolve.wgsl").into()),
        });
        let compute_uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let raster_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edge_raster_bgl"),
            entries: &[
                compute_uniform(0),
                compute_uniform(1),
                cull_ro(2),
                cull_ro(3),
                cull_ro(4),
                compute_uniform(5),
            ],
        });
        let raster_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edge_raster_accum_bgl"),
            entries: &[
                compute_uniform(0),
                rw_storage(1),
                rw_storage(2),
                rw_storage(3),
                rw_storage(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let raster_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edge_raster_pl"),
            bind_group_layouts: &[&raster_bgl, &raster_accum_bgl],
            push_constant_ranges: &[],
        });
        let tile_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&raster_pl),
                module: &raster_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let tile_count_pipeline = tile_pipe("tile_count");
        let tile_scan_pipeline = tile_pipe("tile_scan");
        let tile_emit_pipeline = tile_pipe("tile_emit");
        let tile_raster_pipeline = tile_pipe("tile_raster");
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edge_resolve_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edge_resolve_pl"),
            bind_group_layouts: &[&resolve_bgl],
            push_constant_ranges: &[],
        });
        // Blend One/One: the accumulator already holds premultiplied
        // rgb * edge_alpha sums, added over the cleared background exactly
        // like the hardware additive edge pass.
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge_resolve_pipeline"),
            layout: Some(&resolve_pl),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fs_resolve"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        Renderer {
            camera_buf,
            params_buf,
            view_bg,
            graph_bg,
            graph_bgl,
            edge_types: Vec::new(),
            node_pipeline,
            edge_pipeline,
            pick_pipeline,
            cull_bgl,
            cull_count_pipeline,
            cull_scan_pipeline,
            cull_emit_pipeline,
            raster_bgl,
            raster_accum_bgl,
            tile_count_pipeline,
            tile_scan_pipeline,
            tile_emit_pipeline,
            tile_raster_pipeline,
            resolve_bgl,
            resolve_pipeline,
            accum: None,
            cull_dirty: true,
            last_cam: None,
            last_edge_alpha: None,
            num_nodes: graph.num_nodes as u32,
            draw_edges: true,
            draw_nodes: true,
            edge_raster: false,
        }
    }

    /// Switch between the hardware LineList path and the compute rasterizer.
    /// Safe to flip every frame; both paths share the same dirty tracking.
    pub fn set_edge_raster(&mut self, on: bool) {
        if self.edge_raster != on {
            self.edge_raster = on;
            self.cull_dirty = true;
        }
    }

    pub fn edge_raster(&self) -> bool {
        self.edge_raster
    }

    pub fn update_camera(&mut self, queue: &wgpu::Queue, cam: &CameraUniform) {
        // Skip the upload and cull invalidation when the camera is unchanged —
        // this is what lets a settled graph with a still camera reuse last
        // frame's compacted edge set.
        let changed = self
            .last_cam
            .is_none_or(|c| bytemuck::bytes_of(&c) != bytemuck::bytes_of(cam));
        if changed {
            self.last_cam = Some(*cam);
            self.cull_dirty = true;
            queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
        }
    }

    /// Tell the renderer node positions changed (sim stepped, re-seed, …), so
    /// the compacted visible-edge sets must be rebuilt.
    pub fn mark_scene_dirty(&mut self) {
        self.cull_dirty = true;
    }

    /// (Re)create the tiled-raster GPU state when the surface size changes
    /// (or after set_edge_types dropped it): the 12-byte-per-pixel fixed-point
    /// accumulator, the per-tile scratch, the (tile, edge) pair scratch, and
    /// the per-batch uniform.
    fn ensure_accum(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.accum.as_ref().is_some_and(|a| a.w == w && a.h == h) {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_accum"),
            size: (w as u64 * h as u64 * 3 * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tiles_x = w.div_ceil(TILE);
        let tiles_y = h.div_ceil(TILE);
        let ntiles = tiles_x as u64 * tiles_y as u64;
        let dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edge_accum_dims"),
            contents: bytemuck::cast_slice(&[w, h, tiles_x, tiles_y]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tile_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_tile_counts"),
            size: (ntiles * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tile_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_tile_offsets"),
            size: (ntiles * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Worst-case pairs for the largest edge type in one batch; a batch
        // never needs more than max_edges * (tiles_x + tiles_y + 1).
        let max_edges = self.edge_types.iter().map(|t| t.num_edges as u64).max().unwrap_or(0);
        let max_tiles_per_edge = (tiles_x + tiles_y + 1) as u64;
        let pairs_cap = (max_edges * max_tiles_per_edge).clamp(1, PAIRS_CAP);
        let pairs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_tile_pairs"),
            size: pairs_cap * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let batch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_batch_params"),
            size: MAX_BATCHES as u64 * BATCH_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raster_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edge_accum_raster_bg"),
            layout: &self.raster_accum_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tile_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tile_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pairs.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &batch_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
        });
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edge_accum_resolve_bg"),
            layout: &self.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
            ],
        });
        self.accum = Some(AccumGpu {
            buf,
            batch_buf,
            w,
            h,
            tiles_x,
            tiles_y,
            pairs_cap,
            raster_bg,
            resolve_bg,
        });
        self.cull_dirty = true;
    }

    /// Prepare edge data for this frame on whichever path is active: compact
    /// the visible set (hardware path) or clear + software-rasterize into the
    /// accumulation buffer (compute path). No-op while nothing is dirty.
    /// Must be encoded before the render pass. Returns whether work ran.
    pub fn encode_edge_prep(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        w: u32,
        h: u32,
        enc: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) -> bool {
        if self.edge_raster {
            self.ensure_accum(device, w.max(1), h.max(1));
            self.encode_edge_raster(queue, enc, timestamp_writes)
        } else {
            self.encode_edge_cull(enc, timestamp_writes)
        }
    }

    /// Tiled compute-raster path: clear the accumulator, then per edge type
    /// and per batch run count → scan → emit → tile_raster (see
    /// edge_raster.wgsl). Integer accumulation commutes, so pair order, batch
    /// split, and scheduling cannot change the image.
    fn encode_edge_raster(
        &mut self,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) -> bool {
        if !self.cull_dirty {
            return false;
        }
        self.cull_dirty = false;
        let Some(accum) = &self.accum else {
            return false;
        };

        // Plan batches: (edge type index, start, count) with per-batch pair
        // demand bounded by the pair scratch.
        let max_tiles_per_edge = (accum.tiles_x + accum.tiles_y + 1) as u64;
        let batch_edges = u32::try_from((accum.pairs_cap / max_tiles_per_edge).max(1))
            .unwrap_or(u32::MAX);
        let mut plan: Vec<(usize, u32, u32)> = Vec::new();
        for (ti, et) in self.edge_types.iter().enumerate() {
            if !(et.visible && et.num_edges > 0) {
                continue;
            }
            let mut start = 0u32;
            while start < et.num_edges {
                let count = (et.num_edges - start).min(batch_edges);
                plan.push((ti, start, count));
                start += count;
            }
        }
        if plan.is_empty() {
            enc.clear_buffer(&accum.buf, 0, None);
            return false;
        }
        assert!(
            plan.len() <= MAX_BATCHES,
            "edge raster: {} batches exceeds MAX_BATCHES ({}); grow PAIRS_CAP or MAX_BATCHES",
            plan.len(),
            MAX_BATCHES
        );
        let mut params = vec![0u32; plan.len() * (BATCH_STRIDE as usize / 4)];
        for (i, &(_, start, count)) in plan.iter().enumerate() {
            let o = i * (BATCH_STRIDE as usize / 4);
            params[o] = start;
            params[o + 1] = count;
        }
        queue.write_buffer(&accum.batch_buf, 0, bytemuck::cast_slice(&params));

        enc.clear_buffer(&accum.buf, 0, None);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("edge_raster"),
            timestamp_writes,
        });
        let mut bound_type = usize::MAX;
        for (i, &(ti, _, count)) in plan.iter().enumerate() {
            if ti != bound_type {
                pass.set_bind_group(0, &self.edge_types[ti].raster_bg, &[]);
                bound_type = ti;
            }
            pass.set_bind_group(1, &accum.raster_bg, &[(i as u32) * BATCH_STRIDE as u32]);
            let wgs = count.div_ceil(256);
            let x = wgs.min(65_535);
            let y = wgs.div_ceil(x.max(1));
            pass.set_pipeline(&self.tile_count_pipeline);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.tile_scan_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.tile_emit_pipeline);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.tile_raster_pipeline);
            pass.dispatch_workgroups(accum.tiles_x, accum.tiles_y, 1);
        }
        true
    }

    /// Encode the visible-edge compaction (3 compute dispatches per visible
    /// edge type) if anything invalidated it. Must be encoded before the
    /// render pass that draws edges. Returns whether it ran.
    fn encode_edge_cull(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites>,
    ) -> bool {
        if !self.cull_dirty {
            return false;
        }
        self.cull_dirty = false;
        let any = self
            .edge_types
            .iter()
            .any(|et| et.visible && et.num_edges > 0);
        if !any {
            return false;
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("edge_cull"),
            timestamp_writes,
        });
        for et in &self.edge_types {
            if !(et.visible && et.num_edges > 0) {
                continue;
            }
            let nchunks = et.num_edges.div_ceil(CULL_CHUNK);
            let x = nchunks.min(65_535);
            let y = nchunks.div_ceil(x.max(1));
            pass.set_bind_group(0, &et.cull_bg, &[]);
            pass.set_pipeline(&self.cull_count_pipeline);
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(&self.cull_scan_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.cull_emit_pipeline);
            pass.dispatch_workgroups(x, y, 1);
        }
        true
    }

    pub fn update_params(&mut self, queue: &wgpu::Queue, params: &RenderParams) {
        if self.last_edge_alpha != Some(params.edge_alpha) {
            self.last_edge_alpha = Some(params.edge_alpha);
            if self.edge_raster {
                self.cull_dirty = true;
            }
        }
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    /// Node colors are baked into the compute-raster accumulation; call this
    /// after rewriting the color buffer so the raster path re-draws.
    pub fn mark_colors_dirty(&mut self) {
        if self.edge_raster {
            self.cull_dirty = true;
        }
    }

    /// Pick the node under a cursor pixel by rendering node ids to an R32Uint
    /// target and reading back the single texel. The camera uniform must already
    /// reflect the current view (the app updates it every frame). Returns the
    /// node index, or None for empty space.
    pub fn pick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        cursor_x: u32,
        cursor_y: u32,
    ) -> Option<u32> {
        if self.num_nodes == 0 || width == 0 || height == 0 {
            return None;
        }
        let cx = cursor_x.min(width - 1);
        let cy = cursor_y.min(height - 1);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_tex"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pick") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_bind_group(1, &self.graph_bg, &[]);
            pass.set_pipeline(&self.pick_pipeline);
            pass.draw(0..6, 0..self.num_nodes);
        }

        // Copy the single texel under the cursor into a 256-byte staging buffer.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: cx, y: cy, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));

        let slice = staging.slice(..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range();
        let id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();

        if id == 0 {
            None
        } else {
            Some(id - 1)
        }
    }

    /// Upload the edge types (buffers + per-type bind groups). Replaces any
    /// previously set types.
    pub fn set_edge_types(&mut self, device: &wgpu::Device, graph: &GpuGraph, specs: &[EdgeTypeInput]) {
        self.edge_types.clear();
        self.cull_dirty = true;
        // The pair scratch is sized from the largest edge type; rebuild the
        // tiled-raster state lazily on the next compute-path frame.
        self.accum = None;
        for s in specs {
            // Sort edges by endpoint ids for memory locality: consecutive
            // vertices then fetch nearby positions/colors, and repeated node
            // ids land close together for the post-transform vertex cache.
            // Additive blending is order-independent in the final image (each
            // pixel receives the same set of contributions), and the sort is
            // stable per upload, so frames stay deterministic.
            let mut sorted: Vec<[u32; 2]> = s.edges.to_vec();
            sorted.sort_unstable_by_key(|&[a, b]| ((a.min(b) as u64) << 32) | a.max(b) as u64);
            let flat: Vec<u32> = sorted.iter().flat_map(|&[a, b]| [a, b]).collect();
            let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edge_type_buf"),
                contents: if flat.is_empty() {
                    bytemuck::cast_slice(&[0u32, 0u32])
                } else {
                    bytemuck::cast_slice(&flat)
                },
                usage: wgpu::BufferUsages::STORAGE,
            });
            let (mode, color) = match s.color {
                Some(c) => (1u32, unpack_norm(c)),
                None => (0u32, [0.0; 4]),
            };
            let style_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edge_style_buf"),
                contents: bytemuck::bytes_of(&EdgeStyle { color, mode, _pad: [0; 3] }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("edge_type_bg"),
                layout: &self.graph_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: graph.positions.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: graph.colors.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: graph.sizes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: edge_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: style_buf.as_entire_binding() },
                ],
            });

            let num_edges = s.edges.len() as u32;
            let compact = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("edge_compact_indices"),
                size: (num_edges as u64 * 2 * 4).max(8),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
                mapped_at_creation: false,
            });
            // GPU-written by cull_scan; init to an empty draw so a never-culled
            // type can't draw garbage.
            let indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edge_indirect"),
                contents: bytemuck::cast_slice(&[0u32, 1, 0, 0, 0]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            });
            let nchunks = num_edges.div_ceil(CULL_CHUNK).max(1);
            let chunk_counts = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("edge_cull_chunks"),
                size: nchunks as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let cull_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("edge_cull_bg"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: graph.positions.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: edge_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: compact.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: indirect.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: chunk_counts.as_entire_binding() },
                ],
            });

            let raster_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("edge_raster_bg"),
                layout: &self.raster_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: graph.positions.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: graph.colors.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: edge_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: style_buf.as_entire_binding() },
                ],
            });

            self.edge_types.push(EdgeTypeGpu {
                bind_group,
                cull_bg,
                raster_bg,
                compact,
                indirect,
                num_edges,
                visible: s.visible,
            });
        }
    }

    /// Toggle visibility of edge type `i`.
    pub fn set_edge_visible(&mut self, i: usize, visible: bool) {
        if let Some(et) = self.edge_types.get_mut(i) {
            et.visible = visible;
            if visible {
                // Its compaction may be stale from while it was hidden.
                self.cull_dirty = true;
            }
        }
    }

    /// Record draw calls into an already-begun render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        // Edges under nodes. Hardware path: one indexed indirect draw per
        // visible edge type over the compacted set. Compute path: a fullscreen
        // triangle adds the software-rasterized accumulation onto the frame.
        // Either way encode_edge_prep must have run when the scene changed.
        if self.draw_edges {
            if self.edge_raster {
                if let Some(accum) = &self.accum {
                    pass.set_pipeline(&self.resolve_pipeline);
                    pass.set_bind_group(0, &accum.resolve_bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            } else {
                pass.set_bind_group(0, &self.view_bg, &[]);
                pass.set_pipeline(&self.edge_pipeline);
                for et in &self.edge_types {
                    if et.visible && et.num_edges > 0 {
                        pass.set_bind_group(1, &et.bind_group, &[]);
                        pass.set_index_buffer(et.compact.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed_indirect(&et.indirect, 0);
                    }
                }
            }
        }
        if self.draw_nodes && self.num_nodes > 0 {
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_bind_group(1, &self.graph_bg, &[]);
            pass.set_pipeline(&self.node_pipeline);
            pass.draw(0..6, 0..self.num_nodes);
        }
    }
}
