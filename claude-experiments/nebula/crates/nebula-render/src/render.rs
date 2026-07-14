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

/// GPU state for one edge type: its own edge buffer + bind group.
struct EdgeTypeGpu {
    bind_group: wgpu::BindGroup,
    num_edges: u32,
    visible: bool,
    /// Bundled rendering state, allocated for huge edge sets (see
    /// `BUNDLE_THRESHOLD`). Every edge is binned every frame; only the drawing
    /// aggregates.
    bundle: Option<BundleGpu>,
}

/// Above this many edges a type renders as per-frame computed bundles: every
/// edge is clipped and binned by (source cell, target cell) each frame, and
/// one centroid-to-centroid line per occupied pair carries the accumulated
/// brightness/colors. Pairs holding a single edge reproduce it exactly, so
/// sparse regions and zoomed-in views stay true; only dense flows aggregate.
/// This is a rendering change, not a limit — nothing is skipped or deferred.
const BUNDLE_THRESHOLD: u32 = 1_500_000;

/// Max bin cells per level (bounds the pair matrix). The per-frame sizing
/// loop grows the cell size until the visible grid fits this budget.
const BUNDLE_MAX_NCELLS: u32 = 1_500;
const BUNDLE_MAX_PAIRS: u64 = (BUNDLE_MAX_NCELLS as u64) * (BUNDLE_MAX_NCELLS as u64);
/// Target on-screen size of a FINE bin cell in pixels; the fine level spans
/// (target/2, target] px, the coarse level is exactly 2x.
const BUNDLE_TARGET_CELL_PX: f32 = 128.0;

/// Matches `BParams` in bundle.wgsl / bundle_draw.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BundleParams {
    cells_x: u32,
    cells_y: u32,
    num_edges: u32,
    mode: u32,
    tint: [f32; 4],
    origin: [f32; 2],
    cell: f32,
    weight: f32,
    clip_min: [f32; 2],
    clip_max: [f32; 2],
    edge_alpha: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

/// One LOD scale of a type's bundling: its own pair accumulators + params.
struct BundleLevel {
    params_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    bin_bg: wgpu::BindGroup,
    draw_bg: wgpu::BindGroup,
    /// Pair count of the current frame's grid (dispatch/draw size).
    pairs: u32,
    /// False when this level's cross-fade weight is ~0 (skip bin + draw).
    active: bool,
}

/// Per-type bundling state: two world-anchored power-of-two scales,
/// cross-faded by fractional zoom so nothing ever re-bins during camera
/// motion and scale transitions are invisible.
struct BundleGpu {
    levels: [BundleLevel; 2],
    mode: u32,
    tint: [f32; 4],
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
    // Bundling pipelines + layouts (used by edge types over BUNDLE_THRESHOLD).
    bundle_clear_pipeline: wgpu::ComputePipeline,
    bundle_bin_pipeline: wgpu::ComputePipeline,
    bundle_draw_pipeline: wgpu::RenderPipeline,
    bundle_view_bgl: wgpu::BindGroupLayout,
    bundle_bin_bgl: wgpu::BindGroupLayout,
    bundle_draw_bgl: wgpu::BindGroupLayout,
    num_nodes: u32,
    num_edges: u32,
    pub draw_edges: bool,
    pub draw_nodes: bool,
}

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

        // --- Bundling pipelines ------------------------------------------------
        let bundle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bundle.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bundle.wgsl").into()),
        });
        let bundle_draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bundle_draw.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bundle_draw.wgsl").into()),
        });
        let uniform_cv = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bundle_view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bundle_view_bgl"),
            entries: &[uniform_cv(0), uniform_cv(1)],
        });
        let storage_c = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bundle_bin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bundle_bin_bgl"),
            entries: &[
                storage_c(0, true),  // positions
                storage_c(1, true),  // colors
                storage_c(2, true),  // edges
                storage_c(3, false), // pair_count
                storage_c(4, false), // pair_geom
                storage_c(5, false), // pair_col
            ],
        });
        let storage_v = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bundle_draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bundle_draw_bgl"),
            entries: &[storage_v(0), storage_v(1), storage_v(2)],
        });
        let bundle_compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bundle_compute_pl"),
            bind_group_layouts: &[&bundle_view_bgl, &bundle_bin_bgl],
            push_constant_ranges: &[],
        });
        let make_bundle_compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&bundle_compute_pl),
                module: &bundle_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bundle_clear_pipeline = make_bundle_compute("clear_pairs");
        let bundle_bin_pipeline = make_bundle_compute("bin_edges");
        let bundle_draw_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bundle_draw_pl"),
            bind_group_layouts: &[&bundle_view_bgl, &bundle_draw_bgl],
            push_constant_ranges: &[],
        });
        let bundle_draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bundle_draw_pipeline"),
            layout: Some(&bundle_draw_pl),
            vertex: wgpu::VertexState {
                module: &bundle_draw_shader,
                entry_point: Some("vs_bundle"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bundle_draw_shader,
                entry_point: Some("fs_bundle"),
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
            bundle_clear_pipeline,
            bundle_bin_pipeline,
            bundle_draw_pipeline,
            bundle_view_bgl,
            bundle_bin_bgl,
            bundle_draw_bgl,
            num_nodes: graph.num_nodes as u32,
            num_edges: graph.num_edges as u32,
            draw_edges: true,
            draw_nodes: true,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, cam: &CameraUniform) {
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &RenderParams) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
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
        for s in specs {
            let flat: Vec<u32> = s.edges.iter().flat_map(|&[a, b]| [a, b]).collect();
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
            // Huge edge sets also get bundling accumulators.
            let bundle = if (s.edges.len() as u32) > BUNDLE_THRESHOLD {
                let (bmode, btint) = match s.color {
                    Some(c) => (1u32, unpack_norm(c)),
                    None => (0u32, [0.0f32; 4]),
                };
                let mk_level = || {
                    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("bundle_params"),
                        size: std::mem::size_of::<BundleParams>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mk_storage = |label: &str, bytes: u64| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(label),
                            size: bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        })
                    };
                    let pair_count = mk_storage("bundle_pair_count", BUNDLE_MAX_PAIRS * 4);
                    let pair_geom = mk_storage("bundle_pair_geom", BUNDLE_MAX_PAIRS * 4 * 4);
                    let pair_col = mk_storage("bundle_pair_col", BUNDLE_MAX_PAIRS * 6 * 4);
                    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bundle_view_bg"),
                        layout: &self.bundle_view_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                        ],
                    });
                    let bin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bundle_bin_bg"),
                        layout: &self.bundle_bin_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: graph.positions.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: graph.colors.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: edge_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: pair_count.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: pair_geom.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: pair_col.as_entire_binding() },
                        ],
                    });
                    let draw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bundle_draw_bg"),
                        layout: &self.bundle_draw_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: pair_count.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: pair_geom.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: pair_col.as_entire_binding() },
                        ],
                    });
                    BundleLevel { params_buf, view_bg, bin_bg, draw_bg, pairs: 0, active: false }
                };
                log::info!(
                    "edge type '{}': {} edges -> bundled rendering (world-anchored, 2 LOD scales)",
                    s.name, s.edges.len()
                );
                Some(BundleGpu { levels: [mk_level(), mk_level()], mode: bmode, tint: btint })
            } else {
                None
            };
            self.edge_types.push(EdgeTypeGpu {
                bind_group,
                num_edges: s.edges.len() as u32,
                visible: s.visible,
                bundle,
            });
        }
    }

    /// Toggle visibility of edge type `i`.
    pub fn set_edge_visible(&mut self, i: usize, visible: bool) {
        if let Some(et) = self.edge_types.get_mut(i) {
            et.visible = visible;
        }
    }

    /// Upload per-frame bundle params: pick the two world-space power-of-two
    /// cell scales bracketing the ideal on-screen cell size and their
    /// cross-fade weights, and lay a world-anchored grid over the view for
    /// each. Because grid cells are fixed world rectangles (origin quantized
    /// to whole cells), camera motion never re-bins anything — bundles are
    /// stable geometry — and the octave cross-fade makes scale changes
    /// invisible. Call once per frame before `record_bundle_compute`.
    pub fn update_bundles(
        &mut self,
        queue: &wgpu::Queue,
        cam: &crate::camera::CameraUniform,
        edge_alpha: f32,
    ) {
        let zoom = cam.zoom.max(1e-12);
        let half = [cam.viewport[0] * 0.5 / zoom, cam.viewport[1] * 0.5 / zoom];
        let view_min = [cam.center[0] - half[0], cam.center[1] - half[1]];
        let view_max = [cam.center[0] + half[0], cam.center[1] + half[1]];

        // Grow the target cell size until the fine grid fits the pair budget.
        let mut target_px = BUNDLE_TARGET_CELL_PX;
        let (fine_cell, frac) = loop {
            let ideal = target_px / zoom;
            let fine = 2.0f32.powf(ideal.log2().floor());
            let cx = ((view_max[0] - view_min[0]) / fine).ceil() as u32 + 5;
            let cy = ((view_max[1] - view_min[1]) / fine).ceil() as u32 + 5;
            if cx * cy <= BUNDLE_MAX_NCELLS {
                break (fine, (ideal / fine).log2().clamp(0.0, 1.0));
            }
            target_px *= 1.25;
        };

        // (cell size, cross-fade weight) for the two bracketing scales.
        let scales = [(fine_cell, 1.0 - frac), (fine_cell * 2.0, frac)];
        for et in &mut self.edge_types {
            let Some(b) = et.bundle.as_mut() else { continue };
            for (lvl, &(cell, weight)) in b.levels.iter_mut().zip(scales.iter()) {
                lvl.active = weight > 0.003;
                if !lvl.active {
                    continue;
                }
                // World-anchored grid covering the view + one cell of margin.
                let origin = [
                    (view_min[0] / cell).floor() * cell - cell,
                    (view_min[1] / cell).floor() * cell - cell,
                ];
                let clip_min = [view_min[0] - cell, view_min[1] - cell];
                let clip_max = [view_max[0] + cell, view_max[1] + cell];
                let cells_x = (((clip_max[0] - origin[0]) / cell).ceil() as u32 + 1)
                    .min(BUNDLE_MAX_NCELLS);
                let cells_y = (((clip_max[1] - origin[1]) / cell).ceil() as u32 + 1)
                    .min(BUNDLE_MAX_NCELLS / cells_x.max(1));
                let ncells = cells_x * cells_y;
                lvl.pairs = ncells * ncells;
                let params = BundleParams {
                    cells_x,
                    cells_y,
                    num_edges: et.num_edges,
                    mode: b.mode,
                    tint: b.tint,
                    origin,
                    cell,
                    weight,
                    clip_min,
                    clip_max,
                    edge_alpha,
                    _p0: 0.0,
                    _p1: 0.0,
                    _p2: 0.0,
                };
                queue.write_buffer(&lvl.params_buf, 0, bytemuck::bytes_of(&params));
            }
        }
    }

    /// Re-bin every edge of every bundled visible type for this frame (encode
    /// before the main pass, same encoder). Fully synchronous with the current
    /// camera and positions — no temporal state.
    pub fn record_bundle_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.draw_edges {
            return;
        }
        for et in &self.edge_types {
            let Some(b) = et.bundle.as_ref() else { continue };
            if !et.visible || et.num_edges == 0 {
                continue;
            }
            let edge_groups =
                crate::layout_gpu::dispatch_dims((et.num_edges as u64).div_ceil(256));
            for lvl in &b.levels {
                if !lvl.active {
                    continue;
                }
                let pair_groups =
                    crate::layout_gpu::dispatch_dims((lvl.pairs as u64).div_ceil(256));
                {
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bundle_clear"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&self.bundle_clear_pipeline);
                    cp.set_bind_group(0, &lvl.view_bg, &[]);
                    cp.set_bind_group(1, &lvl.bin_bg, &[]);
                    cp.dispatch_workgroups(pair_groups.0, pair_groups.1, pair_groups.2);
                }
                {
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("bundle_bin"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&self.bundle_bin_pipeline);
                    cp.set_bind_group(0, &lvl.view_bg, &[]);
                    cp.set_bind_group(1, &lvl.bin_bg, &[]);
                    cp.dispatch_workgroups(edge_groups.0, edge_groups.1, edge_groups.2);
                }
            }
        }
    }

    /// Record draw calls into an already-begun render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        // Edges under nodes: one draw per visible edge type (own bind group).
        // Types over BUNDLE_THRESHOLD draw their per-frame computed bundles;
        // the rest draw every line directly.
        if self.draw_edges {
            for et in &self.edge_types {
                if !et.visible || et.num_edges == 0 {
                    continue;
                }
                match et.bundle.as_ref() {
                    Some(b) => {
                        pass.set_pipeline(&self.bundle_draw_pipeline);
                        for lvl in &b.levels {
                            if lvl.active && lvl.pairs > 0 {
                                pass.set_bind_group(0, &lvl.view_bg, &[]);
                                pass.set_bind_group(1, &lvl.draw_bg, &[]);
                                pass.draw(0..2, 0..lvl.pairs);
                            }
                        }
                    }
                    None => {
                        pass.set_pipeline(&self.edge_pipeline);
                        pass.set_bind_group(0, &self.view_bg, &[]);
                        pass.set_bind_group(1, &et.bind_group, &[]);
                        pass.draw(0..(et.num_edges * 2), 0..1);
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
