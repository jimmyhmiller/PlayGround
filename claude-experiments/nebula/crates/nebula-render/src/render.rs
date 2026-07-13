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
}

/// Deterministic Fisher-Yates shuffle (splitmix64). Edge buffers are shuffled
/// once at upload so that the per-frame accumulation slices are unbiased random
/// subsets: the picture converges uniformly everywhere instead of revealing the
/// file's edge order region by region.
fn shuffled(edges: &[[u32; 2]]) -> Vec<[u32; 2]> {
    let mut v = edges.to_vec();
    let mut state: u64 = 0x51ED_BEEF_2026;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..v.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn unpack_norm(c: u32) -> [f32; 4] {
    [
        (c & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 24) & 0xff) as f32 / 255.0,
    ]
}

/// Persistent offscreen accumulation state for progressive edge rendering.
struct EdgeAccum {
    _tex: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    /// Fraction of every visible type's edge buffer accumulated so far (0..=1).
    /// 1.0 = the texture holds every edge; less = a converging unbiased subset.
    frac: f32,
    bind_group: wgpu::BindGroup,
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
    /// Renders edge lines additively into the float accumulation texture.
    edge_accum_pipeline: wgpu::RenderPipeline,
    /// Composites the accumulation texture onto the frame, brightness-normalized.
    composite_pipeline: wgpu::RenderPipeline,
    composite_bgl: wgpu::BindGroupLayout,
    composite_buf: wgpu::Buffer,
    accum: Option<EdgeAccum>,
    pick_pipeline: wgpu::RenderPipeline,
    num_nodes: u32,
    pub draw_edges: bool,
    pub draw_nodes: bool,
    /// Edges rendered into the accumulation texture per frame. Every edge is
    /// always drawn — a graph larger than this just takes total/edges_per_frame
    /// frames to fully converge after the scene stops changing.
    pub edges_per_frame: u32,
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

        // Edges accumulate additively into a float texture: no per-draw 8-bit
        // saturation, and the texture persists across frames so the full edge
        // set converges even when it is too large to draw in one frame.
        let edge_accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge_accum_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba16Float,
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

        // Composite: fullscreen triangle adding the normalized accumulation
        // texture onto the scene.
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let composite_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite_params"),
            contents: bytemuck::bytes_of(&[1.0f32, 0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pl"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_composite"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
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

        Renderer {
            camera_buf,
            params_buf,
            view_bg,
            graph_bg,
            graph_bgl,
            edge_types: Vec::new(),
            node_pipeline,
            edge_accum_pipeline,
            composite_pipeline,
            composite_bgl,
            composite_buf,
            accum: None,
            pick_pipeline,
            num_nodes: graph.num_nodes as u32,
            draw_edges: true,
            draw_nodes: true,
            edges_per_frame: 1_000_000,
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
            let flat: Vec<u32> = shuffled(s.edges).into_iter().flat_map(|[a, b]| [a, b]).collect();
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
            self.edge_types.push(EdgeTypeGpu {
                bind_group,
                num_edges: s.edges.len() as u32,
                visible: s.visible,
            });
        }
        self.invalidate_edge_accum();
    }

    /// Toggle visibility of edge type `i`.
    pub fn set_edge_visible(&mut self, i: usize, visible: bool) {
        if let Some(et) = self.edge_types.get_mut(i) {
            et.visible = visible;
        }
        self.invalidate_edge_accum();
    }

    /// Restart edge accumulation (the accumulated image no longer matches what
    /// the edges would draw).
    pub fn invalidate_edge_accum(&mut self) {
        if let Some(a) = self.accum.as_mut() {
            a.frac = 0.0;
        }
    }

    /// Total edges across visible types.
    pub fn visible_edge_count(&self) -> u64 {
        if !self.draw_edges {
            return 0;
        }
        self.edge_types
            .iter()
            .filter(|t| t.visible)
            .map(|t| t.num_edges as u64)
            .sum()
    }

    /// Edges accumulated so far vs total: `(done, total)`. `done == total`
    /// means the current image contains every single edge.
    pub fn edge_accum_progress(&self) -> (u64, u64) {
        let total = self.visible_edge_count();
        let frac = self.accum.as_ref().map(|a| a.frac).unwrap_or(0.0);
        (self.accumulated_edges(frac), total)
    }

    /// True when the accumulation texture holds the complete edge set (or there
    /// is nothing to draw).
    pub fn edge_accum_converged(&self) -> bool {
        self.visible_edge_count() == 0
            || self.accum.as_ref().map(|a| a.frac >= 1.0).unwrap_or(false)
    }

    fn accumulated_edges(&self, frac: f32) -> u64 {
        self.edge_types
            .iter()
            .filter(|t| t.visible)
            .map(|t| (t.num_edges as f64 * frac as f64).floor() as u64)
            .sum()
    }

    /// Render the next slice of edges into the persistent accumulation texture
    /// (encode before the main pass, same encoder). `invalidate` restarts the
    /// accumulation — call it whenever the picture the edges describe changed:
    /// positions moved, camera moved, colors or edge style changed. Every edge
    /// is drawn eventually; a static scene converges to the exact full image in
    /// ceil(total / edges_per_frame) frames.
    pub fn accumulate_edges(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        invalidate: bool,
    ) {
        let total = self.visible_edge_count();
        if total == 0 || width == 0 || height == 0 {
            return;
        }

        // (Re)create the accumulation texture on first use / resize.
        let needs_tex = self
            .accum
            .as_ref()
            .map(|a| a.width != width || a.height != height)
            .unwrap_or(true);
        if needs_tex {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("edge_accum_tex"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite_bg"),
                layout: &self.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.composite_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            });
            self.accum = Some(EdgeAccum { _tex: tex, view, width, height, frac: 0.0, bind_group });
        }
        let accum = self.accum.as_mut().unwrap();
        if invalidate {
            accum.frac = 0.0;
        }

        let f0 = accum.frac;
        if f0 < 1.0 {
            let step = self.edges_per_frame.max(1) as f32 / total as f32;
            let f1 = (f0 + step).min(1.0);
            accum.frac = f1;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("edge_accum"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &accum.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // First slice after an invalidation clears the texture.
                        load: if f0 <= 0.0 {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.edge_accum_pipeline);
            pass.set_bind_group(0, &self.view_bg, &[]);
            for et in &self.edge_types {
                if !et.visible || et.num_edges == 0 {
                    continue;
                }
                // This frame's slice of the (pre-shuffled) buffer.
                let start = (et.num_edges as f64 * f0 as f64).floor() as u32;
                let end = (et.num_edges as f64 * f1 as f64).floor() as u32;
                if end > start {
                    pass.set_bind_group(1, &et.bind_group, &[]);
                    pass.draw(start * 2..end * 2, 0..1);
                }
            }
        }

        // Normalization factor: total / accumulated, so brightness is constant
        // while the image sharpens toward the full edge set.
        let frac_now = self.accum.as_ref().unwrap().frac;
        let done = self.accumulated_edges(frac_now).max(1);
        let factor = total as f32 / done as f32;
        queue.write_buffer(&self.composite_buf, 0, bytemuck::bytes_of(&[factor, 0.0f32, 0.0, 0.0]));
    }

    /// Add the accumulated edge image onto the current render pass (call before
    /// drawing nodes so edges stay underneath).
    pub fn composite_edges<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if !self.draw_edges {
            return;
        }
        let Some(accum) = self.accum.as_ref() else { return };
        if accum.frac <= 0.0 {
            return;
        }
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &accum.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Record node draw calls into an already-begun render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.draw_nodes && self.num_nodes > 0 {
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_bind_group(1, &self.graph_bg, &[]);
            pass.set_pipeline(&self.node_pipeline);
            pass.draw(0..6, 0..self.num_nodes);
        }
    }
}
