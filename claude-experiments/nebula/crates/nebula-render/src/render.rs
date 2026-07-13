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
/// once at upload so that drawing any prefix of the buffer is a uniform random
/// sample of the edge set — that's what makes the edge budget statistically
/// honest under additive blending.
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
    num_nodes: u32,
    pub draw_edges: bool,
    pub draw_nodes: bool,
    /// Max edges drawn per frame across all visible types. When the graph has
    /// more, a uniform random sample (a prefix of the shuffled buffers) is drawn
    /// and the app compensates by boosting `edge_alpha` — additive blending makes
    /// the sampled image converge to the full one at a fraction of the fill cost.
    pub edge_budget: u32,
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
            num_nodes: graph.num_nodes as u32,
            draw_edges: true,
            draw_nodes: true,
            edge_budget: 1_000_000,
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
    }

    /// Toggle visibility of edge type `i`.
    pub fn set_edge_visible(&mut self, i: usize, visible: bool) {
        if let Some(et) = self.edge_types.get_mut(i) {
            et.visible = visible;
        }
    }

    /// Total edges across visible types (what a budget-free frame would draw).
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

    /// Fraction of each edge buffer drawn this frame (1.0 = everything). The
    /// same fraction applies to every visible type so relative densities between
    /// types stay correct.
    pub fn edge_sample_frac(&self) -> f32 {
        let total = self.visible_edge_count();
        if total == 0 || self.edge_budget as u64 >= total {
            1.0
        } else {
            self.edge_budget as f32 / total as f32
        }
    }

    /// Record draw calls into an already-begun render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_bind_group(0, &self.view_bg, &[]);

        // Edges under nodes: one draw per visible edge type (own bind group).
        // Buffers are pre-shuffled, so drawing the first `frac` of each is a
        // uniform sample of that type's edges.
        if self.draw_edges {
            let frac = self.edge_sample_frac();
            pass.set_pipeline(&self.edge_pipeline);
            for et in &self.edge_types {
                if et.visible && et.num_edges > 0 {
                    let count = ((et.num_edges as f32 * frac).ceil() as u32).min(et.num_edges);
                    if count == 0 {
                        continue;
                    }
                    pass.set_bind_group(1, &et.bind_group, &[]);
                    pass.draw(0..(count * 2), 0..1);
                }
            }
        }
        if self.draw_nodes && self.num_nodes > 0 {
            pass.set_bind_group(1, &self.graph_bg, &[]);
            pass.set_pipeline(&self.node_pipeline);
            pass.draw(0..6, 0..self.num_nodes);
        }
    }
}
