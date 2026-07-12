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

pub struct Renderer {
    camera_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    graph_bg: wgpu::BindGroup,
    node_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
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
        let graph_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("graph_bgl"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2), ro_storage(3)],
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

        Renderer {
            camera_buf,
            params_buf,
            view_bg,
            graph_bg,
            node_pipeline,
            edge_pipeline,
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

    /// Record draw calls into an already-begun render pass.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_bind_group(0, &self.view_bg, &[]);
        pass.set_bind_group(1, &self.graph_bg, &[]);

        // Edges under nodes.
        if self.draw_edges && self.num_edges > 0 {
            pass.set_pipeline(&self.edge_pipeline);
            pass.draw(0..(self.num_edges * 2), 0..1);
        }
        if self.draw_nodes && self.num_nodes > 0 {
            pass.set_pipeline(&self.node_pipeline);
            pass.draw(0..6, 0..self.num_nodes);
        }
    }
}
