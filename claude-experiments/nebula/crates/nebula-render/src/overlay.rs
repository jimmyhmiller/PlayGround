//! In-engine 2D overlay: panels and bitmap text, drawn in screen pixels on top of
//! the graph. No external UI toolkit — just two instanced-quad pipelines and an
//! embedded 8x8 font, which keeps us on a single wgpu version and adds ~zero deps.

use crate::font::FONT8X8;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const GLYPH_W: f32 = 8.0;
pub const GLYPH_H: f32 = 8.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayUniform {
    viewport: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RectInst {
    pos: [f32; 2],
    size: [f32; 2],
    color: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlyphInst {
    pos: [f32; 2],
    size: [f32; 2],
    color: u32,
    ch: u32,
}

pub struct Overlay {
    uniform_buf: wgpu::Buffer,
    font_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    data_bgl: wgpu::BindGroupLayout,
    rect_pipeline: wgpu::RenderPipeline,
    glyph_pipeline: wgpu::RenderPipeline,

    rects: Vec<RectInst>,
    glyphs: Vec<GlyphInst>,
    rect_buf: wgpu::Buffer,
    glyph_buf: wgpu::Buffer,
    rect_cap: usize,
    glyph_cap: usize,
}

impl Overlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_uniform"),
            size: std::mem::size_of::<OverlayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pack the 8x8 font: 128 glyphs * 8 rows, one u32 per row (byte value).
        let mut font_rows: Vec<u32> = Vec::with_capacity(128 * 8);
        for glyph in FONT8X8.iter() {
            for &row in glyph.iter() {
                font_rows.push(row as u32);
            }
        }
        let font_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("font"),
            contents: bytemuck::cast_slice(&font_rows),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_view_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_data_bgl"),
            entries: &[ro(0), ro(1), ro(2)],
        });

        let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_view_bg"),
            layout: &view_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_pl"),
            bind_group_layouts: &[&view_bgl, &data_bgl],
            push_constant_ranges: &[],
        });

        let blend = wgpu::BlendState::ALPHA_BLENDING;
        let make = |vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(vs),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
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
            })
        };
        let rect_pipeline = make("vs_rect", "fs_rect");
        let glyph_pipeline = make("vs_glyph", "fs_glyph");

        let rect_cap = 256usize;
        let glyph_cap = 8192usize;
        let rect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_rects"),
            size: (rect_cap * std::mem::size_of::<RectInst>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let glyph_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_glyphs"),
            size: (glyph_cap * std::mem::size_of::<GlyphInst>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Overlay {
            uniform_buf,
            font_buf,
            view_bg,
            data_bgl,
            rect_pipeline,
            glyph_pipeline,
            rects: Vec::new(),
            glyphs: Vec::new(),
            rect_buf,
            glyph_buf,
            rect_cap,
            glyph_cap,
        }
    }

    pub fn begin(&mut self) {
        self.rects.clear();
        self.glyphs.clear();
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: u32) {
        self.rects.push(RectInst { pos: [x, y], size: [w, h], color, _p: 0 });
    }

    /// Draw text at (x,y) top-left, glyph cell `scale`*8 px. Returns the x after.
    pub fn text(&mut self, x: f32, y: f32, scale: f32, color: u32, s: &str) -> f32 {
        let cw = GLYPH_W * scale;
        let ch = GLYPH_H * scale;
        let mut cx = x;
        for c in s.chars() {
            if c == '\n' {
                continue;
            }
            let code = if (c as u32) < 128 { c as u32 } else { b'?' as u32 };
            if c != ' ' {
                self.glyphs.push(GlyphInst { pos: [cx, y], size: [cw, ch], color, ch: code });
            }
            cx += cw;
        }
        let _ = ch;
        cx
    }

    /// Width in pixels of a string at `scale`.
    pub fn text_width(s: &str, scale: f32) -> f32 {
        s.chars().count() as f32 * GLYPH_W * scale
    }

    /// Upload instance data and record draws. Call inside a render pass. wgpu 27
    /// ref-counts bind groups/pipelines, so the locally-built bind group here is
    /// valid even though it is dropped when this returns.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (f32, f32),
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&OverlayUniform { viewport: [viewport.0, viewport.1], _pad: [0.0; 2] }),
        );

        // Grow buffers if needed.
        if self.rects.len() > self.rect_cap {
            self.rect_cap = self.rects.len().next_power_of_two();
            self.rect_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay_rects"),
                size: (self.rect_cap * std::mem::size_of::<RectInst>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if self.glyphs.len() > self.glyph_cap {
            self.glyph_cap = self.glyphs.len().next_power_of_two();
            self.glyph_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay_glyphs"),
                size: (self.glyph_cap * std::mem::size_of::<GlyphInst>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.rects.is_empty() {
            queue.write_buffer(&self.rect_buf, 0, bytemuck::cast_slice(&self.rects));
        }
        if !self.glyphs.is_empty() {
            queue.write_buffer(&self.glyph_buf, 0, bytemuck::cast_slice(&self.glyphs));
        }

        // A fresh bind group each frame since the buffers may have been recreated.
        let data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_data_bg"),
            layout: &self.data_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.rect_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.glyph_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.font_buf.as_entire_binding() },
            ],
        });

        pass.set_bind_group(0, &self.view_bg, &[]);
        pass.set_bind_group(1, &data_bg, &[]);
        if !self.rects.is_empty() {
            pass.set_pipeline(&self.rect_pipeline);
            pass.draw(0..6, 0..self.rects.len() as u32);
        }
        if !self.glyphs.is_empty() {
            pass.set_pipeline(&self.glyph_pipeline);
            pass.draw(0..6, 0..self.glyphs.len() as u32);
        }
    }
}
