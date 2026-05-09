use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct SliceInstance {
    /// (x, y, w, h) in pixels, top-left origin.
    pub rect_px: [f32; 4],
    /// RGBA in 0..1.
    pub color: [f32; 4],
    /// Stable index into the renderer's `slices` array — used for hover lookup.
    pub instance_id: u32,
    /// Bit flags. 0 = normal. (Reserved for v2 selection state.)
    pub flags: u32,
    pub _pad: [u32; 2],
}

impl SliceInstance {
    pub const ATTRIBS: &'static [wgpu::VertexAttribute] = &[
        // rect_px: vec4<f32>
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x4,
        },
        // color: vec4<f32>
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x4,
        },
        // instance_id: u32
        wgpu::VertexAttribute {
            offset: 32,
            shader_location: 2,
            format: wgpu::VertexFormat::Uint32,
        },
        // flags: u32
        wgpu::VertexAttribute {
            offset: 36,
            shader_location: 3,
            format: wgpu::VertexFormat::Uint32,
        },
    ];

    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SliceInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: Self::ATTRIBS,
    };
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub viewport_size_px: [f32; 2],
    pub hovered: u32,
    pub _pad: u32,
}
