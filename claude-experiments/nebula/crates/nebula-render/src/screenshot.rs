//! Headless offscreen rendering — render one frame into a texture, read it back,
//! and save a PNG. Used both for `--screenshot` verification and for anyone who
//! wants to script image output without a visible window.

use crate::camera::CameraUniform;
use crate::gpu::Gpu;
use crate::render::{RenderParams, Renderer};
use std::path::Path;

/// Render one frame to an offscreen RGBA8 texture and return the pixels plus the
/// row-padded width used for the copy.
pub fn render_to_rgba(
    gpu: &Gpu,
    renderer: &Renderer,
    width: u32,
    height: u32,
    cam: &CameraUniform,
    params: &RenderParams,
) -> Vec<u8> {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    renderer.update_camera(&gpu.queue, cam);
    renderer.update_params(&gpu.queue, params);

    // The offscreen target may differ in format from the swapchain; the renderer
    // was built for the swapchain format, so callers must pass a renderer built
    // for `Rgba8UnormSrgb`. We build a dedicated one in `capture` below.
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("offscreen-enc") });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.015, g: 0.015, b: 0.03, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.draw(&mut pass);
    }

    // Copy texture -> buffer (bytes_per_row must be 256-aligned).
    let bpp = 4u32;
    let unpadded = width * bpp;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buf_size = (padded * height) as u64;
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &out_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    gpu.queue.submit(Some(enc.finish()));

    let slice = out_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok();

    let data = slice.get_mapped_range();
    // Un-pad rows.
    let mut rgba = vec![0u8; (unpadded * height) as usize];
    for y in 0..height as usize {
        let src = y * padded as usize;
        let dst = y * unpadded as usize;
        rgba[dst..dst + unpadded as usize]
            .copy_from_slice(&data[src..src + unpadded as usize]);
    }
    drop(data);
    out_buf.unmap();
    rgba
}

/// Build a renderer for the offscreen sRGB format and save a PNG of the current
/// graph state. Returns the path written.
pub fn capture(
    gpu: &Gpu,
    graph: &crate::scene::GpuGraph,
    width: u32,
    height: u32,
    cam: &CameraUniform,
    params: &RenderParams,
    draw_edges: bool,
    draw_nodes: bool,
    path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = Renderer::new(&gpu.device, format, graph);
    renderer.draw_edges = draw_edges;
    renderer.draw_nodes = draw_nodes;
    let rgba = render_to_rgba(gpu, &renderer, width, height, cam, params);

    let file = std::fs::File::create(path.as_ref())?;
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&rgba)?;
    writer.finish()?;
    log::info!("saved screenshot -> {}", path.as_ref().display());
    Ok(())
}
