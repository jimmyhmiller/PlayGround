//! Headless offscreen rendering — render one frame into a texture, read it back,
//! and save a PNG. Used both for `--screenshot` verification and for anyone who
//! wants to script image output without a visible window.

use crate::gpu::Gpu;
use std::path::Path;

/// Read back an already-rendered texture and save it as a PNG, converting BGRA
/// to RGBA if needed.
pub fn save_texture(
    gpu: &Gpu,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let bpp = 4u32;
    let unpadded = width * bpp;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex_readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tex_readback_enc") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
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

    let mut rgba = vec![0u8; (unpadded * height) as usize];
    for y in 0..height as usize {
        let src = y * padded as usize;
        let dst = y * unpadded as usize;
        rgba[dst..dst + unpadded as usize].copy_from_slice(&data[src..src + unpadded as usize]);
    }
    drop(data);
    out_buf.unmap();

    let is_bgra = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    if is_bgra {
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }

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
