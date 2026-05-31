//! Isolated wgpu stack used to drive the flame-graph `Renderer`.
//!
//! Bevy and `flame-render` pin different wgpu versions (Bevy 0.18 → wgpu 27,
//! flame-render → wgpu 29 via glyphon 0.11). They can't share `wgpu::Device`
//! because the `wgpu` Rust types are version-distinct. Instead, this module
//! creates its own wgpu 29 instance, renders into an offscreen texture, and
//! reads pixels back to CPU. The pixel buffer is then handed to Bevy by
//! writing into a `bevy::Image`'s data field — a copy that crosses no wgpu
//! type boundary.
//!
//! Cost: one RGBA8 readback + upload per redraw. ~8 MB at 1080p. The plugin
//! only re-renders when state changed, so steady-state idle is cheap.

use flame_render::Renderer;

/// All wgpu state needed to render one frame of flame graph into a CPU-side
/// pixel buffer.
pub struct FlameGpu {
    _instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub renderer: Renderer,
    target: RenderTarget,
    /// Current size of the rendered canvas, in pixels.
    pub size: (u32, u32),
    /// Most recent frame, RGBA8 (sRGB), tightly packed (no row stride
    /// padding). Length = `size.0 * size.1 * 4`.
    pub pixels: Vec<u8>,
}

/// Internal offscreen render-to-texture state. Recreated on resize.
struct RenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// Staging buffer for `copy_texture_to_buffer`. Sized to match the
    /// 256-byte-row-alignment requirement of `wgpu::Queue::write_buffer`.
    readback: wgpu::Buffer,
    /// Bytes-per-row in the readback buffer, padded up to a multiple of 256.
    padded_bytes_per_row: u32,
}

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

impl FlameGpu {
    pub fn new(size: (u32, u32)) -> Self {
        pollster::block_on(Self::new_async(size))
    }

    async fn new_async(size: (u32, u32)) -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("flame-bevy: no compatible wgpu adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("flame-bevy device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("flame-bevy: device request failed");

        let target = RenderTarget::new(&device, size);
        let renderer = Renderer::new(&device, &queue, TARGET_FORMAT, size);

        Self {
            _instance: instance,
            device,
            queue,
            renderer,
            target,
            size,
            pixels: vec![0; (size.0 * size.1 * 4) as usize],
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.size == (w, h) {
            return;
        }
        self.target = RenderTarget::new(&self.device, (w, h));
        self.renderer.resize(w, h);
        self.renderer.rebuild_instances();
        self.size = (w, h);
        self.pixels.resize((w * h * 4) as usize, 0);
    }

    /// Render one frame, read back the pixels into `self.pixels`. Returns the
    /// (width, height) of the buffer.
    pub fn render_and_readback(&mut self) -> (u32, u32) {
        self.renderer.render(&self.target.view);

        let (w, h) = self.size;
        let padded = self.target.padded_bytes_per_row;
        let unpadded = w * 4;

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flame-bevy readback"),
                });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map readback buffer synchronously. wgpu's map_async is the only API;
        // we drive it to completion by polling the device.
        let slice = self.target.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Poll until the map completes. Wait::Wait blocks until queue work is done.
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("flame-bevy: device poll failed");
        rx.recv()
            .expect("flame-bevy: readback channel closed")
            .expect("flame-bevy: buffer map failed");

        // Strip row padding into the tightly-packed `self.pixels`.
        {
            let mapped = slice.get_mapped_range();
            if padded == unpadded {
                self.pixels.copy_from_slice(&mapped[..]);
            } else {
                for row in 0..h {
                    let src_off = (row * padded) as usize;
                    let dst_off = (row * unpadded) as usize;
                    self.pixels[dst_off..dst_off + unpadded as usize]
                        .copy_from_slice(&mapped[src_off..src_off + unpadded as usize]);
                }
            }
        }
        self.target.readback.unmap();

        (w, h)
    }
}

impl RenderTarget {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("flame-bevy rtt"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let unpadded = w * 4;
        let padded = unpadded.div_ceil(ALIGN) * ALIGN;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flame-bevy readback"),
            size: (padded as u64) * (h as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            texture,
            view,
            readback,
            padded_bytes_per_row: padded,
        }
    }
}
