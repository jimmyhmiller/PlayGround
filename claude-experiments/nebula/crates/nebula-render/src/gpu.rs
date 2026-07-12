//! wgpu device/surface bring-up. We deliberately request the adapter's *full*
//! limits (not the conservative defaults) so we can allocate the multi-gigabyte
//! storage buffers a large graph needs — the default
//! `max_storage_buffer_binding_size` of 128 MiB would cap us at a few million
//! nodes.

use anyhow::{anyhow, Result};
use std::sync::Arc;
use winit::window::Window;

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'static>,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

impl Gpu {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let size = winit::dpi::PhysicalSize::new(size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow!("no suitable GPU adapter: {e}"))?;

        let info = adapter.get_info();
        log::info!("GPU: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // Ask for everything the hardware allows.
        let adapter_limits = adapter.limits();
        log::info!(
            "limits: max_storage_buffer={} MiB, max_buffer={} MiB, max_compute_invocations={}",
            adapter_limits.max_storage_buffer_binding_size / (1 << 20),
            adapter_limits.max_buffer_size / (1 << 20),
            adapter_limits.max_compute_invocations_per_workgroup,
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nebula-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow!("request_device failed: {e}"))?;

        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB-view we manage ourselves; use the first (usually
        // preferred) format. Rendering assumes linear-ish; good enough for a viewer.
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Gpu {
            instance,
            surface,
            adapter,
            device,
            queue,
            config,
            size,
        })
    }

    pub fn resize(&mut self, new: winit::dpi::PhysicalSize<u32>) {
        if new.width > 0 && new.height > 0 {
            self.size = new;
            self.config.width = new.width;
            self.config.height = new.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
