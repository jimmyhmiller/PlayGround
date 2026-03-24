use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::scene::{PointLight, Scene};
use super::mesh::{generate_surface_mesh, Vertex};
use super::pipeline::create_surface_pipeline;

const MAX_LIGHTS: usize = 8;

/// Uniform data sent to the GPU per-frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    ambient_light: [f32; 4],
    lights: [GpuLight; MAX_LIGHTS],
    num_lights: u32,
    time: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position: [f32; 4],
    color_intensity: [f32; 4], // RGB + intensity
    radius: [f32; 4],          // radius, padding...
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            color_intensity: [0.0; 4],
            radius: [0.0; 4],
        }
    }
}

/// Uniform data sent per-surface.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceUniforms {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    emissive: [f32; 4],
    roughness: f32,
    metallic: f32,
    opacity: f32,
    _pad: f32,
}

/// The GPU renderer. Owns all wgpu state and renders a Scene each frame.
pub struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    global_bind_group_layout: wgpu::BindGroupLayout,
    surface_bind_group_layout: wgpu::BindGroupLayout,
    global_uniform_buffer: wgpu::Buffer,
    depth_texture: wgpu::TextureView,
    time: f32,
}

impl GpuRenderer {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpu-ui device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Bind group layouts
        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("global_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let surface_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("surface_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/surface.wgsl").into()),
        });

        let pipeline = create_surface_pipeline(
            &device,
            format,
            &shader,
            &[&global_bind_group_layout, &surface_bind_group_layout],
        );

        let global_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global_uniforms"),
            size: std::mem::size_of::<GlobalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_texture = Self::create_depth_texture(&device, &config);

        Self {
            device,
            queue,
            surface,
            config,
            pipeline,
            global_bind_group_layout,
            surface_bind_group_layout,
            global_uniform_buffer,
            depth_texture,
            time: 0.0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
        }
    }

    pub fn render(&mut self, scene: &Scene, dt: f32) -> Result<(), wgpu::SurfaceError> {
        self.time += dt;

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&Default::default());

        // Build global uniforms
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view_matrix = Mat4::look_at_rh(
            scene.camera_position,
            scene.camera_target,
            Vec3::Y,
        );
        let proj_matrix = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            100.0,
        );
        let view_proj = proj_matrix * view_matrix;

        let mut gpu_lights = [GpuLight::default(); MAX_LIGHTS];
        for (i, light) in scene.lights.iter().take(MAX_LIGHTS).enumerate() {
            gpu_lights[i] = GpuLight {
                position: [light.position.x, light.position.y, light.position.z, 1.0],
                color_intensity: [light.color.x, light.color.y, light.color.z, light.intensity],
                radius: [light.radius, 0.0, 0.0, 0.0],
            };
        }

        let globals = GlobalUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [
                scene.camera_position.x,
                scene.camera_position.y,
                scene.camera_position.z,
                0.0,
            ],
            ambient_light: [
                scene.ambient_light.x,
                scene.ambient_light.y,
                scene.ambient_light.z,
                0.0,
            ],
            lights: gpu_lights,
            num_lights: scene.lights.len().min(MAX_LIGHTS) as u32,
            time: self.time,
            _pad: [0.0; 2],
        };

        self.queue.write_buffer(
            &self.global_uniform_buffer,
            0,
            bytemuck::cast_slice(&[globals]),
        );

        let global_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global_bind_group"),
            layout: &self.global_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.global_uniform_buffer.as_entire_binding(),
            }],
        });

        // Collect all surfaces to draw
        let surfaces = scene.collect_surfaces();

        // Build per-surface GPU data
        struct DrawCall {
            vertex_buffer: wgpu::Buffer,
            index_buffer: wgpu::Buffer,
            index_count: u32,
            bind_group: wgpu::BindGroup,
        }

        let mut draw_calls = Vec::with_capacity(surfaces.len());

        for (world_transform, surface) in &surfaces {
            let (vertices, indices) = generate_surface_mesh(surface);

            let vertex_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("surface_vb"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            let index_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("surface_ib"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });

            let uniforms = SurfaceUniforms {
                model: world_transform.to_cols_array_2d(),
                base_color: surface.material.base_color.into(),
                emissive: surface.material.emissive.into(),
                roughness: surface.material.roughness,
                metallic: surface.material.metallic,
                opacity: surface.material.opacity,
                _pad: 0.0,
            };

            let uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("surface_ub"),
                        contents: bytemuck::cast_slice(&[uniforms]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface_bind_group"),
                layout: &self.surface_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            draw_calls.push(DrawCall {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
                bind_group,
            });
        }

        // Encode render pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &global_bind_group, &[]);

            for dc in &draw_calls {
                pass.set_bind_group(1, &dc.bind_group, &[]);
                pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..dc.index_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&Default::default())
    }
}
