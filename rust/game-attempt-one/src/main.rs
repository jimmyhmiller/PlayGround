use std::{borrow::Cow, future::Future};

use wgpu::{util::DeviceExt, Adapter, Device, Instance, Queue, RenderPipeline, Surface, SurfaceConfiguration};
use winit::{
    application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent, event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy}, keyboard::SmolStr, window::{Window, WindowId}
};


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}





type Rc<T> = std::sync::Arc<T>;

fn create_graphics(event_loop: &ActiveEventLoop) -> impl Future<Output = Graphics> + 'static {
    #[allow(unused_mut)]
    let mut window_attrs = Window::default_attributes();

    let window = Rc::new(event_loop.create_window(window_attrs).unwrap());
    let instance = wgpu::Instance::default();
    let surface = instance
        .create_surface(window.clone())
        .unwrap();

    async move {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits:wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let surface_config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();

        surface.configure(&device, &surface_config);

        let render_pipeline = create_render_pipeline(&device, &surface, &adapter);

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: &[0; 256],
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: &[0; 256],
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        Graphics {
            window,
            instance,
            surface,
            surface_config,
            adapter,
            device,
            queue,
            render_pipeline,
            vertex_buffer,
            index_buffer,
        }
    }
}

fn create_render_pipeline(device: &Device, surface: &Surface, adapter: &Adapter) -> RenderPipeline {

    let swapchain_capabilities = surface.get_capabilities(adapter);
    let swapchain_format = swapchain_capabilities.formats[0];
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed("
                struct VertexInput {
                    @location(0) position: vec3<f32>,
                    @location(1) color: vec3<f32>,
                };

                struct VertexOutput {
                    @builtin(position) clip_position: vec4<f32>,
                    @location(0) color: vec3<f32>,
                };

                @vertex
                fn vs_main(
                    model: VertexInput,
                ) -> VertexOutput {
                    var out: VertexOutput;
                    out.color = model.color;
                    out.clip_position = vec4<f32>(model.position, 1.0);
                    return out;
                }

                // Fragment shader

                @fragment
                fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                    return vec4<f32>(in.color, 1.0);
                }
            ")),
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                Vertex::desc(),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    render_pipeline
}

#[allow(dead_code)]
struct Graphics {
    window: Rc<Window>,
    instance: Instance,
    surface: Surface<'static>,
    surface_config: SurfaceConfiguration,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    render_pipeline: RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

#[derive(Debug)]
enum Event {
}

enum Direction {
    Clockwise,
    CounterClockwise,
}

struct State {
    rotation_direction: Direction,
    speed: f32,
}

struct Application {
    graphics: Option<Graphics>,
    event_loop_proxy: EventLoopProxy<Event>,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    state: State,
}

impl Application {
    fn new(event_loop: &EventLoop<Event>) -> Self {
        Self {
            graphics: None,
            event_loop_proxy: event_loop.create_proxy(),
            vertices:  vec![
                Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [1.0, 0.0, 0.0] },
                Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [0.5, 0.0, 1.0] },
                Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.0, 0.0, 1.0] },
                Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.0, 1.0, 0.0] },
                Vertex { position: [0.44147372, 0.2347359, 0.0], color: [1.0, 1.0, 0.0] },
            ],
            indices: vec![
                0, 1, 4,
                1, 2, 4,
                2, 3, 4,
            ],
            state: State {
                rotation_direction: Direction::Clockwise,
                speed: 1.0,
            },
        }
    }

    fn tick(&mut self) {
        // rotate our pentagram 1 degree per frame based on the direction
        let angle: f32 = match self.state.rotation_direction {
            Direction::Clockwise => 1.0,
            Direction::CounterClockwise => -1.0,
        };
        let speed = self.state.speed;
        let angle = angle * speed;
        let (sin, cos) = angle.to_radians().sin_cos();
        for vertex in &mut self.vertices {
            let x = vertex.position[0];
            let y = vertex.position[1];
            vertex.position[0] = x * cos - y * sin;
            vertex.position[1] = y * cos + x * sin;
        }
    }

    fn draw(&mut self) {
        let Some(gfx) = &mut self.graphics else {
            // draw call rejected because graphics doesn't exist yet
            return;
        };
        let padded_size = (self.vertices.len() + 3) & !3; // Align to 4 bytes
        let mut padded_data = self.vertices.clone();
        padded_data.resize(padded_size, Vertex { position: [0.0, 0.0, 0.0], color: [0.0, 0.0, 0.0] });

        gfx.queue.write_buffer(&gfx.vertex_buffer, 0, bytemuck::cast_slice(&padded_data));

        let padded_size = (self.indices.len() + 3) & !3; // Align to 4 bytes
        let mut padded_data = self.indices.clone();
        padded_data.resize(padded_size, 0);
        gfx.queue.write_buffer(&gfx.index_buffer, 0, bytemuck::cast_slice(&padded_data));

        let frame = gfx.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = gfx.device.create_command_encoder(&Default::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            render_pass.set_pipeline(&gfx.render_pipeline);
            
            render_pass.set_vertex_buffer(0, gfx.vertex_buffer.slice(0..(self.vertices.len() * std::mem::size_of::<Vertex>()) as u64));
            render_pass.set_index_buffer(gfx.index_buffer.slice(0..(self.indices.len() * std::mem::size_of::<u16>()) as u64), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.indices.len() as u32, 0, 0..1); // 2.
        }

        let command_buffer = encoder.finish();
        gfx.queue.submit([command_buffer]);
        frame.present();
    }

    fn resized(&mut self, size: PhysicalSize<u32>) {
        let Some(gfx) = &mut self.graphics else {
            return;
        };
        gfx.surface_config.width = size.width;
        gfx.surface_config.height = size.height;
        gfx.surface.configure(&gfx.device, &gfx.surface_config);
    }
}

impl ApplicationHandler<Event> for Application {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => self.resized(size),
            WindowEvent::RedrawRequested => {
                self.tick();
                self.draw();
                self.graphics.as_ref().unwrap().window.request_redraw();
            },
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { device_id: _, event, is_synthetic: _ } => {
                match event.logical_key {
                    winit::keyboard::Key::Named(named_key) => {
                        match named_key {
                            winit::keyboard::NamedKey::ArrowLeft => {
                                self.state.rotation_direction = Direction::Clockwise;
                            },
                            winit::keyboard::NamedKey::ArrowRight => {
                                self.state.rotation_direction = Direction::CounterClockwise;
                            },
                            winit::keyboard::NamedKey::ArrowUp => {
                                self.state.speed += 0.1;
                            },
                            winit::keyboard::NamedKey::ArrowDown => {
                                self.state.speed -= 0.1;
                            },
                            _ => {}
                        }
                    }
                    winit::keyboard::Key::Character(k) => {
                        match k.as_str() {
                            "s" => {
                                self.state.speed = 0.0;
                            },
                            _ => {}
                        }
                        
                    }
                    _ => {}
                   
                }
            }
            _ => (),
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_none() {
            let gfx = pollster::block_on(create_graphics(event_loop));
            self.graphics = Some(gfx);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: Event) {
    }
}

pub fn run() {
    let event_loop = EventLoop::with_user_event().build().unwrap();
    let mut app = Application::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}

fn main() {
    run();
}