mod scene;
mod effects;
mod renderer;
mod input;

use std::sync::Arc;
use glam::{Mat4, Vec2, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use scene::*;
use effects::*;
use renderer::GpuRenderer;

// Surface IDs
const WOBBLY_ID: SurfaceId = SurfaceId(1);
const SLIDER_IDS: [(SurfaceId, SurfaceId); 3] = [
    (SurfaceId(100), SurfaceId(101)), // stiffness track, handle
    (SurfaceId(200), SurfaceId(201)), // damping track, handle
    (SurfaceId(300), SurfaceId(301)), // grid size track, handle
];

// Slider layout
const SLIDER_Y: f32 = -0.9;
const SLIDER_Z: f32 = 1.0;
const SLIDER_WIDTH: f32 = 1.2;
const SLIDER_HEIGHT: f32 = 0.06;
const HANDLE_WIDTH: f32 = 0.08;
const HANDLE_HEIGHT: f32 = 0.14;
const SLIDER_X: [f32; 3] = [-1.5, 0.0, 1.5];

// Parameter ranges: (min, max)
const PARAM_RANGES: [(f32, f32); 3] = [
    (20.0, 500.0),   // stiffness
    (1.0, 20.0),     // damping
    (3.0, 12.0),     // grid size
];

enum DragKind {
    Surface {
        surface_id: SurfaceId,
        node_index: usize,
        prev_world: Vec3,
        total_delta: Vec3,
    },
    Slider(usize), // which slider (0, 1, 2)
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<GpuRenderer>,
    scene: Scene,
    wobbly_mgr: WobblyManager,
    last_frame: std::time::Instant,
    time: f32,
    mouse_pos: Vec2,
    drag: Option<DragKind>,
    params: [f32; 3], // stiffness, damping, grid_size
}

impl App {
    fn new() -> Self {
        let params = [180.0, 7.0, 6.0];
        Self {
            window: None,
            renderer: None,
            scene: Self::build_scene(&params),
            wobbly_mgr: WobblyManager::new(),
            last_frame: std::time::Instant::now(),
            time: 0.0,
            mouse_pos: Vec2::ZERO,
            drag: None,
            params,
        }
    }

    fn view_proj(&self) -> Mat4 {
        let r = self.renderer.as_ref().unwrap();
        let aspect = r.config.width as f32 / r.config.height as f32;
        let view = Mat4::look_at_rh(
            self.scene.camera_position,
            self.scene.camera_target,
            Vec3::Y,
        );
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        proj * view
    }

    fn screen_size(&self) -> Vec2 {
        let r = self.renderer.as_ref().unwrap();
        Vec2::new(r.config.width as f32, r.config.height as f32)
    }

    fn screen_to_world(&self, screen_pos: Vec2, ref_world: Vec3) -> Vec3 {
        let inv_vp = self.view_proj().inverse();
        let size = self.screen_size();
        let ndc_x = (screen_pos.x / size.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.y / size.y) * 2.0;

        let near = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        let ray_dir = (far - near).normalize();

        let cam_dir = (self.scene.camera_target - self.scene.camera_position).normalize();
        let n = -cam_dir;
        let d = n.dot(ref_world);
        let denom = n.dot(ray_dir);
        if denom.abs() < 1e-6 {
            return ref_world;
        }
        near + ray_dir * ((d - n.dot(near)) / denom)
    }

    fn handle_x(slider_idx: usize, norm: f32) -> f32 {
        SLIDER_X[slider_idx] - SLIDER_WIDTH / 2.0 + norm * SLIDER_WIDTH
    }

    fn param_to_norm(idx: usize, value: f32) -> f32 {
        let (min, max) = PARAM_RANGES[idx];
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    fn norm_to_param(idx: usize, norm: f32) -> f32 {
        let (min, max) = PARAM_RANGES[idx];
        min + norm.clamp(0.0, 1.0) * (max - min)
    }

    fn world_x_to_norm(world_x: f32, slider_idx: usize) -> f32 {
        let left = SLIDER_X[slider_idx] - SLIDER_WIDTH / 2.0;
        ((world_x - left) / SLIDER_WIDTH).clamp(0.0, 1.0)
    }

    fn update_handles(&mut self) {
        // Handle nodes are at children indices 2, 4, 6
        for i in 0..3 {
            let norm = Self::param_to_norm(i, self.params[i]);
            self.scene.root.children[2 + i * 2].transform.translation.x =
                Self::handle_x(i, norm);
        }
    }

    fn update_wobbly_effect(&mut self) {
        self.scene.root.children[0].effects = vec![
            Effect::Wobbly(WobblyParams {
                stiffness: self.params[0],
                damping: self.params[1],
                grid_size: self.params[2].round() as usize,
            })
        ];
    }

    fn slider_index_for_id(&self, id: SurfaceId) -> Option<usize> {
        for (i, (track, handle)) in SLIDER_IDS.iter().enumerate() {
            if id == *track || id == *handle {
                return Some(i);
            }
        }
        None
    }

    fn update_slider(&mut self, slider_idx: usize) {
        let world = self.screen_to_world(
            self.mouse_pos,
            Vec3::new(SLIDER_X[slider_idx], SLIDER_Y, SLIDER_Z),
        );
        let norm = Self::world_x_to_norm(world.x, slider_idx);
        self.params[slider_idx] = Self::norm_to_param(slider_idx, norm);
        // Snap grid size to integer
        if slider_idx == 2 {
            self.params[2] = self.params[2].round();
        }
        self.update_handles();
        self.update_wobbly_effect();
    }

    fn on_mouse_down(&mut self) {
        if self.renderer.is_none() {
            return;
        }
        let vp = self.view_proj();
        let size = self.screen_size();

        if let Some(hit) = input::hit_test(&self.scene, self.mouse_pos, size, vp) {
            if let Some(slider_idx) = self.slider_index_for_id(hit.surface_id) {
                self.drag = Some(DragKind::Slider(slider_idx));
                self.update_slider(slider_idx);
            } else {
                let surface = self.scene.root.children[hit.node_index]
                    .surface.as_ref().unwrap();
                let grid_size = self.params[2].round() as usize;
                let sim = self.wobbly_mgr.ensure(
                    hit.surface_id, surface.size.x, surface.size.y, grid_size,
                );
                sim.grab(hit.uv.x, hit.uv.y);
                self.drag = Some(DragKind::Surface {
                    surface_id: hit.surface_id,
                    node_index: hit.node_index,
                    prev_world: hit.world_point,
                    total_delta: Vec3::ZERO,
                });
            }
        }
    }

    fn on_mouse_move(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        match self.drag {
            Some(DragKind::Slider(idx)) => {
                self.update_slider(idx);
            }
            Some(DragKind::Surface { prev_world, surface_id, .. }) => {
                let current_world = self.screen_to_world(self.mouse_pos, prev_world);
                let frame_delta = current_world - prev_world;

                if let DragKind::Surface { prev_world: pw, total_delta: td, .. } =
                    self.drag.as_mut().unwrap()
                {
                    *pw = current_world;
                    *td += frame_delta;
                }

                if let Some(sim) = self.wobbly_mgr.get_mut(surface_id) {
                    sim.drag(frame_delta.x, -frame_delta.y);
                }
            }
            None => {}
        }
    }

    fn on_mouse_up(&mut self) {
        if let Some(DragKind::Surface { node_index, total_delta, surface_id, .. }) =
            self.drag.take()
        {
            let node = &mut self.scene.root.children[node_index];
            node.transform.translation += total_delta;

            if let Some(sim) = self.wobbly_mgr.get_mut(surface_id) {
                sim.shift_positions(total_delta.x, -total_delta.y);
                sim.release();
            }
        } else {
            self.drag = None;
        }
    }

    fn build_scene(params: &[f32; 3]) -> Scene {
        let mut scene = Scene::default();

        // Child 0: Wobbly window
        let wobbly_surface = Surface::deformable(WOBBLY_ID, Vec2::new(2.0, 1.5), 20)
            .with_material(Material {
                base_color: Vec4::new(0.15, 0.2, 0.35, 0.85),
                roughness: 0.1,
                metallic: 0.0,
                opacity: 0.85,
                shadow: true,
                reflective: true,
                emissive: Vec4::ZERO,
            });
        let wobbly_node = SceneNode::with_surface(wobbly_surface)
            .at(Vec3::new(0.0, 0.5, 0.0))
            .with_effect(Effect::Wobbly(WobblyParams {
                stiffness: params[0],
                damping: params[1],
                grid_size: params[2].round() as usize,
            }));

        let track_mat = Material {
            base_color: Vec4::new(0.3, 0.3, 0.35, 0.9),
            roughness: 0.6,
            metallic: 0.0,
            opacity: 0.9,
            shadow: false,
            reflective: false,
            emissive: Vec4::ZERO,
        };
        let handle_mat = Material {
            base_color: Vec4::new(0.9, 0.9, 0.95, 1.0),
            roughness: 0.3,
            metallic: 0.2,
            opacity: 1.0,
            shadow: false,
            reflective: false,
            emissive: Vec4::new(0.4, 0.5, 0.8, 0.5),
        };

        let mut root = SceneNode::empty().with_child(wobbly_node);

        // Add 3 sliders: children 1-6 (track, handle, track, handle, track, handle)
        for i in 0..3 {
            let norm = Self::param_to_norm(i, params[i]);
            let (track_id, handle_id) = SLIDER_IDS[i];

            let track = Surface::new(track_id, Vec2::new(SLIDER_WIDTH, SLIDER_HEIGHT))
                .with_material(track_mat.clone());
            let track_node = SceneNode::with_surface(track)
                .at(Vec3::new(SLIDER_X[i], SLIDER_Y, SLIDER_Z));

            let handle = Surface::new(handle_id, Vec2::new(HANDLE_WIDTH, HANDLE_HEIGHT))
                .with_material(handle_mat.clone());
            let handle_node = SceneNode::with_surface(handle)
                .at(Vec3::new(Self::handle_x(i, norm), SLIDER_Y, SLIDER_Z + 0.01));

            root = root.with_child(track_node).with_child(handle_node);
        }

        scene.root = root;

        scene.lights = vec![
            PointLight {
                position: Vec3::new(3.0, 3.0, 4.0),
                color: Vec3::new(1.0, 0.9, 0.8),
                intensity: 2.0,
                radius: 15.0,
            },
            PointLight {
                position: Vec3::new(-3.0, 2.0, 2.0),
                color: Vec3::new(0.3, 0.5, 1.0),
                intensity: 1.5,
                radius: 12.0,
            },
            PointLight {
                position: Vec3::new(0.0, -0.5, 3.0),
                color: Vec3::new(0.8, 0.3, 0.6),
                intensity: 0.8,
                radius: 8.0,
            },
        ];

        scene.ambient_light = Vec3::splat(0.08);
        scene.camera_position = Vec3::new(0.0, 0.5, 4.5);
        scene.camera_target = Vec3::new(0.0, 0.2, 0.0);

        scene
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("gpu-ui — stiffness | damping | grid size")
            .with_inner_size(winit::dpi::LogicalSize::new(1200, 800));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let renderer = pollster::block_on(GpuRenderer::new(window.clone()));
        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                self.on_mouse_move();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed => self.on_mouse_down(),
                    ElementState::Released => self.on_mouse_up(),
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                self.time += dt;

                effects::process_effects(&mut self.scene, &mut self.wobbly_mgr, dt);

                if let Some(light) = self.scene.lights.get_mut(1) {
                    light.position.x = -3.0 + (self.time * 0.5).sin() * 2.0;
                    light.position.y = 2.0 + (self.time * 0.3).cos() * 1.0;
                }

                if let Some(r) = &mut self.renderer {
                    match r.render(&self.scene, dt) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let s = self.window.as_ref().unwrap().inner_size();
                            r.resize(s.width, s.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
