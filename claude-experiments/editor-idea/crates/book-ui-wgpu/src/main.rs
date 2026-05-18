mod geometry;
mod render;
mod scene;

use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec2, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::render::{
    EguiFrame, GlobalsUbo, Renderer, look_at, ortho, perspective, vec3_to_vec4, write_instance,
};
use crate::scene::{
    BTN_H_SIZE, BTN_W, BUTTONS, BtnColor, COLOR_CORAL, COLOR_CREAM, COLOR_OLIVE, COLOR_TAUPE,
    WidgetId, Z_BTN_FACE_PRESSED, Z_BTN_FACE_REST, btn_center, build_button_mesh,
    build_display_mesh, build_panel_mesh, build_paper_mesh, load_font,
};

struct CamCtl {
    /// In-plane rotation of the top-down view. 0 = +Y is "up" on screen.
    yaw: f32,
    dist: f32,
    target: Vec3,
}

impl CamCtl {
    fn default_pose() -> Self {
        Self {
            yaw: 0.0,
            dist: 22.0,
            target: Vec3::new(0.0, -0.4, 0.0),
        }
    }

    fn position(&self) -> Vec3 {
        self.target + Vec3::Z * self.dist
    }

    fn up(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), self.yaw.cos(), 0.0)
    }
}

struct ButtonInfo {
    id: WidgetId,
    rect_center: Vec2,
    home_z: f32,
    mesh_index: usize,
}

struct Settings {
    key_intensity: f32,
    fill_intensity: f32,
    ambient: f32,
    /// Yaw of "direction to the key light" around +Z, in radians.
    /// Reference frame: 0 = +X, π/2 = +Y. Default matches proper_mesh's
    /// `looking_to(Vec3(0.30, 0.30, -0.90))` (so to-light = (-0.30,-0.30,+0.90)).
    key_yaw: f32,
    /// Pitch above the XY horizon, in radians. 0 = horizontal,
    /// π/2 = straight overhead.
    key_pitch: f32,
    tonemap_on: bool,
    gamma_on: bool,
    shadow_bias_min: f32,
    shadow_bias_max: f32,
    key_color: [f32; 3],
    fill_color: [f32; 3],
}

impl Settings {
    fn default() -> Self {
        let to_light = Vec3::new(-0.30, -0.30, 0.90).normalize();
        let pitch = to_light.z.asin();
        let yaw = to_light.y.atan2(to_light.x);
        Self {
            key_intensity: 1.30,
            fill_intensity: 0.42,
            ambient: 0.20,
            key_yaw: yaw,
            key_pitch: pitch,
            tonemap_on: true,
            gamma_on: false,
            shadow_bias_min: 0.0002,
            shadow_bias_max: 0.0008,
            key_color: [1.00, 0.96, 0.90],
            fill_color: [0.85, 0.88, 1.00],
        }
    }

    fn key_to_light(&self) -> Vec3 {
        let cp = self.key_pitch.cos();
        let sp = self.key_pitch.sin();
        Vec3::new(cp * self.key_yaw.cos(), cp * self.key_yaw.sin(), sp).normalize()
    }
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    cam: CamCtl,
    buttons: Vec<ButtonInfo>,
    pressed: Option<WidgetId>,
    cursor: Option<PhysicalPosition<f64>>,
    last_frame: Instant,
    held_keys: HeldKeys,
    settings: Settings,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    /// True while the cursor is over an egui interactable region — we
    /// suppress 3D button picking when this is set.
    pointer_over_ui: bool,
}

#[derive(Default)]
struct HeldKeys {
    left: bool,
    right: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            cam: CamCtl::default_pose(),
            buttons: Vec::new(),
            pressed: None,
            cursor: None,
            last_frame: Instant::now(),
            held_keys: HeldKeys::default(),
            settings: Settings::default(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            pointer_over_ui: false,
        }
    }

    fn init_scene(&mut self) {
        let renderer = self.renderer.as_mut().expect("renderer must be initialised");
        let font = load_font();

        // Paper, panel, display.
        let (verts, idx) = build_paper_mesh();
        renderer
            .meshes
            .push(renderer.upload_mesh(&verts, &idx, Mat4::IDENTITY, COLOR_CREAM));
        let (verts, idx) = build_panel_mesh();
        renderer
            .meshes
            .push(renderer.upload_mesh(&verts, &idx, Mat4::IDENTITY, COLOR_TAUPE));
        let (verts, idx) = build_display_mesh(&font, "0");
        renderer
            .meshes
            .push(renderer.upload_mesh(&verts, &idx, Mat4::IDENTITY, COLOR_OLIVE));

        // Buttons.
        for spec in BUTTONS {
            let (verts, idx) = build_button_mesh(&font, spec);
            let color = match spec.color {
                BtnColor::Cream => COLOR_CREAM,
                BtnColor::Coral => COLOR_CORAL,
            };
            let center = btn_center(spec);
            let model = Mat4::from_translation(Vec3::new(center.x, center.y, 0.0));
            let mesh_index = renderer.meshes.len();
            renderer.meshes.push(renderer.upload_mesh(&verts, &idx, model, color));
            self.buttons.push(ButtonInfo {
                id: spec.id,
                rect_center: center,
                home_z: 0.0,
                mesh_index,
            });
        }
    }

    fn update_camera_uniforms(&self) {
        let renderer = self.renderer.as_ref().expect("renderer");
        let aspect = renderer.config.width as f32 / renderer.config.height.max(1) as f32;
        let proj = perspective(32f32.to_radians(), aspect, 0.1, 100.0);
        let eye = self.cam.position();
        let view = look_at(eye, self.cam.target, self.cam.up());
        let view_proj = proj * view;

        let key_to_light = self.settings.key_to_light();
        let key_shine = -key_to_light;
        let light_eye = Vec3::ZERO - key_shine * 30.0;
        let light_view = look_at(light_eye, Vec3::ZERO, Vec3::Z);
        let light_proj = ortho(-15.0, 15.0, -15.0, 15.0, 0.05, 80.0);
        let light_view_proj = light_proj * light_view;

        // Fill light direction is fixed (it's a cosmetic accent —
        // exposing it as another full slider set is more clutter than
        // value; can revisit).
        let fill_shine = Vec3::new(-0.40, 0.20, -0.85).normalize();
        let fill_to_light = -fill_shine;

        let s = &self.settings;
        let globals = GlobalsUbo {
            view_proj: view_proj.to_cols_array_2d(),
            light_view_proj: light_view_proj.to_cols_array_2d(),
            camera_pos: vec3_to_vec4(eye, 1.0).to_array(),
            key_light_dir: vec3_to_vec4(key_to_light, 0.0).to_array(),
            key_light_color: Vec4::new(
                s.key_color[0],
                s.key_color[1],
                s.key_color[2],
                s.key_intensity,
            )
            .to_array(),
            fill_light_dir: vec3_to_vec4(fill_to_light, 0.0).to_array(),
            fill_light_color: Vec4::new(
                s.fill_color[0],
                s.fill_color[1],
                s.fill_color[2],
                s.fill_intensity,
            )
            .to_array(),
            ambient: Vec4::new(1.0, 1.0, 1.0, s.ambient).to_array(),
            flags: [
                if s.tonemap_on { 1.0 } else { 0.0 },
                if s.gamma_on { 1.0 } else { 0.0 },
                s.shadow_bias_min,
                s.shadow_bias_max,
            ],
        };
        renderer.update_globals(&globals);
    }

    fn animate_buttons(&mut self, dt: f32) {
        const TWEEN_TAU: f32 = 0.06;
        let alpha = 1.0 - (-dt / TWEEN_TAU).exp();
        let press_drop = Z_BTN_FACE_PRESSED - Z_BTN_FACE_REST;
        let renderer = self.renderer.as_mut().expect("renderer");
        let queue = renderer.queue.clone();
        for btn in &self.buttons {
            let target_dz = if Some(btn.id) == self.pressed {
                btn.home_z + press_drop
            } else {
                btn.home_z
            };
            let mesh = &mut renderer.meshes[btn.mesh_index];
            let cur = mesh.model.w_axis.z;
            let mut next = cur + (target_dz - cur) * alpha;
            if (next - target_dz).abs() < 1e-4 {
                next = target_dz;
            }
            if (next - cur).abs() > 1e-7 {
                let new_model = Mat4::from_translation(Vec3::new(
                    btn.rect_center.x,
                    btn.rect_center.y,
                    next,
                ));
                write_instance(&queue, mesh, new_model);
            }
        }
    }

    /// Unproject the cursor onto the plane Z = Z_BTN_FACE_REST.
    fn cursor_world_xy(&self) -> Option<Vec2> {
        let renderer = self.renderer.as_ref()?;
        let cursor = self.cursor?;
        let w = renderer.config.width as f32;
        let h = renderer.config.height as f32;
        if w == 0.0 || h == 0.0 {
            return None;
        }
        let ndc_x = (cursor.x as f32 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor.y as f32 / h) * 2.0;
        let aspect = w / h;
        let proj = perspective(32f32.to_radians(), aspect, 0.1, 100.0);
        let view = look_at(self.cam.position(), self.cam.target, self.cam.up());
        let view_proj = proj * view;
        let inv = view_proj.inverse();
        let near = inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inv * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        let dir = (far - near).normalize();
        if dir.z.abs() < 1e-6 {
            return None;
        }
        let t = (Z_BTN_FACE_REST - near.z) / dir.z;
        if t < 0.0 {
            return None;
        }
        let p = near + dir * t;
        Some(Vec2::new(p.x, p.y))
    }

    fn pick_button(&self) -> Option<WidgetId> {
        let world = self.cursor_world_xy()?;
        let half_w = BTN_W * 0.5;
        let half_h = BTN_H_SIZE * 0.5;
        for btn in &self.buttons {
            let d = world - btn.rect_center;
            if d.x.abs() <= half_w && d.y.abs() <= half_h {
                return Some(btn.id);
            }
        }
        None
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("proper mesh calculator (wgpu)")
            .with_inner_size(winit::dpi::LogicalSize::new(1200, 1000));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());
        let renderer = pollster::block_on(Renderer::new(window.clone()));
        self.renderer = Some(renderer);
        self.init_scene();

        // egui input plumbing for this window.
        let viewport_id = self.egui_ctx.viewport_id();
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            viewport_id,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        ));

        self.last_frame = Instant::now();
        window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        // Hand the event to egui first; if it consumed it, skip our
        // own handling for the same input categories.
        let egui_consumed = if let (Some(window), Some(state)) =
            (self.window.as_ref(), self.egui_state.as_mut())
        {
            let r = state.on_window_event(window, &event);
            if r.repaint {
                window.request_redraw();
            }
            r.consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some(position);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. }
                if !egui_consumed && !self.pointer_over_ui =>
            {
                match state {
                    ElementState::Pressed => {
                        self.pressed = self.pick_button();
                    }
                    ElementState::Released => {
                        self.pressed = None;
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left, .. } => {
                // Always clear pressed on release so a click that
                // started over the UI can't strand a button down.
                self.pressed = None;
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed && !self.pointer_over_ui => {
                let n = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                if n.abs() > 1e-4 {
                    let factor = (1.0 - n * 0.05).clamp(0.92, 1.08);
                    self.cam.dist = (self.cam.dist * factor).clamp(4.0, 60.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.held_keys.left = pressed,
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.held_keys.right = pressed,
                    PhysicalKey::Code(KeyCode::KeyR) if pressed && !event.repeat => {
                        self.cam = CamCtl::default_pose();
                    }
                    PhysicalKey::Code(KeyCode::Escape) if pressed => event_loop.exit(),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                // Continuous camera input.
                if self.held_keys.left { self.cam.yaw -= 1.6 * dt; }
                if self.held_keys.right { self.cam.yaw += 1.6 * dt; }

                let window = self.window.clone().expect("window");
                let pixels_per_point = window.scale_factor() as f32;

                let raw_input = {
                    let state = self.egui_state.as_mut().expect("egui state");
                    state.take_egui_input(&window)
                };

                // Cloning the Context drops the borrow on self.egui_ctx
                // before the closure runs, leaving &mut self.settings
                // free to use inside the UI build.
                let ctx = self.egui_ctx.clone();
                let s = &mut self.settings;
                let full_output = ctx.run(raw_input, |ctx| {
                    egui::Window::new("lighting")
                        .default_pos([10.0, 10.0])
                        .show(ctx, |ui| {
                            ui.collapsing("intensities", |ui| {
                                ui.add(egui::Slider::new(&mut s.key_intensity, 0.0..=3.0).text("key"));
                                ui.add(egui::Slider::new(&mut s.fill_intensity, 0.0..=2.0).text("fill"));
                                ui.add(egui::Slider::new(&mut s.ambient, 0.0..=1.0).text("ambient"));
                            });
                            ui.collapsing("key light direction", |ui| {
                                ui.add(egui::Slider::new(&mut s.key_yaw, -std::f32::consts::PI..=std::f32::consts::PI).text("yaw (rad)"));
                                ui.add(egui::Slider::new(&mut s.key_pitch, 0.0..=(std::f32::consts::FRAC_PI_2 - 0.01)).text("pitch (rad)"));
                            });
                            ui.collapsing("colors", |ui| {
                                ui.horizontal(|ui| { ui.label("key"); ui.color_edit_button_rgb(&mut s.key_color); });
                                ui.horizontal(|ui| { ui.label("fill"); ui.color_edit_button_rgb(&mut s.fill_color); });
                            });
                            ui.collapsing("output", |ui| {
                                ui.checkbox(&mut s.tonemap_on, "Reinhard tonemap");
                                ui.checkbox(&mut s.gamma_on, "gamma (1/2.2)");
                                ui.label("(surface is sRGB; gamma should usually be OFF)");
                            });
                            ui.collapsing("shadow", |ui| {
                                ui.add(egui::Slider::new(&mut s.shadow_bias_min, 0.0..=0.005).text("bias min").logarithmic(true));
                                ui.add(egui::Slider::new(&mut s.shadow_bias_max, 0.0..=0.02).text("bias max").logarithmic(true));
                            });
                            if ui.button("reset").clicked() {
                                *s = Settings::default();
                            }
                        });
                });

                {
                    let state = self.egui_state.as_mut().expect("egui state");
                    state.handle_platform_output(&window, full_output.platform_output.clone());
                }
                self.pointer_over_ui = self.egui_ctx.is_pointer_over_area()
                    || self.egui_ctx.wants_pointer_input();

                self.update_camera_uniforms();
                self.animate_buttons(dt);

                let paint_jobs = self
                    .egui_ctx
                    .tessellate(full_output.shapes, pixels_per_point);

                if let Some(r) = self.renderer.as_mut() {
                    let eg = EguiFrame {
                        paint_jobs: &paint_jobs,
                        textures_delta: &full_output.textures_delta,
                        pixels_per_point,
                    };
                    match r.render(Some(eg)) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let w = r.config.width;
                            let h = r.config.height;
                            r.resize(w, h);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("render error: {e:?}"),
                    }
                }

                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}

