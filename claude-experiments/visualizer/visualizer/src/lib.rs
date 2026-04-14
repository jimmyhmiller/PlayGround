#[allow(dead_code)]
mod anim;
#[allow(dead_code)]
mod animated;
#[allow(dead_code)]
mod collection;
mod dsl;
#[allow(dead_code)]
mod hashmap_viz;
#[allow(dead_code)]
mod scene;
#[allow(dead_code)]
mod scene_demo;
#[allow(dead_code)]
mod state;
mod theme;
mod tweakables;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use dsl::runtime::Program;
use tweakables::Tweakables;
use vello::kurbo::{Affine, Rect};
use vello::peniko::{Color, Fill};
use vello::util::{RenderContext, RenderSurface};
use vello::wgpu;
use vello::{AaConfig, Renderer, RendererOptions, RenderParams, Scene};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::event_loop::EventLoop;
use winit::window::Window;

use wasm_bindgen::prelude::*;

struct App<'s> {
    context: RenderContext,
    renderers: Vec<Option<Renderer>>,
    surface: Option<RenderSurface<'s>>,
    window: Arc<Window>,
    scene: Scene,
    program: Rc<RefCell<Option<Program>>>,
    tweaks: Rc<RefCell<Tweakables>>,
    last_instant: Option<web_time::Instant>,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(surface) = &mut self.surface {
                    self.context
                        .resize_surface(surface, size.width.max(1), size.height.max(1));
                }
                self.window.request_redraw();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(prog) = self.program.borrow_mut().as_mut() {
                    prog.handle_click(&self.tweaks.borrow());
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(prog) = self.program.borrow_mut().as_mut() {
                    prog.set_mouse_pos(position.x, position.y);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let Some(prog) = self.program.borrow_mut().as_mut() {
                    match &event.logical_key {
                        Key::Named(NamedKey::ArrowLeft) => prog.step_back(),
                        Key::Named(NamedKey::ArrowRight) => prog.handle_click(&self.tweaks.borrow()),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = web_time::Instant::now();
                let dt = self
                    .last_instant
                    .map(|prev| now.duration_since(prev).as_secs_f64())
                    .unwrap_or(1.0 / 60.0)
                    .min(0.1);
                self.last_instant = Some(now);

                if let Some(prog) = self.program.borrow_mut().as_mut() {
                    prog.tick(dt, &self.tweaks.borrow());
                }
                self.render();
                self.window.request_redraw();
            }
            _ => {}
        }
    }
}

impl App<'_> {
    fn render(&mut self) {
        let Some(surface) = &self.surface else { return };
        let width = surface.config.width;
        let height = surface.config.height;
        if width == 0 || height == 0 { return; }

        let dev_id = surface.dev_id;
        let device = &self.context.devices[dev_id].device;
        let queue = &self.context.devices[dev_id].queue;

        self.scene.reset();
        if let Some(prog) = self.program.borrow().as_ref() {
            prog.draw(&mut self.scene, &self.tweaks.borrow());
        }

        let theme = theme::current();
        let [br, bg, bb, ba] = theme.background;
        let base_color = Color::new([br, bg, bb, ba]);

        // Grain overlay — pre-baked seamless tile stretched over the viewport.
        if theme.grain_enabled && theme.grain_intensity > 0.0 {
            let brush = theme::grain_brush(theme.grain_intensity);
            // Fill the viewport with an image-brushed rect, letting Repeat tile it.
            let rect = Rect::new(0.0, 0.0, width as f64, height as f64);
            self.scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                &brush,
                None,
                &rect,
            );
        }

        let renderer = self.renderers[dev_id].as_mut().unwrap();
        renderer
            .render_to_texture(device, queue, &self.scene, &surface.target_view,
                &RenderParams {
                    base_color,
                    width, height,
                    antialiasing_method: AaConfig::Area,
                })
            .expect("render failed");

        let tex = surface.surface.get_current_texture().expect("no texture");
        let tex_view = tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        surface.blitter.copy(device, &mut encoder, &surface.target_view, &tex_view);
        queue.submit([encoder.finish()]);
        tex.present();
    }
}

// ── Theme API exposed to JS ──

fn parse_hex(s: &str) -> Option<[f32; 4]> {
    let s = s.trim_start_matches('#');
    let p = |lo: usize, hi: usize| u8::from_str_radix(&s[lo..hi], 16).ok().map(|v| v as f32 / 255.0);
    match s.len() {
        6 => Some([p(0, 2)?, p(2, 4)?, p(4, 6)?, 1.0]),
        8 => Some([p(0, 2)?, p(2, 4)?, p(4, 6)?, p(6, 8)?]),
        _ => None,
    }
}

fn rgba_to_hex(c: [f32; 4]) -> String {
    let to8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", to8(c[0]), to8(c[1]), to8(c[2]))
}

fn theme_to_json(t: &theme::Theme) -> String {
    let mut accents = String::new();
    for (i, c) in t.accents.iter().enumerate() {
        if i > 0 { accents.push(','); }
        accents.push('"');
        accents.push_str(&rgba_to_hex(*c));
        accents.push('"');
    }
    format!(
        r#"{{"name":"{}","background":"{}","stroke":"{}","stroke_width":{},"label":"{}","accents":[{}],"grain_enabled":{},"grain_intensity":{},"spring_stiffness":{},"spring_damping":{},"tween_duration":{},"tween_easing":"{}"}}"#,
        t.name,
        rgba_to_hex(t.background),
        rgba_to_hex(t.stroke),
        t.stroke_width,
        rgba_to_hex(t.label),
        accents,
        t.grain_enabled,
        t.grain_intensity,
        t.spring_stiffness,
        t.spring_damping,
        t.tween_duration,
        t.tween_easing,
    )
}

#[wasm_bindgen]
pub fn theme_get_json() -> String {
    theme_to_json(&theme::current())
}

#[wasm_bindgen]
pub fn theme_set_preset(name: &str) -> bool {
    if let Some(t) = theme::Theme::preset(name) {
        theme::set(t);
        true
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn theme_set_field(key: &str, value: &str) -> bool {
    let mut t = theme::current();
    match key {
        "background" => { if let Some(c) = parse_hex(value) { t.background = c; } else { return false; } }
        "stroke" => { if let Some(c) = parse_hex(value) { t.stroke = c; } else { return false; } }
        "label" => { if let Some(c) = parse_hex(value) { t.label = c; } else { return false; } }
        "stroke_width" => {
            if let Ok(n) = value.parse::<f64>() { t.stroke_width = n; } else { return false; }
        }
        "grain_enabled" => { t.grain_enabled = value == "true" || value == "1"; }
        "grain_intensity" => {
            if let Ok(n) = value.parse::<f32>() { t.grain_intensity = n; } else { return false; }
        }
        "spring_stiffness" => {
            if let Ok(n) = value.parse::<f64>() { t.spring_stiffness = n; } else { return false; }
        }
        "spring_damping" => {
            if let Ok(n) = value.parse::<f64>() { t.spring_damping = n; } else { return false; }
        }
        "tween_duration" => {
            if let Ok(n) = value.parse::<f64>() { t.tween_duration = n; } else { return false; }
        }
        "tween_easing" => {
            t.tween_easing = value.to_string();
        }
        _ => return false,
    }
    theme::set(t);
    true
}

#[wasm_bindgen]
pub fn theme_set_accent(index: usize, hex: &str) -> bool {
    let mut t = theme::current();
    if let Some(c) = parse_hex(hex) {
        while t.accents.len() <= index {
            t.accents.push([1.0, 1.0, 1.0, 1.0]);
        }
        t.accents[index] = c;
        theme::set(t);
        true
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn theme_preset_names() -> String {
    r#"["iso50","byrne","paper","terminal"]"#.to_string()
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not init logger");

    let program: Rc<RefCell<Option<Program>>> = Rc::new(RefCell::new(None));
    let tweaks: Rc<RefCell<Tweakables>> = Rc::new(RefCell::new(Tweakables::new()));

    // JS callback for code changes
    {
        let prog_ref = program.clone();
        let tweaks_ref = tweaks.clone();
        let callback = Closure::wrap(Box::new(move |code: String| {
            match Program::compile(&code, &mut tweaks_ref.borrow_mut()) {
                Ok(prog) => {
                    let n = prog.graph.root.children.len();
                    log::info!("Program compiled: {n} top-level nodes");
                    *prog_ref.borrow_mut() = Some(prog);
                }
                Err(e) => {
                    log::warn!("DSL error: {e}");
                }
            }
        }) as Box<dyn FnMut(String)>);

        let window = web_sys::window().unwrap();
        js_sys::Reflect::set(&window, &JsValue::from_str("__onCodeChange"), callback.as_ref()).unwrap();
        callback.forget();
    }

    // JS callbacks for step forward/back
    {
        let prog_ref = program.clone();
        let cb = Closure::wrap(Box::new(move || {
            if let Some(prog) = prog_ref.borrow_mut().as_mut() {
                prog.step_back();
            }
        }) as Box<dyn FnMut()>);
        js_sys::Reflect::set(&web_sys::window().unwrap(), &JsValue::from_str("__stepBack"), cb.as_ref()).unwrap();
        cb.forget();
    }
    {
        let prog_ref = program.clone();
        let cb = Closure::wrap(Box::new(move || {
            if let Some(prog) = prog_ref.borrow_mut().as_mut() {
                prog.step_forward();
            }
        }) as Box<dyn FnMut()>);
        js_sys::Reflect::set(&web_sys::window().unwrap(), &JsValue::from_str("__stepForward"), cb.as_ref()).unwrap();
        cb.forget();
    }
    {
        let prog_ref = program.clone();
        let tweaks_ref = tweaks.clone();
        let cb = Closure::wrap(Box::new(move || {
            if let Some(prog) = prog_ref.borrow_mut().as_mut() {
                prog.handle_click(&tweaks_ref.borrow());
            }
        }) as Box<dyn FnMut()>);
        js_sys::Reflect::set(&web_sys::window().unwrap(), &JsValue::from_str("__step"), cb.as_ref()).unwrap();
        cb.forget();
    }

    wasm_bindgen_futures::spawn_local(async move {
        let event_loop = EventLoop::new().unwrap();

        #[allow(deprecated)]
        let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());

        {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().unwrap();
            canvas.style().set_css_text("width: 100%; height: 100%;");
            let document = web_sys::window().unwrap().document().unwrap();
            let pane = document.get_element_by_id("canvas-pane")
                .unwrap_or_else(|| document.body().unwrap().into());
            pane.append_child(canvas.as_ref()).unwrap();
        }

        let mut render_cx = RenderContext::new();
        let size = window.inner_size();
        let w = if size.width > 0 { size.width } else { 800 };
        let h = if size.height > 0 { size.height } else { 600 };
        let surface = render_cx
            .create_surface(window.clone(), w, h, wgpu::PresentMode::AutoVsync)
            .await.expect("failed to create surface");

        let dev_id = surface.dev_id;
        let mut renderers: Vec<Option<Renderer>> = Vec::new();
        renderers.resize_with(render_cx.devices.len(), || None);
        let device = &render_cx.devices[dev_id].device;
        renderers[dev_id] = Some(Renderer::new(device, RendererOptions::default()).expect("renderer"));

        // Compile initial editor content
        {
            let initial_code = js_sys::Reflect::get(&web_sys::window().unwrap(), &JsValue::from_str("__editor")).ok()
                .and_then(|e| js_sys::Reflect::get(&e, &JsValue::from_str("state")).ok())
                .and_then(|s| js_sys::Reflect::get(&s, &JsValue::from_str("doc")).ok())
                .and_then(|d| {
                    let f: js_sys::Function = js_sys::Reflect::get(&d, &JsValue::from_str("toString")).ok()?.dyn_into().ok()?;
                    f.call0(&d).ok()
                })
                .and_then(|s| s.as_string());

            if let Some(code) = initial_code {
                if let Ok(prog) = Program::compile(&code, &mut tweaks.borrow_mut()) {
                    *program.borrow_mut() = Some(prog);
                }
            }
        }

        let mut app = App {
            context: render_cx,
            renderers,
            surface: Some(surface),
            window,
            scene: Scene::new(),
            program,
            tweaks,
            last_instant: None,
        };

        event_loop.run_app(&mut app).unwrap();
    });
}
