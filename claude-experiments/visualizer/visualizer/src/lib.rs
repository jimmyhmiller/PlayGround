mod anim;
mod animated;
mod collection;
mod dsl;
mod hashmap_viz;
mod scene;
mod scene_demo;
mod state;
mod tweakables;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use dsl::runtime::Program;
use tweakables::Tweakables;
use vello::peniko::color::palette;
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
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let Some(prog) = self.program.borrow_mut().as_mut() {
                    match &event.logical_key {
                        Key::Named(NamedKey::ArrowLeft) => prog.step_back(),
                        Key::Named(NamedKey::ArrowRight) => prog.step_forward(),
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

        let renderer = self.renderers[dev_id].as_mut().unwrap();
        renderer
            .render_to_texture(device, queue, &self.scene, &surface.target_view,
                &RenderParams {
                    base_color: palette::css::BLACK,
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
