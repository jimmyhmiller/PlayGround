//! Draw on a headless-whiteboard canvas with the mouse.
//!
//! This is the end-to-end proof that the headless library drives a real app: the
//! window forwards raw winit events as [`InputEvent`]s into the [`Editor`], asks
//! it for a [`whiteboard_core::RenderScene`], and hands that command list to the
//! tiny-skia backend, which rasterizes into a softbuffer framebuffer.
//!
//! Controls:
//!   r / o / d   rectangle / ellipse (oval) / diamond tool
//!   l / a       line / arrow tool
//!   f           freedraw tool
//!   v / 1       select tool
//!   space-drag  pan      |  mouse wheel  scroll   |  ctrl+wheel  zoom
//!   u           undo     |  shift+u      redo
//!   Delete      delete selection
//!   Esc         quit

use std::num::NonZeroU32;
use std::rc::Rc;

use softbuffer::{Context, Surface};
use whiteboard_core::editor::Editor;
use whiteboard_core::geometry::{Point, Vec2};
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::render::Backend;
use whiteboard_tiny_skia::{FontMeasurer, TinySkiaBackend};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key as WKey, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

type WhiteboardEditor = Editor<FontMeasurer, whiteboard_core::shape::RoughGenerator>;

struct App {
    window: Option<Rc<Window>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    editor: WhiteboardEditor,
    backend: TinySkiaBackend,
    mods: Modifiers,
    cursor: Point,
    size: (u32, u32),
}

impl App {
    fn new() -> Self {
        App {
            window: None,
            surface: None,
            // The sketchy (rough) generator is the default Excalidraw look.
            editor: Editor::new_rough(FontMeasurer::new()),
            backend: TinySkiaBackend::new(1024, 768),
            mods: Modifiers::default(),
            cursor: Point::ORIGIN,
            size: (1024, 768),
        }
    }

    fn redraw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let (w, h) = self.size;
        if w == 0 || h == 0 {
            return;
        }

        // Render the headless scene to draw commands, then rasterize.
        let scene = self.editor.render();
        self.backend.render(&scene);

        // Blit the tiny-skia pixmap (RGBA, premultiplied) into the softbuffer
        // surface (0RGB u32 per pixel).
        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        let pixmap = self.backend.pixmap();
        let mut buffer = surface.buffer_mut().expect("acquire softbuffer buffer");
        for (dst, px) in buffer.iter_mut().zip(pixmap.pixels()) {
            // tiny-skia stores premultiplied; demultiply is unnecessary for
            // opaque display — pack straight RGB.
            let r = px.red() as u32;
            let g = px.green() as u32;
            let b = px.blue() as u32;
            *dst = (r << 16) | (g << 8) | b;
        }
        buffer.present().expect("present softbuffer");
        window.request_redraw();
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.size = (w, h);
        self.backend = TinySkiaBackend::new(w, h);
        if let Some(surface) = self.surface.as_mut() {
            if let (Some(nw), Some(nh)) = (NonZeroU32::new(w), NonZeroU32::new(h)) {
                surface.resize(nw, nh).expect("resize surface");
            }
        }
    }

    fn forward(&mut self, event: InputEvent) {
        let res = self.editor.handle(event);
        if res.needs_redraw() {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("headless-whiteboard — draw with the mouse")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0));
        let window = Rc::new(event_loop.create_window(attrs).expect("create window"));
        let context = Context::new(window.clone()).expect("softbuffer context");
        let surface = Surface::new(&context, window.clone()).expect("softbuffer surface");
        self.window = Some(window);
        self.surface = Some(surface);
        let (w, h) = self.size;
        self.resize(w, h);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size.width.max(1), size.height.max(1)),
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::ModifiersChanged(m) => self.mods = to_mods(m.state()),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Point::new(position.x, position.y);
                self.forward(InputEvent::PointerMove {
                    pos: self.cursor,
                    mods: self.mods,
                });
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some(button) = to_button(button) else {
                    return;
                };
                let pos = self.cursor;
                let mods = self.mods;
                let event = match state {
                    ElementState::Pressed => InputEvent::PointerDown { pos, button, mods },
                    ElementState::Released => InputEvent::PointerUp { pos, button, mods },
                };
                self.forward(event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        Vec2::new(x as f64 * 30.0, y as f64 * 30.0)
                    }
                    MouseScrollDelta::PixelDelta(p) => Vec2::new(p.x, p.y),
                };
                let pos = self.cursor;
                let mods = self.mods;
                self.forward(InputEvent::Wheel {
                    delta: d,
                    pos,
                    mods,
                });
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                self.on_key(event_loop, event.logical_key);
            }
            _ => {}
        }
    }
}

impl App {
    fn on_key(&mut self, event_loop: &ActiveEventLoop, key: WKey) {
        match key {
            WKey::Named(NamedKey::Escape) => event_loop.exit(),
            WKey::Named(NamedKey::Delete) | WKey::Named(NamedKey::Backspace) => {
                if self.editor.delete_selection() {
                    self.request_redraw();
                }
            }
            WKey::Character(c) => {
                let tool = match c.as_str() {
                    "r" => Some(Tool::Rectangle),
                    "o" => Some(Tool::Ellipse),
                    "d" => Some(Tool::Diamond),
                    "l" => Some(Tool::Line),
                    "a" => Some(Tool::Arrow),
                    "f" => Some(Tool::Freedraw),
                    "v" | "1" => Some(Tool::Select),
                    "u" => {
                        if self.mods.shift {
                            self.editor.redo();
                        } else {
                            self.editor.undo();
                        }
                        self.request_redraw();
                        None
                    }
                    _ => None,
                };
                if let Some(tool) = tool {
                    self.editor.set_tool(tool);
                    self.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn to_button(b: MouseButton) -> Option<PointerButton> {
    match b {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

fn to_mods(m: ModifiersState) -> Modifiers {
    Modifiers {
        shift: m.shift_key(),
        ctrl: m.control_key(),
        alt: m.alt_key(),
        meta: m.super_key(),
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
