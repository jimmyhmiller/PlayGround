//! Render a headless-whiteboard scene on a real GPU surface via Vello.
//!
//! This is the GPU sibling of `examples/winit-draw` (which rasterizes through
//! tiny-skia into a softbuffer framebuffer). Here the pipeline is:
//!
//! ```text
//!   Editor ──render()──▶ RenderScene ──build_vello_scene──▶ vello::Scene
//!          ──Renderer::render_to_texture──▶ intermediate Rgba8 texture
//!          ──TextureBlitter::copy──▶ swapchain surface texture ──present──▶ window
//! ```
//!
//! Building the `vello::Scene` is the headless-testable part (see the
//! `whiteboard-vello` crate's own unit tests). Everything below that point needs
//! a wgpu device/queue and an OS surface, so it can only *run* on a machine with
//! a working GPU — but it must always *compile*. To keep the headless/CI path
//! honest, `--check` exercises the whole CPU side (scene build + Vello scene
//! encode) and exits without ever touching the GPU.
//!
//! Controls mirror `winit-draw`:
//!   r / o / d   rectangle / ellipse (oval) / diamond tool
//!   l / a       line / arrow tool
//!   f           freedraw tool
//!   v / 1       select tool
//!   space-drag  pan      |  mouse wheel  scroll   |  ctrl+wheel  zoom
//!   u           undo     |  shift+u      redo
//!   Delete      delete selection
//!   Esc         quit

use std::sync::Arc;

use vello::peniko::Color as VColor;
use vello::util::{RenderContext, RenderSurface};
use vello::wgpu;
use vello::{AaConfig, Renderer, RendererOptions};
use vello::{RenderParams, Scene};

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::geometry::{Point, Vec2};
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::render::{Color, FillStyle};
use whiteboard_core::shape::RoughGenerator;
use whiteboard_core::text::MonospaceMeasurer;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key as WKey, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

/// Concrete editor: a monospace measurer (no font files needed to lay out text
/// boxes) and the rough/sketchy generator for the Excalidraw look.
type WhiteboardEditor = Editor<MonospaceMeasurer, RoughGenerator>;

/// Background the surface is cleared to before the scene is composited.
const BG: VColor = VColor::from_rgb8(0xf2, 0xf2, 0xf2);

/// Build a small starter scene so the window shows something immediately. IDs and
/// seeds are fixed constants — no global RNG/time — so the scene is identical on
/// every run (matches the project's determinism rule).
fn build_scene(editor: &mut WhiteboardEditor) {
    // A filled rectangle.
    let mut rect = Element::new(
        ElementId::new("rect-1"),
        1,
        80.0,
        80.0,
        220.0,
        140.0,
        ElementKind::Rectangle,
    );
    rect.stroke_color = Color::rgb(0x1e, 0x1e, 0x1e);
    rect.background_color = Color::rgb(0xa5, 0xd8, 0xff);
    rect.fill_style = FillStyle::Solid;
    rect.stroke_width = 2.0;
    editor.add_element(rect);

    // An ellipse with hachure fill.
    let mut ellipse = Element::new(
        ElementId::new("ellipse-1"),
        2,
        360.0,
        120.0,
        180.0,
        180.0,
        ElementKind::Ellipse,
    );
    ellipse.stroke_color = Color::rgb(0xe0, 0x3e, 0x3e);
    ellipse.background_color = Color::rgb(0xff, 0xc9, 0xc9);
    ellipse.fill_style = FillStyle::Hachure;
    ellipse.stroke_width = 2.0;
    editor.add_element(ellipse);

    // A diamond outline.
    let mut diamond = Element::new(
        ElementId::new("diamond-1"),
        3,
        160.0,
        300.0,
        200.0,
        140.0,
        ElementKind::Diamond,
    );
    diamond.stroke_color = Color::rgb(0x2f, 0x9e, 0x44);
    diamond.stroke_width = 3.0;
    editor.add_element(diamond);
}

/// Create an editor with the starter scene wired in.
fn new_editor() -> WhiteboardEditor {
    let mut editor = Editor::new_rough(MonospaceMeasurer::default());
    build_scene(&mut editor);
    editor
}

/// Per-window GPU render state. Held in an `Option` because winit only hands us
/// an event loop (and lets us create a window) once `resumed` fires.
struct RenderState {
    /// `'static` because the surface borrows the window; an `Arc<Window>` keeps
    /// the window alive for exactly as long as the surface.
    window: Arc<Window>,
    surface: RenderSurface<'static>,
}

struct App {
    context: RenderContext,
    /// One Vello renderer per device id (vello's recommended pattern). Lazily
    /// created on first render for the device the surface picked.
    renderers: Vec<Option<Renderer>>,
    state: Option<RenderState>,
    scene: Scene,
    editor: WhiteboardEditor,
    mods: Modifiers,
    cursor: Point,
}

impl App {
    fn new() -> Self {
        App {
            context: RenderContext::new(),
            renderers: Vec::new(),
            state: None,
            scene: Scene::new(),
            editor: new_editor(),
            mods: Modifiers::default(),
            cursor: Point::ORIGIN,
        }
    }

    fn forward(&mut self, event: InputEvent) {
        let res = self.editor.handle(event);
        if res.needs_redraw() {
            self.request_redraw();
        }
    }

    fn request_redraw(&self) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    /// Rebuild the Vello scene from the editor and present it to the surface.
    fn redraw(&mut self) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let width = state.surface.config.width;
        let height = state.surface.config.height;
        if width == 0 || height == 0 {
            return;
        }

        // CPU side: headless library -> draw commands -> vello::Scene.
        let render_scene = self.editor.render();
        self.scene = whiteboard_vello::build_vello_scene(&render_scene);

        let dev_id = state.surface.dev_id;
        let device_handle = &self.context.devices[dev_id];

        // Lazily build the renderer for this device.
        if self.renderers.len() <= dev_id {
            self.renderers.resize_with(dev_id + 1, || None);
        }
        let renderer = self.renderers[dev_id].get_or_insert_with(|| {
            Renderer::new(
                &device_handle.device,
                RendererOptions {
                    use_cpu: false,
                    antialiasing_support: vello::AaSupport::area_only(),
                    num_init_threads: None,
                    pipeline_cache: None,
                },
            )
            .expect("create vello renderer")
        });

        // Render the scene into the surface's intermediate Rgba8 texture.
        renderer
            .render_to_texture(
                &device_handle.device,
                &device_handle.queue,
                &self.scene,
                &state.surface.target_view,
                &RenderParams {
                    base_color: BG,
                    width,
                    height,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .expect("render_to_texture");

        // Acquire the swapchain image and blit the intermediate texture onto it.
        // wgpu 29 returns a status enum rather than a `Result`; for the
        // transient states (timeout/occluded/outdated) we drop this frame and
        // wait for the next redraw rather than panicking.
        let surface_texture = match state.surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                eprintln!("skipping frame: surface not ready ({other:?})");
                return;
            }
        };
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            device_handle
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vello-window blit"),
                });
        state.surface.blitter.copy(
            &device_handle.device,
            &mut encoder,
            &state.surface.target_view,
            &surface_view,
        );
        device_handle.queue.submit([encoder.finish()]);
        surface_texture.present();
        device_handle.device.poll(wgpu::PollType::Poll).ok();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("headless-whiteboard — Vello GPU")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        // Create the wgpu surface. This is async (adapter/device request); we
        // block on it with vello's helper so we don't need a separate runtime.
        let surface_future = self.context.create_surface(
            window.clone(),
            width,
            height,
            wgpu::PresentMode::AutoVsync,
        );
        let surface = pollster_block(surface_future).expect("create vello surface");

        self.state = Some(RenderState { window, surface });
        self.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.as_mut() {
                    let (w, h) = (size.width.max(1), size.height.max(1));
                    self.context.resize_surface(&mut state.surface, w, h);
                    state.window.request_redraw();
                }
            }
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

/// Minimal blocking executor for the single `create_surface` future. We avoid an
/// extra crate dependency (pollster) and the GPU-specialized `block_on_wgpu`
/// (which needs a `Device` we don't have yet here) by spin-polling with a no-op
/// waker. The surface future only awaits adapter/device acquisition, which makes
/// progress independently of being polled, so a busy spin terminates.
fn pollster_block<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    // SAFETY: the vtable's clone/wake/drop are all no-ops over a null data
    // pointer, so the waker never dereferences anything.
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    // SAFETY: `fut` is owned and never moved after being pinned to the stack.
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::hint::spin_loop(),
        }
    }
}

/// Headless self-check: build the scene and encode it through Vello entirely on
/// the CPU (no GPU, no window). Proves the non-GPU half of the pipeline without a
/// display, so CI can run it. Returns the encoded scene so callers can assert on
/// it.
fn check_headless() -> Scene {
    let editor = new_editor();
    let render_scene = editor.render();
    let (scene, report) = whiteboard_vello::build_vello_scene_report(&render_scene);
    assert!(
        report.fills + report.strokes > 0,
        "starter scene must encode at least one fill or stroke (got {report:?})"
    );
    assert!(
        !scene.encoding().is_empty(),
        "encoded vello scene must be non-empty"
    );
    scene
}

fn main() {
    // `--check` runs the headless CPU path and exits 0 without opening a window
    // or touching the GPU — usable in environments without a display/GPU.
    if std::env::args().any(|a| a == "--check") {
        let scene = check_headless();
        let enc = scene.encoding();
        println!(
            "vello-window --check OK: encoded vello scene ({} draw tags, {} path tags)",
            enc.draw_tags.len(),
            enc.path_tags.len()
        );
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starter_scene_has_three_live_elements() {
        let editor = new_editor();
        let live = editor.scene().iter_live().count();
        assert_eq!(live, 3, "rectangle + ellipse + diamond");
    }

    #[test]
    fn headless_check_encodes_non_empty_scene() {
        let scene = check_headless();
        assert!(!scene.encoding().is_empty());
    }

    #[test]
    fn editor_handles_pointer_without_gpu() {
        // The interaction path the window forwards into must work headlessly.
        let mut editor = new_editor();
        editor.set_tool(Tool::Select);
        let res = editor.handle(InputEvent::PointerMove {
            pos: Point::new(10.0, 10.0),
            mods: Modifiers::default(),
        });
        // Just proves the seam is wired and doesn't panic; move alone need not
        // change anything.
        let _ = res.needs_redraw();
    }
}
