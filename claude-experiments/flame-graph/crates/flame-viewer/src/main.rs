//! Native flame graph viewer. Loads a trace via CLI arg and/or drag-and-drop,
//! dispatches to format crates by content sniff, hands the resulting `Profile`
//! to `flame-render`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use flame_core::{LoadError, Profile, ProfileBuilder, TraceSource, TrackKind};
use flame_render::Renderer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const SOURCES: &[&dyn TraceSource] = &[
    // Firefox has the strongest "preprocessedProfileVersion" / "stackTable"
    // signature so its detect() runs first. Speedscope and Chrome have
    // overlapping JSON shapes; both check for unique keys.
    &flame_format_firefox::FirefoxSource,
    &flame_format_speedscope::SpeedscopeSource,
    &flame_format_chrome::ChromeSource,
    &flame_format_folded::FoldedSource,
];

fn load_path(path: &Path) -> Result<flame_core::Profile, LoadError> {
    let raw = std::fs::read(path)?;
    let name = path.file_name().and_then(|s| s.to_str()).map(String::from);

    // Decompress if it's gzip (samply emits .json.gz). Strip the trailing
    // ".gz" from the name so format detection sees `.json` extensions.
    let (bytes, sniff_name): (Vec<u8>, Option<String>) = if is_gzip(&raw) {
        log::info!("decompressing gzip ({} bytes compressed)", raw.len());
        let decompressed = decode_gzip(&raw)
            .map_err(|e| LoadError::Parse(format!("gzip decode: {e}")))?;
        log::info!("decompressed → {} bytes", decompressed.len());
        let stripped = name
            .as_deref()
            .and_then(|n| n.strip_suffix(".gz").map(String::from))
            .or(name);
        (decompressed, stripped)
    } else {
        (raw, name)
    };

    let src = SOURCES
        .iter()
        .find(|s| s.detect(&bytes, sniff_name.as_deref()))
        .ok_or(LoadError::UnknownFormat)?;
    log::info!("loading {} via {}", path.display(), src.name());
    let mut builder = ProfileBuilder::new();
    src.load(&bytes, &mut builder)?;
    Ok(builder.finish())
}

fn is_gzip(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

fn decode_gzip(bytes: &[u8]) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::with_capacity(bytes.len() * 4);
    flate2::read::GzDecoder::new(bytes).read_to_end(&mut out)?;
    Ok(out)
}

/// Synthesize a stress-test profile in memory: `tracks` threads each carrying
/// a single nested call chain `depth` frames deep. No file I/O.
fn synth_stress(tracks: u32, depth: u32) -> Profile {
    let started = std::time::Instant::now();
    let mut b = ProfileBuilder::new();
    let proc = b.add_process(0, "stress");
    let cat = b.intern_category("stress");

    // Pre-intern frame names f0..f{depth} once so we share them across tracks.
    let mut frame_names = Vec::with_capacity(depth as usize);
    for d in 0..depth {
        frame_names.push(b.intern_string(&format!("f{d}")));
    }

    let total_ns: u64 = 1_000_000;
    for t in 0..tracks {
        let label = format!("worker {t}");
        let thread = b.add_thread(Some(proc), t as i64, &label);
        let track = b.add_track(TrackKind::Thread(thread), &label, None);

        // Each level is 0.1% narrower than the one above and centered, so we
        // get a clean pyramid that doesn't drift. At depth 999 the slice is
        // still ~37% of the timeline wide.
        let mut width = total_ns;
        for d in 0..depth {
            let start = (total_ns - width) / 2;
            b.add_complete_slice(
                track,
                d as u16,
                start,
                width,
                frame_names[d as usize],
                cat,
                None,
            );
            width = ((width * 999) / 1000).max(1);
        }
    }

    let p = b.finish();
    log::info!(
        "synthesized stress profile: {} tracks × {} deep, {} slices in {:?}",
        tracks,
        depth,
        p.slices.len(),
        started.elapsed()
    );
    p
}

/// Parse `--stress NxM` (or `--stress N×M`). Returns `(tracks, depth)` if matched.
fn parse_stress_arg(arg: &str) -> Option<(u32, u32)> {
    let s = arg.replace('×', "x");
    let (a, b) = s.split_once('x')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

#[allow(dead_code)]
/// Largest row height that still lets every track's full depth fit vertically
/// in the timeline area. Capped at the default `ROW_HEIGHT_PX` so things stay
/// readable when there's plenty of room.
fn compute_fit_row_height(r: &Renderer) -> f32 {
    let Some(p) = &r.profile else { return flame_render::ROW_HEIGHT_PX };
    let total_rows: u32 = p.tracks.iter().map(|t| t.row_count as u32).sum();
    if total_rows == 0 {
        return flame_render::ROW_HEIGHT_PX;
    }
    // Match the math in flame_render::Renderer::fit_all so the two paths agree.
    let usable_h = (r.viewport.size_px.1 - 60.0 /* status */ - 70.0 /* tab bar */).max(1.0);
    let n_tracks = p.tracks.len() as f32;
    let header_total = 36.0 * n_tracks;
    let gap_total = 8.0 * (n_tracks - 1.0).max(0.0);
    let row_h = (usable_h - header_total - gap_total) / total_rows as f32;
    row_h.clamp(0.5, flame_render::ROW_HEIGHT_PX)
}

/// `Home`, `Escape`, or `0` all act as the "reset to readable defaults" key.
fn is_reset_key(k: &Key) -> bool {
    match k {
        Key::Named(NamedKey::Home) | Key::Named(NamedKey::Escape) => true,
        Key::Character(s) => s.as_str() == "0",
        _ => false,
    }
}

struct App {
    pending_path: Option<PathBuf>,
    pending_profile: Option<Profile>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    cursor: (f32, f32),
    drag_origin: Option<(f32, f32)>,
}

impl App {
    fn new(pending_path: Option<PathBuf>, pending_profile: Option<Profile>) -> Self {
        Self {
            pending_path,
            pending_profile,
            window: None,
            renderer: None,
            cursor: (0.0, 0.0),
            drag_origin: None,
        }
    }

    fn ensure_renderer(&mut self) {
        if self.renderer.is_some() {
            return;
        }
        let Some(window) = self.window.clone() else { return };
        let mut renderer = pollster::block_on(Renderer::new(window));

        if let Some(profile) = self.pending_profile.take() {
            renderer.set_profile(Arc::new(profile));
        } else if let Some(path) = self.pending_path.clone() {
            match load_path(&path) {
                Ok(profile) => {
                    log::info!(
                        "loaded {}: {} slices, duration {:?}",
                        path.display(),
                        profile.slices.len(),
                        std::time::Duration::from_nanos(profile.duration_ns())
                    );
                    renderer.set_profile(Arc::new(profile));
                }
                Err(e) => log::error!("failed to load {}: {e}", path.display()),
            }
        }
        renderer.rebuild_instances();
        self.renderer = Some(renderer);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("resumed: creating window");
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("flame-viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create window"),
        );
        log::info!("window created");
        self.window = Some(window);
        self.ensure_renderer();
        log::info!("renderer ready");
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                    r.rebuild_instances();
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(r) = &mut self.renderer {
                    r.render();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                if let Some(origin) = self.drag_origin {
                    let dx = self.cursor.0 - origin.0;
                    let dy = self.cursor.1 - origin.1;
                    if let Some(r) = &mut self.renderer {
                        r.viewport.pan_x_px(dx);
                        r.viewport.pan_y_px(dy);
                        r.clamp_viewport();
                        r.rebuild_instances();
                    }
                    self.drag_origin = Some(self.cursor);
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                } else if let Some(r) = &mut self.renderer {
                    let hit = r.hit_test(self.cursor.0, self.cursor.1);
                    let prev = r.hovered;
                    r.set_hover(hit);
                    if r.hovered != prev {
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            if let Some(r) = &mut self.renderer {
                                // Tabs at the top span the full window width —
                                // check them before anything else.
                                if let Some(tab) =
                                    r.hit_test_inspector_tab(self.cursor.0, self.cursor.1)
                                {
                                    r.set_tab(tab);
                                    r.rebuild_instances();
                                    if let Some(w) = &self.window {
                                        w.request_redraw();
                                    }
                                } else if let Some(mode) =
                                    r.hit_test_layout_button(self.cursor.0, self.cursor.1)
                                {
                                    r.set_layout_mode(mode);
                                    r.rebuild_instances();
                                    if let Some(w) = &self.window {
                                        w.request_redraw();
                                    }
                                } else if let Some(track_id) =
                                    r.hit_test_track_header(self.cursor.0, self.cursor.1)
                                {
                                    r.toggle_track_collapsed(track_id);
                                    r.rebuild_instances();
                                    if let Some(w) = &self.window {
                                        w.request_redraw();
                                    }
                                } else if r.cursor_in_inspector(self.cursor.0) {
                                    if let Some(slice_idx) =
                                        r.hit_test_inspector(self.cursor.0, self.cursor.1)
                                    {
                                        r.select_slice(Some(slice_idx));
                                        // Center the time viewport on the exemplar.
                                        if let Some(p) = &r.profile {
                                            let s = p.slices.start_ns[slice_idx as usize];
                                            let d = p.slices.dur_ns[slice_idx as usize];
                                            let mid = s as f64 + d as f64 * 0.5;
                                            r.viewport.start_ns = mid
                                                - r.viewport.size_px.0 as f64
                                                    * r.viewport.ns_per_pixel
                                                    * 0.5;
                                        }
                                        r.clamp_viewport();
                                        r.rebuild_instances();
                                        if let Some(w) = &self.window {
                                            w.request_redraw();
                                        }
                                    }
                                    // Don't start a drag inside the inspector.
                                } else {
                                    // Timeline click: select slice under cursor (if any),
                                    // and start a drag-pan.
                                    let hit_inst = r.hit_test(self.cursor.0, self.cursor.1);
                                    let slice_idx = hit_inst
                                        .and_then(|i| r.instance_to_slice(i));
                                    r.select_slice(slice_idx);
                                    self.drag_origin = Some(self.cursor);
                                    r.rebuild_instances();
                                    if let Some(w) = &self.window {
                                        w.request_redraw();
                                    }
                                }
                            }
                        }
                        ElementState::Released => {
                            self.drag_origin = None;
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Two-finger trackpad scroll / mouse wheel pans the viewport.
                // Horizontal scroll → time axis. Vertical scroll → depth (rows).
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x as f32 * 30.0, y as f32 * 30.0),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
                };
                if let Some(r) = &mut self.renderer {
                    if dx != 0.0 {
                        r.viewport.pan_x_px(dx);
                    }
                    if dy != 0.0 {
                        r.viewport.pan_y_px(dy);
                    }
                    r.clamp_viewport();
                    r.rebuild_instances();
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }

            WindowEvent::PinchGesture { delta, .. } => {
                // macOS / trackpad pinch. `delta` is a small relative change per
                // event (positive = pinch out / zoom in). zoom_at takes a factor
                // where < 1 zooms in, so invert.
                if let Some(r) = &mut self.renderer {
                    let factor = (1.0 - delta).clamp(0.5, 2.0);
                    r.viewport.zoom_at(self.cursor.0, factor);
                    // Pinch in (zoom in): grow row height back toward default
                    // if it was squashed by an explicit fit-all reset.
                    if factor < 1.0 && r.viewport.row_height_px < flame_render::ROW_HEIGHT_PX {
                        r.viewport.row_height_px = ((r.viewport.row_height_px as f64 / factor)
                            as f32)
                            .min(flame_render::ROW_HEIGHT_PX);
                    }
                    r.clamp_viewport();
                    r.rebuild_instances();
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let Some(r) = &mut self.renderer {
                    let mut redraw = true;
                    match event.logical_key {
                        ref k if is_reset_key(k) => {
                            // Full reset: fit the entire trace into the viewport
                            // both horizontally AND vertically so every depth on
                            // every track is visible at once. Row height shrinks
                            // (capped at ROW_HEIGHT_PX) to make all rows fit.
                            r.fit_all();
                            r.rebuild_instances();
                        }
                        Key::Character(ref s) if s.as_str().eq_ignore_ascii_case("a") => {
                            // Fit-all: squash row height so every track's full
                            // depth is visible at once.
                            r.fit_all();
                            r.rebuild_instances();
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            let pan = r.viewport.size_px.0 * 0.10;
                            r.viewport.pan_x_px(pan);
                            r.clamp_viewport();
                            r.rebuild_instances();
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            let pan = r.viewport.size_px.0 * 0.10;
                            r.viewport.pan_x_px(-pan);
                            r.clamp_viewport();
                            r.rebuild_instances();
                        }
                        Key::Named(NamedKey::ArrowUp) => {
                            r.viewport.pan_y_px(20.0);
                            r.clamp_viewport();
                            r.rebuild_instances();
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            r.viewport.pan_y_px(-20.0);
                            r.clamp_viewport();
                            r.rebuild_instances();
                        }
                        Key::Character(ref s) if s.as_str() == "+" || s.as_str() == "=" => {
                            r.viewport.zoom_at(r.viewport.size_px.0 * 0.5, 0.7);
                            r.clamp_viewport();
                            r.rebuild_instances();
                        }
                        Key::Character(ref s) if s.as_str() == "-" || s.as_str() == "_" => {
                            r.viewport.zoom_at(r.viewport.size_px.0 * 0.5, 1.43);
                            r.clamp_viewport();
                            r.rebuild_instances();
                        }
                        Key::Character(ref s) if s.as_str().eq_ignore_ascii_case("f") => {
                            r.flip_direction();
                            log::info!("direction → {}", r.direction.label());
                            r.rebuild_instances();
                        }
                        Key::Character(ref s) if matches!(s.as_str(), "1" | "2" | "3" | "4" | "5") => {
                            let idx = s.as_str().parse::<usize>().unwrap_or(1).saturating_sub(1);
                            if let Some(&tab) = flame_render::MainTab::ALL.get(idx) {
                                r.set_tab(tab);
                                r.rebuild_instances();
                            }
                        }
                        _ => redraw = false,
                    }
                    if redraw {
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
            }

            WindowEvent::DroppedFile(path) => {
                match load_path(&path) {
                    Ok(profile) => {
                        log::info!(
                            "loaded {}: {} slices, duration {:?}",
                            path.display(),
                            profile.slices.len(),
                            std::time::Duration::from_nanos(profile.duration_ns())
                        );
                        if let Some(r) = &mut self.renderer {
                            r.set_profile(Arc::new(profile));
                            r.rebuild_instances();
                        }
                    }
                    Err(e) => log::error!("failed to load {}: {e}", path.display()),
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = std::env::args().skip(1).peekable();
    let mut pending_path: Option<PathBuf> = None;
    let mut pending_profile: Option<Profile> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--stress" => {
                let dims = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--stress requires WxD argument, e.g. 1000x1000"))?;
                let (tracks, depth) = parse_stress_arg(&dims)
                    .ok_or_else(|| anyhow::anyhow!("--stress arg must be NxM, got {dims:?}"))?;
                pending_profile = Some(synth_stress(tracks, depth));
            }
            other => {
                if pending_path.is_some() {
                    anyhow::bail!("unexpected extra argument: {other}");
                }
                pending_path = Some(PathBuf::from(other));
            }
        }
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(pending_path, pending_profile);
    event_loop.run_app(&mut app)?;
    Ok(())
}
