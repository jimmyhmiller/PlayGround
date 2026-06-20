//! Age of Models — an Age-of-Empires-style view of live AI/Claude activity.
//!
//! Each project is a city, each session a building, and activity becomes villagers
//! who wander and trade between projects. The data comes from a pluggable
//! [`data::WorldSource`]; by default we read Claude Code's local session logs.
//!
//! Usage:
//!   age_of_models            # read ~/.claude/projects (live)
//!   age_of_models --mock     # synthetic demo data
//!   age_of_models --screenshot out.png   # render a few frames headless, save PNG

mod achievements;
mod data;
mod game;
mod render;
mod util;

use data::claude::ClaudeProjectsSource;
use data::mock::MockSource;
use data::{SourceRunner, WorldSource};
use game::World;
use raylib::prelude::*;

/// Directory that contains `assets/` — fixed at compile time so the binary finds
/// art no matter what the working directory is.
const ASSET_BASE: &str = env!("CARGO_MANIFEST_DIR");

struct Args {
    mock: bool,
    screenshot: Option<String>,
    scan: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args { mock: false, screenshot: None, scan: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--mock" => a.mock = true,
            "--screenshot" => a.screenshot = it.next(),
            "--scan" => a.scan = it.next(),
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();

    // Debug: scan a codebase and print what the biome/achievements would see.
    if let Some(path) = args.scan {
        let cb = data::repo::scan(std::path::Path::new(&path));
        println!("{path}");
        println!("  files={}  loc={}  commits={}  bytes={}", cb.files, cb.loc, cb.commits, cb.bytes);
        println!("  readme={}  tests={}", cb.has_readme, cb.has_tests);
        println!("  languages={:?}", cb.languages);
        println!("  dominant={:?} -> biome {:?}", cb.dominant_lang(), cb.dominant_lang().map(game::Biome::from_lang));
        return;
    }

    let (mut rl, thread) = raylib::init()
        .size(1280, 800)
        .title("Age of Models")
        .resizable()
        .build();
    rl.set_target_fps(60);

    let assets = match render::assets::Assets::load(&mut rl, &thread, ASSET_BASE) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("failed to load assets from {ASSET_BASE}/assets: {e}");
            std::process::exit(1);
        }
    };

    let source: Box<dyn WorldSource> = if args.mock {
        Box::new(MockSource::new())
    } else {
        Box::new(ClaudeProjectsSource::new())
    };
    let mut runner = SourceRunner::spawn(source, 3.0);
    let source_name = runner.source_name().to_string();

    let mut world = World::new();
    let mut cam = Camera2D {
        offset: Vector2::new(640.0, 400.0),
        target: Vector2::new(640.0, 400.0),
        rotation: 0.0,
        zoom: 0.5,
    };
    let mut selected: Option<usize> = None;
    let mut framed_once = false;

    // Left-drag pans; a click without much movement selects.
    let mut press_pos = Vector2::zero();
    let mut dragging = false;
    let mut drag_dist = 0.0f32;

    if let Some(out) = args.screenshot.clone() {
        run_screenshot(&mut rl, &thread, &assets, &mut runner, &mut world, &mut cam, &source_name, &out);
        return;
    }

    while !rl.window_should_close() {
        let dt = rl.get_frame_time();
        let sw = rl.get_screen_width();
        let sh = rl.get_screen_height();
        cam.offset = Vector2::new(sw as f32 / 2.0, sh as f32 / 2.0);

        // Pull fresh data (non-blocking).
        if runner.poll_latest() {
            world.sync(runner.latest());
            if !framed_once && !world.cities.is_empty() {
                frame_all(&mut cam, &world, sw, sh);
                framed_once = true;
            }
        }

        // ---- input ----------------------------------------------------------
        let mouse = rl.get_mouse_position();
        let wheel = rl.get_mouse_wheel_move();
        if wheel.abs() > 0.0 {
            let before = rl.get_screen_to_world2D(mouse, cam);
            cam.zoom = (cam.zoom * (1.0 + wheel * 0.12)).clamp(0.12, 3.5);
            let after = rl.get_screen_to_world2D(mouse, cam);
            cam.target.x += before.x - after.x;
            cam.target.y += before.y - after.y;
        }

        // Keyboard pan.
        let pan = 420.0 * dt / cam.zoom;
        if rl.is_key_down(KeyboardKey::KEY_A) || rl.is_key_down(KeyboardKey::KEY_LEFT) {
            cam.target.x -= pan;
        }
        if rl.is_key_down(KeyboardKey::KEY_D) || rl.is_key_down(KeyboardKey::KEY_RIGHT) {
            cam.target.x += pan;
        }
        if rl.is_key_down(KeyboardKey::KEY_W) || rl.is_key_down(KeyboardKey::KEY_UP) {
            cam.target.y -= pan;
        }
        if rl.is_key_down(KeyboardKey::KEY_S) || rl.is_key_down(KeyboardKey::KEY_DOWN) {
            cam.target.y += pan;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_F) && !world.cities.is_empty() {
            frame_all(&mut cam, &world, sw, sh);
        }

        // Drag-pan vs click-select.
        if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) {
            press_pos = mouse;
            dragging = true;
            drag_dist = 0.0;
        }
        if dragging && rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            let delta = rl.get_mouse_delta();
            drag_dist += delta.x.abs() + delta.y.abs();
            cam.target.x -= delta.x / cam.zoom;
            cam.target.y -= delta.y / cam.zoom;
        }
        if rl.is_mouse_button_released(MouseButton::MOUSE_BUTTON_LEFT) {
            if dragging && drag_dist < 6.0 {
                let w = rl.get_screen_to_world2D(press_pos, cam);
                selected = match world.pick_city(w, 150.0) {
                    Some(i) => {
                        if selected == Some(i) {
                            None
                        } else {
                            Some(i)
                        }
                    }
                    None => None,
                };
            }
            dragging = false;
        }

        // ---- simulate -------------------------------------------------------
        world.update(dt);

        // Building hover (skip when dragging or hovering a UI panel).
        let mouse_world = rl.get_screen_to_world2D(mouse, cam);
        let over_ui = mouse.y < 34.0
            || mouse.y > sh as f32 - 26.0
            || (selected.is_some() && mouse.x > sw as f32 - 384.0);
        let hovered = if dragging || over_ui {
            None
        } else {
            render::pick_building(&world, mouse_world)
        };

        // ---- render ---------------------------------------------------------
        let fps = rl.get_fps();
        let loading = runner.is_loading();
        let sel = selected.filter(|&i| i < world.cities.len());
        let mut d = rl.begin_drawing(&thread);
        render::clear_bg(&mut d);
        {
            let mut m = d.begin_mode2D(cam);
            render::draw_world_space(&mut m, &world, &assets, &cam, (sw, sh), sel);
        }
        render::draw_labels(&mut d, &world, &cam, (sw, sh), sel);
        render::draw_hud(&mut d, &world, &source_name, loading, fps, (sw, sh));
        if let Some(i) = sel {
            render::draw_inspector(&mut d, &world.cities[i], world.now, (sw, sh));
        }
        if let Some((ci, bi)) = hovered {
            if ci < world.cities.len() && bi < world.cities[ci].buildings.len() {
                let city = &world.cities[ci];
                render::draw_building_tooltip(&mut d, city, &city.buildings[bi], world.now, mouse, (sw, sh));
            }
        }
    }
}

/// Fit the camera so every city is on screen.
fn frame_all(cam: &mut Camera2D, world: &World, sw: i32, sh: i32) {
    let (min, max) = world.extent;
    let margin = 280.0;
    let w = (max.x - min.x).max(1.0) + margin * 2.0;
    let h = (max.y - min.y).max(1.0) + margin * 2.0;
    cam.target = Vector2::new((min.x + max.x) / 2.0, (min.y + max.y) / 2.0);
    let zx = sw as f32 / w;
    let zy = sh as f32 / h;
    cam.zoom = zx.min(zy).clamp(0.12, 1.2);
}

/// Headless verification: wait for the first snapshot, simulate a bit so the
/// town settles and villagers move, render, then write a PNG and exit.
fn run_screenshot(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    assets: &render::assets::Assets,
    runner: &mut SourceRunner,
    world: &mut World,
    cam: &mut Camera2D,
    source_name: &str,
    out: &str,
) {
    let sw = rl.get_screen_width();
    let sh = rl.get_screen_height();
    cam.offset = Vector2::new(sw as f32 / 2.0, sh as f32 / 2.0);

    // Wait for the first snapshot, then keep polling ~20s so the incremental repo
    // scans (a handful per poll) finish and biomes/codebase stats are representative.
    let mut waited = 0;
    while waited < 1300 {
        if runner.poll_latest() {
            world.sync(runner.latest());
        }
        // Stop once we have data and every city's codebase has been scanned (mock
        // data is ready immediately; real repos finish within ~20s), or we time out.
        let scanned = !world.cities.is_empty() && world.cities.iter().all(|c| c.codebase.is_some());
        if !runner.is_loading() && (scanned || waited >= 1250) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
        waited += 1;
    }
    frame_all(cam, world, sw, sh);

    // Settle motion.
    for _ in 0..150 {
        world.update(1.0 / 60.0);
    }

    let selected = if world.cities.is_empty() { None } else { Some(0) };
    // Draw several identical frames so both swap buffers hold the image before
    // raylib reads the framebuffer (a single frame screenshots as black).
    for _ in 0..4 {
        world.update(1.0 / 60.0);
        let mut d = rl.begin_drawing(thread);
        render::clear_bg(&mut d);
        {
            let mut m = d.begin_mode2D(*cam);
            render::draw_world_space(&mut m, world, assets, cam, (sw, sh), selected);
        }
        render::draw_labels(&mut d, world, cam, (sw, sh), selected);
        render::draw_hud(&mut d, world, source_name, false, 60, (sw, sh));
        if let Some(i) = selected {
            render::draw_inspector(&mut d, &world.cities[i], world.now, (sw, sh));
        }
    }
    rl.take_screenshot(thread, out);
    println!("wrote screenshot to {out} ({} cities)", world.cities.len());
}
