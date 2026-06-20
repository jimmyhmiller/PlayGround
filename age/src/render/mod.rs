//! Rendering: draws the world with raylib. A camera-space pass paints terrain,
//! buildings and villagers; a screen-space pass paints labels, the HUD, and the
//! selected-city inspector.

pub mod assets;

use crate::achievements::Cat;
use crate::data::Tool;
use crate::game::{Biome, Building, City, Season, Tier, Villager, World, LIVE_WINDOW};
use assets::{src, Assets};
use raylib::prelude::*;

pub const HOUSE_SCALE: f32 = 3.0;
pub const TC_SCALE: f32 = 3.8;
pub const VILLAGER_SCALE: f32 = 2.0;
const GROUND: f32 = 32.0; // grass tile size on screen (world px)

// Palette
const SKY: Color = Color::new(28, 32, 44, 255);
const GOLD: Color = Color::new(255, 196, 64, 255);
const PANEL_BG: Color = Color::new(20, 22, 30, 232);
const PANEL_LINE: Color = Color::new(70, 78, 96, 255);
const TEXT_DIM: Color = Color::new(168, 176, 196, 255);

/// World->screen for a 2D camera (matches raylib's transform), so we can draw
/// crisp constant-size labels in the screen pass.
pub fn world_to_screen(p: Vector2, cam: &Camera2D) -> Vector2 {
    Vector2::new(
        (p.x - cam.target.x) * cam.zoom + cam.offset.x,
        (p.y - cam.target.y) * cam.zoom + cam.offset.y,
    )
}

/// Rough text width for the default font (avoids needing the RaylibHandle here).
fn text_w(text: &str, size: i32) -> i32 {
    (text.chars().count() as f32 * size as f32 * 0.5).round() as i32
}

fn model_color(model: Option<&str>) -> Color {
    match model.unwrap_or("").to_ascii_lowercase() {
        m if m.contains("opus") => Color::new(196, 148, 255, 255), // purple
        m if m.contains("sonnet") => Color::new(120, 200, 255, 255), // blue
        m if m.contains("haiku") => Color::new(120, 230, 170, 255), // green
        _ => Color::new(200, 200, 210, 255),
    }
}

fn short_model(model: Option<&str>) -> &str {
    match model.unwrap_or("") {
        m if m.contains("opus") => "opus",
        m if m.contains("sonnet") => "sonnet",
        m if m.contains("haiku") => "haiku",
        m if m.is_empty() => "?",
        _ => "model",
    }
}

// ---- camera-space pass -------------------------------------------------------

fn draw_tile<D: RaylibDraw>(
    d: &mut D,
    tex: &Texture2D,
    idx: i32,
    x: f32,
    y: f32,
    scale: f32,
    tint: Color,
) {
    let dst = Rectangle::new(x, y, assets::TILE * scale, assets::TILE * scale);
    d.draw_texture_pro(tex, src(idx), dst, Vector2::new(0.0, 0.0), 0.0, tint);
}

/// Draw a building's tile grid, bottom-center anchored at `base`.
fn draw_building<D: RaylibDraw>(d: &mut D, tex: &Texture2D, b: &Building) {
    let preset = if b.is_town_center {
        &assets::TOWN_CENTER
    } else {
        &assets::HOUSES[b.preset.min(assets::HOUSES.len() - 1)]
    };
    let scale = if b.is_town_center { TC_SCALE } else { HOUSE_SCALE };
    let rows = preset.grid.len() as f32;
    let cols = preset.grid.iter().map(|r| r.len()).max().unwrap_or(1) as f32;
    let w = cols * assets::TILE * scale;
    let h = rows * assets::TILE * scale;
    let left = b.pos.x - w / 2.0;
    let top = b.pos.y - h;

    // Soft drop shadow.
    d.draw_ellipse(
        b.pos.x as i32,
        b.pos.y as i32,
        w * 0.42,
        h * 0.10,
        Color::new(0, 0, 0, 60),
    );

    let tint = if b.live {
        Color::WHITE
    } else {
        Color::new(225, 225, 235, 255)
    };
    for (r, row) in preset.grid.iter().enumerate() {
        for (c, &idx) in row.iter().enumerate() {
            let x = left + c as f32 * assets::TILE * scale;
            let y = top + r as f32 * assets::TILE * scale;
            draw_tile(d, tex, idx, x, y, scale, tint);
        }
    }

    // Live buildings get a banner + rising smoke.
    if b.live {
        let fx = b.pos.x + w / 2.0 - 6.0;
        let fy = top - 8.0;
        draw_tile(d, tex, assets::FLAG, fx, fy, 1.4, Color::WHITE);
        for k in 0..3 {
            let t = (b.smoke_t * 0.9 + k as f32 * 0.6) % 1.0;
            let sx = left + w * 0.3 + (t * 12.0) * (k as f32 - 1.0) * 0.6;
            let sy = top - t * 26.0;
            let a = (140.0 * (1.0 - t)) as u8;
            d.draw_circle(sx as i32, sy as i32, 3.0 + t * 4.0, Color::new(210, 210, 220, a));
        }
    }
}

fn draw_villager<D: RaylibDraw>(d: &mut D, tex: &Texture2D, v: &Villager) {
    let bob = (v.anim_t.sin() * 2.0).abs();
    let s = assets::TILE * VILLAGER_SCALE;
    let x = v.pos.x - s / 2.0;
    let y = v.pos.y - s - bob;
    d.draw_ellipse(v.pos.x as i32, v.pos.y as i32, s * 0.32, s * 0.12, Color::new(0, 0, 0, 70));
    let tint = if v.on_trip {
        Color::new(255, 240, 210, 255)
    } else {
        Color::WHITE
    };
    draw_tile(d, tex, v.sprite, x, y, VILLAGER_SCALE, tint);
}

/// Paint the whole world inside the active 2D camera.
pub fn draw_world_space<D: RaylibDraw>(
    d: &mut D,
    world: &World,
    assets: &Assets,
    cam: &Camera2D,
    screen: (i32, i32),
    selected: Option<usize>,
) {
    // Visible world rectangle (with margin) so we only draw what's on screen.
    let inv = 1.0 / cam.zoom;
    let tl = Vector2::new(
        cam.target.x - cam.offset.x * inv,
        cam.target.y - cam.offset.y * inv,
    );
    let br = Vector2::new(
        tl.x + screen.0 as f32 * inv,
        tl.y + screen.1 as f32 * inv,
    );

    // Grass.
    let x0 = (tl.x / GROUND).floor() as i32 - 1;
    let x1 = (br.x / GROUND).ceil() as i32 + 1;
    let y0 = (tl.y / GROUND).floor() as i32 - 1;
    let y1 = (br.y / GROUND).ceil() as i32 + 1;
    for gy in y0..y1 {
        for gx in x0..x1 {
            let h = crate::util::hash64(&(gx, gy));
            let idx = if h % 22 == 0 {
                assets::GRASS[1 + (h as usize >> 4) % 2]
            } else {
                assets::GRASS[0]
            };
            // Olive-mute the field so bright-green trees, buildings and units pop.
            draw_tile(d, &assets.town, idx, gx as f32 * GROUND, gy as f32 * GROUND, GROUND / assets::TILE, Color::new(176, 190, 150, 255));
        }
    }

    // Sparse wilderness trees on a coarse grid (kept clear of city centers).
    let step = 96.0;
    let tx0 = (tl.x / step).floor() as i32 - 1;
    let tx1 = (br.x / step).ceil() as i32 + 1;
    let ty0 = (tl.y / step).floor() as i32 - 1;
    let ty1 = (br.y / step).ceil() as i32 + 1;
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            let h = crate::util::hash64(&(tx, ty, 7u8));
            if h % 5 != 0 {
                continue;
            }
            let px = tx as f32 * step + (h % 53) as f32;
            let py = ty as f32 * step + ((h >> 8) % 53) as f32;
            let p = Vector2::new(px, py);
            if world.cities.iter().any(|c| c.pos.distance_to(p) < 150.0) {
                continue;
            }
            // Mostly trees, with the odd mushroom patch or rock for variety.
            let (deco, sc) = match (h >> 5) % 9 {
                0 => (assets::MUSHROOMS, 2.0),
                1 => (assets::STONES, 2.2),
                _ => (assets::TREES[(h as usize >> 3) % assets::TREES.len()], 2.5),
            };
            let tw = assets::TILE * sc;
            // Shadow grounds the sprite so it reads against same-green grass.
            d.draw_ellipse((px + tw / 2.0) as i32, py as i32, tw * 0.34, tw * 0.12, Color::new(0, 0, 0, 55));
            draw_tile(d, &assets.town, deco, px, py - tw, sc, Color::WHITE);
        }
    }

    // Cities, back-to-front by y.
    let mut order: Vec<usize> = (0..world.cities.len()).collect();
    order.sort_by(|&a, &b| {
        world.cities[a].pos.y.partial_cmp(&world.cities[b].pos.y).unwrap_or(std::cmp::Ordering::Equal)
    });
    for ci in order {
        draw_city(d, &world.cities[ci], assets, world.now, selected == Some(ci));
    }

    let _ = cam;
}

fn draw_city<D: RaylibDraw>(d: &mut D, city: &City, assets: &Assets, _now: f64, selected: bool) {
    // Town plaza: a soft dirt disc under the settlement.
    let radius = 96.0 + (city.buildings.len() as f32).sqrt() * 24.0;
    d.draw_ellipse(
        city.pos.x as i32,
        (city.pos.y + 56.0) as i32,
        radius,
        radius * 0.46,
        Color::new(120, 96, 64, 70),
    );
    if selected {
        d.draw_ellipse_lines(
            city.pos.x as i32,
            (city.pos.y + 56.0) as i32,
            radius + 6.0,
            (radius + 6.0) * 0.46,
            GOLD,
        );
    }
    if city.live > 0 {
        // Gentle activity aura.
        d.draw_ellipse(
            city.pos.x as i32,
            (city.pos.y + 56.0) as i32,
            radius * 0.9,
            radius * 0.42,
            Color::new(255, 200, 90, 26),
        );
    }

    // A little market stall in the plaza — the trade hub villagers head for.
    let ms = 2.2;
    let mw = assets::TILE * ms;
    let mx = city.pos.x - radius * 0.46;
    let my = city.pos.y + 78.0;
    d.draw_ellipse(mx as i32, my as i32, mw * 0.34, mw * 0.12, Color::new(0, 0, 0, 50));
    draw_tile(d, &assets.town, assets::MARKET, mx - mw / 2.0, my - mw, ms, Color::WHITE);

    for b in &city.buildings {
        draw_building(d, &assets.town, b);
    }
    let mut vs: Vec<&Villager> = city.villagers.iter().collect();
    vs.sort_by(|a, b| a.pos.y.partial_cmp(&b.pos.y).unwrap_or(std::cmp::Ordering::Equal));
    for v in vs {
        draw_villager(d, &assets.chars, v);
    }
}

// ---- screen-space pass (labels + HUD) ---------------------------------------

/// Draw city name plates projected to screen space (crisp, constant size).
pub fn draw_labels<D: RaylibDraw>(
    d: &mut D,
    world: &World,
    cam: &Camera2D,
    screen: (i32, i32),
    selected: Option<usize>,
) {
    if cam.zoom < 0.32 {
        return; // too far out — labels would be noise.
    }
    for (i, c) in world.cities.iter().enumerate() {
        let s = world_to_screen(Vector2::new(c.pos.x, c.pos.y), cam);
        if s.x < -120.0 || s.x > screen.0 as f32 + 120.0 || s.y < -40.0 || s.y > screen.1 as f32 + 40.0 {
            continue;
        }
        let label = &c.name;
        let fs = 16;
        let w = text_w(label, fs) + 18;
        let x = s.x as i32 - w / 2;
        let y = (s.y - (TC_SCALE * assets::TILE * 2.0) - 30.0) as i32;
        let sel = selected == Some(i);
        d.draw_rectangle_rounded(
            Rectangle::new(x as f32, y as f32, w as f32, 22.0),
            0.5,
            8,
            if sel { Color::new(60, 52, 24, 235) } else { PANEL_BG },
        );
        if c.live > 0 {
            d.draw_circle(x + 10, y + 11, 4.0, GOLD);
            d.draw_text(label, x + 18, y + 4, fs, Color::WHITE);
        } else {
            d.draw_text(label, x + 9, y + 4, fs, TEXT_DIM);
        }
        // Tiny stat line beneath the name.
        let stat = format!("{} sess  {} msg", c.sessions.len(), c.total_messages);
        let sw = text_w(&stat, 10);
        d.draw_text(&stat, s.x as i32 - sw / 2, y + 24, 10, Color::new(150, 158, 178, 230));
    }
}

/// Top status bar + controls hint.
pub fn draw_hud<D: RaylibDraw>(
    d: &mut D,
    world: &World,
    source_name: &str,
    loading: bool,
    fps: u32,
    screen: (i32, i32),
) {
    let total_sessions: usize = world.cities.iter().map(|c| c.sessions.len()).sum();
    let total_msgs: u32 = world.cities.iter().map(|c| c.total_messages).sum();
    let live_cities = world.cities.iter().filter(|c| c.live > 0).count();
    let live_sessions: usize = world.cities.iter().map(|c| c.live).sum();

    d.draw_rectangle(0, 0, screen.0, 34, PANEL_BG);
    d.draw_line(0, 34, screen.0, 34, PANEL_LINE);
    d.draw_text("AGE OF MODELS", 12, 9, 18, GOLD);

    let info = format!(
        "source: {}   cities: {}   sessions: {}   messages: {}",
        source_name,
        world.cities.len(),
        total_sessions,
        total_msgs,
    );
    d.draw_text(&info, 190, 11, 15, TEXT_DIM);

    let live = format!("LIVE  {} cities / {} sessions", live_cities, live_sessions);
    let lw = text_w(&live, 15) + 24;
    d.draw_circle(screen.0 - lw - 6, 17, 5.0, if live_sessions > 0 { GOLD } else { Color::new(90, 96, 110, 255) });
    d.draw_text(&live, screen.0 - lw + 4, 11, 15, if live_sessions > 0 { Color::WHITE } else { TEXT_DIM });

    // Bottom hint bar.
    let hint = if loading {
        "scanning sessions…".to_string()
    } else {
        format!("{fps} fps   ·   drag / WASD pan   ·   wheel zoom   ·   click a city   ·   F frame all   ·   ESC quit")
    };
    d.draw_rectangle(0, screen.1 - 26, screen.0, 26, PANEL_BG);
    d.draw_line(0, screen.1 - 26, screen.0, screen.1 - 26, PANEL_LINE);
    d.draw_text(&hint, 12, screen.1 - 19, 13, TEXT_DIM);
}

/// Right-hand inspector for the selected city: its sessions, models, activity.
pub fn draw_inspector<D: RaylibDraw>(d: &mut D, city: &City, now: f64, screen: (i32, i32)) {
    let pw = 340;
    let px = screen.0 - pw;
    let py = 34;
    let ph = screen.1 - 34 - 26;
    d.draw_rectangle(px, py, pw, ph, PANEL_BG);
    d.draw_line(px, py, px, py + ph, PANEL_LINE);

    let mut y = py + 14;
    d.draw_text(&city.name, px + 14, y, 22, Color::WHITE);
    y += 28;
    if let Some(path) = &city.path {
        let shown = ellipsize(path, 44);
        d.draw_text(&shown, px + 14, y, 12, Color::new(130, 138, 158, 255));
        y += 20;
    }
    let summary = format!(
        "{} sessions   {} live   {} msgs   {} tools",
        city.sessions.len(),
        city.live,
        city.total_messages,
        city.total_tools,
    );
    d.draw_text(&summary, px + 14, y, 13, TEXT_DIM);
    y += 22;
    d.draw_line(px + 14, y, px + pw - 14, y, PANEL_LINE);
    y += 10;

    for s in city.sessions.iter().take(16) {
        let live = s.is_live(now, LIVE_WINDOW);
        d.draw_circle(px + 20, y + 7, 4.0, if live { GOLD } else { Color::new(96, 102, 116, 255) });
        let title = s.title.clone().unwrap_or_else(|| s.id.chars().take(8).collect());
        d.draw_text(&ellipsize(&title, 36), px + 32, y, 14, if live { Color::WHITE } else { TEXT_DIM });
        y += 18;
        let mc = model_color(s.model.as_deref());
        d.draw_text(short_model(s.model.as_deref()), px + 32, y, 11, mc);
        let meta = format!(
            "{} msg · {} tools · {}",
            s.total_messages(),
            s.tool_uses,
            ago(s.idle_secs(now)),
        );
        d.draw_text(&meta, px + 90, y, 11, Color::new(140, 148, 168, 255));
        y += 22;
        if y > py + ph - 30 {
            break;
        }
    }
    let more = city.sessions.len().saturating_sub(16);
    if more > 0 {
        d.draw_text(&format!("+{more} more…"), px + 32, y, 12, TEXT_DIM);
    }
}

fn ellipsize(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn ago(secs: f64) -> String {
    if !secs.is_finite() {
        return "—".into();
    }
    let s = secs as u64;
    if s < 60 {
        format!("{s}s ago")
    } else if s < 3600 {
        format!("{}m ago", s / 60)
    } else if s < 86400 {
        format!("{}h ago", s / 3600)
    } else {
        format!("{}d ago", s / 86400)
    }
}

pub fn clear_bg<D: RaylibDraw>(d: &mut D) {
    d.clear_background(SKY);
}

// ---- building hover tooltip --------------------------------------------------

/// World-space bounding box of a building sprite (bottom-anchored at `pos`).
pub fn building_bounds(b: &Building) -> Rectangle {
    let preset = if b.is_town_center {
        &assets::TOWN_CENTER
    } else {
        &assets::HOUSES[b.preset.min(assets::HOUSES.len() - 1)]
    };
    let scale = if b.is_town_center { TC_SCALE } else { HOUSE_SCALE };
    let rows = preset.grid.len() as f32;
    let cols = preset.grid.iter().map(|r| r.len()).max().unwrap_or(1) as f32;
    let w = cols * assets::TILE * scale;
    let h = rows * assets::TILE * scale;
    Rectangle::new(b.pos.x - w / 2.0, b.pos.y - h, w, h)
}

/// Find the front-most building under a world point.
pub fn pick_building(world: &World, p: Vector2) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, f32)> = None;
    for (ci, c) in world.cities.iter().enumerate() {
        for (bi, b) in c.buildings.iter().enumerate() {
            let r = building_bounds(b);
            if p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height {
                if best.map_or(true, |(_, _, by)| b.pos.y > by) {
                    best = Some((ci, bi, b.pos.y));
                }
            }
        }
    }
    best.map(|(c, b, _)| (c, b))
}

/// A small panel by the cursor describing the hovered building / session.
pub fn draw_building_tooltip<D: RaylibDraw>(
    d: &mut D,
    city: &City,
    b: &Building,
    now: f64,
    mouse: Vector2,
    screen: (i32, i32),
) {
    let header = if b.is_town_center {
        format!("{}  (town center)", city.name)
    } else {
        b.title.clone().unwrap_or_else(|| "session".to_string())
    };
    let header = ellipsize(&header, 40);
    let model = short_model(b.model.as_deref());
    let line2 = format!("{}  ·  {} msgs", model, b.messages);

    let w = (text_w(&header, 15).max(text_w(&line2, 12)) + 24).max(150);
    let h = 50;
    let mut x = mouse.x as i32 + 16;
    let mut y = mouse.y as i32 + 16;
    if x + w > screen.0 - 8 {
        x = mouse.x as i32 - w - 16;
    }
    if y + h > screen.1 - 30 {
        y = mouse.y as i32 - h - 16;
    }

    d.draw_rectangle_rounded(Rectangle::new(x as f32, y as f32, w as f32, h as f32), 0.18, 8, PANEL_BG);
    d.draw_rectangle_rounded_lines(Rectangle::new(x as f32, y as f32, w as f32, h as f32), 0.18, 8, PANEL_LINE);
    if b.live {
        d.draw_circle(x + 12, y + 14, 4.0, GOLD);
        d.draw_text(&header, x + 22, y + 7, 15, Color::WHITE);
    } else {
        d.draw_text(&header, x + 10, y + 7, 15, TEXT_DIM);
    }
    d.draw_text(&line2, x + 10, y + 27, 12, model_color(b.model.as_deref()));
    let _ = now;
}
