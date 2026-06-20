//! Rendering: draws the world with raylib. A camera-space pass paints terrain,
//! buildings and villagers; a screen-space pass paints labels, the HUD, and the
//! selected-city inspector.

pub mod assets;

use crate::achievements::Cat;
use crate::data::Tool;
use crate::game::terrain::{self, Land, River};
use crate::game::{Biome, Building, City, Road, Season, Villager, World, LIVE_WINDOW};
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

/// Component-wise multiply (used to apply season/biome tints over a base tint).
fn mul_color(a: Color, b: Color) -> Color {
    Color::new(
        ((a.r as u16 * b.r as u16) / 255) as u8,
        ((a.g as u16 * b.g as u16) / 255) as u8,
        ((a.b as u16 * b.b as u16) / 255) as u8,
        a.a,
    )
}

/// Multiplicative tint applied to a whole city by how recently it was worked in.
fn season_tint(s: Season) -> Color {
    match s {
        Season::HighSummer => Color::new(255, 255, 255, 255),
        Season::Summer => Color::new(248, 250, 245, 255),
        Season::LateSummer => Color::new(248, 240, 220, 255),
        Season::Autumn => Color::new(236, 198, 158, 255),
        Season::Winter => Color::new(206, 216, 236, 255),
        Season::Dormant => Color::new(150, 156, 168, 255),
    }
}

/// The ground patch colour for a biome (drawn under the settlement).
fn biome_ground(b: Biome) -> Color {
    match b {
        Biome::Forge => Color::new(150, 120, 96, 150),
        Biome::Coast => Color::new(228, 214, 168, 150),
        Biome::Forest => Color::new(70, 120, 70, 150),
        Biome::Port => Color::new(120, 160, 96, 150),
        Biome::Steppe => Color::new(176, 168, 96, 150),
        Biome::Stone => Color::new(140, 146, 140, 150),
        Biome::Plains => Color::new(120, 160, 130, 150),
        Biome::Vale => Color::new(206, 190, 150, 150),
        Biome::Heartland => Color::new(120, 150, 96, 150),
    }
}

/// The decoration tiles scattered around a biome's settlement (Tiny Town tiles).
fn biome_decor(b: Biome) -> &'static [i32] {
    match b {
        Biome::Forge => &[assets::STONES, 43, 5],
        Biome::Coast => &[28, 5],
        Biome::Forest => &[5, 4, 28, assets::MUSHROOMS],
        Biome::Port => &[57, 5, 28],
        Biome::Steppe => &[27, 28, 5],
        Biome::Stone => &[assets::STONES, 43],
        Biome::Plains => &[28, 5],
        Biome::Vale => &[assets::SIGNPOST, 5, 28],
        Biome::Heartland => &[5, 4, 28],
    }
}

/// Specialized building prop for a session's dominant tool.
/// Returns `(use_chars_atlas, tile, scale)`.
fn tool_prop(t: Tool) -> Option<(bool, i32, f32)> {
    match t {
        Tool::Bash => Some((true, assets::ANVIL, 1.6)),   // forge
        Tool::Read => Some((false, assets::SIGNPOST, 1.7)), // library
        Tool::Search => Some((false, assets::SIGNPOST, 1.5)),
        Tool::Task => Some((true, assets::BANNER, 1.7)),  // barracks
        Tool::Web => Some((false, assets::MARKET, 1.7)),  // harbor / trade
        Tool::Plan => Some((true, assets::POTION, 1.3)),
        Tool::Edit => None, // plain workshop
    }
}

/// The monument icon (Tiny Dungeon tile) representing an achievement category.
fn monument_icon(c: Cat) -> i32 {
    match c {
        Cat::Activity => assets::BANNER,
        Cat::Craft => assets::ANVIL,
        Cat::Codebase => assets::CHEST,
        Cat::Mastery => assets::SWORD,
        Cat::Time => assets::POTION,
        Cat::Wealth => assets::COINS,
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

/// The preset + scale a building draws at (town center grows with tier; below a
/// keep, the "town center" is just the largest hut).
fn building_preset_scale(b: &Building) -> (&'static assets::Building<'static>, f32) {
    if b.is_town_center {
        if b.tier.has_keep() {
            (&assets::TOWN_CENTER, b.tier.keep_scale())
        } else {
            (&assets::HOUSES[0], b.tier.keep_scale())
        }
    } else {
        (&assets::HOUSES[b.preset.min(assets::HOUSES.len() - 1)], HOUSE_SCALE)
    }
}

/// Draw a building's tile grid, bottom-center anchored at `base`, tinted by season.
fn draw_building<D: RaylibDraw>(d: &mut D, assets: &Assets, b: &Building, season: Color) {
    let (preset, scale) = building_preset_scale(b);
    let rows = preset.grid.len() as f32;
    let cols = preset.grid.iter().map(|r| r.len()).max().unwrap_or(1) as f32;
    let w = cols * assets::TILE * scale;
    let h = rows * assets::TILE * scale;
    let left = b.pos.x - w / 2.0;
    let top = b.pos.y - h;

    d.draw_ellipse(b.pos.x as i32, b.pos.y as i32, w * 0.42, h * 0.10, Color::new(0, 0, 0, 60));

    let base = if b.live { Color::WHITE } else { Color::new(225, 225, 235, 255) };
    let tint = mul_color(base, season);
    for (r, row) in preset.grid.iter().enumerate() {
        for (c, &idx) in row.iter().enumerate() {
            let x = left + c as f32 * assets::TILE * scale;
            let y = top + r as f32 * assets::TILE * scale;
            draw_tile(d, &assets.town, idx, x, y, scale, tint);
        }
    }

    // Tool-type prop tells you what kind of work happens here (forge/library/…).
    if !b.is_town_center {
        if let Some((chars, prop, psc)) = tool_prop(b.tool) {
            let pw = assets::TILE * psc;
            let px = b.pos.x + w * 0.30 - pw / 2.0;
            let py = b.pos.y - 1.0;
            let atlas = if chars { &assets.chars } else { &assets.town };
            d.draw_ellipse(px as i32 + (pw / 2.0) as i32, py as i32, pw * 0.3, pw * 0.1, Color::new(0, 0, 0, 50));
            draw_tile(d, atlas, prop, px, py - pw, psc, tint);
        }
    }

    // Live buildings get a banner + rising smoke.
    if b.live {
        let fx = b.pos.x + w / 2.0 - 6.0;
        let fy = top - 8.0;
        draw_tile(d, &assets.town, assets::FLAG, fx, fy, 1.4, Color::WHITE);
        for k in 0..3 {
            let t = (b.smoke_t * 0.9 + k as f32 * 0.6) % 1.0;
            let sx = left + w * 0.3 + (t * 12.0) * (k as f32 - 1.0) * 0.6;
            let sy = top - t * 26.0;
            let a = (140.0 * (1.0 - t)) as u8;
            d.draw_circle(sx as i32, sy as i32, 3.0 + t * 4.0, Color::new(210, 210, 220, a));
        }
    }
}

fn draw_villager<D: RaylibDraw>(d: &mut D, tex: &Texture2D, v: &Villager, season: Color) {
    let bob = (v.anim_t.sin() * 2.0).abs();
    let s = assets::TILE * VILLAGER_SCALE;
    let x = v.pos.x - s / 2.0;
    let y = v.pos.y - s - bob;
    d.draw_ellipse(v.pos.x as i32, v.pos.y as i32, s * 0.32, s * 0.12, Color::new(0, 0, 0, 70));
    let base = if v.on_trip { Color::new(255, 240, 210, 255) } else { Color::WHITE };
    draw_tile(d, tex, v.sprite, x, y, VILLAGER_SCALE, mul_color(base, season));
}

// ---- terrain helpers ---------------------------------------------------------

fn lu(a: i32, b: i32, t: f32) -> i32 {
    (a as f32 + (b - a) as f32 * t.clamp(0.0, 1.0)) as i32
}
fn cu(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Flat ground colour for a land type at height `h`, with a small per-cell jitter.
fn land_color(land: Land, h: f32, jit: i32) -> Color {
    let (r, g, b) = match land {
        Land::Sea => {
            let t = 1.0 - ((0.33 - h) / 0.33).clamp(0.0, 1.0); // shallower -> lighter
            (lu(28, 62, t), lu(68, 124, t), lu(114, 172, t))
        }
        Land::Sand => (226, 208, 152),
        Land::Grass => {
            let t = (h - 0.37) / 0.21;
            (lu(104, 150, t), lu(158, 168, t), lu(76, 98, t))
        }
        Land::Hill => {
            let t = (h - 0.58) / 0.12;
            (lu(120, 150, t), lu(150, 132, t), lu(80, 84, t))
        }
        Land::Mountain => {
            let t = (h - 0.70) / 0.14;
            (lu(144, 118, t), lu(140, 116, t), lu(132, 112, t))
        }
        Land::Snow => (236, 240, 248),
    };
    Color::new(cu(r + jit), cu(g + jit), cu(b + jit), 255)
}

fn rect_visible(p: Vector2, tl: Vector2, br: Vector2, pad: f32) -> bool {
    p.x >= tl.x - pad && p.x <= br.x + pad && p.y >= tl.y - pad && p.y <= br.y + pad
}
fn seg_visible(a: Vector2, b: Vector2, tl: Vector2, br: Vector2, pad: f32) -> bool {
    let minx = a.x.min(b.x) - pad;
    let maxx = a.x.max(b.x) + pad;
    let miny = a.y.min(b.y) - pad;
    let maxy = a.y.max(b.y) + pad;
    maxx >= tl.x && minx <= br.x && maxy >= tl.y && miny <= br.y
}

/// A shaded mountain peak with an optional snow cap, base-anchored at `(cx, base)`.
fn draw_peak<D: RaylibDraw>(d: &mut D, cx: f32, base: f32, size: f32, snow: bool) {
    let half = size * 0.62;
    let apex = Vector2::new(cx, base - size);
    let bl = Vector2::new(cx - half, base);
    let br = Vector2::new(cx + half, base);
    let bm = Vector2::new(cx, base);
    d.draw_ellipse(cx as i32, base as i32, half, half * 0.26, Color::new(0, 0, 0, 50));
    // Two faces so peaks have a lit and a shaded side.
    d.draw_triangle(apex, bl, bm, Color::new(152, 148, 142, 255));
    d.draw_triangle(apex, bm, br, Color::new(108, 104, 100, 255));
    if snow {
        let sy = base - size * 0.6;
        let sh = half * 0.42;
        d.draw_triangle(apex, Vector2::new(cx - sh, sy), Vector2::new(cx + sh, sy), Color::new(238, 242, 250, 255));
    }
}

fn draw_river<D: RaylibDraw>(d: &mut D, r: &River, tl: Vector2, br: Vector2) {
    let pad = r.width * 2.0;
    let deep = Color::new(56, 114, 168, 255);
    let lite = Color::new(98, 160, 208, 255);
    for (col, scale) in [(deep, 1.0_f32), (lite, 0.52)] {
        for w in r.points.windows(2) {
            if seg_visible(w[0], w[1], tl, br, pad) {
                d.draw_line_ex(w[0], w[1], r.width * scale, col);
            }
        }
        for &p in &r.points {
            if rect_visible(p, tl, br, pad) {
                d.draw_circle_v(p, r.width * scale * 0.5, col);
            }
        }
    }
}

fn ribbon<D: RaylibDraw>(d: &mut D, pts: &[Vector2], width: f32, col: Color, tl: Vector2, br: Vector2) {
    for seg in pts.windows(2) {
        if seg_visible(seg[0], seg[1], tl, br, width * 2.0) {
            d.draw_line_ex(seg[0], seg[1], width, col);
        }
    }
    for &p in pts {
        if rect_visible(p, tl, br, width * 2.0) {
            d.draw_circle_v(p, width * 0.5, col);
        }
    }
}

fn draw_road<D: RaylibDraw>(d: &mut D, road: &Road, tl: Vector2, br: Vector2) {
    let w = 16.0;
    // Continuous dirt road (dark edge then lighter fill).
    ribbon(d, &road.points, w, Color::new(92, 70, 46, 255), tl, br);
    ribbon(d, &road.points, w * 0.66, Color::new(178, 144, 98, 255), tl, br);

    // Lay a plank bridge deck over each run of water crossings.
    let n = road.points.len();
    let mut s = 0;
    while s < n {
        if road.water[s] {
            let mut e = s;
            while e + 1 < n && road.water[e + 1] {
                e += 1;
            }
            // Extend one point onto the banks so the deck reaches solid ground.
            let lo = s.saturating_sub(1);
            let hi = (e + 1).min(n - 1);
            let span = &road.points[lo..=hi];
            ribbon(d, span, w + 12.0, Color::new(74, 52, 32, 255), tl, br); // rails / underside
            ribbon(d, span, w + 4.0, Color::new(168, 128, 82, 255), tl, br); // planks
            ribbon(d, span, w * 0.4, Color::new(120, 90, 56, 255), tl, br); // seam
            s = e + 1;
        } else {
            s += 1;
        }
    }
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

    let terrain = &world.terrain;

    // --- terrain ground: heightfield bands (sea/sand/grass/hill/mountain/snow) -
    let step = GROUND.max(5.0 / cam.zoom); // keep cells >= ~5px on screen
    let gx0 = (tl.x / step).floor() as i32 - 1;
    let gx1 = (br.x / step).ceil() as i32 + 1;
    let gy0 = (tl.y / step).floor() as i32 - 1;
    let gy1 = (br.y / step).ceil() as i32 + 1;
    let cell = step.ceil() as i32 + 1;
    for gy in gy0..gy1 {
        for gx in gx0..gx1 {
            let wx = gx as f32 * step;
            let wy = gy as f32 * step;
            let h = terrain.height(wx + step * 0.5, wy + step * 0.5);
            let land = terrain::classify(h);
            let jit = (crate::util::hash64(&(gx, gy)) % 13) as i32 - 6;
            d.draw_rectangle(wx.floor() as i32, wy.floor() as i32, cell, cell, land_color(land, h, jit));
        }
    }

    // --- mountain peaks (coarse grid over high ground) ------------------------
    let ms = 70.0_f32.max(9.0 / cam.zoom);
    let px0 = (tl.x / ms).floor() as i32 - 1;
    let px1 = (br.x / ms).ceil() as i32 + 1;
    let py0 = (tl.y / ms).floor() as i32 - 1;
    let py1 = (br.y / ms).ceil() as i32 + 1;
    for my in py0..py1 {
        for mx in px0..px1 {
            let hh = crate::util::hash64(&(mx, my, 31u8));
            let wx = mx as f32 * ms + (hh % 37) as f32;
            let wy = my as f32 * ms + ((hh >> 8) % 37) as f32;
            let h = terrain.height(wx, wy);
            if h > 0.70 {
                let size = 22.0 + (h - 0.70) * 230.0 + (hh % 11) as f32;
                draw_peak(d, wx, wy, size, h > 0.84);
            }
        }
    }

    // --- rivers, then roads + bridges -----------------------------------------
    for r in terrain.rivers() {
        draw_river(d, r, tl, br);
    }
    for road in &world.roads {
        draw_road(d, road, tl, br);
    }

    // --- countryside: trees on grass/hills, rocks on the mountain feet ---------
    let sstep = 88.0_f32.max(11.0 / cam.zoom);
    let sx0 = (tl.x / sstep).floor() as i32 - 1;
    let sx1 = (br.x / sstep).ceil() as i32 + 1;
    let sy0 = (tl.y / sstep).floor() as i32 - 1;
    let sy1 = (br.y / sstep).ceil() as i32 + 1;
    for ty in sy0..sy1 {
        for tx in sx0..sx1 {
            let h = crate::util::hash64(&(tx, ty, 7u8));
            let px = tx as f32 * sstep + (h % 61) as f32;
            let py = ty as f32 * sstep + ((h >> 8) % 61) as f32;
            let p = Vector2::new(px, py);
            let land = terrain.land_at(px, py);
            let chance = match land {
                Land::Hill => 2,   // denser woods on the hills
                Land::Grass => 5,
                Land::Mountain => 7,
                _ => 0,
            };
            if chance == 0 || h % chance != 0 {
                continue;
            }
            if world.cities.iter().any(|c| c.pos.distance_to(p) < 150.0) || terrain.river_dist(p) < 40.0 {
                continue;
            }
            let (deco, sc) = if land == Land::Mountain {
                (assets::STONES, 2.0)
            } else if (h >> 5) % 8 == 0 {
                (assets::MUSHROOMS, 1.9)
            } else {
                (assets::TREES[(h as usize >> 3) % assets::TREES.len()], 2.4)
            };
            let tw = assets::TILE * sc;
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
    let cx = city.pos.x;
    let cy = city.pos.y + 56.0;
    let season = season_tint(city.season);
    // Bigger cities spread wider.
    let radius = 84.0 + (city.buildings.len() as f32).sqrt() * 22.0 + city.tier.index() as f32 * 10.0;

    // Biome-coloured plaza, tinted by season.
    d.draw_ellipse(cx as i32, cy as i32, radius, radius * 0.46, mul_color(biome_ground(city.biome), season));
    if selected {
        d.draw_ellipse_lines(cx as i32, cy as i32, radius + 6.0, (radius + 6.0) * 0.46, GOLD);
        d.draw_ellipse_lines(cx as i32, cy as i32, radius + 8.0, (radius + 8.0) * 0.46, GOLD);
    }
    if city.live > 0 {
        d.draw_ellipse(cx as i32, cy as i32, radius * 0.9, radius * 0.42, Color::new(255, 200, 90, 26));
    }

    // Fortifications: a fence ring (village+) becomes a stone wall (city+).
    draw_walls(d, assets, city, cx, cy, radius, season);

    // Biome decorations scattered around the settlement edge.
    let decor = biome_decor(city.biome);
    let n_decor = 5 + city.tier.index() * 2;
    for k in 0..n_decor {
        let mut rng = crate::util::Rng::seeded(&(&city.id, k, "decor"));
        let ang = rng.range(0.0, 6.2831);
        let rr = radius * rng.range(0.82, 1.12);
        let px = cx + ang.cos() * rr;
        let py = cy + ang.sin() * rr * 0.5;
        let tile = decor[rng.below(decor.len())];
        let sc = 2.0;
        let tw = assets::TILE * sc;
        d.draw_ellipse((px) as i32, py as i32, tw * 0.3, tw * 0.1, Color::new(0, 0, 0, 45));
        draw_tile(d, &assets.town, tile, px - tw / 2.0, py - tw, sc, season);
    }

    // Market stall — the trade hub villagers head for.
    let ms = 2.2;
    let mw = assets::TILE * ms;
    let mx = cx - radius * 0.46;
    let my = city.pos.y + 78.0;
    d.draw_ellipse(mx as i32, my as i32, mw * 0.34, mw * 0.12, Color::new(0, 0, 0, 50));
    draw_tile(d, &assets.town, assets::MARKET, mx - mw / 2.0, my - mw, ms, season);

    for b in &city.buildings {
        draw_building(d, assets, b, season);
    }
    let mut vs: Vec<&Villager> = city.villagers.iter().collect();
    vs.sort_by(|a, b| a.pos.y.partial_cmp(&b.pos.y).unwrap_or(std::cmp::Ordering::Equal));
    for v in vs {
        draw_villager(d, &assets.chars, v, season);
    }

    // Monuments: one per unlocked achievement, in an arc behind the town.
    draw_monuments(d, assets, city, cx, cy, radius);
}

/// A ring of fortifications whose material upgrades with tier.
fn draw_walls<D: RaylibDraw>(d: &mut D, assets: &Assets, city: &City, cx: f32, cy: f32, radius: f32, season: Color) {
    if city.tier.index() < 2 {
        return; // outposts/hamlets are unwalled
    }
    let (tile, sc) = if city.tier.has_walls() {
        (assets::WALL, 1.8) // stone wall
    } else {
        (assets::SIGNPOST, 1.2) // light fence posts (reuse a slim tile)
    };
    let segs = 14 + city.tier.index() * 3;
    let tw = assets::TILE * sc;
    for k in 0..segs {
        let ang = k as f32 / segs as f32 * 6.2831;
        let px = cx + ang.cos() * (radius + 6.0);
        let py = cy + ang.sin() * (radius + 6.0) * 0.5;
        draw_tile(d, &assets.town, tile, px - tw / 2.0, py - tw, sc, mul_color(Color::new(235, 235, 240, 255), season));
    }
}

/// Unlocked achievements rendered as a tidy "trophy shelf" in front of the city,
/// so they read clearly without colliding with the buildings.
fn draw_monuments<D: RaylibDraw>(d: &mut D, assets: &Assets, city: &City, cx: f32, cy: f32, radius: f32) {
    let shown = city.achievements.len().min(10);
    if shown == 0 {
        return;
    }
    let sc = 1.5;
    let tw = assets::TILE * sc;
    let spacing = tw * 1.05;
    let start_x = cx - (shown as f32 - 1.0) * spacing * 0.5;
    let py = cy + radius * 0.52 + tw * 0.5;
    for (i, ach) in city.achievements.iter().take(shown).enumerate() {
        let px = start_x + i as f32 * spacing;
        d.draw_ellipse(px as i32, py as i32, tw * 0.4, tw * 0.16, Color::new(0, 0, 0, 70));
        d.draw_circle(px as i32, (py - tw * 0.5) as i32, tw * 0.52, Color::new(255, 210, 110, 40));
        draw_tile(d, &assets.chars, monument_icon(ach.cat), px - tw / 2.0, py - tw, sc, Color::WHITE);
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
        // Tiny stat line beneath the name: age tier + session count.
        let stat = format!("{} · {} sess", c.tier.name(), c.sessions.len());
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
        "scanning sessions...".to_string()
    } else {
        format!("{fps} fps   ·   drag / WASD pan   ·   wheel zoom   ·   click a city   ·   F frame all   ·   ESC quit")
    };
    d.draw_rectangle(0, screen.1 - 26, screen.0, 26, PANEL_BG);
    d.draw_line(0, screen.1 - 26, screen.0, screen.1 - 26, PANEL_LINE);
    d.draw_text(&hint, 12, screen.1 - 19, 13, TEXT_DIM);
}

fn cat_color(c: Cat) -> Color {
    match c {
        Cat::Activity => Color::new(255, 196, 64, 255),
        Cat::Craft => Color::new(255, 150, 90, 255),
        Cat::Codebase => Color::new(120, 220, 140, 255),
        Cat::Mastery => Color::new(196, 148, 255, 255),
        Cat::Time => Color::new(120, 200, 255, 255),
        Cat::Wealth => Color::new(240, 220, 120, 255),
    }
}

/// Group thousands with commas: 1234567 -> "1,234,567".
fn commafy(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    let len = s.len();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Right-hand inspector for the selected city: civilization, codebase, resources,
/// achievements, and recent sessions.
pub fn draw_inspector<D: RaylibDraw>(d: &mut D, city: &City, now: f64, screen: (i32, i32)) {
    let pw = 384;
    let px = screen.0 - pw;
    let py = 34;
    let ph = screen.1 - 34 - 26;
    let right = px + pw - 14;
    d.draw_rectangle(px, py, pw, ph, PANEL_BG);
    d.draw_line(px, py, px, py + ph, PANEL_LINE);

    let mut y = py + 12;
    d.draw_text(&city.name, px + 14, y, 22, Color::WHITE);
    y += 27;

    // Civilization line: tier · biome · season.
    let civ = format!("{} · {} · {}", city.tier.name(), city.biome.name(), city.season.name());
    d.draw_text(&civ, px + 14, y, 13, GOLD);
    y += 19;
    if let Some(path) = &city.path {
        d.draw_text(&ellipsize(path, 50), px + 14, y, 11, Color::new(124, 132, 152, 255));
        y += 17;
    }
    let summary = format!(
        "{} sessions · {} live · {} msgs · {} tools",
        city.sessions.len(),
        city.live,
        commafy(city.total_messages as u64),
        commafy(city.total_tools as u64),
    );
    d.draw_text(&summary, px + 14, y, 12, TEXT_DIM);
    y += 20;

    // Codebase facts.
    if let Some(cb) = &city.codebase {
        let langs: Vec<String> =
            cb.languages.iter().take(3).map(|(e, n)| format!(".{e} {n}")).collect();
        let line1 = format!(
            "code: {} LOC · {} files · {} commits",
            commafy(cb.loc),
            commafy(cb.files as u64),
            commafy(cb.commits as u64),
        );
        d.draw_text(&line1, px + 14, y, 12, Color::new(150, 200, 160, 255));
        y += 17;
        let mut flags = langs.join("  ");
        if cb.has_tests {
            flags.push_str("  +tests");
        }
        if cb.has_readme {
            flags.push_str("  +readme");
        }
        d.draw_text(&flags, px + 14, y, 11, Color::new(130, 160, 140, 255));
        y += 18;
    }

    // Resources: the four classic AoE piles, mapped to real metrics.
    let m = &city.metrics;
    let res = format!(
        "food {}   wood {}   gold {}   stone {}",
        commafy(m.messages as u64),
        commafy(m.tools.edit as u64),
        commafy(m.tokens),
        commafy(m.commits as u64),
    );
    d.draw_text(&res, px + 14, y, 12, Color::new(200, 190, 150, 255));
    y += 20;
    d.draw_line(px + 14, y, right, y, PANEL_LINE);
    y += 10;

    // Achievements as colour-coded badge pills, wrapped.
    let total = crate::achievements::CATALOG.len();
    d.draw_text(&format!("ACHIEVEMENTS  {}/{}", city.achievements.len(), total), px + 14, y, 13, Color::WHITE);
    y += 20;
    let mut bx = px + 14;
    let badge_bottom;
    {
        let mut row_y = y;
        for ach in &city.achievements {
            let w = text_w(ach.name, 11) + 16;
            if bx + w > right {
                bx = px + 14;
                row_y += 20;
            }
            if row_y > py + ph - 220 {
                d.draw_text("...", bx, row_y + 3, 12, TEXT_DIM);
                break;
            }
            let col = cat_color(ach.cat);
            d.draw_rectangle_rounded(
                Rectangle::new(bx as f32, row_y as f32, w as f32, 16.0),
                0.5,
                6,
                Color::new(col.r, col.g, col.b, 46),
            );
            d.draw_circle(bx + 7, row_y + 8, 3.0, col);
            d.draw_text(ach.name, bx + 13, row_y + 3, 11, Color::new(225, 228, 236, 255));
            bx += w + 5;
        }
        badge_bottom = row_y + 22;
    }
    y = badge_bottom.max(y);
    d.draw_line(px + 14, y, right, y, PANEL_LINE);
    y += 10;

    // Recent sessions.
    d.draw_text("SESSIONS", px + 14, y, 13, Color::new(200, 206, 222, 255));
    y += 20;
    for s in city.sessions.iter() {
        if y > py + ph - 26 {
            break;
        }
        let live = s.is_live(now, LIVE_WINDOW);
        d.draw_circle(px + 20, y + 7, 4.0, if live { GOLD } else { Color::new(96, 102, 116, 255) });
        let title = s.title.clone().unwrap_or_else(|| s.id.chars().take(8).collect());
        d.draw_text(&ellipsize(&title, 40), px + 32, y, 13, if live { Color::WHITE } else { TEXT_DIM });
        y += 17;
        d.draw_text(short_model(s.model.as_deref()), px + 32, y, 11, model_color(s.model.as_deref()));
        let meta = format!("{} msg · {} tools · {}", s.total_messages(), s.tool_uses, ago(s.idle_secs(now)));
        d.draw_text(&meta, px + 92, y, 11, Color::new(140, 148, 168, 255));
        y += 21;
    }
}

fn ellipsize(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(3)).collect();
        out.push_str("...");
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
    let (preset, scale) = building_preset_scale(b);
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
