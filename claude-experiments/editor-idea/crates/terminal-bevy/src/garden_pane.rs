//! "Claude Garden" pane — per-project pane, hand-drawn pixel-art
//! plants from the CC0 OpenGameArt "Flowers" sheet, project-clustered
//! by cwd, no autospawn.
//!
//! ## Per-project pane, shared event stream
//!
//! Each garden pane is tied to a Bevy project just like every other
//! pane kind — closing the garden in project A doesn't affect project
//! B, and only projects the user explicitly adds a garden to will
//! show one. There is no autospawn.
//!
//! The event stream is **global**: every Claude bus event is fanned
//! out to every garden in existence, regardless of which project's
//! Claude session emitted the event. So one pane can host plants from
//! many projects at once, visually grouped by `cwd` (hashed to an
//! x-cluster center + per-project hue offset).
//!
//! ## Pixel art
//!
//! Plants are pulled from a 78-tile CC0 sprite sheet
//! (`assets/garden/plants.png`, 12×24 px per tile) embedded at compile
//! time. Each tile is hand-drawn and varied — flowers, grass clumps,
//! vines, cattails, mushrooms, sprouts. On spawn a plant picks a
//! random tile (seeded by `plant.seed`), then we apply a per-plant
//! hue shift in HSV space so the same source tile yields different
//! colors per plant.
//!
//! Growth is continuous, not staged: each plant starts at `0.25 ×
//! target_scale` and ramps to its full `target_scale` over
//! `GROW_DURATION` seconds. `target_scale` itself is sampled from a
//! species-typical range (trees tall, grass short, seeds tiny), so
//! plant heights span a real range.
//!
//! ## Hot reload
//!
//! `~/.terminal-bevy/garden-art/` still holds two editable files:
//!   - `palette.txt` — `<char> <rgba hex>` per line. Only affects the
//!     butterfly today (plants come from the embedded sheet).
//!   - `butterfly.txt` — 32×32 pixel grid.
//! Edits are picked up live via a `notify` watcher.
//!
//! ## Persistence
//!
//! Every plant's `{ species, age, x, y_jitter, seed, plot_key,
//! hue_shift_deg, scale, sprite_idx }` lives in the pane snapshot.
//! Restored plants come back at the same x with the same sprite,
//! color, and growth progress. Butterflies are ephemeral by design.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use bevy::asset::RenderAssetUsages;
use bevy::image::{
    Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor,
};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::sprite::Anchor;
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use claude_bus_bevy::ClaudeBusEvent;
use pane_bevy::{
    PaneChrome, PaneKindSpec, PaneRect, PaneRegistry, PaneTitle, MARGIN, TITLE_H,
};

pub const PANE_KIND: &str = "claude_garden";

const PLANTS_PNG: &[u8] = include_bytes!("../assets/garden/plants.png");
const TILE_W: u32 = 12;
const TILE_H: u32 = 24;
const NUM_TILES: u32 = 78;

/// Native sprite size on screen at scale=1.0. 5× upscale of the source
/// tile so individual pixels read clearly. With per-plant scale jitter
/// 0.30-1.55, on-screen plant heights span ~36px (seeds) to ~186px
/// (tall trees) — a real range, visible without leaning in.
const SPRITE_DRAW_W: f32 = TILE_W as f32 * 5.0;
const SPRITE_DRAW_H: f32 = TILE_H as f32 * 5.0;

/// Butterfly is a separate, larger custom 32×32 sprite from
/// `butterfly.txt`.
const BUTTERFLY_W: u32 = 32;
const BUTTERFLY_H: u32 = 32;
const BUTTERFLY_DRAW_SIZE: f32 = BUTTERFLY_W as f32 * 1.5;

/// Seconds for a plant to grow from sprout to full size. Plants start
/// at 0.25× their target scale and ramp linearly to target over this
/// duration.
const GROW_DURATION: f32 = 8.0;
const PLANT_MIN_SCALE_FRAC: f32 = 0.25;

const MAX_PLANTS: usize = 120;
const GROUND_INSET: f32 = 18.0;
const GROUND_H: f32 = 5.0;
const PLANT_Y_JITTER: f32 = 3.0;
const BUTTERFLY_VX: f32 = 70.0;
const BUTTERFLY_WOBBLE_HZ: f32 = 1.4;
const BUTTERFLY_WOBBLE_PX: f32 = 12.0;

const PLOT_SPREAD_FRAC: f32 = 0.15;
const PLOT_SPREAD_MIN_PX: f32 = 40.0;
const PLOT_SPREAD_MAX_PX: f32 = 180.0;
const PER_PLANT_HUE_JITTER_DEG: f32 = 15.0;

const GROUND_COLOR: Color = Color::srgb(0.10, 0.18, 0.10);
const SKY_COLOR: Color = Color::srgb(0.07, 0.10, 0.16);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
enum Species {
    Flower,
    Vine,
    Grass,
    Tree,
    Seed,
}

// ============================================================
// Components / resources
// ============================================================

/// One pre-decoded tile from the sheet. Stored as raw RGBA8 bytes (with
/// chroma-key applied) so per-plant hue shifts can be cheaply applied
/// without re-decoding the PNG.
#[derive(Resource)]
struct GardenArt {
    tile_rgba: Vec<Vec<u8>>,
    butterfly: Handle<Image>,
    art_dir: PathBuf,
    palette: HashMap<char, [u8; 4]>,
}

#[derive(Resource)]
struct ArtWatcher {
    rx: Mutex<std::sync::mpsc::Receiver<PathBuf>>,
    _watcher: RecommendedWatcher,
}

#[derive(Serialize, Deserialize, Clone)]
struct PlantSnapshot {
    species: Species,
    /// Legacy field from the L-system era. Kept on disk so older
    /// snapshots still parse; ignored at runtime when `age > 0`.
    #[serde(default)]
    stage: u8,
    #[serde(default)]
    age: f32,
    x: f32,
    y_jitter: f32,
    #[serde(default)]
    seed: u32,
    #[serde(default)]
    plot_key: u64,
    #[serde(default)]
    hue_shift_deg: f32,
    #[serde(default = "default_scale")]
    scale: f32,
    /// Index into the embedded sheet (0..NUM_TILES). The `default`
    /// returns NUM_TILES as a "missing" sentinel; old snapshots with
    /// no sprite_idx are migrated at load by deriving from `seed`.
    #[serde(default = "default_sprite_idx")]
    sprite_idx: u32,
}

fn default_scale() -> f32 {
    1.0
}

fn default_sprite_idx() -> u32 {
    NUM_TILES // sentinel: "missing, derive from seed"
}

struct Plant {
    species: Species,
    age: f32,
    target_scale: f32,
    sprite_idx: u32,
    x: f32,
    y_jitter: f32,
    seed: u32,
    plot_key: u64,
    hue_shift_deg: f32,
    image: Handle<Image>,
    sprite_entity: Entity,
}

struct Butterfly {
    x: f32,
    y_frac: f32,
    vx: f32,
    age: f32,
    sprite_entity: Entity,
}

#[derive(Component)]
pub struct ClaudeGardenPane {
    plants: Vec<Plant>,
    butterflies: Vec<Butterfly>,
    ground_entity: Entity,
    sky_entity: Entity,
    rng: u32,
}

pub struct ClaudeGardenPanePlugin;

impl Plugin for ClaudeGardenPanePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (register_kind, build_garden_art))
            .add_systems(
                Update,
                (
                    poll_art_reload,
                    resize_garden_backdrop,
                    react_to_events,
                    advance_plants,
                    advance_butterflies,
                )
                    .chain(),
            );
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Claude Garden",
        radial_icon: None,
        default_size: Vec2::new(780.0, 380.0),
        spawn: garden_spawn,
        snapshot: garden_snapshot,
        on_close: None,
    });
}

// ============================================================
// Sprite-sheet loading
// ============================================================

/// Decode the embedded PNG, chroma-key its background, and slice it
/// into per-tile RGBA buffers. Done once at startup; subsequent plant
/// spawns just hue-shift the prepared tile bytes.
fn slice_tiles(sheet_bytes: &[u8]) -> Vec<Vec<u8>> {
    let img = image::load_from_memory(sheet_bytes)
        .expect("garden plants.png must decode")
        .to_rgba8();
    let (w, h) = (img.width(), img.height());
    assert_eq!(
        w,
        TILE_W * NUM_TILES,
        "plants.png width {} doesn't match {} tiles × {}px",
        w,
        NUM_TILES,
        TILE_W
    );
    assert_eq!(h, TILE_H, "plants.png height {} doesn't match {}px tile", h, TILE_H);

    // Chroma-key on the top-left pixel — the sheet uses a solid
    // background that we treat as transparent. (image::load_from_memory
    // gives every pixel alpha=255 for RGB PNGs.)
    let bg_px = img.get_pixel(0, 0).0;
    let bg = [bg_px[0], bg_px[1], bg_px[2]];

    let mut tiles = Vec::with_capacity(NUM_TILES as usize);
    for col in 0..NUM_TILES {
        let mut tile = Vec::with_capacity((TILE_W * TILE_H * 4) as usize);
        for y in 0..TILE_H {
            for x in 0..TILE_W {
                let p = img.get_pixel(col * TILE_W + x, y).0;
                if p[0] == bg[0] && p[1] == bg[1] && p[2] == bg[2] {
                    tile.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    tile.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
            }
        }
        tiles.push(tile);
    }
    tiles
}

/// Build a Bevy `Image` for one plant: apply a hue shift to the source
/// tile's pixels and wrap them in an RGBA8 `Image` asset.
fn make_plant_image(tile_rgba: &[u8], hue_shift_deg: f32) -> Image {
    let mut data: Vec<u8> = Vec::with_capacity(tile_rgba.len());
    for chunk in tile_rgba.chunks(4) {
        if chunk[3] == 0 {
            data.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let shifted = shift_hue([chunk[0], chunk[1], chunk[2], chunk[3]], hue_shift_deg);
            data.extend_from_slice(&shifted);
        }
    }
    image_from_rgba(data, TILE_W, TILE_H)
}

fn image_from_rgba(data: Vec<u8>, w: u32, h: u32) -> Image {
    let mut img = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Nearest,
        mipmap_filter: ImageFilterMode::Nearest,
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        ..ImageSamplerDescriptor::nearest()
    });
    img
}

// ============================================================
// HSV hue shift (per-plant + per-project color variation)
// ============================================================

fn shift_hue(rgba: [u8; 4], degrees: f32) -> [u8; 4] {
    if degrees.abs() < 0.5 {
        return rgba;
    }
    let (h, s, v) = rgb_to_hsv(rgba[0], rgba[1], rgba[2]);
    let h = (h + degrees).rem_euclid(360.0);
    let (r, g, b) = hsv_to_rgb(h, s, v);
    [r, g, b, rgba[3]]
}

fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if (max - g).abs() < 1e-6 {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let h = (h + 360.0).rem_euclid(360.0);
    let s = if max < 1e-6 { 0.0 } else { delta / max };
    let v = max;
    (h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match (h / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let to_u8 = |f: f32| ((f + m) * 255.0).clamp(0.0, 255.0) as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
}

// ============================================================
// Plot key + clustering
// ============================================================

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn plot_key_for_event(ev: &ClaudeBusEvent) -> u64 {
    let payload: Value = match serde_json::from_str(&ev.payload_json) {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let cwd = payload.get("cwd").and_then(|v| v.as_str()).unwrap_or("");
    if cwd.is_empty() {
        0
    } else {
        fnv1a(cwd)
    }
}

/// `(center_x_frac, hue_shift_deg)` for a given plot key. Both
/// deterministic functions of the key so the same cwd ends up at the
/// same position with the same color family across restarts.
fn plot_layout(plot_key: u64) -> (f32, f32) {
    let cx = ((plot_key & 0xFFFF) as f32 / 65535.0) * 0.84 + 0.08;
    let hue = ((plot_key >> 16) % 360) as f32;
    (cx, hue)
}

// ============================================================
// Hot reload (palette + butterfly only)
// ============================================================

fn garden_art_dir() -> Option<PathBuf> {
    let mut p = crate::data_dir()?;
    p.push("garden-art");
    Some(p)
}

fn build_garden_art(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let tile_rgba = slice_tiles(PLANTS_PNG);

    let (art_dir, palette, butterfly_img) = match garden_art_dir() {
        Some(dir) => {
            let _ = std::fs::create_dir_all(&dir);
            write_missing_defaults(&dir);
            let palette = read_palette(&dir);
            let rows = read_sprite_file(&dir, "butterfly.txt").unwrap_or_else(|| {
                DEFAULT_BUTTERFLY.iter().map(|s| s.to_string()).collect()
            });
            let rows_ref: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
            let img = image_from_butterfly_rows(&rows_ref, &palette);
            (dir, palette, img)
        }
        None => {
            warn!("garden: HOME not set, falling back to embedded butterfly art");
            let palette = default_palette();
            let img = image_from_butterfly_rows(DEFAULT_BUTTERFLY, &palette);
            (PathBuf::new(), palette, img)
        }
    };

    let butterfly = images.add(butterfly_img);
    let watcher = if !art_dir.as_os_str().is_empty() {
        spawn_art_watcher(&art_dir)
    } else {
        None
    };

    commands.insert_resource(GardenArt {
        tile_rgba,
        butterfly,
        art_dir,
        palette,
    });
    if let Some(w) = watcher {
        commands.insert_resource(w);
    }
}

fn write_missing_defaults(art_dir: &Path) {
    let palette_path = art_dir.join("palette.txt");
    if !palette_path.exists() {
        let _ = std::fs::write(&palette_path, DEFAULT_PALETTE_FILE);
    }
    let butterfly_path = art_dir.join("butterfly.txt");
    if !butterfly_path.exists() {
        let body = DEFAULT_BUTTERFLY.join("\n");
        let _ = std::fs::write(&butterfly_path, body);
    }
}

fn read_sprite_file(art_dir: &Path, filename: &str) -> Option<Vec<String>> {
    let body = std::fs::read_to_string(art_dir.join(filename)).ok()?;
    Some(body.lines().map(|s| s.to_string()).collect())
}

fn read_palette(art_dir: &Path) -> HashMap<char, [u8; 4]> {
    match std::fs::read_to_string(art_dir.join("palette.txt")) {
        Ok(body) => parse_palette(&body),
        Err(_) => default_palette(),
    }
}

fn parse_palette(body: &str) -> HashMap<char, [u8; 4]> {
    let mut map = HashMap::new();
    map.insert('.', [0, 0, 0, 0]);
    map.insert(' ', [0, 0, 0, 0]);
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let key = parts.next().and_then(|s| s.chars().next());
        let hex = parts.next().map(str::trim);
        if let (Some(key), Some(hex)) = (key, hex) {
            if let Some(rgba) = parse_hex_rgba(hex) {
                map.insert(key, rgba);
            }
        }
    }
    map
}

fn parse_hex_rgba(s: &str) -> Option<[u8; 4]> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 && s.len() != 8 {
        return None;
    }
    let mut out = [0u8; 4];
    out[3] = 255;
    for i in 0..(s.len() / 2) {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn default_palette() -> HashMap<char, [u8; 4]> {
    parse_palette(DEFAULT_PALETTE_FILE)
}

fn image_from_butterfly_rows(rows: &[&str], palette: &HashMap<char, [u8; 4]>) -> Image {
    let mut data: Vec<u8> = Vec::with_capacity((BUTTERFLY_W * BUTTERFLY_H * 4) as usize);
    for y in 0..BUTTERFLY_H {
        let row = rows.get(y as usize).copied().unwrap_or("");
        let mut count: u32 = 0;
        for ch in row.chars().take(BUTTERFLY_W as usize) {
            let px = palette.get(&ch).copied().unwrap_or([0, 0, 0, 0]);
            data.extend_from_slice(&px);
            count += 1;
        }
        while count < BUTTERFLY_W {
            data.extend_from_slice(&[0, 0, 0, 0]);
            count += 1;
        }
    }
    image_from_rgba(data, BUTTERFLY_W, BUTTERFLY_H)
}

fn spawn_art_watcher(art_dir: &Path) -> Option<ArtWatcher> {
    let (tx, rx) = std::sync::mpsc::channel::<PathBuf>();
    let mut watcher = notify::recommended_watcher(
        move |res: notify::Result<notify::Event>| {
            let Ok(ev) = res else { return };
            if !matches!(
                ev.kind,
                EventKind::Modify(_) | EventKind::Create(_) | EventKind::Any
            ) {
                return;
            }
            for path in ev.paths {
                let _ = tx.send(path);
            }
        },
    )
    .map_err(|e| eprintln!("garden: failed to start file watcher: {e}"))
    .ok()?;
    watcher
        .watch(art_dir, RecursiveMode::NonRecursive)
        .map_err(|e| eprintln!("garden: failed to watch {}: {e}", art_dir.display()))
        .ok()?;
    Some(ArtWatcher {
        rx: Mutex::new(rx),
        _watcher: watcher,
    })
}

fn poll_art_reload(
    art: Option<ResMut<GardenArt>>,
    watcher: Option<Res<ArtWatcher>>,
    mut images: ResMut<Assets<Image>>,
    mut gardens: Query<&mut ClaudeGardenPane>,
) {
    let Some(mut art) = art else { return };
    let Some(watcher) = watcher else { return };

    let changed_paths: Vec<PathBuf> = {
        let rx = watcher.rx.lock().expect("art watcher channel poisoned");
        rx.try_iter().collect()
    };
    if changed_paths.is_empty() {
        return;
    }

    let mut unique = std::collections::HashSet::new();
    let mut palette_changed = false;
    let mut butterfly_changed = false;
    for path in changed_paths {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !unique.insert(name.to_string()) {
            continue;
        }
        match name {
            "palette.txt" => palette_changed = true,
            "butterfly.txt" => butterfly_changed = true,
            _ => {}
        }
    }

    if palette_changed {
        art.palette = read_palette(&art.art_dir);
    }

    if palette_changed || butterfly_changed {
        let rows = read_sprite_file(&art.art_dir, "butterfly.txt")
            .unwrap_or_else(|| DEFAULT_BUTTERFLY.iter().map(|s| s.to_string()).collect());
        let rows_ref: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
        let new_image = image_from_butterfly_rows(&rows_ref, &art.palette);
        let handle = art.butterfly.clone();
        if let Some(slot) = images.get_mut(&handle) {
            *slot = new_image;
        }
    }

    // Palette changes don't affect plant sprites (they come from the
    // embedded sheet, not the palette file), so we skip the bulk
    // re-render here. If we want to expose plant color via palette
    // later, do it by adding a per-plant tint multiplier on top of
    // the sheet pixels.
    let _ = gardens;
}

// ============================================================
// Spawn / snapshot
// ============================================================

fn garden_spawn(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Claude Garden")
        .to_string();
    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title;
    }

    let rect = *world
        .get::<PaneRect>(entity)
        .expect("garden pane must have PaneRect by the time spawn runs");
    let (w, h) = inner_size(rect.size);

    let sky_entity = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: SKY_COLOR,
                custom_size: Some(Vec2::new(w.max(1.0), h.max(1.0))),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.0),
        ))
        .id();

    let ground_y = -(h - GROUND_INSET - GROUND_H);
    let ground_entity = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: GROUND_COLOR,
                custom_size: Some(Vec2::new(w.max(1.0), GROUND_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, ground_y, 0.05),
        ))
        .id();

    let seed = (entity.to_bits() as u32)
        .wrapping_mul(2654435761)
        .wrapping_add(0x9E3779B9)
        | 1;

    let restored: Vec<PlantSnapshot> = config
        .get("plants")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let mut garden = ClaudeGardenPane {
        plants: Vec::with_capacity(restored.len()),
        butterflies: Vec::new(),
        ground_entity,
        sky_entity,
        rng: seed,
    };

    for snap in restored.into_iter().take(MAX_PLANTS) {
        let plant = place_plant(world, content_root, snap, h);
        garden.plants.push(plant);
    }

    world.entity_mut(entity).insert(garden);
    let _ = w;
}

fn garden_snapshot(world: &World, entity: Entity) -> Value {
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    let plants: Vec<PlantSnapshot> = world
        .get::<ClaudeGardenPane>(entity)
        .map(|g| {
            g.plants
                .iter()
                .map(|p| PlantSnapshot {
                    species: p.species,
                    stage: 0,
                    age: p.age,
                    x: p.x,
                    y_jitter: p.y_jitter,
                    seed: p.seed,
                    plot_key: p.plot_key,
                    hue_shift_deg: p.hue_shift_deg,
                    scale: p.target_scale,
                    sprite_idx: p.sprite_idx,
                })
                .collect()
        })
        .unwrap_or_default();
    serde_json::json!({
        "title": title,
        "plants": plants,
    })
}

fn inner_size(size: Vec2) -> (f32, f32) {
    let w = (size.x - 2.0 * MARGIN).max(0.0);
    let h = (size.y - TITLE_H - 2.0 * MARGIN).max(0.0);
    (w, h)
}

// ============================================================
// Resize + per-frame
// ============================================================

fn resize_garden_backdrop(
    rects: Query<(&PaneRect, &ClaudeGardenPane), Changed<PaneRect>>,
    mut sprites: Query<&mut Sprite>,
    mut transforms: Query<&mut Transform>,
) {
    for (rect, garden) in &rects {
        let (w, h) = inner_size(rect.size);
        if let Ok(mut sprite) = sprites.get_mut(garden.sky_entity) {
            sprite.custom_size = Some(Vec2::new(w.max(1.0), h.max(1.0)));
        }
        if let Ok(mut sprite) = sprites.get_mut(garden.ground_entity) {
            sprite.custom_size = Some(Vec2::new(w.max(1.0), GROUND_H));
        }
        if let Ok(mut transform) = transforms.get_mut(garden.ground_entity) {
            transform.translation.y = -(h - GROUND_INSET - GROUND_H);
        }

        let baseline = -(h - GROUND_INSET);
        for plant in &garden.plants {
            if let Ok(mut transform) = transforms.get_mut(plant.sprite_entity) {
                transform.translation.y = baseline + plant.y_jitter;
            }
        }
        for bf in &garden.butterflies {
            if let Ok(mut transform) = transforms.get_mut(bf.sprite_entity) {
                transform.translation.y = butterfly_base_y(bf.y_frac, h);
            }
        }
    }
}

fn react_to_events(
    mut events: MessageReader<ClaudeBusEvent>,
    mut commands: Commands,
    art: Res<GardenArt>,
    mut images: ResMut<Assets<Image>>,
    mut gardens: Query<(&PaneChrome, &PaneRect, &mut ClaudeGardenPane)>,
) {
    let buffered: Vec<ClaudeBusEvent> = events.read().cloned().collect();
    if buffered.is_empty() {
        return;
    }
    for (chrome, rect, mut garden) in &mut gardens {
        let (w, h) = inner_size(rect.size);
        if w <= 1.0 || h <= 1.0 {
            continue;
        }
        for ev in &buffered {
            match ev.kind.as_str() {
                "notification" => {
                    spawn_butterfly(
                        &mut commands,
                        chrome.content_root,
                        &art,
                        &mut garden,
                        w,
                        h,
                    );
                }
                "stop" => {
                    // Snap every still-growing plant to mature. We
                    // only mutate `age`; `advance_plants` reads it
                    // next frame and writes the final transform.scale.
                    for plant in &mut garden.plants {
                        if plant.age < GROW_DURATION {
                            plant.age = GROW_DURATION;
                        }
                    }
                }
                _ => {
                    let Some((species, count)) = plantings_for_event(ev) else {
                        continue;
                    };
                    let plot_key = plot_key_for_event(ev);
                    for _ in 0..count {
                        spawn_plant(
                            &mut commands,
                            &mut images,
                            chrome.content_root,
                            &art,
                            &mut garden,
                            species,
                            plot_key,
                            w,
                            h,
                        );
                    }
                }
            }
        }
    }
}

fn plantings_for_event(ev: &ClaudeBusEvent) -> Option<(Species, u32)> {
    match ev.kind.as_str() {
        "pre_tool_use" => {
            let v: Value = serde_json::from_str(&ev.payload_json).ok()?;
            let tool = v.get("tool_name").and_then(|t| t.as_str()).unwrap_or("");
            let species = match tool {
                "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => Species::Flower,
                "Bash" => Species::Vine,
                "Read" | "Grep" | "Glob" => Species::Grass,
                "Task" | "Agent" => Species::Tree,
                _ => Species::Seed,
            };
            Some((species, 1))
        }
        "user_prompt_submit" => Some((Species::Seed, 3)),
        "session_start" => Some((Species::Tree, 1)),
        _ => None,
    }
}

/// Species-typical scale range. Sampled per plant so heights vary
/// within and across species.
fn scale_for_species(species: Species, rng: &mut u32) -> f32 {
    let (lo, hi) = match species {
        Species::Tree => (1.10, 1.55),
        Species::Flower => (0.75, 1.10),
        Species::Vine => (0.65, 0.95),
        Species::Grass => (0.55, 0.85),
        Species::Seed => (0.30, 0.50),
    };
    lo + rand_f32(rng) * (hi - lo)
}

fn spawn_plant(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    content_root: Entity,
    art: &GardenArt,
    garden: &mut ClaudeGardenPane,
    species: Species,
    plot_key: u64,
    w: f32,
    h: f32,
) {
    let (cx_frac, plot_hue) = plot_layout(plot_key);
    let spread = (w * PLOT_SPREAD_FRAC).clamp(PLOT_SPREAD_MIN_PX, PLOT_SPREAD_MAX_PX);
    let center_x = cx_frac * w;
    let offset = (rand_f32(&mut garden.rng) - 0.5) * 2.0 * spread;
    let margin = SPRITE_DRAW_W * 0.5;
    let x = (center_x + offset).clamp(margin, (w - margin).max(margin));
    let jitter = (rand_f32(&mut garden.rng) - 0.5) * 2.0 * PLANT_Y_JITTER;
    let seed = next_rand(&mut garden.rng);
    let sprite_idx = seed % NUM_TILES;
    let per_plant_hue =
        (rand_f32(&mut garden.rng) - 0.5) * 2.0 * PER_PLANT_HUE_JITTER_DEG;
    let hue_shift_deg = (plot_hue + per_plant_hue).rem_euclid(360.0);
    let target_scale = scale_for_species(species, &mut garden.rng);
    let baseline_y = -(h - GROUND_INSET) + jitter;

    let tile = &art.tile_rgba[sprite_idx as usize];
    let image = images.add(make_plant_image(tile, hue_shift_deg));
    let initial_scale = target_scale * PLANT_MIN_SCALE_FRAC;
    let sprite_entity = commands
        .spawn((
            ChildOf(content_root),
            Sprite {
                image: image.clone(),
                custom_size: Some(Vec2::new(SPRITE_DRAW_W, SPRITE_DRAW_H)),
                ..default()
            },
            Anchor::BOTTOM_CENTER,
            Transform {
                translation: Vec3::new(x, baseline_y, 1.0),
                scale: Vec3::splat(initial_scale),
                ..default()
            },
            Visibility::Inherited,
        ))
        .id();

    garden.plants.push(Plant {
        species,
        age: 0.0,
        target_scale,
        sprite_idx,
        x,
        y_jitter: jitter,
        seed,
        plot_key,
        hue_shift_deg,
        image,
        sprite_entity,
    });
    if garden.plants.len() > MAX_PLANTS {
        let oldest = garden.plants.remove(0);
        commands.entity(oldest.sprite_entity).despawn();
        drop(oldest.image);
    }
}

fn place_plant(
    world: &mut World,
    content_root: Entity,
    snap: PlantSnapshot,
    h: f32,
) -> Plant {
    // Migrate sprite_idx: sentinel (NUM_TILES) means "derive from seed",
    // which is what every pre-sprite-sheet snapshot needs.
    let sprite_idx = if snap.sprite_idx >= NUM_TILES {
        snap.seed % NUM_TILES
    } else {
        snap.sprite_idx
    };
    // Migrate age: if absent (0.0) but a non-zero stage was persisted,
    // treat it as a mature plant.
    let age = if snap.age == 0.0 && snap.stage > 0 {
        snap.stage as f32 * 2.0
    } else {
        snap.age
    };
    let target_scale = if snap.scale > 0.0 { snap.scale } else { 1.0 };
    let t = (age / GROW_DURATION).clamp(0.0, 1.0);
    let current_scale = target_scale * (PLANT_MIN_SCALE_FRAC + (1.0 - PLANT_MIN_SCALE_FRAC) * t);

    let tile = world.resource::<GardenArt>().tile_rgba[sprite_idx as usize].clone();
    let img_asset = make_plant_image(&tile, snap.hue_shift_deg);
    let image = world.resource_mut::<Assets<Image>>().add(img_asset);
    let baseline_y = -(h - GROUND_INSET) + snap.y_jitter;
    let sprite_entity = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                image: image.clone(),
                custom_size: Some(Vec2::new(SPRITE_DRAW_W, SPRITE_DRAW_H)),
                ..default()
            },
            Anchor::BOTTOM_CENTER,
            Transform {
                translation: Vec3::new(snap.x, baseline_y, 1.0),
                scale: Vec3::splat(current_scale),
                ..default()
            },
            Visibility::Inherited,
        ))
        .id();
    Plant {
        species: snap.species,
        age,
        target_scale,
        sprite_idx,
        x: snap.x,
        y_jitter: snap.y_jitter,
        seed: snap.seed,
        plot_key: snap.plot_key,
        hue_shift_deg: snap.hue_shift_deg,
        image,
        sprite_entity,
    }
}

fn spawn_butterfly(
    commands: &mut Commands,
    content_root: Entity,
    art: &GardenArt,
    garden: &mut ClaudeGardenPane,
    _w: f32,
    h: f32,
) {
    let y_frac = 0.15 + 0.55 * rand_f32(&mut garden.rng);
    let vx = BUTTERFLY_VX * (0.7 + 0.6 * rand_f32(&mut garden.rng));
    let start_x = -10.0;
    let sprite_entity = commands
        .spawn((
            ChildOf(content_root),
            Sprite {
                image: art.butterfly.clone(),
                custom_size: Some(Vec2::splat(BUTTERFLY_DRAW_SIZE)),
                ..default()
            },
            Anchor::CENTER,
            Transform::from_xyz(start_x, butterfly_base_y(y_frac, h), 2.0),
            Visibility::Inherited,
        ))
        .id();
    garden.butterflies.push(Butterfly {
        x: start_x,
        y_frac,
        vx,
        age: 0.0,
        sprite_entity,
    });
}

fn butterfly_base_y(y_frac: f32, h: f32) -> f32 {
    let usable_h = (h - GROUND_INSET - GROUND_H).max(20.0);
    -(usable_h * y_frac)
}

fn advance_plants(
    time: Res<Time>,
    mut gardens: Query<&mut ClaudeGardenPane>,
    mut transforms: Query<&mut Transform>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    for mut garden in &mut gardens {
        for plant in &mut garden.plants {
            if plant.age >= GROW_DURATION {
                continue;
            }
            plant.age = (plant.age + dt).min(GROW_DURATION);
            let t = plant.age / GROW_DURATION;
            let s =
                plant.target_scale * (PLANT_MIN_SCALE_FRAC + (1.0 - PLANT_MIN_SCALE_FRAC) * t);
            if let Ok(mut transform) = transforms.get_mut(plant.sprite_entity) {
                transform.scale = Vec3::splat(s);
            }
        }
    }
}

fn advance_butterflies(
    time: Res<Time>,
    mut commands: Commands,
    mut gardens: Query<(&PaneRect, &mut ClaudeGardenPane)>,
    mut transforms: Query<&mut Transform>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    for (rect, mut garden) in &mut gardens {
        let (w, h) = inner_size(rect.size);
        let mut i = 0;
        while i < garden.butterflies.len() {
            let bf = &mut garden.butterflies[i];
            bf.age += dt;
            bf.x += bf.vx * dt;
            let wobble = (bf.age * BUTTERFLY_WOBBLE_HZ * std::f32::consts::TAU).sin()
                * BUTTERFLY_WOBBLE_PX;
            if let Ok(mut transform) = transforms.get_mut(bf.sprite_entity) {
                transform.translation.x = bf.x;
                transform.translation.y = butterfly_base_y(bf.y_frac, h) + wobble;
            }
            if bf.x > w + 16.0 {
                commands.entity(bf.sprite_entity).despawn();
                garden.butterflies.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

// ---------- xorshift PRNG ----------

fn next_rand(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn rand_f32(state: &mut u32) -> f32 {
    (next_rand(state) as f32) / (u32::MAX as f32)
}

// ============================================================
// Embedded defaults (palette + butterfly only)
// ============================================================

const DEFAULT_PALETTE_FILE: &str = "\
# Garden palette — edit and save; the app reloads on the fly.
# Currently only the butterfly uses this; plant sprites come from the
# embedded sheet (assets/garden/plants.png) and are hue-shifted per
# plant, so palette edits don't recolor the plants.
# Format: <single char> <rgba hex, 6 or 8 chars>

p f0a5c8ff
P cd5faaff
Y f5dc6eff
g 87d273ff
G 378746ff
l b4eb96ff
b 875532ff
B 55371eff
o c8af69ff
r cd4b4bff
w f5f5f5ff
";

const DEFAULT_BUTTERFLY: &[&str] = &[
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    ".......pPp..........pPp.........",
    "......pPYPp........pPYPp........",
    "......pPYwPp......pPYwPp........",
    "......pPYPpYp....pYpPYPp........",
    ".......pPp...b....b...pPp.......",
    ".............b....b.............",
    ".............b....b.............",
    "..............b..b..............",
    "..............b..b..............",
    "...............bb...............",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
    "................................",
];
