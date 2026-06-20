//! Loads the Kenney CC0 atlases and gives names to the 16×16 tiles we use.
//!
//! Both atlases are a clean 12×11 grid of 16px tiles, so a tile is addressed by a
//! single index `i` -> column `i % 12`, row `i / 12`. See `ATTRIBUTION.md`.

use raylib::prelude::*;

/// Edge length of one source tile, in atlas pixels.
pub const TILE: f32 = 16.0;
/// Columns per atlas (both Tiny Town and Tiny Dungeon are 12 wide).
pub const COLS: i32 = 12;

/// Source rectangle for tile index `i` within a 12-wide atlas.
pub fn src(i: i32) -> Rectangle {
    let c = (i % COLS) as f32;
    let r = (i / COLS) as f32;
    Rectangle::new(c * TILE, r * TILE, TILE, TILE)
}

// ---- Tiny Town tiles (terrain / buildings / props) ---------------------------

/// Plain grass + two flowered grass variants.
pub const GRASS: [i32; 3] = [1, 0, 2];
/// Single trees (mix of pine / round / small / autumn).
pub const TREES: [i32; 4] = [5, 4, 28, 27];
pub const MUSHROOMS: i32 = 29;
/// A ring of pale stones — used as a "quarry"/resource decoration.
pub const STONES: i32 = 43;
/// Red-and-white banner on a pole — a town flag.
pub const FLAG: i32 = 95;
/// A market stall.
pub const MARKET: i32 = 57;

/// A building is a small grid of Tiny-Town tiles, drawn bottom-anchored.
pub struct Building<'a> {
    pub grid: &'a [&'a [i32]],
}

/// The big project HQ: a little crenellated castle with a gate.
pub const TOWN_CENTER: Building = Building { grid: &[&[96, 97, 98], &[108, 103, 110]] };

/// House presets (roof gable over a doored facade). Index by model/variant.
pub const HOUSES: [Building; 4] = [
    Building { grid: &[&[63], &[85]] }, // blue roof / tan door
    Building { grid: &[&[67], &[89]] }, // red roof  / gray door
    Building { grid: &[&[63], &[89]] }, // blue roof / gray door
    Building { grid: &[&[67], &[85]] }, // red roof  / tan door
];

// ---- Tiny Dungeon tiles (villager characters) --------------------------------

pub const KNIGHT: i32 = 96; // heavy / opus
pub const VILLAGER: i32 = 88; // worker / sonnet
pub const RANGER: i32 = 112; // light / haiku
pub const MAGE: i32 = 84; // unknown model
pub const VILLAGER_ALTS: [i32; 4] = [85, 98, 99, 87];

/// Pick a villager sprite that hints at the model doing the work.
pub fn villager_sprite_for_model(model: Option<&str>, variant: usize) -> i32 {
    let m = model.unwrap_or("").to_ascii_lowercase();
    if m.contains("opus") {
        KNIGHT
    } else if m.contains("sonnet") {
        VILLAGER
    } else if m.contains("haiku") {
        RANGER
    } else if m.is_empty() {
        VILLAGER_ALTS[variant % VILLAGER_ALTS.len()]
    } else {
        MAGE
    }
}

/// Owns the loaded GPU textures.
pub struct Assets {
    pub town: Texture2D,
    pub chars: Texture2D,
}

impl Assets {
    /// Load atlases relative to `base` (the directory containing `assets/`).
    pub fn load(rl: &mut RaylibHandle, thread: &RaylibThread, base: &str) -> Result<Assets, String> {
        let town = rl
            .load_texture(thread, &format!("{base}/assets/kenney_tiny_town/tilemap_packed.png"))
            .map_err(|e| e.to_string())?;
        let chars = rl
            .load_texture(thread, &format!("{base}/assets/kenney_tiny_dungeon/tilemap_packed.png"))
            .map_err(|e| e.to_string())?;
        // Crisp pixel-art scaling.
        town.set_texture_filter(thread, TextureFilter::TEXTURE_FILTER_POINT);
        chars.set_texture_filter(thread, TextureFilter::TEXTURE_FILTER_POINT);
        Ok(Assets { town, chars })
    }
}
