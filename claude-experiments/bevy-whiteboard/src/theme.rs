//! Theme system. A `Theme` resource carries every palette decision (paper,
//! ink, accent, per-node-type fills, data-colour swatches) so swapping the
//! resource re-skins the entire app at runtime. Add a new preset by writing
//! another `Theme::xyz()` constructor and listing it in [`Theme::all`].
//!
//! Down-stream code reads from the resource each frame; persistent assets
//! that store baked colours (Bevy `ColorMaterial` handles, mesh borders, …)
//! are refreshed by [`apply_theme_to_canvas`] when the theme changes.

use bevy::prelude::*;

pub struct ThemePlugin;

impl Plugin for ThemePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Theme::iso50())
            .add_systems(Update, sync_clear_color);
    }
}

/// Six "data colour" slots — used by generators/clients to tag emitted
/// packets and by sinks/queues/workers to filter on type. The slots are
/// stable across themes (slot 0 is always the dominant accent in that
/// theme's palette, slot 5 is always the darkest).
pub const DATA_SLOT_COUNT: usize = 6;

#[derive(Resource, Clone, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub paper: Color,
    pub paper_alt: Color,
    pub node_bg: Color,
    pub ink: Color,
    pub ink_soft: Color,
    pub muted: Color,
    pub rule: Color,
    pub accent: Color,
    pub accent_soft: Color,
    pub data: [Color; DATA_SLOT_COUNT],
    pub node_fill: NodeFillSet,
}

#[derive(Clone, Debug)]
pub struct NodeFillSet {
    pub generator: Color,
    pub client: Color,
    pub worker: Color,
    pub router: Color,
    pub queue: Color,
    pub sink: Color,
}

impl Theme {
    pub fn iso50() -> Self {
        // Dusty cream poster stock + dark olive-brown ink + burnt orange. The
        // node fills come straight from the design (mustard / olive / burnt
        // orange / teal / plum / ink).
        Self {
            name: "iso50",
            paper: hex(0xe8dfc4),
            paper_alt: hex(0xded3b7),
            node_bg: hex(0xf3eadb),
            ink: hex(0x2b2a24),
            ink_soft: hex(0x6a665a),
            muted: hex(0xa09a85),
            rule: hex(0xc6bc9f),
            accent: hex(0xc95a3a),
            accent_soft: hex(0xd4a23a),
            data: [
                hex(0xc95a3a), // coral / burnt orange
                hex(0xd4a23a), // amber / mustard
                hex(0x7a8a4a), // moss / olive
                hex(0x4c7a7a), // sky / dusty teal
                hex(0x8a6a9c), // plum
                hex(0x2b2a24), // ink
            ],
            node_fill: NodeFillSet {
                generator: hex(0xd9a84a),
                client: hex(0x7a8a4a),
                worker: hex(0xc95a3a),
                router: hex(0x4c7a7a),
                queue: hex(0x8a6a9c),
                sink: hex(0x2b2a24),
            },
        }
    }

    pub fn original() -> Self {
        // The pre-redesign look — bright whites, near-black borders, saturated
        // primaries. Useful for A/B sanity-checking that the theme system
        // actually swaps everything.
        Self {
            name: "original",
            paper: hex(0xf7f7f2),
            paper_alt: Color::srgb(1.0, 1.0, 1.0),
            node_bg: Color::srgb(0.98, 0.98, 0.98),
            ink: Color::srgb(0.10, 0.10, 0.12),
            ink_soft: Color::srgb(0.30, 0.30, 0.35),
            muted: Color::srgb(0.55, 0.55, 0.60),
            rule: Color::srgb(0.80, 0.80, 0.82),
            accent: Color::srgb(0.25, 0.55, 0.90),
            accent_soft: Color::srgb(0.65, 0.80, 0.95),
            data: [
                Color::srgb(0.90, 0.30, 0.30),
                Color::srgb(0.95, 0.70, 0.20),
                Color::srgb(0.40, 0.75, 0.35),
                Color::srgb(0.25, 0.55, 0.90),
                Color::srgb(0.65, 0.40, 0.85),
                Color::srgb(0.25, 0.25, 0.30),
            ],
            node_fill: NodeFillSet {
                generator: Color::srgb(0.98, 0.98, 0.98),
                client: Color::srgb(0.94, 0.96, 0.99),
                worker: Color::srgb(0.92, 0.92, 0.96),
                router: Color::srgb(0.90, 0.95, 0.92),
                queue: Color::srgb(0.96, 0.92, 0.88),
                sink: Color::srgb(0.95, 0.95, 0.90),
            },
        }
    }

    pub fn dark() -> Self {
        // Inverse of iso50: ink-on-paper flipped to ink-bg with cream
        // accents. Same node-fill hue family, just lifted for contrast.
        Self {
            name: "dark",
            paper: hex(0x1c1b18),
            paper_alt: hex(0x26241f),
            node_bg: hex(0x2c2a24),
            ink: hex(0xe8dfc4),
            ink_soft: hex(0x9a917a),
            muted: hex(0x6a665a),
            rule: hex(0x40392f),
            accent: hex(0xe87049),
            accent_soft: hex(0xe5b455),
            data: [
                hex(0xe87049),
                hex(0xe5b455),
                hex(0x9bb05c),
                hex(0x67a3a3),
                hex(0xb088c4),
                hex(0xe8dfc4),
            ],
            node_fill: NodeFillSet {
                generator: hex(0xb88a3a),
                client: hex(0x6a7a3a),
                worker: hex(0xb04a30),
                router: hex(0x3c6868),
                queue: hex(0x6e547e),
                sink: hex(0x40392f),
            },
        }
    }

    pub fn all() -> [fn() -> Theme; 3] {
        [Theme::iso50, Theme::original, Theme::dark]
    }

    pub fn next(&self) -> Theme {
        let names: Vec<&'static str> = Theme::all().iter().map(|f| f().name).collect();
        let idx = names.iter().position(|n| *n == self.name).unwrap_or(0);
        let next = (idx + 1) % names.len();
        Theme::all()[next]()
    }
}

fn hex(rgb: u32) -> Color {
    let r = ((rgb >> 16) & 0xff) as f32 / 255.0;
    let g = ((rgb >> 8) & 0xff) as f32 / 255.0;
    let b = (rgb & 0xff) as f32 / 255.0;
    Color::srgb(r, g, b)
}

fn sync_clear_color(theme: Res<Theme>, mut clear: ResMut<ClearColor>) {
    if theme.is_changed() {
        clear.0 = theme.paper;
    }
}
