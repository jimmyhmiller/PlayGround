//! Font registry — name → `Handle<Font>` resolution.
//!
//! Theme tokens like [`crate::tokens::FONT_FAMILY_HEADING`] hold a
//! family *name* (`"serif"`, `"sans"`, `"mono"`). At render time the
//! widget asks the registry for the handle.
//!
//! ## Adding a family
//!
//! Drop a `.ttf` or `.otf` into `crates/style-bevy/assets/fonts/` and
//! add a line to [`bundled_fonts`]. Today only JetBrains Mono is
//! bundled; `"serif"` and `"sans"` both fall back to it until you
//! drop in real files. Unknown names also fall back to mono — the
//! engine never crashes for a missing font, it just renders in mono.

use std::collections::HashMap;

use bevy::prelude::*;

/// Maps a family name to a `Handle<Font>`. Populated at Startup.
#[derive(Resource, Default, Clone)]
pub struct FontRegistry {
    by_name: HashMap<String, Handle<Font>>,
    /// Fallback used when a name doesn't resolve. Always points at the
    /// bundled mono font.
    fallback: Handle<Font>,
}

impl FontRegistry {
    /// Look up a family by name. Returns the fallback (mono) when the
    /// name isn't registered — callers never see `None`.
    pub fn resolve(&self, name: &str) -> Handle<Font> {
        self.by_name
            .get(name)
            .cloned()
            .unwrap_or_else(|| self.fallback.clone())
    }

    /// Resolve one of the three role tokens. Equivalent to
    /// `resolve(theme.str_value(role))`.
    pub fn for_role(&self, theme: &crate::Theme, role: crate::TokenId) -> Handle<Font> {
        self.resolve(theme.str_value(role))
    }

    /// Raw bytes for a bundled family. Used by callers that need to
    /// measure glyph advances (skrifa, etc.) and can't go through the
    /// `Handle<Font>`. Returns `None` for unknown names.
    pub fn bytes(&self, name: &str) -> Option<&'static [u8]> {
        BUNDLED_FONTS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, b)| *b)
    }

    /// All registered family names. Used by the theme editor's font
    /// picker. Sorted for stable display order.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.by_name.keys().cloned().collect();
        v.sort();
        v
    }
}

/// Per-family bundled bytes. New families add a row here.
//
// Until real serif / sans fonts are dropped into the assets dir, both
// roles route to mono so the renderer still produces output. To add a
// real font: place `Foo-Regular.ttf` next to JetBrainsMono and add a
// row here (`("foo", include_bytes!("../assets/fonts/Foo-Regular.ttf"))`).
const BUNDLED_FONTS: &[(&str, &[u8])] = &[
    ("mono", include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf")),
    ("sans", include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf")),
    ("serif", include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf")),
];

pub struct FontRegistryPlugin;

impl Plugin for FontRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FontRegistry>()
            .add_systems(Startup, register_bundled_fonts);
    }
}

/// Idempotently populate the registry with bundled fonts. Safe to call
/// from multiple Startup systems — entries that already exist are kept,
/// new ones are appended. The first registered family becomes the
/// fallback if no fallback is set yet.
pub fn ensure_initialized(registry: &mut FontRegistry, fonts: &mut Assets<Font>) {
    let mut first: Option<Handle<Font>> = None;
    for (name, bytes) in BUNDLED_FONTS {
        if registry.by_name.contains_key(*name) {
            if first.is_none() {
                first = registry.by_name.get(*name).cloned();
            }
            continue;
        }
        let font = match Font::try_from_bytes(bytes.to_vec()) {
            Ok(f) => f,
            Err(e) => {
                warn!("[font-registry] bundled font {:?} failed to parse: {}", name, e);
                continue;
            }
        };
        let handle = fonts.add(font);
        if first.is_none() {
            first = Some(handle.clone());
        }
        registry.by_name.insert((*name).to_string(), handle);
    }
    if let Some(h) = first {
        // Don't clobber an explicitly-set fallback if one already exists.
        if registry.fallback == Handle::default() {
            registry.fallback = h;
        }
    }
}

fn register_bundled_fonts(
    mut registry: ResMut<FontRegistry>,
    mut fonts: ResMut<Assets<Font>>,
) {
    ensure_initialized(&mut registry, &mut fonts);
}
