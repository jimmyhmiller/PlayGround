//! Typography: font cascade loader, `Bold` / `Mono` marker components, and
//! [`caps_spaced`] for tracked-caps labels. Install [`TypographyPlugin`] and
//! every `TextFont` in the app gets the primary (Jost / Futura / Avenir Next
//! / system fallback) font stamped onto it each frame. Add the `Bold` marker
//! to a text entity to get the bold weight instead; add `Mono` to get a
//! monospace face (nice for live numeric readouts).
//!
//! # Glyph fallback
//!
//! Bevy 0.18's bundled default font (a FiraMono subset) is missing glyphs we
//! care about — µ, ×, ÷, •, ⚠, ‖, Σ, the various arrows. Without the
//! [`TypographyPlugin`] these render as tofu. We replace Bevy's
//! `CosmicFontSystem` at startup with one constructed from the host's font
//! database, so cosmic-text can pick a glyph from a system font whenever the
//! primary face doesn't have it.
//!
//! # Why we re-stamp every frame
//!
//! Bevy spawns text entities with `TextFont::default()` (handle = the default
//! asset id). Inserting at `AssetId::default()` doesn't help: cosmic-text
//! caches font faces against the handle on first render, so a late swap at
//! the same id is ignored. Loading fresh handles and a per-frame system that
//! reassigns them onto any `TextFont` still holding the default works
//! reliably. The system is cheap — `Mut<T>` only flags the component as
//! changed when dereferenced.

use bevy::prelude::*;

pub struct TypographyPlugin;

impl Plugin for TypographyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FontSearchPaths>()
            .add_systems(
                Startup,
                (
                    enable_system_font_fallback,
                    load_primary_font,
                    load_bold_font,
                    load_mono_font,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (apply_primary_font, apply_bold_font, apply_mono_font).chain(),
            );
    }
}

/// Jost (medium weight) embedded as bytes. Used as the last-resort primary
/// font when nothing in [`FontSearchPaths::primary`] resolves — makes the
/// library work out of the box regardless of the consumer's CWD or assets
/// layout. Licensed under OFL (bundled in `crates/poster-ui/assets/fonts/`).
const JOST_MEDIUM_TTF: &[u8] = include_bytes!("../assets/fonts/Jost-Medium.ttf");
const JOST_BOLD_TTF: &[u8] = include_bytes!("../assets/fonts/Jost-Bold.ttf");

/// Marker: this text entity wants the bold weight. Paired with a `TextFont`,
/// the per-frame applier re-stamps the bold font handle.
#[derive(Component)]
pub struct Bold;

/// Marker: this text entity wants the monospace face. Use for counters,
/// time readouts, anything whose width should stay fixed as digits change.
#[derive(Component)]
pub struct Mono;

/// Paths probed when loading the primary / bold / mono faces at startup.
/// The first path that both reads and parses wins. Consumers can override
/// by inserting a custom `FontSearchPaths` resource before `Startup`.
///
/// Defaults target Jost (bundled in `assets/fonts/` if the app ships it),
/// falling back to macOS system Futura → Avenir Next → unicode safety nets.
#[derive(Resource, Clone)]
pub struct FontSearchPaths {
    pub primary: Vec<String>,
    pub bold: Vec<String>,
    pub mono: Vec<String>,
}

impl Default for FontSearchPaths {
    fn default() -> Self {
        // By default we don't probe the filesystem for primary / bold — the
        // embedded Jost is the canonical iso50 face. If consumers want a
        // different body font, they insert their own `FontSearchPaths`
        // before `Startup` (e.g. `{ primary: vec!["assets/my-font.ttf".into()], ..default() }`).
        //
        // `mono` still probes system fonts because Jost isn't monospaced and
        // most hosts have Menlo / Monaco / equivalent available.
        Self {
            primary: vec![],
            bold: vec![],
            mono: vec![
                "assets/fonts/JetBrainsMono-Regular.ttf".into(),
                "/System/Library/Fonts/Menlo.ttc".into(),
                "/System/Library/Fonts/SFNSMono.ttf".into(),
                "/System/Library/Fonts/Monaco.ttf".into(),
                "/System/Library/Fonts/Courier.ttc".into(),
            ],
        }
    }
}

/// Bevy 0.18 doesn't expose `letter-spacing` on text, so we approximate the
/// iso50 design's wide-tracked caps by interspersing a thin space (U+2009)
/// between glyphs. Reads well at small point sizes — use for headers, section
/// titles, button labels.
pub fn caps_spaced(s: &str) -> String {
    let upper = s.to_uppercase();
    let mut out = String::with_capacity(upper.len() * 3);
    for (i, ch) in upper.chars().enumerate() {
        if i > 0 {
            out.push('\u{2009}');
        }
        out.push(ch);
    }
    out
}

// ---------------- font loading ----------------

#[derive(Resource, Default)]
struct PrimaryFont(Handle<bevy::text::Font>);

#[derive(Resource, Default)]
struct BoldFont(Handle<bevy::text::Font>);

#[derive(Resource, Default)]
struct MonoFont(Handle<bevy::text::Font>);

fn enable_system_font_fallback(mut fs: ResMut<bevy::text::CosmicFontSystem>) {
    // Bevy's default `CosmicFontSystem` holds an empty font database, so when
    // the primary face lacks a glyph there's nothing to fall back to. Replace
    // it with one constructed from the host's installed fonts.
    let new_fs = cosmic_text::FontSystem::new();
    eprintln!(
        "[poster-ui] system-font fallback enabled ({} faces loaded)",
        new_fs.db().len()
    );
    fs.0 = new_fs;
}

fn try_load_path(paths: &[String], fonts: &mut Assets<bevy::text::Font>, label: &str) -> Option<Handle<bevy::text::Font>> {
    for path in paths {
        let Ok(bytes) = std::fs::read(path) else { continue };
        match bevy::text::Font::try_from_bytes(bytes) {
            Ok(font) => {
                let handle = fonts.add(font);
                eprintln!("[poster-ui] loaded {} font '{}'", label, path);
                return Some(handle);
            }
            Err(e) => {
                eprintln!("[poster-ui] '{}' not a valid font: {:?}", path, e);
            }
        }
    }
    None
}

fn load_embedded(fonts: &mut Assets<bevy::text::Font>, bytes: &[u8], label: &str) -> Handle<bevy::text::Font> {
    match bevy::text::Font::try_from_bytes(bytes.to_vec()) {
        Ok(font) => {
            let handle = fonts.add(font);
            eprintln!("[poster-ui] loaded {} font (embedded Jost)", label);
            handle
        }
        Err(e) => {
            eprintln!("[poster-ui] embedded {} font failed to parse: {:?}", label, e);
            Handle::default()
        }
    }
}

fn load_primary_font(
    paths: Res<FontSearchPaths>,
    mut fonts: ResMut<Assets<bevy::text::Font>>,
    mut commands: Commands,
) {
    // Consumer paths win if any resolve (lets apps swap the primary face via
    // `FontSearchPaths`). Otherwise fall through to the embedded Jost so the
    // library looks right on a fresh checkout.
    let handle = try_load_path(&paths.primary, &mut fonts, "primary")
        .unwrap_or_else(|| load_embedded(&mut fonts, JOST_MEDIUM_TTF, "primary"));
    commands.insert_resource(PrimaryFont(handle));
}

fn load_bold_font(
    paths: Res<FontSearchPaths>,
    mut fonts: ResMut<Assets<bevy::text::Font>>,
    mut commands: Commands,
) {
    let handle = try_load_path(&paths.bold, &mut fonts, "bold")
        .unwrap_or_else(|| load_embedded(&mut fonts, JOST_BOLD_TTF, "bold"));
    commands.insert_resource(BoldFont(handle));
}

fn load_mono_font(
    paths: Res<FontSearchPaths>,
    mut fonts: ResMut<Assets<bevy::text::Font>>,
    mut commands: Commands,
) {
    // No embedded mono fallback — if none of the system paths resolve, leave
    // the handle empty and the `Mono` applier becomes a no-op. The primary
    // font (Jost) still renders numerics fine, just proportionally.
    let handle = try_load_path(&paths.mono, &mut fonts, "mono").unwrap_or_default();
    commands.insert_resource(MonoFont(handle));
}

/// Stamp the primary font onto every `TextFont`. Runs every frame so
/// late-spawned text picks it up. The `Bold` and `Mono` systems run after and
/// override their targeted entities.
fn apply_primary_font(font: Res<PrimaryFont>, mut q: Query<&mut TextFont>) {
    if font.0 == Handle::default() {
        return;
    }
    for mut tf in q.iter_mut() {
        if tf.font != font.0 {
            tf.font = font.0.clone();
        }
    }
}

fn apply_bold_font(font: Res<BoldFont>, mut q: Query<&mut TextFont, With<Bold>>) {
    if font.0 == Handle::default() {
        return;
    }
    for mut tf in q.iter_mut() {
        if tf.font != font.0 {
            tf.font = font.0.clone();
        }
    }
}

fn apply_mono_font(font: Res<MonoFont>, mut q: Query<&mut TextFont, With<Mono>>) {
    if font.0 == Handle::default() {
        return;
    }
    for mut tf in q.iter_mut() {
        if tf.font != font.0 {
            tf.font = font.0.clone();
        }
    }
}
