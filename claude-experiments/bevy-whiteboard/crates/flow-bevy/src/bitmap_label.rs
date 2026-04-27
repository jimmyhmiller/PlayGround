//! Atlas-backed text labels that bypass `bevy_text` / `cosmic-text`.
//! A single shared `GlyphAtlas` rasterises printable ASCII at
//! startup and lazily inserts any other codepoint a label asks for
//! (CoreText cascade on macOS — see `glyph_atlas.rs`). Each label
//! is a parent entity owning a small pool of pre-spawned child
//! `Sprite`s, one per character slot. The `sync_bitmap_labels`
//! system picks up `Changed<BitmapLabel>` and rewrites the
//! children's `TextureAtlas.index`, transform, color, and
//! visibility — no despawn/respawn in the hot path.
//!
//! Trade-offs vs `Text2d`:
//!   + Per-frame cost is ~zero for unchanged labels (Bevy's change
//!     detection skips them entirely).
//!   + Changed labels cost ~O(chars) — pure component writes.
//!   - Monospace only (atlas grid is a fixed cell width).
//!   - First-time use of a codepoint pays one rasterise + GPU
//!     re-upload of the atlas image; tens of microseconds, then
//!     cached forever.

use bevy::image::TextureAtlasLayout;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use crate::glyph_atlas::GlyphAtlas;

/// Font + sizing for the shared atlas. SF Mono is a system font on
/// macOS that ships with most installations. If we ever need to
/// bundle our own, drop the `.otf` into `assets/fonts/` and read
/// from there.
const FONT_PATH: &str = "/Library/Fonts/SF-Mono-Regular.otf";
const DEFAULT_FONT_SIZE: f32 = 11.0;
const DEFAULT_LINE_HEIGHT: f32 = 14.0;

/// Cached metrics + asset handles published alongside the atlas so
/// labels can lay out their child sprites without `&mut Atlas`.
/// Codepoint → slot lookup still requires the atlas itself (sync
/// system holds `ResMut<GlyphAtlas>` for that).
#[derive(Resource, Clone)]
pub struct AtlasMetrics {
    pub image: Handle<Image>,
    pub layout: Handle<TextureAtlasLayout>,
    pub cell_w: f32,
    pub cell_h: f32,
}

pub struct BitmapLabelPlugin;
impl Plugin for BitmapLabelPlugin {
    fn build(&self, app: &mut App) {
        // Init eagerly here (not as a Startup system) so canvas /
        // example loaders running in `Startup` can already pull
        // `AtlasMetrics` as a system parameter. AssetPlugin must
        // already be installed by `DefaultPlugins`, which comes
        // before us in our plugin order.
        let world = app.world_mut();
        // Asset plugin must already be registered by the time we
        // build — `DefaultPlugins` runs first in our plugin chain.
        let font_bytes: &'static [u8] = match std::fs::read(FONT_PATH) {
            Ok(bytes) => Box::leak(bytes.into_boxed_slice()),
            Err(e) => {
                bevy::log::error!(
                    "BitmapLabelPlugin: failed to read {} ({}); labels will be tofu",
                    FONT_PATH, e
                );
                return;
            }
        };
        let cell_w = measure_advance(font_bytes, DEFAULT_FONT_SIZE, '0');
        let cell_h = DEFAULT_LINE_HEIGHT;
        // Lift `Assets<Image>` out so we can hold both maps mutably.
        let atlas = world.resource_scope::<Assets<Image>, _>(|world, mut images| {
            let mut layouts = world.resource_mut::<Assets<TextureAtlasLayout>>();
            GlyphAtlas::new(
                font_bytes,
                DEFAULT_FONT_SIZE,
                cell_w,
                cell_h,
                &mut images,
                &mut layouts,
            )
        });
        world.insert_resource(AtlasMetrics {
            image: atlas.image.clone(),
            layout: atlas.layout.clone(),
            cell_w,
            cell_h,
        });
        world.insert_resource(atlas);

        app.add_systems(PostUpdate, sync_bitmap_labels);
    }
}


fn measure_advance(font_bytes: &[u8], font_size: f32, ch: char) -> f32 {
    use swash::FontRef;
    let font = FontRef::from_index(font_bytes, 0).expect("font parse");
    let glyph_id = font.charmap().map(ch);
    if glyph_id == 0 {
        return font_size * 0.6;
    }
    let metrics = font.metrics(&[]).scale(font_size);
    if metrics.average_width > 0.0 {
        metrics.average_width
    } else {
        font_size * 0.6
    }
}

/// Component on the parent entity. Children are atlas-sprite chars.
/// `Changed<BitmapLabel>` only fires when `text` is actually
/// mutated — guard your writes with `if label.text != new { ... }`.
#[derive(Component)]
pub struct BitmapLabel {
    pub text: String,
    pub color: Color,
    pub align: TextAlign,
    /// Capacity — how many child sprite slots were pre-spawned.
    /// `text.chars().count()` beyond this gets clipped.
    pub capacity: usize,
    pub cell_w: f32,
    pub cell_h: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TextAlign { Left, Center, Right }

/// Marker on each child sprite. `index` is the position within the
/// label (0 = leftmost).
#[derive(Component, Clone, Copy)]
pub struct BitmapLabelChar { pub index: usize }

/// Helper: spawn a `BitmapLabel` parent + `capacity` child sprites
/// underneath an existing parent's child-spawner. Used from
/// `nodes::spawn_node_entity` to attach a label to a node body.
pub fn spawn_bitmap_label_child(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    metrics: &AtlasMetrics,
    text: &str,
    color: Color,
    font_size: f32,
    align: TextAlign,
    z: f32,
    y_offset: f32,
) {
    let scale = font_size / DEFAULT_FONT_SIZE;
    let cell_w = metrics.cell_w * scale;
    let cell_h = metrics.cell_h * scale;
    let capacity = text.chars().count().max(8);

    let label = parent.spawn((
        BitmapLabel {
            text: text.to_string(),
            color,
            align,
            capacity,
            cell_w,
            cell_h,
        },
        Transform::from_xyz(0.0, y_offset, z),
        Visibility::Inherited,
    )).id();

    parent.commands().entity(label).with_children(|child_parent| {
        spawn_label_chars(child_parent, metrics, capacity, color, cell_w, cell_h);
    });
}

/// Spawn `capacity` initially-hidden tofu sprites under the current
/// child-spawner. Public so `nodes::spawn_node_entity` can reuse it
/// when it spawns labels with custom marker components attached.
pub fn spawn_label_chars(
    child_parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    metrics: &AtlasMetrics,
    capacity: usize,
    color: Color,
    cell_w: f32,
    cell_h: f32,
) {
    for i in 0..capacity {
        child_parent.spawn((
            BitmapLabelChar { index: i },
            Sprite {
                image: metrics.image.clone(),
                texture_atlas: Some(TextureAtlas {
                    layout: metrics.layout.clone(),
                    index: 0,
                }),
                color,
                custom_size: Some(Vec2::new(cell_w, cell_h)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.0),
            Visibility::Hidden,
        ));
    }
}

/// Per-frame sync: picks up `Changed<BitmapLabel>` parents, looks
/// up each char in the atlas (lazily inserting on miss), and writes
/// the result into the pre-spawned child sprites. Takes
/// `ResMut<GlyphAtlas>` because lookup may insert a new glyph.
fn sync_bitmap_labels(
    atlas: Option<ResMut<GlyphAtlas>>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<TextureAtlasLayout>>,
    parents: Query<(&BitmapLabel, &Children), Changed<BitmapLabel>>,
    mut chars: Query<(&BitmapLabelChar, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    let Some(mut atlas) = atlas else { return };
    for (label, children) in parents.iter() {
        let chars_iter: Vec<char> = label.text.chars().take(label.capacity).collect();
        let n = chars_iter.len();
        let total_w = n as f32 * label.cell_w;
        let x_origin = match label.align {
            TextAlign::Left => 0.0,
            TextAlign::Center => -total_w * 0.5,
            TextAlign::Right => -total_w,
        };
        for child in children.iter() {
            let Ok((meta, mut sprite, mut transform, mut vis)) = chars.get_mut(child) else {
                continue;
            };
            let i = meta.index;
            if i < n {
                let slot = atlas.lookup_or_insert(chars_iter[i], &mut images, &mut layouts) as usize;
                if let Some(ta) = sprite.texture_atlas.as_mut() {
                    if ta.index != slot {
                        ta.index = slot;
                    }
                }
                if sprite.color != label.color {
                    sprite.color = label.color;
                }
                if sprite.custom_size != Some(Vec2::new(label.cell_w, label.cell_h)) {
                    sprite.custom_size = Some(Vec2::new(label.cell_w, label.cell_h));
                }
                let want_x = x_origin + i as f32 * label.cell_w;
                if transform.translation.x != want_x {
                    transform.translation.x = want_x;
                }
                if !matches!(*vis, Visibility::Inherited | Visibility::Visible) {
                    *vis = Visibility::Inherited;
                }
            } else if !matches!(*vis, Visibility::Hidden) {
                *vis = Visibility::Hidden;
            }
        }
    }
}
