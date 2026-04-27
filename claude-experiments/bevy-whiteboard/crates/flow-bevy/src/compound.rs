//! Compound nodes on the canvas — visibility scoping and a small
//! visual abstraction so the renderer can draw a compound as a single
//! body without knowing the inner topology.
//!
//! ## What this module owns
//!
//! - [`CompoundMembership`] — a flat `child NodeId → enclosing compound
//!   NodeId` map, recomputed whenever a canvas is loaded. Source of
//!   truth for "is this node inside something."
//! - [`CurrentScope`] — the user's current zoom-into-compound state.
//!   `None` = top of the world, `Some(nid)` = the inside of compound
//!   `nid`. Phase 2 (double-click + Esc) toggles it; phase 1 visibility
//!   already honors it.
//! - [`CompoundVisual`] — the visual descriptor. Today the only
//!   variant is [`CompoundVisual::LabeledBox`]; the enum is here so a
//!   richer authoring surface (a stack/expression DSL, a script, etc.)
//!   can be added by appending variants without touching the spawn
//!   path or anyone else's call site.
//! - [`Inside`] component — marker we drop on every Bevy entity whose
//!   `flow::NodeId` belongs to a compound. `sync_compound_visibility`
//!   reads it together with [`CurrentScope`] to decide each frame which
//!   inner / boundary entities are visible.
//!
//! ## What this module deliberately does NOT do
//!
//! - It does not spawn entities. The canvas loader does that and
//!   stamps `Inside` / `EdgeInside` on the right ones during the same
//!   pass.
//! - It does not author compound visuals. The current shape is a hard
//!   fallback; visual authoring (ie. coming from `visual.json` or the
//!   future stack DSL) attaches a [`CompoundVisual`] component when
//!   the canvas is loaded. The renderer reads whichever one is there.
//!
//! ## Membership rule
//!
//! A node is inside compound `C` iff its name starts with `C.name + "::"`
//! (for an outer compound, that means the *closest* such prefix wins).
//! This matches what the DSL expansion pass produces: `Life::Cell_3_4`
//! is inside `Life`, `Outer::Inner::Leaf` is inside `Inner` first and
//! transitively inside `Outer`. We keep only the innermost mapping in
//! [`CompoundMembership`] — outer ancestors are reachable by walking
//! the chain.

use std::collections::BTreeMap;

use bevy::prelude::*;
use flow::sim::Sim;
use flow::NodeId;

use crate::bitmap_label::{AtlasMetrics, BitmapLabel, TextAlign, spawn_label_chars};
use crate::bridge::{EntityMaps, FlowNodeRef};
use crate::camera::{MainCamera, cursor_to_world};
use crate::nodes::{BodyShape, NodeAssetCache, hit_size};
use crate::sim_driver::SimSnapshotRes;
use crate::theme::Theme;
use crate::tool::{ActiveTool, Tool};

pub struct CompoundPlugin;
impl Plugin for CompoundPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CompoundMembership>()
            .init_resource::<CurrentScope>()
            .init_resource::<CompoundParamRegistry>()
            .init_resource::<DoubleClickTracker>()
            .add_systems(Update, (
                drill_in_on_double_click,
                drill_out_on_escape,
                sync_compound_visibility,
                sync_grid_cell_paint,
            ).chain());
    }
}

/// Authoring-time compound metadata, indexed by qualified compound
/// name. Populated at canvas-load and re-populated on every reload;
/// read by the inspector to render compound-param rows.
///
/// **Read-only today.** Editing + rebuild is the natural next step:
/// the resource would gain an `overrides` map and an "apply" entry
/// point that re-runs `parse → expand-with-overrides → lower` and
/// respawns the world.
#[derive(Resource, Default, Debug, Clone)]
pub struct CompoundParamRegistry {
    pub by_name: std::collections::BTreeMap<String, Vec<flow::dsl::expand::CompoundParamEntry>>,
}

/// `child NodeId → innermost-enclosing compound NodeId`.
///
/// Recomputed wholesale every time the canvas reloads. Node entities
/// that are inside any compound carry an [`Inside`] marker that mirrors
/// the *inner-most* entry from this map; outer ancestry is recovered
/// by walking the map (`while let Some(parent) = membership.get(&nid)`).
#[derive(Resource, Default, Debug, Clone)]
pub struct CompoundMembership {
    pub parent: BTreeMap<NodeId, NodeId>,
}

impl CompoundMembership {
    /// Innermost enclosing compound for `nid`, if any.
    pub fn parent_of(&self, nid: NodeId) -> Option<NodeId> {
        self.parent.get(&nid).copied()
    }

    /// True iff `nid` is itself an "inside" node — i.e. some compound
    /// owns it. Compound bodies themselves return false (they may be
    /// nested inside *another* compound, but the inner-of map answers
    /// "is this hidden from the top level").
    pub fn is_inside_anything(&self, nid: NodeId) -> bool {
        self.parent.contains_key(&nid)
    }

    /// Walks the parent chain, yielding ancestors innermost-first,
    /// stopping at the outer-most compound. Used by visibility logic
    /// when scoping into deeply nested compounds.
    pub fn ancestors(&self, nid: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let mut cur = self.parent_of(nid);
        std::iter::from_fn(move || {
            let n = cur?;
            cur = self.parent_of(n);
            Some(n)
        })
    }
}

/// Build a [`CompoundMembership`] from the sim by matching node-name
/// prefixes against compound names. The DSL expansion pass guarantees
/// inner names look like `<Outer>::<...>::<Leaf>`, so a single linear
/// pass over `(compound_name, node_name)` pairs suffices.
///
/// Tie-breaks on prefix length: the longest matching compound prefix
/// wins, so for `Outer::Inner::Leaf` we record `Inner` (innermost), not
/// `Outer` (which is still recoverable via the parent chain through
/// `Inner`).
pub fn compute_membership(sim: &Sim) -> CompoundMembership {
    // Collect (name → NodeId) for compounds and (name → NodeId) for
    // every node, in deterministic order.
    let compounds: Vec<(String, NodeId)> = sim
        .nodes
        .iter()
        .filter(|(_, n)| n.is_compound())
        .map(|(id, n)| (n.name.clone(), *id))
        .collect();

    let mut parent: BTreeMap<NodeId, NodeId> = BTreeMap::new();

    for (nid, node) in sim.nodes.iter() {
        // A compound itself can be nested inside another compound; we
        // *do* record that ancestry. But a compound is never recorded
        // as its own parent.
        let mut best: Option<(usize, NodeId)> = None;
        for (cname, cid) in &compounds {
            if *cid == *nid {
                continue;
            }
            let prefix = format!("{}::", cname);
            if node.name.starts_with(&prefix) {
                let len = cname.len();
                if best.map_or(true, |(blen, _)| len > blen) {
                    best = Some((len, *cid));
                }
            }
        }
        if let Some((_, cid)) = best {
            parent.insert(*nid, cid);
        }
    }
    CompoundMembership { parent }
}

/// Where the user's view is currently anchored. `None` = top-level
/// canvas (compounds appear as a single body, their innards hidden).
/// `Some(nid)` = "drilled into" the compound `nid`; that compound's
/// direct children are revealed and everything else is hidden.
///
/// Setters / consumers live in the (forthcoming) double-click handler;
/// this module only owns the resource and the visibility system that
/// reacts to it.
#[derive(Resource, Default, Debug, Clone)]
pub struct CurrentScope(pub Option<NodeId>);

/// Visual descriptor for a compound's outward face. **Intentionally
/// minimal today** — the variant set will grow as authoring needs do.
///
/// Why an enum and not just hard-coded behaviour: the renderer keys off
/// this type, so swapping in a richer authoring surface (a stack-based
/// drawing DSL, a script, a class-style visual reference, etc.) means
/// adding a variant + matching it in one place rather than untangling
/// renderer internals.
#[derive(Component, Clone, Debug)]
pub enum CompoundVisual {
    /// A simple ink-bordered rectangle with the compound's display
    /// name centered inside. Authoring: caller picks a fallback size;
    /// the body label uses the unqualified compound name (everything
    /// after the last `::`).
    LabeledBox { size: Vec2, label: String },
    /// Render the compound as a regular grid of small bodies, one per
    /// inner member node, painted by reading a slot off each member.
    /// The compound's own outline still draws around the grid as the
    /// "frame," so the user can see where the compound ends and what
    /// the rest of the canvas is.
    ///
    /// `member_resolver(col, row)` produces the inner node's full
    /// name; we pre-resolve all of them at spawn time so the painter
    /// can do a flat `NodeId` lookup each frame instead of redoing
    /// string substitution.
    Grid {
        columns: u32,
        rows: u32,
        cell_size: f32,
        gap: f32,
        /// `(col, row) -> NodeId`. Pre-resolved at canvas-load time;
        /// any unresolved entries stay `None` and get drawn as dim
        /// "missing-cell" placeholders so the user sees the gap.
        members: Vec<Option<flow::NodeId>>,
        paint: GridCellPaint,
    },
}

/// How a single grid cell is colored. Mirrors the existing
/// `BinarySlotPaint` semantics so the JSON authoring shape stays
/// uniform across class paints and compound paints.
#[derive(Clone, Debug)]
pub struct GridCellPaint {
    pub slot: String,
    pub on: Color,
    pub off: Color,
}

impl CompoundVisual {
    pub fn default_for(compound_name: &str) -> Self {
        // Strip any enclosing `Outer::` prefix — we want "Life", not
        // "Outer::Life", on the body.
        let label = compound_name.rsplit("::").next().unwrap_or(compound_name).to_string();
        CompoundVisual::LabeledBox {
            size: Vec2::new(220.0, 140.0),
            label,
        }
    }
}

/// Substitute `{x}` / `{y}` placeholders in a member-pattern string.
/// Unrecognized braces (anything other than `{x}` or `{y}`) are
/// preserved verbatim so a future richer pattern format can grow into
/// the same syntax. No escapes — patterns are fully under the canvas
/// author's control and don't need to defend against literal `{x}`.
pub fn resolve_member_pattern(pattern: &str, x: u32, y: u32) -> String {
    let mut out = String::with_capacity(pattern.len());
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            let mut buf = String::new();
            let mut closed = false;
            while let Some(&pc) = chars.peek() {
                chars.next();
                if pc == '}' { closed = true; break; }
                buf.push(pc);
            }
            if !closed {
                out.push('{');
                out.push_str(&buf);
                continue;
            }
            match buf.as_str() {
                "x" => out.push_str(&x.to_string()),
                "y" => out.push_str(&y.to_string()),
                other => {
                    out.push('{');
                    out.push_str(other);
                    out.push('}');
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Markers + visibility
// ─────────────────────────────────────────────────────────────────────

/// Marker stamped on each node entity that lives inside a compound.
/// Carries the **innermost** enclosing compound's `NodeId`. The
/// visibility system reads this against [`CurrentScope`] to decide
/// per-frame whether the entity is rendered.
#[derive(Component, Clone, Copy, Debug)]
pub struct Inside(pub NodeId);

/// Marker stamped on edge entities both of whose endpoints sit inside
/// the same compound (i.e. internal wiring). Carries that compound's
/// `NodeId`. Boundary edges (one endpoint inside, one outside) are
/// **not** marked — they remain visible in either scope today; future
/// work will reroute them visually to attach to the compound's body.
#[derive(Component, Clone, Copy, Debug)]
pub struct EdgeInside(pub NodeId);

/// Marker stamped on the compound's own body entity. Mirrors the
/// `flow::NodeId` so the double-click drill-in handler can tell the
/// difference between "user clicked a compound" (zoom into it) and
/// "user clicked a leaf" (selection / inspector).
#[derive(Component, Clone, Copy, Debug)]
pub struct CompoundBodyMarker(pub NodeId);

/// Marker stamped on each child mesh of a [`CompoundVisual::Grid`]
/// body. Carries the source node + the slot to read + the on/off
/// colors. The per-frame painter ([`sync_grid_cell_paint`]) reads
/// this and swaps the entity's material handle accordingly. Same
/// caching strategy as `nodes::sync_binary_slot_paint`: deduplicate
/// material assets via [`NodeAssetCache`] so a 100×100 grid doesn't
/// allocate 10 000 unique materials each frame.
#[derive(Component, Clone, Debug)]
pub struct GridCellPaintRef {
    pub source: Option<NodeId>,
    pub slot: String,
    pub on: Color,
    pub off: Color,
}

/// Per-frame visibility update. Cheap when nothing changed; only does
/// work the first frame after the canvas loads or [`CurrentScope`]
/// flips. Intentionally bypasses any "dirty" gating for now — it's a
/// single iteration over compound-tagged entities and the vast
/// majority of canvases have a few hundred at most.
fn sync_compound_visibility(
    scope: Res<CurrentScope>,
    mut nodes: Query<(&Inside, &mut Visibility), (Without<EdgeInside>, Without<CompoundBodyMarker>)>,
    mut edges: Query<(&EdgeInside, &mut Visibility), (Without<Inside>, Without<CompoundBodyMarker>)>,
    mut bodies: Query<(&CompoundBodyMarker, Option<&Inside>, &mut Visibility), Without<EdgeInside>>,
) {
    for (inside, mut vis) in nodes.iter_mut() {
        *vis = compute_inside_visibility(inside.0, &scope.0);
    }
    for (inside, mut vis) in edges.iter_mut() {
        *vis = compute_inside_visibility(inside.0, &scope.0);
    }
    // A compound body is visible iff (a) we're at the scope where it
    // appears as one face — i.e. the user is *outside* this compound —
    // AND (b) any enclosing parent compound is the current scope (so
    // drilling into Outer reveals Inner's body, not Outer::Inner::Leaf).
    for (marker, inside, mut vis) in bodies.iter_mut() {
        let parent_scope = inside.map(|i| i.0);
        *vis = if scope.0 == Some(marker.0) {
            // Drilled INTO this compound — hide its outer face so the
            // interior reads cleanly.
            Visibility::Hidden
        } else if parent_scope == scope.0 {
            // Either both are None (top-level) or both point at the
            // same enclosing compound (drilled into our parent).
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// "Should this inside-compound entity be visible right now?" Pulled
/// out so node and edge updates share the same rule, and so it's easy
/// to reuse from the (incoming) drill-in handler when it preflights a
/// scope change.
fn compute_inside_visibility(parent: NodeId, scope: &Option<NodeId>) -> Visibility {
    match scope {
        // Top-level view: nothing inside any compound is visible.
        None => Visibility::Hidden,
        // Drilled-in view: only entities directly inside the scope's
        // compound are visible. (Multi-level nesting will need to walk
        // the membership chain — wired up when we add nested-drill-in
        // UI; for now LCD is fine because only one nesting level is
        // exercised.)
        Some(s) if *s == parent => Visibility::Visible,
        Some(_) => Visibility::Hidden,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Spawn
// ─────────────────────────────────────────────────────────────────────

/// Spawn the Bevy entity that visually represents a compound. This is
/// the **easy thing**: a paper-coloured rectangle with an ink border
/// and the compound's display name centered inside, plus a soft drop
/// shadow so it reads as a body and not a frame.
///
/// Designed to be the only spawn point for compound visuals so that
/// when [`CompoundVisual`] grows new variants — a stack-DSL view, a
/// child-driven layout, etc. — there's exactly one match arm to
/// extend. Today only [`CompoundVisual::LabeledBox`] exists.
pub fn spawn_compound_body_entity(
    commands: &mut Commands,
    cache: &mut NodeAssetCache,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    maps: &mut EntityMaps,
    theme: &Theme,
    metrics: &AtlasMetrics,
    flow_id: NodeId,
    visual: CompoundVisual,
    full_name: String,
    pos: Vec2,
) -> Entity {
    match visual {
        CompoundVisual::Grid { columns, rows, cell_size, gap, members, paint } => {
            // Outer frame just big enough to contain the grid plus a
            // little padding on every side so the cells don't touch
            // the ink border.
            let pad = 12.0;
            let pitch = cell_size + gap;
            let inner_w = (columns as f32) * pitch - gap;
            let inner_h = (rows as f32) * pitch - gap;
            let frame_size = Vec2::new(inner_w + 2.0 * pad, inner_h + 2.0 * pad);

            let frame_shape = BodyShape::Rect(frame_size);
            let body_handle = cache.body_mesh(&frame_shape, meshes);
            let border_handle = cache.body_mesh(&pad_rect(&frame_shape, 3.0), meshes);
            let shadow_handle = cache.body_mesh(&pad_rect(&frame_shape, 3.0), meshes);
            let paper_mat = cache.material(theme.paper, materials);
            let ink_mat = cache.material(theme.ink, materials);
            let ink_soft_mat = cache.material(theme.ink_soft, materials);

            let entity = commands.spawn((
                Mesh2d(body_handle),
                MeshMaterial2d(paper_mat.clone()),
                Transform::from_translation(pos.extend(1.0)),
                FlowNodeRef(flow_id),
                CompoundBodyMarker(flow_id),
                frame_shape,
            )).id();
            maps.node_to_entity.insert(flow_id, entity);
            maps.entity_to_node.insert(entity, flow_id);

            // Pre-compute the cell mesh once — every cell is the same
            // square, so they all share one mesh handle.
            let cell_shape = BodyShape::Rect(Vec2::new(cell_size, cell_size));
            let cell_mesh = cache.body_mesh(&cell_shape, meshes);

            commands.entity(entity).with_children(|parent| {
                // Frame border + shadow.
                parent.spawn((
                    Mesh2d(border_handle),
                    MeshMaterial2d(ink_mat.clone()),
                    Transform::from_xyz(0.0, 0.0, -0.1),
                ));
                parent.spawn((
                    Mesh2d(shadow_handle),
                    MeshMaterial2d(ink_soft_mat),
                    Transform::from_xyz(6.0, -6.0, -0.2),
                ));

                // Grid cells. Layout: row 0 at the top, increasing y
                // visually downward — matches how Life canvases store
                // their cell positions in `visual.json`. The inner
                // origin is the top-left of the cell rectangle area.
                let origin_x = -inner_w * 0.5 + cell_size * 0.5;
                let origin_y =  inner_h * 0.5 - cell_size * 0.5;
                for row in 0..rows {
                    for col in 0..columns {
                        let idx = (row * columns + col) as usize;
                        let source = members.get(idx).copied().flatten();
                        let cx = origin_x + (col as f32) * pitch;
                        let cy = origin_y - (row as f32) * pitch;
                        // Default to "off" — the per-frame painter
                        // will flip to "on" as soon as the snapshot
                        // shows the source slot is non-zero.
                        let mat = cache.material(paint.off, materials);
                        parent.spawn((
                            Mesh2d(cell_mesh.clone()),
                            MeshMaterial2d(mat),
                            Transform::from_xyz(cx, cy, 0.05),
                            GridCellPaintRef {
                                source,
                                slot: paint.slot.clone(),
                                on: paint.on,
                                off: paint.off,
                            },
                        ));
                    }
                }
            });

            let _ = full_name;
            entity
        }
        CompoundVisual::LabeledBox { size, label } => {
            let shape = BodyShape::Rect(size);
            let body_handle = cache.body_mesh(&shape, meshes);
            let border_handle = cache.body_mesh(&pad_rect(&shape, 3.0), meshes);
            let shadow_handle = cache.body_mesh(&pad_rect(&shape, 3.0), meshes);
            let paper_mat = cache.material(theme.paper, materials);
            let ink_mat = cache.material(theme.ink, materials);
            let ink_soft_mat = cache.material(theme.ink_soft, materials);

            let entity = commands.spawn((
                Mesh2d(body_handle),
                MeshMaterial2d(paper_mat),
                Transform::from_translation(pos.extend(1.0)),
                FlowNodeRef(flow_id),
                CompoundBodyMarker(flow_id),
                shape,
            )).id();
            maps.node_to_entity.insert(flow_id, entity);
            maps.entity_to_node.insert(entity, flow_id);

            commands.entity(entity).with_children(|parent| {
                // Ink outline behind the body.
                parent.spawn((
                    Mesh2d(border_handle),
                    MeshMaterial2d(ink_mat.clone()),
                    Transform::from_xyz(0.0, 0.0, -0.1),
                ));
                // Subtle drop shadow so the body reads as a panel
                // rather than a thin frame.
                parent.spawn((
                    Mesh2d(shadow_handle),
                    MeshMaterial2d(ink_soft_mat),
                    Transform::from_xyz(6.0, -6.0, -0.2),
                ));
                // Centered label. We deliberately render just the
                // unqualified compound name (e.g. "Life") rather than
                // the full path — when the user is at the top level
                // the path is always one level, and inside a deeper
                // scope the breadcrumb (forthcoming) carries the rest.
                let label_text = label.to_uppercase();
                let label_capacity = label_text.chars().count().max(8);
                let label_entity = parent.spawn((
                    BitmapLabel {
                        text: label_text,
                        color: theme.ink,
                        align: TextAlign::Center,
                        capacity: label_capacity,
                        cell_w: metrics.cell_w,
                        cell_h: metrics.cell_h,
                    },
                    Transform::from_xyz(0.0, 0.0, 0.1),
                    Visibility::Inherited,
                )).id();
                parent.commands().entity(label_entity).with_children(|cp| {
                    spawn_label_chars(cp, metrics, label_capacity, theme.ink, metrics.cell_w, metrics.cell_h);
                });
            });

            // Suppress the unused-warning on the full qualified name
            // — we keep the parameter so the future visual variants
            // can use it (e.g. a script-driven body that wants to
            // resolve sibling names relative to the compound).
            let _ = full_name;
            entity
        }
    }
}

/// Pad a rectangular body to draw the surrounding ink border / shadow
/// at the right size. Mirrors `nodes::padded_shape`'s rect arm; kept
/// local so this module doesn't depend on a crate-private helper.
fn pad_rect(shape: &BodyShape, pad: f32) -> BodyShape {
    match shape {
        BodyShape::Rect(v) => BodyShape::Rect(Vec2::new(v.x + pad * 2.0, v.y + pad * 2.0)),
        BodyShape::Circle(r) => BodyShape::Circle(r + pad),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Drill-in / drill-out
// ─────────────────────────────────────────────────────────────────────

/// Time-window state for distinguishing single from double clicks on
/// compound bodies. Click-history isn't a primitive Bevy gives us, so
/// we keep the smallest possible state: which entity was last clicked,
/// and at what wall-clock time. A second click on the same entity
/// within [`DOUBLE_CLICK_WINDOW_SECS`] becomes a drill-in.
///
/// Lives outside the regular drag-state machinery deliberately — the
/// drag handler is in `nodes.rs` and we don't want to bend that around
/// compound-specific concerns. Drill-in is a clean second pass over
/// the same input frame.
#[derive(Resource, Default)]
pub struct DoubleClickTracker {
    last: Option<(Entity, f64)>,
}

const DOUBLE_CLICK_WINDOW_SECS: f64 = 0.4;

/// Detect double-clicks on compound bodies and update [`CurrentScope`]
/// to drill into the clicked compound. Single-clicks fall through to
/// the existing select / drag pipeline (we don't consume the input —
/// `nodes::begin_drag` runs later in the same frame and handles
/// selection just fine).
fn drill_in_on_double_click(
    buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    active: Res<ActiveTool>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    compounds: Query<(Entity, &Transform, &BodyShape, &CompoundBodyMarker)>,
    ui: Query<&Interaction>,
    mut tracker: ResMut<DoubleClickTracker>,
    mut scope: ResMut<CurrentScope>,
) {
    if !matches!(active.0, Tool::Select) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };

    // Find the topmost compound under the cursor (compound bodies
    // don't currently overlap, but we keep iteration order
    // deterministic to mirror `begin_drag`'s pick).
    let mut picked: Option<(Entity, NodeId)> = None;
    for (e, tf, shape, marker) in compounds.iter() {
        let size = hit_size(shape);
        let min = tf.translation.truncate() - size * 0.5;
        let max = tf.translation.truncate() + size * 0.5;
        if world.x >= min.x && world.x <= max.x && world.y >= min.y && world.y <= max.y {
            picked = Some((e, marker.0));
        }
    }
    let Some((entity, nid)) = picked else {
        // Click in empty space: clear the click history but don't
        // change scope. Esc is the dedicated drill-out gesture.
        tracker.last = None;
        return;
    };

    let now = time.elapsed_secs_f64();
    let is_double = matches!(
        tracker.last,
        Some((prev_entity, prev_time))
            if prev_entity == entity && (now - prev_time) <= DOUBLE_CLICK_WINDOW_SECS
    );
    if is_double {
        scope.0 = Some(nid);
        tracker.last = None; // reset so a 3rd quick click doesn't re-fire
    } else {
        tracker.last = Some((entity, now));
    }
}

/// Per-frame paint sync for grid cells. Same shape as
/// `nodes::sync_binary_slot_paint`: read each cell's source-slot value
/// from the latest sim snapshot and swap material handles between two
/// cached colors. Only does work for grid-cell entities, so canvases
/// without a grid-shaped compound pay nothing.
fn sync_grid_cell_paint(
    snapshot: Res<SimSnapshotRes>,
    mut cache: ResMut<NodeAssetCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut cells: Query<(&GridCellPaintRef, &mut MeshMaterial2d<ColorMaterial>)>,
) {
    for (paint, mut mat) in cells.iter_mut() {
        let on = paint
            .source
            .and_then(|nid| snapshot.0.nodes.get(&nid))
            .and_then(|n| n.slots.get(&paint.slot))
            .map(|v| match v {
                flow::Value::Int(i) => *i != 0,
                flow::Value::Bool(b) => *b,
                _ => false,
            })
            .unwrap_or(false);
        let want = if on { paint.on } else { paint.off };
        let want_handle = cache.material(want, &mut materials);
        if mat.0.id() != want_handle.id() {
            mat.0 = want_handle;
        }
    }
}

/// Pop the current scope on Escape. Single-level drill-out for now —
/// when nested compounds are exercised, this will become "walk up one
/// level" using [`CompoundMembership::ancestors`].
fn drill_out_on_escape(
    keys: Res<ButtonInput<KeyCode>>,
    membership: Res<CompoundMembership>,
    mut scope: ResMut<CurrentScope>,
) {
    if !keys.just_pressed(KeyCode::Escape) { return; }
    let Some(current) = scope.0 else { return; };
    // If the current scope is itself nested inside another compound,
    // pop to that one. Otherwise pop all the way to the top level.
    scope.0 = membership.parent_of(current);
}
