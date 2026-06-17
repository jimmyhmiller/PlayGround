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
//! - [`Scoped`] component — **the** marker we drop on every Bevy
//!   entity that visualizes a sim entity. Carries the `NodeId` whose
//!   scope it inherits (the node itself for a node entity, an edge's
//!   canonical owner for an edge / packet / probe). One marker, one
//!   visibility system ([`sync_scoped_visibility`]), every layer.
//!
//! ## What this module deliberately does NOT do
//!
//! - It does not spawn entities. The canvas loader does that and
//!   stamps `Scoped` on the right ones during the same pass; subsystems
//!   that draw via single GPU entities (the packet cloud, the edge
//!   line-list mesh) apply the same predicate inline at draw time
//!   using [`canonical_edge_owner`] and [`CompoundMembership`].
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
            .init_resource::<CachedCanvasAst>()
            .init_resource::<CompoundOverrides>()
            .init_resource::<DoubleClickTracker>()
            .init_resource::<CanvasPositions>()
            .init_resource::<crate::canvas::CurrentCanvasVisual>()
            .add_message::<RebuildCompound>()
            .init_resource::<SimTopologyFingerprint>()
            .add_systems(Update, (
                drill_in_on_double_click,
                drill_out_on_escape,
                detect_dynamic_topology_change,
                sync_canvas_population,
                sync_grid_cell_paint,
            ).chain())
            // Rebuild handler runs before the population sync so new
            // sim entities are visible to it on the same frame.
            .add_systems(Update, handle_rebuild_compound.before(sync_canvas_population));
    }
}

/// Authoring-time compound metadata, indexed by qualified compound
/// name. Populated at canvas-load and re-populated on every reload;
/// read by the inspector to render compound-param rows.
#[derive(Resource, Default, Debug, Clone)]
pub struct CompoundParamRegistry {
    pub by_name: std::collections::BTreeMap<String, Vec<flow::dsl::expand::CompoundParamEntry>>,
}

/// Parsed (un-expanded) main.flow AST, cached at canvas-load so the
/// surgical-rebuild path can re-expand any compound with new
/// overrides without touching the disk. Wrapped in `Arc` so cheap
/// clones land on event readers without copying the whole tree.
#[derive(Resource, Default, Clone)]
pub struct CachedCanvasAst(pub std::sync::Arc<flow::dsl::ast::File>);

/// User-supplied param overrides per compound. Each compound name
/// keys into a `param_name → CtValue` map; missing entries fall back
/// to the param's declared default at expansion time.
///
/// The slider widgets mutate this resource; emitting [`RebuildCompound`]
/// is what actually applies the new values to the running canvas.
/// Decoupling the two means slider drag stays interactive even while
/// a rebuild is in flight, and "set the value but don't rebuild yet"
/// (e.g. mid-drag) is a single-resource write.
#[derive(Resource, Default, Debug, Clone)]
pub struct CompoundOverrides {
    pub by_compound: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, flow::dsl::expand::CtValue>,
    >,
}

/// Fired when a compound's interior should be torn down and rebuilt
/// from its source AST with the current overrides applied. Surgical:
/// only entities and edges *inside* the named compound are touched —
/// top-level state, sim time, scenarios, viewport, selection of
/// non-interior entities — all preserved.
#[derive(Message, Debug, Clone)]
pub struct RebuildCompound(pub String);

/// `child NodeId → innermost-enclosing compound NodeId`.
///
/// Recomputed wholesale every time the canvas reloads. Every visual
/// entity carries a [`Scoped`] marker pointing at the NodeId whose
/// scope it inherits; this map answers "what compound does that
/// NodeId live in" in one lookup. Outer ancestry is recovered by
/// walking the map repeatedly.
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

/// Cheap fingerprint of the sim's node/edge topology, used to notice
/// when nodes are spawned or despawned *dynamically* (by `Effect::Spawn`
/// / `Effect::Despawn` inside the running sim — e.g. an autoscaler) as
/// opposed to by canvas load or compound rebuild. Those static paths
/// already recompute [`CompoundMembership`]; the dynamic path has no
/// other hook, so [`detect_dynamic_topology_change`] watches this.
#[derive(Resource, Default)]
pub struct SimTopologyFingerprint {
    nodes: usize,
    edges: usize,
    /// Monotonic id high-water mark — distinguishes "spawned N, despawned
    /// N" (same counts, different ids) from a true no-op.
    next_node_id: u64,
}

/// Notice runtime spawn/despawn and refresh [`CompoundMembership`] so the
/// population reconciler picks up the new/removed nodes. Membership is a
/// pure function of node names, so dynamically-spawned nodes (whose names
/// inherit their spawner's compound prefix) land in the right scope
/// automatically. Writing the resource only when the fingerprint moved
/// keeps `is_resource_changed` from firing every frame.
pub fn detect_dynamic_topology_change(
    mut driver: ResMut<crate::sim_driver::SimDriverRes>,
    mut fingerprint: ResMut<SimTopologyFingerprint>,
    mut membership: ResMut<CompoundMembership>,
) {
    // Copy the prior fingerprint out before the closure: `with_sim_mut`
    // takes a `'static` closure (it may run on the sim worker thread),
    // so it can't borrow `fingerprint`.
    let (prev_nodes, prev_edges, prev_next) =
        (fingerprint.nodes, fingerprint.edges, fingerprint.next_node_id);
    let (nodes, edges, next_id, new_membership) = driver.0.with_sim_mut(move |sim| {
        let changed = sim.nodes.len() != prev_nodes
            || sim.edges.len() != prev_edges
            || sim.next_node_id_hint() != prev_next;
        let m = if changed { Some(compute_membership(sim)) } else { None };
        (sim.nodes.len(), sim.edges.len(), sim.next_node_id_hint(), m)
    });
    let Some(new_membership) = new_membership else { return };
    fingerprint.nodes = nodes;
    fingerprint.edges = edges;
    fingerprint.next_node_id = next_id;
    // Assigning through `DerefMut` flags the resource changed, waking
    // `sync_canvas_population` this same frame (it runs after us in the
    // chain).
    *membership = new_membership;
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

/// Last-known canvas position of every node, keyed by `NodeId`.
///
/// `sync_canvas_population` despawns out-of-scope node entities when the
/// user drills into a compound, and respawns them on drill-out. The
/// position otherwise lives only on the (now-gone) Bevy `Transform`, so
/// without this the respawn falls back to a default grid and the layout
/// scrambles. We snapshot each node's live position right before
/// despawning it and restore it on respawn — which also preserves any
/// dragging the user did, since the captured `Transform` reflects it.
#[derive(Resource, Default, Debug, Clone)]
pub struct CanvasPositions(pub BTreeMap<NodeId, Vec2>);

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

/// **The** scope marker. Stamped on every Bevy entity that visualizes
/// a sim entity (node body, edge line, traveling packet, edge label,
/// probe readout, …) and carries the `NodeId` whose scope it inherits.
///
/// For a node, that's the node's own id. For an edge, packet, or
/// edge-attached label, it's the edge's *canonical owner* —
/// [`canonical_edge_owner`] picks the inner endpoint when both
/// endpoints belong to the same compound (so internal edges hide with
/// their compound) and the outer endpoint otherwise (so boundary edges
/// stay visible when their outside end is in scope).
///
/// One marker, one visibility system, one rule: every visual subsystem
/// stamps `Scoped` at spawn time and the central [`sync_scoped_visibility`]
/// reconciles them with [`CurrentScope`] + [`CompoundMembership`].
/// Adding a new visual subsystem? It's one `.insert(Scoped(owner))`
/// at its spawn site — nothing else.
#[derive(Component, Clone, Copy, Debug)]
pub struct Scoped(pub NodeId);

/// Pick the canonical owning node for an edge. The choice determines
/// which scope the edge (and any visuals attached to it: packets,
/// labels, …) lives in.
///
/// Rule: if both endpoints are inside the *same* compound, that's an
/// internal edge → owner is either endpoint (same scope answer). If
/// exactly one endpoint is inside a compound, that's a boundary edge
/// → owner is the *outer* endpoint, so the edge remains visible at
/// the top level. If neither endpoint is inside anything, owner is
/// either.
///
/// The function never returns the compound body's own id — that would
/// give edges a separate scope from the nodes they connect, which is
/// the bug we're explicitly avoiding.
pub fn canonical_edge_owner(from: NodeId, to: NodeId, m: &CompoundMembership) -> NodeId {
    match (m.parent_of(from), m.parent_of(to)) {
        (Some(a), Some(b)) if a == b => from,   // internal — both inside same compound
        (Some(_), None) => to,                  // boundary — outer wins
        (None, Some(_)) => from,                // boundary — outer wins
        (Some(_), Some(_)) => from,             // cross-compound (shouldn't happen, pick one)
        (None, None) => from,                   // both top-level
    }
}

/// Marker stamped on the compound's own body entity. Mirrors the
/// `flow::NodeId` so the double-click drill-in handler can tell the
/// difference between "user clicked a compound" (zoom into it) and
/// "user clicked a leaf" (selection / inspector). The body **also**
/// carries a `Scoped`, set to its parent compound (or itself if at top
/// level) so the unified visibility system handles it like everything
/// else; see the special-case branch in [`sync_scoped_visibility`] for
/// the "hide self when drilled into self" rule.
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

/// **The** canvas population reconciler. Diffs `EntityMaps` against
/// what [`is_in_scope_node`] / [`is_in_scope_edge`] say should exist
/// right now, then despawns out-of-scope entities and spawns
/// missing in-scope ones. Runs whenever scope or membership changes
/// (the only two inputs to the predicate).
///
/// This is what implements "inner cells don't exist on the canvas
/// at the top level" — they're not Hidden, they're not present.
/// Selection, drag, packet rendering, edge drawing all naturally
/// stop being concerns because there's no entity for them to find.
pub fn sync_canvas_population(world: &mut World) {
    let scope_changed = world.is_resource_changed::<CurrentScope>();
    let membership_changed = world.is_resource_changed::<CompoundMembership>();
    if !scope_changed && !membership_changed { return; }

    let scope = world.resource::<CurrentScope>().clone();
    let membership = world.resource::<CompoundMembership>().clone();

    // ── Snapshot what the sim says exists.
    let (sim_nodes, sim_edges): (
        Vec<(NodeId, String, Option<String>, Option<usize>, bool)>,
        Vec<(flow::EdgeId, NodeId, NodeId)>,
    ) = {
        let mut driver = world.resource_mut::<crate::sim_driver::SimDriverRes>();
        driver.0.with_sim_mut(|sim| {
            let nodes: Vec<_> = sim.nodes.iter().map(|(id, n)| {
                // `class_name()` follows the leaf TemplateId — None for
                // compound shims. Fall back to `compound_class_of` so
                // downstream visual selection can recognize a shim as
                // a built-in gadget composite (e.g. "GeneratorComposite")
                // and pick the gadget body shape on respawn.
                let class = sim
                    .class_name(*id)
                    .map(|s| s.to_owned())
                    .or_else(|| sim.compound_class_of.get(id).cloned());
                let color = match n.slots.get("color") {
                    Some(flow::Value::Int(i)) => Some(*i as usize),
                    _ => None,
                };
                (*id, n.name.clone(), class, color, n.is_compound())
            }).collect();
            let edges: Vec<_> = sim.edges.iter().map(|(eid, e)| (*eid, e.from, e.to)).collect();
            (nodes, edges)
        })
    };

    // ── Compute target set for this scope.
    let mut target_nodes: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for (nid, _, _, _, _) in &sim_nodes {
        if is_in_scope_node(*nid, &scope, &membership) {
            target_nodes.insert(*nid);
        }
    }
    let mut target_edges: std::collections::HashSet<flow::EdgeId> = std::collections::HashSet::new();
    for (eid, from, to) in &sim_edges {
        if is_in_scope_edge(*from, *to, &scope, &membership) {
            target_edges.insert(*eid);
        }
    }

    // ── Diff against EntityMaps.
    let (current_nodes, current_edges): (Vec<NodeId>, Vec<flow::EdgeId>) = {
        let maps = world.resource::<crate::bridge::EntityMaps>();
        (
            maps.node_to_entity.keys().copied().collect(),
            maps.edge_to_entity.keys().copied().collect(),
        )
    };

    // Node despawns: in maps but not in target.
    let mut node_despawns: Vec<(NodeId, bevy::prelude::Entity)> = Vec::new();
    {
        let maps = world.resource::<crate::bridge::EntityMaps>();
        for nid in &current_nodes {
            if !target_nodes.contains(nid) {
                if let Some(e) = maps.node_to_entity.get(nid).copied() {
                    node_despawns.push((*nid, e));
                }
            }
        }
    }
    // Edge despawns: in maps but not in target.
    let mut edge_despawns: Vec<(flow::EdgeId, bevy::prelude::Entity)> = Vec::new();
    {
        let maps = world.resource::<crate::bridge::EntityMaps>();
        for eid in &current_edges {
            if !target_edges.contains(eid) {
                if let Some(e) = maps.edge_to_entity.get(eid).copied() {
                    edge_despawns.push((*eid, e));
                }
            }
        }
    }

    // ── Apply despawns through Commands so the relationship system
    // (ChildOf / Children) cleans up bidirectionally without
    // warnings. Direct world.despawn leaves dangling parent→child
    // references in the same-frame batch.
    // Snapshot the position of every node we're about to despawn, so a
    // later respawn (drill back out) restores it instead of falling back
    // to a default grid slot. Captures dragged positions too.
    {
        let mut positions = std::mem::take(&mut world.resource_mut::<CanvasPositions>().0);
        for (nid, e) in &node_despawns {
            if let Some(tf) = world.entity(*e).get::<Transform>() {
                positions.insert(*nid, tf.translation.truncate());
            }
        }
        world.resource_mut::<CanvasPositions>().0 = positions;
    }

    let needs_despawn = !node_despawns.is_empty() || !edge_despawns.is_empty();
    if needs_despawn {
        let mut state: bevy::ecs::system::SystemState<(
            Commands,
            ResMut<crate::bridge::EntityMaps>,
        )> = bevy::ecs::system::SystemState::new(world);
        let (mut commands, mut maps) = state.get_mut(world);
        for (nid, e) in &node_despawns {
            commands.entity(*e).despawn();
            maps.node_to_entity.remove(nid);
            maps.entity_to_node.remove(e);
        }
        for (eid, e) in &edge_despawns {
            commands.entity(*e).despawn();
            maps.edge_to_entity.remove(eid);
            maps.entity_to_edge.remove(e);
        }
        state.apply(world);
    }

    // ── Compute spawns: in target but not in maps.
    let to_spawn_nodes: Vec<&(NodeId, String, Option<String>, Option<usize>, bool)> = {
        let maps = world.resource::<crate::bridge::EntityMaps>();
        sim_nodes
            .iter()
            .filter(|(nid, _, _, _, _)| {
                target_nodes.contains(nid) && !maps.node_to_entity.contains_key(nid)
            })
            .collect()
    };
    let to_spawn_edges: Vec<&(flow::EdgeId, NodeId, NodeId)> = {
        let maps = world.resource::<crate::bridge::EntityMaps>();
        sim_edges
            .iter()
            .filter(|(eid, _, _)| {
                target_edges.contains(eid) && !maps.edge_to_entity.contains_key(eid)
            })
            .collect()
    };

    if to_spawn_nodes.is_empty() && to_spawn_edges.is_empty() { return; }

    // ── Apply spawns.
    let visual = world.resource::<crate::canvas::CurrentCanvasVisual>().0.clone();
    // Names → NodeId for compound visuals' member-pattern lookup.
    let node_data_by_name: std::collections::BTreeMap<String, NodeId> = sim_nodes
        .iter()
        .map(|(id, name, _, _, _)| (name.clone(), *id))
        .collect();
    // Live param map (defaults ⊕ overrides) for `Grid` dimensions.
    let compound_param_values = {
        let registry = world.resource::<CompoundParamRegistry>().clone();
        let overrides = world.resource::<CompoundOverrides>().clone();
        crate::canvas::merged_compound_params(&registry, &overrides)
    };
    // Restore previously-captured positions (set when these nodes were
    // despawned on drill-in) so the layout survives a drill round-trip.
    let saved_positions = world.resource::<CanvasPositions>().0.clone();

    let mut state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<crate::nodes::NodeAssetCache>,
        ResMut<Assets<Mesh>>,
        ResMut<Assets<ColorMaterial>>,
        ResMut<crate::bridge::EntityMaps>,
        ResMut<crate::tool::NodeColors>,
        Res<crate::theme::Theme>,
        Res<crate::bitmap_label::AtlasMetrics>,
    )> = bevy::ecs::system::SystemState::new(world);
    {
        let (mut commands, mut cache, mut meshes, mut materials, mut maps, mut node_colors, theme, metrics) =
            state.get_mut(world);
        for (i, (nid, name, class_name, color_slot_raw, is_compound)) in to_spawn_nodes.iter().enumerate() {
            crate::canvas::spawn_one_canvas_node_at(
                &mut commands, &mut cache, &mut meshes, &mut materials,
                &mut maps, &mut node_colors, &theme, &metrics,
                &visual, &node_data_by_name, &compound_param_values,
                saved_positions.get(nid).copied(),
                *nid, name, class_name.as_deref(), *color_slot_raw, *is_compound, i,
            );
        }
        for (eid, from, to) in &to_spawn_edges {
            crate::canvas::spawn_one_canvas_edge(
                &mut commands, &mut maps, &membership, *eid, *from, *to,
            );
        }
    }
    state.apply(world);
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
    // Spawn a minimal body entity (Mesh2d/material set to a temporary
    // default — the real geometry comes from the apply step below).
    // This pattern means the rebuild path can call the same
    // `apply_compound_visual` helper to refresh body geometry +
    // chrome + content without re-creating the Entity, so selection
    // / drag state persists.
    let placeholder_shape = BodyShape::Rect(Vec2::splat(1.0));
    let placeholder_mesh = cache.body_mesh(&placeholder_shape, meshes);
    let paper_mat = cache.material(theme.paper, materials);
    let entity = commands
        .spawn((
            Mesh2d(placeholder_mesh),
            MeshMaterial2d(paper_mat),
            Transform::from_translation(pos.extend(1.0)),
            FlowNodeRef(flow_id),
            CompoundBodyMarker(flow_id),
            placeholder_shape,
        ))
        .id();
    maps.node_to_entity.insert(flow_id, entity);
    maps.entity_to_node.insert(entity, flow_id);

    apply_compound_visual(commands, cache, meshes, materials, theme, metrics, entity, &visual);
    let _ = full_name;
    entity
}

/// **The** scope predicate. A sim node should have a Bevy canvas
/// entity right now iff this returns `true`. There are no "hidden
/// ghosts" — entities outside the current scope don't exist on the
/// canvas at all. Selection, dragging, hit-testing, packet rendering
/// all stop being concerns the moment the entity isn't there.
///
/// Rule: a node renders iff its enclosing compound matches the
/// current scope. Top-level nodes (no enclosing compound) render at
/// the top scope (`CurrentScope == None`). Inner nodes of compound
/// `C` render at scope `Some(C)`. Compound bodies follow the same
/// rule — at top scope they render as "an outer face you can click
/// to drill into"; when drilled into themselves they don't exist
/// (you're inside, looking at the interior).
pub fn is_in_scope_node(nid: NodeId, scope: &CurrentScope, membership: &CompoundMembership) -> bool {
    membership.parent_of(nid) == scope.0
}

/// Edge variant of [`is_in_scope_node`]: an edge has a Bevy entity
/// iff both endpoints are in scope. Boundary edges (one inside, one
/// outside a compound) don't render at either scope today — visually
/// rerouting them to attach to the compound's body is a documented
/// follow-up.
pub fn is_in_scope_edge(
    from: NodeId,
    to: NodeId,
    scope: &CurrentScope,
    membership: &CompoundMembership,
) -> bool {
    is_in_scope_node(from, scope, membership) && is_in_scope_node(to, scope, membership)
}

/// Pad a rectangular body to draw the surrounding ink border / shadow
/// at the right size. Mirrors `nodes::padded_shape`'s rect arm; kept
/// local so this module doesn't depend on a crate-private helper.
fn pad_rect(shape: &BodyShape, pad: f32) -> BodyShape {
    // Mirror of `nodes::padded_shape`; kept local. Compounds only
    // expose Rect/Circle as outward visuals today, so the extra
    // primitive variants are passed through unchanged (they shouldn't
    // appear here in practice but a wildcard keeps the match total).
    match shape {
        BodyShape::Rect(v) => BodyShape::Rect(Vec2::new(v.x + pad * 2.0, v.y + pad * 2.0)),
        BodyShape::Circle(r) => BodyShape::Circle(r + pad),
        other => *other,
    }
}

/// **The** entry point for setting / refreshing what a compound body
/// looks like. Wipes the body's existing children, recomputes its
/// frame size + shape from the [`CompoundVisual`], and spawns fresh
/// chrome (border, shadow, optional label) plus per-cell content.
///
/// Called from initial spawn (right after the body Entity is created)
/// and from the surgical-rebuild handler (so the body grows /
/// shrinks when its `width` / `height` params change). One source of
/// truth means the body's appearance is always derived from the
/// current visual, never partially-stale.
pub fn apply_compound_visual(
    commands: &mut Commands,
    cache: &mut NodeAssetCache,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    theme: &Theme,
    metrics: &AtlasMetrics,
    body_entity: Entity,
    visual: &CompoundVisual,
) {
    // Wipe whatever children were there last (border, shadow, label,
    // grid cells). Cheaper than diffing — the body's children are a
    // few-tens of entities for grids in the dozens; for thousand-cell
    // grids we can revisit when it bites.
    commands.entity(body_entity).despawn_related::<Children>();

    match visual {
        CompoundVisual::Grid { columns, rows, cell_size, gap, members, paint } => {
            let pad = 12.0;
            let pitch = cell_size + gap;
            let inner_w = (*columns as f32) * pitch - gap;
            let inner_h = (*rows as f32) * pitch - gap;
            let frame_size = Vec2::new(inner_w + 2.0 * pad, inner_h + 2.0 * pad);
            let frame_shape = BodyShape::Rect(frame_size);
            let body_handle = cache.body_mesh(&frame_shape, meshes);
            let border_handle = cache.body_mesh(&pad_rect(&frame_shape, 3.0), meshes);
            let shadow_handle = cache.body_mesh(&pad_rect(&frame_shape, 3.0), meshes);
            let paper_mat = cache.material(theme.paper, materials);
            let ink_mat = cache.material(theme.ink, materials);
            let ink_soft_mat = cache.material(theme.ink_soft, materials);

            // Refresh the body's own mesh + shape (not just children)
            // so hit-testing, shadows, and any `BodyShape`-driven layout
            // pick up the new dimensions on rebuild.
            commands
                .entity(body_entity)
                .insert(Mesh2d(body_handle))
                .insert(MeshMaterial2d(paper_mat))
                .insert(frame_shape);

            let cell_shape = BodyShape::Rect(Vec2::new(*cell_size, *cell_size));
            let cell_mesh = cache.body_mesh(&cell_shape, meshes);
            let origin_x = -inner_w * 0.5 + cell_size * 0.5;
            let origin_y =  inner_h * 0.5 - cell_size * 0.5;

            commands.entity(body_entity).with_children(|parent| {
                parent.spawn((
                    Mesh2d(border_handle),
                    MeshMaterial2d(ink_mat),
                    Transform::from_xyz(0.0, 0.0, -0.1),
                ));
                parent.spawn((
                    Mesh2d(shadow_handle),
                    MeshMaterial2d(ink_soft_mat),
                    Transform::from_xyz(6.0, -6.0, -0.2),
                ));
                for row in 0..*rows {
                    for col in 0..*columns {
                        let idx = (row * columns + col) as usize;
                        let source = members.get(idx).copied().flatten();
                        let cx = origin_x + (col as f32) * pitch;
                        let cy = origin_y - (row as f32) * pitch;
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
        }
        CompoundVisual::LabeledBox { size, label } => {
            let shape = BodyShape::Rect(*size);
            let body_handle = cache.body_mesh(&shape, meshes);
            let border_handle = cache.body_mesh(&pad_rect(&shape, 3.0), meshes);
            let shadow_handle = cache.body_mesh(&pad_rect(&shape, 3.0), meshes);
            let paper_mat = cache.material(theme.paper, materials);
            let ink_mat = cache.material(theme.ink, materials);
            let ink_soft_mat = cache.material(theme.ink_soft, materials);

            commands
                .entity(body_entity)
                .insert(Mesh2d(body_handle))
                .insert(MeshMaterial2d(paper_mat))
                .insert(shape);

            let label_text = label.to_uppercase();
            let label_capacity = label_text.chars().count().max(8);
            let label_color = theme.ink;
            let label_cell_w = metrics.cell_w;
            let label_cell_h = metrics.cell_h;
            let metrics_clone = metrics.clone();

            commands.entity(body_entity).with_children(|parent| {
                parent.spawn((
                    Mesh2d(border_handle),
                    MeshMaterial2d(ink_mat),
                    Transform::from_xyz(0.0, 0.0, -0.1),
                ));
                parent.spawn((
                    Mesh2d(shadow_handle),
                    MeshMaterial2d(ink_soft_mat),
                    Transform::from_xyz(6.0, -6.0, -0.2),
                ));
                let label_entity = parent.spawn((
                    BitmapLabel {
                        text: label_text,
                        color: label_color,
                        align: TextAlign::Center,
                        capacity: label_capacity,
                        cell_w: label_cell_w,
                        cell_h: label_cell_h,
                    },
                    Transform::from_xyz(0.0, 0.0, 0.1),
                    Visibility::Inherited,
                )).id();
                parent.commands().entity(label_entity).with_children(|cp| {
                    spawn_label_chars(cp, &metrics_clone, label_capacity, label_color, label_cell_w, label_cell_h);
                });
            });
        }
    }
}

/// Back-compat alias — kept so callers that only want grid
/// repopulation don't need to think about chrome refresh. New code
/// should call [`apply_compound_visual`] directly.
pub fn populate_grid_cells_under(
    commands: &mut Commands,
    cache: &mut NodeAssetCache,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    theme: &Theme,
    metrics: &AtlasMetrics,
    body_entity: Entity,
    visual: &CompoundVisual,
) {
    apply_compound_visual(commands, cache, meshes, materials, theme, metrics, body_entity, visual);
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

/// Surgical-rebuild handler. Reads `RebuildCompound` events and, for
/// each one, demolishes the named compound's current interior in both
/// the sim and the Bevy world, re-expands the compound from the
/// cached AST with the current overrides, lowers the new interior
/// into the existing sim, and spawns Bevy entities for the new sim
/// nodes/edges.
///
/// **Surgical** — top-level entities, sim time, scenarios, viewport,
/// snapshot, and any *other* compound's interior are untouched. The
/// compound body's Bevy entity itself stays, so selection on it
/// survives a rebuild.
///
/// Implemented as an exclusive system because it needs `&mut World`
/// (despawn, query, then mutate resources, then despawn again) and
/// the access pattern doesn't compose into a single `SystemParam`.
fn handle_rebuild_compound(world: &mut World) {
    // Drain events first; we mutate them out so a re-fire doesn't
    // double-rebuild on the next frame.
    let pending: Vec<String> = {
        let mut messages = world.resource_mut::<bevy::ecs::message::Messages<RebuildCompound>>();
        let collected: Vec<String> = messages.drain().map(|m| m.0).collect();
        collected
    };
    if pending.is_empty() { return; }

    for compound_name in pending {
        if let Err(e) = rebuild_one_compound(world, &compound_name) {
            bevy::log::warn!("rebuild compound `{}`: {}", compound_name, e);
        }
    }
}

fn rebuild_one_compound(world: &mut World, compound_name: &str) -> Result<(), String> {
    use crate::bridge::EntityMaps;
    use crate::sim_driver::SimDriverRes;

    // ── Step 1: re-expand the subtree with current overrides.
    //
    // `expand_compound_subtree` returns the interior + the compound's
    // own port-shim. We drop the top-level port-shim so `lower_into`
    // doesn't create a duplicate body NodeId — the original body's
    // identity stays stable across rebuilds.
    let new_items = {
        let ast = world.resource::<CachedCanvasAst>().0.clone();
        let overrides = world.resource::<CompoundOverrides>()
            .by_compound
            .get(compound_name)
            .cloned()
            .unwrap_or_default();
        let raw = flow::dsl::expand::expand_compound_subtree(&ast, compound_name, &overrides)?;
        raw.into_iter()
            .filter(|item| match item {
                flow::dsl::ast::Item::Compound(c) => c.name != compound_name,
                _ => true,
            })
            .collect::<Vec<_>>()
    };

    // ── Step 2: sim demolition + lowering, in one driver borrow so
    // intermediate state is never visible.
    let body_id: Option<flow::NodeId>;
    {
        let mut driver = world.resource_mut::<SimDriverRes>();
        let compound_name_owned = compound_name.to_string();
        let r = driver.0.with_sim_mut(move |sim| {
            let _removed = sim.despawn_compound_interior(&compound_name_owned, false);
            let bid = sim.nodes.iter()
                .find(|(_, n)| n.name == compound_name_owned)
                .map(|(id, _)| *id);
            let synth = flow::dsl::ast::File { items: new_items };
            flow::dsl::lower_into(sim, &synth)
                .map(|_| bid)
                .map_err(|e| format!("lower_into: {}", e))
        });
        body_id = r?;
    }

    // ── Step 3: recompute membership and assign it. The assignment
    // triggers `is_resource_changed::<CompoundMembership>`, which is
    // what wakes `sync_canvas_population` next frame to reconcile
    // Bevy entities against the new sim state — despawn old cells
    // (whose NodeIds no longer exist), spawn new ones if the
    // compound is currently in scope. We don't despawn or spawn here;
    // that would duplicate the population sync's work.
    let new_membership: CompoundMembership = {
        let mut driver = world.resource_mut::<SimDriverRes>();
        driver.0.with_sim_mut(|sim| compute_membership(sim))
    };
    *world.resource_mut::<CompoundMembership>() = new_membership;

    // ── Step 4: refresh the compound body's grid view (if its Bevy
    // entity is currently alive — i.e. the body is in scope). The
    // sim's old cells are gone, so the body's existing grid mini-cells
    // hold stale `GridCellPaintRef.source` references. `apply_compound_visual`
    // wipes the body's children and re-spawns them pointing at the
    // new NodeIds.
    let body_entity = body_id.and_then(|id| {
        world.resource::<EntityMaps>().node_to_entity.get(&id).copied()
    });
    let Some(body_entity) = body_entity else { return Ok(()); };

    let visual = world.resource::<crate::canvas::CurrentCanvasVisual>().0.clone();
    let node_data_by_name: BTreeMap<String, flow::NodeId> = {
        let mut driver = world.resource_mut::<SimDriverRes>();
        driver.0.with_sim_mut(|sim| {
            sim.nodes.iter().map(|(id, n)| (n.name.clone(), *id)).collect()
        })
    };
    let compound_param_values = {
        let registry = world.resource::<CompoundParamRegistry>().clone();
        let overrides = world.resource::<CompoundOverrides>().clone();
        crate::canvas::merged_compound_params(&registry, &overrides)
    };

    let body_name = node_data_by_name
        .iter()
        .find(|(_, id)| **id == body_id.unwrap())
        .map(|(n, _)| n.clone());
    let Some(body_name) = body_name else { return Ok(()); };

    let mut state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<crate::nodes::NodeAssetCache>,
        ResMut<Assets<Mesh>>,
        ResMut<Assets<ColorMaterial>>,
        Res<crate::theme::Theme>,
        Res<crate::bitmap_label::AtlasMetrics>,
    )> = bevy::ecs::system::SystemState::new(world);
    {
        let (mut commands, mut cache, mut meshes, mut materials, theme, metrics) =
            state.get_mut(world);
        let empty = std::collections::BTreeMap::new();
        let params = compound_param_values.get(&body_name).unwrap_or(&empty);
        let cv = crate::canvas::build_compound_visual(
            &body_name,
            visual.compounds.get(&body_name),
            &node_data_by_name,
            &theme,
            params,
        );
        apply_compound_visual(
            &mut commands, &mut cache, &mut meshes, &mut materials,
            &theme, &metrics, body_entity, &cv,
        );
    }
    state.apply(world);

    Ok(())
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
