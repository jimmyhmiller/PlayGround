//! Canvas file format: a directory (`foo.whiteboard/`) bundling the
//! DSL source describing a sim, visual state for its nodes, and optional
//! frozen state snapshots.
//!
//! Directory layout:
//!
//! ```text
//! my_canvas.whiteboard/
//! ├── manifest.json        # metadata + which scenario/snapshot to boot
//! ├── main.flow            # params + edges + compounds + scenarios
//! ├── components/          # custom node classes defined by this canvas
//! │   ├── custom_worker.flow
//! │   └── custom_router.flow
//! ├── visual.json          # positions, colors, viewport (optional)
//! └── snapshots/           # optional frozen Sim states (future task)
//! ```
//!
//! Stock gadgets (generator, client, worker, etc.) are auto-registered
//! by [`load_canvas`] before any user-supplied components, so a canvas
//! never needs to redeclare them.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use flow::{Sim, SimSnapshot, Value};

use crate::bitmap_label::AtlasMetrics;
use crate::bridge::{EntityMaps, FlowEdgeRef};
use crate::gadgets::{self, Kind};
use crate::nodes::{BinarySlotPaint, BodyShape, NodeAssetCache, NodeCounter, spawn_node_entity};
use crate::sim_driver::SimDriverRes;
use crate::theme::Theme;
use crate::tool::NodeColors;

// ────────────────────────────────────────────────────────────────────
// Manifest
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version. Bumped when an incompatible change to the format
    /// ships so loaders can refuse or migrate old files explicitly.
    pub format_version: u32,
    #[serde(default)]
    pub canvas: CanvasMeta,
    /// Load-time controls: component order, etc. Optional — defaults fit
    /// a single-file canvas with no custom components.
    #[serde(default)]
    pub load: LoadOpts,
    /// What to boot the sim with: either a named scenario or a named
    /// snapshot. Omit for a canvas that loads into a dormant state.
    #[serde(default)]
    pub default: DefaultBoot,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CanvasMeta {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub created: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadOpts {
    /// Explicit load order for `components/*.flow`. Each entry is the
    /// component file's stem (without the `.flow` extension). Files not
    /// listed here are appended in alphabetical order.
    #[serde(default)]
    pub components: Vec<String>,
}

/// What to do after the canvas is fully loaded. A canvas with both
/// fields set is rejected — scenarios and snapshots are mutually
/// exclusive starting points.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefaultBoot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
}

// ────────────────────────────────────────────────────────────────────
// Visual state
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Visual {
    #[serde(default = "default_visual_version")]
    pub format_version: u32,
    /// Per-class appearance overrides. The map is keyed by class name
    /// (as declared in the DSL) and lets a canvas describe shape/size
    /// and dynamic paint behavior for custom classes that flow-bevy's
    /// built-in `Kind` enum doesn't know about. Built-in palette
    /// classes (Worker, Client, Queue, etc.) ignore the entry — their
    /// shape comes from `Kind`.
    #[serde(default)]
    pub classes: BTreeMap<String, ClassVisual>,
    /// Per-compound visual descriptors. Keyed by the compound's
    /// fully-qualified name (e.g. `"Life"` or `"Outer::Inner"`).
    /// Missing entries get a [`crate::compound::CompoundVisual::LabeledBox`] fallback.
    #[serde(default)]
    pub compounds: BTreeMap<String, CompoundVisualSpec>,
    /// Keyed by node name (as declared in DSL). Missing nodes fall back
    /// to default layout from the UI layer.
    #[serde(default)]
    pub nodes: BTreeMap<String, VisualNode>,
    #[serde(default)]
    pub viewport: Viewport,
    /// Boot the canvas with all edges and packets hidden. Useful for
    /// stress-test canvases (Game of Life, large grids) where the raw
    /// visual layer is unreadable at startup. The user can toggle the
    /// state at runtime with the `H` key either way.
    #[serde(default)]
    pub hide_all: bool,
}

fn default_visual_version() -> u32 { 1 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisualNode {
    /// `[x, y]` in world units. Stored as array so the JSON is compact.
    #[serde(default)]
    pub pos: [f32; 2],
    /// Optional hex string like `"#e86a4c"`. `None` = let theme pick
    /// based on node kind / data-palette slot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// Per-class appearance for canvas-defined node classes. All fields are
/// optional — a class spec can declare just a shape, just a paint
/// behavior, or both. Missing fields fall back to flow-bevy's built-in
/// defaults for the nearest matching `Kind` (`Worker` for unknown
/// classes).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassVisual {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<ShapeSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paint: Option<PaintSpec>,
}

/// Body geometry. `square` and `circle` carry a `size` in world units —
/// a square's edge length, a circle's diameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShapeSpec {
    Square { size: f32 },
    Circle { size: f32 },
}

/// Dynamic body fill driven by sim state.
///
/// `binary_slot` flips between two colors based on whether the named
/// slot is non-zero (`Int != 0` or `Bool == true`). Colors are CSS-style
/// hex strings (`"#000000"`); invalid strings fall back to theme
/// `ink`/`paper`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PaintSpec {
    BinarySlot { slot: String, on: String, off: String },
}

/// How a compound renders its outward face. Mirrors
/// [`crate::compound::CompoundVisual`] but lives in the on-disk visual
/// schema; the loader converts one into the other.
///
/// **Why a separate enum**: `CompoundVisual` is a runtime-side type
/// (carries `bevy::Vec2`, etc.) while `CompoundVisualSpec` is the
/// JSON-serializable authoring shape. Keeping them separate avoids
/// forcing serde derives onto every Bevy primitive the renderer might
/// later need.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompoundVisualSpec {
    /// A simple rectangular body with the compound name centered.
    /// (Same as the no-spec fallback, but lets a canvas pin a specific
    /// size / label override.)
    LabeledBox {
        #[serde(default = "default_compound_box_w")]
        width: f32,
        #[serde(default = "default_compound_box_h")]
        height: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// Render the compound as an explicit grid of small bodies, one
    /// per inner member. The author supplies a `member_pattern`
    /// containing `{x}` / `{y}` placeholders that are filled in for
    /// each `(col, row)` and looked up in the sim's node index. Each
    /// cell is painted by reading a slot from its source node — the
    /// same `binary_slot` semantics as the existing class paint.
    ///
    /// This is the minimum useful "compound visual driven by its
    /// parts" — enough to make Game of Life look like Game of Life on
    /// the canvas without exposing the inner cells as separate bodies.
    /// Future variants will let visuals do richer things (per-slot
    /// gradients, vector glyphs, scripted layouts); this one was the
    /// concrete need on the table today.
    Grid {
        columns: u32,
        rows: u32,
        cell_size: f32,
        #[serde(default)]
        gap: f32,
        member_pattern: String,
        paint: PaintSpec,
    },
}

fn default_compound_box_w() -> f32 { 220.0 }
fn default_compound_box_h() -> f32 { 140.0 }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Viewport {
    #[serde(default)]
    pub pos: [f32; 2],
    #[serde(default = "default_zoom")]
    pub zoom: f32,
}

fn default_zoom() -> f32 { 1.0 }

// ────────────────────────────────────────────────────────────────────
// Loader
// ────────────────────────────────────────────────────────────────────

/// Fully-loaded canvas ready to hand to the bridge. The sim already has
/// stock gadgets + custom components registered, main.flow wiring
/// applied, and the default scenario (if any) scheduled.
pub struct LoadedCanvas {
    pub manifest: Manifest,
    pub sim: Sim,
    pub visual: Visual,
    /// Absolute path the canvas was loaded from — kept so callers can
    /// save back to the same location without the user re-picking a path.
    pub path: PathBuf,
    /// Authoring-time compound metadata (param names + evaluated
    /// defaults) extracted from `main.flow` before expansion. Surfaced
    /// to the inspector so the user can see the construction params
    /// of any compound on the canvas. Read-only at the moment;
    /// editing + rebuild is the next step.
    pub compound_params: Vec<flow::dsl::expand::CompoundParamSummary>,
}

/// Read a `foo.whiteboard/` directory and hydrate it into a running sim.
///
/// `seed` is the RNG seed for the sim. Two loads with the same seed and
/// canvas produce bitwise-identical runs, so canvas + seed is a
/// reproducible experiment specification.
pub fn load_canvas(path: impl AsRef<Path>, seed: u64) -> Result<LoadedCanvas, String> {
    let path = path.as_ref().to_path_buf();
    if !path.is_dir() {
        return Err(format!(
            "canvas path `{}` is not a directory",
            path.display()
        ));
    }

    let manifest = load_manifest(&path)?;
    if manifest.format_version != 1 {
        return Err(format!(
            "unsupported manifest format_version {} (this build supports 1)",
            manifest.format_version
        ));
    }
    if manifest.default.scenario.is_some() && manifest.default.snapshot.is_some() {
        return Err("manifest.default: set exactly one of `scenario` or `snapshot`, not both".into());
    }

    let mut sim = Sim::new(seed);
    gadgets::install_default_params(&mut sim);

    // Stock gadget classes first — user-supplied components and main.flow
    // can reference them by name.
    flow::dsl::register_classes(&mut sim, gadgets::GADGETS_DSL)
        .map_err(|e| format!("registering stock gadgets: {}", e))?;

    // User-supplied component classes (custom node types this canvas
    // adds). Each component file declares one or more `node X { }`
    // blocks that get *registered as classes only* — no instances are
    // created. main.flow then names instances explicitly via
    // `node Cell_0_0 : LifeCell { ... }`. Using `register_classes`
    // (rather than the full `lower_into`) avoids spawning a stray
    // singleton `LifeCell` instance that has no position in
    // visual.json and lands at a default fallback.
    let comp_order = resolve_component_order(&path, &manifest.load.components)?;
    for stem in &comp_order {
        let src = fs::read_to_string(path.join("components").join(format!("{}.flow", stem)))
            .map_err(|e| format!("reading component `{}`: {}", stem, e))?;
        flow::dsl::register_classes(&mut sim, &src)
            .map_err(|e| format!("registering component `{}`: {}", stem, e))?;
    }

    // main.flow: the actual wiring. Parsed, then lowered *into* the
    // existing sim so it sees stock + component classes.
    let main_path = path.join("main.flow");
    let main_src = fs::read_to_string(&main_path)
        .map_err(|e| format!("reading {}: {}", main_path.display(), e))?;
    let raw = flow::dsl::parse(&main_src)
        .map_err(|e| format!("parsing main.flow: {}", e))?;
    // Capture compound param metadata BEFORE expansion consumes it —
    // expansion folds defaults into the residual AST and the inspector
    // would otherwise have nowhere to learn `width: 5, height: 5`
    // from. Same `&raw` reused for expansion below so we don't pay
    // for a re-parse.
    let compound_params = flow::dsl::expand::collect_compound_params(&raw);
    let file = flow::dsl::expand::expand(&raw)
        .map_err(|e| format!("expanding main.flow: {}", e))?;
    let lowered = flow::dsl::lower_into(&mut sim, &file)
        .map_err(|e| format!("lowering main.flow: {}", e))?;

    // Default boot: snapshot replaces the freshly-built sim wholesale
    // (the snapshot carries full state including in-flight packets and
    // RNG, so there's no "merge" — the main.flow load was just needed
    // to validate structure). Otherwise run the named or main scenario.
    if let Some(name) = &manifest.default.snapshot {
        let snap = load_snapshot(&path, name)?;
        sim = snap.into_sim();
    } else if let Some(name) = &manifest.default.scenario {
        sim.run_scenario(name)
            .map_err(|e| format!("manifest.default.scenario: {}", e))?;
    } else if lowered.auto_run_main {
        sim.run_scenario("main").unwrap();
    }

    let visual = load_visual(&path)?;

    Ok(LoadedCanvas {
        manifest,
        sim,
        visual,
        path,
        compound_params,
    })
}

/// Startup system for [`crate::CanvasSeedPlugin`]: pulls the path out
/// of the `PendingCanvas` resource (injected by the plugin), loads the
/// canvas, replaces the default `FlowSim.sim`, and spawns a Bevy
/// entity for every sim node and edge. Visual positions come from
/// `visual.json` when the canvas provides one, otherwise falls back
/// to a crude grid layout so the graph is at least visible.
///
/// Errors are logged at `error!` level and leave the canvas empty —
/// the user can still load a built-in example from the palette.
pub(crate) fn seed_from_path(
    mut commands: Commands,
    mut cache: ResMut<NodeAssetCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut driver: ResMut<SimDriverRes>,
    mut maps: ResMut<EntityMaps>,
    mut counter: ResMut<NodeCounter>,
    mut node_colors: ResMut<NodeColors>,
    mut hide_all: ResMut<crate::edges::HideAll>,
    theme: Res<Theme>,
    metrics: Res<AtlasMetrics>,
    mut pending: ResMut<crate::PendingCanvas>,
) {
    let Some(path) = pending.0.take() else { return; };
    let canvas = match load_canvas(&path, 1) {
        Ok(c) => c,
        Err(e) => {
            bevy::log::error!("load_canvas({}): {}", path.display(), e);
            return;
        }
    };
    hide_all.0 = canvas.visual.hide_all;
    bevy::log::info!(
        "loaded canvas `{}` ({} nodes, {} edges)",
        canvas.manifest.canvas.name,
        canvas.sim.nodes.len(),
        canvas.sim.edges.len()
    );

    // Swap the whole sim onto the worker / direct driver and pull
    // back the data we need to spawn entities. One round-trip
    // through `with_sim_mut`; in worker mode the cost is one tick of
    // latency, which is invisible to the user.
    let new_sim = canvas.sim;
    let load_data = driver.0.with_sim_mut(move |sim| {
        *sim = new_sim;
        let membership = crate::compound::compute_membership(sim);
        let mut nodes: Vec<(flow::NodeId, String, Option<String>, Option<usize>, bool)> = Vec::new();
        for (nid, node) in sim.nodes.iter() {
            let class_name = sim.class_name(*nid).map(|s| s.to_owned());
            let color_slot = match node.slots.get("color") {
                Some(Value::Int(n)) => Some(*n as usize),
                _ => None,
            };
            nodes.push((*nid, node.name.clone(), class_name, color_slot, node.is_compound()));
        }
        let edges: Vec<(flow::EdgeId, flow::NodeId, flow::NodeId)> = sim.edges.iter()
            .map(|(eid, e)| (*eid, e.from, e.to))
            .collect();
        let node_count = sim.nodes.len() as u32;
        (nodes, edges, node_count, membership)
    });
    let (node_data, edge_data, node_count, membership) = load_data;
    counter.0 = node_count;
    commands.insert_resource(membership.clone());
    // Surface compound authoring params so the inspector can show
    // them when a compound is selected.
    let mut by_name = std::collections::BTreeMap::new();
    for summary in canvas.compound_params.iter() {
        by_name.insert(summary.name.clone(), summary.params.clone());
    }
    commands.insert_resource(crate::compound::CompoundParamRegistry { by_name });

    // Flat name → NodeId lookup so compound visuals can resolve member
    // patterns (`Life::Cell_{x}_{y}` → real NodeIds) at spawn time.
    let node_data_by_name: std::collections::BTreeMap<String, flow::NodeId> =
        node_data.iter().map(|(nid, name, _, _, _)| (name.clone(), *nid)).collect();

    // Spawn an entity for each node. Deterministic iteration order
    // (sim returns BTreeMap order via `nodes.iter()`) so fall-back
    // grid positions stay stable across loads of the same canvas.
    //
    // Compound nodes get a separate spawn path that produces a
    // labeled-box body instead of routing through the leaf gadget
    // logic — see `crate::compound::spawn_compound_body_entity`. Inner
    // nodes (anything `membership` reports as inside a compound) still
    // spawn here, but they're tagged with [`Inside`] so the visibility
    // system can hide them at the top level and reveal them when the
    // user drills into the compound.
    let visual = &canvas.visual;
    for (i, (nid, name, class_name, color_slot_raw, is_compound)) in node_data.into_iter().enumerate() {
        let pos = visual
            .nodes
            .get(&name)
            .map(|v| Vec2::new(v.pos[0], v.pos[1]))
            .unwrap_or_else(|| default_grid_position(i));

        if is_compound {
            let visual = build_compound_visual(&name, visual.compounds.get(&name), &node_data_by_name, &theme);
            let entity = crate::compound::spawn_compound_body_entity(
                &mut commands,
                &mut cache,
                &mut meshes,
                &mut materials,
                &mut maps,
                &theme,
                &metrics,
                nid,
                visual,
                name.clone(),
                pos,
            );
            // A compound that's nested inside another compound is
            // itself an "inside" entity — same visibility rules.
            if let Some(parent) = membership.parent_of(nid) {
                commands
                    .entity(entity)
                    .insert(crate::compound::Inside(parent))
                    .insert(Visibility::Hidden);
            }
            continue;
        }

        let kind = class_name
            .as_deref()
            .map(class_to_kind)
            .unwrap_or(Kind::Worker);
        let class_visual = class_name
            .as_deref()
            .and_then(|c| visual.classes.get(c));
        let shape_override = class_visual
            .and_then(|cv| cv.shape.as_ref())
            .map(shape_spec_to_body_shape);
        let self_painted = class_visual.and_then(|cv| cv.paint.as_ref()).is_some();
        let entity = spawn_node_entity(
            &mut commands,
            &mut cache,
            &mut meshes,
            &mut materials,
            &mut maps,
            &theme,
            &metrics,
            nid,
            kind,
            shape_override,
            self_painted,
            color_slot_raw.is_some(),
            name.clone(),
            pos,
        );
        if let Some(slot_raw) = color_slot_raw {
            let color_slot = slot_raw % theme.data.len();
            node_colors.0.insert(nid, theme.data[color_slot]);
        }
        if let Some(paint) = class_visual.and_then(|cv| cv.paint.as_ref()) {
            commands
                .entity(entity)
                .insert(paint_spec_to_component(paint, nid, &theme));
        }
        // Tag for compound-scope visibility. Hidden by default at top
        // level (CurrentScope = None); the visibility system flips it
        // visible when the user drills in.
        if let Some(parent) = membership.parent_of(nid) {
            commands
                .entity(entity)
                .insert(crate::compound::Inside(parent))
                .insert(Visibility::Hidden);
        }
    }

    // Spawn entities for every existing sim edge. No topology changes
    // here — the DSL already populated them.
    //
    // Edges whose endpoints are *both* inside the same compound are
    // tagged with [`EdgeInside`] so the visibility system hides them
    // at the top level. Boundary edges (one endpoint inside, one
    // outside) stay visible — visually rerouting them to attach to
    // the compound's body is a follow-up.
    for (eid, from, to) in edge_data {
        if from == to {
            // Self-loops are sim plumbing (tick / done); the UI draws
            // nothing for them and no entity is needed for routing.
            continue;
        }
        let ent = commands.spawn(FlowEdgeRef(eid)).id();
        maps.edge_to_entity.insert(eid, ent);
        maps.entity_to_edge.insert(ent, eid);
        let from_parent = membership.parent_of(from);
        let to_parent = membership.parent_of(to);
        if let (Some(fp), Some(tp)) = (from_parent, to_parent) {
            if fp == tp {
                commands
                    .entity(ent)
                    .insert(crate::compound::EdgeInside(fp))
                    .insert(Visibility::Hidden);
            }
        }
    }
}

/// Fall-back layout for a canvas that has no `visual.json`. A simple
/// 6-column grid — readable enough that the user can drag nodes into
/// place and save their own visual file.
fn default_grid_position(index: usize) -> Vec2 {
    const COLS: usize = 6;
    const COL_GAP: f32 = 180.0;
    const ROW_GAP: f32 = 140.0;
    let col = (index % COLS) as f32;
    let row = (index / COLS) as f32;
    Vec2::new(
        -((COLS as f32 - 1.0) * COL_GAP) / 2.0 + col * COL_GAP,
        200.0 - row * ROW_GAP,
    )
}

/// Convert a `visual.json` shape declaration into the runtime
/// [`BodyShape`] component used by the rendering systems.
fn shape_spec_to_body_shape(spec: &ShapeSpec) -> BodyShape {
    match spec {
        // Square `size` is edge length; circle `size` is diameter, so
        // halve to a radius for `BodyShape::Circle`.
        ShapeSpec::Square { size } => BodyShape::Rect(Vec2::splat(*size)),
        ShapeSpec::Circle { size } => BodyShape::Circle(*size * 0.5),
    }
}

/// Convert a `visual.json` paint declaration into a runtime component
/// to attach to a node. Hex colors that fail to parse fall back to
/// theme `ink` (on) / `paper` (off) so a typo can't make a node go
/// invisible.
fn paint_spec_to_component(spec: &PaintSpec, node: flow::NodeId, theme: &Theme) -> BinarySlotPaint {
    match spec {
        PaintSpec::BinarySlot { slot, on, off } => BinarySlotPaint {
            node,
            slot: slot.clone(),
            on: parse_hex_color(on).unwrap_or(theme.ink),
            off: parse_hex_color(off).unwrap_or(theme.paper),
        },
    }
}

/// Parse a `"#rrggbb"` (or `"rrggbb"`) hex string into a Bevy [`Color`].
/// Returns `None` if the string isn't a valid 6- or 8-digit hex.
fn parse_hex_color(s: &str) -> Option<Color> {
    let h = s.strip_prefix('#').unwrap_or(s);
    bevy::color::Srgba::hex(h).ok().map(Into::into)
}

/// Translate a `visual.json` compound spec into the runtime
/// [`crate::compound::CompoundVisual`] used by the renderer. Resolves
/// `member_pattern` against `node_index` so the spawn path doesn't
/// have to redo string substitution.
///
/// Missing entries fall back to a neutral [`crate::compound::CompoundVisual::default_for`]
/// box; bad hex colors fall back to theme `ink` / `paper` (matches
/// `paint_spec_to_component`'s policy — typos shouldn't make the
/// visual disappear).
fn build_compound_visual(
    name: &str,
    spec: Option<&CompoundVisualSpec>,
    node_index: &std::collections::BTreeMap<String, flow::NodeId>,
    theme: &Theme,
) -> crate::compound::CompoundVisual {
    let Some(spec) = spec else {
        return crate::compound::CompoundVisual::default_for(name);
    };
    match spec {
        CompoundVisualSpec::LabeledBox { width, height, label } => {
            let label = label.clone().unwrap_or_else(||
                name.rsplit("::").next().unwrap_or(name).to_string()
            );
            crate::compound::CompoundVisual::LabeledBox {
                size: Vec2::new(*width, *height),
                label,
            }
        }
        CompoundVisualSpec::Grid { columns, rows, cell_size, gap, member_pattern, paint } => {
            let mut members = Vec::with_capacity((columns * rows) as usize);
            for row in 0..*rows {
                for col in 0..*columns {
                    let resolved = crate::compound::resolve_member_pattern(member_pattern, col, row);
                    members.push(node_index.get(&resolved).copied());
                }
            }
            let (slot, on, off) = match paint {
                PaintSpec::BinarySlot { slot, on, off } => (
                    slot.clone(),
                    parse_hex_color(on).unwrap_or(theme.ink),
                    parse_hex_color(off).unwrap_or(theme.paper),
                ),
            };
            crate::compound::CompoundVisual::Grid {
                columns: *columns,
                rows: *rows,
                cell_size: *cell_size,
                gap: *gap,
                members,
                paint: crate::compound::GridCellPaint { slot, on, off },
            }
        }
    }
}

/// Map a DSL class name to a visual [`Kind`]. The seven stock gadget
/// class names match directly; anything else (custom components) falls
/// back to `Kind::Worker`, which has a neutral rectangular shape that
/// reads as "generic node" without implying any specific semantics.
///
/// Custom classes that want a different look should declare a
/// [`ClassVisual`] entry in `visual.json` rather than being hard-coded
/// here — flow-bevy's `Kind` enum is reserved for the built-in palette.
pub fn class_to_kind(class: &str) -> Kind {
    match class {
        "Generator" => Kind::Generator,
        "Client" => Kind::Client,
        "BackoffClient" => Kind::BackoffClient,
        "Worker" => Kind::Worker,
        "Router" => Kind::Router,
        "Queue" => Kind::Queue,
        "Sink" => Kind::Sink,
        _ => Kind::Worker,
    }
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let p = path.join("manifest.json");
    let src = fs::read_to_string(&p)
        .map_err(|e| format!("reading {}: {}", p.display(), e))?;
    serde_json::from_str(&src)
        .map_err(|e| format!("parsing {}: {}", p.display(), e))
}

fn load_snapshot(path: &Path, name: &str) -> Result<SimSnapshot, String> {
    let p = path.join("snapshots").join(format!("{}.json", name));
    let src = fs::read_to_string(&p)
        .map_err(|e| format!("reading snapshot {}: {}", p.display(), e))?;
    serde_json::from_str(&src)
        .map_err(|e| format!("parsing {}: {}", p.display(), e))
}

/// Save a snapshot of the sim's current state into
/// `<canvas>/snapshots/<name>.json`. The canvas directory must exist;
/// `snapshots/` is created on demand.
pub fn save_snapshot(
    canvas_path: &Path,
    name: &str,
    sim: &Sim,
    description: impl Into<String>,
) -> Result<(), String> {
    let mut snap = SimSnapshot::capture(sim, name);
    snap.description = description.into();
    let dir = canvas_path.join("snapshots");
    fs::create_dir_all(&dir)
        .map_err(|e| format!("creating {}: {}", dir.display(), e))?;
    let p = dir.join(format!("{}.json", name));
    let json = serde_json::to_string_pretty(&snap)
        .map_err(|e| format!("serializing snapshot: {}", e))?;
    fs::write(&p, json)
        .map_err(|e| format!("writing {}: {}", p.display(), e))?;
    Ok(())
}

fn load_visual(path: &Path) -> Result<Visual, String> {
    let p = path.join("visual.json");
    if !p.exists() {
        return Ok(Visual::default());
    }
    let src = fs::read_to_string(&p)
        .map_err(|e| format!("reading {}: {}", p.display(), e))?;
    serde_json::from_str(&src)
        .map_err(|e| format!("parsing {}: {}", p.display(), e))
}

/// Resolve the final component load order. `explicit` wins in the given
/// order; any `.flow` files in `components/` not listed are appended in
/// alphabetical order so authors don't have to list every file just to
/// add a new one.
fn resolve_component_order(
    path: &Path,
    explicit: &[String],
) -> Result<Vec<String>, String> {
    let components_dir = path.join("components");
    if !components_dir.exists() {
        if !explicit.is_empty() {
            return Err(format!(
                "manifest.load.components lists `{}` but `components/` dir doesn't exist",
                explicit.join(", ")
            ));
        }
        return Ok(Vec::new());
    }

    // Collect every .flow stem in the dir.
    let mut present: Vec<String> = Vec::new();
    for entry in fs::read_dir(&components_dir)
        .map_err(|e| format!("reading {}: {}", components_dir.display(), e))?
    {
        let entry = entry.map_err(|e| format!("iterating components: {}", e))?;
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("flow") {
            if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                present.push(stem.to_string());
            }
        }
    }
    present.sort();

    // Validate explicit entries and build final order.
    let mut order = Vec::with_capacity(present.len());
    for name in explicit {
        if !present.contains(name) {
            return Err(format!(
                "manifest.load.components refers to missing component `{}` (not found in components/)",
                name
            ));
        }
        if !order.contains(name) {
            order.push(name.clone());
        }
    }
    for name in present {
        if !order.contains(&name) {
            order.push(name);
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn mktemp() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "canvas_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write(p: &Path, contents: &str) {
        let mut f = fs::File::create(p).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    fn minimal_canvas(dir: &Path) {
        write(
            &dir.join("manifest.json"),
            r#"{ "format_version": 1, "canvas": { "name": "Test" } }"#,
        );
        write(
            &dir.join("main.flow"),
            r#"
node Counter {
    slots { hits: Int = 0 }
    rule on_ping { on ping(_) do { hits := hits + 1 } }
}
scenario { at 0ns: inject Counter <- ping(nil) }
"#,
        );
    }

    #[test]
    fn loads_a_minimal_canvas() {
        let dir = mktemp();
        minimal_canvas(&dir);

        let canvas = load_canvas(&dir, 0).unwrap();
        assert_eq!(canvas.manifest.canvas.name, "Test");
        // Stock gadgets registered.
        assert!(canvas.sim.has_class("Generator"));
        // User-declared class registered + instantiated.
        assert!(canvas.sim.has_class("Counter"));
        assert!(canvas.sim.node_by_name("Counter").is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn main_scenario_auto_runs_when_no_default() {
        let dir = mktemp();
        minimal_canvas(&dir);
        let mut canvas = load_canvas(&dir, 0).unwrap();
        canvas.sim.run_until(1);
        let id = canvas.sim.node_by_name("Counter").unwrap();
        assert_eq!(
            canvas.sim.nodes[&id].slots["hits"],
            flow::Value::Int(1),
            "unnamed `scenario {{ }}` should auto-run as \"main\""
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn custom_component_is_registered() {
        let dir = mktemp();
        fs::create_dir_all(dir.join("components")).unwrap();
        write(
            &dir.join("manifest.json"),
            r#"{ "format_version": 1 }"#,
        );
        write(
            &dir.join("components").join("special.flow"),
            r#"
node Special {
    slots { hits: Int = 0 }
    rule on_ping { on ping(_) do { hits := hits + 1 } }
}
"#,
        );
        write(
            &dir.join("main.flow"),
            r#"
# main.flow names an explicit instance of the Special class declared
# in components/. components/ files only register classes — instances
# must be named in main.flow.
node sp : Special { hits: 0 }
scenario { at 0ns: inject sp <- ping(nil) }
"#,
        );

        let mut canvas = load_canvas(&dir, 0).unwrap();
        assert!(canvas.sim.has_class("Special"));
        canvas.sim.run_until(1);
        let id = canvas.sim.node_by_name("sp").unwrap();
        assert_eq!(canvas.sim.nodes[&id].slots["hits"], flow::Value::Int(1));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_default_scenario_overrides_main() {
        let dir = mktemp();
        write(
            &dir.join("manifest.json"),
            r#"{
                "format_version": 1,
                "default": { "scenario": "warmup" }
            }"#,
        );
        write(
            &dir.join("main.flow"),
            r#"
node C {
    slots { hits: Int = 0 }
    rule on_ping { on ping(_) do { hits := hits + 1 } }
}
scenario warmup {
    at 0ns: inject C <- ping(nil)
    at 0ns: inject C <- ping(nil)
}
"#,
        );

        let mut canvas = load_canvas(&dir, 0).unwrap();
        canvas.sim.run_until(1);
        let id = canvas.sim.node_by_name("C").unwrap();
        assert_eq!(canvas.sim.nodes[&id].slots["hits"], flow::Value::Int(2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn visual_json_parses_when_present() {
        let dir = mktemp();
        minimal_canvas(&dir);
        write(
            &dir.join("visual.json"),
            r##"{
                "format_version": 1,
                "nodes": {
                    "Counter": { "pos": [100.0, -50.0], "color": "#abc123" }
                },
                "viewport": { "pos": [0.0, 0.0], "zoom": 1.5 }
            }"##,
        );
        let canvas = load_canvas(&dir, 0).unwrap();
        let vn = canvas.visual.nodes.get("Counter").unwrap();
        assert_eq!(vn.pos, [100.0, -50.0]);
        assert_eq!(vn.color.as_deref(), Some("#abc123"));
        assert_eq!(canvas.visual.viewport.zoom, 1.5);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn loads_checked_in_sample_canvas() {
        // The repo ships a sample canvas at examples/sample.whiteboard
        // that exercises every piece: main.flow with stock-shaped nodes,
        // a probes block, on_spawn self-loops, visual.json positions.
        // Loading it here catches regressions in the end-to-end path.
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .join("examples")
            .join("sample.whiteboard");
        if !p.exists() {
            // Not running from a full checkout (rare); skip rather than fail.
            return;
        }
        let canvas = load_canvas(&p, 1).unwrap();
        assert_eq!(canvas.sim.nodes.len(), 2);
        assert!(canvas.sim.node_by_name("Client").is_some());
        assert!(canvas.sim.node_by_name("Worker").is_some());
        // Visual positions parsed.
        assert_eq!(canvas.visual.nodes.len(), 2);
        assert_eq!(
            canvas.visual.nodes.get("Client").unwrap().pos,
            [-200.0, 0.0]
        );
    }

    #[test]
    fn snapshot_save_then_load_restores_slot_values() {
        let dir = mktemp();
        minimal_canvas(&dir);

        // Load once, advance sim, save a snapshot.
        let mut canvas = load_canvas(&dir, 0).unwrap();
        canvas.sim.run_until(1);
        let id = canvas.sim.node_by_name("Counter").unwrap();
        assert_eq!(canvas.sim.nodes[&id].slots["hits"], flow::Value::Int(1));
        save_snapshot(&dir, "warm", &canvas.sim, "after first tick").unwrap();

        // Now point the manifest at the snapshot and reload — the slot
        // value from the snapshot overrides the fresh default.
        write(
            &dir.join("manifest.json"),
            r#"{ "format_version": 1, "default": { "snapshot": "warm" } }"#,
        );
        // Remove the unnamed scenario from main.flow so the reloaded
        // canvas doesn't run it a second time and clobber the snapshot.
        write(
            &dir.join("main.flow"),
            r#"
node Counter {
    slots { hits: Int = 0 }
    rule on_ping { on ping(_) do { hits := hits + 1 } }
}
"#,
        );
        let canvas2 = load_canvas(&dir, 0).unwrap();
        let id2 = canvas2.sim.node_by_name("Counter").unwrap();
        assert_eq!(canvas2.sim.nodes[&id2].slots["hits"], flow::Value::Int(1));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_both_scenario_and_snapshot() {
        let dir = mktemp();
        minimal_canvas(&dir);
        write(
            &dir.join("manifest.json"),
            r#"{
                "format_version": 1,
                "default": { "scenario": "main", "snapshot": "warm" }
            }"#,
        );
        let err = match load_canvas(&dir, 0) {
            Err(e) => e,
            Ok(_) => panic!("expected load error"),
        };
        assert!(err.contains("exactly one"), "got: {}", err);
        let _ = fs::remove_dir_all(&dir);
    }
}
