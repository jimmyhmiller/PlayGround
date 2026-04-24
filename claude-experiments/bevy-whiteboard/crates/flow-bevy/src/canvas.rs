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

use crate::bridge::{EntityMaps, FlowEdgeRef, FlowSim};
use crate::gadgets::{self, Kind};
use crate::nodes::{NodeCounter, spawn_node_entity};
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
    /// Keyed by node name (as declared in DSL). Missing nodes fall back
    /// to default layout from the UI layer.
    #[serde(default)]
    pub nodes: BTreeMap<String, VisualNode>,
    #[serde(default)]
    pub viewport: Viewport,
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
    // adds). Each component file is lowered into the shared sim so its
    // `node X { }` block both registers the class and creates the
    // singleton instance `X` — same behaviour as the rest of the DSL.
    // Multiple instances of one class require either separate files or
    // top-level `node` blocks in main.flow referencing the shared class
    // by name.
    let comp_order = resolve_component_order(&path, &manifest.load.components)?;
    for stem in &comp_order {
        let src = fs::read_to_string(path.join("components").join(format!("{}.flow", stem)))
            .map_err(|e| format!("reading component `{}`: {}", stem, e))?;
        let comp_file = flow::dsl::parse(&src)
            .map_err(|e| format!("parsing component `{}`: {}", stem, e))?;
        flow::dsl::lower_into(&mut sim, &comp_file)
            .map_err(|e| format!("lowering component `{}`: {}", stem, e))?;
    }

    // main.flow: the actual wiring. Parsed, then lowered *into* the
    // existing sim so it sees stock + component classes.
    let main_path = path.join("main.flow");
    let main_src = fs::read_to_string(&main_path)
        .map_err(|e| format!("reading {}: {}", main_path.display(), e))?;
    let file = flow::dsl::parse(&main_src)
        .map_err(|e| format!("parsing main.flow: {}", e))?;
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut flow: ResMut<FlowSim>,
    mut maps: ResMut<EntityMaps>,
    mut counter: ResMut<NodeCounter>,
    mut node_colors: ResMut<NodeColors>,
    theme: Res<Theme>,
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
    bevy::log::info!(
        "loaded canvas `{}` ({} nodes, {} edges)",
        canvas.manifest.canvas.name,
        canvas.sim.nodes.len(),
        canvas.sim.edges.len()
    );

    flow.sim = canvas.sim;
    flow.consumed_log_index = flow.sim.log.total_recorded;
    counter.0 = flow.sim.nodes.len() as u32;

    // Spawn an entity for each node. Deterministic iteration order
    // (BTreeMap by NodeId) so fall-back grid positions stay stable
    // across loads of the same canvas.
    let visual = &canvas.visual;
    let node_ids: Vec<flow::NodeId> = flow.sim.nodes.keys().copied().collect();
    for (i, nid) in node_ids.iter().copied().enumerate() {
        let (name, kind, color_slot) = {
            let node = &flow.sim.nodes[&nid];
            let kind = node.class.as_deref().map(class_to_kind).unwrap_or(Kind::Worker);
            let color_slot = match node.slots.get("color") {
                Some(Value::Int(n)) => (*n as usize) % theme.data.len(),
                _ => 0,
            };
            (node.name.clone(), kind, color_slot)
        };
        let pos = visual
            .nodes
            .get(&name)
            .map(|v| Vec2::new(v.pos[0], v.pos[1]))
            .unwrap_or_else(|| default_grid_position(i));
        spawn_node_entity(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut maps,
            &theme,
            nid,
            kind,
            name,
            pos,
        );
        if !matches!(kind, Kind::Router) {
            node_colors.0.insert(nid, theme.data[color_slot]);
        }
    }

    // Spawn entities for every existing sim edge. No topology changes
    // here — the DSL already populated them.
    for eid in flow.sim.edges.keys().copied().collect::<Vec<_>>() {
        let edge = &flow.sim.edges[&eid];
        if edge.from == edge.to {
            // Self-loops are sim plumbing (tick / done); the UI draws
            // nothing for them and no entity is needed for routing.
            continue;
        }
        let ent = commands.spawn(FlowEdgeRef(eid)).id();
        maps.edge_to_entity.insert(eid, ent);
        maps.entity_to_edge.insert(ent, eid);
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

/// Map a DSL class name to a visual [`Kind`]. The seven stock gadget
/// class names match directly; anything else (custom components) falls
/// back to `Kind::Worker`, which has a neutral rectangular shape that
/// reads as "generic node" without implying any specific semantics.
/// A future richer UI could let custom classes declare their own
/// shape/color in DSL attributes.
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
        assert!(canvas.sim.templates.contains_key("Generator"));
        // User-declared class registered + instantiated.
        assert!(canvas.sim.templates.contains_key("Counter"));
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
# main.flow wires a Special instance and pings it.
node Wrapper {
    slots { x: Int = 0 }
}
# Special auto-instantiates under its own name when the component file
# is loaded, so main.flow can reference "Special" in its scenario.
scenario { at 0ns: inject Special <- ping(nil) }
"#,
        );

        let mut canvas = load_canvas(&dir, 0).unwrap();
        assert!(canvas.sim.templates.contains_key("Special"));
        canvas.sim.run_until(1);
        let id = canvas.sim.node_by_name("Special").unwrap();
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
