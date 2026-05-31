//! Per-project Issues pane.
//!
//! One unified data model with optional richness — a row can be a
//! quick checkbox-todo (just title + done) or expand into a named
//! issue with body text and arbitrary key/value fields. All issues
//! for a project live in `~/.terminal-bevy/issues/<project_id>.json`,
//! so multiple Issues panes bound to the same project share the same
//! list. Switching the active project hides issues panes from other
//! projects via pane-bevy's standard project visibility.
//!
//! Editing model: each piece of text (title, body, field key, field
//! value) renders as a Text2d in the row layout. Click a piece to
//! enter edit mode — the row layout is rebuilt with a TextInput
//! widget in place of the Text2d at that slot. Enter commits, Esc
//! cancels, click-outside also commits.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextBounds};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use pane_bevy::{
    content_area, focus_text_input, pt_to_content_local, spawn_text_input,
    spawn_text_input_multiline, FocusedPane, FocusedTextInput, InputConsumed, PaneContentNoClip,
    PaneContentPressed, PaneFont, PaneFontMetrics, PaneHotZones, PaneKindSpec, PaneRect,
    PaneRegistry, PaneTag, PaneTitle, TextInput, TextInputEvent, TextInputStyle,
};

use crate::projects::Projects;

const PANE_KIND: &str = "issues";

// ---------- Layout constants ----------

const HEADER_H: f32 = 26.0;
const ROW_H: f32 = 26.0;
const FIELD_ROW_H: f32 = 22.0;
const BODY_ROW_H: f32 = 60.0;
const ROW_PAD_X: f32 = 8.0;
const CHECKBOX_SIZE: f32 = 14.0;
const TEXT_FONT_SIZE: f32 = 13.0;
const SMALL_FONT_SIZE: f32 = 11.0;
const ADD_BTN_H: f32 = 24.0;
/// Line height for wrapped title / body / field-value text. Used both
/// for the Bevy `LineHeight` component and for our local height
/// computation, so they always agree.
const WRAP_LINE_H: f32 = TEXT_FONT_SIZE * 1.3;
const SMALL_WRAP_LINE_H: f32 = SMALL_FONT_SIZE * 1.3;

// ---------- Data model ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    pub id: u64,
    pub title: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub done: bool,
    /// Custom user-defined fields. BTreeMap keeps key order stable.
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
    #[serde(default)]
    pub expanded: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProjectIssuesFile {
    #[serde(default)]
    pub next_id: u64,
    #[serde(default)]
    pub issues: Vec<Issue>,
}

#[derive(Resource, Default)]
pub struct IssuesStore {
    pub by_project: HashMap<u64, ProjectIssuesFile>,
    /// Project ids that have been loaded from disk this session.
    pub loaded: HashSet<u64>,
    /// Project ids whose in-memory state hasn't been flushed.
    pub dirty: HashSet<u64>,
}

impl IssuesStore {
    pub fn ensure_loaded(&mut self, project_id: u64) {
        if self.loaded.contains(&project_id) {
            return;
        }
        let file = load_project_issues(project_id);
        self.by_project.insert(project_id, file);
        self.loaded.insert(project_id);
    }

    pub fn get(&self, project_id: u64) -> Option<&ProjectIssuesFile> {
        self.by_project.get(&project_id)
    }

    pub fn get_mut(&mut self, project_id: u64) -> Option<&mut ProjectIssuesFile> {
        self.by_project.get_mut(&project_id)
    }

    pub fn mark_dirty(&mut self, project_id: u64) {
        self.dirty.insert(project_id);
    }
}

fn storage_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    p.push("issues");
    Some(p)
}

fn project_file_path(project_id: u64) -> Option<PathBuf> {
    Some(storage_dir()?.join(format!("{}.json", project_id)))
}

fn load_project_issues(project_id: u64) -> ProjectIssuesFile {
    let Some(path) = project_file_path(project_id) else {
        return ProjectIssuesFile::default();
    };
    let Ok(bytes) = fs::read(&path) else {
        return ProjectIssuesFile::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        eprintln!(
            "[issues] failed to parse {}: {} — starting empty",
            path.display(),
            e
        );
        ProjectIssuesFile::default()
    })
}

fn save_project_issues(project_id: u64, data: &ProjectIssuesFile) {
    let Some(dir) = storage_dir() else {
        return;
    };
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[issues] mkdir {}: {}", dir.display(), e);
        return;
    }
    let file = dir.join(format!("{}.json", project_id));
    let tmp = dir.join(format!("{}.json.tmp", project_id));
    let bytes = match serde_json::to_vec_pretty(data) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[issues] serialize {} failed: {}", project_id, e);
            return;
        }
    };
    if let Err(e) = fs::write(&tmp, &bytes) {
        eprintln!("[issues] write {} failed: {}", tmp.display(), e);
        return;
    }
    if let Err(e) = fs::rename(&tmp, &file) {
        eprintln!("[issues] rename {} failed: {}", tmp.display(), e);
    }
}

// ---------- Editing state ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditTarget {
    Title(u64),
    Body(u64),
    FieldKey(u64, usize),
    FieldValue(u64, usize),
}

#[derive(Component, Default)]
pub struct IssuesPane {
    /// Project this pane is bound to. Set at spawn; never changes
    /// after that (the pane belongs to one project for its lifetime).
    pub project_id: u64,
    /// What's currently being edited, if anything.
    pub editing: Option<EditTarget>,
    /// Live TextInput entity when in edit mode. Reused across re-layout
    /// while the edit target is unchanged (so the caret/selection
    /// survive the body box growing); despawned + recreated when the
    /// target changes or edit mode ends.
    pub edit_input: Option<Entity>,
    /// The `EditTarget` the live `edit_input` was built for. Lets
    /// `rebuild_rows` tell "same field, just reflow — reuse the input"
    /// from "different field — rebuild it".
    pub edit_target_built: Option<EditTarget>,
    /// True when the pane content needs a full rebuild (issue list
    /// changed, expand toggled, project switched, edit mode changed).
    pub dirty_layout: bool,
}

#[derive(Component, Copy, Clone, Debug)]
pub enum IssueHit {
    /// "+ Add" button at the top.
    AddNew,
    /// Checkbox on a row.
    Checkbox(u64),
    /// Title text — click to edit, click chevron region to expand.
    Title(u64),
    /// Chevron toggling expanded.
    ExpandToggle(u64),
    /// Delete an issue.
    DeleteIssue(u64),
    /// Body text — click to edit.
    Body(u64),
    /// Field key text — click to edit.
    FieldKey(u64, usize),
    /// Field value text — click to edit.
    FieldValue(u64, usize),
    /// Delete a field.
    DeleteField(u64, usize),
    /// "+ Add Field" button inside an expanded issue.
    AddField(u64),
}

#[derive(Component)]
struct IssueRowEntity;

// ---------- Plugin ----------

pub struct IssuesPanePlugin;

impl Plugin for IssuesPanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<IssuesStore>()
            .add_systems(Startup, register_kind)
            .add_systems(
                Update,
                (
                    mark_dirty_on_project_change,
                    handle_content_press,
                    commit_edit_on_focus_change,
                    handle_text_input_events,
                    rebuild_rows,
                    save_dirty_projects,
                    update_issues_hot_zones,
                )
                    .chain(),
            );
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Issues",
        radial_icon: Some("☐"),
        default_size: Vec2::new(480.0, 540.0),
        spawn: issues_spawn,
        snapshot: issues_snapshot,
        on_close: None,
    });
}

// ---------- spawn / snapshot ----------

fn issues_spawn(world: &mut World, entity: Entity, _content_root: Entity, config: &Value) {
    // project_id is the active project at spawn time. Restoration
    // re-uses the saved project_id verbatim.
    let project_id = config
        .get("project_id")
        .and_then(|v| v.as_u64())
        .or_else(|| world.resource::<Projects>().active)
        .unwrap_or(0);

    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        if t.0 == "Issues" || t.0.is_empty() {
            t.0 = "Issues".to_string();
        }
    }

    world.entity_mut(entity).insert(IssuesPane {
        project_id,
        editing: None,
        edit_input: None,
        edit_target_built: None,
        dirty_layout: true,
    });
    // Ensure this project's issues are loaded so the first rebuild
    // has data to show.
    world.resource_mut::<IssuesStore>().ensure_loaded(project_id);
}

fn issues_snapshot(world: &World, entity: Entity) -> Value {
    let project_id = world
        .get::<IssuesPane>(entity)
        .map(|p| p.project_id)
        .unwrap_or(0);
    serde_json::json!({ "project_id": project_id })
}

// ---------- Dirty propagation ----------

fn mark_dirty_on_project_change(
    mut panes: Query<&mut IssuesPane>,
    rect_q: Query<(), (With<PaneTag>, Changed<PaneRect>)>,
    pane_entities: Query<Entity, With<IssuesPane>>,
) {
    // Resize changes the column width → relayout needed.
    if rect_q.is_empty() {
        return;
    }
    for e in &pane_entities {
        if let Ok(mut p) = panes.get_mut(e) {
            p.dirty_layout = true;
        }
    }
}

// ---------- Input ----------

#[allow(clippy::too_many_arguments)]
fn handle_content_press(
    mut events: MessageReader<PaneContentPressed>,
    mut panes: Query<&mut IssuesPane>,
    hits: Query<(&IssueHit, &GlobalTransform, &HitSize)>,
    pane_rects: Query<&PaneRect>,
    inputs: Query<&TextInput>,
    mut store: ResMut<IssuesStore>,
) {
    for ev in events.read() {
        let Ok(mut pane) = panes.get_mut(ev.pane) else {
            continue;
        };
        let Ok(rect) = pane_rects.get(ev.pane) else {
            continue;
        };
        // Find the topmost hit-box under the cursor inside this pane.
        let mut picked: Option<IssueHit> = None;
        for (hit, gt, size) in &hits {
            let world = gt.translation();
            // World y-up; convert to pane-local window-space. The pane
            // origin in world coords is at (rect.pos - half_window).
            // Instead of computing that, just convert the click's
            // window-space pos to content-local using the helper, and
            // compare against the row's content-local coords stored
            // in HitSize.local_origin.
            let _ = world;
            // `ev.local_pt` is already content-local in canvas-space;
            // recomputing from window_pt + canvas-space rect would
            // mis-hit the moment the canvas is panned/zoomed.
            let local = ev.local_pt;
            let origin = size.local_origin;
            if local.x >= origin.x
                && local.x <= origin.x + size.size.x
                && local.y >= origin.y
                && local.y <= origin.y + size.size.y
            {
                picked = Some(*hit);
                // Don't break — last-wins so deeper rebuilds rendered
                // later in the same frame can override; for our layout
                // disjoint hit-rects mean order doesn't matter, so the
                // first match is fine in practice.
                break;
            }
        }
        // A click that leaves (or changes) the field being edited
        // commits its current value first, so edits aren't lost. A click
        // that lands back on the same field (self-hit) keeps editing —
        // no commit, so an empty new-issue title isn't dropped just by
        // clicking inside it.
        let stays_on_same_field = picked
            .as_ref()
            .is_some_and(|h| hit_matches_target(h, pane.editing));
        if !stays_on_same_field {
            commit_active_edit(&pane, &inputs, &mut store);
        }

        let Some(hit) = picked else {
            // Click inside pane but not on any row element — exit
            // edit mode if any (value already committed above).
            if pane.editing.is_some() {
                pane.editing = None;
                pane.dirty_layout = true;
            }
            continue;
        };
        apply_hit(&mut pane, hit, &mut store);
    }
}

/// Mutate the store + pane editing state based on what the user clicked.
fn apply_hit(pane: &mut IssuesPane, hit: IssueHit, store: &mut IssuesStore) {
    let project_id = pane.project_id;
    store.ensure_loaded(project_id);
    let Some(data) = store.get_mut(project_id) else {
        return;
    };
    match hit {
        IssueHit::AddNew => {
            let id = next_id(data);
            data.issues.insert(
                0,
                Issue {
                    id,
                    title: String::new(),
                    body: String::new(),
                    done: false,
                    fields: BTreeMap::new(),
                    expanded: false,
                },
            );
            pane.editing = Some(EditTarget::Title(id));
            pane.dirty_layout = true;
            store.mark_dirty(project_id);
        }
        IssueHit::Checkbox(id) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                issue.done = !issue.done;
                pane.dirty_layout = true;
                store.mark_dirty(project_id);
            }
        }
        IssueHit::Title(id) => {
            pane.editing = Some(EditTarget::Title(id));
            pane.dirty_layout = true;
        }
        IssueHit::ExpandToggle(id) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                issue.expanded = !issue.expanded;
                pane.dirty_layout = true;
                store.mark_dirty(project_id);
            }
        }
        IssueHit::DeleteIssue(id) => {
            data.issues.retain(|i| i.id != id);
            pane.editing = None;
            pane.dirty_layout = true;
            store.mark_dirty(project_id);
        }
        IssueHit::Body(id) => {
            pane.editing = Some(EditTarget::Body(id));
            pane.dirty_layout = true;
        }
        IssueHit::FieldKey(id, idx) => {
            pane.editing = Some(EditTarget::FieldKey(id, idx));
            pane.dirty_layout = true;
        }
        IssueHit::FieldValue(id, idx) => {
            pane.editing = Some(EditTarget::FieldValue(id, idx));
            pane.dirty_layout = true;
        }
        IssueHit::DeleteField(id, idx) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                if let Some(key) = nth_field_key(issue, idx) {
                    issue.fields.remove(&key);
                    pane.dirty_layout = true;
                    store.mark_dirty(project_id);
                }
            }
        }
        IssueHit::AddField(id) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                // Pick a unique placeholder key.
                let mut k = String::from("field");
                let mut n = 1;
                while issue.fields.contains_key(&k) {
                    n += 1;
                    k = format!("field{}", n);
                }
                let idx = issue.fields.len();
                issue.fields.insert(k, String::new());
                pane.editing = Some(EditTarget::FieldKey(id, idx));
                pane.dirty_layout = true;
                store.mark_dirty(project_id);
            }
        }
    }
}

fn next_id(data: &mut ProjectIssuesFile) -> u64 {
    if data.next_id == 0 {
        data.next_id = 1;
    }
    let id = data.next_id;
    data.next_id += 1;
    id
}

/// Snapshot the n-th key in a BTreeMap (matches the order we use to
/// number fields in the UI).
fn nth_field_key(issue: &Issue, idx: usize) -> Option<String> {
    issue.fields.keys().nth(idx).cloned()
}

// ---------- TextInput commit / cancel ----------

fn handle_text_input_events(
    mut events: MessageReader<TextInputEvent>,
    mut panes: Query<&mut IssuesPane>,
    inputs: Query<&TextInput>,
    mut store: ResMut<IssuesStore>,
) {
    for ev in events.read() {
        let (entity, commit) = match ev {
            TextInputEvent::Submit { entity } => (*entity, true),
            TextInputEvent::Cancel { entity } => (*entity, false),
            // A keystroke changed the buffer. The wrapped line count may
            // have changed, so re-flow the owning pane (rebuild reuses
            // the live input entity, so the caret survives).
            TextInputEvent::Changed { entity } => {
                if let Some(mut pane) = panes.iter_mut().find(|p| p.edit_input == Some(*entity)) {
                    pane.dirty_layout = true;
                }
                continue;
            }
        };
        // Find which pane owns this input.
        let Some(mut pane) = panes
            .iter_mut()
            .find(|p| p.edit_input == Some(entity))
        else {
            continue;
        };
        let project_id = pane.project_id;
        let editing = pane.editing;
        if commit {
            if let Ok(input) = inputs.get(entity) {
                let value = input.text();
                if let Some(target) = editing {
                    commit_edit(&mut store, project_id, target, &value);
                }
            }
        }
        pane.editing = None;
        pane.edit_input = None;
        pane.edit_target_built = None;
        pane.dirty_layout = true;
    }
}

fn commit_edit(store: &mut IssuesStore, project_id: u64, target: EditTarget, value: &str) {
    let Some(data) = store.get_mut(project_id) else {
        return;
    };
    let trimmed = value.trim();
    match target {
        EditTarget::Title(id) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                if trimmed.is_empty() {
                    // Empty title — drop the issue if it has no other
                    // content. Otherwise keep the empty title (user
                    // can re-edit).
                    if issue.body.is_empty() && issue.fields.is_empty() {
                        let drop_id = issue.id;
                        data.issues.retain(|i| i.id != drop_id);
                    }
                } else {
                    issue.title = trimmed.to_string();
                }
            }
        }
        EditTarget::Body(id) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                issue.body = value.to_string();
            }
        }
        EditTarget::FieldKey(id, idx) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                let Some(old_key) = nth_field_key(issue, idx) else {
                    return;
                };
                let new_key = trimmed.to_string();
                if new_key.is_empty() || new_key == old_key {
                    return;
                }
                if let Some(val) = issue.fields.remove(&old_key) {
                    issue.fields.insert(new_key, val);
                }
            }
        }
        EditTarget::FieldValue(id, idx) => {
            if let Some(issue) = data.issues.iter_mut().find(|i| i.id == id) {
                if let Some(key) = nth_field_key(issue, idx) {
                    issue.fields.insert(key, value.to_string());
                }
            }
        }
    }
    store.mark_dirty(project_id);
}

/// Commit the pane's current in-progress edit (if any) to the store,
/// reading the live value out of its `TextInput`. Does NOT clear the
/// editing state — callers decide whether the edit continues.
fn commit_active_edit(pane: &IssuesPane, inputs: &Query<&TextInput>, store: &mut IssuesStore) {
    if let (Some(target), Some(input)) = (pane.editing, pane.edit_input) {
        if let Ok(ti) = inputs.get(input) {
            commit_edit(store, pane.project_id, target, &ti.text());
        }
    }
}

/// Whether a clicked `IssueHit` refers to the same slot currently being
/// edited — used to tell "clicked back into the same field" (keep
/// editing) from "moved to a different field" (commit + switch).
fn hit_matches_target(hit: &IssueHit, target: Option<EditTarget>) -> bool {
    match (hit, target) {
        (IssueHit::Title(a), Some(EditTarget::Title(b))) => *a == b,
        (IssueHit::Body(a), Some(EditTarget::Body(b))) => *a == b,
        (IssueHit::FieldKey(a, i), Some(EditTarget::FieldKey(b, j))) => *a == b && *i == j,
        (IssueHit::FieldValue(a, i), Some(EditTarget::FieldValue(b, j))) => *a == b && *i == j,
        _ => false,
    }
}

/// The clickable `IssueHit` for an edit slot. The active edit field's
/// background carries this so clicking inside the field re-selects it
/// (a no-op) instead of falling through to "clicked empty space → exit
/// edit mode".
fn hit_for_target(target: EditTarget) -> IssueHit {
    match target {
        EditTarget::Title(id) => IssueHit::Title(id),
        EditTarget::Body(id) => IssueHit::Body(id),
        EditTarget::FieldKey(id, idx) => IssueHit::FieldKey(id, idx),
        EditTarget::FieldValue(id, idx) => IssueHit::FieldValue(id, idx),
    }
}

/// Commit + exit edit mode when keyboard focus leaves the pane that's
/// editing (the user clicked another pane, or empty canvas). Without
/// this the `TextInput` stays focused and keystrokes keep flowing into
/// the now-hidden field.
fn commit_edit_on_focus_change(
    mut commands: Commands,
    focused_pane: Res<FocusedPane>,
    mut focused_input: ResMut<FocusedTextInput>,
    mut panes: Query<(Entity, &mut IssuesPane)>,
    inputs: Query<&TextInput>,
    mut store: ResMut<IssuesStore>,
) {
    for (entity, mut pane) in &mut panes {
        if pane.editing.is_none() {
            continue;
        }
        // Still the focused pane → keep editing. `FocusedPane` only
        // changes on a pane click, so between clicks this is stable and
        // we never blur spuriously.
        if focused_pane.0 == Some(entity) {
            continue;
        }
        commit_active_edit(&pane, &inputs, &mut store);
        if let Some(input) = pane.edit_input {
            if focused_input.0 == Some(input) {
                focus_text_input(&mut commands, &mut focused_input, [], None);
            }
        }
        pane.editing = None;
        pane.edit_input = None;
        pane.edit_target_built = None;
        pane.dirty_layout = true;
    }
}

// ---------- Layout / render ----------

/// Per-hit-target sidecar with the rectangle in pane-local content
/// coords (x right, y down, origin at content_root). Click handler
/// uses this to point-in-rect test against the click's local pos.
#[derive(Component, Copy, Clone, Debug)]
struct HitSize {
    pub local_origin: Vec2,
    pub size: Vec2,
}

/// Mirror each issues pane's `HitSize` children into `PaneHotZones` so
/// pinned-pane hit-testing can route clicks to checkboxes, chevrons,
/// inline edit fields, etc. All HitSize entities are spawned as direct
/// children of `chrome.content_root`, so a one-hop ChildOf lookup is
/// enough to attribute them to the owning pane.
fn update_issues_hot_zones(
    panes: Query<(Entity, &pane_bevy::PaneChrome), With<IssuesPane>>,
    mut zones_q: Query<&mut PaneHotZones>,
    hits: Query<(&HitSize, &ChildOf), With<IssueHit>>,
) {
    let by_root: std::collections::HashMap<Entity, Entity> = panes
        .iter()
        .map(|(e, c)| (c.content_root, e))
        .collect();
    for (e, _) in panes.iter() {
        if let Ok(mut z) = zones_q.get_mut(e) {
            z.clear();
        }
    }
    for (size, child_of) in &hits {
        let Some(&pane) = by_root.get(&child_of.0) else { continue };
        let Ok(mut z) = zones_q.get_mut(pane) else { continue };
        z.push(Rect::from_corners(
            size.local_origin,
            size.local_origin + size.size,
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn rebuild_rows(
    mut commands: Commands,
    mut panes: Query<(Entity, &mut IssuesPane, &PaneRect, &pane_bevy::PaneChrome)>,
    existing_rows: Query<(Entity, &ChildOf), With<IssueRowEntity>>,
    inputs_q: Query<&TextInput>,
    store: Res<IssuesStore>,
    font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    mut focused: ResMut<FocusedTextInput>,
) {
    if !theme.is_changed() && panes.iter().all(|(_, p, _, _)| !p.dirty_layout) {
        return;
    }

    use style_bevy::tokens as t;
    let c = |id| Color::LinearRgba(theme.color(id));
    let fg = c(t::FG);
    let fg_muted = c(t::FG_MUTED);
    let accent = c(t::ACCENT);
    let divider = c(t::CHROME_DIVIDER);
    let row_hover_bg = c(t::SIDEBAR_ROW_ACTIVE_BG);
    let input_bg = c(t::INPUT_BG);

    for (pane_entity, mut pane, rect, chrome) in &mut panes {
        if !pane.dirty_layout && !theme.is_changed() {
            continue;
        }
        // If we're still editing the same target as last layout, reuse
        // the live input entity across this rebuild so the caret and
        // selection survive (the body box grows by re-flowing the rows
        // around a persistent input). Otherwise the input is rebuilt.
        let reuse: Option<Entity> = match (pane.editing, pane.edit_target_built, pane.edit_input) {
            (Some(cur), Some(built), Some(input))
                if cur == built && inputs_q.get(input).is_ok() =>
            {
                Some(input)
            }
            _ => None,
        };
        // Live edit text (for the reused input) drives the editing
        // slot's wrapped height so the box grows as you type.
        let live_edit_text: Option<String> =
            reuse.and_then(|e| inputs_q.get(e).ok()).map(|ti| ti.text());

        // Despawn previous rows for this pane — except the reused input
        // (and its text/caret children, which hang off it, not off
        // content_root, so the parent walk leaves them alone).
        for (row, child_of) in &existing_rows {
            if child_of.0 == chrome.content_root && Some(row) != reuse {
                commands.entity(row).despawn();
            }
        }
        if reuse.is_none() {
            // Not reusing: the old input (if any) was despawned above.
            if let Some(old) = pane.edit_input.take() {
                if focused.0 == Some(old) {
                    focused.0 = None;
                }
            }
            pane.edit_target_built = None;
        }
        pane.dirty_layout = false;

        let (_origin, content_size) = content_area(rect);
        let content_w = content_size.x;
        if content_w <= 0.0 {
            continue;
        }

        let Some(data) = store.get(pane.project_id) else {
            continue;
        };

        let mut y = 0.0_f32;

        // Header row: "+ Add Issue" button.
        spawn_row_bg(
            &mut commands,
            chrome.content_root,
            Vec2::new(0.0, y),
            Vec2::new(content_w, HEADER_H),
            row_hover_bg.with_alpha(0.0), // invisible bg, just for layout
        );
        let add_btn_w = 110.0_f32.min(content_w - 2.0 * ROW_PAD_X);
        let add_btn_x = ROW_PAD_X;
        let add_btn_y = y + (HEADER_H - ADD_BTN_H) * 0.5;
        commands.spawn((
            IssueRowEntity,
            ChildOf(chrome.content_root),
            Sprite {
                color: accent.with_alpha(0.18),
                custom_size: Some(Vec2::new(add_btn_w, ADD_BTN_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(add_btn_x, -add_btn_y, 0.1),
            IssueHit::AddNew,
            HitSize {
                local_origin: Vec2::new(add_btn_x, add_btn_y),
                size: Vec2::new(add_btn_w, ADD_BTN_H),
            },
        ));
        commands.spawn((
            IssueRowEntity,
            ChildOf(chrome.content_root),
            Text2d::new("+ New issue"),
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(ADD_BTN_H),
            TextColor(accent),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(add_btn_x + 10.0, -(add_btn_y + ADD_BTN_H * 0.5), 0.2),
        ));
        y += HEADER_H;

        // Bottom hairline under header.
        commands.spawn((
            IssueRowEntity,
            ChildOf(chrome.content_root),
            Sprite {
                color: divider,
                custom_size: Some(Vec2::new(content_w, 1.0)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -y, 0.05),
        ));
        y += 1.0;

        let style = text_input_style(font.0.clone(), &metrics, fg, accent);

        for issue in &data.issues {
            y = spawn_issue_block(
                &mut commands,
                chrome.content_root,
                issue,
                content_w,
                y,
                &font,
                &metrics,
                &style,
                fg,
                fg_muted,
                accent,
                divider,
                input_bg,
                pane.editing,
                pane_entity,
                &mut focused,
                &mut pane,
                reuse,
                live_edit_text.as_deref(),
            );
        }
    }
}

/// Count the wrapped lines `text` will occupy if rendered into a slot
/// `wrap_px` wide, given a monospace advance of `char_px` per char.
/// Mirrors Bevy/cosmic-text's word-wrap behavior closely enough for
/// the host's monospace fonts: greedy word fill, char-break for words
/// that exceed the wrap width on their own. Preserves explicit `\n`.
fn wrap_line_count(text: &str, wrap_px: f32, char_px: f32) -> usize {
    if text.is_empty() {
        return 1;
    }
    if char_px <= 0.0 || wrap_px <= char_px {
        // Degenerate width — fall back to explicit-newline count so we
        // never report zero lines.
        return text.lines().count().max(1);
    }
    let max_chars = (wrap_px / char_px).floor() as usize;
    if max_chars == 0 {
        return text.lines().count().max(1);
    }
    let mut total = 0usize;
    for hard_line in text.split('\n') {
        if hard_line.is_empty() {
            total += 1;
            continue;
        }
        let mut col = 0usize;
        let mut line_count = 1usize;
        let mut first_word = true;
        for word in hard_line.split(' ') {
            let wlen = word.chars().count();
            if wlen == 0 {
                // Run of spaces — counts as one char's worth of column.
                if !first_word {
                    if col + 1 > max_chars {
                        line_count += 1;
                        col = 0;
                    } else {
                        col += 1;
                    }
                }
                first_word = false;
                continue;
            }
            // Width needed to place this word on the current line
            // (including the joining space, except for the first word).
            let needed = if first_word { wlen } else { wlen + 1 };
            if col + needed <= max_chars {
                col += needed;
            } else if wlen <= max_chars {
                // Wrap word onto a fresh line.
                line_count += 1;
                col = wlen;
            } else {
                // Word longer than the wrap width — char-break it.
                // Start on a fresh line if the current one isn't empty.
                if col > 0 {
                    line_count += 1;
                    col = 0;
                }
                let mut remaining = wlen;
                while remaining > max_chars {
                    line_count += 1;
                    remaining -= max_chars;
                }
                col = remaining;
            }
            first_word = false;
        }
        total += line_count;
    }
    total.max(1)
}

fn text_input_style(
    font: Handle<Font>,
    metrics: &PaneFontMetrics,
    fg: Color,
    accent: Color,
) -> TextInputStyle {
    TextInputStyle {
        font,
        font_size: TEXT_FONT_SIZE,
        // Match WRAP_LINE_H so a multiline input's rendered line
        // spacing, caret row stepping, and the height the layout
        // reserves per wrapped line all agree.
        line_height: WRAP_LINE_H,
        cell_width: metrics.char_width(TEXT_FONT_SIZE),
        color_idle: fg,
        color_focused: fg,
        color_caret: accent,
        color_selection: accent.with_alpha(0.35),
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_issue_block(
    commands: &mut Commands,
    parent: Entity,
    issue: &Issue,
    content_w: f32,
    mut y: f32,
    font: &Res<PaneFont>,
    metrics: &PaneFontMetrics,
    input_style: &TextInputStyle,
    fg: Color,
    fg_muted: Color,
    accent: Color,
    divider: Color,
    input_bg: Color,
    editing: Option<EditTarget>,
    _pane_entity: Entity,
    focused: &mut ResMut<FocusedTextInput>,
    pane: &mut IssuesPane,
    reuse: Option<Entity>,
    live_edit_text: Option<&str>,
) -> f32 {
    let row_top = y;

    // Compute the title row's actual height first — once the title
    // wraps to multiple lines, the entire top row needs to grow so the
    // body / next issue don't overlap. We measure wrap up front because
    // every offset below (delete button y, expansion start y) is keyed
    // off the row's bottom edge.
    let title_char_w = metrics.char_width(TEXT_FONT_SIZE);
    let chev_w = 14.0_f32;
    let title_x = ROW_PAD_X + CHECKBOX_SIZE + 6.0 + chev_w + 4.0;
    let delete_w = 18.0_f32;
    let title_w = (content_w - title_x - delete_w - ROW_PAD_X).max(20.0);
    let title_display: String = if issue.title.is_empty() {
        "(untitled)".to_string()
    } else {
        issue.title.clone()
    };
    let editing_title = editing == Some(EditTarget::Title(issue.id));
    let title_lines = if editing_title {
        // While editing, size from the live wrapped text at the input's
        // own wrap width (its content width) so the edit box keeps the
        // same multi-line shape the display had.
        let live = live_edit_text.unwrap_or(&issue.title);
        wrap_line_count(live, (title_w - 4.0).max(20.0), title_char_w)
    } else {
        wrap_line_count(&title_display, title_w, title_char_w)
    };
    let title_text_h = title_lines as f32 * WRAP_LINE_H;
    let row_h = (title_text_h + 9.0).max(ROW_H);

    // Checkbox.
    let cb_x = ROW_PAD_X;
    // Checkbox and chevron stay pinned to the first text line so they
    // don't drift downward when the title wraps to more lines.
    let cb_y = y + (ROW_H - CHECKBOX_SIZE) * 0.5;
    let cb_color = if issue.done { accent } else { fg_muted };
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color: cb_color.with_alpha(if issue.done { 1.0 } else { 0.0 }),
            custom_size: Some(Vec2::splat(CHECKBOX_SIZE)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(cb_x, -cb_y, 0.1),
        IssueHit::Checkbox(issue.id),
        HitSize {
            local_origin: Vec2::new(cb_x, cb_y),
            size: Vec2::splat(CHECKBOX_SIZE),
        },
    ));
    // Checkbox border (1px outline using 4 thin sprites).
    spawn_outline(commands, parent, Vec2::new(cb_x, cb_y), Vec2::splat(CHECKBOX_SIZE), cb_color);
    // Checkbox glyph when done.
    if issue.done {
        commands.spawn((
            IssueRowEntity,
            ChildOf(parent),
            Text2d::new("✓"),
            TextFont {
                font: font.0.clone(),
                font_size: 12.0,
                ..default()
            },
            LineHeight::Px(CHECKBOX_SIZE),
            TextColor(Color::WHITE),
            Anchor::CENTER,
            Transform::from_xyz(cb_x + CHECKBOX_SIZE * 0.5, -(cb_y + CHECKBOX_SIZE * 0.5), 0.3),
        ));
    }

    // Chevron toggle (▸ / ▾) — clickable region next to the checkbox.
    let chev_x = cb_x + CHECKBOX_SIZE + 6.0;
    let chev_y = y + (ROW_H - CHECKBOX_SIZE) * 0.5;
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Text2d::new(if issue.expanded { "▾" } else { "▸" }),
        TextFont {
            font: font.0.clone(),
            font_size: 12.0,
            ..default()
        },
        LineHeight::Px(CHECKBOX_SIZE),
        TextColor(fg_muted),
        Anchor::TOP_LEFT,
        Transform::from_xyz(chev_x, -chev_y, 0.2),
        IssueHit::ExpandToggle(issue.id),
        HitSize {
            local_origin: Vec2::new(chev_x, chev_y),
            size: Vec2::new(chev_w, CHECKBOX_SIZE),
        },
    ));

    // Title. Position the first line of text so it sits on the same
    // baseline as the (fixed-size) checkbox/chevron, then let wrapped
    // lines extend downward and grow `row_h`.
    let title_y = y + (ROW_H - WRAP_LINE_H) * 0.5;
    let title_h = title_text_h.max(ROW_H - 4.0);
    if editing_title {
        let title_box_h = (title_text_h + 4.0).max(ROW_H - 4.0);
        spawn_inline_input(
            commands,
            parent,
            Vec2::new(title_x, title_y),
            Vec2::new(title_w, title_box_h),
            input_bg,
            &issue.title,
            input_style.clone(),
            focused,
            pane,
            EditTarget::Title(issue.id),
            reuse,
            true,
        );
    } else {
        let title_color = if issue.done { fg_muted } else { fg };
        commands.spawn((
            IssueRowEntity,
            ChildOf(parent),
            Text2d::new(title_display),
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(WRAP_LINE_H),
            TextColor(title_color),
            Anchor::TOP_LEFT,
            Transform::from_xyz(title_x, -title_y, 0.1),
            // Use our measured wrap width (which reserves the delete
            // column) instead of letting pane-bevy's enforcer pick a
            // wider one — otherwise our line count under-reports and
            // the row clips.
            TextBounds {
                width: Some(title_w),
                height: None,
            },
            PaneContentNoClip,
            IssueHit::Title(issue.id),
            HitSize {
                local_origin: Vec2::new(title_x, title_y),
                size: Vec2::new(title_w, title_h),
            },
        ));
    }

    // Delete (×) button. Pinned to the first line of the title.
    let del_x = content_w - ROW_PAD_X - delete_w;
    let del_y = y + (ROW_H - 16.0) * 0.5;
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Text2d::new("×"),
        TextFont {
            font: font.0.clone(),
            font_size: 14.0,
            ..default()
        },
        LineHeight::Px(16.0),
        TextColor(fg_muted),
        Anchor::TOP_LEFT,
        Transform::from_xyz(del_x, -del_y, 0.2),
        IssueHit::DeleteIssue(issue.id),
        HitSize {
            local_origin: Vec2::new(del_x, del_y),
            size: Vec2::new(delete_w, 16.0),
        },
    ));

    y += row_h;

    // Expanded section.
    if issue.expanded {
        let indent_x = title_x;
        let inner_w = (content_w - indent_x - ROW_PAD_X).max(20.0);

        // Body block. Wraps in the available column; height grows with
        // line count.
        let body_text = if issue.body.is_empty() {
            "(no description — click to add)".to_string()
        } else {
            issue.body.clone()
        };
        let body_wrap_w = (inner_w - 12.0).max(20.0); // 4px frame + 6px text indent on each side margin
        let body_lines = if editing == Some(EditTarget::Body(issue.id)) {
            // While editing, the box grows to fit the wrapped live text
            // at the input's own wrap width (its content width), so what
            // we render lines up with the height we reserve.
            let live = live_edit_text.unwrap_or(&issue.body);
            let edit_wrap_w = (inner_w - 4.0).max(20.0);
            wrap_line_count(live, edit_wrap_w, title_char_w)
        } else {
            wrap_line_count(&body_text, body_wrap_w, title_char_w)
        };
        let body_text_h = body_lines as f32 * WRAP_LINE_H;
        let body_y = y + 4.0;
        let body_h = (body_text_h + 8.0).max(BODY_ROW_H);
        if editing == Some(EditTarget::Body(issue.id)) {
            spawn_inline_input(
                commands,
                parent,
                Vec2::new(indent_x, body_y),
                Vec2::new(inner_w, body_h - 4.0),
                input_bg,
                &issue.body,
                input_style.clone(),
                focused,
                pane,
                EditTarget::Body(issue.id),
                reuse,
                true,
            );
        } else {
            // Body bg so the click target reads as a slot.
            commands.spawn((
                IssueRowEntity,
                ChildOf(parent),
                Sprite {
                    color: input_bg.with_alpha(0.5),
                    custom_size: Some(Vec2::new(inner_w, body_h - 4.0)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(indent_x, -body_y, 0.05),
                IssueHit::Body(issue.id),
                HitSize {
                    local_origin: Vec2::new(indent_x, body_y),
                    size: Vec2::new(inner_w, body_h - 4.0),
                },
            ));
            let body_color = if issue.body.is_empty() { fg_muted } else { fg };
            commands.spawn((
                IssueRowEntity,
                ChildOf(parent),
                Text2d::new(body_text),
                TextFont {
                    font: font.0.clone(),
                    font_size: TEXT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(WRAP_LINE_H),
                TextColor(body_color),
                Anchor::TOP_LEFT,
                Transform::from_xyz(indent_x + 6.0, -(body_y + 4.0), 0.1),
                TextBounds {
                    width: Some(body_wrap_w),
                    height: None,
                },
                PaneContentNoClip,
            ));
        }
        y += body_h;

        // Field rows.
        let key_col_w = (inner_w * 0.32).clamp(60.0, 140.0);
        let value_col_w = inner_w - key_col_w - delete_w - 8.0;
        let small_char_w = metrics.char_width(SMALL_FONT_SIZE);
        for (idx, (key, value)) in issue.fields.iter().enumerate() {
            let value_display = if value.is_empty() {
                "(empty)".to_string()
            } else {
                value.clone()
            };
            let editing_this_key = editing == Some(EditTarget::FieldKey(issue.id, idx));
            let editing_this_val = editing == Some(EditTarget::FieldValue(issue.id, idx));
            // The value editor uses the regular (TEXT_FONT_SIZE) input
            // font, so while editing we size from the live wrapped text
            // at that font's metrics; the display uses the smaller font.
            let (value_lines, value_line_h) = if editing_this_val {
                let live = live_edit_text.unwrap_or(value.as_str());
                (
                    wrap_line_count(live, (value_col_w - 4.0).max(20.0), title_char_w),
                    WRAP_LINE_H,
                )
            } else {
                (
                    wrap_line_count(&value_display, value_col_w, small_char_w),
                    SMALL_WRAP_LINE_H,
                )
            };
            let value_text_h = value_lines as f32 * value_line_h;
            let field_h = (value_text_h).max(FIELD_ROW_H - 4.0);
            let row_y = y + 2.0;

            // Key — always single line in practice; pin to first line.
            let key_line_h = (FIELD_ROW_H - 4.0).min(field_h);
            if editing_this_key {
                spawn_inline_input(
                    commands,
                    parent,
                    Vec2::new(indent_x, row_y),
                    Vec2::new(key_col_w, key_line_h),
                    input_bg,
                    key,
                    input_style.clone(),
                    focused,
                    pane,
                    EditTarget::FieldKey(issue.id, idx),
                    reuse,
                    false,
                );
            } else {
                commands.spawn((
                    IssueRowEntity,
                    ChildOf(parent),
                    Text2d::new(key.clone()),
                    TextFont {
                        font: font.0.clone(),
                        font_size: SMALL_FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(key_line_h),
                    TextColor(fg_muted),
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(indent_x, -row_y, 0.1),
                    IssueHit::FieldKey(issue.id, idx),
                    HitSize {
                        local_origin: Vec2::new(indent_x, row_y),
                        size: Vec2::new(key_col_w, key_line_h),
                    },
                ));
            }

            // Value — wraps to value_col_w; height grows with line count.
            let value_x = indent_x + key_col_w + 4.0;
            if editing_this_val {
                spawn_inline_input(
                    commands,
                    parent,
                    Vec2::new(value_x, row_y),
                    Vec2::new(value_col_w, field_h),
                    input_bg,
                    value,
                    input_style.clone(),
                    focused,
                    pane,
                    EditTarget::FieldValue(issue.id, idx),
                    reuse,
                    true,
                );
            } else {
                let val_color = if value.is_empty() { fg_muted } else { fg };
                commands.spawn((
                    IssueRowEntity,
                    ChildOf(parent),
                    Text2d::new(value_display),
                    TextFont {
                        font: font.0.clone(),
                        font_size: SMALL_FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(SMALL_WRAP_LINE_H),
                    TextColor(val_color),
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(value_x, -row_y, 0.1),
                    TextBounds {
                        width: Some(value_col_w),
                        height: None,
                    },
                    PaneContentNoClip,
                    IssueHit::FieldValue(issue.id, idx),
                    HitSize {
                        local_origin: Vec2::new(value_x, row_y),
                        size: Vec2::new(value_col_w, field_h),
                    },
                ));
            }

            // Field delete (×) — pinned to first line.
            let fdel_x = value_x + value_col_w + 4.0;
            commands.spawn((
                IssueRowEntity,
                ChildOf(parent),
                Text2d::new("×"),
                TextFont {
                    font: font.0.clone(),
                    font_size: 12.0,
                    ..default()
                },
                LineHeight::Px(key_line_h),
                TextColor(fg_muted),
                Anchor::TOP_LEFT,
                Transform::from_xyz(fdel_x, -row_y, 0.2),
                IssueHit::DeleteField(issue.id, idx),
                HitSize {
                    local_origin: Vec2::new(fdel_x, row_y),
                    size: Vec2::new(delete_w, key_line_h),
                },
            ));

            y += (field_h + 4.0).max(FIELD_ROW_H);
        }

        // "+ Add Field" button.
        let af_w = 90.0_f32.min(inner_w);
        let af_y = y + 2.0;
        let af_h = ADD_BTN_H - 4.0;
        commands.spawn((
            IssueRowEntity,
            ChildOf(parent),
            Sprite {
                color: accent.with_alpha(0.10),
                custom_size: Some(Vec2::new(af_w, af_h)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(indent_x, -af_y, 0.1),
            IssueHit::AddField(issue.id),
            HitSize {
                local_origin: Vec2::new(indent_x, af_y),
                size: Vec2::new(af_w, af_h),
            },
        ));
        commands.spawn((
            IssueRowEntity,
            ChildOf(parent),
            Text2d::new("+ Add field"),
            TextFont {
                font: font.0.clone(),
                font_size: SMALL_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(af_h),
            TextColor(accent),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(indent_x + 6.0, -(af_y + af_h * 0.5), 0.2),
        ));
        y += ADD_BTN_H;
    }

    // Bottom hairline.
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color: divider,
            custom_size: Some(Vec2::new(content_w, 1.0)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(0.0, -y, 0.05),
    ));
    y += 1.0;
    let _ = row_top;
    y
}

fn spawn_row_bg(
    commands: &mut Commands,
    parent: Entity,
    local_origin: Vec2,
    size: Vec2,
    color: Color,
) {
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color,
            custom_size: Some(size),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(local_origin.x, -local_origin.y, 0.0),
    ));
}

/// Spawn a 1px outline frame using four thin sprites — gives us a
/// hollow rectangle without needing a custom material.
fn spawn_outline(
    commands: &mut Commands,
    parent: Entity,
    origin: Vec2,
    size: Vec2,
    color: Color,
) {
    // top
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color,
            custom_size: Some(Vec2::new(size.x, 1.0)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, 0.2),
    ));
    // bottom
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color,
            custom_size: Some(Vec2::new(size.x, 1.0)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -(origin.y + size.y - 1.0), 0.2),
    ));
    // left
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color,
            custom_size: Some(Vec2::new(1.0, size.y)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, 0.2),
    ));
    // right
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color,
            custom_size: Some(Vec2::new(1.0, size.y)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x + size.x - 1.0, -origin.y, 0.2),
    ));
}

#[allow(clippy::too_many_arguments)]
fn spawn_inline_input(
    commands: &mut Commands,
    parent: Entity,
    origin: Vec2,
    size: Vec2,
    bg: Color,
    initial: &str,
    style: TextInputStyle,
    focused: &mut ResMut<FocusedTextInput>,
    pane: &mut IssuesPane,
    target: EditTarget,
    reuse: Option<Entity>,
    multiline: bool,
) {
    // Background frame so the active edit slot reads as an input. It
    // carries the slot's own hit so clicking inside the field re-selects
    // it (a no-op) rather than dropping out of edit mode.
    commands.spawn((
        IssueRowEntity,
        ChildOf(parent),
        Sprite {
            color: bg,
            custom_size: Some(size),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(origin.x, -origin.y, 0.05),
        hit_for_target(target),
        HitSize {
            local_origin: origin,
            size,
        },
    ));

    if let Some(existing) = reuse {
        // Re-flow only: keep the live input (and its caret/selection).
        // Reposition it in case rows above shifted; the fresh bg above
        // already grew to the new wrapped height. The input's own wrap
        // width is fixed at spawn — a pane resize mid-edit won't re-wrap
        // until the edit ends, which is acceptable.
        commands
            .entity(existing)
            .insert(Transform::from_xyz(origin.x + 4.0, -(origin.y + 2.0), 0.2));
        pane.edit_input = Some(existing);
        pane.edit_target_built = Some(target);
        return;
    }

    let transform = Transform::from_xyz(origin.x + 4.0, -(origin.y + 2.0), 0.2);
    let input = if multiline {
        spawn_text_input_multiline(commands, parent, initial, style, size.x - 4.0, transform)
    } else {
        spawn_text_input(commands, parent, initial, style, size.x - 4.0, transform)
    };
    commands.entity(input).insert(IssueRowEntity);
    pane.edit_input = Some(input);
    pane.edit_target_built = Some(target);
    focus_text_input(commands, focused, [], Some(input));
}

// ---------- Save ----------

/// Persist dirty projects whenever the user releases left mouse or
/// Esc — mirrors the existing project-state debounce.
fn save_dirty_projects(
    mut store: ResMut<IssuesStore>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut keyboard: MessageReader<KeyboardInput>,
) {
    // Drain so we don't accumulate stale events; we only care about
    // any-keypress flush.
    let any_key = keyboard.read().any(|e| e.state.is_pressed());
    let mouse_up = buttons.just_released(MouseButton::Left);
    let escape = keys.just_pressed(KeyCode::Escape);
    let force_flush = mouse_up || escape || any_key;
    if !force_flush {
        return;
    }
    if store.dirty.is_empty() {
        return;
    }
    let pids: Vec<u64> = store.dirty.iter().copied().collect();
    for pid in pids {
        if let Some(data) = store.by_project.get(&pid) {
            save_project_issues(pid, data);
        }
    }
    store.dirty.clear();
}

// Silence the InputConsumed import in case the click handler ever
// needs to gate by it; keeping the import keeps the diff minimal.
#[allow(dead_code)]
fn _input_consumed_witness(_c: Res<InputConsumed>, _p: Res<FocusedPane>) {}
