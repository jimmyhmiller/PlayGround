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
use bevy::text::LineHeight;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use pane_bevy::{
    content_area, focus_text_input, pt_to_content_local, spawn_text_input, FocusedPane,
    FocusedTextInput, InputConsumed, PaneContentPressed, PaneFont, PaneFontMetrics, PaneHotZones,
    PaneKindSpec, PaneRect, PaneRegistry, PaneTag, PaneTitle, TextInput, TextInputEvent,
    TextInputStyle,
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
    /// Live TextInput entity when in edit mode. We despawn + recreate
    /// it whenever `editing` changes or rows are rebuilt.
    pub edit_input: Option<Entity>,
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
            let local = pt_to_content_local(ev.window_pt, rect);
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
        let Some(hit) = picked else {
            // Click inside pane but not on any row element — exit
            // edit mode if any.
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
            TextInputEvent::Changed { .. } => continue,
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
    store: Res<IssuesStore>,
    font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    mut focused: ResMut<FocusedTextInput>,
) {
    // If nothing changed and theme didn't change, fast-out.
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
        // Despawn previous rows for this pane.
        for (row, child_of) in &existing_rows {
            if child_of.0 == chrome.content_root {
                commands.entity(row).despawn();
            }
        }
        // Drop the previous edit input if any (the despawn above
        // already removed it since it was a child of content_root).
        if let Some(_old) = pane.edit_input.take() {
            // Don't double-despawn; it's gone with the parent walk
            // above. Clearing edit_input is enough.
            if focused.0 == pane.edit_input {
                focused.0 = None;
            }
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
            );
        }
    }
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
        line_height: TEXT_FONT_SIZE * 1.4,
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
) -> f32 {
    let row_top = y;

    // Checkbox.
    let cb_x = ROW_PAD_X;
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
    let chev_w = 14.0;
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

    // Title.
    let title_x = chev_x + chev_w + 4.0;
    let delete_w = 18.0;
    let title_w = (content_w - title_x - delete_w - ROW_PAD_X).max(20.0);
    let title_y = y + 2.0;
    let title_h = ROW_H - 4.0;
    if editing == Some(EditTarget::Title(issue.id)) {
        spawn_inline_input(
            commands,
            parent,
            Vec2::new(title_x, title_y),
            Vec2::new(title_w, title_h),
            input_bg,
            &issue.title,
            input_style.clone(),
            focused,
            pane,
        );
    } else {
        let title_color = if issue.done { fg_muted } else { fg };
        let display = if issue.title.is_empty() {
            "(untitled)".to_string()
        } else {
            issue.title.clone()
        };
        commands.spawn((
            IssueRowEntity,
            ChildOf(parent),
            Text2d::new(display),
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(ROW_H),
            TextColor(title_color),
            Anchor::TOP_LEFT,
            Transform::from_xyz(title_x, -title_y, 0.1),
            IssueHit::Title(issue.id),
            HitSize {
                local_origin: Vec2::new(title_x, title_y),
                size: Vec2::new(title_w, title_h),
            },
        ));
    }

    // Delete (×) button.
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

    y += ROW_H;

    // Expanded section.
    if issue.expanded {
        let indent_x = title_x;
        let inner_w = (content_w - indent_x - ROW_PAD_X).max(20.0);

        // Body block.
        let body_y = y + 4.0;
        let body_h = BODY_ROW_H;
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
            let display = if issue.body.is_empty() {
                "(no description — click to add)".to_string()
            } else {
                issue.body.clone()
            };
            let body_color = if issue.body.is_empty() { fg_muted } else { fg };
            commands.spawn((
                IssueRowEntity,
                ChildOf(parent),
                Text2d::new(display),
                TextFont {
                    font: font.0.clone(),
                    font_size: TEXT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(TEXT_FONT_SIZE * 1.3),
                TextColor(body_color),
                Anchor::TOP_LEFT,
                Transform::from_xyz(indent_x + 6.0, -(body_y + 4.0), 0.1),
            ));
        }
        y += BODY_ROW_H;

        // Field rows.
        let key_col_w = (inner_w * 0.32).clamp(60.0, 140.0);
        let value_col_w = inner_w - key_col_w - delete_w - 8.0;
        for (idx, (key, value)) in issue.fields.iter().enumerate() {
            let row_y = y + 2.0;
            let row_h = FIELD_ROW_H - 4.0;

            // Key
            if editing == Some(EditTarget::FieldKey(issue.id, idx)) {
                spawn_inline_input(
                    commands,
                    parent,
                    Vec2::new(indent_x, row_y),
                    Vec2::new(key_col_w, row_h),
                    input_bg,
                    key,
                    input_style.clone(),
                    focused,
                    pane,
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
                    LineHeight::Px(row_h),
                    TextColor(fg_muted),
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(indent_x, -row_y, 0.1),
                    IssueHit::FieldKey(issue.id, idx),
                    HitSize {
                        local_origin: Vec2::new(indent_x, row_y),
                        size: Vec2::new(key_col_w, row_h),
                    },
                ));
            }

            // Value
            let value_x = indent_x + key_col_w + 4.0;
            if editing == Some(EditTarget::FieldValue(issue.id, idx)) {
                spawn_inline_input(
                    commands,
                    parent,
                    Vec2::new(value_x, row_y),
                    Vec2::new(value_col_w, row_h),
                    input_bg,
                    value,
                    input_style.clone(),
                    focused,
                    pane,
                );
            } else {
                let display = if value.is_empty() {
                    "(empty)".to_string()
                } else {
                    value.clone()
                };
                let val_color = if value.is_empty() { fg_muted } else { fg };
                commands.spawn((
                    IssueRowEntity,
                    ChildOf(parent),
                    Text2d::new(display),
                    TextFont {
                        font: font.0.clone(),
                        font_size: SMALL_FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(row_h),
                    TextColor(val_color),
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(value_x, -row_y, 0.1),
                    IssueHit::FieldValue(issue.id, idx),
                    HitSize {
                        local_origin: Vec2::new(value_x, row_y),
                        size: Vec2::new(value_col_w, row_h),
                    },
                ));
            }

            // Field delete (×).
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
                LineHeight::Px(row_h),
                TextColor(fg_muted),
                Anchor::TOP_LEFT,
                Transform::from_xyz(fdel_x, -row_y, 0.2),
                IssueHit::DeleteField(issue.id, idx),
                HitSize {
                    local_origin: Vec2::new(fdel_x, row_y),
                    size: Vec2::new(delete_w, row_h),
                },
            ));

            y += FIELD_ROW_H;
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
) {
    // Background frame so the active edit slot reads as an input.
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
    ));

    let input = spawn_text_input(
        commands,
        parent,
        initial,
        style,
        size.x - 4.0,
        Transform::from_xyz(origin.x + 4.0, -(origin.y + 2.0), 0.2),
    );
    commands.entity(input).insert(IssueRowEntity);
    pane.edit_input = Some(input);
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
