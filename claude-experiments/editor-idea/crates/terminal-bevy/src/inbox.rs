//! Per-project inbox.
//!
//! Storage: one JSONL file per project at
//! `~/.terminal-bevy/inbox/<project_id>.jsonl`. Each line is an
//! [`InboxMessage`]. External tools (other projects via IPC, scripts,
//! CI hooks) append messages by calling [`append_message`]; the inbox
//! pane reads the file on a slow tick and rerenders when contents
//! change.
//!
//! The pane also lets the user flag a message for "send to Claude" —
//! that appends a copy of the message to
//! `~/.terminal-bevy/claude-outbox/<project_id>.jsonl`. A separate Bun
//! MCP channel server (under `tools/inbox-channel/`) tails that file
//! and pushes each new line into the user's Claude Code session as a
//! `<channel source="terminal-bevy">` notification. See the channels
//! reference at https://code.claude.com/docs/en/channels-reference.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextLayout};
use bevy::time::common_conditions::on_timer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use pane_bevy::{
    content_area, pt_to_content_local, PaneContentPressed, PaneFont, PaneHotZones, PaneKindSpec,
    PaneRect, PaneRegistry, PaneTitle,
};

use crate::projects::Projects;

const PANE_KIND: &str = "inbox";

const HEADER_H: f32 = 26.0;
const ROW_H: f32 = 26.0;
const EXPANDED_PAD: f32 = 8.0;
const ROW_PAD_X: f32 = 8.0;
const TEXT_FONT_SIZE: f32 = 13.0;
const SMALL_FONT_SIZE: f32 = 11.0;
const ACTION_BTN_H: f32 = 22.0;

// ---------- Data ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InboxMessage {
    pub id: u64,
    /// Unix milliseconds.
    pub ts: u64,
    #[serde(default)]
    pub sender: String,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub read: bool,
}

#[derive(Default)]
pub struct ProjectInbox {
    pub messages: Vec<InboxMessage>,
    /// Wall-clock metadata stamp from the last successful file read, used
    /// to detect external writes (other tools, the channel bridge).
    pub last_seen_mtime: Option<std::time::SystemTime>,
    pub expanded: HashSet<u64>,
}

#[derive(Resource, Default)]
pub struct InboxStore {
    pub by_project: HashMap<u64, ProjectInbox>,
    /// Projects we've at least tried to load this session.
    pub loaded: HashSet<u64>,
}

impl InboxStore {
    pub fn get(&self, project_id: u64) -> Option<&ProjectInbox> {
        self.by_project.get(&project_id)
    }

    pub fn ensure_loaded(&mut self, project_id: u64) {
        if self.loaded.contains(&project_id) {
            return;
        }
        let messages = read_messages(project_id);
        let mtime = inbox_path(project_id)
            .and_then(|p| fs::metadata(p).ok())
            .and_then(|m| m.modified().ok());
        self.by_project.insert(
            project_id,
            ProjectInbox {
                messages,
                last_seen_mtime: mtime,
                expanded: HashSet::new(),
            },
        );
        self.loaded.insert(project_id);
    }
}

fn data_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    Some(p)
}

fn inbox_dir() -> Option<PathBuf> {
    Some(data_root()?.join("inbox"))
}

fn claude_outbox_dir() -> Option<PathBuf> {
    Some(data_root()?.join("claude-outbox"))
}

pub fn inbox_path(project_id: u64) -> Option<PathBuf> {
    Some(inbox_dir()?.join(format!("{}.jsonl", project_id)))
}

pub fn claude_outbox_path(project_id: u64) -> Option<PathBuf> {
    Some(claude_outbox_dir()?.join(format!("{}.jsonl", project_id)))
}

fn read_messages(project_id: u64) -> Vec<InboxMessage> {
    let Some(path) = inbox_path(project_id) else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| match serde_json::from_str::<InboxMessage>(line) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("[inbox] parse {}: {} (line: {:?})", path.display(), e, line);
                None
            }
        })
        .collect()
}

fn write_messages(project_id: u64, messages: &[InboxMessage]) -> std::io::Result<()> {
    let Some(dir) = inbox_dir() else {
        return Ok(());
    };
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.jsonl", project_id));
    let tmp = dir.join(format!("{}.jsonl.tmp", project_id));
    let mut out = String::new();
    for m in messages {
        out.push_str(&serde_json::to_string(m).unwrap_or_default());
        out.push('\n');
    }
    fs::write(&tmp, out.as_bytes())?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

/// External entry point used by the IPC layer and other tools: append
/// a single message to a project's inbox. Returns the assigned id.
pub fn append_message(
    project_id: u64,
    sender: impl Into<String>,
    subject: Option<String>,
    body: impl Into<String>,
) -> std::io::Result<u64> {
    let Some(dir) = inbox_dir() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "no HOME env",
        ));
    };
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.jsonl", project_id));
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    // Use ts as id baseline + monotonic-bump if collision (rare).
    let id = ts;
    let msg = InboxMessage {
        id,
        ts,
        sender: sender.into(),
        subject,
        body: body.into(),
        read: false,
    };
    let line = serde_json::to_string(&msg).unwrap_or_default();
    let mut f = fs::OpenOptions::new().create(true).append(true).open(&path)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(id)
}

/// Forward a message into the project's Claude-channel outbox.
pub fn forward_to_claude(project_id: u64, msg: &InboxMessage) -> std::io::Result<()> {
    let Some(dir) = claude_outbox_dir() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "no HOME env",
        ));
    };
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.jsonl", project_id));
    let line = serde_json::to_string(msg).unwrap_or_default();
    let mut f = fs::OpenOptions::new().create(true).append(true).open(&path)?;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}

// ---------- Pane ----------

#[derive(Component)]
pub struct InboxPane {
    pub project_id: u64,
    pub dirty_layout: bool,
    /// When false (default) the pane lists only unread messages — read
    /// ones drop out of the list the moment they're marked read. Toggled
    /// on to reveal the full history.
    pub show_read: bool,
}

#[derive(Component, Copy, Clone, Debug)]
pub enum InboxHit {
    ExpandToggle(u64),
    MarkRead(u64),
    SendToClaude(u64),
    Delete(u64),
    MarkAllRead,
    /// Flip between unread-only (default) and full history.
    ToggleShowRead,
}

#[derive(Component, Copy, Clone, Debug)]
struct HitSize {
    local_origin: Vec2,
    size: Vec2,
}

#[derive(Component)]
struct InboxRowEntity;

pub struct InboxPanePlugin;

impl Plugin for InboxPanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InboxStore>()
            .add_systems(Startup, register_kind)
            .add_systems(
                Update,
                (
                    poll_disk.run_if(on_timer(std::time::Duration::from_millis(1000))),
                    handle_content_press,
                    rebuild_rows,
                    update_inbox_hot_zones,
                )
                    .chain(),
            );
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Inbox",
        radial_icon: Some("✉"),
        default_size: Vec2::new(520.0, 460.0),
        spawn: inbox_spawn,
        snapshot: inbox_snapshot,
        on_close: None,
    });
}

fn inbox_spawn(world: &mut World, entity: Entity, _content_root: Entity, config: &Value) {
    let project_id = config
        .get("project_id")
        .and_then(|v| v.as_u64())
        .or_else(|| world.resource::<Projects>().active)
        .unwrap_or(0);
    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        if t.0.is_empty() || t.0 == "Inbox" {
            t.0 = "Inbox".to_string();
        }
    }
    world.entity_mut(entity).insert(InboxPane {
        project_id,
        dirty_layout: true,
        show_read: false,
    });
    world.resource_mut::<InboxStore>().ensure_loaded(project_id);
}

fn inbox_snapshot(world: &World, entity: Entity) -> Value {
    let project_id = world
        .get::<InboxPane>(entity)
        .map(|p| p.project_id)
        .unwrap_or(0);
    serde_json::json!({ "project_id": project_id })
}

// ---------- Disk poll ----------

/// Periodically re-read each loaded project's inbox file. If mtime
/// changed (an external writer appended) we reload and mark all panes
/// for that project dirty.
fn poll_disk(
    mut store: ResMut<InboxStore>,
    mut panes: Query<&mut InboxPane>,
) {
    let pids: Vec<u64> = store.loaded.iter().copied().collect();
    for pid in pids {
        let Some(path) = inbox_path(pid) else { continue };
        let mtime = fs::metadata(&path).ok().and_then(|m| m.modified().ok());
        let prev = store.by_project.get(&pid).and_then(|i| i.last_seen_mtime);
        if mtime != prev {
            let messages = read_messages(pid);
            if let Some(inbox) = store.by_project.get_mut(&pid) {
                // Preserve expanded set for ids still present.
                let still: HashSet<u64> =
                    messages.iter().map(|m| m.id).collect();
                inbox.expanded.retain(|id| still.contains(id));
                inbox.messages = messages;
                inbox.last_seen_mtime = mtime;
            }
            for mut pane in &mut panes {
                if pane.project_id == pid {
                    pane.dirty_layout = true;
                }
            }
        }
    }
}

// ---------- Input ----------

fn handle_content_press(
    mut events: MessageReader<PaneContentPressed>,
    mut panes: Query<&mut InboxPane>,
    hits: Query<(&InboxHit, &HitSize)>,
    pane_rects: Query<&PaneRect>,
    mut store: ResMut<InboxStore>,
) {
    for ev in events.read() {
        let Ok(mut pane) = panes.get_mut(ev.pane) else {
            continue;
        };
        let Ok(_rect) = pane_rects.get(ev.pane) else {
            continue;
        };
        // `ev.local_pt` is already content-local in canvas-space;
        // recomputing from window_pt + canvas-space rect would
        // mis-hit the moment the canvas is panned/zoomed.
        let local = ev.local_pt;
        let mut picked: Option<InboxHit> = None;
        for (hit, size) in &hits {
            if local.x >= size.local_origin.x
                && local.x <= size.local_origin.x + size.size.x
                && local.y >= size.local_origin.y
                && local.y <= size.local_origin.y + size.size.y
            {
                picked = Some(*hit);
                break;
            }
        }
        let Some(hit) = picked else { continue };
        apply_hit(&mut pane, hit, &mut store);
    }
}

/// Mirror each inbox pane's `HitSize` children into `PaneHotZones` so
/// pinned-pane hit-testing can route clicks to inbox row buttons. The
/// content_root → pane map is rebuilt every frame — cheap because the
/// number of inbox panes is tiny, and avoids any stale-mapping risk
/// when the pane is closed and respawned.
fn update_inbox_hot_zones(
    panes: Query<(Entity, &pane_bevy::PaneChrome), With<InboxPane>>,
    mut zones_q: Query<&mut PaneHotZones>,
    hits: Query<(&HitSize, &ChildOf), With<InboxHit>>,
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

fn apply_hit(pane: &mut InboxPane, hit: InboxHit, store: &mut InboxStore) {
    let pid = pane.project_id;
    store.ensure_loaded(pid);
    let Some(inbox) = store.by_project.get_mut(&pid) else {
        return;
    };
    let mut writeback = false;
    match hit {
        InboxHit::ExpandToggle(id) => {
            if inbox.expanded.contains(&id) {
                inbox.expanded.remove(&id);
            } else {
                inbox.expanded.insert(id);
                // Opening counts as a read.
                if let Some(m) = inbox.messages.iter_mut().find(|m| m.id == id) {
                    if !m.read {
                        m.read = true;
                        writeback = true;
                    }
                }
            }
            pane.dirty_layout = true;
        }
        InboxHit::MarkRead(id) => {
            if let Some(m) = inbox.messages.iter_mut().find(|m| m.id == id) {
                m.read = !m.read;
                writeback = true;
                pane.dirty_layout = true;
            }
        }
        InboxHit::SendToClaude(id) => {
            if let Some(m) = inbox.messages.iter().find(|m| m.id == id).cloned() {
                if let Err(e) = forward_to_claude(pid, &m) {
                    eprintln!("[inbox] forward_to_claude: {}", e);
                }
                // Mark read on send.
                if let Some(mm) = inbox.messages.iter_mut().find(|m| m.id == id) {
                    if !mm.read {
                        mm.read = true;
                        writeback = true;
                    }
                }
                pane.dirty_layout = true;
            }
        }
        InboxHit::Delete(id) => {
            inbox.messages.retain(|m| m.id != id);
            inbox.expanded.remove(&id);
            writeback = true;
            pane.dirty_layout = true;
        }
        InboxHit::MarkAllRead => {
            for m in &mut inbox.messages {
                if !m.read {
                    m.read = true;
                    writeback = true;
                }
            }
            pane.dirty_layout = true;
        }
        InboxHit::ToggleShowRead => {
            pane.show_read = !pane.show_read;
            pane.dirty_layout = true;
        }
    }
    if writeback {
        let _ = write_messages(pid, &inbox.messages);
        // Refresh our own mtime stamp so poll_disk doesn't immediately
        // re-read what we just wrote.
        if let Some(path) = inbox_path(pid) {
            inbox.last_seen_mtime = fs::metadata(path).ok().and_then(|m| m.modified().ok());
        }
    }
}

// ---------- Render ----------

#[allow(clippy::too_many_arguments)]
fn rebuild_rows(
    mut commands: Commands,
    mut panes: Query<(&mut InboxPane, &PaneRect, &pane_bevy::PaneChrome)>,
    existing_rows: Query<(Entity, &ChildOf), With<InboxRowEntity>>,
    store: Res<InboxStore>,
    font: Res<PaneFont>,
    theme: Res<style_bevy::Theme>,
) {
    let theme_changed = theme.is_changed();
    if !theme_changed && panes.iter().all(|(p, _, _)| !p.dirty_layout) {
        return;
    }

    use style_bevy::tokens as t;
    let c = |id| Color::LinearRgba(theme.color(id));
    let fg = c(t::FG);
    let fg_muted = c(t::FG_MUTED);
    let accent = c(t::ACCENT);
    let divider = c(t::CHROME_DIVIDER);
    let unread_bg = c(t::SIDEBAR_ROW_ACTIVE_BG);
    let surface = c(t::PANE_BG);

    for (mut pane, rect, chrome) in &mut panes {
        if !pane.dirty_layout && !theme_changed {
            continue;
        }
        for (row, child_of) in &existing_rows {
            if child_of.0 == chrome.content_root {
                commands.entity(row).despawn();
            }
        }
        pane.dirty_layout = false;

        let (_origin, content_size) = content_area(rect);
        let content_w = content_size.x;
        if content_w <= 0.0 {
            continue;
        }
        let Some(inbox) = store.get(pane.project_id) else {
            continue;
        };

        let mut y = 0.0_f32;
        let show_read = pane.show_read;

        // Header: title + unread count + "mark all read" button.
        let unread = inbox.messages.iter().filter(|m| !m.read).count();
        let read_count = inbox.messages.len() - unread;
        let header_label = if unread == 0 {
            "Inbox · all read".to_string()
        } else {
            format!("Inbox · {} unread", unread)
        };
        commands.spawn((
            InboxRowEntity,
            ChildOf(chrome.content_root),
            Text2d::new(header_label),
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(HEADER_H),
            TextColor(fg_muted),
            Anchor::CENTER_LEFT,
            TextLayout::new_with_no_wrap(),
            pane_bevy::PaneContentNoClip,
            Transform::from_xyz(ROW_PAD_X, -(y + HEADER_H * 0.5), 0.1),
        ));
        // mark-all-read button on the right
        let mark_w = 110.0_f32.min(content_w * 0.4);
        let mark_x = content_w - ROW_PAD_X - mark_w;
        let mark_y = y + (HEADER_H - ACTION_BTN_H) * 0.5;
        commands.spawn((
            InboxRowEntity,
            ChildOf(chrome.content_root),
            Sprite {
                color: accent.with_alpha(0.12),
                custom_size: Some(Vec2::new(mark_w, ACTION_BTN_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(mark_x, -mark_y, 0.1),
            InboxHit::MarkAllRead,
            HitSize {
                local_origin: Vec2::new(mark_x, mark_y),
                size: Vec2::new(mark_w, ACTION_BTN_H),
            },
        ));
        commands.spawn((
            InboxRowEntity,
            ChildOf(chrome.content_root),
            Text2d::new("Mark all read"),
            TextFont {
                font: font.0.clone(),
                font_size: SMALL_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(ACTION_BTN_H),
            TextColor(accent),
            Anchor::CENTER,
            TextLayout::new_with_no_wrap(),
            pane_bevy::PaneContentNoClip,
            Transform::from_xyz(
                mark_x + mark_w * 0.5,
                -(mark_y + ACTION_BTN_H * 0.5),
                0.2,
            ),
        ));
        // Show/hide-read toggle, just left of "Mark all read". Only shown
        // when there's something to toggle (read messages exist, or we're
        // currently revealing them).
        if show_read || read_count > 0 {
            let toggle_label = if show_read {
                "Hide read".to_string()
            } else {
                format!("Show read ({})", read_count)
            };
            let toggle_w = 110.0_f32.min(content_w * 0.4);
            let toggle_x = mark_x - 8.0 - toggle_w;
            if toggle_x > ROW_PAD_X {
                commands.spawn((
                    InboxRowEntity,
                    ChildOf(chrome.content_root),
                    Sprite {
                        color: fg_muted.with_alpha(0.10),
                        custom_size: Some(Vec2::new(toggle_w, ACTION_BTN_H)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(toggle_x, -mark_y, 0.1),
                    InboxHit::ToggleShowRead,
                    HitSize {
                        local_origin: Vec2::new(toggle_x, mark_y),
                        size: Vec2::new(toggle_w, ACTION_BTN_H),
                    },
                ));
                commands.spawn((
                    InboxRowEntity,
                    ChildOf(chrome.content_root),
                    Text2d::new(toggle_label),
                    TextFont {
                        font: font.0.clone(),
                        font_size: SMALL_FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(ACTION_BTN_H),
                    TextColor(fg_muted),
                    Anchor::CENTER,
                    TextLayout::new_with_no_wrap(),
                    pane_bevy::PaneContentNoClip,
                    Transform::from_xyz(
                        toggle_x + toggle_w * 0.5,
                        -(mark_y + ACTION_BTN_H * 0.5),
                        0.2,
                    ),
                ));
            }
        }
        y += HEADER_H;
        commands.spawn((
            InboxRowEntity,
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

        // Unread-only by default; the user can reveal read messages via
        // the header toggle. A currently-expanded row stays visible even
        // once read, so expanding a message (which marks it read) doesn't
        // make it vanish under the user — it drops out when collapsed.
        // Newest-first.
        let mut rows: Vec<&InboxMessage> = inbox
            .messages
            .iter()
            .filter(|m| show_read || !m.read || inbox.expanded.contains(&m.id))
            .collect();
        rows.sort_by(|a, b| b.ts.cmp(&a.ts));

        if rows.is_empty() {
            let empty_label = if inbox.messages.is_empty() {
                "(empty — POST to the inbox to see messages here)"
            } else {
                // Messages exist, just none unread and not showing read.
                "(all caught up — nothing unread)"
            };
            commands.spawn((
                InboxRowEntity,
                ChildOf(chrome.content_root),
                Text2d::new(empty_label),
                TextFont {
                    font: font.0.clone(),
                    font_size: SMALL_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(ROW_H),
                TextColor(fg_muted),
                Anchor::CENTER_LEFT,
                Transform::from_xyz(ROW_PAD_X, -(y + ROW_H * 0.5), 0.1),
            ));
            continue;
        }

        for m in rows {
            let expanded = inbox.expanded.contains(&m.id);

            // Row bg — unread is highlighted.
            if !m.read {
                commands.spawn((
                    InboxRowEntity,
                    ChildOf(chrome.content_root),
                    Sprite {
                        color: unread_bg,
                        custom_size: Some(Vec2::new(content_w, ROW_H)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(0.0, -y, 0.02),
                ));
            }

            // Click target spanning the row top (toggles expand).
            commands.spawn((
                InboxRowEntity,
                ChildOf(chrome.content_root),
                Sprite {
                    color: surface.with_alpha(0.0),
                    custom_size: Some(Vec2::new(content_w, ROW_H)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(0.0, -y, 0.01),
                InboxHit::ExpandToggle(m.id),
                HitSize {
                    local_origin: Vec2::new(0.0, y),
                    size: Vec2::new(content_w, ROW_H),
                },
            ));

            // Chevron.
            commands.spawn((
                InboxRowEntity,
                ChildOf(chrome.content_root),
                Text2d::new(if expanded { "▾" } else { "▸" }),
                TextFont {
                    font: font.0.clone(),
                    font_size: 12.0,
                    ..default()
                },
                LineHeight::Px(ROW_H),
                TextColor(fg_muted),
                Anchor::CENTER_LEFT,
                Transform::from_xyz(ROW_PAD_X, -(y + ROW_H * 0.5), 0.1),
            ));

            // Sender + subject summary.
            let summary = format_summary(m);
            let title_color = if m.read { fg_muted } else { fg };
            commands.spawn((
                InboxRowEntity,
                ChildOf(chrome.content_root),
                Text2d::new(summary),
                TextFont {
                    font: font.0.clone(),
                    font_size: TEXT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(ROW_H),
                TextColor(title_color),
                Anchor::CENTER_LEFT,
                TextLayout::new_with_no_wrap(),
                pane_bevy::PaneContentNoClip,
                Transform::from_xyz(ROW_PAD_X + 18.0, -(y + ROW_H * 0.5), 0.1),
            ));

            // Right-edge timestamp.
            let ts_text = format_ts(m.ts);
            commands.spawn((
                InboxRowEntity,
                ChildOf(chrome.content_root),
                Text2d::new(ts_text),
                TextFont {
                    font: font.0.clone(),
                    font_size: SMALL_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(ROW_H),
                TextColor(fg_muted),
                Anchor::CENTER_RIGHT,
                TextLayout::new_with_no_wrap(),
                pane_bevy::PaneContentNoClip,
                Transform::from_xyz(content_w - ROW_PAD_X, -(y + ROW_H * 0.5), 0.1),
            ));

            y += ROW_H;

            if expanded {
                // Body block.
                let body_x = ROW_PAD_X + 18.0;
                let body_w = content_w - body_x - ROW_PAD_X;
                let body_text = if m.body.is_empty() {
                    "(no body)".to_string()
                } else {
                    m.body.clone()
                };
                // Approximate body height: 1 line of body + actions row.
                let body_h = (TEXT_FONT_SIZE * 1.3 * body_line_count(&body_text) as f32)
                    .max(TEXT_FONT_SIZE * 1.3);
                commands.spawn((
                    InboxRowEntity,
                    ChildOf(chrome.content_root),
                    Text2d::new(body_text),
                    TextFont {
                        font: font.0.clone(),
                        font_size: TEXT_FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(TEXT_FONT_SIZE * 1.3),
                    TextColor(fg),
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(body_x, -(y + EXPANDED_PAD * 0.5), 0.1),
                ));
                y += body_h + EXPANDED_PAD;

                // Actions row.
                let actions_y = y;
                let btn_h = ACTION_BTN_H;
                let mut bx = body_x;
                let mk_btn =
                    |commands: &mut Commands,
                     parent: Entity,
                     x: f32,
                     y: f32,
                     w: f32,
                     label: &str,
                     hit: InboxHit,
                     color: Color| {
                        commands.spawn((
                            InboxRowEntity,
                            ChildOf(parent),
                            Sprite {
                                color: color.with_alpha(0.14),
                                custom_size: Some(Vec2::new(w, btn_h)),
                                ..default()
                            },
                            Anchor::TOP_LEFT,
                            Transform::from_xyz(x, -y, 0.1),
                            hit,
                            HitSize {
                                local_origin: Vec2::new(x, y),
                                size: Vec2::new(w, btn_h),
                            },
                        ));
                        commands.spawn((
                            InboxRowEntity,
                            ChildOf(parent),
                            Text2d::new(label.to_string()),
                            TextFont {
                                font: font.0.clone(),
                                font_size: SMALL_FONT_SIZE,
                                ..default()
                            },
                            LineHeight::Px(btn_h),
                            TextColor(color),
                            Anchor::CENTER,
                            TextLayout::new_with_no_wrap(),
                            pane_bevy::PaneContentNoClip,
                            Transform::from_xyz(x + w * 0.5, -(y + btn_h * 0.5), 0.2),
                        ));
                    };
                // Send to Claude
                mk_btn(
                    &mut commands,
                    chrome.content_root,
                    bx,
                    actions_y,
                    140.0,
                    "Send to Claude",
                    InboxHit::SendToClaude(m.id),
                    accent,
                );
                bx += 140.0 + 6.0;
                // Mark read/unread
                mk_btn(
                    &mut commands,
                    chrome.content_root,
                    bx,
                    actions_y,
                    100.0,
                    if m.read { "Mark unread" } else { "Mark read" },
                    InboxHit::MarkRead(m.id),
                    fg_muted,
                );
                bx += 100.0 + 6.0;
                // Delete
                mk_btn(
                    &mut commands,
                    chrome.content_root,
                    bx,
                    actions_y,
                    70.0,
                    "Delete",
                    InboxHit::Delete(m.id),
                    fg_muted,
                );
                y += btn_h + EXPANDED_PAD;
            }

            // Divider between messages.
            commands.spawn((
                InboxRowEntity,
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
        }
    }
}

fn format_summary(m: &InboxMessage) -> String {
    let sender = if m.sender.is_empty() {
        "—".to_string()
    } else {
        m.sender.clone()
    };
    match &m.subject {
        Some(s) if !s.is_empty() => format!("{} · {}", sender, s),
        _ => {
            // First line of body, truncated.
            let first_line = m.body.lines().next().unwrap_or("").trim();
            if first_line.is_empty() {
                sender
            } else {
                format!("{} · {}", sender, truncate(first_line, 80))
            }
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

fn body_line_count(s: &str) -> usize {
    s.lines().count().max(1)
}

fn format_ts(ts_millis: u64) -> String {
    // Show HH:MM if today, else short date. Use chrono if available?
    // Stick to local time via SystemTime → chrono-less formatting:
    // just show seconds since epoch reduced to mod-day. Keep it minimal.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let delta = now.saturating_sub(ts_millis);
    let secs = delta / 1000;
    if secs < 60 {
        format!("{}s ago", secs)
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

// Silence the MouseButton import in case the click handler later
// needs it; keep here so the existing pattern across pane kinds
// matches.
#[allow(dead_code)]
fn _witness(_b: Res<ButtonInput<MouseButton>>) {}
