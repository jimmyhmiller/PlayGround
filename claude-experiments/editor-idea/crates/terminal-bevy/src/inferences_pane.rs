//! "Inferences" pane — live tail of the inference layer.
//!
//! Spawnable from the radial menu. Filters the same bus stream that
//! the [`claude_events_pane`] watches, but keeps only the two event
//! kinds the inference layer cares about today:
//!
//! - `terminal.cwd_changed` — the trigger; shown as `→ <cwd>`
//! - `inference.project_default_cwd_suggested` — the verdict; shown
//!   as `✓` (good_default true) or `✗` (false) followed by confidence,
//!   project name, and one-sentence reason.
//!
//! The point of this pane is mainly *visibility while we tune the
//! classifier*: are we firing on the right triggers, what does the
//! model actually say, how often does it disagree with itself across
//! similar paths. It is intentionally read-only and adds no new state
//! beyond a per-pane rolling buffer; the bus log is still the
//! authoritative record.
//!
//! Wired in alongside `ClaudeEventsPanePlugin` from [`TerminalPlugin`].

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextLayout};
use serde_json::Value;

use pane_bevy::{PaneFont, PaneKindSpec, PaneRegistry, PaneTitle};

use claude_bus_bevy::ClaudeBusEvent;

const PANE_KIND: &str = "inferences";
const TEXT_FONT_SIZE: f32 = 12.0;
const TEXT_LINE_HEIGHT: f32 = TEXT_FONT_SIZE * 1.35;
/// Soft cap on retained lines per pane.
const MAX_LINES_PER_PANE: usize = 400;
const TEXT_INNER_PAD_X: f32 = 8.0;
const TEXT_INNER_PAD_Y: f32 = 6.0;

#[derive(Component)]
pub struct InferencesPane {
    pub lines: VecDeque<String>,
    pub text_entity: Entity,
    pub dirty: bool,
}

pub struct InferencesPanePlugin;

impl Plugin for InferencesPanePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register_kind)
            .add_systems(Update, (pump_into_panes, sync_visuals).chain());
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Inferences",
        // Eye glyph — "what the editor is noticing about you".
        radial_icon: Some("◉"),
        default_size: Vec2::new(700.0, 360.0),
        spawn: spawn,
        snapshot: snapshot,
        on_close: None,
    });
}

fn spawn(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    let font = world
        .get_resource::<PaneFont>()
        .expect("PaneFont resource must be present before spawning an inferences pane")
        .0
        .clone();

    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Inferences")
        .to_string();
    let text_color = world
        .get_resource::<style_bevy::Theme>()
        .map(|t| Color::LinearRgba(t.color(style_bevy::tokens::FG)))
        .unwrap_or(Color::srgb(0.85, 0.87, 0.9));

    let text_entity = world
        .spawn((
            ChildOf(content_root),
            Text2d::new(String::new()),
            TextFont {
                font,
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(TEXT_LINE_HEIGHT),
            TextColor(text_color),
            Anchor::TOP_LEFT,
            TextLayout::new_with_no_wrap(),
            Transform::from_xyz(TEXT_INNER_PAD_X, -TEXT_INNER_PAD_Y, 0.0),
        ))
        .id();

    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title;
    }

    world.entity_mut(entity).insert(InferencesPane {
        lines: VecDeque::new(),
        text_entity,
        dirty: false,
    });
}

fn snapshot(world: &World, entity: Entity) -> Value {
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    serde_json::json!({ "title": title })
}

fn pump_into_panes(
    mut events: MessageReader<ClaudeBusEvent>,
    mut panes: Query<&mut InferencesPane>,
) {
    // Filter + format in one pass so we don't materialize lines for
    // events the user can't see anyway.
    let mut new_lines: Vec<String> = Vec::new();
    for ev in events.read() {
        match ev.kind.as_str() {
            "terminal.cwd_changed" => {
                if let Some(line) = format_cwd_changed(ev) {
                    new_lines.push(line);
                }
            }
            "inference.project_default_cwd_suggested" => {
                if let Some(line) = format_suggestion(ev) {
                    new_lines.push(line);
                }
            }
            _ => {}
        }
    }
    if new_lines.is_empty() {
        return;
    }
    for mut pane in &mut panes {
        for line in &new_lines {
            pane.lines.push_back(line.clone());
        }
        while pane.lines.len() > MAX_LINES_PER_PANE {
            pane.lines.pop_front();
        }
        pane.dirty = true;
    }
}

fn sync_visuals(
    mut panes: Query<&mut InferencesPane>,
    mut texts: Query<&mut Text2d>,
) {
    for mut pane in &mut panes {
        if !pane.dirty {
            continue;
        }
        let combined: String = pane
            .lines
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if let Ok(mut t) = texts.get_mut(pane.text_entity) {
            t.0 = combined;
        }
        pane.dirty = false;
    }
}

fn format_cwd_changed(ev: &ClaudeBusEvent) -> Option<String> {
    let v: Value = serde_json::from_str(&ev.payload_json).ok()?;
    let cwd = v.get("cwd").and_then(|c| c.as_str())?;
    Some(format!(
        "{}  →  T{}  cd {}",
        format_time(ev.ts),
        ev.terminal_session_id,
        shorten_home(cwd),
    ))
}

fn format_suggestion(ev: &ClaudeBusEvent) -> Option<String> {
    let v: Value = serde_json::from_str(&ev.payload_json).ok()?;
    let good = v.get("good_default").and_then(|g| g.as_bool()).unwrap_or(false);
    let conf = v.get("confidence").and_then(|c| c.as_f64()).unwrap_or(0.0);
    let project = v
        .get("project_name")
        .and_then(|p| p.as_str())
        .unwrap_or("?");
    let cwd = v.get("cwd").and_then(|c| c.as_str()).unwrap_or("?");
    let reason = v
        .get("reason")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .trim();
    let glyph = if good { "✓" } else { "✗" };
    let conf_pct = (conf * 100.0).round() as i32;
    // Two-line: verdict header + indented reason. Easier to scan than
    // a single ultra-wide line.
    let header = format!(
        "{}  {}  T{}  {:>3}%  {}  ({})",
        format_time(ev.ts),
        glyph,
        ev.terminal_session_id,
        conf_pct,
        project,
        shorten_home(cwd),
    );
    if reason.is_empty() {
        Some(header)
    } else {
        Some(format!("{}\n     {}", header, trunc(reason, 100)))
    }
}

/// Replace the leading $HOME with `~` for readability. Falls back to
/// the raw path if HOME isn't set.
fn shorten_home(p: &str) -> String {
    if let Some(h) = std::env::var_os("HOME").and_then(|h| h.into_string().ok()) {
        if let Some(rest) = p.strip_prefix(&h) {
            if rest.is_empty() {
                return "~".into();
            }
            return format!("~{}", rest);
        }
    }
    p.to_string()
}

/// `HH:MM:SS` local-ish time derived from the Unix ts. We don't pull
/// in `chrono` for this — a u64 modulo gives the user a fine-grained
/// sense of "when" without claiming to be wall-clock accurate to the
/// minute (the bus timestamp is seconds since epoch, but we don't
/// know the user's timezone offset here so this is UTC).
fn format_time(ts: u64) -> String {
    let secs = ts % 86_400;
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut end = n;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}
