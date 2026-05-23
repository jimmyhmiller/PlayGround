//! "Claude Events" pane — live tail of the claude-bus.
//!
//! Spawnable from the radial menu; appears as a scrolling text pane
//! that grows downward as new events arrive. Each pane keeps its own
//! ring buffer so multiple Events panes can coexist (e.g. one
//! per-project filtering by `session_id`). For now there's no
//! filtering UI — every pane sees every event.
//!
//! The pane is intentionally read-only and unstyled to start: one
//! event per line, monospace, oldest at the top, newest at the
//! bottom. The hard parts (subscription, reconnect, replay) are
//! delegated to `claude-bus-bevy`'s `BusEventPlugin`, which the host
//! app installs once.

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextLayout};
use serde_json::Value;

use pane_bevy::{PaneFont, PaneKindSpec, PaneRegistry, PaneTitle};

use claude_bus_bevy::ClaudeBusEvent;

const PANE_KIND: &str = "claude_events";
const TEXT_FONT_SIZE: f32 = 12.0;
const TEXT_LINE_HEIGHT: f32 = TEXT_FONT_SIZE * 1.3;
/// Soft cap on retained lines per pane. Past this we trim from the
/// front so the Text2d node doesn't accumulate forever.
const MAX_LINES_PER_PANE: usize = 500;
const TEXT_INNER_PAD_X: f32 = 8.0;
const TEXT_INNER_PAD_Y: f32 = 6.0;
const COLOR_TEXT: Color = Color::srgb(0.85, 0.87, 0.9);

/// Per-pane state. Lives on the pane entity (the same one PaneTag is
/// on). `text_entity` is the child Text2d we mutate as events stream
/// in. `lines` is the rolling buffer.
#[derive(Component)]
pub struct ClaudeEventsPane {
    pub lines: VecDeque<String>,
    pub text_entity: Entity,
    /// Set whenever `lines` mutated; the visual-sync system clears it
    /// after updating Text2d. Lets us avoid rewriting the whole Text2d
    /// every frame when nothing changed.
    pub dirty: bool,
}

pub struct ClaudeEventsPanePlugin;

impl Plugin for ClaudeEventsPanePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register_kind).add_systems(
            Update,
            // Append-then-sync: pump_into_panes mutates `lines`; the
            // sync system rewrites Text2d.dirty only when needed.
            (pump_into_panes, sync_visuals).chain(),
        );
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Claude Events",
        radial_icon: Some("✦"),
        default_size: Vec2::new(620.0, 380.0),
        spawn: claude_events_spawn,
        snapshot: claude_events_snapshot,
        on_close: None,
    });
}

/// Spawn callback invoked by `pane-bevy` when this kind is created
/// (from the radial menu, the restore loop, or any other path). The
/// chrome (bg, title, resize handle) is already in place; we just add
/// the Text2d body and attach our component.
fn claude_events_spawn(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    let font = world
        .get_resource::<PaneFont>()
        .expect("PaneFont resource must be present before spawning a claude-events pane")
        .0
        .clone();

    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Claude Events")
        .to_string();

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
            TextColor(COLOR_TEXT),
            Anchor::TOP_LEFT,
            // No-wrap so seq + kind columns stay aligned. Right-overflow
            // is invisibly clipped by the per-pane camera viewport
            // (see pane-bevy's top-of-file docs) — no per-line
            // truncation needed here.
            TextLayout::new_with_no_wrap(),
            Transform::from_xyz(TEXT_INNER_PAD_X, -TEXT_INNER_PAD_Y, 0.0),
        ))
        .id();

    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title;
    }

    world.entity_mut(entity).insert(ClaudeEventsPane {
        lines: VecDeque::new(),
        text_entity,
        dirty: false,
    });
}

/// Snapshot is intentionally tiny — we don't persist the rolling
/// buffer (events are reproducible from the bus + JSONL anyway). Just
/// the title, so a restart keeps the user's rename.
fn claude_events_snapshot(world: &World, entity: Entity) -> Value {
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    serde_json::json!({ "title": title })
}

/// Collect all events received this frame, then append a formatted
/// line to every Claude Events pane. We materialize into a `Vec` first
/// because Bevy's `MessageReader` is a one-shot stream; once read by
/// this system, the next pane query wouldn't see them.
fn pump_into_panes(
    mut events: MessageReader<ClaudeBusEvent>,
    mut panes: Query<&mut ClaudeEventsPane>,
) {
    let new_lines: Vec<String> = events.read().map(format_event_line).collect();
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

/// Rewrite each dirty pane's Text2d in one go. Done as a separate
/// system because `Text2d` lives on a child entity (not the pane
/// entity), and Bevy's borrow rules make it cleaner to do it after
/// mutating the buffer.
fn sync_visuals(
    mut panes: Query<&mut ClaudeEventsPane>,
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

/// One event → one line. Aim for "scannable at a glance" rather than
/// "full structural detail": seq, kind, session id (truncated), then a
/// payload hint that varies by kind.
fn format_event_line(ev: &ClaudeBusEvent) -> String {
    let sess = if ev.terminal_session_id.is_empty() {
        "—".to_string()
    } else {
        format!("T{}", ev.terminal_session_id)
    };
    let hint = payload_hint(&ev.kind, &ev.payload_json);
    format!(
        "{:>5} {:<18} {:<4} {}",
        ev.seq, ev.kind, sess, hint
    )
}

/// Per-kind one-liner extracted from the payload. Parses on demand —
/// we don't pay JSON cost for events that don't show a hint. Falls
/// back to a short slice of the raw payload so unknown kinds still say
/// something useful.
fn payload_hint(kind: &str, payload_json: &str) -> String {
    let v: Value = match serde_json::from_str(payload_json) {
        Ok(v) => v,
        Err(_) => return trunc(payload_json, 80),
    };
    match kind {
        "pre_tool_use" | "post_tool_use" => {
            let tool = v.get("tool_name").and_then(|t| t.as_str()).unwrap_or("?");
            format!("{}", tool)
        }
        "user_prompt_submit" => v
            .get("prompt")
            .and_then(|p| p.as_str())
            .map(|s| trunc(s.trim(), 80))
            .unwrap_or_else(|| "—".into()),
        "notification" => v
            .get("message")
            .and_then(|m| m.as_str())
            .map(|s| trunc(s, 80))
            .unwrap_or_else(|| "—".into()),
        "session_start" => v
            .get("model")
            .and_then(|m| m.get("id"))
            .and_then(|i| i.as_str())
            .unwrap_or("—")
            .into(),
        "session_end" => v
            .get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("—")
            .into(),
        "stop" => v
            .get("last_assistant_message")
            .and_then(|m| m.as_str())
            .map(|s| trunc(s.trim(), 80))
            .unwrap_or_else(|| "—".into()),
        _ => trunc(payload_json, 80),
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        // Char-boundary aware truncation — Rust will panic on a
        // multi-byte slice otherwise.
        let mut end = n;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}
