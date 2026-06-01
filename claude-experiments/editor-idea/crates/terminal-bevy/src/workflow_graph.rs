//! Live workflow-execution graph pane.
//!
//! Watches a Claude Code workflow run directory:
//! `…/<project>/<session>/subagents/workflows/<runId>/` (or any session's
//! `subagents/` dir for plain Agent-tool subagents) and renders the
//! agents as a live node/edge graph via the generic [`graph_view`].
//!
//! Data sources (all append-only / written live, so we poll cheaply on a
//! background thread and push snapshots into the ECS over a channel):
//!   - `journal.jsonl`        — `{type:"started"|"result", agentId, ...}`
//!   - `agent-<id>.meta.json` — `{agentType}` (+ file mtime ≈ start time)
//!   - `agent-<id>.jsonl`     — per-agent transcript (first user msg = label)
//!
//! Structure: agents are clustered into "waves" by start time (a wave =
//! a parallel batch / pipeline stage) and laid out as left-to-right
//! columns. Edges connect consecutive waves. The live journal does not
//! encode parent/phase edges (its `key` is a content hash), so the wave
//! clustering is the faithful structure we can reconstruct live; the
//! exact phase grouping sharpens in the final consolidated run json.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use bevy::prelude::*;
use pane_bevy::{PaneKindMarker, PaneKindSpec, PaneRect, PaneRegistry, PaneTitle};
use serde_json::Value;

use crate::graph_view::{self, GEdge, GNode, GraphModel, GraphPalette, GraphView, NodeStatus};
use crate::{MonoFont, FONT_SIZE};

pub const PANE_KIND: &str = "workflow-graph";

/// Gap (seconds) between successive agent start times that starts a new
/// wave/column. Agents that start closer than this are treated as one
/// parallel batch.
const WAVE_GAP_SECS: f64 = 10.0;

const POLL_INTERVAL: Duration = Duration::from_millis(700);

// ---------- Plugin / registry ----------

pub struct WorkflowGraphPlugin;

impl Plugin for WorkflowGraphPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register_kind)
            .add_systems(Update, drain_and_render);
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Workflow",
        radial_icon: Some("\u{2387}"), // ⎇ branch glyph
        default_size: Vec2::new(900.0, 620.0),
        spawn: spawn_from_config,
        snapshot,
        on_close: Some(on_close),
    });
}

// ---------- Component ----------

#[derive(Component)]
pub struct WorkflowGraphPane {
    run_dir: PathBuf,
    content_root: Entity,
    rx: Mutex<Receiver<GraphModel>>,
    stop: Arc<AtomicBool>,
    model: GraphModel,
    dirty: bool,
    last_size: Vec2,
}

// ---------- Spawn / snapshot / close ----------

fn spawn_from_config(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    // The CLI overloads `command` as the run-dir path; `run_dir` is the
    // explicit key used on snapshot/restore.
    let run_dir = config
        .get("run_dir")
        .or_else(|| config.get("command"))
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_default();

    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Workflow")
        .to_string();
    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title;
    }

    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<GraphModel>();

    if run_dir.as_os_str().is_empty() {
        // No directory to watch — surface it clearly rather than sit blank.
        let _ = tx.send(GraphModel {
            caption: "workflow-graph: no run_dir configured".into(),
            ..default()
        });
    } else {
        let dir = run_dir.clone();
        let stop_thread = stop.clone();
        std::thread::Builder::new()
            .name("wf-graph-watch".into())
            .spawn(move || watch_loop(dir, tx, stop_thread))
            .expect("spawn wf-graph-watch thread");
    }

    world.entity_mut(entity).insert((
        WorkflowGraphPane {
            run_dir,
            content_root,
            rx: Mutex::new(rx),
            stop,
            model: GraphModel::default(),
            dirty: true,
            last_size: Vec2::ZERO,
        },
        GraphView::default(),
    ));
}

fn snapshot(world: &World, entity: Entity) -> Value {
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    let run_dir = world
        .get::<WorkflowGraphPane>(entity)
        .map(|p| p.run_dir.to_string_lossy().into_owned())
        .unwrap_or_default();
    serde_json::json!({ "title": title, "run_dir": run_dir })
}

fn on_close(world: &mut World, entity: Entity) {
    if let Some(p) = world.get::<WorkflowGraphPane>(entity) {
        p.stop.store(true, Ordering::Relaxed);
    }
}

// ---------- Render system ----------

fn drain_and_render(
    mut commands: Commands,
    theme: Res<style_bevy::Theme>,
    pane_font: Option<Res<pane_bevy::PaneFont>>,
    mono_font: Option<Res<MonoFont>>,
    mut q: Query<(
        &PaneKindMarker,
        &PaneRect,
        &mut WorkflowGraphPane,
        &mut GraphView,
    )>,
) {
    let Some(font) = pane_font
        .map(|f| f.0.clone())
        .or_else(|| mono_font.map(|f| f.0.clone()))
    else {
        return;
    };
    let palette = palette(&theme);

    for (kind, rect, mut pane, mut view) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }

        // Drain to the most recent snapshot from the watcher thread,
        // releasing the lock before we touch the rest of `pane`.
        let latest = {
            let mut latest = None;
            if let Ok(rx) = pane.rx.lock() {
                while let Ok(m) = rx.try_recv() {
                    latest = Some(m);
                }
            }
            latest
        };
        if let Some(m) = latest {
            if m != pane.model {
                pane.model = m;
                pane.dirty = true;
            }
        }

        let content_size = Vec2::new(
            (rect.size.x - 2.0 * pane_bevy::MARGIN).max(0.0),
            (rect.size.y - pane_bevy::TITLE_H - 2.0 * pane_bevy::MARGIN).max(0.0),
        );
        let resized = (content_size - pane.last_size).length() > 0.5;

        if !pane.dirty && !resized {
            continue;
        }
        pane.last_size = content_size;
        pane.dirty = false;

        let content_root = pane.content_root;
        let model = pane.model.clone();
        graph_view::render(
            &mut commands,
            content_root,
            &font,
            &palette,
            content_size,
            &model,
            &mut view,
        );
    }
}

// ---------- Palette ----------

fn palette(theme: &style_bevy::Theme) -> GraphPalette {
    use style_bevy::tokens as t;
    let c = |id| Color::LinearRgba(theme.color(id));
    GraphPalette {
        node_bg: c(t::SURFACE_2),
        label: c(t::FG),
        sublabel: c(t::FG_MUTED),
        edge: c(t::CHROME_DIVIDER),
        caption: c(t::FG_MUTED),
        pending: c(t::STATUS_IDLE),
        running: c(t::STATUS_RUNNING),
        done: c(t::STATUS_SUCCESS),
        failed: c(t::STATUS_FAILED),
    }
}

// ---------- Background watcher ----------

fn watch_loop(dir: PathBuf, tx: mpsc::Sender<GraphModel>, stop: Arc<AtomicBool>) {
    let mut last: Option<GraphModel> = None;
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let model = build_model(&dir);
        if last.as_ref() != Some(&model) {
            if tx.send(model.clone()).is_err() {
                return; // pane gone
            }
            last = Some(model);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

#[derive(Default)]
struct AgentRow {
    label: String,
    agent_type: String,
    start: f64,
    started: bool,
    has_result: bool,
    failed: bool,
}

/// Read the run directory and assemble a [`GraphModel`].
fn build_model(dir: &Path) -> GraphModel {
    let mut rows: HashMap<String, AgentRow> = HashMap::new();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return GraphModel {
            caption: format!("workflow-graph: cannot read {}", dir.display()),
            ..default()
        };
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(id) = name
            .strip_prefix("agent-")
            .and_then(|s| s.strip_suffix(".meta.json"))
        {
            let row = rows.entry(id.to_string()).or_default();
            if let Ok(txt) = std::fs::read_to_string(entry.path()) {
                if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                    row.agent_type = v
                        .get("agentType")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                }
            }
            row.start = mtime_secs(&entry.path());
        } else if let Some(id) = name
            .strip_prefix("agent-")
            .and_then(|s| s.strip_suffix(".jsonl"))
        {
            let path = entry.path();
            let label = agent_label(&path).unwrap_or_else(|| short_id(id));
            let row = rows.entry(id.to_string()).or_default();
            row.label = label;
            if row.start == 0.0 {
                row.start = mtime_secs(&path);
            }
        }
    }

    // Journal supplies authoritative started/result lifecycle.
    let journal = dir.join("journal.jsonl");
    if let Ok(file) = std::fs::File::open(&journal) {
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let Some(id) = v.get("agentId").and_then(|x| x.as_str()) else {
                continue;
            };
            let row = rows.entry(id.to_string()).or_default();
            match v.get("type").and_then(|x| x.as_str()) {
                Some("started") => row.started = true,
                Some("result") => {
                    row.has_result = true;
                    row.failed = result_is_failure(v.get("result"));
                }
                _ => {}
            }
            if row.label.is_empty() {
                row.label = short_id(id);
            }
        }
    }

    if rows.is_empty() {
        return GraphModel {
            caption: format!(
                "workflow-graph: watching {} (no agents yet)",
                dir.file_name().map(|s| s.to_string_lossy()).unwrap_or_default()
            ),
            ..default()
        };
    }

    // Order by start time; cluster into waves (columns).
    let mut ids: Vec<String> = rows.keys().cloned().collect();
    ids.sort_by(|a, b| {
        rows[a]
            .start
            .partial_cmp(&rows[b].start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });

    let mut nodes: Vec<GNode> = Vec::new();
    let mut cols: Vec<Vec<String>> = Vec::new(); // ids per column
    let mut col: u32 = 0;
    let mut row_in_col: u32 = 0;
    let mut prev_start: Option<f64> = None;
    let (mut n_done, mut n_running, mut n_failed) = (0u32, 0u32, 0u32);

    for id in &ids {
        let r = &rows[id];
        let new_wave = match prev_start {
            Some(p) => r.start - p > WAVE_GAP_SECS,
            None => false,
        };
        if new_wave {
            col += 1;
            row_in_col = 0;
        }
        prev_start = Some(r.start);

        let status = if r.failed {
            n_failed += 1;
            NodeStatus::Failed
        } else if r.has_result {
            n_done += 1;
            NodeStatus::Done
        } else if r.started {
            n_running += 1;
            NodeStatus::Running
        } else {
            NodeStatus::Pending
        };

        let status_word = match status {
            NodeStatus::Pending => "pending",
            NodeStatus::Running => "running\u{2026}",
            NodeStatus::Done => "done",
            NodeStatus::Failed => "failed",
        };
        let sublabel = if r.agent_type.is_empty() {
            format!("{} · {}", short_id(id), status_word)
        } else {
            format!("{} · {}", r.agent_type, status_word)
        };

        while cols.len() <= col as usize {
            cols.push(Vec::new());
        }
        cols[col as usize].push(id.clone());

        nodes.push(GNode {
            id: id.clone(),
            label: r.label.clone(),
            sublabel,
            status,
            col,
            row: row_in_col,
        });
        row_in_col += 1;
    }

    // Edges between consecutive waves.
    let mut edges: Vec<GEdge> = Vec::new();
    for w in 1..cols.len() {
        let prev = &cols[w - 1];
        let next = &cols[w];
        if prev.len() * next.len() <= 16 {
            for f in prev {
                for t in next {
                    edges.push(GEdge {
                        from: f.clone(),
                        to: t.clone(),
                    });
                }
            }
        } else if let Some(hub) = prev.first() {
            // Avoid an unreadable mesh: fan from the wave's first node.
            for t in next {
                edges.push(GEdge {
                    from: hub.clone(),
                    to: t.clone(),
                });
            }
        }
    }

    let run_name = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let caption = format!(
        "{run_name}  ·  {total} agents  ·  {n_done} done  ·  {n_running} running{failed}",
        total = nodes.len(),
        failed = if n_failed > 0 {
            format!("  ·  {n_failed} failed")
        } else {
            String::new()
        },
    );

    GraphModel {
        nodes,
        edges,
        caption,
    }
}

fn result_is_failure(result: Option<&Value>) -> bool {
    match result {
        Some(Value::Object(m)) => m.get("ok").and_then(|v| v.as_bool()) == Some(false),
        Some(Value::String(s)) => {
            let s = s.to_ascii_lowercase();
            s.contains("panic") || s.starts_with("error") || s.contains("failed")
        }
        _ => false,
    }
}

/// First user-message text in an agent transcript, first line, trimmed
/// and truncated for use as a node label.
fn agent_label(path: &Path) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok).take(40) {
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if v.get("type").and_then(|x| x.as_str()) != Some("user") {
            continue;
        }
        let content = v.get("message").and_then(|m| m.get("content"))?;
        let text = match content {
            Value::String(s) => Some(s.clone()),
            Value::Array(arr) => arr.iter().find_map(|c| {
                if c.get("type").and_then(|x| x.as_str()) == Some("text") {
                    c.get("text").and_then(|x| x.as_str()).map(str::to_string)
                } else {
                    None
                }
            }),
            _ => None,
        }?;
        let first = text.lines().find(|l| !l.trim().is_empty())?.trim();
        return Some(truncate(first, 40));
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        let mut out: String = chars[..max.saturating_sub(1)].iter().collect();
        out.push('\u{2026}');
        out
    }
}

fn short_id(id: &str) -> String {
    let trimmed = id.trim_start_matches('a');
    trimmed.chars().take(6).collect()
}

fn mtime_secs(path: &Path) -> f64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// Keep FONT_SIZE referenced so a future caret/measure tweak has it handy
// and the import doesn't rot; cheap const, no runtime cost.
#[allow(dead_code)]
const _FONT_SIZE_REF: f32 = FONT_SIZE;
