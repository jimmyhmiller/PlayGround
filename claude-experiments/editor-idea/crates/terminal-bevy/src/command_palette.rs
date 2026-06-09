//! Command palette — a VSCode/Sublime-style centered overlay that fuzzy-
//! searches the [`ActionRegistry`](crate::actions::ActionRegistry) and
//! runs the chosen action. Opened with **Cmd+Shift+P**; Esc / Enter close it.
//!
//! It also hosts the **DeepSeek "Ask"** flow: when the query doesn't (or
//! does) match actions, an "Ask DeepSeek" row sends the query — plus the
//! current project/cwd/pane context — to the model, which replies with a
//! plan of tool calls drawn from the app's IPC surface
//! ([`crate::tools`]). Safe calls run immediately (via
//! [`crate::ipc::dispatch_local`], the same socket the CLIs use); risky
//! ones are listed and wait for Enter.
//!
//! ## Why native (not a widget pane)
//!
//! The palette is a *modal overlay*: it owns the keyboard, sits above
//! everything, stays centered. It reuses the widget **Element** vocabulary
//! for its look (inheriting theme tokens) but renders natively onto
//! [`MENU_OVERLAY_LAYER`].
//!
//! ## Render-layer correctness
//!
//! `widget_bevy::render::render` spawns primitives with no `RenderLayers`,
//! and pane-bevy's propagation only stamps subtrees under a `PaneLayer`
//! ancestor — which a native overlay isn't. So [`render_palette`] is an
//! **exclusive system**: it spawns the content root, renders into the
//! world command buffer, flushes, then stamps
//! `RenderLayers::layer(MENU_OVERLAY_LAYER)` over the whole subtree in the
//! same run. No frame shows palette glyphs on the main camera.
//!
//! ## Input isolation
//!
//! While the palette is open, `compute_keyboard_owner` (terminal-bevy)
//! sets `pane_bevy::KeyboardOwner::Modal`, which every keyboard consumer
//! respects — pane typing and global chords are suppressed centrally, so
//! the palette needs no per-handler focus juggling of its own.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Mutex;

use bevy::camera::visibility::RenderLayers;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;

use pane_bevy::{PaneFont, PaneFontMetrics, PaneRegistry};
use widget_bevy::protocol::{Align, Border, Edges, Element, Shadow, Style, Weight};
use widget_bevy::render::{self, LayoutCtx, WidgetPalette};
use widget_bevy::WidgetTargets;

use crate::actions::{ActionInvocations, ActionRegistry};
use crate::ipc;
use crate::projects::Projects;
use crate::tools;
use crate::MENU_OVERLAY_LAYER;

/// Z within the overlay layer — above the drawer (550) and radial (600).
const PALETTE_Z: f32 = 700.0;
const PALETTE_W: f32 = 600.0;
const TOP_MARGIN: f32 = 96.0;
const MAX_ROWS: usize = 10;
/// Synthetic result-row id for the "Ask DeepSeek" entry.
const ASK_ID: &str = "__ask_deepseek__";

/// Result of a DeepSeek worker call.
type AskResult = Result<tools::ToolPlan, String>;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum PaletteMode {
    /// Fuzzy action search (default).
    #[default]
    Actions,
    /// Waiting on a DeepSeek response.
    Busy,
    /// Showing the model's plan (safe calls already ran; risky ones
    /// pending Enter).
    Plan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowStatus {
    /// Risky call awaiting confirmation.
    Pending,
    /// Dispatched successfully.
    Ran,
    /// Rejected or failed to dispatch.
    Failed,
}

/// One row of a DeepSeek plan.
struct PlanRow {
    label: String,
    reason: String,
    /// Validated request for pending (risky) rows; `None` if rejected.
    req: Option<ipc::IpcRequest>,
    status: RowStatus,
    detail: String,
}

/// One row in the filtered action list.
#[derive(Clone)]
pub struct PaletteRow {
    pub id: &'static str,
    pub title: String,
    pub category: &'static str,
    pub keybind: Option<String>,
}

/// External request (e.g. from IPC) to open the palette.
#[derive(Resource, Default)]
pub struct PaletteOpenRequest {
    pub requested: bool,
    pub seed: Option<String>,
    /// Immediately fire the DeepSeek "Ask" flow with the seeded query.
    pub ask: bool,
}

/// Channel back from the DeepSeek worker thread. `Mutex` makes the
/// `Receiver` `Sync` so this stays an ordinary `Resource`.
#[derive(Resource, Default)]
pub struct DeepSeekChannel {
    rx: Mutex<Option<Receiver<AskResult>>>,
}

/// Per-action pick counts, persisted to disk. Used to bias ranking so a
/// frequently-chosen action floats above equal/near-equal fuzzy matches
/// (and an empty query lists your most-used first).
#[derive(Resource, Default)]
pub struct PaletteUsage {
    counts: HashMap<String, u32>,
}

/// Each pick is worth this many score points, capped so usage biases
/// ties / small gaps without overriding a clearly-better fuzzy match.
const USAGE_WEIGHT: i32 = 2;
const USAGE_CAP: u32 = 8;

impl PaletteUsage {
    fn load() -> Self {
        let Some(path) = usage_path() else {
            return Self::default();
        };
        match std::fs::read(&path) {
            Ok(bytes) => Self {
                counts: serde_json::from_slice(&bytes).unwrap_or_default(),
            },
            Err(_) => Self::default(),
        }
    }

    fn count(&self, id: &str) -> u32 {
        self.counts.get(id).copied().unwrap_or(0)
    }

    /// Score contribution for `id` — capped so habits win ties but not
    /// strong matches.
    fn bonus(&self, id: &str) -> i32 {
        self.count(id).min(USAGE_CAP) as i32 * USAGE_WEIGHT
    }

    /// Record a pick and persist immediately (the file is tiny).
    fn bump(&mut self, id: &str) {
        *self.counts.entry(id.to_string()).or_insert(0) += 1;
        if let Some(path) = usage_path() {
            if let Ok(bytes) = serde_json::to_vec_pretty(&self.counts) {
                let _ = std::fs::write(path, bytes);
            }
        }
    }
}

fn usage_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    p.push("palette_usage.json");
    Some(p)
}

#[derive(Resource, Default)]
pub struct CommandPalette {
    pub open: bool,
    pub mode: PaletteMode,
    pub query: String,
    pub results: Vec<PaletteRow>,
    pub selected: usize,
    /// Header line for Busy / Plan modes.
    message: String,
    plan_rows: Vec<PlanRow>,
    root: Option<Entity>,
    last_sig: u64,
}

impl CommandPalette {
    /// Hash of the visible state — [`render_palette`] only rebuilds when
    /// this changes.
    fn signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.open.hash(&mut h);
        (self.mode as u8).hash(&mut h);
        self.query.hash(&mut h);
        self.selected.hash(&mut h);
        self.message.hash(&mut h);
        self.results.len().hash(&mut h);
        for r in &self.plan_rows {
            r.label.hash(&mut h);
            (r.status as u8).hash(&mut h);
        }
        h.finish()
    }
}

pub struct CommandPalettePlugin;

impl Plugin for CommandPalettePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CommandPalette>()
            .init_resource::<PaletteOpenRequest>()
            .init_resource::<DeepSeekChannel>()
            .insert_resource(PaletteUsage::load())
            .add_systems(Update, (palette_input, poll_deepseek).chain())
            .add_systems(Update, render_palette.after(palette_input));
    }
}

// ---------- Fuzzy matching ----------

/// Subsequence fuzzy scorer; rewards consecutive runs and word-start hits.
fn fuzzy_score(needle: &str, haystack: &str) -> Option<i32> {
    let n: Vec<char> = needle.chars().flat_map(|c| c.to_lowercase()).collect();
    if n.is_empty() {
        return Some(0);
    }
    let h: Vec<char> = haystack.chars().collect();
    let mut hi = 0usize;
    let mut score = 0i32;
    let mut consec = 0i32;
    for &nc in &n {
        let mut found = false;
        while hi < h.len() {
            let hc = h[hi].to_ascii_lowercase();
            let at = hi;
            hi += 1;
            if hc == nc {
                score += 1 + consec;
                if at == 0 || matches!(h[at - 1], ' ' | '_' | '.' | '/' | '-') {
                    score += 4;
                }
                consec += 1;
                found = true;
                break;
            } else {
                consec = 0;
            }
        }
        if !found {
            return None;
        }
    }
    Some(score)
}

/// Filter + rank the registry, then append the synthetic "Ask DeepSeek"
/// row when the query is non-empty. Ranking blends the fuzzy score with a
/// capped usage bonus so frequently-picked actions float up (and break
/// ties against alphabetical/registration order).
fn refresh_results(palette: &mut CommandPalette, registry: &ActionRegistry, usage: &PaletteUsage) {
    let q = palette.query.trim();
    let mut scored: Vec<(i32, PaletteRow)> = registry
        .iter()
        .filter_map(|a| {
            let hay = if a.keywords.is_empty() {
                a.title.to_string()
            } else {
                format!("{} {}", a.title, a.keywords.join(" "))
            };
            let s = fuzzy_score(q, &hay)? + usage.bonus(a.id);
            Some((
                s,
                PaletteRow {
                    id: a.id,
                    title: a.title.to_string(),
                    category: a.category,
                    keybind: a.default_keybind.map(|k| k.label()),
                },
            ))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    let mut rows: Vec<PaletteRow> = scored.into_iter().map(|(_, r)| r).collect();
    if !q.is_empty() {
        rows.push(PaletteRow {
            id: ASK_ID,
            title: format!("Ask DeepSeek: {}", q),
            category: "AI",
            keybind: None,
        });
    }
    palette.results = rows;
    if palette.selected >= palette.results.len() {
        palette.selected = 0;
    }
}

// ---------- Input ----------

#[allow(clippy::too_many_arguments)]
fn palette_input(
    mut keys: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    registry: Res<ActionRegistry>,
    pane_registry: Res<PaneRegistry>,
    projects: Res<Projects>,
    mut palette: ResMut<CommandPalette>,
    mut invocations: ResMut<ActionInvocations>,
    mut deepseek: ResMut<DeepSeekChannel>,
    mut usage: ResMut<PaletteUsage>,
    mut open_req: ResMut<PaletteOpenRequest>,
) {
    // External (IPC) open request.
    if open_req.requested {
        open_req.requested = false;
        if !palette.open {
            open(&mut palette, &registry, &usage);
        }
        if let Some(seed) = open_req.seed.take() {
            palette.query = seed;
            refresh_results(&mut palette, &registry, &usage);
        }
        if std::mem::take(&mut open_req.ask) && !palette.query.trim().is_empty() {
            start_ask(&mut palette, &projects, &pane_registry, &mut deepseek);
        }
    }

    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);

    for ev in keys.read() {
        if !ev.state.is_pressed() {
            continue;
        }

        // Cmd+Shift+P toggles, open or closed (VSCode/Sublime palette key).
        if cmd && shift && ev.key_code == KeyCode::KeyP {
            if palette.open {
                close(&mut palette);
            } else {
                open(&mut palette, &registry, &usage);
            }
            continue;
        }

        if !palette.open {
            continue;
        }

        match &ev.logical_key {
            Key::Escape => close(&mut palette),
            Key::Enter => match palette.mode {
                PaletteMode::Actions => {
                    if let Some(row) = palette.results.get(palette.selected).cloned() {
                        if row.id == ASK_ID {
                            start_ask(&mut palette, &projects, &pane_registry, &mut deepseek);
                        } else {
                            usage.bump(row.id);
                            invocations.request(row.id, None);
                            close(&mut palette);
                        }
                    }
                }
                PaletteMode::Plan => {
                    let pending = palette
                        .plan_rows
                        .iter()
                        .any(|r| r.status == RowStatus::Pending);
                    if pending {
                        run_pending(&mut palette);
                    } else {
                        close(&mut palette);
                    }
                }
                PaletteMode::Busy => {}
            },
            Key::ArrowDown if palette.mode == PaletteMode::Actions => {
                if !palette.results.is_empty() {
                    palette.selected = (palette.selected + 1).min(palette.results.len() - 1);
                }
            }
            Key::ArrowUp if palette.mode == PaletteMode::Actions => {
                palette.selected = palette.selected.saturating_sub(1);
            }
            Key::Backspace if palette.mode == PaletteMode::Actions => {
                palette.query.pop();
                refresh_results(&mut palette, &registry, &usage);
            }
            Key::Space if palette.mode == PaletteMode::Actions => {
                palette.query.push(' ');
                refresh_results(&mut palette, &registry, &usage);
            }
            Key::Character(s) if palette.mode == PaletteMode::Actions && !cmd && !ctrl => {
                palette.query.push_str(s.as_str());
                refresh_results(&mut palette, &registry, &usage);
            }
            _ => {}
        }
    }
}

fn open(palette: &mut CommandPalette, registry: &ActionRegistry, usage: &PaletteUsage) {
    palette.open = true;
    palette.mode = PaletteMode::Actions;
    palette.query.clear();
    palette.message.clear();
    palette.plan_rows.clear();
    palette.selected = 0;
    refresh_results(palette, registry, usage);
    // No focus juggling: while `palette.open`, `compute_keyboard_owner`
    // sets `KeyboardOwner::Modal`, which centrally suppresses pane typing
    // and global chords. Focus returns to its pane automatically on close.
}

fn close(palette: &mut CommandPalette) {
    palette.open = false;
    palette.mode = PaletteMode::Actions;
    palette.plan_rows.clear();
}

// ---------- DeepSeek ----------

/// Kick off a DeepSeek call: assemble context, resolve the API config,
/// spawn the blocking call on a worker thread, and switch to Busy.
fn start_ask(
    palette: &mut CommandPalette,
    projects: &Projects,
    pane_registry: &PaneRegistry,
    deepseek: &mut DeepSeekChannel,
) {
    let cfg = match inference_bevy::llm::LlmConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            palette.mode = PaletteMode::Plan;
            palette.message = format!("DeepSeek unavailable: {e}");
            palette.plan_rows.clear();
            return;
        }
    };
    let context = assemble_context(projects, pane_registry);
    let prompt = palette.query.trim().to_string();
    let system = tools::system_prompt();
    let user = format!("Context:\n{context}\nRequest: {prompt}");

    let (tx, rx) = std::sync::mpsc::channel::<AskResult>();
    let spawned = std::thread::Builder::new()
        .name("deepseek-palette".into())
        .spawn(move || {
            let r = inference_bevy::llm::classify::<tools::ToolPlan>(&cfg, &system, &user)
                .map_err(|e| e.to_string());
            let _ = tx.send(r);
        });
    if spawned.is_err() {
        palette.mode = PaletteMode::Plan;
        palette.message = "Could not spawn DeepSeek worker".into();
        return;
    }
    *deepseek.rx.lock().unwrap() = Some(rx);
    palette.mode = PaletteMode::Busy;
    palette.message = "Asking DeepSeek…".into();
    palette.plan_rows.clear();
}

/// Concise context handed to the model.
fn assemble_context(projects: &Projects, pane_registry: &PaneRegistry) -> String {
    let mut s = String::new();
    if let Some(active) = projects.active {
        s.push_str(&format!(
            "Active project: {}\n",
            projects.name_of(active).unwrap_or("?")
        ));
        if let Some(cwd) = projects.default_cwd_of(active) {
            s.push_str(&format!("Active project cwd: {cwd}\n"));
        }
    }
    let names: Vec<&str> = projects.list.iter().map(|p| p.name.as_str()).collect();
    if !names.is_empty() {
        s.push_str(&format!("Known projects: {}\n", names.join(", ")));
    }
    let kinds: Vec<&str> = pane_registry.iter().map(|k| k.kind).collect();
    s.push_str(&format!("Pane kinds (for 'kind' args): {}\n", kinds.join(", ")));
    s
}

/// Poll the worker channel; when a plan arrives, run safe calls and queue
/// risky ones.
fn poll_deepseek(mut palette: ResMut<CommandPalette>, deepseek: Res<DeepSeekChannel>) {
    if palette.mode != PaletteMode::Busy {
        return;
    }
    let got: Option<AskResult> = {
        let mut guard = deepseek.rx.lock().unwrap();
        match guard.as_ref() {
            Some(rx) => match rx.try_recv() {
                Ok(r) => {
                    *guard = None;
                    Some(r)
                }
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    *guard = None;
                    Some(Err("DeepSeek worker exited".into()))
                }
            },
            None => None,
        }
    };
    let Some(result) = got else { return };
    match result {
        Ok(plan) => apply_plan(&mut palette, plan),
        Err(e) => {
            palette.mode = PaletteMode::Plan;
            palette.message = format!("DeepSeek error: {e}");
            palette.plan_rows.clear();
        }
    }
}

/// Validate each call; run safe ones immediately; queue risky ones.
fn apply_plan(palette: &mut CommandPalette, plan: tools::ToolPlan) {
    eprintln!(
        "[palette] deepseek plan: message={:?} calls={}",
        plan.message,
        plan.calls.len()
    );
    for c in &plan.calls {
        eprintln!("[palette]   call tool={} args={}", c.tool, c.args);
    }
    palette.message = if plan.message.is_empty() {
        "DeepSeek plan".into()
    } else {
        plan.message
    };
    palette.plan_rows.clear();
    for call in plan.calls {
        let spec = tools::spec_of(&call.tool);
        let mut row = PlanRow {
            label: call.tool.clone(),
            reason: call.reason.clone(),
            req: None,
            status: RowStatus::Failed,
            detail: String::new(),
        };
        match (spec, tools::to_ipc_request(&call)) {
            (Some(spec), Ok(req)) => {
                if spec.risk == tools::Risk::Risky {
                    row.req = Some(req);
                    row.status = RowStatus::Pending;
                } else {
                    match ipc::dispatch_local(&req) {
                        Ok(()) => {
                            eprintln!("[palette]   dispatched safe tool {}", call.tool);
                            row.status = RowStatus::Ran;
                        }
                        Err(e) => {
                            eprintln!("[palette]   dispatch failed for {}: {e}", call.tool);
                            row.status = RowStatus::Failed;
                            row.detail = e.to_string();
                        }
                    }
                }
            }
            (None, _) => {
                eprintln!("[palette]   rejected unknown tool {}", call.tool);
                row.detail = "not an available tool".into();
            }
            (_, Err(e)) => {
                eprintln!("[palette]   invalid args for {}: {e}", call.tool);
                row.detail = e;
            }
        }
        palette.plan_rows.push(row);
    }
    palette.mode = PaletteMode::Plan;
}

/// Dispatch all still-pending (risky, confirmed) rows.
fn run_pending(palette: &mut CommandPalette) {
    for row in &mut palette.plan_rows {
        if row.status != RowStatus::Pending {
            continue;
        }
        if let Some(req) = &row.req {
            match ipc::dispatch_local(req) {
                Ok(()) => row.status = RowStatus::Ran,
                Err(e) => {
                    row.status = RowStatus::Failed;
                    row.detail = e.to_string();
                }
            }
        }
    }
}

// ---------- Render (exclusive) ----------

fn render_palette(world: &mut World) {
    let open = world.resource::<CommandPalette>().open;
    let sig = world.resource::<CommandPalette>().signature();
    let prev_root = world.resource::<CommandPalette>().root;
    let last_sig = world.resource::<CommandPalette>().last_sig;

    if !open {
        if let Some(root) = prev_root {
            despawn_tree(world, root);
            world.resource_mut::<CommandPalette>().root = None;
        }
        return;
    }
    if prev_root.is_some() && sig == last_sig {
        return;
    }
    if let Some(root) = prev_root {
        despawn_tree(world, root);
    }

    let win_h = {
        let mut q = world.query::<&Window>();
        match q.iter(world).next() {
            Some(w) => w.height(),
            None => return,
        }
    };

    let theme = world.resource::<style_bevy::Theme>().clone();
    let fonts = world.resource::<style_bevy::FontRegistry>().clone();
    let font = world.resource::<PaneFont>().0.clone();
    let metrics = *world.resource::<PaneFontMetrics>();
    let colors = WidgetPalette::from_theme(&theme);

    let el = build_palette_element(world.resource::<CommandPalette>());

    let top_left = Vec2::new(-PALETTE_W * 0.5, win_h * 0.5 - TOP_MARGIN);
    let root = world
        .spawn((
            Transform::from_xyz(top_left.x, top_left.y, PALETTE_Z),
            Visibility::Visible,
            RenderLayers::layer(MENU_OVERLAY_LAYER),
        ))
        .id();

    let ctx = LayoutCtx {
        font,
        metrics,
        owner_pane: root,
        content_root: root,
        content_size: Vec2::new(PALETTE_W, win_h),
        palette: colors,
        theme,
        fonts,
        focused_input: None,
        caret_visible: true,
        hovered_click_id: None,
    };
    let mut targets = WidgetTargets::default();
    {
        let mut commands = world.commands();
        render::render(
            &mut commands,
            &ctx,
            &mut targets,
            &el,
            Vec2::ZERO,
            PALETTE_W,
            0.0,
        );
    }
    world.flush();
    stamp_layer(world, root, MENU_OVERLAY_LAYER);

    let mut p = world.resource_mut::<CommandPalette>();
    p.root = Some(root);
    p.last_sig = sig;
}

fn build_palette_element(palette: &CommandPalette) -> Element {
    let children = match palette.mode {
        PaletteMode::Actions => actions_children(palette),
        PaletteMode::Busy => vec![
            query_text(&palette.query),
            text_colored(&palette.message, "accent", 15.0),
        ],
        PaletteMode::Plan => plan_children(palette),
    };

    Element::Frame {
        gap: 4.0,
        pad: 0.0,
        children,
        style: Some(Style {
            background: Some("surface_2".into()),
            radius: Some("radius_lg".into()),
            border: Some(Border {
                color: "surface_3".into(),
                width: 1.0,
            }),
            padding: Some(Edges::all(14.0)),
            width: Some(format!("{}", PALETTE_W as i32)),
            shadow: Some(Shadow {
                token: Some("shadow_lg".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
    }
}

fn actions_children(palette: &CommandPalette) -> Vec<Element> {
    let mut children = vec![query_text(&palette.query)];
    if palette.results.is_empty() {
        children.push(text_muted("no matching actions", 14.0));
    }
    for (i, row) in palette.results.iter().take(MAX_ROWS).enumerate() {
        let hint = row.keybind.clone().unwrap_or_else(|| row.category.to_string());
        children.push(list_row(
            row.id,
            &row.title,
            &hint,
            i == palette.selected,
            "fg",
        ));
    }
    children
}

fn plan_children(palette: &CommandPalette) -> Vec<Element> {
    let mut children = vec![text_colored(&palette.message, "accent", 16.0)];
    if palette.plan_rows.is_empty() {
        children.push(text_muted("(no actions)", 14.0));
    }
    for row in &palette.plan_rows {
        let (glyph, color) = match row.status {
            RowStatus::Ran => ("✓", "fg"),
            RowStatus::Pending => ("⏎", "accent"),
            RowStatus::Failed => ("✗", "fg_muted"),
        };
        let mut label = format!("{glyph}  {}", row.label);
        if !row.reason.is_empty() {
            label.push_str(&format!(" — {}", row.reason));
        }
        let hint = if row.detail.is_empty() {
            match row.status {
                RowStatus::Ran => "ran".to_string(),
                RowStatus::Pending => "needs confirm".to_string(),
                RowStatus::Failed => "skipped".to_string(),
            }
        } else {
            row.detail.clone()
        };
        children.push(list_row("plan", &label, &hint, false, color));
    }
    let pending = palette
        .plan_rows
        .iter()
        .any(|r| r.status == RowStatus::Pending);
    children.push(text_muted(
        if pending {
            "Enter to run the highlighted calls · Esc to dismiss"
        } else {
            "Esc to dismiss"
        },
        12.0,
    ));
    children
}

// ---------- Element helpers ----------

fn query_text(query: &str) -> Element {
    Element::Text {
        value: format!("› {}▏", query),
        color: Some("fg".into()),
        size: Some(20.0),
        weight: Some(Weight::Normal),
        family: None,
        selectable: false,
    }
}

fn list_row(id: &str, title: &str, hint: &str, selected: bool, title_color: &str) -> Element {
    let title_el = Element::Frame {
        gap: 0.0,
        pad: 0.0,
        children: vec![Element::Text {
            value: title.to_string(),
            color: Some(title_color.into()),
            size: Some(15.0),
            weight: Some(Weight::Normal),
            family: None,
            selectable: false,
        }],
        style: Some(Style {
            flex_grow: Some(1.0),
            ..Default::default()
        }),
    };
    let hint_el = Element::Text {
        value: hint.to_string(),
        color: Some("fg_muted".into()),
        size: Some(13.0),
        weight: Some(Weight::Normal),
        family: None,
        selectable: false,
    };
    Element::ListItem {
        id: id.to_string(),
        selected,
        gap: 8.0,
        pad: 8.0,
        children: vec![Element::Hstack {
            gap: 8.0,
            pad: 0.0,
            align: Align::Center,
            children: vec![title_el, hint_el],
            style: Some(Style {
                width: Some("100%".into()),
                ..Default::default()
            }),
        }],
        style: None,
    }
}

fn text_muted(s: &str, size: f32) -> Element {
    text_colored(s, "fg_muted", size)
}

fn text_colored(s: &str, color: &str, size: f32) -> Element {
    Element::Text {
        value: s.to_string(),
        color: Some(color.into()),
        size: Some(size),
        weight: Some(Weight::Normal),
        family: None,
        selectable: false,
    }
}

// ---------- Subtree helpers ----------

fn stamp_layer(world: &mut World, root: Entity, layer: usize) {
    let mut stack = vec![root];
    while let Some(e) = stack.pop() {
        let kids: Vec<Entity> = world
            .get::<Children>(e)
            .map(|c| c.iter().collect::<Vec<Entity>>())
            .unwrap_or_default();
        if let Ok(mut em) = world.get_entity_mut(e) {
            em.insert(RenderLayers::layer(layer));
        }
        stack.extend(kids);
    }
}

fn despawn_tree(world: &mut World, root: Entity) {
    // `despawn` cascades to descendants, so despawning the root is enough
    // — walking and despawning each child would double-despawn and log
    // "entity is invalid" warnings.
    let _ = world.despawn(root);
}
