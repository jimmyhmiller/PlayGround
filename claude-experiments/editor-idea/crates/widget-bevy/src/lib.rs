//! Widget panes: external processes that emit a UI-as-data stream.
//!
//! Each pane spawns a subprocess and pipes NDJSON between it and the
//! host. The widget owns the UI tree; the host owns layout, rendering,
//! and input dispatch. See `protocol.rs` for the message shapes.
//!
//! Config keys consumed by `spawn`:
//!   - `command` (string)  — shell command line to run; falls back to
//!                            `WIDGET_BEVY_DEFAULT_CMD` env var, then a
//!                            placeholder frame if neither is set.
//!   - `args`    (string[]) — when present, runs `command args…` directly
//!                            (no shell). When absent, `command` is fed
//!                            to `sh -c`.
//!   - `title`   (string)  — initial pane title bar text.
//!   - `cwd`     (string)  — working directory for the child.
//!   - `state`   (any)     — last `state` blob the widget published;
//!                            sent back as `init.state` next launch.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;
use pane_bevy::{
    MARGIN, PaneContentPressed, PaneFont, PaneFontMetrics, PaneKindMarker, PaneKindSpec, PaneRect,
    PaneRegistry, PaneTitle, TITLE_H, FocusedTextInput, TextInput, TextInputEvent, TextInputStyle,
    focus_text_input, spawn_text_input,
};
use serde_json::Value;

pub mod protocol;
pub mod render;

use protocol::{Element, HostEvent, Weight, WidgetMsg};

/// Stable identifier for widget panes. Used in `PaneKindMarker` and
/// snapshots.
pub const PANE_KIND: &str = "widget";

/// Env var consulted when a pane is spawned without an explicit command
/// (i.e. from the radial menu rather than a saved snapshot).
pub const DEFAULT_CMD_ENV: &str = "WIDGET_BEVY_DEFAULT_CMD";

// ---------- Components ----------

/// Widget config + per-pane state that persists across process exits.
#[derive(Component)]
pub struct Widget {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    /// Most recent `state` message the child published. Mirrored into
    /// snapshots so the next launch can resume.
    pub last_state: Value,
    /// Timestamp of most recent content press, for double-click detection.
    last_press_time: Option<f64>,
}

const DOUBLE_CLICK_SECS: f64 = 0.35;
const EDIT_FONT_SIZE: f32 = 13.0;
const EDIT_LINE_HEIGHT: f32 = 16.0;
const EDIT_PAD: f32 = 8.0;
const EDIT_INPUT_H: f32 = 22.0;
const EDIT_HINT_FONT_SIZE: f32 = 11.0;
const EDIT_LABEL_FONT_SIZE: f32 = 11.0;

/// Marker + entity refs for the "edit command" overlay shown when the
/// user double-clicks a widget. While present, `rerender_widgets`
/// suppresses normal frame rendering.
#[derive(Component)]
pub struct WidgetEditMode {
    pub command_input: Entity,
    pub label: Entity,
    pub hint: Entity,
    pub bg: Entity,
}

/// Handle to the running child process. Absent when the widget has no
/// command configured (placeholder mode) or after the child exits.
#[derive(Component)]
pub struct WidgetProcess {
    pub child: Child,
}

/// Channels to/from the IO threads. `rx` carries parsed widget→host
/// messages; `tx` carries host→widget JSON lines (already serialized).
#[derive(Component)]
pub struct WidgetIO {
    pub rx: Mutex<Receiver<WidgetMsg>>,
    pub tx: Sender<String>,
}

/// What the host knows about the visual: the active frame, an optional
/// pending frame waiting to be rendered, and the size we last rendered
/// at so we can detect resizes.
#[derive(Component, Default)]
pub struct WidgetRender {
    pub pending_frame: Option<Element>,
    pub current_frame: Option<Element>,
    pub last_size: Vec2,
    pub init_sent: bool,
}

/// Hit-test geometry collected while rendering the current frame.
#[derive(Component, Default)]
pub struct WidgetTargets {
    pub clicks: Vec<ClickTarget>,
    pub links: Vec<LinkTarget>,
}

pub struct ClickTarget {
    pub id: String,
    /// Local to the content_root (y-down, pixels from top-left of the
    /// content area). Same frame as `PaneContentPressed.local_pt`.
    pub rect: Rect,
}

pub struct LinkTarget {
    pub url: String,
    pub rect: Rect,
}

/// Cached `content_root` entity so render systems don't have to walk
/// the pane chrome to find it.
#[derive(Component)]
pub struct WidgetContentRoot(pub Entity);

/// Set by any system that just spawned new sprites under a widget's
/// content_root (`rerender_widgets`, `enter_edit_mode`). Consumed by
/// `clip_widget_sprites` so it knows to do a sweep this frame even
/// when no `PaneRect` changed.
#[derive(Resource, Default)]
pub struct WidgetClipDirty(pub bool);

// ---------- Plugin / registry ----------

pub struct WidgetPlugin;

impl Plugin for WidgetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WidgetClipDirty>()
            .add_systems(Startup, register_kind)
            .add_systems(
                Update,
                (
                    tick_widget_io,
                    rerender_widgets,
                    handle_widget_press,
                    handle_widget_edit_events,
                    poll_widget_children,
                )
                    .chain(),
            )
            .add_systems(PostUpdate, clip_widget_sprites);
    }
}

/// Walks every widget pane's content_root subtree and clamps every
/// Sprite's `custom_size` so it can't escape the pane's content area.
///
/// This is the single automatic bound-enforcement for widget sprites —
/// individual `render_*` calls don't have to know the pane's current
/// size, they just emit sprites at their intrinsic size and this
/// system trims anything that overflows on the next frame.
///
/// Text2d is already clipped by pane-bevy's `enforce_pane_content_bounds`,
/// so we don't touch it here. The two systems together cover both axes
/// and both visual primitives.
fn clip_widget_sprites(
    panes: Query<(&PaneKindMarker, &PaneRect, &WidgetContentRoot)>,
    changed_panes: Query<(), (With<PaneKindMarker>, Changed<PaneRect>)>,
    mut needs_clip: ResMut<WidgetClipDirty>,
    children_q: Query<&Children>,
    transforms: Query<&Transform>,
    mut sprites: Query<&mut Sprite>,
) {
    // Same idle-fast-path as pane_bevy::enforce_pane_content_bounds:
    // walking every widget subtree every frame to clamp sizes that
    // haven't changed is pure waste. Re-walk only when a pane just
    // resized or `rerender_widgets` (or edit-mode) just spawned new
    // sprites under a content_root. Use a `ResMut<WidgetClipDirty>`
    // signal for the latter — we can't ask for `Added<Sprite>` here
    // because that conflicts with `&mut Sprite`.
    let new_content = needs_clip.0;
    needs_clip.0 = false;
    if changed_panes.is_empty() && !new_content {
        return;
    }
    for (kind, rect, root) in &panes {
        if kind.0 != PANE_KIND {
            continue;
        }
        let content_w = (rect.size.x - 2.0 * MARGIN).max(0.0);
        let content_h = (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0);

        // Walk subtree depth-first. `offset` accumulates Bevy local
        // translations from `content_root` outward: x is right, y is up
        // (negative y = down inside the pane).
        let mut stack: Vec<(Entity, Vec2)> = Vec::with_capacity(16);
        if let Ok(children) = children_q.get(root.0) {
            for c in children.iter() {
                let t = transforms
                    .get(c)
                    .map(|t| Vec2::new(t.translation.x, t.translation.y))
                    .unwrap_or(Vec2::ZERO);
                stack.push((c, t));
            }
        }

        while let Some((entity, offset)) = stack.pop() {
            // top_offset is "distance below content_root top, in px".
            let top_offset = (-offset.y).max(0.0);
            let left_offset = offset.x.max(0.0);
            let avail_w = (content_w - left_offset).max(0.0);
            let avail_h = (content_h - top_offset).max(0.0);

            if let Ok(mut sprite) = sprites.get_mut(entity) {
                if let Some(want) = sprite.custom_size {
                    let new = Vec2::new(want.x.min(avail_w), want.y.min(avail_h));
                    if (new.x - want.x).abs() > f32::EPSILON
                        || (new.y - want.y).abs() > f32::EPSILON
                    {
                        sprite.custom_size = Some(new);
                    }
                }
            }

            if let Ok(children) = children_q.get(entity) {
                for child in children.iter() {
                    let ct = transforms
                        .get(child)
                        .map(|t| Vec2::new(t.translation.x, t.translation.y))
                        .unwrap_or(Vec2::ZERO);
                    stack.push((child, offset + ct));
                }
            }
        }
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Widget",
        radial_icon: Some("◫"),
        default_size: Vec2::new(360.0, 240.0),
        spawn: widget_spawn,
        snapshot: widget_snapshot,
        on_close: Some(widget_on_close),
    });
}

// ---------- Spawn ----------

fn widget_spawn(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Widget")
        .to_string();
    let command = config
        .get("command")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| std::env::var(DEFAULT_CMD_ENV).ok())
        .unwrap_or_default();
    let args: Vec<String> = config
        .get("args")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let cwd = config
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    let state = config.get("state").cloned().unwrap_or(Value::Null);

    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title.clone();
    } else {
        world.entity_mut(entity).insert(PaneTitle(title));
    }

    let widget = Widget {
        command: command.clone(),
        args: args.clone(),
        cwd: cwd.clone(),
        last_state: state.clone(),
        last_press_time: None,
    };
    let mut render_state = WidgetRender::default();
    let targets = WidgetTargets::default();

    if command.trim().is_empty() {
        render_state.current_frame = Some(placeholder_frame());
        world.entity_mut(entity).insert((
            widget,
            render_state,
            targets,
            WidgetContentRoot(content_root),
        ));
        return;
    }

    match spawn_widget_process(&command, &args, cwd.as_deref()) {
        Ok((process, io)) => {
            world.entity_mut(entity).insert((
                widget,
                render_state,
                targets,
                WidgetContentRoot(content_root),
                process,
                io,
            ));
        }
        Err(e) => {
            eprintln!("[widget] spawn failed: {}", e);
            render_state.current_frame = Some(error_frame(&format!("spawn failed: {}", e)));
            world.entity_mut(entity).insert((
                widget,
                render_state,
                targets,
                WidgetContentRoot(content_root),
            ));
        }
    }
}

fn spawn_widget_process(
    cmd: &str,
    args: &[String],
    cwd: Option<&Path>,
) -> std::io::Result<(WidgetProcess, WidgetIO)> {
    let mut command = if args.is_empty() {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    } else {
        let mut c = Command::new(cmd);
        c.args(args);
        c
    };
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(p) = cwd {
        command.current_dir(p);
    }
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdin = child.stdin.take().expect("stdin was piped");

    let (msg_tx, msg_rx) = mpsc::channel::<WidgetMsg>();
    let (line_tx, line_rx) = mpsc::channel::<String>();

    // Stdout: parse NDJSON, forward parsed messages.
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(s) = line else { return };
            if s.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WidgetMsg>(&s) {
                Ok(m) => {
                    if msg_tx.send(m).is_err() {
                        return;
                    }
                }
                Err(e) => eprintln!("[widget] parse error: {} | line: {}", e, s),
            }
        }
    });

    // Stderr: log only — useful for `set -x` style debugging.
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(s) = line {
                eprintln!("[widget stderr] {}", s);
            }
        }
    });

    // Stdin writer thread — receives already-serialized JSON lines.
    let mut stdin_writer = stdin;
    std::thread::spawn(move || {
        while let Ok(line) = line_rx.recv() {
            if stdin_writer.write_all(line.as_bytes()).is_err() {
                return;
            }
            if stdin_writer.write_all(b"\n").is_err() {
                return;
            }
            if stdin_writer.flush().is_err() {
                return;
            }
        }
    });

    Ok((
        WidgetProcess { child },
        WidgetIO {
            rx: Mutex::new(msg_rx),
            tx: line_tx,
        },
    ))
}

// ---------- Snapshot / close ----------

fn widget_snapshot(world: &World, entity: Entity) -> Value {
    let Some(w) = world.get::<Widget>(entity) else {
        return Value::Null;
    };
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    let mut out = serde_json::Map::new();
    out.insert("title".into(), Value::String(title));
    out.insert("command".into(), Value::String(w.command.clone()));
    if !w.args.is_empty() {
        out.insert(
            "args".into(),
            Value::Array(w.args.iter().cloned().map(Value::String).collect()),
        );
    }
    if let Some(p) = &w.cwd {
        out.insert("cwd".into(), Value::String(p.to_string_lossy().into_owned()));
    }
    if !w.last_state.is_null() {
        out.insert("state".into(), w.last_state.clone());
    }
    Value::Object(out)
}

fn widget_on_close(world: &mut World, entity: Entity) {
    // Best-effort graceful shutdown: send a "close" event, then kill.
    // The stdin writer thread may not get a chance to flush before drop,
    // but the kill guarantees the child goes away.
    if let Some(io) = world.get::<WidgetIO>(entity)
        && let Ok(json) = serde_json::to_string(&HostEvent::Close)
    {
        let _ = io.tx.send(json);
    }
    if let Some(mut wp) = world.get_mut::<WidgetProcess>(entity) {
        let _ = wp.child.kill();
    }
}

// ---------- Systems ----------

/// Drain inbound messages, send `init` once, and send `resize` when the
/// content area changes.
fn tick_widget_io(
    mut q: Query<(
        &PaneKindMarker,
        &PaneRect,
        &mut Widget,
        &mut WidgetRender,
        Option<&WidgetIO>,
    )>,
    mut titles: Query<&mut PaneTitle>,
    pane_q: Query<Entity, With<PaneKindMarker>>,
) {
    // Walk pane entities so we can update PaneTitle by entity. Bevy 0.18
    // doesn't allow mixing &mut Widget and &mut PaneTitle on the same
    // entity from the same Query when they coexist, so we look up titles
    // separately by entity id.
    let entities: Vec<Entity> = pane_q.iter().collect();
    for entity in entities {
        let Ok((kind, rect, mut w, mut render_state, io_opt)) = q.get_mut(entity) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }

        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );

        if let Some(io) = io_opt {
            // Init once we know the pane size.
            if !render_state.init_sent {
                let init = HostEvent::Init {
                    width: content_size.x,
                    height: content_size.y,
                    state: w.last_state.clone(),
                };
                if let Ok(json) = serde_json::to_string(&init) {
                    let _ = io.tx.send(json);
                }
                render_state.init_sent = true;
            }

            // Resize on size change (after init has gone out).
            if render_state.last_size != content_size && render_state.last_size != Vec2::ZERO {
                let resize = HostEvent::Resize {
                    width: content_size.x,
                    height: content_size.y,
                };
                if let Ok(json) = serde_json::to_string(&resize) {
                    let _ = io.tx.send(json);
                }
            }

            // Drain everything available without blocking.
            if let Ok(rx) = io.rx.lock() {
                loop {
                    match rx.try_recv() {
                        Ok(WidgetMsg::Frame { root }) => {
                            render_state.pending_frame = Some(root);
                        }
                        Ok(WidgetMsg::State { value }) => {
                            w.last_state = value;
                        }
                        Ok(WidgetMsg::Title { value }) => {
                            if let Ok(mut t) = titles.get_mut(entity) {
                                if t.0 != value {
                                    t.0 = value;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        }
    }
}

/// Despawn old content children and rebuild the tree whenever the
/// frame changes or the content area resizes. Skipped while a widget is
/// in edit mode (the overlay owns the content_root subtree).
fn rerender_widgets(
    mut commands: Commands,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    mut clip_dirty: ResMut<WidgetClipDirty>,
    mut q: Query<(
        &PaneKindMarker,
        &PaneRect,
        &WidgetContentRoot,
        &mut WidgetRender,
        &mut WidgetTargets,
        Option<&WidgetEditMode>,
    )>,
    children_q: Query<&Children>,
) {
    for (kind, rect, root, mut render_state, mut targets, edit) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        if edit.is_some() {
            continue;
        }

        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );

        let frame_came_in = render_state.pending_frame.is_some();
        if let Some(p) = render_state.pending_frame.take() {
            render_state.current_frame = Some(p);
        }
        let size_changed = render_state.last_size != content_size;
        let needs_render =
            (frame_came_in || size_changed) && render_state.current_frame.is_some();

        if !needs_render {
            // Track size even when we don't render so we don't fire a
            // spurious "size changed" the first time a frame arrives.
            if size_changed {
                render_state.last_size = content_size;
            }
            continue;
        }

        if let Ok(children) = children_q.get(root.0) {
            for c in children.iter() {
                commands.entity(c).despawn();
            }
        }
        targets.clicks.clear();
        targets.links.clear();

        let ctx = render::LayoutCtx {
            font: pane_font.0.clone(),
            metrics: *metrics,
            content_root: root.0,
            content_size,
        };
        let frame_clone = render_state.current_frame.clone().unwrap();
        render::render(
            &mut commands,
            &ctx,
            &mut targets,
            &frame_clone,
            Vec2::ZERO,
            content_size.x,
            0.0,
        );

        render_state.last_size = content_size;
        clip_dirty.0 = true;
    }
}

fn handle_widget_press(
    mut commands: Commands,
    time: Res<Time>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    mut focused: ResMut<FocusedTextInput>,
    mut clip_dirty: ResMut<WidgetClipDirty>,
    mut presses: MessageReader<PaneContentPressed>,
    kinds: Query<&PaneKindMarker>,
    pane_rects: Query<&PaneRect>,
    mut widgets: Query<(
        &mut Widget,
        &WidgetTargets,
        Option<&WidgetIO>,
        &WidgetContentRoot,
        Option<&WidgetEditMode>,
    )>,
) {
    for ev in presses.read() {
        let Ok(kind) = kinds.get(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        let Ok((mut widget, targets, io, root, edit)) = widgets.get_mut(ev.pane) else {
            continue;
        };

        // In edit mode, every press is consumed by the overlay. Blur on
        // clicks outside the input rect; focus on clicks inside.
        if let Some(edit) = edit {
            let input_rect = Rect::from_corners(
                Vec2::new(EDIT_PAD, EDIT_PAD + EDIT_LINE_HEIGHT + 4.0),
                Vec2::new(f32::MAX, EDIT_PAD + EDIT_LINE_HEIGHT + 4.0 + EDIT_INPUT_H),
            );
            let target = if input_rect.contains(ev.local_pt) {
                Some(edit.command_input)
            } else {
                None
            };
            focus_text_input(&mut commands, &mut focused, [], target);
            continue;
        }

        // First: normal widget-frame click handling (buttons, links).
        let click = targets.clicks.iter().find(|t| t.rect.contains(ev.local_pt));
        let link = targets.links.iter().find(|t| t.rect.contains(ev.local_pt));

        if let Some(ct) = click {
            if let Some(io) = io {
                let evt = HostEvent::Click { id: ct.id.clone() };
                if let Ok(json) = serde_json::to_string(&evt) {
                    let _ = io.tx.send(json);
                }
            }
            widget.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }
        if let Some(lt) = link {
            open_url(&lt.url);
            widget.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }

        // Empty-space press: double-click → enter edit mode.
        let now = time.elapsed_secs_f64();
        let is_double = widget
            .last_press_time
            .is_some_and(|t| now - t < DOUBLE_CLICK_SECS);
        widget.last_press_time = Some(now);
        if is_double {
            enter_edit_mode(
                &mut commands,
                &mut focused,
                ev.pane,
                root.0,
                &widget,
                pane_font.0.clone(),
                content_size_of(&pane_rects, ev.pane),
                &metrics,
            );
            clip_dirty.0 = true;
        }
    }
}

fn content_size_of(pane_rects: &Query<&PaneRect>, pane: Entity) -> Vec2 {
    let Ok(rect) = pane_rects.get(pane) else {
        return Vec2::ZERO;
    };
    Vec2::new(
        (rect.size.x - 2.0 * MARGIN).max(0.0),
        (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
    )
}

fn enter_edit_mode(
    commands: &mut Commands,
    focused: &mut FocusedTextInput,
    pane: Entity,
    content_root: Entity,
    widget: &Widget,
    font: Handle<Font>,
    content_size: Vec2,
    metrics: &PaneFontMetrics,
) {
    // Clear out any frame children — the overlay owns the content_root
    // subtree while we're in edit mode. We use Commands::despawn rather
    // than walking Children explicitly; the next rerender (after exit)
    // will rebuild the frame from current_frame.
    commands.entity(content_root).despawn_related::<Children>();

    // Background fills the content area. clip_widget_sprites would
    // clamp us anyway, but starting at the right size avoids the
    // one-frame flash of an oversize sprite.
    let bg = commands
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: Color::srgba(0.08, 0.085, 0.10, 0.92),
                custom_size: Some(content_size),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.05),
        ))
        .id();

    let label = commands
        .spawn((
            ChildOf(content_root),
            Text2d::new("Command"),
            TextFont {
                font: font.clone(),
                font_size: EDIT_LABEL_FONT_SIZE,
                ..default()
            },
            TextColor(Color::srgb(0.65, 0.68, 0.74)),
            LineHeight::Px(EDIT_LINE_HEIGHT),
            Anchor::TOP_LEFT,
            Transform::from_xyz(EDIT_PAD, -EDIT_PAD, 0.1),
        ))
        .id();

    let input_y = -(EDIT_PAD + EDIT_LINE_HEIGHT + 4.0);
    let style = TextInputStyle {
        font: font.clone(),
        font_size: EDIT_FONT_SIZE,
        line_height: EDIT_INPUT_H,
        cell_width: metrics.char_width(EDIT_FONT_SIZE),
        color_idle: Color::srgb(0.85, 0.86, 0.90),
        color_focused: Color::srgb(0.97, 0.98, 1.00),
        color_caret: Color::srgb(0.55, 0.85, 1.0),
        color_selection: Color::srgba(0.42, 0.62, 0.92, 0.35),
    };
    let input_width = (content_size.x - 2.0 * EDIT_PAD).max(40.0);
    let command_input = spawn_text_input(
        commands,
        content_root,
        &widget.command,
        style,
        input_width,
        Transform::from_xyz(EDIT_PAD, input_y, 0.2),
    );

    let hint_y = input_y - EDIT_INPUT_H - 4.0;
    let hint = commands
        .spawn((
            ChildOf(content_root),
            Text2d::new("Enter to save, Esc to cancel"),
            TextFont {
                font,
                font_size: EDIT_HINT_FONT_SIZE,
                ..default()
            },
            TextColor(Color::srgb(0.50, 0.52, 0.58)),
            LineHeight::Px(EDIT_LINE_HEIGHT),
            Anchor::TOP_LEFT,
            Transform::from_xyz(EDIT_PAD, hint_y, 0.1),
        ))
        .id();

    commands.entity(pane).insert(WidgetEditMode {
        command_input,
        label,
        hint,
        bg,
    });

    focus_text_input(commands, focused, [], Some(command_input));
}

/// React to Submit/Cancel from the command-edit TextInput. Submit
/// applies the new command and respawns the subprocess; Cancel just
/// tears down the overlay.
fn handle_widget_edit_events(
    mut commands: Commands,
    mut events: MessageReader<TextInputEvent>,
    mut focused: ResMut<FocusedTextInput>,
    text_inputs: Query<&TextInput>,
    mut state_q: Query<(&mut Widget, &mut WidgetRender)>,
    io_q: Query<&WidgetIO>,
    pane_for_input: Query<(Entity, &WidgetEditMode)>,
) {
    for ev in events.read() {
        let (submit, entity_input) = match *ev {
            TextInputEvent::Submit { entity } => (true, entity),
            TextInputEvent::Cancel { entity } => (false, entity),
            TextInputEvent::Changed { .. } => continue,
        };
        let Some((pane, _)) = pane_for_input
            .iter()
            .find(|(_, e)| e.command_input == entity_input)
        else {
            continue;
        };
        // Re-fetch the edit-mode component so we can call exit_edit_mode
        // with a stable reference (the borrow from pane_for_input would
        // conflict with state_q's mutable borrow).
        let edit_snapshot = pane_for_input
            .get(pane)
            .ok()
            .map(|(_, e)| (e.command_input, e.label, e.hint, e.bg));

        if submit {
            let new_cmd = text_inputs
                .get(entity_input)
                .map(|ti| ti.text())
                .unwrap_or_default();
            apply_command_change(&mut commands, pane, new_cmd, &mut state_q, &io_q);
        }

        let Some((ci, lbl, hint, bg)) = edit_snapshot else {
            continue;
        };
        let Ok((_, mut render_state)) = state_q.get_mut(pane) else {
            continue;
        };
        if focused.0 == Some(ci) {
            focus_text_input(&mut commands, &mut focused, [], None);
        }
        commands.entity(ci).despawn();
        commands.entity(lbl).despawn();
        commands.entity(hint).despawn();
        commands.entity(bg).despawn();
        commands.entity(pane).remove::<WidgetEditMode>();
        render_state.last_size = Vec2::ZERO;
    }
}

/// Replace the pane's command with `new_cmd`, kill any running child,
/// and spawn a fresh one. Empty command → placeholder frame.
fn apply_command_change(
    commands: &mut Commands,
    pane: Entity,
    new_cmd: String,
    state_q: &mut Query<(&mut Widget, &mut WidgetRender)>,
    io_q: &Query<&WidgetIO>,
) {
    if let Ok(io) = io_q.get(pane) {
        if let Ok(json) = serde_json::to_string(&HostEvent::Close) {
            let _ = io.tx.send(json);
        }
    }
    // Kill + remove the old child via an exclusive-world hop, since the
    // outer system can't get &mut WidgetProcess (would alias with the
    // other mut queries).
    commands.queue(move |world: &mut World| {
        if let Some(mut wp) = world.get_mut::<WidgetProcess>(pane) {
            let _ = wp.child.kill();
        }
        world
            .entity_mut(pane)
            .remove::<WidgetProcess>()
            .remove::<WidgetIO>();
    });

    let (cmd_str, args_vec, cwd_opt) = {
        let Ok((mut widget, mut render_state)) = state_q.get_mut(pane) else {
            return;
        };
        widget.command = new_cmd.clone();
        widget.last_state = Value::Null;
        render_state.init_sent = false;
        render_state.pending_frame = None;
        if new_cmd.trim().is_empty() {
            render_state.current_frame = Some(placeholder_frame());
            return;
        }
        render_state.current_frame = None;
        (widget.command.clone(), widget.args.clone(), widget.cwd.clone())
    };

    match spawn_widget_process(&cmd_str, &args_vec, cwd_opt.as_deref()) {
        Ok((process, io)) => {
            commands.entity(pane).insert((process, io));
        }
        Err(e) => {
            eprintln!("[widget] respawn failed: {}", e);
            if let Ok((_, mut render_state)) = state_q.get_mut(pane) {
                render_state.current_frame = Some(error_frame(&format!("spawn failed: {}", e)));
            }
        }
    }
}

fn poll_widget_children(
    mut commands: Commands,
    mut q: Query<(
        Entity,
        &PaneKindMarker,
        &mut WidgetProcess,
        &mut WidgetRender,
        &Widget,
    )>,
) {
    for (entity, kind, mut wp, mut render_state, widget) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        match wp.child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code();
                eprintln!("[widget] exited code={:?}; respawning", code);
                commands
                    .entity(entity)
                    .remove::<WidgetProcess>()
                    .remove::<WidgetIO>();

                if widget.command.trim().is_empty() {
                    if render_state.current_frame.is_none() {
                        render_state.current_frame = Some(placeholder_frame());
                        render_state.pending_frame = None;
                        render_state.last_size = Vec2::ZERO;
                    }
                    continue;
                }

                // Force a fresh init/resize handshake so the new child
                // gets the current pane width on its first frame.
                render_state.init_sent = false;
                render_state.pending_frame = None;

                match spawn_widget_process(
                    &widget.command,
                    &widget.args,
                    widget.cwd.as_deref(),
                ) {
                    Ok((process, io)) => {
                        commands.entity(entity).insert((process, io));
                    }
                    Err(e) => {
                        eprintln!("[widget] respawn failed: {}", e);
                        render_state.current_frame =
                            Some(error_frame(&format!("respawn failed: {}", e)));
                        render_state.last_size = Vec2::ZERO;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("[widget] try_wait failed: {}", e);
            }
        }
    }
}

// ---------- Helpers ----------

fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        eprintln!("[widget] open_url unsupported on this OS: {}", url);
    }
}

fn placeholder_frame() -> Element {
    Element::Vstack {
        gap: 8.0,
        pad: 12.0,
        children: vec![
            Element::Text {
                value: "Widget not configured".into(),
                color: Some("#cc8".into()),
                size: Some(14.0),
                weight: Some(Weight::Bold),
            },
            Element::Text {
                value: format!("Set {} or save a snapshot with a command.", DEFAULT_CMD_ENV),
                color: Some("#888".into()),
                size: None,
                weight: None,
            },
        ],
    }
}

fn error_frame(msg: &str) -> Element {
    Element::Vstack {
        gap: 6.0,
        pad: 12.0,
        children: vec![
            Element::Text {
                value: "Widget error".into(),
                color: Some("#e55".into()),
                size: Some(14.0),
                weight: Some(Weight::Bold),
            },
            Element::Text {
                value: msg.into(),
                color: Some("#aaa".into()),
                size: None,
                weight: None,
            },
        ],
    }
}
