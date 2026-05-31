//! "Run command" button pane.
//!
//! Three visual modes (the chrome's title bar always shows the user-
//! given title and is rendered by pane-bevy):
//!
//! Draft (newly created or after double-click to edit):
//! ```text
//!     ┌──────────────────────────────────────────┐
//!     │  Title    [______________________]       │
//!     │  Command  [______________________]       │
//!     │                            [ Save ]      │
//!     └──────────────────────────────────────────┘
//! ```
//!
//! Saved, narrow content width — just the play button:
//! ```text
//!     ┌─────────┐
//!     │   ▶     │
//!     └─────────┘
//! ```
//!
//! Saved, wide content width — play + command + details disclosure:
//! ```text
//!     ┌──────────────────────────────────────────┐
//!     │  ▶  $ <command>                          │
//!     │  ─────────────────────────────────────── │
//!     │  ▸ Details                               │
//!     └──────────────────────────────────────────┘
//! ```
//!
//! Double-clicking anywhere except the play button re-enters draft mode
//! with the current title and command pre-populated.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextBounds};
use pane_bevy::{
    focus_text_input, spawn_text_input, FocusedTextInput, PaneContentNoClip, PaneContentPressed,
    PaneHotZones, PaneKindMarker, PaneKindSpec, PaneProject, PaneRect, PaneRegistry, PaneTag,
    PaneTitle, TextInput, TextInputEvent, TextInputStyle,
};
use serde_json::Value;

use crate::projects::Projects;
use crate::{MonoFont, MonoMetrics, FONT_SIZE};

pub const PANE_KIND: &str = "run-button";

const OUTPUT_LINES_CAP: usize = 2000;
const OUTPUT_LINE_HEIGHT: f32 = OUTPUT_FONT_SIZE * 1.45;

const DOUBLE_CLICK_SECS: f64 = 0.35;

// ---------- Components ----------

#[derive(Component, Debug)]
pub struct RunButton {
    pub cwd: PathBuf,
    pub status: RunStatus,
    pub command: String,
    /// True when the title/command form is showing instead of the saved
    /// view (newly-created panes start here; double-click re-enters).
    pub draft: bool,
    pub output_expanded: bool,
    pub output: VecDeque<String>,
    /// Index into `output` of the topmost line currently shown in the
    /// details panel. Held stable as the deque grows so the user keeps
    /// reading the same lines; decremented when `trim_output` pops from
    /// the front so the view stays anchored to the same content.
    pub output_scroll_top: usize,
    /// True while we should auto-scroll the details panel to keep the
    /// most recent line visible. Cleared when the user wheels up; reset
    /// when they wheel back to the bottom.
    pub output_follow_tail: bool,
    /// Time of most recent content press, for double-click detection.
    last_press_time: Option<f64>,
}

#[derive(Debug)]
pub enum RunStatus {
    Idle,
    Running { child: Child },
    Finished { success: bool, code: Option<i32> },
}

impl RunStatus {
    fn is_running(&self) -> bool {
        matches!(self, RunStatus::Running { .. })
    }
}

#[derive(Component)]
struct OutputStream(Mutex<Receiver<String>>);

#[derive(Component)]
struct RunButtonChrome {
    // --- Form (draft) entities ---
    title_label: Entity,
    title_input: Entity,
    title_input_bg: Entity,
    command_label: Entity,
    command_input: Entity,
    command_input_bg: Entity,
    cwd_label: Entity,
    cwd_input: Entity,
    cwd_input_bg: Entity,
    save_btn_bg: Entity,
    save_btn_text: Entity,

    // --- Saved entities ---
    play_btn: Entity,
    prompt_glyph: Entity,
    command_text: Entity,
    divider: Entity,
    details_toggle: Entity,
    output_text: Entity,
}

// ---------- Colors / sizes ----------

/// Theme-derived colors for one run-button render pass. Resolved
/// once at the top of `run_button_spawn_from_config` and
/// `sync_run_button_visual` so a preset switch retones every input,
/// label, divider, and status indicator in one shot.
struct RunButtonPalette {
    idle: Color,
    running: Color,
    success: Color,
    failed: Color,
    command_text: Color,
    prompt: Color,
    output: Color,
    divider: Color,
    form_label: Color,
    input_text: Color,
    input_focused: Color,
    input_bg: Color,
    caret: Color,
    selection: Color,
    save_bg: Color,
    save_text: Color,
    details_dim: Color,
}

fn run_button_palette(theme: &style_bevy::Theme) -> RunButtonPalette {
    use style_bevy::tokens as t;
    let c = |id| Color::LinearRgba(theme.color(id));
    RunButtonPalette {
        idle: c(t::STATUS_IDLE),
        running: c(t::STATUS_RUNNING),
        success: c(t::STATUS_SUCCESS),
        failed: c(t::STATUS_FAILED),
        command_text: c(t::FG),
        prompt: c(t::ACCENT),
        output: c(t::FG_MUTED),
        divider: c(t::CHROME_DIVIDER),
        form_label: c(t::FG_MUTED),
        input_text: c(t::INPUT_TEXT),
        input_focused: c(t::INPUT_TEXT_FOCUSED),
        input_bg: c(t::INPUT_BG),
        caret: c(t::CARET),
        selection: c(t::SELECTION),
        save_bg: c(t::BUTTON_PRIMARY_BG),
        save_text: c(t::BUTTON_PRIMARY_LABEL),
        details_dim: c(t::FG_MUTED),
    }
}

const ICON_FONT_SIZE: f32 = 22.0;
const COMMAND_FONT_SIZE: f32 = 13.0;
const OUTPUT_FONT_SIZE: f32 = 11.0;
const FORM_LABEL_FONT_SIZE: f32 = 12.0;
const FORM_INPUT_FONT_SIZE: f32 = 13.0;
const SAVE_FONT_SIZE: f32 = 13.0;
const DETAILS_FONT_SIZE: f32 = 12.0;

// --- Saved (wide) layout ---
const ROW_Y: f32 = 4.0;
const ROW_H: f32 = 28.0;
const PLAY_X: f32 = 6.0;
const PLAY_W: f32 = 28.0;
const PROMPT_X: f32 = 38.0;
const COMMAND_X: f32 = 54.0;
const COMMAND_TEXT_Y: f32 = ROW_Y + 6.0;
const DIVIDER_Y: f32 = ROW_Y + ROW_H + 2.0;
const DETAILS_TOGGLE_Y: f32 = DIVIDER_Y + 4.0;
const DETAILS_TOGGLE_X: f32 = 8.0;
const DETAILS_TOGGLE_H: f32 = 22.0;
const DETAILS_TOGGLE_HIT_W: f32 = 90.0;
const OUTPUT_Y: f32 = DETAILS_TOGGLE_Y + DETAILS_TOGGLE_H + 2.0;

/// Below this content width the saved view collapses to just the play
/// button. Chosen to roughly match the play+arrow+a few command chars.
const SMALL_WIDTH_THRESHOLD: f32 = 180.0;

// --- Form layout ---
const FORM_PAD_X: f32 = 12.0;
const FORM_PAD_Y_TOP: f32 = 10.0;
const FORM_LABEL_W: f32 = 64.0;
const FORM_ROW_H: f32 = 24.0;
const FORM_ROW_GAP: f32 = 8.0;
const FORM_INPUT_PAD_X: f32 = 6.0;
const FORM_INPUT_PAD_Y: f32 = 4.0;

const TITLE_LABEL_Y: f32 = FORM_PAD_Y_TOP;
const TITLE_INPUT_Y: f32 = TITLE_LABEL_Y;
const COMMAND_LABEL_Y: f32 = TITLE_LABEL_Y + FORM_ROW_H + FORM_ROW_GAP;
const COMMAND_INPUT_Y: f32 = COMMAND_LABEL_Y;
const CWD_LABEL_Y: f32 = COMMAND_LABEL_Y + FORM_ROW_H + FORM_ROW_GAP;
const CWD_INPUT_Y: f32 = CWD_LABEL_Y;
const SAVE_BTN_Y: f32 = CWD_LABEL_Y + FORM_ROW_H + FORM_ROW_GAP;

const SAVE_BTN_W: f32 = 70.0;
const SAVE_BTN_H: f32 = 24.0;
const SAVE_BTN_RIGHT_PAD: f32 = 12.0;

/// The TextInput's caret is rendered at `col * cell_width`, where
/// `cell_width` was measured for the host font at `FONT_SIZE`. Inputs
/// render at a smaller size, so we scale.
fn input_cell_width(measured: f32, font_size: f32) -> f32 {
    measured * (font_size / FONT_SIZE)
}

// ---------- Plugin / registry ----------

pub struct RunButtonPlugin;

impl Plugin for RunButtonPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, register_run_button_kind)
            .add_systems(
                Update,
                (
                    handle_run_button_press,
                    handle_text_input_events,
                    scroll_run_button_output,
                    drain_output_streams,
                    poll_run_button_children,
                    sync_run_button_visual,
                    update_run_button_hot_zones,
                )
                    .chain(),
            );
    }
}

fn register_run_button_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Run",
        radial_icon: Some("▶"),
        default_size: Vec2::new(380.0, 188.0),
        spawn: run_button_spawn_from_config,
        snapshot: run_button_snapshot,
        on_close: Some(run_button_on_close),
    });
}

fn default_cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn run_button_spawn_from_config(
    world: &mut World,
    entity: Entity,
    content_root: Entity,
    config: &Value,
) {
    let palette = run_button_palette(
        world
            .get_resource::<style_bevy::Theme>()
            .expect("Theme resource missing; StylePlugin must run before spawning a RunButton"),
    );
    // A snapshot always emits "command"; its absence means this is a
    // freshly-created pane that should start in draft mode.
    let is_restore = config.get("command").is_some();

    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Run")
        .to_string();
    let command = config
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let cwd = config
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .or_else(|| {
            // New (non-restore) pane in a project: default to the
            // project's remembered root, so a Run pane drops into the
            // same place a fresh terminal would. Restores always honor
            // the snapshotted cwd above.
            let pid = world.get::<PaneProject>(entity).map(|p| p.0)?;
            world
                .get_resource::<Projects>()?
                .default_cwd_of(pid)
                .map(PathBuf::from)
        })
        .unwrap_or_else(default_cwd);

    let font = world
        .get_resource::<pane_bevy::PaneFont>()
        .map(|f| f.0.clone())
        .or_else(|| world.get_resource::<MonoFont>().map(|f| f.0.clone()))
        .expect("PaneFont or MonoFont must be present before spawning a RunButton");
    let measured_cell = world
        .get_resource::<MonoMetrics>()
        .map(|m| m.cell_width)
        .expect("MonoMetrics must be present");

    // ----- Form: title row -----
    let title_label = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("Title"),
            TextFont {
                font: font.clone(),
                font_size: FORM_LABEL_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(FORM_LABEL_FONT_SIZE * 1.4),
            TextColor(palette.form_label),
            Anchor::TOP_LEFT,
            Transform::from_xyz(FORM_PAD_X, -(TITLE_LABEL_Y + FORM_INPUT_PAD_Y), 0.0),
            Visibility::Hidden,
        ))
        .id();

    let title_input_bg = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: palette.input_bg,
                custom_size: Some(Vec2::new(200.0, FORM_ROW_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(FORM_PAD_X + FORM_LABEL_W, -TITLE_INPUT_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let title_input = {
        let style = TextInputStyle {
            font: font.clone(),
            font_size: FORM_INPUT_FONT_SIZE,
            line_height: FORM_INPUT_FONT_SIZE * 1.4,
            cell_width: input_cell_width(measured_cell, FORM_INPUT_FONT_SIZE),
            color_idle: palette.input_text,
            color_focused: palette.input_focused,
            color_caret: palette.caret,
            color_selection: palette.selection,
        };
        let mut commands_queue = world.commands();
        spawn_text_input(
            &mut commands_queue,
            content_root,
            &title,
            style,
            200.0,
            Transform::from_xyz(
                FORM_PAD_X + FORM_LABEL_W + FORM_INPUT_PAD_X,
                -(TITLE_INPUT_Y + FORM_INPUT_PAD_Y),
                0.1,
            ),
        )
    };
    world.flush();
    if let Ok(mut e) = world.get_entity_mut(title_input) {
        e.insert(Visibility::Hidden);
    }

    // ----- Form: command row -----
    let command_label = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("Command"),
            TextFont {
                font: font.clone(),
                font_size: FORM_LABEL_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(FORM_LABEL_FONT_SIZE * 1.4),
            TextColor(palette.form_label),
            Anchor::TOP_LEFT,
            Transform::from_xyz(FORM_PAD_X, -(COMMAND_LABEL_Y + FORM_INPUT_PAD_Y), 0.0),
            Visibility::Hidden,
        ))
        .id();

    let command_input_bg = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: palette.input_bg,
                custom_size: Some(Vec2::new(200.0, FORM_ROW_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(FORM_PAD_X + FORM_LABEL_W, -COMMAND_INPUT_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let command_input = {
        let style = TextInputStyle {
            font: font.clone(),
            font_size: FORM_INPUT_FONT_SIZE,
            line_height: FORM_INPUT_FONT_SIZE * 1.4,
            cell_width: input_cell_width(measured_cell, FORM_INPUT_FONT_SIZE),
            color_idle: palette.input_text,
            color_focused: palette.input_focused,
            color_caret: palette.caret,
            color_selection: palette.selection,
        };
        let mut commands_queue = world.commands();
        spawn_text_input(
            &mut commands_queue,
            content_root,
            &command,
            style,
            200.0,
            Transform::from_xyz(
                FORM_PAD_X + FORM_LABEL_W + FORM_INPUT_PAD_X,
                -(COMMAND_INPUT_Y + FORM_INPUT_PAD_Y),
                0.1,
            ),
        )
    };
    world.flush();
    if let Ok(mut e) = world.get_entity_mut(command_input) {
        e.insert(Visibility::Hidden);
    }

    // ----- Form: cwd row -----
    let cwd_label = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("CWD"),
            TextFont {
                font: font.clone(),
                font_size: FORM_LABEL_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(FORM_LABEL_FONT_SIZE * 1.4),
            TextColor(palette.form_label),
            Anchor::TOP_LEFT,
            Transform::from_xyz(FORM_PAD_X, -(CWD_LABEL_Y + FORM_INPUT_PAD_Y), 0.0),
            Visibility::Hidden,
        ))
        .id();

    let cwd_input_bg = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: palette.input_bg,
                custom_size: Some(Vec2::new(200.0, FORM_ROW_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(FORM_PAD_X + FORM_LABEL_W, -CWD_INPUT_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let cwd_input = {
        let style = TextInputStyle {
            font: font.clone(),
            font_size: FORM_INPUT_FONT_SIZE,
            line_height: FORM_INPUT_FONT_SIZE * 1.4,
            cell_width: input_cell_width(measured_cell, FORM_INPUT_FONT_SIZE),
            color_idle: palette.input_text,
            color_focused: palette.input_focused,
            color_caret: palette.caret,
            color_selection: palette.selection,
        };
        let mut commands_queue = world.commands();
        spawn_text_input(
            &mut commands_queue,
            content_root,
            &cwd.to_string_lossy(),
            style,
            200.0,
            Transform::from_xyz(
                FORM_PAD_X + FORM_LABEL_W + FORM_INPUT_PAD_X,
                -(CWD_INPUT_Y + FORM_INPUT_PAD_Y),
                0.1,
            ),
        )
    };
    world.flush();
    if let Ok(mut e) = world.get_entity_mut(cwd_input) {
        e.insert(Visibility::Hidden);
    }

    // ----- Form: save button -----
    let save_btn_bg = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: palette.save_bg,
                custom_size: Some(Vec2::new(SAVE_BTN_W, SAVE_BTN_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -SAVE_BTN_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let save_btn_text = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("Save"),
            TextFont {
                font: font.clone(),
                font_size: SAVE_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(SAVE_FONT_SIZE * 1.4),
            TextColor(palette.save_text),
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -(SAVE_BTN_Y + 4.0), 0.1),
            Visibility::Hidden,
        ))
        .id();

    // ----- Saved: play button -----
    let play_btn = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("▶"),
            TextFont {
                font: font.clone(),
                font_size: ICON_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(ICON_FONT_SIZE),
            TextColor(palette.idle),
            Anchor::TOP_LEFT,
            Transform::from_xyz(PLAY_X, -ROW_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let prompt_glyph = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("$"),
            TextFont {
                font: font.clone(),
                font_size: COMMAND_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(COMMAND_FONT_SIZE * 1.4),
            TextColor(palette.prompt),
            Anchor::TOP_LEFT,
            Transform::from_xyz(PROMPT_X, -COMMAND_TEXT_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let command_text = world
        .spawn((
            ChildOf(content_root),
            Text2d::new(command.clone()),
            TextFont {
                font: font.clone(),
                font_size: COMMAND_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(COMMAND_FONT_SIZE * 1.4),
            TextColor(palette.command_text),
            Anchor::TOP_LEFT,
            Transform::from_xyz(COMMAND_X, -COMMAND_TEXT_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let divider = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: palette.divider,
                custom_size: Some(Vec2::new(800.0, 1.0)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -DIVIDER_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let details_toggle = world
        .spawn((
            ChildOf(content_root),
            Text2d::new("\u{25B8} Details"),
            TextFont {
                font: font.clone(),
                font_size: DETAILS_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(DETAILS_FONT_SIZE * 1.4),
            TextColor(palette.details_dim),
            Anchor::TOP_LEFT,
            Transform::from_xyz(DETAILS_TOGGLE_X, -DETAILS_TOGGLE_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    let output_text = world
        .spawn((
            ChildOf(content_root),
            Text2d::new(String::new()),
            TextFont {
                font,
                font_size: OUTPUT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(OUTPUT_FONT_SIZE * 1.45),
            TextColor(palette.output),
            Anchor::TOP_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            TextBounds {
                width: Some(0.0),
                height: Some(0.0),
            },
            // sync_run_button_visual manages our TextBounds explicitly
            // so we get whole-line vertical clipping (no half-line
            // bleed below the pane) and so per-line truncation isn't
            // re-wrapped by the auto-bounds layout.
            PaneContentNoClip,
            Transform::from_xyz(PLAY_X, -OUTPUT_Y, 0.0),
            Visibility::Hidden,
        ))
        .id();

    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title;
    }

    world.entity_mut(entity).insert((
        RunButton {
            cwd,
            status: RunStatus::Idle,
            command,
            draft: !is_restore,
            output_expanded: false,
            output: VecDeque::new(),
            output_scroll_top: 0,
            output_follow_tail: true,
            last_press_time: None,
        },
        RunButtonChrome {
            title_label,
            title_input,
            title_input_bg,
            command_label,
            command_input,
            command_input_bg,
            cwd_label,
            cwd_input,
            cwd_input_bg,
            save_btn_bg,
            save_btn_text,
            play_btn,
            prompt_glyph,
            command_text,
            divider,
            details_toggle,
            output_text,
        },
    ));
}

fn run_button_snapshot(world: &World, entity: Entity) -> Value {
    let Some(rb) = world.get::<RunButton>(entity) else {
        return Value::Null;
    };
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    serde_json::json!({
        "title": title,
        "command": rb.command,
        "cwd": rb.cwd.to_string_lossy(),
    })
}

fn run_button_on_close(world: &mut World, entity: Entity) {
    if let Some(mut rb) = world.get_mut::<RunButton>(entity)
        && let RunStatus::Running { child } = &mut rb.status
    {
        let _ = child.kill();
    }
}

// ---------- Geometry helpers ----------

fn save_btn_x(content_w: f32) -> f32 {
    (content_w - SAVE_BTN_RIGHT_PAD - SAVE_BTN_W).max(FORM_PAD_X)
}

fn input_box_width(content_w: f32) -> f32 {
    (content_w - FORM_PAD_X - FORM_LABEL_W - FORM_PAD_X).max(40.0)
}

fn hit_rect(local: Vec2, x0: f32, y0: f32, w: f32, h: f32) -> bool {
    local.x >= x0 && local.x < x0 + w && local.y >= y0 && local.y < y0 + h
}

fn hit_play_button(local: Vec2) -> bool {
    hit_rect(local, 0.0, ROW_Y, PLAY_X + PLAY_W, ROW_H)
}

fn hit_save_button(local: Vec2, content_w: f32) -> bool {
    hit_rect(local, save_btn_x(content_w), SAVE_BTN_Y, SAVE_BTN_W, SAVE_BTN_H)
}

fn hit_title_input_row(local: Vec2, content_w: f32) -> bool {
    hit_rect(
        local,
        FORM_PAD_X + FORM_LABEL_W,
        TITLE_INPUT_Y,
        input_box_width(content_w),
        FORM_ROW_H,
    )
}

fn hit_command_input_row(local: Vec2, content_w: f32) -> bool {
    hit_rect(
        local,
        FORM_PAD_X + FORM_LABEL_W,
        COMMAND_INPUT_Y,
        input_box_width(content_w),
        FORM_ROW_H,
    )
}

fn hit_cwd_input_row(local: Vec2, content_w: f32) -> bool {
    hit_rect(
        local,
        FORM_PAD_X + FORM_LABEL_W,
        CWD_INPUT_Y,
        input_box_width(content_w),
        FORM_ROW_H,
    )
}

fn hit_details_toggle(local: Vec2) -> bool {
    hit_rect(
        local,
        DETAILS_TOGGLE_X,
        DETAILS_TOGGLE_Y,
        DETAILS_TOGGLE_HIT_W,
        DETAILS_TOGGLE_H,
    )
}

/// Publish the run-button's clickable regions to `PaneHotZones` each
/// frame so pinned panes can still receive presses on play / save /
/// inputs / details-toggle. Rects are in content-local coords (same
/// frame as `PaneContentPressed.local_pt`).
fn update_run_button_hot_zones(
    mut q: Query<(&PaneKindMarker, &PaneRect, &RunButton, &mut PaneHotZones)>,
) {
    for (kind, rect, rb, mut zones) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        zones.clear();
        let content_w = (rect.size.x - 2.0 * pane_bevy::MARGIN).max(0.0);
        let is_small = content_w < SMALL_WIDTH_THRESHOLD;
        if rb.draft {
            // Form mode: save button + both input rows.
            zones.push(Rect::from_corners(
                Vec2::new(save_btn_x(content_w), SAVE_BTN_Y),
                Vec2::new(save_btn_x(content_w) + SAVE_BTN_W, SAVE_BTN_Y + SAVE_BTN_H),
            ));
            let input_x0 = FORM_PAD_X + FORM_LABEL_W;
            let input_w = input_box_width(content_w);
            zones.push(Rect::from_corners(
                Vec2::new(input_x0, TITLE_INPUT_Y),
                Vec2::new(input_x0 + input_w, TITLE_INPUT_Y + FORM_ROW_H),
            ));
            zones.push(Rect::from_corners(
                Vec2::new(input_x0, COMMAND_INPUT_Y),
                Vec2::new(input_x0 + input_w, COMMAND_INPUT_Y + FORM_ROW_H),
            ));
            zones.push(Rect::from_corners(
                Vec2::new(input_x0, CWD_INPUT_Y),
                Vec2::new(input_x0 + input_w, CWD_INPUT_Y + FORM_ROW_H),
            ));
        } else {
            // Saved mode: play button always, details toggle when wide.
            zones.push(Rect::from_corners(
                Vec2::new(0.0, ROW_Y),
                Vec2::new(PLAY_X + PLAY_W, ROW_Y + ROW_H),
            ));
            if !is_small {
                zones.push(Rect::from_corners(
                    Vec2::new(DETAILS_TOGGLE_X, DETAILS_TOGGLE_Y),
                    Vec2::new(
                        DETAILS_TOGGLE_X + DETAILS_TOGGLE_HIT_W,
                        DETAILS_TOGGLE_Y + DETAILS_TOGGLE_H,
                    ),
                ));
            }
        }
    }
}

// ---------- Press dispatch ----------

#[allow(clippy::too_many_arguments)]
fn handle_run_button_press(
    mut presses: MessageReader<PaneContentPressed>,
    mut commands: Commands,
    time: Res<Time>,
    mut focused: ResMut<FocusedTextInput>,
    kinds: Query<&PaneKindMarker>,
    rects: Query<&pane_bevy::PaneRect>,
    mut rbs: Query<(&mut RunButton, &RunButtonChrome)>,
    mut text_inputs: Query<&mut TextInput>,
    mut titles: Query<&mut PaneTitle>,
) {
    for ev in presses.read() {
        let Ok(kind) = kinds.get(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        let Ok((mut rb, chrome)) = rbs.get_mut(ev.pane) else {
            continue;
        };
        let Ok(rect) = rects.get(ev.pane) else {
            continue;
        };
        let content_w = (rect.size.x - 2.0 * pane_bevy::MARGIN).max(0.0);
        let local = ev.local_pt;
        let is_small = content_w < SMALL_WIDTH_THRESHOLD;

        if rb.draft {
            handle_form_press(
                &mut rb,
                chrome,
                ev.pane,
                local,
                content_w,
                &mut commands,
                &mut focused,
                &mut text_inputs,
                &mut titles,
            );
            // Always reset the double-click clock once we leave saved mode.
            rb.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }

        // --- Saved mode dispatch ---

        // Play button always wins.
        if hit_play_button(local) {
            focus_text_input(&mut commands, &mut focused, [], None);
            toggle_run(&mut rb, ev.pane, &mut commands);
            rb.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }

        // Details toggle (wide only).
        if !is_small && hit_details_toggle(local) {
            rb.output_expanded = !rb.output_expanded;
            focus_text_input(&mut commands, &mut focused, [], None);
            rb.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }

        // Pinned panes don't get the double-click re-edit affordance —
        // they're background decoration, only their hot-zones fire.
        if ev.pinned {
            continue;
        }

        // Anywhere else: maybe a double-click → re-edit.
        let now = time.elapsed_secs_f64();
        let is_double = rb
            .last_press_time
            .is_some_and(|t| now - t < DOUBLE_CLICK_SECS);
        rb.last_press_time = Some(now);
        if is_double {
            enter_draft(
                &mut rb,
                chrome,
                &mut commands,
                &mut focused,
                &mut text_inputs,
                &mut titles,
                ev.pane,
            );
        } else {
            focus_text_input(&mut commands, &mut focused, [], None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_form_press(
    rb: &mut RunButton,
    chrome: &RunButtonChrome,
    pane: Entity,
    local: Vec2,
    content_w: f32,
    commands: &mut Commands,
    focused: &mut FocusedTextInput,
    text_inputs: &mut Query<&mut TextInput>,
    titles: &mut Query<&mut PaneTitle>,
) {
    if hit_save_button(local, content_w) {
        commit_form(rb, chrome, pane, commands, focused, text_inputs, titles);
        return;
    }

    if hit_title_input_row(local, content_w) {
        if let Ok(mut ti) = text_inputs.get_mut(chrome.title_input) {
            let local_x =
                (local.x - (FORM_PAD_X + FORM_LABEL_W + FORM_INPUT_PAD_X)).max(0.0);
            pane_bevy::click_to_caret(&mut ti, local_x);
        }
        focus_text_input(commands, focused, [], Some(chrome.title_input));
        return;
    }

    if hit_command_input_row(local, content_w) {
        if let Ok(mut ti) = text_inputs.get_mut(chrome.command_input) {
            let local_x =
                (local.x - (FORM_PAD_X + FORM_LABEL_W + FORM_INPUT_PAD_X)).max(0.0);
            pane_bevy::click_to_caret(&mut ti, local_x);
        }
        focus_text_input(commands, focused, [], Some(chrome.command_input));
        return;
    }

    if hit_cwd_input_row(local, content_w) {
        if let Ok(mut ti) = text_inputs.get_mut(chrome.cwd_input) {
            let local_x =
                (local.x - (FORM_PAD_X + FORM_LABEL_W + FORM_INPUT_PAD_X)).max(0.0);
            pane_bevy::click_to_caret(&mut ti, local_x);
        }
        focus_text_input(commands, focused, [], Some(chrome.cwd_input));
        return;
    }

    focus_text_input(commands, focused, [], None);
}

fn enter_draft(
    rb: &mut RunButton,
    chrome: &RunButtonChrome,
    commands: &mut Commands,
    focused: &mut FocusedTextInput,
    text_inputs: &mut Query<&mut TextInput>,
    titles: &mut Query<&mut PaneTitle>,
    pane: Entity,
) {
    let title_now = titles
        .get(pane)
        .map(|t| t.0.clone())
        .unwrap_or_else(|_| "Run".to_string());
    if let Ok(mut ti) = text_inputs.get_mut(chrome.title_input) {
        ti.set_text(&title_now);
    }
    if let Ok(mut ti) = text_inputs.get_mut(chrome.command_input) {
        ti.set_text(&rb.command);
    }
    if let Ok(mut ti) = text_inputs.get_mut(chrome.cwd_input) {
        ti.set_text(&rb.cwd.to_string_lossy());
    }
    rb.draft = true;
    focus_text_input(commands, focused, [], Some(chrome.command_input));
}

fn commit_form(
    rb: &mut RunButton,
    chrome: &RunButtonChrome,
    pane: Entity,
    commands: &mut Commands,
    focused: &mut FocusedTextInput,
    text_inputs: &mut Query<&mut TextInput>,
    titles: &mut Query<&mut PaneTitle>,
) {
    let new_title = text_inputs
        .get(chrome.title_input)
        .map(|ti| ti.text())
        .unwrap_or_default();
    let new_title = if new_title.trim().is_empty() {
        "Run".to_string()
    } else {
        new_title
    };
    let new_command = text_inputs
        .get(chrome.command_input)
        .map(|ti| ti.text())
        .unwrap_or_default();
    let new_cwd = text_inputs
        .get(chrome.cwd_input)
        .map(|ti| ti.text())
        .unwrap_or_default();

    rb.command = new_command;
    let trimmed_cwd = new_cwd.trim();
    if !trimmed_cwd.is_empty() {
        rb.cwd = PathBuf::from(trimmed_cwd);
    }
    rb.draft = false;
    if let Ok(mut t) = titles.get_mut(pane) {
        if t.0 != new_title {
            t.0 = new_title;
        }
    } else {
        commands.entity(pane).insert(PaneTitle(new_title));
    }
    focus_text_input(commands, focused, [], None);
}

/// Enter on either input commits; Esc reverts and exits draft only when
/// there's a previously-saved command (otherwise the user has nothing
/// to revert to and we keep the form open).
fn handle_text_input_events(
    mut events: MessageReader<TextInputEvent>,
    mut commands: Commands,
    mut focused: ResMut<FocusedTextInput>,
    mut rbs: Query<(Entity, &mut RunButton, &RunButtonChrome)>,
    mut text_inputs: Query<&mut TextInput>,
    mut titles: Query<&mut PaneTitle>,
) {
    for ev in events.read() {
        match ev {
            TextInputEvent::Submit { entity } => {
                let Some(pane) = find_pane_for_input(&rbs, *entity) else {
                    continue;
                };
                let Ok((_, mut rb, chrome)) = rbs.get_mut(pane) else {
                    continue;
                };
                if !rb.draft {
                    continue;
                }
                if *entity == chrome.title_input {
                    // Tab-like: jump to command input.
                    focus_text_input(
                        &mut commands,
                        &mut focused,
                        [],
                        Some(chrome.command_input),
                    );
                } else if *entity == chrome.command_input {
                    focus_text_input(
                        &mut commands,
                        &mut focused,
                        [],
                        Some(chrome.cwd_input),
                    );
                } else {
                    commit_form(
                        &mut rb,
                        chrome,
                        pane,
                        &mut commands,
                        &mut focused,
                        &mut text_inputs,
                        &mut titles,
                    );
                }
            }
            TextInputEvent::Cancel { entity } => {
                let Some(pane) = find_pane_for_input(&rbs, *entity) else {
                    continue;
                };
                let Ok((_, mut rb, chrome)) = rbs.get_mut(pane) else {
                    continue;
                };
                if !rb.draft || rb.command.trim().is_empty() {
                    // Nothing to revert to — keep the form open.
                    continue;
                }
                let saved_title = titles
                    .get(pane)
                    .map(|t| t.0.clone())
                    .unwrap_or_else(|_| "Run".to_string());
                if let Ok(mut ti) = text_inputs.get_mut(chrome.title_input) {
                    ti.set_text(&saved_title);
                }
                if let Ok(mut ti) = text_inputs.get_mut(chrome.command_input) {
                    ti.set_text(&rb.command);
                }
                if let Ok(mut ti) = text_inputs.get_mut(chrome.cwd_input) {
                    ti.set_text(&rb.cwd.to_string_lossy());
                }
                rb.draft = false;
                focus_text_input(&mut commands, &mut focused, [], None);
            }
            TextInputEvent::Changed { .. } => {}
        }
    }
}

fn find_pane_for_input(
    rbs: &Query<(Entity, &mut RunButton, &RunButtonChrome)>,
    input: Entity,
) -> Option<Entity> {
    for (pane, _, chrome) in rbs.iter() {
        if chrome.title_input == input
            || chrome.command_input == input
            || chrome.cwd_input == input
        {
            return Some(pane);
        }
    }
    None
}

fn toggle_run(rb: &mut RunButton, pane: Entity, commands: &mut Commands) {
    if rb.status.is_running() {
        if let RunStatus::Running { child } = &mut rb.status {
            let _ = child.kill();
        }
        return;
    }
    let command = rb.command.clone();
    if command.trim().is_empty() {
        eprintln!("[run-button] empty command — ignoring run");
        return;
    }
    eprintln!("[run-button] $ {}  (cwd={})", command, rb.cwd.display());
    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(&command)
        .current_dir(&rb.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[run-button] spawn failed: {}", e);
            rb.status = RunStatus::Finished {
                success: false,
                code: None,
            };
            rb.output.push_back(format!("error: {}", e));
            trim_output(&mut *rb);
            return;
        }
    };
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let (tx, rx) = mpsc::channel::<String>();
    let tx_err = tx.clone();
    std::thread::spawn(move || stream_lines(stdout, tx));
    std::thread::spawn(move || stream_lines(stderr, tx_err));
    rb.status = RunStatus::Running { child };
    rb.output.clear();
    rb.output_scroll_top = 0;
    rb.output_follow_tail = true;
    commands.entity(pane).insert(OutputStream(Mutex::new(rx)));
}

fn stream_lines<R: Read + Send + 'static>(reader: R, tx: Sender<String>) {
    let buf = BufReader::new(reader);
    for line in buf.lines() {
        match line {
            Ok(s) => {
                if tx.send(s).is_err() {
                    return;
                }
            }
            Err(_) => return,
        }
    }
}

fn trim_output(rb: &mut RunButton) {
    while rb.output.len() > OUTPUT_LINES_CAP {
        rb.output.pop_front();
        // Keep the scrolled-up view anchored to the same content as
        // older lines fall off the front of the buffer.
        if !rb.output_follow_tail {
            rb.output_scroll_top = rb.output_scroll_top.saturating_sub(1);
        }
    }
}

/// How many output lines fit in the details panel given the current
/// pane height. Used by both the visual sync and the scroll handler so
/// they agree on the viewport size.
fn output_lines_capacity(rect: &PaneRect) -> usize {
    let content_h = (rect.size.y - pane_bevy::TITLE_H - 2.0 * pane_bevy::MARGIN).max(0.0);
    let avail = (content_h - OUTPUT_Y).max(0.0);
    (avail / OUTPUT_LINE_HEIGHT).floor() as usize
}

// ---------- Output draining ----------

fn drain_output_streams(mut q: Query<(&mut RunButton, &OutputStream)>) {
    for (mut rb, stream) in &mut q {
        let Ok(rx) = stream.0.lock() else {
            continue;
        };
        let mut got_any = false;
        while let Ok(line) = rx.try_recv() {
            rb.output.push_back(line);
            got_any = true;
        }
        if got_any {
            trim_output(&mut *rb);
        }
    }
}

// ---------- Child polling ----------

fn poll_run_button_children(
    mut commands: Commands,
    mut buttons: Query<(Entity, &mut RunButton, Option<&OutputStream>)>,
) {
    for (entity, mut rb, stream) in &mut buttons {
        let RunStatus::Running { child } = &mut rb.status else {
            continue;
        };
        match child.try_wait() {
            Ok(Some(status)) => {
                let success = status.success();
                let code = status.code();
                eprintln!("[run-button] exited code={:?} success={}", code, success);
                rb.status = RunStatus::Finished { success, code };
                if let Some(s) = stream
                    && let Ok(rx) = s.0.lock()
                {
                    while let Ok(line) = rx.try_recv() {
                        rb.output.push_back(line);
                    }
                    trim_output(&mut *rb);
                }
                commands.entity(entity).remove::<OutputStream>();
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("[run-button] try_wait failed: {}", e);
                rb.status = RunStatus::Finished {
                    success: false,
                    code: None,
                };
                commands.entity(entity).remove::<OutputStream>();
            }
        }
    }
}

// ---------- Visual sync ----------

fn sync_run_button_visual(
    theme: Res<style_bevy::Theme>,
    // Dropped the Or<Changed<...>> filter so theme changes also retone
    // the run button in the same frame. The body work is cheap (few
    // sprite color compares per pane); cost is amortized by Bevy
    // running in Reactive mode anyway when nothing's happening.
    buttons: Query<(&RunButton, &RunButtonChrome, &pane_bevy::PaneRect)>,
    mut text_q: Query<&mut Text2d>,
    mut color_q: Query<&mut TextColor>,
    mut sprite_q: Query<&mut Sprite>,
    mut transform_q: Query<&mut Transform>,
    mut vis_q: Query<&mut Visibility>,
    mut input_q: Query<&mut TextInput>,
    mut bounds_q: Query<&mut TextBounds>,
) {
    let palette = run_button_palette(&theme);
    for (rb, chrome, rect) in &buttons {
        let content_w = (rect.size.x - 2.0 * pane_bevy::MARGIN).max(0.0);
        let is_small = content_w < SMALL_WIDTH_THRESHOLD;
        let draft = rb.draft;
        let show_output = rb.output_expanded && !rb.output.is_empty() && !is_small;

        // --- Form visibility ---
        let form_vis = if draft { Visibility::Inherited } else { Visibility::Hidden };
        for e in [
            chrome.title_label,
            chrome.title_input,
            chrome.title_input_bg,
            chrome.command_label,
            chrome.command_input,
            chrome.command_input_bg,
            chrome.cwd_label,
            chrome.cwd_input,
            chrome.cwd_input_bg,
            chrome.save_btn_bg,
            chrome.save_btn_text,
        ] {
            if let Ok(mut v) = vis_q.get_mut(e) {
                *v = form_vis;
            }
        }

        // Input box widths grow with the pane.
        let box_w = input_box_width(content_w);
        if let Ok(mut s) = sprite_q.get_mut(chrome.title_input_bg) {
            s.custom_size = Some(Vec2::new(box_w, FORM_ROW_H));
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.command_input_bg) {
            s.custom_size = Some(Vec2::new(box_w, FORM_ROW_H));
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.cwd_input_bg) {
            s.custom_size = Some(Vec2::new(box_w, FORM_ROW_H));
        }
        let inner_w = (box_w - 2.0 * FORM_INPUT_PAD_X).max(20.0);
        if let Ok(mut ti) = input_q.get_mut(chrome.title_input) {
            ti.width = inner_w;
        }
        if let Ok(mut ti) = input_q.get_mut(chrome.command_input) {
            ti.width = inner_w;
        }
        if let Ok(mut ti) = input_q.get_mut(chrome.cwd_input) {
            ti.width = inner_w;
        }

        // Save button position (right-aligned).
        let sx = save_btn_x(content_w);
        let save_center_x = sx + SAVE_BTN_W * 0.5;
        if let Ok(mut tr) = transform_q.get_mut(chrome.save_btn_bg) {
            tr.translation.x = sx;
        }
        if let Ok(mut tr) = transform_q.get_mut(chrome.save_btn_text) {
            // Approx-center "Save" text within the button. Text2d
            // anchored top-left, so push right by ~half the text width.
            tr.translation.x = save_center_x - 16.0;
        }

        // --- Saved-mode visibility ---
        let saved = !draft;
        let show_play = saved;
        let show_command_row = saved && !is_small;
        let show_details_toggle = saved && !is_small;
        let show_divider = show_command_row;

        let action_color = match &rb.status {
            RunStatus::Idle => palette.idle,
            RunStatus::Running { .. } => palette.running,
            RunStatus::Finished { success: true, .. } => palette.success,
            RunStatus::Finished { success: false, .. } => palette.failed,
        };
        let action_glyph = if rb.status.is_running() {
            "\u{25A0}"
        } else {
            "▶"
        };
        if let Ok(mut t) = text_q.get_mut(chrome.play_btn) {
            if t.0 != action_glyph {
                t.0 = action_glyph.into();
            }
        }
        if let Ok(mut c) = color_q.get_mut(chrome.play_btn) {
            c.0 = action_color;
        }
        // Center the play button in small mode.
        if let Ok(mut tr) = transform_q.get_mut(chrome.play_btn) {
            tr.translation.x = if is_small {
                ((content_w - PLAY_W) * 0.5).max(PLAY_X)
            } else {
                PLAY_X
            };
        }
        set_vis(&mut vis_q, chrome.play_btn, show_play);

        set_vis(&mut vis_q, chrome.prompt_glyph, show_command_row);
        set_vis(&mut vis_q, chrome.command_text, show_command_row);
        set_vis(&mut vis_q, chrome.divider, show_divider);
        set_vis(&mut vis_q, chrome.details_toggle, show_details_toggle);

        if show_command_row
            && let Ok(mut t) = text_q.get_mut(chrome.command_text)
            && t.0 != rb.command
        {
            t.0 = rb.command.clone();
        }

        let toggle_glyph = if show_output { "\u{25BE} Details" } else { "\u{25B8} Details" };
        if let Ok(mut t) = text_q.get_mut(chrome.details_toggle) {
            if t.0 != toggle_glyph {
                t.0 = toggle_glyph.into();
            }
        }

        if let Ok(mut s) = sprite_q.get_mut(chrome.divider) {
            s.custom_size = Some(Vec2::new(content_w, 1.0));
        }

        // Output (only visible when expanded).
        set_vis(&mut vis_q, chrome.output_text, show_output);
        let cap = output_lines_capacity(rect);
        let n = rb.output.len();
        let visible = n.min(cap);
        let max_top = n - visible;
        let top = if rb.output_follow_tail {
            max_top
        } else {
            rb.output_scroll_top.min(max_top)
        };

        // Horizontal overflow is clipped by the per-pane camera
        // viewport (see pane-bevy's top-of-file docs), so we no longer
        // pre-truncate each line — just join and let the renderer cut.
        let joined = rb
            .output
            .iter()
            .skip(top)
            .take(visible)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if let Ok(mut t) = text_q.get_mut(chrome.output_text)
            && t.0 != joined
        {
            t.0 = joined;
        }

        // Snap TextBounds.height to a whole number of lines so the last
        // visible line doesn't appear half-cut at the pane bottom (the
        // viewport would otherwise show a mid-glyph cut, which looks
        // worse than the line just not appearing). The output_text
        // entity carries `PaneContentNoClip` to opt out of pane-bevy's
        // default bound-setting, leaving height management to us here.
        // Width is left as the available area as a hint — it's a no-op
        // for our `TextLayout::new_with_no_wrap` text but keeps the
        // bounds shape sane.
        let avail_w = (content_w - PLAY_X).max(0.0);
        let new_bounds = TextBounds {
            width: Some(avail_w),
            height: Some(visible as f32 * OUTPUT_LINE_HEIGHT),
        };
        if let Ok(mut b) = bounds_q.get_mut(chrome.output_text)
            && (b.width != new_bounds.width || b.height != new_bounds.height)
        {
            *b = new_bounds;
        }
    }
}

/// Mouse-wheel scroll on the details panel of a run-button pane:
/// wheel up reveals older lines, wheel down moves toward the latest.
/// Uses pane-bevy's topmost-pane lookup so panes layered on top of a
/// run-button win the scroll target. Trackpad pixel events accumulate
/// across frames so a slow swipe still registers a whole line.
fn scroll_run_button_output(
    mut wheel: MessageReader<MouseWheel>,
    mut accum: Local<f32>,
    windows: Query<&Window>,
    viewport: Res<pane_bevy::PaneViewport>,
    keys: Res<ButtonInput<KeyCode>>,
    panes_q: Query<(Entity, &PaneRect, Option<&Visibility>, &PaneKindMarker), With<PaneTag>>,
    mut rbs: Query<&mut RunButton>,
) {
    // Cmd+scroll is canvas pan, not pane scroll.
    if keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight) {
        wheel.clear();
        *accum = 0.0;
        return;
    }
    let mut delta_lines: f32 = 0.0;
    for ev in wheel.read() {
        let lines = match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / OUTPUT_LINE_HEIGHT,
        };
        delta_lines += lines;
    }
    if delta_lines == 0.0 {
        return;
    }
    *accum += delta_lines;
    let whole = accum.trunc() as isize;
    if whole == 0 {
        return;
    }
    *accum -= whole as f32;

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };

    let visible_rects: Vec<(Entity, PaneRect)> = panes_q
        .iter()
        .filter(|(_, _, vis, kind)| {
            kind.0 == PANE_KIND && !matches!(vis, Some(Visibility::Hidden))
        })
        .map(|(e, r, _, _)| (e, *r))
        .collect();
    let Some(target) = pane_bevy::topmost_pane_at(viewport.window_to_canvas(pt), &visible_rects)
    else {
        return;
    };
    let target_rect = visible_rects
        .iter()
        .find(|(e, _)| *e == target)
        .map(|(_, r)| *r);
    let Some(rect) = target_rect else {
        return;
    };

    let Ok(mut rb) = rbs.get_mut(target) else {
        return;
    };
    if !rb.output_expanded || rb.output.is_empty() {
        return;
    }

    let cap = output_lines_capacity(&rect);
    let n = rb.output.len();
    let max_top = n.saturating_sub(n.min(cap));

    // Anchor relative to what's *currently visible* so the first scroll
    // up out of follow-tail doesn't snap to scroll_top=0 (the oldest
    // line) and feel like the wheel went the wrong way.
    let current_top = if rb.output_follow_tail {
        max_top
    } else {
        rb.output_scroll_top.min(max_top)
    };
    let new_top = if whole > 0 {
        // wheel up = show older lines = top moves earlier in deque
        current_top.saturating_sub(whole as usize)
    } else {
        // wheel down = newer lines = top moves later
        current_top.saturating_add((-whole) as usize).min(max_top)
    };
    rb.output_scroll_top = new_top;
    rb.output_follow_tail = new_top >= max_top;
}

fn set_vis(vis_q: &mut Query<&mut Visibility>, e: Entity, on: bool) {
    if let Ok(mut v) = vis_q.get_mut(e) {
        *v = if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}
