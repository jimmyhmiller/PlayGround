//! Minimal Bevy-native terminal emulator built on libghostty-vt.
//!
//! Multi-instance: each terminal is an Entity with chrome (bg, title bar,
//! content root, cursor, resize handle), draggable and resizable like the
//! editors in `editor-bevy`. Focus + mouse logic mirror that crate so the
//! two will later share a `text-surface-bevy` extraction (step 2 of the
//! plan). For now this is a fork, not a refactor.
//!
//! ## Threading
//!
//! `libghostty_vt::Terminal` is `!Send + !Sync`, so we can't store it as a
//! Bevy `Component`. Instead a single `NonSend<TerminalStore>` resource
//! owns a `HashMap<Entity, TerminalData>`. Entities still carry Send
//! components (rect, chrome, rev counter, row-entity pool); the
//! non-Send runtime lives in the store keyed by the same entity. Systems
//! that need both iterate entities and look up the store by id.
//!
//! ## v0 scope
//!
//! - Direct key encoding (no libghostty key encoder / Kitty kb protocol):
//!   printable chars, Enter/Tab/Backspace/Escape, arrows (xterm style),
//!   Home/End/PageUp/PageDown, Delete, ctrl+letter → control codes.
//! - fg-color text runs only, no per-cell bg
//! - no wide-char handling (1 cell per char assumed)
//! - no mouse reporting to the child, no selection / scrollback panning

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextSpan};

use libghostty_vt::{
    render::{CellIterator, Dirty, RenderState, RowIterator},
    style::RgbColor,
    terminal::Mode,
    Terminal,
};

pub mod pty;
pub mod vt;
use pty::{Child, Pty, PtyError, PtySize};
use vt::CellPx;

pub const FONT_SIZE: f32 = 14.0;
pub const LINE_HEIGHT: f32 = 18.0;
pub const MARGIN: f32 = 8.0;
pub const TITLE_H: f32 = 22.0;
pub const HANDLE_SIZE: f32 = 14.0;
pub const MIN_TERMINAL_SIZE: Vec2 = Vec2::new(240.0, 160.0);
pub const SCROLLBACK_LINES: usize = 1000;

const EMBEDDED_FONT: &[u8] =
    include_bytes!("../../editor-bevy/assets/fonts/JetBrainsMono-Regular.ttf");

// ---------- Per-entity runtime (NonSend) ----------

/// Everything the libghostty side needs. Lives in `TerminalStore` keyed
/// by the owning Bevy `Entity`.
pub struct TerminalData {
    /// The libghostty Terminal, heap-pinned in a `Box` so its address
    /// never changes after handler registration. See `vt.rs` module
    /// docs for why this matters — the 0.1.1 binding is buggy about
    /// moves-after-registration for several of its callback slots.
    pub terminal: Box<Terminal<'static, 'static>>,
    pub render_state: RenderState<'static>,
    pub row_it: RowIterator<'static>,
    pub cell_it: CellIterator<'static>,
    pub pty: Pty,
    pub child: Child,
    /// Bytes the terminal's VT parser wants sent back to the pty
    /// (DA1/DSR responses, etc.). Filled by the on_pty_write effect.
    pub pty_response: Rc<RefCell<Vec<u8>>>,
    pub grid_cols: u16,
    pub grid_rows: u16,
}

#[derive(Default)]
pub struct TerminalStore {
    pub map: HashMap<Entity, TerminalData>,
}

// ---------- Components (Send) ----------

#[derive(Component)]
pub struct TerminalTag;

/// Position + size + z in window space, top-left origin, y-down. Same
/// layout as `editor-bevy::EditorRect`.
#[derive(Component, Copy, Clone, Debug)]
pub struct TerminalRect {
    pub pos: Vec2,
    pub size: Vec2,
    pub z: f32,
}

/// Child entities making up the visible chrome for one terminal.
#[derive(Component)]
pub struct TerminalChrome {
    pub bg: Entity,
    pub title_bar: Entity,
    pub content_root: Entity,
    pub cursor: Entity,
    pub resize_handle: Entity,
}

/// Single `Text2d` entity per terminal that holds the entire visible
/// grid as one newline-separated text block, coloured via `TextSpan`
/// children.
///
/// Was previously one `Text2d` per row (24+ entities). Under a heavy
/// stream (`cat` of a big file) every row was dirty every frame, which
/// forced Bevy's text layout pipeline to re-shape 24 separate text
/// blocks per frame — the dominant cost. One text block = one shape
/// per frame regardless of grid size.
#[derive(Component)]
pub struct ContentText(pub Entity);

/// Hash of the most recently rendered runs; skips rebuild when the
/// visible grid didn't change.
#[derive(Component, Default)]
pub struct ContentRender {
    pub last_signature: u64,
    /// Number of TextSpan child entities currently under the Text2d.
    /// Lets us reuse existing span entities on the next rebuild and
    /// only spawn/despawn the delta.
    pub span_count: usize,
}

/// Bumped whenever the Terminal for this entity is mutated (vt bytes
/// processed, resize). `sync_grid` rebuilds row spans when it differs
/// from the value we last rendered.
#[derive(Component, Default)]
pub struct TerminalRev(pub u64);

#[derive(Resource)]
pub struct MonoFont(pub Handle<Font>);

#[derive(Resource, Copy, Clone)]
pub struct MonoMetrics {
    pub cell_width: f32,
}

#[derive(Resource, Default)]
pub struct FocusedTerminal(pub Option<Entity>);

#[derive(Resource, Default)]
pub enum MouseMode {
    #[default]
    Idle,
    WindowDrag {
        terminal: Entity,
        grab_offset: Vec2,
    },
    WindowResize {
        terminal: Entity,
        anchor_pos: Vec2,
    },
}

// ---------- Plugin ----------

pub struct TerminalPlugin;

impl Plugin for TerminalPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::srgb(0.05, 0.06, 0.08)))
            .insert_resource(FocusedTerminal::default())
            .insert_resource(MouseMode::default())
            .insert_resource(bevy::winit::WinitSettings {
                focused_mode: bevy::winit::UpdateMode::Continuous,
                unfocused_mode: bevy::winit::UpdateMode::Continuous,
            })
            .add_systems(Startup, setup_camera_and_font)
            .add_systems(PostStartup, release_os_focus)
            .add_systems(
                Update,
                (
                    handle_mouse,
                    handle_resize,
                    process_pty_all,
                    handle_keyboard,
                    sync_grid,
                    position_root,
                )
                    .chain(),
            );
    }
}

#[cfg(target_os = "macos")]
fn release_os_focus() {
    use objc2_app_kit::NSApplication;
    use objc2_foundation::MainThreadMarker;
    if let Some(mtm) = MainThreadMarker::new() {
        let app = NSApplication::sharedApplication(mtm);
        unsafe { app.deactivate() };
    }
}

#[cfg(not(target_os = "macos"))]
fn release_os_focus() {}

fn measure_cell_width(font_bytes: &[u8], font_size: f32) -> f32 {
    use skrifa::instance::{LocationRef, Size};
    use skrifa::{FontRef, MetadataProvider};
    let font = FontRef::from_index(font_bytes, 0).expect("embedded font must parse");
    let metrics = font.glyph_metrics(Size::new(font_size), LocationRef::default());
    let gid = font.charmap().map('M').expect("font must contain 'M'");
    metrics
        .advance_width(gid)
        .expect("'M' must have an advance width")
}

/// Exposed so callers can `.after(setup_camera_and_font)` their own
/// startup systems that spawn terminals.
pub fn setup_camera_and_font(world: &mut World) {
    world
        .resource_mut::<bevy::text::CosmicFontSystem>()
        .0
        .db_mut()
        .load_system_fonts();

    world.spawn(Camera2d);

    let font_handle = world
        .resource_mut::<Assets<Font>>()
        .add(Font::try_from_bytes(EMBEDDED_FONT.to_vec()).expect("JetBrainsMono must parse"));
    world.insert_resource(MonoFont(font_handle));

    let cell_width = measure_cell_width(EMBEDDED_FONT, FONT_SIZE);
    world.insert_resource(MonoMetrics { cell_width });

    // Init the NonSend store once — spawners populate it per entity.
    world.insert_non_send_resource(TerminalStore::default());
}

// ---------- Spawn ----------

/// Create one terminal entity with its chrome + spawn a shell on its pty.
/// Returns the entity so the caller can set focus, tweak z, etc.
///
/// Must be called from an exclusive system (`&mut World`) because the
/// Terminal + closures are `!Send`.
pub fn spawn_terminal(world: &mut World, rect: TerminalRect) -> Entity {
    let cell_width = world.resource::<MonoMetrics>().cell_width;
    let font_handle = world.resource::<MonoFont>().0.clone();

    let (cols, rows) = grid_size_for_rect(rect.size, cell_width);
    let (terminal, pty_response) = vt::build_terminal(
        cols,
        rows,
        SCROLLBACK_LINES,
        CellPx {
            width: cell_width as u32,
            height: LINE_HEIGHT as u32,
        },
    );

    let (pty, child) = Pty::spawn(PtySize {
        cols,
        rows,
        cell_width_px: cell_width as u16,
        cell_height_px: LINE_HEIGHT as u16,
    })
    .expect("Pty::spawn failed");

    let data = TerminalData {
        terminal,
        render_state: RenderState::new().expect("RenderState::new"),
        row_it: RowIterator::new().expect("RowIterator::new"),
        cell_it: CellIterator::new().expect("CellIterator::new"),
        pty,
        child,
        pty_response,
        grid_cols: cols,
        grid_rows: rows,
    };

    // Parent entity + chrome children.
    let terminal_entity = world
        .spawn((
            TerminalTag,
            rect,
            TerminalRev::default(),
            Transform::default(),
            Visibility::default(),
        ))
        .id();

    let bg = world
        .spawn((
            ChildOf(terminal_entity),
            Sprite {
                color: Color::srgb(0.08, 0.10, 0.13),
                custom_size: Some(rect.size),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.0),
        ))
        .id();
    let title_bar = world
        .spawn((
            ChildOf(terminal_entity),
            Sprite {
                color: Color::srgb(0.18, 0.20, 0.24),
                custom_size: Some(Vec2::new(rect.size.x, TITLE_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.1),
        ))
        .id();
    let content_root = world
        .spawn((
            ChildOf(terminal_entity),
            Transform::from_xyz(MARGIN, -(TITLE_H + MARGIN), 0.2),
            Visibility::default(),
        ))
        .id();

    // One Text2d holds all visible rows, separated by '\n'. TextSpan
    // children colour individual runs. See `ContentText` docs for why.
    let content_text = world
        .spawn((
            ChildOf(content_root),
            Text2d::new(String::new()),
            TextFont {
                font: font_handle.clone(),
                font_size: FONT_SIZE,
                ..default()
            },
            LineHeight::Px(LINE_HEIGHT),
            TextColor(Color::srgb(0.85, 0.86, 0.88)),
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.0),
            ContentRender::default(),
        ))
        .id();

    let cursor = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: Color::srgba(0.6, 0.85, 1.0, 0.7),
                custom_size: Some(Vec2::new(cell_width, LINE_HEIGHT)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 1.0),
        ))
        .id();
    let resize_handle = world
        .spawn((
            ChildOf(terminal_entity),
            Sprite {
                color: Color::srgb(0.35, 0.40, 0.48),
                custom_size: Some(Vec2::splat(HANDLE_SIZE)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                rect.size.x - HANDLE_SIZE,
                -(rect.size.y - HANDLE_SIZE),
                0.3,
            ),
        ))
        .id();

    world.entity_mut(terminal_entity).insert((
        TerminalChrome {
            bg,
            title_bar,
            content_root,
            cursor,
            resize_handle,
        },
        ContentText(content_text),
    ));

    world
        .get_non_send_resource_mut::<TerminalStore>()
        .expect("TerminalStore resource (did setup_camera_and_font run?)")
        .map
        .insert(terminal_entity, data);

    // Suppress unused font warning in this scope; actual handle is
    // resolved when rows are spawned by sync_grid.
    let _ = font_handle;

    terminal_entity
}

fn grid_size_for_rect(size: Vec2, cell_width: f32) -> (u16, u16) {
    let content_w = (size.x - 2.0 * MARGIN).max(0.0);
    let content_h = (size.y - TITLE_H - 2.0 * MARGIN).max(0.0);
    let cols = ((content_w / cell_width).floor() as u16).max(1);
    let rows = ((content_h / LINE_HEIGHT).floor() as u16).max(1);
    (cols, rows)
}

fn content_area(rect: &TerminalRect) -> (Vec2, Vec2) {
    let origin = Vec2::new(MARGIN, -(TITLE_H + MARGIN));
    let size = Vec2::new(
        (rect.size.x - 2.0 * MARGIN).max(0.0),
        (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
    );
    (origin, size)
}

// ---------- Resize ----------

/// Recompute grid dimensions for any terminal whose rect disagrees with
/// its stored grid. Forwards resize to libghostty + the pty.
fn handle_resize(
    metrics: Res<MonoMetrics>,
    mut store: NonSendMut<TerminalStore>,
    mut rev_q: Query<(Entity, &TerminalRect, &mut TerminalRev)>,
) {
    for (entity, rect, mut rev) in &mut rev_q {
        let Some(data) = store.map.get_mut(&entity) else {
            continue;
        };
        let (cols, rows) = grid_size_for_rect(rect.size, metrics.cell_width);
        if cols == data.grid_cols && rows == data.grid_rows {
            continue;
        }
        let _ = data.terminal.resize(
            cols,
            rows,
            metrics.cell_width as u32,
            LINE_HEIGHT as u32,
        );
        data.pty.resize(PtySize {
            cols,
            rows,
            cell_width_px: metrics.cell_width as u16,
            cell_height_px: LINE_HEIGHT as u16,
        });
        data.grid_cols = cols;
        data.grid_rows = rows;
        rev.0 = rev.0.wrapping_add(1);
    }
}

// ---------- PTY ↔ Terminal pumping ----------

fn process_pty_all(
    mut store: NonSendMut<TerminalStore>,
    mut rev_q: Query<(Entity, &mut TerminalRev)>,
) {
    for (entity, mut rev) in &mut rev_q {
        let Some(data) = store.map.get_mut(&entity) else {
            continue;
        };
        if !matches!(data.child, Child::Active(_)) {
            continue;
        }
        let read_result = data.pty.read_into(&mut data.terminal);

        // Flush any queued responses the VT parser wants sent back.
        {
            let mut response = data.pty_response.borrow_mut();
            if !response.is_empty() {
                data.pty.write(&response);
                response.clear();
            }
        }

        match read_result {
            Ok(()) => rev.0 = rev.0.wrapping_add(1),
            Err(PtyError::EndOfStream) => {
                if let Child::Active(pid) = data.child {
                    data.child = Child::Exited(pid);
                }
            }
            Err(PtyError::Other(e)) => {
                eprintln!("pty read error: {e}");
                if let Child::Active(pid) = data.child {
                    data.child = Child::Exited(pid);
                }
            }
        }
    }
}

// ---------- Keyboard ----------

/// Translate Bevy key events to VT bytes for the focused terminal.
///
/// Direct mapping (not libghostty's key encoder) for v0 simplicity and
/// to fix space/printable keys landing as `Key::Space` / `Key::Character`
/// rather than going through an encoder path that requires a separate
/// text stream.
fn handle_keyboard(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    focused: Res<FocusedTerminal>,
    mut store: NonSendMut<TerminalStore>,
) {
    let Some(target) = focused.0 else {
        events.read().for_each(|_| {});
        return;
    };
    let Some(data) = store.map.get_mut(&target) else {
        events.read().for_each(|_| {});
        return;
    };
    if !matches!(data.child, Child::Active(_)) {
        events.read().for_each(|_| {});
        return;
    }

    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);

    // DECCKM — cursor keys app mode. Off = xterm CSI (`\x1b[A`);
    // on = SS3 (`\x1bOA`). Most shells only send SS3 while readline is
    // in certain modes, but we honor it if the program enables it.
    let app_cursor = data.terminal.mode(Mode::DECCKM).unwrap_or(false);

    let mut out: Vec<u8> = Vec::with_capacity(16);

    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }

        // Ctrl + printable letter → control byte (Ctrl+A = 0x01, etc.).
        if ctrl && !alt {
            if let KeyCode::KeyA
            | KeyCode::KeyB
            | KeyCode::KeyC
            | KeyCode::KeyD
            | KeyCode::KeyE
            | KeyCode::KeyF
            | KeyCode::KeyG
            | KeyCode::KeyH
            | KeyCode::KeyI
            | KeyCode::KeyJ
            | KeyCode::KeyK
            | KeyCode::KeyL
            | KeyCode::KeyM
            | KeyCode::KeyN
            | KeyCode::KeyO
            | KeyCode::KeyP
            | KeyCode::KeyQ
            | KeyCode::KeyR
            | KeyCode::KeyS
            | KeyCode::KeyT
            | KeyCode::KeyU
            | KeyCode::KeyV
            | KeyCode::KeyW
            | KeyCode::KeyX
            | KeyCode::KeyY
            | KeyCode::KeyZ = ev.key_code
            {
                let b = keycode_to_ctrl_byte(ev.key_code);
                out.push(b);
                continue;
            }
        }

        // Named keys we know the VT encoding for. Arrows / Home / End
        // honor DECCKM.
        if let Some(bytes) = named_key_bytes(&ev.key_code, app_cursor) {
            out.extend_from_slice(bytes);
            continue;
        }

        // Printable text via Bevy's logical_key.
        match &ev.logical_key {
            Key::Character(s) => {
                let mut bytes: Vec<u8> = s.as_str().as_bytes().to_vec();
                // Alt+letter sends ESC-prefixed byte (meta convention).
                if alt && !ctrl {
                    out.push(0x1b);
                }
                out.append(&mut bytes);
            }
            Key::Space => {
                if alt && !ctrl {
                    out.push(0x1b);
                }
                out.push(b' ');
            }
            _ => {
                let _ = shift; // informational — most shifting already baked into Character.
            }
        }
    }

    if !out.is_empty() {
        data.pty.write(&out);
    }
}

fn keycode_to_ctrl_byte(code: KeyCode) -> u8 {
    // Ctrl+A = 0x01 ... Ctrl+Z = 0x1a.
    let base = match code {
        KeyCode::KeyA => 1,
        KeyCode::KeyB => 2,
        KeyCode::KeyC => 3,
        KeyCode::KeyD => 4,
        KeyCode::KeyE => 5,
        KeyCode::KeyF => 6,
        KeyCode::KeyG => 7,
        KeyCode::KeyH => 8,
        KeyCode::KeyI => 9,
        KeyCode::KeyJ => 10,
        KeyCode::KeyK => 11,
        KeyCode::KeyL => 12,
        KeyCode::KeyM => 13,
        KeyCode::KeyN => 14,
        KeyCode::KeyO => 15,
        KeyCode::KeyP => 16,
        KeyCode::KeyQ => 17,
        KeyCode::KeyR => 18,
        KeyCode::KeyS => 19,
        KeyCode::KeyT => 20,
        KeyCode::KeyU => 21,
        KeyCode::KeyV => 22,
        KeyCode::KeyW => 23,
        KeyCode::KeyX => 24,
        KeyCode::KeyY => 25,
        KeyCode::KeyZ => 26,
        _ => 0,
    };
    base
}

fn named_key_bytes(code: &KeyCode, app_cursor: bool) -> Option<&'static [u8]> {
    Some(match code {
        KeyCode::Enter | KeyCode::NumpadEnter => b"\r",
        KeyCode::Tab => b"\t",
        KeyCode::Backspace => b"\x7f",
        KeyCode::Escape => b"\x1b",
        KeyCode::Delete => b"\x1b[3~",
        KeyCode::Insert => b"\x1b[2~",
        KeyCode::PageUp => b"\x1b[5~",
        KeyCode::PageDown => b"\x1b[6~",
        KeyCode::ArrowUp => {
            if app_cursor {
                b"\x1bOA"
            } else {
                b"\x1b[A"
            }
        }
        KeyCode::ArrowDown => {
            if app_cursor {
                b"\x1bOB"
            } else {
                b"\x1b[B"
            }
        }
        KeyCode::ArrowRight => {
            if app_cursor {
                b"\x1bOC"
            } else {
                b"\x1b[C"
            }
        }
        KeyCode::ArrowLeft => {
            if app_cursor {
                b"\x1bOD"
            } else {
                b"\x1b[D"
            }
        }
        KeyCode::Home => {
            if app_cursor {
                b"\x1bOH"
            } else {
                b"\x1b[H"
            }
        }
        KeyCode::End => {
            if app_cursor {
                b"\x1bOF"
            } else {
                b"\x1b[F"
            }
        }
        _ => return None,
    })
}

// ---------- Mouse / chrome ----------

#[derive(Copy, Clone)]
enum Region {
    TitleBar,
    ResizeHandle,
    Content,
}

fn region_at(pt: Vec2, rect: &TerminalRect) -> Option<Region> {
    if pt.x < rect.pos.x || pt.x > rect.pos.x + rect.size.x {
        return None;
    }
    if pt.y < rect.pos.y || pt.y > rect.pos.y + rect.size.y {
        return None;
    }
    let handle_x0 = rect.pos.x + rect.size.x - HANDLE_SIZE;
    let handle_y0 = rect.pos.y + rect.size.y - HANDLE_SIZE;
    if pt.x >= handle_x0 && pt.y >= handle_y0 {
        return Some(Region::ResizeHandle);
    }
    if pt.y < rect.pos.y + TITLE_H {
        return Some(Region::TitleBar);
    }
    Some(Region::Content)
}

fn topmost_terminal_at(pt: Vec2, rects: &[(Entity, TerminalRect)]) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for &(e, r) in rects {
        if pt.x >= r.pos.x
            && pt.x <= r.pos.x + r.size.x
            && pt.y >= r.pos.y
            && pt.y <= r.pos.y + r.size.y
        {
            if best.map_or(true, |(_, z)| r.z > z) {
                best = Some((e, r.z));
            }
        }
    }
    best.map(|(e, _)| e)
}

fn handle_mouse(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut mode: ResMut<MouseMode>,
    mut focused: ResMut<FocusedTerminal>,
    mut terminals: Query<(Entity, &mut TerminalRect), With<TerminalTag>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };

    if buttons.just_released(MouseButton::Left) {
        *mode = MouseMode::Idle;
    }

    if buttons.just_pressed(MouseButton::Left) {
        let rects: Vec<(Entity, TerminalRect)> =
            terminals.iter().map(|(e, r)| (e, *r)).collect();
        let Some(target) = topmost_terminal_at(pt, &rects) else {
            return;
        };
        focused.0 = Some(target);
        bring_to_front(target, &mut terminals);

        let rect = *terminals.get(target).unwrap().1;
        match region_at(pt, &rect) {
            Some(Region::TitleBar) => {
                *mode = MouseMode::WindowDrag {
                    terminal: target,
                    grab_offset: pt - rect.pos,
                };
            }
            Some(Region::ResizeHandle) => {
                *mode = MouseMode::WindowResize {
                    terminal: target,
                    anchor_pos: rect.pos,
                };
            }
            Some(Region::Content) | None => {}
        }
        return;
    }

    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    match *mode {
        MouseMode::WindowDrag {
            terminal,
            grab_offset,
        } => {
            if let Ok((_, mut rect)) = terminals.get_mut(terminal) {
                rect.pos = pt - grab_offset;
            }
        }
        MouseMode::WindowResize {
            terminal,
            anchor_pos,
        } => {
            if let Ok((_, mut rect)) = terminals.get_mut(terminal) {
                let raw = pt - anchor_pos;
                rect.size = Vec2::new(
                    raw.x.max(MIN_TERMINAL_SIZE.x),
                    raw.y.max(MIN_TERMINAL_SIZE.y),
                );
            }
        }
        MouseMode::Idle => {}
    }
}

fn bring_to_front(
    target: Entity,
    terminals: &mut Query<(Entity, &mut TerminalRect), With<TerminalTag>>,
) {
    let max_z = terminals.iter().map(|(_, r)| r.z).fold(0.0_f32, f32::max);
    if let Ok((_, mut rect)) = terminals.get_mut(target) {
        if rect.z < max_z {
            rect.z = max_z + 1.0;
        }
    }
}

fn position_root(
    windows: Query<&Window>,
    terminals: Query<(&TerminalRect, &TerminalChrome), With<TerminalTag>>,
    parents: Query<Entity, With<TerminalTag>>,
    mut t_q: Query<&mut Transform>,
    mut sprite_q: Query<&mut Sprite>,
) {
    let Ok(win) = windows.single() else {
        return;
    };
    let win_size = Vec2::new(win.width(), win.height());

    for entity in &parents {
        let Ok((rect, chrome)) = terminals.get(entity) else {
            continue;
        };
        if let Ok(mut t) = t_q.get_mut(entity) {
            t.translation.x = rect.pos.x - win_size.x * 0.5;
            t.translation.y = win_size.y * 0.5 - rect.pos.y;
            t.translation.z = rect.z;
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.bg) {
            s.custom_size = Some(rect.size);
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.title_bar) {
            s.custom_size = Some(Vec2::new(rect.size.x, TITLE_H));
        }
        if let Ok(mut t) = t_q.get_mut(chrome.resize_handle) {
            t.translation.x = rect.size.x - HANDLE_SIZE;
            t.translation.y = -(rect.size.y - HANDLE_SIZE);
        }
    }
}

// ---------- Rendering ----------

/// Render the whole grid as a single multi-line `Text2d`, reusing
/// TextSpan entities across frames and mutating their contents in
/// place. This is the fast path for high-throughput output like
/// `cat` — one cosmic-text layout per terminal per frame instead of
/// one per row.
fn sync_grid(
    metrics: Res<MonoMetrics>,
    font: Res<MonoFont>,
    mut store: NonSendMut<TerminalStore>,
    terminals: Query<(Entity, &TerminalRect, &TerminalChrome, &ContentText)>,
    mut content_q: Query<&mut ContentRender>,
    mut text_q: Query<&mut Text2d>,
    mut span_q: Query<(&mut TextSpan, &mut TextColor)>,
    children_q: Query<&Children>,
    mut cursor_t_q: Query<&mut Transform>,
    mut vis_q: Query<&mut Visibility>,
    mut commands: Commands,
) {
    for (entity, rect, chrome, content) in &terminals {
        let Some(data) = store.map.get_mut(&entity) else {
            continue;
        };
        let TerminalData {
            terminal,
            render_state,
            row_it,
            cell_it,
            grid_rows,
            ..
        } = data;
        let rows_n = *grid_rows;

        let snapshot = match render_state.update(terminal) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Cursor — update every frame since libghostty's per-row dirty
        // bits don't cover cursor motion through unchanged cells.
        let cursor_visible = snapshot.cursor_visible().unwrap_or(false);
        let cursor_pos = snapshot.cursor_viewport().ok().flatten();
        if let Ok(mut v) = vis_q.get_mut(chrome.cursor) {
            *v = if cursor_visible && cursor_pos.is_some() {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
        if let Some(p) = cursor_pos {
            if let Ok(mut t) = cursor_t_q.get_mut(chrome.cursor) {
                t.translation.x = p.x as f32 * metrics.cell_width;
                t.translation.y = -(p.y as f32) * LINE_HEIGHT;
                t.translation.z = 1.0;
            }
        }

        let colors = snapshot.colors().ok();
        let default_fg = colors.map(|c| c.foreground).unwrap_or(RgbColor {
            r: 220,
            g: 220,
            b: 220,
        });

        let dirty = snapshot.dirty().unwrap_or(Dirty::Full);
        // Nothing changed — skip all cell iteration + span rebuild.
        if matches!(dirty, Dirty::Clean) {
            continue;
        }

        // Clamp width so row text isn't spawned past the content edge.
        let (_, content_size) = content_area(rect);
        let max_cols = ((content_size.x / metrics.cell_width).floor() as usize).max(1);

        // Build runs across the whole grid. A run is a contiguous span
        // of same-fg text. Newlines are emitted as part of the run that
        // happens to end the row, which keeps span count down.
        let mut runs: Vec<(String, RgbColor)> = Vec::with_capacity(rows_n as usize * 2);
        let mut current: Option<(String, RgbColor)> = None;

        let mut row_iter = match row_it.update(&snapshot) {
            Ok(it) => it,
            Err(_) => continue,
        };
        let mut rows_emitted = 0u16;
        while let Some(row) = row_iter.next() {
            if rows_emitted >= rows_n {
                break;
            }
            // Render this row's cells, accumulate into `runs`.
            let mut cell_iter = match cell_it.update(row) {
                Ok(it) => it,
                Err(_) => {
                    // Pad with empty run so line index lines up.
                    push_into_runs(&mut runs, &mut current, "\n", default_fg);
                    rows_emitted += 1;
                    continue;
                }
            };
            let mut col_count = 0usize;
            // Last-non-blank column so we can skip trailing spaces in
            // each row — saves huge amounts of text on mostly-empty rows.
            let mut row_text_buf = String::new();
            let mut row_colors: Vec<(usize, RgbColor)> = Vec::new(); // (start_idx, color)
            while let Some(cell) = cell_iter.next() {
                if col_count >= max_cols {
                    break;
                }
                let graphemes_len = cell.graphemes_len().unwrap_or(0);
                let fg = cell.fg_color().ok().flatten().unwrap_or(default_fg);
                if row_colors.last().map_or(true, |(_, c)| !rgb_eq(*c, fg)) {
                    row_colors.push((row_text_buf.len(), fg));
                }
                if graphemes_len == 0 {
                    row_text_buf.push(' ');
                } else if let Ok(chars) = cell.graphemes() {
                    for c in chars {
                        row_text_buf.push(c);
                    }
                } else {
                    row_text_buf.push(' ');
                }
                col_count += 1;
            }
            // Trim trailing spaces from the row.
            let trimmed_end = row_text_buf.trim_end_matches(' ').len();
            row_text_buf.truncate(trimmed_end);

            // Emit row's runs. `row_colors` holds byte offsets recorded
            // before we trimmed trailing spaces, so clamp both ends to
            // the post-trim length. Trailing spaces are ASCII so trim
            // never crosses a UTF-8 boundary, and the cell-push record
            // sites were always valid boundaries — clamping to
            // `row_text_buf.len()` stays on a boundary.
            let row_len = row_text_buf.len();
            for i in 0..row_colors.len() {
                let (start, color) = row_colors[i];
                let start = start.min(row_len);
                let end = row_colors
                    .get(i + 1)
                    .map(|(e, _)| *e)
                    .unwrap_or(row_len)
                    .min(row_len);
                if start >= end {
                    continue;
                }
                let slice = &row_text_buf[start..end];
                push_into_runs(&mut runs, &mut current, slice, color);
            }
            push_into_runs(&mut runs, &mut current, "\n", default_fg);
            rows_emitted += 1;
            let _ = row.set_dirty(false);
        }
        if let Some(prev) = current.take() {
            runs.push(prev);
        }

        // Signature — skip ECS mutation if the runs didn't actually
        // change since last frame.
        let mut sig: u64 = 1469598103934665603;
        for (text, fg) in &runs {
            for b in text.as_bytes() {
                sig ^= *b as u64;
                sig = sig.wrapping_mul(1099511628211);
            }
            sig ^= ((fg.r as u64) << 16) | ((fg.g as u64) << 8) | (fg.b as u64);
            sig = sig.wrapping_mul(1099511628211);
        }
        if let Ok(cr) = content_q.get(content.0) {
            if cr.last_signature == sig {
                let _ = snapshot.set_dirty(Dirty::Clean);
                continue;
            }
        }

        // Reuse existing span entities: mutate text+color for the first
        // N, spawn extras, despawn leftovers.
        let current_spans: Vec<Entity> = children_q
            .get(content.0)
            .map(|c| c.iter().collect())
            .unwrap_or_default();

        // First span is the Text2d's base content. Keep it empty and
        // use spans for all colored content — simpler than splitting
        // the first run across Text2d and TextSpan.
        if let Ok(mut t2d) = text_q.get_mut(content.0) {
            if !t2d.0.is_empty() {
                t2d.0.clear();
            }
        }

        let mut span_idx = 0usize;
        for (text, fg) in &runs {
            let color = TextColor(rgb_to_color(*fg));
            if span_idx < current_spans.len() {
                let ent = current_spans[span_idx];
                if let Ok((mut span, mut tc)) = span_q.get_mut(ent) {
                    if span.0 != *text {
                        span.0.clear();
                        span.0.push_str(text);
                    }
                    tc.0 = color.0;
                }
            } else {
                commands.spawn((
                    ChildOf(content.0),
                    TextSpan::new(text.clone()),
                    TextFont {
                        font: font.0.clone(),
                        font_size: FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(LINE_HEIGHT),
                    color,
                ));
            }
            span_idx += 1;
        }
        // Despawn surplus.
        for &ent in current_spans.iter().skip(span_idx) {
            commands.entity(ent).despawn();
        }
        if let Ok(mut cr) = content_q.get_mut(content.0) {
            cr.last_signature = sig;
            cr.span_count = span_idx;
        } else {
            commands.entity(content.0).insert(ContentRender {
                last_signature: sig,
                span_count: span_idx,
            });
        }

        let _ = snapshot.set_dirty(Dirty::Clean);
    }
}

/// Push `text` with `fg` into `runs`, coalescing with the current run
/// when colours match. `current` is the in-progress run; the caller is
/// responsible for draining it into `runs` at the end.
fn push_into_runs(
    runs: &mut Vec<(String, RgbColor)>,
    current: &mut Option<(String, RgbColor)>,
    text: &str,
    fg: RgbColor,
) {
    if text.is_empty() {
        return;
    }
    match current.as_mut() {
        Some((buf, run_fg)) if rgb_eq(*run_fg, fg) => buf.push_str(text),
        _ => {
            if let Some(prev) = current.take() {
                runs.push(prev);
            }
            *current = Some((text.to_string(), fg));
        }
    }
}

// ---------- helpers ----------

fn rgb_eq(a: RgbColor, b: RgbColor) -> bool {
    a.r == b.r && a.g == b.g && a.b == b.b
}

fn rgb_to_color(c: RgbColor) -> Color {
    Color::srgb(c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0)
}
