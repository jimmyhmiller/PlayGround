//! Central **action registry** — the single source of truth for "things
//! the app can do." One `Action` is enumerable by the command palette,
//! placeable on the radial ring, bindable to a global keyboard chord,
//! and (later) exposable to DeepSeek as a tool. Before this module the
//! same capability was spread across four near-identical keyboard-
//! shortcut systems in `lib.rs` and a pane-spawn-only radial menu.
//!
//! ## Shape
//!
//! - [`Action`] is `Copy` (all-`'static` fields + an [`ActionRun`] that
//!   is either a data-carrying `SpawnPane` or a bare `fn` pointer),
//!   mirroring `pane_bevy::PaneKindSpec`.
//! - [`ActionRegistry`] is a resource keyed by id, preserving insertion
//!   order so the palette lists actions deterministically.
//! - Producers (keybinds, radial, palette) push an [`ActionInvocation`]
//!   onto [`ActionInvocations`]; the exclusive [`run_requested_actions`]
//!   system drains it and performs each effect with full `&mut World`
//!   access. Producers run in [`ActionProducerSet`], dispatch after it.

use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use std::collections::HashMap;

use pane_bevy::{KeyboardOwner, PaneRegistry};

use crate::projects::{NewPaneRequest, PendingActions, Projects};

/// A global keyboard chord. Left/right modifier variants are folded
/// together by the matcher, so a chord only records *whether* each
/// modifier is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyChord {
    pub cmd: bool,
    pub shift: bool,
    pub alt: bool,
    pub ctrl: bool,
    pub key: KeyCode,
}

impl KeyChord {
    pub const fn cmd(key: KeyCode) -> Self {
        Self { cmd: true, shift: false, alt: false, ctrl: false, key }
    }
    pub const fn cmd_shift(key: KeyCode) -> Self {
        Self { cmd: true, shift: true, alt: false, ctrl: false, key }
    }

    /// Human-readable hint shown in the palette (e.g. `⌘⇧T`).
    pub fn label(&self) -> String {
        let mut s = String::new();
        if self.ctrl { s.push('⌃'); }
        if self.alt { s.push('⌥'); }
        if self.shift { s.push('⇧'); }
        if self.cmd { s.push('⌘'); }
        s.push_str(key_name(self.key));
        s
    }
}

fn key_name(key: KeyCode) -> &'static str {
    use KeyCode::*;
    match key {
        KeyA => "A", KeyB => "B", KeyC => "C", KeyD => "D", KeyE => "E",
        KeyF => "F", KeyG => "G", KeyH => "H", KeyI => "I", KeyJ => "J",
        KeyK => "K", KeyL => "L", KeyM => "M", KeyN => "N", KeyO => "O",
        KeyP => "P", KeyQ => "Q", KeyR => "R", KeyS => "S", KeyT => "T",
        KeyU => "U", KeyV => "V", KeyW => "W", KeyX => "X", KeyY => "Y",
        KeyZ => "Z", Backslash => "\\", _ => "?",
    }
}

/// How an action performs its effect when invoked.
#[derive(Clone, Copy)]
pub enum ActionRun {
    /// Spawn a registered pane kind into the active project. Auto-
    /// generated for every `PaneKindSpec`. `origin` (cursor) comes from
    /// the invocation; `config` is `Null`.
    SpawnPane {
        kind: &'static str,
        size: Option<Vec2>,
    },
    /// Arbitrary world mutation. The handler receives an [`ActionCtx`]
    /// and can do anything an old shortcut system did.
    Custom(fn(&mut ActionCtx)),
}

/// One thing the app can do.
#[derive(Clone, Copy)]
pub struct Action {
    /// Stable dispatch id, e.g. `"pane.spawn.terminal"`, `"view.toggle_cube"`.
    pub id: &'static str,
    /// Human label shown in the palette / radial.
    pub title: &'static str,
    /// Palette section, e.g. `"Panes"`, `"View"`, `"AI"`.
    pub category: &'static str,
    /// Extra fuzzy-search aliases beyond the title.
    pub keywords: &'static [&'static str],
    /// `Some(glyph)` makes the action eligible for the radial ring.
    pub radial_icon: Option<&'static str>,
    /// `Some(chord)` registers a global keyboard shortcut.
    pub default_keybind: Option<KeyChord>,
    /// The effect.
    pub run: ActionRun,
}

/// Registry resource. Keyed by id; `order` preserves first-registration
/// order for a stable palette listing.
#[derive(Resource, Default)]
pub struct ActionRegistry {
    by_id: HashMap<&'static str, Action>,
    order: Vec<&'static str>,
}

impl ActionRegistry {
    pub fn register(&mut self, action: Action) {
        if self.by_id.insert(action.id, action).is_none() {
            self.order.push(action.id);
        }
    }

    pub fn get(&self, id: &str) -> Option<&Action> {
        self.by_id.get(id)
    }

    /// Iterate in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &Action> {
        self.order.iter().map(move |id| &self.by_id[id])
    }

    /// Actions eligible for the radial ring (those with an icon).
    pub fn radial_items(&self) -> impl Iterator<Item = &Action> {
        self.iter().filter(|a| a.radial_icon.is_some())
    }
}

/// A queued request to run an action, drained by [`run_requested_actions`].
pub struct ActionInvocation {
    /// Action id. Owned because radial snapshots / palette results hand
    /// us strings, not `'static` literals.
    pub id: String,
    /// Cursor position for `SpawnPane` origin (radial). `None` = use the
    /// kind's normal cascade placement.
    pub origin: Option<Vec2>,
}

/// Pending action invocations for this frame.
#[derive(Resource, Default)]
pub struct ActionInvocations(pub Vec<ActionInvocation>);

impl ActionInvocations {
    pub fn request(&mut self, id: impl Into<String>, origin: Option<Vec2>) {
        self.0.push(ActionInvocation { id: id.into(), origin });
    }
}

/// Context handed to [`ActionRun::Custom`] handlers.
pub struct ActionCtx<'w> {
    pub world: &'w mut World,
    /// Cursor at invocation time (radial), if any.
    pub origin: Option<Vec2>,
}

/// Systems that *enqueue* action invocations (keybind matcher, radial
/// pick, palette Enter). [`run_requested_actions`] runs after this set
/// so picks dispatch the same frame.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActionProducerSet;

/// Extension trait mirroring `app.add_systems` — register a bespoke
/// action at plugin-build time. Requires [`ActionsPlugin`] added first.
pub trait AppActionsExt {
    fn add_action(&mut self, action: Action) -> &mut Self;
}

impl AppActionsExt for App {
    fn add_action(&mut self, action: Action) -> &mut Self {
        self.world_mut()
            .resource_mut::<ActionRegistry>()
            .register(action);
        self
    }
}

pub struct ActionsPlugin;

impl Plugin for ActionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActionRegistry>()
            .init_resource::<ActionInvocations>()
            // Pane registrations land in `Startup`; synthesize their
            // spawn-actions once those are all in.
            .add_systems(PostStartup, generate_pane_spawn_actions)
            .add_systems(Update, dispatch_action_keybinds.in_set(ActionProducerSet))
            .add_systems(Update, run_requested_actions.after(ActionProducerSet));
    }
}

/// Synthesize one `pane.spawn.<kind>` action per registered pane kind so
/// adding a pane plugin makes it appear in the radial *and* palette with
/// no extra bookkeeping. The kind's `radial_icon` carries over verbatim.
fn generate_pane_spawn_actions(registry: Res<PaneRegistry>, mut actions: ResMut<ActionRegistry>) {
    let specs: Vec<(&'static str, &'static str, Option<&'static str>)> = registry
        .iter()
        .map(|s| (s.kind, s.display_name, s.radial_icon))
        .collect();
    for (kind, display_name, radial_icon) in specs {
        // Leak the id once at startup — matches the `Box::leak` already
        // used for dynamic pane kinds in the SpawnWidget IPC handler.
        let id: &'static str = Box::leak(format!("pane.spawn.{kind}").into_boxed_str());
        actions.register(Action {
            id,
            title: display_name,
            category: "Panes",
            keywords: &[],
            radial_icon,
            default_keybind: None,
            run: ActionRun::SpawnPane { kind, size: None },
        });
    }
}

/// Read keyboard events once and enqueue any action whose chord matches.
fn dispatch_action_keybinds(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    registry: Res<ActionRegistry>,
    owner: Res<KeyboardOwner>,
    mut invocations: ResMut<ActionInvocations>,
) {
    if owner.is_modal() {
        // A text-entry modal (command palette / rename) owns the keyboard;
        // don't fire global chords while typing. Drain so events don't
        // pile up for next frame.
        events.clear();
        return;
    }
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);
    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        let chord = KeyChord { cmd, shift, alt, ctrl, key: ev.key_code };
        for action in registry.iter() {
            if action.default_keybind == Some(chord) {
                invocations.request(action.id, None);
            }
        }
    }
}

/// Exclusive system: drain queued invocations and perform each effect.
/// Looks each action up, copies it out (releasing the registry borrow),
/// then dispatches with `&mut World`.
pub fn run_requested_actions(world: &mut World) {
    let queued = std::mem::take(&mut world.resource_mut::<ActionInvocations>().0);
    for inv in queued {
        let Some(action) = world.resource::<ActionRegistry>().get(&inv.id).copied() else {
            warn!("action {:?} requested but not registered", inv.id);
            continue;
        };
        match action.run {
            ActionRun::SpawnPane { kind, size } => {
                let Some(active) = world.resource::<Projects>().active else {
                    continue;
                };
                world
                    .resource_mut::<PendingActions>()
                    .new_panes
                    .push(NewPaneRequest {
                        kind,
                        project_id: active,
                        origin: inv.origin,
                        size,
                        config: serde_json::Value::Null,
                    });
            }
            ActionRun::Custom(f) => {
                let mut ctx = ActionCtx { world, origin: inv.origin };
                f(&mut ctx);
            }
        }
    }
}
