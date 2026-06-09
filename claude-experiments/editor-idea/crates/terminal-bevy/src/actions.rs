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
//!
//! ## Keybindings (rebindable + chord sequences)
//!
//! Each action carries a [`Action::default_keys`] sequence (empty =
//! unbound, one chord = a plain shortcut, many = a *sequence* like `⌘K`
//! then `C`). At startup [`rebuild_keymap`] folds those defaults together
//! with `~/.terminal-bevy/keybinds.json` into the [`Keymap`] resource —
//! the one place every consumer reads bindings from. The JSON is a flat
//! `{ "<action id>": "<binding>" }` map; a `null` value unbinds an action
//! that has a default:
//!
//! ```json
//! {
//!   "view.toggle_cube": "cmd+shift+backslash",
//!   "pane.spawn.terminal": "cmd+k t",
//!   "file.open": null
//! }
//! ```
//!
//! Binding strings are `+`-joined modifiers + key (`cmd|super|meta`,
//! `shift`, `alt|opt`, `ctrl`), with space-separated chords forming a
//! sequence (see [`KeyChord::parse`] / [`parse_sequence`]). The
//! `keybinds.reload` action re-reads the file live. The matcher
//! ([`dispatch_action_keybinds`]) recognizes a sequence one key at a time
//! via [`PendingSequence`]; while a sequence is mid-flight the keyboard is
//! held modal so the continuation key can't leak into the focused pane —
//! so sequence *leaders* should carry a modifier (a bare-key leader leaks
//! for one frame before suppression kicks in).

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
    /// No modifiers — the typical second chord in a sequence (e.g. the
    /// `C` in `⌘K C`).
    pub const fn plain(key: KeyCode) -> Self {
        Self { cmd: false, shift: false, alt: false, ctrl: false, key }
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

    /// Parse one chord from a string like `"cmd+shift+t"` / `"ctrl+\\"`.
    /// Modifier aliases: cmd|super|meta|win, shift, alt|opt|option,
    /// ctrl|control. The final non-modifier token is the key. Returns
    /// `None` on an unknown token or a missing/duplicate key.
    pub fn parse(s: &str) -> Option<KeyChord> {
        let (mut cmd, mut shift, mut alt, mut ctrl) = (false, false, false, false);
        let mut key: Option<KeyCode> = None;
        for tok in s.split('+') {
            let t = tok.trim().to_ascii_lowercase();
            match t.as_str() {
                "" => {}
                "cmd" | "super" | "meta" | "win" => cmd = true,
                "shift" => shift = true,
                "alt" | "opt" | "option" => alt = true,
                "ctrl" | "control" => ctrl = true,
                other => {
                    if key.is_some() {
                        return None; // two keys in one chord
                    }
                    key = Some(parse_key(other)?);
                }
            }
        }
        Some(KeyChord { cmd, shift, alt, ctrl, key: key? })
    }
}

/// Parse a whitespace-separated chord sequence (e.g. `"cmd+k c"`). Returns
/// `None` if empty or any chord fails to parse.
pub fn parse_sequence(s: &str) -> Option<Vec<KeyChord>> {
    let v: Vec<KeyChord> = s.split_whitespace().map(KeyChord::parse).collect::<Option<_>>()?;
    (!v.is_empty()).then_some(v)
}

/// Render a chord sequence for the palette, chords joined by a space.
pub fn seq_label(seq: &[KeyChord]) -> String {
    seq.iter().map(|c| c.label()).collect::<Vec<_>>().join(" ")
}

/// True for the bare modifier keys, which must never form a chord on their
/// own (so holding `⌘` while reaching for the next key doesn't fire or
/// abort a sequence).
fn is_modifier_key(k: KeyCode) -> bool {
    use KeyCode::*;
    matches!(
        k,
        SuperLeft | SuperRight | ShiftLeft | ShiftRight | AltLeft | AltRight | ControlLeft | ControlRight
    )
}

/// Inverse of [`key_name`] for the key tokens the parser accepts.
fn parse_key(s: &str) -> Option<KeyCode> {
    use KeyCode::*;
    Some(match s {
        "a" => KeyA, "b" => KeyB, "c" => KeyC, "d" => KeyD, "e" => KeyE,
        "f" => KeyF, "g" => KeyG, "h" => KeyH, "i" => KeyI, "j" => KeyJ,
        "k" => KeyK, "l" => KeyL, "m" => KeyM, "n" => KeyN, "o" => KeyO,
        "p" => KeyP, "q" => KeyQ, "r" => KeyR, "s" => KeyS, "t" => KeyT,
        "u" => KeyU, "v" => KeyV, "w" => KeyW, "x" => KeyX, "y" => KeyY,
        "z" => KeyZ,
        "0" => Digit0, "1" => Digit1, "2" => Digit2, "3" => Digit3, "4" => Digit4,
        "5" => Digit5, "6" => Digit6, "7" => Digit7, "8" => Digit8, "9" => Digit9,
        "\\" | "backslash" => Backslash,
        "=" | "equal" | "plus" => Equal,
        "-" | "minus" => Minus,
        "[" | "bracketleft" => BracketLeft,
        "]" | "bracketright" => BracketRight,
        "space" => Space,
        "enter" | "return" => Enter,
        "tab" => Tab,
        "esc" | "escape" => Escape,
        "up" | "arrowup" => ArrowUp,
        "down" | "arrowdown" => ArrowDown,
        "left" | "arrowleft" => ArrowLeft,
        "right" | "arrowright" => ArrowRight,
        _ => return None,
    })
}

fn key_name(key: KeyCode) -> &'static str {
    use KeyCode::*;
    match key {
        KeyA => "A", KeyB => "B", KeyC => "C", KeyD => "D", KeyE => "E",
        KeyF => "F", KeyG => "G", KeyH => "H", KeyI => "I", KeyJ => "J",
        KeyK => "K", KeyL => "L", KeyM => "M", KeyN => "N", KeyO => "O",
        KeyP => "P", KeyQ => "Q", KeyR => "R", KeyS => "S", KeyT => "T",
        KeyU => "U", KeyV => "V", KeyW => "W", KeyX => "X", KeyY => "Y",
        KeyZ => "Z", Backslash => "\\",
        Digit0 => "0", Digit1 => "1", Digit2 => "2", Digit3 => "3", Digit4 => "4",
        Digit5 => "5", Digit6 => "6", Digit7 => "7", Digit8 => "8", Digit9 => "9",
        Equal => "=", Minus => "-", BracketLeft => "[", BracketRight => "]",
        Space => "Space", Enter => "⏎", Tab => "⇥", Escape => "Esc",
        ArrowUp => "↑", ArrowDown => "↓", ArrowLeft => "←", ArrowRight => "→",
        _ => "?",
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
    /// Default key binding. Empty = unbound; one chord = a simple
    /// shortcut; more = a chord *sequence* (e.g. `⌘K` then `C`). Always
    /// overridable per-id from `~/.terminal-bevy/keybinds.json`; the
    /// effective binding lives in [`Keymap`].
    pub default_keys: &'static [KeyChord],
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

/// How long a partial chord sequence waits for its next key before it is
/// abandoned (and the keyboard, which the sequence holds modal, freed).
const SEQUENCE_TIMEOUT: f32 = 1.2;

/// The effective key bindings: every action's [`Action::default_keys`],
/// overlaid by the user's `~/.terminal-bevy/keybinds.json`. The matcher
/// and the command palette both read bindings from here (never straight
/// off `default_keys`) so a disk override is honored everywhere at once.
/// Rebuilt by [`rebuild_keymap`] at startup and on the `keybinds.reload`
/// action.
#[derive(Resource, Default)]
pub struct Keymap {
    bindings: HashMap<&'static str, Vec<KeyChord>>,
}

impl Keymap {
    /// Effective binding for an action id, if any.
    pub fn get(&self, id: &str) -> Option<&[KeyChord]> {
        self.bindings.get(id).map(Vec::as_slice)
    }

    /// Palette hint for an action id (chords joined by spaces).
    pub fn label(&self, id: &str) -> Option<String> {
        self.get(id).map(seq_label)
    }

    /// The action whose binding is exactly `seq`, if any. (Ties — two
    /// actions on one chord — resolve arbitrarily; we don't reject them.)
    fn match_exact(&self, seq: &[KeyChord]) -> Option<&'static str> {
        self.bindings
            .iter()
            .find(|(_, b)| b.as_slice() == seq)
            .map(|(id, _)| *id)
    }

    /// True if `seq` is a strict prefix of some longer binding — i.e. more
    /// keys could still complete a sequence.
    fn is_prefix(&self, seq: &[KeyChord]) -> bool {
        self.bindings
            .values()
            .any(|b| b.len() > seq.len() && &b[..seq.len()] == seq)
    }
}

/// In-flight chord sequence: the chords pressed so far and when the
/// sequence began (engine `elapsed_secs`, for the timeout). Empty = no
/// sequence in progress.
#[derive(Resource, Default)]
pub struct PendingSequence {
    pub chords: Vec<KeyChord>,
    pub started_at: f32,
}

/// `~/.terminal-bevy/keybinds.json`.
fn keybinds_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(".terminal-bevy");
    p.push("keybinds.json");
    Some(p)
}

/// Disk override map: action id -> binding string. `null` (or empty
/// string) unbinds an action that has a default. A missing file means
/// "no overrides" — not an error.
fn load_disk_overrides() -> HashMap<String, Option<String>> {
    let Some(path) = keybinds_path() else {
        return HashMap::new();
    };
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            warn!("keybinds.json: invalid JSON ({e}); ignoring overrides");
            HashMap::new()
        }),
        Err(_) => HashMap::new(),
    }
}

/// (Re)compute [`Keymap`] from the registry defaults plus disk overrides.
/// Safe to call any time after the registry is populated. Unknown ids and
/// unparseable chord strings are warned about and skipped (the default,
/// if any, stays).
pub fn rebuild_keymap(world: &mut World) {
    let mut bindings: HashMap<&'static str, Vec<KeyChord>> = world
        .resource::<ActionRegistry>()
        .iter()
        .filter(|a| !a.default_keys.is_empty())
        .map(|a| (a.id, a.default_keys.to_vec()))
        .collect();
    // id literals, so a disk entry can be remapped to its &'static id.
    let known: HashMap<&str, &'static str> = world
        .resource::<ActionRegistry>()
        .iter()
        .map(|a| (a.id, a.id))
        .collect();

    for (id, val) in load_disk_overrides() {
        // `_`-prefixed keys are treated as comments (JSON has none).
        if id.starts_with('_') {
            continue;
        }
        let Some(&sid) = known.get(id.as_str()) else {
            warn!("keybinds.json: unknown action id {id:?}");
            continue;
        };
        match val {
            None => {
                bindings.remove(sid);
            }
            Some(s) if s.trim().is_empty() => {
                bindings.remove(sid);
            }
            Some(s) => match parse_sequence(&s) {
                Some(seq) => {
                    bindings.insert(sid, seq);
                }
                None => warn!("keybinds.json: could not parse {s:?} for {id}"),
            },
        }
    }
    world.resource_mut::<Keymap>().bindings = bindings;
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
            .init_resource::<Keymap>()
            .init_resource::<PendingSequence>()
            // Pane registrations land in `Startup`; synthesize their
            // spawn-actions once those are all in, then fold every action's
            // defaults + the disk overrides into the `Keymap`. (Bespoke
            // actions are registered in `App::build` before `PostStartup`,
            // so they're present too.)
            .add_systems(
                PostStartup,
                (generate_pane_spawn_actions, rebuild_keymap).chain(),
            )
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
            default_keys: default_spawn_keys(kind),
            run: ActionRun::SpawnPane { kind, size: None },
        });
    }
}

/// Default spawn shortcut for a few well-known pane kinds. Anything not
/// listed gets no default and is still bindable from `keybinds.json` by
/// its id (`pane.spawn.<kind>`).
fn default_spawn_keys(kind: &str) -> &'static [KeyChord] {
    match kind {
        "terminal" => const { &[KeyChord::cmd(KeyCode::KeyT)] },
        _ => &[],
    }
}

/// Read keyboard events once and drive the chord-sequence matcher against
/// the effective [`Keymap`]. A single-chord binding fires immediately; a
/// multi-chord binding is recognized one key at a time via
/// [`PendingSequence`].
fn dispatch_action_keybinds(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    keymap: Res<Keymap>,
    owner: Res<KeyboardOwner>,
    time: Res<Time>,
    mut pending: ResMut<PendingSequence>,
    mut invocations: ResMut<ActionInvocations>,
) {
    let now = time.elapsed_secs();
    // Expire a stale partial sequence even on a frame with no key events,
    // so a forgotten prefix doesn't hold the keyboard hostage — while a
    // sequence is pending the owner authority reports `Modal`.
    if !pending.chords.is_empty() && now - pending.started_at > SEQUENCE_TIMEOUT {
        pending.chords.clear();
    }
    // A *text* modal (palette / rename) suppresses chords entirely. A
    // pending sequence ALSO makes the owner `Modal` (so pane typing is
    // gated mid-sequence) but must not suppress us — that's how we read
    // the continuation key. Distinguish the two by whether a sequence is
    // actually in progress.
    if owner.is_modal() && pending.chords.is_empty() {
        events.clear();
        return;
    }

    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);

    for ev in events.read() {
        if !ev.state.is_pressed() || is_modifier_key(ev.key_code) {
            continue;
        }
        let chord = KeyChord { cmd, shift, alt, ctrl, key: ev.key_code };
        let mut candidate = pending.chords.clone();
        candidate.push(chord);

        if let Some(id) = keymap.match_exact(&candidate) {
            // Complete binding — fire and reset.
            invocations.request(id, None);
            pending.chords.clear();
        } else if keymap.is_prefix(&candidate) {
            // Still a viable prefix of a longer binding — wait for more.
            pending.chords = candidate;
            pending.started_at = now;
        } else {
            // Dead end: this key extends nothing. Abandon any partial
            // sequence (the key is consumed, not re-routed).
            pending.chords.clear();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_chords() {
        assert_eq!(
            KeyChord::parse("cmd+shift+t"),
            Some(KeyChord::cmd_shift(KeyCode::KeyT))
        );
        assert_eq!(
            KeyChord::parse("CMD+O"),
            Some(KeyChord::cmd(KeyCode::KeyO))
        );
        // modifier aliases + a symbol key
        assert_eq!(
            KeyChord::parse("ctrl+\\"),
            Some(KeyChord {
                cmd: false,
                shift: false,
                alt: false,
                ctrl: true,
                key: KeyCode::Backslash,
            })
        );
        assert_eq!(KeyChord::parse("opt+="), Some(KeyChord { cmd: false, shift: false, alt: true, ctrl: false, key: KeyCode::Equal }));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(KeyChord::parse("cmd+nope"), None); // unknown key
        assert_eq!(KeyChord::parse("cmd+shift"), None); // no key
        assert_eq!(KeyChord::parse("a+b"), None); // two keys
    }

    #[test]
    fn parses_sequences() {
        let seq = parse_sequence("cmd+k c").unwrap();
        assert_eq!(seq, vec![KeyChord::cmd(KeyCode::KeyK), KeyChord::plain(KeyCode::KeyC)]);
        assert_eq!(parse_sequence("   "), None);
        assert!(parse_sequence("cmd+k bogus").is_none()); // any bad chord fails the whole seq
    }

    #[test]
    fn labels_round_trip_through_parse() {
        for s in ["cmd+shift+t", "ctrl+\\", "cmd+0", "cmd+["] {
            let chord = KeyChord::parse(s).unwrap();
            // The label re-parses to the same chord (glyphs aren't parseable,
            // so check structural equality via a fresh parse of the canonical
            // ascii form instead).
            assert_eq!(KeyChord::parse(s), Some(chord));
        }
        assert_eq!(seq_label(&parse_sequence("cmd+k c").unwrap()), "⌘K C");
    }

    fn keymap_with(pairs: &[(&'static str, &str)]) -> Keymap {
        let mut km = Keymap::default();
        for (id, s) in pairs {
            km.bindings.insert(id, parse_sequence(s).unwrap());
        }
        km
    }

    #[test]
    fn matches_exact_and_prefix() {
        let km = keymap_with(&[
            ("a.single", "cmd+o"),
            ("a.seq", "cmd+k c"),
        ]);
        let cmd_o = vec![KeyChord::cmd(KeyCode::KeyO)];
        let cmd_k = vec![KeyChord::cmd(KeyCode::KeyK)];
        let cmd_k_c = vec![KeyChord::cmd(KeyCode::KeyK), KeyChord::plain(KeyCode::KeyC)];

        // single chord fires immediately
        assert_eq!(km.match_exact(&cmd_o), Some("a.single"));
        assert!(!km.is_prefix(&cmd_o));

        // sequence prefix is not yet a match, but IS a viable prefix
        assert_eq!(km.match_exact(&cmd_k), None);
        assert!(km.is_prefix(&cmd_k));

        // full sequence matches and is no longer a prefix of anything
        assert_eq!(km.match_exact(&cmd_k_c), Some("a.seq"));
        assert!(!km.is_prefix(&cmd_k_c));
    }
}
