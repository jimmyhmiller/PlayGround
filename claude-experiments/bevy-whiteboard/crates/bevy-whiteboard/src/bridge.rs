//! Bridges the pure `sim` core into Bevy. The `Sim` is the authoritative
//! source of truth for behavior; Bevy entities exist for rendering and
//! interaction. Packets you see traveling on the canvas are purely visual
//! animations spawned from `SimEvent::Traveled` — they don't feed back into
//! the simulation.

use crate::edges::Edge;
use crate::nodes::{NodeKind, SimNode};
use crate::sim::{self, EdgeId, NodeId, NS_PER_S, Sim, SimEvent};
use bevy::prelude::*;
use std::collections::HashMap;

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimResource>()
            .init_resource::<EntityMaps>()
            .init_resource::<TickEvents>()
            .init_resource::<SimSpeed>()
            .init_resource::<EdgeVisualState>()
            .add_systems(
                Startup,
                (enable_system_font_fallback, load_unicode_font, load_bold_font, load_mono_font).chain(),
            )
            .add_systems(Update, (apply_unicode_font_to_text, apply_bold_font, apply_mono_font).chain())
            .add_systems(
                Update,
                (
                    sim_speed_controls,
                    advance_sim,
                    spawn_traveling_packets,
                    animate_packets,
                    despawn_finished_packets,
                )
                    .chain(),
            );
    }
}

/// How the real-time clock maps onto sim time. `multiplier = 1.0` means one
/// real second advances the sim by one sim second. Ns-scale boards will want
/// multipliers far below 1 (e.g. `1e-6` runs the sim one microsecond of sim
/// time per real second — watchable). `paused` freezes the sim entirely.
#[derive(Resource)]
pub struct SimSpeed {
    pub multiplier: f64,
    pub paused: bool,
    /// When set, the next `advance_sim` tick treats the sim as unpaused for
    /// one frame and injects this much sim time. Used by the step-forward key.
    pub step_once_ns: Option<u64>,
}

impl Default for SimSpeed {
    fn default() -> Self {
        Self { multiplier: 1.0, paused: false, step_once_ns: None }
    }
}

#[derive(Resource, Default)]
pub struct SimResource(pub Sim);

/// Bidirectional mapping between Bevy entities and sim ids. The sim owns
/// behavioral state; Bevy entities own position, rendering, and interaction.
#[derive(Resource, Default)]
pub struct EntityMaps {
    pub entity_to_node: HashMap<Entity, NodeId>,
    pub node_to_entity: HashMap<NodeId, Entity>,
    pub entity_to_edge: HashMap<Entity, EdgeId>,
    pub edge_to_entity: HashMap<EdgeId, Entity>,
}

/// Attached to each node entity. Lookup into `SimResource`.
#[derive(Component, Clone, Copy)]
pub struct SimNodeRef(pub NodeId);

/// Attached to each edge entity.
#[derive(Component, Clone, Copy)]
pub struct SimEdgeRef(pub EdgeId);

/// Purely visual — one entity per sim `Traveled` event, animated from
/// edge.from to edge.to over `VISUAL_PACKET_SECONDS`.
#[derive(Component)]
pub struct TravelingPacket {
    pub edge_entity: Entity,
    pub t: f32,
    pub duration: f32,
    pub color: Color,
    /// When true, this is a response traveling backward along the edge —
    /// rendered as a ring rather than a filled disc, and animated from
    /// `edge.to` back to `edge.from`.
    pub is_response: bool,
}

/// Events produced by the sim during the most recent advance. Consumed by
/// `spawn_traveling_packets`; other systems may read this too if useful.
#[derive(Resource, Default)]
pub struct TickEvents(pub Vec<SimEvent>);

/// Per-edge throttle for visual packet spawning. The sim can emit billions of
/// `Traveled` events per real second on ns-scale boards; spawning an entity
/// for each would swamp the renderer. We cap each edge to one visual spawn
/// every `MIN_SPAWN_INTERVAL` of real time. Truth about flow rates still
/// lives on the probe labels — this is cosmetic ("visuals can lie").
#[derive(Resource, Default)]
pub struct EdgeVisualState {
    /// Real-time seconds (from `Time::elapsed_secs`) of the last spawn per edge.
    pub last_spawn: HashMap<EdgeId, f32>,
}

pub const MIN_SPAWN_INTERVAL: f32 = 0.05;

/// How long (real-time seconds) a packet takes to visually cross an edge,
/// regardless of the edge's length. This is cosmetic — the sim already
/// decided the outcome by the time we spawn one of these.
pub const VISUAL_PACKET_SECONDS: f32 = 0.6;

// ---- Color bridging ------------------------------------------------------

pub fn bevy_to_sim_color(c: Color) -> sim::Color {
    let s = c.to_srgba();
    let r = (s.red.clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (s.green.clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (s.blue.clamp(0.0, 1.0) * 255.0).round() as u32;
    sim::Color((r << 16) | (g << 8) | b)
}

pub fn sim_to_bevy_color(c: sim::Color) -> Color {
    let r = ((c.0 >> 16) & 0xff) as f32 / 255.0;
    let g = ((c.0 >> 8) & 0xff) as f32 / 255.0;
    let b = (c.0 & 0xff) as f32 / 255.0;
    Color::srgb(r, g, b)
}

// ---- Registration helpers (called from node/edge spawn code) -------------

/// Registers a newly-spawned node entity with the sim. Returns the sim NodeId
/// so the caller can attach a `SimNodeRef` component.
pub fn register_node(
    sim_res: &mut SimResource,
    maps: &mut EntityMaps,
    entity: Entity,
    kind: NodeKind,
    color: Color,
) -> NodeId {
    let sc = bevy_to_sim_color(color);
    let nid = match kind {
        // Defaults: 2 pkts/s generator, 500ms worker — same as the old API.
        NodeKind::Generator => sim_res.0.add_generator(sc, NS_PER_S / 2),
        NodeKind::Client => sim_res.0.add_client(sc, NS_PER_S),
        NodeKind::Worker => sim_res.0.add_worker(sc, NS_PER_S / 2),
        NodeKind::Sink => sim_res.0.add_sink(sc),
        NodeKind::Router => sim_res.0.add_router(),
        NodeKind::Queue => sim_res.0.add_queue(sc, usize::MAX),
        NodeKind::Custom => {
            panic!("register_node: Custom must be bound via bind_existing_node");
        }
        NodeKind::Steps => {
            // Empty by default — the user adds rows by clicking the
            // Client/Worker palette tools inside the container.
            sim_res.0.add_steps(sc, Vec::new())
        }
    };
    maps.entity_to_node.insert(entity, nid);
    maps.node_to_entity.insert(nid, entity);
    nid
}

/// Bind a Bevy entity to an already-existing sim node id. Used when a
/// composite is created via `Sim::group_into_composite` — the node
/// already lives in the sim; we just need entity-maps bookkeeping.
pub fn bind_existing_node(maps: &mut EntityMaps, entity: Entity, nid: NodeId) {
    maps.entity_to_node.insert(entity, nid);
    maps.node_to_entity.insert(nid, entity);
}

/// Bind a Bevy edge entity to an already-existing sim edge id.
pub fn bind_existing_edge(maps: &mut EntityMaps, entity: Entity, eid: EdgeId) {
    maps.entity_to_edge.insert(entity, eid);
    maps.edge_to_entity.insert(eid, entity);
}

/// Registers a newly-spawned edge entity with the sim. Returns the new EdgeId.
pub fn register_edge(
    sim_res: &mut SimResource,
    maps: &mut EntityMaps,
    edge_entity: Entity,
    from_ent: Entity,
    to_ent: Entity,
) -> Option<EdgeId> {
    let from = *maps.entity_to_node.get(&from_ent)?;
    let to = *maps.entity_to_node.get(&to_ent)?;
    let eid = sim_res.0.connect(from, to);
    maps.entity_to_edge.insert(edge_entity, eid);
    maps.edge_to_entity.insert(eid, edge_entity);
    Some(eid)
}

// ---- Systems -------------------------------------------------------------

fn advance_sim(
    time: Res<Time>,
    mut sim_res: ResMut<SimResource>,
    mut events: ResMut<TickEvents>,
    mut speed: ResMut<SimSpeed>,
) {
    events.0.clear();
    // Cap real-frame delta at 100ms so a paused-then-resumed window (or
    // alt-tabbed tab) doesn't dump a massive chunk of sim time in one frame.
    let real_dt = time.delta_secs().min(0.1) as f64;

    let dt_ns: u64 = if let Some(step) = speed.step_once_ns.take() {
        step
    } else if speed.paused {
        0
    } else {
        let scaled = real_dt * speed.multiplier * NS_PER_S as f64;
        scaled.max(0.0).min(u64::MAX as f64) as u64
    };

    if dt_ns > 0 {
        events.0.extend(sim_res.0.advance_ns(dt_ns));
    }
}

/// Keybindings for sim-speed control:
/// - Space: pause/unpause.
/// - `[` / `]`: divide / multiply speed by 2 (clamped to a huge range).
/// - `0`: reset to 1× real-time.
/// - `.`: while paused, advance by one real-frame's worth of sim at the
///   current multiplier (acts like a single step).
/// A `Handle<Font>` pointing to the unicode-capable font we loaded at
/// startup. Used by `apply_unicode_font_to_text` to stamp every text entity
/// whose `TextFont.font` is still the default (tofu-producing) handle.
#[derive(Resource, Default)]
struct UnicodeFont(Handle<bevy::text::Font>);

/// Optional bold-weight font. When loaded, [`apply_bold_font`] re-stamps it
/// onto any `TextFont` paired with the [`Bold`] marker, overriding the
/// default body-weight stamp. Falls back silently — UI just stays in the
/// medium weight if the bold asset can't be loaded.
#[derive(Resource, Default)]
pub struct BoldFont(pub Handle<bevy::text::Font>);

/// Marker: this text entity wants the bold weight. Add alongside `TextFont`
/// at spawn time. The per-frame applier handles the rest.
#[derive(Component)]
pub struct Bold;

/// Marker: this text entity wants the monospace face. Live numeric
/// readouts (counts, durations, rates) use `Mono` so their width
/// doesn't jitter as digits change. The per-frame
/// [`apply_mono_font`] system stamps the mono font onto matching
/// text widgets. Falls back silently if the font didn't load.
#[derive(Component)]
pub struct Mono;

/// Handle to the loaded monospace font. Default = unloaded.
#[derive(Resource, Default)]
pub struct MonoFont(pub Handle<bevy::text::Font>);

fn load_mono_font(mut fonts: ResMut<Assets<bevy::text::Font>>, mut commands: Commands) {
    // Probe order: bundled first (if we ever ship one), then the
    // well-known system monos. First hit wins.
    const CANDIDATES: &[&str] = &[
        "assets/fonts/JetBrainsMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Courier.ttc",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = bevy::text::Font::try_from_bytes(bytes) {
                commands.insert_resource(MonoFont(fonts.add(font)));
                return;
            }
        }
    }
    commands.insert_resource(MonoFont::default());
}

fn apply_mono_font(mono: Res<MonoFont>, mut q: Query<&mut TextFont, With<Mono>>) {
    if mono.0 == Handle::default() {
        return;
    }
    for mut tf in q.iter_mut() {
        if tf.font != mono.0 {
            tf.font = mono.0.clone();
        }
    }
}

fn load_bold_font(
    mut fonts: ResMut<Assets<bevy::text::Font>>,
    mut commands: Commands,
) {
    let path = "assets/fonts/Jost-Bold.ttf";
    match std::fs::read(path) {
        Ok(bytes) => match bevy::text::Font::try_from_bytes(bytes) {
            Ok(font) => {
                let handle = fonts.add(font);
                commands.insert_resource(BoldFont(handle));
            }
            Err(_) => {
                commands.insert_resource(BoldFont::default());
            }
        },
        Err(_) => {
            commands.insert_resource(BoldFont::default());
        }
    }
}

fn apply_bold_font(
    bold: Res<BoldFont>,
    mut q: Query<&mut TextFont, With<Bold>>,
) {
    if bold.0 == Handle::default() {
        return;
    }
    for mut tf in q.iter_mut() {
        if tf.font != bold.0 {
            tf.font = bold.0.clone();
        }
    }
}

/// Bevy's bundled `default_font` (FiraMono subset) is missing glyphs for
/// characters we use in labels — µ, ×, ÷, •, ⚠, ‖, Σ — so each renders as a
/// tofu box. Inserting at `AssetId::default()` doesn't help: cosmic-text
/// caches font faces against the handle on first render, so a late swap at
/// the same id is ignored. Instead we load the font at a *fresh* handle and
/// a per-frame system reassigns that handle onto any `TextFont` still using
/// the default.
///
/// macOS ships `Arial Unicode.ttf` with complete BMP coverage. Falls back
/// silently on other platforms — text just stays in FiraMono.
/// Replace Bevy's default `CosmicFontSystem` — which is constructed with
/// `FontSystem::new_with_locale_and_db` holding an empty font database — with
/// one that loads every font installed on the system. Cosmic's text layout
/// picks glyphs per-character; when our primary font lacks one (e.g. Arial
/// Unicode has • but not ⚠, Helvetica has neither but has †), Cosmic falls
/// back to whichever system font does. Without this, missing glyphs render as
/// tofu regardless of which primary font we pick.
fn enable_system_font_fallback(mut fs: ResMut<bevy::text::CosmicFontSystem>) {
    let new_fs = cosmic_text::FontSystem::new();
    eprintln!(
        "[bridge] enabled system-font fallback ({} faces loaded)",
        new_fs.db().len()
    );
    fs.0 = new_fs;
}

fn load_unicode_font(
    mut fonts: ResMut<Assets<bevy::text::Font>>,
    mut commands: Commands,
) {
    // Probe order matters: the first one that loads becomes the primary
    // typeface for the entire UI (the per-frame applier stamps it onto every
    // `TextFont`). The iso50 design wants Futura — Avenir Next is the
    // closest still-clean fallback; the rest are pure unicode-coverage
    // safety nets so missing glyphs don't tofu.
    // Probe the medium-weight body font first — most of the UI is body text
    // and Bold-everywhere reads as shouty. Bold is loaded separately below
    // and applied opt-in via the [`Bold`] marker.
    const CANDIDATES: &[&str] = &[
        "assets/fonts/Jost-Medium.ttf",
        "assets/fonts/Jost-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/Futura.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Geneva.ttf",
    ];
    for path in CANDIDATES {
        match std::fs::read(path) {
            Ok(bytes) => {
                let len = bytes.len();
                match bevy::text::Font::try_from_bytes(bytes) {
                    Ok(font) => {
                        let handle = fonts.add(font);
                        // Use eprintln so the message lands even without
                        // RUST_LOG configured — the diagnostic is otherwise
                        // invisible and we spent a round-trip figuring that out.
                        eprintln!(
                            "[bridge] loaded unicode font '{}' ({} bytes) at handle {:?}",
                            path, len, handle.id()
                        );
                        commands.insert_resource(UnicodeFont(handle));
                        return;
                    }
                    Err(e) => {
                        eprintln!("[bridge] '{}' parsed as NOT a valid font: {:?}", path, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("[bridge] could not read '{}': {}", path, e);
            }
        }
    }
    eprintln!("[bridge] no unicode font could be loaded; labels will tofu");
    commands.insert_resource(UnicodeFont::default());
}

/// Stamps the unicode font handle onto every `TextFont` component. Runs
/// every frame so newly spawned text (palette buttons, node labels, probes,
/// the status overlay) picks it up. We unconditionally overwrite rather
/// than compare-and-swap — Bevy's `Mut<T>` only flags the component as
/// changed when actually dereferenced mutably, so guarding the write on
/// `font != our_handle` keeps this cheap. `fonts` is `Res`, not `ResMut`,
/// so the underlying asset isn't touched.
fn apply_unicode_font_to_text(
    font: Res<UnicodeFont>,
    mut q: Query<&mut TextFont>,
) {
    if font.0 == Handle::default() {
        return;
    }
    for mut tf in q.iter_mut() {
        if tf.font != font.0 {
            tf.font = font.0.clone();
        }
    }
}

fn sim_speed_controls(
    keys: Res<ButtonInput<KeyCode>>,
    sim_res: Res<SimResource>,
    mut speed: ResMut<SimSpeed>,
) {
    if keys.just_pressed(KeyCode::Space) {
        speed.paused = !speed.paused;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        speed.multiplier = (speed.multiplier * 0.5).max(1e-15);
        eprintln!("[sim] speed = {:e}×", speed.multiplier);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        speed.multiplier = (speed.multiplier * 2.0).min(1e9);
        eprintln!("[sim] speed = {:e}×", speed.multiplier);
    }
    if keys.just_pressed(KeyCode::Digit0) {
        speed.multiplier = 1.0;
        eprintln!("[sim] speed = {:e}×", speed.multiplier);
    }
    if keys.just_pressed(KeyCode::Period) && speed.paused {
        // Step to the next scheduled sim event. On a mixed-scale board, a
        // fixed-time step is useless — "one frame of sim time" either fires
        // nothing (ms-scale board at high slowdown) or fires millions of
        // events (ns-scale board). Jumping to the next event always produces
        // exactly one visible thing happening.
        let now = sim_res.0.now_ns;
        let step = sim_res
            .0
            .next_event_ns()
            .map(|t| t.saturating_sub(now).max(1))
            .unwrap_or(1);
        speed.step_once_ns = Some(step);
    }
}

fn spawn_traveling_packets(
    time: Res<Time>,
    events: Res<TickEvents>,
    maps: Res<EntityMaps>,
    mut vis: ResMut<EdgeVisualState>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let now_real = time.elapsed_secs();
    for ev in &events.0 {
        let SimEvent::Traveled { edge, color, is_response } = ev else {
            continue;
        };
        let Some(&edge_entity) = maps.edge_to_entity.get(edge) else {
            continue;
        };
        // Coalesce bursty traffic: only spawn one visual packet per edge per
        // `MIN_SPAWN_INTERVAL` of real time. Drops everything else — probes
        // still reflect the true sim rate. Requests and responses share the
        // same bucket so a ping-pong pair only spawns one visual per window
        // (picking whichever arrived first).
        let last = vis.last_spawn.get(edge).copied().unwrap_or(f32::NEG_INFINITY);
        if now_real - last < MIN_SPAWN_INTERVAL {
            continue;
        }
        vis.last_spawn.insert(*edge, now_real);
        let bevy_color = sim_to_bevy_color(*color);
        let mesh = if *is_response {
            // Hollow ring — the outer radius matches the filled-disc packet so
            // response visuals are recognizable at a glance.
            meshes.add(Annulus::new(3.5, 6.0))
        } else {
            meshes.add(Circle::new(6.0))
        };
        commands.spawn((
            TravelingPacket {
                edge_entity,
                t: 0.0,
                duration: VISUAL_PACKET_SECONDS,
                color: bevy_color,
                is_response: *is_response,
            },
            Mesh2d(mesh),
            MeshMaterial2d(materials.add(bevy_color)),
            Transform::from_xyz(0.0, 0.0, 0.5),
            Visibility::default(),
        ));
    }
}

fn animate_packets(
    time: Res<Time>,
    edges: Query<&Edge>,
    nodes_q: Query<&Transform, (With<SimNode>, Without<TravelingPacket>)>,
    mut packets: Query<(&mut TravelingPacket, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (mut p, mut tf) in packets.iter_mut() {
        p.t += dt / p.duration;
        let Ok(edge) = edges.get(p.edge_entity) else {
            continue;
        };
        let (Ok(a), Ok(b)) = (nodes_q.get(edge.from), nodes_q.get(edge.to)) else {
            continue;
        };
        let (start, end) = if p.is_response {
            (b.translation.truncate(), a.translation.truncate())
        } else {
            (a.translation.truncate(), b.translation.truncate())
        };
        let pos = start.lerp(end, p.t.clamp(0.0, 1.0));
        tf.translation.x = pos.x;
        tf.translation.y = pos.y;
    }
}

fn despawn_finished_packets(mut commands: Commands, packets: Query<(Entity, &TravelingPacket)>) {
    for (e, p) in packets.iter() {
        if p.t >= 1.0 {
            commands.entity(e).despawn();
        }
    }
}
