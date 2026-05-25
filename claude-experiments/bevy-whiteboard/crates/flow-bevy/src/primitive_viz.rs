//! High-fidelity mechanical visualizations for the 8 Life-cell primitive
//! gadgets. Each one trades the generic shape+glyph rendering for a
//! purpose-built moving-parts visual:
//!
//! - Tick           — swinging pendulum (heartbeat)
//! - Switch         — railway points with rotating blade
//! - ConstantPacket — stamping press with engraved value
//! - ConstantSignal — telegraph striker (signals are visually distinct)
//! - FanOut         — sunburst manifold with nozzle spokes
//! - Egress         — bulkhead / wall plug
//! - Aggregator     — 3×3-minus-center abacus (mirrors Conway neighbourhood)
//! - Filter         — coin sorter with a notched template
//!
//! Implementation sketch: each primitive class gets a [`PrimitiveKind`]
//! marker attached during canvas spawn. A per-frame system reads slot
//! state (from the sim snapshot) and packet activity (from `NewEvents`)
//! and updates child-entity transforms / materials accordingly. The
//! generic body shape is suppressed via [`PrimitiveBody`] — these
//! primitives draw their own housing.

use std::collections::HashMap;

use bevy::prelude::*;
use flow::{Event, NodeId};

use crate::bridge::{FlowNodeRef, NewEvents, SimClock};
use crate::sim_driver::SimSnapshotRes;

pub struct PrimitiveVizPlugin;
impl Plugin for PrimitiveVizPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                ingest_packet_events,
                construct_primitive_visuals,
                animate_tick,
                animate_switch,
                animate_constant_packet,
                animate_constant_signal,
                animate_fanout,
                animate_egress,
                animate_aggregator,
                animate_filter,
            )
                .chain(),
        );
    }
}

/// Which of the 8 primitive visuals this node is rendered as. Attached
/// during canvas spawn for any node whose class matches a stock
/// primitive. Drives the per-primitive animation systems and tells the
/// generic node-render path to skip the default glyph/inner-label.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveKind {
    Tick,
    Switch,
    ConstantPacket,
    ConstantSignal,
    FanOut,
    Egress,
    Aggregator,
    Filter,
}

impl PrimitiveKind {
    /// Maps a class name to its primitive viz kind, if any. Used by the
    /// canvas spawn path so it can attach [`PrimitiveKind`] alongside the
    /// generic node entity.
    pub fn from_class(class: &str) -> Option<Self> {
        Some(match class {
            "Tick" => PrimitiveKind::Tick,
            "Switch" => PrimitiveKind::Switch,
            "ConstantPacket" => PrimitiveKind::ConstantPacket,
            "ConstantSignal" => PrimitiveKind::ConstantSignal,
            "FanOut" => PrimitiveKind::FanOut,
            "Egress" => PrimitiveKind::Egress,
            "Aggregator" => PrimitiveKind::Aggregator,
            "Filter" => PrimitiveKind::Filter,
            _ => return None,
        })
    }
}

/// Records when packets last arrived at this primitive, in sim-ns.
/// Animation systems read `visual_now - last_arrival_visual` to drive
/// fade-out pulses. Per-primitive: some watch arrivals (Aggregator),
/// some watch emits (Tick), some watch both (Switch — emits are how it
/// shows which way the blade routed).
#[derive(Component, Default)]
pub struct PrimitivePulse {
    /// Wall-clock-equivalent time (`SimClock.visual_now`) at which the
    /// most recent packet arrived. `f64::MIN` means "never seen one."
    pub last_arrival: f64,
    /// Same axis as `last_arrival` but for emits leaving this node.
    pub last_emit: f64,
    /// Per-port last-emit visual_now. The Switch uses this to flash the
    /// pass-vs-divert branch that actually fired. Empty unless the
    /// primitive needs per-port granularity.
    pub last_emit_by_port: HashMap<String, f64>,
    /// Sim-time of the most recent packet emit by this node (in ns).
    /// Used by the Tick visual to detect "a new beat just happened"
    /// even when multiple sim updates fall inside one frame.
    pub last_emit_sim_ns: u64,
}

/// Marker on the parent entity indicating its child-entity construction
/// is complete. Lets `construct_primitive_visuals` be idempotent — it
/// only builds children once per spawn.
#[derive(Component)]
pub struct PrimitiveBuilt;

// ============================================================================
// Per-primitive child-entity markers (one per moving part)
// ============================================================================

#[derive(Component)]
pub struct TickPendulum;
#[derive(Component)]
pub struct TickGlow;

/// Holds the blade's current rotation angle (radians, around Z). The
/// tween system reads + writes this and pushes the result into the
/// entity's `Transform.rotation` — avoids round-tripping through Quat
/// (whose `to_euler` is gimbal-prone) and keeps the angle deterministic
/// across headless tests where `Time::delta_secs` is 0.
#[derive(Component)]
pub struct SwitchBlade {
    pub current_angle: f32,
}
#[derive(Component)]
pub struct SwitchPassRail;
#[derive(Component)]
pub struct SwitchDivertRail;

#[derive(Component)]
pub struct PressHead;
#[derive(Component)]
pub struct PressFace; // the body of the press, carries the engraved value

#[derive(Component)]
pub struct StrikerArm;
#[derive(Component)]
pub struct StrikerContact;

#[derive(Component)]
pub struct FanoutSpoke {
    /// 0..n_spokes, used to evenly distribute rotation.
    pub idx: u32,
}
#[derive(Component)]
pub struct FanoutHub;

#[derive(Component)]
pub struct EgressWall;
#[derive(Component)]
pub struct EgressPort;
#[derive(Component)]
pub struct EgressShimmer;

#[derive(Component)]
pub struct AggregatorBead {
    /// 0..8 — which neighbour slot this bead represents. Laid out as a
    /// 3x3 minus the center cell, so idx ∈ {0,1,2, 3, 4, 5,6,7}:
    /// 0 1 2
    /// 3 . 4
    /// 5 6 7
    pub slot: u8,
}
#[derive(Component)]
pub struct AggregatorReadout; // the count text in the center

#[derive(Component)]
pub struct FilterNotch;
#[derive(Component)]
pub struct FilterPassChute;
#[derive(Component)]
pub struct FilterRejectChute;

// ============================================================================
// Event ingest — packet arrivals/emits → PrimitivePulse
// ============================================================================

/// Mirrors `NewEvents` into the per-entity `PrimitivePulse`. Runs every
/// frame; for each `PacketDelivered { to }` or `PacketEmitted { from }`
/// event matching a primitive entity, updates that entity's pulse to
/// the current `visual_now`.
fn ingest_packet_events(
    new_events: Res<NewEvents>,
    clock: Res<SimClock>,
    maps: Res<crate::bridge::EntityMaps>,
    mut q: Query<&mut PrimitivePulse>,
) {
    if new_events.0.is_empty() {
        return;
    }
    let now = clock.visual_now;
    for ev in &new_events.0 {
        match ev {
            Event::PacketDelivered { to, .. } => {
                if let Some(&ent) = maps.node_to_entity.get(to) {
                    if let Ok(mut pulse) = q.get_mut(ent) {
                        pulse.last_arrival = now;
                    }
                }
            }
            Event::PacketEmitted { from, at_ns, .. } => {
                if let Some(&ent) = maps.node_to_entity.get(from) {
                    if let Ok(mut pulse) = q.get_mut(ent) {
                        pulse.last_emit = now;
                        pulse.last_emit_sim_ns = *at_ns;
                    }
                }
            }
            Event::RuleFired { node, rule, .. } => {
                // Map rule names to logical ports for per-port pulses.
                // Switch: "pass" / "divert"; Filter: "pass_*" / "reject_*".
                if let Some(&ent) = maps.node_to_entity.get(node) {
                    if let Ok(mut pulse) = q.get_mut(ent) {
                        let port = if rule.starts_with("pass") {
                            Some("pass")
                        } else if rule.starts_with("divert") {
                            Some("divert")
                        } else if rule.starts_with("reject") {
                            Some("reject")
                        } else {
                            None
                        };
                        if let Some(p) = port {
                            pulse.last_emit_by_port.insert(p.to_string(), now);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

// ============================================================================
// Child-entity construction
// ============================================================================

/// For each newly-tagged primitive entity, build its mechanical child
/// entities. Idempotent via the [`PrimitiveBuilt`] tag.
#[allow(clippy::too_many_arguments)]
fn construct_primitive_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    snapshot: Res<SimSnapshotRes>,
    q: Query<(Entity, &PrimitiveKind, &FlowNodeRef), Without<PrimitiveBuilt>>,
) {
    for (entity, kind, node_ref) in q.iter() {
        commands.entity(entity).insert(PrimitiveBuilt);
        match *kind {
            PrimitiveKind::Tick => build_tick(&mut commands, entity, &mut meshes, &mut materials),
            PrimitiveKind::Switch => build_switch(&mut commands, entity, &mut meshes, &mut materials),
            PrimitiveKind::ConstantPacket => {
                let v = read_slot_i64(&snapshot, node_ref.0, "value").unwrap_or(0);
                build_constant_packet(&mut commands, entity, &mut meshes, &mut materials, v);
            }
            PrimitiveKind::ConstantSignal => {
                let v = read_slot_i64(&snapshot, node_ref.0, "value").unwrap_or(0);
                build_constant_signal(&mut commands, entity, &mut meshes, &mut materials, v);
            }
            PrimitiveKind::FanOut => build_fanout(&mut commands, entity, &mut meshes, &mut materials),
            PrimitiveKind::Egress => build_egress(&mut commands, entity, &mut meshes, &mut materials),
            PrimitiveKind::Aggregator => build_aggregator(&mut commands, entity, &mut meshes, &mut materials),
            PrimitiveKind::Filter => {
                let m = read_slot_i64(&snapshot, node_ref.0, "match").unwrap_or(0);
                build_filter(&mut commands, entity, &mut meshes, &mut materials, m);
            }
        }
    }
}

fn read_slot_i64(snapshot: &SimSnapshotRes, node: NodeId, slot: &str) -> Option<i64> {
    snapshot
        .0
        .nodes
        .get(&node)
        .and_then(|n| n.slots.get(slot))
        .and_then(|v| match v {
            flow::Value::Int(i) => Some(*i),
            flow::Value::Bool(b) => Some(*b as i64),
            _ => None,
        })
}

// ============================================================================
// Builders — geometry stubs (filled in below per primitive)
// ============================================================================

fn rect_mesh(w: f32, h: f32, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
    meshes.add(Rectangle::new(w, h))
}
fn circle_mesh(r: f32, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
    meshes.add(Circle::new(r))
}
fn mat(c: Color, materials: &mut Assets<ColorMaterial>) -> Handle<ColorMaterial> {
    materials.add(ColorMaterial::from(c))
}

const INK: Color = Color::srgb(0.10, 0.10, 0.10);
const BRASS: Color = Color::srgb(0.78, 0.62, 0.30);
const COPPER: Color = Color::srgb(0.78, 0.45, 0.25);
const WARM: Color = Color::srgb(0.86, 0.52, 0.28);
const COOL: Color = Color::srgb(0.30, 0.45, 0.65);
const STEEL: Color = Color::srgb(0.55, 0.58, 0.62);
const RAIL_DARK: Color = Color::srgb(0.30, 0.32, 0.34);
const RAIL_BRIGHT: Color = Color::srgb(0.95, 0.78, 0.30);

// ---- Tick (pendulum) ----

fn build_tick(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    // The parent already renders a Circle (radius 44 from stock visual).
    // We add a glow disc behind, and a pendulum arm + bob on top.
    commands.entity(parent).with_children(|p| {
        // Breathing glow (z behind body).
        p.spawn((
            Mesh2d(circle_mesh(54.0, meshes)),
            MeshMaterial2d(mat(Color::srgba(0.95, 0.78, 0.30, 0.25), materials)),
            Transform::from_xyz(0.0, 0.0, -0.05),
            TickGlow,
        ));
        // Pivot dot at center.
        p.spawn((
            Mesh2d(circle_mesh(4.0, meshes)),
            MeshMaterial2d(mat(INK, materials)),
            Transform::from_xyz(0.0, 0.0, 0.3),
        ));
        // Pendulum arm + bob. The arm pivots at (0,0); the bob hangs
        // 28px down its length. We rotate the whole subgroup.
        let arm = p
            .spawn((
                Transform::from_xyz(0.0, 0.0, 0.25),
                Visibility::Inherited,
                TickPendulum,
            ))
            .id();
        p.commands().entity(arm).with_children(|cp| {
            // Arm rod.
            cp.spawn((
                Mesh2d(rect_mesh(3.0, 30.0, meshes)),
                MeshMaterial2d(mat(INK, materials)),
                Transform::from_xyz(0.0, -15.0, 0.0),
            ));
            // Bob.
            cp.spawn((
                Mesh2d(circle_mesh(6.5, meshes)),
                MeshMaterial2d(mat(BRASS, materials)),
                Transform::from_xyz(0.0, -30.0, 0.05),
            ));
        });
    });
}

// ---- Switch (railway points) ----

fn build_switch(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    // Parent is a Diamond at size 96 (radius 48). We draw two outgoing
    // rails (pass = straight right, divert = down-right) plus a rotating
    // blade that physically swings between them.
    commands.entity(parent).with_children(|p| {
        // Inbound rail (left).
        p.spawn((
            Mesh2d(rect_mesh(40.0, 4.0, meshes)),
            MeshMaterial2d(mat(RAIL_DARK, materials)),
            Transform::from_xyz(-22.0, 0.0, 0.15),
        ));
        // Pass rail — straight through to the right.
        p.spawn((
            Mesh2d(rect_mesh(40.0, 4.0, meshes)),
            MeshMaterial2d(mat(RAIL_DARK, materials)),
            Transform::from_xyz(22.0, 0.0, 0.15),
            SwitchPassRail,
        ));
        // Divert rail — angled downward. Rotated -30°; offset placed at
        // the end of the rotation so it visibly leaves the body.
        let divert_t = Transform {
            translation: Vec3::new(20.0, -10.0, 0.15),
            rotation: Quat::from_rotation_z(-0.52),
            scale: Vec3::ONE,
        };
        p.spawn((
            Mesh2d(rect_mesh(40.0, 4.0, meshes)),
            MeshMaterial2d(mat(RAIL_DARK, materials)),
            divert_t,
            SwitchDivertRail,
        ));
        // Pivot.
        p.spawn((
            Mesh2d(circle_mesh(3.5, meshes)),
            MeshMaterial2d(mat(INK, materials)),
            Transform::from_xyz(0.0, 0.0, 0.3),
        ));
        // Blade — pivots at the input point (0,0). Animated to point
        // toward pass-rail (0°) when passing=1 or divert (-30°) when 0.
        p.spawn((
            Mesh2d(rect_mesh(34.0, 6.0, meshes)),
            MeshMaterial2d(mat(RAIL_BRIGHT, materials)),
            Transform::from_xyz(17.0, 0.0, 0.25),
            SwitchBlade { current_angle: 0.0 },
        ));
    });
}

// ---- ConstantPacket (stamping press) ----

fn build_constant_packet(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    value: i64,
) {
    let warm = value != 0;
    let face_color = if warm { WARM } else { COOL };
    commands.entity(parent).with_children(|p| {
        // Anvil base (small darker rect).
        p.spawn((
            Mesh2d(rect_mesh(54.0, 8.0, meshes)),
            MeshMaterial2d(mat(STEEL, materials)),
            Transform::from_xyz(0.0, -16.0, 0.10),
        ));
        // Press face — circular plate with the value engraved.
        p.spawn((
            Mesh2d(circle_mesh(14.0, meshes)),
            MeshMaterial2d(mat(face_color, materials)),
            Transform::from_xyz(0.0, 4.0, 0.20),
            PressFace,
        ));
        // Press head (moves down on arrival).
        p.spawn((
            Mesh2d(rect_mesh(26.0, 16.0, meshes)),
            MeshMaterial2d(mat(STEEL, materials)),
            Transform::from_xyz(0.0, 22.0, 0.25),
            PressHead,
        ));
    });
}

// ---- ConstantSignal (telegraph striker) ----

fn build_constant_signal(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    value: i64,
) {
    let warm = value != 0;
    let head_color = if warm { Color::srgb(0.78, 0.20, 0.18) } else { COOL };
    commands.entity(parent).with_children(|p| {
        // Base contact plate.
        p.spawn((
            Mesh2d(rect_mesh(40.0, 6.0, meshes)),
            MeshMaterial2d(mat(STEEL, materials)),
            Transform::from_xyz(0.0, -10.0, 0.10),
            StrikerContact,
        ));
        // Striker arm — pivots near the right edge, hammer head on left.
        let arm = p
            .spawn((
                Transform::from_xyz(14.0, 0.0, 0.20),
                Visibility::Inherited,
                StrikerArm,
            ))
            .id();
        p.commands().entity(arm).with_children(|cp| {
            cp.spawn((
                Mesh2d(rect_mesh(34.0, 4.0, meshes)),
                MeshMaterial2d(mat(INK, materials)),
                Transform::from_xyz(-17.0, 0.0, 0.0),
            ));
            cp.spawn((
                Mesh2d(rect_mesh(10.0, 14.0, meshes)),
                MeshMaterial2d(mat(head_color, materials)),
                Transform::from_xyz(-30.0, 0.0, 0.05),
            ));
        });
    });
}

// ---- FanOut (sunburst manifold) ----

fn build_fanout(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    // Parent body is Circle size 72 → radius 36. We draw 8 spokes
    // radiating outward and a central hub.
    const N_SPOKES: u32 = 8;
    commands.entity(parent).with_children(|p| {
        for i in 0..N_SPOKES {
            let theta = (i as f32) * std::f32::consts::TAU / (N_SPOKES as f32);
            let t = Transform {
                translation: Vec3::new(20.0 * theta.cos(), 20.0 * theta.sin(), 0.12),
                rotation: Quat::from_rotation_z(theta),
                scale: Vec3::ONE,
            };
            p.spawn((
                Mesh2d(rect_mesh(28.0, 4.0, meshes)),
                MeshMaterial2d(mat(STEEL, materials)),
                t,
                FanoutSpoke { idx: i },
            ));
        }
        // Hub.
        p.spawn((
            Mesh2d(circle_mesh(12.0, meshes)),
            MeshMaterial2d(mat(COPPER, materials)),
            Transform::from_xyz(0.0, 0.0, 0.20),
            FanoutHub,
        ));
        p.spawn((
            Mesh2d(circle_mesh(4.0, meshes)),
            MeshMaterial2d(mat(INK, materials)),
            Transform::from_xyz(0.0, 0.0, 0.25),
        ));
    });
}

// ---- Egress (bulkhead) ----

fn build_egress(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    // Parent is Square size 64. We render a vertical "wall" strip down
    // its left edge with a port on the right side.
    commands.entity(parent).with_children(|p| {
        // Wall (a vertical bar through the body, slightly off-center).
        p.spawn((
            Mesh2d(rect_mesh(8.0, 56.0, meshes)),
            MeshMaterial2d(mat(STEEL, materials)),
            Transform::from_xyz(-6.0, 0.0, 0.10),
            EgressWall,
        ));
        // Port flange — circle on the right side.
        p.spawn((
            Mesh2d(circle_mesh(14.0, meshes)),
            MeshMaterial2d(mat(COPPER, materials)),
            Transform::from_xyz(8.0, 0.0, 0.15),
            EgressPort,
        ));
        p.spawn((
            Mesh2d(circle_mesh(7.0, meshes)),
            MeshMaterial2d(mat(INK, materials)),
            Transform::from_xyz(8.0, 0.0, 0.20),
        ));
        // Shimmer overlay (alpha-only, no transform changes per-frame —
        // animation tweaks its material color).
        p.spawn((
            Mesh2d(circle_mesh(18.0, meshes)),
            MeshMaterial2d(mat(Color::srgba(0.95, 0.85, 0.45, 0.0), materials)),
            Transform::from_xyz(8.0, 0.0, 0.22),
            EgressShimmer,
        ));
    });
}

// ---- Aggregator (3×3-minus-center abacus) ----

fn build_aggregator(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    // Parent is Pentagon size 100 → radius 50. We lay an 8-bead layout
    // (3×3 minus center) over the upper portion of the body.
    const SPACING: f32 = 16.0;
    let positions: [(f32, f32); 8] = [
        (-SPACING, SPACING),  // 0
        (0.0, SPACING),       // 1
        (SPACING, SPACING),   // 2
        (-SPACING, 0.0),      // 3
        (SPACING, 0.0),       // 4
        (-SPACING, -SPACING), // 5
        (0.0, -SPACING),      // 6
        (SPACING, -SPACING),  // 7
    ];
    commands.entity(parent).with_children(|p| {
        for (i, (x, y)) in positions.iter().enumerate() {
            // Slot well (drawn dim — bead lights up).
            p.spawn((
                Mesh2d(circle_mesh(5.0, meshes)),
                MeshMaterial2d(mat(Color::srgb(0.40, 0.36, 0.42), materials)),
                Transform::from_xyz(*x, *y, 0.15),
                AggregatorBead { slot: i as u8 },
            ));
        }
        // Center readout pad (where the running sum will live — text
        // not implemented in this pass; the bead count is the readout).
        p.spawn((
            Mesh2d(circle_mesh(7.0, meshes)),
            MeshMaterial2d(mat(INK, materials)),
            Transform::from_xyz(0.0, 0.0, 0.10),
            AggregatorReadout,
        ));
    });
}

// ---- Filter (coin sorter) ----

fn build_filter(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    match_val: i64,
) {
    // Parent is a downward-pointing Triangle size 96. We add:
    // - A template plate near the top with a notch sized to match_val
    // - A pass chute below the notch
    // - A reject chute angled off to the side
    commands.entity(parent).with_children(|p| {
        // Notch plate — width scaled by match value so the shape is
        // readable as the configured sum. (Clamped so it always fits.)
        let notch_w = (match_val.clamp(1, 9) as f32) * 3.0 + 6.0;
        // Left half of plate.
        p.spawn((
            Mesh2d(rect_mesh(20.0, 6.0, meshes)),
            MeshMaterial2d(mat(STEEL, materials)),
            Transform::from_xyz(-14.0 - notch_w * 0.5, 8.0, 0.18),
        ));
        // Right half of plate.
        p.spawn((
            Mesh2d(rect_mesh(20.0, 6.0, meshes)),
            MeshMaterial2d(mat(STEEL, materials)),
            Transform::from_xyz(14.0 + notch_w * 0.5, 8.0, 0.18),
        ));
        // Notch indicator (drawn as a recessed darker rect below the gap).
        p.spawn((
            Mesh2d(rect_mesh(notch_w, 3.0, meshes)),
            MeshMaterial2d(mat(Color::srgb(0.25, 0.20, 0.15), materials)),
            Transform::from_xyz(0.0, 4.5, 0.20),
            FilterNotch,
        ));
        // Pass chute (straight down through the body).
        p.spawn((
            Mesh2d(rect_mesh(notch_w * 0.7, 18.0, meshes)),
            MeshMaterial2d(mat(Color::srgba(0.90, 0.78, 0.45, 0.45), materials)),
            Transform::from_xyz(0.0, -8.0, 0.12),
            FilterPassChute,
        ));
        // Reject chute (angled off to the right).
        let reject_t = Transform {
            translation: Vec3::new(16.0, -4.0, 0.12),
            rotation: Quat::from_rotation_z(-0.7),
            scale: Vec3::ONE,
        };
        p.spawn((
            Mesh2d(rect_mesh(22.0, 4.0, meshes)),
            MeshMaterial2d(mat(Color::srgba(0.78, 0.40, 0.30, 0.45), materials)),
            reject_t,
            FilterRejectChute,
        ));
    });
    let _ = parent; // silence if unused
}

// ============================================================================
// Animations
// ============================================================================

/// Tick: pendulum swings sinusoidally with period = sim's `period_ns`
/// slot. Glow oscillates at the same rate. On each beat (detected via
/// `last_emit_sim_ns` jump) we briefly brighten the glow.
fn animate_tick(
    snapshot: Res<SimSnapshotRes>,
    clock: Res<SimClock>,
    parents: Query<(&FlowNodeRef, &PrimitivePulse, &Children, &PrimitiveKind)>,
    mut pendulums: Query<&mut Transform, (With<TickPendulum>, Without<TickGlow>)>,
    mut glows: Query<&mut MeshMaterial2d<ColorMaterial>, With<TickGlow>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (node_ref, pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::Tick {
            continue;
        }
        let period_ns = read_slot_i64(&snapshot, node_ref.0, "period_ns").unwrap_or(200_000_000) as f64;
        let period_s = (period_ns * 1e-9).max(0.05);
        let omega = std::f32::consts::TAU / period_s as f32;
        let t = clock.visual_now as f32;
        // Pendulum: sinusoidal swing ±0.6 rad.
        let angle = 0.6 * (omega * t).sin();
        // Recent-beat brightening: 250ms decay.
        let since_beat = (clock.visual_now - pulse.last_emit).max(0.0) as f32;
        let beat_intensity = (1.0 - (since_beat / 0.25)).clamp(0.0, 1.0);
        for child in children.iter() {
            if let Ok(mut tf) = pendulums.get_mut(child) {
                tf.rotation = Quat::from_rotation_z(angle);
            }
            if let Ok(mat_handle) = glows.get_mut(child) {
                if let Some(m) = materials.get_mut(&mat_handle.0) {
                    let base = 0.18;
                    let pulse_alpha = base + 0.55 * beat_intensity;
                    m.color = Color::srgba(0.95, 0.78, 0.30, pulse_alpha);
                }
            }
        }
    }
}

/// Switch: blade rotates between 0 (pass = straight) and -0.52 rad
/// (divert = downward-angled). Tween toward target each frame. The
/// rails brighten on the side that just emitted.
fn animate_switch(
    snapshot: Res<SimSnapshotRes>,
    clock: Res<SimClock>,
    parents: Query<(&FlowNodeRef, &PrimitivePulse, &Children, &PrimitiveKind)>,
    mut blades: Query<(&mut Transform, &mut SwitchBlade)>,
    pass_rails: Query<&MeshMaterial2d<ColorMaterial>, (With<SwitchPassRail>, Without<SwitchDivertRail>)>,
    divert_rails: Query<&MeshMaterial2d<ColorMaterial>, (With<SwitchDivertRail>, Without<SwitchPassRail>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut last_visual_now: Local<f64>,
) {
    // Derive dt from visual_now diff. Robust to headless tests where
    // Time::delta_secs is 0; falls back to a sensible 16ms minimum so
    // a single-frame test step still produces visible motion.
    let now = clock.visual_now;
    let mut dt = (now - *last_visual_now) as f32;
    *last_visual_now = now;
    if dt <= 0.0 {
        dt = 0.016;
    }
    for (node_ref, pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::Switch {
            continue;
        }
        let passing = read_slot_i64(&snapshot, node_ref.0, "passing").unwrap_or(1);
        let target_angle = if passing != 0 { 0.0_f32 } else { -0.52_f32 };
        // Recent-emit pulse (250ms).
        let since_pass = (clock.visual_now - pulse.last_emit_by_port.get("pass").copied().unwrap_or(f64::MIN)).max(0.0) as f32;
        let since_divert = (clock.visual_now - pulse.last_emit_by_port.get("divert").copied().unwrap_or(f64::MIN)).max(0.0) as f32;
        let pass_brt = (1.0 - (since_pass / 0.25)).clamp(0.0, 1.0);
        let divert_brt = (1.0 - (since_divert / 0.25)).clamp(0.0, 1.0);
        // Whichever was most recent — approximate by using emit pulse on
        // the active rail when we don't have per-port data.
        let active_pulse = if passing != 0 { pass_brt.max((1.0 - (clock.visual_now - pulse.last_emit).max(0.0) as f32 / 0.25).clamp(0.0, 1.0)) } else { pass_brt };
        let divert_active = if passing == 0 { divert_brt.max((1.0 - (clock.visual_now - pulse.last_emit).max(0.0) as f32 / 0.25).clamp(0.0, 1.0)) } else { divert_brt };
        for child in children.iter() {
            if let Ok((mut tf, mut blade)) = blades.get_mut(child) {
                // Tween at 12/s toward target. Exponential approach so
                // a single big dt still lands close to target rather
                // than overshooting.
                let k = (dt * 12.0).min(1.0);
                blade.current_angle += (target_angle - blade.current_angle) * k;
                tf.rotation = Quat::from_rotation_z(blade.current_angle);
            }
            if let Ok(mat_handle) = pass_rails.get(child) {
                if let Some(m) = materials.get_mut(&mat_handle.0) {
                    let base = if passing != 0 { 0.45 } else { 0.18 };
                    let r = base + 0.45 * active_pulse;
                    let g = base * 0.85 + 0.40 * active_pulse;
                    let b = 0.15 + 0.05 * active_pulse;
                    m.color = Color::srgb(r.min(0.95), g.min(0.85), b.min(0.30));
                }
            }
            if let Ok(mat_handle) = divert_rails.get(child) {
                if let Some(m) = materials.get_mut(&mat_handle.0) {
                    let base = if passing == 0 { 0.45 } else { 0.18 };
                    let r = base + 0.45 * divert_active;
                    let g = base * 0.85 + 0.40 * divert_active;
                    let b = 0.15 + 0.05 * divert_active;
                    m.color = Color::srgb(r.min(0.95), g.min(0.85), b.min(0.30));
                }
            }
        }
    }
}

/// ConstantPacket: press head slams down on arrival, springs back up.
fn animate_constant_packet(
    clock: Res<SimClock>,
    parents: Query<(&PrimitivePulse, &Children, &PrimitiveKind)>,
    mut heads: Query<&mut Transform, With<PressHead>>,
) {
    for (pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::ConstantPacket {
            continue;
        }
        let since = (clock.visual_now - pulse.last_arrival).max(0.0) as f32;
        // Slam profile: t in [0, 0.2s]: head moves from y=22 → y=8.
        // t in [0.2, 0.5]: springs back to 22.
        let y = if since < 0.2 {
            22.0 - 14.0 * (since / 0.2)
        } else if since < 0.5 {
            8.0 + 14.0 * ((since - 0.2) / 0.3)
        } else {
            22.0
        };
        for child in children.iter() {
            if let Ok(mut tf) = heads.get_mut(child) {
                tf.translation.y = y;
            }
        }
    }
}

/// ConstantSignal: striker arm swings down on arrival, rebounds.
fn animate_constant_signal(
    clock: Res<SimClock>,
    parents: Query<(&PrimitivePulse, &Children, &PrimitiveKind)>,
    mut arms: Query<&mut Transform, With<StrikerArm>>,
) {
    for (pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::ConstantSignal {
            continue;
        }
        let since = (clock.visual_now - pulse.last_arrival).max(0.0) as f32;
        // Strike profile: hammer arcs from +0.4 rad (cocked) down to -0.2 (strike) and back.
        let angle = if since < 0.12 {
            0.4 - 0.6 * (since / 0.12)
        } else if since < 0.40 {
            -0.2 + 0.6 * ((since - 0.12) / 0.28)
        } else {
            0.4
        };
        for child in children.iter() {
            if let Ok(mut tf) = arms.get_mut(child) {
                tf.rotation = Quat::from_rotation_z(angle);
            }
        }
    }
}

/// FanOut: spokes briefly brighten on arrival, hub pulses, all 8 spokes
/// receive the broadcast together so they pulse in sync.
fn animate_fanout(
    clock: Res<SimClock>,
    parents: Query<(&PrimitivePulse, &Children, &PrimitiveKind)>,
    spokes: Query<&MeshMaterial2d<ColorMaterial>, (With<FanoutSpoke>, Without<FanoutHub>)>,
    hubs: Query<&MeshMaterial2d<ColorMaterial>, (With<FanoutHub>, Without<FanoutSpoke>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::FanOut {
            continue;
        }
        let since = (clock.visual_now - pulse.last_arrival).max(0.0) as f32;
        // 300ms decay.
        let brt = (1.0 - (since / 0.30)).clamp(0.0, 1.0);
        for child in children.iter() {
            if let Ok(h) = spokes.get(child) {
                if let Some(m) = materials.get_mut(&h.0) {
                    let base = 0.55;
                    let v = base + 0.40 * brt;
                    m.color = Color::srgb(v, v, v.min(0.85));
                }
            }
            if let Ok(h) = hubs.get(child) {
                if let Some(m) = materials.get_mut(&h.0) {
                    let r = 0.78 + 0.20 * brt;
                    let g = 0.45 + 0.30 * brt;
                    let b = 0.25 + 0.10 * brt;
                    m.color = Color::srgb(r.min(1.0), g.min(0.95), b.min(0.55));
                }
            }
        }
    }
}

/// Egress: shimmer ring fades in on packet transit.
fn animate_egress(
    clock: Res<SimClock>,
    parents: Query<(&PrimitivePulse, &Children, &PrimitiveKind)>,
    shimmers: Query<&MeshMaterial2d<ColorMaterial>, With<EgressShimmer>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::Egress {
            continue;
        }
        // Use whichever pulse is more recent (arrival or emit).
        let since = (clock.visual_now - pulse.last_arrival.max(pulse.last_emit)).max(0.0) as f32;
        let brt = (1.0 - (since / 0.45)).clamp(0.0, 1.0);
        for child in children.iter() {
            if let Ok(h) = shimmers.get(child) {
                if let Some(m) = materials.get_mut(&h.0) {
                    m.color = Color::srgba(0.95, 0.85, 0.45, 0.60 * brt);
                }
            }
        }
    }
}

/// Aggregator: beads light up to match `seen` slot. On reset (seen
/// drops back to 0) the beads all dim together.
fn animate_aggregator(
    snapshot: Res<SimSnapshotRes>,
    parents: Query<(&FlowNodeRef, &Children, &PrimitiveKind)>,
    beads: Query<(&AggregatorBead, &MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (node_ref, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::Aggregator {
            continue;
        }
        let seen = read_slot_i64(&snapshot, node_ref.0, "seen").unwrap_or(0).max(0) as u8;
        for child in children.iter() {
            if let Ok((bead, mat_handle)) = beads.get(child) {
                let lit = bead.slot < seen;
                if let Some(m) = materials.get_mut(&mat_handle.0) {
                    m.color = if lit {
                        Color::srgb(0.95, 0.78, 0.30)
                    } else {
                        Color::srgb(0.40, 0.36, 0.42)
                    };
                }
            }
        }
    }
}

/// Filter: pass chute flashes on .pass emit, reject chute flashes on
/// .reject emit. Without per-port pulse data we approximate from
/// last_emit alone, splitting based on whether the most recent arrival's
/// inbound value matched. (Stub: just pulses both faintly on activity.)
fn animate_filter(
    clock: Res<SimClock>,
    parents: Query<(&PrimitivePulse, &Children, &PrimitiveKind)>,
    pass_chutes: Query<&MeshMaterial2d<ColorMaterial>, (With<FilterPassChute>, Without<FilterRejectChute>)>,
    reject_chutes: Query<&MeshMaterial2d<ColorMaterial>, (With<FilterRejectChute>, Without<FilterPassChute>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (pulse, children, kind) in parents.iter() {
        if *kind != PrimitiveKind::Filter {
            continue;
        }
        let since_pass = (clock.visual_now - pulse.last_emit_by_port.get("pass").copied().unwrap_or(f64::MIN)).max(0.0) as f32;
        let since_reject = (clock.visual_now - pulse.last_emit_by_port.get("reject").copied().unwrap_or(f64::MIN)).max(0.0) as f32;
        let pass_brt = (1.0 - since_pass / 0.30).clamp(0.0, 1.0);
        let reject_brt = (1.0 - since_reject / 0.30).clamp(0.0, 1.0);
        for child in children.iter() {
            if let Ok(h) = pass_chutes.get(child) {
                if let Some(m) = materials.get_mut(&h.0) {
                    m.color = Color::srgba(0.90, 0.78, 0.45, 0.30 + 0.55 * pass_brt);
                }
            }
            if let Ok(h) = reject_chutes.get(child) {
                if let Some(m) = materials.get_mut(&h.0) {
                    m.color = Color::srgba(0.78, 0.40, 0.30, 0.30 + 0.55 * reject_brt);
                }
            }
        }
    }
}
