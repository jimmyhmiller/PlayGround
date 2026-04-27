//! GPU-instanced packet renderer.
//!
//! All visible packets render in a single draw call backed by a custom
//! `Material2d`. The CPU side just rewrites a storage buffer of
//! `PacketInstance` records each frame and bumps `clock.now_real`; the
//! vertex shader does the lerp from→to and rasterizes a small soft
//! circle per packet (see `shaders/packet_cloud.wgsl`).
//!
//! Mesh structure: a fixed-size index-only mesh covering up to
//! `MAX_PACKETS` quads. Each vertex's `vertex_index` is divided by 6
//! to yield an instance id; out-of-range slots collapse off-clip in
//! the shader so we never need to resize the mesh as the live count
//! changes.
//!
//! `TravelingPacket` entities are still spawned (kept lightweight —
//! no mesh/material) so existing tests can keep asserting on the
//! per-packet view; this module is purely the rendering path.

use bevy::asset::embedded_asset;
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use bevy::render::batching::NoAutomaticBatching;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

use crate::bridge::{EntityMaps, FlowNodeRef, SimClock};
use crate::edges::VisualTimelineRes;
use crate::theme::Theme;
use crate::tool::NodeColors;

/// Hard cap on simultaneously-rendered packets. The mesh is sized once
/// to this; storage buffer is allocated once to this. Going past it
/// just clips the tail of the visible set.
const MAX_PACKETS: usize = 100_000;

/// Half-edge of the packet quad in world units. Final visible dot has
/// roughly this radius before the shader's circular alpha mask kicks
/// in. Tuned to match the previous `Circle::new(6.0)` look.
const PACKET_RADIUS: f32 = 6.0;

/// Per-packet payload sent to the GPU. Layout matches
/// `PacketInstance` in `shaders/packet_cloud.wgsl`. Field names match
/// the WGSL — `from`/`to` are reserved in WGSL so we use
/// `start`/`end`.
#[derive(Clone, Copy, ShaderType, Default)]
struct PacketInstance {
    start: Vec2,
    end: Vec2,
    emit_real: f32,
    arrive_real: f32,
    color: Vec4,
}

/// Frame-shared uniform: the wall clock, the count of populated
/// instance slots, and the quad radius (so we can change it at
/// runtime without recompiling the shader).
#[derive(Clone, Copy, ShaderType, Default)]
struct PacketClock {
    now_real: f32,
    active_count: u32,
    quad_radius: f32,
    _pad: f32,
}

/// Wraps the per-frame instance array in a Bevy ShaderType so we can
/// `set_data` it onto the storage-buffer asset every frame. The
/// `#[shader(size(runtime))]` attribute is required for the trailing
/// runtime-sized array layout that storage buffers use in WGSL.
#[derive(Clone, ShaderType, Default)]
struct PacketInstances {
    #[shader(size(runtime))]
    items: Vec<PacketInstance>,
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct PacketCloudMaterial {
    #[storage(0, read_only)]
    instances: Handle<ShaderStorageBuffer>,
    #[uniform(1)]
    clock: PacketClock,
}

impl Material2d for PacketCloudMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://flow_bevy/shaders/packet_cloud.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://flow_bevy/shaders/packet_cloud.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

#[derive(Component)]
pub struct PacketCloud;

pub struct PacketCloudPlugin;

impl Plugin for PacketCloudPlugin {
    fn build(&self, app: &mut App) {
        // Bake the WGSL into the binary so there's no runtime asset
        // discovery — Bevy resolves `embedded://flow_bevy/...` via
        // the macro registration below.
        embedded_asset!(app, "shaders/packet_cloud.wgsl");

        app.add_plugins(Material2dPlugin::<PacketCloudMaterial>::default())
            .add_systems(Startup, spawn_packet_cloud)
            .add_systems(Update, update_packet_cloud);
    }
}

fn spawn_packet_cloud(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<PacketCloudMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let mesh = build_index_only_mesh(MAX_PACKETS);
    // Allocate the storage buffer pre-sized for the max packet count so
    // the GPU buffer is created once at startup and we just memcpy
    // into it each frame via `set_data`.
    let mut storage = ShaderStorageBuffer::default();
    storage.asset_usage = RenderAssetUsages::default();
    storage.set_data(PacketInstances {
        items: vec![PacketInstance::default(); MAX_PACKETS],
    });
    let mat = PacketCloudMaterial {
        instances: buffers.add(storage),
        clock: PacketClock {
            now_real: 0.0,
            active_count: 0,
            quad_radius: PACKET_RADIUS,
            _pad: 0.0,
        },
    };
    // Z=3 to match the previous per-packet z so packets render above
    // edges/nodes the same way.
    //
    // `NoFrustumCulling`: the dummy mesh has all vertex positions at
    // origin → AABB is a single point. Real packet positions come
    // from the storage buffer at draw time, which the culler doesn't
    // know about. Without this the cloud disappears whenever the
    // camera is anywhere off origin.
    // `NoAutomaticBatching`: Bevy 0.18 batches Mesh2d entities into a
    // shared vertex buffer with base-vertex offsets. That breaks our
    // shader's `vid / 6` math because `@builtin(vertex_index)` then
    // counts from the global merged buffer, not 0..N*6 of our quad
    // mesh. Opting out keeps each packet-cloud draw as its own draw
    // call with vertex_index starting at 0.
    commands.spawn((
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(mats.add(mat)),
        Transform::from_xyz(0.0, 0.0, 3.0),
        Visibility::Visible,
        NoFrustumCulling,
        NoAutomaticBatching,
        PacketCloud,
    ));
}

/// Build the dummy mesh: `MAX_PACKETS * 6` vertices, no index buffer.
///
/// Why no indices: under `draw_indexed`, `@builtin(vertex_index)` in
/// the vertex shader returns the *indexed value*, not the linear step
/// number. We rely on `vertex_index` walking 0..N*6 linearly so the
/// shader can derive `instance_id = vid / 6` and `corner_id = vid % 6`.
/// A non-indexed `draw` call gives us that.
///
/// The position attribute is required by the 2D mesh pipeline even
/// though the shader recomputes everything from `vertex_index`; we
/// hand it zeros.
fn build_index_only_mesh(max_packets: usize) -> Mesh {
    let n_verts = max_packets * 6;
    let positions: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; n_verts];

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh
}

/// Each frame: walk the visual timeline, pack visible packets into
/// the storage buffer, advance the clock uniform.
fn update_packet_cloud(
    timeline: Res<VisualTimelineRes>,
    clock: Res<SimClock>,
    theme: Res<Theme>,
    node_colors: Res<NodeColors>,
    nodes: Query<&Transform, With<FlowNodeRef>>,
    maps: Res<EntityMaps>,
    cloud: Query<&MeshMaterial2d<PacketCloudMaterial>, With<PacketCloud>>,
    mut mats: ResMut<Assets<PacketCloudMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    hide_all: Res<crate::edges::HideAll>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    let Ok(handle) = cloud.single() else { return };
    let Some(mat) = mats.get_mut(&handle.0) else { return };

    crate::time_phase!(perf, "packet_cloud.update", {
    let now = clock.visual_now;
    let mut packed: Vec<PacketInstance> =
        Vec::with_capacity(timeline.packets.len().min(MAX_PACKETS));

    // Hide-all toggle (`H` key) zeroes out the active count so the
    // shader rasterizes nothing, mirroring the gating that `edges.rs`
    // applies to arrow drawing and per-packet entity spawning.
    if !hide_all.0 {
    for vp in &timeline.packets {
        if packed.len() >= MAX_PACKETS {
            break;
        }
        if now < vp.emit_real || now >= vp.arrive_real {
            continue;
        }
        let Some(&from_e) = maps.node_to_entity.get(&vp.from) else { continue };
        let Some(&to_e) = maps.node_to_entity.get(&vp.to) else { continue };
        let Ok(from_t) = nodes.get(from_e) else { continue };
        let Ok(to_t) = nodes.get(to_e) else { continue };

        let color = crate::edges::packet_color(
            &vp.payload,
            node_colors.0.get(&vp.from).copied(),
            &theme.data,
            theme.accent,
        );
        let lin = color.to_linear();

        packed.push(PacketInstance {
            start: from_t.translation.truncate(),
            end: to_t.translation.truncate(),
            emit_real: vp.emit_real as f32,
            arrive_real: vp.arrive_real as f32,
            color: Vec4::new(lin.red, lin.green, lin.blue, lin.alpha),
        });
    }
    }

    let count = packed.len();

    if let Some(storage) = buffers.get_mut(&mat.instances) {
        storage.set_data(PacketInstances { items: packed });
    }

    mat.clock = PacketClock {
        now_real: now as f32,
        active_count: count as u32,
        quad_radius: PACKET_RADIUS,
        _pad: 0.0,
    };
    });
}

#[cfg(test)]
mod tests {
    //! End-to-end tests for the packet-cloud renderer's data plumbing.
    //!
    //! These run headless (no GPU backend) so they can't verify pixels,
    //! but they DO verify that the systems wire up correctly: the
    //! `PacketCloud` entity gets spawned, `update_packet_cloud` runs,
    //! the storage-buffer asset is updated with the right data, and
    //! the clock uniform's `active_count` matches the number of
    //! visible packets in the timeline.
    use super::*;
    use flow_bevy_internal_test_helpers::*;

    #[test]
    fn packet_cloud_active_count_grows_with_traffic() {
        // Load a real example via the full seed path — that registers
        // Bevy entities for every node, so `EntityMaps.node_to_entity`
        // is populated and the filter in `update_packet_cloud` finds
        // both endpoints.
        let mut app = make_app();
        // The default test `k=1.0` makes 1ms edges animate in 1ms
        // wall-clock — so narrow that no realistic check time is
        // mid-flight. Use the production default `k=200` so each
        // 1ms edge gives us a 200ms visible window.
        {
            let mut tl = app
                .world_mut()
                .resource_mut::<crate::edges::VisualTimelineRes>();
            tl.0.k = 200.0;
        }
        app.world_mut()
            .resource_mut::<bevy::ecs::message::Messages<crate::examples::LoadExample>>()
            .write(crate::examples::LoadExample(crate::examples::Example::ClientWorker));
        app.update();
        app.update();

        // Drive a bunch of sim activity and let the bridge ingest it.
        for _ in 0..5 {
            advance_sim_ns(&mut app, 100_000_000);
            app.update();
        }

        let timeline_len = app
            .world()
            .resource::<crate::edges::VisualTimelineRes>()
            .0
            .packets
            .len();
        assert!(
            timeline_len > 0,
            "timeline should have ingested at least one packet after sim activity"
        );

        // Pull the cloud's material handle out via a mutable query.
        let cloud_handle: Handle<PacketCloudMaterial> = {
            let world = app.world_mut();
            let mut q = world
                .query_filtered::<&MeshMaterial2d<PacketCloudMaterial>, With<PacketCloud>>();
            q.iter(world)
                .next()
                .expect("PacketCloud entity missing")
                .0
                .clone()
        };
        let mat = app
            .world()
            .resource::<Assets<PacketCloudMaterial>>()
            .get(&cloud_handle)
            .expect("material missing");
        assert!(
            mat.clock.active_count > 0,
            "active_count should be > 0 with packets in flight, got {}",
            mat.clock.active_count
        );

        // And the storage-buffer data should have been written.
        let storage = app
            .world()
            .resource::<Assets<bevy::render::storage::ShaderStorageBuffer>>()
            .get(&mat.instances)
            .expect("storage buffer missing");
        assert!(
            storage.data.as_ref().is_some_and(|d| !d.is_empty()),
            "storage buffer should hold serialized instance data"
        );
    }
}

/// Helpers exposed only to the in-crate tests above. Mirrors the
/// integration-test helpers in `tests/common/mod.rs` but lives here so
/// `cfg(test)` unit tests can see crate-private items.
#[cfg(test)]
mod flow_bevy_internal_test_helpers {
    use bevy::prelude::*;

    pub fn make_app() -> App {
        let mut app = poster_ui::testing::test_app_headless();
        app.add_plugins(crate::FlowBevyPlugins);
        app.world_mut()
            .resource_mut::<crate::bridge::SimClock>()
            .multiplier = 1.0;
        {
            let mut tl = app
                .world_mut()
                .resource_mut::<crate::edges::VisualTimelineRes>();
            tl.0.k = 1.0;
        }
        // Tests must drive the sim deterministically — swap the
        // worker-mode driver the bridge installs for a Direct one.
        {
            let world = app.world_mut();
            let mut sim = flow::Sim::new(1);
            crate::gadgets::install_default_params(&mut sim);
            let (driver, events_rx) = crate::sim_driver::SimDriver::direct(sim, 1.0);
            world.insert_resource(crate::sim_driver::SimDriverRes(driver));
            world.insert_resource(crate::sim_driver::SimEventRx(
                std::sync::Mutex::new(events_rx),
            ));
        }
        app.update();
        app.update();
        app
    }

    pub fn advance_sim_ns(app: &mut App, duration_ns: u64) {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<crate::sim_driver::SimDriverRes>();
        driver.0.advance_direct(duration_ns);
    }
}
