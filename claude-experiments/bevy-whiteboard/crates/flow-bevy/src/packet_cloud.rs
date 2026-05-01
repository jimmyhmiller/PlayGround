//! GPU-instanced packet renderer.
//!
//! All visible packets render in a single draw call backed by a custom
//! `Material2d`. Each frame the CPU encodes the active packets into a
//! storage buffer and updates `clock.now_real`; the vertex shader does
//! the lerp from→to and rasterizes a small soft circle per packet (see
//! `shaders/packet_cloud.wgsl`).
//!
//! Mesh structure: a fixed-size mesh of `MAX_PACKETS * 6` vertices
//! whose per-vertex `ATTRIBUTE_POSITION` carries
//! `(corner.x, corner.y, instance_id_as_f32)`. The shader reads packet
//! id and corner from that attribute directly. Out-of-range slots
//! (`instance_id >= active_count`) collapse off-clip so we never need
//! to resize the mesh as the live count changes.
//!
//! `TravelingPacket` entities are still spawned (kept lightweight —
//! no mesh/material) so existing tests can keep asserting on the
//! per-packet view; this module is purely the rendering path.

use bevy::asset::embedded_asset;
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{AsBindGroup, BufferUsages, ShaderType};
use bevy::render::renderer::RenderQueue;
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{Extract, Render, RenderApp, RenderSystems};
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

impl PacketCloudMaterial {
    /// Number of packets the most recent `update_packet_cloud` pass
    /// packed into the storage buffer. Exposed for tests and (in the
    /// future) UI overlays that want to surface the active packet
    /// count without poking shader internals.
    pub fn active_count(&self) -> u32 {
        self.clock.active_count
    }
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

/// Main-world handoff for the per-frame instance bytes.
/// `update_packet_cloud` fills `bytes`; an extract system moves them
/// into `ExtractedPacketUpload`, and a `PrepareResources` system writes
/// them into the GPU buffer via `RenderQueue::write_buffer` — so the
/// underlying `wgpu::Buffer` is allocated *once* at startup and reused
/// every frame. Calling `ShaderStorageBuffer::set_data` per frame would
/// instead make `prepare_asset` reallocate the buffer every frame and
/// re-invalidate the material's bind group, which is a lot of churn
/// for no reason.
#[derive(Resource, Default)]
struct PendingPacketUpload {
    bytes: Vec<u8>,
    handle: Option<Handle<ShaderStorageBuffer>>,
}

#[derive(Resource, Default)]
struct ExtractedPacketUpload {
    bytes: Vec<u8>,
    handle_id: Option<AssetId<ShaderStorageBuffer>>,
}

pub struct PacketCloudPlugin;

impl Plugin for PacketCloudPlugin {
    fn build(&self, app: &mut App) {
        // Bake the WGSL into the binary so there's no runtime asset
        // discovery — Bevy resolves `embedded://flow_bevy/...` via
        // the macro registration below.
        embedded_asset!(app, "shaders/packet_cloud.wgsl");

        app.add_plugins(Material2dPlugin::<PacketCloudMaterial>::default())
            .init_resource::<PendingPacketUpload>()
            .add_systems(Startup, spawn_packet_cloud)
            .add_systems(Update, update_packet_cloud);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ExtractedPacketUpload>()
                .add_systems(ExtractSchedule, extract_packet_upload)
                .add_systems(
                    Render,
                    upload_packet_buffer.in_set(RenderSystems::PrepareResources),
                );
        }
    }
}

fn spawn_packet_cloud(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<PacketCloudMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut pending: ResMut<PendingPacketUpload>,
) {
    let mesh = build_packet_cloud_mesh(MAX_PACKETS);
    // Allocate the storage buffer once at MAX_PACKETS capacity, with
    // `COPY_DST` so we can rewrite contents each frame via
    // `RenderQueue::write_buffer` without ever reallocating. Leaving
    // `data` as `None` makes `prepare_asset` take the empty branch and
    // call `create_buffer` (not `create_buffer_with_data`), so the
    // buffer is created uninitialized — the first frame's
    // `write_buffer` populates it before the shader reads.
    let max_size = encode_packet_instances(vec![PacketInstance::default(); MAX_PACKETS])
        .len() as u64;
    let mut storage = ShaderStorageBuffer::default();
    storage.buffer_description.label = Some("packet_cloud_instances");
    storage.buffer_description.size = max_size;
    storage.buffer_description.usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    storage.buffer_description.mapped_at_creation = false;
    storage.asset_usage = RenderAssetUsages::default();
    let handle = buffers.add(storage);
    pending.handle = Some(handle.clone());
    let mat = PacketCloudMaterial {
        instances: handle,
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
    // `NoFrustumCulling`: the mesh's vertex positions encode corner +
    // instance id, not real world coordinates — actual packet
    // positions come from the storage buffer at draw time. The
    // mesh-derived AABB is therefore meaningless to the culler;
    // without this opt-out the cloud disappears whenever the camera
    // pans off the AABB.
    commands.spawn((
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(mats.add(mat)),
        Transform::from_xyz(0.0, 0.0, 3.0),
        Visibility::Visible,
        NoFrustumCulling,
        PacketCloud,
    ));
}

/// Build the cloud's vertex mesh: `MAX_PACKETS * 6` vertices, one
/// triangle list, no index buffer. Per-vertex `ATTRIBUTE_POSITION`
/// carries `(corner.x, corner.y, instance_id_as_f32)` so the shader
/// reads packet id and corner from the attribute directly.
///
/// Why a position attribute and not `@builtin(vertex_index)`: Bevy's
/// `MeshAllocator` packs every Mesh2d into one shared vertex buffer
/// and the 2D draw uses our slice's start as `firstVertex`. So
/// `@builtin(vertex_index)` returns a *global* index — when the slice
/// start isn't a multiple of 6, the 6 vertices of a single quad
/// straddle two storage-buffer entries and the quad stretches into a
/// streak between two packets' paths. Vertex attribute fetches are
/// local to our buffer slice, which sidesteps the offset entirely.
fn build_packet_cloud_mesh(max_packets: usize) -> Mesh {
    let n_verts = max_packets * 6;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let instance_id = (i / 6) as f32;
        let [cx, cy] = quad_corner(i % 6);
        positions.push([cx, cy, instance_id]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh
}

/// Two triangles per quad: corners 0..5 →
/// (-1,-1), (1,-1), (1,1), (1,1), (-1,1), (-1,-1).
fn quad_corner(i: usize) -> [f32; 2] {
    match i {
        0 | 5 => [-1.0, -1.0],
        1 => [1.0, -1.0],
        2 | 3 => [1.0, 1.0],
        4 => [-1.0, 1.0],
        _ => unreachable!("quad_corner called with i={i} > 5"),
    }
}

/// Each frame: walk the visual timeline, encode visible packets into
/// `PendingPacketUpload.bytes` for the render world to write into the
/// GPU buffer, and advance the clock uniform.
fn update_packet_cloud(
    timeline: Res<VisualTimelineRes>,
    clock: Res<SimClock>,
    theme: Res<Theme>,
    node_colors: Res<NodeColors>,
    nodes: Query<&Transform, With<FlowNodeRef>>,
    maps: Res<EntityMaps>,
    cloud: Query<&MeshMaterial2d<PacketCloudMaterial>, With<PacketCloud>>,
    mut mats: ResMut<Assets<PacketCloudMaterial>>,
    hide_all: Res<crate::edges::HideAll>,
    membership: Res<crate::compound::CompoundMembership>,
    current_scope: Res<crate::compound::CurrentScope>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
    mut pending: ResMut<PendingPacketUpload>,
) {
    let Ok(handle) = cloud.single() else { return };
    let Some(mat) = mats.get_mut(&handle.0) else { return };

    crate::time_phase!(perf, "packet_cloud.update", {
    let now = clock.visual_now;
    let mut packed: Vec<PacketInstance> = Vec::with_capacity(MAX_PACKETS);

    // Hide-all toggle (`H` key) zeroes out the active count so the
    // shader rasterizes nothing, mirroring the gating that `edges.rs`
    // applies to arrow drawing and per-packet entity spawning.
    let _ = now;
    if !hide_all.0 {
    for (vp, _prog) in timeline.visible.iter() {
        if packed.len() >= MAX_PACKETS {
            break;
        }
        // Scope filter — same rule as `Scoped` / `sync_scoped_visibility`,
        // applied here because the packet cloud is one GPU entity
        // (no per-packet Bevy entity to stamp). The packet inherits
        // its edge's canonical owner; we drop it if that owner's
        // scope doesn't match the current view.
        let owner = crate::compound::canonical_edge_owner(vp.from, vp.to, &membership);
        if membership.parent_of(owner) != current_scope.0 {
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

    pending.bytes = encode_packet_instances(packed);

    mat.clock = PacketClock {
        now_real: now as f32,
        active_count: count as u32,
        quad_radius: PACKET_RADIUS,
        _pad: 0.0,
    };
    });
}

/// Encode a `Vec<PacketInstance>` into the WGSL `array<PacketInstance>`
/// storage-buffer layout via encase, matching what `set_data` would do
/// — but landing the bytes in our own buffer so we can hand them to
/// `RenderQueue::write_buffer` instead of round-tripping through Bevy's
/// `ShaderStorageBuffer` asset re-prepare path.
fn encode_packet_instances(items: Vec<PacketInstance>) -> Vec<u8> {
    use bevy::render::render_resource::encase::StorageBuffer;
    let mut wrapper = StorageBuffer::<Vec<u8>>::new(Vec::new());
    wrapper.write(&PacketInstances { items }).unwrap();
    wrapper.into_inner()
}

/// Copy the encoded bytes from the main world into the render world's
/// `ExtractedPacketUpload`. `Extract` is read-only so we clone — bytes
/// are at most `MAX_PACKETS * stride` (~5 MB) and typically far less.
fn extract_packet_upload(
    pending: Extract<Res<PendingPacketUpload>>,
    mut extracted: ResMut<ExtractedPacketUpload>,
) {
    extracted.bytes.clear();
    extracted.bytes.extend_from_slice(&pending.bytes);
    extracted.handle_id = pending.handle.as_ref().map(|h| h.id());
}

/// Write the extracted bytes into the persistent GPU buffer in place.
/// Runs in `PrepareResources`, after `prepare_assets::<GpuShaderStorageBuffer>`
/// (which `RenderSystems::PrepareAssets` ensures has already created
/// the buffer for our handle).
fn upload_packet_buffer(
    extracted: Res<ExtractedPacketUpload>,
    ssbos: Res<RenderAssets<GpuShaderStorageBuffer>>,
    queue: Res<RenderQueue>,
) {
    if extracted.bytes.is_empty() {
        return;
    }
    let Some(id) = extracted.handle_id else { return };
    let Some(gpu) = ssbos.get(id) else { return };
    queue.write_buffer(&gpu.buffer, 0, &extracted.bytes);
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
            tl.set_k(200.0);
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
            .strategy
            .as_replay()
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

        // And the per-frame instance bytes should have been encoded
        // into `PendingPacketUpload` for the render world to upload.
        let pending = app.world().resource::<PendingPacketUpload>();
        assert!(
            !pending.bytes.is_empty(),
            "PendingPacketUpload.bytes should hold serialized instance data"
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
            tl.set_k(1.0);
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
