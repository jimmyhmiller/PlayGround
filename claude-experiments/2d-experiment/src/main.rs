mod editor;
mod enemies;

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use bevy_inspector_egui::bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_inspector_egui::InspectorOptions;
use bevy_inspector_egui::prelude::ReflectInspectorOptions;
use serde::{Deserialize, Serialize};

use editor::*;
use enemies::*;

pub const LEVEL_PATH: &str = "levels/level.ron";
const WALL_HEIGHT: f32 = 48.0;
const FIXED_HZ: f64 = 60.0;

// =====================================================================
// Tuning
// =====================================================================

#[derive(Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
pub struct Tuning {
    #[inspector(min = 0.0, max = 1000.0)]
    pub max_speed: f32,
    #[inspector(min = 0.0, max = 10000.0)]
    pub accel: f32,
    #[inspector(min = 0.0, max = 10000.0)]
    pub friction: f32,
    #[inspector(min = 50.0, max = 2000.0)]
    pub light_range: f32,
    #[inspector(min = 1.0, max = 180.0)]
    pub light_half_angle_deg: f32,
    #[inspector(min = 0.0, max = 4.0)]
    pub light_intensity: f32,
    #[inspector(min = 0.0, max = 1.0)]
    pub ambient: f32,
    #[inspector(min = 0.0, max = 30.0)]
    pub camera_smoothing: f32,
    #[inspector(min = 1.0, max = 4.0)]
    pub run_multiplier: f32,
    #[inspector(min = 0.1, max = 1.0)]
    pub sneak_multiplier: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            max_speed: 280.0,
            accel: 3200.0,
            friction: 3600.0,
            light_range: 520.0,
            light_half_angle_deg: 83.0,
            light_intensity: 1.3,
            ambient: 0.0,
            camera_smoothing: 8.0,
            run_multiplier: 1.75,
            sneak_multiplier: 0.4,
        }
    }
}

// =====================================================================
// Components
// =====================================================================

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct PlayerSpawn(pub Vec2);

#[derive(Component, Copy, Clone)]
pub struct Collider {
    pub half: Vec2,
}

#[derive(Component, Default, Deref, DerefMut)]
pub struct DesiredDirection(Vec2);

#[derive(Component, Default)]
pub struct AimDirection(pub Vec2);

#[derive(Component)]
struct SpeedMode {
    multiplier: f32,
}

impl Default for SpeedMode {
    fn default() -> Self { Self { multiplier: 1.0 } }
}

#[derive(Component, Default, Deref, DerefMut)]
pub struct Velocity(pub Vec2);

#[derive(Component, Default, Deref, DerefMut)]
pub struct PhysicalTranslation(pub Vec2);

#[derive(Component, Default, Deref, DerefMut)]
pub struct PreviousPhysicalTranslation(pub Vec2);

#[derive(Component)]
pub struct WallFootprint {
    pub top_face: Entity,
    pub front_face: Entity,
}

// =====================================================================
// Game mode
// =====================================================================

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum GameMode {
    #[default]
    Playing,
    Editing,
}

#[derive(Serialize, Deserialize)]
pub struct WallData {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Serialize, Deserialize, Default)]
pub struct LevelData {
    pub walls: Vec<WallData>,
    #[serde(default)]
    pub monsters: Vec<MonsterData>,
    #[serde(default)]
    pub sentinels: Vec<SentinelData>,
}

#[derive(Component)]
pub struct YSorted {
    pub ground_offset: f32,
}

impl YSorted {
    pub fn player() -> Self { Self { ground_offset: 0.0 } }
}

#[derive(Component)]
pub struct LightOverlay;

#[derive(Component)]
pub struct ShadowMesh;

// =====================================================================
// Cone-light material
// =====================================================================

#[derive(ShaderType, Clone, Copy, Default)]
struct ExtraLightGpu {
    pos: Vec2,
    dir: Vec2,
    cos_half_angle: f32,
    range: f32,
    intensity: f32,
    _pad: f32,
}

#[derive(ShaderType, Clone, Copy, Default)]
struct ConeLightParams {
    player_pos: Vec2,
    aim_dir: Vec2,
    cos_half_angle: f32,
    range: f32,
    ambient: f32,
    intensity: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
struct ConeLightMaterial {
    #[uniform(0)]
    params: ConeLightParams,
    #[storage(1, read_only)]
    extras: Handle<ShaderStorageBuffer>,
}

impl Material2d for ConeLightMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/cone_light.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Default)]
struct CaveFloorMaterial {}

impl Material2d for CaveFloorMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/cave_floor.wgsl".into()
    }
}

// =====================================================================
// App
// =====================================================================

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "zelda-like".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<ConeLightMaterial>::default())
        .add_plugins(Material2dPlugin::<CaveFloorMaterial>::default())
        .add_plugins(EguiPlugin::default())
        .register_type::<Monster>()
        .register_type::<Sentinel>()
        .insert_resource(ClearColor(Color::srgb(0.10, 0.12, 0.15)))
        .insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
        .init_resource::<Tuning>()
        .register_type::<Tuning>()
        .init_state::<GameMode>()
        .init_resource::<EditorState>()
        .add_systems(Startup, setup)
        .add_systems(
            FixedUpdate,
            (
                advance_physics,
                monster_ai,
                sentinel_ai,
                monster_attack,
            )
                .chain()
                .run_if(in_state(GameMode::Playing)),
        )
        .add_systems(
            RunFixedMainLoop,
            (
                (
                    (accumulate_input, update_aim_from_mouse)
                        .run_if(in_state(GameMode::Playing)),
                )
                    .in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
                (
                    (
                        interpolate_rendered_transform,
                        camera_follow_player.run_if(in_state(GameMode::Playing)),
                        y_sort,
                        update_cone_light,
                        update_shadow_geometry,
                        follow_camera,
                    )
                        .chain(),
                )
                    .in_set(RunFixedMainLoopSystems::AfterFixedMainLoop),
            ),
        )
        .add_systems(Update, toggle_mode)
        .add_systems(OnEnter(GameMode::Editing), on_enter_editing)
        .add_systems(OnEnter(GameMode::Playing), on_enter_playing)
        .add_systems(
            EguiPrimaryContextPass,
            editor_ui.run_if(in_state(GameMode::Editing)),
        )
        .add_systems(
            Update,
            (
                editor_camera_pan,
                editor_camera_zoom,
                editor_handle_select,
                editor_handle_placement,
                editor_handle_delete,
                editor_undo_redo,
                editor_save,
                editor_draw_monsters,
                editor_draw_selection,
            )
                .run_if(in_state(GameMode::Editing)),
        )
        .run();
}

// =====================================================================
// Setup
// =====================================================================

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut cone_materials: ResMut<Assets<ConeLightMaterial>>,
    mut cave_materials: ResMut<Assets<CaveFloorMaterial>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    commands.spawn((Camera2d, Name::new("Camera")));

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(8192.0, 8192.0))),
        MeshMaterial2d(cave_materials.add(CaveFloorMaterial::default())),
        Transform::from_xyz(0.0, 0.0, -10.0),
        Name::new("Cave Floor"),
    ));

    let player_half = Vec2::new(14.0, 14.0);
    commands.spawn((
        Player,
        PlayerSpawn(Vec2::ZERO),
        Name::new("Player"),
        Mesh2d(meshes.add(Rectangle::new(player_half.x * 2.0, player_half.y * 2.0))),
        MeshMaterial2d(color_materials.add(Color::srgb(0.45, 0.92, 0.55))),
        Transform::from_xyz(0.0, 0.0, 1.0),
        Collider { half: player_half },
        DesiredDirection::default(),
        AimDirection(Vec2::X),
        Velocity::default(),
        PhysicalTranslation::default(),
        PreviousPhysicalTranslation::default(),
        SpeedMode::default(),
        YSorted::player(),
    ));

    load_level_or_default(&mut commands, &mut meshes, &mut color_materials);

    let overlay_mesh = meshes.add(Rectangle::new(4096.0, 4096.0));
    let extras_buffer = storage_buffers.add(ShaderStorageBuffer::default());
    let overlay_mat = cone_materials.add(ConeLightMaterial {
        params: ConeLightParams::default(),
        extras: extras_buffer,
    });
    commands.spawn((
        LightOverlay,
        Mesh2d(overlay_mesh),
        MeshMaterial2d(overlay_mat),
        Transform::from_xyz(0.0, 0.0, 900.0),
        Name::new("Light Overlay"),
    ));

    let shadow_mesh = meshes.add(empty_shadow_mesh());
    let shadow_mat = color_materials.add(Color::srgb(0.0, 0.0, 0.0));
    commands.spawn((
        ShadowMesh,
        Mesh2d(shadow_mesh),
        MeshMaterial2d(shadow_mat),
        Transform::from_xyz(0.0, 0.0, 901.0),
        Name::new("Shadow Mesh"),
    ));
}

// =====================================================================
// Helpers
// =====================================================================

pub fn ray_segment_hit(origin: Vec2, dir: Vec2, a: Vec2, b: Vec2) -> Option<f32> {
    let s = b - a;
    let denom = dir.x * s.y - dir.y * s.x;
    if denom.abs() < 1e-6 {
        return None;
    }
    let diff = a - origin;
    let t = (diff.x * s.y - diff.y * s.x) / denom;
    let u = (diff.x * dir.y - diff.y * dir.x) / denom;
    if t >= 0.0 && u >= 0.0 && u <= 1.0 {
        Some(t)
    } else {
        None
    }
}

fn empty_shadow_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

pub fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    center: Vec2,
    size: Vec2,
) -> Entity {
    let top_color = Color::srgb(0.58, 0.55, 0.50);
    let front_color = Color::srgb(0.28, 0.26, 0.24);
    let sort_y = center.y - size.y * 0.5;

    let top_face = commands
        .spawn((
            Mesh2d(meshes.add(Rectangle::new(size.x, size.y))),
            MeshMaterial2d(materials.add(top_color)),
            Transform::from_xyz(center.x, center.y, 0.0),
            YSorted { ground_offset: sort_y - center.y },
            Name::new("Wall Top"),
        ))
        .id();

    let front_center = Vec2::new(center.x, center.y - size.y * 0.5 - WALL_HEIGHT * 0.5);
    let front_face = commands
        .spawn((
            Mesh2d(meshes.add(Rectangle::new(size.x, WALL_HEIGHT))),
            MeshMaterial2d(materials.add(front_color)),
            Transform::from_xyz(front_center.x, front_center.y, 0.0),
            YSorted { ground_offset: sort_y - front_center.y },
            Name::new("Wall Front"),
        ))
        .id();

    commands
        .spawn((
            WallFootprint { top_face, front_face },
            Collider { half: size * 0.5 },
            Transform::from_xyz(center.x, center.y, 0.0),
            GlobalTransform::default(),
            Visibility::Hidden,
            Name::new("Wall"),
        ))
        .id()
}

// =====================================================================
// Input
// =====================================================================

fn accumulate_input(
    keys: Res<ButtonInput<KeyCode>>,
    tuning: Res<Tuning>,
    mut q: Query<(&mut DesiredDirection, &mut SpeedMode), With<Player>>,
) {
    let sneaking = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let running = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let multiplier = if sneaking {
        tuning.sneak_multiplier
    } else if running {
        tuning.run_multiplier
    } else {
        1.0
    };

    for (mut dir, mut mode) in &mut q {
        let mut d = Vec2::ZERO;
        if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp)    { d.y += 1.0; }
        if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown)  { d.y -= 1.0; }
        if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft)  { d.x -= 1.0; }
        if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) { d.x += 1.0; }
        dir.0 = d.normalize_or_zero();
        mode.multiplier = multiplier;
    }
}

fn update_aim_from_mouse(
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut players: Query<(&Transform, &mut AimDirection), With<Player>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_tf)) = cameras.single() else { return };
    let Some(cursor) = window.cursor_position() else { return };
    let Ok(world) = camera.viewport_to_world_2d(cam_tf, cursor) else { return };
    for (tf, mut aim) in &mut players {
        let to_cursor = world - tf.translation.truncate();
        if to_cursor.length_squared() > 1.0 {
            aim.0 = to_cursor.normalize();
        }
    }
}

// =====================================================================
// Physics + collision
// =====================================================================

fn advance_physics(
    fixed_time: Res<Time<Fixed>>,
    tuning: Res<Tuning>,
    walls: Query<(&Transform, &Collider), (With<WallFootprint>, Without<Player>)>,
    mut players: Query<
        (
            &mut PhysicalTranslation,
            &mut PreviousPhysicalTranslation,
            &mut Velocity,
            &DesiredDirection,
            &SpeedMode,
            &Collider,
        ),
        With<Player>,
    >,
) {
    let dt = fixed_time.delta_secs();
    let wall_list: Vec<(Vec2, Vec2)> = walls
        .iter()
        .map(|(tf, c)| (tf.translation.truncate(), c.half))
        .collect();

    for (mut current, mut previous, mut velocity, desired, mode, collider) in &mut players {
        previous.0 = current.0;

        let target = desired.0 * tuning.max_speed * mode.multiplier;
        let delta_v = target - velocity.0;
        let rate = if desired.0 == Vec2::ZERO {
            tuning.friction
        } else {
            tuning.accel
        };
        let max_step = rate * dt;
        let step = if delta_v.length() <= max_step {
            delta_v
        } else {
            delta_v.normalize_or_zero() * max_step
        };
        velocity.0 += step;

        let move_delta = velocity.0 * dt;
        let resolved =
            resolve_collision(current.0, collider.half, move_delta, &wall_list, &mut velocity.0);
        current.0 = resolved;
    }
}

pub fn resolve_collision(
    pos: Vec2,
    half: Vec2,
    delta: Vec2,
    walls: &[(Vec2, Vec2)],
    velocity: &mut Vec2,
) -> Vec2 {
    let eps = 0.001;
    let mut p = pos;

    p.x += delta.x;
    for (wc, wh) in walls {
        if aabb_overlap(p, half, *wc, *wh) {
            if delta.x > 0.0 {
                p.x = wc.x - wh.x - half.x - eps;
            } else if delta.x < 0.0 {
                p.x = wc.x + wh.x + half.x + eps;
            }
            velocity.x = 0.0;
        }
    }

    p.y += delta.y;
    for (wc, wh) in walls {
        if aabb_overlap(p, half, *wc, *wh) {
            if delta.y > 0.0 {
                p.y = wc.y - wh.y - half.y - eps;
            } else if delta.y < 0.0 {
                p.y = wc.y + wh.y + half.y + eps;
            }
            velocity.y = 0.0;
        }
    }

    p
}

fn aabb_overlap(pa: Vec2, ha: Vec2, pb: Vec2, hb: Vec2) -> bool {
    (pa.x - pb.x).abs() < (ha.x + hb.x) && (pa.y - pb.y).abs() < (ha.y + hb.y)
}

fn interpolate_rendered_transform(
    fixed_time: Res<Time<Fixed>>,
    mut q: Query<(
        &mut Transform,
        &PhysicalTranslation,
        &PreviousPhysicalTranslation,
    )>,
) {
    let alpha = fixed_time.overstep_fraction();
    for (mut transform, current, previous) in &mut q {
        let rendered = previous.0.lerp(current.0, alpha);
        transform.translation.x = rendered.x;
        transform.translation.y = rendered.y;
    }
}

// =====================================================================
// Y-sort
// =====================================================================

fn y_sort(mut q: Query<(&mut Transform, &YSorted)>) {
    for (mut tf, sort) in &mut q {
        let ground_y = tf.translation.y + sort.ground_offset;
        tf.translation.z = 0.5 - ground_y * 0.0005;
    }
}

// =====================================================================
// Cone light
// =====================================================================

fn update_cone_light(
    tuning: Res<Tuning>,
    players: Query<(&Transform, &AimDirection), With<Player>>,
    sentinel_lights: Query<(&Transform, &Sentinel, &SentinelState)>,
    overlays: Query<&MeshMaterial2d<ConeLightMaterial>, With<LightOverlay>>,
    mut materials: ResMut<Assets<ConeLightMaterial>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let Ok((tf, aim)) = players.single() else { return };

    let mut extras: Vec<ExtraLightGpu> = Vec::new();
    for (stf, sentinel, sstate) in &sentinel_lights {
        let dir = Vec2::new(sstate.aim_angle.cos(), sstate.aim_angle.sin());
        extras.push(ExtraLightGpu {
            pos: stf.translation.truncate(),
            dir,
            cos_half_angle: sentinel.light_half_angle_deg.to_radians().cos(),
            range: sentinel.light_range,
            intensity: sentinel.light_intensity,
            _pad: 0.0,
        });
    }
    // Storage buffers can't be zero-sized on some backends.
    if extras.is_empty() {
        extras.push(ExtraLightGpu::default());
    }

    for handle in &overlays {
        if let Some(mat) = materials.get_mut(&handle.0) {
            mat.params.player_pos = tf.translation.truncate();
            mat.params.aim_dir = aim.0;
            mat.params.cos_half_angle = tuning.light_half_angle_deg.to_radians().cos();
            mat.params.range = tuning.light_range;
            mat.params.ambient = tuning.ambient;
            mat.params.intensity = tuning.light_intensity;

            if let Some(buf) = storage_buffers.get_mut(&mat.extras) {
                buf.set_data(extras.as_slice());
            }
        }
    }
}

// =====================================================================
// Shadow geometry
// =====================================================================

fn update_shadow_geometry(
    players: Query<&Transform, With<Player>>,
    walls: Query<(&Transform, &Collider), With<WallFootprint>>,
    shadow_query: Query<&Mesh2d, With<ShadowMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(player_tf) = players.single() else { return };
    let light = player_tf.translation.truncate();
    let far_dist: f32 = 5000.0;
    const EPS: f32 = 0.0001;
    const FILLER_RAYS: usize = 16;
    const WALL_TOP_REVEAL: f32 = 20.0;

    let mut segments: Vec<(Vec2, Vec2)> = Vec::new();
    let mut endpoints: Vec<Vec2> = Vec::new();
    for (wtf, col) in &walls {
        let c = wtf.translation.truncate();
        let h = col.half;
        let bl = c + Vec2::new(-h.x, -h.y);
        let br = c + Vec2::new( h.x, -h.y);
        let tr = c + Vec2::new( h.x,  h.y);
        let tl = c + Vec2::new(-h.x,  h.y);
        segments.push((bl, br));
        segments.push((br, tr));
        segments.push((tr, tl));
        segments.push((tl, bl));
        endpoints.extend_from_slice(&[bl, br, tr, tl]);
    }

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    if !endpoints.is_empty() {
        let mut angles: Vec<f32> = Vec::with_capacity(endpoints.len() * 3 + FILLER_RAYS);
        for p in &endpoints {
            let a = (*p - light).to_angle();
            angles.push(a - EPS);
            angles.push(a);
            angles.push(a + EPS);
        }
        for i in 0..FILLER_RAYS {
            let a = -std::f32::consts::PI
                + (i as f32 / FILLER_RAYS as f32) * std::f32::consts::TAU;
            angles.push(a);
        }

        let mut hits: Vec<(f32, Vec2)> = angles
            .iter()
            .map(|&a| {
                let dir = Vec2::new(a.cos(), a.sin());
                let mut best_t = far_dist;
                for &(sa, sb) in &segments {
                    if let Some(t) = ray_segment_hit(light, dir, sa, sb) {
                        if t > 0.0 && t < best_t {
                            best_t = t;
                        }
                    }
                }
                (a, light + dir * best_t)
            })
            .collect();

        hits.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        let n = hits.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let (a_i, p_i) = hits[i];
            let (a_j, p_j) = hits[j];
            let dir_i = Vec2::new(a_i.cos(), a_i.sin());
            let dir_j = Vec2::new(a_j.cos(), a_j.sin());
            let f_i = light + dir_i * far_dist;
            let f_j = light + dir_j * far_dist;
            let p_i = p_i + dir_i * WALL_TOP_REVEAL;
            let p_j = p_j + dir_j * WALL_TOP_REVEAL;
            let base = positions.len() as u32;
            positions.push([p_i.x, p_i.y, 0.0]);
            positions.push([p_j.x, p_j.y, 0.0]);
            positions.push([f_j.x, f_j.y, 0.0]);
            positions.push([f_i.x, f_i.y, 0.0]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    for handle in &shadow_query {
        if let Some(mesh) = meshes.get_mut(&handle.0) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
            mesh.insert_indices(Indices::U32(indices.clone()));
        }
    }
}

// =====================================================================
// Camera
// =====================================================================

fn camera_follow_player(
    time: Res<Time>,
    tuning: Res<Tuning>,
    players: Query<&Transform, (With<Player>, Without<Camera2d>)>,
    mut cameras: Query<&mut Transform, With<Camera2d>>,
) {
    let Ok(player) = players.single() else { return };
    let Ok(mut cam) = cameras.single_mut() else { return };
    let target = player.translation.truncate();
    let current = cam.translation.truncate();
    let k = tuning.camera_smoothing;
    let alpha = if k <= 0.0 {
        1.0
    } else {
        1.0 - (-k * time.delta_secs()).exp()
    };
    let next = current.lerp(target, alpha);
    cam.translation.x = next.x;
    cam.translation.y = next.y;
}

fn follow_camera(
    cameras: Query<&Transform, (With<Camera2d>, Without<LightOverlay>)>,
    mut overlays: Query<&mut Transform, With<LightOverlay>>,
) {
    let Ok(cam) = cameras.single() else { return };
    for mut tf in &mut overlays {
        tf.translation.x = cam.translation.x;
        tf.translation.y = cam.translation.y;
    }
}

// =====================================================================
// Level save / load
// =====================================================================

fn load_level_or_default(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    if let Ok(s) = std::fs::read_to_string(LEVEL_PATH) {
        match ron::de::from_str::<LevelData>(&s) {
            Ok(level) => {
                info!("loaded {} walls, {} monsters, {} sentinels from {}", level.walls.len(), level.monsters.len(), level.sentinels.len(), LEVEL_PATH);
                for w in level.walls {
                    spawn_wall(
                        commands,
                        meshes,
                        materials,
                        Vec2::new(w.cx, w.cy),
                        Vec2::new(w.w, w.h),
                    );
                }
                for m in level.monsters {
                    spawn_monster(
                        commands,
                        meshes,
                        materials,
                        Vec2::new(m.x, m.y),
                        Monster {
                            speed: m.speed,
                            detection_range: m.detection_range,
                            attack_reach: m.attack_reach,
                            strength: m.strength,
                        },
                    );
                }
                for s in level.sentinels {
                    spawn_sentinel(
                        commands,
                        meshes,
                        materials,
                        Vec2::new(s.x, s.y),
                        Sentinel {
                            light_range: s.light_range,
                            light_half_angle_deg: s.light_half_angle_deg,
                            light_intensity: s.light_intensity,
                            speed: s.speed,
                            attack_reach: s.attack_reach,
                            sweep_speed: s.sweep_speed,
                        },
                    );
                }
                return;
            }
            Err(e) => warn!("failed to parse {}: {e}", LEVEL_PATH),
        }
    }

    let defaults: [(Vec2, Vec2); 5] = [
        (Vec2::new(-300.0, 150.0), Vec2::new(240.0, 40.0)),
        (Vec2::new(250.0, -50.0), Vec2::new(60.0, 220.0)),
        (Vec2::new(-150.0, -220.0), Vec2::new(320.0, 40.0)),
        (Vec2::new(420.0, 220.0), Vec2::new(120.0, 120.0)),
        (Vec2::new(-450.0, -40.0), Vec2::new(40.0, 260.0)),
    ];
    for (center, size) in defaults {
        spawn_wall(commands, meshes, materials, center, size);
    }
}
