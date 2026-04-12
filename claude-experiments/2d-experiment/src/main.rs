use bevy::asset::RenderAssetUsages;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::camera::Projection;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use bevy_inspector_egui::quick::ResourceInspectorPlugin;
use bevy_inspector_egui::InspectorOptions;
use bevy_inspector_egui::prelude::ReflectInspectorOptions;
use serde::{Deserialize, Serialize};

const LEVEL_PATH: &str = "levels/level.ron";
const WALL_HEIGHT: f32 = 48.0;

const FIXED_HZ: f64 = 60.0;

// =====================================================================
// Tuning — edited live via the inspector window in the top-right.
// =====================================================================

#[derive(Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct Tuning {
    #[inspector(min = 0.0, max = 1000.0)]
    max_speed: f32,
    #[inspector(min = 0.0, max = 10000.0)]
    accel: f32,
    #[inspector(min = 0.0, max = 10000.0)]
    friction: f32,
    #[inspector(min = 50.0, max = 2000.0)]
    light_range: f32,
    #[inspector(min = 1.0, max = 180.0)]
    light_half_angle_deg: f32,
    #[inspector(min = 0.0, max = 4.0)]
    light_intensity: f32,
    #[inspector(min = 0.0, max = 1.0)]
    ambient: f32,
    #[inspector(min = 0.0, max = 30.0)]
    camera_smoothing: f32,
    #[inspector(min = 1.0, max = 4.0)]
    run_multiplier: f32,
    #[inspector(min = 0.1, max = 1.0)]
    sneak_multiplier: f32,
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
struct Player;

#[derive(Component, Copy, Clone)]
struct Collider {
    half: Vec2,
}

#[derive(Component, Default, Deref, DerefMut)]
struct DesiredDirection(Vec2);

#[derive(Component, Default)]
struct AimDirection(Vec2);

#[derive(Component)]
struct SpeedMode {
    multiplier: f32,
}

impl Default for SpeedMode {
    fn default() -> Self { Self { multiplier: 1.0 } }
}

#[derive(Component, Default, Deref, DerefMut)]
struct Velocity(Vec2);

#[derive(Component, Default, Deref, DerefMut)]
struct PhysicalTranslation(Vec2);

#[derive(Component, Default, Deref, DerefMut)]
struct PreviousPhysicalTranslation(Vec2);

#[derive(Component)]
struct WallFootprint {
    top_face: Entity,
    front_face: Entity,
}

// =====================================================================
// Game mode + editor state
// =====================================================================

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
enum GameMode {
    #[default]
    Playing,
    Editing,
}

#[derive(Resource, Default)]
struct EditorState {
    drag_start: Option<Vec2>,
}

#[derive(Serialize, Deserialize)]
struct WallData {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

#[derive(Serialize, Deserialize, Default)]
struct LevelData {
    walls: Vec<WallData>,
}

// Y-sort by `transform.y + ground_offset`. Use the offset so wall visuals
// (top cap, front face) sort by their shared ground-plane front edge rather
// than each visual's own center.
#[derive(Component)]
struct YSorted {
    ground_offset: f32,
}

impl YSorted {
    fn player() -> Self { Self { ground_offset: 0.0 } }
}

// Marker for the fullscreen light overlay — its position follows the camera.
#[derive(Component)]
struct LightOverlay;

// Marker for the shadow geometry mesh — regenerated each frame.
#[derive(Component)]
struct ShadowMesh;

// =====================================================================
// Cone-light material
// =====================================================================

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
        .add_plugins(ResourceInspectorPlugin::<Tuning>::default())
        .insert_resource(ClearColor(Color::srgb(0.10, 0.12, 0.15)))
        .insert_resource(Time::<Fixed>::from_hz(FIXED_HZ))
        .init_resource::<Tuning>()
        .register_type::<Tuning>()
        .init_state::<GameMode>()
        .init_resource::<EditorState>()
        .add_systems(Startup, setup)
        .add_systems(
            FixedUpdate,
            advance_physics.run_if(in_state(GameMode::Playing)),
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
            Update,
            (
                editor_camera_pan.run_if(in_state(GameMode::Editing)),
                editor_camera_zoom.run_if(in_state(GameMode::Editing)),
                editor_handle_placement.run_if(in_state(GameMode::Editing)),
                editor_handle_delete.run_if(in_state(GameMode::Editing)),
                editor_save.run_if(in_state(GameMode::Editing)),
            ),
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
) {
    commands.spawn(Camera2d);

    // Procedural cave floor — one big quad, shader samples world-space noise.
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(8192.0, 8192.0))),
        MeshMaterial2d(cave_materials.add(CaveFloorMaterial::default())),
        Transform::from_xyz(0.0, 0.0, -10.0),
    ));

    // Player — green square.
    let player_half = Vec2::new(14.0, 14.0);
    commands.spawn((
        Player,
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

    // Walls — load from disk, falling back to a hand-authored starter layout.
    load_level_or_default(&mut commands, &mut meshes, &mut color_materials);

    // Fullscreen cone-light overlay. The mesh is enormous so it covers whatever
    // the camera is looking at; we also follow the camera in `follow_camera`.
    let overlay_mesh = meshes.add(Rectangle::new(4096.0, 4096.0));
    let overlay_mat = cone_materials.add(ConeLightMaterial {
        params: ConeLightParams::default(),
    });
    commands.spawn((
        LightOverlay,
        Mesh2d(overlay_mesh),
        MeshMaterial2d(overlay_mat),
        Transform::from_xyz(0.0, 0.0, 900.0),
    ));

    // Shadow geometry mesh — regenerated each frame by `update_shadow_geometry`.
    // Rendered above the darkness overlay so it over-darkens areas where walls
    // block the cone.
    let shadow_mesh = meshes.add(empty_shadow_mesh());
    let shadow_mat = color_materials.add(Color::srgb(0.0, 0.0, 0.0));
    commands.spawn((
        ShadowMesh,
        Mesh2d(shadow_mesh),
        MeshMaterial2d(shadow_mat),
        Transform::from_xyz(0.0, 0.0, 901.0),
    ));
}

// Ray-segment intersection: returns the ray parameter `t` of the hit, or None.
// Ray is `origin + t * dir` with t >= 0; segment is `a + u * (b - a)` with u ∈ [0,1].
fn ray_segment_hit(origin: Vec2, dir: Vec2, a: Vec2, b: Vec2) -> Option<f32> {
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

fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    center: Vec2,
    size: Vec2,
) -> Entity {
    let top_color = Color::srgb(0.58, 0.55, 0.50);
    let front_color = Color::srgb(0.28, 0.26, 0.24);

    // Both wall pieces sort against the footprint's front (south) edge — the
    // wall's true "ground line".
    let sort_y = center.y - size.y * 0.5;

    let top_face = commands
        .spawn((
            Mesh2d(meshes.add(Rectangle::new(size.x, size.y))),
            MeshMaterial2d(materials.add(top_color)),
            Transform::from_xyz(center.x, center.y, 0.0),
            YSorted { ground_offset: sort_y - center.y },
        ))
        .id();

    let front_center = Vec2::new(center.x, center.y - size.y * 0.5 - WALL_HEIGHT * 0.5);
    let front_face = commands
        .spawn((
            Mesh2d(meshes.add(Rectangle::new(size.x, WALL_HEIGHT))),
            MeshMaterial2d(materials.add(front_color)),
            Transform::from_xyz(front_center.x, front_center.y, 0.0),
            YSorted { ground_offset: sort_y - front_center.y },
        ))
        .id();

    commands
        .spawn((
            WallFootprint { top_face, front_face },
            Collider { half: size * 0.5 },
            Transform::from_xyz(center.x, center.y, 0.0),
            GlobalTransform::default(),
            Visibility::Hidden,
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
    // Sneak takes priority so you can't "run-sneak".
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

        // Accel toward target velocity; friction when no input.
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

        // Integrate with axis-separated AABB resolution so the player slides along walls.
        let move_delta = velocity.0 * dt;
        let resolved =
            resolve_collision(current.0, collider.half, move_delta, &wall_list, &mut velocity.0);
        current.0 = resolved;
    }
}

fn resolve_collision(
    pos: Vec2,
    half: Vec2,
    delta: Vec2,
    walls: &[(Vec2, Vec2)],
    velocity: &mut Vec2,
) -> Vec2 {
    let eps = 0.001;
    let mut p = pos;

    // X axis
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

    // Y axis
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
// Y-sort — lower-Y sprites draw on top of higher-Y sprites.
// =====================================================================

fn y_sort(mut q: Query<(&mut Transform, &YSorted)>) {
    for (mut tf, sort) in &mut q {
        // Sort key is the entity's ground-plane front edge. Lower y → larger z → drawn on top.
        let ground_y = tf.translation.y + sort.ground_offset;
        tf.translation.z = 0.5 - ground_y * 0.0005;
    }
}

// =====================================================================
// Cone light — push tuning + player state into the shader uniforms.
// =====================================================================

fn update_cone_light(
    tuning: Res<Tuning>,
    players: Query<(&Transform, &AimDirection), With<Player>>,
    overlays: Query<&MeshMaterial2d<ConeLightMaterial>, With<LightOverlay>>,
    mut materials: ResMut<Assets<ConeLightMaterial>>,
) {
    let Ok((tf, aim)) = players.single() else { return };
    for handle in &overlays {
        if let Some(mat) = materials.get_mut(&handle.0) {
            mat.params.player_pos = tf.translation.truncate();
            mat.params.aim_dir = aim.0;
            mat.params.cos_half_angle = tuning.light_half_angle_deg.to_radians().cos();
            mat.params.range = tuning.light_range;
            mat.params.ambient = tuning.ambient;
            mat.params.intensity = tuning.light_intensity;
        }
    }
}

// Rebuild the shadow mesh each frame using the visibility-polygon technique
// from https://ncase.me/sight-and-light/. Cast a ray from the player toward
// every wall corner (plus ±ε to slip past corners), take the nearest wall hit,
// sort by angle, and fill the area OUTSIDE the resulting visibility polygon
// with dark quads — each quad spans from one hit point to the next and extends
// radially out to a far boundary. A handful of evenly-spaced filler rays cap
// the max angular gap so empty directions still get proper far coverage.
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
    // Push the shadow start a little past each wall hit so a thin strip of the
    // wall top facing the player stays lit. Enough to read the wall, not so
    // much that light visibly bleeds over.
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

        // For each adjacent pair of sorted rays, emit a quad that covers the
        // shadow wedge between them: from the hit points outward along the
        // rays to `far_dist`. The last→first wraparound closes the ring.
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
    // Exponential smoothing: frame-rate independent lerp.
    // A smoothing factor of 0 means the camera is hard-locked to the player;
    // higher values snap in faster. Infinite → instant.
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
                info!("loaded {} walls from {}", level.walls.len(), LEVEL_PATH);
                for w in level.walls {
                    spawn_wall(
                        commands,
                        meshes,
                        materials,
                        Vec2::new(w.cx, w.cy),
                        Vec2::new(w.w, w.h),
                    );
                }
                return;
            }
            Err(e) => warn!("failed to parse {}: {e}", LEVEL_PATH),
        }
    }

    // Fallback layout for first run.
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

fn editor_save(
    keys: Res<ButtonInput<KeyCode>>,
    walls: Query<(&Transform, &Collider), With<WallFootprint>>,
) {
    let mod_key = keys.pressed(KeyCode::ControlLeft)
        || keys.pressed(KeyCode::ControlRight)
        || keys.pressed(KeyCode::SuperLeft)
        || keys.pressed(KeyCode::SuperRight);
    if !(mod_key && keys.just_pressed(KeyCode::KeyS)) {
        return;
    }
    let data = LevelData {
        walls: walls
            .iter()
            .map(|(tf, c)| WallData {
                cx: tf.translation.x,
                cy: tf.translation.y,
                w: c.half.x * 2.0,
                h: c.half.y * 2.0,
            })
            .collect(),
    };
    if let Err(e) = std::fs::create_dir_all("levels") {
        warn!("couldn't create levels dir: {e}");
        return;
    }
    match ron::ser::to_string_pretty(&data, ron::ser::PrettyConfig::default()) {
        Ok(s) => match std::fs::write(LEVEL_PATH, s) {
            Ok(()) => info!("saved {} walls to {}", data.walls.len(), LEVEL_PATH),
            Err(e) => warn!("save failed: {e}"),
        },
        Err(e) => warn!("serialize failed: {e}"),
    }
}

// =====================================================================
// Editor — mode toggle, camera, placement, deletion
// =====================================================================

fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    current: Res<State<GameMode>>,
    mut next: ResMut<NextState<GameMode>>,
) {
    if keys.just_pressed(KeyCode::F1) {
        let new = match current.get() {
            GameMode::Playing => GameMode::Editing,
            GameMode::Editing => GameMode::Playing,
        };
        next.set(new);
    }
}

fn on_enter_editing(mut overlays: Query<&mut Visibility, With<LightOverlay>>) {
    for mut v in &mut overlays {
        *v = Visibility::Hidden;
    }
}

fn on_enter_playing(
    mut overlays: Query<&mut Visibility, With<LightOverlay>>,
    mut cameras: Query<&mut Projection, With<Camera2d>>,
) {
    for mut v in &mut overlays {
        *v = Visibility::Inherited;
    }
    // Snap zoom back so the gameplay framing is consistent regardless of where
    // the editor left it.
    for mut proj in &mut cameras {
        if let Projection::Orthographic(ref mut o) = *proj {
            o.scale = 1.0;
        }
    }
}

fn editor_camera_pan(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<(&mut Transform, &Projection), With<Camera2d>>,
) {
    let Ok((mut tf, proj)) = cameras.single_mut() else { return };
    let scale = match proj {
        Projection::Orthographic(o) => o.scale,
        _ => 1.0,
    };
    let mut d = Vec2::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp)    { d.y += 1.0; }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown)  { d.y -= 1.0; }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft)  { d.x -= 1.0; }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) { d.x += 1.0; }
    let speed = 700.0 * scale;
    let step = d.normalize_or_zero() * speed * time.delta_secs();
    tf.translation.x += step.x;
    tf.translation.y += step.y;
}

fn editor_camera_zoom(
    mut scroll: MessageReader<MouseWheel>,
    mut cameras: Query<&mut Projection, With<Camera2d>>,
) {
    let Ok(mut proj) = cameras.single_mut() else { return };
    let mut total = 0.0f32;
    for ev in scroll.read() {
        total += ev.y;
    }
    if total == 0.0 {
        return;
    }
    if let Projection::Orthographic(ref mut o) = *proj {
        o.scale = (o.scale * (1.0 - total * 0.025)).clamp(0.3, 10.0);
    }
}

fn cursor_world(
    windows: &Query<&Window>,
    cameras: &Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) -> Option<Vec2> {
    let window = windows.single().ok()?;
    let (camera, cam_tf) = cameras.single().ok()?;
    let pixel = window.cursor_position()?;
    camera.viewport_to_world_2d(cam_tf, pixel).ok()
}

fn editor_handle_placement(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut state: ResMut<EditorState>,
    mut gizmos: Gizmos,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    let Some(cursor) = cursor_world(&windows, &cameras) else { return };

    if buttons.just_pressed(MouseButton::Left) {
        state.drag_start = Some(cursor);
    }

    if let Some(start) = state.drag_start {
        let center = (start + cursor) * 0.5;
        let size = (cursor - start).abs();
        gizmos.rect_2d(
            Isometry2d::from_translation(center),
            size.max(Vec2::splat(1.0)),
            Color::srgb(1.0, 0.85, 0.15),
        );

        if buttons.just_released(MouseButton::Left) {
            state.drag_start = None;
            if size.x >= 10.0 && size.y >= 10.0 {
                spawn_wall(&mut commands, &mut meshes, &mut materials, center, size);
            }
        }
    }
}

fn editor_handle_delete(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    walls: Query<(Entity, &Transform, &Collider, &WallFootprint)>,
) {
    if !buttons.just_pressed(MouseButton::Right) {
        return;
    }
    let Some(cursor) = cursor_world(&windows, &cameras) else { return };
    for (entity, tf, col, wall) in &walls {
        let center = tf.translation.truncate();
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            commands.entity(wall.top_face).despawn();
            commands.entity(wall.front_face).despawn();
            commands.entity(entity).despawn();
            break;
        }
    }
}
