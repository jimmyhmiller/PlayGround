use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use bevy_inspector_egui::quick::ResourceInspectorPlugin;
use bevy_inspector_egui::InspectorOptions;
use bevy_inspector_egui::prelude::ReflectInspectorOptions;

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
struct WallFootprint;

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

// =====================================================================
// Cone-light material
// =====================================================================

#[derive(ShaderType, Clone, Copy, Default, Debug)]
struct ConeLightParams {
    player_pos: Vec2,
    aim_dir: Vec2,
    cos_half_angle: f32,
    range: f32,
    ambient: f32,
    intensity: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
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
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, advance_physics)
        .add_systems(
            RunFixedMainLoop,
            (
                (accumulate_input, update_aim_from_mouse)
                    .in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
                (
                    interpolate_rendered_transform,
                    camera_follow_player,
                    y_sort,
                    update_cone_light,
                    follow_camera,
                )
                    .chain()
                    .in_set(RunFixedMainLoopSystems::AfterFixedMainLoop),
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

    // Walls — each gets a footprint (collision), a top face (lighter), and a front face (darker).
    let wall_specs: [(Vec2, Vec2); 5] = [
        (Vec2::new(-300.0, 150.0), Vec2::new(240.0, 40.0)),
        (Vec2::new(250.0, -50.0), Vec2::new(60.0, 220.0)),
        (Vec2::new(-150.0, -220.0), Vec2::new(320.0, 40.0)),
        (Vec2::new(420.0, 220.0), Vec2::new(120.0, 120.0)),
        (Vec2::new(-450.0, -40.0), Vec2::new(40.0, 260.0)),
    ];
    for (center, size) in wall_specs {
        spawn_wall(
            &mut commands,
            &mut meshes,
            &mut color_materials,
            center,
            size,
        );
    }

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
}

fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    center: Vec2,
    size: Vec2,
) {
    let wall_height = 48.0;

    // Collision + sort anchor lives on this invisible "footprint" entity.
    // Top face: drawn at footprint center, lighter.
    // Front face: drawn below, darker — fakes the side of the wall.
    let top_color = Color::srgb(0.58, 0.55, 0.50);
    let front_color = Color::srgb(0.28, 0.26, 0.24);

    // Footprint (collision)
    commands.spawn((
        WallFootprint,
        Collider { half: size * 0.5 },
        Transform::from_xyz(center.x, center.y, 0.0),
        GlobalTransform::default(),
        Visibility::Hidden,
    ));

    // Both wall pieces sort against the footprint's front (south) edge. This is
    // the wall's true "ground line" — the player should draw in front of the wall
    // the moment their feet are south of it.
    let sort_y = center.y - size.y * 0.5;

    // Top face — drawn at the footprint.
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(size.x, size.y))),
        MeshMaterial2d(materials.add(top_color)),
        Transform::from_xyz(center.x, center.y, 0.0),
        YSorted { ground_offset: sort_y - center.y },
    ));

    // Front face — sits visually below the footprint's front edge.
    let front_center = Vec2::new(center.x, center.y - size.y * 0.5 - wall_height * 0.5);
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(size.x, wall_height))),
        MeshMaterial2d(materials.add(front_color)),
        Transform::from_xyz(front_center.x, front_center.y, 0.0),
        YSorted { ground_offset: sort_y - front_center.y },
    ));
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
            mat.params = ConeLightParams {
                player_pos: tf.translation.truncate(),
                aim_dir: aim.0,
                cos_half_angle: tuning.light_half_angle_deg.to_radians().cos(),
                range: tuning.light_range,
                ambient: tuning.ambient,
                intensity: tuning.light_intensity,
            };
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
