use bevy::camera::Projection;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy_inspector_egui::bevy_egui::{EguiContexts, egui};
use serde::{Deserialize, Serialize};

use crate::{
    Collider, GameMode, LightOverlay, ShadowMesh, Tuning, WallFootprint,
    spawn_wall,
};
use crate::enemies::*;

// =====================================================================
// Editor types
// =====================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Reflect)]
pub enum EditorTool {
    #[default]
    Wall,
    Monster,
    Sentinel,
}

impl EditorTool {
    pub fn label(&self) -> &'static str {
        match self {
            EditorTool::Wall => "Wall",
            EditorTool::Monster => "Monster",
            EditorTool::Sentinel => "Sentinel",
        }
    }
}

pub const EDITOR_CATEGORIES: &[(&str, &[EditorTool])] = &[
    ("Structures", &[EditorTool::Wall]),
    ("Enemies", &[EditorTool::Monster, EditorTool::Sentinel]),
];

#[derive(Clone, Debug)]
pub enum EditorAction {
    PlaceWall { center: Vec2, size: Vec2 },
    DeleteWall { center: Vec2, size: Vec2 },
    PlaceMonster { pos: Vec2, monster: MonsterSnapshot },
    DeleteMonster { pos: Vec2, monster: MonsterSnapshot },
    PlaceSentinel { pos: Vec2, sentinel: SentinelSnapshot },
    DeleteSentinel { pos: Vec2, sentinel: SentinelSnapshot },
}

#[derive(Clone, Debug)]
pub struct MonsterSnapshot {
    pub speed: f32,
    pub detection_range: f32,
    pub attack_reach: f32,
    pub strength: f32,
}

impl From<&Monster> for MonsterSnapshot {
    fn from(m: &Monster) -> Self {
        Self {
            speed: m.speed,
            detection_range: m.detection_range,
            attack_reach: m.attack_reach,
            strength: m.strength,
        }
    }
}

impl MonsterSnapshot {
    pub fn to_monster(&self) -> Monster {
        Monster {
            speed: self.speed,
            detection_range: self.detection_range,
            attack_reach: self.attack_reach,
            strength: self.strength,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SentinelSnapshot {
    pub light_range: f32,
    pub light_half_angle_deg: f32,
    pub light_intensity: f32,
    pub speed: f32,
    pub attack_reach: f32,
    pub sweep_speed: f32,
}

impl From<&Sentinel> for SentinelSnapshot {
    fn from(s: &Sentinel) -> Self {
        Self {
            light_range: s.light_range,
            light_half_angle_deg: s.light_half_angle_deg,
            light_intensity: s.light_intensity,
            speed: s.speed,
            attack_reach: s.attack_reach,
            sweep_speed: s.sweep_speed,
        }
    }
}

impl SentinelSnapshot {
    pub fn to_sentinel(&self) -> Sentinel {
        Sentinel {
            light_range: self.light_range,
            light_half_angle_deg: self.light_half_angle_deg,
            light_intensity: self.light_intensity,
            speed: self.speed,
            attack_reach: self.attack_reach,
            sweep_speed: self.sweep_speed,
        }
    }
}

#[derive(Resource, Default)]
pub struct EditorState {
    pub drag_start: Option<Vec2>,
    pub tool: EditorTool,
    pub undo_stack: Vec<EditorAction>,
    pub redo_stack: Vec<EditorAction>,
    pub selected: Option<Entity>,
    pub ui_has_pointer: bool,
}

#[derive(Serialize, Deserialize)]
pub struct MonsterData {
    pub x: f32,
    pub y: f32,
    pub speed: f32,
    pub detection_range: f32,
    pub attack_reach: f32,
    pub strength: f32,
}

#[derive(Serialize, Deserialize)]
pub struct SentinelData {
    pub x: f32,
    pub y: f32,
    pub light_range: f32,
    pub light_half_angle_deg: f32,
    pub light_intensity: f32,
    pub speed: f32,
    pub attack_reach: f32,
    pub sweep_speed: f32,
}

// =====================================================================
// Editor systems
// =====================================================================

pub fn toggle_mode(
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

pub fn on_enter_editing(
    mut overlays: Query<&mut Visibility, With<LightOverlay>>,
    mut shadows: Query<&mut Visibility, (With<ShadowMesh>, Without<LightOverlay>)>,
) {
    for mut v in &mut overlays {
        *v = Visibility::Hidden;
    }
    for mut v in &mut shadows {
        *v = Visibility::Hidden;
    }
}

pub fn on_enter_playing(
    mut overlays: Query<&mut Visibility, With<LightOverlay>>,
    mut shadows: Query<&mut Visibility, (With<ShadowMesh>, Without<LightOverlay>)>,
    mut cameras: Query<&mut Projection, With<Camera2d>>,
) {
    for mut v in &mut overlays {
        *v = Visibility::Inherited;
    }
    for mut v in &mut shadows {
        *v = Visibility::Inherited;
    }
    for mut proj in &mut cameras {
        if let Projection::Orthographic(ref mut o) = *proj {
            o.scale = 1.0;
        }
    }
}

pub fn editor_camera_pan(
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

pub fn editor_camera_zoom(
    mut scroll: MessageReader<MouseWheel>,
    mut cameras: Query<&mut Projection, With<Camera2d>>,
    state: Res<EditorState>,
) {
    if state.ui_has_pointer {
        for _ in scroll.read() {}
        return;
    }
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

pub fn editor_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<EditorState>,
    mut tuning: ResMut<Tuning>,
    mut monsters: Query<&mut Monster, Without<Sentinel>>,
    mut sentinels_q: Query<&mut Sentinel, Without<Monster>>,
    names: Query<&Name>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::SidePanel::left("editor_panel")
        .default_width(200.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Place");
                ui.separator();

                for &(category, tools) in EDITOR_CATEGORIES {
                    egui::CollapsingHeader::new(category)
                        .default_open(true)
                        .show(ui, |ui| {
                            for &tool in tools {
                                let selected = state.tool == tool;
                                if ui.selectable_label(selected, tool.label()).clicked() {
                                    state.tool = tool;
                                }
                            }
                        });
                }

                ui.add_space(8.0);

                if let Some(entity) = state.selected {
                    let label = names
                        .get(entity)
                        .map(|n| n.as_str().to_owned())
                        .unwrap_or_else(|_| format!("{entity:?}"));

                    ui.heading("Selected");
                    ui.separator();
                    ui.label(label);

                    if let Ok(mut monster) = monsters.get_mut(entity) {
                        ui.add_space(4.0);
                        ui.label("Monster");
                        ui.add(egui::Slider::new(&mut monster.speed, 0.0..=500.0).text("Speed"));
                        ui.add(egui::Slider::new(&mut monster.detection_range, 0.0..=800.0).text("Detection"));
                        ui.add(egui::Slider::new(&mut monster.attack_reach, 0.0..=100.0).text("Reach"));
                        ui.add(egui::Slider::new(&mut monster.strength, 0.0..=100.0).text("Strength"));
                    }

                    if let Ok(mut sentinel) = sentinels_q.get_mut(entity) {
                        ui.add_space(4.0);
                        ui.label("Sentinel");
                        ui.add(egui::Slider::new(&mut sentinel.light_range, 50.0..=500.0).text("Light Range"));
                        ui.add(egui::Slider::new(&mut sentinel.light_half_angle_deg, 1.0..=90.0).text("Light Angle"));
                        ui.add(egui::Slider::new(&mut sentinel.light_intensity, 0.0..=4.0).text("Intensity"));
                        ui.add(egui::Slider::new(&mut sentinel.speed, 0.0..=500.0).text("Speed"));
                        ui.add(egui::Slider::new(&mut sentinel.attack_reach, 0.0..=100.0).text("Reach"));
                        ui.add(egui::Slider::new(&mut sentinel.sweep_speed, 0.0..=180.0).text("Sweep Speed"));
                    }

                    if ui.button("Deselect").clicked() {
                        state.selected = None;
                    }

                    ui.add_space(8.0);
                }

                egui::CollapsingHeader::new("Tuning")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut tuning.max_speed, 0.0..=1000.0).text("Max Speed"));
                        ui.add(egui::Slider::new(&mut tuning.accel, 0.0..=10000.0).text("Accel"));
                        ui.add(egui::Slider::new(&mut tuning.friction, 0.0..=10000.0).text("Friction"));
                        ui.add(egui::Slider::new(&mut tuning.light_range, 50.0..=2000.0).text("Light Range"));
                        ui.add(egui::Slider::new(&mut tuning.light_half_angle_deg, 1.0..=180.0).text("Light Angle"));
                        ui.add(egui::Slider::new(&mut tuning.light_intensity, 0.0..=4.0).text("Light Intensity"));
                        ui.add(egui::Slider::new(&mut tuning.ambient, 0.0..=1.0).text("Ambient"));
                        ui.add(egui::Slider::new(&mut tuning.camera_smoothing, 0.0..=30.0).text("Cam Smoothing"));
                        ui.add(egui::Slider::new(&mut tuning.run_multiplier, 1.0..=4.0).text("Run Mult"));
                        ui.add(egui::Slider::new(&mut tuning.sneak_multiplier, 0.1..=1.0).text("Sneak Mult"));
                    });

                ui.add_space(8.0);

                egui::CollapsingHeader::new("Controls")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.small("Left-click: place / select");
                        ui.small("Right-click: delete");
                        ui.small("Cmd+Z: undo");
                        ui.small("Cmd+Shift+Z: redo");
                        ui.small("Cmd+S: save level");
                        ui.small("F1: back to game");
                    });
            });
        });

    state.ui_has_pointer = ctx.is_pointer_over_area();
}

pub fn editor_handle_placement(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut state: ResMut<EditorState>,
    mut gizmos: Gizmos,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    if state.ui_has_pointer { return; }
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) { return; }
    let Some(cursor) = cursor_world(&windows, &cameras) else { return };

    match state.tool {
        EditorTool::Wall => {
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
                        state.redo_stack.clear();
                        state.undo_stack.push(EditorAction::PlaceWall { center, size });
                    }
                }
            }
        }
        EditorTool::Monster => {
            if buttons.just_pressed(MouseButton::Left) {
                let snap = MonsterSnapshot::from(&Monster::default());
                spawn_monster(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    cursor,
                    Monster::default(),
                );
                state.redo_stack.clear();
                state.undo_stack.push(EditorAction::PlaceMonster { pos: cursor, monster: snap });
            }
        }
        EditorTool::Sentinel => {
            if buttons.just_pressed(MouseButton::Left) {
                let snap = SentinelSnapshot::from(&Sentinel::default());
                spawn_sentinel(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    cursor,
                    Sentinel::default(),
                );
                state.redo_stack.clear();
                state.undo_stack.push(EditorAction::PlaceSentinel { pos: cursor, sentinel: snap });
            }
        }
    }
}

pub fn editor_handle_delete(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    walls: Query<(Entity, &Transform, &Collider, &WallFootprint)>,
    monsters: Query<(Entity, &Transform, &Collider, &Monster), Without<Sentinel>>,
    sentinels_q: Query<(Entity, &Transform, &Collider, &Sentinel), Without<Monster>>,
    mut state: ResMut<EditorState>,
) {
    if state.ui_has_pointer { return; }
    if !buttons.just_pressed(MouseButton::Right) {
        return;
    }
    let Some(cursor) = cursor_world(&windows, &cameras) else { return };

    for (entity, tf, col, sentinel) in &sentinels_q {
        let center = tf.translation.truncate();
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            let snap = SentinelSnapshot::from(sentinel);
            commands.entity(entity).despawn();
            if state.selected == Some(entity) { state.selected = None; }
            state.redo_stack.clear();
            state.undo_stack.push(EditorAction::DeleteSentinel { pos: center, sentinel: snap });
            return;
        }
    }
    for (entity, tf, col, monster) in &monsters {
        let center = tf.translation.truncate();
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            let snap = MonsterSnapshot::from(monster);
            commands.entity(entity).despawn();
            if state.selected == Some(entity) { state.selected = None; }
            state.redo_stack.clear();
            state.undo_stack.push(EditorAction::DeleteMonster { pos: center, monster: snap });
            return;
        }
    }
    for (entity, tf, col, wall) in &walls {
        let center = tf.translation.truncate();
        let size = col.half * 2.0;
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            commands.entity(wall.top_face).despawn();
            commands.entity(wall.front_face).despawn();
            commands.entity(entity).despawn();
            if state.selected == Some(entity) { state.selected = None; }
            state.redo_stack.clear();
            state.undo_stack.push(EditorAction::DeleteWall { center, size });
            return;
        }
    }
}

pub fn editor_undo_redo(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<EditorState>,
    walls: Query<(Entity, &Transform, &Collider, &WallFootprint)>,
    monsters: Query<(Entity, &Transform, &Collider, &Monster)>,
    sentinels_q: Query<(Entity, &Transform, &Collider, &Sentinel)>,
) {
    let mod_key = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight)
        || keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    if !mod_key {
        return;
    }
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if shift && keys.just_pressed(KeyCode::KeyZ) {
        let Some(action) = state.redo_stack.pop() else { return };
        apply_action(&action, &mut commands, &mut meshes, &mut materials, &walls, &monsters, &sentinels_q);
        state.undo_stack.push(action);
    } else if keys.just_pressed(KeyCode::KeyZ) {
        let Some(action) = state.undo_stack.pop() else { return };
        let inverse = invert_action(&action);
        apply_action(&inverse, &mut commands, &mut meshes, &mut materials, &walls, &monsters, &sentinels_q);
        state.redo_stack.push(action);
    }
}

fn invert_action(action: &EditorAction) -> EditorAction {
    match action {
        EditorAction::PlaceWall { center, size } => EditorAction::DeleteWall { center: *center, size: *size },
        EditorAction::DeleteWall { center, size } => EditorAction::PlaceWall { center: *center, size: *size },
        EditorAction::PlaceMonster { pos, monster } => EditorAction::DeleteMonster { pos: *pos, monster: monster.clone() },
        EditorAction::DeleteMonster { pos, monster } => EditorAction::PlaceMonster { pos: *pos, monster: monster.clone() },
        EditorAction::PlaceSentinel { pos, sentinel } => EditorAction::DeleteSentinel { pos: *pos, sentinel: sentinel.clone() },
        EditorAction::DeleteSentinel { pos, sentinel } => EditorAction::PlaceSentinel { pos: *pos, sentinel: sentinel.clone() },
    }
}

fn apply_action(
    action: &EditorAction,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    walls: &Query<(Entity, &Transform, &Collider, &WallFootprint)>,
    monsters: &Query<(Entity, &Transform, &Collider, &Monster)>,
    sentinels: &Query<(Entity, &Transform, &Collider, &Sentinel)>,
) {
    match action {
        EditorAction::PlaceWall { center, size } => {
            spawn_wall(commands, meshes, materials, *center, *size);
        }
        EditorAction::DeleteWall { center, size: _ } => {
            for (entity, tf, _col, wall) in walls {
                let wc = tf.translation.truncate();
                if (wc - *center).length() < 1.0 {
                    commands.entity(wall.top_face).despawn();
                    commands.entity(wall.front_face).despawn();
                    commands.entity(entity).despawn();
                    break;
                }
            }
        }
        EditorAction::PlaceMonster { pos, monster } => {
            spawn_monster(commands, meshes, materials, *pos, monster.to_monster());
        }
        EditorAction::DeleteMonster { pos, .. } => {
            for (entity, tf, _col, _m) in monsters {
                let mc = tf.translation.truncate();
                if (mc - *pos).length() < 1.0 {
                    commands.entity(entity).despawn();
                    break;
                }
            }
        }
        EditorAction::PlaceSentinel { pos, sentinel } => {
            spawn_sentinel(commands, meshes, materials, *pos, sentinel.to_sentinel());
        }
        EditorAction::DeleteSentinel { pos, .. } => {
            for (entity, tf, _col, _s) in sentinels {
                let sc = tf.translation.truncate();
                if (sc - *pos).length() < 1.0 {
                    commands.entity(entity).despawn();
                    break;
                }
            }
        }
    }
}

pub fn editor_handle_select(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut state: ResMut<EditorState>,
    monsters: Query<(Entity, &Transform, &Collider), (With<Monster>, Without<Sentinel>)>,
    sentinels_q: Query<(Entity, &Transform, &Collider), (With<Sentinel>, Without<Monster>)>,
    walls: Query<(Entity, &Transform, &Collider), With<WallFootprint>>,
) {
    if state.ui_has_pointer { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if !keys.pressed(KeyCode::ShiftLeft) && !keys.pressed(KeyCode::ShiftRight) { return; }

    let Some(cursor) = cursor_world(&windows, &cameras) else { return };

    for (entity, tf, col) in &sentinels_q {
        let center = tf.translation.truncate();
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            state.selected = Some(entity);
            return;
        }
    }
    for (entity, tf, col) in &monsters {
        let center = tf.translation.truncate();
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            state.selected = Some(entity);
            return;
        }
    }
    for (entity, tf, col) in &walls {
        let center = tf.translation.truncate();
        if (cursor.x - center.x).abs() < col.half.x
            && (cursor.y - center.y).abs() < col.half.y
        {
            state.selected = Some(entity);
            return;
        }
    }
    state.selected = None;
}

pub fn editor_draw_monsters(
    mut gizmos: Gizmos,
    monsters: Query<(&Transform, &Monster)>,
    sentinels_q: Query<(&Transform, &Sentinel, &SentinelState)>,
) {
    for (tf, monster) in &monsters {
        let pos = tf.translation.truncate();
        gizmos.circle_2d(
            Isometry2d::from_translation(pos),
            monster.detection_range,
            Color::srgba(1.0, 0.3, 0.3, 0.3),
        );
        gizmos.circle_2d(
            Isometry2d::from_translation(pos),
            monster.attack_reach,
            Color::srgba(1.0, 0.0, 0.0, 0.6),
        );
    }
    for (tf, sentinel, sstate) in &sentinels_q {
        let pos = tf.translation.truncate();
        let half_rad = sentinel.light_half_angle_deg.to_radians();
        let left = Vec2::new(
            (sstate.aim_angle + half_rad).cos(),
            (sstate.aim_angle + half_rad).sin(),
        );
        let right = Vec2::new(
            (sstate.aim_angle - half_rad).cos(),
            (sstate.aim_angle - half_rad).sin(),
        );
        let color = Color::srgba(0.85, 0.65, 0.15, 0.5);
        gizmos.line_2d(pos, pos + left * sentinel.light_range, color);
        gizmos.line_2d(pos, pos + right * sentinel.light_range, color);
        let segments = 12;
        for i in 0..segments {
            let a1 = sstate.aim_angle - half_rad + (2.0 * half_rad * i as f32 / segments as f32);
            let a2 = sstate.aim_angle - half_rad + (2.0 * half_rad * (i + 1) as f32 / segments as f32);
            let p1 = pos + Vec2::new(a1.cos(), a1.sin()) * sentinel.light_range;
            let p2 = pos + Vec2::new(a2.cos(), a2.sin()) * sentinel.light_range;
            gizmos.line_2d(p1, p2, color);
        }
    }
}

pub fn editor_draw_selection(
    mut gizmos: Gizmos,
    state: Res<EditorState>,
    transforms: Query<(&Transform, &Collider)>,
) {
    let Some(entity) = state.selected else { return };
    let Ok((tf, col)) = transforms.get(entity) else {
        return;
    };
    let center = tf.translation.truncate();
    let size = col.half * 2.0 + Vec2::splat(4.0);
    gizmos.rect_2d(
        Isometry2d::from_translation(center),
        size,
        Color::srgb(0.2, 0.8, 1.0),
    );
}

pub fn editor_save(
    keys: Res<ButtonInput<KeyCode>>,
    walls: Query<(&Transform, &Collider), With<WallFootprint>>,
    monsters: Query<(&Transform, &Monster)>,
    sentinels: Query<(&Transform, &Sentinel)>,
) {
    let mod_key = keys.pressed(KeyCode::ControlLeft)
        || keys.pressed(KeyCode::ControlRight)
        || keys.pressed(KeyCode::SuperLeft)
        || keys.pressed(KeyCode::SuperRight);
    if !(mod_key && keys.just_pressed(KeyCode::KeyS)) {
        return;
    }
    let data = crate::LevelData {
        walls: walls
            .iter()
            .map(|(tf, c)| crate::WallData {
                cx: tf.translation.x,
                cy: tf.translation.y,
                w: c.half.x * 2.0,
                h: c.half.y * 2.0,
            })
            .collect(),
        monsters: monsters
            .iter()
            .map(|(tf, m)| MonsterData {
                x: tf.translation.x,
                y: tf.translation.y,
                speed: m.speed,
                detection_range: m.detection_range,
                attack_reach: m.attack_reach,
                strength: m.strength,
            })
            .collect(),
        sentinels: sentinels
            .iter()
            .map(|(tf, s)| SentinelData {
                x: tf.translation.x,
                y: tf.translation.y,
                light_range: s.light_range,
                light_half_angle_deg: s.light_half_angle_deg,
                light_intensity: s.light_intensity,
                speed: s.speed,
                attack_reach: s.attack_reach,
                sweep_speed: s.sweep_speed,
            })
            .collect(),
    };
    if let Err(e) = std::fs::create_dir_all("levels") {
        warn!("couldn't create levels dir: {e}");
        return;
    }
    match ron::ser::to_string_pretty(&data, ron::ser::PrettyConfig::default()) {
        Ok(s) => match std::fs::write(crate::LEVEL_PATH, s) {
            Ok(()) => info!("saved {} walls, {} monsters, {} sentinels to {}", data.walls.len(), data.monsters.len(), data.sentinels.len(), crate::LEVEL_PATH),
            Err(e) => warn!("save failed: {e}"),
        },
        Err(e) => warn!("serialize failed: {e}"),
    }
}
