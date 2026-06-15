//! Bevy proof-of-concept for the headless whiteboard.
//!
//! This is the integration this library was built for: a Bevy ECS app holds a
//! `whiteboard_core::Editor`, forwards Bevy's mouse/keyboard input into it, and
//! renders the editor's backend-neutral `DrawCommand` list using Bevy's own
//! drawing — gizmos for strokes and the selection overlay, triangulated `Mesh2d`
//! entities for fills. The whiteboard library never touches Bevy; Bevy is just
//! another backend.
//!
//! Controls (same as the other examples):
//!   r/o/d  rectangle/ellipse/diamond   l/a  line/arrow   f  freedraw
//!   v/1 select   e eraser   k laser    u undo   shift+u redo   Delete delete
//!   mouse drag  draw/move/resize        Esc  quit
//!
//! Coordinate note: the whiteboard works in screen-style coords (y grows down);
//! Bevy's 2D world has y growing up with the origin centered. `to_bevy` maps a
//! whiteboard scene point into Bevy world space for the active window size.

use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::MouseButtonInput;
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use whiteboard_core::editor::Editor;
use whiteboard_core::geometry::{Path as WbPath, PathSegment, Point as WbPoint};
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::render::{Color as WbColor, DrawCommand, Paint, RenderScene};
use whiteboard_core::shape::RoughGenerator;
use whiteboard_core::text::MonospaceMeasurer;

type WbEditor = Editor<MonospaceMeasurer, RoughGenerator>;

#[derive(Resource)]
struct Board {
    editor: WbEditor,
    cursor: WbPoint,
    mods: Modifiers,
    /// Entities spawned for fills this frame; despawned and rebuilt each update.
    fill_entities: Vec<Entity>,
}

/// Marks fill mesh entities so they can be cleared each frame.
#[derive(Component)]
struct FillMesh;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "headless-whiteboard · Bevy PoC".into(),
                resolution: (1024.0, 768.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.99, 0.99, 0.98)))
        .insert_resource(Board {
            editor: Editor::new_rough(MonospaceMeasurer::default()),
            cursor: WbPoint::ORIGIN,
            mods: Modifiers::default(),
            fill_entities: Vec::new(),
        })
        .add_systems(Startup, (setup, seed_scene))
        .add_systems(
            Update,
            (
                mouse_input,
                keyboard_input,
                render_scene.after(mouse_input).after(keyboard_input),
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Drop in a couple of shapes so the window isn't empty on launch.
fn seed_scene(mut board: ResMut<Board>) {
    use whiteboard_core::element::{Element, ElementId, ElementKind};
    let mut rect = Element::new(
        ElementId::from("seed-rect"),
        11,
        120.0,
        120.0,
        220.0,
        140.0,
        ElementKind::Rectangle,
    );
    rect.background_color = WbColor::rgb(255, 224, 178);
    rect.stroke_color = WbColor::rgb(230, 81, 0);
    rect.stroke_width = 2.0;
    board.editor.add_element(rect);

    let mut ell = Element::new(
        ElementId::from("seed-ell"),
        29,
        420.0,
        140.0,
        200.0,
        120.0,
        ElementKind::Ellipse,
    );
    ell.stroke_color = WbColor::rgb(13, 71, 161);
    ell.stroke_width = 2.0;
    board.editor.add_element(ell);
}

/// Map a whiteboard scene point (y-down, top-left origin) into Bevy world space
/// (y-up, centered origin) for the current window.
fn to_bevy(p: WbPoint, win_w: f32, win_h: f32) -> Vec2 {
    Vec2::new(p.x as f32 - win_w / 2.0, win_h / 2.0 - p.y as f32)
}

/// Whiteboard screen position from a Bevy cursor position (already in physical
/// top-left coords from the window event), passed straight through — the
/// whiteboard also uses top-left y-down screen coords.
fn cursor_to_wb(cursor: Vec2) -> WbPoint {
    WbPoint::new(cursor.x as f64, cursor.y as f64)
}

fn mouse_input(
    mut board: ResMut<Board>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut motion: EventReader<CursorMoved>,
    mut buttons: EventReader<MouseButtonInput>,
) {
    let Ok(_window) = windows.single() else {
        return;
    };

    for ev in motion.read() {
        board.cursor = cursor_to_wb(ev.position);
        let (pos, mods) = (board.cursor, board.mods);
        board.editor.handle(InputEvent::PointerMove { pos, mods });
    }

    for ev in buttons.read() {
        let Some(button) = to_wb_button(ev.button) else {
            continue;
        };
        let (pos, mods) = (board.cursor, board.mods);
        let event = match ev.state {
            ButtonState::Pressed => InputEvent::PointerDown { pos, button, mods },
            ButtonState::Released => InputEvent::PointerUp { pos, button, mods },
        };
        board.editor.handle(event);
    }
}

fn to_wb_button(b: MouseButton) -> Option<PointerButton> {
    match b {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

fn keyboard_input(
    mut board: ResMut<Board>,
    keys: Res<ButtonInput<KeyCode>>,
    mut key_evs: EventReader<KeyboardInput>,
    mut exit: EventWriter<AppExit>,
) {
    // Track modifier state.
    board.mods = Modifiers {
        shift: keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight),
        ctrl: keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight),
        alt: keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight),
        meta: keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight),
    };

    for ev in key_evs.read() {
        if ev.state != ButtonState::Pressed {
            continue;
        }
        match &ev.key_code {
            KeyCode::Escape => {
                exit.write(AppExit::Success);
            }
            KeyCode::Delete | KeyCode::Backspace => {
                board.editor.delete_selection();
            }
            KeyCode::KeyU => {
                if board.mods.shift {
                    board.editor.redo();
                } else {
                    board.editor.undo();
                }
            }
            code => {
                if let Some(tool) = tool_for(code) {
                    board.editor.set_tool(tool);
                }
            }
        }
    }
}

fn tool_for(code: &KeyCode) -> Option<Tool> {
    Some(match code {
        KeyCode::KeyR => Tool::Rectangle,
        KeyCode::KeyO => Tool::Ellipse,
        KeyCode::KeyD => Tool::Diamond,
        KeyCode::KeyL => Tool::Line,
        KeyCode::KeyA => Tool::Arrow,
        KeyCode::KeyF => Tool::Freedraw,
        KeyCode::KeyE => Tool::Eraser,
        KeyCode::KeyK => Tool::Laser,
        KeyCode::KeyV | KeyCode::Digit1 => Tool::Select,
        _ => return None,
    })
}

/// Render the editor's draw-command list each frame: strokes + fill outlines via
/// gizmos, fills via rebuilt triangulated meshes.
fn render_scene(
    mut board: ResMut<Board>,
    mut gizmos: Gizmos,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    old_fills: Query<Entity, With<FillMesh>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let (w, h) = (window.width(), window.height());

    // Clear last frame's fill meshes.
    for e in &old_fills {
        commands.entity(e).despawn();
    }
    board.fill_entities.clear();

    let scene: RenderScene = board.editor.render_with_overlay();

    // A simple transform stack (the whiteboard emits balanced push/pop pairs).
    let mut stack: Vec<whiteboard_core::geometry::Transform> =
        vec![whiteboard_core::geometry::Transform::IDENTITY];

    for cmd in &scene.commands {
        let top = *stack.last().unwrap();
        match cmd {
            DrawCommand::PushTransform(t) => stack.push(top.then(t)),
            DrawCommand::PopTransform => {
                if stack.len() > 1 {
                    stack.pop();
                }
            }
            // Clipping is not modelled in this PoC; ignore the clip rects.
            DrawCommand::PushClip(_) | DrawCommand::PopClip => {}
            DrawCommand::StrokePath { path, paint, .. } => {
                let color = paint_color(paint);
                draw_stroke(&mut gizmos, path, &top, color, w, h);
            }
            DrawCommand::FillPath { path, paint } => {
                let color = paint_color(paint);
                if let Some(entity) = spawn_fill(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    path,
                    &top,
                    color,
                    w,
                    h,
                ) {
                    board.fill_entities.push(entity);
                }
            }
            // Text and images are out of scope for this PoC's renderer.
            DrawCommand::DrawText { .. } | DrawCommand::DrawImage { .. } => {}
        }
    }
}

fn paint_color(paint: &Paint) -> Color {
    let Paint::Solid(c) = paint;
    Color::srgba(
        c.r as f32 / 255.0,
        c.g as f32 / 255.0,
        c.b as f32 / 255.0,
        c.a as f32 / 255.0,
    )
}

/// Flatten a whiteboard path into world-space polylines (one per subpath),
/// applying the active transform and the scene→Bevy mapping.
fn flatten(
    path: &WbPath,
    t: &whiteboard_core::geometry::Transform,
    w: f32,
    h: f32,
) -> Vec<Vec<Vec2>> {
    let mut subpaths: Vec<Vec<Vec2>> = Vec::new();
    let mut current: Vec<Vec2> = Vec::new();
    let mut last = WbPoint::ORIGIN;
    let map = |p: WbPoint| to_bevy(t.apply(p), w, h);

    for seg in &path.segments {
        match seg {
            PathSegment::MoveTo(p) => {
                if current.len() >= 2 {
                    subpaths.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
                last = *p;
                current.push(map(*p));
            }
            PathSegment::LineTo(p) => {
                last = *p;
                current.push(map(*p));
            }
            PathSegment::CubicTo { c1, c2, to } => {
                // Flatten the cubic to a handful of line segments.
                const STEPS: usize = 16;
                for i in 1..=STEPS {
                    let u = i as f64 / STEPS as f64;
                    current.push(map(cubic_point(last, *c1, *c2, *to, u)));
                }
                last = *to;
            }
            PathSegment::Close => {
                if let Some(first) = current.first().copied() {
                    current.push(first);
                }
                if current.len() >= 2 {
                    subpaths.push(std::mem::take(&mut current));
                }
            }
        }
    }
    if current.len() >= 2 {
        subpaths.push(current);
    }
    subpaths
}

fn cubic_point(p0: WbPoint, c1: WbPoint, c2: WbPoint, p3: WbPoint, u: f64) -> WbPoint {
    let v = 1.0 - u;
    let (a, b, c, d) = (v * v * v, 3.0 * v * v * u, 3.0 * v * u * u, u * u * u);
    WbPoint::new(
        a * p0.x + b * c1.x + c * c2.x + d * p3.x,
        a * p0.y + b * c1.y + c * c2.y + d * p3.y,
    )
}

fn draw_stroke(
    gizmos: &mut Gizmos,
    path: &WbPath,
    t: &whiteboard_core::geometry::Transform,
    color: Color,
    w: f32,
    h: f32,
) {
    for sub in flatten(path, t, w, h) {
        gizmos.linestrip(sub.into_iter().map(|v| v.extend(0.0)), color);
    }
}

/// Build a triangulated mesh (centroid fan) for the first subpath of a fill and
/// spawn it. Convex shapes (rect/ellipse/diamond) triangulate correctly this way.
#[allow(clippy::too_many_arguments)]
fn spawn_fill(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    path: &WbPath,
    t: &whiteboard_core::geometry::Transform,
    color: Color,
    w: f32,
    h: f32,
) -> Option<Entity> {
    let sub = flatten(path, t, w, h).into_iter().next()?;
    if sub.len() < 3 {
        return None;
    }
    // Centroid.
    let centroid = sub.iter().copied().fold(Vec2::ZERO, |a, b| a + b) / sub.len() as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(sub.len() + 1);
    positions.push([centroid.x, centroid.y, 0.0]);
    for v in &sub {
        positions.push([v.x, v.y, 0.0]);
    }
    let mut indices: Vec<u32> = Vec::with_capacity(sub.len() * 3);
    let n = sub.len() as u32;
    for i in 0..n {
        let next = (i + 1) % n;
        indices.push(0);
        indices.push(1 + i);
        indices.push(1 + next);
    }

    let mut mesh = Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));

    let entity = commands
        .spawn((
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(materials.add(color)),
            // Fills sit behind strokes (drawn via gizmos which are on top).
            Transform::from_xyz(0.0, 0.0, -1.0),
            FillMesh,
        ))
        .id();
    Some(entity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::geometry::Transform as WbTransform;

    #[test]
    fn scene_to_bevy_centers_and_flips_y() {
        // Scene origin (top-left) maps to the top-left of Bevy world space:
        // x = -w/2, y = +h/2.
        let p = to_bevy(WbPoint::new(0.0, 0.0), 800.0, 600.0);
        assert_eq!(p, Vec2::new(-400.0, 300.0));
        // Center of the scene maps to the world origin.
        let c = to_bevy(WbPoint::new(400.0, 300.0), 800.0, 600.0);
        assert_eq!(c, Vec2::ZERO);
        // Y is flipped: moving down in scene moves down in world.
        let down = to_bevy(WbPoint::new(400.0, 400.0), 800.0, 600.0);
        assert!(down.y < c.y);
    }

    #[test]
    fn cubic_endpoints_are_exact() {
        let p0 = WbPoint::new(0.0, 0.0);
        let c1 = WbPoint::new(0.0, 10.0);
        let c2 = WbPoint::new(10.0, 10.0);
        let p3 = WbPoint::new(10.0, 0.0);
        assert_eq!(cubic_point(p0, c1, c2, p3, 0.0), p0);
        assert_eq!(cubic_point(p0, c1, c2, p3, 1.0), p3);
    }

    #[test]
    fn flatten_polygon_yields_closed_subpath() {
        let path = WbPath::polygon(&[
            WbPoint::new(0.0, 0.0),
            WbPoint::new(20.0, 0.0),
            WbPoint::new(10.0, 20.0),
        ]);
        let subs = flatten(&path, &WbTransform::IDENTITY, 200.0, 200.0);
        assert_eq!(subs.len(), 1);
        // Closed: first == last point.
        let s = &subs[0];
        assert!((s.first().unwrap().distance(*s.last().unwrap())) < 1e-3);
    }

    #[test]
    fn flatten_cubic_produces_many_points() {
        let mut b = WbPath::builder();
        b.move_to(WbPoint::new(0.0, 0.0)).cubic_to(
            WbPoint::new(0.0, 30.0),
            WbPoint::new(30.0, 30.0),
            WbPoint::new(30.0, 0.0),
        );
        let path = b.build();
        let subs = flatten(&path, &WbTransform::IDENTITY, 100.0, 100.0);
        // The cubic is flattened into many line segments, not just 2 endpoints.
        assert!(
            subs[0].len() > 8,
            "cubic flattened to {} pts",
            subs[0].len()
        );
    }

    #[test]
    fn paint_color_maps_channels() {
        let c = paint_color(&Paint::solid(WbColor::rgba(255, 128, 0, 255)));
        let srgba = c.to_srgba();
        assert!((srgba.red - 1.0).abs() < 1e-6);
        assert!((srgba.green - 0.5).abs() < 0.02);
        assert!(srgba.blue.abs() < 1e-6);
    }
}
