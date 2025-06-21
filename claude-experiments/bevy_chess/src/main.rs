use bevy::prelude::*;

#[derive(Component, Clone, Copy, PartialEq)]
pub enum PieceColor {
    White,
    Black,
}

#[derive(Component, Clone, Copy, PartialEq)]
pub enum PieceType {
    King,
    Queen,
    Bishop,
    Knight,
    Rook,
    Pawn,
}

#[derive(Component)]
pub struct Piece {
    pub color: PieceColor,
    pub piece_type: PieceType,
    pub x: u8,
    pub y: u8,
}

#[derive(Component)]
pub struct Square {
    pub x: u8,
    pub y: u8,
}

#[derive(Resource, Default)]
struct SelectedSquare {
    entity: Option<Entity>,
}

#[derive(Resource, Default)]
struct SelectedPiece {
    entity: Option<Entity>,
}

#[derive(Resource)]
struct PlayerTurn(PieceColor);

impl Default for PlayerTurn {
    fn default() -> Self {
        Self(PieceColor::White)
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Chess!".into(),
                resolution: (1600., 1600.).into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<SelectedSquare>()
        .init_resource::<SelectedPiece>()
        .init_resource::<PlayerTurn>()
        .add_systems(Startup, (setup, create_board, create_pieces))
        .add_systems(Update, (color_squares, move_piece))
        .add_systems(Update, handle_input.before(move_piece))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(4.0, 8.0, 8.0).looking_at(Vec3::new(4.0, 0.0, 4.0), Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn create_board(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(Cuboid::new(1.0, 0.2, 1.0));
    let white_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.9, 0.9),
        ..default()
    });
    let black_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.1, 0.1),
        ..default()
    });

    for i in 0..8 {
        for j in 0..8 {
            let material = if (i + j + 1) % 2 == 0 {
                white_material.clone()
            } else {
                black_material.clone()
            };

            commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material),
                Transform::from_translation(Vec3::new(i as f32, 0.0, j as f32)),
                Square { x: i, y: j },
            ));
        }
    }
}

fn create_pieces(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Use simple shapes for all pieces
    let king_mesh = meshes.add(Cylinder::new(0.25, 1.0));      // Tallest
    let queen_mesh = meshes.add(Cylinder::new(0.2, 0.9));      // Second tallest
    let bishop_mesh = meshes.add(Cylinder::new(0.15, 0.7));    // Medium
    let knight_mesh = meshes.add(Cuboid::new(0.3, 0.8, 0.2));  // Rectangular
    let rook_mesh = meshes.add(Cuboid::new(0.3, 0.6, 0.3));    // Square
    let pawn_mesh = meshes.add(Cylinder::new(0.1, 0.5));       // Smallest

    // White back rank (row 0) - keeping original coordinate system
    spawn_piece(&mut commands, &mut materials, (0, 0), PieceType::Rook, PieceColor::White, rook_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 1), PieceType::Knight, PieceColor::White, knight_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 2), PieceType::Bishop, PieceColor::White, bishop_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 3), PieceType::Queen, PieceColor::White, queen_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 4), PieceType::King, PieceColor::White, king_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 5), PieceType::Bishop, PieceColor::White, bishop_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 6), PieceType::Knight, PieceColor::White, knight_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (0, 7), PieceType::Rook, PieceColor::White, rook_mesh.clone());

    // White pawns (row 1)
    for i in 0..8 {
        spawn_piece(&mut commands, &mut materials, (1, i), PieceType::Pawn, PieceColor::White, pawn_mesh.clone());
    }

    // Black back rank (row 7)
    spawn_piece(&mut commands, &mut materials, (7, 0), PieceType::Rook, PieceColor::Black, rook_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 1), PieceType::Knight, PieceColor::Black, knight_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 2), PieceType::Bishop, PieceColor::Black, bishop_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 3), PieceType::Queen, PieceColor::Black, queen_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 4), PieceType::King, PieceColor::Black, king_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 5), PieceType::Bishop, PieceColor::Black, bishop_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 6), PieceType::Knight, PieceColor::Black, knight_mesh.clone());
    spawn_piece(&mut commands, &mut materials, (7, 7), PieceType::Rook, PieceColor::Black, rook_mesh.clone());

    // Black pawns (row 6)
    for i in 0..8 {
        spawn_piece(&mut commands, &mut materials, (6, i), PieceType::Pawn, PieceColor::Black, pawn_mesh.clone());
    }
}

fn spawn_piece(
    commands: &mut Commands,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    position: (u8, u8),
    piece_type: PieceType,
    color: PieceColor,
    mesh: Handle<Mesh>,
) {
    // Create a unique material for each piece
    let material = materials.add(StandardMaterial {
        base_color: match color {
            PieceColor::White => Color::srgb(0.9, 0.9, 0.9),
            PieceColor::Black => Color::srgb(0.1, 0.1, 0.1),
        },
        ..default()
    });
    
    let transform = Transform::from_translation(Vec3::new(
        position.0 as f32,
        0.4,
        position.1 as f32,
    ));
    
    // Spawn the main piece 
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        transform,
        Piece {
            color,
            piece_type,
            x: position.0,
            y: position.1,
        },
    ));
}

fn handle_input(
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut selected_square: ResMut<SelectedSquare>,
    mut selected_piece: ResMut<SelectedPiece>,
    square_query: Query<(Entity, &Square, &Transform), Without<Piece>>,
    piece_query: Query<(Entity, &Piece, &Transform)>,
) {
    if mouse_button_input.just_pressed(MouseButton::Left) {
        let window = windows.single();
        if let Some(cursor_position) = window.cursor_position() {
            let (camera, camera_transform) = camera_q.single();
            
            if let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_position) {
                let mut selected_entity = None;
                let mut is_piece = false;
                
                // Convert screen coordinates to world coordinates on the chess board plane (y=0)
                let plane_y = 0.0;
                if ray.direction.y != 0.0 {
                    let t = (plane_y - ray.origin.y) / ray.direction.y;
                    let hit_point = ray.origin + ray.direction * t;
                    
                    // Check if we hit the chess board (0-7 range)
                    if hit_point.x >= -0.5 && hit_point.x <= 7.5 && hit_point.z >= -0.5 && hit_point.z <= 7.5 {
                        let board_x = hit_point.x.round() as u8;
                        let board_z = hit_point.z.round() as u8;
                        
                        // First check if there's a piece at this position
                        for (entity, piece, _) in piece_query.iter() {
                            if piece.x == board_x && piece.y == board_z {
                                selected_entity = Some(entity);
                                is_piece = true;
                                break;
                            }
                        }
                        
                        // If no piece found, find the square
                        if selected_entity.is_none() {
                            for (entity, square, _) in square_query.iter() {
                                if square.x == board_x && square.y == board_z {
                                    selected_entity = Some(entity);
                                    is_piece = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                if let Some(entity) = selected_entity {
                    if is_piece {
                        selected_piece.entity = Some(entity);
                        selected_square.entity = None;
                    } else {
                        selected_square.entity = Some(entity);
                    }
                }
            }
        }
    }
}

fn move_piece(
    mut commands: Commands,
    mut selected_piece: ResMut<SelectedPiece>,
    mut selected_square: ResMut<SelectedSquare>,
    mut piece_query: Query<(Entity, &mut Piece, &mut Transform)>,
    square_query: Query<&Square>,
    mut turn: ResMut<PlayerTurn>,
) {
    if let (Some(piece_entity), Some(square_entity)) = (selected_piece.entity, selected_square.entity) {
        if let Ok(target_square) = square_query.get(square_entity) {
            // Check if it's the correct player's turn
            if let Ok((_, piece, _)) = piece_query.get(piece_entity) {
                if piece.color == turn.0 {
                    let target_x = target_square.x;
                    let target_y = target_square.y;
                    
                    // Check if there's a piece at the target square and remove it (capture)
                    let mut captured_entity = None;
                    for (entity, other_piece, _) in piece_query.iter() {
                        if other_piece.x == target_x && other_piece.y == target_y && entity != piece_entity {
                            captured_entity = Some(entity);
                            break;
                        }
                    }
                    
                    if let Some(entity) = captured_entity {
                        commands.entity(entity).despawn();
                    }
                    
                    // Move the piece
                    if let Ok((_, mut piece, mut piece_transform)) = piece_query.get_mut(piece_entity) {
                        piece.x = target_x;
                        piece.y = target_y;
                        piece_transform.translation.x = target_x as f32;
                        piece_transform.translation.z = target_y as f32;
                        
                        
                        // Switch turns
                        turn.0 = match turn.0 {
                            PieceColor::White => PieceColor::Black,
                            PieceColor::Black => PieceColor::White,
                        };
                        
                        // Clear selections after successful move
                        selected_piece.entity = None;
                        selected_square.entity = None;
                    }
                }
                // For invalid moves (wrong turn), keep the selections so user can see the red square
            }
        }
    }
}

fn color_squares(
    mut materials: ResMut<Assets<StandardMaterial>>,
    square_query: Query<(&Square, &MeshMaterial3d<StandardMaterial>)>,
    piece_query: Query<(Entity, &Piece, &MeshMaterial3d<StandardMaterial>), Without<Square>>,
    selected_square: Res<SelectedSquare>,
    selected_piece: Res<SelectedPiece>,
) {
    // Reset all square colors
    for (square, material_handle) in &square_query {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            let base_color = if (square.x + square.y + 1) % 2 == 0 {
                Color::srgb(1.0, 0.9, 0.9)
            } else {
                Color::srgb(0.0, 0.1, 0.1)
            };
            material.base_color = base_color;
            material.emissive = LinearRgba::NONE; // Reset emissive
        }
    }

    // Highlight selected square with bright red
    if let Some(selected_entity) = selected_square.entity {
        if let Ok((_, material_handle)) = square_query.get(selected_entity) {
            if let Some(material) = materials.get_mut(&material_handle.0) {
                material.base_color = Color::srgb(1.0, 0.2, 0.2); // Bright red
                material.emissive = LinearRgba::rgb(0.3, 0.0, 0.0); // Red glow
            }
        }
    }

    // Reset all piece colors and highlight only the selected piece
    for (entity, piece, material_handle) in piece_query.iter() {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            if Some(entity) == selected_piece.entity {
                // Make selected piece bright yellow
                material.base_color = Color::srgb(1.0, 1.0, 0.0);
                material.emissive = LinearRgba::rgb(0.5, 0.5, 0.0);
            } else {
                // Reset to original colors based on piece color
                let original_color = match piece.color {
                    PieceColor::White => Color::srgb(0.9, 0.9, 0.9),
                    PieceColor::Black => Color::srgb(0.1, 0.1, 0.1),
                };
                material.base_color = original_color;
                material.emissive = LinearRgba::NONE;
            }
        }
    }
}
