use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point as SdlPoint;
use sdl2::render::Canvas;
use sdl2::EventPump;
use std::collections::HashSet;
use std::ops::Add;
use std::ops::Mul;

use sdl2::video::Window;

extern crate sdl2;
extern crate wolf;

use wolf::world_map::get_world_map;

// code borrowed from https://lodev.org/cgtutor/raycasting.html

fn clear_canvas(canvas: &mut Canvas<Window>) {
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    // fills the canvas with the color we set in `set_draw_color`.
    canvas.clear();
}

#[derive(Debug, PartialEq, Clone)]
struct Point<T> {
    x: T,
    y: T,
}

impl<T: Add<Output = T>> Add for Point<T> {
    type Output = Point<T>;

    fn add(self, other: Point<T>) -> Point<T> {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<T: Mul<Output = T>> Mul for Point<T> {
    type Output = Point<T>;

    fn mul(self, other: Point<T>) -> Point<T> {
        Point {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl Mul<f64> for Point<f64> {
    type Output = Point<f64>;

    fn mul(self, other: f64) -> Point<f64> {
        Point {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

type Camera = Point<f64>;
type Facing = Point<f64>;
type Position = Point<f64>;
type PositionDelta = Point<f64>;

#[derive(Debug, PartialEq, Clone)]
struct Player {
    facing: Facing,
    position: Position,
    camera: Camera,
    rotation_speed: f64,
    move_speed: f64,
}

fn rotate_point(point: &Point<f64>, rotation_speed: f64) -> Point<f64> {
    Point {
        x: point.x * rotation_speed.cos() - point.y * rotation_speed.sin(),
        y: point.x * rotation_speed.sin() + point.y * rotation_speed.cos(),
    }
}

fn rotate_player(player: &Player, dir: f64) -> Player {
    let mut new_player = player.clone();
    new_player.facing = rotate_point(&player.facing, dir * player.rotation_speed);
    new_player.camera = rotate_point(&player.camera, dir * player.rotation_speed);
    new_player
}

const EMPTY_DELTA: PositionDelta = PositionDelta { x: 0.0, y: 0.0 };

// I'm really not sure if this way of doing things is better
fn move_position(world: &Vec<Vec<usize>>, player: &Player, delta: PositionDelta) -> Player {
    let mut new_player = player.clone();
    if world[(player.position.x + delta.x) as usize][(player.position.y + delta.y) as usize] == 0 {
        new_player.position = new_player.position + delta;
    } else {
        new_player.position = new_player.position + EMPTY_DELTA;
    }
    new_player
}

fn darken_color(color: Color) -> Color {
    Color::RGB(color.r / 2, color.g / 2, color.b / 2)
}

fn handle_key_presses(
    event_pump: &EventPump,
    player: &Player,
    players: &mut Vec<Player>,
    world_map: &Vec<Vec<usize>>,
) -> (bool, Player) {
    let keys: HashSet<Keycode> = event_pump
        .keyboard_state()
        .pressed_scancodes()
        .filter_map(Keycode::from_scancode)
        .collect();

    if keys.contains(&Keycode::B) {
        (true, players.pop().unwrap())
    } else if keys.contains(&Keycode::Right) {
        (false, rotate_player(player, -1.0))
    } else if keys.contains(&Keycode::Left) {
        (false, rotate_player(player, 1.0))
    } else if keys.contains(&Keycode::Up) {
        let delta = PositionDelta {
            x: player.facing.x * player.move_speed,
            y: player.facing.y * player.move_speed,
        };
        (false, move_position(&world_map, &player, delta))
    } else if keys.contains(&Keycode::Down) {
        let delta = PositionDelta {
            x: -player.facing.x * player.move_speed,
            y: -player.facing.y * player.move_speed,
        };
        (false, move_position(&world_map, &player, delta))
    } else {
        (false, player.clone())
    }
}

fn main() {
    let world_map = get_world_map();

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let map_width: i32 = 1680;
    let map_height: i32 = 1050;

    let window = video_subsystem
        .window("Example", map_width as u32, map_height as u32)
        .fullscreen()
        .build()
        .unwrap();

    // Let's create a Canvas which we will use to draw in our Window
    let mut canvas: Canvas<Window> = window.into_canvas().present_vsync().build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut running = true;

    let mut timer = sdl_context.timer().unwrap();
    let mut player = Player {
        facing: Facing { x: -1.0, y: 0.0 },
        position: Position { x: 12.0, y: 22.0 },
        camera: Camera { x: 0.0, y: 0.66 },
        rotation_speed: 0.0,
        move_speed: 0.0,
    };
    let mut time = timer.ticks();
    let mut old_time;

    let mut players: Vec<Player> = vec![];

    while running {
        old_time = time;
        time = timer.ticks();
        let frame_time = (time - old_time) as f64 / 1000.0;
        player.rotation_speed = frame_time * 5.0;
        player.move_speed = frame_time * 3.0;

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    running = false;
                }

                Event::MouseButtonDown { x: _, y: _, .. } => {}
                _ => {}
            }
        }

        let (rewind, new_player) =
            handle_key_presses(&event_pump, &player, &mut players, &world_map);

        if !rewind && new_player != player {
            players.push(new_player.clone());
        }

        player = new_player;

        clear_canvas(&mut canvas);

        for x in 0..map_width {
            let camera_x = 2.0 * (x as f64) / (map_width as f64) - 1.0; // x-coordinate in player.camera space

            let ray_player_facing = player.facing.clone() + player.camera.clone() * camera_x;

            let mut map = Point {
                x: player.position.x as i32,
                y: player.position.y as i32,
            };

            let mut side_dist = Point { x: 0.0, y: 0.0 };

            let delta_dist = Point {
                x: (1.0 / ray_player_facing.x).abs(),
                y: (1.0 / ray_player_facing.y).abs(),
            };

            let mut perp_wall_dist: f64;

            // what direction to step in x or y-direction (either +1 or -1)
            let mut step = Point { x: 0, y: 0 };

            let mut hit = 0; // was there a wall hit?
            let mut side: i32 = 1; // was a NS or a EW wall hit?

            if ray_player_facing.x < 0.0 {
                step.x = -1;
                side_dist.x = (player.position.x - (map.x as f64)) * delta_dist.x;
            } else {
                step.x = 1;
                side_dist.x = (map.x as f64 + 1.0 - player.position.x) * delta_dist.x;
            }
            if ray_player_facing.y < 0.0 {
                step.y = -1;
                side_dist.y = (player.position.y - map.y as f64) * delta_dist.y;
            } else {
                step.y = 1;
                side_dist.y = (map.y as f64 + 1.0 - player.position.y) * delta_dist.y;
            }

            while hit == 0 {
                // jump to next map square, OR in x-direction, OR in y-direction
                if side_dist.x < side_dist.y {
                    side_dist.x += delta_dist.x;
                    map.x += step.x as i32;
                    side = 0;
                } else {
                    side_dist.y += delta_dist.y;
                    map.y += step.y as i32;
                    side = 1;
                }
                // Check if ray has hit a wall
                if world_map[map.x as usize][map.y as usize] > 0 {
                    hit = 1;
                }
            }
            if side == 0 {
                perp_wall_dist = (map.x as f64 - player.position.x + (1.0 - step.x as f64) / 2.0)
                    / ray_player_facing.x;
            } else {
                perp_wall_dist = (map.y as f64 - player.position.y + (1.0 - step.y as f64) / 2.0)
                    / ray_player_facing.y;
            }
            // Calculate height of line to draw on screen
            let line_height = map_height as f64 / perp_wall_dist;

            // calculate lowest and highest pixel to fill in current stripe
            let mut draw_start = -line_height / 2.0 + map_height as f64 / 2.0;
            if draw_start < 0.0 {
                draw_start = 0.0;
            }
            let mut draw_end = line_height / 2.0 + map_height as f64 / 2.0;
            if draw_end >= map_height as f64 {
                draw_end = map_height as f64 - 1.0;
            }

            let mut color = match world_map[map.x as usize][map.y as usize] {
                1 => Color::RGB(255, 0, 0),
                2 => Color::RGB(0, 255, 0),
                3 => Color::RGB(0, 0, 255),
                4 => Color::RGB(255, 255, 255),
                _ => Color::RGB(255, 255, 0),
            };

            if side == 1 {
                color = darken_color(color)
            }

            canvas.set_draw_color(color);

            canvas
                .draw_line(
                    SdlPoint::new(x, draw_start as i32),
                    SdlPoint::new(x, draw_end as i32),
                )
                .unwrap();
        }

        canvas.present();
    }
}
